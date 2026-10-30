"""
Day 105: Decision Trees From Scratch -- Entropy and Gini on Real Data

Fifteen weeks of neural networks; Week 16 returns to classical machine
learning, where trees still win many tabular problems. A decision tree asks
one yes/no question about one feature at a time ('is mean radius <= 15.0?'),
splitting the data into purer and purer groups, and predicts the majority
class of the group a new example lands in.

Everything is written from scratch in NumPy:
  * impurity: Gini (1 - sum p^2) and entropy (-sum p log2 p)
  * the best split: every feature, every threshold between neighbouring values,
    found with sorted cumulative counts instead of a loop over thresholds
  * the tree: recursive splitting until a depth limit, a minimum leaf size, or
    a pure node

The data:
  * Breast Cancer Wisconsin (diagnostic), bundled with scikit-learn: 569
    tumours, 30 measurements from cell images, malignant or benign.
  * Titanic (the repo's own Week 3 copy), for a tree small enough to read.

  1. Impurity, and the root split each criterion chooses.
  2. A correctness check: the fast split search against brute force.
  3. Depth: training and test accuracy from depth 1 to unlimited.
  4. Choosing the depth by 5-fold cross-validation (from scratch).
  5. Gini vs entropy.
  6. Instability: the same tree on bootstrap resamples (why Day 107 averages many).
  7. A readable tree: Titanic.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported exercise quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 99's bootstrap
from collections import Counter                        # Counter: root features across resamples

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import pandas as pd                                    # pandas: reading the Titanic CSV
from sklearn.datasets import load_breast_cancer        # the bundled dataset only (no scikit-learn models are used)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, os.path.join(HERE, "..", "week-15-practical-nlp-pretrained"))  # Day 99's exercise
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day99_exercise as d99                        # Day 99: bootstrap_ci
TITANIC = os.path.join(HERE, "..", "week-03-intro-to-machine-learning", "titanic.csv")  # the repo's own copy
DEPTHS = list(range(1, 11)) + [None]                    # None = grow until pure or min_leaf stops it
MIN_LEAF = 1                                            # smallest allowed leaf (1 = no limit)


# =============================================================================
# IMPURITY AND THE BEST SPLIT -- NEW
# =============================================================================
def gini(counts):                                       # NEW: 1 - sum p_k^2, for class counts along the last axis
    n = counts.sum(-1, keepdims=True)                     # examples in the group
    p = counts / np.maximum(n, 1)                         # class proportions
    return 1 - (p ** 2).sum(-1)


def entropy(counts):                                    # NEW: -sum p_k log2 p_k (0 log 0 = 0)
    n = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):  # log2(0) is -inf, but it is multiplied by 0
        return np.where(p > 0, -p * np.log2(p), 0.0).sum(-1)  # each term is >= 0, so a pure group gives exactly 0


def best_split(X, y, n_classes, impurity, min_leaf=1):  # NEW: (feature, threshold, gain) of the best split, or None
    n, best = len(y), None                                # examples; best so far
    parent = impurity(np.bincount(y, minlength=n_classes)[None])[0]  # impurity before splitting
    for f in range(X.shape[1]):                           # every feature
        order = np.argsort(X[:, f], kind="stable"); xs, ys = X[order, f], y[order]  # sorted values and their labels
        left = np.cumsum(np.eye(n_classes, dtype=int)[ys], axis=0)[:-1]  # class counts left of each gap (first i+1 examples)
        right = np.bincount(y, minlength=n_classes) - left  # and right of it
        nl = np.arange(1, n)                              # left sizes
        ok = (xs[1:] != xs[:-1]) & (nl >= min_leaf) & (n - nl >= min_leaf)  # only between DIFFERENT values, leaves big enough
        if not ok.any():
            continue
        child = (nl * impurity(left) + (n - nl) * impurity(right)) / n  # weighted impurity after the split
        gain = np.where(ok, parent - child, -np.inf)      # impurity removed
        i = int(np.argmax(gain))
        if best is None or gain[i] > best[2] + 1e-12:     # strictly better (ties keep the earlier feature)
            best = (f, (xs[i] + xs[i + 1]) / 2, float(gain[i]))  # threshold halfway between neighbours
    return best if best and best[2] > 1e-12 else None     # no split that helps


# =============================================================================
# THE TREE -- NEW
# =============================================================================
def grow(X, y, n_classes, impurity, depth=0, max_depth=None, min_leaf=MIN_LEAF):  # NEW: a nested dict
    counts = np.bincount(y, minlength=n_classes)          # this node's classes
    node = {"counts": counts, "pred": int(np.argmax(counts))}  # a leaf predicts its majority class
    if (max_depth is not None and depth >= max_depth) or counts.max() == len(y):  # depth limit, or already pure
        return node
    s = best_split(X, y, n_classes, impurity, min_leaf)
    if s is None:                                         # nothing helps
        return node
    f, t, g = s; m = X[:, f] <= t                         # the question, and who answers yes
    node.update(feature=f, threshold=t, gain=g,
                left=grow(X[m], y[m], n_classes, impurity, depth + 1, max_depth, min_leaf),
                right=grow(X[~m], y[~m], n_classes, impurity, depth + 1, max_depth, min_leaf))
    return node


def predict(node, X):                                   # NEW: follow the questions down to a leaf, for every row
    out = np.empty(len(X), dtype=int)
    for i, x in enumerate(X):
        n = node
        while "feature" in n:                               # until a leaf
            n = n["left"] if x[n["feature"]] <= n["threshold"] else n["right"]
        out[i] = n["pred"]
    return out


def size(node):                                         # (leaves, depth)
    if "feature" not in node:
        return 1, 0
    (a, da), (b, db) = size(node["left"]), size(node["right"])
    return a + b, 1 + max(da, db)


def show(node, names, classes, indent="  "):            # NEW: the tree as nested if/else rules
    if "feature" not in node:
        return [f"{indent}-> {classes[node['pred']]}  {node['counts'].tolist()}"]
    q = f"{names[node['feature']]} <= {node['threshold']:.3g}"
    return ([f"{indent}if {q}:"] + show(node["left"], names, classes, indent + "    ") +
            [f"{indent}else:"] + show(node["right"], names, classes, indent + "    "))


def stratified_split(y, test_frac, seed):               # indices for a split that keeps the class balance
    rng = np.random.RandomState(seed); tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); k = int(round(len(idx) * test_frac))
        te += list(idx[:k]); tr += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


def kfold_accuracy(X, y, impurity, max_depth, k=5, seed=0):  # NEW: mean validation accuracy over k folds
    folds = np.array_split(np.random.RandomState(seed).permutation(len(y)), k)
    accs = []
    for i in range(k):
        va = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        t = grow(X[tr], y[tr], 2, impurity, max_depth=max_depth)
        accs.append(np.mean(predict(t, X[va]) == y[va]))
    return np.mean(accs), np.std(accs)


# =============================================================================
# DEMOS
# =============================================================================
def demo_impurity(X, y, names):
    print("=" * 70)
    print("1. IMPURITY, AND THE ROOT SPLIT")
    print("=" * 70)
    for p in (0.5, 0.9, 1.0):
        c = np.array([[p, 1 - p]]) * 100                    # a group of 100 with this class mix
        print(f"  a group {p:.0%} / {1 - p:.0%}: Gini {gini(c)[0]:.3f}, entropy {entropy(c)[0]:.3f} bits")
    print(f"\nBreast cancer: {len(y)} tumours, {X.shape[1]} features, {np.mean(y == 1):.3f} benign")
    for name, imp in (("Gini", gini), ("entropy", entropy)):
        f, t, g = best_split(X, y, 2, imp)
        m = X[:, f] <= t
        print(f"  {name:<8} root split: {names[f]} <= {t:.4g} (impurity removed {g:.4f}); left {np.bincount(y[m], minlength=2).tolist()}, "
              f"right {np.bincount(y[~m], minlength=2).tolist()} (malignant, benign)")
    print("Actual output from this lesson's run.")


def demo_check():
    print("\n" + "=" * 70)
    print("2. A CORRECTNESS CHECK -- the fast split search against brute force")
    print("=" * 70)
    rng = np.random.RandomState(0); agree = 0
    for trial in range(30):                               # 30 small random problems, with repeated values
        X = rng.randint(0, 6, (25, 3)).astype(float); y = rng.randint(0, 3, 25)
        fast = best_split(X, y, 3, gini)
        best = None                                         # brute force: every feature, every threshold, by loops
        parent = gini(np.bincount(y, minlength=3)[None])[0]
        for f in range(3):
            for t in np.unique(X[:, f])[:-1]:
                m = X[:, f] <= t
                g = parent - (m.sum() * gini(np.bincount(y[m], minlength=3)[None])[0] + (~m).sum() * gini(np.bincount(y[~m], minlength=3)[None])[0]) / 25
                if best is None or g > best[1] + 1e-12:
                    best = (f, g)
        agree += fast is not None and abs(fast[2] - best[1]) < 1e-12
    print(f"30 random problems (25 rows, 3 features, 3 classes): the fast search finds the brute-force best gain in {agree} of 30")
    print("Actual output from this lesson's run.")


def demo_depth(X, y, tr, te):
    print("\n" + "=" * 70)
    print("3. DEPTH -- training and test accuracy")
    print("=" * 70)
    print(f"stratified split: {len(tr)} training, {len(te)} test tumours (seed 0); Gini; minimum leaf {MIN_LEAF}")
    rows = []
    print(f"\n{'max depth':>10}{'leaves':>8}{'depth':>7}{'train acc':>11}{'test acc':>10}")
    for d in DEPTHS:
        t = grow(X[tr], y[tr], 2, gini, max_depth=d); leaves, depth = size(t)
        a_tr, a_te = np.mean(predict(t, X[tr]) == y[tr]), np.mean(predict(t, X[te]) == y[te])
        rows.append((d, leaves, depth, a_tr, a_te))
        print(f"{str(d):>10}{leaves:>8}{depth:>7}{a_tr:>11.4f}{a_te:>10.4f}")
    print("Actual output from this lesson's run.")
    return rows                                             # the plot uses them


def demo_cv(X, y, tr, te):
    print("\n" + "=" * 70)
    print("4. CHOOSING THE DEPTH BY 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    cv = {d: kfold_accuracy(X[tr], y[tr], gini, d) for d in DEPTHS}  # on the training part only
    for d, (m, s) in cv.items():
        print(f"  max depth {str(d):>4}: cross-validated accuracy {m:.4f} (+- {s:.4f} across folds)")
    best = max(DEPTHS, key=lambda d: (round(cv[d][0], 10), -(d or 99)))  # highest mean; ties go to the shallower tree
    t = grow(X[tr], y[tr], 2, gini, max_depth=best)
    c = (predict(t, X[te]) == y[te]).astype(float); lo, hi = d99.bootstrap_ci(c)
    print(f"chosen: max depth {best}; test accuracy {c.mean():.4f}, 95% interval [{lo:.4f}, {hi:.4f}] (Day 99's bootstrap)")
    print("Actual output from this lesson's run.")
    return cv, best                                          # later sections use them


def demo_criteria(X, y, tr, te, best):
    print("\n" + "=" * 70)
    print("5. GINI VS ENTROPY")
    print("=" * 70)
    preds = {}
    for name, imp in (("Gini", gini), ("entropy", entropy)):
        for d in (best, None):
            t = grow(X[tr], y[tr], 2, imp, max_depth=d); p = predict(t, X[te]); preds[(name, d)] = p
            print(f"  {name:<8} max depth {str(d):>4}: {size(t)[0]:>3} leaves, test accuracy {np.mean(p == y[te]):.4f}")
    for d in (best, None):
        print(f"  the two criteria agree on {np.mean(preds[('Gini', d)] == preds[('entropy', d)]):.4f} of test tumours at max depth {d}")
    print("Actual output from this lesson's run.")


def demo_instability(X, y, tr, te, names, best):
    print("\n" + "=" * 70)
    print("6. INSTABILITY -- the same recipe on 30 bootstrap resamples")
    print("=" * 70)
    rng = np.random.RandomState(1); roots, accs, P = Counter(), [], []
    for b in range(30):
        s = rng.randint(0, len(tr), len(tr)); idx = tr[s]   # a resample of the training part, with replacement
        t = grow(X[idx], y[idx], 2, gini, max_depth=best)
        roots[names[t["feature"]]] += 1; p = predict(t, X[te]); P.append(p); accs.append(np.mean(p == y[te]))
    P = np.array(P); vote = (P.mean(0) > 0.5).astype(int)
    print("root question chosen across resamples: " + ", ".join(f"{k} {v}" for k, v in roots.most_common()))
    print(f"test accuracy of single trees: mean {np.mean(accs):.4f}, min {np.min(accs):.4f}, max {np.max(accs):.4f}")
    print(f"test tumours on which the 30 trees disagree at all: {np.mean((P.min(0) != P.max(0))):.4f}")
    print(f"a majority vote of the 30 trees (bagging, Day 107's subject): test accuracy {np.mean(vote == y[te]):.4f}")
    print("Actual output from this lesson's run.")
    return roots, accs                                       # the plot uses them


def demo_titanic():
    print("\n" + "=" * 70)
    print("7. A READABLE TREE -- Titanic")
    print("=" * 70)
    d = pd.read_csv(TITANIC)                              # the repo's own copy
    feats = ["Pclass", "is_female", "Age", "SibSp", "Parch", "Fare"]
    d["is_female"] = (d["Sex"] == "female").astype(float)
    d["Age"] = d["Age"].fillna(d["Age"].median())         # 177 missing ages: the median (a simplification)
    X, y = d[feats].to_numpy(float), d["Survived"].to_numpy()
    tr, te = stratified_split(y, 0.3, 0)
    t = grow(X[tr], y[tr], 2, gini, max_depth=3)
    print(f"{len(y)} passengers ({len(tr)} training, {len(te)} test), {y.mean():.3f} survived; Gini, max depth 3; "
          "counts are [died, survived] among training passengers")
    print("\n".join(show(t, feats, ["died", "survived"])))
    women = (X[te, 1] == 1).astype(int)                   # the one-rule baseline: women survive
    print(f"test accuracy: tree {np.mean(predict(t, X[te]) == y[te]):.4f}; 'women survive, men do not' {np.mean(women == y[te]):.4f}")
    print("Actual output from this lesson's run.")


def plot_results(rows, cv, roots):                      # three panels: impurity | accuracy vs depth | root features
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))           # canvas
    p = np.linspace(0, 1, 201); c = np.stack([p, 1 - p], -1)
    ax[0].plot(p, gini(c), label="Gini"); ax[0].plot(p, entropy(c), label="entropy (bits)"); ax[0].plot(p, entropy(c) / 2, ":", label="entropy / 2")
    ax[0].set_xlabel("share of class 1 in a group"); ax[0].set_ylabel("impurity"); ax[0].set_title("Two impurity measures", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    x = np.arange(len(rows)); lab = [str(r[0]) if r[0] else "none" for r in rows]
    ax[1].plot(x, [r[3] for r in rows], marker="o", label="training"); ax[1].plot(x, [r[4] for r in rows], marker="o", label="test")
    ax[1].plot(x, [cv[r[0]][0] for r in rows], marker="s", linestyle="--", label="5-fold CV (training part)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_xlabel("max depth"); ax[1].set_ylabel("accuracy"); ax[1].set_ylim(0.85, 1.005)
    ax[1].set_title("Breast cancer: accuracy against depth", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    k = [a for a, _ in roots.most_common()]; ax[2].barh(k, [roots[a] for a in k], color="#9467bd"); ax[2].invert_yaxis()
    ax[2].set_xlabel("resamples (of 30)"); ax[2].set_title("Root question across bootstrap resamples", fontsize=10); ax[2].grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig("tree_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved tree_results.png")


def draw_diagram():                                     # one node's search, and the recursion
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 105: growing a decision tree", ha="center", fontsize=12, fontweight="bold")
    box(2.2, 2.2, 3.6, 2.6, "at each node:\nfor every feature\n  sort the values\n  cumulative class counts\n  impurity left / right at every gap\nkeep the largest impurity drop", "#FAEEDA", fs=8)
    box(7.2, 3.1, 3.2, 0.8, "is feature f <= threshold t ?", "#DCEBF7")
    box(5.9, 1.3, 2.4, 0.9, "yes: grow again\non these rows", "#E5F2E1")
    box(8.5, 1.3, 2.4, 0.9, "no: grow again\non these rows", "#E5F2E1")
    box(11.5, 2.2, 2.6, 1.6, "stop when:\nmax depth reached\nnode is pure\nno split helps", "#EEEDFE")
    ax.annotate("", xy=(5.56, 3.1), xytext=(4.04, 2.6), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(5.9, 1.79), xytext=(6.7, 2.66), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(8.5, 1.79), xytext=(7.7, 2.66), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(10.16, 2.2), xytext=(9.74, 1.5), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("tree_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved tree_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 106 is pruning: growing the full tree and cutting it back (cost-complexity pruning), with a real "
          "demonstration of over- and underfitting.")


def main():                                             # runs every section in order
    bc = load_breast_cancer(); X, y, names = bc.data, bc.target, list(bc.feature_names)  # 0 = malignant, 1 = benign
    tr, te = stratified_split(y, 0.3, 0)                   # one fixed split
    demo_impurity(X, y, names)                             # 1. impurity
    demo_check()                                           # 2. correctness
    rows = demo_depth(X, y, tr, te)                        # 3. depth
    cv, best = demo_cv(X, y, tr, te)                       # 4. cross-validation
    demo_criteria(X, y, tr, te, best)                      # 5. Gini vs entropy
    roots, accs = demo_instability(X, y, tr, te, names, best)  # 6. instability
    demo_titanic()                                         # 7. Titanic
    plot_results(rows, cv, roots)                          # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 106


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
