"""
Day 106: Decision-Tree Pruning -- Real Over- and Underfitting

Day 105 limited a tree before it grew (a maximum depth). Pruning does the
opposite: grow the whole tree, then cut back the branches that were not worth
their size. Too few leaves underfits, too many overfits, and pruning walks
between the two one cut at a time.

Everything is written from scratch in NumPy, on Day 105's own tree:
  * cost-complexity pruning (Breiman et al., 1984): cost = errors + alpha * leaves
  * the weakest link: the branch whose leaves each save the fewest errors
  * alpha chosen by 5-fold cross-validation, and the one-standard-error rule

  1. The cost, on a tree you can check by hand.
  2. A correctness check: the pruning path against every possible pruned tree.
  3. Over- and underfitting: Titanic, from one leaf to the full tree.
  4. Choosing alpha by cross-validation, and the pruned tree as rules.
  5. Is it better? Thirty splits: full tree, pruning, depth limit, leaf-size limit.
  6. Label noise: where pruning earns its keep.
  7. The horizon effect: why stopping early can fail where pruning does not.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lesson quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 105
from collections import Counter                        # Counter: which grid values cross-validation chose

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import pandas as pd                                    # pandas: reading the Titanic CSV
from sklearn.datasets import load_breast_cancer        # the bundled dataset only (no scikit-learn models are used)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 105 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day105_decision_trees as d105                # Day 105: gini, grow, predict, size, show, stratified_split
gini, grow, predict, size, show = d105.gini, d105.grow, d105.predict, d105.size, d105.show  # shorter names, unchanged code
stratified_split = d105.stratified_split                # also unchanged
TITANIC = d105.TITANIC                                  # the repo's own copy
FEATS = ["Pclass", "is_female", "Age", "SibSp", "Parch", "Fare"]  # Day 105's Titanic features
SPLITS = 30                                             # repeated train/test splits in section 5
NOISE = [0.0, 0.1, 0.2, 0.3, 0.4]                       # shares of training labels flipped in section 6
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, None]                # the depth limit's grid in section 5
LEAVES = [1, 2, 5, 10, 20, 40, 80]                        # the leaf-size limit's grid in section 5
GAINS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]           # the minimum-gain thresholds of section 7


# =============================================================================
# COST-COMPLEXITY PRUNING -- NEW
# =============================================================================
def node_errors(node):                                  # NEW: training errors if this node were a leaf
    return int(node["counts"].sum() - node["counts"].max())  # everyone outside its majority class


def clone(node):                                        # NEW: a copy of a tree, so cutting never touches the original
    c = dict(node)                                        # this node's fields
    if "feature" in node:                                 # a question: copy both sides
        c["left"], c["right"] = clone(node["left"]), clone(node["right"])
    return c


def branch(node):                                       # NEW: (training errors, leaves) of everything below this node
    if "feature" not in node:                             # a leaf
        return node_errors(node), 1
    (ea, la), (eb, lb) = branch(node["left"]), branch(node["right"])  # add up both sides
    return ea + eb, la + lb


def questions(node):                                    # NEW: every internal node, root first
    if "feature" not in node:                             # a leaf asks nothing
        return []
    return [node] + questions(node["left"]) + questions(node["right"])


def cut(node):                                          # NEW: turn a node into a leaf, in place
    for k in ("feature", "threshold", "gain", "left", "right"):  # forget the question and everything below
        node.pop(k, None)


def cost_path(tree):                                    # NEW: the weakest-link sequence: alphas and their trees
    n, t = int(tree["counts"].sum()), clone(tree)         # training rows; a copy to cut
    alphas, trees = [0.0], [clone(t)]                     # the full tree costs nothing to keep at alpha = 0
    while "feature" in t:                                 # until only the root is left
        nodes = questions(t)                                # the branches that could go
        g = []                                              # each one's alpha
        for q in nodes:
            e, leaves = branch(q)                             # errors and leaves if it stays
            g.append((node_errors(q) - e) / n / (leaves - 1))  # errors saved per extra leaf
        a = min(g)                                          # the weakest link
        for q, v in zip(nodes, g):                          # every branch tied with it goes at the same alpha
            if v <= a + 1e-12:
                cut(q)
        alphas.append(a); trees.append(clone(t))            # record this step
    return alphas, trees


def at_alpha(alphas, trees, a):                         # NEW: the tree in charge at this alpha
    return trees[int(np.searchsorted(np.array(alphas), a + 1e-12, side="right")) - 1]  # the last one with alpha <= a


def cost(tree, a, n):                                   # NEW: error rate + alpha * leaves
    e, leaves = branch(tree)
    return e / n + a * leaves


def every_pruning(node):                                # NEW: all pruned versions of a tree (the brute force)
    leaf = {k: v for k, v in node.items() if k not in ("feature", "threshold", "gain", "left", "right")}  # this node cut
    if "feature" not in node:
        return [leaf]
    out = [leaf]                                          # cutting here is one choice
    for a in every_pruning(node["left"]):                 # keeping the question: any left version with any right version
        for b in every_pruning(node["right"]):
            out.append({**node, "left": a, "right": b})
    return out


def accuracy(tree, X, y):                               # share of rows the tree gets right
    return float(np.mean(predict(tree, X) == y))


def betas(alphas):                                      # NEW: one alpha inside each tree's range (geometric means)
    a = np.array(alphas)                                  # the path's alphas, already increasing
    m = np.sqrt(np.maximum(a[:-1], 1e-12) * a[1:])        # between neighbours
    return np.append(m, a[-1])                            # and the last one, which leaves only the root


def cv_alpha(X, y, k=5, seed=0):                        # NEW: score every alpha of the full path by k-fold CV
    n_classes = int(y.max()) + 1                          # classes in this data
    full = grow(X, y, n_classes, gini)                    # the tree that will really be pruned
    alphas, trees = cost_path(full)
    b = betas(alphas)                                     # the candidates
    folds = np.array_split(np.random.RandomState(seed).permutation(len(y)), k)  # the same folds for every candidate
    scores = np.zeros((k, len(b)))                        # accuracy per fold and candidate
    for i in range(k):
        va = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])  # this round's parts
        al, ts = cost_path(grow(X[tr], y[tr], n_classes, gini))  # its own full tree and its own path
        for j, a in enumerate(b):
            scores[i, j] = accuracy(at_alpha(al, ts, a), X[va], y[va])  # the fold's pruned tree at this alpha
    return alphas, trees, b, scores.mean(0), scores.std(0) / np.sqrt(k)  # the mean, and its standard error


def stop_early(node, min_gain):                         # NEW: pre-pruning by a minimum gain: stop at the first weak question
    t = clone(node)                                       # never touch the original
    for q in questions(t):                                # root first, so a cut removes everything below it
        if "feature" in q and q["gain"] < min_gain:
            cut(q)
    return t


def load_titanic():                                     # Day 105's Titanic features, unchanged
    d = pd.read_csv(TITANIC)                              # the repo's own copy
    d["is_female"] = (d["Sex"] == "female").astype(float)  # a text column to 0/1
    d["Age"] = d["Age"].fillna(d["Age"].median())         # 177 missing ages: the median (a simplification)
    return d[FEATS].to_numpy(float), d["Survived"].to_numpy()


# =============================================================================
# DEMOS
# =============================================================================
def demo_cost():
    print("=" * 70)
    print("1. THE COST, ON A TREE YOU CAN CHECK BY HAND")
    print("=" * 70)
    X = np.array([[1, 1], [2, 1], [3, 2], [4, 2], [5, 3], [6, 3], [7, 4], [8, 4], [9, 5], [10, 5]], dtype=float)
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])          # ten rows: a noisy middle
    t = grow(X, y, 2, gini)                               # the full tree
    print("\n".join(show(t, ["x0", "x1"], ["class 0", "class 1"])))
    print(f"full tree: {branch(t)[1]} leaves, {branch(t)[0]} training errors of {len(y)}")
    alphas, trees = cost_path(t)
    for a, s in zip(alphas, trees):
        e, leaves = branch(s)
        print(f"  from alpha = {a:.4f}: {leaves} leaves, {e} errors, error rate {e / len(y):.2f}")
    print("Actual output from this lesson's run.")


def demo_check():
    print("\n" + "=" * 70)
    print("2. A CORRECTNESS CHECK -- the path against every possible pruned tree")
    print("=" * 70)
    X, y = load_titanic()
    rng = np.random.RandomState(0); agree = total = 0; counts = []  # 40 small trees from real rows
    for trial in range(40):
        s = rng.choice(len(y), 40, replace=False)           # 40 random passengers
        t = grow(X[s], y[s], 2, gini, max_depth=4)          # small enough to enumerate
        alphas, trees = cost_path(t)
        every = every_pruning(t); counts.append(len(every))
        for a in rng.uniform(0, 0.06, 25):                  # 25 random prices per leaf
            brute = min(cost(c, a, len(s)) for c in every)    # the true minimum
            total += 1; agree += abs(cost(at_alpha(alphas, trees, a), a, len(s)) - brute) < 1e-12  # does the path reach it?
    print(f"40 trees from 40 random Titanic passengers each (depth up to 4): between {min(counts)} and {max(counts)} possible pruned trees")
    print(f"at 25 random alphas each, the path's tree has the lowest possible cost in {agree} of {total} cases")
    print("Actual output from this lesson's run.")


def demo_titanic(X, y, tr, te):
    print("\n" + "=" * 70)
    print("3. OVER- AND UNDERFITTING -- Titanic, from one leaf to the full tree")
    print("=" * 70)
    full = grow(X[tr], y[tr], 2, gini)
    alphas, trees = cost_path(full)
    print(f"{len(tr)} training, {len(te)} test passengers; full tree: {size(full)[0]} leaves, depth {size(full)[1]}; "
          f"{len(trees)} trees on the pruning path")
    _, group = np.unique(X[tr], axis=0, return_inverse=True)  # passengers with identical features
    group = group.ravel()                                 # one group number per passenger
    floor = sum(min(np.sum(y[tr][group == g] == 0), np.sum(y[tr][group == g] == 1)) for g in np.unique(group))  # unavoidable errors
    print(f"training errors of the full tree: {branch(full)[0]}; unavoidable, because passengers identical in all six features had different outcomes: {floor}")
    idx = sorted(set(np.linspace(0, len(trees) - 1, 13).round().astype(int).tolist()))  # thirteen steps along the path
    print(f"\n{'alpha':>9}{'leaves':>8}{'train acc':>11}{'test acc':>10}")
    for i in idx:
        print(f"{alphas[i]:>9.5f}{size(trees[i])[0]:>8}{accuracy(trees[i], X[tr], y[tr]):>11.4f}{accuracy(trees[i], X[te], y[te]):>10.4f}")
    path = [(size(s)[0], accuracy(s, X[tr], y[tr]), accuracy(s, X[te], y[te])) for s in trees]  # every step, for the figure
    best = max(path, key=lambda r: r[2])
    print(f"highest test accuracy on the path (looked at for this table only, not used to choose): {best[2]:.4f} at {best[0]} leaves")
    print("Actual output from this lesson's run.")
    return path                                              # the plot uses it


def demo_cv(X, y, tr, te):
    print("\n" + "=" * 70)
    print("4. CHOOSING ALPHA BY 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    alphas, trees, b, m, se = cv_alpha(X[tr], y[tr])
    i = int(np.argmax(m))                                 # the best mean
    ok = np.where(m >= m[i] - se[i])[0]                   # everything within one standard error of it
    j = int(ok[np.argmax(b[ok])])                         # the largest alpha among them, so the smallest tree
    for name, k in (("best CV mean", i), ("one-SE rule", j)):
        s = at_alpha(alphas, trees, b[k])
        print(f"  {name:<13}: alpha {b[k]:.5f}, CV accuracy {m[k]:.4f} (+- {se[k]:.4f}), {size(s)[0]} leaves, "
              f"train {accuracy(s, X[tr], y[tr]):.4f}, test {accuracy(s, X[te], y[te]):.4f}")
    s = at_alpha(alphas, trees, b[j])
    c = (predict(s, X[te]) == y[te]).astype(float); lo, hi = d105.d99.bootstrap_ci(c)
    print(f"one-SE tree: 95% interval for test accuracy [{lo:.4f}, {hi:.4f}] (Day 99's bootstrap)")
    print("the one-SE tree as rules (counts are [died, survived] among training passengers):")
    print("\n".join(show(s, FEATS, ["died", "survived"])))
    print("Actual output from this lesson's run.")
    return [(size(at_alpha(alphas, trees, a))[0], v) for a, v in zip(b, m)], b[i], b[j]  # the plot uses the first


def cv_grid(X, y, make, grid, k=5, seed=0):             # NEW: the grid value with the best k-fold accuracy (ties: earliest)
    folds = np.array_split(np.random.RandomState(seed).permutation(len(y)), k)
    scores = []
    for g in grid:
        acc = []
        for i in range(k):
            va = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
            acc.append(accuracy(make(X[tr], y[tr], g), X[va], y[va]))
        scores.append(np.mean(acc))
    return grid[int(np.argmax(scores))]


def compare(X, y, seed):                                # NEW: one split, four ways of controlling the tree
    tr, te = stratified_split(y, 0.3, seed); out = {}     # a fresh split
    full = grow(X[tr], y[tr], 2, gini)
    out["full tree"] = full
    alphas, trees, b, m, se = cv_alpha(X[tr], y[tr], seed=seed)
    i = int(np.argmax(m)); ok = np.where(m >= m[i] - se[i])[0]; j = int(ok[np.argmax(b[ok])])  # as in section 4
    out["pruned, best CV"] = at_alpha(alphas, trees, b[i])
    out["pruned, one-SE"] = at_alpha(alphas, trees, b[j])
    d = cv_grid(X[tr], y[tr], lambda a, c, g: grow(a, c, 2, gini, max_depth=g), DEPTHS, seed=seed)
    out["depth limit"] = grow(X[tr], y[tr], 2, gini, max_depth=d)
    k = cv_grid(X[tr], y[tr], lambda a, c, g: grow(a, c, 2, gini, min_leaf=g), LEAVES, seed=seed)
    out["leaf-size limit"] = grow(X[tr], y[tr], 2, gini, min_leaf=k)
    return {n: (accuracy(t, X[te], y[te]), size(t)[0]) for n, t in out.items()}, d, k  # scores, and the grid choices


def demo_compare(data):
    print("\n" + "=" * 70)
    print(f"5. IS IT BETTER? {SPLITS} SPLITS, FOUR WAYS OF CONTROLLING A TREE")
    print("=" * 70)
    print(f"every choice made by 5-fold CV on the training part; test used once per split; depth grid {DEPTHS}, leaf-size grid {LEAVES}")
    res = {}
    for name, (X, y) in data.items():
        runs = [compare(X, y, s) for s in range(SPLITS)]    # one result per split
        rows = [r[0] for r in runs]                         # the scores
        res[name] = {m: np.array([r[m] for r in rows]) for m in rows[0]}
        print(f"\n{name}: {len(y)} rows")
        print(f"{'method':<18}{'test acc':>10}{'std':>8}{'leaves':>8}")
        for m, a in res[name].items():
            print(f"{m:<18}{a[:, 0].mean():>10.4f}{a[:, 0].std():>8.4f}{a[:, 1].mean():>8.1f}")
        for m in ("pruned, best CV", "pruned, one-SE", "depth limit", "leaf-size limit"):
            diff = res[name][m][:, 0] - res[name]["full tree"][:, 0]
            lo, hi = d105.d99.bootstrap_ci(diff)            # Day 99's bootstrap over the splits
            print(f"  {m:<18} minus full tree: {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; better on {int((diff > 0).sum())}, worse on {int((diff < 0).sum())} of {SPLITS} splits")
        diff = res[name]["pruned, best CV"][:, 0] - res[name]["depth limit"][:, 0]  # the direct question: post against pre
        lo, hi = d105.d99.bootstrap_ci(diff)
        print(f"  pruned, best CV minus depth limit: {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; better on {int((diff > 0).sum())}, worse on {int((diff < 0).sum())} of {SPLITS} splits")
        print(f"  chosen depth limits: {sorted(Counter(str(r[1]) for r in runs).items())}")
        print(f"  chosen leaf sizes: {sorted(Counter(r[2] for r in runs).items())}")
    print("the splits share rows, so they are not independent: these intervals are too narrow, a rough guide only")
    print("Actual output from this lesson's run.")
    return res                                               # the plot uses it


def demo_noise(X, y):
    print("\n" + "=" * 70)
    print("6. LABEL NOISE -- breast cancer with training labels flipped at random")
    print("=" * 70)
    print(f"test labels stay correct; mean of {SPLITS} splits; one-SE pruning chosen by 5-fold CV on the noisy training part")
    print(f"\n{'flipped':>8}{'full acc':>10}{'full leaves':>13}{'pruned acc':>12}{'pruned leaves':>15}{'gain':>9}")
    rows = []
    for p in NOISE:
        a_f, n_f, a_p, n_p = [], [], [], []
        for s in range(SPLITS):
            tr, te = stratified_split(y, 0.3, s); rng = np.random.RandomState(100 + s)
            yn = y.copy(); flip = rng.choice(tr, int(round(p * len(tr))), replace=False); yn[flip] = 1 - yn[flip]  # noisy copies
            full = grow(X[tr], yn[tr], 2, gini)
            alphas, trees, b, m, se = cv_alpha(X[tr], yn[tr], seed=s)
            i = int(np.argmax(m)); ok = np.where(m >= m[i] - se[i])[0]; j = int(ok[np.argmax(b[ok])])  # one-SE, as before
            t = at_alpha(alphas, trees, b[j])
            a_f.append(accuracy(full, X[te], y[te])); n_f.append(size(full)[0])
            a_p.append(accuracy(t, X[te], y[te])); n_p.append(size(t)[0])
        rows.append((p, np.mean(a_f), np.mean(a_p)))
        print(f"{p:>8.0%}{np.mean(a_f):>10.4f}{np.mean(n_f):>13.1f}{np.mean(a_p):>12.4f}{np.mean(n_p):>15.1f}{np.mean(a_p) - np.mean(a_f):>+9.4f}")
    print("Actual output from this lesson's run.")
    return rows                                              # the plot uses them


def xor_data(n, seed):                                  # two informative features and four useless ones
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, 6))
    y = ((X[:, 0] > 0.5) != (X[:, 1] > 0.5)).astype(int)   # the class is an exclusive or
    flip = rng.uniform(size=n) < 0.05; y[flip] = 1 - y[flip]  # 5% label noise
    return X, y


def demo_horizon():
    print("\n" + "=" * 70)
    print("7. THE HORIZON EFFECT -- an exclusive-or, stopped early or pruned")
    print("=" * 70)
    acc = {g: [] for g in GAINS}; n = {g: [] for g in GAINS}  # one list per fixed threshold
    other = {"stop, gain chosen by CV": ([], []), "grow all, prune (best CV)": ([], []), "grow all, prune (one-SE)": ([], [])}  # the rest
    root = []; good_root = 0; picks = Counter()           # the first split, and the thresholds CV chose
    for s in range(SPLITS):
        Xa, ya = xor_data(400, s); Xb, yb = xor_data(2000, 1000 + s)  # synthetic, on purpose: the structure is known
        full = grow(Xa, ya, 2, gini); root.append(full["gain"]); good_root += full["feature"] in (0, 1)
        for g in GAINS:
            t = stop_early(full, g); acc[g].append(accuracy(t, Xb, yb)); n[g].append(size(t)[0])
        alphas, trees, b, m, se = cv_alpha(Xa, ya, seed=s)
        i = int(np.argmax(m)); ok = np.where(m >= m[i] - se[i])[0]; j = int(ok[np.argmax(b[ok])])
        g = cv_grid(Xa, ya, lambda a, c, v: stop_early(grow(a, c, 2, gini), v), GAINS, seed=s); picks[g] += 1  # a fair threshold
        for name, t in (("stop, gain chosen by CV", stop_early(full, g)), ("grow all, prune (best CV)", at_alpha(alphas, trees, b[i])),
                        ("grow all, prune (one-SE)", at_alpha(alphas, trees, b[j]))):
            other[name][0].append(accuracy(t, Xb, yb)); other[name][1].append(size(t)[0])
    print(f"synthetic on purpose: y = (x0 > 0.5) xor (x1 > 0.5), 5% noise, four useless features; 400 training rows, 2,000 test rows, {SPLITS} datasets")
    print(f"the first split removes only {np.mean(root):.4f} Gini on average (the largest was {np.max(root):.4f}), though two questions nearly solve the problem")
    print(f"the first split used a useful feature (x0 or x1) in {good_root} of {SPLITS} datasets; four of the six features are useless")
    print(f"\n{'rule':<28}{'test acc':>10}{'leaves':>9}")
    for g in GAINS:
        print(f"{'stop if gain < ' + str(g):<28}{np.mean(acc[g]):>10.4f}{np.mean(n[g]):>9.1f}")
    for name, (a, k) in other.items():
        print(f"{name:<28}{np.mean(a):>10.4f}{np.mean(k):>9.1f}")
    print(f"thresholds chosen by CV: {sorted(picks.items())}")
    print("Actual output from this lesson's run.")


def plot_results(path, cvcurve, res, noise):            # three panels: the U-shape | four methods | label noise
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))           # canvas
    p = sorted(path)                                          # by leaves
    ax[0].plot([r[0] for r in p], [r[1] for r in p], marker=".", label="training")
    ax[0].plot([r[0] for r in p], [r[2] for r in p], marker=".", label="test")
    c = sorted(cvcurve); ax[0].plot([r[0] for r in c], [r[1] for r in c], linestyle="--", label="5-fold CV (training part)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("leaves (log scale)"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("Titanic: the pruning path, full tree on the right", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    names = list(next(iter(res.values())).keys()); x = np.arange(len(names)); w = 0.38
    for i, (d, r) in enumerate(res.items()):
        v = [r[m][:, 0].mean() for m in names]
        ax[1].bar(x + (i - 0.5) * w, v, w, label=d)
        for k, m in enumerate(names):
            ax[1].text(x[k] + (i - 0.5) * w, v[k] + 0.003, f"{r[m][:, 1].mean():.0f}", ha="center", fontsize=7.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels([m.replace(", ", ",\n").replace(" limit", "\nlimit") for m in names], fontsize=8)
    ax[1].set_ylim(0.7, 1.02); ax[1].set_ylabel("mean test accuracy"); ax[1].legend(fontsize=8, loc="upper center", ncol=2); ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_title(f"{SPLITS} splits (number above a bar = mean leaves)", fontsize=10)
    q = [100 * r[0] for r in noise]
    ax[2].plot(q, [r[1] for r in noise], marker="o", label="full tree"); ax[2].plot(q, [r[2] for r in noise], marker="o", label="pruned (one-SE)")
    ax[2].set_xlabel("training labels flipped (%)"); ax[2].set_ylabel("accuracy on correct test labels"); ax[2].grid(alpha=0.3)
    ax[2].set_title("Breast cancer under label noise", fontsize=10); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("pruning_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved pruning_results.png")


def draw_diagram():                                     # the cost, the weakest link, and the nested trees
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 106: cost-complexity pruning", ha="center", fontsize=12, fontweight="bold")
    box(1.7, 2.3, 2.8, 2.2, "grow the full tree\n(Day 105)\n\nleaves as pure as\nthe data allows,\nmany leaves", "#FAEEDA")
    box(5.3, 2.3, 3.4, 2.2, "for every branch t:\n\ng(t) = errors saved by t\n/ extra leaves of t\n\ncut the smallest g", "#DCEBF7")
    box(9.0, 2.3, 2.8, 2.2, "a sequence of\nsmaller trees\nT0 > T1 > ... > root,\neach best for a\nrange of alpha", "#E5F2E1")
    box(11.9, 2.3, 1.9, 2.2, "pick alpha\nby 5-fold CV;\nuse the test\nset once", "#EEEDFE")
    ax.annotate("", xy=(3.56, 2.3), xytext=(3.14, 2.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(7.56, 2.3), xytext=(7.04, 2.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(10.91, 2.3), xytext=(10.44, 2.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("repeat until only the root is left", xy=(5.3, 1.17), xytext=(5.3, 0.55), ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("pruning_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved pruning_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 107 is random forests: instead of cutting one tree back, grow many and average them, with a real measurement of the variance that removes.")


def main():                                             # runs every section in order
    bc = load_breast_cancer(); Xc, yc = bc.data, bc.target  # 0 = malignant, 1 = benign
    Xt, yt = load_titanic()                                # the noisier dataset
    tr, te = stratified_split(yt, 0.3, 0)                  # Day 105's Titanic split
    demo_cost()                                            # 1. by hand
    demo_check()                                           # 2. correctness
    path = demo_titanic(Xt, yt, tr, te)                    # 3. over- and underfitting
    cvcurve, _, _ = demo_cv(Xt, yt, tr, te)                # 4. choosing alpha
    res = demo_compare({"Titanic": (Xt, yt), "Breast cancer": (Xc, yc)})  # 5. four methods
    noise = demo_noise(Xc, yc)                             # 6. label noise
    demo_horizon()                                         # 7. the horizon effect
    plot_results(path, cvcurve, res, noise)                # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 107


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
