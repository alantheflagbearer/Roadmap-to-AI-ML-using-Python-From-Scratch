"""
Day 107: Random Forests From Scratch -- Bagging and a Real Variance-Reduction Demo

Day 105 found a single tree unstable: resample the data and the tree changes.
Day 106 fought that by making one tree smaller. A random forest does the
opposite: it grows many large trees that disagree, and averages them.

Everything is written from scratch in NumPy, on Day 105's split search:
  * bagging: each tree sees a bootstrap resample of the rows
  * random feature subsets: each question may use only mtry random features
  * out-of-bag (OOB) rows: the rows a tree never saw act as its free test set
  * a real measurement of variance, on repeated training sets

  1. The bootstrap: how many rows does a resample really contain?
  2. Correctness checks, including scikit-learn's forest as an outside check.
  3. Variance: measured, tree by tree, against bias.
  4. How many trees? Accuracy and the OOB estimate as the forest grows.
  5. OOB as a free validation set: does a forest still overfit?
  6. mtry: strong trees against different trees.
  7. Useless features: where the random subsets hurt.
  8. Is it better? Trees, pruning, bagging and forests on three datasets.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Days 105 and 106

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_breast_cancer, load_digits  # bundled datasets
from sklearn.ensemble import RandomForestClassifier    # used ONLY as an outside check of the from-scratch forest
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Days 105 and 106 sit next to this file
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing
    import day106_pruning as d106                       # Day 106: cv_alpha, at_alpha, load_titanic
d105 = d106.d105                                        # Day 105: gini, best_split, grow, predict, stratified_split
gini, best_split, stratified_split = d105.gini, d105.best_split, d105.stratified_split  # shorter names, unchanged code
TREES = 100                                             # trees in a forest, unless a section says otherwise
RUNS, SUB, BAG = 30, 300, 64                            # section 3: training sets, their size, trees per forest
COUNTS = [1, 2, 4, 8, 16, 32, 64]                       # section 3: forest sizes
MTRYS = [1, 2, 3, 5, 8, 12, 20, 30]                     # section 6: features allowed per question
NOISE = [0, 10, 30, 100, 300]                           # section 7: useless columns added
LEAVES = [1, 2, 5, 10, 20, 40]                          # section 5: minimum leaf sizes
SPLITS = {"Titanic": 20, "Breast cancer": 20, "Digits": 6}  # section 8: random splits per dataset


# =============================================================================
# THE FOREST -- NEW
# =============================================================================
def grow_random(X, y, n_classes, mtry, rng, depth=0, max_depth=None, min_leaf=1):  # NEW: a tree whose questions see mtry features
    counts = np.bincount(y, minlength=n_classes)          # this node's classes
    node = {"counts": counts, "pred": int(np.argmax(counts))}  # Day 105's node
    if (max_depth is not None and depth >= max_depth) or counts.max() == len(y):  # depth limit, or already pure
        return node
    order = rng.permutation(X.shape[1])                   # the features in random order
    feats = order[:mtry]; s = best_split(X[:, feats], y, n_classes, gini, min_leaf); k = mtry  # the first mtry of them
    while s is None and k < X.shape[1]:                   # nothing useful there: look at one more feature at a time
        feats = order[k:k + 1]; s = best_split(X[:, feats], y, n_classes, gini, min_leaf); k += 1
    if s is None:                                         # no feature helps
        return node
    f, t = int(feats[s[0]]), s[1]; m = X[:, f] <= t       # the question, in the original feature numbers
    node.update(feature=f, threshold=t, gain=s[2],
                left=grow_random(X[m], y[m], n_classes, mtry, rng, depth + 1, max_depth, min_leaf),
                right=grow_random(X[~m], y[~m], n_classes, mtry, rng, depth + 1, max_depth, min_leaf))
    return node


def proba(node, X, n_classes):                          # NEW: class probabilities for many rows at once
    out = np.zeros((len(X), n_classes))                   # one row of probabilities per example
    stack = [(node, np.arange(len(X)))]                   # nodes still to visit, with the rows that reached them
    while stack:
        n, idx = stack.pop()
        if "feature" not in n:                              # a leaf: every row here gets its class shares
            out[idx] = n["counts"] / n["counts"].sum()
            continue
        m = X[idx, n["feature"]] <= n["threshold"]          # who answers yes
        stack.append((n["left"], idx[m])); stack.append((n["right"], idx[~m]))  # both sides, one level down
    return out


def fit_forest(X, y, n_classes, n_trees, mtry, seed, max_depth=None, min_leaf=1):  # NEW: (trees, the bootstrap of each)
    rng = np.random.RandomState(seed); n = len(y); trees, bags = [], []  # one random stream for the whole forest
    for _ in range(n_trees):
        idx = rng.randint(0, n, n); bags.append(idx)        # a bootstrap resample of the rows
        trees.append(grow_random(X[idx], y[idx], n_classes, mtry, rng, max_depth=max_depth, min_leaf=min_leaf))
    return trees, bags


def forest_proba(trees, X, n_classes):                  # NEW: the average of every tree's probabilities
    return np.mean([proba(t, X, n_classes) for t in trees], axis=0)


def oob_accuracy(trees, bags, X, y, n_classes):         # NEW: each row judged only by trees that never saw it
    votes = np.zeros((len(y), n_classes))                 # summed probabilities
    for t, idx in zip(trees, bags):
        out = np.setdiff1d(np.arange(len(y)), idx)          # the rows this tree did not get
        votes[out] += proba(t, X[out], n_classes)
    seen = votes.sum(1) > 0                               # rows that were left out at least once
    return float(np.mean(np.argmax(votes[seen], 1) == y[seen])), float(seen.mean())


def accuracy(p, y):                                     # share of rows whose largest probability is the right class
    return float(np.mean(np.argmax(p, 1) == y))


def sqrt_features(p):                                   # the usual mtry for classification
    return max(1, int(np.sqrt(p)))


class InOrder:                                          # stands in for a random generator: features in their usual order
    def permutation(self, n):
        return np.arange(n)


def data():                                             # the three datasets of this lesson
    bc = load_breast_cancer(); dg = load_digits()
    return {"Titanic": d106.load_titanic(), "Breast cancer": (bc.data, bc.target), "Digits": (dg.data, dg.target)}


# =============================================================================
# DEMOS
# =============================================================================
def demo_bootstrap():
    print("=" * 70)
    print("1. THE BOOTSTRAP -- what a resample really contains")
    print("=" * 70)
    rng = np.random.RandomState(0)
    print(f"{'rows':>8}{'distinct share':>16}{'(1 - 1/n)^n left out':>23}")
    for n in (10, 100, 398, 10000):
        share = np.mean([len(np.unique(rng.randint(0, n, n))) / n for _ in range(2000 if n < 1000 else 100)])  # measured
        print(f"{n:>8}{share:>16.4f}{(1 - 1 / n) ** n:>23.4f}")
    print(f"as n grows the share left out approaches 1/e = {np.exp(-1):.4f}, so each tree never sees about a third of the rows")
    print("Actual output from this lesson's run.")


def demo_check(sets):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS CHECKS")
    print("=" * 70)
    X, y = sets["Breast cancer"]; tr, te = stratified_split(y, 0.3, 0)
    a = d105.grow(X[tr], y[tr], 2, gini)                  # Day 105's tree
    b = grow_random(X[tr], y[tr], 2, X.shape[1], np.random.RandomState(0))  # every feature, in a random order, no resample
    c = grow_random(X[tr], y[tr], 2, X.shape[1], InOrder())  # every feature, in Day 105's order, no resample
    pa, pb, pc = d105.predict(a, X[te]), np.argmax(proba(b, X[te], 2), 1), np.argmax(proba(c, X[te], 2), 1)
    print(f"mtry = all features, no bootstrap, features in Day 105's order: {np.mean(pa == pc):.4f} of test predictions equal Day 105's tree")
    print(f"the same with features in a random order: {np.mean(pa == pb):.4f}, because tied splits go to whichever feature is tried first")
    p = forest_proba(fit_forest(X[tr], y[tr], 2, 5, 5, 0)[0], X[te], 2)
    print(f"forest probabilities sum to 1 for every row: largest error {np.abs(p.sum(1) - 1).max():.1e}")
    a, b = [], []                                         # from scratch against scikit-learn, same settings
    for s in range(10):
        tr, te = stratified_split(y, 0.3, s)
        a.append(accuracy(forest_proba(fit_forest(X[tr], y[tr], 2, TREES, sqrt_features(X.shape[1]), s)[0], X[te], 2), y[te]))
        b.append(RandomForestClassifier(TREES, max_features="sqrt", random_state=s).fit(X[tr], y[tr]).score(X[te], y[te]))
    print(f"breast cancer, 10 splits, {TREES} trees: from scratch {np.mean(a):.4f}, scikit-learn {np.mean(b):.4f}, mean gap {np.mean(np.array(a) - np.array(b)):+.4f}")
    print("Actual output from this lesson's run.")


def decomposition(P, y):                                # NEW: variance, and squared error of the mean prediction
    mean = P.mean(0)                                      # the average prediction per test row, across training sets
    return float(((P - mean) ** 2).mean()), float(((mean - y) ** 2).mean())


def demo_variance(X, y, tr, te):
    print("\n" + "=" * 70)
    print("3. VARIANCE -- measured on repeated training sets")
    print("=" * 70)
    print(f"Titanic: {RUNS} random training sets of {SUB} passengers from the {len(tr)} training rows, the same {len(te)} test passengers, {BAG} trees each")
    print("per test row and across training sets: variance of p(survived), and (mean p - label)^2, which holds the bias and the noise in the label")
    out = {}
    for name, mtry in (("bagging (all 6 features)", 6), ("forest (mtry 2)", 2)):
        P = np.zeros((RUNS, BAG, len(te)))                  # training set, tree, test row
        for r in range(RUNS):
            s = np.random.RandomState(r).choice(tr, SUB, replace=False)  # one training set
            trees, _ = fit_forest(X[s], y[s], 2, BAG, mtry, r)
            P[r] = np.stack([proba(t, X[te], 2)[:, 1] for t in trees])  # every tree's prediction
        print(f"\n{name}")
        print(f"{'trees':>7}{'variance':>11}{'bias+noise':>12}{'Brier':>9}{'accuracy':>10}")
        rows = []
        for b in COUNTS:
            F = P[:, :b].mean(1)                              # the forest of the first b trees, per training set
            v, e = decomposition(F, y[te]); acc = np.mean((F > 0.5) == y[te])
            rows.append((b, v))
            print(f"{b:>7}{v:>11.4f}{e:>12.4f}{v + e:>9.4f}{acc:>10.4f}")
        q = P - P.mean((0, 1), keepdims=True)               # deviation from the overall mean, per test row
        sig = float((q ** 2).mean())                        # one tree's variance
        s = q.sum(1)                                        # for the covariance between two trees of one forest
        cov = float(((s ** 2 - (q ** 2).sum(1)) / (BAG * (BAG - 1))).mean())  # over all pairs of different trees
        rho = cov / sig
        print(f"one tree's variance {sig:.4f}; correlation between two trees of a forest {rho:.3f}; so the floor for many trees is rho * sigma^2 = {cov:.4f}")
        out[name] = (rows, cov, sig)
    print("Actual output from this lesson's run.")
    return out                                               # the plot uses it


def demo_curve(sets):
    print("\n" + "=" * 70)
    print("4. HOW MANY TREES? accuracy and OOB as a forest grows to 300")
    print("=" * 70)
    curves = {}
    for name in ("Titanic", "Breast cancer"):
        X, y = sets[name]; tr, te = stratified_split(y, 0.3, 0)
        trees, bags = fit_forest(X[tr], y[tr], 2, 300, sqrt_features(X.shape[1]), 0)
        S = np.cumsum([proba(t, X[te], 2) for t in trees], axis=0)  # running sum of test probabilities
        rows = []
        print(f"\n{name}: {len(tr)} training, {len(te)} test")
        print(f"{'trees':>7}{'test acc':>10}{'OOB acc':>10}{'rows with an OOB vote':>23}")
        for b in (1, 2, 5, 10, 25, 50, 100, 200, 300):
            o, seen = oob_accuracy(trees[:b], bags[:b], X[tr], y[tr], 2)
            a = accuracy(S[b - 1], y[te]); rows.append((b, a, o))
            print(f"{b:>7}{a:>10.4f}{o:>10.4f}{seen:>23.4f}")
        curves[name] = rows
    print("Actual output from this lesson's run.")
    return curves                                            # the plot uses them


def demo_oob(X, y):
    print("\n" + "=" * 70)
    print("5. OOB AS A FREE VALIDATION SET -- does a forest still overfit?")
    print("=" * 70)
    print(f"Titanic, 10 splits, {TREES} trees, mtry 2: forests with a minimum leaf size, unpruned at 1")
    print(f"\n{'min leaf':>9}{'train acc':>11}{'OOB acc':>9}{'test acc':>10}{'OOB - test':>12}")
    res = {}
    for k in LEAVES:
        a, o, t = [], [], []
        for s in range(10):
            tr, te = stratified_split(y, 0.3, s)
            trees, bags = fit_forest(X[tr], y[tr], 2, TREES, 2, s, min_leaf=k)
            a.append(accuracy(forest_proba(trees, X[tr], 2), y[tr])); o.append(oob_accuracy(trees, bags, X[tr], y[tr], 2)[0])
            t.append(accuracy(forest_proba(trees, X[te], 2), y[te]))
        res[k] = (np.array(o), np.array(t))
        print(f"{k:>9}{np.mean(a):>11.4f}{np.mean(o):>9.4f}{np.mean(t):>10.4f}{np.mean(o) - np.mean(t):>+12.4f}")
    O = np.stack([res[k][0] for k in LEAVES]); T = np.stack([res[k][1] for k in LEAVES])  # leaf size by split
    pick = O.argmax(0)                                    # the OOB choice on each split
    chosen = T[pick, np.arange(10)]
    print(f"choosing the leaf size by OOB on each split: test {chosen.mean():.4f}, against {T[0].mean():.4f} at leaf size 1; no cross-validation needed")
    print(f"sizes chosen: {[LEAVES[i] for i in pick]}")
    print("Actual output from this lesson's run.")


def demo_mtry(X, y):
    print("\n" + "=" * 70)
    print("6. MTRY -- strong trees against different trees")
    print("=" * 70)
    print(f"breast cancer, 10 splits, {TREES} trees: 'disagree' is the share of test rows on which two trees of a forest differ")
    print(f"\n{'mtry':>6}{'one tree':>10}{'disagree':>10}{'forest':>9}")
    rows = []
    for m in MTRYS:
        one, dis, acc = [], [], []
        for s in range(10):
            tr, te = stratified_split(y, 0.3, s)
            trees, _ = fit_forest(X[tr], y[tr], 2, TREES, m, s)
            P = np.stack([proba(t, X[te], 2) for t in trees])  # tree, row, class
            lab = P.argmax(2)                                 # each tree's answers
            one.append(np.mean(lab == y[te]))
            dis.append(np.mean(lab[:50] != lab[50:]))         # fifty pairs of trees
            acc.append(accuracy(P.mean(0), y[te]))
        rows.append((m, np.mean(one), np.mean(dis), np.mean(acc)))
        print(f"{m:>6}{np.mean(one):>10.4f}{np.mean(dis):>10.4f}{np.mean(acc):>9.4f}")
    print(f"mtry {sqrt_features(X.shape[1])} is the usual default for {X.shape[1]} features; mtry {X.shape[1]} is plain bagging")
    print("Actual output from this lesson's run.")
    return rows                                              # the plot uses them


def demo_noise(X, y):
    print("\n" + "=" * 70)
    print("7. USELESS FEATURES -- where the random subsets hurt")
    print("=" * 70)
    print(f"breast cancer with random columns added, 5 splits, {TREES} trees; a question may see sqrt(p) features, or all of them")
    print(f"\n{'added':>7}{'p':>6}{'sqrt(p)':>9}{'chance of a real one':>22}{'one tree':>10}{'forest':>9}{'bagging':>9}")
    for k in NOISE:
        extra = np.random.RandomState(k).normal(size=(len(y), k))  # columns with no signal at all
        Z = np.hstack([X, extra]); p = Z.shape[1]; m = sqrt_features(p)
        one, frs, bag = [], [], []
        for s in range(5):
            tr, te = stratified_split(y, 0.3, s)
            one.append(np.mean(d105.predict(d105.grow(Z[tr], y[tr], 2, gini), Z[te]) == y[te]))
            frs.append(accuracy(forest_proba(fit_forest(Z[tr], y[tr], 2, TREES, m, s)[0], Z[te], 2), y[te]))
            bag.append(accuracy(forest_proba(fit_forest(Z[tr], y[tr], 2, TREES, p, s)[0], Z[te], 2), y[te]))
        chance = 1 - np.prod([(k - i) / (p - i) for i in range(m)]) if k >= m else 1.0  # at least one real feature in the subset
        print(f"{k:>7}{p:>6}{m:>9}{chance:>22.3f}{np.mean(one):>10.4f}{np.mean(frs):>9.4f}{np.mean(bag):>9.4f}")
    print("Actual output from this lesson's run.")


def demo_compare(sets):
    print("\n" + "=" * 70)
    print(f"8. IS IT BETTER? trees, pruning, bagging and forests, {TREES} trees each")
    print("=" * 70)
    print("the pruned tree is Day 106's (alpha by 5-fold CV); test used once per split; the same splits for every method")
    res = {}
    for name, (X, y) in sets.items():
        k = int(y.max()) + 1; p = X.shape[1]; m = sqrt_features(p); rows = []  # classes, features, mtry
        for s in range(SPLITS[name]):
            tr, te = stratified_split(y, 0.3, s)
            full = d105.grow(X[tr], y[tr], k, gini)
            al, ts, b, cv, se = d106.cv_alpha(X[tr], y[tr], seed=s)
            rows.append((np.mean(d105.predict(full, X[te]) == y[te]),
                         np.mean(d105.predict(d106.at_alpha(al, ts, b[int(np.argmax(cv))]), X[te]) == y[te]),
                         accuracy(forest_proba(fit_forest(X[tr], y[tr], k, TREES, p, s)[0], X[te], k), y[te]),
                         accuracy(forest_proba(fit_forest(X[tr], y[tr], k, TREES, m, s)[0], X[te], k), y[te]),
                         RandomForestClassifier(TREES, max_features="sqrt", random_state=s).fit(X[tr], y[tr]).score(X[te], y[te])))
        r = np.array(rows); res[name] = r
        print(f"\n{name}: {len(y)} rows, {p} features, {k} classes, {SPLITS[name]} splits, mtry {m}")
        for i, n in enumerate(("full tree", "pruned tree", "bagging", "random forest", "scikit-learn forest")):
            print(f"  {n:<20}{r[:, i].mean():>8.4f}  (std {r[:, i].std():.4f})")
        for a, b, n in ((3, 1, "forest minus pruned tree"), (3, 2, "forest minus bagging")):
            d = r[:, a] - r[:, b]; lo, hi = d105.d99.bootstrap_ci(d)
            print(f"  {n:<26}{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; better on {int((d > 0).sum())}, worse on {int((d < 0).sum())} of {SPLITS[name]}")
    print("the splits share rows, so these intervals are too narrow, a rough guide only")
    print("Actual output from this lesson's run.")
    return res


def plot_results(var, curves, sweep):                   # three panels: variance | accuracy against trees | mtry
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))           # canvas
    for (name, (rows, cov, sig)), c in zip(var.items(), ("#1f77b4", "#ff7f0e")):
        ax[0].plot([r[0] for r in rows], [r[1] for r in rows], marker="o", color=c, label=name)
        ax[0].axhline(cov, color=c, linestyle=":", linewidth=1)  # the floor
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("trees in the forest (log scale)"); ax[0].set_ylabel("variance of p(survived)")
    ax[0].set_title("Titanic: variance falls to a floor (dotted)", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for (name, rows), c in zip(curves.items(), ("#1f77b4", "#2ca02c")):
        ax[1].plot([r[0] for r in rows], [r[1] for r in rows], marker="o", color=c, label=f"{name}: test")
        ax[1].plot([r[0] for r in rows], [r[2] for r in rows], marker="s", linestyle="--", color=c, label=f"{name}: OOB")
    ax[1].set_xscale("log"); ax[1].set_xlabel("trees in the forest (log scale)"); ax[1].set_ylabel("accuracy")
    ax[1].set_title("The gain stops after a few dozen trees", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    x = [r[0] for r in sweep]
    ax[2].plot(x, [r[1] for r in sweep], marker="o", label="one tree"); ax[2].plot(x, [r[3] for r in sweep], marker="o", label="forest")
    ax[2].set_xlabel("mtry: features per question"); ax[2].set_ylabel("test accuracy"); ax[2].grid(alpha=0.3)
    b = ax[2].twinx(); b.plot(x, [r[2] for r in sweep], marker="s", linestyle="--", color="#2ca02c", label="two trees disagree")
    b.set_ylabel("share of rows two trees disagree on")
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax[2].set_title("Breast cancer: strong trees against different trees", fontsize=10)
    plt.tight_layout(); plt.savefig("forest_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved forest_results.png")


def draw_diagram():                                     # rows, trees, average
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 107: a random forest", ha="center", fontsize=12, fontweight="bold")
    box(1.3, 2.0, 2.0, 1.6, "training rows\n(n of them)", "#FAEEDA")
    ys = (3.2, 2.0, 0.8)
    for i, y in enumerate(ys):
        box(4.3, y, 2.5, 0.85, f"resample {i + 1}: n rows\nwith replacement", "#DCEBF7")
        box(7.6, y, 2.7, 0.85, f"tree {i + 1}: each question sees\nonly mtry random features", "#E5F2E1")
        ax.annotate("", xy=(3.0, y), xytext=(2.34, 2.0), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(6.2, y), xytext=(5.59, y), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(10.1, 2.0), xytext=(8.99, y), arrowprops=dict(arrowstyle="->", color="#333333"))
    box(11.4, 2.0, 2.4, 1.6, "average the\nprobabilities;\nrows a tree missed\ngive the OOB score", "#EEEDFE")
    plt.savefig("forest_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved forest_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("9. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 108 is feature importance: a forest cannot be read like one tree, so which features did it use? Impurity-based importance against permutation importance.")


def main():                                             # runs every section in order
    sets = data()                                          # the three datasets
    Xt, yt = sets["Titanic"]; Xc, yc = sets["Breast cancer"]
    tr, te = stratified_split(yt, 0.3, 0)                  # Day 105's Titanic split
    demo_bootstrap()                                       # 1. the bootstrap
    demo_check(sets)                                       # 2. correctness
    var = demo_variance(Xt, yt, tr, te)                    # 3. variance
    curves = demo_curve(sets)                              # 4. how many trees
    demo_oob(Xt, yt)                                       # 5. OOB
    sweep = demo_mtry(Xc, yc)                              # 6. mtry
    demo_noise(Xc, yc)                                     # 7. useless features
    demo_compare(sets)                                     # 8. the comparison
    plot_results(var, curves, sweep)                       # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 9. Day 108


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
