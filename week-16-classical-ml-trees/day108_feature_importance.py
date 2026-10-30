"""
Day 108: Feature Importance -- Permutation Against Impurity-Based

A forest of 100 trees cannot be read like Day 105's flowchart, so people ask it
which features mattered. Two popular answers are written from scratch here, and
tested on cases where the right answer is known.

  * impurity-based importance (MDI): add up how much impurity each feature
    removed across all the trees; cheap, and computed from the training data
  * permutation importance: shuffle one column of held-out data and measure how
    much accuracy falls; it treats the model as a black box
  * drop-column importance: retrain without the feature; slow, the reference

  1. Correctness checks: by hand, and against scikit-learn.
  2. Breast cancer: the two rankings side by side.
  3. Correlated features: why permutation can call everything useless.
  4. Useless columns: a real ID number and a random coin, on Titanic.
  5. A duplicated feature: how importance gets shared.
  6. Stability: the same question on 10 splits.
  7. Using it: digits, and how few pixels a forest needs.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 107

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import pandas as pd                                    # pandas: PassengerId from the Titanic CSV
from sklearn.ensemble import RandomForestClassifier    # used ONLY as an outside check
from sklearn.tree import DecisionTreeClassifier        # used ONLY as an outside check
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 107 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day107_random_forests as d107                # Day 107: fit_forest, forest_proba, accuracy, sqrt_features, data
d106, d105 = d107.d106, d107.d105                       # Days 106 and 105: Titanic loading, trees, stratified_split
fit_forest, forest_proba, accuracy = d107.fit_forest, d107.forest_proba, d107.accuracy  # shorter names, unchanged code
stratified_split = d105.stratified_split                # also unchanged
TREES = 100                                             # trees per forest
REPEATS = 10                                            # shuffles per feature
SPLITS = 10                                             # splits in sections 4 to 6
KEEP = [1, 2, 4, 8, 16, 32, 64]                         # section 7: pixels kept


# =============================================================================
# THREE KINDS OF IMPORTANCE -- NEW
# =============================================================================
def tree_importance(tree, p):                           # NEW: impurity removed by each feature in one tree, summing to 1
    out = np.zeros(p); stack = [tree]; n = tree["counts"].sum()  # one total per feature; rows at the root
    while stack:
        node = stack.pop()
        if "feature" in node:                               # a question: credit its feature, weighted by the rows that reach it
            out[node["feature"]] += node["counts"].sum() / n * node["gain"]
            stack += [node["left"], node["right"]]
    return out / max(out.sum(), 1e-12)                    # shares, so a tree with little to remove still counts equally


def impurity_importance(trees, p):                      # NEW: the forest's average
    return np.mean([tree_importance(t, p) for t in trees], axis=0)


def permutation_importance(trees, X, y, k, seed, groups=None):  # NEW: accuracy lost when a column (or group) is shuffled
    rng = np.random.RandomState(seed); base = accuracy(forest_proba(trees, X, k), y)  # the score before any shuffle
    groups = groups if groups is not None else [[j] for j in range(X.shape[1])]  # by default, one feature at a time
    out = np.zeros((len(groups), REPEATS))                # one row per group, one column per shuffle
    for g, cols in enumerate(groups):
        for r in range(REPEATS):
            Z = X.copy(); Z[:, cols] = X[rng.permutation(len(X))][:, cols]  # shuffle these columns together
            out[g, r] = base - accuracy(forest_proba(trees, Z, k), y)
    return out.mean(1), out.std(1)                        # the mean loss, and its spread between shuffles


def ranks(a):                                           # 1 for the smallest value (ties in order)
    r = np.empty(len(a)); r[np.argsort(a, kind="stable")] = np.arange(1, len(a) + 1)
    return r


def spearman(a, b):                                     # NEW: correlation of the rankings
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def titanic_extended():                                 # Titanic plus two columns that cannot matter
    X, y = d106.load_titanic()
    pid = pd.read_csv(d105.TITANIC)["PassengerId"].to_numpy(float)  # a row number, real and meaningless
    coin = np.random.RandomState(0).randint(0, 2, len(y)).astype(float)  # a fair coin per passenger
    return np.column_stack([X, pid, coin]), y, d106.FEATS + ["PassengerId", "coin"]


# =============================================================================
# DEMOS
# =============================================================================
def demo_check(X, y):
    print("=" * 70)
    print("1. CORRECTNESS CHECKS")
    print("=" * 70)
    hand = {"counts": np.array([10, 10]), "feature": 0, "gain": 0.3,  # 20 rows; feature 0 first, then feature 1 on 8 rows
            "left": {"counts": np.array([9, 3])},
            "right": {"counts": np.array([1, 7]), "feature": 1, "gain": 0.1, "left": {"counts": np.array([1, 0])}, "right": {"counts": np.array([0, 7])}}}
    print(f"by hand: feature 0 removes 0.3 at the root (20 of 20 rows), feature 1 removes 0.1 on 8 of 20 rows: 0.3 / 0.34 = {0.3 / 0.34:.4f}, 0.04 / 0.34 = {0.04 / 0.34:.4f}")
    print(f"tree_importance gives {tree_importance(hand, 3).round(4).tolist()} (the third feature was never used)")
    tr, te = stratified_split(y, 0.3, 0); p = X.shape[1]
    ours = tree_importance(d105.grow(X[tr], y[tr], 2, d105.gini), p)
    theirs = DecisionTreeClassifier(random_state=0).fit(X[tr], y[tr]).feature_importances_
    print(f"one tree, breast cancer: largest gap to scikit-learn's feature_importances_ {np.abs(ours - theirs).max():.4f} (tied splits may differ)")
    a = impurity_importance(fit_forest(X[tr], y[tr], 2, 300, 5, 0)[0], p)
    b = RandomForestClassifier(300, max_features="sqrt", random_state=0).fit(X[tr], y[tr]).feature_importances_
    print(f"forest of 300: rank correlation with scikit-learn's forest {spearman(a, b):.3f}; largest gap {np.abs(a - b).max():.4f}")
    print("Actual output from this lesson's run.")


def demo_rankings(X, y, names):
    print("\n" + "=" * 70)
    print("2. BREAST CANCER -- the two rankings side by side")
    print("=" * 70)
    tr, te = stratified_split(y, 0.3, 0); p = X.shape[1]
    trees, _ = fit_forest(X[tr], y[tr], 2, TREES, d107.sqrt_features(p), 0)
    mdi = impurity_importance(trees, p); perm, sd = permutation_importance(trees, X[te], y[te], 2, 0)
    print(f"forest of {TREES}, test accuracy {accuracy(forest_proba(trees, X[te], 2), y[te]):.4f}; permutation on the {len(te)} test tumours, {REPEATS} shuffles each")
    print(f"\n{'feature':<24}{'MDI':>8}{'MDI rank':>10}{'perm':>9}{'+-':>8}{'perm rank':>11}")
    ra, rb = p + 1 - ranks(mdi), p + 1 - ranks(perm)      # 1 = most important
    for j in np.argsort(-mdi)[:10]:
        print(f"{names[j]:<24}{mdi[j]:>8.4f}{int(ra[j]):>10}{perm[j]:>9.4f}{sd[j]:>8.4f}{int(rb[j]):>11}")
    print(f"rank correlation of the two lists: {spearman(mdi, perm):.3f}; features whose shuffle cost no accuracy at all: {int((perm <= 0).sum())} of {p}")
    print(f"largest single loss {perm.max():.4f}, which is {perm.max() * len(te):.1f} tumours of {len(te)}, though the features share a lot of information")
    print("Actual output from this lesson's run.")
    return trees, mdi, perm, te                              # later sections use them


def clusters(X, limit=0.9):                             # NEW: groups of features with |r| above a limit
    r = np.abs(np.corrcoef(X.T)); left = list(range(X.shape[1])); out = []  # correlations; features not yet placed
    while left:
        j = left.pop(0); g = [j] + [i for i in left if r[j, i] > limit]  # the first free feature and its near-copies
        left = [i for i in left if i not in g]; out.append(g)
    return out


def demo_correlated(X, y, names, trees, te):
    print("\n" + "=" * 70)
    print("3. CORRELATED FEATURES -- shuffle one, and its near-copies cover for it")
    print("=" * 70)
    g = clusters(X[stratified_split(y, 0.3, 0)[0]])       # found on the training rows
    big = [c for c in g if len(c) > 1]
    print(f"greedy groups with |r| > 0.9 on the training rows: {len(g)} groups from {X.shape[1]} features, {len(big)} of them with more than one member")
    single = permutation_importance(trees, X[te], y[te], 2, 0)[0]
    whole = permutation_importance(trees, X[te], y[te], 2, 0, groups=g)[0]
    print(f"\n{'largest groups':<56}{'features':>9}{'best single':>12}{'together':>10}")
    for i in np.argsort(-whole)[:5]:
        lab = ", ".join(names[j] for j in g[i]); lab = lab if len(lab) <= 53 else lab[:50] + "..."
        print(f"{lab:<56}{len(g[i]):>9}{single[g[i]].max():>12.4f}{whole[i]:>10.4f}")
    print("Actual output from this lesson's run.")


def demo_useless():
    print("\n" + "=" * 70)
    print("4. USELESS COLUMNS -- a real ID number and a random coin, on Titanic")
    print("=" * 70)
    X, y, names = titanic_extended(); p = X.shape[1]; k = len(names)
    mdi, tr_p, te_p, drop = [], [], [], []                # one row per split
    for s in range(SPLITS):
        tr, te = stratified_split(y, 0.3, s)
        trees, _ = fit_forest(X[tr], y[tr], 2, TREES, 3, s)  # mtry 3 of 8
        base = accuracy(forest_proba(trees, X[te], 2), y[te])
        mdi.append(impurity_importance(trees, p))
        tr_p.append(permutation_importance(trees, X[tr], y[tr], 2, s)[0]); te_p.append(permutation_importance(trees, X[te], y[te], 2, s)[0])
        row = []
        for j in range(p):                                  # retrain without each feature
            keep = [i for i in range(p) if i != j]
            t2, _ = fit_forest(X[tr][:, keep], y[tr], 2, TREES, 3, s)
            row.append(base - accuracy(forest_proba(t2, X[te][:, keep], 2), y[te]))
        drop.append(row)
    mdi, tr_p, te_p, drop = map(np.array, (mdi, tr_p, te_p, drop))
    print(f"{SPLITS} splits, {TREES} trees, mtry 3 of {p}; permutation on training and on test rows; drop-column retrains without the feature")
    print(f"\n{'feature':<13}{'distinct':>9}{'MDI':>8}{'perm train':>12}{'perm test':>11}{'drop-column':>13}{'+- SE':>8}")
    for j in np.argsort(-mdi.mean(0)):
        print(f"{names[j]:<13}{len(np.unique(X[:, j])):>9}{mdi[:, j].mean():>8.4f}{tr_p[:, j].mean():>12.4f}{te_p[:, j].mean():>11.4f}{drop[:, j].mean():>+13.4f}{drop[:, j].std() / np.sqrt(SPLITS):>8.4f}")
    print("Actual output from this lesson's run.")
    return names, mdi.mean(0), te_p.mean(0), drop.mean(0)     # the plot uses them


def demo_duplicate():
    print("\n" + "=" * 70)
    print("5. A DUPLICATED FEATURE -- is_female twice")
    print("=" * 70)
    X, y = d106.load_titanic(); names = d106.FEATS
    Z = np.column_stack([X, X[:, 1]]); labels = names + ["is_female copy"]  # an exact copy, as the last column
    a_m, a_p, b_m, b_p = [], [], [], []
    for s in range(SPLITS):
        tr, te = stratified_split(y, 0.3, s)
        t1, _ = fit_forest(X[tr], y[tr], 2, TREES, 2, s); t2, _ = fit_forest(Z[tr], y[tr], 2, TREES, 2, s)
        a_m.append(impurity_importance(t1, 6)); a_p.append(permutation_importance(t1, X[te], y[te], 2, s)[0])
        b_m.append(impurity_importance(t2, 7)); b_p.append(permutation_importance(t2, Z[te], y[te], 2, s)[0])
    a_m, a_p, b_m, b_p = map(lambda v: np.mean(v, axis=0), (a_m, a_p, b_m, b_p))
    print(f"{SPLITS} splits, {TREES} trees, mtry 2")
    print(f"{'':<16}{'MDI':>8}{'perm test':>11}")
    print(f"{'alone':<16}{a_m[1]:>8.4f}{a_p[1]:>11.4f}")
    print(f"{'with a copy':<16}{b_m[1]:>8.4f}{b_p[1]:>11.4f}")
    print(f"{'the copy':<16}{b_m[6]:>8.4f}{b_p[6]:>11.4f}")
    print(f"{'the two added':<16}{b_m[1] + b_m[6]:>8.4f}{b_p[1] + b_p[6]:>11.4f}")
    print("Actual output from this lesson's run.")


def demo_stability(X, y):
    print("\n" + "=" * 70)
    print("6. STABILITY -- the same question on 10 splits")
    print("=" * 70)
    p = X.shape[1]; M, Q = [], []
    for s in range(SPLITS):
        tr, te = stratified_split(y, 0.3, s)
        trees, _ = fit_forest(X[tr], y[tr], 2, TREES, d107.sqrt_features(p), s)
        M.append(impurity_importance(trees, p)); Q.append(permutation_importance(trees, X[te], y[te], 2, s)[0])
    pairs = [(a, b) for a in range(SPLITS) for b in range(a + 1, SPLITS)]  # every pair of splits
    for name, V in (("MDI", M), ("permutation", Q)):
        top = [set(np.argsort(-v)[:5].tolist()) for v in V]
        print(f"{name:<12}: rank correlation between splits {np.mean([spearman(V[a], V[b]) for a, b in pairs]):.3f}; top-5 lists share {np.mean([len(top[a] & top[b]) for a, b in pairs]):.2f} of 5 features on average")
    print("Actual output from this lesson's run.")


def demo_digits():
    print("\n" + "=" * 70)
    print("7. USING IT -- digits: how few pixels does a forest need?")
    print("=" * 70)
    X, y = d107.data()["Digits"]; p = X.shape[1]; rows = {"MDI ranking": [], "random pixels": []}
    maps = []
    print(f"{len(y)} images of 8x8 pixels, 10 classes; 3 splits; {TREES // 2} trees; every ranking made on training rows only, then a new forest is trained")
    for s in range(3):
        tr, te = stratified_split(y, 0.3, s)
        t, _ = fit_forest(X[tr], y[tr], 10, TREES // 2, d107.sqrt_features(p), s)
        imp = impurity_importance(t, p); maps.append(imp); order = np.argsort(-imp)
        rand = np.random.RandomState(s).permutation(p)
        for name, o in (("MDI ranking", order), ("random pixels", rand)):
            r = []
            for k in KEEP:
                c = o[:k]; f, _ = fit_forest(X[tr][:, c], y[tr], 10, TREES // 2, d107.sqrt_features(k), s)
                r.append(accuracy(forest_proba(f, X[te][:, c], 10), y[te]))
            rows[name].append(r)
    print(f"\n{'pixels kept':>12}" + "".join(f"{k:>8}" for k in KEEP))
    means = {n: np.mean(r, axis=0) for n, r in rows.items()}
    for n, m in means.items():
        print(f"{n:<14}" + "".join(f"{v:>8.4f}" for v in m)[2:])
    mp = np.mean(maps, axis=0)
    print(f"pixels never used by any tree: {int((mp == 0).sum())} of {p}; those pixels are non-blank in {np.mean(X[:, mp == 0] > 0):.4f} of images, the others in {np.mean(X[:, mp > 0] > 0):.4f}")
    print("Actual output from this lesson's run.")
    return mp, means                                         # the plot uses them


def plot_results(names, mdi, perm, useless, dmap, curves):  # four panels
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.2))             # canvas
    o = np.argsort(-mdi)[:10]; y = np.arange(len(o)); a = ax[0, 0]
    a.barh(y - 0.2, mdi[o] / mdi.max(), 0.4, label="MDI"); a.barh(y + 0.2, np.maximum(perm[o], 0) / perm.max(), 0.4, label="permutation")
    a.set_yticks(y); a.set_yticklabels([names[j] for j in o], fontsize=8); a.invert_yaxis(); a.set_xlabel("relative to each method's largest")
    a.set_title("Breast cancer: the same forest, two rankings", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis="x")
    n, m, q, d = useless; o = np.argsort(-m); y = np.arange(len(o)); a = ax[0, 1]
    a.barh(y - 0.27, m[o], 0.27, label="MDI"); a.barh(y, q[o], 0.27, label="permutation (test)"); a.barh(y + 0.27, d[o], 0.27, label="drop-column")
    a.set_yticks(y); a.set_yticklabels([n[j] for j in o], fontsize=8); a.invert_yaxis(); a.set_xlabel("importance (each in its own units)")
    a.set_title("Titanic with two useless columns", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis="x")
    a = ax[1, 0]; im = a.imshow(dmap.reshape(8, 8), cmap="viridis"); plt.colorbar(im, ax=a, fraction=0.046); a.set_xticks([]); a.set_yticks([])
    a.set_title("Digits: MDI per pixel", fontsize=10)
    a = ax[1, 1]
    for k, v in curves.items():
        a.plot(KEEP, v, marker="o", label=k)
    a.set_xscale("log", base=2); a.set_xlabel("pixels kept (log scale)"); a.set_ylabel("test accuracy, new forest")
    a.set_title("Digits: keeping only the top pixels", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("importance_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved importance_results.png")


def draw_diagram():                                     # the two procedures side by side
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 108: two ways to ask a forest what mattered", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.2, 3.4, "impurity (MDI)", fontsize=10, fontweight="bold", va="center"); ax.text(0.2, 1.3, "permutation", fontsize=10, fontweight="bold", va="center")
    box(3.6, 3.4, 2.5, 1.1, "walk every tree's\nquestions", "#FAEEDA")
    box(7.0, 3.4, 2.9, 1.1, "credit each feature with\nthe impurity it removed", "#DCEBF7")
    box(10.8, 3.4, 3.4, 1.1, "cheap, but measured on\ntraining rows", "#E5F2E1")
    box(3.6, 1.3, 2.5, 1.1, "held-out rows:\nshuffle one column", "#FAEEDA")
    box(7.0, 1.3, 2.9, 1.1, "predict again\nwith the same forest", "#DCEBF7")
    box(10.8, 1.3, 3.4, 1.1, "accuracy lost = importance;\nblind where features overlap", "#E5F2E1")
    for y in (3.4, 1.3):
        ax.annotate("", xy=(5.51, y), xytext=(4.89, y), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(9.06, y), xytext=(8.49, y), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("importance_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved importance_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 109 is gradient boosting from scratch: trees grown one after another, each one correcting the errors of those before it.")


def main():                                             # runs every section in order
    sets = d107.data(); X, y = sets["Breast cancer"]       # the three datasets
    from sklearn.datasets import load_breast_cancer       # for the feature names only
    names = list(load_breast_cancer().feature_names)
    demo_check(X, y)                                       # 1. correctness
    trees, mdi, perm, te = demo_rankings(X, y, names)      # 2. rankings
    demo_correlated(X, y, names, trees, te)                # 3. correlated features
    useless = demo_useless()                               # 4. useless columns
    demo_duplicate()                                       # 5. a duplicate
    demo_stability(X, y)                                   # 6. stability
    dmap, curves = demo_digits()                           # 7. using it
    plot_results(names, mdi, perm, useless, dmap, curves)  # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 109


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
