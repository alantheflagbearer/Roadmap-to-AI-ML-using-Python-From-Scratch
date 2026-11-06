"""
Day 120: Hierarchical Clustering From Scratch -- a Real Dendrogram

k-means needs k in advance. Agglomerative clustering does not: start with every
point as its own cluster and repeatedly merge the two closest clusters until one
is left. The full history of merges is a tree, the dendrogram; cutting it at any
height gives a clustering. Everything depends on what 'closest clusters' means:
the linkage.

Everything is written from scratch in NumPy:
  * agglomerative clustering with single, complete, average and Ward linkage,
    using the Lance-Williams update (new distances from old ones, no recomputing)
  * cutting the tree into k clusters, and the cophenetic correlation (how well the
    tree's merge heights preserve the original distances)
  * the dendrogram itself, laid out and drawn by hand

The data, with labels used only to judge: iris, wine (standardised) and
handwritten digits.

  1. By hand: five points.
  2. Correctness: against SciPy's linkage.
  3. Four linkages on three datasets.
  4. Single linkage's chaining.
  5. Where to cut: the largest gap in merge heights.
  6. Ward against k-means (Day 119).
  7. The cost of a whole tree.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lesson quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 119
import time                                            # time: the cost of a full tree

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from scipy.cluster.hierarchy import cophenet, linkage  # used ONLY as an outside check
from sklearn.datasets import load_digits, load_iris, load_wine  # bundled datasets
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 119 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day119_k_means as d119                       # Day 119: sqdist, kmeans, ari, standardise
sqdist, ari, standardise = d119.sqdist, d119.ari, d119.standardise  # unchanged
LINKS = ["single", "complete", "average", "ward"]        # the four linkages


# =============================================================================
# AGGLOMERATIVE CLUSTERING -- NEW
# =============================================================================
def agglomerate(X, method):                             # NEW: the merge history, in SciPy's format: (a, b, height, size) per merge
    n = len(X); D = np.sqrt(np.maximum(sqdist(X, X), 0))  # distances between single points
    if method == "ward":
        D = D ** 2                                          # Ward's update works on squared distances
    D[np.arange(n), np.arange(n)] = np.inf                # a cluster is never merged with itself
    size = np.ones(n); ids = np.arange(n); alive = np.ones(n, bool); Z = []
    for step in range(n - 1):
        M = np.where(alive[:, None] & alive[None, :], D, np.inf)  # only clusters still in play
        i, j = np.unravel_index(np.argmin(M), M.shape); i, j = min(i, j), max(i, j)
        d = M[i, j]; ni, nj = size[i], size[j]
        Z.append((min(ids[i], ids[j]), max(ids[i], ids[j]), np.sqrt(d) if method == "ward" else d, ni + nj))
        nk = size                                               # the sizes of every other cluster
        if method == "single":
            new = np.minimum(D[i], D[j])                          # Lance-Williams: the nearer of the two
        elif method == "complete":
            new = np.maximum(D[i], D[j])                          # the further
        elif method == "average":
            new = (ni * D[i] + nj * D[j]) / (ni + nj)             # the size-weighted mean
        else:
            new = ((ni + nk) * D[i] + (nj + nk) * D[j] - nk * D[i, j]) / (ni + nj + nk)  # Ward: growth in squared error
        D[i, :] = new; D[:, i] = new; D[i, i] = np.inf          # the merged cluster takes row i
        alive[j] = False; size[i] = ni + nj; ids[i] = n + step  # row j retires; the new cluster gets the next id
    return np.array(Z)


def cut(Z, n, k):                                       # NEW: labels from undoing the last k - 1 merges
    parent = np.arange(2 * n - 1)                          # every node points to itself at first
    for s, (a, b, _, _) in enumerate(Z[:n - k]):           # replay the first n - k merges
        parent[int(a)] = parent[int(b)] = n + s
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    r = np.array([root(i) for i in range(n)])
    return np.unique(r, return_inverse=True)[1]           # renumber 0 .. k - 1


def cophenetic(Z, n):                                   # NEW: for every pair of points, the height at which they first share a cluster
    members = {i: [i] for i in range(n)}; C = np.zeros((n, n))
    for s, (a, b, h, _) in enumerate(Z):
        A, B = members.pop(int(a)), members.pop(int(b))
        C[np.ix_(A, B)] = h; C[np.ix_(B, A)] = h            # every cross pair meets here
        members[n + s] = A + B
    return C


def coph_corr(X, Z):                                    # NEW: correlation between original distances and merge heights
    n = len(X); D = np.sqrt(np.maximum(sqdist(X, X), 0)); C = cophenetic(Z, n); iu = np.triu_indices(n, 1)
    return float(np.corrcoef(D[iu], C[iu])[0, 1])


def leaf_order(Z, n):                                   # NEW: the left-to-right order of the points under the tree
    kids = {n + s: (int(a), int(b)) for s, (a, b, _, _) in enumerate(Z)}
    def walk(x):
        return [x] if x < n else walk(kids[x][0]) + walk(kids[x][1])
    return walk(2 * n - 2)


def draw_dendrogram(ax, Z, n, labels=None, colour_k=None):  # NEW: each merge as an upside-down U above its two children
    order = leaf_order(Z, n); xpos = {leaf: i for i, leaf in enumerate(order)}; ypos = {i: 0.0 for i in range(n)}
    groups = cut(Z, n, colour_k) if colour_k else None; cmap = plt.get_cmap("tab10")
    for s, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b); xa, xb = xpos[a], xpos[b]
        same = None
        if groups is not None:
            ga = {groups[l] for l in (leaf_order_sub(Z, n, a))}; gb = {groups[l] for l in leaf_order_sub(Z, n, b)}
            same = ga.pop() if len(ga) == 1 and ga == gb else None
        c = cmap(same % 10) if same is not None else "#555555"
        ax.plot([xa, xa, xb, xb], [ypos[a], h, h, ypos[b]], color=c, linewidth=0.8)
        xpos[n + s] = (xa + xb) / 2; ypos[n + s] = h
    ax.set_xticks([]); ax.set_xlim(-1, n)


_SUB = {}


def leaf_order_sub(Z, n, node):                         # the leaves under any node (cached per tree)
    key = (id(Z), node)
    if key not in _SUB:
        if node < n:
            _SUB[key] = [node]
        else:
            a, b = Z[node - n][:2]; _SUB[key] = leaf_order_sub(Z, n, int(a)) + leaf_order_sub(Z, n, int(b))
    return _SUB[key]


def datasets():                                         # three labelled datasets
    ir, wi, dg = load_iris(), load_wine(), load_digits()
    return {"iris": (standardise(ir.data), ir.target), "wine": (standardise(wi.data), wi.target), "digits": (dg.data, dg.target)}


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- five points")
    print("=" * 70)
    X = np.array([[0.0], [1.0], [3.0], [7.0], [8.5]])     # five points on a line
    print("points 0, 1, 3, 7, 8.5; each row of the history is (cluster, cluster, height, size); clusters 5, 6, ... are merges")
    for m in ("single", "complete"):
        Z = agglomerate(X, m)
        print(f"  {m:<9}: " + "; ".join(f"({int(a)}, {int(b)}) at {h:.2f}" for a, b, h, _ in Z))
    print("by hand, single: 0-1 at 1; then {0,1}-3 at 2 (nearest pair 1-3); 7-8.5 at 1.5 comes before that; the two groups at 4 (3 to 7)")
    print("complete uses the FURTHEST pair: {0,1}-3 at 3 (0 to 3), and the last merge at 8.5 (0 to 8.5)")
    print("Actual output from this lesson's run.")


def demo_check(sets):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- against SciPy's linkage")
    print("=" * 70)
    X, _ = sets["iris"]
    for m in LINKS:
        ours = agglomerate(X, m); theirs = linkage(X, m)
        print(f"  {m:<9}: merge heights, largest gap {np.abs(np.sort(ours[:, 2]) - np.sort(theirs[:, 2])).max():.1e}; "
              f"cophenetic correlation {coph_corr(X, ours):.6f} against SciPy's {cophenet(theirs, np.sqrt(np.maximum(sqdist(X, X), 0))[np.triu_indices(len(X), 1)])[0]:.6f}")
    print("(heights are compared in sorted order: when two merges tie, the two versions may take them in a different order)")
    D = np.sqrt(np.maximum(sqdist(X, X), 0))[np.triu_indices(len(X), 1)]
    print(f"iris has {len(D) - len(np.unique(np.round(D, 10)))} tied distances (iris measurements are rounded to 0.1 cm); with ties, complete linkage can build a different tree")
    Xj = X + np.random.RandomState(0).normal(scale=1e-6, size=X.shape)  # a tiny jitter breaks every tie
    print("the same check after adding noise of size 1e-6 to break the ties: largest gap " + ", ".join(
        f"{m} {np.abs(np.sort(agglomerate(Xj, m)[:, 2]) - np.sort(linkage(Xj, m)[:, 2])).max():.1e}" for m in LINKS))
    print("Actual output from this lesson's run.")


def demo_links(sets):
    print("\n" + "=" * 70)
    print("3. FOUR LINKAGES ON THREE DATASETS -- each tree cut into the true number of groups")
    print("=" * 70)
    print(f"{'data':<8}{'linkage':<10}{'ARI':>8}{'cluster sizes':>40}{'cophenetic':>12}")
    out = {}
    for name, (X, y) in sets.items():
        k = len(np.unique(y))
        for m in LINKS:
            Z = agglomerate(X, m); lab = cut(Z, len(X), k); out[(name, m)] = (Z, ari(lab, y))
            sizes = sorted(np.bincount(lab).tolist(), reverse=True); s = str(sizes if len(sizes) <= 4 else sizes[:4] + ["..."])
            print(f"{name:<8}{m:<10}{ari(lab, y):>8.4f}{s:>40}{coph_corr(X, Z):>12.4f}")
    print("Actual output from this lesson's run.")
    return out


def demo_chaining(sets):
    print("\n" + "=" * 70)
    print("4. SINGLE LINKAGE'S CHAINING -- digits")
    print("=" * 70)
    X, y = sets["digits"]; Z = agglomerate(X, "single")
    for k in (2, 10, 50):
        lab = cut(Z, len(X), k); s = np.bincount(lab)
        print(f"  cut into {k:>2} clusters: the largest holds {s.max()} of {len(X)} images; clusters of a single image: {int((s == 1).sum())}")
    print("single linkage merges through any chain of near neighbours, so one cluster swallows almost everything and outliers stay alone")
    print("Actual output from this lesson's run.")


def demo_gap(sets):
    print("\n" + "=" * 70)
    print("5. WHERE TO CUT -- the largest gap between successive merge heights, Ward linkage")
    print("=" * 70)
    for name, (X, y) in sets.items():
        Z = agglomerate(X, "ward"); h = Z[:, 2]; gaps = np.diff(h)[-15:]  # gaps among the last 15 merges
        k = int(len(gaps) - np.argmax(gaps) + 1); lab = cut(Z, len(X), k)  # cutting in the biggest gap undoes every merge above it
        print(f"  {name:<7}: last five merge heights {np.round(h[-5:], 2).tolist()}; the biggest gap suggests k = {k} (true {len(np.unique(y))}); ARI {ari(lab, y):.4f}")
    print("Actual output from this lesson's run.")


def demo_vs_kmeans(sets):
    print("\n" + "=" * 70)
    print("6. WARD AGAINST k-MEANS (Day 119) -- both minimise within-cluster squared distance")
    print("=" * 70)
    for name, (X, y) in sets.items():
        k = len(np.unique(y)); wl = cut(agglomerate(X, "ward"), len(X), k); km = d119.kmeans(X, k, 0)[0]
        inertia = lambda lab: float(sum(((X[lab == c] - X[lab == c].mean(0)) ** 2).sum() for c in np.unique(lab)))
        print(f"  {name:<7}: inertia Ward {inertia(wl):.1f}, k-means {inertia(km):.1f}; ARI with truth Ward {ari(wl, y):.4f}, k-means {ari(km, y):.4f}; "
              f"agreement between the two {ari(wl, km):.4f}")
    print("Ward builds greedily, merge by merge, and never revisits a merge; k-means keeps moving points, so it usually reaches a lower inertia for one k")
    print("Actual output from this lesson's run.")


def demo_cost(sets):
    print("\n" + "=" * 70)
    print("7. THE COST OF A WHOLE TREE")
    print("=" * 70)
    X, _ = sets["digits"]
    for n in (200, 400, 800, 1600):
        t0 = time.perf_counter(); agglomerate(X[:n], "ward"); t = time.perf_counter() - t0
        print(f"  {n:>5} images: {t:.2f} s, and a distance table of {n * n * 8 / 1e6:.1f} MB")
    print("this simple version searches the whole table at every merge: time grows with n cubed, memory with n squared (within this run)")
    print("Actual output from this lesson's run.")


def plot_results(sets, links):                          # a real dendrogram, and linkages compared
    fig = plt.figure(figsize=(16, 8.6)); G = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    X, y = sets["iris"]; Z = links[("iris", "ward")][0]
    a = fig.add_subplot(G[0, :]); draw_dendrogram(a, Z, len(X), colour_k=3)
    order = leaf_order(Z, len(X)); cm = plt.get_cmap("Set1")
    a.scatter(range(len(X)), np.full(len(X), -0.6), c=[cm(y[i]) for i in order], s=10, marker="s")  # the true species under each leaf
    a.set_ylabel("merge height (Ward)"); a.set_title("Iris, Ward linkage: the dendrogram (colours: cut into 3; squares below: the true species)", fontsize=10)
    a = fig.add_subplot(G[1, 0]); names = list(sets); x = np.arange(len(names)); w = 0.2
    for i, m in enumerate(LINKS):
        a.bar(x + (i - 1.5) * w, [links[(n, m)][1] for n in names], w, label=m)
    a.set_xticks(x); a.set_xticklabels(names); a.set_ylabel("ARI with the true groups"); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    a.set_title("Four linkages, cut at the true number of groups", fontsize=10)
    a = fig.add_subplot(G[1, 1]); X, y = sets["wine"]; Z = links[("wine", "single")][0]; draw_dendrogram(a, Z, len(X), colour_k=3)
    a.set_ylabel("merge height (single)"); a.set_title("Wine, single linkage: chaining (cut into 3)", fontsize=10)
    plt.tight_layout(); plt.savefig("hierarchical_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved hierarchical_results.png")


def draw_diagram():                                     # merging, and the four linkages
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 120: agglomerative clustering", ha="center", fontsize=12, fontweight="bold")
    box(1.8, 2.5, 2.8, 1.6, "every point starts\nas its own cluster", "#FAEEDA")
    box(5.4, 2.5, 3.2, 1.6, "merge the two\nclosest clusters;\nrecord the height", "#DCEBF7")
    box(9.0, 2.5, 3.0, 1.6, "update distances to the\nnew cluster\n(Lance-Williams)", "#E5F2E1")
    box(11.9, 2.5, 1.8, 1.6, "until one\ncluster:\nthe tree", "#EEEDFE")
    for x0, x1 in ((3.24, 3.76), (7.04, 7.46), (10.54, 10.96)):
        ax.annotate("", xy=(x1, 2.5), xytext=(x0, 2.5), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 0.9, "'closest clusters': single = nearest pair, complete = furthest pair, average = mean pair, Ward = least growth in squared error", ha="center", fontsize=8.5)
    plt.savefig("hierarchical_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved hierarchical_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 121 is DBSCAN from scratch: clusters as dense regions of any shape, with points in sparse regions left as noise.")


def main():                                             # runs every section in order
    sets = datasets()                                      # three labelled datasets
    demo_hand()                                            # 1. by hand
    demo_check(sets)                                       # 2. correctness
    links = demo_links(sets)                               # 3. four linkages
    demo_chaining(sets)                                    # 4. chaining
    demo_gap(sets)                                         # 5. where to cut
    demo_vs_kmeans(sets)                                   # 6. Ward against k-means
    demo_cost(sets)                                        # 7. cost
    plot_results(sets, links)                              # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 121


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
