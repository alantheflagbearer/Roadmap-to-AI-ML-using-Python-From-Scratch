"""
Day 119: k-Means From Scratch -- a Real Elbow-Method Demonstration

Week 18 turns to unsupervised learning: finding structure in data with no
labels at all. k-means is the classic: choose k centres, give every point to
its nearest centre, move each centre to the mean of its points, and repeat.
The famous difficulty is choosing k. The 'elbow method' plots the total squared
distance against k and looks for a bend. Today tests whether that bend is there
on real data whose true groups are known.

Everything is written from scratch in NumPy:
  * Lloyd's algorithm, with random and k-means++ starts, and restarts
  * inertia (total squared distance to the centres), the silhouette score, and
    the adjusted Rand index (agreement with known labels, corrected for chance)
  * an automatic 'elbow' finder: the k furthest from the line joining the ends of the curve

The data, with labels kept aside and used only to judge the clusters:
  * iris (3 species), wine (3 cultivars), handwritten digits (10 digits)

  1. By hand: six points, two centres.
  2. Correctness: every step lowers the inertia, and scikit-learn agrees.
  3. Starting points: random against k-means++.
  4. The elbow, on three real datasets.
  5. Scaling: one feature measured in the hundreds.
  6. Where k-means fails: shapes it cannot represent.
  7. What the clusters look like: digits.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.cluster import KMeans                     # used ONLY as an outside check
from sklearn.datasets import load_digits, load_iris, load_wine  # bundled datasets
from sklearn.metrics import adjusted_rand_score, silhouette_score  # used ONLY as an outside check
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

KS = list(range(1, 11))                                 # section 4: numbers of clusters
SEEDS = 20                                              # section 3: random starts compared
RESTARTS = 10                                           # restarts per fit (the best inertia is kept)


# =============================================================================
# k-MEANS -- NEW
# =============================================================================
def sqdist(A, B):                                       # every squared distance (Day 113's)
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def init_centres(X, k, rng, method):                    # NEW: random rows, or k-means++
    if method == "random":
        return X[rng.choice(len(X), k, replace=False)].copy()
    C = [X[rng.randint(len(X))]]                          # k-means++: the first centre at random ...
    for _ in range(k - 1):
        d = np.maximum(sqdist(X, np.array(C)).min(1), 0)    # ... then each next one far from those chosen,
        C.append(X[rng.choice(len(X), p=d / d.sum())])      # with probability proportional to squared distance
    return np.array(C)


def lloyd(X, C, max_iter=300):                          # NEW: assign, update, repeat; returns labels, centres, inertia per step
    history = []
    for _ in range(max_iter):
        D = sqdist(X, C); lab = D.argmin(1); history.append(float(np.maximum(D.min(1), 0).sum()))  # nearest centre, and the cost
        new = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(len(C))])  # an empty cluster keeps its centre
        if np.allclose(new, C):
            break
        C = new
    return lab, C, history


def kmeans(X, k, seed, method="k-means++", restarts=RESTARTS):  # NEW: the best of several starts
    rng = np.random.RandomState(seed); best = None
    for _ in range(restarts):
        lab, C, h = lloyd(X, init_centres(X, k, rng, method))
        if best is None or h[-1] < best[2][-1]:
            best = (lab, C, h)
    return best


def silhouette(X, lab):                                 # NEW: (b - a) / max(a, b) per point, averaged
    D = np.sqrt(np.maximum(sqdist(X, X), 0)); s = np.zeros(len(X)); ks = np.unique(lab)
    for i in range(len(X)):
        same = lab == lab[i]; n = same.sum() - 1
        if n == 0:
            continue                                        # a point alone in its cluster scores 0
        a = D[i, same].sum() / n                            # mean distance to its own cluster
        b = min(D[i, lab == c].mean() for c in ks if c != lab[i])  # to the nearest other cluster
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def ari(a, b):                                          # NEW: adjusted Rand index: pair agreement, corrected for chance
    _, ia = np.unique(a, return_inverse=True); _, ib = np.unique(b, return_inverse=True)
    T = np.zeros((ia.max() + 1, ib.max() + 1)); np.add.at(T, (ia, ib), 1)  # the contingency table
    c2 = lambda x: x * (x - 1) / 2                        # pairs among x items
    s = c2(T).sum(); sa = c2(T.sum(1)).sum(); sb = c2(T.sum(0)).sum(); e = sa * sb / c2(len(a))
    return float((s - e) / (0.5 * (sa + sb) - e))


def elbow(ks, inertia):                                 # NEW: the k furthest below the straight line from the first point to the last
    x = np.array(ks, float); y = np.array(inertia, float)
    x, y = (x - x[0]) / (x[-1] - x[0]), (y - y[-1]) / (y[0] - y[-1])  # both axes to 0..1
    return ks[int(np.argmax((1 - x) - y))]                # distance below the chord y = 1 - x


def standardise(X):                                     # every feature to mean 0, spread 1
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def datasets():                                         # three real datasets, with their true groups
    ir, wi, dg = load_iris(), load_wine(), load_digits()
    return {"iris": (standardise(ir.data), ir.target), "wine": (standardise(wi.data), wi.target), "digits": (dg.data, dg.target)}


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- six points, two centres")
    print("=" * 70)
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]); C = np.array([[1.0], [2.0]])  # a poor start: both centres on the left
    print("points 1, 2, 3, 10, 11, 12 on a line; centres start at 1 and 2")
    for step in range(3):
        lab = sqdist(X, C).argmin(1); cost = float(sqdist(X, C).min(1).sum())
        print(f"  step {step + 1}: groups {[X[lab == j, 0].tolist() for j in range(2)]}, inertia {cost:.2f}; new centres {[round(float(X[lab == j, 0].mean()), 3) for j in range(2)]}")
        C = np.array([[X[lab == j, 0].mean()] for j in range(2)])
    print("by hand: the best split is {1, 2, 3} and {10, 11, 12}, centres 2 and 11, inertia 2 + 2 = 4")
    print("Actual output from this lesson's run.")


def demo_check(sets):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- every step lowers the inertia, and scikit-learn agrees")
    print("=" * 70)
    X, y = sets["digits"]; up = 0; steps = 0
    for s in range(SEEDS):
        _, _, h = lloyd(X, init_centres(X, 10, np.random.RandomState(s), "random")); steps += len(h) - 1
        up += int(np.sum(np.diff(h) > 1e-9))
    print(f"digits, k = 10, {SEEDS} random starts: {steps} steps in all; steps where the inertia went up: {up}")
    rng = np.random.RandomState(0); C0 = init_centres(X, 10, rng, "k-means++")
    lab, C, h = lloyd(X, C0.copy()); sk = KMeans(10, init=C0, n_init=1, algorithm="lloyd", tol=0, max_iter=300).fit(X)
    print(f"from the same starting centres: inertia {h[-1]:.2f} here, {sk.inertia_:.2f} in scikit-learn; adjusted Rand index between the two labellings {ari(lab, sk.labels_):.4f}")
    D0 = ((X[:, None, :] - C0[None]) ** 2).sum(2); two = np.sort(D0, 1)[:, :2]; tied = np.where(two[:, 0] == two[:, 1])[0]  # exact distances at step 1
    print(f"why they differ: at step 1, {len(tied)} image(s) lie exactly as far from two centres (image {tied[0]}: squared distance {two[tied[0], 0]:.0f} to both); pixels are whole numbers, so exact ties happen, and rounding decides")
    m = X.mean(0); lab_c, _, h_c = lloyd(X - m, C0 - m)    # scikit-learn subtracts the mean before computing distances
    print(f"the same steps on mean-centred data, as scikit-learn computes them: inertia {h_c[-1]:.2f}; ARI with scikit-learn's labelling {ari(lab_c, sk.labels_):.4f}")
    print(f"silhouette {silhouette(X, lab):.6f} against scikit-learn's {silhouette_score(X, lab):.6f}; ARI with the true digits {ari(lab, y):.6f} against {adjusted_rand_score(y, lab):.6f}")
    print("Actual output from this lesson's run.")


def demo_init(sets):
    print("\n" + "=" * 70)
    print(f"3. STARTING POINTS -- random against k-means++, digits, k = 10, {SEEDS} single starts each")
    print("=" * 70)
    X, y = sets["digits"]; res = {}
    for method in ("random", "k-means++"):
        rows = [kmeans(X, 10, s, method, restarts=1) for s in range(SEEDS)]
        inert = np.array([r[2][-1] for r in rows]); aris = np.array([ari(r[0], y) for r in rows]); steps = np.array([len(r[2]) for r in rows])
        res[method] = inert
        print(f"  {method:<10}: inertia {inert.min():.0f} to {inert.max():.0f} (mean {inert.mean():.0f}); ARI with the digits {aris.min():.4f} to {aris.max():.4f}; steps {steps.mean():.1f}")
    best = kmeans(X, 10, 0); print(f"  the best of {RESTARTS} k-means++ starts: inertia {best[2][-1]:.0f}, ARI {ari(best[0], y):.4f}")
    print("every start ends in a local minimum; different starts end in different ones, which is why restarts are standard")
    print("Actual output from this lesson's run.")
    return res


def demo_elbow(sets):
    print("\n" + "=" * 70)
    print("4. THE ELBOW, ON THREE REAL DATASETS")
    print("=" * 70)
    print(f"k-means++ with {RESTARTS} restarts for every k; the true number of groups is known (3, 3 and 10) but never shown to k-means")
    out = {}
    for name, (X, y) in sets.items():
        I, S, A = [], [], []
        for k in KS:
            lab, C, h = kmeans(X, k, 0); I.append(h[-1]); A.append(ari(lab, y)); S.append(silhouette(X, lab) if k > 1 else float("nan"))
        out[name] = (I, S, A)
        print(f"\n{name}: {len(y)} rows, {X.shape[1]} features, {len(np.unique(y))} true groups")
        print(f"  {'k':>4}{'inertia':>12}{'drop':>9}{'silhouette':>12}{'ARI':>8}")
        for i, k in enumerate(KS):
            drop = f"{1 - I[i] / I[i - 1]:>9.3f}" if i else f"{'':>9}"
            print(f"  {k:>4}{I[i]:>12.1f}{drop}{S[i]:>12.4f}{A[i]:>8.4f}")
        print(f"  the automatic elbow: k = {elbow(KS, I)}; best silhouette: k = {KS[int(np.nanargmax(S))]}; best agreement with the truth: k = {KS[int(np.argmax(A))]}")
    print("Actual output from this lesson's run.")
    return out


def demo_scaling():
    print("\n" + "=" * 70)
    print("5. SCALING -- wine, where one feature is measured in the hundreds")
    print("=" * 70)
    w = load_wine(); X, y = w.data, w.target; sd = X.std(0)
    print(f"'{w.feature_names[int(np.argmax(sd))]}' has spread {sd.max():.1f}; the next largest, '{w.feature_names[int(np.argsort(sd)[-2])]}', {np.sort(sd)[-2]:.1f}; the smallest {sd.min():.3f}")
    for name, Z in (("raw", X), ("standardised", standardise(X))):
        lab, C, h = kmeans(Z, 3, 0); print(f"  {name:<13}: ARI with the 3 cultivars {ari(lab, y):.4f}")
    print("Actual output from this lesson's run.")


def demo_shapes():
    print("\n" + "=" * 70)
    print("6. WHERE k-MEANS FAILS -- shapes it cannot represent (synthetic on purpose: the true groups must be known exactly)")
    print("=" * 70)
    rng = np.random.RandomState(0); t = rng.uniform(0, np.pi, 300)
    moons = np.r_[np.c_[np.cos(t[:150]), np.sin(t[:150])], np.c_[1 - np.cos(t[150:]), 0.5 - np.sin(t[150:])]] + rng.normal(0, 0.08, (300, 2))
    ym = np.r_[np.zeros(150, int), np.ones(150, int)]
    big = rng.normal(0, 2.0, (450, 2)); small = rng.normal([5, 0], 0.4, (50, 2)); Xs = np.r_[big, small]; ys = np.r_[np.zeros(450, int), np.ones(50, int)]
    for name, X, y in (("two interleaved moons", moons, ym), ("a wide cluster beside a small tight one", Xs, ys)):
        lab, C, h = kmeans(X, 2, 0); print(f"  {name:<40}: ARI {ari(lab, y):.4f}; cluster sizes {np.bincount(lab).tolist()} (true {np.bincount(y).tolist()})")
    print("k-means draws straight boundaries halfway between centres, so it cuts curved groups and splits big groups to balance distances")
    print("Actual output from this lesson's run.")
    return moons, kmeans(moons, 2, 0)[0]


def demo_digits(sets):
    print("\n" + "=" * 70)
    print("7. WHAT THE CLUSTERS LOOK LIKE -- digits, k = 10")
    print("=" * 70)
    X, y = sets["digits"]; lab, C, h = kmeans(X, 10, 0)
    print(f"{'cluster':>8}{'size':>6}{'most common digit':>19}{'share':>8}")
    for j in np.argsort([-np.bincount(y[lab == j], minlength=10).max() / max((lab == j).sum(), 1) for j in range(10)]):
        c = np.bincount(y[lab == j], minlength=10); print(f"{j:>8}{c.sum():>6}{int(c.argmax()):>19}{c.max() / c.sum():>8.3f}")
    print(f"labelling each cluster by its most common digit gives {np.mean(np.array([np.bincount(y[lab == j], minlength=10).argmax() for j in range(10)])[lab] == y):.4f} of images right, with no labels used to form the clusters")
    print("Actual output from this lesson's run.")
    return C                                                 # the plot shows the centres


def plot_results(elb, init, moons, mlab, C):            # five panels: elbows | scores | starts | moons | the digit centres
    fig = plt.figure(figsize=(16, 8.4)); G = fig.add_gridspec(2, 3)  # a 2 x 3 grid of panels
    a = fig.add_subplot(G[0, 0])
    for name, (I, S, A) in elb.items():
        a.plot(KS, np.array(I) / I[0], marker="o", label=name)
    a.set_xlabel("k"); a.set_ylabel("inertia, as a share of k = 1"); a.set_title("The elbow curves", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = fig.add_subplot(G[0, 1])
    for name, (I, S, A) in elb.items():
        a.plot(KS[1:], S[1:], marker="o", label=f"{name}: silhouette"); a.plot(KS, A, marker="s", linestyle="--", label=f"{name}: ARI with truth")
    a.set_xlabel("k"); a.set_ylabel("score"); a.set_title("Silhouette, and agreement with the true groups", fontsize=10); a.legend(fontsize=7); a.grid(alpha=0.3)
    a = fig.add_subplot(G[0, 2]); a.boxplot([init["random"], init["k-means++"]], tick_labels=["random starts", "k-means++ starts"])
    a.set_ylabel("final inertia"); a.set_title("Digits: where single starts end", fontsize=10); a.grid(alpha=0.3, axis="y")
    a = fig.add_subplot(G[1, 0]); a.scatter(moons[:, 0], moons[:, 1], c=mlab, cmap="coolwarm", s=8)
    a.set_title("Two moons: k-means cuts straight across", fontsize=10); a.set_xticks([]); a.set_yticks([])
    sub = G[1, 1:].subgridspec(2, 5)                        # the lower right: ten small images
    for j in range(10):
        a = fig.add_subplot(sub[j // 5, j % 5]); a.imshow(C[j].reshape(8, 8), cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
        if j == 2:
            a.set_title("Digits: the 10 cluster centres", fontsize=10)
    plt.tight_layout(); plt.savefig("kmeans_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved kmeans_results.png")


def draw_diagram():                                     # the loop
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 119: k-means", ha="center", fontsize=12, fontweight="bold")
    box(1.8, 2.4, 3.0, 1.5, "choose k starting centres\n(k-means++: spread out)", "#FAEEDA")
    box(5.5, 2.4, 3.0, 1.5, "give every point to\nits nearest centre", "#DCEBF7")
    box(9.2, 2.4, 3.0, 1.5, "move each centre to\nthe mean of its points", "#E5F2E1")
    box(11.9, 2.4, 1.8, 1.5, "until nothing\nmoves", "#EEEDFE")
    ax.annotate("", xy=(3.96, 2.4), xytext=(3.34, 2.4), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(7.66, 2.4), xytext=(7.04, 2.4), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(10.96, 2.4), xytext=(10.74, 2.4), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(5.5, 1.61), xytext=(9.2, 1.61), arrowprops=dict(arrowstyle="->", color="#333333", connectionstyle="bar,fraction=-0.12"))
    ax.text(7.35, 0.65, "each step can only lower the total squared distance (the inertia)", ha="center", fontsize=8.5)
    plt.savefig("kmeans_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved kmeans_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 120 is hierarchical clustering from scratch: no k chosen in advance, but a whole tree of merges, drawn as a real dendrogram.")


def main():                                             # runs every section in order
    sets = datasets()                                      # three labelled datasets
    demo_hand()                                            # 1. by hand
    demo_check(sets)                                       # 2. correctness
    init = demo_init(sets)                                 # 3. starts
    elb = demo_elbow(sets)                                 # 4. the elbow
    demo_scaling()                                         # 5. scaling
    moons, mlab = demo_shapes()                            # 6. failures
    C = demo_digits(sets)                                  # 7. digits
    plot_results(elb, init, moons, mlab, C)                # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 120


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
