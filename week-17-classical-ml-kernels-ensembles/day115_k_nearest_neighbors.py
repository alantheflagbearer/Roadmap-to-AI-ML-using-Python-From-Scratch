"""
Day 115: k-Nearest Neighbours From Scratch -- a Real Curse-of-Dimensionality Demonstration

k-nearest neighbours (kNN) has no training at all: to classify a point, find the
k training points closest to it and take a vote. It is the simplest model in
this course, and it depends entirely on one idea, that "close" means "similar".
In many dimensions that idea quietly breaks down. That is the curse of
dimensionality, and today measures it.

Everything is written from scratch in NumPy:
  * all distances at once, the k nearest by partial sorting, and a vote
  * principal component analysis (PCA) by the singular value decomposition

  1. By hand: five points.
  2. Correctness: against scikit-learn's KNeighborsClassifier.
  3. Choosing k on digits.
  4. Scaling: the unit of each feature decides who is "near".
  5. The curse, on random data: distances become all alike.
  6. The curse, in volume: how far a neighbourhood must reach.
  7. The curse, on real data: digits with useless dimensions added.
  8. Fighting back with PCA, and what kNN costs at prediction time.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 112
import time                                            # time: prediction cost

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_breast_cancer, load_digits  # bundled datasets
from sklearn.neighbors import KNeighborsClassifier     # used ONLY as an outside check
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 112 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day112_support_vector_machines as d112       # Day 112: standardise, stratified_split, Day 99's bootstrap
standardise, stratified_split, d99 = d112.standardise, d112.stratified_split, d112.d99  # unchanged
KS = [1, 3, 5, 7, 9, 15, 25, 51]                        # section 3: neighbours
DIMS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]       # sections 5 and 6: dimensions
NOISE = [0, 16, 64, 256, 1024]                          # section 7: useless dimensions added to digits
COMPS = [2, 5, 10, 15, 20, 30, 40, 64]                  # section 8: PCA components
SPLITS = 5                                              # repeated splits


# =============================================================================
# kNN AND PCA -- NEW
# =============================================================================
def sqdist(A, B):                                       # all squared distances at once (Day 113's)
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def knn_predict(Xtr, ytr, X, k, chunk=500):             # NEW: majority vote of the k nearest training points
    n_classes = int(ytr.max()) + 1; out = []
    for s in range(0, len(X), chunk):                     # in chunks, so the distance table stays small
        D = sqdist(X[s:s + chunk], Xtr)                     # rows: new points, columns: training points
        idx = np.argpartition(D, k - 1, axis=1)[:, :k] if k < len(ytr) else np.tile(np.arange(len(ytr)), (len(D), 1))  # the k smallest, unordered
        votes = np.zeros((len(D), n_classes))
        for j in range(k):
            votes[np.arange(len(D)), ytr[idx[:, j]]] += 1    # one vote per neighbour
        d = np.take_along_axis(D, idx, 1)                   # their distances, to break ties
        tie = (votes == votes.max(1, keepdims=True)).sum(1) > 1
        for r in np.where(tie)[0]:                          # a tie: the tied class with the nearest neighbour wins
            best = np.where(votes[r] == votes[r].max())[0]
            order = idx[r][np.argsort(d[r])]
            out_r = next(ytr[i] for i in order if ytr[i] in best); votes[r, out_r] += 0.5
        out.append(np.argmax(votes, 1))
    return np.concatenate(out)


def pca_fit(X, k):                                      # NEW: the k directions of largest variance, by SVD
    mu = X.mean(0); _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k], (s ** 2) / (s ** 2).sum()          # centre, directions, share of variance per direction


def digits():                                           # 1,797 images, pixels 0..16
    d = load_digits(); return d.data.astype(float), d.target


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- five points")
    print("=" * 70)
    X = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5]], dtype=float); y = np.array([0, 0, 1, 1, 1])
    q = np.array([[0.4, 0.4]])
    print("training: (0,0) class 0, (1,0) class 0, (0,1) class 1, (5,5) class 1, (6,5) class 1; query (0.4, 0.4)")
    print(f"squared distances: {sqdist(q, X)[0].round(2).tolist()}")
    for k in (1, 3, 5):
        print(f"  k = {k}: class {knn_predict(X, y, q, k)[0]}", end="")
        print("  (nearest is (0,0))" if k == 1 else "  (two of the three nearest are class 0)" if k == 3 else "  (all five vote: three are class 1)")
    print("Actual output from this lesson's run.")


def demo_check():
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- against scikit-learn's KNeighborsClassifier")
    print("=" * 70)
    X, y = digits(); tr, te = stratified_split(y, 0.3, 0)
    for k in (1, 5, 15):
        ours = knn_predict(X[tr], y[tr], X[te], k)
        theirs = KNeighborsClassifier(k, algorithm="brute").fit(X[tr], y[tr]).predict(X[te])
        print(f"digits, k = {k:>2}: the same prediction on {np.mean(ours == theirs):.4f} of {len(te)} test images; accuracy {np.mean(ours == y[te]):.4f} against {np.mean(theirs == y[te]):.4f}")
    print("ties between equally common classes are the only place two correct versions can differ")
    print("Actual output from this lesson's run.")


def demo_k():
    print("\n" + "=" * 70)
    print(f"3. CHOOSING k -- digits, {SPLITS} splits")
    print("=" * 70)
    X, y = digits(); R = []
    for s in range(SPLITS):
        tr, te = stratified_split(y, 0.3, s); row = []
        for k in KS:
            tr_acc = np.mean(knn_predict(X[tr], y[tr], X[tr], k) == y[tr])  # each training image is its own nearest neighbour
            row.append((tr_acc, np.mean(knn_predict(X[tr], y[tr], X[te], k) == y[te])))
        R.append(row)
    R = np.array(R).mean(0)
    print(f"{'k':>5}{'train acc':>11}{'test acc':>10}")
    for k, (a, b) in zip(KS, R):
        print(f"{k:>5}{a:>11.4f}{b:>10.4f}")
    print("k = 1 is perfect on training rows because each row finds itself; larger k smooths, and eventually blurs the classes")
    print("Actual output from this lesson's run.")
    return R                                                  # the plot uses it


def demo_scaling():
    print("\n" + "=" * 70)
    print(f"4. SCALING -- breast cancer, k = 5, {SPLITS} splits")
    print("=" * 70)
    bc = load_breast_cancer(); X, y = bc.data, bc.target; names = list(bc.feature_names)
    sd = X.std(0); j = np.argsort(-sd)[:3]
    print("the largest spreads: " + ", ".join(f"{names[i]} {sd[i]:.1f}" for i in j) + f"; the smallest: {names[np.argmin(sd)]} {sd.min():.4f}")
    raw, std = [], []
    for s in range(SPLITS):
        tr, te = stratified_split(y, 0.3, s)
        raw.append(np.mean(knn_predict(X[tr], y[tr], X[te], 5) == y[te]))
        A, B = standardise(X[tr], X[te]); std.append(np.mean(knn_predict(A, y[tr], B, 5) == y[te]))
    print(f"raw measurements: {np.mean(raw):.4f}; standardised: {np.mean(std):.4f}")
    a = X[:, names.index("worst area")]; d = np.abs(a[:, None] - a[None, :]) ** 2; D = sqdist(X, X)
    print(f"on raw data, 'worst area' alone makes up {np.median(d[np.triu_indices(len(a), 1)] / np.maximum(D[np.triu_indices(len(a), 1)], 1e-12)):.4f} "
          "of the squared distance between two tumours (median over all pairs): one feature decides who is near")
    print("Actual output from this lesson's run.")


def demo_concentration():
    print("\n" + "=" * 70)
    print("5. THE CURSE ON RANDOM DATA -- distances become all alike")
    print("=" * 70)
    print("1,000 points uniform in the unit cube; for 100 query points: (farthest - nearest) / nearest, and nearest / mean distance")
    print(f"{'dimensions':>11}{'relative contrast':>19}{'nearest / mean':>16}")
    rows = []
    for d in DIMS:
        rng = np.random.RandomState(d); P = rng.uniform(size=(1000, d)); Q = rng.uniform(size=(100, d))
        D = np.sqrt(np.maximum(sqdist(Q, P), 0)); lo, hi, mu = D.min(1), D.max(1), D.mean(1)
        rows.append((d, np.median((hi - lo) / lo), np.median(lo / mu)))
        print(f"{d:>11}{rows[-1][1]:>19.4f}{rows[-1][2]:>16.4f}")
    print("as dimensions grow, the nearest point is barely nearer than the average one: 'nearest' stops meaning much")
    print("Actual output from this lesson's run.")
    return rows                                               # the plot uses them


def demo_volume():
    print("\n" + "=" * 70)
    print("6. THE CURSE IN VOLUME -- how far a neighbourhood must reach")
    print("=" * 70)
    print("to hold 1% of points spread evenly in a unit cube, a smaller cube needs edge 0.01^(1/d); and the measured distance to the nearest of 1,000 points")
    print(f"{'dimensions':>11}{'edge for 1%':>13}{'nearest of 1,000':>18}{'cube diagonal':>15}")
    for d in DIMS:
        rng = np.random.RandomState(100 + d); P = rng.uniform(size=(1000, d)); Q = rng.uniform(size=(100, d))
        nn = np.sqrt(np.maximum(sqdist(Q, P), 0)).min(1)
        print(f"{d:>11}{0.01 ** (1 / d):>13.4f}{np.median(nn):>18.4f}{np.sqrt(d):>15.4f}")
    print("in 100 dimensions the 'local' 1% cube spans 95% of every side: a neighbourhood is no longer local")
    print("Actual output from this lesson's run.")


def demo_noise():
    print("\n" + "=" * 70)
    print(f"7. THE CURSE ON REAL DATA -- digits with useless dimensions added, k = 5, {SPLITS} splits")
    print("=" * 70)
    X, y = digits(); print("each added dimension is uniform noise on 0..16, the same range as a pixel")
    print(f"{'added':>7}{'dimensions':>12}{'accuracy':>10}{'nearest / mean distance':>25}")
    rows = []
    for k in NOISE:
        acc, ratio = [], []
        for s in range(SPLITS):
            Z = np.hstack([X, np.random.RandomState(s).uniform(0, 16, (len(y), k))]); tr, te = stratified_split(y, 0.3, s)
            acc.append(np.mean(knn_predict(Z[tr], y[tr], Z[te], 5) == y[te]))
            D = np.sqrt(np.maximum(sqdist(Z[te][:100], Z[tr]), 0)); ratio.append(np.median(D.min(1) / D.mean(1)))
        rows.append((k, np.mean(acc), np.mean(ratio)))
        print(f"{k:>7}{64 + k:>12}{np.mean(acc):>10.4f}{np.mean(ratio):>25.4f}")
    print("Actual output from this lesson's run.")
    return rows                                               # the plot uses them


def demo_pca():
    print("\n" + "=" * 70)
    print(f"8. FIGHTING BACK WITH PCA, AND THE COST OF A LAZY MODEL -- k = 5, {SPLITS} splits")
    print("=" * 70)
    X, y = digits(); _, _, share = pca_fit(X, 64)
    print(f"digits: the first 10 PCA directions hold {share[:10].sum():.4f} of the variance; the first 30, {share[:30].sum():.4f}")
    print(f"{'components':>11}{'clean digits':>14}{'digits + 256 noise':>20}")
    rows = []
    for c in COMPS:
        a, b = [], []
        for s in range(SPLITS):
            tr, te = stratified_split(y, 0.3, s)
            mu, V, _ = pca_fit(X[tr], c); a.append(np.mean(knn_predict((X[tr] - mu) @ V.T, y[tr], (X[te] - mu) @ V.T, 5) == y[te]))  # fitted on training rows only
            Z = np.hstack([X, np.random.RandomState(s).uniform(0, 16, (len(y), 256))])
            mu, V, _ = pca_fit(Z[tr], c); b.append(np.mean(knn_predict((Z[tr] - mu) @ V.T, y[tr], (Z[te] - mu) @ V.T, 5) == y[te]))
        rows.append((c, np.mean(a), np.mean(b)))
        print(f"{c:>11}{np.mean(a):>14.4f}{np.mean(b):>20.4f}")
    print(f"PCA keeps the directions with most variance: a noise dimension has variance {16 ** 2 / 12:.2f}, an average pixel {X.var(0).mean():.2f} and the busiest pixel {X.var(0).max():.2f},")
    print("  so noise is not simply ranked last, and only part of it is filtered out")
    tr, te = stratified_split(y, 0.3, 0)
    for n in (250, 500, 1000, 1257):                      # the cost of prediction grows with the training set
        sub = tr[:n]; t0 = time.perf_counter()
        for _ in range(5):
            knn_predict(X[sub], y[sub], X[te], 5)
        print(f"  {n:>5} training images: predicting {len(te)} test images takes {(time.perf_counter() - t0) / 5 * 1000:.1f} ms (mean of 5, within this run)")
    print("Actual output from this lesson's run.")
    return rows                                               # the plot uses them


def plot_results(R, conc, noise, pca):                  # four panels
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))           # canvas
    ax[0].plot(KS, R[:, 0], marker="o", label="training"); ax[0].plot(KS, R[:, 1], marker="o", label="test"); ax[0].set_xscale("log")
    ax[0].set_xlabel("k (log scale)"); ax[0].set_ylabel("accuracy"); ax[0].set_title("Digits: choosing k", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot([r[0] for r in conc], [r[2] for r in conc], marker="o", color="#9467bd"); ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("dimensions (log scale)"); ax[1].set_ylabel("nearest / mean distance"); ax[1].set_title("Random points: the nearest is not near", fontsize=10); ax[1].grid(alpha=0.3)
    x = [64 + r[0] for r in noise]
    ax[2].plot(x, [r[1] for r in noise], marker="o", label="accuracy"); ax[2].plot(x, [r[2] for r in noise], marker="s", linestyle="--", label="nearest / mean distance")
    ax[2].set_xscale("log"); ax[2].set_xlabel("dimensions: 64 pixels + noise (log scale)"); ax[2].set_title("Digits with useless dimensions", fontsize=10); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    ax[3].plot([r[0] for r in pca], [r[1] for r in pca], marker="o", label="clean digits"); ax[3].plot([r[0] for r in pca], [r[2] for r in pca], marker="s", label="digits + 256 noise")
    ax[3].set_xlabel("PCA components kept"); ax[3].set_ylabel("test accuracy"); ax[3].set_title("Fighting back with PCA", fontsize=10); ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("knn_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved knn_results.png")


def draw_diagram():                                     # the method and where it breaks
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 115: k-nearest neighbours, and the curse of dimensionality", ha="center", fontsize=12, fontweight="bold")
    box(1.7, 2.7, 2.8, 1.4, "'training':\nstore every row", "#FAEEDA")
    box(5.0, 2.7, 3.0, 1.4, "a new point: its distance\nto every stored row", "#DCEBF7")
    box(8.4, 2.7, 3.0, 1.4, "keep the k nearest;\nthey vote", "#E5F2E1")
    box(11.6, 2.7, 2.4, 1.4, "the class with\nmost votes", "#EEEDFE")
    for x0, x1 in ((3.14, 3.46), (6.54, 6.86), (9.94, 10.36)):
        ax.annotate("", xy=(x1, 2.7), xytext=(x0, 2.7), arrowprops=dict(arrowstyle="->", color="#333333"))
    box(6.7, 0.85, 9.4, 0.8, "in many dimensions every distance is about the same, so 'nearest' carries little information", "#F4C7C3")
    ax.annotate("", xy=(5.0, 1.27), xytext=(5.0, 1.98), arrowprops=dict(arrowstyle="->", color="#a33"))
    plt.savefig("knn_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved knn_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("9. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 116 is stacking and blending: training a model to combine other models' predictions, with a real meta-model demonstration.")


def main():                                             # runs every section in order
    demo_hand()                                            # 1. by hand
    demo_check()                                           # 2. correctness
    R = demo_k()                                           # 3. choosing k
    demo_scaling()                                         # 4. scaling
    conc = demo_concentration()                            # 5. concentration
    demo_volume()                                          # 6. volume
    noise = demo_noise()                                   # 7. noise dimensions
    pca = demo_pca()                                       # 8. PCA
    plot_results(R, conc, noise, pca)                      # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 9. Day 116


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
