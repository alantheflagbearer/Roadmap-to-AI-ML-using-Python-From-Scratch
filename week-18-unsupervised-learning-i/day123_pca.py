"""
Day 123: Principal Component Analysis From Scratch -- Variance Explained

A table with 64 columns rarely has 64 independent directions of information.
Principal component analysis (PCA) finds the directions along which the data
varies most: the eigenvectors of its covariance matrix, ordered by their
eigenvalues (the variance along each). Keeping the first few gives a smaller
table that holds most of the variation.

Everything is written from scratch in NumPy:
  * the covariance matrix, and its eigenvectors through np.linalg.eigh
  * power iteration with deflation, to find the same eigenvectors with nothing
    but matrix-vector products
  * projecting onto k components, reconstructing from them, and the variance
    each component explains

Day 115 used PCA through the SVD as a tool; Day 124 connects the two. The data:
handwritten digits, breast cancer and wine, all bundled with scikit-learn.

  1. By hand: five points in two dimensions.
  2. Correctness: against scikit-learn, and power iteration against eigh.
  3. Variance explained on three real datasets.
  4. Reconstruction: digits from k components.
  5. Scaling: wine, raw against standardised.
  6. PCA before a model: kNN and k-means on digits.
  7. Where PCA misleads: the signal in a low-variance direction.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Days 115 and 119

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.decomposition import PCA                  # used ONLY as an outside check
from sklearn.datasets import load_breast_cancer, load_digits, load_wine  # bundled datasets
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "week-17-classical-ml-kernels-ensembles"))  # Weeks 17 and 18
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing
    import day119_k_means as d119                       # Day 119: kmeans, ari, standardise
    import day115_k_nearest_neighbors as d115           # Day 115: knn_predict
ari, standardise = d119.ari, d119.standardise           # unchanged


# =============================================================================
# PCA -- NEW
# =============================================================================
def pca_fit(X):                                         # NEW: mean, eigenvalues (largest first) and eigenvectors (as columns) of the covariance
    mu = X.mean(0); A = X - mu                            # centre every feature
    C = A.T @ A / (len(X) - 1)                            # the covariance matrix, d x d
    vals, vecs = np.linalg.eigh(C)                        # eigh: for symmetric matrices; it returns the eigenvalues in ascending order
    order = np.argsort(vals)[::-1]                        # largest first
    return mu, np.maximum(vals[order], 0), vecs[:, order]  # (rounding can leave tiny negative eigenvalues)


def transform(X, mu, V, k):                             # NEW: coordinates along the first k components
    return (X - mu) @ V[:, :k]


def reconstruct(Z, mu, V, k):                           # NEW: back to the original space from k coordinates
    return Z @ V[:, :k].T + mu


def power_iteration(C, k, iters=5000, tol=1e-12, seed=0):  # NEW: the top k eigenpairs with matrix-vector products only
    rng = np.random.RandomState(seed); C = C.copy(); vals, vecs, steps = [], [], []
    for _ in range(k):
        v = rng.normal(size=len(C)); v /= np.linalg.norm(v)
        for t in range(1, iters + 1):
            w = C @ v; w /= np.linalg.norm(w)                 # multiply and renormalise: the largest direction grows fastest
            done = np.linalg.norm(w - v) < tol
            v = w
            if done:
                break
        lam = float(v @ C @ v); vals.append(lam); vecs.append(v); steps.append(t)
        C -= lam * np.outer(v, v)                           # deflation: remove the direction found, so the next one is on top
    return np.array(vals), np.array(vecs).T, steps


def explained(vals):                                    # NEW: each component's share of the total variance
    return vals / vals.sum()


def needed(vals, share):                                # NEW: how many components hold at least `share` of the variance
    return int(np.searchsorted(np.cumsum(explained(vals)), share - 1e-12) + 1)


def datasets():                                         # three real datasets
    dg, bc, wi = load_digits(), load_breast_cancer(), load_wine()
    return {"digits": (dg.data, dg.target), "breast cancer": (standardise(bc.data), bc.target), "wine": (standardise(wi.data), wi.target)}


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- five points in two dimensions")
    print("=" * 70)
    X = np.array([[-2.0, -1.5], [-1.0, -1.0], [0.0, 0.0], [1.0, 0.5], [2.0, 2.0]])
    mu, vals, V = pca_fit(X); C = np.cov(X.T)
    a, b, c = C[0, 0], C[0, 1], C[1, 1]
    lam = (a + c) / 2 + np.array([1, -1]) * np.sqrt(((a - c) / 2) ** 2 + b ** 2)  # the eigenvalues of a 2 x 2 symmetric matrix
    print(f"covariance [[{a:.4f}, {b:.4f}], [{b:.4f}, {c:.4f}]]")
    print(f"by hand: eigenvalues = (a + c)/2 +- sqrt(((a - c)/2)^2 + b^2) = {lam[0]:.4f} and {lam[1]:.4f}; eigh gives {vals[0]:.4f} and {vals[1]:.4f}")
    v = V[:, 0] * np.sign(V[0, 0])
    print(f"first component ({v[0]:.4f}, {v[1]:.4f}): at {np.degrees(np.arctan2(v[1], v[0])):.1f} degrees, along the line the points follow; "
          f"it explains {explained(vals)[0]:.4f} of the variance")
    print(f"coordinates along it: {(np.round(transform(X, mu, V, 1)[:, 0] * np.sign(V[0, 0]), 4) + 0.0).tolist()}")  # + 0.0 turns -0.0 into 0.0
    print("Actual output from this lesson's run.")


def demo_check(sets):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- against scikit-learn, and power iteration against eigh")
    print("=" * 70)
    X, _ = sets["digits"]; mu, vals, V = pca_fit(X); sk = PCA().fit(X)
    print(f"digits: explained-variance shares, largest gap from scikit-learn {np.abs(explained(vals) - sk.explained_variance_ratio_).max():.1e}; "
          f"first 10 components, smallest |cos| with scikit-learn's {np.abs((V[:, :10] * sk.components_[:10].T).sum(0)).min():.10f} (signs may differ)")
    C = np.cov(X.T); pv, pV, steps = power_iteration(C, 5)
    print(f"power iteration, top 5: largest eigenvalue gap {np.abs(pv - vals[:5]).max() / vals[0]:.1e} (relative); "
          f"smallest |cos| with eigh's {np.abs((pV * V[:, :5]).sum(0)).min():.8f}")
    print(f"  steps to converge: {steps}; ratios of each eigenvalue to the next: {np.round(vals[1:6] / vals[:5], 3).tolist()}")
    print("power iteration converges at the speed of that ratio: the closer two eigenvalues, the more steps")
    print("Actual output from this lesson's run.")


def demo_variance(sets):
    print("\n" + "=" * 70)
    print("3. VARIANCE EXPLAINED -- how many components hold how much")
    print("=" * 70)
    print(f"  {'data':<14}{'features':>9}{'1st share':>11}{'for 80%':>9}{'for 90%':>9}{'for 95%':>9}{'for 99%':>9}")
    out = {}
    for name, (X, y) in sets.items():
        _, vals, _ = pca_fit(X); out[name] = vals
        print(f"  {name:<14}{X.shape[1]:>9}{explained(vals)[0]:>11.4f}" + "".join(f"{needed(vals, s):>9}" for s in (0.8, 0.9, 0.95, 0.99)))
    print("(breast cancer and wine standardised; digits as raw pixels)")
    print("Actual output from this lesson's run.")
    return out


def demo_reconstruct(sets):
    print("\n" + "=" * 70)
    print("4. RECONSTRUCTION -- digits from k components")
    print("=" * 70)
    X, _ = sets["digits"]; mu, vals, V = pca_fit(X); n = len(X); rows = []
    print(f"  {'k':>4}{'variance kept':>15}{'mean squared error per pixel':>30}{'sum of the dropped eigenvalues / 64':>37}")
    for k in (1, 2, 5, 10, 20, 40, 64):
        R = reconstruct(transform(X, mu, V, k), mu, V, k); mse = float(((X - R) ** 2).mean())
        rows.append((k, mse))
        print(f"  {k:>4}{explained(vals)[:k].sum():>15.4f}{mse:>30.4f}{vals[k:].sum() * (n - 1) / n / 64:>37.4f}")
    print("the error left is exactly the variance in the directions dropped: PCA keeps the most variance any k directions can")
    print("Actual output from this lesson's run.")
    return mu, V


def demo_scaling():
    print("\n" + "=" * 70)
    print("5. SCALING -- wine, raw against standardised")
    print("=" * 70)
    w = load_wine()
    for name, X in (("raw", w.data), ("standardised", standardise(w.data))):
        mu, vals, V = pca_fit(X); top = np.argsort(-np.abs(V[:, 0]))[:2]
        print(f"  {name:<13}: 1st component explains {explained(vals)[0]:.4f}; its largest weights: " +
              ", ".join(f"'{w.feature_names[i]}' {abs(V[i, 0]):.3f}" for i in top) +
              f"; k-means on 2 components, ARI {ari(d119.kmeans(transform(X, mu, V, 2), 3, 0)[0], w.target):.4f}")
    print("raw, the first component is almost exactly 'proline', the feature with the largest numbers; scaling decides what 'most variance' means")
    print("Actual output from this lesson's run.")


def demo_models(sets):
    print("\n" + "=" * 70)
    print("6. PCA BEFORE A MODEL -- digits, kNN (k = 3) and k-means (k = 10) on k components")
    print("=" * 70)
    X, y = sets["digits"]; o = np.random.RandomState(0).permutation(len(y)); te, tr = o[:540], o[540:]
    mu, vals, V = pca_fit(X[tr])                          # components from the training images only
    print(f"  {'components':>11}{'variance kept':>15}{'kNN accuracy':>14}{'k-means ARI':>13}")
    rows = []
    for k in (2, 5, 10, 20, 40, 64):
        A, B = transform(X[tr], mu, V, k), transform(X[te], mu, V, k)
        acc = float(np.mean(d115.knn_predict(A, y[tr], B, 3) == y[te])); a = ari(d119.kmeans(transform(X, mu, V, k), 10, 0)[0], y)
        rows.append((k, acc, a))
        print(f"  {k:>11}{explained(vals)[:k].sum():>15.4f}{acc:>14.4f}{a:>13.4f}")
    print("Actual output from this lesson's run.")
    return rows


def demo_misleads():
    print("\n" + "=" * 70)
    print("7. WHERE PCA MISLEADS -- the class signal in a low-variance direction (made on purpose, so the truth is known)")
    print("=" * 70)
    rng = np.random.RandomState(0); y = np.r_[np.zeros(200, int), np.ones(200, int)]
    X = np.c_[rng.normal(0, 3.0, 400), np.where(y == 1, 0.5, -0.5) + rng.normal(0, 0.2, 400)]  # wide noise; narrow but class-separating
    mu, vals, V = pca_fit(X); o = rng.permutation(400); te, tr = o[:120], o[120:]
    for k, name in ((1, "the 1st component only"), (2, "both components")):
        A, B = transform(X[tr], mu, V, k), transform(X[te], mu, V, k)
        print(f"  {name:<24}: variance kept {explained(vals)[:k].sum():.4f}; kNN accuracy {np.mean(d115.knn_predict(A, y[tr], B, 3) == y[te]):.4f}")
    A, B = transform(X[tr], mu, V, 2)[:, 1:], transform(X[te], mu, V, 2)[:, 1:]
    print(f"  {'the 2nd component only':<24}: variance kept {explained(vals)[1]:.4f}; kNN accuracy {np.mean(d115.knn_predict(A, y[tr], B, 3) == y[te]):.4f}")
    print("PCA never looks at the labels: 'most variance' and 'most useful' are different questions")
    print("Actual output from this lesson's run.")
    return X, y, V


def plot_results(sets, var, mu, V, rows, mis):          # six panels
    fig = plt.figure(figsize=(16, 9.6)); G = fig.add_gridspec(2, 3)  # a 2 x 3 grid of panels
    a = fig.add_subplot(G[0, 0])
    for name, vals in var.items():
        a.plot(np.arange(1, len(vals) + 1), np.cumsum(explained(vals)), marker=".", label=name)
    a.axhline(0.9, color="k", linestyle=":", linewidth=0.8); a.set_xlabel("components kept"); a.set_ylabel("share of variance kept")
    a.set_title("Variance explained", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    sub = G[0, 1].subgridspec(2, 4)                         # the first eight components, shown as images
    for j in range(8):
        b = fig.add_subplot(sub[j // 4, j % 4]); b.imshow(V[:, j].reshape(8, 8), cmap="RdBu_r"); b.set_xticks([]); b.set_yticks([])
        b.set_title(f"component {j + 1}", fontsize=8)
    X, y = sets["digits"]; Z = transform(X, X.mean(0), pca_fit(X)[2], 2)
    a = fig.add_subplot(G[0, 2]); s = a.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5); fig.colorbar(s, ax=a, label="digit")
    a.set_title("Digits on the first 2 components", fontsize=10); a.set_xlabel("component 1"); a.set_ylabel("component 2")
    sub = G[1, 0].subgridspec(2, 4)                         # one digit rebuilt from k components
    img = X[0]; Vf = pca_fit(X)[2]; m = X.mean(0)
    for j, k in enumerate((1, 2, 5, 10, 20, 30, 40, 64)):
        b = fig.add_subplot(sub[j // 4, j % 4]); b.imshow(reconstruct(transform(img[None], m, Vf, k), m, Vf, k).reshape(8, 8), cmap="gray_r")
        b.set_xticks([]); b.set_yticks([]); b.set_title(f"k = {k}", fontsize=8)
    a = fig.add_subplot(G[1, 1]); r = np.array(rows); a.plot(r[:, 0], r[:, 1], marker="o", label="kNN accuracy")
    a.plot(r[:, 0], r[:, 2], marker="s", label="k-means ARI"); a.set_xscale("log"); a.minorticks_off(); a.set_xticks(r[:, 0]); a.set_xticklabels([str(int(k)) for k in r[:, 0]])
    a.set_xlabel("components kept"); a.set_title("Digits: models on k components", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = fig.add_subplot(G[1, 2]); Xm, ym, Vm = mis; a.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="coolwarm", s=6)
    for j, (c, lab) in enumerate((("k", "1st component"), ("g", "2nd component"))):
        a.annotate("", xy=Xm.mean(0) + 4 * Vm[:, j] / (1 + 3 * j), xytext=Xm.mean(0), arrowprops=dict(arrowstyle="->", color=c, linewidth=2))
        a.text(*(Xm.mean(0) + 4.3 * Vm[:, j] / (1 + 3 * j)), lab, color=c, fontsize=8)
    a.set_title("Most variance is not most useful", fontsize=10); a.set_xlabel("x1"); a.set_ylabel("x2")
    plt.tight_layout(); plt.savefig("pca_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved pca_results.png")


def draw_diagram():                                     # the pipeline
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 123: principal component analysis", ha="center", fontsize=12, fontweight="bold")
    box(1.7, 2.5, 2.6, 1.6, "centre each feature\n(and scale it, if\nunits differ)", "#FAEEDA")
    box(4.9, 2.5, 2.8, 1.6, "covariance matrix\nC = A^T A / (n - 1)", "#DCEBF7")
    box(8.2, 2.5, 3.0, 1.6, "eigenvectors of C,\nlargest eigenvalue first:\nthe components", "#E5F2E1")
    box(11.6, 2.5, 2.4, 1.6, "keep k:\nproject, or\nreconstruct", "#EEEDFE")
    for x0, x1 in ((3.04, 3.46), (6.34, 6.66), (9.74, 10.36)):
        ax.annotate("", xy=(x1, 2.5), xytext=(x0, 2.5), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 0.9, "each eigenvalue is the variance along its component; the components are perpendicular, and PCA never sees the labels",
            ha="center", fontsize=8.5)
    plt.savefig("pca_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved pca_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 124 is the singular value decomposition and its relationship to PCA, with a real matrix-factorization demonstration.")


def main():                                             # runs every section in order
    sets = datasets()
    demo_hand()                                            # 1. by hand
    demo_check(sets)                                       # 2. correctness
    var = demo_variance(sets)                              # 3. variance explained
    mu, V = demo_reconstruct(sets)                         # 4. reconstruction
    demo_scaling()                                         # 5. scaling
    rows = demo_models(sets)                               # 6. PCA before a model
    mis = demo_misleads()                                  # 7. where PCA misleads
    plot_results(sets, var, mu, V, rows, mis)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
