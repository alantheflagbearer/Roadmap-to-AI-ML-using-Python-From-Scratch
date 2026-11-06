"""
Day 124: The Singular Value Decomposition, and Its Relationship to PCA

Every matrix A, of any shape, factors as A = U S V^T: U and V have perpendicular
unit columns, and S is diagonal with the singular values, largest first. Keeping
the k largest gives the best rank-k approximation of A that exists. Applied to
centred data, the SVD is PCA: V holds the components, and S^2 / (n - 1) the
variances, computed without ever forming the covariance matrix.

Everything is written from scratch in NumPy, with np.linalg.svd as the check:
  * the SVD through the eigenvectors of A^T A (the route Day 123 took)
  * the top singular vectors by power iteration on A and A^T, never forming A^T A
  * the rank-k approximation, and low-rank matrix completion
  * least squares through the pseudoinverse, plain and truncated

The data: handwritten digits, a real photograph bundled with scikit-learn
(china.jpg, in grey), and the Diabetes progression data.

  1. By hand: a 2 x 2 matrix.
  2. SVD is PCA: the same components and variances as Day 123.
  3. Three routes, and what squaring costs: an ill-conditioned matrix.
  4. A photograph at rank k.
  5. Filling in missing pixels: low-rank matrix completion.
  6. Least squares through the pseudoinverse.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lesson quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 123

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_diabetes, load_digits, load_sample_image  # bundled data (the photograph needs Pillow to read)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 123 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day123_pca as d123                           # Day 123: pca_fit


# =============================================================================
# THE SVD -- NEW
# =============================================================================
def svd_via_eig(A):                                     # NEW: U, s, V from the eigenvectors of A^T A (Day 123's route)
    vals, V = np.linalg.eigh(A.T @ A)                     # A^T A = V S^2 V^T
    o = np.argsort(vals)[::-1]; vals, V = vals[o], V[:, o]
    s = np.sqrt(np.maximum(vals, 0))                      # singular values are the square roots
    U = (A @ V) / np.where(s > 0, s, 1)                   # A v = s u, so u = A v / s
    return U, s, V


def svd_power(A, k, iters=3000, tol=1e-12, seed=0):     # NEW: the top k singular triplets by alternating multiplication
    rng = np.random.RandomState(seed); U, s, V = [], [], []
    for _ in range(k):
        v = rng.normal(size=A.shape[1]); v /= np.linalg.norm(v)
        for _ in range(iters):
            u = A @ v - sum(sj * uj * (vj @ v) for sj, uj, vj in zip(s, U, V))       # A v, with the triplets already found removed
            u /= np.linalg.norm(u)
            w = A.T @ u - sum(sj * vj * (uj @ u) for sj, uj, vj in zip(s, U, V))     # A^T u, likewise
            sigma = np.linalg.norm(w); w /= sigma
            done = np.linalg.norm(w - v) < tol
            v = w
            if done:
                break
        U.append(u); s.append(sigma); V.append(v)
    return np.array(U).T, np.array(s), np.array(V).T


def low_rank(U, s, V, k):                               # NEW: the best rank-k approximation, U_k S_k V_k^T
    return (U[:, :k] * s[:k]) @ V[:, :k].T


def complete(M, mask, k, iters=2000, tol=1e-6):          # NEW: fill the unknown entries with a rank-k fit, repeatedly
    X = np.where(mask, M, M[mask].mean())                 # start: unknown entries set to the mean of the known ones
    for t in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :k] * s[:k]) @ Vt[:k]                     # the rank-k fit to the current guess
        change = np.linalg.norm((L - X)[~mask]) / max(np.linalg.norm(X[~mask]), 1e-12)
        X = np.where(mask, M, L)                            # keep what is known; take the fit where unknown
        if change < tol:
            break
    return X, t + 1


def pinv_solve(A, b, k=None):                           # NEW: least squares through the SVD; k keeps only the largest k singular values
    U, s, Vt = np.linalg.svd(A, full_matrices=False); k = len(s) if k is None else k
    return Vt[:k].T @ ((U[:, :k].T @ b) / s[:k])          # x = V S^-1 U^T b


def photo():                                            # the bundled photograph, in grey, values 0 to 1
    return load_sample_image("china.jpg").mean(2) / 255.0


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- A = [[3, 0], [4, 5]]")
    print("=" * 70)
    A = np.array([[3.0, 0.0], [4.0, 5.0]]); U, s, V = svd_via_eig(A)
    print("A^T A = [[25, 20], [20, 25]]; its eigenvalues are 25 + 20 = 45 and 25 - 20 = 5, so the singular values are sqrt(45) and sqrt(5)")
    print(f"by hand: {np.sqrt(45):.4f} and {np.sqrt(5):.4f}; from A^T A here: {s[0]:.4f} and {s[1]:.4f}; np.linalg.svd: {np.linalg.svd(A)[1].round(4).tolist()}")
    print(f"U S V^T rebuilds A: {bool(np.allclose((U * s) @ V.T, A))}; columns of U perpendicular: {bool(np.allclose(U.T @ U, np.eye(2)))}")
    print(f"V's first column ({abs(V[0, 0]):.4f}, {abs(V[1, 0]):.4f}) = (1, 1)/sqrt(2): A stretches that direction most, by {s[0]:.4f}")
    print("Actual output from this lesson's run.")


def demo_pca():
    print("\n" + "=" * 70)
    print("2. SVD IS PCA -- digits, centred")
    print("=" * 70)
    X = load_digits().data; mu, vals, Vp = d123.pca_fit(X); A = X - mu
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"s^2 / (n - 1) against Day 123's eigenvalues: largest gap {np.abs(s ** 2 / (len(X) - 1) - vals).max():.1e} (largest eigenvalue {vals[0]:.1f})")
    print(f"right singular vectors against Day 123's components, first 20: smallest |cos| {np.abs((Vt[:20].T * Vp[:, :20]).sum(0)).min():.10f}")
    print(f"U S equals the component scores (X - mean) V: {bool(np.allclose(np.abs(U[:, :20] * s[:20]), np.abs(A @ Vp[:, :20]), atol=1e-6))}")
    print("the SVD reaches PCA without forming the covariance matrix A^T A / (n - 1)")
    print("Actual output from this lesson's run.")


def demo_routes():
    print("\n" + "=" * 70)
    print("3. THREE ROUTES, AND WHAT SQUARING COSTS")
    print("=" * 70)
    X = load_digits().data; A = X - X.mean(0); s_ref = np.linalg.svd(A, compute_uv=False)
    _, s_eig, _ = svd_via_eig(A); _, s_pow, _ = svd_power(A, 5)
    print(f"digits (well conditioned at the top): top 5 singular values, relative gap from np.linalg.svd: through A^T A {np.abs(s_eig[:5] - s_ref[:5]).max() / s_ref[0]:.1e}, "
          f"by power iteration {np.abs(s_pow - s_ref[:5]).max() / s_ref[0]:.1e}")
    n = 12; H = 1.0 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1)  # the 12 x 12 Hilbert matrix: famously ill-conditioned
    s_ref = np.linalg.svd(H, compute_uv=False); _, s_eig, _ = svd_via_eig(H)
    print(f"Hilbert 12 x 12: singular values from {s_ref[0]:.3f} down to {s_ref[-1]:.2e}; condition number {s_ref[0] / s_ref[-1]:.1e}")
    print(f"  {'singular value':>16}{'np.linalg.svd':>16}{'through A^T A':>16}{'relative error':>16}")
    for i in (0, 5, 8, 10, 11):
        print(f"  {i + 1:>16}{s_ref[i]:>16.3e}{s_eig[i]:>16.3e}{abs(s_eig[i] - s_ref[i]) / s_ref[i]:>16.1e}")
    print("A^T A squares the condition number: values below about sqrt(machine precision) x the largest are lost in rounding")
    print("Actual output from this lesson's run.")
    return s_ref, s_eig


def demo_photo():
    print("\n" + "=" * 70)
    print("4. A PHOTOGRAPH AT RANK k -- china.jpg in grey, 427 x 640 pixels")
    print("=" * 70)
    P = photo(); U, s, Vt = np.linalg.svd(P, full_matrices=False); V = Vt.T
    total = np.sqrt((s ** 2).sum())
    print(f"  {'rank':>5}{'relative error':>16}{'from the dropped singular values':>34}{'numbers stored':>16}{'share of pixels':>17}")
    rows = []
    for k in (1, 5, 10, 20, 50, 100, 200):
        L = low_rank(U, s, V, k); err = np.linalg.norm(P - L) / np.linalg.norm(P); pred = np.sqrt((s[k:] ** 2).sum()) / total
        stored = k * (P.shape[0] + P.shape[1] + 1); rows.append((k, err))
        print(f"  {k:>5}{err:>16.4f}{pred:>34.4f}{stored:>16,}{stored / P.size:>17.3f}")
    print("the error of the best rank-k picture is exactly the size of the singular values left out (the Eckart-Young theorem)")
    print("Actual output from this lesson's run.")
    return P, U, s, V, rows


def demo_complete():
    print("\n" + "=" * 70)
    print("5. FILLING IN MISSING PIXELS -- half the photograph hidden at random (every 2nd pixel row and column kept: 214 x 320)")
    print("=" * 70)
    P = photo()[::2, ::2]; rng = np.random.RandomState(0); mask = rng.uniform(size=P.shape) < 0.5  # True = known
    rmse = lambda X: float(np.sqrt(np.mean((X - P)[~mask] ** 2)))
    col = np.where(mask, P, (np.where(mask, P, 0).sum(0) / mask.sum(0))[None, :])  # baseline: each column's known mean
    print(f"known pixels {mask.mean():.3f}; error on the hidden pixels (RMSE, 0 to 1 scale): filled with column means {rmse(col):.4f}")
    rows = []
    for k in (2, 5, 10, 20, 40, 80):
        X, t = complete(P, mask, k); rows.append((k, rmse(X)))
        print(f"  rank {k:>3}: RMSE {rows[-1][1]:.4f} after {t} rounds" + ("  (the 2,000-round limit: not converged)" if t == 2000 else ""))
    print("the high ranks after only 200 rounds: " + ", ".join(f"rank {k} RMSE {rmse(complete(P, mask, k, iters=200)[0]):.4f}" for k in (20, 40, 80)) +
          ": they get worse the longer they run")
    print("too low a rank blurs; too high a rank has the freedom to match the known pixels in ways that do not carry over to the hidden ones")
    print("Actual output from this lesson's run.")
    return P, mask, rows


def demo_lstsq():
    print("\n" + "=" * 70)
    print("6. LEAST SQUARES THROUGH THE PSEUDOINVERSE -- Diabetes, 10 features plus an intercept")
    print("=" * 70)
    d = load_diabetes(); X, y = d.data, d.target; o = np.random.RandomState(0).permutation(len(y)); te, tr = o[:132], o[132:]
    A = np.c_[np.ones(len(X)), X]; s = np.linalg.svd(A[tr], compute_uv=False)
    x_svd = pinv_solve(A[tr], y[tr]); x_ref = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
    print(f"pseudoinverse against np.linalg.lstsq: largest coefficient gap {np.abs(x_svd - x_ref).max():.1e}")
    print(f"singular values of the training matrix: largest {s[0]:.2f}, smallest {s[-1]:.4f}")
    v = np.linalg.svd(A[tr], full_matrices=False)[2][-1]; top = np.argsort(-np.abs(v))[:2]; names = ["intercept"] + list(d.feature_names)
    print(f"the smallest one's direction is mostly '{names[top[0]]}' ({v[top[0]]:+.2f}) and '{names[top[1]]}' ({v[top[1]]:+.2f}); "
          f"their correlation {np.corrcoef(X[tr][:, top[0] - 1], X[tr][:, top[1] - 1])[0, 1]:.3f}: one nearly repeats the other")
    print(f"  {'kept':>5}{'largest |coefficient|':>23}{'test RMSE':>11}")
    for k in (11, 10, 9, 8, 6, 4):
        x = pinv_solve(A[tr], y[tr], k)
        print(f"  {k:>5}{np.abs(x).max():>23.1f}{np.sqrt(np.mean((A[te] @ x - y[te]) ** 2)):>11.3f}")
    print("dropping the smallest singular values tames the coefficients; it changes the test error only a little here")
    print("Actual output from this lesson's run.")


def plot_results(routes, ph, comp):                      # six panels
    P, U, s, V, rows = ph; fig = plt.figure(figsize=(16, 9.6)); G = fig.add_gridspec(2, 3)  # a 2 x 3 grid of panels
    a = fig.add_subplot(G[0, 0]); a.imshow(P, cmap="gray"); a.set_title("The photograph (rank 427)", fontsize=10); a.axis("off")
    a = fig.add_subplot(G[0, 1]); a.imshow(low_rank(U, s, V, 20), cmap="gray", vmin=0, vmax=1); a.axis("off")
    a.set_title(f"Rank 20: {20 * (P.shape[0] + P.shape[1] + 1) / P.size:.3f} of the numbers", fontsize=10)
    a = fig.add_subplot(G[0, 2]); a.semilogy(np.arange(1, len(s) + 1), s, color="#1f77b4"); a.set_xlabel("index")
    a.set_ylabel("singular value (log scale)"); a.set_title("The photograph's singular values", fontsize=10); a.grid(alpha=0.3)
    Pm, mask, crow = comp; X, _ = complete(Pm, mask, int(min(crow, key=lambda r: r[1])[0]))
    a = fig.add_subplot(G[1, 0]); a.imshow(np.where(mask, Pm, np.nan), cmap="gray"); a.axis("off"); a.set_title("Half the pixels hidden", fontsize=10)
    a = fig.add_subplot(G[1, 1]); a.imshow(X, cmap="gray", vmin=0, vmax=1); a.axis("off")
    a.set_title(f"Filled in at rank {int(min(crow, key=lambda r: r[1])[0])}", fontsize=10)
    a = fig.add_subplot(G[1, 2]); s_ref, s_eig = routes; i = np.arange(1, len(s_ref) + 1)
    a.semilogy(i, s_ref, marker="o", label="np.linalg.svd"); a.semilogy(i, s_eig, marker="x", linestyle="--", label="through A^T A")
    a.set_xlabel("index"); a.set_ylabel("singular value (log scale)"); a.set_title("Hilbert 12 x 12: squaring loses the small ones", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("svd_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved svd_results.png")


def draw_diagram():                                     # A = U S V^T
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 124: A = U S V^T", ha="center", fontsize=12, fontweight="bold")
    box(1.6, 2.5, 2.4, 1.8, "A\n(n x d):\nany matrix", "#FAEEDA")
    box(4.6, 2.5, 2.4, 1.8, "U (n x r):\nperpendicular\nunit columns", "#DCEBF7")
    box(7.4, 2.5, 2.4, 1.8, "S: singular values,\nlargest first,\non a diagonal", "#E5F2E1")
    box(10.15, 2.5, 2.2, 1.8, "V^T (r x d):\nperpendicular\nunit rows", "#DCEBF7")
    ax.text(3.1, 2.5, "=", ha="center", va="center", fontsize=16)
    for x in (5.98, 8.82):
        ax.text(x, 2.5, "x", ha="center", va="center", fontsize=12)
    box(12.35, 2.5, 1.2, 1.8, "keep\nthe first k:\nbest rank k", "#EEEDFE", fs=8)
    ax.text(6.5, 0.75, "on centred data: V = the PCA components, S^2 / (n - 1) = their variances, U S = the scores", ha="center", fontsize=8.5)
    plt.savefig("svd_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved svd_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 125 is the Week 18 build: t-SNE and UMAP from scratch, compared on real high-dimensional data, with the week's clustering parts.")


def main():                                             # runs every section in order
    demo_hand()                                            # 1. by hand
    demo_pca()                                             # 2. SVD is PCA
    routes = demo_routes()                                 # 3. three routes
    ph = demo_photo()                                      # 4. the photograph
    comp = demo_complete()                                 # 5. matrix completion
    demo_lstsq()                                           # 6. least squares
    plot_results(routes, ph, comp)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
