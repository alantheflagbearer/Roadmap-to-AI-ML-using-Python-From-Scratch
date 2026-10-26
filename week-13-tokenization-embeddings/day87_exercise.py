"""
Day 87 Exercise -- PCA in One Function

The fourth of Week 13's parts for Day 90's tiny GPT-style model: after
training, Day 90 draws its learned character table in two dimensions with
this function.

    pca(X, k)  ->  Z (n, k) the data along the k directions of largest spread,
                   the directions themselves (d, k), and each direction's share of the spread

Checked four ways: the directions are orthonormal, the result matches NumPy's
SVD, keeping all directions loses nothing, and on data that really lies near
a 2-D plane, two directions keep almost all of the spread.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def pca(X, k):                                          # NEW: the k directions of largest spread, and the data in them
    Xc = X - X.mean(axis=0)                               # centre every column
    C = Xc.T @ Xc / (len(X) - 1)                          # covariance matrix (d, d)
    vals, vecs = np.linalg.eigh(C)                        # eigen-decomposition of a symmetric matrix, smallest first
    order = np.argsort(vals)[::-1]                        # largest spread first
    vals, vecs = vals[order], vecs[:, order]              # reordered
    return Xc @ vecs[:, :k], vecs[:, :k], vals / vals.sum()  # projections, directions, share of spread per direction


rng = np.random.RandomState(0)                          # fixed rng
X = rng.randn(200, 6) @ rng.randn(6, 6)                   # 200 points in 6-D, correlated columns
Z, W, share = pca(X, 6)                                   # every direction
print(f"directions orthonormal: largest |W^T W - I| = {np.abs(W.T @ W - np.eye(6)).max():.1e}")
U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)  # NumPy's answer
print(f"against SVD: directions {np.abs(np.abs(W) - np.abs(Vt.T)).max():.1e}, projections {np.abs(np.abs(Z) - np.abs(U * S)).max():.1e}")
print(f"all 6 directions: reconstruction error {np.abs(Z @ W.T + X.mean(axis=0) - X).max():.1e}; shares sum to {share.sum():.6f}")
plane = rng.randn(300, 2) @ rng.randn(2, 6) + 0.05 * rng.randn(300, 6)  # 6-D points near a 2-D plane
_, _, s2 = pca(plane, 2)                                  # its shares
print(f"points near a 2-D plane: the first two directions keep {s2[:2].sum():.4f} of the spread; the rest "
      + ", ".join(f"{v:.4f}" for v in s2[2:]))
