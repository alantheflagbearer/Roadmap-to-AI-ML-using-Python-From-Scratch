"""
Day 124 Exercise -- The Singular Value Decomposition

The sixth of Week 18's parts.

    svd_via_eig(A)            U, s, V through the eigenvectors of A^T A
    svd_power(A, k)           the top k singular triplets by alternating multiplication
    low_rank(U, s, V, k)      the best rank-k approximation
    complete(M, mask, k)      fill unknown entries with a rank-k fit, repeatedly
    pinv_solve(A, b, k)       least squares through the SVD, optionally truncated

Checked by hand on a 2 x 2 matrix, and on cases with a known answer: the rank-k
error equals the dropped singular values, an exactly rank-2 matrix with half its
entries hidden is recovered, and the pseudoinverse agrees with lstsq.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def svd_via_eig(A):                                     # NEW: Day 124's version, unchanged
    vals, V = np.linalg.eigh(A.T @ A)
    o = np.argsort(vals)[::-1]; vals, V = vals[o], V[:, o]
    s = np.sqrt(np.maximum(vals, 0))
    U = (A @ V) / np.where(s > 0, s, 1)
    return U, s, V


def svd_power(A, k, iters=3000, tol=1e-12, seed=0):     # NEW: Day 124's version, unchanged
    rng = np.random.RandomState(seed); U, s, V = [], [], []
    for _ in range(k):
        v = rng.normal(size=A.shape[1]); v /= np.linalg.norm(v)
        for _ in range(iters):
            u = A @ v - sum(sj * uj * (vj @ v) for sj, uj, vj in zip(s, U, V))
            u /= np.linalg.norm(u)
            w = A.T @ u - sum(sj * vj * (uj @ u) for sj, uj, vj in zip(s, U, V))
            sigma = np.linalg.norm(w); w /= sigma
            done = np.linalg.norm(w - v) < tol
            v = w
            if done:
                break
        U.append(u); s.append(sigma); V.append(v)
    return np.array(U).T, np.array(s), np.array(V).T


def low_rank(U, s, V, k):                               # NEW: Day 124's version, unchanged
    return (U[:, :k] * s[:k]) @ V[:, :k].T


def complete(M, mask, k, iters=2000, tol=1e-6):          # NEW: Day 124's version, unchanged
    X = np.where(mask, M, M[mask].mean())
    for t in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :k] * s[:k]) @ Vt[:k]
        change = np.linalg.norm((L - X)[~mask]) / max(np.linalg.norm(X[~mask]), 1e-12)
        X = np.where(mask, M, L)
        if change < tol:
            break
    return X, t + 1


def pinv_solve(A, b, k=None):                           # NEW: Day 124's version, unchanged
    U, s, Vt = np.linalg.svd(A, full_matrices=False); k = len(s) if k is None else k
    return Vt[:k].T @ ((U[:, :k].T @ b) / s[:k])


A = np.array([[3.0, 0.0], [4.0, 5.0]]); U, s, V = svd_via_eig(A)
print(f"[[3, 0], [4, 5]]: singular values {np.round(s, 4).tolist()}; by hand sqrt(45) = {np.sqrt(45):.4f}, sqrt(5) = {np.sqrt(5):.4f}")
rng = np.random.RandomState(0); B = rng.normal(size=(30, 20)); U, s, Vt = np.linalg.svd(B, full_matrices=False); V = Vt.T
e = np.linalg.norm(B - low_rank(U, s, V, 5)); print(f"rank-5 error {e:.6f}; square root of the dropped singular values squared {np.sqrt((s[5:] ** 2).sum()):.6f}")
pu, ps, pv = svd_power(B, 3); print(f"power iteration, top 3: {np.round(ps, 6).tolist()}; np.linalg.svd: {np.round(s[:3], 6).tolist()}")
R = rng.normal(size=(40, 2)) @ rng.normal(size=(2, 30)); mask = rng.uniform(size=R.shape) < 0.5
X, t = complete(R, mask, 2); print(f"exactly rank-2 matrix, half hidden: largest error on hidden entries after {t} rounds {np.abs(X - R)[~mask].max():.1e}")
b = rng.normal(size=30); print(f"pseudoinverse against lstsq: largest gap {np.abs(pinv_solve(B, b) - np.linalg.lstsq(B, b, rcond=None)[0]).max():.1e}")
