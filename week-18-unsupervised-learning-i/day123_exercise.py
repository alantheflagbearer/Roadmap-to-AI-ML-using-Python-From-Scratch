"""
Day 123 Exercise -- Principal Component Analysis

The fifth of Week 18's parts.

    pca_fit(X)                       mean, eigenvalues (largest first) and eigenvectors of the covariance
    transform(X, mu, V, k)           coordinates along the first k components
    reconstruct(Z, mu, V, k)         back to the original space
    power_iteration(C, k)            the top k eigenpairs with matrix-vector products only

Checked by hand: points on a line have one component that explains everything;
an axis-aligned cloud has the axes as its components and the variances as its
eigenvalues; keeping every component reconstructs the data exactly; and power
iteration agrees with eigh.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def pca_fit(X):                                         # NEW: Day 123's version, unchanged
    mu = X.mean(0); A = X - mu
    C = A.T @ A / (len(X) - 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    return mu, np.maximum(vals[order], 0), vecs[:, order]


def transform(X, mu, V, k):                             # NEW: Day 123's version, unchanged
    return (X - mu) @ V[:, :k]


def reconstruct(Z, mu, V, k):                           # NEW: Day 123's version, unchanged
    return Z @ V[:, :k].T + mu


def power_iteration(C, k, iters=5000, tol=1e-12, seed=0):  # NEW: Day 123's version, unchanged
    rng = np.random.RandomState(seed); C = C.copy(); vals, vecs, steps = [], [], []
    for _ in range(k):
        v = rng.normal(size=len(C)); v /= np.linalg.norm(v)
        for t in range(1, iters + 1):
            w = C @ v; w /= np.linalg.norm(w)
            done = np.linalg.norm(w - v) < tol
            v = w
            if done:
                break
        lam = float(v @ C @ v); vals.append(lam); vecs.append(v); steps.append(t)
        C -= lam * np.outer(v, v)
    return np.array(vals), np.array(vecs).T, steps


t = np.arange(5.0); X = np.c_[t, 2 * t]                 # five points on the line y = 2x
mu, vals, V = pca_fit(X)
print(f"points on y = 2x: eigenvalues {np.round(vals, 6).tolist()}; first component {np.round(np.abs(V[:, 0]), 4).tolist()}, "
      f"by hand (1, 2)/sqrt(5) = {np.round(np.array([1, 2]) / np.sqrt(5), 4).tolist()}")
rng = np.random.RandomState(0); Y = rng.normal(0, 1, (2000, 3)) * np.array([3.0, 1.0, 0.5])  # spreads 3, 1 and 0.5 along the axes
mu, vals, V = pca_fit(Y)
print(f"axis-aligned cloud with spreads 3, 1, 0.5: eigenvalues {np.round(vals, 3).tolist()} (by hand about 9, 1, 0.25); "
      f"components are the axes: {bool(np.allclose(np.abs(V), np.eye(3), atol=0.05))}")
print(f"all 3 components rebuild the data exactly: {bool(np.allclose(reconstruct(transform(Y, mu, V, 3), mu, V, 3), Y))}; "
      f"1 component leaves a mean squared distance per point of {((Y - reconstruct(transform(Y, mu, V, 1), mu, V, 1)) ** 2).sum(1).mean():.3f} (by hand about 1 + 0.25)")
pv, pV, steps = power_iteration(np.cov(Y.T), 3)
print(f"power iteration: eigenvalues {np.round(pv, 3).tolist()}, the same as eigh: {bool(np.allclose(pv, vals))}; steps {steps}")
