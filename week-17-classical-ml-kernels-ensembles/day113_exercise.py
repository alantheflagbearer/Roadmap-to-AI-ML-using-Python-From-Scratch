"""
Day 113 Exercise -- Kernels and Random Fourier Features

The second of Week 17's parts for Day 118's build. Each kernel returns the full
matrix K(A, B), ready for Day 112's smo.

    linear(A, B), poly(A, B, degree, c), rbf(A, B, gamma)
    phi_poly2(X)          the explicit features a degree-2 polynomial kernel stands for
    rff(X, D, gamma, seed)   a map z with z(x).z(x') close to rbf(x, x')

Checked on a small table: the polynomial kernel equals the explicit dot
products, the RBF matrix has ones on its diagonal, is symmetric and has no
negative eigenvalues, and random features approach it as D grows.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def linear(A, B):                                       # NEW: plain dot products
    return A @ B.T


def poly(A, B, degree=2, c=1.0):                        # NEW: (x.z + c)^degree
    return (A @ B.T + c) ** degree


def sqdist(A, B):                                       # all squared distances at once
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def rbf(A, B, gamma):                                   # NEW: exp(-gamma |x - z|^2)
    return np.exp(-gamma * np.maximum(sqdist(A, B), 0))


def phi_poly2(X):                                       # NEW: the features behind (x.z + 1)^2
    n, p = X.shape; cols = [np.ones(n)] + [np.sqrt(2) * X[:, i] for i in range(p)]
    cols += [X[:, i] ** 2 for i in range(p)]
    cols += [np.sqrt(2) * X[:, i] * X[:, j] for i in range(p) for j in range(i + 1, p)]
    return np.column_stack(cols)


def rff(X, D, gamma, seed):                             # NEW: random Fourier features for the RBF kernel
    rng = np.random.RandomState(seed)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(X.shape[1], D))
    c = rng.uniform(0, 2 * np.pi, D)
    return lambda Z: np.sqrt(2.0 / D) * np.cos(Z @ W + c)


X = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5], [-2.0, 1.0]])  # four points in two dimensions
F = phi_poly2(X)
print(f"two features become {F.shape[1]}: 1, sqrt2 x1, sqrt2 x2, x1^2, x2^2, sqrt2 x1 x2")
print(f"(x.z + 1)^2 against phi(x).phi(z): largest gap {np.abs(poly(X, X) - F @ F.T).max():.1e}")
print(f"by hand for the first two points: x.z = 1*0 + 2*(-1) = -2, so (x.z + 1)^2 = 1; kernel {poly(X[:1], X[1:2])[0, 0]:.1f}")
K = rbf(X, X, 0.5); e = np.linalg.eigvalsh(K)
print(f"RBF, gamma 0.5: diagonal {np.diag(K).round(4).tolist()}, symmetric {bool(np.allclose(K, K.T))}, smallest eigenvalue {e.min():.4f}")
print(f"by hand: |x1 - x2|^2 = 1 + 9 = 10, exp(-0.5 * 10) = {np.exp(-5):.6f}; kernel {K[0, 1]:.6f}")
for D in (10, 100, 1000, 10000):
    z = rff(X, D, 0.5, 0); Z = z(X)
    print(f"random features D = {D:>5}: largest error against the exact kernel {np.abs(Z @ Z.T - K).max():.4f}")
