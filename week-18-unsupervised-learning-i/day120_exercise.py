"""
Day 120 Exercise -- Agglomerative Clustering

The second of Week 18's parts.

    agglomerate(X, method)   the merge history (a, b, height, size), for "single", "complete", "average" or "ward"
    cut(Z, n, k)             labels from undoing the last k - 1 merges
    cophenetic(Z, n)         the height at which each pair of points first share a cluster

Checked by hand on five points on a line, and on the edge cases: cutting into
n clusters gives every point its own label, cutting into 1 gives one label,
and merge heights never decrease for single, complete, average or Ward linkage.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sqdist(A, B):                                       # every squared distance
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def agglomerate(X, method):                             # NEW: Day 120's Lance-Williams version, unchanged
    n = len(X); D = np.sqrt(np.maximum(sqdist(X, X), 0))
    if method == "ward":
        D = D ** 2
    D[np.arange(n), np.arange(n)] = np.inf
    size = np.ones(n); ids = np.arange(n); alive = np.ones(n, bool); Z = []
    for step in range(n - 1):
        M = np.where(alive[:, None] & alive[None, :], D, np.inf)
        i, j = np.unravel_index(np.argmin(M), M.shape); i, j = min(i, j), max(i, j)
        d = M[i, j]; ni, nj = size[i], size[j]
        Z.append((min(ids[i], ids[j]), max(ids[i], ids[j]), np.sqrt(d) if method == "ward" else d, ni + nj))
        nk = size
        if method == "single":
            new = np.minimum(D[i], D[j])
        elif method == "complete":
            new = np.maximum(D[i], D[j])
        elif method == "average":
            new = (ni * D[i] + nj * D[j]) / (ni + nj)
        else:
            new = ((ni + nk) * D[i] + (nj + nk) * D[j] - nk * D[i, j]) / (ni + nj + nk)
        D[i, :] = new; D[:, i] = new; D[i, i] = np.inf
        alive[j] = False; size[i] = ni + nj; ids[i] = n + step
    return np.array(Z)


def cut(Z, n, k):                                       # NEW: labels for k clusters
    parent = np.arange(2 * n - 1)
    for s, (a, b, _, _) in enumerate(Z[:n - k]):
        parent[int(a)] = parent[int(b)] = n + s
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    r = np.array([root(i) for i in range(n)])
    return np.unique(r, return_inverse=True)[1]


def cophenetic(Z, n):                                   # NEW: merge height for every pair
    members = {i: [i] for i in range(n)}; C = np.zeros((n, n))
    for s, (a, b, h, _) in enumerate(Z):
        A, B = members.pop(int(a)), members.pop(int(b))
        C[np.ix_(A, B)] = h; C[np.ix_(B, A)] = h
        members[n + s] = A + B
    return C


X = np.array([[0.0], [1.0], [3.0], [7.0], [8.5]])
Z = agglomerate(X, "single")
print(f"single linkage on 0, 1, 3, 7, 8.5: merges {[(int(a), int(b), round(float(h), 2)) for a, b, h, _ in Z]}")
print("by hand: 0-1 at 1, 7-8.5 at 1.5, then 3 joins {0,1} at 2, and the two groups meet at 4 (3 to 7)")
print(f"cut into 2: {cut(Z, 5, 2).tolist()}; into 5: {cut(Z, 5, 5).tolist()}; into 1: {cut(Z, 5, 1).tolist()}")
C = cophenetic(Z, 5)
print(f"cophenetic height between 0 and 8.5: {C[0, 4]:.1f} (the last merge); between 0 and 3: {C[0, 2]:.1f}")
rng = np.random.RandomState(0); R = rng.normal(size=(40, 3))
print("merge heights never decrease: " + ", ".join(f"{m} {bool(np.all(np.diff(agglomerate(R, m)[:, 2]) >= -1e-12))}" for m in ("single", "complete", "average", "ward")))
