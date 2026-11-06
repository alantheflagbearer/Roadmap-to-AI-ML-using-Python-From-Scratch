"""
Day 119 Exercise -- k-Means, and Ways to Judge a Clustering

The first of Week 18's parts.

    sqdist(A, B)          every squared distance
    init_centres(X, k, rng, method)   "random" rows, or "k-means++"
    lloyd(X, C)           assign and update until nothing moves: (labels, centres, inertia per step)
    kmeans(X, k, seed, method, restarts)   the best of several starts
    silhouette(X, labels), ari(a, b), elbow(ks, inertia)

Checked by hand on six points, and on the edge cases: the adjusted Rand index
ignores how clusters are numbered, is 0 for a random labelling on average, and
the elbow finder picks the bend of a curve built to have one.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sqdist(A, B):                                       # every squared distance
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def init_centres(X, k, rng, method):                    # NEW: Day 119's starts, unchanged
    if method == "random":
        return X[rng.choice(len(X), k, replace=False)].copy()
    C = [X[rng.randint(len(X))]]
    for _ in range(k - 1):
        d = np.maximum(sqdist(X, np.array(C)).min(1), 0)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    return np.array(C)


def lloyd(X, C, max_iter=300):                          # NEW: Day 119's loop, unchanged
    history = []
    for _ in range(max_iter):
        D = sqdist(X, C); lab = D.argmin(1); history.append(float(np.maximum(D.min(1), 0).sum()))
        new = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(len(C))])
        if np.allclose(new, C):
            break
        C = new
    return lab, C, history


def kmeans(X, k, seed, method="k-means++", restarts=10):  # NEW: the best of several starts
    rng = np.random.RandomState(seed); best = None
    for _ in range(restarts):
        lab, C, h = lloyd(X, init_centres(X, k, rng, method))
        if best is None or h[-1] < best[2][-1]:
            best = (lab, C, h)
    return best


def silhouette(X, lab):                                 # NEW: (b - a) / max(a, b), averaged
    D = np.sqrt(np.maximum(sqdist(X, X), 0)); s = np.zeros(len(X)); ks = np.unique(lab)
    for i in range(len(X)):
        same = lab == lab[i]; n = same.sum() - 1
        if n == 0:
            continue
        a = D[i, same].sum() / n
        b = min(D[i, lab == c].mean() for c in ks if c != lab[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def ari(a, b):                                          # NEW: the adjusted Rand index
    _, ia = np.unique(a, return_inverse=True); _, ib = np.unique(b, return_inverse=True)
    T = np.zeros((ia.max() + 1, ib.max() + 1)); np.add.at(T, (ia, ib), 1)
    c2 = lambda x: x * (x - 1) / 2
    s = c2(T).sum(); sa = c2(T.sum(1)).sum(); sb = c2(T.sum(0)).sum(); e = sa * sb / c2(len(a))
    return float((s - e) / (0.5 * (sa + sb) - e))


def elbow(ks, inertia):                                 # NEW: the k furthest below the chord
    x = np.array(ks, float); y = np.array(inertia, float)
    x, y = (x - x[0]) / (x[-1] - x[0]), (y - y[-1]) / (y[0] - y[-1])
    return ks[int(np.argmax((1 - x) - y))]


X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
lab, C, h = lloyd(X, np.array([[1.0], [2.0]]))
print(f"six points from a poor start: groups {[X[lab == j, 0].tolist() for j in range(2)]}, centres {C[:, 0].tolist()}, inertia by step {[round(v, 2) for v in h]}")
print(f"by hand: centres 2 and 11, inertia (1 + 0 + 1) + (1 + 0 + 1) = 4")
print(f"ARI ignores the numbering: [0,0,1,1] against [1,1,0,0] -> {ari(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])):.3f}")
rng = np.random.RandomState(0); t = rng.randint(0, 3, 3000)
print(f"a random labelling against the truth, averaged over 20 tries: {np.mean([ari(t, rng.randint(0, 3, 3000)) for _ in range(20)]):+.4f}")
print(f"silhouette of the six points in their two groups: {silhouette(X, lab):.4f} (tight groups far apart: near 1)")
print(f"elbow of a curve that falls fast until k = 4 and slowly after: k = {elbow([1, 2, 3, 4, 5, 6, 7, 8], [100, 60, 30, 10, 9, 8, 7, 6])}")
