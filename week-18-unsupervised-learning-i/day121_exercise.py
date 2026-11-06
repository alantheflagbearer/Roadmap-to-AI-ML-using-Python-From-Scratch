"""
Day 121 Exercise -- DBSCAN

The third of Week 18's parts.

    neighbours(X, eps)        for every point, the indices within eps (itself included)
    dbscan(X, eps, min_pts)   labels 0, 1, ... for clusters and -1 for noise, and which points are core
    k_distance(X, k)          each point's distance to its k-th nearest point, the guide for choosing eps

Checked by hand on eleven points on a line, and on the edge cases: an eps
smaller than every gap makes every point noise, an eps larger than the data
makes one cluster, and min_pts = 1 makes every point core, so nothing is noise.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this

NOISE = -1                                              # the label for noise points


def sqdist(A, B):                                       # every squared distance (Day 113's)
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def neighbours(X, eps, chunk=1000):                     # NEW: Day 121's version, unchanged
    out = []
    for s in range(0, len(X), chunk):
        D = sqdist(X[s:s + chunk], X)
        out += [np.where(row <= eps * eps)[0] for row in D]
    return out


def dbscan(X, eps, min_pts):                            # NEW: Day 121's version, unchanged
    N = neighbours(X, eps); core = np.array([len(nb) >= min_pts for nb in N])
    lab = np.full(len(X), NOISE); c = 0
    for i in range(len(X)):
        if lab[i] != NOISE or not core[i]:
            continue
        lab[i] = c; stack = [i]
        while stack:
            j = stack.pop()
            if core[j]:
                for v in N[j]:
                    if lab[v] == NOISE:
                        lab[v] = c; stack.append(v)
        c += 1
    return lab, core


def k_distance(X, k, chunk=1000):                       # NEW: Day 121's version, unchanged
    out = []
    for s in range(0, len(X), chunk):
        D = np.maximum(sqdist(X[s:s + chunk], X), 0)
        out.append(np.sqrt(np.partition(D, k - 1, axis=1)[:, k - 1]))
    return np.concatenate(out)


x = np.array([0, 0.5, 1, 1.5, 5, 5.4, 5.8, 9, 12, 12.3, 20.0]); X = x[:, None]
lab, core = dbscan(X, 0.6, 3)
print(f"eleven points, eps 0.6, min_pts 3: labels {lab.tolist()}")
print(f"core points: {x[core].tolist()}; by hand: 0.5, 1 and 5.4 (three points within 0.6 each)")
print(f"noise: {x[lab == NOISE].tolist()}; by hand: 9, 12, 12.3 and 20")
print(f"2nd-nearest distance (itself first) of 0, 9 and 12: {np.round(k_distance(X, 2)[[0, 7, 8]], 2).tolist()}; by hand: 0.5, 3.0 and 0.3")
print(f"eps 0.1: every point noise {bool(np.all(dbscan(X, 0.1, 2)[0] == NOISE))}; eps 100: one cluster {bool(np.all(dbscan(X, 100, 3)[0] == 0))}; "
      f"min_pts 1: no noise {bool(np.all(dbscan(X, 0.6, 1)[0] != NOISE))}")
