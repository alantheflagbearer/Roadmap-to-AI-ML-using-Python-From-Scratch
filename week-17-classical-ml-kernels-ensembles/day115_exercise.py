"""
Day 115 Exercise -- k-Nearest Neighbours and PCA

The fourth of Week 17's parts for Day 118's build.

    sqdist(A, B)          every squared distance between the rows of A and of B
    knn_predict(Xtr, ytr, X, k)   majority vote of the k nearest; ties go to the class with the nearest neighbour
    pca_fit(X, k)         (centre, the k directions of largest variance, share of variance per direction)

Checked by hand on five points, and on the edge cases: k equal to the whole
training set predicts the majority class everywhere, a tie is broken by the
nearest neighbour, and PCA's directions are orthonormal.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sqdist(A, B):                                       # NEW: all squared distances at once
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def knn_predict(Xtr, ytr, X, k, chunk=500):             # NEW: Day 115's vote, unchanged
    n_classes = int(ytr.max()) + 1; out = []
    for s in range(0, len(X), chunk):
        D = sqdist(X[s:s + chunk], Xtr)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k] if k < len(ytr) else np.tile(np.arange(len(ytr)), (len(D), 1))
        votes = np.zeros((len(D), n_classes))
        for j in range(k):
            votes[np.arange(len(D)), ytr[idx[:, j]]] += 1
        d = np.take_along_axis(D, idx, 1)
        tie = (votes == votes.max(1, keepdims=True)).sum(1) > 1
        for r in np.where(tie)[0]:
            best = np.where(votes[r] == votes[r].max())[0]
            order = idx[r][np.argsort(d[r])]
            out_r = next(ytr[i] for i in order if ytr[i] in best); votes[r, out_r] += 0.5
        out.append(np.argmax(votes, 1))
    return np.concatenate(out)


def pca_fit(X, k):                                      # NEW: the k directions of largest variance, by SVD
    mu = X.mean(0); _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k], (s ** 2) / (s ** 2).sum()


X = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5]], dtype=float); y = np.array([0, 0, 1, 1, 1])
q = np.array([[0.4, 0.4], [5.5, 4.0]])
print(f"query (0.4, 0.4): k=1 -> {knn_predict(X, y, q[:1], 1)[0]}, k=3 -> {knn_predict(X, y, q[:1], 3)[0]}, k=5 -> {knn_predict(X, y, q[:1], 5)[0]} (by hand: 0, 0, 1)")
print(f"query (5.5, 4.0): k=1 -> {knn_predict(X, y, q[1:], 1)[0]} (by hand: 1)")
print(f"k = all 5 points: every query gets the majority class: {knn_predict(X, y, np.array([[0.0, 0.0], [9.0, 9.0]]), 5).tolist()}")
T = np.array([[0.0], [1.0], [3.0], [4.0]]); t = np.array([0, 1, 0, 1])
print(f"a 2-2 tie at x = 1.6 (k = 4): the nearest neighbour is x = 1, class 1 -> {knn_predict(T, t, np.array([[1.6]]), 4)[0]}")
rng = np.random.RandomState(0); Z = rng.normal(size=(200, 3)) @ np.diag([3.0, 1.0, 0.1])  # spreads 3, 1 and 0.1
mu, V, share = pca_fit(Z, 3)
print(f"PCA on spreads 3, 1, 0.1: variance shares {share.round(4).tolist()} (by hand about 9 : 1 : 0.01 of 10.01)")
print(f"directions are orthonormal: largest gap of V V^T from the identity {np.abs(V @ V.T - np.eye(3)).max():.1e}")
