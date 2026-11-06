"""
Day 125: The Week 18 Build -- t-SNE and UMAP From Scratch

PCA (Day 123) draws high-dimensional data on a flat sheet by keeping the
directions of most variance. t-SNE and UMAP do something different: they try
to keep each point's NEIGHBOURS as neighbours on the page, and give up on
distances further away. Both are written from scratch here, checked against
scikit-learn's t-SNE and the umap-learn library, and compared with PCA. The
week's six exercise parts are copied in verbatim (one banner per file) and do
the rest: PCA starts t-SNE, and k-means, Ward linkage, a Gaussian mixture and
DBSCAN cluster the maps.

  1. Checks: the parts are verbatim.
  2. t-SNE's pieces: perplexity calibration, and a gradient check.
  3. t-SNE on digits, against scikit-learn.
  4. UMAP on digits, against umap-learn.
  5. Three maps compared: what each keeps.
  6. Clustering the maps with the week's parts.
  7. Warnings: perplexity, and clusters drawn from pure noise.

Data: the handwritten digits, bundled with scikit-learn (labels used only to
judge), and random points made on purpose.

Every line written for today carries its own comment, continuing Day 52's
convention; the copied parts keep their own.
"""
import ast                                             # ast: reading function sources out of the exercise files
import os                                              # os: paths
import time                                            # time: what each map costs

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_digits               # bundled dataset
from sklearn.manifold import TSNE, trustworthiness     # used ONLY as outside checks
import umap                                            # umap-learn: used ONLY as an outside check
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
NOISE = -1                                              # Day 121's noise label, which its part needs
LOG2PI = np.log(2 * np.pi)                              # Day 122's constant, which its part needs
PERPLEXITY = 30                                         # t-SNE's effective number of neighbours
NEIGHBOURS = 15                                         # UMAP's neighbours
MIN_DIST = 0.1                                          # UMAP's minimum distance on the page


SKIP = {(120, 'sqdist'), (121, 'sqdist')}                 # parts left out: the same code as Day 119's sqdist


# =============================================================================
# from day119_exercise.py, verbatim: sqdist, init_centres, lloyd, kmeans, silhouette, ari, elbow
# =============================================================================
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


# =============================================================================
# from day120_exercise.py, verbatim: agglomerate, cut, cophenetic
# =============================================================================
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


# =============================================================================
# from day121_exercise.py, verbatim: neighbours, dbscan, k_distance
# =============================================================================
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


# =============================================================================
# from day122_exercise.py, verbatim: log_gauss, logsumexp, e_step, m_step
# =============================================================================
def log_gauss(X, mu, S):                                # NEW: Day 122's version, unchanged
    L = np.linalg.cholesky(S)
    Z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (Z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * LOG2PI


def logsumexp(A):                                       # NEW: Day 122's version, unchanged
    m = A.max(1, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(1, keepdims=True)))[:, 0]


def e_step(X, w, M, S):                                 # NEW: Day 122's version, unchanged
    logp = np.stack([np.log(w[j]) + log_gauss(X, M[j], S[j]) for j in range(len(w))], 1)
    ll = logsumexp(logp)
    return np.exp(logp - ll[:, None]), float(ll.sum())


def m_step(X, R, kind="full", reg=1e-6):                # NEW: Day 122's version, unchanged
    Nk = R.sum(0) + 1e-300; w = Nk / len(X); M = (R.T @ X) / Nk[:, None]; d = X.shape[1]; S = []
    for j in range(R.shape[1]):
        D = X - M[j]; Sj = (R[:, j, None] * D).T @ D / Nk[j]
        if kind == "diag":
            Sj = np.diag(np.diag(Sj))
        elif kind == "spherical":
            Sj = np.eye(d) * np.trace(Sj) / d
        S.append(Sj + reg * np.eye(d))
    return w, M, np.array(S)


# =============================================================================
# from day123_exercise.py, verbatim: pca_fit, transform, reconstruct, power_iteration
# =============================================================================
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


# =============================================================================
# from day124_exercise.py, verbatim: svd_via_eig, svd_power, low_rank, complete, pinv_solve
# =============================================================================
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


# =============================================================================
# t-SNE -- NEW
# =============================================================================
def p_conditional(D2, perplexity, iters=64, tol=1e-6):  # NEW: P(j | i) with each row's width set so its perplexity is the target
    n = len(D2); beta = np.ones(n); lo = np.zeros(n); hi = np.full(n, np.inf); target = np.log(perplexity)
    off = ~np.eye(n, dtype=bool)                          # a point is not its own neighbour
    E = np.where(off, D2 - np.where(off, D2, np.inf).min(1, keepdims=True), 0.0)  # each row shifted by its smallest distance (it cancels below)
    for _ in range(iters):                               # binary search on beta = 1 / (2 sigma^2), every row at once
        P = np.where(off, np.exp(-E * beta[:, None]), 0.0); s = P.sum(1)
        H = np.log(s) + beta * (P * E).sum(1) / s           # the entropy of each row, in nats
        too_flat = H > target                               # entropy too high: the row is too spread out, so raise beta
        lo = np.where(too_flat, beta, lo); hi = np.where(too_flat, hi, beta)
        beta = np.where(np.isinf(hi), beta * 2, (lo + hi) / 2)
        if np.abs(H - target).max() < tol:
            break
    return P / s[:, None], np.exp(H)                      # rows sum to 1; each row's perplexity


def joint_p(X, perplexity):                             # NEW: symmetric P(i, j), summing to 1
    D2 = np.maximum(sqdist(X, X), 0); Pc, perp = p_conditional(D2, perplexity)
    return np.maximum((Pc + Pc.T) / (2 * len(X)), 1e-12), perp


def tsne_grad(Y, P):                                    # NEW: the KL divergence and its gradient for a layout Y
    num = 1.0 / (1.0 + np.maximum(sqdist(Y, Y), 0)); np.fill_diagonal(num, 0)  # Student-t similarities
    Q = np.maximum(num / num.sum(), 1e-12); PQ = (P - Q) * num
    grad = 4 * (PQ.sum(1)[:, None] * Y - PQ @ Y)          # 4 sum_j (p_ij - q_ij)(y_i - y_j)/(1 + |y_i - y_j|^2)
    return float((P * np.log(P / Q)).sum()), grad


def tsne(X, perplexity=PERPLEXITY, iters=1000, seed=0, exaggeration=12.0, exag_iters=250):  # NEW: van der Maaten's recipe
    n = len(X); P, _ = joint_p(X, perplexity); lr = max(n / exaggeration / 4, 50)  # scikit-learn's 'auto' learning rate
    mu, vals, V = pca_fit(X); Y = transform(X, mu, V, 2); Y = Y / Y[:, 0].std() * 1e-4  # Day 123's PCA, scaled tiny, as the start
    upd = np.zeros_like(Y); gains = np.ones_like(Y); kl = []
    for t in range(iters):
        ex = exaggeration if t < exag_iters else 1.0       # early exaggeration: pull neighbours together first
        _, g = tsne_grad(Y, P * ex)
        gains = np.where(upd * g < 0, gains + 0.2, gains * 0.8).clip(0.01)  # per-coordinate step sizes: grow while the direction holds
        upd = (0.5 if t < exag_iters else 0.8) * upd - lr * gains * g  # momentum
        Y = Y + upd
        if (t + 1) % 50 == 0:
            kl.append(tsne_grad(Y, P)[0])
    return Y, kl


# =============================================================================
# UMAP -- NEW (a compact version: dense matrices, and whole-epoch updates instead of one edge at a time)
# =============================================================================
def find_ab(min_dist=MIN_DIST, spread=1.0):            # NEW: a, b so that 1 / (1 + a d^(2b)) follows UMAP's target curve
    d = np.linspace(0, 3 * spread, 300); f = np.where(d < min_dist, 1.0, np.exp(-(d - min_dist) / spread))
    best = (np.inf, None)
    for b in np.linspace(0.5, 1.5, 201):                  # a fine grid over b; for each b, a grid over a
        for a in np.exp(np.linspace(np.log(0.5), np.log(5.0), 201)):
            e = ((1 / (1 + a * d ** (2 * b)) - f) ** 2).sum()
            if e < best[0]:
                best = (e, (a, b))
    return best[1]


def fuzzy_graph(X, k=NEIGHBOURS):                       # NEW: UMAP's weighted k-nearest-neighbour graph, made symmetric
    n = len(X); D = np.sqrt(np.maximum(sqdist(X, X), 0)); np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, 1)[:, :k]; d = np.take_along_axis(D, idx, 1)  # each point's k nearest, and their distances
    rho = d[:, 0]; lo = np.zeros(n); hi = np.full(n, np.inf); sig = np.ones(n); target = np.log2(k)
    for _ in range(64):                                   # binary search: sum_j exp(-(d_ij - rho_i) / sigma_i) = log2(k)
        s = np.exp(-np.maximum(d - rho[:, None], 0) / sig[:, None]).sum(1)
        big = s > target                                  # too much weight: shrink sigma
        hi = np.where(big, sig, hi); lo = np.where(big, lo, sig); sig = np.where(np.isinf(hi), sig * 2, (lo + hi) / 2)
    W = np.zeros((n, n)); W[np.arange(n)[:, None], idx] = np.exp(-np.maximum(d - rho[:, None], 0) / sig[:, None])
    return W + W.T - W * W.T                              # fuzzy union: an edge is there if either end sees it


def spectral_init(W, dims=2):                           # NEW: the start: the graph's smoothest non-trivial directions
    deg = W.sum(1); Dm = 1 / np.sqrt(deg); L = np.eye(len(W)) - Dm[:, None] * W * Dm[None, :]  # normalised Laplacian
    vals, vecs = np.linalg.eigh(L); Y = vecs[:, 1:dims + 1]  # skip the first (constant) eigenvector
    return 10 * Y / np.abs(Y).max()                       # scaled to [-10, 10], as umap-learn does


def umap_layout(W, a, b, epochs=500, neg=5, seed=0):    # NEW: attract along edges, repel random pairs, with a shrinking step
    rng = np.random.RandomState(seed); Y = spectral_init(W); n = len(W)
    i, j = np.nonzero(np.triu(W)); w = W[i, j] / W[i, j].max()  # each edge once, its weight scaled to (0, 1]
    for e in range(epochs):
        alpha = 1.0 - e / epochs                            # the step shrinks to 0
        use = rng.uniform(size=len(w)) < w                  # heavier edges are used in more epochs
        a_i, a_j = i[use], j[use]; diff = Y[a_i] - Y[a_j]; d2 = (diff ** 2).sum(1)
        c = (-2 * a * b * d2 ** (b - 1)) / (1 + a * d2 ** b); c[d2 == 0] = 0
        g = np.clip(c[:, None] * diff, -4, 4)                 # attraction, clipped as umap-learn does
        dY = np.zeros_like(Y); np.add.at(dY, a_i, alpha * g); np.add.at(dY, a_j, -alpha * g)
        r_i = np.repeat(a_i, neg); r_k = rng.randint(n, size=len(r_i))  # negative samples: random partners
        diff = Y[r_i] - Y[r_k]; d2 = (diff ** 2).sum(1)
        c = 2 * b / ((0.001 + d2) * (1 + a * d2 ** b)); c[r_i == r_k] = 0
        np.add.at(dY, r_i, alpha * np.clip(c[:, None] * diff, -4, 4))  # repulsion
        Y = Y + dY
    return Y


def umap_embed(X, k=NEIGHBOURS, min_dist=MIN_DIST, epochs=500, seed=0):  # NEW: graph, curve and layout
    a, b = find_ab(min_dist); return umap_layout(fuzzy_graph(X, k), a, b, epochs, seed=seed)


# =============================================================================
# JUDGING A MAP -- NEW
# =============================================================================
def knn_sets(X, k):                                     # each point's k nearest others
    D = np.maximum(sqdist(X, X), 0); np.fill_diagonal(D, np.inf); return np.argsort(D, 1)[:, :k], D


def trust(X, Y, k=10):                                  # NEW: trustworthiness: penalises map neighbours that were far apart
    n = len(X); _, DX = knn_sets(X, k); ranks = np.argsort(np.argsort(DX, 1), 1) + 1  # rank of every point in each row of the original
    NY, _ = knn_sets(Y, k); r = np.take_along_axis(ranks, NY, 1)
    return float(1 - 2 / (n * k * (2 * n - 3 * k - 1)) * np.maximum(r - k, 0).sum())


def kept(X, Y, k=10):                                   # NEW: share of each point's k nearest neighbours that stay among its k nearest
    NX, _ = knn_sets(X, k); NY, _ = knn_sets(Y, k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(NX, NY)]))


def centre_corr(X, Y, y):                               # NEW: do the digit groups keep their relative distances?
    cx = np.array([X[y == c].mean(0) for c in range(10)]); cy = np.array([Y[y == c].mean(0) for c in range(10)])
    iu = np.triu_indices(10, 1); return float(np.corrcoef(np.sqrt(sqdist(cx, cx)[iu]), np.sqrt(np.maximum(sqdist(cy, cy), 0)[iu]))[0, 1])


def gmm(X, k, seed=0, reg=1e-6, iters=300):            # Day 122's EM, rebuilt from its two parts and Day 119's k-means
    R = np.eye(k)[kmeans(X, k, seed)[0]]; prev = -np.inf
    for _ in range(iters):
        w, M, S = m_step(X, R, "full", reg); R, ll = e_step(X, w, M, S)
        if ll - prev < 1e-6 * len(X):
            break
        prev = ll
    return R.argmax(1)


def cluster_all(Z, y):                                  # the week's four clusterers on one representation
    eps = float(np.percentile(k_distance(Z, 10), 90))     # eps: the 90th percentile of the distance to the 10th nearest point
    out = {"k-means": kmeans(Z, 10, 0)[0], "Ward": cut(agglomerate(Z, "ward"), len(Z), 10), "Gaussian mixture": gmm(Z, 10),
           "DBSCAN": dbscan(Z, eps, 10)[0]}
    return {m: ari(lab, y) for m, lab in out.items()}, out["DBSCAN"], eps


# =============================================================================
# DEMOS
# =============================================================================
def demo_checks():
    print("=" * 70)
    print("1. CHECKS -- the parts are verbatim")
    print("=" * 70)
    me = open(os.path.abspath(__file__), encoding="utf-8").read(); found, total, dup = 0, 0, []
    for d in range(119, 125):
        src = open(os.path.join(HERE, f"day{d}_exercise.py"), encoding="utf-8").read(); lines = src.split("\n")
        for node in ast.parse(src).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if (d, node.name) in SKIP:
                    dup.append(f"{node.name} (day{d})"); continue
                total += 1; found += "\n".join(lines[node.lineno - 1:node.end_lineno]) in me
    print(f"functions from day119_exercise.py to day124_exercise.py found here, character for character: {found} of {total}")
    print(f"left out as the same code as Day 119's sqdist: {', '.join(dup)}")
    print("Actual output from this lesson's run.")


def demo_pieces(X):
    print("\n" + "=" * 70)
    print("2. t-SNE'S PIECES -- perplexity calibration, and a gradient check")
    print("=" * 70)
    P, perp = joint_p(X, PERPLEXITY)
    print(f"digits: every point's perplexity after the binary search: {perp.min():.4f} to {perp.max():.4f} (target {PERPLEXITY}); P sums to {P.sum():.6f}")
    rng = np.random.RandomState(0); Xs = X[:40]; Ps, _ = joint_p(Xs, 10); Y = rng.normal(size=(40, 2)); kl, g = tsne_grad(Y, Ps); h = 1e-6; num = np.zeros_like(Y)
    for i in range(40):
        for c in range(2):
            Yp = Y.copy(); Yp[i, c] += h; Ym = Y.copy(); Ym[i, c] -= h
            num[i, c] = (tsne_grad(Yp, Ps)[0] - tsne_grad(Ym, Ps)[0]) / (2 * h)  # nudge each coordinate both ways
    print(f"40 digits, a random layout: KL {kl:.4f}; the gradient formula against finite differences, largest gap {np.abs(num - g).max():.1e} "
          f"(largest component {np.abs(g).max():.3f})")
    print("Actual output from this lesson's run.")


def demo_tsne(X, y):
    print("\n" + "=" * 70)
    print(f"3. t-SNE ON DIGITS -- perplexity {PERPLEXITY}, 1,000 steps, against scikit-learn (exact method, the same PCA start)")
    print("=" * 70)
    t0 = time.perf_counter(); Y, kl = tsne(X); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); sk = TSNE(2, perplexity=PERPLEXITY, method="exact", init="pca", random_state=0, max_iter=1000).fit(X); t2 = time.perf_counter() - t0
    print(f"  {'':<14}{'final KL':>10}{'trustworthiness':>17}{'seconds':>9}")
    print(f"  {'here':<14}{kl[-1]:>10.4f}{trust(X, Y):>17.4f}{t1:>9.1f}")
    print(f"  {'scikit-learn':<14}{sk.kl_divergence_:>10.4f}{trust(X, sk.embedding_):>17.4f}{t2:>9.1f}")
    print(f"KL every 200 steps here: {[round(v, 4) for v in kl[3::4]]}; early exaggeration ends at step 250")
    print(f"trustworthiness written here against scikit-learn's function, on this map: {trust(X, Y):.6f} against {trustworthiness(X, Y, n_neighbors=10):.6f}")
    print("Actual output from this lesson's run.")
    return Y


def demo_umap(X, y):
    print("\n" + "=" * 70)
    print(f"4. UMAP ON DIGITS -- {NEIGHBOURS} neighbours, min_dist {MIN_DIST}, 500 epochs, against umap-learn")
    print("=" * 70)
    a, b = find_ab(); ra, rb = umap.umap_.find_ab_params(1.0, MIN_DIST)
    print(f"the curve 1 / (1 + a d^(2b)): a {a:.4f}, b {b:.4f} here (a grid search); umap-learn's fit: a {ra:.4f}, b {rb:.4f}")
    t0 = time.perf_counter(); U = umap_embed(X); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); U2 = umap.UMAP(n_neighbors=NEIGHBOURS, min_dist=MIN_DIST, random_state=0).fit_transform(X); t2 = time.perf_counter() - t0
    print(f"  {'':<14}{'trustworthiness':>17}{'10 nearest kept':>17}{'seconds':>9}")
    print(f"  {'here':<14}{trust(X, U):>17.4f}{kept(X, U):>17.4f}{t1:>9.1f}")
    print(f"  {'umap-learn':<14}{trust(X, U2):>17.4f}{kept(X, U2):>17.4f}{t2:>9.1f}")
    print("(umap-learn's time includes compiling its code with numba on first use)")
    print("Actual output from this lesson's run.")
    return U, U2


def demo_compare(X, y, maps):
    print("\n" + "=" * 70)
    print("5. THREE MAPS COMPARED -- what each keeps (digits, 2 dimensions)")
    print("=" * 70)
    print(f"  {'map':<20}{'trustworthiness':>17}{'10 nearest kept':>17}{'digit-centre distances (corr)':>31}")
    for name, Z in maps.items():
        print(f"  {name:<20}{trust(X, Z):>17.4f}{kept(X, Z):>17.4f}{centre_corr(X, Z, y):>31.4f}")
    print("trustworthiness and 'kept' judge each point's neighbourhood; the last column, distances between the 10 digit groups' centres")
    print("Actual output from this lesson's run.")


def demo_cluster(X, y, maps):
    print("\n" + "=" * 70)
    print("6. CLUSTERING THE MAPS WITH THE WEEK'S PARTS -- ARI with the true digits, k = 10")
    print("=" * 70)
    mu, vals, V = pca_fit(X); reps = {"PCA, 10 components": transform(X, mu, V, 10)}; reps.update(maps)
    print(f"  {'representation':<22}" + "".join(f"{m:>18}" for m in ("k-means", "Ward", "Gaussian mixture", "DBSCAN")) + "   DBSCAN noise")
    out = {}
    for name, Z in reps.items():
        r, lab, eps = cluster_all(Z, y); out[name] = r
        print(f"  {name:<22}" + "".join(f"{r[m]:>18.4f}" for m in ("k-means", "Ward", "Gaussian mixture", "DBSCAN")) + f"   {np.mean(lab == NOISE):>11.3f}")
    print("k-means: Day 119; Ward: Day 120; Gaussian mixture: Day 122's E and M steps; DBSCAN: Day 121, eps from Day 121's k-distance; PCA: Day 123")
    print("Actual output from this lesson's run.")
    return out


def demo_warnings(X, y, Y30):
    print("\n" + "=" * 70)
    print("7. WARNINGS -- perplexity changes the picture; and does t-SNE invent clusters in pure noise?")
    print("=" * 70)
    sweep = {30: Y30}
    for p in (5, 100):
        sweep[p] = tsne(X, perplexity=p)[0]
    print(f"  {'perplexity':>11}{'trustworthiness':>17}{'k-means ARI':>13}{'digit-centre corr':>19}")
    for p in (5, 30, 100):
        Z = sweep[p]; print(f"  {p:>11}{trust(X, Z):>17.4f}{ari(kmeans(Z, 10, 0)[0], y):>13.4f}{centre_corr(X, Z, y):>19.4f}")
    rng = np.random.RandomState(0); N = rng.uniform(size=(1000, 10))  # 1,000 points spread evenly through a 10-dimensional cube: no clusters at all
    Yn = tsne(N)[0]; mu, vals, V = pca_fit(N); Pn = transform(N, mu, V, 2)  # the same noise, by t-SNE and by PCA
    print("pure noise (1,000 uniform points in 10 dimensions), k-means with k = 5; silhouette near 0 means no real groups:")
    for name, Z in (("the points themselves", N), ("PCA, 2 components", Pn), ("t-SNE map", Yn)):
        print(f"  {name:<22}: silhouette {silhouette(Z, kmeans(Z, 5, 0)[0]):.4f}")
    print("any flat 2-D cloud scores higher than a 10-D one; t-SNE is judged against PCA's 2-D picture of the same points")
    print("Actual output from this lesson's run.")
    return sweep, N, Yn


def plot_results(y, maps, sweep, Yn, clus):             # six panels
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))              # canvas
    for a, (name, Z) in zip(ax[0], [(n, maps[n]) for n in ("PCA, 2 components", "t-SNE (here)", "UMAP (here)")]):
        a.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=4); a.set_title(f"Digits: {name}", fontsize=10); a.set_xticks([]); a.set_yticks([])
        for c in range(10):
            m = np.median(Z[y == c], 0); a.text(m[0], m[1], str(c), fontsize=11, fontweight="bold", ha="center", va="center")
    for a, p in zip(ax[1, :2], (5, 100)):
        Z = sweep[p]; a.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=4); a.set_title(f"t-SNE, perplexity {p}", fontsize=10); a.set_xticks([]); a.set_yticks([])
    a = ax[1, 2]; a.scatter(Yn[:, 0], Yn[:, 1], s=4, color="#555555"); a.set_title("t-SNE of pure noise (uniform, 10 dimensions)", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig("tsne_umap_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved tsne_umap_results.png")


def draw_diagram():                                     # neighbours in, neighbours out
    fig, ax = plt.subplots(figsize=(13, 4.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.35, "Day 125: keeping neighbours on a flat page", ha="center", fontsize=12, fontweight="bold")
    box(2.0, 3.0, 3.4, 1.3, "t-SNE: P = Gaussian neighbour\nprobabilities, width set by\nperplexity", "#FAEEDA")
    box(6.5, 3.0, 3.6, 1.3, "on the page: Student-t similarities Q;\nmove points down the gradient\nof KL(P || Q)", "#DCEBF7")
    box(2.0, 1.3, 3.4, 1.3, "UMAP: a weighted k-nearest-\nneighbour graph, made\nsymmetric", "#FAEEDA")
    box(6.5, 1.3, 3.6, 1.3, "on the page: pull along edges,\npush random pairs apart;\nstart from the graph's spectrum", "#DCEBF7")
    box(11.0, 2.15, 3.2, 2.6, "judge the map:\ntrustworthiness,\nneighbours kept,\ndistances between groups;\nthen cluster it with\nthe week's parts", "#E5F2E1")
    for y0 in (3.0, 1.3):
        ax.annotate("", xy=(4.66, y0), xytext=(3.74, y0), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(9.36, 2.15 + (y0 - 2.15) * 0.4), xytext=(8.34, y0), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("tsne_umap_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved tsne_umap_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 126 begins Week 19, unsupervised learning II: autoencoders from scratch, with a real reconstruction demonstration.")


def main():                                             # runs every section in order
    X, y = load_digits(return_X_y=True)
    demo_checks()                                          # 1. verbatim parts
    demo_pieces(X)                                         # 2. t-SNE's pieces
    Y = demo_tsne(X, y)                                    # 3. t-SNE
    U, U2 = demo_umap(X, y)                                # 4. UMAP
    mu, vals, V = pca_fit(X)
    maps = {"PCA, 2 components": transform(X, mu, V, 2), "t-SNE (here)": Y, "UMAP (here)": U, "UMAP (umap-learn)": U2}
    demo_compare(X, y, maps)                               # 5. three maps
    clus = demo_cluster(X, y, {k: maps[k] for k in ("t-SNE (here)", "UMAP (here)")})  # 6. clustering
    sweep, N, Yn = demo_warnings(X, y, Y)                  # 7. warnings
    plot_results(y, maps, sweep, Yn, clus)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
