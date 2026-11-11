"""
Day 155 Exercise -- Matrix Factorization: SVD and Learned Factors

The second of Week 23's parts for Day 160's neural recommender.

    top_svd(A, k)                      the k largest singular values and vectors, by subspace iteration
    low_rank(A, k)                     the best rank-k approximation of A
    hard_impute(A, M, k)               fill the empty cells with the rank-k fit, refit, repeat
    mf_predict(F, u, i)                mean + user bias + film bias + user factors . film factors
    mf_objective(F, R, M, lam)         squared error on known ratings + lam x squared sizes, once per rating
    mf_gradients(F, R, M, lam)         the objective's gradient for every parameter
    als_mf(R, M, k, lam)               alternating least squares: each half-step an exact minimum
    sgd_mf(u, i, r, nu, ni, k, lam)    Funk-style stochastic gradient descent, in small batches

Checked by hand: singular values of a known matrix and of a random one, a
rank-1 table rebuilt exactly, one missing cell of a rank-1 table filled in, a
prediction and an objective worked on paper, a numerical gradient check, ALS
never raising its objective, and SGD learning a rank-1 table.
"""
import numpy as np                                     # numpy: every array operation below is built on this


def top_svd(A, k, iters=40, seed=0):                    # NEW: Day 155's version, unchanged
    rng = np.random.RandomState(seed)                                                 # random starting directions
    Q = np.linalg.qr(A @ rng.randn(A.shape[1], k + 10))[0]                            # k + 10 orthonormal columns in A's column space
    for _ in range(iters):                                                            # each round multiplies by A A^T: big directions grow fastest
        Q = np.linalg.qr(A @ (A.T @ Q))[0]                                            # re-orthonormalise so the columns stay independent
    B = Q.T @ A                                                                       # A squeezed into the small subspace: (k + 10) x columns
    w, V = np.linalg.eigh(B @ B.T)                                                    # eigenvalues of a small symmetric matrix = squared singular values
    o = np.argsort(-w)[:k]                                                            # the k largest
    s = np.sqrt(np.maximum(w[o], 0.0))                                                # singular values
    return Q @ V[:, o], s, (V[:, o].T @ B) / np.maximum(s, 1e-12)[:, None]           # U (rows x k), s (k), Vt (k x columns)


def low_rank(A, k, seed=0):                             # NEW: Day 155's version, unchanged
    U, s, Vt = top_svd(A, k, seed=seed)                                               # the top k singular triplets
    return (U * s) @ Vt                                                               # U diag(s) Vt


def hard_impute(A, M, k, iters=30, seed=0):             # NEW: Day 155's version, unchanged
    Z = np.where(M, A, 0.0)                                                           # start with 0 in the empty cells
    for _ in range(iters):                                                            # each round: factorize, then refill
        L = low_rank(Z, k, seed)                                                      # rank-k fit of the current table
        Z = np.where(M, A, L)                                                         # known cells stay; empty cells take the fit
    return L                                                                          # the last rank-k fit: predictions for every cell


def mf_predict(F, u, i):                                # NEW: Day 155's version, unchanged
    return F["mu"] + F["bu"][u] + F["bi"][i] + (F["P"][u] * F["Q"][i]).sum(1)         # one number per (user, film) pair


def mf_objective(F, R, M, lam):                         # NEW: Day 155's version, unchanged
    pred = F["mu"] + F["bu"][:, None] + F["bi"][None, :] + F["P"] @ F["Q"].T          # every cell's prediction
    nu, ni = M.sum(1), M.sum(0)                                                       # each user's and film's rating count
    sq = ((R - pred) ** 2 * M).sum()                                                  # error on known cells only
    pen = lam * (nu @ ((F["P"] ** 2).sum(1) + F["bu"] ** 2) + ni @ ((F["Q"] ** 2).sum(1) + F["bi"] ** 2))  # the penalty: lam x squared sizes, once per rating
    return sq + pen                                                                   # a user with 100 ratings pays the penalty 100 times


def mf_gradients(F, R, M, lam):                         # NEW: Day 155's version, unchanged
    E = (R - F["mu"] - F["bu"][:, None] - F["bi"][None, :] - F["P"] @ F["Q"].T) * M  # errors on known cells, 0 elsewhere
    nu, ni = M.sum(1), M.sum(0)                                                       # rating counts
    return {"P": -2 * E @ F["Q"] + 2 * lam * nu[:, None] * F["P"],                     # d/dP of sum e^2 + lam n_u |p_u|^2
            "Q": -2 * E.T @ F["P"] + 2 * lam * ni[:, None] * F["Q"],                  # the same for the film factors
            "bu": -2 * E.sum(1) + 2 * lam * nu * F["bu"],                             # and the user biases
            "bi": -2 * E.sum(0) + 2 * lam * ni * F["bi"]}                             # and the film biases


def als_mf(R, M, k, lam, sweeps=15, seed=0, track=None):  # NEW: Day 155's version, unchanged
    rng = np.random.RandomState(seed); nu, ni = M.shape                               # random small starting factors
    F = dict(mu=R[M].mean(), P=0.1 * rng.randn(nu, k), Q=0.1 * rng.randn(ni, k), bu=np.zeros(nu), bi=np.zeros(ni))  # the mean is fixed; biases start at 0
    hist = []                                                                         # (objective, extra measures) after each sweep
    for _ in range(sweeps):                                                           # each sweep: all users, then all films
        for a in range(nu):                                                           # users: films' factors fixed, one small least squares each
            j = np.flatnonzero(M[a]); X = np.c_[F["Q"][j], np.ones(len(j))]           # design: the film factors and a 1 for the user bias
            y = R[a, j] - F["mu"] - F["bi"][j]                                        # what the user's part has to explain
            w = np.linalg.solve(X.T @ X + lam * len(j) * np.eye(k + 1), X.T @ y)      # ridge regression, penalty lam per rating
            F["P"][a], F["bu"][a] = w[:k], w[k]                                       # the user's factors and bias
        for b in range(ni):                                                           # films: users' factors fixed
            j = np.flatnonzero(M[:, b])                                               # who rated film b
            if len(j) == 0:                                                           # nobody: the penalty alone puts everything at 0
                F["Q"][b], F["bi"][b] = 0.0, 0.0                                      # an unrated film gets no factors and no bias
                continue                                                              # next film
            X = np.c_[F["P"][j], np.ones(len(j))]; y = R[j, b] - F["mu"] - F["bu"][j]  # design: the user factors and a 1 for the film bias
            w = np.linalg.solve(X.T @ X + lam * len(j) * np.eye(k + 1), X.T @ y)      # ridge regression, penalty lam per rating
            F["Q"][b], F["bi"][b] = w[:k], w[k]                                       # the film's factors and bias
        hist.append((mf_objective(F, R, M, lam),) + (track(F) if track else ()))     # the objective can only go down
    return F, hist                                                                    # the factors, and one record per sweep


def sgd_mf(u, i, r, nu, ni, k, lam, lr=0.01, epochs=60, batch=128, seed=0, track=None):  # NEW: Day 155's version, unchanged
    rng = np.random.RandomState(seed)                                                 # starting factors and the shuffles
    F = dict(mu=r.mean(), P=0.1 * rng.randn(nu, k), Q=0.1 * rng.randn(ni, k), bu=np.zeros(nu), bi=np.zeros(ni))  # the mean is fixed; biases start at 0
    hist = []                                                                         # measures after each epoch
    for _ in range(epochs):                                                           # one pass over the known ratings
        o = rng.permutation(len(r))                                                   # a new random order each epoch
        for s in range(0, len(o), batch):                                             # small batches of ratings
            b = o[s:s + batch]; a, c = u[b], i[b]                                     # this batch's users and films
            e = r[b] - mf_predict(F, a, c)                                            # its errors
            gp = e[:, None] * F["Q"][c] - lam * F["P"][a]                             # downhill for each user's factors
            gq = e[:, None] * F["P"][a] - lam * F["Q"][c]                             # and each film's
            np.add.at(F["bu"], a, lr * (e - lam * F["bu"][a]))                        # add.at: a user twice in a batch gets both updates
            np.add.at(F["bi"], c, lr * (e - lam * F["bi"][c]))                        # film biases, the same way
            np.add.at(F["P"], a, lr * gp); np.add.at(F["Q"], c, lr * gq)              # and the factors
        hist.append(track(F) if track else ())                                        # measures after this epoch, if asked for
    return F, hist                                                                    # the factors, and one record per epoch


A = np.zeros((4, 3)); A[0, 0], A[1, 1], A[2, 2] = 3.0, 2.0, 1.0
print(f"singular values of a 4 x 3 table with 3, 2, 1 on its diagonal: 3, 2; top_svd gives {np.round(top_svd(A, 2)[1], 10).tolist()}")
B = np.random.RandomState(0).randn(30, 20)
print(f"a random 30 x 20 table, top 5: largest difference from numpy's SVD {np.abs(top_svd(B, 5)[1] - np.linalg.svd(B, compute_uv=False)[:5]).max():.1e}")
a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.5, 1.0, 4.0])
print(f"a rank-1 table (outer product of 1,2,3 and 2,0.5,1,4) rebuilt at rank 1: error {np.abs(low_rank(np.outer(a, b), 1) - np.outer(a, b)).max():.1e}")
T = np.outer(a, b); Mk = np.ones_like(T, bool); Mk[2, 3] = False                     # hide the cell 3 x 4 = 12
print(f"the same table with cell (3, 4) hidden (true value 12): hard_impute at rank 1 fills {hard_impute(T, Mk, 1, iters=200)[2, 3]:.4f}")
F = dict(mu=3.0, bu=np.array([0.5]), bi=np.array([-0.2]), P=np.array([[1.0, 2.0]]), Q=np.array([[0.5, 0.25]]))
print(f"prediction: 3 + 0.5 - 0.2 + (1 x 0.5 + 2 x 0.25) = 4.3; mf_predict gives {mf_predict(F, np.array([0]), np.array([0]))[0]:.4f}")
R1 = np.array([[5.0]]); M1 = np.array([[True]])
print(f"objective with rating 5, lam 0.1: (5 - 4.3)^2 + 0.1 x (1 + 4 + 0.25 + 0.0625 + 0.25 + 0.04) = 0.49 + 0.56025 = 1.05025; "
      f"gives {mf_objective(F, R1, M1, 0.1):.5f}")
rng = np.random.RandomState(1); Rs = np.where(rng.rand(5, 4) < 0.7, rng.randint(1, 6, (5, 4)), 0).astype(float); Ms = Rs > 0
G = dict(mu=Rs[Ms].mean(), P=rng.randn(5, 2), Q=rng.randn(4, 2), bu=rng.randn(5), bi=rng.randn(4)); g = mf_gradients(G, Rs, Ms, 0.2); worst = 0.0
for q in ("P", "Q", "bu", "bi"):
    for idx in np.ndindex(G[q].shape):
        o = G[q][idx]; G[q][idx] = o + 1e-6; fp = mf_objective(G, Rs, Ms, 0.2); G[q][idx] = o - 1e-6
        fm = mf_objective(G, Rs, Ms, 0.2); G[q][idx] = o; worst = max(worst, abs((fp - fm) / 2e-6 - g[q][idx]))
print(f"numerical gradient check, every parameter of a 5 x 4 table: largest |numeric - analytic| {worst:.1e}")
_, h = als_mf(Rs, Ms, 2, 0.2, sweeps=6)
print(f"ALS objective after each of 6 sweeps never rises: {all(h[j + 1][0] <= h[j][0] + 1e-12 for j in range(5))} ({h[0][0]:.4f} -> {h[-1][0]:.4f})")
uu, ii = np.nonzero(np.ones((3, 4), bool)); rr = 3.0 + 0.5 * np.outer(a - 2, b - 1.875)[uu, ii]
Fs, _ = sgd_mf(uu, ii, rr, 3, 4, 1, 0.0, lr=0.05, epochs=400, batch=4)
print(f"SGD on a full 3 x 4 table that is a mean plus a rank-1 pattern: largest error after 400 epochs {np.abs(mf_predict(Fs, uu, ii) - rr).max():.1e}")
