"""
Day 155: Matrix Factorization from Scratch -- an SVD-Based Recommender

Day 154 compared users and films directly. Matrix factorization instead
describes every user and every film by a short list of k hidden numbers
(factors), chosen so that a user's numbers times a film's numbers rebuild the
ratings that are known. The same numbers then predict the ratings that are not.

Two routes are built from scratch in NumPy, on the same MovieLens 100K split as
Day 154 (Day 154's parts are imported from day154_exercise.py):
  * SVD: fill the empty cells, take the best rank-k approximation (the singular
    value decomposition), and read predictions off it;
  * learned factors: fit the factors to the known ratings only, with a penalty
    on their size, by alternating least squares (ALS) or by stochastic gradient
    descent (SGD, the method Simon Funk made famous in the Netflix Prize).

  1. Is the ratings table low-rank? SVD from scratch, checked, against a control.
  2. SVD as a recommender: what the empty cells are filled with.
  3. Fill, factorize, repeat: imputation without a penalty.
  4. Learning the factors from known ratings only: ALS.
  5. SGD (Funk SVD) against ALS: the same objective, two routes.
  6. Against Day 154's neighbours, a control, and what the factors mean.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported part quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: to print film titles with accents on any console
import time                                            # time: how long each step takes

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

if hasattr(sys.stdout, "reconfigure"):                  # Python 3.7+: make printing accented titles safe everywhere
    sys.stdout.reconfigure(encoding="utf-8")                                          # so titles such as 'Belle de jour' print on any console
HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # so Day 154's part can be imported from here
with contextlib.redirect_stdout(io.StringIO()):         # day154_exercise prints its hand checks when imported
    import day154_exercise as d154                      # Day 154: data, split, ratings table, biases, similarity, item-based

LU, LI = 5, 2                                           # Day 154's bias penalties, chosen there on the same validation split
ITEM = dict(k=40, shrink=800)                           # Day 154's item-based setting (adjusted cosine), chosen there on validation
SVD_KS = (1, 2, 3, 5, 10, 20, 50, 100)                  # ranks tried for the SVD recommenders
MF_KS = (10, 20, 50, 100)                               # factor counts tried for ALS
MF_LAMS = (0.05, 0.075, 0.1, 0.125, 0.15)               # penalties tried for ALS


# =============================================================================
# THE PARTS -- NEW (copied unchanged into day155_exercise.py)
# =============================================================================
def top_svd(A, k, iters=40, seed=0):                    # NEW: the k largest singular values and vectors, by subspace iteration
    rng = np.random.RandomState(seed)                                                 # random starting directions
    Q = np.linalg.qr(A @ rng.randn(A.shape[1], k + 10))[0]                            # k + 10 orthonormal columns in A's column space
    for _ in range(iters):                                                            # each round multiplies by A A^T: big directions grow fastest
        Q = np.linalg.qr(A @ (A.T @ Q))[0]                                            # re-orthonormalise so the columns stay independent
    B = Q.T @ A                                                                       # A squeezed into the small subspace: (k + 10) x columns
    w, V = np.linalg.eigh(B @ B.T)                                                    # eigenvalues of a small symmetric matrix = squared singular values
    o = np.argsort(-w)[:k]                                                            # the k largest
    s = np.sqrt(np.maximum(w[o], 0.0))                                                # singular values
    return Q @ V[:, o], s, (V[:, o].T @ B) / np.maximum(s, 1e-12)[:, None]           # U (rows x k), s (k), Vt (k x columns)


def low_rank(A, k, seed=0):                             # NEW: the best rank-k approximation of A
    U, s, Vt = top_svd(A, k, seed=seed)                                               # the top k singular triplets
    return (U * s) @ Vt                                                               # U diag(s) Vt


def hard_impute(A, M, k, iters=30, seed=0):             # NEW: fill the empty cells with the rank-k fit, refit, repeat
    Z = np.where(M, A, 0.0)                                                           # start with 0 in the empty cells
    for _ in range(iters):                                                            # each round: factorize, then refill
        L = low_rank(Z, k, seed)                                                      # rank-k fit of the current table
        Z = np.where(M, A, L)                                                         # known cells stay; empty cells take the fit
    return L                                                                          # the last rank-k fit: predictions for every cell


def mf_predict(F, u, i):                                # NEW: mean + user bias + film bias + user factors . film factors
    return F["mu"] + F["bu"][u] + F["bi"][i] + (F["P"][u] * F["Q"][i]).sum(1)         # one number per (user, film) pair


def mf_objective(F, R, M, lam):                         # NEW: squared error on known ratings + lam x squared sizes, once per rating
    pred = F["mu"] + F["bu"][:, None] + F["bi"][None, :] + F["P"] @ F["Q"].T          # every cell's prediction
    nu, ni = M.sum(1), M.sum(0)                                                       # each user's and film's rating count
    sq = ((R - pred) ** 2 * M).sum()                                                  # error on known cells only
    pen = lam * (nu @ ((F["P"] ** 2).sum(1) + F["bu"] ** 2) + ni @ ((F["Q"] ** 2).sum(1) + F["bi"] ** 2))  # the penalty: lam x squared sizes, once per rating
    return sq + pen                                                                   # a user with 100 ratings pays the penalty 100 times


def mf_gradients(F, R, M, lam):                         # NEW: the objective's gradient for every parameter (full batch)
    E = (R - F["mu"] - F["bu"][:, None] - F["bi"][None, :] - F["P"] @ F["Q"].T) * M  # errors on known cells, 0 elsewhere
    nu, ni = M.sum(1), M.sum(0)                                                       # rating counts
    return {"P": -2 * E @ F["Q"] + 2 * lam * nu[:, None] * F["P"],                     # d/dP of sum e^2 + lam n_u |p_u|^2
            "Q": -2 * E.T @ F["P"] + 2 * lam * ni[:, None] * F["Q"],                  # the same for the film factors
            "bu": -2 * E.sum(1) + 2 * lam * nu * F["bu"],                             # and the user biases
            "bi": -2 * E.sum(0) + 2 * lam * ni * F["bi"]}                             # and the film biases


def als_mf(R, M, k, lam, sweeps=15, seed=0, track=None):  # NEW: alternating least squares; each half-step is an exact minimum
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


def sgd_mf(u, i, r, nu, ni, k, lam, lr=0.01, epochs=60, batch=128, seed=0, track=None):  # NEW: Funk-style SGD, in small batches
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


# =============================================================================
# HELPERS FOR THE EXPERIMENTS
# =============================================================================
def setup():                                            # Day 154's data and split, rebuilt exactly
    u, i, r, t = d154.load_movielens(); titles, genres, gnames = d154.load_items()    # the ratings, and the films' titles
    nu, ni = u.max() + 1, i.max() + 1                                                 # users and films, counted from 0
    tr, te = d154.split_ratings(len(r), 0.2, seed=0)                                  # Day 154's 80/20 split
    fi, va = d154.split_ratings(len(tr), 0.1, seed=1); fi, va = tr[fi], tr[va]        # and its validation part
    return u, i, r, titles, nu, ni, tr, te, fi, va                                    # everything the experiments need


def base_model(R, M):                                   # Day 154's bias model and its prediction everywhere
    mu, bu, bi = d154.fit_biases(R, M, LU, LI)                                        # the penalties Day 154 chose
    return mu + bu[:, None] + bi[None, :]                                             # mean + user bias + film bias, every cell


def score(p, r):                                        # RMSE of clipped predictions
    return d154.rmse(np.clip(p, 1, 5), r)                                             # predictions kept inside 1-5 stars


def top10(Q, M, u, i, r, te):                           # precision@10 against test films rated 4-5 (Day 154's measure)
    Q = np.where(M, -np.inf, Q); liked = np.zeros(M.shape, bool); g = r[te] >= 4; liked[u[te][g], i[te][g]] = True  # rated films can never be picked; test films rated 4-5 are hits
    who = np.flatnonzero(liked.sum(1) > 0); top = np.argsort(-Q, axis=1, kind="stable")[:, :10]  # users with a hit to find; their 10 best-scored films
    return np.take_along_axis(liked, top, 1)[who].sum(1).mean() / 10, np.median(M.sum(0)[top[who]])  # mean precision@10, and the median rating count of the films shown


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_lowrank(u, i, r, nu, ni, fi):                                                # section 1
    print("=" * 70)
    print("1. IS THE RATINGS TABLE LOW-RANK? -- SVD from scratch, checked, and a control")
    print("=" * 70)
    R, M = d154.to_matrix(u[fi], i[fi], r[fi], nu, ni)                                # the training part without the validation ratings
    A = (R - base_model(R, M)) * M                                                    # what the bias model leaves, 0 where unrated
    t0 = time.time(); U, s, Vt = top_svd(A, 20); t1 = time.time() - t0                # our SVD, timed
    s_np = np.linalg.svd(A, compute_uv=False)                                         # numpy's full SVD, for the check
    print(f"top 20 singular values of the {nu} x {ni} residual table: subspace iteration ({t1:.2f} s) against numpy's SVD,")
    print(f"  largest relative difference {np.abs(s - s_np[:20]).max() / s_np[0]:.1e}; U orthonormal to {np.abs(U.T @ U - np.eye(20)).max():.1e}")
    print(f"  rank-20 reconstruction: |A - U S Vt| = {np.linalg.norm(A - (U * s) @ Vt):.4f}; the best possible, from the "
          f"discarded singular values, {np.sqrt((s_np[20:] ** 2).sum()):.4f}")        # (the Eckart-Young bound)
    v = A[M]; rng = np.random.RandomState(0); Ash = np.zeros_like(A)                  # the known residuals, and an empty table for the control
    Ash[M] = v[rng.permutation(len(v))]                                               # control: the same values, shuffled among the known cells
    s_sh = np.linalg.svd(Ash, compute_uv=False)                                       # the control's singular values
    tot, tsh = (s_np ** 2).sum(), (s_sh ** 2).sum()                                   # each table's total squared size
    print("share of the table's squared size in the top k directions (the control: the same residuals shuffled among known cells):")
    for k in (1, 5, 10, 20, 50):                                                      # a few ranks
        print(f"  k {k:>3}: real {(s_np[:k] ** 2).sum() / tot:.3f}   shuffled {(s_sh[:k] ** 2).sum() / tsh:.3f}")
    print("Actual output from this lesson's run.")
    return s_np, s_sh                                                                 # both spectra, for the figure


def demo_svd(u, i, r, nu, ni, tr, te, fi, va):                                        # section 2
    print("\n" + "=" * 70)
    print("2. SVD AS A RECOMMENDER -- what goes in the empty cells")
    print("=" * 70)
    Rf, Mf = d154.to_matrix(u[fi], i[fi], r[fi], nu, ni); Bf = base_model(Rf, Mf)     # training without validation, and its bias model
    print(f"{'rank k':>8} {'empty = 0':>11} {'empty = bias model':>19}   (validation RMSE)")
    E0, E1 = [], []                                                                   # validation errors for the two fillings
    for k in SVD_KS:                                                                  # every rank
        L0 = low_rank(Rf, k); L1 = low_rank((Rf - Bf) * Mf, k)                       # raw ratings with 0s; residuals with 0s (= the bias model)
        E0.append(score(L0[u[va], i[va]], r[va])); E1.append(score(Bf[u[va], i[va]] + L1[u[va], i[va]], r[va]))  # score both on validation
        print(f"{k:>8} {E0[-1]:>11.4f} {E1[-1]:>19.4f}")
    k = SVD_KS[int(np.argmin(E1))]                                                    # the best rank for the bias-model filling
    print(f"chosen on validation: empty cells = the bias model, rank {k}")
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni); B = base_model(R, M)          # refit on all the training ratings
    P = B + low_rank((R - B) * M, k)                                                  # bias model + rank-k fit of what it leaves
    print(f"test RMSE {score(P[u[te], i[te]], r[te]):.4f} (bias model alone {score(B[u[te], i[te]], r[te]):.4f}); "
          f"with empty = 0 at rank {SVD_KS[int(np.argmin(E0))]} the best validation RMSE was {min(E0):.4f}")  # the zero-filled route's best, for comparison
    print("Actual output from this lesson's run.")
    return P, k, E0, E1                                                               # test-ready predictions, the rank, both validation curves


def demo_impute(u, i, r, nu, ni, tr, te, fi, va):                                     # section 3
    print("\n" + "=" * 70)
    print("3. FILL, FACTORIZE, REPEAT -- imputation with no penalty")
    print("=" * 70)
    Rf, Mf = d154.to_matrix(u[fi], i[fi], r[fi], nu, ni); Bf = base_model(Rf, Mf); Af = (Rf - Bf) * Mf  # training without validation; the residuals to impute
    t0 = time.time()                                                                  # start the clock
    print(f"{'rank k':>8} {'training RMSE':>14} {'validation RMSE':>16}   (30 rounds)")
    E = []                                                                            # validation errors
    for k in (1, 2, 3, 5, 10, 20, 50):                                                # every rank
        L = hard_impute(Af, Mf, k)                                                    # 30 rounds of fill and refit
        E.append(score(Bf[u[va], i[va]] + L[u[va], i[va]], r[va]))                    # scored on validation
        print(f"{k:>8} {score(Bf[u[fi], i[fi]] + L[u[fi], i[fi]], r[fi]):>14.4f} {E[-1]:>16.4f}")
    k = (1, 2, 3, 5, 10, 20, 50)[int(np.argmin(E))]                                   # the best rank
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni); B = base_model(R, M)          # refit on all the training ratings
    P = B + hard_impute((R - B) * M, M, k)                                            # bias model + imputed residuals
    print(f"chosen on validation: rank {k}; test RMSE {score(P[u[te], i[te]], r[te]):.4f} ({time.time() - t0:.0f} s)")
    print("Actual output from this lesson's run.")
    return P, E                                                                       # predictions everywhere, validation curve


def demo_als(u, i, r, nu, ni, tr, te, fi, va):                                        # section 4
    print("\n" + "=" * 70)
    print("4. LEARNING THE FACTORS FROM KNOWN RATINGS ONLY -- alternating least squares")
    print("=" * 70)
    rng = np.random.RandomState(0); Rs = np.where(rng.rand(6, 5) < 0.6, rng.randint(1, 6, (6, 5)), 0).astype(float)  # a small table, about 60% filled with 1-5
    Ms = Rs > 0; F = dict(mu=Rs[Ms].mean(), P=rng.randn(6, 2), Q=rng.randn(5, 2), bu=rng.randn(6), bi=rng.randn(5))  # random parameters
    G = mf_gradients(F, Rs, Ms, 0.3); worst = 0.0                                     # the analytic gradient
    for q in ("P", "Q", "bu", "bi"):                                                  # every parameter, nudged both ways
        for idx in np.ndindex(F[q].shape):                                            # every entry of this parameter
            o = F[q][idx]; F[q][idx] = o + 1e-6; fp = mf_objective(F, Rs, Ms, 0.3); F[q][idx] = o - 1e-6  # nudge up and down
            fm = mf_objective(F, Rs, Ms, 0.3); F[q][idx] = o; worst = max(worst, abs((fp - fm) / 2e-6 - G[q][idx]))  # central difference against the formula
    print(f"gradient check on a 6 x 5 table, every parameter: largest |numeric - analytic| {worst:.1e}")
    Fa, h = als_mf(Rs, Ms, 2, 0.3, sweeps=5)                                          # five sweeps on the small table
    G = mf_gradients(Fa, Rs, Ms, 0.3)                                                 # gradients at the end
    print(f"ALS on the same table: objective after each sweep {', '.join(f'{x[0]:.4f}' for x in h)}")
    print(f"  after a user sweep the user gradients are 0 by construction; after the film sweep, film gradients: "
          f"{max(np.abs(G['Q']).max(), np.abs(G['bi']).max()):.1e}")                  # near 0: the films' half-step solved exactly
    Rf, Mf = d154.to_matrix(u[fi], i[fi], r[fi], nu, ni)                              # training without the validation ratings
    t0 = time.time(); E = np.zeros((len(MF_KS), len(MF_LAMS)))                        # start the clock; one cell per (factors, penalty)
    for a, k in enumerate(MF_KS):                                                     # every factor count
        for b, lam in enumerate(MF_LAMS):                                             # every penalty
            F, _ = als_mf(Rf, Mf, k, lam)                                             # fit on the training part
            E[a, b] = score(mf_predict(F, u[va], i[va]), r[va])                       # score on validation
    print(f"validation RMSE, 15 sweeps each ({time.time() - t0:.0f} s for the grid):")
    print(f"  {'factors':>8} " + " ".join(f"{'lam ' + format(x, 'g'):>10}" for x in MF_LAMS))
    for a, k in enumerate(MF_KS):                                                     # one row per factor count
        print(f"  {k:>8} " + " ".join(f"{e:>10.4f}" for e in E[a]))
    a, b = np.unravel_index(np.argmin(E), E.shape); k, lam = MF_KS[a], MF_LAMS[b]     # the lowest validation error
    F2, _ = als_mf(Rf, Mf, 2 * MF_KS[-1], lam)                                        # a check beyond the grid's edge
    print(f"chosen: {k} factors, lam {lam} ({E[a, b]:.4f}); the grid's edge, {2 * MF_KS[-1]} factors at lam {lam}: "
          f"{score(mf_predict(F2, u[va], i[va]), r[va]):.4f}")                        # (not chosen: the grid's choice stands)
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni)                                # all the training ratings
    F, h = als_mf(R, M, k, lam, track=lambda F: (score(mf_predict(F, u[te], i[te]), r[te]),))  # refit, recording test RMSE per sweep (only the last is reported)
    print(f"refit on all training ratings: objective {h[0][0]:.1f} after sweep 1, {h[4][0]:.1f} after 5, {h[-1][0]:.1f} after 15; "
          f"falls every sweep: {all(h[j + 1][0] <= h[j][0] for j in range(len(h) - 1))}")  # ALS can never raise the objective
    print(f"test RMSE {h[-1][1]:.4f}, MAE {d154.mae(np.clip(mf_predict(F, u[te], i[te]), 1, 5), r[te]):.4f}")
    print("Actual output from this lesson's run.")
    return F, k, lam, E                                                               # the model, the choice and the grid


def demo_sgd(u, i, r, nu, ni, tr, te, fi, va, k, lam):                                # section 5
    print("\n" + "=" * 70)
    print("5. SGD (FUNK SVD) AGAINST ALS -- the same objective, two routes")
    print("=" * 70)
    Rf, Mf = d154.to_matrix(u[fi], i[fi], r[fi], nu, ni)                              # training without the validation ratings
    vtrack = lambda F: (score(mf_predict(F, u[fi], i[fi]), r[fi]), score(mf_predict(F, u[va], i[va]), r[va]))  # training and validation RMSE
    t0 = time.time()                                                                  # start the clock
    curves = {}                                                                       # one trajectory per method
    for name, (lm, ep) in (("SGD, lam " + format(lam, "g"), (lam, 120)), ("SGD, lam " + format(lam / 4, "g"), (lam / 4, 60))):  # SGD at the chosen penalty, and at a quarter of it
        F, h = sgd_mf(u[fi], i[fi], r[fi], nu, ni, k, lm, epochs=ep, track=vtrack); curves[name] = (np.array(h), lm)  # train, recording RMSE after every epoch
    Fa, ha = als_mf(Rf, Mf, k, lam, track=vtrack); curves["ALS, lam " + format(lam, "g")] = (np.array([x[1:] for x in ha]), lam)  # ALS at the chosen penalty, recorded per sweep
    print(f"{k} factors, learning rate 0.01, batches of 128 ({time.time() - t0:.0f} s); RMSE on training / validation after each pass:")
    for name, (H, _) in curves.items():                                               # one summary per method
        j = int(np.argmin(H[:, 1]))                                                   # the pass with the lowest validation error
        marks = ", ".join(f"{p}: {H[p - 1, 0]:.3f} / {H[p - 1, 1]:.4f}" for p in (1, 5, 15, 30, 60, 120) if p <= len(H))  # a few passes along the way
        print(f"  {name:<14} {marks}")
        print(f"  {'':<14} best validation {H[j, 1]:.4f} at pass {j + 1}")
    Hs = curves["SGD, lam " + format(lam, "g")][0]; ep = int(np.argmin(Hs[:, 1])) + 1  # early stopping: validation's best epoch
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni)                                # all the training ratings
    F, _ = sgd_mf(u[tr], i[tr], r[tr], nu, ni, k, lam, epochs=ep)                     # SGD for that many epochs
    Fals, _ = als_mf(R, M, k, lam)                                                    # ALS for 15 sweeps
    print(f"refit on all training ratings, SGD for {ep} epochs (validation's best): test RMSE {score(mf_predict(F, u[te], i[te]), r[te]):.4f}; "
          f"ALS {score(mf_predict(Fals, u[te], i[te]), r[te]):.4f}")                  # both on the test ratings
    print(f"  objective (lower is better): SGD {mf_objective(F, R, M, lam):.1f}, ALS {mf_objective(Fals, R, M, lam):.1f}")
    print("Actual output from this lesson's run.")
    return F, curves                                                                  # the SGD model and the curves


def demo_compare(u, i, r, nu, ni, tr, te, titles, Fals, Fsgd, Psvd, Pimp, k):         # section 6
    print("\n" + "=" * 70)
    print("6. AGAINST DAY 154'S NEIGHBOURS, A CONTROL, AND WHAT THE FACTORS MEAN")
    print("=" * 70)
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni); B = base_model(R, M)          # all the training ratings; the bias model
    X = (R - (R.sum(1) / np.maximum(M.sum(1), 1))[:, None]) * M                       # ratings minus each user's mean (adjusted cosine)
    S, _ = d154.similarity(X, M, shrink=ITEM["shrink"])                               # Day 154's adjusted cosine, shrunk
    c, _ = d154.predict_item_knn((R - B) * M, M, S, u[te], i[te], ITEM["k"])          # Day 154's item-based correction
    Fsh = dict(Fals); Fsh["P"] = Fals["P"][np.random.RandomState(0).permutation(nu)]  # control: each user gets another user's factors
    rows = {"bias model (Day 154)": B[u[te], i[te]], "item-based, k 40 (Day 154)": B[u[te], i[te]] + c,  # every model's test predictions, in table order
            f"SVD, empty = bias model": Psvd[u[te], i[te]], "fill, factorize, repeat": Pimp[u[te], i[te]],  # (the SVD routes)
            f"learned factors, ALS ({k})": mf_predict(Fals, u[te], i[te]), f"learned factors, SGD ({k})": mf_predict(Fsgd, u[te], i[te]),  # (the learned factors)
            "control: ALS, users' factors shuffled": mf_predict(Fsh, u[te], i[te])}   # (the control last)
    print(f"{'test, 20,000 ratings':>40}    RMSE     MAE")
    for name, p in rows.items():                                                      # one row per model
        print(f"{name:>40}  {score(p, r[te]):.4f}  {d154.mae(np.clip(p, 1, 5), r[te]):.4f}")
    a, b = rows["item-based, k 40 (Day 154)"], rows[f"learned factors, ALS ({k})"]    # the two best models
    print(f"item-based and ALS predictions: correlation {np.corrcoef(a, b)[0, 1]:.3f}; their errors: correlation "
          f"{np.corrcoef(a - r[te], b - r[te])[0, 1]:.3f}")                           # do they make the same mistakes?
    cnt = M.sum(0); Q = Fals["Q"]; ok = np.flatnonzero(cnt >= 50)                    # films with 50+ training ratings
    _, _, Vt = np.linalg.svd(Q[ok] - Q[ok].mean(0), full_matrices=False)              # the main directions of the film factors
    for d in range(2):                                                                # the two main directions
        z = (Q[ok] - Q[ok].mean(0)) @ Vt[d]; o = np.argsort(z)                        # each film's position along it, sorted
        print(f"direction {d + 1} of the film factors (films with 50+ ratings), one end: " + "; ".join(titles[ok[j]] for j in o[:3]))
        print(f"  {'':>41}other end: " + "; ".join(titles[ok[j]] for j in o[-3:][::-1]))
    sw = titles.index("Star Wars (1977)"); qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)  # film factors scaled to length 1
    sim = qn[ok] @ qn[sw]; o = [j for j in np.argsort(-sim) if ok[j] != sw][:5]       # cosine of every film with Star Wars; the closest five
    print("closest to Star Wars (1977) by the angle between factor vectors (50+ ratings): " + "; ".join(f"{titles[ok[j]]} {sim[j]:.2f}" for j in o))
    full = Fals["mu"] + Fals["bu"][:, None] + Fals["bi"][None, :] + Fals["P"] @ Fals["Q"].T  # ALS predictions for every cell
    pa, ma = top10(full, M, u, i, r, te); pp, mp = top10(np.broadcast_to(cnt.astype(float), M.shape), M, u, i, r, te)  # top-10 lists from them, and from popularity
    pf, mf = top10(Fals["P"] @ Fals["Q"].T, M, u, i, r, te)                           # and from the factors alone
    print(f"top-10 lists (Day 154's measure): ALS predicted rating precision@10 {pa:.4f} (median {ma:.0f} ratings per film); "
          f"factors only, no biases {pf:.4f} ({mf:.0f}); popularity {pp:.4f} ({mp:.0f})")  # (median rating count of the films shown)
    print("Actual output from this lesson's run.")
    return rows                                                                       # test predictions, for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(s_np, s_sh, E0, E1, Eimp, E, curves, rows, rt, out):                 # six panels
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))                                  # two rows of three
    ax = axs[0, 0]; k = np.arange(1, 101)                                             # how much of the table the top k directions hold
    ax.plot(k, np.cumsum(s_np[:100] ** 2) / (s_np ** 2).sum(), label="real residuals")  # real residuals
    ax.plot(k, np.cumsum(s_sh[:100] ** 2) / (s_sh ** 2).sum(), label="control: shuffled among known cells")  # the shuffled control
    ax.set_xlabel("top k directions"); ax.set_ylabel("share of squared size"); ax.legend(fontsize=8)  # labels and legend
    ax.set_title("real residuals: only a little more concentrated than the control", fontsize=10)       # title
    ax = axs[0, 1]                                                                    # validation RMSE of the SVD routes
    ax.plot(SVD_KS, E1, marker="o", label="SVD, empty = bias model")                  # SVD with the bias model in the empty cells
    ax.plot((1, 2, 3, 5, 10, 20, 50), Eimp, marker="s", label="fill, factorize, repeat")  # fill, factorize, repeat
    ax.axhline(np.min(E), color="#55A868", ls="--", label="learned factors (ALS), best")  # the learned factors' best, for reference
    ax.set_xscale("log"); ax.set_xlabel("rank k"); ax.set_ylabel("validation RMSE"); ax.legend(fontsize=8)  # a log axis for the rank
    ax.set_title(f"SVD routes (empty = 0 is off the chart: {min(E0):.2f})", fontsize=10)  # the zero-filled route, quoted in the title
    ax = axs[0, 2]                                                                    # the ALS grid
    for a, kk in enumerate(MF_KS):                                                    # one line per factor count
        ax.plot(MF_LAMS, E[a], marker="o", label=f"{kk} factors")                     # validation RMSE against the penalty
    ax.set_xlabel("penalty lam"); ax.set_ylabel("validation RMSE"); ax.legend(fontsize=8); ax.set_title("ALS: choosing factors and penalty", fontsize=10)  # labels, legend, title
    ax = axs[1, 0]                                                                    # training curves
    for (name, (H, _)), col in zip(curves.items(), ("#4C72B0", "#C44E52", "#55A868")):  # one colour per method
        x = np.arange(1, len(H) + 1)                                                  # pass numbers
        ax.plot(x, H[:, 1], color=col, label=name + ", validation"); ax.plot(x, H[:, 0], color=col, ls=":", label=name + ", training")  # validation solid, training dotted
    ax.set_ylim(0.4, 1.3); ax.set_xlabel("pass over the data (epoch or sweep)"); ax.set_ylabel("RMSE"); ax.legend(fontsize=7, ncol=2, loc="upper right")  # axes and legend
    ax.set_title("training keeps falling; validation turns", fontsize=10)             # title
    ax = axs[1, 1]; names = list(rows); vals = [score(rows[m], rt) for m in names]    # test RMSE per model
    ax.barh(range(len(names)), vals, color=["#BBBBBB", "#999999", "#4C72B0", "#8172B2", "#55A868", "#64B5CD", "#C44E52"])  # one bar each
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()  # names, top to bottom
    ax.set_xlim(0.88, max(vals) + 0.04); ax.set_xlabel("test RMSE (stars)")           # zoom in on the differences
    for y, v in enumerate(vals):                                                      # the value at each bar
        ax.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=8)                    # written next to it
    ax.set_title("test error, every model", fontsize=10)                              # title
    ax = axs[1, 2]; ax.semilogy(np.arange(1, 51), s_np[:50], marker=".", label="real residuals")  # the singular values themselves
    ax.semilogy(np.arange(1, 51), s_sh[:50], marker=".", label="control: shuffled")   # and the control's
    ax.set_xlabel("singular value number"); ax.set_ylabel("singular value (log)"); ax.legend(fontsize=8)  # labels and legend
    ax.set_title("the first few singular values stand out", fontsize=10)                  # title
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)                     # save to disk


def draw_diagram():                                     # the two routes to factors
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 15); ax.set_ylim(0.2, 5.2); ax.axis("off")  # a wide, empty canvas

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))  # the box
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)                    # its text

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))  # an arrow from (x0, y0) to (x1, y1)
    box(1.3, 2.6, 2.0, 1.6, "ratings table\nusers x films\n(6% filled)", "#FFF2CC")   # the input
    box(4.3, 3.9, 2.4, 1.0, "fill the empty cells\n(with the bias model)", "#DDEBF7")  # SVD route, step 1
    box(7.3, 3.9, 2.4, 1.0, "SVD: best rank-k\napproximation", "#DDEBF7")             # SVD route, step 2
    box(10.3, 3.9, 2.4, 1.0, "(optional) put the fit\nin the empty cells,\nrepeat", "#DDEBF7")  # SVD route, step 3
    box(4.3, 1.3, 2.4, 1.0, "user factors P\nfilm factors Q\n(k numbers each)", "#DDEBF7")  # learned route, step 1
    box(7.3, 1.3, 2.4, 1.0, "fit to known ratings\nonly, with a penalty\non the sizes", "#DDEBF7")  # learned route, step 2
    box(10.3, 1.3, 2.4, 1.0, "ALS: exact steps\nor SGD: small steps", "#DDEBF7")      # learned route, step 3
    box(13.4, 2.6, 2.4, 1.4, "prediction =\nmean + user bias +\nfilm bias + p_u . q_i", "#E2F0D9")  # the shared prediction
    for y in (3.9, 1.3):                                                              # both rows
        arrow(5.5, y, 6.1, y); arrow(8.5, y, 9.1, y)                                  # arrows along each row
    arrow(11.5, 3.7, 12.2, 3.0); arrow(11.5, 1.5, 12.2, 2.2); arrow(2.3, 3.0, 3.1, 3.7); arrow(2.3, 2.2, 3.1, 1.5)  # into the prediction, and from the table
    ax.text(4.0, 4.75, "SVD route: treats the filled-in guesses as if they were data", fontsize=9.5, style="italic")  # the top row's catch
    ax.text(4.0, 0.4, "learned-factor route (Funk SVD): only the known ratings count", fontsize=9.5, style="italic")  # the bottom row's idea
    ax.set_title("Day 155: two routes to a low-rank model of the ratings table", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "matrix_factorization_diagram.png"), dpi=120); plt.close(fig)  # save to disk


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 156: content-based filtering. Neighbours and factors need ratings; a film nobody has rated has")
    print("none. Content-based filtering recommends from what the films are (here, their genres and year), building")
    print("a profile of each user's tastes, and so can place a brand-new film.")


def main():                                             # runs every section in order
    u, i, r, titles, nu, ni, tr, te, fi, va = setup()                                 # Day 154's data and split
    s_np, s_sh = demo_lowrank(u, i, r, nu, ni, fi)                                    # 1. low rank
    Psvd, ksvd, E0, E1 = demo_svd(u, i, r, nu, ni, tr, te, fi, va)                    # 2. SVD recommender
    Pimp, Eimp = demo_impute(u, i, r, nu, ni, tr, te, fi, va)                         # 3. fill, factorize, repeat
    Fals, k, lam, E = demo_als(u, i, r, nu, ni, tr, te, fi, va)                       # 4. ALS
    Fsgd, curves = demo_sgd(u, i, r, nu, ni, tr, te, fi, va, k, lam)                  # 5. SGD against ALS
    rows = demo_compare(u, i, r, nu, ni, tr, te, titles, Fals, Fsgd, Psvd, Pimp, k)   # 6. comparison
    plot_results(s_np, s_sh, E0, E1, Eimp, E, curves, rows, r[te], os.path.join(HERE, "matrix_factorization_results.png"))  # the results figure
    draw_diagram()                                                                    # the method
    print("\nsaved matrix_factorization_results.png\nsaved matrix_factorization_diagram.png")
    note_on_what_comes_next()                                                         # a preview of Day 156


if __name__ == "__main__":
    main()                                                                            # run everything
