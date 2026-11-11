"""
Day 154: Collaborative Filtering -- User-Based and Item-Based, on a Real Ratings Matrix

A recommender guesses how much a person will like something they have not tried
yet. Collaborative filtering does it from ratings alone: people who agreed in
the past tend to agree again (user-based), and films that were rated alike tend
to be liked by the same people (item-based). Nothing about the films themselves
is used.

Week 23 is about recommender systems. Like Weeks 16-22, Days 154-159 each add a
full lesson and a short exercise file holding that day's parts, and Day 160
builds a neural recommender from those parts.

The data is MovieLens 100K: 100,000 ratings (1 to 5 stars) of 1,682 films by 943
users, collected by GroupLens at the University of Minnesota in 1997-98.
  F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and Context.
  ACM Transactions on Interactive Intelligent Systems 5(4), 2015.
Its licence does not allow it to be copied into this repository, so the script
downloads it (4.9 MB) to ~/movielens_data the first time it runs.

Everything is written from scratch in NumPy.

  1. The data: how empty a ratings matrix is.
  2. A split, and four baselines that use no neighbours at all.
  3. Similarity, checked by hand, and the small-overlap problem.
  4. User-based collaborative filtering.
  5. Item-based collaborative filtering, against the baselines and a control.
  6. From predicted ratings to a top-10 list.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: file paths
import sys                                             # sys: to print film titles with accents on any console
import time                                            # time: how long each step takes
import zipfile                                         # zipfile: to unpack the downloaded archive
import urllib.request                                  # urllib: to download the data the first time

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

if hasattr(sys.stdout, "reconfigure"):                  # Python 3.7+: make printing accented titles safe everywhere
    sys.stdout.reconfigure(encoding="utf-8")                                          # so titles such as 'Belle de jour' print on any console

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
DATA = os.path.join(os.path.expanduser("~"), "movielens_data")  # where the data is kept: outside the repository
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"  # GroupLens' official copy
KS = (5, 10, 20, 40, 80, 160)                           # neighbourhood sizes tried on validation
SHRINKS = (0, 10, 25, 50, 100, 200, 400, 800)                     # similarity shrinkage strengths tried on validation
LAMS = (0, 2, 5, 10, 25, 50)                            # bias penalties tried on validation
DAMPS = (0, 0.1, 0.25, 0.5, 1, 2, 4)                    # top-10 damping constants tried on validation


# =============================================================================
# THE PARTS -- NEW (copied unchanged into day154_exercise.py)
# =============================================================================
def load_movielens(folder=DATA):                        # NEW: the 100,000 ratings, downloaded on first use
    path = os.path.join(folder, "ml-100k", "u.data")                                  # one rating per line: user, film, stars, time
    if not os.path.exists(path):                                                      # first run: fetch and unpack the archive
        os.makedirs(folder, exist_ok=True)                                            # the data folder, if missing
        z = os.path.join(folder, "ml-100k.zip")                                       # where the archive is saved
        urllib.request.urlretrieve(URL, z)                                            # 4.9 MB from GroupLens
        zipfile.ZipFile(z).extractall(folder)                                         # creates folder/ml-100k/
    a = np.loadtxt(path, dtype=np.int64)                                              # 100,000 rows of 4 whole numbers
    return a[:, 0] - 1, a[:, 1] - 1, a[:, 2].astype(float), a[:, 3]                   # users and films from 0; stars; seconds since 1970


def load_items(folder=DATA):                            # NEW: each film's title and its 19 genre flags
    rows = [ln.split("|") for ln in open(os.path.join(folder, "ml-100k", "u.item"), encoding="latin-1").read().strip().split("\n")]  # one row per film, fields split at '|'
    names = [ln.split("|")[0] for ln in open(os.path.join(folder, "ml-100k", "u.genre"), encoding="latin-1").read().strip().split("\n")]  # the 19 genre names, in column order
    titles = [r[1] for r in rows]                                                     # e.g. "Star Wars (1977)"
    genres = np.array([[int(g) for g in r[5:24]] for r in rows])                     # 1 if the film has that genre
    return titles, genres, names                                                      # titles, a films x 19 table of 0/1, the genre names


def split_ratings(n, frac, seed):                       # NEW: a random split of the ratings (not of the users)
    p = np.random.RandomState(seed).permutation(n)                                    # every rating's position, shuffled
    m = int(round(n * frac))                                                          # how many go to the held-out part
    return np.sort(p[m:]), np.sort(p[:m])                                             # (kept, held out), each in file order


def to_matrix(u, i, r, nu, ni):                         # NEW: the ratings as a users x films table, and where it is filled
    R = np.zeros((nu, ni)); M = np.zeros((nu, ni), bool)                              # 0 and False where nothing was rated
    R[u, i] = r; M[u, i] = True                                                       # fill in the known ratings
    return R, M                                                                       # the table, and the filled cells


def fit_biases(R, M, lam_u, lam_i, iters=20):           # NEW: rating = mean + user bias + film bias, with penalties
    mu = R[M].mean()                                                                  # the average of every known rating
    bu = np.zeros(R.shape[0]); bi = np.zeros(R.shape[1])                              # start every bias at 0
    nu, ni = M.sum(1), M.sum(0)                                                       # ratings per user, per film
    for _ in range(iters):                                                            # alternate: each step is an exact minimum
        bi = ((R - mu - bu[:, None]) * M).sum(0) / np.maximum(ni + lam_i, 1e-9)       # films, with the users' biases fixed (0 if unrated)
        bu = ((R - mu - bi[None, :]) * M).sum(1) / np.maximum(nu + lam_u, 1e-9)       # users, with the films' biases fixed
    return mu, bu, bi                                                                 # the global mean and the two sets of biases


def similarity(X, M, shrink=0.0):                       # NEW: cosine between the columns of X, over rows where both are filled
    Mf = M.astype(float); X2 = X * X                                                  # the mask as numbers; squared entries
    num = X.T @ X                                                                     # sum over shared rows of x_a x_b (X is 0 where unfilled)
    den = np.sqrt(X2.T @ Mf) * np.sqrt(Mf.T @ X2)                                     # the two lengths, each over the shared rows only
    S = np.divide(num, den, out=np.zeros_like(num), where=den > 0)                    # 0 when there is nothing to compare
    n = Mf.T @ Mf                                                                     # how many rows each pair shares
    if shrink:                                                                        # few shared rows: pull the similarity toward 0
        S = S * n / (n + shrink)                                                      # n / (n + shrink) is near 0 for small n, near 1 for large n
    np.fill_diagonal(S, 0.0)                                                          # a column is never its own neighbour
    return S, n                                                                       # similarities, and the shared counts behind them


def _groups(keys):                                      # NEW: positions of each distinct key, without a full scan per key
    order = np.argsort(keys, kind="stable"); ks = keys[order]                         # keys sorted, ties kept in order
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])                           # where each new key begins
    return [(ks[a], order[a:b]) for a, b in zip(starts, np.r_[starts[1:], len(ks)])]  # (key, positions of that key) for every key


def _weighted(s, d, k, damp=0.0):                       # NEW: average of d weighted by the k largest positive s, row by row
    kk = min(k, s.shape[1])                                                           # cannot use more neighbours than exist
    top = np.argpartition(-s, kk - 1, axis=1)[:, :kk]                                 # the kk largest similarities in each row (unordered)
    st = np.take_along_axis(s, top, 1); dt = np.take_along_axis(d, top, 1)            # their similarities and their values
    st = np.where(st > 0, st, 0.0)                                                    # only positive similarities count
    w = st.sum(1)                                                                     # total weight in each row
    return np.where(w > 0, (st * dt).sum(1) / np.where(w > 0, w + damp, 1.0), 0.0), (st > 0).sum(1)  # damp > 0: little weight, little correction


def predict_user_knn(D, M, S, users, items, k):         # NEW: user-based: what similar users thought of this film
    out = np.zeros(len(users)); used = np.zeros(len(users), int)                      # correction, and neighbours used
    for ii, q in _groups(items):                                                      # one film at a time
        raters = np.flatnonzero(M[:, ii])                                             # who rated this film
        if len(raters) == 0:                                                          # nobody: no correction
            continue                                                                  # skip to the next film
        s = S[np.ix_(users[q], raters)]                                               # each asking user against each rater
        d = np.broadcast_to(D[raters, ii], s.shape)                                   # the raters' deviations from their own means
        out[q], used[q] = _weighted(s, d, k)                                          # weighted average of the k most similar raters
    return out, used                                                                  # the correction to add to each user's mean, and how many neighbours gave it


def predict_item_knn(D, M, S, users, items, k, damp=0.0):  # NEW: item-based: how this user rated films like this one
    out = np.zeros(len(users)); used = np.zeros(len(users), int)                      # correction, and neighbours used
    for uu, q in _groups(users):                                                      # one user at a time
        rated = np.flatnonzero(M[uu])                                                 # the films this user rated
        s = S[np.ix_(items[q], rated)]                                                # each film asked about against each film rated
        d = np.broadcast_to(D[uu, rated], s.shape)                                    # the user's deviations on the films rated
        out[q], used[q] = _weighted(s, d, k, damp)                                    # weighted average of the k most similar films the user rated
    return out, used                                                                  # the correction to add to the offset, and how many neighbours gave it


def rmse(p, r):                                         # NEW: root mean squared error, in stars
    return float(np.sqrt(np.mean((p - r) ** 2)))                                      # large errors count more than in MAE


def mae(p, r):                                          # NEW: mean absolute error, in stars
    return float(np.mean(np.abs(p - r)))                                              # the typical size of an error


# =============================================================================
# HELPERS FOR THE EXPERIMENTS
# =============================================================================
def user_means(R, M):                                   # each user's average rating
    return R.sum(1) / np.maximum(M.sum(1), 1)                                         # users with no ratings get 0 (never asked here)


def item_means(R, M, fill):                             # each film's average rating (fill for films nobody rated)
    c = M.sum(0)                                                                      # ratings per film
    return np.where(c > 0, R.sum(0) / np.maximum(c, 1), fill)                         # films nobody rated get the fill value


def choose_biases(R, M, u, i, r):                       # the bias penalties with the lowest validation RMSE
    best = None                                                                       # nothing tried yet
    for lu in LAMS:                                                                   # every pair of penalties
        for li in LAMS:                                                               # film penalties
            mu, bu, bi = fit_biases(R, M, lu, li)                                     # fit on the training part only
            e = rmse(np.clip(mu + bu[u] + bi[i], 1, 5), r)                            # score on the validation ratings
            if best is None or e < best[0]:                                           # keep the lowest error
                best = (e, lu, li)                                                    # with the penalties that gave it
    return best                                                                       # (validation RMSE, user penalty, film penalty)


def shuffled(S, seed):                                  # the same similarity values, given to random pairs
    n = S.shape[0]; iu = np.triu_indices(n, 1)                                        # every pair once
    v = S[iu][np.random.RandomState(seed).permutation(len(iu[0]))]                    # their values, shuffled
    T = np.zeros_like(S); T[iu] = v; return T + T.T                                   # symmetric again, 0 on the diagonal


def item_parts(R, M, lam_u, lam_i, centre):             # the residuals an item-based model averages, and its offset
    mu, bu, bi = fit_biases(R, M, lam_u, lam_i)                                       # the bias model, fitted on R
    base = mu + bu[:, None] + bi[None, :]                                             # the bias model's prediction everywhere
    C = {"raw ratings (cosine)": 0.0, "minus the film's mean (Pearson)": item_means(R, M, mu)[None, :],  # what to subtract before comparing films
         "minus the user's mean (adjusted cosine)": user_means(R, M)[:, None], "minus the bias model": base}[centre]  # (each choice is a different similarity)
    return (R - C) * M, (R - base) * M, base                                          # compared values, residuals to average, offset


CENTRES = ("raw ratings (cosine)", "minus the film's mean (Pearson)", "minus the user's mean (adjusted cosine)", "minus the bias model")  # the four similarities compared in section 5


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_data(u, i, r, t, titles, genres, gnames):                                    # section 1
    print("=" * 70)
    print("1. THE DATA -- MovieLens 100K, a real ratings matrix")
    print("=" * 70)
    nu, ni = u.max() + 1, i.max() + 1                                                 # users and films, counted from 0
    print(f"{len(r):,} ratings by {nu} users of {ni} films; the {nu} x {ni} table has {nu * ni:,} cells")
    print(f"  filled: {len(r) / (nu * ni):.4f} of the cells ({len(r) / (nu * ni) * 100:.1f}%); the rest is what a recommender must guess")
    c = np.bincount(r.astype(int), minlength=6)[1:]                                   # how many 1s, 2s, ... 5s
    print("  stars:  " + "   ".join(f"{s}: {c[s - 1]:,} ({c[s - 1] / len(r):.3f})" for s in range(1, 6)) + f";  mean {r.mean():.3f}")
    pu, pi = np.bincount(u, minlength=nu), np.bincount(i, minlength=ni)               # ratings per user, per film
    print(f"  ratings per user: fewest {pu.min()}, median {np.median(pu):.0f}, most {pu.max()}")
    print(f"  ratings per film: fewest {pi.min()}, median {np.median(pi):.0f}, most {pi.max()} ({titles[pi.argmax()]})")
    top = np.sort(pi)[::-1]                                                           # films' rating counts, largest first
    print(f"  the most-rated 10% of films ({ni // 10}) hold {top[:ni // 10].sum() / len(r):.3f} of all ratings; "
          f"{(pi < 5).sum()} films have fewer than 5")                                # the long tail
    g = genres.sum(0); o = np.argsort(-g)[:3]                                         # films per genre; the three commonest
    print("  commonest genres: " + ", ".join(f"{gnames[k]} {g[k]}" for k in o) + f" (a film can have several)")
    d = (t.max() - t.min()) / 86400                                                   # seconds -> days
    print(f"  collected over {d:.0f} days, {str(np.datetime64(int(t.min()), 's'))[:10]} to {str(np.datetime64(int(t.max()), 's'))[:10]}")
    print("Actual output from this lesson's run.")
    return pi                                                                         # ratings per film, for later sections


def demo_baselines(u, i, r, nu, ni):                                                  # section 2
    print("\n" + "=" * 70)
    print("2. A SPLIT, AND FOUR BASELINES THAT USE NO NEIGHBOURS")
    print("=" * 70)
    tr, te = split_ratings(len(r), 0.2, seed=0)                                       # 80% to learn from, 20% to test
    fi, va = split_ratings(len(tr), 0.1, seed=1); fi, va = tr[fi], tr[va]             # 10% of the training part for choosing settings
    print(f"ratings split at random (seed 0): {len(tr):,} training, {len(te):,} test; of the training, {len(va):,} held back for validation")
    print(f"  every test user also has training ratings: {np.isin(np.unique(u[te]), np.unique(u[tr])).all()}; "
          f"test films never rated in training: {(~np.isin(np.unique(i[te]), np.unique(i[tr]))).sum()}")  # films that cannot be predicted from their own ratings
    Rf, Mf = to_matrix(u[fi], i[fi], r[fi], nu, ni)                                   # the training part without the validation ratings
    e, lu, li = choose_biases(Rf, Mf, u[va], i[va], r[va])                            # every pair of penalties, scored on validation
    print(f"bias penalties chosen on validation: user {lu}, film {li} (validation RMSE {e:.4f})")
    R, M = to_matrix(u[tr], i[tr], r[tr], nu, ni)                                     # refit on all the training ratings
    mu = R[M].mean(); um = user_means(R, M); im = item_means(R, M, mu)                # the three averages, from all the training ratings
    mu, bu, bi = fit_biases(R, M, lu, li)                                             # the bias model with the chosen penalties
    rows = {"the global mean": np.full(len(te), mu), "each user's mean": um[u[te]], "each film's mean": im[i[te]],  # each baseline's prediction for every test rating
            "mean + user bias + film bias": np.clip(mu + bu[u[te]] + bi[i[te]], 1, 5)}  # (clipped to the 1-5 star range)
    print(f"{'test, ' + format(len(te), ',') + ' ratings':>32}    RMSE     MAE")
    for k, p in rows.items():                                                         # one row per baseline
        print(f"{k:>32}  {rmse(p, r[te]):.4f}  {mae(p, r[te]):.4f}")
    mu0, bu0, bi0 = fit_biases(R, M, 0, 0)                                            # the bias model with no penalties at all
    w = np.argmax(bi0); w2 = np.argmax(bi)                                            # the film each version rates highest
    print(f"without penalties the 'best' film is #{w + 1} (bias {bi0[w]:+.2f} from {M[:, w].sum()} rating(s)); "
          f"with them, #{w2 + 1} (bias {bi[w2]:+.2f} from {M[:, w2].sum()})")         # a film's bias needs many ratings to be trusted
    print("Actual output from this lesson's run.")
    return tr, te, fi, va, lu, li, rows                                               # the split, the penalties, the baselines' predictions


def demo_similarity(u, i, r, tr, nu, ni, titles, lu, li):                             # section 3
    print("\n" + "=" * 70)
    print("3. SIMILARITY -- checked by hand, and the small-overlap problem")
    print("=" * 70)
    X = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]], float)  # 5 users x 4 films, 0 = unrated
    Mx = X > 0                                                                        # where a rating exists
    S, n = similarity(X, Mx)                                                          # every pair of films
    a, b = X[:, 0], X[:, 3]; sh = Mx[:, 0] & Mx[:, 3]                                 # films 1 and 4, and the users who rated both
    hand = (a[sh] @ b[sh]) / np.sqrt((a[sh] ** 2).sum() * (b[sh] ** 2).sum())         # the cosine formula, written out for films 1 and 4
    print(f"a 5-user, 4-film example: films 1 and 4 share {int(n[0, 3])} raters; cosine {S[0, 3]:.4f}, by hand {hand:.4f}")
    F = np.random.RandomState(0).randint(1, 6, (30, 6)).astype(float)                 # a full table: every user rated every film
    Sp, _ = similarity(F - F.mean(0), np.ones_like(F, bool))                          # centred by each film's mean = Pearson when every cell is filled
    C = np.corrcoef(F.T); np.fill_diagonal(C, 0.0)                                    # numpy's Pearson correlations, diagonal zeroed like ours
    print(f"a full 30 x 6 table, centred by each film's mean: against numpy's corrcoef, largest difference {np.abs(Sp - C).max():.1e}")
    R, M = to_matrix(u[tr], i[tr], r[tr], nu, ni)                                     # the training ratings as a table
    D, _, _ = item_parts(R, M, lu, li, "minus the bias model")                        # residuals from the bias model
    S0, n = similarity(D, M)                                                          # every pair of films, no shrinkage
    iu = np.triu_indices(ni, 1); s, c = S0[iu], n[iu]                                 # each pair once: similarities and shared counts
    print(f"films' similarities on the training ratings (residuals from the bias model): {len(s):,} pairs, {(c == 0).mean():.3f} share no rater")
    for lo, hi in ((1, 2), (3, 9), (10, 49), (50, 10 ** 9)):                          # four bands of shared raters
        k = (c >= lo) & (c <= hi)                                                     # the pairs in this band
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"                              # a label such as 10-49 or 50+
        print(f"  pairs sharing {lab:>5} raters: {k.sum():>9,}; |similarity| > 0.9 in {(np.abs(s[k]) > 0.9).mean():.3f}, mean |s| {np.abs(s[k]).mean():.3f}")
    sw = titles.index("Star Wars (1977)")                                             # film 50 in the data
    for lab, SS in (("no shrinkage", S0), ("shrinkage 100", S0 * n / (n + 100))):     # without, then with, shrinkage
        o = np.argsort(-SS[sw])[:5]                                                   # its five most similar films
        print(f"most similar to Star Wars (1977), {lab}:")
        for j in o:                                                                   # one line per neighbour
            print(f"    {SS[sw, j]:+.3f}  shared raters {int(n[sw, j]):>3}  {titles[j]}")
    print("Actual output from this lesson's run.")
    return S0, n                                                                      # the unshrunk similarities and the shared counts, for the figure


def grid(R, M, D, S0, n, offset, u, i, r, predict, k_list, shrinks):  # validation RMSE for every (shrink, k)
    E = np.zeros((len(shrinks), len(k_list)))                                         # validation RMSE, one row per shrink, one column per k
    for a, lam in enumerate(shrinks):                                                 # every shrink strength
        S = S0 * n / (n + lam) if lam else S0                                         # the cheap way to shrink: scale the stored similarities
        for b, k in enumerate(k_list):                                                # every neighbourhood size
            c, _ = predict(D, M, S, u, i, k)                                          # the correction for every validation rating
            E[a, b] = rmse(np.clip(offset + c, 1, 5), r)                              # added to the offset, clipped to 1-5
    a, b = np.unravel_index(np.argmin(E), E.shape)                                    # the lowest error's row and column
    return E, shrinks[a], k_list[b]                                                   # the table, and the chosen shrink and k


def print_grid(E, title):                                                             # a validation table, one row per shrink
    print(f"  {title}: validation RMSE")
    print(f"  {'shrink':>8} " + " ".join(f"{'k ' + str(k):>8}" for k in KS))
    for a, lam in enumerate(SHRINKS):                                                 # one row per shrink
        print(f"  {lam:>8} " + " ".join(f"{e:>8.4f}" for e in E[a]))


def demo_user(u, i, r, tr, te, fi, va, nu, ni):                                       # section 4
    print("\n" + "=" * 70)
    print("4. USER-BASED -- the k most similar users who rated the film")
    print("=" * 70)
    t0 = time.time()                                                                  # start the clock
    Rf, Mf = to_matrix(u[fi], i[fi], r[fi], nu, ni); um = user_means(Rf, Mf)          # training without the validation ratings; each user's mean
    D = (Rf - um[:, None]) * Mf                                                       # each rating minus that user's mean
    S0, n = similarity(D.T, Mf.T)                                                     # users x users
    E, lam, k = grid(Rf, Mf, D, S0, n, um[u[va]], u[va], i[va], r[va], predict_user_knn, KS, SHRINKS)  # every (shrink, k) on validation
    print_grid(E, "user-based")                                                       # the table
    print(f"  chosen: k {k}, shrink {lam} ({time.time() - t0:.1f} s for the grid)")
    R, M = to_matrix(u[tr], i[tr], r[tr], nu, ni); um = user_means(R, M)              # refit on all the training ratings
    D = (R - um[:, None]) * M                                                         # each rating minus the user's mean
    S, n = similarity(D.T, M.T, shrink=lam)                                           # users x users, with the chosen shrink
    c, used = predict_user_knn(D, M, S, u[te], i[te], k)                              # the k most similar users who rated each test film
    p = np.clip(um[u[te]] + c, 1, 5)                                                  # user's mean + correction, clipped to 1-5
    print(f"test: RMSE {rmse(p, r[te]):.4f}, MAE {mae(p, r[te]):.4f}; neighbours used: median {np.median(used):.0f}, "
          f"none for {(used == 0).mean():.3f} of test ratings (they get the user's mean)")  # pairs where no similar user rated the film
    print("Actual output from this lesson's run.")
    return p, E                                                                       # test predictions, and the grid for the figure


def demo_item(u, i, r, tr, te, fi, va, nu, ni, lu, li, rows, p_user, pop):            # section 5
    print("\n" + "=" * 70)
    print("5. ITEM-BASED -- the k films most like this one that the user rated")
    print("=" * 70)
    t0 = time.time()                                                                  # start the clock
    Rf, Mf = to_matrix(u[fi], i[fi], r[fi], nu, ni)                                   # training without the validation ratings
    best = {}                                                                         # the best result for each similarity
    for centre in CENTRES:                                                            # which ratings the similarity compares
        X, D, base = item_parts(Rf, Mf, lu, li, centre)                               # compared values, residuals, offset
        S0, n = similarity(X, Mf)                                                     # every pair of films
        E, lam, k = grid(Rf, Mf, D, S0, n, base[u[va], i[va]], u[va], i[va], r[va], predict_item_knn, KS, SHRINKS)  # every (shrink, k) on validation
        best[centre] = (E.min(), lam, k, E)                                           # (error, shrink, k, the whole table)
        print(f"  similarity on {centre:<40} best validation RMSE {E.min():.4f} (k {k}, shrink {lam})")
    centre = min(best, key=lambda c: best[c][0]); _, lam, k, E = best[centre]         # the similarity with the lowest validation error
    print_grid(E, f"item-based, {centre}")                                            # its table
    print(f"  chosen: {centre}, k {k}, shrink {lam} ({time.time() - t0:.1f} s for the grids)")
    X, D, base = item_parts(Rf, Mf, lu, li, centre); S0, n = similarity(X, Mf)       # the chosen similarity, unshrunk
    c, _ = predict_item_knn(D, Mf, S0 * n, u[va], i[va], k)                           # weights s x n: the limit of s x n / (n + shrink), rescaled
    print(f"  the limit as shrink grows (weights s x n; the 1 / shrink cancels in the weighted average): validation RMSE "
          f"{rmse(np.clip(base[u[va], i[va]] + c, 1, 5), r[va]):.4f} at k {k}")
    R, M = to_matrix(u[tr], i[tr], r[tr], nu, ni)                                     # refit on all the training ratings
    X, D, base = item_parts(R, M, lu, li, centre)                                     # with the chosen similarity
    S, n = similarity(X, M, shrink=lam)                                               # with the chosen shrink
    c, used = predict_item_knn(D, M, S, u[te], i[te], k)                              # the k most similar films each test user rated
    p = np.clip(base[u[te], i[te]] + c, 1, 5)                                         # bias model + correction, clipped to 1-5
    cs, _ = predict_item_knn(D, M, shuffled(S, 0), u[te], i[te], k)                   # control: same weights, random neighbours
    ps = np.clip(base[u[te], i[te]] + cs, 1, 5)                                       # the control's predictions
    allrows = dict(rows); allrows["user-based"] = p_user; allrows["item-based"] = p   # every model's test predictions, in table order
    allrows["control: item-based, shuffled similarities"] = ps                        # the control last
    print(f"{'test, all models':>44}    RMSE     MAE")
    for name, q in allrows.items():                                                   # one row per model
        print(f"{name:>44}  {rmse(q, r[te]):.4f}  {mae(q, r[te]):.4f}")
    print(f"item-based neighbours used: median {np.median(used):.0f}; none for {(used == 0).mean():.3f} of test ratings (they get the bias model)")
    nr = M.sum(1)[u[te]]                                                              # how many training ratings each test user has
    print("test RMSE by how many training ratings the user has:")
    print(f"  {'ratings':>10} {'test pairs':>10} {'bias model':>11} {'user-based':>11} {'item-based':>11}")
    bm = allrows["mean + user bias + film bias"]                                      # the bias model's predictions
    for lo, hi in ((0, 49), (50, 149), (150, 10 ** 9)):                               # light, medium and heavy users
        g = (nr >= lo) & (nr <= hi); lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+" # the users in this band
        print(f"  {lab:>10} {g.sum():>10,} {rmse(bm[g], r[te][g]):>11.4f} {rmse(p_user[g], r[te][g]):>11.4f} {rmse(p[g], r[te][g]):>11.4f}")
    pf = pop[i[te]]                                                                   # each test film's total number of ratings
    print("test RMSE by how many ratings the film has (in all the data):")
    for lo, hi in ((0, 19), (20, 99), (100, 10 ** 9)):                                # rare, middling and popular films
        g = (pf >= lo) & (pf <= hi); lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+" # the films in this band
        print(f"  {lab:>10} {g.sum():>10,} {rmse(bm[g], r[te][g]):>11.4f} {rmse(p_user[g], r[te][g]):>11.4f} {rmse(p[g], r[te][g]):>11.4f}")
    print("Actual output from this lesson's run.")
    return allrows, (centre, lam, k), best                                            # predictions, chosen setting, all grids


def item_scores(R, M, lu, li, setting, damp=0.0):      # item-based predicted ratings for every unrated pair (-inf where rated)
    centre, lam, k = setting                                                          # the chosen similarity, shrink and k
    X, D, base = item_parts(R, M, lu, li, centre)                                     # what the similarity compares, and the residuals
    S, _ = similarity(X, M, shrink=lam)                                               # films x films, shrunk
    uu, ii = np.nonzero(~M)                                                           # every film each user has not rated
    c, _ = predict_item_knn(D, M, S, uu, ii, k, damp)                                 # the correction for every unrated pair
    P = np.full(M.shape, -np.inf); P[uu, ii] = base[uu, ii] + c                       # rated films can never be picked
    return P, S, D, base                                                              # scores, similarities, residuals, bias model


def precision_at_10(Q, liked):                          # mean precision@10 and recall@10 over users with a liked held-out film
    who = np.flatnonzero(liked.sum(1) > 0)                                            # users with something to find
    top = np.argsort(-Q, axis=1, kind="stable")[:, :10]                               # the 10 best-scored films per user
    h = np.take_along_axis(liked, top, 1)[who]                                        # which of them are hits
    return (h.sum(1) / 10).mean(), (h.sum(1) / liked[who].sum(1)).mean(), top[who], len(who)  # precision@10, recall@10, the lists, and how many users


def demo_topn(u, i, r, tr, te, fi, va, nu, ni, lu, li, setting, titles):              # section 6
    print("\n" + "=" * 70)
    print("6. FROM PREDICTED RATINGS TO A TOP-10 LIST")
    print("=" * 70)
    t0 = time.time()                                                                  # start the clock
    centre, lam, k = setting                                                          # the setting chosen in section 5
    R, M = to_matrix(u[tr], i[tr], r[tr], nu, ni)                                     # all the training ratings
    P, S, D, base = item_scores(R, M, lu, li, setting)                                # predicted ratings for every unrated pair
    cnt = M.sum(0).astype(float)                                                      # training ratings per film
    liked = np.zeros((nu, ni), bool); g = r[te] >= 4; liked[u[te][g], i[te][g]] = True  # test films rated 4 or 5
    print("for each user, the 10 unrated films with the highest score; a hit is a test film the user rated 4 or 5")
    rows = {"item-based predicted rating": P, "bias model (mean + biases)": np.where(M, -np.inf, base),  # each ranking's scores; rated films get -inf
            "popularity (most rated)": np.where(M, -np.inf, cnt[None, :]),            # how many training ratings each film has
            "random": np.where(M, -np.inf, np.random.RandomState(0).rand(nu, ni))}    # a random order, for comparison
    out = {}                                                                          # results per ranking
    for name, Q in rows.items():                                                      # every ranking
        out[name] = precision_at_10(Q, liked)                                         # scored against the liked test films
    print(f"{out['random'][3]} users have at least one such film ({time.time() - t0:.1f} s to score every unrated pair)")
    print(f"{'ranked by':>34}  precision@10  recall@10  ratings per recommended film (median)")
    for name, (pr, rc, top, _) in out.items():                                        # one row per ranking
        print(f"{name:>34}  {pr:>12.4f}  {rc:>9.4f}  {np.median(cnt[top]):>8.0f}")
    x = 0; top = np.argsort(-P[x])[:5]                                                # user 1's list
    print(f"user 1 rated {M[x].sum()} films in training; item-based top 5, the rated film that pulled each up most, and the")
    print("total similarity behind it (the weighted average divides by this, so a tiny total still gives a full correction):")
    rated = np.flatnonzero(M[x])                                                      # user 1's training films
    for j in top:                                                                     # each recommended film
        nb = rated[np.argsort(-S[j, rated])[:k]]; sj = np.where(S[j, nb] > 0, S[j, nb], 0.0)  # the k neighbours used
        b = nb[np.argmax(sj * D[x, nb])]                                              # the largest single push upward
        print(f"    {P[x, j]:.2f}  {titles[j][:34]:<34} ({int(cnt[j]):>3} ratings) total {sj.sum():.3f}  because of {titles[b][:24]} ({int(R[x, b])})")
    print("a fix: add a constant d to that total (dividing by total + d), so a film with little evidence keeps close to the")
    print("bias model; d chosen on validation precision@10 (training without the validation ratings; hits = validation films rated 4-5)")
    Rf, Mf = to_matrix(u[fi], i[fi], r[fi], nu, ni)                                   # training without the validation ratings
    lv = np.zeros((nu, ni), bool); g = r[va] >= 4; lv[u[va][g], i[va][g]] = True      # validation films rated 4 or 5
    best = None                                                                       # nothing tried yet
    for dmp in DAMPS:                                                                 # every damping constant
        pv = precision_at_10(item_scores(Rf, Mf, lu, li, setting, dmp)[0], lv)[0]     # validation precision@10 with this d
        print(f"  d {dmp:<5} validation precision@10 {pv:.4f}")
        if best is None or pv > best[0]:                                              # keep the highest
            best = (pv, dmp)                                                          # (validation precision, d)
    Pd = item_scores(R, M, lu, li, setting, best[1])[0]                               # refit on all the training ratings with the chosen d
    out["item-based, damped (d " + str(best[1]) + ")"] = precision_at_10(Pd, liked)   # scored on the test films
    pr, rc, top, _ = out["item-based, damped (d " + str(best[1]) + ")"]               # its numbers
    print(f"test, d {best[1]}: precision@10 {pr:.4f}, recall@10 {rc:.4f}, ratings per recommended film (median) {np.median(cnt[top]):.0f}")
    print(f"  still {out['popularity (most rated)'][0] / pr:.1f} times below popularity ({out['popularity (most rated)'][0]:.4f}); "
          f"test RMSE with d {best[1]}: {rmse(np.clip(base[u[te], i[te]] + predict_item_knn(D, M, S, u[te], i[te], k, best[1])[0], 1, 5), r[te]):.4f}")  # the damping's effect on rating accuracy
    print("Actual output from this lesson's run.")
    return {m: v[:2] for m, v in out.items()}                                         # precision and recall per ranking, for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(pop, pu, S0, n, Eu, best, lam, allrows, rt, topn, out):              # the six panels of the results figure
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))                                  # two rows of three
    ax = axs[0, 0]; ax.loglog(np.arange(1, len(pop) + 1), np.sort(pop)[::-1], label="films", color="#4C72B0")  # rating counts, largest first, on log axes
    ax.loglog(np.arange(1, len(pu) + 1), np.sort(pu)[::-1], label="users", color="#C44E52")  # the same for users
    ax.set_xlabel("rank (most ratings first)"); ax.set_ylabel("number of ratings"); ax.legend(fontsize=8)  # axis labels
    ax.set_title("a long tail: a few films get most ratings", fontsize=10)            # panel title
    ax = axs[0, 1]; iu = np.triu_indices(S0.shape[0], 1); c, s = n[iu], S0[iu]; k = c > 0  # every pair of films once; pairs that share a rater
    rs = np.random.RandomState(0).choice(np.flatnonzero(k), 40000, replace=False)     # a sample, so the plot stays light
    ax.scatter(c[rs], s[rs], s=2, alpha=0.25, color="#4C72B0", label="no shrinkage")  # similarity against shared raters
    ax.scatter(c[rs], (s * c / (c + 100))[rs], s=2, alpha=0.25, color="#DD8452", label="shrinkage 100 (as in section 3)")  # the same pairs, shrunk
    ax.set_xscale("log"); ax.set_xlabel("raters the two films share"); ax.set_ylabel("similarity")  # a log axis for the counts
    ax.legend(fontsize=8, markerscale=4); ax.set_title("few shared raters: extreme similarities", fontsize=10)  # legend and title
    ax = axs[0, 2]                                                                    # validation curves
    ax.plot(KS, Eu.min(0), marker="o", label="user-based (best shrink per k)")        # user-based: best over the shrinks at each k
    for name, mk in zip(CENTRES, "s^vD"):                                             # one curve per item-based similarity
        ax.plot(KS, best[name][3].min(0), marker=mk, label="item-based, " + name)     # best over the shrinks at each k
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.set_xlabel("neighbours k"); ax.set_ylabel("validation RMSE")  # k on a log axis
    ax.set_ylim(top=1.05); ax.legend(fontsize=6.5); ax.set_title("choosing k on validation", fontsize=10)  # room for the legend; title
    ax = axs[1, 0]; names = list(allrows); vals = [rmse(allrows[m], rt) for m in names]  # test RMSE per model
    ax.barh(range(len(names)), vals, color=["#BBBBBB"] * 4 + ["#4C72B0", "#55A868", "#C44E52"])  # baselines grey, models coloured, control red
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()  # model names, top to bottom
    ax.set_xlim(0.85, max(vals) + 0.06); ax.set_xlabel("test RMSE (stars)")       # zoom in on the differences
    for y, v in enumerate(vals):                                                      # the value at the end of each bar
        ax.text(v + 0.003, y, f"{v:.4f}", va="center", fontsize=8)                    # written next to the bar
    ax.set_title("test error, every model", fontsize=10)                              # panel title
    ax = axs[1, 1]                                                                    # predictions against the true star rating
    for s_ in range(1, 6):                                                            # one box per true rating
        ax.boxplot(allrows["item-based"][rt == s_], positions=[s_], widths=0.5, showfliers=False)  # the middle half of the predictions
    ax.plot([1, 5], [1, 5], color="gray", lw=0.7, ls="--"); ax.set_xlabel("true rating"); ax.set_ylabel("item-based prediction")  # a perfect prediction would lie on this line
    ax.set_title("predictions pulled toward the middle", fontsize=10)                 # panel title
    ax = axs[1, 2]; names = list(topn); x = np.arange(len(names))                     # top-10 precision per ranking
    ax.bar(x, [topn[m][0] for m in names], color=["#55A868", "#BBBBBB", "#DD8452", "#999999", "#8172B2"])  # one bar per ranking
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" predicted rating", "").replace(" (mean + biases)", "").replace(" (most rated)", "").replace(", damped", ",\ndamped") for m in names], fontsize=8)  # short labels
    for xx, m in zip(x, names):                                                       # the value above each bar
        ax.text(xx, topn[m][0] + 0.003, f"{topn[m][0]:.4f}", ha="center", fontsize=8) # written above the bar
    ax.set_ylabel("precision@10"); ax.set_title("top-10 lists: hits among test films rated 4-5", fontsize=10)  # axis label and title
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)                     # save to disk


def draw_diagram():                                     # user-based and item-based, side by side
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 15); ax.set_ylim(0.2, 5.2); ax.axis("off")  # a wide, empty canvas

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))  # the box
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)                    # its text

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))  # an arrow from (x0, y0) to (x1, y1)
    box(1.3, 2.6, 2.0, 1.6, "ratings table\nusers x films\n(6% filled)", "#FFF2CC")   # the input
    box(4.3, 3.9, 2.4, 1.0, "centre each user's\nratings on their mean", "#DDEBF7")   # user-based, step 1
    box(7.3, 3.9, 2.4, 1.0, "similarity of users\nover the films\nboth rated", "#DDEBF7")  # user-based, step 2
    box(10.3, 3.9, 2.4, 1.0, "k most similar users\nwho rated film i", "#DDEBF7")     # user-based, step 3
    box(4.3, 1.3, 2.4, 1.0, "to compare films: minus\nthe user's mean; to\naverage: minus the\nbias model", "#DDEBF7") # item-based, step 1
    box(7.3, 1.3, 2.4, 1.0, "similarity of films\nover the users\nwho rated both", "#DDEBF7")  # item-based, step 2
    box(10.3, 1.3, 2.4, 1.0, "k films most like i\nthat user u rated", "#DDEBF7")     # item-based, step 3
    box(13.4, 3.9, 2.4, 1.0, "user's mean +\nweighted average\nof their deviations", "#E2F0D9")  # user-based prediction
    box(13.4, 1.3, 2.4, 1.0, "bias model +\nweighted average\nof u's residuals", "#E2F0D9")  # item-based prediction
    for y in (3.9, 1.3):                                                              # both rows
        arrow(5.5, y, 6.1, y); arrow(8.5, y, 9.1, y); arrow(11.5, y, 12.2, y)         # the three arrows along each row
    arrow(2.3, 3.0, 3.1, 3.7); arrow(2.3, 2.2, 3.1, 1.5)                              # the table feeds both rows
    ax.text(5.8, 4.75, "user-based: people who agreed before will agree again", fontsize=9.5, style="italic")  # the idea behind the top row
    ax.text(5.8, 0.4, "item-based: films rated alike are liked by the same people", fontsize=9.5, style="italic")  # the idea behind the bottom row
    ax.text(8.8, 2.6, "shrinkage: similarity x n / (n + shrink), n = how many ratings the two share",  # the shrinkage formula, between the rows
            ha="center", fontsize=8.5, color="#444444")                               # (continued)
    ax.set_title("Day 154: predicting user u's rating of film i from other ratings", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "collaborative_filtering_diagram.png"), dpi=120); plt.close(fig)  # save to disk


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 155: matrix factorization. Instead of comparing users or films directly, describe each by a short list")
    print("of hidden numbers, learned so that their products rebuild the known ratings; SVD and its regularised")
    print("relative (Funk SVD) are built from scratch and compared with today's neighbours.")


def main():                                             # runs every section in order
    u, i, r, t = load_movielens(); titles, genres, gnames = load_items()              # the ratings, and the films
    nu, ni = u.max() + 1, i.max() + 1                                                 # users and films, counted from 0
    pop = demo_data(u, i, r, t, titles, genres, gnames)                               # 1. the data
    tr, te, fi, va, lu, li, rows = demo_baselines(u, i, r, nu, ni)                    # 2. split and baselines
    S0, n = demo_similarity(u, i, r, tr, nu, ni, titles, lu, li)                      # 3. similarity
    p_user, Eu = demo_user(u, i, r, tr, te, fi, va, nu, ni)                           # 4. user-based
    allrows, setting, best = demo_item(u, i, r, tr, te, fi, va, nu, ni, lu, li, rows, p_user, pop)  # 5. item-based
    topn = demo_topn(u, i, r, tr, te, fi, va, nu, ni, lu, li, setting, titles)                # 6. top-10 lists
    plot_results(pop, np.bincount(u), S0, n, Eu, best, setting[1], allrows, r[te], topn,  # the results figure
                 os.path.join(HERE, "collaborative_filtering_results.png"))           # (saved next to the script)
    draw_diagram()                                                                    # the method
    print("\nsaved collaborative_filtering_results.png\nsaved collaborative_filtering_diagram.png")
    note_on_what_comes_next()                                                         # a preview of Day 155


if __name__ == "__main__":
    main()                                                                            # run everything
