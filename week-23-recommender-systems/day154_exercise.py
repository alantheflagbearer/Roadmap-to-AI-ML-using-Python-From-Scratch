"""
Day 154 Exercise -- Collaborative Filtering, User-Based and Item-Based

The first of Week 23's parts for Day 160's neural recommender.

    load_movielens()                     MovieLens 100K: users, films, stars, times (downloaded on first use)
    load_items()                         each film's title and its 19 genre flags, and the genre names
    split_ratings(n, frac, seed)         a random split of the ratings into kept and held out
    to_matrix(u, i, r, nu, ni)           the ratings as a users x films table, and a mask of the filled cells
    fit_biases(R, M, lam_u, lam_i)       rating = mean + user bias + film bias, with penalties
    similarity(X, M, shrink)             cosine between columns over the rows both have, with shrinkage
    predict_user_knn(D, M, S, u, i, k)   the weighted average of the k most similar users' deviations
    predict_item_knn(D, M, S, u, i, k)   the weighted average of the user's residuals on the k most similar films
    rmse(p, r), mae(p, r)                the two error measures, in stars

Data: MovieLens 100K, F. M. Harper and J. A. Konstan, The MovieLens Datasets:
History and Context, ACM TiiS 5(4), 2015. Downloaded to ~/movielens_data,
never stored in this repository.

Checked by hand: the dataset's sizes, a split, biases on a table that is
exactly mean + user part + film part, a cosine worked on paper and its
shrunk value, a user-based and an item-based prediction worked on paper,
the damping constant, and the two error measures.
"""
import os                                              # os: file paths
import zipfile                                         # zipfile: to unpack the downloaded archive
import urllib.request                                  # urllib: to download the data the first time
import numpy as np                                     # numpy: every array operation below is built on this
DATA = os.path.join(os.path.expanduser("~"), "movielens_data")  # where the data is kept: outside the repository
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"  # GroupLens' official copy


def load_movielens(folder=DATA):                        # NEW: Day 154's version, unchanged
    path = os.path.join(folder, "ml-100k", "u.data")                                  # one rating per line: user, film, stars, time
    if not os.path.exists(path):                                                      # first run: fetch and unpack the archive
        os.makedirs(folder, exist_ok=True)                                            # the data folder, if missing
        z = os.path.join(folder, "ml-100k.zip")                                       # where the archive is saved
        urllib.request.urlretrieve(URL, z)                                            # 4.9 MB from GroupLens
        zipfile.ZipFile(z).extractall(folder)                                         # creates folder/ml-100k/
    a = np.loadtxt(path, dtype=np.int64)                                              # 100,000 rows of 4 whole numbers
    return a[:, 0] - 1, a[:, 1] - 1, a[:, 2].astype(float), a[:, 3]                   # users and films from 0; stars; seconds since 1970


def load_items(folder=DATA):                            # NEW: Day 154's version, unchanged
    rows = [ln.split("|") for ln in open(os.path.join(folder, "ml-100k", "u.item"), encoding="latin-1").read().strip().split("\n")]  # one row per film, fields split at '|'
    names = [ln.split("|")[0] for ln in open(os.path.join(folder, "ml-100k", "u.genre"), encoding="latin-1").read().strip().split("\n")]  # the 19 genre names, in column order
    titles = [r[1] for r in rows]                                                     # e.g. "Star Wars (1977)"
    genres = np.array([[int(g) for g in r[5:24]] for r in rows])                     # 1 if the film has that genre
    return titles, genres, names                                                      # titles, a films x 19 table of 0/1, the genre names


def split_ratings(n, frac, seed):                       # NEW: Day 154's version, unchanged
    p = np.random.RandomState(seed).permutation(n)                                    # every rating's position, shuffled
    m = int(round(n * frac))                                                          # how many go to the held-out part
    return np.sort(p[m:]), np.sort(p[:m])                                             # (kept, held out), each in file order


def to_matrix(u, i, r, nu, ni):                         # NEW: Day 154's version, unchanged
    R = np.zeros((nu, ni)); M = np.zeros((nu, ni), bool)                              # 0 and False where nothing was rated
    R[u, i] = r; M[u, i] = True                                                       # fill in the known ratings
    return R, M                                                                       # the table, and the filled cells


def fit_biases(R, M, lam_u, lam_i, iters=20):           # NEW: Day 154's version, unchanged
    mu = R[M].mean()                                                                  # the average of every known rating
    bu = np.zeros(R.shape[0]); bi = np.zeros(R.shape[1])                              # start every bias at 0
    nu, ni = M.sum(1), M.sum(0)                                                       # ratings per user, per film
    for _ in range(iters):                                                            # alternate: each step is an exact minimum
        bi = ((R - mu - bu[:, None]) * M).sum(0) / np.maximum(ni + lam_i, 1e-9)       # films, with the users' biases fixed (0 if unrated)
        bu = ((R - mu - bi[None, :]) * M).sum(1) / np.maximum(nu + lam_u, 1e-9)       # users, with the films' biases fixed
    return mu, bu, bi                                                                 # the global mean and the two sets of biases


def similarity(X, M, shrink=0.0):                       # NEW: Day 154's version, unchanged
    Mf = M.astype(float); X2 = X * X                                                  # the mask as numbers; squared entries
    num = X.T @ X                                                                     # sum over shared rows of x_a x_b (X is 0 where unfilled)
    den = np.sqrt(X2.T @ Mf) * np.sqrt(Mf.T @ X2)                                     # the two lengths, each over the shared rows only
    S = np.divide(num, den, out=np.zeros_like(num), where=den > 0)                    # 0 when there is nothing to compare
    n = Mf.T @ Mf                                                                     # how many rows each pair shares
    if shrink:                                                                        # few shared rows: pull the similarity toward 0
        S = S * n / (n + shrink)                                                      # n / (n + shrink) is near 0 for small n, near 1 for large n
    np.fill_diagonal(S, 0.0)                                                          # a column is never its own neighbour
    return S, n                                                                       # similarities, and the shared counts behind them


def _groups(keys):                                      # NEW: Day 154's version, unchanged
    order = np.argsort(keys, kind="stable"); ks = keys[order]                         # keys sorted, ties kept in order
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])                           # where each new key begins
    return [(ks[a], order[a:b]) for a, b in zip(starts, np.r_[starts[1:], len(ks)])]  # (key, positions of that key) for every key


def _weighted(s, d, k, damp=0.0):                       # NEW: Day 154's version, unchanged
    kk = min(k, s.shape[1])                                                           # cannot use more neighbours than exist
    top = np.argpartition(-s, kk - 1, axis=1)[:, :kk]                                 # the kk largest similarities in each row (unordered)
    st = np.take_along_axis(s, top, 1); dt = np.take_along_axis(d, top, 1)            # their similarities and their values
    st = np.where(st > 0, st, 0.0)                                                    # only positive similarities count
    w = st.sum(1)                                                                     # total weight in each row
    return np.where(w > 0, (st * dt).sum(1) / np.where(w > 0, w + damp, 1.0), 0.0), (st > 0).sum(1)  # damp > 0: little weight, little correction


def predict_user_knn(D, M, S, users, items, k):         # NEW: Day 154's version, unchanged
    out = np.zeros(len(users)); used = np.zeros(len(users), int)                      # correction, and neighbours used
    for ii, q in _groups(items):                                                      # one film at a time
        raters = np.flatnonzero(M[:, ii])                                             # who rated this film
        if len(raters) == 0:                                                          # nobody: no correction
            continue                                                                  # skip to the next film
        s = S[np.ix_(users[q], raters)]                                               # each asking user against each rater
        d = np.broadcast_to(D[raters, ii], s.shape)                                   # the raters' deviations from their own means
        out[q], used[q] = _weighted(s, d, k)                                          # weighted average of the k most similar raters
    return out, used                                                                  # the correction to add to each user's mean, and how many neighbours gave it


def predict_item_knn(D, M, S, users, items, k, damp=0.0):  # NEW: Day 154's version, unchanged
    out = np.zeros(len(users)); used = np.zeros(len(users), int)                      # correction, and neighbours used
    for uu, q in _groups(users):                                                      # one user at a time
        rated = np.flatnonzero(M[uu])                                                 # the films this user rated
        s = S[np.ix_(items[q], rated)]                                                # each film asked about against each film rated
        d = np.broadcast_to(D[uu, rated], s.shape)                                    # the user's deviations on the films rated
        out[q], used[q] = _weighted(s, d, k, damp)                                    # weighted average of the k most similar films the user rated
    return out, used                                                                  # the correction to add to the offset, and how many neighbours gave it


def rmse(p, r):                                         # NEW: Day 154's version, unchanged
    return float(np.sqrt(np.mean((p - r) ** 2)))                                      # large errors count more than in MAE


def mae(p, r):                                          # NEW: Day 154's version, unchanged
    return float(np.mean(np.abs(p - r)))                                              # the typical size of an error


u, i, r, t = load_movielens(); titles, genres, gnames = load_items()
print(f"MovieLens 100K: 100,000 ratings, 943 users, 1,682 films, 19 genres; gives {len(r):,}, {u.max() + 1}, {i.max() + 1:,}, {len(gnames)}")
print(f"film 50 (index 49) is Star Wars (1977); gives {titles[49]}")
kept, out = split_ratings(10, 0.2, seed=0)
print(f"split of 10 ratings, 20% held out: 8 kept, 2 out, no overlap, all 10 used; gives {len(kept)}, {len(out)}, "
      f"{len(np.intersect1d(kept, out))}, {len(np.union1d(kept, out))}")
a, b = np.array([0.0, 1.0, -1.0]), np.array([0.0, -0.5, 0.5])                        # user parts, film parts
R = 3.0 + a[:, None] + b[None, :]; M = np.ones_like(R, bool)                          # a table that is exactly mean + user + film
mu, bu, bi = fit_biases(R, M, 0, 0)
print(f"biases on a full table that is exactly 3 + user part + film part: rebuilt to within {np.abs(mu + bu[:, None] + bi[None, :] - R).max():.1e}")
X = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]], float)  # 5 users x 4 films, 0 = unrated
S, n = similarity(X, X > 0)
print("films 1 and 4 share users 1-4: (5x1 + 4x1 + 1x5 + 1x4) / (sqrt(25+16+1+1) x sqrt(1+1+25+16)) = 18 / 43 = 0.4186;")
print(f"  similarity gives {S[0, 3]:.4f} from {int(n[0, 3])} shared raters; with shrink 4, x 4 / (4 + 4): 0.2093, gives {similarity(X, X > 0, 4)[0][0, 3]:.4f}")
D = np.array([[1.0, -1.0, 0.0]]); Mi = np.array([[True, True, False]])                 # one user: +1 on film A, -1 on film B, film C unrated
Si = np.array([[0, 0, 0.5], [0, 0, 0.25], [0.5, 0.25, 0]])                           # C is 0.5 like A and 0.25 like B
one = lambda k, d=0.0: predict_item_knn(D, Mi, Si, np.array([0]), np.array([2]), k, d)[0][0]
print(f"item-based, film C: k 2 gives (0.5 x 1 + 0.25 x -1) / 0.75 = 0.3333, k 1 gives 1; with damping 0.25, 0.25 / 1.0 = 0.25;")
print(f"  gives {one(2):.4f}, {one(1):.4f}, {one(2, 0.25):.4f}")
Du = np.array([[0.0], [2.0], [-1.0]]); Mu = np.array([[False], [True], [True]])        # users 2 and 3 rated the film: +2 and -1 from their means
Su = np.array([[0, 0.6, 0.2], [0.6, 0, 0], [0.2, 0, 0]])                              # user 1 is 0.6 like user 2, 0.2 like user 3
c = predict_user_knn(Du, Mu, Su, np.array([0]), np.array([0]), 2)[0][0]
print(f"user-based, user 1: (0.6 x 2 + 0.2 x -1) / 0.8 = 1.25 above user 1's mean; gives {c:.4f}")
print(f"errors of predictions 1, 2 against 2, 4: RMSE sqrt((1 + 4) / 2) = 1.5811, MAE 1.5; gives {rmse(np.array([1.0, 2]), np.array([2.0, 4])):.4f}, "
      f"{mae(np.array([1.0, 2]), np.array([2.0, 4])):.4f}")
