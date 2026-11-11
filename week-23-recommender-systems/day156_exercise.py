"""
Day 156 Exercise -- Content-Based Filtering

The third of Week 23's parts for Day 160's neural recommender.

    load_years()                         each film's release year from u.item (-1 when missing)
    film_features(genres, years)         19 genre flags + 8 decade flags per film
    content_similarity(X)                cosine of every pair of films' feature vectors
    fit_shared_profile(X, T, M, lam)     one ridge regression for everyone
    fit_profiles(X, T, M, lam, prior)    a ridge regression per user, shrunk toward the prior
    content_predict(W, X, u, i)          profile . film features for each (user, film) pair
    cold_split(i, ni, frac, seed)        hold out whole films, with every rating they have

Checked by hand: Star Wars's year, the features of two made-up films, a cosine
of 0.5, a shared profile and a user profile worked on paper, a prediction, and
a cold split that holds out every rating of the chosen films and no other.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported part quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: to find Day 154's part

import numpy as np                                     # numpy: every array operation below is built on this

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so Day 154's part can be imported from here
with contextlib.redirect_stdout(io.StringIO()):         # day154_exercise prints its hand checks when imported
    import day154_exercise as d154                      # Day 154: the data folder (DATA) and the loaders


def load_years(folder=d154.DATA):                       # NEW: Day 156's version, unchanged
    path = os.path.join(folder, "ml-100k", "u.item")                                  # Day 154 downloaded it
    rows = open(path, encoding="latin-1").read().strip().split("\n")                  # one film per line, latin-1 like load_items
    return np.array([int(x.split("|")[2][-4:]) if x.split("|")[2] else -1 for x in rows])  # the date ends in the year


def film_features(genres, years):                       # NEW: Day 156's version, unchanged
    dec = np.clip((years - 1920) // 10, 0, 7)                                         # 0 = before 1930, ..., 7 = 1990s
    D = np.eye(8)[dec] * (years > 0)[:, None]                                         # one decade flag; none if the year is missing
    return np.c_[genres, D].astype(float)                                             # films x 27


def content_similarity(X):                              # NEW: Day 156's version, unchanged
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)              # each row scaled to length 1
    S = Xn @ Xn.T                                                                     # every pair's cosine at once
    np.fill_diagonal(S, 0.0)                                                          # a film is never its own neighbour
    return S                                                                          # films x films


def fit_shared_profile(X, T, M, lam):                   # NEW: Day 156's version, unchanged
    uu, ii = np.nonzero(M)                                                            # every known (user, film) pair
    A = X[ii]                                                                         # the film's features, one row per rating
    return np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ T[uu, ii])      # ridge solution, 27 numbers


def fit_profiles(X, T, M, lam, prior=None):             # NEW: Day 156's version, unchanged
    w0 = np.zeros(X.shape[1]) if prior is None else prior                             # what an unknown user's profile is
    W = np.tile(w0, (M.shape[0], 1))                                                  # every user starts at the prior
    for a in range(M.shape[0]):                                                       # one user at a time
        j = np.flatnonzero(M[a]); A = X[j]                                            # the films user a rated, as features
        W[a] += np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ (T[a, j] - A @ w0))  # ridge on what the prior misses
    return W                                                                          # users x 27


def content_predict(W, X, u, i):                        # NEW: Day 156's version, unchanged
    return (W[u] * X[i]).sum(1)                                                       # one number per pair


def cold_split(i, ni, frac, seed, exclude=None):        # NEW: Day 156's version, unchanged
    pool = np.arange(ni) if exclude is None else np.flatnonzero(~exclude)              # films still available
    rng = np.random.RandomState(seed); cold = np.zeros(ni, bool)                      # seeded, so the split repeats
    cold[rng.permutation(pool)[:int(round(frac * ni))]] = True                       # frac of all films chosen
    return cold, np.flatnonzero(cold[i])                                              # the films, and their ratings' positions


titles, genres, gnames = d154.load_items(); years = load_years()
sw = titles.index("Star Wars (1977)")
print(f"Star Wars (1977): load_years gives {years[sw]}; films with no year: {(years < 0).sum()} of {len(years)}")
g = np.zeros((2, 19), int); g[0, [1, 15]] = 1; g[1, 8] = 1                            # film 0: Action + Sci-Fi; film 1: Drama
F = film_features(g, np.array([1977, -1]))
print(f"film_features, Action + Sci-Fi from 1977: flags at {np.flatnonzero(F[0]).tolist()} (1, 15 and 19 + 5 = 24); "
      f"Drama, year missing: {np.flatnonzero(F[1]).tolist()} (8 only)")
S = content_similarity(np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
print(f"cosine of (1, 1, 0) and (1, 0, 1) = 1 / (sqrt 2 x sqrt 2) = 0.5; content_similarity gives {S[0, 1]:.4f}, diagonal {S[0, 0]:.1f}")
X = np.eye(2); T = np.array([[1.0, -2.0], [3.0, 0.0]]); M = np.array([[True, True], [True, False]])
w = fit_shared_profile(X, T, M, 2.0)
print(f"shared profile, film A residuals 1 and 3, film B residual -2, penalty 2: A (1 + 3) / (2 + 2) = 1.0, "
      f"B -2 / (1 + 2) = -0.6667; gives {np.round(w, 4).tolist()}")
Wp = fit_profiles(X, T, M, 2.0, prior=w)
print(f"user 2's profile (rated A only, 3), prior {w[0]:.1f}, penalty 2: A 1.0 + (3 - 1.0) x 1 / (1 + 2) = 1.6667, "
      f"B stays at the prior -0.6667; gives {np.round(Wp[1], 4).tolist()}")
print(f"prediction for user 2 on film B: profile . features = -0.6667; content_predict gives "
      f"{content_predict(Wp, X, np.array([1]), np.array([1]))[0]:.4f}")
i = np.array([0, 1, 2, 3, 1, 2, 4, 0, 3, 4]); cold, pos = cold_split(i, 5, 0.4, seed=0)
print(f"cold_split of 5 films, 40%: films {np.flatnonzero(cold).tolist()} held out; their ratings at {pos.tolist()}; "
      f"every one of those ratings is a held-out film: {bool(cold[i[pos]].all())}; none missed: {int(cold[i].sum()) == len(pos)}")
