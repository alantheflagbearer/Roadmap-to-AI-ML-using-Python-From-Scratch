"""
Day 158 Exercise -- Implicit Feedback and Cold Start

The fifth of Week 23's parts for Day 160's neural recommender.

    load_users(folder)                      age, gender (1 = female) and occupation of every user
    implicit_als(C, k, lam)                 Hu-Koren-Volinsky ALS: confidence where watched, a weak 0 elsewhere
    implicit_objective(X, Y, C, lam)        the weighted squared error over every cell, plus the penalty
    fold_in(Y, items, conf, lam)            a new user's factors from the films they watched, Y fixed
    cooccurrence(B)                         cosine between films' watcher sets
    topk_metrics(S, seen, held, k)          precision@k and recall@k against held-out films

Checked by hand: the objective of a 1 x 2 table, a cosine on three users, a
top-2 list, the shortcut user step against the full weighted ridge, and the
objective falling on every sweep.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep Day 154's checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path

import numpy as np                                     # numpy: every array operation below is built on this

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so Day 154's parts can be imported from here
with contextlib.redirect_stdout(io.StringIO()):         # Day 154's exercise prints its checks when imported
    import day154_exercise as d154                      # for DATA, the folder load_users reads from


def load_users(folder=d154.DATA):                       # NEW: Day 158's version, unchanged
    rows = [x.split("|") for x in open(os.path.join(folder, "ml-100k", "u.user"), encoding="latin-1").read().strip().split("\n")]  # one list of fields per line of u.user
    occ = sorted({x[3] for x in rows})                                                # the occupations, in alphabetical order
    return (np.array([int(x[1]) for x in rows]), np.array([x[2] == "F" for x in rows]).astype(int),  # ages and genders
            np.array([occ.index(x[3]) for x in rows]))                                # occupation numbers


def implicit_als(C, k, lam, sweeps=10, seed=0, track=None):  # NEW: Day 158's version, unchanged
    nu, ni = C.shape; rng = np.random.RandomState(seed)                               # seeded, so a run repeats
    X, Y = 0.01 * rng.randn(nu, k), 0.01 * rng.randn(ni, k)                           # user and film factors
    hist = []                                                                         # one record per sweep, if asked for
    for _ in range(sweeps):                                                           # each sweep: users, then films
        for A, B, Cm in ((X, Y, C), (Y, X, C.T)):                                     # solve the rows of A with B fixed
            G = B.T @ B + lam * np.eye(k)                                             # every empty cell's weight-1 term, shared by all rows
            for a in range(A.shape[0]):                                               # one row (user or film) at a time
                j = np.flatnonzero(Cm[a]); c = Cm[a, j]                               # where this row has interactions, and their confidence
                Bj = B[j]                                                             # the other side's factors there
                A[a] = np.linalg.solve(G + (Bj * (c - 1)[:, None]).T @ Bj, Bj.T @ c)  # weighted ridge: extra weight c - 1 where watched
        hist.append(track(X, Y) if track else ())                                     # measures after this sweep
    return X, Y, hist                                                                 # the factors, and the records


def implicit_objective(X, Y, C, lam):                   # NEW: Day 158's version, unchanged
    P = C > 0                                                                         # 1 where watched, 0 elsewhere
    W = np.where(P, C, 1.0)                                                           # confidence: c where watched, 1 elsewhere
    return (W * (P - X @ Y.T) ** 2).sum() + lam * ((X ** 2).sum() + (Y ** 2).sum())   # weighted squared error + penalty


def fold_in(Y, items, conf, lam):                       # NEW: Day 158's version, unchanged
    k = Y.shape[1]; Bj = Y[items]                                                     # the watched films' factors
    G = Y.T @ Y + lam * np.eye(k)                                                     # the same shared term as in implicit_als
    return np.linalg.solve(G + (Bj * (conf - 1)[:, None]).T @ Bj, Bj.T @ conf)        # one user step: no retraining


def cooccurrence(B):                                    # NEW: Day 158's version, unchanged
    n = np.sqrt(B.sum(0))                                                             # each film's length: sqrt(number of watchers)
    S = (B.T @ B) / np.maximum(np.outer(n, n), 1e-12)                                 # shared watchers / sqrt(n_a n_b)
    np.fill_diagonal(S, 0.0)                                                          # a film is not its own neighbour
    return S                                                                          # films x films


def topk_metrics(S, seen, held, k=10):                  # NEW: Day 158's version, unchanged
    S = np.where(seen, -np.inf, S); who = np.flatnonzero(held.sum(1) > 0)             # already-watched films are never shown
    top = np.argsort(-S, axis=1, kind="stable")[:, :k]                                # each user's k best-scored films
    h = np.take_along_axis(held, top, 1)[who].sum(1)                                  # hits per user
    return h.mean() / k, (h / held[who].sum(1)).mean(), top[who]                      # precision, recall, the lists


if __name__ == "__main__":                              # the checks run only when this file is run, not when Day 160 imports it
    age, fem, occ = load_users()                                                      # the real users' table
    print(f"load_users: {len(age)} users, ages {age.min()}-{age.max()}, {fem.sum()} women, {occ.max() + 1} occupations")
    X = np.array([[1.0]]); Y = np.array([[1.0], [0.5]]); C = np.array([[3.0, 0.0]])  # one user, two films, the first watched
    print("objective, one user and two films, x = 1, y = (1, 0.5), confidence 3 on film 0, penalty 0.1:")
    print(f"  watched cell 3 x (1 - 1)^2 = 0; empty cell 1 x (0 - 0.5)^2 = 0.25; penalty 0.1 x (1 + 1 + 0.25) = 0.225; "
          f"total 0.475; implicit_objective gives {implicit_objective(X, Y, C, 0.1):.4f}")
    B = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])                  # film 0: users 0, 1; film 1: user 0; film 2: user 2
    S = cooccurrence(B)                                                               # films x films
    print(f"cosine: films 0 and 1 share 1 watcher of 2 and 1, so 1 / sqrt(2 x 1) = 0.7071; films 0 and 2 share none: "
          f"cooccurrence gives {S[0, 1]:.4f} and {S[0, 2]:.4f}")
    p, r, top = topk_metrics(np.array([[9.0, 5.0, 4.0, 1.0]]), np.array([[True, False, False, False]]),
                             np.array([[False, True, False, True]]), k=2)             # film 0 already seen; films 1 and 3 held out
    print(f"top-2 of scores 9, 5, 4, 1 with film 0 seen: films 1 and 2; held out 1 and 3, so 1 hit: precision 0.5, recall 0.5; "
          f"topk_metrics gives {p:.2f}, {r:.2f}, list {top[0].tolist()}")
    rng = np.random.RandomState(0); Y = rng.randn(5, 2); c = np.array([0.0, 4.0, 0.0, 2.0, 0.0])  # 5 films; the user watched 1 and 3
    W = np.where(c > 0, c, 1.0)                                                       # weight c where watched, 1 elsewhere
    full = np.linalg.solve(Y.T @ (W[:, None] * Y) + 0.5 * np.eye(2), Y.T @ (W * (c > 0)))  # the weighted ridge over all 5 films
    print(f"fold_in against the full weighted ridge over every film: largest difference "
          f"{np.abs(full - fold_in(Y, np.array([1, 3]), np.array([4.0, 2.0]), 0.5)).max():.1e}")
    wrong = np.linalg.solve(Y.T @ Y + 0.5 * np.eye(2), Y[[1, 3]].T @ np.array([4.0, 2.0]))  # a mistake: the c - 1 term left out
    print(f"  the same check can fail: leaving out the extra weight c - 1 misses by {np.abs(full - wrong).max():.3f}")
    C = np.where(rng.rand(12, 8) < 0.4, 4.0, 0.0)                                     # 12 users x 8 films, confidence 4 where watched
    _, _, h = implicit_als(C, 2, 0.5, sweeps=6, track=lambda X, Y: implicit_objective(X, Y, C, 0.5))  # the objective after each sweep
    print(f"implicit_als on 12 x 8: objective after each sweep {', '.join(f'{x:.3f}' for x in h)}; "
          f"never rises: {all(h[q + 1] <= h[q] + 1e-9 for q in range(len(h) - 1))}")
