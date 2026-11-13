"""
Day 172 Exercise -- Cross-Validation for Time Series and Grouped Data

The fifth of Week 25's parts for Day 174's reproducible tuning study.

    shuffled_folds(n, k, seed)       ordinary k-fold: rows dealt into folds at random
    blocked_folds(n, k, gap)         contiguous blocks, with `gap` rows either side of the test block unused
    forward_folds(t, origins, h)     train on times up to each origin, test the next h
    group_folds(groups, k, seed)     every row of a group in the same fold
    cross_val(folds, fit_score)      the mean score over the folds, and each fold's score

Each splitter returns a list of (training rows, test rows) pairs of row
numbers. Checked by hand: every row tested exactly once, a blocked fold
with and without a gap, one forward fold, no group split across training
and test, and a leak on a toy random walk that shuffled folds hide and
forward folds show.
"""
import numpy as np                                     # numpy: every array operation below is built on this



def shuffled_folds(n, k, seed):                         # ordinary k-fold: rows dealt into k folds at random
    f = np.random.RandomState(seed).permutation(n) % k                               # each row's fold
    return [(np.flatnonzero(f != j), np.flatnonzero(f == j)) for j in range(k)]       # (training rows, test rows) per fold


def blocked_folds(n, k, gap=0):                         # k contiguous blocks; optionally drop `gap` rows either side of the test block
    b = np.arange(n) * k // n                                                         # each row's block, in time order
    out = []                                                                          # one split per block
    for j in range(k):                                                                # each block in turn is the test block
        te = np.flatnonzero(b == j); lo, hi = te[0] - gap, te[-1] + gap               # the test rows; the purged zone around them
        tr = np.flatnonzero((b != j) & ((np.arange(n) < lo) | (np.arange(n) > hi)))   # everything outside it trains
        out.append((tr, te))                                                          # one split
    return out                                                                        # k splits


def forward_folds(t, origins, horizon):                 # forward chaining: train on the past, test the next `horizon` steps
    t = np.asarray(t)                                                                 # each row's time
    return [(np.flatnonzero(t <= o), np.flatnonzero((t > o) & (t <= o + horizon))) for o in origins]  # one split per origin


def group_folds(groups, k, seed):                       # every row of a group in the same fold
    g = np.unique(groups); f = dict(zip(g, np.random.RandomState(seed).permutation(len(g)) % k))  # each group's fold
    fold = np.array([f[x] for x in groups])                                           # each row inherits its group's fold
    return [(np.flatnonzero(fold != j), np.flatnonzero(fold == j)) for j in range(k)]  # (training rows, test rows) per fold


def cross_val(folds, fit_score):                        # fit_score(train rows, test rows) -> one number; mean and per-fold scores
    s = np.array([fit_score(tr, te) for tr, te in folds])                             # one score per fold
    return s.mean(), s                                                                # the estimate and its pieces


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    F = shuffled_folds(10, 5, 0)                                                      # 10 rows, 5 folds
    print(f"shuffled_folds(10, 5): test sizes {[len(te) for tr, te in F]}, every row tested once: "
          f"{sorted(np.concatenate([te for tr, te in F]).tolist()) == list(range(10))} (by hand [2, 2, 2, 2, 2], True)")
    tr, te = blocked_folds(10, 5)[2]                                                  # the middle block
    print(f"blocked_folds(10, 5), fold 3: test {te.tolist()}, train {tr.tolist()} (by hand [4, 5] and the other 8)")
    tr, te = blocked_folds(10, 5, gap=1)[2]                                           # the same with a gap of 1
    print(f"  with gap 1: train {tr.tolist()} (by hand [0, 1, 2, 7, 8, 9]: rows 3 and 6 touch the test block)")
    tr, te = forward_folds(np.arange(10), [5], 2)[0]                                  # one origin
    print(f"forward_folds(0..9, origin 5, horizon 2): train {tr.tolist()}, test {te.tolist()} (by hand 0..5 and [6, 7])")
    g = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4])                                      # 5 groups of 1 to 3 rows
    ok = all(not set(g[tr]) & set(g[te]) for tr, te in group_folds(g, 2, 0))         # no group on both sides of any fold
    print(f"group_folds on groups {g.tolist()}, 2 folds: no group on both sides: {ok} (by hand True)")
    m, s = cross_val(shuffled_folds(10, 5, 0), lambda tr, te: len(te))               # a 'score' that is just the test size
    print(f"cross_val with score = test size: mean {m:.1f}, per fold {s.tolist()} (by hand 2.0)")
    y = np.cumsum(np.random.RandomState(0).randn(400))                                # a toy random walk: tomorrow is today plus noise
    near = lambda tr, te: np.mean(np.abs(y[te] - y[tr][np.abs(te[:, None] - tr[None, :]).argmin(1)]))  # copy the nearest training day
    a = cross_val(shuffled_folds(400, 5, 0), near)[0]                                 # neighbours in time are in training
    b = cross_val(forward_folds(np.arange(400), range(299, 399, 20), 20), near)[0]    # the next 20 days, from the past only
    print(f"random walk, 'copy the nearest training day': shuffled folds {a:.2f}, forward folds {b:.2f} "
          f"(by hand: shuffled copies a neighbour usually 1 step away, so about E|N(0,1)| = 0.80, a little more when it is 2; "
          f"forward copies a day 1 to 20 steps back, 0.80 x the mean of sqrt(1)..sqrt(20) = 2.46 on average, from only 5 origins here)")
