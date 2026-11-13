"""
Day 168 Exercise -- Grid Search and Random Search

The first of Week 25's parts for Day 174's reproducible tuning study.

    to_value(u, kind, lo, hi)   a number in [0, 1) -> a value on a log, whole-number or log-whole-number scale
    grid_unit(k, d, shift)      a k x k x ... grid in the unit cube, optionally shifted within its slices
    random_unit(n, d, rng)      n random points in the unit cube
    decode(U, space)            unit-cube points -> configurations
    best_so_far(scores)         the best score after each trial
    importance(U, scores)       the share of the scores' spread each hyperparameter explains alone

Checked by hand: the three scales at a few points, a 2 x 2 grid with and
without a shift, a running minimum, and an importance where one setting
decides everything.
"""
import itertools                                       # itertools: product, every combination of grid values

import numpy as np                                     # numpy: every array operation below is built on this

SPACE = (("learning rate", "log", 0.01, 1.0),           # the search space: name, scale, lowest, highest
         ("depth", "int", 1, 8),                        # tree depth: whole numbers 1 to 8
         ("min leaf", "logint", 1, 300))                # fewest rows a leaf may hold: whole numbers on a log scale


def to_value(u, kind, lo, hi):                          # a number in [0, 1) -> a hyperparameter value on its scale
    if kind == "log":                                                                 # spread evenly on a log scale
        return float(lo * (hi / lo) ** u)                                             # u = 0 gives lo, u -> 1 gives hi
    if kind == "int":                                                                 # whole numbers, each equally likely
        return int(min(hi, np.floor(lo + u * (hi - lo + 1))))                         # hi - lo + 1 equal slices of [0, 1)
    return int(min(hi, np.floor(np.exp(np.log(lo) + u * (np.log(hi + 1) - np.log(lo))))))  # logint: whole numbers, log-spaced


def grid_unit(k, d, shift=None):                        # a k x k x ... grid in the unit cube; shift moves each axis within its slices
    s = np.zeros(d) if shift is None else np.asarray(shift, float)                    # no shift: the centre of each of k slices
    axes = [(np.arange(k) + 0.5 + s[j]) / k for j in range(d)]                        # e.g. k = 3, no shift: 1/6, 1/2, 5/6
    return np.array(list(itertools.product(*axes)))                                   # every combination: k**d rows


def random_unit(n, d, rng):                             # n random points in the unit cube
    return rng.rand(n, d)                                                             # each coordinate uniform on [0, 1)


def decode(U, space=SPACE):                             # unit-cube points -> configurations (one dictionary each)
    return [{nm: to_value(u, kind, lo, hi) for u, (nm, kind, lo, hi) in zip(row, space)} for row in U]  # one value per hyperparameter


def best_so_far(scores):                                # the best (lowest) score after each trial
    return np.minimum.accumulate(np.asarray(scores, float))                           # running minimum


def importance(U, scores, bins=5):                      # how much of the scores' spread each hyperparameter explains on its own
    s = np.asarray(scores, float); tot = np.sum((s - s.mean()) ** 2)                  # the total spread
    out = []                                                                          # one share per hyperparameter
    for j in range(U.shape[1]):                                                       # each hyperparameter
        b = np.minimum((U[:, j] * bins).astype(int), bins - 1)                        # its slice of [0, 1)
        m = np.array([s[b == i].mean() if np.any(b == i) else 0.0 for i in range(bins)])  # mean score per slice
        n = np.bincount(b, minlength=bins)                                            # rows per slice
        out.append(float(np.sum(n * (m - s.mean()) ** 2) / tot))                      # between-slice spread / total spread (eta squared)
    return np.array(out)                                                              # 0 = no effect, 1 = explains everything


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    print(f"to_value log 0.01 to 1: u = 0 -> {to_value(0, 'log', 0.01, 1):.4f} (by hand 0.01), u = 0.5 -> {to_value(0.5, 'log', 0.01, 1):.4f} "
          f"(0.01 x 100^0.5 = 0.1)")
    print(f"to_value int 1 to 8: u = 0 -> {to_value(0, 'int', 1, 8)}, u = 0.5 -> {to_value(0.5, 'int', 1, 8)} (floor(1 + 0.5 x 8) = 5), "
          f"u = 0.999 -> {to_value(0.999, 'int', 1, 8)} (8)")
    print(f"to_value logint 1 to 300: u = 0.5 -> {to_value(0.5, 'logint', 1, 300)} (floor(e^(0.5 ln 301)) = floor(17.35) = 17)")
    print(f"grid_unit(2, 2): {grid_unit(2, 2).tolist()} (by hand the centres 0.25 and 0.75 in every combination)")
    print(f"grid_unit(2, 2, shift [0.4, 0]): {grid_unit(2, 2, [0.4, 0]).tolist()} (first axis (0.5 + 0.4)/2 = 0.45 and (1.5 + 0.4)/2 = 0.95)")
    print(f"random_unit(2, 3, seeded): {np.round(random_unit(2, 3, np.random.RandomState(0)), 4).tolist()} (2 rows of 3, all in [0, 1))")
    print(f"decode([[0.5, 0.5, 0.5]]): {decode(np.array([[0.5, 0.5, 0.5]]))} (by hand 0.1, 5, 17)")
    print(f"best_so_far([3, 1, 2, 0.5]): {best_so_far([3, 1, 2, 0.5]).tolist()} (by hand 3, 1, 1, 0.5)")
    U = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]]); s = np.array([0.0, 0.0, 1.0, 1.0])  # the score follows the first setting only
    print(f"importance with 2 bins, score = 1 exactly when the first setting is high: {importance(U, s, bins=2).tolist()} (by hand 1 and 0)")
