"""
Day 165 Exercise -- Calibration

The fifth of Week 24's parts for Day 167's report card.

    reliability(y, p), ece(y, p)        the reliability table and the expected calibration error
    log_loss(y, p)                      mean negative log-likelihood
    platt_fit(p, y), platt_apply(w, p)  Platt scaling, by Newton's method with step halving
    isotonic_fit(p, y), isotonic_apply  isotonic regression by pool-adjacent-violators
    cross_fit(p, y, half, fit, apply)   fit on one half, apply to the other, and swap

Checked by hand: a reliability table and ECE for four rows, a log loss, a
pool-adjacent-violators run on four rows, Platt scaling recovering a slope and
intercept it was built from, and the small case where plain Newton runs away
but step halving lands on the answer worked on paper.
"""
import numpy as np                                     # numpy: every array operation below is built on this

BINS = 10                                               # equal-width probability bins for the reliability diagram and ECE
EPS = 1e-6                                              # probabilities are kept this far from 0 and 1 before a logarithm


def reliability(y, p, bins=BINS):                       # per bin: (rows, mean predicted probability, share that happened)
    b = np.minimum((p * bins).astype(int), bins - 1)                                  # bin of each row; p = 1 goes in the top bin
    n = np.bincount(b, minlength=bins)                                                # rows per bin
    mp = np.bincount(b, p, bins) / np.maximum(n, 1)                                   # mean predicted probability per bin
    my = np.bincount(b, y, bins) / np.maximum(n, 1)                                   # share of positives per bin
    return n, np.where(n > 0, mp, np.nan), np.where(n > 0, my, np.nan)                # empty bins are NaN


def ece(y, p, bins=BINS):                               # expected calibration error: the rows-weighted gap between the two
    n, mp, my = reliability(y, p, bins); ok = n > 0                                   # the reliability table; non-empty bins
    return float(np.sum(n[ok] * np.abs(mp[ok] - my[ok])) / len(y))                   # 0 = predicted and observed agree in every bin


def log_loss(y, p):                                     # mean negative log-likelihood of the 0/1 outcomes
    p = np.clip(p, EPS, 1 - EPS)                                                      # log(0) would be minus infinity
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))                   # lower is better


def logit(p):                                           # probability to log-odds
    p = np.clip(p, EPS, 1 - EPS)                                                      # so 0 and 1 stay finite (about -13.8 and +13.8)
    return np.log(p / (1 - p))                                                        # log(p / (1 - p))


def platt_loss(A, y, w):                                # Platt's log loss for weights w, written so it never overflows
    z = A @ w                                                                         # the new log-odds
    return float(np.mean(np.logaddexp(0, z) - y * z))                                 # log(1 + e^z) - y z, computed safely


def platt_fit(p, y, iters=100, damped=True):            # Platt scaling: a logistic regression of y on logit(p); returns (a, b)
    A = np.c_[logit(p), np.ones(len(p))]; w = np.array([1.0, 0.0])                   # one feature and an intercept; start at 'no change'
    for _ in range(iters):                                                            # Newton steps
        q = 1 / (1 + np.exp(-np.clip(A @ w, -500, 500)))                              # current probabilities (clipped to avoid overflow)
        g = A.T @ (q - y)                                                             # gradient of the summed log loss
        H = (A * (q * (1 - q))[:, None]).T @ A + 1e-9 * np.eye(2)                     # its Hessian, with a tiny ridge
        step = np.linalg.solve(H, g); t = 1.0                                         # the Newton step, at full length
        if damped:                                                                    # step halving: never accept a step that raises the loss
            while platt_loss(A, y, w - t * step) > platt_loss(A, y, w) and t > 1e-10:  # too far?
                t /= 2                                                                # halve it
        w = w - t * step                                                              # take the (possibly shortened) step
        if np.abs(t * step).max() < 1e-10:                                            # converged
            break                                                                     # stop early
    return w                                                                          # slope a and intercept b


def platt_apply(w, p):                                  # calibrated probability: sigmoid(a logit(p) + b)
    return 1 / (1 + np.exp(-np.clip(w[0] * logit(p) + w[1], -500, 500)))             # clipped, so exp never overflows


def isotonic_fit(p, y):                                 # isotonic regression by pool-adjacent-violators; returns (right edges, values)
    xs, inv = np.unique(p, return_inverse=True)                                       # distinct scores, sorted; each row's position
    wt = np.bincount(inv).astype(float); v = np.bincount(inv, y) / wt                 # rows and share of positives at each score
    blocks = []                                                                       # [value, weight, index of its last score]
    for i in range(len(xs)):                                                          # each score, smallest first
        blocks.append([v[i], wt[i], i])                                               # a new block of one score
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:                     # the order is violated (or tied)
            v2, w2, i2 = blocks.pop(); v1, w1, _ = blocks.pop()                       # take the last two blocks
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, i2])             # and pool them into their weighted mean
    return np.array([xs[b[2]] for b in blocks]), np.array([b[0] for b in blocks])     # each block's largest score, and its value


def isotonic_apply(m, p):                               # a step function: each score takes the value of the first block that reaches it
    right, val = m                                                                    # block edges and values
    return val[np.minimum(np.searchsorted(right, p, side="left"), len(val) - 1)]      # scores above every edge take the top value


def cross_fit(p, y, half, fit, apply):                  # fit on one half, apply to the other, then swap: every row calibrated once
    out = np.zeros(len(p))                                                            # calibrated probabilities
    for A, B in ((half, ~half), (~half, half)):                                       # both directions
        out[B] = apply(fit(p[A], y[A]), p[B])                                         # rows in B never helped fit their own calibrator
    return out                                                                        # one calibrated probability per row


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    y = np.array([0, 1, 1, 0]); p = np.array([0.05, 0.15, 0.95, 0.92])               # four rows
    n, mp, my = reliability(y, p)                                                     # their reliability table
    print("reliability of four rows: bins 0, 1 and 9 (two rows: mean 0.935, one of two happened)")
    print(f"  got rows {n.tolist()}, bin 9 mean {mp[9]:.3f} share {my[9]:.2f}")
    print("ece: by hand (|0.05 - 0| + |0.15 - 1| + 2 x |0.935 - 0.5|) / 4 = (0.05 + 0.85 + 0.87) / 4 = 0.4425; "
          f"got {ece(y, p):.4f}")
    print(f"log_loss of y (1, 0) at p (0.8, 0.3): by hand (-ln 0.8 - ln 0.7) / 2 = (0.2231 + 0.3567) / 2 = 0.2899; "
          f"got {log_loss(np.array([1, 0]), np.array([0.8, 0.3])):.4f}")
    m = isotonic_fit(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 1]))         # 0, 1, 0, 1 in score order
    print("isotonic_fit of y (0, 1, 0, 1) at scores 0.1 to 0.4: 1 then 0 breaks the order, so they pool to 0.5;")
    print(f"  by hand edges (0.1, 0.3, 0.4) values (0, 0.5, 1); got edges {m[0].tolist()} values {m[1].tolist()}")
    print(f"  applied to 0.05, 0.25, 0.35, 0.5: {isotonic_apply(m, np.array([0.05, 0.25, 0.35, 0.5])).tolist()}")
    rng = np.random.RandomState(0); p = rng.uniform(0.02, 0.98, 20000)                # scores spread over (0, 1)
    y = (rng.rand(20000) < 1 / (1 + np.exp(-(2 * logit(p) - 1)))).astype(float)      # outcomes built with a = 2, b = -1
    print(f"platt_fit on 20,000 rows built with slope 2 and intercept -1: got {np.round(platt_fit(p, y), 3).tolist()}")
    p = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]); y = np.array([0, 0, 1, 0, 1, 1])   # scores of only 0 and 1, one mistake each side
    print("platt_fit when every score is 0 or 1 (logit about -13.82 or +13.82), one of three wrong on each side:")
    print("  by hand the answer gives 1/3 and 2/3: intercept 0, slope ln 2 / 13.8155 = 0.0502")
    print(f"  plain Newton: {np.round(platt_fit(p, y, damped=False), 4).tolist()}; "
          f"with step halving: {np.round(platt_fit(p, y), 4).tolist()}")
    half = np.array([True, False] * 3)                                                # rows 0, 2, 4 and rows 1, 3, 5
    print("cross_fit with a mean-only calibrator: rows 0, 2, 4 have outcomes 0, 1, 1 (rate 2/3) and rows 1, 3, 5 have 0, 0, 1")
    print(f"  (rate 1/3); each row gets the other half's rate, so by hand 0.333, 0.667, 0.333, 0.667, 0.333, 0.667; got "
          f"{cross_fit(p, y, half, lambda a, b: b.mean(), lambda w, a: np.full(len(a), w)).round(3).tolist()}")
