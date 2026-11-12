"""
Day 163 Exercise -- Confidence Intervals and the Bootstrap

The third of Week 24's parts for Day 167's report card.

    wald_ci(x, n), wilson_ci(x, n)      intervals for a rate (x successes out of n)
    bootstrap_reps(stat, arrays, B)     a statistic recomputed on B resamples of the rows
    percentile_ci(reps)                 the middle 95% of those values
    jackknife(stat, arrays)             the statistic with each row left out in turn
    bca_ci(theta, reps, jack)           the bias-corrected and accelerated interval

Checked by hand: a Wald and a Wilson interval worked on paper (including
zero successes), a bootstrap of three numbers whose every resample can be
listed, a jackknife of four numbers, and BCa on data with no skew.
"""
from statistics import NormalDist                      # the standard normal's quantiles (the standard library's)

import numpy as np                                     # numpy: every array operation below is built on this

ALPHA = 0.05                                            # 95% intervals all day
B_BOOT = 2000                                           # bootstrap resamples, unless stated


def wald_ci(x, n, alpha=ALPHA):                         # the textbook interval for a rate: x successes out of n
    z = NormalDist().inv_cdf(1 - alpha / 2); p = x / n                               # the cut-off; the observed rate
    se = np.sqrt(p * (1 - p) / n)                                                     # its estimated spread
    return p - z * se, p + z * se                                                     # works on arrays of x too


def wilson_ci(x, n, alpha=ALPHA):                       # Wilson's interval: inverts the test instead of the estimate
    z = NormalDist().inv_cdf(1 - alpha / 2); p = x / n                               # the cut-off; the observed rate
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)                                  # pulled towards 1/2
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)      # never zero, even when x = 0
    return centre - half, centre + half                                               # works on arrays of x too


def bootstrap_reps(stat, arrays, B=B_BOOT, seed=0):     # stat recomputed on B resamples of the rows (all arrays share the rows)
    rng = np.random.RandomState(seed); n = len(arrays[0])                            # a seeded generator; the number of rows
    out = []                                                                          # one value per resample
    for _ in range(B):                                                                # each resample
        idx = rng.randint(0, n, n)                                                    # n rows drawn with replacement
        out.append(stat(*[a[idx] for a in arrays]))                                   # the statistic on them
    return np.array(out)                                                              # B values


def percentile_ci(reps, alpha=ALPHA):                   # the middle 95% of the bootstrap values
    return float(np.quantile(reps, alpha / 2)), float(np.quantile(reps, 1 - alpha / 2))  # 2.5th and 97.5th percentiles


def jackknife(stat, arrays):                            # the statistic with each row left out in turn
    n = len(arrays[0])                                                                # the number of rows
    return np.array([stat(*[np.delete(a, i) for a in arrays]) for i in range(n)])     # n values


def bca_ci(theta, reps, jack, alpha=ALPHA):             # bias-corrected and accelerated percentile interval
    nd = NormalDist()                                                                 # the standard normal
    if np.all(reps == reps[0]):                                                       # every resample gave the same value
        return float(theta), float(theta)                                             # no spread to correct: a single point
    z0 = nd.inv_cdf(min(max(np.mean(reps < theta), 1e-6), 1 - 1e-6))                  # bias: how off-centre the bootstrap is
    d = jack.mean() - jack; den = 6 * np.sum(d ** 2) ** 1.5                          # jackknife differences
    acc = np.sum(d ** 3) / den if den > 0 else 0.0                                    # acceleration: how the spread changes with the value
    out = []                                                                          # the two ends
    for q in (alpha / 2, 1 - alpha / 2):                                              # lower, then upper
        zq = nd.inv_cdf(q)                                                            # the plain percentile's z
        out.append(float(np.quantile(reps, nd.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq))))))  # moved percentile
    return tuple(out)                                                                 # (low, high)


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    lo, hi = wald_ci(5, 100)                                                          # 5 successes in 100
    print("wald_ci, 5 of 100: by hand 0.05 +/- 1.96 x sqrt(0.05 x 0.95 / 100) = 0.05 +/- 0.0427 = [0.0073, 0.0927]")
    print(f"  got [{lo:.4f}, {hi:.4f}]; and for 0 of 50: {[round(float(v), 4) for v in wald_ci(0, 50)]} (a zero-width interval)")
    lo, hi = wilson_ci(0, 50)                                                         # no successes at all
    print("wilson_ci, 0 of 50: by hand centre (0 + 3.8415/100) / (1 + 3.8415/50) = 0.0357, half-width "
          "1.96 x sqrt(3.8415 / 10000) / 1.0768 = 0.0357, so [0.0000, 0.0713]")
    print(f"  got [{abs(lo):.4f}, {hi:.4f}]")
    x = np.array([1.0, 2.0, 3.0]); reps = bootstrap_reps(np.mean, (x,))               # every resample of 3 numbers is one of 27
    print("bootstrap_reps of the mean of (1, 2, 3): by hand 27 equally likely resamples, mean 2, spread sqrt((2/3)/3) = 0.4714; "
          "a mean of 1 has chance 1/27 = 0.037, above 0.025, so the 95% percentile interval is [1, 3]")
    print(f"  got mean {reps.mean():.3f}, spread {reps.std():.3f}, interval {percentile_ci(reps)}")
    jk = jackknife(np.mean, (np.array([1.0, 2.0, 3.0, 10.0]),))                       # leave each of four out
    print(f"jackknife of the mean of (1, 2, 3, 10): by hand 15/3, 14/3, 13/3, 6/3 = 5.0, 4.667, 4.333, 2.0; got {np.round(jk, 3).tolist()}")
    x = np.arange(1.0, 10.0); reps = bootstrap_reps(np.mean, (x,), 4000)              # 1 to 9: no skew
    print("bca_ci on 1..9 (symmetric, so the acceleration is 0 and BCa should sit close to the percentile interval):")
    print(f"  percentile {np.round(percentile_ci(reps), 3).tolist()}, BCa {np.round(bca_ci(x.mean(), reps, jackknife(np.mean, (x,))), 3).tolist()}")
    print(f"  BCa sits one step lower: {np.mean(reps == 5):.3f} of resample means land exactly on 5 and only the {np.mean(reps < 5):.3f} "
          "strictly below count, so its bias correction z0 is below 0")
    z = np.zeros(20)                                                                  # twenty zeros: no spread at all
    print(f"bca_ci when every value is 0: {bca_ci(0.0, bootstrap_reps(np.mean, (z,), 100), jackknife(np.mean, (z,)))} (a single point)")
