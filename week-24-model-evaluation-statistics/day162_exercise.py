"""
Day 162 Exercise -- Hypothesis Tests for A/B Experiments and for Comparing Models

The second of Week 24's parts for Day 167's report card.

    load_hillstrom()                       Hillstrom's email experiment, one array per column
    p_two_sided(z)                         chance of a normal value at least |z| from 0
    two_prop_test(x1, n1, x0, n0)          two rates: difference, z, p, 95% interval
    welch_test(a, b)                       two means: difference, t, p, 95% interval
    permutation_test(a, b, R, seed)        p by reshuffling the group labels
    n_per_arm(p1, p0), power_two_prop(n, p1, p0)   planning: customers needed, and power
    mcnemar(e1, e2)                        two models' mistakes on the same rows

Checked by hand: a two-proportion test, a Welch test and a sample size worked
on paper, a permutation test small enough to list every reshuffle, and
McNemar's exact binomial tail.
"""
import csv                                             # csv: the data file quotes fields that contain commas
import math                                            # math: erfc and lgamma
import os                                              # os: file paths
import urllib.request                                  # urllib: fetch the data on first run
from statistics import NormalDist                      # the standard normal's quantiles (the standard library's)

import numpy as np                                     # numpy: every array operation below is built on this

DATA = os.path.join(os.path.expanduser("~"), "week24_data")  # where the data is kept: outside the repository
HILL_URL = ("http://www.minethatdata.com/"
            "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")  # the author's own copy
ARMS = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")    # the three groups, as the file names them
ALPHA = 0.05                                            # the significance level used all day
Z975 = NormalDist().inv_cdf(0.975)                      # 1.96: the normal quantile for 95% intervals
R_PERM = 10000                                          # shuffles per permutation test


def load_hillstrom():                                   # the email experiment: one array per column
    os.makedirs(DATA, exist_ok=True)                                                  # the data folder, if it is missing
    path = os.path.join(DATA, "hillstrom.csv")                                        # where the file is kept
    if not os.path.exists(path):                                                      # first run: fetch it
        urllib.request.urlretrieve(HILL_URL, path)                                    # 4.4 MB from minethatdata.com
    rows = list(csv.reader(open(path, encoding="utf-8", newline="")))                 # every line, split at commas outside quotes
    d = {c: np.array([r[k] for r in rows[1:]]) for k, c in enumerate(rows[0])}       # the first line names the columns
    for c in ("recency", "history", "mens", "womens", "newbie", "visit", "conversion", "spend"):  # the numeric columns
        d[c] = d[c].astype(float)                                                     # text -> numbers
    return d                                                                          # 64,000 customers


def p_two_sided(z):                                     # chance of a normal value at least this far from 0, either side
    return math.erfc(abs(z) / math.sqrt(2))                                           # erfc stays accurate far into the tail


def two_prop_test(x1, n1, x0, n0):                      # difference of two rates: (diff, z, p, low, high)
    p1, p0 = x1 / n1, x0 / n0                                                         # the two observed rates
    pp = (x1 + x0) / (n1 + n0)                                                        # the pooled rate: the rate if there is no difference
    se0 = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))                                # spread of the difference if there is none
    z = (p1 - p0) / se0 if se0 > 0 else 0.0                                           # how many spreads away from 0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)                           # spread around the observed difference
    return p1 - p0, z, p_two_sided(z), p1 - p0 - Z975 * se, p1 - p0 + Z975 * se       # with a 95% interval


def welch_test(a, b):                                   # difference of two means, variances not assumed equal
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))                   # spread of the difference
    t = (a.mean() - b.mean()) / se                                                    # how many spreads away from 0
    return a.mean() - b.mean(), t, p_two_sided(t), a.mean() - b.mean() - Z975 * se, a.mean() - b.mean() + Z975 * se  # normal tails: groups are large


def permutation_test(a, b, R=R_PERM, seed=0):           # p by reshuffling the group labels: (diff, p)
    x = np.r_[a, b]; obs = a.mean() - b.mean()                                        # everyone together; the real difference
    rng = np.random.RandomState(seed); hits = 0                                       # a seeded shuffler; a counter
    for _ in range(R):                                                                # each shuffle
        p = rng.permutation(len(x))                                                   # labels handed out again at random
        hits += abs(x[p[:len(a)]].mean() - x[p[len(a):]].mean()) >= abs(obs) - 1e-12  # as extreme as the real one?
    return obs, (hits + 1) / (R + 1)                                                  # +1: the real labelling is one of the orders


def n_per_arm(p1, p0, alpha=ALPHA, power=0.8):          # customers per group to detect p1 vs p0
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)        # 1.96 and 0.84
    pb = (p1 + p0) / 2                                                                # the average rate
    return (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p0 * (1 - p0))) ** 2 / (p1 - p0) ** 2  # the classic formula


def power_two_prop(n, p1, p0, alpha=ALPHA):             # chance a test with n per group finds a true p1 vs p0
    za = NormalDist().inv_cdf(1 - alpha / 2); pb = (p1 + p0) / 2                      # the cut-off; the average rate
    se0 = math.sqrt(2 * pb * (1 - pb) / n); se1 = math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / n)  # spread under no effect, and under the effect
    return NormalDist().cdf((abs(p1 - p0) - za * se0) / se1)                          # chance the difference clears the cut-off


def mcnemar(e1, e2):                                    # two models' errors on the same rows: (b, c, chi2, p, exact p)
    b = int(np.sum(e1 & ~e2)); c = int(np.sum(~e1 & e2))                              # rows only model 1 gets wrong; only model 2
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0                          # with the continuity correction
    n, k = b + c, min(b, c)                                                           # disagreements; the smaller count
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]  # log P(X = i), X ~ Binomial(n, 1/2)
    m = max(lp) if lp else 0.0                                                        # largest term, for a safe sum
    exact = min(1.0, 2 * math.exp(m) * sum(math.exp(v - m) for v in lp)) if n else 1.0  # both tails
    return b, c, chi2, math.erfc(math.sqrt(chi2 / 2)), exact                          # chi-square with 1 degree of freedom = z squared


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    diff, z, p, lo, hi = two_prop_test(30, 100, 20, 100)                              # 30% against 20%, 100 each
    print("two_prop_test, 30/100 vs 20/100: by hand pooled 0.25, spread sqrt(0.25 x 0.75 x 2/100) = 0.0612, z = 0.1/0.0612 = 1.633, "
          "p = 0.1025; interval 0.1 +/- 1.96 x sqrt(0.21/100 + 0.16/100) = [-0.0192, 0.2192]")
    print(f"  got z {z:.3f}, p {p:.4f}, interval [{lo:.4f}, {hi:.4f}]")
    diff, t, p, lo, hi = welch_test(np.array([1.0, 2, 3, 4]), np.array([2.0, 4, 6, 8]))  # means 2.5 and 5
    print("welch_test, (1, 2, 3, 4) vs (2, 4, 6, 8): by hand variances 1.667 and 6.667, spread sqrt(1.667/4 + 6.667/4) = 1.443, "
          "t = -2.5/1.443 = -1.732, normal p 0.0833 (with 4 rows a t table would give a larger p; the lesson's groups have thousands)")
    print(f"  got t {t:.3f}, p {p:.4f}")
    obs, p = permutation_test(np.array([1.0, 2, 3]), np.array([4.0, 5, 6]))            # small enough to list every split
    print("permutation_test, (1, 2, 3) vs (4, 5, 6): by hand 20 ways to choose 3 of 6, and only 2 are as extreme "
          "((1, 2, 3) and (4, 5, 6)), so p = 2/20 = 0.1")
    print(f"  got difference {obs:.1f}, p {p:.4f} from {R_PERM} shuffles (their own spread: sqrt(0.1 x 0.9 / {R_PERM}) = "
          f"{math.sqrt(0.1 * 0.9 / R_PERM):.4f})")
    n = n_per_arm(0.2, 0.1)                                                           # 20% against 10%
    print("n_per_arm, 0.2 vs 0.1: by hand (1.960 x sqrt(2 x 0.15 x 0.85) + 0.842 x sqrt(0.16 + 0.09))^2 / 0.1^2 "
          "= (0.9897 + 0.4208)^2 / 0.01 = 199.0")
    print(f"  got {n:.1f}; power_two_prop at 199 per group: {power_two_prop(199, 0.2, 0.1):.4f} (by hand 0.80)")
    e1 = np.r_[np.ones(10, bool), np.zeros(2, bool), np.ones(5, bool), np.zeros(30, bool)]  # model 1: wrong on 10 + 5 rows
    e2 = np.r_[np.zeros(10, bool), np.ones(2, bool), np.ones(5, bool), np.zeros(30, bool)]  # model 2: wrong on 2 + 5 rows
    b, c, chi2, p, ex = mcnemar(e1, e2)                                               # only the 12 disagreements count
    print("mcnemar, b = 10 rows only model 1 got wrong, c = 2 only model 2: by hand chi2 (|10 - 2| - 1)^2 / 12 = 4.083, p 0.0433; "
          "exact 2 x (1 + 12 + 66) / 2^12 = 0.0386")
    print(f"  got b {b}, c {c}, chi2 {chi2:.3f}, p {p:.4f}, exact {ex:.4f}")
    d = load_hillstrom()                                                              # the real data
    print(f"load_hillstrom: {len(d['visit'])} customers; " + ", ".join(f"{a} {int(np.sum(d['segment'] == a))}" for a in ARMS))
