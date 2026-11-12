"""
Day 164 Exercise -- Multiple Comparisons

The fourth of Week 24's parts for Day 167's report card.

    segments(d)                         22 customer segments of Hillstrom's experiment
    segment_tests(d, A, B, S)           every segment x outcome, group A against group B: 66 p values
    bonferroni(p), holm(p)              reject while keeping the chance of any false discovery at 5%
    benjamini_hochberg(p)               reject while keeping the false share of discoveries at 5%
    METHODS                             the three rules and 'no correction', by name

Checked by hand: the three rules on five and three p values worked on paper,
including one case where Holm finds more than Bonferroni and one where
Benjamini-Hochberg's step-up accepts a p value that sits above its own line;
the segments of each kind cover every customer exactly once.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path

import numpy as np                                     # numpy: every array operation below is built on this

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Day 162 lives in this same folder
with contextlib.redirect_stdout(io.StringIO()):         # its exercise prints hand checks when imported
    import day162_exercise as d162                      # Day 162: Hillstrom's experiment and its tests

ALPHA = 0.05                                            # the error rate each method promises
OUTCOMES = ("visit", "conversion", "spend")             # the three outcomes tested in every segment
EMPTY = [0]                                             # spend tests with no buyer in either group (a list, so functions can add to it)


def segments(d):                                        # 22 customer segments, each a 0/1 mask over all 64,000 customers
    S = {}                                                                            # name -> mask
    for c in ("channel", "zip_code"):                                                 # two category columns
        for v in np.unique(d[c]):                                                     # each category
            S[f"{c} {v}"] = d[c] == v                                                 # its customers
    S["new customer"], S["returning"] = d["newbie"] == 1, d["newbie"] == 0            # new this year or not
    S["men's buyer only"] = (d["mens"] == 1) & (d["womens"] == 0)                     # bought men's products only
    S["women's buyer only"] = (d["mens"] == 0) & (d["womens"] == 1)                   # bought women's products only
    S["bought both"] = (d["mens"] == 1) & (d["womens"] == 1)                          # bought both
    for v in np.unique(d["history_segment"]):                                         # seven bands of last year's spend
        S[f"history {v[3:]}"] = d["history_segment"] == v                             # "$0 - $100" and so on
    for lo, hi in ((1, 3), (4, 6), (7, 9), (10, 12)):                                 # months since the last purchase
        S[f"recency {lo}-{hi}"] = (d["recency"] >= lo) & (d["recency"] <= hi)         # four bands
    return S                                                                          # 22 segments


def segment_tests(d, A, B, S):                          # every segment x outcome: group A against group B; (p values, names)
    ps, names = [], []                                                                # results
    for sn, sm in S.items():                                                          # each segment
        a, b = A & sm, B & sm                                                         # the segment's customers in each group
        for col in OUTCOMES:                                                          # each outcome
            if col == "spend" and d[col][a].max() == 0 and d[col][b].max() == 0:      # nobody bought in either group: Welch would divide 0 by 0
                ps.append(1.0); EMPTY[0] += 1                                         # no evidence of a difference; counted, to be reported
            elif col == "spend":                                                      # an amount: Welch's test
                ps.append(d162.welch_test(d[col][a], d[col][b])[2])                   # item [2] of the result is the two-sided p value
            else:                                                                     # yes/no: the two-proportion test
                ps.append(d162.two_prop_test(d[col][a].sum(), a.sum(), d[col][b].sum(), b.sum())[2])  # successes and sizes in each group; [2] is the p value
            names.append(f"{sn}, {col}")                                              # e.g. "recency 4-6, spend"
    return np.array(ps), names                                                        # 66 p values


def bonferroni(p, alpha=ALPHA):                         # reject where p <= alpha / m
    return p <= alpha / len(p)                                                        # one strict threshold for every test


def holm(p, alpha=ALPHA):                               # step down: the k-th smallest p against alpha / (m - k)
    order = np.argsort(p); m = len(p); rej = np.zeros(m, bool)                        # smallest first; nothing rejected yet
    for k, i in enumerate(order):                                                     # k = 0 is the smallest p
        if p[i] > alpha / (m - k):                                                    # the first one that fails
            break                                                                     # stops everything after it
        rej[i] = True                                                                 # otherwise rejected
    return rej                                                                        # True = significant


def benjamini_hochberg(p, alpha=ALPHA):                 # step up: the largest k with p_(k) <= k alpha / m, and all below it
    order = np.argsort(p); m = len(p)                                                 # smallest first
    ok = p[order] <= alpha * np.arange(1, m + 1) / m                                  # each sorted p against its rising line
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0                               # the last sorted p under the line
    rej = np.zeros(m, bool); rej[order[:k]] = True                                    # it and every smaller one
    return rej                                                                        # True = significant (controls the false discovery rate)


METHODS = {"no correction": lambda p: p < ALPHA, "Bonferroni": bonferroni, "Holm": holm,  # each rule as a function from p values to True/False
           "Benjamini-Hochberg": benjamini_hochberg}    # name -> rejections


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    p = np.array([0.3, 0.025, 0.005, 0.035, 0.028])                                   # five p values, deliberately unsorted
    print("five p values (0.3, 0.025, 0.005, 0.035, 0.028), m = 5:")
    print("  Bonferroni: only p <= 0.05/5 = 0.01, so only 0.005; got", bonferroni(p).astype(int).tolist())
    print("  Holm: 0.005 <= 0.05/5 = 0.01 yes; 0.025 <= 0.05/4 = 0.0125 no, stop; got", holm(p).astype(int).tolist())
    print("  Benjamini-Hochberg: sorted 0.005, 0.025, 0.028, 0.035, 0.3 against 0.01, 0.02, 0.03, 0.04, 0.05;")
    print("    0.025 > 0.02 fails its own line, but 0.035 <= 0.04 passes, so the four smallest are all rejected; got",
          benjamini_hochberg(p).astype(int).tolist())
    p = np.array([0.01, 0.02, 0.3])                                                   # three p values
    print("three p values (0.01, 0.02, 0.3), m = 3:")
    print("  Bonferroni: p <= 0.0167 keeps 0.01 only; got", bonferroni(p).astype(int).tolist())
    print("  Holm: 0.01 <= 0.0167 yes; 0.02 <= 0.05/2 = 0.025 yes; 0.3 <= 0.05 no; got", holm(p).astype(int).tolist())
    d = d162.load_hillstrom(); S = segments(d)                                        # the real data and its segments
    print(f"segments: {len(S)}; 22 x 3 outcomes = {len(S) * 3} tests")
    for kind in ("channel", "zip_code", "history", "recency"):                        # each kind of segment
        tot = sum(int(v.sum()) for k, v in S.items() if k.startswith(kind))           # customers over its segments
        print(f"  {kind} segments cover {tot} of {len(d['spend'])} customers")
    tot = sum(int(S[k].sum()) for k in ("men's buyer only", "women's buyer only", "bought both"))  # the product segments
    print(f"  men's only + women's only + both cover {tot} of {len(d['spend'])} customers")
    c0 = d["segment"] == "No E-Mail"; u = np.random.RandomState(0).rand(len(c0)) < 0.5  # Section 1's A/A split
    ps, names = segment_tests(d, c0 & u, c0 & ~u, S)                                  # its 66 tests
    print(f"segment_tests on the lesson's first A/A split: {len(ps)} p values, smallest {ps.min():.4f} ({names[int(np.argmin(ps))]})")
