"""
Day 166 Exercise -- Fairness Metrics

The sixth of Week 24's parts for Day 167's report card.

    group_rates(y, d, groups)           per group: size, base rate, selection rate, TPR, FPR, precision
    gap(y, d, groups, metric)           one rate in men minus the same rate in women
    threshold_for_rate(p, rate)         the score that selects a given share of rows
    threshold_for_tpr(p, y, tpr)        the score that selects a given share of the positives
    decide(p, groups, thr)              0/1 decisions with one threshold per group
    fpr_from_identity(base, tpr, ppv)   the false positive rate those three imply
    feature_of_column(ref)              which Adult feature each of Day 161's one-hot columns came from

Checked by hand: every rate of two groups of four people, two thresholds on
lists small enough to count, per-group decisions, the identity on those
groups, and the 87 one-hot columns of Day 161's matrix.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path

import numpy as np                                     # numpy: every array operation below is built on this

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Day 161 lives in this same folder
with contextlib.redirect_stdout(io.StringIO()):         # its exercise prints hand checks when imported
    import day161_exercise as d161                      # Day 161: the Adult data and its feature lists

GROUPS = ("Female", "Male")                             # the two values of Adult's sex column
METRICS = ("sel", "tpr", "fpr", "ppv")                  # selection rate, true and false positive rates, precision


def group_rates(y, d, groups):                          # per group: rows, base rate and the four rates of the 0/1 decisions d
    out = {}                                                                          # group -> its rates
    for g in np.unique(groups):                                                       # each group
        yy, dd = y[groups == g], d[groups == g]                                       # its outcomes and decisions
        out[g] = {"n": len(yy), "base": yy.mean(),                                    # size; share truly positive
                  "sel": dd.mean(),                                                   # share selected (demographic parity compares this)
                  "tpr": dd[yy == 1].mean() if (yy == 1).any() else np.nan,           # selected among the truly positive
                  "fpr": dd[yy == 0].mean() if (yy == 0).any() else np.nan,           # selected among the truly negative
                  "ppv": yy[dd].mean() if dd.any() else np.nan}                       # truly positive among the selected (precision)
    return out                                                                        # e.g. out["Female"]["tpr"]


def gap(y, d, groups, metric, a="Male", b="Female"):    # one rate in group a minus the same rate in group b
    r = group_rates(y, d, groups)                                                     # every rate of every group
    return r[a][metric] - r[b][metric]                                                # positive = group a higher


def threshold_for_rate(p, rate):                        # the score above which a share `rate` of these rows falls
    return np.quantile(p, 1 - rate)                                                   # e.g. rate 0.2: the 80th percentile


def threshold_for_tpr(p, y, tpr):                       # the score above which a share `tpr` of the positives falls
    return np.quantile(p[y == 1], 1 - tpr)                                            # a quantile of the positives' scores only


def decide(p, groups, thr):                             # 0/1 decisions with one threshold per group (thr: group -> threshold)
    d = np.zeros(len(p), bool)                                                        # nobody selected yet
    for g, t in thr.items():                                                          # each group's own threshold
        d[groups == g] = p[groups == g] > t                                           # applied to its rows only
    return d                                                                          # one decision per row


def fpr_from_identity(base, tpr, ppv):                  # the false positive rate implied by base rate, TPR and precision
    return base / (1 - base) * (1 - ppv) / ppv * tpr                                  # FPR = odds(base) x (1 - PPV)/PPV x TPR


def feature_of_column(ref):                             # the name of the Adult feature behind each of Day 161's 87 one-hot columns
    return np.repeat(np.array(d161.NUM + d161.CAT),                                   # five numbers, then the categories
                     [1] * len(d161.NUM) + [len(np.unique(ref[c])) for c in d161.CAT])  # one column per number, one per category seen


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0]); d = np.array([1, 0, 1, 0, 1, 1, 0, 0], bool)  # outcomes and decisions
    g = np.array(["Female"] * 4 + ["Male"] * 4)                                       # two groups of four
    R = group_rates(y, d, g)                                                          # every rate of both groups
    print("women: outcomes 1, 1, 0, 0, decisions 1, 0, 1, 0 -> by hand base 0.5, selected 0.5, TPR 1/2, FPR 1/2, precision 1/2")
    print(f"  got {[round(float(R['Female'][k]), 4) for k in ('base', 'sel', 'tpr', 'fpr', 'ppv')]}")
    print("men: outcomes 1, 0, 0, 0, decisions 1, 1, 0, 0 -> by hand base 0.25, selected 0.5, TPR 1, FPR 1/3, precision 1/2")
    print(f"  got {[round(float(R['Male'][k]), 4) for k in ('base', 'sel', 'tpr', 'fpr', 'ppv')]}")
    print(f"gap men - women: TPR by hand 1 - 0.5 = 0.5, got {gap(y, d, g, 'tpr'):.4f}; FPR 1/3 - 1/2 = -0.1667, got {gap(y, d, g, 'fpr'):.4f}")
    print(f"fpr_from_identity for the men: odds 0.25/0.75 = 1/3, x (1 - 0.5)/0.5 = 1, x TPR 1 = 0.3333; got {fpr_from_identity(0.25, 1.0, 0.5):.4f}")
    p = np.arange(1, 11) / 10                                                         # scores 0.1, 0.2, ..., 1.0
    t = threshold_for_rate(p, 0.3)                                                    # select 30%
    print(f"threshold_for_rate(0.1 to 1.0, 30%): the 70th percentile, by hand 0.7 + 0.3 x 0.1 = 0.73; got {t:.2f}, selecting {int(np.sum(p > t))} of 10")
    p, y = np.array([0.2, 0.4, 0.6, 0.8]), np.array([1, 0, 1, 1])                   # positives score 0.2, 0.6, 0.8
    t = threshold_for_tpr(p, y, 2 / 3)                                                # two of the three positives
    print(f"threshold_for_tpr(TPR 2/3): positives 0.2, 0.6, 0.8; the 1/3 quantile, by hand 0.2 + (2/3) x 0.4 = 0.4667; got {t:.4f}, "
          f"TPR {np.mean(p[y == 1] > t):.4f}")
    print(f"decide with women at 0.2 and men at 0.5, scores 0.3, 0.6 (women) and 0.3, 0.6 (men): by hand 1, 1, 0, 1; got "
          f"{decide(np.array([0.3, 0.6, 0.3, 0.6]), np.array(['Female', 'Female', 'Male', 'Male']), {'Female': 0.2, 'Male': 0.5}).astype(int).tolist()}")
    tr, te = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Day 161's data and split
    f = feature_of_column({k: v[fi] for k, v in tr.items()})                          # the feature behind each column
    print(f"feature_of_column: {len(f)} columns (Day 161's matrix has {d161.onehot_matrix(tr, {k: v[fi] for k, v in tr.items()}).shape[1]}); "
          f"sex has {int(np.sum(f == 'sex'))}, relationship {int(np.sum(f == 'relationship'))}, marital-status {int(np.sum(f == 'marital-status'))}")
