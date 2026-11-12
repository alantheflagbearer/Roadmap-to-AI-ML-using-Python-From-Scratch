"""
Day 162: Hypothesis Testing for ML, with a Real A/B Test

Is a difference real, or could chance alone have made it? Today answers that
on a real randomised experiment: Kevin Hillstrom's MineThatData email test
(2008). 64,000 customers of a retailer were split at random into three
groups: one got an email advertising men's products, one an email for women's
products, and one got nothing. Two weeks later the retailer recorded who
visited the website, who bought, and how much they spent.

  1. the experiment, and a check that the randomisation worked
  2. two-proportion z tests and Welch's test, with confidence intervals
  3. permutation tests: the same question with no formula
  4. A/A tests (does the machinery say "no effect" when there is none?), and peeking
  5. power: how many customers an experiment needs, by formula and by simulation
  6. comparing two ML models on the same rows: McNemar's test (UCI Adult, Day 161's split)

Data: Kevin Hillstrom, "MineThatData E-Mail Analytics and Data Mining
Challenge" (2008), downloaded on first run to ~/week24_data (outside the
repository); UCI Adult through Day 161's exercise file.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import csv                                             # csv: the data file quotes fields that contain commas
import io                                              # io: StringIO, where that output goes
import math                                            # math: erfc, for normal tail probabilities
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: how long each step takes
import urllib.request                                  # urllib: fetch the data on first run
from statistics import NormalDist                      # the standard normal's quantiles (the standard library's)

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, os.path.join(ROOT, "week-16-classical-ml-trees"))                 # Day 109's parts
sys.path.insert(0, HERE)                                                             # Day 161's parts, in this folder
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day109_exercise as d109                      # Day 109: regression trees and gradient boosting
    import day161_exercise as d161                      # Day 161: the Adult data, Week 24's split, logistic regression

DATA = os.path.join(os.path.expanduser("~"), "week24_data")  # where the data is kept: outside the repository
HILL_URL = ("http://www.minethatdata.com/"
            "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")  # the author's own copy
ARMS = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")    # the three groups, as the file names them
ALPHA = 0.05                                            # the significance level used all day
Z975 = NormalDist().inv_cdf(0.975)                      # 1.96: the normal quantile for 95% intervals
R_PERM = 10000                                          # shuffles per permutation test
S_AA = 2000                                             # A/A experiments
LOOK = 500                                              # peeking: a look after every 500 customers per group
R_POW = 1000                                            # simulated experiments per sample size
S_SAME = 100                                            # repeats of the same-procedure control for McNemar


# =============================================================================
# THE PARTS (copied unchanged into day162_exercise.py, for Day 167's report card)
# =============================================================================
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


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def groups(d):                                          # row numbers of each group
    return {a: np.where(d["segment"] == a)[0] for a in ARMS}                          # name -> rows


def covariates(d):                                      # what was known before the emails went out
    cov = {c: d[c] for c in ("recency", "history", "mens", "womens", "newbie")}       # months since last purchase, $ last year, past buying, new customer
    for c in ("channel", "zip_code"):                                                 # two category columns
        for v in np.unique(d[c]):                                                     # each category
            cov[f"{c} {v}"] = (d[c] == v).astype(float)                               # a 0/1 column
    return cov                                                                        # 11 columns


def demo_data(d, g):                                                                  # section 1
    print("\n" + "=" * 70)
    print("1. THE EXPERIMENT, AND A CHECK THAT THE RANDOMISATION WORKED")
    print("=" * 70)
    print(f"Hillstrom's email experiment: {len(d['visit'])} customers, two weeks after the emails")
    for a in ARMS:                                                                    # each group
        r = g[a]                                                                      # its rows
        print(f"  {a:14s} {len(r):6d} customers: visited {d['visit'][r].mean():.4f}, bought {d['conversion'][r].mean():.4f}, "
              f"spent ${d['spend'][r].mean():.3f} each")
    cov = covariates(d); c0 = g["No E-Mail"]; ps = []; smd = []                       # before-email columns; the control group
    for a in ARMS[:2]:                                                                # each emailed group against the control
        for x in cov.values():                                                        # each column
            ps.append(welch_test(x[g[a]], x[c0])[2])                                  # a test that should find nothing
            smd.append((x[g[a]].mean() - x[c0].mean()) / math.sqrt((x[g[a]].var() + x[c0].var()) / 2))  # standardised difference
    ps = np.array(ps); smd = np.abs(smd)                                              # arrays
    print(f"balance: {len(ps)} tests of before-email columns ({len(cov)} columns x 2 emailed groups): "
          f"{int(np.sum(ps < ALPHA))} below {ALPHA} (chance alone expects {len(ps) * ALPHA:.1f}); smallest p {ps.min():.3f}")
    print(f"largest standardised difference {smd.max():.4f} (a common rule of thumb worries above 0.1)")
    print("Actual output from this lesson's run.")


def demo_tests(d, g):                                                                 # section 2
    print("\n" + "=" * 70)
    print("2. TWO-PROPORTION Z TESTS AND WELCH'S TEST, WITH 95% INTERVALS")
    print("=" * 70)
    c0 = g["No E-Mail"]; out = []                                                     # the control; rows for the figure
    for a in ARMS[:2]:                                                                # each email against no email
        r = g[a]                                                                      # its rows
        for col in ("visit", "conversion"):                                           # two yes/no outcomes
            diff, z, p, lo, hi = two_prop_test(d[col][r].sum(), len(r), d[col][c0].sum(), len(c0))  # the z test
            print(f"  {a} vs none, {col:10s}: {diff:+.4f} [{lo:+.4f}, {hi:+.4f}], z {z:.2f}, p {p:.1e} "
                  f"({diff / d[col][c0].mean():+.0%} relative)")
            out.append((a, col, diff / d[col][c0].mean(), lo / d[col][c0].mean(), hi / d[col][c0].mean()))  # relative, for the figure
        diff, t, p, lo, hi = welch_test(d["spend"][r], d["spend"][c0])                # spend per customer
        print(f"  {a} vs none, spend     : ${diff:+.3f} [{lo:+.3f}, {hi:+.3f}], t {t:.2f}, p {p:.1e} "
              f"({diff / d['spend'][c0].mean():+.0%} relative)")
        out.append((a, "spend", diff / d["spend"][c0].mean(), lo / d["spend"][c0].mean(), hi / d["spend"][c0].mean()))  # relative, for the figure
    m, w = g["Mens E-Mail"], g["Womens E-Mail"]                                       # the two emails against each other
    diff, z, p, lo, hi = two_prop_test(d["visit"][m].sum(), len(m), d["visit"][w].sum(), len(w))  # z test: visits, men's email vs women's
    print(f"men's email vs women's email, visit: {diff:+.4f} [{lo:+.4f}, {hi:+.4f}], p {p:.1e}")
    diff, t, p, lo, hi = welch_test(d["spend"][m], d["spend"][w])                         # Welch: spend, men's email vs women's
    print(f"men's email vs women's email, spend: ${diff:+.3f} [{lo:+.3f}, {hi:+.3f}], p {p:.3f}")
    print("among buyers only (not randomised: the email changes who buys, so these groups are different people):")
    for a in ARMS:                                                                    # each group's buyers
        s = d["spend"][g[a]][d["conversion"][g[a]] == 1]                              # what each buyer spent
        print(f"  {a:14s} {len(s):4d} buyers, mean ${s.mean():.2f}, median ${np.median(s):.2f}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_permutation(d, g):                                                           # section 3
    print("\n" + "=" * 70)
    print("3. PERMUTATION TESTS: THE SAME QUESTION, NO FORMULA")
    print("=" * 70)
    m, w, c0 = g["Mens E-Mail"], g["Womens E-Mail"], g["No E-Mail"]                    # the three groups
    buy = d["conversion"] == 1                                                        # buyers
    cases = [("bought, men's vs women's email", d["conversion"][m], d["conversion"][w], "z"),
             ("spend, men's vs women's email", d["spend"][m], d["spend"][w], "welch"),
             ("spend among buyers, women's email vs none", d["spend"][w][buy[w]], d["spend"][c0][buy[c0]], "welch")]  # three questions
    print(f"{R_PERM} shuffles each; formula's p against the shuffles' p")
    keep = None                                                                       # one null distribution, for the figure
    for k, (name, a, b, kind) in enumerate(cases):                                    # each question
        t0 = time.time()                                                              # a clock
        pf = two_prop_test(a.sum(), len(a), b.sum(), len(b))[2] if kind == "z" else welch_test(a, b)[2]  # the formula
        obs, pp = permutation_test(a, b)                                              # the shuffles
        print(f"  {name}: difference {obs:+.4f}; {'z test' if kind == 'z' else 'Welch'} p {pf:.4f}, permutation p {pp:.4f} "
              f"({time.time() - t0:.0f} s)")
        if k == 0:                                                                    # the first question's shuffles never matched it
            print(f"    (the smallest p that {R_PERM} shuffles can give is 1/{R_PERM + 1} = {1 / (R_PERM + 1):.4f})")
        if k == 1:                                                                    # the spend question
            print(f"    (the shuffles' p carries its own sampling spread: about {math.sqrt(pp * (1 - pp) / R_PERM):.4f} at this p)")
            x = np.r_[a, b]; rng = np.random.RandomState(0); sh = []                  # the same shuffles again, kept this time
            for _ in range(R_PERM):                                                   # each shuffle
                p = rng.permutation(len(x)); sh.append(x[p[:len(a)]].mean() - x[p[len(a):]].mean())  # its difference
            keep = (obs, np.array(sh))                                                # the real difference and the shuffled ones
    print("Actual output from this lesson's run.")
    return keep                                                                       # (observed, shuffled differences)


def demo_aa(d, g):                                                                    # section 4
    print("\n" + "=" * 70)
    print("4. A/A TESTS: NO EFFECT EXISTS, SO 5% SHOULD BE 'SIGNIFICANT'; AND PEEKING")
    print("=" * 70)
    c0 = g["No E-Mail"]; h = len(c0) // 2; rng = np.random.RandomState(1)             # the control group, halved at random
    P = {"visit": [], "conversion": [], "spend": []}; looks = np.arange(LOOK, h + 1, LOOK)  # p values; the peeking checkpoints
    ever = {"visit": np.zeros(len(looks)), "conversion": np.zeros(len(looks))}        # how many experiments stopped by each look
    for s in range(S_AA):                                                             # each A/A experiment
        p = rng.permutation(c0); A, B = p[:h], p[h:2 * h]                             # two random halves of the untreated
        for col in ("visit", "conversion"):                                           # yes/no outcomes
            P[col].append(two_prop_test(d[col][A].sum(), h, d[col][B].sum(), h)[2])   # the full-size test
            ca, cb = np.cumsum(d[col][A]), np.cumsum(d[col][B])                       # running totals, as customers arrive
            sig = [two_prop_test(ca[n - 1], n, cb[n - 1], n)[2] < ALPHA for n in looks]  # a test at every look
            ever[col] += np.maximum.accumulate(sig)                                   # stopped by look k = any significant so far
        P["spend"].append(welch_test(d["spend"][A], d["spend"][B])[2])                # spend
    print(f"{S_AA} A/A experiments: the {len(c0)} untreated customers split at random into two groups of {h}")
    for col, v in P.items():                                                          # each outcome
        v = np.array(v); P[col] = v                                                   # as an array
        print(f"  {col:10s}: p < 0.05 in {np.mean(v < 0.05):.4f}, p < 0.01 in {np.mean(v < 0.01):.4f}")
    print(f"peeking: a test after every {LOOK} customers per group ({len(looks)} looks), stopping at the first p < 0.05")
    for col in ever:                                                                  # each yes/no outcome
        r = ever[col] / S_AA                                                          # share stopped by each look
        print(f"  {col:10s}: falsely 'significant' at some look {r[-1]:.4f} (1 look {r[0]:.4f}, 5 looks {r[4]:.4f}, 10 looks {r[9]:.4f}); "
              f"at the last look alone {np.mean(P[col] < 0.05):.4f}")
    print("Actual output from this lesson's run.")
    return P, looks, {k: v / S_AA for k, v in ever.items()}                          # for the figure


def demo_power(d, g):                                                                 # section 5
    print("\n" + "=" * 70)
    print("5. POWER: HOW MANY CUSTOMERS DOES AN EXPERIMENT NEED?")
    print("=" * 70)
    c0 = g["No E-Mail"]; rng = np.random.RandomState(2); out = []                     # control; a seeded generator; results
    print("customers per group for 80% power at alpha 0.05, if the true rates are the ones observed:")
    for a in ARMS[:2]:                                                                # each email
        for col in ("visit", "conversion"):                                           # each outcome
            print(f"  {a} vs none, {col:10s}: {d[col][g[a]].mean():.4f} vs {d[col][c0].mean():.4f} -> "
                  f"{math.ceil(n_per_arm(d[col][g[a]].mean(), d[col][c0].mean()))} per group")
    print(f"checked by simulation: {R_POW} experiments of n per group, drawn with replacement from the real groups")
    plan = [("Mens E-Mail", "visit", (100, 200, 300, 500, 800)), ("Mens E-Mail", "conversion", (500, 1000, 2000, 3000, 4000, 6000, 8000)),
            ("Womens E-Mail", "conversion", (2000, 5000, 10000, 15000, 20000))]       # three effects, from easy to hard
    for a, col, ns in plan:                                                           # each effect
        p1, p0 = d[col][g[a]].mean(), d[col][c0].mean(); row = []                     # its true rates (here: the groups' rates)
        for n in ns:                                                                  # each size
            x1 = d[col][g[a]][rng.randint(0, len(g[a]), (R_POW, n))].sum(1)            # buyers among n drawn from the emailed group, R times
            x0 = d[col][c0][rng.randint(0, len(c0), (R_POW, n))].sum(1)               # and among n drawn from the control
            sim = np.mean([two_prop_test(u, n, v, n)[2] < ALPHA for u, v in zip(x1, x0)])  # share of experiments that find it
            row.append((n, sim, power_two_prop(n, p1, p0)))                           # simulated and formula
        out.append((a, col, row))                                                     # for the figure
        print(f"  {a} vs none, {col}: " + "; ".join(f"n {n}: {s:.3f} / {f:.3f}" for n, s, f in row) + "  (simulated / formula)")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_mcnemar():                                                                   # section 6
    print("\n" + "=" * 70)
    print("6. COMPARING TWO MODELS ON THE SAME ROWS: McNEMAR'S TEST")
    print("=" * 70)
    tr, te = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Day 161's data and split
    ref = {k: v[fi] for k, v in tr.items()}; y = tr["y"]; yv = y[va]                  # codes from fitting rows; targets
    Xt, Xo = d161.tree_matrix(tr, ref), d161.onehot_matrix(tr, ref)                   # both feature matrices
    w, _ = d161.fit_logistic(Xo[fi], y[fi]); err = {"logistic": (d161.predict_logistic(w, Xo[va]) > 0.5) != yv}  # logistic regression's mistakes
    for rounds in (100, 200):                                                         # boosting with two lengths
        m, _ = d109.fit_boost(Xt[fi], y[fi], "log", rounds, 0.1, 3)                   # Day 109's boosting, depth 3, rate 0.1
        err[f"boost {rounds}"] = (d109.predict_boost(m, Xt[va]) > 0) != yv            # score above 0 = probability above 0.5
    print(f"validation rows {len(va)}; error rates: " + ", ".join(f"{k} {v.mean():.4f}" for k, v in err.items()))
    out = []                                                                          # for the figure
    for a, b in (("logistic", "boost 200"), ("boost 100", "boost 200")):              # two comparisons
        bb, cc, chi2, p, ex = mcnemar(err[a], err[b])                                 # paired: only the disagreements count
        pu = two_prop_test(err[a].sum(), len(va), err[b].sum(), len(va))[2]           # unpaired, as if the rows were different people
        print(f"  {a} vs {b}: only {a} wrong {bb}, only {b} wrong {cc}; McNemar chi2 {chi2:.2f}, p {p:.2e} "
              f"(exact {ex:.2e}); unpaired z test p {pu:.4f}")
        out.append((f"{a}\nvs {b}", bb, cc, p, pu))                                   # for the figure
    print(f"same procedure, different training rows: {S_SAME} times, the fitting rows cut into two random halves,")
    print("one model trained on each, McNemar on the validation rows (no real difference in the procedure):")
    same = {}                                                                         # p values per model type
    for name in ("logistic", "deep tree"):                                            # a stable model and an unstable one
        ps = []; t0 = time.time()                                                     # p values; a clock
        for s in range(S_SAME):                                                       # each repeat
            p = np.random.RandomState(100 + s).permutation(fi); h = len(fi) // 2       # a random cut of the fitting rows
            e = []                                                                    # the two models' mistakes
            for idx in (np.sort(p[:h]), np.sort(p[h:])):                              # each half
                if name == "logistic":                                                # logistic regression
                    wh, _ = d161.fit_logistic(Xo[idx], y[idx]); pr = d161.predict_logistic(wh, Xo[va])  # fit, then predict validation rows
                else:                                                                 # one tree grown until its leaves are pure
                    t = d109.grow_reg(Xt[idx], y[idx].astype(float), np.arange(len(idx)), 64)  # grown until its leaves are pure
                    d109.set_leaves(t, lambda i: y[idx][i].mean()); pr = d109.predict_reg(t, Xt[va])  # leaf value = share above $50K; predict
                e.append((pr > 0.5) != yv)                                            # its validation mistakes
            ps.append(mcnemar(e[0], e[1])[3])                                         # the paired test
        same[name] = np.array(ps)                                                     # all repeats
        print(f"  {name:9s}: p < 0.05 in {np.mean(same[name] < 0.05):.2f} of {S_SAME} (a test of fixed models expects 0.05; "
              f"spread of that share about {math.sqrt(0.05 * 0.95 / S_SAME):.3f}) ({time.time() - t0:.0f} s)")
    print("Actual output from this lesson's run.")
    return out, same                                                                  # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(eff, null, aa, pw, mc):                # six panels from the experiments' returned values
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # a 2 x 3 grid
    a = ax[0, 0]; lab = []                                                            # effects with intervals
    for k, (arm, col, r, lo, hi) in enumerate(eff):                                   # each comparison
        c = "tab:blue" if arm.startswith("Mens") else "tab:orange"                    # colour by email
        a.errorbar(r * 100, k, xerr=[[(r - lo) * 100], [(hi - r) * 100]], fmt="o", color=c, capsize=4)  # point and 95% interval
        lab.append(f"{arm.split()[0].replace('Womens', 'Women')}, {col}".replace("Mens", "Men"))  # row label
    a.axvline(0, color="k", lw=0.8); a.set_yticks(range(len(eff))); a.set_yticklabels(lab); a.invert_yaxis()  # zero line; labels top-down
    a.set_xlabel("change against no email (% of the no-email rate)"); a.set_title("every effect, with its 95% interval")  # axis label and title
    a = ax[0, 1]; obs, sh = null                                                      # the permutation null
    a.hist(sh, bins=60, color="0.7"); a.axvline(obs, color="tab:red", lw=2, label=f"observed {obs:+.3f}")  # shuffled differences; the real one
    a.axvline(-obs, color="tab:red", lw=2, ls="--", label="as extreme, other side")  # two-sided
    a.set_xlabel("difference in mean spend ($), men's minus women's email"); a.set_ylabel("shuffles"); a.legend(fontsize=9)  # labels and legend
    a.set_title("permutation test: where the real difference falls")                      # title
    a = ax[0, 2]; P, looks, ever = aa                                                 # A/A p values
    for col, c in (("visit", "tab:blue"), ("spend", "tab:green"), ("conversion", "tab:orange")):  # three outcomes
        a.hist(P[col], bins=20, range=(0, 1), histtype="step", lw=1.8, color=c, label=col)  # their p values
    a.axhline(S_AA / 20, color="k", ls=":", lw=1, label="flat: what chance gives")    # the uniform level
    a.set_xlabel("p value"); a.set_ylabel("A/A experiments"); a.legend(fontsize=9)        # labels and legend
    a.set_title("A/A p values: flat, as chance gives (conversion lumpy)")                 # title
    a = ax[1, 0]                                                                      # peeking
    for col, c in (("visit", "tab:blue"), ("conversion", "tab:orange")):              # two outcomes
        a.plot(np.arange(1, len(looks) + 1), ever[col] * 100, "o-", color=c, label=col)  # false alarms by number of looks
    a.axhline(5, color="k", ls=":", lw=1, label="the promised 5%")                     # nominal
    a.set_xlabel(f"looks taken (one per {LOOK} customers per group)"); a.set_ylabel("A/A experiments stopped as 'significant' (%)")  # axis labels
    a.legend(fontsize=9); a.set_title("peeking: every extra look adds false alarms")      # legend and title
    a = ax[1, 1]                                                                      # power curves
    for (arm, col, row), c in zip(pw, ("tab:blue", "tab:orange", "tab:purple")):      # three effects
        n = np.array([r[0] for r in row])                                             # the sizes simulated
        a.plot(n, [r[1] for r in row], "o", color=c)                                  # simulated
        a.plot(n, [r[2] for r in row], "-", color=c, label=f"{arm.split()[0].replace('Womens', 'Women').replace('Mens', 'Men')}, {col}")  # formula
    a.axhline(0.8, color="k", ls=":", lw=1); a.set_xscale("log")                      # the 80% target
    a.set_xlabel("customers per group"); a.set_ylabel("power"); a.legend(fontsize=9)      # labels and legend
    a.set_title("power: formula (lines) and simulation (dots)")                           # title
    a = ax[1, 2]; out, same = mc                                                      # McNemar
    x = np.arange(len(out)); a.bar(x - 0.18, [o[1] for o in out], 0.36, label="only the first model wrong")  # b
    a.bar(x + 0.18, [o[2] for o in out], 0.36, label="only the second model wrong")   # c
    for k, o in enumerate(out):                                                       # annotate the two p values
        a.text(k, max(o[1], o[2]) * 1.04, f"paired p {o[3]:.1e}\nunpaired p {o[4]:.1e}", ha="center", fontsize=9)  # both p values above the bars
    a.set_xticks(x); a.set_xticklabels([o[0] for o in out]); a.set_ylim(0, max(max(o[1], o[2]) for o in out) * 1.3)  # ticks, labels and room for the text
    a.set_ylabel("validation rows"); a.legend(fontsize=9, loc="upper right")              # label and legend
    a.set_title("McNemar: only the rows where the models disagree count")                 # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "hypothesis_testing_results.png"), dpi=110); plt.close(fig)  # save the figure
    print("saved hypothesis_testing_results.png")


def draw_diagram():                                     # the A/B test as a pipeline, with its three safeguards
    fig, ax = plt.subplots(figsize=(15, 5.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.6); ax.axis("off")  # a blank canvas

    def box(x, y, w, h, text, color):                   # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="0.3"))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)       # its text

    def arrow(x0, y0, x1, y1):                          # an arrow between boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))  # an arrow from (x0, y0) to (x1, y1)

    top = [("customers\nassigned at random", "#e8f0fb"), ("check balance on\nbefore-email columns", "#e6f4ea"),
           ("one test, at the\nsize fixed in advance", "#fdecea"), ("difference, 95% interval\nand p value", "#fff4d6"),
           ("decide", "#f2e6f7")]                                                     # the pipeline
    xs = [0.3, 3.3, 6.3, 9.3, 12.3]                                                   # left edges
    for (t, c), x in zip(top, xs):                                                    # draw each step
        box(x, 3.4, 2.4, 1.2, t, c)                                                       # one step
    for x in xs[:-1]:                                                                 # arrows between steps
        arrow(x + 2.45, 4.0, x + 2.95, 4.0)                                               # to the next step
    low = [(0.3, "power: customers needed\nfor an 80% chance\nto see the effect"), (6.3, "no peeking: every extra\nlook is another chance\nof a false alarm"),
           (9.3, "A/A test: with no effect,\np < 0.05 only 5%\nof the time")]          # the safeguards, under the step each protects
    for x, t in low:                                                                  # draw each, with an arrow up
        box(x, 0.9, 2.4, 1.3, t, "#f4f4f4"); arrow(x + 1.2, 2.25, x + 1.2, 3.35)  # the safeguard and its arrow
    ax.text(7.5, 5.25, "Day 162: an A/B test, and the three checks that keep its 5% honest", ha="center", fontsize=12)  # title
    fig.savefig(os.path.join(HERE, "hypothesis_testing_diagram.png"), dpi=110, bbox_inches="tight"); plt.close(fig)  # save the diagram
    print("saved hypothesis_testing_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    d = load_hillstrom(); g = groups(d)                                               # the experiment
    demo_data(d, g)                                                                   # 1
    eff = demo_tests(d, g)                                                            # 2
    null = demo_permutation(d, g)                                                     # 3
    aa = demo_aa(d, g)                                                                # 4
    pw = demo_power(d, g)                                                             # 5
    mc = demo_mcnemar()                                                               # 6
    print()
    plot_results(eff, null, aa, pw, mc); draw_diagram()                               # the figures
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 163: confidence intervals and the bootstrap. Today's intervals came from formulas that")
    print("assume large, well-behaved samples; tomorrow resamples the data itself, and checks how often")
    print("each kind of interval really contains the truth.")


if __name__ == "__main__":                                                                # only when run directly, not when imported
    main()                                                                                # run everything
