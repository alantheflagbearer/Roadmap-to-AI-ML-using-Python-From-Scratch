"""
Day 164: The Multiple-Comparisons Problem, and an Honest p-Hacking Demonstration

One test at the 5% level is wrong 5% of the time when there is no effect.
Sixty-six tests are wrong far more often. Today shows how easily that happens
on real data, and how three corrections fix it:

  1. p-hacking on data with no effect: one A/A split tested in 22 segments x 3 outcomes
  2. Bonferroni, Holm and Benjamini-Hochberg, built and checked on 1,000 A/A splits
  3. a known-truth benchmark: real email effects mixed with A/A nulls, in smaller samples
  4. does a 'discovery' replicate on fresh customers?
  5. the same trap in ML: choosing the best of 100 models on one validation set

Data: Hillstrom's email experiment through Day 162's exercise file; UCI Adult,
Week 24's split and logistic regression through Day 161's.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: how long each step takes

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Days 161 and 162, in this folder
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day161_exercise as d161                      # Day 161: the Adult data, Week 24's split, logistic regression
    import day162_exercise as d162                      # Day 162: Hillstrom's experiment and its tests

ALPHA = 0.05                                            # the error rate each method promises
OUTCOMES = ("visit", "conversion", "spend")             # the three outcomes tested in every segment
S_AA = 1000                                             # A/A splits
S_TRUTH = 300                                           # repeats of the known-truth benchmark
FRAC = 0.3                                              # share of customers in each benchmark sample
SURE = 1e-4                                             # a full-data p below this counts as a certainly real effect
N_MODELS = 100                                          # candidate models in section 5
S_SPLIT = 500                                           # random halvings of the validation rows in section 5
EMPTY = [0]                                             # spend tests with no buyer in either group (a list, so functions can add to it)


# =============================================================================
# THE PARTS (copied unchanged into day164_exercise.py, for Day 167's report card)
# =============================================================================
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


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_phack(d, S):                                                                 # section 1
    print("\n" + "=" * 70)
    print("1. p-HACKING ON DATA WITH NO EFFECT")
    print("=" * 70)
    c0 = d["segment"] == "No E-Mail"; rng = np.random.RandomState(0)                  # the untreated customers
    u = rng.rand(len(c0)) < 0.5; A, B = c0 & u, c0 & ~u                              # a fake 'new email' half, and the rest
    p, names = segment_tests(d, A, B, S)                                              # 22 segments x 3 outcomes
    print(f"one A/A split of the {int(c0.sum())} no-email customers into 'A' ({int(A.sum())}) and 'B' ({int(B.sum())}); "
          f"{len(p)} tests ({len(S)} segments x {len(OUTCOMES)} outcomes)")
    print(f"  'significant' at 0.05: {int(np.sum(p < ALPHA))} (chance alone expects {len(p) * ALPHA:.1f}):")
    for i in np.argsort(p)[:int(np.sum(p < ALPHA))]:                                   # the 'findings', strongest first
        print(f"    {names[i]}: p {p[i]:.4f}")
    print("  after correction: " + ", ".join(f"{k} {int(f(p).sum())}" for k, f in METHODS.items() if k != "no correction"))
    print("Actual output from this lesson's run.")
    return p                                                                          # for the figure


def demo_aa(d, S):                                                                    # section 2
    print("\n" + "=" * 70)
    print("2. THE CORRECTIONS ON 1,000 A/A SPLITS (EVERY 'DISCOVERY' IS FALSE)")
    print("=" * 70)
    c0 = d["segment"] == "No E-Mail"; rng = np.random.RandomState(1); t0 = time.time(); EMPTY[0] = 0  # untreated; seeded; reset the counter
    anyrej = {k: 0 for k in METHODS}; nrej = {k: [] for k in METHODS}; minp = []       # per method: splits with a false discovery; counts
    for s in range(S_AA):                                                             # each A/A split
        u = rng.rand(len(c0)) < 0.5; p, _ = segment_tests(d, c0 & u, c0 & ~u, S)       # 66 tests with no real effect
        minp.append(p.min())                                                          # the best-looking finding
        for k, f in METHODS.items():                                                  # each method
            r = f(p); anyrej[k] += r.any(); nrej[k].append(r.sum())                   # any false discovery? how many?
    print(f"{S_AA} splits, {len(p)} tests each ({time.time() - t0:.0f} s); family-wise error = share of splits with at least one false discovery")
    for k in METHODS:                                                                 # each method
        print(f"  {k:18s}: family-wise error {anyrej[k] / S_AA:.3f}, false discoveries per split {np.mean(nrej[k]):.2f}")
    print(f"if the {len(p)} tests were independent: 1 - 0.95^{len(p)} = {1 - 0.95 ** len(p):.3f}; {EMPTY[0]} of the {S_AA * len(S)} spend tests had no buyer in either group (p = 1)")
    print("Actual output from this lesson's run.")
    return np.array(minp), {k: v / S_AA for k, v in anyrej.items()}                  # for the figure


def demo_truth(d, S):                                                                 # section 3
    print("\n" + "=" * 70)
    print("3. A KNOWN-TRUTH BENCHMARK: REAL EFFECTS MIXED WITH NULLS")
    print("=" * 70)
    c0, m = d["segment"] == "No E-Mail", d["segment"] == "Mens E-Mail"                # control; the men's email
    pfull, names = segment_tests(d, m, c0, S)                                         # every segment x outcome, all customers
    real = pfull < SURE                                                               # certainly real on the full data
    print(f"truth from all 64,000 customers: men's email vs none is certainly real (p < {SURE:g}) in {int(real.sum())} of "
          f"{len(pfull)} segment x outcome tests; the other {int((~real).sum())} are left out (truth uncertain)")
    print(f"each repeat: a random {FRAC:.0%} of customers; {int(real.sum())} real tests (men's email vs none) plus "
          f"{len(pfull)} null tests (an A/A split of the no-email customers)")
    rng = np.random.RandomState(2); t0 = time.time(); EMPTY[0] = 0                    # seeded; a clock; reset the counter
    stats = {k: [] for k in METHODS}                                                  # per method: (true found, false found)
    for s in range(S_TRUTH):                                                          # each repeat
        keep = rng.rand(len(c0)) < FRAC; u = rng.rand(len(c0)) < 0.5                  # the sample; the A/A coin
        pr, _ = segment_tests(d, m & keep, c0 & keep, S)                              # real comparisons in the sample
        pn, _ = segment_tests(d, c0 & keep & u, c0 & keep & ~u, S)                    # null comparisons in the sample
        p = np.r_[pr[real], pn]; is_real = np.r_[np.ones(real.sum(), bool), np.zeros(len(pn), bool)]  # one family of tests
        for k, f in METHODS.items():                                                  # each method
            r = f(p); stats[k].append((np.sum(r & is_real), np.sum(r & ~is_real)))   # discoveries, true and false
    print(f"{S_TRUTH} repeats ({time.time() - t0:.0f} s); FDR = mean share of discoveries that are false; FWER = share of repeats with any")
    out = {}                                                                          # for the figure
    for k in METHODS:                                                                 # each method
        tf = np.array(stats[k]); tot = tf.sum(1)                                      # true, false; total discoveries
        fdp = np.where(tot > 0, tf[:, 1] / np.maximum(tot, 1), 0.0)                   # false share in each repeat (0 if none found)
        out[k] = (tf[:, 0].mean(), tf[:, 1].mean(), fdp.mean(), np.mean(tf[:, 1] > 0))  # the four summaries
        print(f"  {k:18s}: real effects found {out[k][0]:5.2f} of {int(real.sum())}, false discoveries {out[k][1]:.2f}, "
              f"FDR {out[k][2]:.4f}, FWER {out[k][3]:.3f}")
    print(f"({EMPTY[0]} of the {2 * S_TRUTH * len(S)} spend tests had no buyer in either group and were given p = 1)")
    print("Actual output from this lesson's run.")
    return out, int(real.sum())                                                       # for the figure


def demo_replicate(d, S):                                                             # section 4
    print("\n" + "=" * 70)
    print("4. DOES THE BEST-LOOKING FINDING REPLICATE ON FRESH CUSTOMERS?")
    print("=" * 70)
    c0, m = d["segment"] == "No E-Mail", d["segment"] == "Mens E-Mail"                # control; the men's email
    rng = np.random.RandomState(3); out = {}; EMPTY[0] = 0                            # a seeded generator; results; reset the counter
    for label, treated in (("A/A (no effect)", None), ("men's email vs none", m)):    # a null world and a real one
        rep = []; found = again = 0                                                   # the winner's p values; every discovery, and its replications
        for s in range(200):                                                          # each repeat
            half = rng.rand(len(c0)) < 0.5                                            # discovery customers vs fresh customers
            if treated is None:                                                       # A/A: four random quarters of the control
                u = rng.rand(len(c0)) < 0.5                                           # the A/A coin
                g1 = (c0 & half & u, c0 & half & ~u); g2 = (c0 & ~half & u, c0 & ~half & ~u)  # two A/A pairs: one in the discovery half, one in the fresh half
            else:                                                                     # real: the email against none, in each half
                g1 = (treated & half, c0 & half); g2 = (treated & ~half, c0 & ~half)      # the email against none in each half
            p1, _ = segment_tests(d, *g1, S); best = np.argmin(p1)                    # discovery: the smallest of 66 p values
            p2, _ = segment_tests(d, *g2, S)                                          # the same 66 tests on fresh customers
            rep.append((p1[best], p2[best]))                                          # the winner, before and after
            found += np.sum(p1 < ALPHA); again += np.sum((p1 < ALPHA) & (p2 < ALPHA))  # every discovery at 0.05, and how many repeat
        rep = np.array(rep); out[label] = (np.mean(rep[:, 0] < ALPHA), np.mean(rep[:, 1] < ALPHA), again / max(found, 1))  # three rates, for the figure
        print(f"  {label}: over 200 repeats")
        print(f"    the best of 66 in the discovery half: median p {np.median(rep[:, 0]):.1e}, below 0.05 in {np.mean(rep[:, 0] < ALPHA):.3f} "
              f"of repeats; the same test on fresh customers: below 0.05 in {np.mean(rep[:, 1] < ALPHA):.3f}")
        print(f"    every test below 0.05 in the discovery half ({found / 200:.1f} per repeat): {again / max(found, 1):.3f} "
              f"of them below 0.05 again on fresh customers")
    print(f"({EMPTY[0]} of the {2 * 2 * 200 * len(S)} spend tests here had no buyer in either group and were given p = 1)")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_models():                                                                    # section 5
    print("\n" + "=" * 70)
    print("5. THE SAME TRAP IN ML: THE BEST OF 100 MODELS ON ONE VALIDATION SET")
    print("=" * 70)
    tr, te = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Day 161's data and split
    ref = {k: v[fi] for k, v in tr.items()}; y = tr["y"]; yv = y[va]                  # codes from fitting rows; targets
    X = d161.onehot_matrix(tr, ref)                                                   # 87 one-hot columns
    groups = np.repeat(np.arange(12), [1] * 5 + [len(np.unique(ref[c])) for c in d161.CAT])  # which of the 12 features each column came from
    rng = np.random.RandomState(4); C = []; t0 = time.time()                          # a seeded generator; correct/incorrect per model
    for k in range(N_MODELS):                                                         # each candidate model
        use = rng.rand(12) < 0.5                                                      # a random half of the 12 features
        cols = np.flatnonzero(use[groups])                                            # their one-hot columns
        w, _ = d161.fit_logistic(X[fi][:, cols], y[fi])                               # trained on the fitting rows
        C.append((d161.predict_logistic(w, X[va][:, cols]) > 0.5) == yv)               # its validation mistakes
    C = np.array(C, float)                                                            # models x validation rows
    print(f"{N_MODELS} logistic regressions, each on a random half of the 12 features ({time.time() - t0:.0f} s); "
          f"validation accuracy {C.mean(1).min():.4f} to {C.mean(1).max():.4f}")
    gaps = []; pick = []                                                              # optimism of the winner; its identity
    for s in range(S_SPLIT):                                                          # each random halving of the validation rows
        h = rng.rand(len(va)) < 0.5                                                   # selection half, and a fresh half
        a, b = C[:, h].mean(1), C[:, ~h].mean(1)                                      # every model's accuracy on each half
        w = np.argmax(a); gaps.append(a[w] - b[w]); pick.append(w)                    # the winner on the first half, then on the second
    gaps = np.array(gaps)                                                             # one optimism value per halving
    print(f"choose the best on one half of the validation rows, score it on the other half ({S_SPLIT} random halvings):")
    print(f"  the winner's accuracy drops by {gaps.mean():+.4f} on average (spread of that mean {gaps.std(ddof=1) / np.sqrt(len(gaps)):.4f}); "
          f"{len(set(pick))} different models won at least once")
    print(f"  for comparison, a model fixed in advance drops by {np.mean([C[0, h].mean() - C[0, ~h].mean() for h in (rng.rand(S_SPLIT, len(va)) < 0.5)]):+.4f}")
    print("Actual output from this lesson's run.")
    return gaps                                                                       # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(p1, aa, truth, rep, gaps):             # six panels from the experiments' returned values
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # a 2 x 3 grid
    a = ax[0, 0]; ps = np.sort(p1); m = len(ps); k = np.arange(1, m + 1)              # one A/A split's sorted p values
    a.plot(k, ps, "o", ms=4, color="0.3", label="sorted p values (A/A)")              # the p values
    a.plot(k, np.full(m, ALPHA), "--", color="tab:red", label="0.05 (no correction)")  # the naive line
    a.plot(k, ALPHA * k / m, "-", color="tab:green", label="Benjamini-Hochberg line")  # BH's rising line
    a.plot(k, np.full(m, ALPHA / m), ":", color="tab:blue", label="Bonferroni 0.05/66")  # Bonferroni
    a.set_yscale("log"); a.set_xlabel("rank of the p value"); a.set_ylabel("p value"); a.set_ylim(1e-5, 2)  # labels; room below the lines
    a.legend(fontsize=8, loc="lower center", ncol=2)                                   # the legend, under the lines
    a.set_title("one A/A split: 66 tests against three rules")                        # title
    a = ax[0, 1]; minp, fw = aa                                                       # the best-looking finding per split
    a.hist(minp, bins=np.geomspace(1e-4, 1, 30), color="0.7"); a.set_xscale("log")    # smallest p of each split
    a.axvline(ALPHA, color="tab:red", ls="--", label=f"0.05: {np.mean(minp < ALPHA):.0%} of splits")  # the naive line
    a.axvline(ALPHA / 66, color="tab:blue", ls=":", label=f"0.05/66: {np.mean(minp < ALPHA / 66):.1%}")  # Bonferroni's line
    a.set_xlabel("smallest of 66 p values, per A/A split"); a.set_ylabel("splits"); a.legend(fontsize=9, loc="upper left")  # labels and legend
    a.set_title("the best 'finding' when nothing is real")                            # title
    a = ax[0, 2]; out, nreal = truth; names = list(out)                               # the benchmark
    x = np.arange(len(names)); a.bar(x - 0.2, [out[k][0] for k in names], 0.4, color="tab:green", label=f"real effects found (of {nreal})")  # real effects found, per method
    a.bar(x + 0.2, [out[k][1] for k in names], 0.4, color="tab:red", label="false discoveries")  # false discoveries
    a.set_xticks(x); a.set_xticklabels([k.replace("Benjamini-Hochberg", "B-H").replace("no correction", "none") for k in names])  # labels
    a.set_ylabel("per repeat (mean)"); a.legend(fontsize=9); a.set_title("known truth: what each rule finds")  # label, legend, title
    a = ax[1, 0]                                                                      # FDR and FWER per method
    a.bar(x - 0.2, [out[k][2] for k in names], 0.4, label="false discovery rate"); a.bar(x + 0.2, [out[k][3] for k in names], 0.4, label="family-wise error")  # two bars per method
    a.axhline(ALPHA, color="k", ls=":", lw=1); a.set_xticks(x); a.text(2.5, ALPHA + 0.01, "0.05", fontsize=9)  # the 5% target
    a.set_xticklabels([k.replace("Benjamini-Hochberg", "B-H").replace("no correction", "none") for k in names])  # labels
    a.legend(fontsize=9); a.set_title("known truth: the error each rule promises to control")  # legend and title
    a = ax[1, 1]; xs = np.arange(3); lab = ["best of 66:\nsignificant", "best of 66:\nfresh customers", "every discovery:\nfresh customers"]  # replication
    for j, ((label, r), c) in enumerate(zip(rep.items(), ("tab:red", "tab:green"))):  # null and real
        a.bar(xs + (j - 0.5) * 0.4, r, 0.4, color=c, label=label)                     # three rates per world
    a.axhline(ALPHA, color="k", ls=":", lw=1); a.set_xticks(xs); a.set_xticklabels(lab, fontsize=9)  # the 5% line; labels
    a.set_ylim(0, 1.2); a.set_ylabel("share below 0.05"); a.legend(fontsize=9, loc="upper right", ncol=2)  # axis and legend
    a.set_title("discoveries replicate only when real")                               # title
    a = ax[1, 2]                                                                      # the winner's curse
    a.hist(gaps * 100, bins=40, color="tab:purple", alpha=0.8); a.axvline(gaps.mean() * 100, color="k", lw=2, label=f"mean {gaps.mean() * 100:+.2f} points")  # the optimism of each halving, and its mean
    a.axvline(0, color="k", ls=":", lw=1); a.set_xlabel("winner's accuracy: selection half minus fresh half (points)")  # zero line; label
    a.set_ylabel("random halvings"); a.legend(fontsize=9); a.set_title("best of 100 models: its score was flattered")  # label, legend, title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "multiple_comparisons_results.png"), dpi=110); plt.close(fig)  # save
    print("saved multiple_comparisons_results.png")


def draw_diagram():                                     # one family of tests, three ways to read it
    fig, ax = plt.subplots(figsize=(15, 5.8)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.8); ax.axis("off")  # a blank canvas

    def box(x, y, w, h, text, color):                   # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="0.3"))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)  # its text

    def arrow(x0, y0, x1, y1):                          # an arrow between boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))  # the arrow

    box(0.3, 2.3, 2.6, 1.3, "one experiment", "#e8f0fb")                            # the data
    box(3.6, 2.3, 2.8, 1.3, "22 segments x 3 outcomes\n= 66 p values", "#e6f4ea")    # the family
    rules = [("no correction: p < 0.05\n(a false alarm in most\nexperiments)", "#fdecea", 4.3),  # rule text, box colour, height
             ("Bonferroni / Holm: p < 0.05/66\n(any false alarm: at most 5%)", "#fff4d6", 2.35),  # the strict rules
             ("Benjamini-Hochberg: rising line\n(false share of discoveries: at most 5%)", "#f2e6f7", 0.4)]  # three rules
    for t, c, y in rules:                                                             # each rule
        box(7.4, y, 3.6, 1.15, t, c); arrow(6.45, 2.95, 7.35, y + 0.57)                # its box and arrow from the family
    box(11.9, 2.3, 2.8, 1.3, "replicate on\nfresh data", "#f4f4f4")                   # the final check
    for (t, c, y), dy in zip(rules, (0.4, 0.0, -0.4)):                               # each rule feeds replication
        arrow(11.05, y + 0.57, 11.85, 2.95 + dy)                                      # its arrow, landing at its own height
    arrow(2.95, 2.95, 3.55, 2.95)                                                     # experiment -> family
    ax.text(7.5, 5.6, "Day 164: one family of tests, three ways to read it", ha="center", fontsize=12)  # title
    fig.savefig(os.path.join(HERE, "multiple_comparisons_diagram.png"), dpi=110, bbox_inches="tight"); plt.close(fig)  # save
    print("saved multiple_comparisons_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    d = d162.load_hillstrom(); S = segments(d)                                        # Day 162's experiment; the segments
    p1 = demo_phack(d, S)                                                             # 1
    aa = demo_aa(d, S)                                                                # 2
    truth = demo_truth(d, S)                                                          # 3
    rep = demo_replicate(d, S)                                                        # 4
    gaps = demo_models()                                                              # 5
    print()
    plot_results(p1, aa, truth, rep, gaps); draw_diagram()                            # the figures
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 165: calibration. When a model says 70%, does the event happen 70% of the time? Reliability")
    print("diagrams for Day 161's models, the expected calibration error, and two fixes (Platt scaling and")
    print("isotonic regression) fitted on held-out rows.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
