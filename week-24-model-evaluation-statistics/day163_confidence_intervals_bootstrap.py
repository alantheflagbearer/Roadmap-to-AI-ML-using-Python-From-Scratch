"""
Day 163: Confidence Intervals and the Bootstrap, with Real Resampling

A 95% confidence interval promises to contain the true value in 95% of
experiments. Today checks that promise instead of trusting it. The trick:
treat one whole group of Hillstrom's 64,000 customers as the "population",
whose true values are known, draw thousands of small samples from it, build
an interval from each, and count how often the intervals contain the truth.

  1. intervals for a rate: the textbook (Wald) interval, Wilson's, and the bootstrap
  2. intervals for a mean when the data are mostly zeros: normal, percentile, BCa
  3. the bootstrap where no formula exists: a median, a ratio, and how many resamples
  4. intervals for a model's score (UCI Adult): rows resampled, pairs, and refitting
  5. where the bootstrap fails: the largest value

Data: Hillstrom's email experiment through Day 162's exercise file; UCI
Adult, Week 24's split and logistic regression through Day 161's; Day 109's
boosting; Day 117's ROC AUC.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import math                                            # math: square roots
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: how long each step takes
from statistics import NormalDist                      # the standard normal's quantiles (the standard library's)

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, os.path.join(ROOT, "week-16-classical-ml-trees"))                 # Day 109's parts
sys.path.insert(0, os.path.join(ROOT, "week-17-classical-ml-kernels-ensembles"))     # Day 117's parts
sys.path.insert(0, HERE)                                                             # Days 161 and 162, in this folder
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day109_exercise as d109                      # Day 109: gradient boosting
    import day117_exercise as d117                      # Day 117: ROC AUC
    import day161_exercise as d161                      # Day 161: the Adult data, Week 24's split, logistic regression
    import day162_exercise as d162                      # Day 162: Hillstrom's experiment

ALPHA = 0.05                                            # 95% intervals all day
Z975 = NormalDist().inv_cdf(0.975)                      # 1.96
B_BOOT = 2000                                           # bootstrap resamples, unless stated
S_RATE = 4000                                           # simulated samples per size, for rates
S_MEAN = 300                                            # simulated samples per size, for means (each needs its own bootstrap)
RATE_NS = (50, 100, 300, 1000, 3000)                    # sample sizes for section 1
MEAN_NS = (200, 1000, 5000)                             # sample sizes for section 2


# =============================================================================
# THE PARTS (copied unchanged into day163_exercise.py, for Day 167's report card)
# =============================================================================
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


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_rates(d):                                                                    # section 1
    print("\n" + "=" * 70)
    print("1. INTERVALS FOR A RATE: WALD, WILSON, AND THE BOOTSTRAP")
    print("=" * 70)
    c0 = d["segment"] == "No E-Mail"; rng = np.random.RandomState(1); out = {}        # the population: the no-email group
    print(f"population: the {int(c0.sum())} no-email customers; {S_RATE} samples per size, drawn with replacement;")
    print("coverage = share of 95% intervals that contain the population's true rate (should be 0.95)")
    for col in ("conversion", "visit"):                                               # a rare outcome and a common one
        pop = d[col][c0]; truth = pop.mean(); rows = []                               # the population and its true rate
        print(f"  {col} (true rate {truth:.4f}):")
        for n in RATE_NS:                                                             # each sample size
            x = pop[rng.randint(0, len(pop), (S_RATE, n))].sum(1)                     # successes in each sample
            ivs = {"Wald": wald_ci(x, n), "Wilson": wilson_ci(x, n)}                  # the two formulas
            bx = rng.binomial(n, (x / n)[:, None], (S_RATE, 1000)) / n                # bootstrap of a 0/1 sample = binomial draws at its rate
            ivs["bootstrap"] = (np.quantile(bx, 0.025, axis=1), np.quantile(bx, 0.975, axis=1))  # percentile interval per sample
            cov = {k: np.mean((lo <= truth) & (truth <= hi)) for k, (lo, hi) in ivs.items()}  # coverage of each kind
            rows.append((n,) + tuple(cov.values()) + (np.mean(x == 0),))              # for the figure
            print(f"    n {n:4d}: " + ", ".join(f"{k} {v:.3f}" for k, v in cov.items()) +
                  f"; samples with no successes {np.mean(x == 0):.3f}")
        out[col] = np.array(rows)                                                     # this outcome's table
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_means(d):                                                                    # section 2
    print("\n" + "=" * 70)
    print("2. INTERVALS FOR A MEAN OF MOSTLY ZEROS: NORMAL, PERCENTILE, BCa")
    print("=" * 70)
    pop = d["spend"][d["segment"] == "Mens E-Mail"]; truth = pop.mean()               # the population: the men's-email group's spend
    print(f"population: spend of the {len(pop)} men's-email customers: mean ${truth:.3f}, {np.mean(pop == 0):.4f} spent nothing, "
          f"largest ${pop.max():.0f}")
    print(f"{S_MEAN} samples per size; each interval from 1000 bootstrap resamples")
    rng = np.random.RandomState(2); rows = []                                         # a seeded generator; results
    for n in MEAN_NS:                                                                 # each sample size
        hits = np.zeros(3); width = np.zeros(3); none = 0; t0 = time.time()          # covered; widths; samples with no buyer; a clock
        for s in range(S_MEAN):                                                       # each simulated sample
            x = pop[rng.randint(0, len(pop), n)]; m = x.mean(); none += np.all(x == 0)  # the sample; its mean; no buyer at all?
            se = x.std(ddof=1) / math.sqrt(n)                                         # the formula's spread
            reps = bootstrap_reps(np.mean, (x,), 1000, seed=1000 * n + s)             # the bootstrap
            jack = (x.sum() - x) / (n - 1)                                            # the mean's jackknife has a closed form
            ivs = [(m - Z975 * se, m + Z975 * se), percentile_ci(reps), bca_ci(m, reps, jack)]  # the three intervals
            for k, (lo, hi) in enumerate(ivs):                                        # score each
                hits[k] += lo <= truth <= hi; width[k] += hi - lo                     # covered? how wide?
        rows.append((n,) + tuple(hits / S_MEAN) + (none / S_MEAN,))                   # for the figure
        print(f"  n {n:4d}: coverage normal {hits[0] / S_MEAN:.3f}, percentile {hits[1] / S_MEAN:.3f}, BCa {hits[2] / S_MEAN:.3f}; "
              f"mean width ${width[0] / S_MEAN:.3f} / ${width[1] / S_MEAN:.3f} / ${width[2] / S_MEAN:.3f}; "
              f"samples with no buyer {none / S_MEAN:.3f} ({time.time() - t0:.0f} s)")
    print(f"(with {S_MEAN} samples, a coverage of 0.95 is measured with a spread of about {math.sqrt(0.95 * 0.05 / S_MEAN):.3f})")
    print("Actual output from this lesson's run.")
    return np.array(rows)                                                             # for the figure


def demo_noformula(d):                                                                # section 3
    print("\n" + "=" * 70)
    print("3. THE BOOTSTRAP WHERE NO SIMPLE FORMULA EXISTS")
    print("=" * 70)
    m, c0 = d["segment"] == "Mens E-Mail", d["segment"] == "No E-Mail"               # two groups
    buy = d["spend"][m & (d["conversion"] == 1)]                                      # what each men's-email buyer spent
    med = float(np.median(buy)); reps = bootstrap_reps(np.median, (buy,))             # the median; its bootstrap
    jack = jackknife(np.median, (buy,))                                               # leave-one-out medians
    print(f"median spend of the {len(buy)} men's-email buyers: ${med:.2f}; 95% percentile [{percentile_ci(reps)[0]:.2f}, "
          f"{percentile_ci(reps)[1]:.2f}], BCa [{bca_ci(med, reps, jack)[0]:.2f}, {bca_ci(med, reps, jack)[1]:.2f}]")
    print(f"  the bootstrap medians take only {len(np.unique(reps))} different values: a median can only be one of the data's values")
    a, b = d["spend"][m], d["spend"][c0]                                              # spend per customer, both groups
    lift = a.mean() / b.mean() - 1                                                    # relative lift
    ra = bootstrap_reps(np.mean, (a,), seed=1); rb = bootstrap_reps(np.mean, (b,), seed=2)  # each group resampled on its own
    lr = ra / rb - 1                                                                  # the lift in each resample
    print(f"relative lift in spend, men's email vs none: {lift:+.3f}; 95% percentile [{percentile_ci(lr)[0]:+.3f}, {percentile_ci(lr)[1]:+.3f}] "
          f"(each group resampled separately, {B_BOOT} times)")
    print("how many resamples? the median's 95% interval, 10 times with different seeds:")
    spread = []                                                                       # for the figure
    for B in (200, 1000, 5000):                                                       # three resample counts
        ends = np.array([percentile_ci(bootstrap_reps(np.median, (buy,), B, seed=50 + k)) for k in range(10)])  # 10 intervals
        spread.append((B, ends))                                                      # keep them
        print(f"  B {B:4d}: lower end {ends[:, 0].min():.2f} to {ends[:, 0].max():.2f}, upper end {ends[:, 1].min():.2f} to {ends[:, 1].max():.2f}")
    print("Actual output from this lesson's run.")
    return (lift, lr), spread                                                         # for the figure


def demo_models():                                                                    # section 4
    print("\n" + "=" * 70)
    print("4. INTERVALS FOR A MODEL'S SCORE (UCI ADULT, WEEK 24'S VALIDATION ROWS)")
    print("=" * 70)
    tr, te = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Day 161's data and split
    ref = {k: v[fi] for k, v in tr.items()}; y = tr["y"]; yv = y[va]                  # codes from fitting rows; targets
    Xt, Xo = d161.tree_matrix(tr, ref), d161.onehot_matrix(tr, ref)                   # both feature matrices
    fits = {"logistic": lambda r: (lambda w: d161.predict_logistic(w, Xo[va]))(d161.fit_logistic(Xo[r], y[r])[0]),
            "boosting": lambda r: d109.sigmoid(d109.predict_boost(d109.fit_boost(Xt[r], y[r], "log", 200, 0.1, 3)[0], Xt[va]))}  # train on rows r, predict validation
    P = {k: f(fi) for k, f in fits.items()}                                           # the two fitted models' probabilities
    C = {k: ((p > 0.5) == yv).astype(float) for k, p in P.items()}                    # 1 where correct
    rng = np.random.RandomState(3); idx = rng.randint(0, len(va), (B_BOOT, len(va)))  # B resamples of the validation rows
    out = {}                                                                          # intervals, for the figure
    for k in P:                                                                       # each model
        acc = C[k][idx].mean(1)                                                       # accuracy in each resample, all at once
        out[k] = (C[k].mean(), percentile_ci(acc))                                    # point and interval
        print(f"  {k:8s}: accuracy {C[k].mean():.4f} [{out[k][1][0]:.4f}, {out[k][1][1]:.4f}], AUC {d117.roc_auc(yv, P[k]):.4f}")
    dp = C["boosting"][idx].mean(1) - C["logistic"][idx].mean(1)                      # paired: the same resampled rows for both
    idx2 = rng.randint(0, len(va), (B_BOOT, len(va)))                                 # a second, independent set of resamples
    du = C["boosting"][idx].mean(1) - C["logistic"][idx2].mean(1)                     # unpaired: different rows for each model
    diff = C["boosting"].mean() - C["logistic"].mean()                                # the observed gap
    out["paired"] = (diff, percentile_ci(dp)); out["unpaired"] = (diff, percentile_ci(du))  # both intervals
    print(f"accuracy gap, boosting minus logistic: {diff:+.4f}; paired [{out['paired'][1][0]:+.4f}, {out['paired'][1][1]:+.4f}] "
          f"(width {np.ptp(out['paired'][1]):.4f}), unpaired [{out['unpaired'][1][0]:+.4f}, {out['unpaired'][1][1]:+.4f}] "
          f"(width {np.ptp(out['unpaired'][1]):.4f})")
    t0 = time.time(); ad = []                                                         # AUC gap, paired, fewer resamples (each AUC sorts)
    for r in idx[:1000]:                                                              # 1000 resamples
        ad.append(d117.roc_auc(yv[r], P["boosting"][r]) - d117.roc_auc(yv[r], P["logistic"][r]))  # both models, same rows
    out["auc gap"] = (d117.roc_auc(yv, P["boosting"]) - d117.roc_auc(yv, P["logistic"]), percentile_ci(np.array(ad)))
    print(f"AUC gap, boosting minus logistic: {out['auc gap'][0]:+.4f}; paired [{out['auc gap'][1][0]:+.4f}, "
          f"{out['auc gap'][1][1]:+.4f}] from 1000 resamples ({time.time() - t0:.0f} s)")
    print("refitting: each model retrained on bootstrap resamples of the fitting rows, scored on the same validation rows")
    refit = {}                                                                        # validation accuracies of the refits
    for k, B in (("logistic", 50), ("boosting", 20)):                                 # boosting is slow: fewer refits
        t0 = time.time(); r = np.random.RandomState(4)                                # a clock; a seeded generator
        refit[k] = np.array([np.mean(((fits[k](fi[r.randint(0, len(fi), len(fi))])) > 0.5) == yv) for _ in range(B)])  # B refits
        print(f"  {k:8s}: {B} refits, validation accuracy spread {refit[k].std(ddof=1):.4f}, against {np.std(C[k][idx].mean(1), ddof=1):.4f} "
              f"from resampling the validation rows ({time.time() - t0:.0f} s)")
    print("Actual output from this lesson's run.")
    return out, refit                                                                 # for the figure


def demo_failure(d):                                                                  # section 5
    print("\n" + "=" * 70)
    print("5. WHERE THE BOOTSTRAP FAILS: THE LARGEST VALUE")
    print("=" * 70)
    pop = d["history"][d["segment"] == "Mens E-Mail"]; rng = np.random.RandomState(5)  # last year's spend, the men's-email group
    truth = {"largest": pop.max(), "95th percentile": np.quantile(pop, 0.95)}         # the population's true values
    stats = {"largest": np.max, "95th percentile": lambda x: np.quantile(x, 0.95)}    # the two statistics
    n, S = 500, 500; hits = {k: 0 for k in stats}; keep = None                        # sample size; samples; covered counts
    for s in range(S):                                                                # each simulated sample
        x = pop[rng.randint(0, len(pop), n)]                                          # 500 customers
        for k, f in stats.items():                                                    # each statistic
            reps = bootstrap_reps(f, (x,), 1000, seed=s)                              # its bootstrap
            lo, hi = percentile_ci(reps); hits[k] += lo <= truth[k] <= hi             # covered?
            if s == 0 and k == "largest":                                             # keep one example for the figure
                keep = (x.max(), reps, truth[k])                                      # the sample's largest, the bootstrap, the truth
    print(f"population: last year's spend of {len(pop)} customers; {S} samples of {n}, 1000 resamples each")
    for k in stats:                                                                   # each statistic
        print(f"  {k:15s}: true ${truth[k]:.2f}, 95% percentile interval covered it in {hits[k] / S:.3f} of samples")
    x0, reps, _ = keep                                                                # the example
    print(f"one sample: largest ${x0:.2f}; {np.mean(reps == x0):.3f} of its bootstrap resamples repeat that same largest value")
    print("Actual output from this lesson's run.")
    return keep                                                                       # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(rates, means, nof, models, fail):      # six panels from the experiments' returned values
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # a 2 x 3 grid
    a = ax[0, 0]                                                                      # rates
    for col, ls in (("conversion", "-"), ("visit", ":")):                             # rare (solid) and common (dotted)
        r = rates[col]                                                                # its table
        for j, (name, c) in enumerate((("Wald", "tab:red"), ("Wilson", "tab:green"), ("bootstrap", "tab:blue"))):  # three intervals
            a.plot(r[:, 0], r[:, 1 + j], ls, marker="o", color=c, label=f"{name}, {col}")  # coverage by size
    a.axhline(0.95, color="k", lw=0.8); a.set_xscale("log"); a.set_ylim(0.2, 1.0)      # the promise
    a.set_xticks(RATE_NS); a.set_xticklabels([str(n) for n in RATE_NS]); a.minorticks_off()  # plain size labels
    a.set_xlabel("customers in the sample"); a.set_ylabel("coverage"); a.legend(fontsize=8, ncol=2)  # labels and legend
    a.set_title("a rate: purchases (0.6%) solid, visits (11%) dotted")                    # title
    a = ax[0, 1]                                                                      # means
    for j, (name, c) in enumerate((("normal", "tab:red"), ("percentile", "tab:blue"), ("BCa", "tab:purple"))):  # three intervals
        a.plot(means[:, 0], means[:, 1 + j], "o-", color=c, label=name)               # coverage by size
    a.axhline(0.95, color="k", lw=0.8); a.set_xscale("log")                           # the promise
    a.set_xticks(MEAN_NS); a.set_xticklabels([str(n) for n in MEAN_NS]); a.minorticks_off()  # plain size labels
    a.set_xlabel("customers in the sample"); a.set_ylabel("coverage"); a.legend(fontsize=9)  # labels and legend
    a.set_title("mean spend (99% zeros): small samples fall short")                       # title
    a = ax[0, 2]; (lift, lr), spread = nof                                            # the lift's bootstrap
    a.hist(lr * 100, bins=60, color="0.7"); a.axvline(lift * 100, color="tab:red", lw=2, label=f"observed {lift:+.0%}")  # the bootstrap lifts; the observed one
    for v in percentile_ci(lr):                                                       # the interval's ends
        a.axvline(v * 100, color="tab:blue", ls="--")                                 # dashed
    a.set_xlabel("relative lift in spend, men's email vs none (%)"); a.set_ylabel("resamples"); a.legend(fontsize=9)  # labels and legend
    a.set_title("bootstrap of a ratio (dashed: 95% interval)")                            # title
    a = ax[1, 0]                                                                      # how many resamples
    for k, (B, ends) in enumerate(spread):                                            # each B
        a.plot([k - 0.1] * len(ends), ends[:, 0], "o", color="tab:blue", alpha=0.6)   # lower ends
        a.plot([k + 0.1] * len(ends), ends[:, 1], "o", color="tab:orange", alpha=0.6)  # upper ends
    a.set_xticks(range(len(spread))); a.set_xticklabels([f"B = {B}" for B, _ in spread])  # labels
    a.set_ylabel("interval end ($)"); a.set_title("median spend of buyers: lower (blue), upper (orange) ends, 10 seeds")  # label and title
    a = ax[1, 1]; out, refit = models                                                 # model scores
    lab = ["logistic accuracy", "boosting accuracy", "gap, paired", "gap, unpaired"]   # rows
    keys = ["logistic", "boosting", "paired", "unpaired"]                             # matching keys
    for i, k in enumerate(keys):                                                      # each interval
        v, (lo, hi) = out[k]                                                          # point and ends
        a.errorbar(0, i, xerr=[[(v - lo) * 100], [(hi - v) * 100]], fmt="o", capsize=4, color="tab:blue")  # interval around its own estimate
    for i, k in enumerate(("logistic", "boosting")):                                  # refit spreads as +/- 1.96 sd
        sd = refit[k].std(ddof=1); a.errorbar(0, i + 0.25, xerr=Z975 * sd * 100, fmt="s", capsize=4, color="tab:orange")  # +/- 1.96 sd of the refits
    a.set_yticks(range(len(lab))); a.set_yticklabels(lab); a.invert_yaxis()           # labels top-down
    a.axvline(0, color="k", lw=0.8)                                                   # each estimate sits at 0
    a.set_xlabel("distance from the estimate (accuracy points)"); a.set_title("blue: validation rows resampled; orange: refits")  # label and title
    a = ax[1, 2]; x0, reps, truth = fail                                              # the failure case
    a.hist(reps, bins=40, color="0.7"); a.axvline(truth, color="tab:red", lw=2, label=f"true largest ${truth:.0f}")  # the bootstrap largest values; the truth
    a.axvline(x0, color="tab:blue", ls="--", label=f"sample's largest ${x0:.0f}")    # the sample's own largest
    a.set_xlabel("largest of a resample ($)"); a.set_ylabel("resamples"); a.legend(fontsize=9)  # labels and legend
    a.set_title("the largest value: no resample exceeds the sample's")                    # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "bootstrap_results.png"), dpi=110); plt.close(fig)  # save the figure
    print("saved bootstrap_results.png")


def draw_diagram():                                     # the bootstrap, and the coverage check around it
    fig, ax = plt.subplots(figsize=(15, 5.8)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.8); ax.axis("off")  # a blank canvas

    def box(x, y, w, h, text, color):                   # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="0.3"))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)  # its text

    def arrow(x0, y0, x1, y1):                          # an arrow between boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))  # the arrow

    top = [("one sample\nof n rows", "#e8f0fb"), ("draw n rows\nwith replacement,\nB times", "#e6f4ea"),
           ("the statistic on\neach resample", "#fdecea"), ("B values: the\nbootstrap distribution", "#fff4d6"),
           ("95% interval:\npercentile or BCa", "#f2e6f7")]                           # the bootstrap
    xs = [0.3, 3.3, 6.3, 9.3, 12.3]                                                   # left edges
    for (t, c), x in zip(top, xs):                                                    # each step
        box(x, 3.3, 2.4, 1.3, t, c)                                                   # its box
    for x in xs[:-1]:                                                                 # arrows between steps
        arrow(x + 2.45, 3.95, x + 2.95, 3.95)                                         # to the next step
    box(0.3, 0.7, 2.4, 1.3, "a whole group\nas the population\n(true value known)", "#f4f4f4")  # the coverage check
    box(6.3, 0.7, 5.4, 1.3, "repeat for thousands of samples:\ncoverage = share of intervals\nthat contain the true value", "#f4f4f4")  # the coverage check
    arrow(1.5, 2.05, 1.5, 3.25)                                                       # population -> sample
    arrow(13.5, 3.25, 11.75, 1.45)                                                    # interval -> coverage count
    ax.text(7.5, 5.45, "Day 163: the bootstrap (top), and the check of its 95% promise (bottom)", ha="center", fontsize=12)  # title
    fig.savefig(os.path.join(HERE, "bootstrap_diagram.png"), dpi=110, bbox_inches="tight"); plt.close(fig)  # save
    print("saved bootstrap_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    d = d162.load_hillstrom()                                                         # Day 162's experiment
    rates = demo_rates(d)                                                             # 1
    means = demo_means(d)                                                             # 2
    nof = demo_noformula(d)                                                           # 3
    models = demo_models()                                                            # 4
    fail = demo_failure(d)                                                            # 5
    print()
    plot_results(rates, means, nof, models, fail); draw_diagram()                     # the figures
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 164: the multiple-comparisons problem. Hillstrom's customers are cut into many segments")
    print("and each is tested; some come out 'significant' by chance alone. Bonferroni, Holm and")
    print("Benjamini-Hochberg are built and checked on data where the truth is known.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
