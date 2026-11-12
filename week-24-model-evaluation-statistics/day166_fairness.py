"""
Day 166: Fairness Metrics -- Demographic Parity, Equalized Odds, and Why They Conflict

A model can be accurate overall and still treat groups differently. Today
measures that for two UCI Adult models, by sex, and tests the usual remedies:

  1. the same model across women and men: selection rate, true and false
     positive rates, precision, with bootstrap intervals and a random-group control
  2. 'fairness through unawareness': delete the sex column, then its proxies
  3. why the criteria conflict: a calibrated model, unequal error rates, and the
     identity that ties them to the groups' base rates
  4. one threshold per group, chosen on half the rows and checked on the other
  5. small groups: race x sex, and how wide their intervals are

Models: Day 161's logistic regression and Day 109's gradient boosting, trained
on Week 24's fitting rows and scored on the 6,032 validation rows. Adult's test
file is used once, on Day 167.

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
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, os.path.join(ROOT, "week-16-classical-ml-trees"))                 # Day 109's parts
sys.path.insert(0, os.path.join(ROOT, "week-17-classical-ml-kernels-ensembles"))     # Day 117's parts
sys.path.insert(0, HERE)                                                             # Days 161, 163 and 165, in this folder
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day109_exercise as d109                      # Day 109: gradient boosting
    import day117_exercise as d117                      # Day 117: ROC AUC
    import day161_exercise as d161                      # Day 161: the Adult data, Week 24's split, logistic regression
    import day163_exercise as d163                      # Day 163: the bootstrap and percentile intervals
    import day165_exercise as d165                      # Day 165: the expected calibration error

GROUPS = ("Female", "Male")                             # the two values of Adult's sex column
METRICS = ("sel", "tpr", "fpr", "ppv")                  # selection rate, true and false positive rates, precision
B = 1000                                                # bootstrap resamples
SPLITS = 50                                             # random halvings in section 4


# =============================================================================
# THE PARTS (copied unchanged into day166_exercise.py, for Day 167's report card)
# =============================================================================
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


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def train_models():                                     # Week 24's data, split and two models
    tr, te = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Day 161's data and split
    ref = {k: v[fi] for k, v in tr.items()}; y = tr["y"]                              # codes from fitting rows; targets
    Xt, Xo = d161.tree_matrix(tr, ref), d161.onehot_matrix(tr, ref)                   # both feature matrices
    t0 = time.time(); P = {}                                                          # a clock; probabilities per model
    w, _ = d161.fit_logistic(Xo[fi], y[fi]); P["logistic"] = d161.predict_logistic(w, Xo[va])  # Day 161's logistic regression
    m, _ = d109.fit_boost(Xt[fi], y[fi], "log", 200, 0.1, 3); P["boosting"] = d109.sigmoid(d109.predict_boost(m, Xt[va]))  # Day 109's boosting
    print(f"two models trained on the {len(fi)} fitting rows ({time.time() - t0:.0f} s); scored on {len(va)} validation rows")
    return tr, ref, Xo, y, fi, va, P                                                  # everything the sections need


def demo_measure(P, yv, sv):                                                          # section 1
    print("\n" + "=" * 70)
    print("1. ONE MODEL, TWO GROUPS: THE RATES THAT FAIRNESS CRITERIA COMPARE (THRESHOLD 0.5)")
    print("=" * 70)
    for g in GROUPS:                                                                  # each group's size and base rate
        print(f"  {g:6s}: {int((sv == g).sum())} validation people, {yv[sv == g].mean():.4f} earn above $50K")
    out = {}                                                                          # for the figure
    for k in ("logistic", "boosting"):                                                # each model
        d = P[k] > 0.5; R = group_rates(yv, d, sv); out[k] = R                        # decisions; rates per group
        print(f"  {k}: accuracy {np.mean(d == yv):.4f}")
        for g in GROUPS:                                                              # each group
            print(f"    {g:6s}: selected {R[g]['sel']:.4f}, TPR {R[g]['tpr']:.4f}, FPR {R[g]['fpr']:.4f}, precision {R[g]['ppv']:.4f}")
        print(f"    women's selection rate / men's = {R['Female']['sel'] / R['Male']['sel']:.3f} (the '80% rule' asks for at least 0.8)")
        for mt in METRICS:                                                            # each gap, with a bootstrap interval
            reps = d163.bootstrap_reps(lambda a, b_, c: gap(a, b_, c, mt), (yv, d, sv), B, seed=1)  # resampled rows
            lo, hi = d163.percentile_ci(reps); obs = gap(yv, d, sv, mt)               # 95% interval; the observed gap
            rng = np.random.RandomState(2)                                            # control: shuffle the group labels
            ctl = [gap(yv, d, rng.permutation(sv), mt) for _ in range(200)]           # gaps between random groups of the same sizes
            print(f"    gap men - women in {mt}: {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]; between random groups: spread {np.std(ctl):.4f}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_unaware(tr, ref, Xo, y, fi, va):                                             # section 2
    print("\n" + "=" * 70)
    print("2. 'FAIRNESS THROUGH UNAWARENESS': DELETE THE SEX COLUMN, THEN ITS PROXIES")
    print("=" * 70)
    feat = feature_of_column(ref); yv, sv = y[va], tr["sex"][va]                      # each column's feature; targets; sex
    s = (tr["sex"] == "Male").astype(float); out = []                                 # sex as a 0/1 target; results
    for drop in ([], ["sex"], ["sex", "relationship"], ["sex", "relationship", "marital-status"]):  # ever more removed
        cols = ~np.isin(feat, drop)                                                   # the columns kept
        w, _ = d161.fit_logistic(Xo[fi][:, cols], y[fi]); p = d161.predict_logistic(w, Xo[va][:, cols])  # retrained without them
        c2 = cols & (feat != "sex")                                                   # every kept column except sex itself
        w2, _ = d161.fit_logistic(Xo[fi][:, c2], s[fi])                               # how well do they reveal sex?
        auc_s = d117.roc_auc(s[va], d161.predict_logistic(w2, Xo[va][:, c2]))         # 0.5 = not at all, 1 = perfectly
        d = p > 0.5; R = group_rates(yv, d, sv); out.append((drop, R, np.mean(d == yv)))  # decisions and rates
        print(f"  removed {', '.join(drop) if drop else 'nothing':37s}: accuracy {np.mean(d == yv):.4f}, AUC {d117.roc_auc(yv, p):.4f}; "
              f"sex predictable from the rest with AUC {auc_s:.4f}")
        print(f"    selected women {R['Female']['sel']:.4f}, men {R['Male']['sel']:.4f}; TPR women {R['Female']['tpr']:.4f}, "
              f"men {R['Male']['tpr']:.4f}; FPR women {R['Female']['fpr']:.4f}, men {R['Male']['fpr']:.4f}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_conflict(P, yv, sv):                                                         # section 3
    print("\n" + "=" * 70)
    print("3. WHY THE CRITERIA CONFLICT: CALIBRATED, YET UNEQUAL ERROR RATES")
    print("=" * 70)
    for k in ("logistic", "boosting"):                                                # each model
        p = P[k]; R = group_rates(yv, p > 0.5, sv)                                    # rates at 0.5
        print(f"  {k}:")
        for g in GROUPS:                                                              # each group
            m = sv == g; r = R[g]                                                     # its rows and rates
            print(f"    {g:6s}: mean probability {p[m].mean():.4f} against a rate of {yv[m].mean():.4f} (ECE {d165.ece(yv[m], p[m]):.4f}); "
                  f"precision {r['ppv']:.4f}, TPR {r['tpr']:.4f}, FPR {r['fpr']:.4f}; identity gives FPR "
                  f"{fpr_from_identity(r['base'], r['tpr'], r['ppv']):.4f}")
    R = group_rates(yv, P["logistic"] > 0.5, sv)                                      # the logistic model's rates
    o = {g: R[g]["base"] / (1 - R[g]["base"]) for g in GROUPS}                        # each group's odds of earning above $50K
    print(f"  base-rate odds: women {o['Female']:.4f}, men {o['Male']:.4f} (ratio {o['Male'] / o['Female']:.2f}); with equal "
          f"precision and equal TPR the identity would force men's FPR to be {o['Male'] / o['Female']:.2f} times women's")
    print("Actual output from this lesson's run.")


def demo_thresholds(P, yv, sv):                                                       # section 4
    print("\n" + "=" * 70)
    print("4. ONE THRESHOLD PER GROUP, CHOSEN ON HALF THE ROWS, CHECKED ON THE OTHER HALF")
    print("=" * 70)
    p = P["boosting"]; rng = np.random.RandomState(3); res = {}                       # the better model; a seeded generator
    for s in range(SPLITS):                                                           # random halvings of the validation rows
        A = rng.rand(len(p)) < 0.5; Bh = ~A                                           # choose on A, check on B
        sel = np.mean(p[A] > 0.5); tpr = np.mean(p[A][yv[A] == 1] > 0.5)             # the single-threshold model's overall rates on A
        rules = {"one threshold 0.5": {g: 0.5 for g in GROUPS},                       # the same threshold for everyone
                 "parity (equal selection)": {g: threshold_for_rate(p[A & (sv == g)], sel) for g in GROUPS},  # each group selected at the overall rate
                 "equal opportunity (equal TPR)": {g: threshold_for_tpr(p[A & (sv == g)], yv[A & (sv == g)], tpr) for g in GROUPS}}  # each group's TPR at the overall TPR
        for name, thr in rules.items():                                               # each rule
            d = decide(p[Bh], sv[Bh], thr); R = group_rates(yv[Bh], d, sv[Bh])       # its decisions and rates on half B
            row = [np.mean(d == yv[Bh])] + [R[g][mt] for g in GROUPS for mt in METRICS] + [thr["Female"], thr["Male"]]  # one row of results
            res.setdefault(name, []).append(row)                                      # collected over the halvings
    print(f"means over {SPLITS} random halvings (boosting; thresholds chosen on half A, everything below measured on half B):")
    out = {}                                                                          # for the figure
    for name, rows in res.items():                                                    # each rule
        m = np.mean(rows, 0); sd = np.std(rows, 0); out[name] = m                     # means and spreads over the halvings
        da = np.array(rows)[:, 0] - np.array(res["one threshold 0.5"])[:, 0]          # accuracy change on the same halvings
        print(f"  {name}: thresholds women {m[9]:.3f}, men {m[10]:.3f}; accuracy {m[0]:.4f} (spread {sd[0]:.4f}); "
              f"change against one threshold on the same halves {da.mean():+.4f} (spread {da.std():.4f})")
        print(f"    women: selected {m[1]:.3f}, TPR {m[2]:.3f}, FPR {m[3]:.3f}, precision {m[4]:.3f}")
        print(f"    men:   selected {m[5]:.3f}, TPR {m[6]:.3f}, FPR {m[7]:.3f}, precision {m[8]:.3f}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_small(P, yv, tr, va):                                                        # section 5
    print("\n" + "=" * 70)
    print("5. SMALL GROUPS: RACE x SEX, AND HOW WIDE THEIR INTERVALS ARE")
    print("=" * 70)
    d = P["boosting"] > 0.5; race, sv = tr["race"][va], tr["sex"][va]                 # decisions; race and sex
    out = []                                                                          # for the figure
    for r in np.unique(race):                                                         # each race
        for g in GROUPS:                                                              # each sex
            m = (race == r) & (sv == g); n = int(m.sum()); npos = int(yv[m].sum())    # the group's rows and positives
            lo, hi = d163.wilson_ci(d[m].sum(), n); lo = max(lo, 0.0)                 # Day 163's Wilson interval (lower end floored at 0)
            reps = d163.bootstrap_reps(np.mean, (d[m].astype(float),), B, seed=4)     # the first draft's bootstrap, kept for comparison
            blo, bhi = d163.percentile_ci(reps)                                       # its 95% interval
            tpr = f"{d[m & (yv == 1)].mean():.3f} (of {npos})" if npos else "undefined (nobody positive)"  # TPR, if it exists
            out.append((f"{r[:12]} {g[0]}", n, d[m].mean(), lo, hi))                  # for the figure
            print(f"  {r:18s} {g:6s}: {n:4d} people, base rate {yv[m].mean():.3f}, selected {d[m].mean():.3f} Wilson "
                  f"[{lo:.3f}, {hi:.3f}] (bootstrap [{blo:.3f}, {bhi:.3f}]), TPR {tpr}")
    print(f"  {len(out)} groups x 4 rates = {len(out) * 4} comparisons: Day 164's multiple-comparisons problem returns")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(m1, un, th, sm):                       # six panels from the experiments' returned values
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5)); x = np.arange(len(METRICS)); wd = 0.38  # a 2 x 3 grid; bar positions
    lab = ["selected", "TPR", "FPR", "precision"]                                     # metric labels
    for a, k in ((ax[0, 0], "logistic"), (ax[0, 1], "boosting")):                     # the two models at 0.5
        for j, (g, c) in enumerate(zip(GROUPS, ("tab:purple", "tab:olive"))):         # women, then men
            a.bar(x + (j - 0.5) * wd, [m1[k][g][mt] for mt in METRICS], wd, color=c, label=g)  # their four rates
        a.set_xticks(x); a.set_xticklabels(lab); a.set_ylim(0, 1); a.legend(fontsize=9)  # labels
        a.set_title(f"{k} at threshold 0.5")                                         # title
    a = ax[0, 2]; xs = np.arange(len(un))                                             # unawareness
    a.plot(xs, [R["Female"]["sel"] for _, R, _ in un], "o-", color="tab:purple", label="women selected")  # women's selection
    a.plot(xs, [R["Male"]["sel"] for _, R, _ in un], "o-", color="tab:olive", label="men selected")  # men's selection
    a.plot(xs, [acc for _, _, acc in un], "s--", color="k", label="accuracy")         # accuracy
    a.set_xticks(xs); a.set_xticklabels(["all", "- sex", "- relationship", "- marital"], fontsize=9)  # what was removed
    a.set_ylim(0, 1); a.legend(fontsize=9); a.set_title("deleting sex and its proxies (logistic)")  # legend, title
    a = ax[1, 0]; names = list(th)                                                    # per-group thresholds
    for j, (g, c) in enumerate(zip(GROUPS, ("tab:purple", "tab:olive"))):             # women, then men
        a.bar(np.arange(3) + (j - 0.5) * wd, [th[n][1 + 4 * j] for n in names], wd, color=c, label=f"{g} selected")  # selection
        a.scatter(np.arange(3) + (j - 0.5) * wd, [th[n][2 + 4 * j] for n in names], marker="D", color="k", zorder=3,  # TPR of each group under each rule
                  label="TPR" if j == 0 else None)                                    # TPR as a diamond on each bar
    a.set_xticks(np.arange(3)); a.set_xticklabels(["one 0.5", "parity", "equal TPR"]); a.set_ylim(0, 1)  # labels
    a.legend(fontsize=9); a.set_title("per-group thresholds: selection and TPR (half B)")  # legend, title
    a = ax[1, 1]                                                                      # precision and FPR under each rule
    for j, (g, c) in enumerate(zip(GROUPS, ("tab:purple", "tab:olive"))):             # women, then men
        a.bar(np.arange(3) + (j - 0.5) * wd, [th[n][4 + 4 * j] for n in names], wd, color=c, label=f"{g} precision")  # precision
        a.scatter(np.arange(3) + (j - 0.5) * wd, [th[n][3 + 4 * j] for n in names], marker="v", color="r", zorder=3,  # FPR of each group under each rule
                  label="FPR" if j == 0 else None)                                    # FPR as a red triangle
    a.set_xticks(np.arange(3)); a.set_xticklabels(["one 0.5", "parity", "equal TPR"]); a.set_ylim(0, 1)  # labels
    a.legend(fontsize=9); a.set_title("...and what that does to precision and FPR")  # legend, title
    a = ax[1, 2]; ys = np.arange(len(sm))                                             # small groups
    a.errorbar([s for _, _, s, _, _ in sm], ys, xerr=[[s - lo for _, _, s, lo, _ in sm], [hi - s for _, _, s, _, hi in sm]],  # interval ends as distances below and above
               fmt="o", color="tab:blue", capsize=3)                                  # selection rate with its 95% interval
    a.set_yticks(ys); a.set_yticklabels([f"{nm} ({n})" for nm, n, _, _, _ in sm], fontsize=8)  # group and size
    a.set_xlabel("selected at 0.5 (boosting), 95% interval"); a.set_title("race x sex: small groups, wide intervals")  # labels
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fairness_results.png"), dpi=110); plt.close(fig)  # save
    print("saved fairness_results.png")


def draw_diagram():                                     # one model, three criteria, and the identity that links them
    fig, ax = plt.subplots(figsize=(15, 5.5)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.6); ax.axis("off")  # a blank canvas

    def box(x, y, w, h, text, color):                   # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="0.3"))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)  # its text

    def arrow(x0, y0, x1, y1):                          # an arrow between boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))  # the arrow

    box(0.2, 2.0, 2.6, 1.2, "scores and decisions\nfor each group", "#e8f0fb")       # the start
    crit = [("demographic parity:\nequal share selected", "#fdecea", 3.9),  # text, colour, height
            ("equalized odds:\nequal TPR and FPR", "#fff4d6", 2.0),  # the error-rate criterion
            ("calibration within groups:\nequal precision at equal scores", "#e6f4ea", 0.1)]  # three criteria
    for t, c, y in crit:                                                              # each criterion
        box(4.0, y, 3.6, 1.1, t, c); arrow(2.85, 2.6, 3.95, y + 0.55)                 # its box and arrow
    box(8.8, 1.9, 5.9, 1.4, "FPR = odds(base rate) x (1 - precision)/precision x TPR\n"  # the identity
        "different base rates: not all three can hold at once\n(unless the model makes no mistakes)", "#f2e6f7")  # what it implies
    for t, c, y in crit:                                                              # each criterion feeds the identity
        arrow(7.65, y + 0.55, 8.75, 2.6 + (y - 2.0) * 0.15)                           # its arrow, landing at its own height
    ax.text(7.5, 5.45, "Day 166: three fairness criteria, and why they conflict", ha="center", fontsize=12)  # title
    fig.savefig(os.path.join(HERE, "fairness_diagram.png"), dpi=110, bbox_inches="tight"); plt.close(fig)  # save
    print("saved fairness_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    tr, ref, Xo, y, fi, va, P = train_models()                                        # the data and two models
    yv, sv = y[va], tr["sex"][va]                                                     # validation targets and sex
    m1 = demo_measure(P, yv, sv)                                                      # 1
    un = demo_unaware(tr, ref, Xo, y, fi, va)                                         # 2
    demo_conflict(P, yv, sv)                                                          # 3
    th = demo_thresholds(P, yv, sv)                                                   # 4
    sm = demo_small(P, yv, tr, va)                                                    # 5
    print()
    plot_results(m1, un, th, sm); draw_diagram()                                      # the figures
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 167: the report card. SHAP and LIME built from scratch, then all six of the week's tools")
    print("applied to one model, scored once on Adult's untouched test file.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
