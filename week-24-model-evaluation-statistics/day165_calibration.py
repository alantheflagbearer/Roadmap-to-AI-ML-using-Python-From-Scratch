"""
Day 165: Calibration -- When a Model Says 70%, Does It Happen 70% of the Time?

A model can rank people well and still give wrong probabilities. Today
measures that on UCI Adult with four of Week 24's models, and fixes it:

  1. reliability diagrams and the expected calibration error (ECE), next to the
     ECE a perfectly calibrated model would show by chance
  2. two fixes fitted on held-out rows: Platt scaling and isotonic regression
  3. the trap: fitting the fix on the rows the model was trained on
  4. how many rows each fix needs
  5. why it matters: totals, decisions, and calibration within groups

Models: Day 161's logistic regression, the same trained on a class-balanced
sample, Day 109's gradient boosting, and Day 161's deep tree. Nothing is chosen
on Adult's test file; Day 167 uses it once.

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
sys.path.insert(0, HERE)                                                             # Day 161, in this folder
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day109_exercise as d109                      # Day 109: trees and gradient boosting
    import day117_exercise as d117                      # Day 117: ROC AUC
    import day161_exercise as d161                      # Day 161: the Adult data, Week 24's split, logistic regression

BINS = 10                                               # equal-width probability bins for the reliability diagram and ECE
EPS = 1e-6                                              # probabilities are kept this far from 0 and 1 before a logarithm
NAMES = ("logistic", "balanced logistic", "boosting", "deep tree")  # the four models, in print order


# =============================================================================
# THE PARTS (copied unchanged into day165_exercise.py, for Day 167's report card)
# =============================================================================
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


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def train_models():                                     # Day 161's data and split; four models' probabilities on every training-file row
    tr, te = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Day 161's data and split
    ref = {k: v[fi] for k, v in tr.items()}; y = tr["y"]                              # codes from fitting rows; targets
    Xt, Xo = d161.tree_matrix(tr, ref), d161.onehot_matrix(tr, ref)                   # both feature matrices
    P = {}; t0 = time.time()                                                          # probabilities per model; a clock
    w, _ = d161.fit_logistic(Xo[fi], y[fi]); P["logistic"] = d161.predict_logistic(w, Xo)  # Day 161's logistic regression
    rng = np.random.RandomState(0); pos = fi[y[fi] == 1]                              # the fitting rows above $50K
    bal = np.r_[pos, rng.choice(fi[y[fi] == 0], len(pos), replace=False)]             # with an equal number of the rest, drawn at random
    w, _ = d161.fit_logistic(Xo[bal], y[bal]); P["balanced logistic"] = d161.predict_logistic(w, Xo)  # trained on a 50/50 sample
    m, _ = d109.fit_boost(Xt[fi], y[fi], "log", 200, 0.1, 3); P["boosting"] = d109.sigmoid(d109.predict_boost(m, Xt))  # 200 trees of depth 3
    t = d109.grow_reg(Xt[fi], y[fi].astype(float), np.arange(len(fi)), 64)          # one tree grown until its leaves are pure
    d109.set_leaves(t, lambda i: y[fi][i].mean()); P["deep tree"] = d109.predict_reg(t, Xt)  # leaf value = share above $50K
    print(f"four models trained on the {len(fi)} fitting rows ({time.time() - t0:.0f} s); the balanced one on {len(bal)} "
          f"rows, half above $50K (the fitting rows have {y[fi].mean():.4f})")
    return P, y, fi, va, tr["sex"]                                                    # probabilities for every row; targets; split; sex


def demo_measure(P, y, va):                                                           # section 1
    print("\n" + "=" * 70)
    print("1. RELIABILITY DIAGRAMS AND THE EXPECTED CALIBRATION ERROR (6,032 VALIDATION ROWS)")
    print("=" * 70)
    yv = y[va]; rng = np.random.RandomState(0); floor = {}                            # validation targets; a seeded generator
    for k in NAMES:                                                                   # each model
        p = P[k][va]                                                                  # its validation probabilities
        sim = [ece(rng.rand(len(p)) < p, p) for _ in range(200)]                      # outcomes drawn from p itself: a perfectly calibrated world
        floor[k] = (np.mean(sim), np.std(sim))                                        # the ECE that chance alone gives
        print(f"  {k:18s}: ECE {ece(yv, p):.4f} (if perfectly calibrated: {floor[k][0]:.4f} +/- {floor[k][1]:.4f}); mean "
              f"probability {p.mean():.4f} against a rate of {yv.mean():.4f}; AUC {d117.roc_auc(yv, p):.4f}, Brier "
              f"{d161.brier(yv, p):.4f}, log loss {log_loss(yv, p):.4f}")
    for k in ("balanced logistic", "deep tree"):                                      # the two badly calibrated ones, bin by bin
        n, mp, my = reliability(yv, P[k][va])                                         # the reliability table
        print(f"  {k}, bin by bin (rows: said -> happened):")
        print("    " + "  ".join(f"{int(a)}: {b:.2f}->{c:.2f}" for a, b, c in zip(n, mp, my) if a > 0))
    p = P["logistic"][va]                                                             # the same ECE with other bin counts
    print(f"  the ECE depends on the bins: logistic with 5, 10, 20, 50 bins: "
          + ", ".join(f"{ece(yv, p, b):.4f}" for b in (5, 10, 20, 50)))
    print("Actual output from this lesson's run.")
    return floor                                                                      # for the figure


def demo_fix(P, y, va):                                                               # section 2
    print("\n" + "=" * 70)
    print("2. TWO FIXES FITTED ON HELD-OUT ROWS: PLATT SCALING AND ISOTONIC REGRESSION")
    print("=" * 70)
    yv = y[va]; half = np.random.RandomState(1).rand(len(va)) < 0.5                  # the validation rows in two random halves
    print(f"cross-fitted: each calibrator is fitted on one half of the validation rows ({int(half.sum())} or "
          f"{int((~half).sum())}) and applied to the other")
    p = P["deep tree"][va]                                                            # the first draft's failure, kept
    w0, w1 = platt_fit(p[half], yv[half], damped=False), platt_fit(p[half], yv[half])  # plain Newton, then with step halving
    print(f"  first draft (plain Newton) on the deep tree: slope {w0[0]:.3g}, intercept {w0[1]:.3g}; "
          f"with step halving: slope {w1[0]:.4f}, intercept {w1[1]:.4f}")
    print(f"  the halves' own rates differ by chance: {yv[half].mean():.4f} and {yv[~half].mean():.4f} above $50K")
    C, E = {}, {}                                                                     # calibrated probabilities; ECE per half, per model and method
    rng = np.random.RandomState(2)                                                    # for the per-half chance floor
    for k in NAMES:                                                                   # each model
        p = P[k][va]; C[k] = {"raw": p}                                               # before
        C[k]["Platt"] = cross_fit(p, yv, half, platt_fit, platt_apply)                # Platt scaling
        C[k]["isotonic"] = cross_fit(p, yv, half, isotonic_fit, isotonic_apply)        # isotonic regression
        E[k] = {}                                                                     # per version: (mean ECE of the halves, its chance floor)
        print(f"  {k}:")
        for c, q in C[k].items():                                                     # each version
            ea, eb = ece(yv[half], q[half]), ece(yv[~half], q[~half])                 # each half scored on its own
            fl = np.mean([ece(rng.rand(h.sum()) < q[h], q[h]) for h in (half, ~half) for _ in range(100)])  # these probabilities, perfectly calibrated
            E[k][c] = ((ea + eb) / 2, fl)                                             # kept for the figure
            print(f"    {c:9s}: ECE on half A {ea:.4f}, half B {eb:.4f} (chance floor {fl:.4f}; both halves pooled: "
                  f"{ece(yv, q):.4f}); Brier {d161.brier(yv, q):.4f}, log loss {log_loss(yv, q):.4f}, mean {q.mean():.4f}")
    q = C["boosting"]["Platt"]                                                        # where the extra error comes from
    print(f"  boosting after Platt: mean probability {q[half].mean():.4f} on half A (rate {yv[half].mean():.4f}), "
          f"{q[~half].mean():.4f} on half B (rate {yv[~half].mean():.4f}): each half gets close to the other half's rate")
    print("  AUC within half B, where one map is used for every row (logistic): " + ", ".join(
        f"{c} {d117.roc_auc(yv[~half], C['logistic'][c][~half]):.4f}" for c in C["logistic"]))
    print("Actual output from this lesson's run.")
    return C, E, half                                                                 # for later sections and the figure


def demo_trap(P, y, fi, va):                                                          # section 3
    print("\n" + "=" * 70)
    print("3. THE TRAP: FITTING THE FIX ON THE ROWS THE MODEL WAS TRAINED ON")
    print("=" * 70)
    yv = y[va]                                                                        # validation targets
    for k in ("deep tree", "boosting", "balanced logistic"):                          # a memoriser, and two that are not
        ptr = P[k][fi]                                                                # the model's probabilities on its own training rows
        out = {c: ece(yv, a(f(ptr, y[fi]), P[k][va])) for c, (f, a) in              # fitted on training rows, scored on validation
               {"Platt": (platt_fit, platt_apply), "isotonic": (isotonic_fit, isotonic_apply)}.items()}  # the two fixes, by name
        print(f"  {k:18s}: training-row log loss {log_loss(y[fi], ptr):.4f} against validation {log_loss(yv, P[k][va]):.4f}; "
              f"validation ECE {ece(yv, P[k][va]):.4f} raw, after a fix fitted on the training rows: Platt {out['Platt']:.4f}, "
              f"isotonic {out['isotonic']:.4f}")
    print("Actual output from this lesson's run.")


def demo_size(P, y, va, half):                                                        # section 4
    print("\n" + "=" * 70)
    print("4. HOW MANY ROWS DOES EACH FIX NEED?")
    print("=" * 70)
    yv = y[va]; R = {}                                                                # validation targets; results for the figure
    for k in ("balanced logistic", "boosting"):                                       # a badly and a well calibrated model
        pa, ya, pb, yb = P[k][va][half], yv[half], P[k][va][~half], yv[~half]         # calibrate on half A, score on half B
        rng = np.random.RandomState(5); fl = np.mean([ece(rng.rand(len(pb)) < pb, pb) for _ in range(200)])  # half B's chance floor
        print(f"  {k}: ECE on half B ({len(pb)} rows) raw {ece(yb, pb):.4f}; perfectly calibrated would give about {fl:.4f}")
        R[k] = {}                                                                     # per method: mean ECE per size
        for c, (f, a) in {"Platt": (platt_fit, platt_apply), "isotonic": (isotonic_fit, isotonic_apply)}.items():  # each fix
            R[k][c] = []                                                              # one mean per size
            for n in (100, 300, 1000, len(pa)):                                       # calibration rows
                e = [ece(yb, a(f(pa[idx], ya[idx]), pb)) for idx in                   # 50 random calibration sets of n rows
                     (np.random.RandomState(100 + s).choice(len(pa), n, replace=False) for s in range(50 if n < len(pa) else 1))]  # one set of every row when n is the whole half
                R[k][c].append(np.mean(e))                                            # their mean ECE
            print(f"    {c:9s}: mean ECE on half B with 100, 300, 1000, {len(pa)} calibration rows: "
                  + ", ".join(f"{v:.4f}" for v in R[k][c]))
    print("Actual output from this lesson's run.")
    return R, len(pa)                                                                 # for the figure


def demo_use(P, C, y, va, sex):                                                       # section 5
    print("\n" + "=" * 70)
    print("5. WHY IT MATTERS: TOTALS, DECISIONS, AND GROUPS")
    print("=" * 70)
    yv = y[va]; sv = sex[va]                                                          # validation targets and sex
    print(f"  {int(yv.sum())} of the {len(yv)} validation people earn above $50K")
    for k, c in (("logistic", "raw"), ("balanced logistic", "raw"), ("balanced logistic", "Platt")):  # three versions
        q = C[k][c]; act = q > 0.5                                                    # probabilities; 'act' where above 50%
        print(f"  {k} ({c}): predicted total {q.sum():.0f}; above 50%: {int(act.sum())} people, of whom {yv[act].mean():.3f} "
              f"really are; accuracy at 0.5: {np.mean(act == yv):.4f}")
    rng = np.random.RandomState(6)                                                    # for each group's chance floor
    for k in ("logistic", "boosting"):                                                # the two calibrated models
        p = P[k][va]                                                                  # their probabilities
        for g in np.unique(sv):                                                       # women, then men
            m = sv == g; fl = np.mean([ece(rng.rand(m.sum()) < p[m], p[m]) for _ in range(200)])  # the group's chance floor
            print(f"  {k:8s} {g:6s} ({int(m.sum())}): mean probability {p[m].mean():.4f} against a rate of {yv[m].mean():.4f}; "
                  f"ECE {ece(yv[m], p[m]):.4f} (chance floor {fl:.4f})")
    print("Actual output from this lesson's run.")


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(P, C, E, y, va, half, R, na):          # six panels from the experiments' returned values
    yv = y[va]; fig, ax = plt.subplots(2, 3, figsize=(17, 10))                        # validation targets; a 2 x 3 grid
    cols = {"logistic": "tab:blue", "balanced logistic": "tab:orange", "boosting": "tab:green", "deep tree": "tab:red"}  # a colour per model
    a = ax[0, 0]; a.plot([0, 1], [0, 1], ":", color="k", lw=1)                        # the diagonal: said = happened
    for k in NAMES:                                                                   # each model
        n, mp, my = reliability(yv, P[k][va]); ok = n >= 20                           # bins with at least 20 rows
        a.plot(mp[ok], my[ok], "o-", color=cols[k], label=f"{k} (ECE {ece(yv, P[k][va]):.3f})")  # its reliability curve
    a.set_xlabel("predicted probability (bin mean)"); a.set_ylabel("share that earned above $50K"); a.legend(fontsize=8)  # labels
    a.set_title("reliability before any fix (bins with 20+ rows)")                    # title
    a = ax[0, 1]                                                                      # how the deep tree's probabilities are spread
    for k in ("balanced logistic", "deep tree", "boosting"):                          # three models
        a.hist(P[k][va], bins=np.linspace(0, 1, 21), histtype="step", lw=2, color=cols[k], label=k)  # their probabilities
    a.set_yscale("log"); a.set_xlabel("predicted probability"); a.set_ylabel("validation rows"); a.legend(fontsize=9)  # labels
    a.set_title("what each model says (log scale)")                                   # title
    a = ax[0, 2]; x = np.arange(len(NAMES)); wd = 0.2                                 # ECE before and after, with the chance floor
    for j, (c, col) in enumerate(zip(("raw", "Platt", "isotonic"), ("0.6", "tab:purple", "tab:cyan"))):  # three versions
        a.bar(x + (j - 1) * wd, [E[k][c][0] for k in NAMES], wd, color=col, label=c)  # their ECE, each half scored on its own
        a.scatter(x + (j - 1) * wd, [E[k][c][1] for k in NAMES], marker="_", s=150, color="k", zorder=3,  # a short black dash at each bar
                  label="if perfectly calibrated" if j == 0 else None)       # each version's chance floor
    a.set_xticks(x); a.set_xticklabels(["logistic", "balanced", "boosting", "deep tree"]); a.set_ylabel("ECE")  # labels
    a.legend(fontsize=9); a.set_title("ECE per half of the validation rows, before and after")  # legend and title
    for a, k in ((ax[1, 0], "balanced logistic"), (ax[1, 1], "deep tree")):           # the two fixed models, bin by bin
        a.plot([0, 1], [0, 1], ":", color="k", lw=1)                                  # the diagonal
        for c, col in (("raw", "0.5"), ("Platt", "tab:purple"), ("isotonic", "tab:cyan")):  # three versions
            n, mp, my = reliability(yv[~half], C[k][c][~half]); ok = n >= 20          # half B only (its map came from half A); bins with 20+ rows
            a.plot(mp[ok], my[ok], "o-", color=col, label=f"{c} (ECE {ece(yv[~half], C[k][c][~half]):.3f})")  # its curve
        a.set_xlabel("predicted probability (bin mean)"); a.set_ylabel("share above $50K"); a.legend(fontsize=9)  # labels
        a.set_title(f"{k}: half B, fixes fitted on half A")                         # title
    a = ax[1, 2]; sizes = [100, 300, 1000, na]                                        # calibration-set size
    for k, ls in (("balanced logistic", "-"), ("boosting", "--")):                    # two models
        for c, col in (("Platt", "tab:purple"), ("isotonic", "tab:cyan")):           # two fixes
            a.plot(sizes, R[k][c], ls, marker="o", color=col, label=f"{k}, {c}")      # mean ECE per size
    a.set_xscale("log"); a.set_xticks(sizes); a.set_xticklabels([str(s) for s in sizes]); a.minorticks_off()  # sizes as ticks
    a.set_xlabel("calibration rows"); a.set_ylabel("mean ECE on the other half"); a.legend(fontsize=8)  # labels
    a.set_title("how many rows each fix needs")                                      # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "calibration_results.png"), dpi=110); plt.close(fig)  # save
    print("saved calibration_results.png")


def draw_diagram():                                     # train, measure, fix on held-out rows, check
    fig, ax = plt.subplots(figsize=(15, 4.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis("off")  # a blank canvas

    def box(x, y, w, h, text, color):                   # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="0.3"))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)  # its text

    def arrow(x0, y0, x1, y1):                          # an arrow between boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))  # the arrow

    box(0.2, 2.6, 2.5, 1.2, "train the model\n(fitting rows)", "#e8f0fb")             # step 1
    box(3.4, 2.6, 2.9, 1.2, "probabilities on\nheld-out rows", "#e6f4ea")             # step 2
    box(7.0, 2.6, 3.0, 1.2, "fit Platt or isotonic\non half A", "#fff4d6")             # step 3
    box(10.7, 2.6, 4.0, 1.2, "score on half B: reliability,\nECE, Brier, log loss, AUC", "#f2e6f7")  # step 4
    for x0, x1 in ((2.75, 3.35), (6.35, 6.95), (10.05, 10.65)):                       # arrows along the top row
        arrow(x0, 3.2, x1, 3.2)                                                       # left to right
    box(7.0, 0.4, 3.0, 1.2, "swap the halves, and score\neach half on its own", "#f4f4f4")  # cross-fitting
    arrow(8.5, 2.55, 8.5, 1.65)                                                       # fix -> swap
    box(0.2, 0.4, 6.1, 1.2, "not on the training rows: a model that memorised\nthem looks perfect there", "#fdecea")  # the trap
    ax.text(7.5, 4.45, "Day 165: calibrating a model's probabilities on rows it never saw", ha="center", fontsize=12)  # title
    fig.savefig(os.path.join(HERE, "calibration_diagram.png"), dpi=110, bbox_inches="tight"); plt.close(fig)  # save
    print("saved calibration_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    P, y, fi, va, sex = train_models()                                                # the four models
    demo_measure(P, y, va)                                                            # 1
    C, E, half = demo_fix(P, y, va)                                                   # 2
    demo_trap(P, y, fi, va)                                                           # 3
    R, na = demo_size(P, y, va, half)                                                 # 4
    demo_use(P, C, y, va, sex)                                                        # 5
    print()
    plot_results(P, C, E, y, va, half, R, na); draw_diagram()                         # the figures
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 166: fairness. The same Adult models, checked across groups: demographic parity, equalized")
    print("odds and calibration within groups, why they cannot all hold at once, and what a threshold per")
    print("group changes.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
