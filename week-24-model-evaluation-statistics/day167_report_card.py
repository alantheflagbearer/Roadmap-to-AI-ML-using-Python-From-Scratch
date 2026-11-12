"""
Day 167: The Report Card -- SHAP and LIME from Scratch, and Week 24's Six Tools on One Model

Days 161-166 each built one evaluation tool. Today they are used together, once,
on a model chosen before Adult's test file is opened:

  0. the plan, fixed on the validation rows: which model, which threshold,
     which tests, and how they will be corrected
  1. performance with intervals (Day 163)
  2. is it better than the simpler model? (Day 162's McNemar test)
  3. over-fitting and variance (Day 161)
  4. calibration (Day 165)
  5. fairness by sex (Day 166)
  6. the card's six tests, corrected for multiple comparisons (Day 164)
  7. SHAP from scratch: exact Shapley values, checked against a closed form,
     then KernelSHAP's sampled approximation, checked against the exact values
  8. LIME from scratch, and how much its answer depends on its settings
  9. the report card

Models: Day 161's logistic regression and Day 109's gradient boosting, trained
on Week 24's 24,130 fitting rows only. The test file (15,060 rows) is scored
once.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: how long each step takes
from math import factorial                             # factorial: the Shapley weights

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, os.path.join(ROOT, "week-16-classical-ml-trees"))                 # Day 109's parts
sys.path.insert(0, os.path.join(ROOT, "week-17-classical-ml-kernels-ensembles"))     # Day 117's parts
sys.path.insert(0, HERE)                                                             # Days 161-166, in this folder
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day109_exercise as d109                      # Day 109: gradient boosting
    import day117_exercise as d117                      # Day 117: ROC AUC
    import day161_exercise as d161                      # Day 161: the data, the split, logistic regression, bias and variance
    import day162_exercise as d162                      # Day 162: two-proportion test, McNemar's test
    import day163_exercise as d163                      # Day 163: Wilson intervals, the bootstrap
    import day164_exercise as d164                      # Day 164: Bonferroni, Holm, Benjamini-Hochberg
    import day165_exercise as d165                      # Day 165: reliability and the expected calibration error
    import day166_exercise as d166                      # Day 166: rates per group and their gaps

FEAT = d161.NUM + d161.CAT                              # the 12 features, in the order of Day 161's tree matrix
NF = len(FEAT)                                          # 12
THRESHOLD = 0.5                                         # the decision threshold, fixed in the plan
B = 1000                                                # bootstrap resamples
SIMS = 200                                              # simulated perfectly calibrated worlds, for the ECE's chance floor
BG = 200                                                # background rows for SHAP and LIME (the first draft used 50; section 7)
KERNEL_M = 500                                          # KernelSHAP: coalitions sampled (each with its complement)
LIME_N = 1000                                           # LIME: perturbed samples per explanation
N_EXPLAIN = 200                                         # test rows explained for the global picture


# =============================================================================
# THE PARTS
# =============================================================================
def fast_auc(y, p):                                     # Day 117's rank form of ROC AUC, with tied ranks averaged by sorting once
    u, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)               # distinct scores, each row's position, their counts
    r = (np.cumsum(cnt) - cnt + (cnt - 1) / 2)[inv]                                   # each row's rank from 0, ties sharing their average
    n1 = y.sum(); n0 = len(y) - n1                                                    # positives and negatives
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))                   # share of (positive, negative) pairs ordered correctly


ALL = ((np.arange(2 ** NF)[:, None] >> np.arange(NF)[None, :]) & 1).astype(bool)    # every coalition of the 12 features: 4096 rows of True/False


def coalition_values(f, x, bg, masks, colfeat):         # v(S) for each coalition S: the mean of f over background rows with S's features taken from x
    out = np.empty(len(masks))                                                        # one value per coalition
    for i in range(0, len(masks), 256):                                               # 256 coalitions at a time, to bound memory
        keep = masks[i:i + 256][:, colfeat]                                           # the coalitions, spelled out per matrix column
        H = np.where(keep[:, None, :], x[None, None, :], bg[None, :, :])              # coalitions x background rows x columns: hybrid rows
        out[i:i + 256] = f(H.reshape(-1, H.shape[-1])).reshape(len(keep), len(bg)).mean(1)  # the model on every hybrid, averaged over the background
    return out                                                                        # v for each coalition


def shap_exact(f, x, bg, colfeat):                      # Shapley values by the definition: every coalition, weighted
    v = coalition_values(f, x, bg, ALL, colfeat)                                      # all 4096 coalition values
    code = ALL @ (1 << np.arange(NF)); size = ALL.sum(1)                              # each coalition's number (its bits) and size
    wt = np.array([factorial(s) * factorial(NF - s - 1) / factorial(NF) for s in range(NF)])  # weight of a coalition of size s
    phi = np.zeros(NF)                                                                # one value per feature
    for j in range(NF):                                                               # each feature
        S = code[~ALL[:, j]]                                                          # coalitions without feature j (ALL[k] is coalition k)
        phi[j] = np.sum(wt[size[S]] * (v[S + (1 << j)] - v[S]))                       # its weighted average contribution when added
    return phi, v[0], v[-1]                                                           # values; the empty coalition (background mean) and the full one, f(x)


def shap_kernel(f, x, bg, colfeat, M=KERNEL_M, seed=0):  # KernelSHAP: sampled coalitions, then a weighted regression
    rng = np.random.RandomState(seed)                                                 # a seeded generator
    sizes = np.arange(1, NF); pk = (NF - 1) / (sizes * (NF - sizes)); pk = pk / pk.sum()  # coalition sizes drawn in proportion to the Shapley kernel
    Z = np.zeros((2 * M, NF), bool)                                                   # the sampled coalitions
    for i in range(M):                                                                # each draw
        Z[2 * i, rng.choice(NF, rng.choice(sizes, p=pk), replace=False)] = True       # a random size, then that many random features
        Z[2 * i + 1] = ~Z[2 * i]                                                      # and its complement, which cancels much of the noise
    v = coalition_values(f, x, bg, np.vstack([np.zeros((1, NF), bool), np.ones((1, NF), bool), Z]), colfeat)  # empty, full, and the samples
    v0, D, vz = v[0], v[1] - v[0], v[2:]                                              # background mean; total to share out; sampled values
    A = Z[:, :-1].astype(float) - Z[:, -1:].astype(float)                             # the last feature eliminated, so the values add up to D exactly
    phi = np.linalg.lstsq(A, vz - v0 - Z[:, -1] * D, rcond=None)[0]                   # least squares (the kernel is already in the sampling)
    return np.r_[phi, D - phi.sum()], v0, v[1]                                        # all 12 values; the background mean; f(x)


def shap_linear(w, x, bg, colfeat):                     # the known answer for a linear model: weight x (value - background mean), summed per feature
    return np.bincount(colfeat, w[1:] * (x - bg.mean(0)), NF)                         # one value per feature


def lime(f, x, bg, colfeat, N=LIME_N, width=None, seed=0):  # LIME: a weighted linear model fitted to the model near x
    rng = np.random.RandomState(seed); width = width or 0.75 * np.sqrt(NF)            # a seeded generator; LIME's usual kernel width
    Z = rng.rand(N, NF) < 0.5; Z[0] = True                                            # each feature kept from x or not, at random; row 0 is x itself
    H = np.where(Z[:, colfeat], x[None, :], bg[rng.randint(0, len(bg), N)])          # features not kept are taken from a random background row
    wt = np.exp(-(~Z).sum(1) / width ** 2)                                            # closeness: exp(-distance^2 / width^2), distance^2 = features changed
    A = np.c_[np.ones(N), Z.astype(float)]                                            # an intercept and one 0/1 column per feature
    R = 1e-3 * np.eye(NF + 1); R[0, 0] = 0.0                                          # a tiny ridge, not on the intercept
    return np.linalg.solve(A.T @ (A * wt[:, None]) + R, A.T @ (wt * f(H)))[1:]        # the weighted fit's coefficients: one per feature


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def train_models():                                     # Week 24's data, split and two models, trained on the fitting rows only
    tr, te = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Day 161's data and split
    ref = {k: v[fi] for k, v in tr.items()}; y = tr["y"]                              # codes from fitting rows; targets
    Xt, Xo = d161.tree_matrix(tr, ref), d161.onehot_matrix(tr, ref)                   # both feature matrices
    t0 = time.time()                                                                  # a clock
    w, _ = d161.fit_logistic(Xo[fi], y[fi])                                           # Day 161's logistic regression
    m, _ = d109.fit_boost(Xt[fi], y[fi], "log", 200, 0.1, 3)                          # Day 109's boosting: 200 trees of depth 3
    print(f"two models trained on the {len(fi)} fitting rows ({time.time() - t0:.0f} s)")
    M = {"logistic": (lambda X: d161.predict_logistic(w, X), "o"), "boosting": (lambda X: d109.sigmoid(d109.predict_boost(m, X)), "t")}  # name -> (probabilities, matrix)
    return dict(tr=tr, te=te, fi=fi, va=va, ref=ref, y=y, Xt=Xt, Xo=Xo, w=w, m=m, M=M)  # everything the sections need


def demo_plan(S):                                       # section 0: the evidence and the plan, from the validation rows only
    print("\n" + "=" * 70)
    print("0. THE PLAN, FIXED ON THE VALIDATION ROWS BEFORE THE TEST FILE IS OPENED")
    print("=" * 70)
    yv = S["y"][S["va"]]; S["pv"] = {}                                                # validation targets; validation probabilities
    for k, (f, mat) in S["M"].items():                                                # each model
        p = f((S["Xo"] if mat == "o" else S["Xt"])[S["va"]]); S["pv"][k] = p          # its validation probabilities
        print(f"  validation, {k:8s}: accuracy {np.mean((p > THRESHOLD) == yv):.4f}, AUC {fast_auc(yv, p):.4f}, ECE {d165.ece(yv, p):.4f}")
    print("  model: boosting (more accurate on validation, and calibrated); baseline: logistic regression")
    print(f"  threshold {THRESHOLD}; the six tests in section 6 are corrected with Holm at 0.05 (Benjamini-Hochberg shown beside it)")
    print("  the test file is scored once, below; nothing is changed after seeing it")
    print("Actual output from this lesson's run.")


def demo_performance(S):                                # section 1 (Day 163)
    print("\n" + "=" * 70)
    print("1. PERFORMANCE ON THE TEST FILE, WITH INTERVALS (DAY 163)")
    print("=" * 70)
    te = S["te"]; yt = te["y"]                                                        # the test rows and their targets
    S["Tt"], S["To"] = d161.tree_matrix(te, S["ref"]), d161.onehot_matrix(te, S["ref"])  # coded with the fitting rows' categories and scales
    S["pt"] = {k: f(S["To"] if mat == "o" else S["Tt"]) for k, (f, mat) in S["M"].items()}  # test probabilities: the one time the test file is scored
    print(f"  {len(yt)} test rows, {yt.mean():.4f} earn above $50K")
    p = S["pt"]["boosting"]                                                           # the chosen model
    print(f"  fast_auc against Day 117's roc_auc on these rows: {fast_auc(yt, p):.10f} vs {d117.roc_auc(yt, p):.10f}")
    for k in ("boosting", "logistic"):                                                # each model
        p = S["pt"][k]; c = int(np.sum((p > THRESHOLD) == yt))                        # its probabilities; correct decisions
        lo, hi = d163.wilson_ci(c, len(yt))                                           # Wilson interval for the accuracy
        reps = d163.bootstrap_reps(fast_auc, (yt, p), B, seed=1); alo, ahi = d163.percentile_ci(reps)  # bootstrap interval for the AUC
        print(f"  {k:8s}: accuracy {c / len(yt):.4f} [{lo:.4f}, {hi:.4f}]; AUC {fast_auc(yt, p):.4f} [{alo:.4f}, {ahi:.4f}]")
    print("Actual output from this lesson's run.")


def demo_baseline(S):                                   # section 2 (Day 162)
    print("\n" + "=" * 70)
    print("2. IS BOOSTING BETTER THAN LOGISTIC REGRESSION? (DAY 162'S McNEMAR TEST)")
    print("=" * 70)
    yt = S["te"]["y"]; pb, pl = S["pt"]["boosting"], S["pt"]["logistic"]              # targets; both models' probabilities
    eb, el = (pb > THRESHOLD) != yt, (pl > THRESHOLD) != yt                           # each model's errors
    b, c, chi2, p, ex = d162.mcnemar(eb, el); S["p_mcnemar"] = ex                     # rows only boosting gets wrong; only logistic
    print(f"  only boosting wrong {b}, only logistic wrong {c}: chi-square {chi2:.2f}, p {p:.2e}, exact p {ex:.2e}")
    reps = d163.bootstrap_reps(lambda a, u, v: fast_auc(a, u) - fast_auc(a, v), (yt, pb, pl), B, seed=2)  # the same resampled rows for both
    lo, hi = d163.percentile_ci(reps)                                                 # paired interval for the AUC difference
    print(f"  AUC difference {fast_auc(yt, pb) - fast_auc(yt, pl):+.4f} [{lo:+.4f}, {hi:+.4f}] (paired bootstrap)")
    print("Actual output from this lesson's run.")


def demo_variance(S):                                   # section 3 (Day 161)
    print("\n" + "=" * 70)
    print("3. OVER-FITTING AND VARIANCE (DAY 161)")
    print("=" * 70)
    fi, y, yt = S["fi"], S["y"], S["te"]["y"]                                         # fitting rows; targets
    for k, (f, mat) in S["M"].items():                                                # each final model
        X = S["Xo"] if mat == "o" else S["Xt"]                                        # its matrix
        print(f"  {k:8s}: Brier on its training rows {d161.brier(y[fi], f(X[fi])):.4f}, on the test file {d161.brier(yt, S['pt'][k]):.4f}")
    q = np.array_split(np.random.RandomState(3).permutation(fi), 4); t0 = time.time()  # four disjoint quarters of the fitting rows
    PL = np.array([d161.predict_logistic(d161.fit_logistic(S["Xo"][i], y[i])[0], S["To"]) for i in q])  # logistic trained on each quarter
    PB = np.array([d109.sigmoid(d109.predict_boost(d109.fit_boost(S["Xt"][i], y[i], "log", 200, 0.1, 3)[0], S["Tt"])) for i in q])  # boosting on each
    for k, P in (("logistic", PL), ("boosting", PB)):                                 # each model family
        br, bn, var = d161.bias_variance(P, yt)                                       # Day 161's decomposition on the test rows
        print(f"  {k:8s} trained on quarters ({len(q[0])} rows each): Brier {br:.4f} = bias^2 + noise {bn:.4f} + variance {var:.4f}")
    print(f"  (quarter models took {time.time() - t0:.0f} s)")
    print("Actual output from this lesson's run.")


def demo_calibration(S):                                # section 4 (Day 165)
    print("\n" + "=" * 70)
    print("4. CALIBRATION ON THE TEST FILE (DAY 165)")
    print("=" * 70)
    yt, sx = S["te"]["y"], S["te"]["sex"]; rng = np.random.RandomState(4)            # targets; sex; a seeded generator
    for k in ("boosting", "logistic"):                                                # each model
        p = S["pt"][k]; fl = np.mean([d165.ece(rng.rand(len(p)) < p, p) for _ in range(SIMS)])  # its probabilities; chance floor
        print(f"  {k:8s}: ECE {d165.ece(yt, p):.4f} (chance floor {fl:.4f}); mean probability {p.mean():.4f} against a rate of {yt.mean():.4f}")
    p = S["pt"]["boosting"]                                                           # the chosen model, per sex
    for g in d166.GROUPS:                                                             # each sex
        m = sx == g; fl = np.mean([d165.ece(rng.rand(m.sum()) < p[m], p[m]) for _ in range(SIMS)])  # its rows; its floor
        print(f"  boosting, {g:6s}: ECE {d165.ece(yt[m], p[m]):.4f} (chance floor {fl:.4f}); mean probability {p[m].mean():.4f} against {yt[m].mean():.4f}")
    S["rel"] = d165.reliability(yt, p)                                                # for the figure
    print("Actual output from this lesson's run.")


def demo_fairness(S):                                   # section 5 (Day 166)
    print("\n" + "=" * 70)
    print("5. FAIRNESS BY SEX ON THE TEST FILE (DAY 166)")
    print("=" * 70)
    yt, sx = S["te"]["y"], S["te"]["sex"]; d = S["pt"]["boosting"] > THRESHOLD        # targets; sex; decisions
    yv, sv = S["y"][S["va"]], S["tr"]["sex"][S["va"]]; dv = S["pv"]["boosting"] > THRESHOLD  # the same on validation
    R, Rv = d166.group_rates(yt, d, sx), d166.group_rates(yv, dv, sv); S["fair"] = (R, Rv)  # rates per group, test and validation
    for g in d166.GROUPS:                                                             # each sex
        print(f"  {g:6s}: {R[g]['n']} test rows, base rate {R[g]['base']:.4f}; selected {R[g]['sel']:.4f}, TPR {R[g]['tpr']:.4f}, "
              f"FPR {R[g]['fpr']:.4f}, precision {R[g]['ppv']:.4f}")
    for mt in d166.METRICS:                                                           # each gap, men minus women
        reps = d163.bootstrap_reps(lambda a, b_, c: d166.gap(a, b_, c, mt), (yt, d, sx), B, seed=5)  # resampled test rows
        lo, hi = d163.percentile_ci(reps)                                             # its 95% interval
        print(f"  gap men - women in {mt}: test {d166.gap(yt, d, sx, mt):+.4f} [{lo:+.4f}, {hi:+.4f}]; validation {d166.gap(yv, dv, sv, mt):+.4f}")
    print(f"  women's selection rate / men's = {R['Female']['sel'] / R['Male']['sel']:.3f}")
    print("Actual output from this lesson's run.")


def demo_tests(S):                                      # section 6 (Day 164)
    print("\n" + "=" * 70)
    print("6. THE CARD'S SIX TESTS, CORRECTED FOR MULTIPLE COMPARISONS (DAY 164)")
    print("=" * 70)
    yt, sx = S["te"]["y"], S["te"]["sex"]; d = S["pt"]["boosting"] > THRESHOLD        # targets; sex; decisions
    yv = S["y"][S["va"]]; dv = S["pv"]["boosting"] > THRESHOLD                         # validation targets and decisions
    names = ["boosting vs logistic (McNemar)"]; p = [S["p_mcnemar"]]                  # test 1, from section 2
    _, _, pa, _, _ = d162.two_prop_test(int(np.sum(d == yt)), len(yt), int(np.sum(dv == yv)), len(yv))  # test 2: accuracy, test file against validation
    names.append("accuracy, test vs validation"); p.append(pa)                        # its name and p value
    w, mn = sx == "Female", sx == "Male"                                              # the two groups
    for mt, base in (("sel", np.ones(len(yt), bool)), ("tpr", yt == 1), ("fpr", yt == 0), ("ppv", d)):  # tests 3-6: the four rates by sex
        hit = yt == 1 if mt == "ppv" else d                                           # what counts: a true positive for precision, else a selection
        _, _, pg, _, _ = d162.two_prop_test(int(np.sum(hit & base & mn)), int(np.sum(base & mn)), int(np.sum(hit & base & w)), int(np.sum(base & w)))  # men's hits and count, then women's
        names.append(f"{mt} gap by sex"); p.append(pg)                                # its name and p value
    p = np.array(p); rej = {k: f(p) for k, f in d164.METHODS.items()}; S["tests"] = (names, p, rej)  # every rule's decisions
    for i, n in enumerate(names):                                                     # one line per test
        print(f"  {n:32s}: p {p[i]:.2e}; " + ", ".join(f"{k} {'reject' if rej[k][i] else 'keep'}" for k in rej))
    print("Actual output from this lesson's run.")


def demo_shap(S):                                       # section 7
    print("\n" + "=" * 70)
    print("7. SHAP FROM SCRATCH: EXACT, CHECKED AGAINST A CLOSED FORM; KERNEL, CHECKED AGAINST EXACT")
    print("=" * 70)
    fo = d166.feature_of_column(S["ref"]); OH = np.array([FEAT.index(c) for c in fo]); TR = np.arange(NF)  # each matrix column's feature
    pick = lambda seed, k: S["fi"][np.random.RandomState(seed).choice(len(S["fi"]), k, replace=False)]  # k random fitting rows
    bgi = pick(5, BG); bgT, bgO = S["Xt"][bgi], S["Xo"][bgi]                          # the background, in both matrices
    rows = np.random.RandomState(6).choice(len(S["te"]["y"]), N_EXPLAIN, replace=False)  # the test rows to explain
    fl = lambda X: S["w"][0] + X @ S["w"][1:]                                         # logistic regression's log-odds
    fb = lambda X: d109.predict_boost(S["m"], X)                                      # boosting's log-odds
    S.update(OH=OH, bgT=bgT, bgO=bgO, rows=rows, fl=fl, fb=fb)                        # kept for section 8
    print(f"  explaining log-odds; a feature 'absent' takes its value from one of {BG} background rows (averaged over all {BG})")
    for r in rows[:3]:                                                                # the known answer first
        phi, v0, v1 = shap_exact(fl, S["To"][r], bgO, OH)                             # by the definition
        print(f"  logistic, test row {r}: largest difference from the closed form {np.abs(phi - shap_linear(S['w'], S['To'][r], bgO, OH)).max():.1e}; "
              f"sum of values {phi.sum():+.4f} = f(x) - background mean {v1 - v0:+.4f}")
    t0 = time.time(); EX = []                                                         # boosting, exactly, on five rows
    for r in rows[:5]:                                                                # each row
        phi, v0, v1 = shap_exact(fb, S["Tt"][r], bgT, TR); EX.append(phi)             # its exact values
        print(f"  boosting, test row {r}: sum of values {phi.sum():+.4f} = f(x) - background mean {v1 - v0:+.4f}; "
              f"largest: {FEAT[np.argmax(np.abs(phi))]} {phi[np.argmax(np.abs(phi))]:+.3f}")
    print(f"  (exact values for 5 rows: {time.time() - t0:.0f} s, 4096 coalitions x {BG} background rows each)")
    S["kerr"] = {}                                                                    # KernelSHAP's error at each budget
    for M in (25, 100, KERNEL_M):                                                     # coalitions sampled
        e = [np.abs(shap_kernel(fb, S["Tt"][r], bgT, TR, M=M, seed=s)[0] - EX[i]).max() for i, r in enumerate(rows[:5]) for s in range(2)]  # 5 rows x 2 seeds
        S["kerr"][M] = (np.mean(e), np.max(e))                                        # mean and worst
        print(f"  KernelSHAP with {2 * M:4d} coalitions: largest error against exact {np.max(e):.4f}, mean {np.mean(e):.4f} (5 rows x 2 seeds)")
    S["bgdiff"] = {}                                                                  # how much the background itself matters
    for k in (50, BG):                                                                # the first draft's size, and today's
        a, b = S["Xt"][pick(100 + k, k)], S["Xt"][pick(200 + k, k)]                   # two independent backgrounds of this size
        e = [np.abs(shap_exact(fb, S["Tt"][r], a, TR)[0] - shap_exact(fb, S["Tt"][r], b, TR)[0]).max() for r in rows[:5]]  # same rows, either background
        S["bgdiff"][k] = (np.mean(e), np.max(e))                                      # mean and worst
        print(f"  two backgrounds of {k:3d} rows: exact values differ by up to {np.max(e):.4f} (mean of the per-row largest {np.mean(e):.4f})")
    j = FEAT.index("capital-gain")                                                    # the feature the first draft tripped on
    for seed in (5, 7):                                                               # the first draft's background, and another
        bi = pick(seed, 50); P = np.array([shap_kernel(fb, S["Tt"][r], S["Xt"][bi], TR, M=100)[0] for r in rows]); imp = np.abs(P).mean(0)  # a quick global picture
        cg = S["tr"]["capital-gain"][bi].astype(float)                                # that background's capital gains
        print(f"  first draft, 50 background rows (seed {seed}): top three " + ", ".join(f"{FEAT[i]} {imp[i]:.3f}" for i in np.argsort(-imp)[:3])
              + f"; capital-gain {imp[j]:.3f}, rank {int(np.sum(imp > imp[j])) + 1}; the background has {int(np.sum(cg > 0))} rows with a capital gain, largest {cg.max():.0f}")
    t0 = time.time()                                                                  # the global picture, at today's settings
    PHI = np.array([shap_kernel(fb, S["Tt"][r], bgT, TR)[0] for r in rows]); S["PHI"] = PHI  # KernelSHAP on 200 test rows
    imp = np.abs(PHI).mean(0); order = np.argsort(-imp); S["imp"] = imp               # mean |value| per feature, largest first
    print(f"  mean |SHAP value| over {N_EXPLAIN} test rows ({time.time() - t0:.0f} s): "
          + ", ".join(f"{FEAT[j]} {imp[j]:.3f}" for j in order))
    b2 = S["Xt"][pick(8, BG)]; P2 = np.array([shap_kernel(fb, S["Tt"][r], b2, TR, M=100)[0] for r in rows]); i2 = np.abs(P2).mean(0)  # check: another background of the same size
    print(f"  check with a second background of {BG} rows: " + ", ".join(f"{FEAT[j]} {i2[j]:.3f}" for j in np.argsort(-i2))
          + f"; largest change {np.abs(i2 - imp).max():.3f}")
    sx = S["te"]["sex"][rows]                                                         # the explained rows' sex
    for c in ("sex", "relationship", "marital-status"):                               # sex and its two strongest proxies (Day 166)
        j = FEAT.index(c)                                                             # its position
        print(f"  mean SHAP value of {c:14s}: women {PHI[sx == 'Female', j].mean():+.3f} ({int(np.sum(sx == 'Female'))} rows), "
              f"men {PHI[sx == 'Male', j].mean():+.3f} ({int(np.sum(sx == 'Male'))} rows)")
    S["EX"] = EX                                                                      # kept for section 8
    print("Actual output from this lesson's run.")


def demo_lime(S):                                       # section 8
    print("\n" + "=" * 70)
    print("8. LIME FROM SCRATCH, AND HOW MUCH ITS ANSWER DEPENDS ON ITS SETTINGS")
    print("=" * 70)
    TR = np.arange(NF); r = S["rows"][0]; x = S["Tt"][r]; ex = S["EX"][0]             # the first explained row and its exact SHAP values
    top = np.argsort(-np.abs(ex))[:4]                                                 # its four largest features
    print(f"  test row {r}, exact SHAP: " + ", ".join(f"{FEAT[j]} {ex[j]:+.3f}" for j in top))
    S["lime_seeds"] = {}                                                              # for the figure
    for N in (LIME_N, 5 * LIME_N):                                                    # two sample sizes
        L = np.array([lime(S["fb"], x, S["bgT"], TR, N=N, seed=s) for s in range(5)]); S["lime_seeds"][N] = L  # five seeds
        print(f"  LIME, {N} samples, 5 seeds: " + "; ".join(f"{FEAT[j]} {L[:, j].min():+.3f} to {L[:, j].max():+.3f}" for j in top)
              + f"; largest spread across seeds {np.max(L.max(0) - L.min(0)):.3f} ({FEAT[np.argmax(L.max(0) - L.min(0))]})")
    S["lime_width"] = {}                                                              # for the figure
    for wd in (0.5, 1.0, 0.75 * np.sqrt(NF), 10.0):                                   # kernel widths, LIME's usual one third
        L = lime(S["fb"], x, S["bgT"], TR, N=5 * LIME_N, width=wd); S["lime_width"][wd] = L  # one explanation per width
        print(f"  width {wd:4.2f}: " + ", ".join(f"{FEAT[j]} {L[j]:+.3f}" for j in top) + f"; largest difference from SHAP {np.abs(L - ex).max():.3f}")
    L = lime(S["fl"], S["To"][r], S["bgO"], S["OH"], N=5 * LIME_N)                    # LIME on the linear model
    print(f"  on logistic regression (linear in these on/off features), LIME vs the closed-form SHAP: largest difference "
          f"{np.abs(L - shap_linear(S['w'], S['To'][r], S['bgO'], S['OH'])).max():.3f}")
    ov, cr = [], []                                                                   # agreement with SHAP over 20 rows
    for i, r in enumerate(S["rows"][:20]):                                            # each row
        L = lime(S["fb"], S["Tt"][r], S["bgT"], TR); P = S["PHI"][i]                  # LIME at its defaults; KernelSHAP's values
        ov.append(len(set(np.argsort(-np.abs(L))[:3]) & set(np.argsort(-np.abs(P))[:3])))  # shared features among each one's top three
        cr.append(np.corrcoef(L, P)[0, 1])                                            # how closely the 12 values agree
    print(f"  over 20 rows at the defaults: top-3 features shared with SHAP {np.mean(ov):.2f} of 3 on average "
          f"(all 3 in {int(np.sum(np.array(ov) == 3))} rows); correlation of the 12 values {np.mean(cr):.3f} on average, lowest {np.min(cr):.3f}")
    print("Actual output from this lesson's run.")


def demo_card(S):                                       # section 9
    print("\n" + "=" * 70)
    print("9. THE REPORT CARD")
    print("=" * 70)
    yt = S["te"]["y"]; p = S["pt"]["boosting"]; R, _ = S["fair"]; names, pv, rej = S["tests"]  # what the card summarises
    c = int(np.sum((p > THRESHOLD) == yt)); lo, hi = d163.wilson_ci(c, len(yt))       # accuracy and its interval
    print(f"  model             boosting, 200 trees of depth 3, threshold {THRESHOLD}, trained on {len(S['fi'])} rows")
    print(f"  performance       accuracy {c / len(yt):.4f} [{lo:.4f}, {hi:.4f}], AUC {fast_auc(yt, p):.4f}")
    print(f"  vs baseline       McNemar exact p {S['p_mcnemar']:.2e}: {'better' if rej['Holm'][0] else 'not shown better'} after Holm")
    print(f"  calibration       ECE {d165.ece(yt, p):.4f}")
    print(f"  fairness          selected women {R['Female']['sel']:.4f}, men {R['Male']['sel']:.4f}; TPR women {R['Female']['tpr']:.4f}, men {R['Male']['tpr']:.4f}")
    print(f"  tests             {int(rej['Holm'].sum())} of {len(pv)} significant after Holm: " + ", ".join(n for n, k in zip(names, rej["Holm"]) if k))
    j = np.argsort(-S["imp"])[:3]                                                     # the three largest features
    print("  explanations      largest mean |SHAP|: " + ", ".join(f"{FEAT[i]} {S['imp'][i]:.3f}" for i in j))
    print("Actual output from this lesson's run.")


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(S):                                    # six panels from what the sections kept
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # a 2 x 3 grid
    a = ax[0, 0]; o = np.argsort(S["imp"])                                            # global importance, smallest at the bottom
    a.barh(np.arange(NF), S["imp"][o], color="tab:blue"); a.set_yticks(np.arange(NF)); a.set_yticklabels([FEAT[i] for i in o], fontsize=9)  # bars
    a.set_xlabel("mean |SHAP value| (log-odds)"); a.set_title(f"boosting: global importance, {N_EXPLAIN} test rows")  # labels
    a = ax[0, 1]; Ms = sorted(S["kerr"])                                              # KernelSHAP against exact
    a.loglog([2 * m for m in Ms], [S["kerr"][m][1] for m in Ms], "o-", label="largest error")  # worst case
    a.loglog([2 * m for m in Ms], [S["kerr"][m][0] for m in Ms], "s--", label="mean error")  # typical case
    for k, ls in ((50, "-."), (BG, ":")):                                             # the background's own effect, for scale
        a.axhline(S["bgdiff"][k][0], color="k", ls=ls, lw=1, label=f"two backgrounds of {k} rows differ by (mean)")  # a horizontal line
    a.set_xlabel("coalitions sampled"); a.set_ylabel("difference in a SHAP value"); a.legend(fontsize=9)  # labels
    a.set_title("sampling error against the background's effect")                   # title
    a = ax[0, 2]; ex = S["EX"][0]; xs = np.arange(NF)                                 # LIME across seeds, one row
    a.bar(xs, ex, color="0.8", label="exact SHAP")                                    # the exact values
    for k, (N, L) in enumerate(S["lime_seeds"].items()):                              # each sample size
        for s in range(len(L)):                                                       # each seed
            a.plot(xs + (k - 0.5) * 0.3, L[s], "o" if k == 0 else "^", ms=4, color=("tab:red", "tab:green")[k],  # its values
                   label=f"LIME, {N} samples, 5 seeds" if s == 0 else None)               # a legend entry once per sample size
    a.set_xticks(xs); a.set_xticklabels(FEAT, rotation=60, ha="right", fontsize=8); a.legend(fontsize=8)  # labels
    a.set_title("one row: LIME's answer changes with the seed")                        # title
    a = ax[1, 0]                                                                      # LIME's kernel width
    for wd, L in S["lime_width"].items():                                             # each width
        a.plot(xs, L, "o-", ms=4, label=f"width {wd:.2f}")                            # its explanation
    a.bar(xs, ex, color="0.85", label="exact SHAP")                                   # against the exact values
    a.set_xticks(xs); a.set_xticklabels(FEAT, rotation=60, ha="right", fontsize=8); a.legend(fontsize=8)  # labels
    a.set_title("the same row: LIME's answer changes with the width")                 # title
    a = ax[1, 1]; n, mp, my = S["rel"]; ok = n > 0                                    # reliability on the test file
    a.plot([0, 1], [0, 1], "k:", lw=1); a.plot(mp[ok], my[ok], "o-", color="tab:blue")  # the diagonal and the curve
    a.set_xlabel("mean predicted probability"); a.set_ylabel("share that happened"); a.set_title("boosting: reliability on the test file")  # labels
    a = ax[1, 2]; R, Rv = S["fair"]; lab = ["selected", "TPR", "FPR", "precision"]; x = np.arange(4)  # fairness, test against validation
    for j, (g, c) in enumerate(zip(d166.GROUPS, ("tab:purple", "tab:olive"))):         # women, then men
        a.bar(x + (j - 0.5) * 0.38, [R[g][m] for m in d166.METRICS], 0.38, color=c, label=f"{g}, test")  # test rates
        a.scatter(x + (j - 0.5) * 0.38, [Rv[g][m] for m in d166.METRICS], marker="_", s=300, color="k", zorder=3,  # validation rates
                  label="validation" if j == 0 else None)                                 # one legend entry for the marks
    a.set_xticks(x); a.set_xticklabels(lab); a.set_ylim(0, 1); a.legend(fontsize=9); a.set_title("boosting by sex: test bars, validation marks")  # labels
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "report_card_results.png"), dpi=110); plt.close(fig)  # save
    print("saved report_card_results.png")


def draw_diagram():                                     # the week's six tools around one model, scored once
    fig, ax = plt.subplots(figsize=(15, 6.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 6.7); ax.axis("off")  # a blank canvas

    def box(x, y, w, h, text, color):                   # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="0.3"))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)  # its text

    def arrow(x0, y0, x1, y1):                          # an arrow between boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))  # the arrow
    box(0.2, 2.65, 2.6, 1.3, "the plan, fixed on\nvalidation rows:\nmodel, threshold, tests", "#e8f0fb")  # the plan
    box(3.8, 2.65, 2.4, 1.3, "test file,\nscored once", "#fdecea")                    # the one look
    arrow(2.85, 3.3, 3.75, 3.3)                                                       # plan to test
    tools = [("161 over-fitting and variance", "#fff4d6"), ("162 McNemar against the baseline", "#fff4d6"),  # the six tools
             ("163 intervals", "#fff4d6"), ("164 the card's tests, corrected", "#fff4d6"), ("165 calibration", "#fff4d6"),  # the rest of the week's tools
             ("166 fairness by sex", "#fff4d6"), ("167 SHAP (exact, kernel) and LIME", "#e6f4ea")]  # and today's explanations
    for i, (t, c) in enumerate(tools):                                                # each tool
        y = 5.55 - i * 0.85; box(7.2, y, 3.6, 0.6, t, c)                              # its box
        arrow(6.25, 3.3, 7.15, y + 0.3)                                               # from the test file
        arrow(10.85, y + 0.3, 11.75, 3.3 + (y + 0.3 - 3.3) * 0.25)                   # into the card, landing at its own height
    box(11.8, 2.65, 3.0, 1.3, "the report card", "#f2e6f7")                           # the card
    ax.text(7.5, 6.5, "Day 167: one model, seven checks, one look at the test file", ha="center", fontsize=12)  # title
    fig.savefig(os.path.join(HERE, "report_card_diagram.png"), dpi=110, bbox_inches="tight"); plt.close(fig)  # save
    print("saved report_card_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    S = train_models()                                                                # the data and both models
    demo_plan(S)                                                                      # 0
    demo_performance(S)                                                               # 1
    demo_baseline(S)                                                                  # 2
    demo_variance(S)                                                                  # 3
    demo_calibration(S)                                                               # 4
    demo_fairness(S)                                                                  # 5
    demo_tests(S)                                                                     # 6
    demo_shap(S)                                                                      # 7
    demo_lime(S)                                                                      # 8
    demo_card(S)                                                                      # 9
    print()
    plot_results(S); draw_diagram()                                                   # the figures
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Week 25: hyperparameter tuning and experimentation, starting with grid and random search")
    print("built from scratch on the same kind of held-out rows.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
