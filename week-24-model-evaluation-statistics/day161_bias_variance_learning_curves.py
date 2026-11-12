"""
Day 161: The Bias-Variance Tradeoff Revisited, with Real Learning Curves

Week 24 is about judging models honestly. It starts with the oldest question:
is a model wrong because it is too simple (bias) or because it learned the
quirks of its training set (variance)? Both are measured here on the UCI Adult
census data (predict whether income is above $50K), for three models:

  logistic regression   fitted by Newton's method (new today)
  gradient boosting     Day 109's, imported from its exercise file
  one deep tree         Day 109's regression tree, grown until its leaves are pure

  1. the data, the split, and the models, with a convergence check
  2. learning curves: training and validation error as the training set grows
  3. bias and variance measured directly, from many disjoint training sets
  4. a complexity sweep (tree depth), and a first-draft error in the decomposition
  5. can a learning curve be extrapolated? fitted on small sizes, checked on the full set

Data: Ron Kohavi and Barry Becker (1996), UCI Adult, downloaded on first run
to ~/week24_data (outside the repository). Week 24's split, used all week: the official training file, minus
rows with missing values, is cut 80/20 into fitting and validation rows (seed
0); the official test file is not touched until later in the week.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: how long each step takes
import urllib.request                                  # urllib: fetch the data on first run

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, os.path.join(ROOT, "week-16-classical-ml-trees"))                 # Day 109's parts
sys.path.insert(0, os.path.join(ROOT, "week-17-classical-ml-kernels-ensembles"))     # Day 116's parts
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day109_exercise as d109                      # Day 109: regression trees and gradient boosting
    import day116_exercise as d116                      # Day 116: softmax regression (the first draft's optimiser)

DATA = os.path.join(os.path.expanduser("~"), "week24_data")  # where the data is kept: outside the repository
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"  # UCI's copy of the Adult files
COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
        "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]  # the files' 15 columns
NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]   # numbers used as they are
CAT = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]  # categories
LAM = 1e-3                                              # logistic regression's ridge penalty (fixed, not tuned today)
GB = dict(rounds=200, lr=0.1, depth=3)                  # gradient boosting's settings (fixed, not tuned today)
SIZES = (250, 500, 1000, 2000, 4000, 8000, 16000)       # training-set sizes for the learning curves (plus all fitting rows)
REPEATS = 5                                             # random subsets per size
BV_SIZES = (250, 500, 1000, 2000, 4000, 8000)           # sizes for the direct bias-variance measurement
DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20)      # tree depths for the complexity sweep


# =============================================================================
# THE PARTS (copied unchanged into day161_exercise.py, for Day 167's report card)
# =============================================================================
def load_adult():                                       # the two official files, rows with missing values dropped
    out = []                                                                          # training part, then test part
    os.makedirs(DATA, exist_ok=True)                                                  # the data folder, if it is missing
    for name in ("adult.data", "adult.test"):                                         # both files
        if not os.path.exists(os.path.join(DATA, name)):                              # first run: fetch it
            urllib.request.urlretrieve(ADULT_URL + name, os.path.join(DATA, name))    # 4.0 MB and 2.0 MB from UCI
        rows = [[x.strip() for x in ln.split(",")] for ln in open(os.path.join(DATA, name), encoding="utf-8")]  # split every line
        rows = [r for r in rows if len(r) == 15 and "?" not in r]                     # whole rows only (the test file starts with a note)
        d = {c: np.array([r[k] for r in rows]) for k, c in enumerate(COLS)}           # one array per column, as text
        d["y"] = np.array([r[14].rstrip(".") == ">50K" for r in rows], int)           # 1 if income above $50K (the test file adds a full stop)
        out.append(d)                                                                 # keep it
    return out                                                                        # (training dict, test dict)


def split_rows(n, frac=0.2, seed=0):                    # Week 24's split: (fitting rows, validation rows)
    perm = np.random.RandomState(seed).permutation(n); nv = int(round(frac * n))      # a shuffled order; validation size
    return np.sort(perm[nv:]), np.sort(perm[:nv])                                     # sorted, so rows keep their file order


def tree_matrix(d, ref):                                # one column per feature; categories coded by their rate in ref
    cols = [d[c].astype(float) for c in NUM]                                          # numbers as they are
    for c in CAT:                                                                     # each category column
        cats = np.unique(ref[c]); rate = {v: ref["y"][ref[c] == v].mean() for v in cats}  # share above $50K, per category, in ref
        cols.append(np.array([rate.get(v, ref["y"].mean()) for v in d[c]]))           # unseen categories get the overall rate
    return np.stack(cols, 1)                                                          # rows x 12 features


def onehot_matrix(d, ref):                              # standardised numbers and one column per category, for linear models
    cols = []                                                                         # blocks of columns
    for c in NUM:                                                                     # each number
        x, r = d[c].astype(float), ref[c].astype(float)                               # this data, and the reference data
        if c.startswith("capital"):                                                   # capital gains and losses are mostly 0 and very skewed
            x, r = np.log1p(x), np.log1p(r)                                           # so they are logged first
        cols.append(((x - r.mean()) / r.std())[:, None])                              # standardised with the reference's mean and spread
    for c in CAT:                                                                     # each category column
        cols.append((d[c][:, None] == np.unique(ref[c])[None, :]).astype(float))      # one 0/1 column per category seen in ref
    return np.hstack(cols)                                                            # rows x 87 columns


def fit_logistic(X, y, lam=LAM, iters=50):              # logistic regression by Newton's method; returns (weights, steps taken)
    A = np.c_[np.ones(len(X)), X]; w = np.zeros(A.shape[1])                           # a column of ones for the intercept; start at 0
    R = lam * np.eye(A.shape[1]); R[0, 0] = 0.0                                       # ridge penalty, not on the intercept
    for it in range(1, iters + 1):                                                    # Newton steps
        p = 1.0 / (1.0 + np.exp(-A @ w))                                              # current probabilities
        g = A.T @ (p - y) / len(y) + R @ w                                            # gradient of mean log loss + penalty
        H = (A * (p * (1 - p))[:, None]).T @ A / len(y) + R                           # its Hessian (curvature)
        step = np.linalg.solve(H, g); w = w - step                                    # Newton's step
        if np.abs(step).max() < 1e-9:                                                 # converged
            break                                                                     # stop early
    return w, it                                                                      # weights, and how many steps it took


def predict_logistic(w, X):                             # probabilities from fitted weights
    return 1.0 / (1.0 + np.exp(-(w[0] + X @ w[1:])))                                  # sigmoid of the linear score


def brier(y, p):                                        # mean squared error of a probability (the Brier score)
    return float(np.mean((p - y) ** 2))                                               # 0 is perfect; 0.25 is always saying 50%


def bias_variance(P, y):                                # P: B models' predictions (B x rows); returns (Brier, bias^2 + noise, variance)
    B = len(P); var = P.var(0, ddof=1)                                                # spread across models, per row (unbiased)
    bias_noise = np.mean((P.mean(0) - y) ** 2) - var.mean() / B                       # the average model's error, minus the variance it still carries
    return float(np.mean((P - y) ** 2)), float(bias_noise), float(var.mean() * (B - 1) / B)  # the three parts; Brier = the other two exactly


# =============================================================================
# THE THREE MODELS, AS fit(X, y) -> predict(X)
# =============================================================================
def logistic(X, y):                                     # logistic regression (one-hot features)
    w, _ = fit_logistic(X, y)                                                         # Newton
    return lambda Z: predict_logistic(w, Z)                                           # its predictor


def boosting(X, y):                                     # Day 109's gradient boosting with log loss (tree features)
    m, _ = d109.fit_boost(X, y, "log", GB["rounds"], GB["lr"], GB["depth"])           # 200 trees of depth 3
    return lambda Z: d109.sigmoid(d109.predict_boost(m, Z))                           # probabilities


def deep_tree(X, y, depth=64):                          # one regression tree on the 0/1 target: leaf value = share above $50K
    t = d109.grow_reg(X, y.astype(float), np.arange(len(y)), depth); d109.set_leaves(t, lambda i: y[i].mean())  # grown until pure
    return lambda Z: d109.predict_reg(t, Z)                                           # each row gets its leaf's share


# =============================================================================
# HELPERS
# =============================================================================
def setup():                                            # the data, Week 24's split, both feature matrices
    tr, te = load_adult(); fi, va = split_rows(len(tr["y"]))                          # official training file, cut 80/20
    ref = {k: v[fi] for k, v in tr.items()}                                           # codes and scalings come from fitting rows only
    return tr, te, fi, va, tree_matrix(tr, ref), onehot_matrix(tr, ref), tr["y"]      # everything the experiments need


def feats(name, Xt, Xo):                                # which feature matrix a model uses
    return Xo if name == "logistic regression" else Xt                                # linear: one-hot; trees: one column per feature


MODELS = {"logistic regression": logistic, "gradient boosting": boosting, "one deep tree": deep_tree}  # name -> fit


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_data(tr, te, fi, va, Xt, Xo, y):                                             # section 1
    print("\n" + "=" * 70)
    print("1. THE DATA, THE SPLIT, AND THE MODELS")
    print("=" * 70)
    print(f"UCI Adult: {len(tr['y'])} training-file rows and {len(te['y'])} test-file rows without missing values")
    print(f"share above $50K: training file {tr['y'].mean():.3f}, test file {te['y'].mean():.3f}")
    print(f"Week 24's split of the training file: {len(fi)} fitting rows, {len(va)} validation rows (seed 0)")
    print(f"features: {Xt.shape[1]} columns for trees, {Xo.shape[1]} one-hot columns for logistic regression")
    yf, yv = y[fi], y[va]; t0 = time.time()                                           # targets; a clock
    w, it = fit_logistic(Xo[fi], yf)                                                  # Newton's method
    A = np.c_[np.ones(len(fi)), Xo[fi]]; g = A.T @ (predict_logistic(w, Xo[fi]) - yf) / len(fi)  # gradient of the data part at the answer
    g[1:] += LAM * w[1:]                                                              # plus the penalty's
    pv = predict_logistic(w, Xo[va])                                                  # validation probabilities
    print(f"logistic regression, Newton: {it} steps, largest gradient left {np.abs(g).max():.1e}; validation log loss "
          f"{-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)):.4f}, error {np.mean((pv > 0.5) != yv):.4f} ({time.time() - t0:.1f} s)")
    for steps, lr in ((500, 0.5), (500, 0.1), (5000, 0.1)):                           # the first draft: Day 116's Adam optimiser
        W, b = d116.fit_softmax(Xo[fi], yf, 2, lam=LAM, steps=steps, lr=lr)           # two-class softmax = logistic regression
        q = np.clip(d116.softmax(Xo[va] @ W + b)[:, 1], 1e-12, 1 - 1e-12)             # validation probabilities
        print(f"  first draft, Day 116's Adam, {steps} steps at rate {lr}: validation log loss "
              f"{-np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q)):.4f}, error {np.mean((q > 0.5) != yv):.4f}")
    print(f"always predicting the fitting rows' share ({yf.mean():.3f}): validation Brier {brier(yv, np.full(len(va), yf.mean())):.4f}")
    print("Actual output from this lesson's run.")


def demo_curves(fi, va, Xt, Xo, y):                                                   # section 2
    print("\n" + "=" * 70)
    print("2. LEARNING CURVES: ERROR AS THE TRAINING SET GROWS")
    print("=" * 70)
    sizes = SIZES + (len(fi),); out = {}                                              # every size, ending with all fitting rows
    print(f"Brier score (mean squared error of the probability), mean of {REPEATS} random subsets "
          f"(1 at the full {len(fi)}); training / validation")
    for name, fit in MODELS.items():                                                  # each model
        X = feats(name, Xt, Xo); rows = []; t0 = time.time()                          # its features; results; a clock
        for n in sizes:                                                               # each training-set size
            reps = REPEATS if n < len(fi) else 1                                      # the full set can only be drawn once
            r = []                                                                    # results for this size
            for s in range(reps):                                                     # each random subset
                idx = np.sort(np.random.RandomState(1000 * s + n).choice(fi, n, replace=False))  # n fitting rows
                f = fit(X[idx], y[idx])                                               # train
                pt, pv = f(X[idx]), f(X[va])                                          # predict training and validation rows
                r.append((brier(y[idx], pt), brier(y[va], pv), np.mean((pt > 0.5) != y[idx]), np.mean((pv > 0.5) != y[va])))  # Brier and error, training and validation
            r = np.array(r); rows.append((n,) + tuple(r.mean(0)) + (r[:, 1].std(),))  # mean over subsets, and the validation spread
        out[name] = np.array(rows)                                                    # this model's curve
        print(f"  {name} ({time.time() - t0:.0f} s)")
        print("    " + "; ".join(f"{int(a)}: {b:.3f} / {c:.3f}" for a, b, c, *_ in rows))
    print("validation error rate at the full size: " + ", ".join(f"{k} {v[-1, 4]:.4f}" for k, v in out.items()))
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def disjoint_sets(fi, n, B, seed):                      # B training sets of n rows that share no rows
    p = np.random.RandomState(seed).permutation(fi)                                   # the fitting rows, shuffled
    return [np.sort(p[b * n:(b + 1) * n]) for b in range(B)]                          # consecutive slices


def demo_bv(fi, va, Xt, Xo, y):                                                       # section 3
    print("\n" + "=" * 70)
    print("3. BIAS AND VARIANCE, MEASURED FROM MANY DISJOINT TRAINING SETS")
    print("=" * 70)
    print("Brier on validation = (bias^2 + noise) + variance; B = disjoint training sets of n rows")
    out = {}                                                                          # for the figure
    for name, fit in MODELS.items():                                                  # each model
        X = feats(name, Xt, Xo); rows = []                                            # its features; results
        for n in BV_SIZES:                                                            # each size
            B = min(20, len(fi) // n)                                                 # as many disjoint sets as fit, up to 20
            P = np.array([fit(X[idx], y[idx])(X[va]) for idx in disjoint_sets(fi, n, B, n)])  # B models' validation predictions
            rows.append((n, B) + bias_variance(P, y[va]))                             # the decomposition
        out[name] = np.array(rows)                                                    # this model's table
        print(f"  {name}:")
        print("    " + "; ".join(f"n {int(n)} (B {int(B)}): {b:.4f} = {bn:.4f} + {v:.4f}" for n, B, b, bn, v in rows))
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_depth(fi, va, Xt, y):                                                        # section 4
    print("\n" + "=" * 70)
    print("4. A COMPLEXITY SWEEP: TREE DEPTH, AND A FIRST-DRAFT ERROR")
    print("=" * 70)
    n, B = 2000, 12                                                                   # 12 disjoint sets of 2000 rows
    sets = disjoint_sets(fi, n, B, 7); rows = []                                      # the same sets at every depth
    print(f"one tree, {B} disjoint training sets of {n} rows; validation Brier = (bias^2 + noise) + variance")
    for depth in DEPTHS:                                                              # each depth
        P = np.array([deep_tree(Xt[idx], y[idx], depth)(Xt[va]) for idx in sets])     # 12 trees' predictions
        naive = np.mean((P.mean(0) - y[va]) ** 2)                                     # the first draft's bias term: the average model's error
        b, bn, v = bias_variance(P, y[va])                                            # the corrected decomposition
        rows.append((depth, b, bn, v, naive))                                         # keep it
        print(f"  depth {depth:>2}: {b:.4f} = {bn:.4f} + {v:.4f}   first draft's bias^2 + noise: {naive:.4f}")
    rows = np.array(rows); k = int(np.argmin(rows[:, 1]))                             # the best depth on validation
    print(f"best depth {int(rows[k, 0])} (Brier {rows[k, 1]:.4f}); from depth 1 to 20 the corrected bias^2 + noise falls "
          f"{rows[0, 2]:.4f} -> {rows[-1, 2]:.4f}, the first draft's {rows[0, 4]:.4f} -> {rows[-1, 4]:.4f}")
    print("Actual output from this lesson's run.")
    return rows                                                                       # for the figure


def power_fit(n, e):                                    # e = a + b n^(-c), fitted by least squares over a grid of c
    best = None                                                                       # (sum of squares, a, b, c)
    for c in np.linspace(0.05, 1.5, 291):                                             # every exponent on the grid
        A = np.c_[np.ones(len(n)), n ** -c]; (a, b), *_ = np.linalg.lstsq(A, e, rcond=None)  # a and b for this c
        sse = float(np.sum((A @ np.array([a, b]) - e) ** 2))                          # how well it fits
        if best is None or sse < best[0]:                                             # the best so far
            best = (sse, a, b, c)                                                         # keep it
    return best[1:]                                                                   # (a, b, c)


def demo_extrapolate(curves):                                                         # section 5
    print("\n" + "=" * 70)
    print("5. CAN A LEARNING CURVE BE EXTRAPOLATED?")
    print("=" * 70)
    print("validation Brier fitted as a + b n^(-c) on the small sizes only, then predicted at the full size")
    out = {}                                                                          # for the figure
    for cut in (2000, 4000):                                                          # two cut-offs: is the answer sensitive to it?
        print(f"fitted on sizes up to {cut}:")
        for name, r in curves.items():                                                # each model's curve
            m = r[:, 0] <= cut; a, b, c = power_fit(r[m, 0], r[m, 2])                 # fit on the small sizes only
            pred = a + b * r[-1, 0] ** -c                                             # the prediction at the full size
            if cut == 4000:                                                           # the figure shows the 4000 fit
                out[name] = (a, b, c)                                                     # this model's fit
            print(f"  {name:>19}: a {a:.4f}, b {b:.3f}, c {c:.2f}; predicted at {int(r[-1, 0])}: {pred:.4f}, "
                  f"actual {r[-1, 2]:.4f} (error {pred - r[-1, 2]:+.4f})")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(curves, bv, depth, ext, base, path):                                 # six panels
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # two rows of three
    for q, (name, r) in enumerate(curves.items()):                                    # each model's learning curve
        ax[0, 0].plot(r[:, 0], r[:, 1], "--", color=f"C{q}", lw=1.2)                  # training Brier (dashed)
        ax[0, 0].errorbar(r[:, 0], r[:, 2], yerr=r[:, 5], fmt="o-", ms=3, color=f"C{q}", label=name)  # validation, with the spread
    ax[0, 0].axhline(base, color="k", ls=":", lw=1, label="always the base rate")    # the do-nothing model
    ax[0, 0].set_xscale("log"); ax[0, 0].set_ylim(-0.005, 0.25); ax[0, 0].legend(fontsize=7)  # log sizes
    ax[0, 0].set_xlabel("training rows"); ax[0, 0].set_ylabel("Brier score")          # labels
    ax[0, 0].set_title("learning curves: validation (solid), training (dashed)", fontsize=10)  # title
    for q, (name, r) in enumerate(bv.items()):                                        # bias and variance by size
        ax[0, 1].plot(r[:, 0], r[:, 3], "o-", ms=3, color=f"C{q}", label=f"{name}: bias^2 + noise")  # solid
        ax[0, 1].plot(r[:, 0], r[:, 4], "s--", ms=3, color=f"C{q}", label=f"{name}: variance")  # dashed
    ax[0, 1].set_xscale("log"); ax[0, 1].legend(fontsize=6.5); ax[0, 1].set_xlabel("training rows")  # labels
    ax[0, 1].minorticks_off(); ax[0, 1].set_xticks(BV_SIZES); ax[0, 1].set_xticklabels([str(s) for s in BV_SIZES])  # one plain label per size
    ax[0, 1].set_ylabel("share of validation Brier"); ax[0, 1].set_title("variance shrinks with data; logistic's bias does not", fontsize=10)  # labels and title
    d = depth                                                                         # the depth sweep
    ax[0, 2].plot(d[:, 0], d[:, 1], "o-", ms=3, color="k", label="validation Brier")  # total
    ax[0, 2].plot(d[:, 0], d[:, 2], "o-", ms=3, color="C0", label="bias^2 + noise")   # bias part
    ax[0, 2].plot(d[:, 0], d[:, 3], "o-", ms=3, color="C3", label="variance")         # variance part
    ax[0, 2].set_xlabel("tree depth"); ax[0, 2].legend(fontsize=7); ax[0, 2].set_title("one tree, 2000 rows: the U shape", fontsize=10)  # labels and title
    ax[1, 0].plot(d[:, 0], d[:, 4], "o--", ms=3, color="C1", label="first draft: error of the average model")  # naive
    ax[1, 0].plot(d[:, 0], d[:, 2], "o-", ms=3, color="C0", label="corrected for 12 models")  # corrected
    ax[1, 0].set_xlabel("tree depth"); ax[1, 0].set_ylabel("bias^2 + noise"); ax[1, 0].legend(fontsize=7)  # labels
    ax[1, 0].set_title("the first draft's bias rose with depth", fontsize=10)         # title
    for q, (name, r) in enumerate(curves.items()):                                    # error rate curves
        ax[1, 1].plot(r[:, 0], r[:, 4], "o-", ms=3, color=f"C{q}", label=name)        # validation error
        ax[1, 1].plot(r[:, 0], r[:, 3], "--", color=f"C{q}", lw=1.2)                  # training error
    ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("training rows"); ax[1, 1].set_ylabel("error rate (threshold 0.5)")  # labels
    ax[1, 1].legend(fontsize=7); ax[1, 1].set_title("the same curves as error rates", fontsize=10)  # title
    for q, (name, r) in enumerate(curves.items()):                                    # extrapolation
        a, b, c = ext[name]; nn = np.logspace(np.log10(250), np.log10(r[-1, 0]), 100)  # the fitted curve
        ax[1, 2].plot(nn, a + b * nn ** -c, "-", color=f"C{q}", lw=1, label=f"{name}: fitted on <= 4000")  # fit
        ax[1, 2].plot(r[:, 0], r[:, 2], "o", ms=4, color=f"C{q}", mfc="none")         # actual points
    ax[1, 2].set_xscale("log"); ax[1, 2].set_xlabel("training rows"); ax[1, 2].set_ylabel("validation Brier")  # labels
    ax[1, 2].legend(fontsize=7); ax[1, 2].set_title("extrapolating from small training sets", fontsize=10)  # title
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)                    # save to disk


def draw_diagram():                                     # how learning curves and the decomposition are measured
    fig, ax = plt.subplots(figsize=(15, 5.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.6); ax.axis("off")  # a wide, empty canvas
    p = 0.08                                                                          # the boxes' rounded padding

    def box(x, y, w, h, text, fc):                                                    # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={p}", fc=fc, ec="k", lw=1.2))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)   # its text

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))  # from (x0, y0) to (x1, y1)
    box(0.3, 2.3, 2.2, 1.0, "Adult training file\n24,130 fitting rows", "#e8f0fe")    # the data
    box(3.4, 3.8, 2.6, 1.0, "random subsets\nof n rows, n = 250...", "#e6f4ea")       # learning curves
    box(3.4, 0.8, 2.6, 1.0, "B disjoint sets\nof n rows each", "#fde8e8")              # decomposition
    box(6.9, 3.8, 2.4, 1.0, "fit, then score\ntraining and validation", "#e6f4ea")     # score
    box(6.9, 0.8, 2.4, 1.0, "B models predict the\nsame validation rows", "#fde8e8")   # predictions
    box(10.2, 3.8, 2.2, 1.0, "learning curve:\nerror against n", "#fff4d6")            # result 1
    box(10.2, 0.8, 2.2, 1.0, "spread across models\n= variance", "#fff4d6")            # result 2
    box(13.0, 0.65, 1.8, 1.3, "error of the\naverage model,\nminus variance / B\n= bias^2 + noise", "#fff4d6")  # result 3
    arrow(2.5 + p, 3.0, 3.4 - p, 4.2); arrow(2.5 + p, 2.6, 3.4 - p, 1.4)              # data into both
    arrow(6.0 + p, 4.3, 6.9 - p, 4.3); arrow(6.0 + p, 1.3, 6.9 - p, 1.3)              # sets into fitting
    arrow(9.3 + p, 4.3, 10.2 - p, 4.3); arrow(9.3 + p, 1.3, 10.2 - p, 1.3)            # into results
    arrow(12.4 + p, 1.3, 13.0 - p, 1.3)                                               # variance beside bias
    ax.set_title("Day 161: learning curves (top) and a direct bias-variance measurement (bottom)", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "bias_variance_diagram.png"), dpi=120); plt.close(fig)  # save to disk


def note_on_what_comes_next():  # a preview of Day 162
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 162: hypothesis testing for ML. A real randomised email experiment (Hillstrom's 64,000")
    print("customers) is analysed as an A/B test: two-proportion tests, permutation tests, an A/A check, power,")
    print("and McNemar's test for comparing two models on the same rows.")


def main():                                             # runs every section in order
    tr, te, fi, va, Xt, Xo, y = setup()                                               # data and features
    demo_data(tr, te, fi, va, Xt, Xo, y)                                              # 1. data and models
    curves = demo_curves(fi, va, Xt, Xo, y)                                           # 2. learning curves
    bv = demo_bv(fi, va, Xt, Xo, y)                                                   # 3. bias and variance by size
    depth = demo_depth(fi, va, Xt, y)                                                 # 4. the depth sweep
    ext = demo_extrapolate(curves)                                                    # 5. extrapolation
    base = brier(y[va], np.full(len(va), y[fi].mean()))                               # the do-nothing model's Brier
    plot_results(curves, bv, depth, ext, base, os.path.join(HERE, "bias_variance_results.png"))  # the results figure
    draw_diagram()                                                                    # the method
    print("\nsaved bias_variance_results.png\nsaved bias_variance_diagram.png")
    note_on_what_comes_next()                                                         # a preview


if __name__ == "__main__":  # only when run directly, not when imported
    main()                                                                            # run everything
