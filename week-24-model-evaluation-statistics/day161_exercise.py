"""
Day 161 Exercise -- Week 24's Data, a Logistic Regression, and Bias-Variance

The first of Week 24's parts for Day 167's report card.

    load_adult()                  the UCI Adult files, rows with missing values dropped
    split_rows(n)                 Week 24's split: 80% fitting rows, 20% validation rows (seed 0)
    tree_matrix(d, ref), onehot_matrix(d, ref)   features for trees, and for linear models
    fit_logistic(X, y, lam), predict_logistic(w, X)   logistic regression by Newton's method
    brier(y, p)                   mean squared error of a probability
    bias_variance(P, y)           Brier = (bias^2 + noise) + variance, from B models' predictions

Checked by hand: a tiny category coding, a logistic regression whose answer is
known, and a decomposition of two models' predictions worked on paper.
"""
import os                                              # os: file paths
import urllib.request                                  # urllib: fetch the data on first run

import numpy as np                                     # numpy: every array operation below is built on this

DATA = os.path.join(os.path.expanduser("~"), "week24_data")  # where the data is kept: outside the repository
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"  # UCI's copy of the Adult files
COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
        "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]  # the files' 15 columns
NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]   # numbers used as they are
CAT = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]  # categories
LAM = 1e-3                                              # logistic regression's ridge penalty (fixed, not tuned today)


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


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    d = {"age": np.array(["20", "40", "60", "30"]), "sex": np.array(["F", "M", "M", "F"]), "y": np.array([0, 1, 1, 0])}  # 4 tiny rows
    for c in ("education-num", "capital-gain", "capital-loss", "hours-per-week"):     # the other numbers (any varied values)
        d[c] = np.array(["1", "2", "3", "4"])                                        # 1 to 4, as text like the files
    for c in ("workclass", "marital-status", "occupation", "relationship", "race", "native-country"):  # the other categories, one value
        d[c] = np.array(["x"] * 4)                                                    # a single category
    T = tree_matrix(d, d)                                                             # category codes = share above $50K
    print(f"tree_matrix: sex coded by its rate: F -> {T[0, 10]:.1f}, M -> {T[1, 10]:.1f} (by hand: 0/2 and 2/2)")
    O = onehot_matrix(d, d)                                                           # standardised age, one column per sex
    print(f"onehot_matrix: age (20, 40, 60, 30) standardised -> {np.round(O[:, 0], 3).tolist()} "
          f"(by hand: mean 37.5, spread 14.79); sex columns F, M -> {O[:, -3:-1].astype(int).tolist()}")
    x = np.repeat([0.0, 1.0], 100)[:, None]; y = np.r_[np.zeros(80), np.ones(20), np.zeros(30), np.ones(70)]  # 20% and 70% above
    w, it = fit_logistic(x, y, lam=0.0)                                               # no penalty: the answer is known exactly
    print(f"fit_logistic on two groups with 20% and 70% positives: probabilities {predict_logistic(w, np.array([[0.0], [1.0]])).round(4).tolist()} "
          f"in {it} steps (by hand: 0.2 and 0.7, so intercept log(0.2/0.8) = {np.log(0.25):.4f}, got {w[0]:.4f})")
    P = np.array([[0.2, 0.9], [0.6, 0.5]]); yy = np.array([0.0, 1.0])                 # two models, two rows
    b, bn, v = bias_variance(P, yy)                                                   # the decomposition
    print("bias_variance, models (0.2, 0.9) and (0.6, 0.5), targets (0, 1): by hand Brier (0.04 + 0.01 + 0.36 + 0.25)/4 = 0.165; "
          "average model (0.4, 0.7) errs (0.16 + 0.09)/2 = 0.125; spread per row 0.08 and 0.08 (n - 1 = 1), so variance 0.08 x 1/2 = 0.04 "
          "and bias^2 + noise 0.125 - 0.08/2 = 0.085")
    print(f"  bias_variance: Brier {b:.3f} = {bn:.3f} + {v:.3f}")
    tr, te = load_adult(); fi, va = split_rows(len(tr["y"]))                          # the real data
    print(f"load_adult: {len(tr['y'])} training-file rows, {len(te['y'])} test-file rows; split_rows: {len(fi)} fitting, {len(va)} validation")
