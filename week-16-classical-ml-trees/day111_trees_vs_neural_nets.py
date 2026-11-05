"""
Day 111: The Week 16 Build -- Trees Against Neural Networks on Tabular Data

The week's six exercise parts are copied in verbatim (below, one banner per
file) and assembled into four tree models: a pruned tree, a random forest,
gradient boosting from scratch and XGBoost. Against them stands a neural
network written today in NumPy: a multilayer perceptron (MLP) with ReLU, Adam,
weight decay and early stopping. With its hidden layers removed, the same code
is logistic or linear regression, which serves as the baseline.

The question is the one practitioners argue about: on tables of mixed
measurements, do trees really beat neural networks, and if so, why?

  1. Checks: the parts are verbatim, and the MLP's gradients are right.
  2. The head-to-head: Titanic, breast cancer, Diabetes and digits.
  3. Useless columns.
  4. Rotation: mixing the features together.
  5. Monotone distortions: squashing and stretching each feature.
  6. Training-set size.
  7. What each model relies on (Day 108's permutation importance).

Every line written for today carries its own comment, continuing Day 52's
convention; the copied parts keep their own.
"""
import ast                                             # ast: reading function sources out of the exercise files
import contextlib                                      # contextlib: redirect_stdout, to keep imports quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 99's bootstrap
import time                                            # time: training time per model

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import pandas as pd                                    # pandas: reading the Titanic CSV
import xgboost as xgb                                  # the library of Day 110's part
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits  # bundled datasets only
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, os.path.join(HERE, "..", "week-15-practical-nlp-pretrained"))  # Day 99's exercise
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day99_exercise as d99                        # Day 99: bootstrap_ci
TITANIC = os.path.join(HERE, "..", "week-03-intro-to-machine-learning", "titanic.csv")  # the repo's own copy
FEATS = ["Pclass", "is_female", "Age", "SibSp", "Parch", "Fare"]  # Day 105's Titanic features
SPLITS = {"Titanic": 20, "Breast cancer": 20, "Diabetes": 20, "Digits": 5}  # repeated random splits per dataset
PERTURB_SPLITS = 5                                      # splits in sections 3 to 6
NOISE = [0, 30, 100, 300]                               # section 3: useless columns added
SIZES = {"Breast cancer": [50, 100, 200, 398], "Digits": [50, 100, 200, 400, 1257]}  # section 6: training rows (the last is all of them)
GRID = [(32, 1e-4), (32, 0.1), (128, 1e-4), (128, 0.1)]  # the MLP's (hidden units, weight decay) choices
XGB = dict(eta=0.1, max_depth=3, seed=0, nthread=1, tree_method="hist")  # Day 110's simple settings


# =============================================================================
# from day105_exercise.py, verbatim: gini, entropy, best_split
# =============================================================================
def gini(counts):                                       # NEW: 1 - sum p_k^2
    n = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    return 1 - (p ** 2).sum(-1)


def entropy(counts):                                    # NEW: -sum p_k log2 p_k
    n = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):  # log2(0) is -inf, but it is multiplied by 0
        return np.where(p > 0, -p * np.log2(p), 0.0).sum(-1)  # each term is >= 0, so a pure group gives exactly 0


def best_split(X, y, n_classes, impurity, min_leaf=1):  # NEW: Day 105's sorted-cumulative-counts search, unchanged
    n, best = len(y), None
    parent = impurity(np.bincount(y, minlength=n_classes)[None])[0]
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="stable"); xs, ys = X[order, f], y[order]
        left = np.cumsum(np.eye(n_classes, dtype=int)[ys], axis=0)[:-1]
        right = np.bincount(y, minlength=n_classes) - left
        nl = np.arange(1, n)
        ok = (xs[1:] != xs[:-1]) & (nl >= min_leaf) & (n - nl >= min_leaf)
        if not ok.any():
            continue
        child = (nl * impurity(left) + (n - nl) * impurity(right)) / n
        gain = np.where(ok, parent - child, -np.inf)
        i = int(np.argmax(gain))
        if best is None or gain[i] > best[2] + 1e-12:
            best = (f, (xs[i] + xs[i + 1]) / 2, float(gain[i]))
    return best if best and best[2] > 1e-12 else None


# =============================================================================
# from day106_exercise.py, verbatim: node_errors, clone, branch, questions, cut, cost_path, at_alpha
# =============================================================================
def node_errors(node):                                  # NEW: training errors if this node were a leaf
    return int(node["counts"].sum() - node["counts"].max())


def clone(node):                                        # NEW: a copy, so cutting never touches the original
    c = dict(node)
    if "feature" in node:
        c["left"], c["right"] = clone(node["left"]), clone(node["right"])
    return c


def branch(node):                                       # NEW: (training errors, leaves) below this node
    if "feature" not in node:
        return node_errors(node), 1
    (ea, la), (eb, lb) = branch(node["left"]), branch(node["right"])
    return ea + eb, la + lb


def questions(node):                                    # NEW: every internal node, root first
    if "feature" not in node:
        return []
    return [node] + questions(node["left"]) + questions(node["right"])


def cut(node):                                          # NEW: turn a node into a leaf, in place
    for k in ("feature", "threshold", "gain", "left", "right"):
        node.pop(k, None)


def cost_path(tree):                                    # NEW: Day 106's weakest-link sequence, unchanged
    n, t = int(tree["counts"].sum()), clone(tree)
    alphas, trees = [0.0], [clone(t)]
    while "feature" in t:
        nodes = questions(t)
        g = []
        for q in nodes:
            e, leaves = branch(q)
            g.append((node_errors(q) - e) / n / (leaves - 1))  # errors saved per extra leaf
        a = min(g)
        for q, v in zip(nodes, g):
            if v <= a + 1e-12:
                cut(q)
        alphas.append(a); trees.append(clone(t))
    return alphas, trees


def at_alpha(alphas, trees, a):                         # NEW: the last tree whose alpha is <= a
    return trees[int(np.searchsorted(np.array(alphas), a + 1e-12, side="right")) - 1]


# =============================================================================
# from day107_exercise.py, verbatim: grow_random, proba, fit_forest, forest_proba, oob_accuracy
# =============================================================================
def grow_random(X, y, n_classes, mtry, rng, depth=0, max_depth=None, min_leaf=1):  # NEW: Day 107's tree, unchanged
    counts = np.bincount(y, minlength=n_classes)
    node = {"counts": counts, "pred": int(np.argmax(counts))}
    if (max_depth is not None and depth >= max_depth) or counts.max() == len(y):
        return node
    order = rng.permutation(X.shape[1])
    feats = order[:mtry]; s = best_split(X[:, feats], y, n_classes, gini, min_leaf); k = mtry
    while s is None and k < X.shape[1]:                   # nothing useful there: look at one more feature at a time
        feats = order[k:k + 1]; s = best_split(X[:, feats], y, n_classes, gini, min_leaf); k += 1
    if s is None:
        return node
    f, t = int(feats[s[0]]), s[1]; m = X[:, f] <= t
    node.update(feature=f, threshold=t, gain=s[2],
                left=grow_random(X[m], y[m], n_classes, mtry, rng, depth + 1, max_depth, min_leaf),
                right=grow_random(X[~m], y[~m], n_classes, mtry, rng, depth + 1, max_depth, min_leaf))
    return node


def proba(node, X, n_classes):                          # NEW: class probabilities, one tree
    out = np.zeros((len(X), n_classes))
    stack = [(node, np.arange(len(X)))]
    while stack:
        n, idx = stack.pop()
        if "feature" not in n:
            out[idx] = n["counts"] / n["counts"].sum()
            continue
        m = X[idx, n["feature"]] <= n["threshold"]
        stack.append((n["left"], idx[m])); stack.append((n["right"], idx[~m]))
    return out


def fit_forest(X, y, n_classes, n_trees, mtry, seed, max_depth=None, min_leaf=1):  # NEW: trees and their bootstraps
    rng = np.random.RandomState(seed); n = len(y); trees, bags = [], []
    for _ in range(n_trees):
        idx = rng.randint(0, n, n); bags.append(idx)
        trees.append(grow_random(X[idx], y[idx], n_classes, mtry, rng, max_depth=max_depth, min_leaf=min_leaf))
    return trees, bags


def forest_proba(trees, X, n_classes):                  # NEW: the average of every tree's probabilities
    return np.mean([proba(t, X, n_classes) for t in trees], axis=0)


def oob_accuracy(trees, bags, X, y, n_classes):         # NEW: (accuracy, share of rows that were left out at least once)
    votes = np.zeros((len(y), n_classes))
    for t, idx in zip(trees, bags):
        out = np.setdiff1d(np.arange(len(y)), idx)
        votes[out] += proba(t, X[out], n_classes)
    seen = votes.sum(1) > 0
    return float(np.mean(np.argmax(votes[seen], 1) == y[seen])), float(seen.mean())


# =============================================================================
# from day108_exercise.py, verbatim: tree_importance, impurity_importance, permutation_importance, ranks, spearman
# =============================================================================
def tree_importance(tree, p):                           # NEW: Day 108's impurity importance for one tree, unchanged
    out = np.zeros(p); stack = [tree]; n = tree["counts"].sum()
    while stack:
        node = stack.pop()
        if "feature" in node:
            out[node["feature"]] += node["counts"].sum() / n * node["gain"]
            stack += [node["left"], node["right"]]
    return out / max(out.sum(), 1e-12)


def impurity_importance(trees, p):                      # NEW: the forest's average
    return np.mean([tree_importance(t, p) for t in trees], axis=0)


def permutation_importance(score, X, y, seed, repeats=10):  # NEW: mean score lost, for any model
    rng = np.random.RandomState(seed); base = score(X, y)
    out = np.zeros((X.shape[1], repeats))
    for j in range(X.shape[1]):
        for r in range(repeats):
            Z = X.copy(); Z[:, j] = X[rng.permutation(len(X)), j]  # shuffle one column
            out[j, r] = base - score(Z, y)
    return out.mean(1)


def ranks(a):                                           # 1 for the smallest value
    r = np.empty(len(a)); r[np.argsort(a, kind="stable")] = np.arange(1, len(a) + 1)
    return r


def spearman(a, b):                                     # NEW: correlation of the rankings
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


# =============================================================================
# from day109_exercise.py, verbatim: reg_split, grow_reg, set_leaves, predict_reg, sigmoid, fit_boost, predict_boost
# =============================================================================
def reg_split(X, r, min_leaf=1):                        # NEW: Day 109's split search, unchanged
    n, best = len(r), None
    total = r.sum(); base = total ** 2 / n
    for f in range(X.shape[1]):
        o = np.argsort(X[:, f], kind="stable"); xs, rs = X[o, f], r[o]
        sl = np.cumsum(rs)[:-1]; nl = np.arange(1, n)
        ok = (xs[1:] != xs[:-1]) & (nl >= min_leaf) & (n - nl >= min_leaf)
        if not ok.any():
            continue
        gain = np.where(ok, sl ** 2 / nl + (total - sl) ** 2 / (n - nl) - base, -np.inf)
        i = int(np.argmax(gain))
        if best is None or gain[i] > best[2] + 1e-12:
            best = (f, (xs[i] + xs[i + 1]) / 2, float(gain[i]))
    return best if best and best[2] > 1e-12 else None


def grow_reg(X, r, idx, max_depth, min_leaf=1, depth=0):  # NEW: a tree that remembers each leaf's rows
    node = {"idx": idx}
    if depth >= max_depth or len(idx) < 2:
        return node
    s = reg_split(X[idx], r[idx], min_leaf)
    if s is None:
        return node
    m = X[idx, s[0]] <= s[1]
    node.update(feature=s[0], threshold=s[1], left=grow_reg(X, r, idx[m], max_depth, min_leaf, depth + 1),
                right=grow_reg(X, r, idx[~m], max_depth, min_leaf, depth + 1))
    return node


def set_leaves(node, value):                            # NEW: give every leaf value(its rows)
    if "feature" not in node:
        node["value"] = value(node["idx"]); return
    set_leaves(node["left"], value); set_leaves(node["right"], value)


def predict_reg(node, X):                               # NEW: follow many rows down at once
    out = np.zeros(len(X)); stack = [(node, np.arange(len(X)))]
    while stack:
        n, idx = stack.pop()
        if "feature" not in n:
            out[idx] = n["value"]
            continue
        m = X[idx, n["feature"]] <= n["threshold"]
        stack.append((n["left"], idx[m])); stack.append((n["right"], idx[~m]))
    return out


def sigmoid(F):                                         # log-odds to probability
    return 1 / (1 + np.exp(-F))


def fit_boost(X, y, loss, rounds, lr, max_depth, evals=(), min_leaf=1):  # NEW: Day 109's loop, unchanged
    y = y.astype(float); idx = np.arange(len(y))
    base = y.mean() if loss == "squared" else np.log(y.mean() / (1 - y.mean()))
    F = np.full(len(y), base); trees = []
    E = [np.zeros((rounds + 1, len(v))) + base for v in evals]
    for m in range(rounds):
        if loss == "squared":
            r = y - F
            leaf = lambda i: r[i].mean()
        else:
            p = sigmoid(F); r = y - p; h = p * (1 - p)
            leaf = lambda i: r[i].sum() / max(h[i].sum(), 1e-12)
        t = grow_reg(X, r, idx, max_depth, min_leaf); set_leaves(t, leaf); trees.append(t)
        F = F + lr * predict_reg(t, X)
        for e, v in zip(E, evals):
            e[m + 1] = e[m] + lr * predict_reg(t, v)
    return {"base": base, "lr": lr, "trees": trees, "train": F}, E


def predict_boost(model, X):                            # NEW: the base plus every tree's small step
    F = np.full(len(X), model["base"])
    for t in model["trees"]:
        F = F + model["lr"] * predict_reg(t, X)
    return F


# =============================================================================
# from day110_exercise.py, verbatim: newton_split, grow_newton, predict_lt, fit_newton, fit_xgb, margin
# =============================================================================
def newton_split(X, g, h, lam):                         # NEW: Day 110's gain, unchanged
    n, best = len(g), None
    G, H = g.sum(), h.sum(); parent = G ** 2 / (H + lam)
    for f in range(X.shape[1]):
        o = np.argsort(X[:, f], kind="stable"); xs = X[o, f]
        GL = np.cumsum(g[o])[:-1]; HL = np.cumsum(h[o])[:-1]
        ok = xs[1:] != xs[:-1]
        if not ok.any():
            continue
        gain = np.where(ok, 0.5 * (GL ** 2 / (HL + lam) + (G - GL) ** 2 / (H - HL + lam) - parent), -np.inf)
        i = int(np.argmax(gain))
        if best is None or gain[i] > best[2] + 1e-12:
            best = (f, (xs[i] + xs[i + 1]) / 2, float(gain[i]))
    return best if best and best[2] > 1e-12 else None


def grow_newton(X, g, h, idx, max_depth, lam, depth=0):  # NEW: every node knows its leaf value
    node = {"value": float(-g[idx].sum() / (h[idx].sum() + lam))}
    if depth >= max_depth or len(idx) < 2:
        return node
    s = newton_split(X[idx], g[idx], h[idx], lam)
    if s is None:
        return node
    m = X[idx, s[0]] <= s[1]
    node.update(feature=s[0], threshold=s[1], gain=s[2], left=grow_newton(X, g, h, idx[m], max_depth, lam, depth + 1),
                right=grow_newton(X, g, h, idx[~m], max_depth, lam, depth + 1))
    return node


def predict_lt(node, X):                                # NEW: XGBoost's convention
    out = np.zeros(len(X)); stack = [(node, np.arange(len(X)))]
    Z = X.astype(np.float32)
    while stack:
        n, idx = stack.pop()
        if "feature" not in n:
            out[idx] = n["value"]
            continue
        m = Z[idx, n["feature"]] < np.float32(n["threshold"])
        stack.append((n["left"], idx[m])); stack.append((n["right"], idx[~m]))
    return out


def fit_newton(X, y, loss, rounds, lr, max_depth, lam, Xe, base):  # NEW: Newton boosting
    y = y.astype(float); idx = np.arange(len(y)); F = np.full(len(y), base); Fe = np.full(len(Xe), base); trees = []
    for _ in range(rounds):
        if loss == "squared":
            g, h = F - y, np.ones(len(y))
        else:
            p = sigmoid(F); g, h = p - y, p * (1 - p)
        t = grow_newton(X, g, h, idx, max_depth, lam)
        F = F + lr * predict_lt(t, X); Fe = Fe + lr * predict_lt(t, Xe); trees.append(t)
    return F, Fe, trees


def fit_xgb(X, y, params, rounds, Xv=None, yv=None, stop=None):  # NEW: the library, with optional early stopping
    d = xgb.DMatrix(X, label=y)
    ev = [(xgb.DMatrix(Xv, label=yv), "val")] if Xv is not None else []
    return xgb.train(params, d, rounds, evals=ev, early_stopping_rounds=stop, verbose_eval=False)


def margin(model, X):                                   # NEW: the raw prediction, at the best round
    it = getattr(model, "best_iteration", None)
    return model.predict(xgb.DMatrix(X), output_margin=True, iteration_range=(0, it + 1) if it is not None else (0, 0))


# =============================================================================
# THE NEURAL NETWORK -- NEW
# =============================================================================
class MLP:                                              # NEW: ReLU layers, Adam with weight decay, early stopping
    def __init__(self, sizes, task, seed):
        rng = np.random.RandomState(seed)                 # its own random stream
        self.task = task                                  # "binary", "multi" or "regression"
        self.W = [rng.randn(a, b) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]  # He initialisation
        self.b = [np.zeros(b) for b in sizes[1:]]         # biases start at zero
        self.rng = rng                                    # used again for shuffling

    def forward(self, X):                               # the output layer's raw values; keeps every layer for backward
        self.A = [X]                                      # layer inputs, starting with the data
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = self.A[-1] @ W + b                          # a weighted sum per unit
            self.A.append(np.maximum(Z, 0) if i < len(self.W) - 1 else Z)  # ReLU, except on the last layer
        return self.A[-1]

    def loss_grad(self, Z, y):                          # NEW: the loss, and its gradient with respect to Z
        n = len(Z)
        if self.task == "binary":                         # log loss on a log-odds output
            return float(np.mean(np.logaddexp(0, Z[:, 0]) - y * Z[:, 0])), (0.5 * (1 + np.tanh(Z / 2)) - y[:, None]) / n  # tanh form: no overflow
        if self.task == "multi":                          # softmax cross-entropy
            Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)  # stable softmax
            Y = np.eye(Z.shape[1])[y]                       # one-hot labels
            return float(-np.mean(np.log(P[np.arange(n), y] + 1e-12))), (P - Y) / n
        return float(np.mean((Z[:, 0] - y) ** 2) / 2), (Z - y[:, None]) / n  # squared error, halved

    def grads(self, X, y):                              # NEW: backpropagation
        loss, dZ = self.loss_grad(self.forward(X), y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):          # from the last layer back to the first
            gW[i] = self.A[i].T @ dZ; gb[i] = dZ.sum(0)     # this layer's gradients
            if i:
                dZ = (dZ @ self.W[i].T) * (self.A[i] > 0)     # back through the ReLU
        return loss, gW, gb

    def fit(self, X, y, Xv, yv, lr=1e-3, wd=1e-4, epochs=400, batch=64, patience=30):  # NEW: Adam, early stopping
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8     # standardise with training statistics only
        if self.task == "regression":
            self.ym, self.ys = y.mean(), y.std()            # the target is standardised too
            y, yv = (y - self.ym) / self.ys, (yv - self.ym) / self.ys
        X, Xv = (X - self.mu) / self.sd, (Xv - self.mu) / self.sd
        P = self.W + self.b; M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]; t = 0  # Adam's state
        best, best_p, wait = np.inf, None, 0
        for ep in range(epochs):
            order = self.rng.permutation(len(X))            # a new order every epoch
            for s in range(0, len(X), batch):
                i = order[s:s + batch]; _, gW, gb = self.grads(X[i], y[i]); t += 1
                for k, (p, g) in enumerate(zip(P, gW + gb)):
                    M[k] = 0.9 * M[k] + 0.1 * g; V[k] = 0.999 * V[k] + 0.001 * g * g  # Day 37's Adam averages
                    step = (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
                    p -= lr * (step + (wd * p if k < len(self.W) else 0))  # weight decay on weights, not biases
            v = self.loss_grad(self.forward(Xv), yv)[0]     # validation loss after this epoch
            if v < best - 1e-6:
                best, best_p, wait = v, [p.copy() for p in P], 0  # remember the best weights
            else:
                wait += 1
                if wait >= patience:                          # Day 38's early stopping
                    break
        for p, q in zip(P, best_p):
            p[...] = q                                        # restore the best weights, in place
        self.epochs, self.best = ep + 1, best
        return self

    def predict(self, X):                               # probabilities for classification, values for regression
        Z = self.forward((X - self.mu) / self.sd)
        if self.task == "binary":
            return 0.5 * (1 + np.tanh(Z[:, 0] / 2))         # the sigmoid, in a form that cannot overflow
        if self.task == "multi":
            Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)
        return Z[:, 0] * self.ys + self.ym


# =============================================================================
# THE MODELS, EACH AS fit(Xtr, ytr, seed) -> predict(X) -- NEW
# =============================================================================
class InOrder:                                          # a stand-in generator: features in their usual order (Day 107)
    def permutation(self, n):
        return np.arange(n)


def task_of(y):                                         # the problem type, from the labels
    if y.dtype.kind == "f":
        return "regression"
    return "binary" if y.max() == 1 else "multi"


def carve(n, seed, frac=0.25):                          # a validation part from the training rows
    o = np.random.RandomState(seed).permutation(n); v = int(n * frac)
    return o[v:], o[:v]


def fit_mlp(X, y, seed, hidden=True):                   # NEW: the MLP (or, without hidden layers, a linear model), tuned on validation
    task = task_of(y); out = int(y.max()) + 1 if task == "multi" else 1
    a, v = carve(len(y), seed); best = None
    for h, wd in (GRID if hidden else [(0, 1e-4), (0, 0.1)]):
        sizes = [X.shape[1], h, h, out] if hidden else [X.shape[1], out]  # two hidden layers, or none
        m = MLP(sizes, task, seed).fit(X[a], y[a], X[v], y[v], wd=wd)
        if best is None or m.best < best.best:
            best = m                                          # the lowest validation loss wins
    return best.predict


def fit_pruned(X, y, seed):                             # Days 105-106: a full tree, pruned by 5-fold CV over alpha
    k = int(y.max()) + 1; p = X.shape[1]
    full = grow_random(X, y, k, p, InOrder())             # every feature, in order: Day 105's tree
    alphas, trees = cost_path(full)
    a = np.array(alphas); b = np.append(np.sqrt(np.maximum(a[:-1], 1e-12) * a[1:]), a[-1])  # Day 106's candidates
    folds = np.array_split(np.random.RandomState(seed).permutation(len(y)), 5); S = np.zeros(len(b))
    for i in range(5):
        va = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        al, ts = cost_path(grow_random(X[tr], y[tr], k, p, InOrder()))
        S += [np.mean(np.argmax(proba(at_alpha(al, ts, c), X[va], k), 1) == y[va]) for c in b]
    t = at_alpha(alphas, trees, b[int(np.argmax(S))])
    return lambda Z: proba(t, Z, k)


def fit_forest_oob(X, y, seed):                         # Day 107: 100 trees, leaf size 1 or 5 chosen by OOB
    k = int(y.max()) + 1; m = max(1, int(np.sqrt(X.shape[1]))); best = None
    for leaf in (1, 5):
        trees, bags = fit_forest(X, y, k, 100, m, seed, min_leaf=leaf)
        o = oob_accuracy(trees, bags, X, y, k)[0]
        if best is None or o > best[0]:
            best = (o, trees)
    return lambda Z: forest_proba(best[1], Z, k)


def logloss(F, y):                                      # Day 109's stopping score, from log-odds
    return float(np.mean(np.logaddexp(0, -F * (2 * y - 1))))


def fit_boost_es(X, y, seed):                           # Day 109: boosting from scratch, stopped on validation
    loss = "squared" if task_of(y) == "regression" else "logistic"
    a, v = carve(len(y), seed)
    m, E = fit_boost(X[a], y[a], loss, 500, 0.1, 3, evals=(X[v],))
    score = [np.mean((f - y[v]) ** 2) if loss == "squared" else logloss(f, y[v].astype(float)) for f in E[0]]
    m["trees"] = m["trees"][:int(np.argmin(score))]       # keep the best round
    if loss == "squared":
        return lambda Z: predict_boost(m, Z)
    return lambda Z: sigmoid(predict_boost(m, Z))


def fit_xgb_es(X, y, seed):                             # Day 110: XGBoost, simple settings, early stopping
    task = task_of(y); a, v = carve(len(y), seed)
    obj = {"binary": dict(objective="binary:logistic"), "multi": dict(objective="multi:softprob", num_class=int(y.max()) + 1),
           "regression": dict(objective="reg:squarederror")}[task]
    m = fit_xgb(X[a], y[a], dict(XGB, **obj), 1000, X[v], y[v], 30)
    if task == "binary":
        return lambda Z: sigmoid(margin(m, Z))
    if task == "multi":
        return lambda Z: m.predict(xgb.DMatrix(Z), iteration_range=(0, m.best_iteration + 1))
    return lambda Z: margin(m, Z)


MODELS = {"pruned tree": fit_pruned, "random forest": fit_forest_oob, "boosting (scratch)": fit_boost_es, "XGBoost": fit_xgb_es,
          "linear model": lambda X, y, s: fit_mlp(X, y, s, hidden=False), "MLP": fit_mlp}  # name -> fit function
TREES = ["pruned tree", "random forest", "boosting (scratch)", "XGBoost"]  # the tree side


def score(pred, y):                                     # accuracy for classification, RMSE for regression
    if task_of(y) == "regression":
        return float(np.sqrt(np.mean((pred - y) ** 2)))
    return float(np.mean((pred > 0.5).astype(int) == y)) if pred.ndim == 1 else float(np.mean(np.argmax(pred, 1) == y))


def usable(name, y):                                    # which models each problem allows
    if name in ("pruned tree", "random forest"):
        return task_of(y) != "regression"                 # Days 105-107's trees classify
    if name == "boosting (scratch)":
        return task_of(y) != "multi"                      # Day 109's version is binary or regression
    return True


def split(y, seed):                                     # a stratified split for classes, a random one for regression
    rng = np.random.RandomState(seed)
    if task_of(y) == "regression":
        o = rng.permutation(len(y)); k = int(round(len(y) * 0.3)); return np.sort(o[k:]), np.sort(o[:k])
    tr, te = [], []
    for c in np.unique(y):                                # Day 105's stratified split
        idx = rng.permutation(np.where(y == c)[0]); k = int(round(len(idx) * 0.3)); te += list(idx[:k]); tr += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


def load_titanic(extra=False):                          # Day 106's Titanic; with extra=True, Day 108's two useless columns
    d = pd.read_csv(TITANIC)
    d["is_female"] = (d["Sex"] == "female").astype(float)
    d["Age"] = d["Age"].fillna(d["Age"].median())         # 177 missing ages: the median (a simplification)
    X, y = d[FEATS].to_numpy(float), d["Survived"].to_numpy()
    if extra:
        coin = np.random.RandomState(0).randint(0, 2, len(y)).astype(float)
        return np.column_stack([X, d["PassengerId"].to_numpy(float), coin]), y, FEATS + ["PassengerId", "coin"]
    return X, y


def datasets():                                         # the four problems
    bc, db, dg = load_breast_cancer(), load_diabetes(scaled=False), load_digits()
    return {"Titanic": load_titanic(), "Breast cancer": (bc.data, bc.target), "Diabetes": (db.data, db.target.astype(float)),
            "Digits": (dg.data, dg.target)}


# =============================================================================
# DEMOS
# =============================================================================
def demo_checks():
    print("=" * 70)
    print("1. CHECKS -- the parts are verbatim, and the MLP's gradients are right")
    print("=" * 70)
    me = open(os.path.abspath(__file__), encoding="utf-8").read(); found = total = 0
    for d in range(105, 111):                             # every function of every exercise file
        src = open(os.path.join(HERE, f"day{d}_exercise.py"), encoding="utf-8").read()
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name != "node":  # 'node' is Day 106's helper for its hand check
                total += 1; found += "\n".join(src.split("\n")[node.lineno - 1:node.end_lineno]) in me
    print(f"functions from day105_exercise.py to day110_exercise.py found here, character for character: {found} of {total}")
    rng = np.random.RandomState(0); X = rng.normal(size=(7, 4)); eps = 1e-6
    for task, y, out in (("binary", rng.randint(0, 2, 7), 1), ("multi", rng.randint(0, 3, 7), 3), ("regression", rng.normal(size=7), 1)):
        m = MLP([4, 5, 5, out], task, 0); _, gW, _ = m.grads(X, y); worst = 0.0
        for i in range(len(m.W)):
            for j in np.ndindex(*m.W[i].shape):             # every weight, nudged both ways
                m.W[i][j] += eps; up = m.loss_grad(m.forward(X), y)[0]; m.W[i][j] -= 2 * eps
                down = m.loss_grad(m.forward(X), y)[0]; m.W[i][j] += eps
                worst = max(worst, abs((up - down) / (2 * eps) - gW[i][j]))
        print(f"MLP gradient check, {task:<10}: largest gap between backpropagation and finite differences {worst:.1e}")
    print("Actual output from this lesson's run.")


def demo_head_to_head(sets):
    print("\n" + "=" * 70)
    print("2. THE HEAD-TO-HEAD -- six models, four problems")
    print("=" * 70)
    print("each model tunes itself on its own training part only; the test part is used once; accuracy, or RMSE for Diabetes (lower is better)")
    print("MLP: two hidden layers, 4 settings chosen on validation; linear model: the same code without hidden layers")
    res, secs = {}, {}
    for name, (X, y) in sets.items():
        R = {m: [] for m in MODELS if usable(m, y)}; T = {m: [] for m in R}
        for s in range(SPLITS[name]):
            tr, te = split(y, s)
            for m in R:
                t0 = time.perf_counter(); f = MODELS[m](X[tr], y[tr], s); T[m].append(time.perf_counter() - t0)
                R[m].append(score(f(X[te]), y[te]))
        res[name] = {m: np.array(v) for m, v in R.items()}; secs[name] = {m: float(np.median(v)) for m, v in T.items()}
        reg = task_of(y) == "regression"
        print(f"\n{name}: {len(y)} rows, {X.shape[1]} features, {SPLITS[name]} splits")
        print(f"  {'model':<20}{'RMSE' if reg else 'accuracy':>10}{'std':>8}{'seconds':>9}")
        for m, v in res[name].items():
            print(f"  {m:<20}{v.mean():>10.4f}{v.std():>8.4f}{secs[name][m]:>9.2f}")
        trees = [m for m in TREES if m in R]
        best = (min if reg else max)(trees, key=lambda m: res[name][m].mean())  # the best tree model here
        d = res[name]["MLP"] - res[name][best]; lo, hi = d99.bootstrap_ci(d)
        better = int((d < 0).sum() if reg else (d > 0).sum()); worse = int((d > 0).sum() if reg else (d < 0).sum())
        print(f"  MLP minus the best tree model ({best}): {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; MLP better on {better}, worse on {worse} of {SPLITS[name]}")
    print("the splits share rows, so these intervals are too narrow, a rough guide only; seconds are medians per fit, within this run")
    print("Actual output from this lesson's run.")
    return res                                               # the plot uses it


def perturbed(make, X, y, label, models=("random forest", "XGBoost", "MLP")):  # NEW: the same models on a changed table
    out = {}
    for m in models:
        v = []
        for s in range(PERTURB_SPLITS):
            tr, te = split(y, s); A, B = make(X[tr], X[te], s)  # the change is fitted on training rows where it needs fitting
            v.append(score(MODELS[m](A, y[tr], s)(B), y[te]))
        out[m] = float(np.mean(v))
    print(f"  {label:<34}" + "".join(f"{out[m]:>15.4f}" for m in models))
    return out


def demo_useless(X, y):
    print("\n" + "=" * 70)
    print("3. USELESS COLUMNS -- breast cancer with random columns added")
    print("=" * 70)
    print(f"{PERTURB_SPLITS} splits; the columns are random numbers with no relation to the label")
    print(f"  {'':<34}{'random forest':>15}{'XGBoost':>15}{'MLP':>15}")
    rows = []
    for k in NOISE:
        add = lambda A, B, s, k=k: (np.hstack([A, np.random.RandomState(1000 + s).normal(size=(len(A), k))]),
                                    np.hstack([B, np.random.RandomState(2000 + s).normal(size=(len(B), k))]))
        rows.append((k, perturbed(add, X, y, f"{k} useless columns ({X.shape[1] + k} in all)")))
    print("Actual output from this lesson's run.")
    return rows                                              # the plot uses them


def rotation(p, seed):                                  # a random orthogonal matrix
    q, r = np.linalg.qr(np.random.RandomState(seed).normal(size=(p, p)))
    return q * np.sign(np.diag(r))                        # sign fix, so the result is uniformly random


def demo_rotation(sets):
    print("\n" + "=" * 70)
    print("4. ROTATION -- every new feature a mix of all the old ones")
    print("=" * 70)
    print(f"{PERTURB_SPLITS} splits; features standardised with training statistics, then rotated by a random orthogonal matrix")
    print("(the same information, so a model that does not care about the axes should not change)")
    print(f"  {'':<34}{'random forest':>15}{'XGBoost':>15}{'MLP':>15}")
    out = {}
    for name in ("Breast cancer", "Digits"):
        X, y = sets[name]
        def std(A, B, s):                                   # standardised only
            mu, sd = A.mean(0), A.std(0) + 1e-8; return (A - mu) / sd, (B - mu) / sd
        def rot(A, B, s):                                   # standardised, then rotated
            A, B = std(A, B, s); Q = rotation(A.shape[1], s); return A @ Q, B @ Q
        out[name] = (perturbed(std, X, y, f"{name}: original axes"), perturbed(rot, X, y, f"{name}: rotated"))
    print("Actual output from this lesson's run.")
    return out                                               # the plot uses it


def demo_monotone(X, y):
    print("\n" + "=" * 70)
    print("5. MONOTONE DISTORTIONS -- squash or stretch each feature without changing its order")
    print("=" * 70)
    tr, te = split(y, 0); k = 2; p = X.shape[1]
    z = lambda A: (A - X[tr].mean(0)) / (X[tr].std(0) + 1e-8)  # a fixed standardisation, the same for both parts
    warp = lambda A: np.exp(3 * z(A))                     # strictly increasing: every order is kept, distances are not
    a = proba(grow_random(X[tr], y[tr], k, p, InOrder()), X[te], k)
    b = proba(grow_random(warp(X[tr]), y[tr], k, p, InOrder()), warp(X[te]), k)
    c = proba(grow_random(X[tr], y[tr], k, p, InOrder()), X[tr], k)
    d = proba(grow_random(warp(X[tr]), y[tr], k, p, InOrder()), warp(X[tr]), k)
    print(f"breast cancer, one split: Day 105's tree on raw and on exp(3z) features gives the same probabilities on {np.mean(np.all(c == d, 1)):.4f} of TRAINING rows")
    print(f"  and on {np.mean(np.all(a == b, 1)):.4f} of test rows: thresholds sit halfway between neighbouring training values, and halfway is not kept by exp,")
    print("  so a test row that falls between two training values can land on the other side")
    print(f"{PERTURB_SPLITS} splits:")
    print(f"  {'':<34}{'random forest':>15}{'XGBoost':>15}{'MLP':>15}")
    same = lambda A, B, s: (A, B)
    w = lambda A, B, s: (np.exp(3 * (A - A.mean(0)) / (A.std(0) + 1e-8)), np.exp(3 * (B - A.mean(0)) / (A.std(0) + 1e-8)))
    r = lambda A, B, s: (np.argsort(np.argsort(A, 0), 0) / len(A), np.array([np.searchsorted(np.sort(A[:, j]), B[:, j]) for j in range(A.shape[1])]).T / len(A))
    out = [perturbed(same, X, y, "raw features"), perturbed(w, X, y, "exp(3z): long right tails"), perturbed(r, X, y, "ranks: every feature uniform")]
    print("Actual output from this lesson's run.")
    return out


def demo_size(sets):
    print("\n" + "=" * 70)
    print("6. TRAINING-SET SIZE -- who needs more rows?")
    print("=" * 70)
    print(f"{PERTURB_SPLITS} splits; the test part is the usual 30%; training rows drawn from the rest")
    out = {}
    for name in ("Breast cancer", "Digits"):
        X, y = sets[name]; print(f"\n{name}:")
        print(f"  {'training rows':<16}" + "".join(f"{m:>15}" for m in ("random forest", "XGBoost", "MLP", "linear model")))
        rows = []
        for n in SIZES[name]:
            v = {m: [] for m in ("random forest", "XGBoost", "MLP", "linear model")}
            for s in range(PERTURB_SPLITS):
                tr, te = split(y, s); tr = np.random.RandomState(s).choice(tr, n, replace=False)  # n training rows
                for m in v:
                    v[m].append(score(MODELS[m](X[tr], y[tr], s)(X[te]), y[te]))
            rows.append((n, {m: float(np.mean(a)) for m, a in v.items()}))
            print(f"  {n:<16}" + "".join(f"{rows[-1][1][m]:>15.4f}" for m in v))
        out[name] = rows
    print("Actual output from this lesson's run.")
    return out                                               # the plot uses it


def demo_reliance():
    print("\n" + "=" * 70)
    print("7. WHAT EACH MODEL RELIES ON -- Day 108's permutation importance, on held-out rows")
    print("=" * 70)
    X, y, names = load_titanic(extra=True)
    print(f"Titanic with Day 108's PassengerId and coin columns; {PERTURB_SPLITS} splits; accuracy lost when a column is shuffled")
    models = ("random forest", "XGBoost", "MLP"); imp = {m: [] for m in models}
    for s in range(PERTURB_SPLITS):
        tr, te = split(y, s)
        for m in models:
            f = MODELS[m](X[tr], y[tr], s)
            imp[m].append(permutation_importance(lambda A, b, f=f: score(f(A), b), X[te], y[te], s))  # Day 108's part
    imp = {m: np.mean(v, axis=0) for m, v in imp.items()}
    print(f"\n  {'feature':<14}" + "".join(f"{m:>15}" for m in models))
    for j in np.argsort(-imp["XGBoost"]):
        print(f"  {names[j]:<14}" + "".join(f"{imp[m][j]:>15.4f}" for m in models))
    print(f"rank correlation (Day 108's spearman): forest-XGBoost {spearman(imp['random forest'], imp['XGBoost']):.3f}, "
          f"forest-MLP {spearman(imp['random forest'], imp['MLP']):.3f}, XGBoost-MLP {spearman(imp['XGBoost'], imp['MLP']):.3f}")
    print("Actual output from this lesson's run.")


def plot_results(res, useless, rot, size):              # four panels
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.2))             # canvas
    a = ax[0, 0]; names = [n for n in res if n != "Diabetes"]; models = list(MODELS); w = 0.13; x = np.arange(len(names))
    for i, m in enumerate(models):
        v = [res[n][m].mean() if m in res[n] else 0 for n in names]
        a.bar(x + (i - 2.5) * w, v, w, label=m)
    a.set_xticks(x); a.set_xticklabels(names); a.set_ylim(0.75, 1.0); a.set_ylabel("mean test accuracy"); a.grid(alpha=0.3, axis="y")
    a.legend(fontsize=7.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.07)); a.set_title("The head-to-head (Diabetes, a regression, is in the table)", fontsize=10)
    a = ax[0, 1]
    for m in ("random forest", "XGBoost", "MLP"):
        a.plot([r[0] for r in useless], [r[1][m] for r in useless], marker="o", label=m)
    a.set_xscale("symlog", linthresh=10); a.set_xlabel("useless columns added"); a.set_ylabel("test accuracy"); a.grid(alpha=0.3)
    a.set_title("Breast cancer with useless columns", fontsize=10); a.legend(fontsize=8)
    a = ax[1, 0]; lab = []; vals = {m: [] for m in ("random forest", "XGBoost", "MLP")}
    for n, (o, r) in rot.items():
        lab += [f"{n}\noriginal", f"{n}\nrotated"]
        for m in vals:
            vals[m] += [o[m], r[m]]
    x = np.arange(len(lab))
    for i, m in enumerate(vals):
        a.bar(x + (i - 1) * 0.26, vals[m], 0.26, label=m)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=8); a.set_ylim(0.75, 1.0); a.set_ylabel("test accuracy"); a.grid(alpha=0.3, axis="y")
    a.set_title("Rotating the features", fontsize=10); a.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    a = ax[1, 1]
    for n, ls in (("Breast cancer", "-"), ("Digits", "--")):
        for m, c in (("random forest", "#1f77b4"), ("XGBoost", "#ff7f0e"), ("MLP", "#2ca02c")):
            a.plot([r[0] for r in size[n]], [r[1][m] for r in size[n]], marker="o", linestyle=ls, color=c, label=f"{n}: {m}")
    a.set_xscale("log", base=2); a.set_xlabel("training rows (log scale)"); a.set_ylabel("test accuracy"); a.grid(alpha=0.3)
    a.set_title("Training-set size", fontsize=10); a.legend(fontsize=7.5)
    plt.tight_layout(h_pad=2.5); plt.savefig("trees_vs_nets_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved trees_vs_nets_results.png")


def draw_diagram():                                     # the week's parts and the build
    fig, ax = plt.subplots(figsize=(13, 4.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.3):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.35, "Day 111: the week's parts, assembled", ha="center", fontsize=12, fontweight="bold")
    parts = ["105: impurity,\nbest split", "106: pruning\npath", "107: forest,\nOOB", "108: permutation\nimportance", "109: boosting,\nfrom scratch", "110: Newton\nboosting, XGBoost"]
    for i, t in enumerate(parts):
        box(1.1 + i * 2.16, 3.35, 1.95, 0.85, t, "#FAEEDA", fs=7.8)
    box(3.3, 1.8, 4.4, 0.95, "tree models: pruned tree, forest,\nboosting from scratch, XGBoost", "#DCEBF7")
    box(9.7, 1.8, 4.4, 0.95, "new today: an MLP in NumPy\n(and, without hidden layers, a linear model)", "#E5F2E1")
    box(6.5, 0.45, 6.2, 0.6, "the same splits, the same test rows, the same rules for tuning", "#EEEDFE")
    for i in range(6):
        x = 1.1 + i * 2.16
        if i == 3:
            ax.annotate("", xy=(8.1, 2.29), xytext=(x, 2.91), arrowprops=dict(arrowstyle="->", color="#333333"))  # importance works for any model
        else:
            ax.annotate("", xy=(min(max(x, 1.4), 5.2), 2.29), xytext=(x, 2.91), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(5.0, 0.76), xytext=(3.3, 1.31), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(8.0, 0.76), xytext=(9.7, 1.31), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("trees_vs_nets_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved trees_vs_nets_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 112 begins Week 17, classical ML II: support vector machines from scratch, on real linearly separable and non-separable data.")


def main():                                             # runs every section in order
    sets = datasets()                                      # the four problems
    demo_checks()                                          # 1. checks
    res = demo_head_to_head(sets)                          # 2. the head-to-head
    useless = demo_useless(*sets["Breast cancer"])         # 3. useless columns
    rot = demo_rotation(sets)                              # 4. rotation
    demo_monotone(*sets["Breast cancer"])                  # 5. monotone distortions
    size = demo_size(sets)                                 # 6. training-set size
    demo_reliance()                                        # 7. what each model relies on
    plot_results(res, useless, rot, size)                  # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 112


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
