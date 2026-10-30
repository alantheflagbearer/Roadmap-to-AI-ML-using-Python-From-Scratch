"""
Day 109 Exercise -- Gradient Boosting

The fifth of Week 16's parts for Day 111's build (trees against neural networks
on tabular data).

    reg_split(X, r)            the split that most reduces squared error
    grow_reg(X, r, idx, max_depth), set_leaves(node, value), predict_reg(node, X)
    fit_boost(X, y, loss, rounds, lr, max_depth, evals)   squared error or log loss
    predict_boost(model, X)    the model's prediction (log-odds for log loss)

Checked by hand on six numbers and on the edge cases: zero rounds gives the best
constant, a rate of 1 with deep trees memorises, and predict_boost equals the
training prediction fit_boost tracked as it went.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


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


X = np.arange(1.0, 7.0)[:, None]; y = np.array([1.0, 1.0, 2.0, 2.0, 6.0, 8.0])
m, E = fit_boost(X, y, "squared", 1, 0.5, 1, evals=(X,))
print(f"mean of y = {y.mean():.3f}; residuals of rows 5, 6 = {y[4] - y.mean():.3f}, {y[5] - y.mean():.3f}; the rest average {(y[:4] - y.mean()).mean():.3f}")
print(f"one stump, rate 0.5: left step {0.5 * (y[:4] - y.mean()).mean():+.3f}, right step {0.5 * (y[4:] - y.mean()).mean():+.3f}; predictions {m['train'].round(3).tolist()}")
print(f"zero rounds gives the best constant: {fit_boost(X, y, 'squared', 0, 0.5, 1)[0]['train'].round(3).tolist()}")
d, _ = fit_boost(X, y, "squared", 1, 1.0, 5)
print(f"rate 1, deep enough to isolate rows: training error {np.abs(d['train'] - y).max():.1e}")
c = (y > 2).astype(int)
b, _ = fit_boost(X, c, "logistic", 50, 0.3, 1)
print(f"log loss, 50 rounds: probabilities {sigmoid(b['train']).round(3).tolist()} for labels {c.tolist()}")
print(f"predict_boost equals the tracked training prediction: largest gap {np.abs(predict_boost(b, X) - b['train']).max():.1e}")
