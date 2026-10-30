"""
Day 110 Exercise -- Newton Boosting and the XGBoost Wrapper

The sixth and last of Week 16's parts for Day 111's build (trees against neural
networks on tabular data).

    newton_split(X, g, h, lam)   the split with the largest Newton gain
    grow_newton(X, g, h, idx, max_depth, lam)   a tree whose nodes hold -G / (H + lambda)
    predict_lt(node, X)          follow rows down by XGBoost's rule (a row ON a threshold goes right)
    fit_newton(X, y, loss, rounds, lr, max_depth, lam, Xe, base)   (prediction on X, on Xe, trees)
    fit_xgb(X, y, params, rounds, Xv, yv, stop)   the library, with optional early stopping
    margin(model, X)             the library's raw prediction, at its best round

Checked by hand on a tiny table, and against XGBoost itself: with the same
settings, the from-scratch trees give the same predictions.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import xgboost as xgb                                  # the library under test


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


def sigmoid(F):                                         # log-odds to probability
    return 1 / (1 + np.exp(-F))


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


X = np.array([[1.0], [2.0], [3.0], [4.0]]); y = np.array([0.0, 0.0, 10.0, 10.0])  # a split at 2.5 is clearly right
g, h = 0.0 - y, np.ones(4)                              # from F = 0: gradients -y, curvature 1
s = newton_split(X, g, h, 1.0)
print(f"from F = 0, lambda = 1: split at {s[1]}; right leaf -G / (H + lambda) = 20 / 3 = {20 / 3:.4f}")
print(f"gain by hand: 0.5 * (0 + 400 / 3 - 400 / 5) = {0.5 * (400 / 3 - 400 / 5):.4f}; newton_split: {s[2]:.4f}")
t = grow_newton(X, g, h, np.arange(4), 1, 1.0)
print(f"leaves: {predict_lt(t, X).round(4).tolist()}")

rng = np.random.RandomState(0)                          # a small random problem, for the comparison
Xr = rng.uniform(size=(200, 4)); yr = ((Xr[:, 0] > 0.5) ^ (Xr[:, 1] > 0.5)).astype(int)
base = float(yr.mean()); F0 = float(np.log(base / (1 - base)))
_, ours, _ = fit_newton(Xr, yr, "logistic", 30, 0.3, 3, 1.0, Xr, F0)
m = fit_xgb(Xr, yr, dict(objective="binary:logistic", eta=0.3, max_depth=3, reg_lambda=1.0, min_child_weight=0, gamma=0, base_score=base,
                         tree_method="exact", seed=0, nthread=1), 30)
print(f"30 rounds on 200 random rows, log loss: largest gap to XGBoost {np.abs(ours - margin(m, Xr)).max():.6f}")
