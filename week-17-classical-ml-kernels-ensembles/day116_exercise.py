"""
Day 116 Exercise -- Out-of-Fold Predictions and a Stacking Meta-Model

The fifth of Week 17's parts for Day 118's build.

    softmax(Z), fit_softmax(F, y, k, lam, steps, lr)   multinomial logistic regression (the meta-model)
    features(P, k)        meta-features: log-odds for two classes, log-probabilities otherwise
    oof(fit, X, y, k, seed, folds)   every row predicted by a model that never saw it
    stack_fit(fits, X, y, k, seed), stack_proba(model, X)   a stacked model

A base model is any function fit(X, y, k, seed) -> proba(X). Checked on a
tiny problem: out-of-fold predictions never come from a model that saw the row,
and the meta-model learns to trust a useful model over a useless one.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this

EPS = 1e-6                                              # probabilities are clipped before taking logs


def softmax(Z):                                         # NEW: rows of scores to rows of probabilities
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def fit_softmax(F, y, k, lam=1e-3, steps=500, lr=0.5):  # NEW: Day 116's softmax regression, unchanged
    W = np.zeros((F.shape[1], k)); b = np.zeros(k); Y = np.eye(k)[y]
    m = [np.zeros_like(W), np.zeros_like(b)]; v = [np.zeros_like(W), np.zeros_like(b)]
    for t in range(1, steps + 1):
        G = (softmax(F @ W + b) - Y) / len(y)
        for i, (p, g) in enumerate(((W, F.T @ G + lam * W), (b, G.sum(0)))):
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def features(P, k):                                     # NEW: log-odds or log-probabilities
    P = np.clip(P, EPS, 1 - EPS)
    return np.log(P[:, 1] / P[:, 0])[:, None] if k == 2 else np.log(P)


def oof(fit, X, y, k, seed, folds=5):                   # NEW: out-of-fold predictions
    parts = np.array_split(np.random.RandomState(seed).permutation(len(y)), folds); P = np.zeros((len(y), k))
    for i in range(folds):
        va = parts[i]; tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        P[va] = fit(X[tr], y[tr], k, seed)(X[va])
    return P


def stack_fit(fits, X, y, k, seed):                     # NEW: base models on all rows, the meta-model on their out-of-fold predictions
    base = [f(X, y, k, seed) for f in fits]
    W, b = fit_softmax(np.hstack([features(oof(f, X, y, k, seed), k) for f in fits]), y, k)
    return base, W, b, k


def stack_proba(model, X):                              # NEW: the stacked prediction
    base, W, b, k = model
    return softmax(np.hstack([features(p(X), k) for p in base]) @ W + b)


def useful(X, y, k, seed):                              # a base model that reads the first column
    return lambda Z: np.stack([1 / (1 + np.exp(3 * Z[:, 0])), 1 / (1 + np.exp(-3 * Z[:, 0]))], 1)


def useless(X, y, k, seed):                             # one that always says 50/50
    return lambda Z: np.full((len(Z), 2), 0.5)


rng = np.random.RandomState(0); X = rng.normal(size=(200, 2)); y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
seen = []
def spy(Xa, ya, k, seed):                               # records which rows each fold's model was trained on
    seen.append(set(map(tuple, Xa.round(6)))); return useless(Xa, ya, k, seed)
oof(spy, X, y, 2, 0)
parts = np.array_split(np.random.RandomState(0).permutation(200), 5)
print(f"out-of-fold: rows a fold's model saw that it then predicted: {sum(len(seen[i] & set(map(tuple, X[parts[i]].round(6)))) for i in range(5))}")
model = stack_fit([useful, useless], X, y, 2, 0); W = model[1]
print(f"meta-weights (class 1 against class 0): useful model {W[0, 1] - W[0, 0]:+.3f}, useless model {W[1, 1] - W[1, 0]:+.3f} (its log-odds are always 0)")
print(f"stacked accuracy on its training rows: {np.mean(np.argmax(stack_proba(model, X), 1) == y):.3f}; probabilities sum to 1: {np.allclose(stack_proba(model, X).sum(1), 1)}")
