"""
Day 108 Exercise -- Impurity and Permutation Importance

The fourth of Week 16's parts for Day 111's build (trees against neural networks
on tabular data). Trees are Day 105's nested dictionaries.

    tree_importance(tree, p)          impurity removed by each feature, as shares summing to 1
    impurity_importance(trees, p)     the forest's average of those shares
    permutation_importance(score, X, y, seed, repeats)   score lost when each column is shuffled
    spearman(a, b)                    correlation of two rankings

permutation_importance takes any function score(X, y), so it works for a forest,
a neural network or anything else. Checked by hand and on known answers: a column
the model ignores loses exactly nothing, and a column it depends on loses a lot.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


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


tree = {"counts": np.array([10, 10]), "feature": 0, "gain": 0.3,   # the tree from the lesson's hand check
        "left": {"counts": np.array([9, 3])},
        "right": {"counts": np.array([1, 7]), "feature": 1, "gain": 0.1,
                  "left": {"counts": np.array([1, 0])}, "right": {"counts": np.array([0, 7])}}}
print(f"by hand: 0.3 / 0.34 = {0.3 / 0.34:.4f}, 0.04 / 0.34 = {0.04 / 0.34:.4f}; tree_importance: {tree_importance(tree, 3).round(4).tolist()}")
print(f"two identical trees average to the same shares: {impurity_importance([tree, tree], 3).round(4).tolist()}")

rng = np.random.RandomState(0)
X = rng.normal(size=(500, 3))                           # three columns; the rule uses only the first
y = (X[:, 0] > 0).astype(int)
score = lambda A, b: float(np.mean((A[:, 0] > 0).astype(int) == b))  # a perfect model of that rule
imp = permutation_importance(score, X, y, seed=0)
print(f"a perfect rule on column 0: importance {imp.round(4).tolist()} (about 0.5, because a shuffled column is right half the time)")
print(f"spearman: same order {spearman(np.array([1., 2, 3]), np.array([10., 20, 30])):.1f}; reversed {spearman(np.array([1., 2, 3]), np.array([3., 2, 1])):.1f}")
