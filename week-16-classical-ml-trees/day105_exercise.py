"""
Day 105 Exercise -- Impurity and the Best Split

The first of Week 16's parts for Day 111's build (trees against neural
networks on tabular data).

    gini(counts)          1 - sum p_k^2 for class counts along the last axis
    entropy(counts)       -sum p_k log2 p_k, with 0 log 0 = 0
    best_split(X, y, n_classes, impurity, min_leaf)   (feature, threshold, impurity removed), or None

Checked by hand on a tiny table whose best question can be worked out on
paper, and on the edge cases: a pure node, identical feature values, and a
minimum leaf size that forbids the best split.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


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


# eight patients: (age, temperature), and whether they had the flu (1) or not (0)
X = np.array([[25, 36.6], [30, 38.9], [45, 37.0], [50, 39.2], [35, 38.5], [60, 36.8], [28, 39.0], [41, 36.9]])
y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
print(f"Gini of the whole table (4 flu, 4 not): {gini(np.bincount(y)[None])[0]:.3f}; entropy {entropy(np.bincount(y)[None])[0]:.3f} bits")
for name, imp in (("Gini", gini), ("entropy", entropy)):
    f, t, g = best_split(X, y, 2, imp)
    print(f"{name:<8}: best question 'feature {f} ({['age', 'temperature'][f]}) <= {t:.2f}', impurity removed {g:.3f} "
          "(temperature separates the classes perfectly, so all impurity goes)")
print(f"a pure node: best_split -> {best_split(X[y == 1], y[y == 1], 2, gini)}")
print(f"identical feature values: best_split -> {best_split(np.ones((6, 2)), np.array([0, 1, 0, 1, 0, 1]), 2, gini)}")
print(f"min_leaf = 5 on 8 rows (no split leaves 5 on both sides): best_split -> {best_split(X, y, 2, gini, min_leaf=5)}")
f, t, g = best_split(X, y, 2, gini, min_leaf=3)
print(f"min_leaf = 3: still possible -> feature {f}, threshold {t:.2f}, impurity removed {g:.3f}")
