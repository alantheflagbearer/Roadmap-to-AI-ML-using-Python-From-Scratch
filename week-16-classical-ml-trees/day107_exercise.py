"""
Day 107 Exercise -- A Small Random Forest

The third of Week 16's parts for Day 111's build (trees against neural networks
on tabular data). It uses Day 105's exercise for the split search.

    grow_random(X, y, n_classes, mtry, rng)   a tree whose questions each see mtry random features
    proba(node, X, n_classes)                 class probabilities for many rows at once
    fit_forest(X, y, n_classes, n_trees, mtry, seed)   (trees, the bootstrap of each)
    forest_proba(trees, X, n_classes)         the average of the trees' probabilities
    oob_accuracy(trees, bags, X, y, n_classes)   each row judged only by trees that never saw it

Checked on Day 105's eight patients and on the edge cases: a forest is the same
for the same seed, probabilities sum to 1, and a useless random feature cannot
stop a question from being asked.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported exercise quiet
import io                                              # io: StringIO, where that output goes

import numpy as np                                     # numpy: every array/matrix operation below is built on this

with contextlib.redirect_stdout(io.StringIO()):         # Day 105's exercise prints when imported
    from day105_exercise import best_split, gini, X, y  # the split search, the impurity, and the eight patients


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


trees, bags = fit_forest(X, y, 2, 25, 1, seed=0)        # eight patients, 25 trees, one feature per question
p = forest_proba(trees, X, 2)
print(f"eight patients, 25 trees, mtry 1: training accuracy {np.mean(np.argmax(p, 1) == y):.3f}, OOB accuracy {oob_accuracy(trees, bags, X, y, 2)[0]:.3f}")
print(f"probabilities sum to 1: largest error {np.abs(p.sum(1) - 1).max():.1e}")
again = forest_proba(fit_forest(X, y, 2, 25, 1, seed=0)[0], X, 2)
other = forest_proba(fit_forest(X, y, 2, 25, 1, seed=1)[0], X, 2)
print(f"same seed: largest difference {np.abs(p - again).max():.1e}; another seed: {np.abs(p - other).max():.3f}")
print(f"rows left out per tree (expected about 8 x 0.34 = 2.7): {np.mean([8 - len(set(b.tolist())) for b in bags]):.1f}")
useless = np.column_stack([np.ones(8), X[:, 1]])        # a constant column beside the temperature
roots = [grow_random(useless, y, 2, 1, np.random.RandomState(s))["feature"] for s in range(20)]
print(f"mtry 1 with a constant column: the root still uses the real feature in {sum(r == 1 for r in roots)} of 20 seeds")
