"""
Day 106 Exercise -- The Weakest-Link Path

The second of Week 16's parts for Day 111's build (trees against neural
networks on tabular data). A tree is Day 105's nested dictionary: every node has
"counts", and a question also has "feature", "threshold", "gain", "left", "right".

    branch(node)          (training errors, leaves) of everything below a node
    cost_path(tree)       the increasing alphas, and the tree that is best from each one on
    at_alpha(alphas, trees, a)   the tree in charge at this alpha

Checked by hand on a three-leaf tree whose alphas can be worked out on paper,
and on the edge cases: a tree that is one leaf, a split that saves no errors,
and that the original tree is never changed.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


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


def node(a, b, **q):                                    # a small helper for writing trees by hand
    return {"counts": np.array([a, b]), "pred": int(b > a), **q}


# 20 rows. Root: 12 / 8. Left branch: 10 / 2 with one good split; right branch: 2 / 6 and pure below.
tree = node(12, 8, feature=0, threshold=0.5, gain=0.1,
            left=node(10, 2, feature=1, threshold=0.5, gain=0.05, left=node(10, 0), right=node(0, 2)),
            right=node(2, 6))
alphas, trees = cost_path(tree)
print("hand check: leaves and errors along the path")
for a, t in zip(alphas, trees):
    print(f"  from alpha {a:.3f}: {branch(t)[1]} leaves, {branch(t)[0]} errors")
print("left branch saves 2 errors of 20 with 1 extra leaf: 2 / 20 / 1 = 0.100; root then saves 8 - 4 = 4: 4 / 20 / 1 = 0.200")
print(f"at alpha 0.05 -> {branch(at_alpha(alphas, trees, 0.05))[1]} leaves; at 0.1 -> {branch(at_alpha(alphas, trees, 0.1))[1]}; at 0.5 -> {branch(at_alpha(alphas, trees, 0.5))[1]}")
print(f"the original still has {branch(tree)[1]} leaves")
one = cost_path(node(5, 1))
print(f"a single leaf: alphas {one[0]}, {len(one[1])} tree")
same = node(6, 2, feature=0, threshold=0.5, gain=0.01, left=node(3, 1), right=node(3, 1))
al, ts = cost_path(same)
print(f"a split that saves no errors (both children keep the majority): cut at alpha {al[1]:.3f}, {branch(ts[1])[1]} leaf")
