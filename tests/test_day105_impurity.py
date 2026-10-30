"""
Tests for Day 105's impurity functions and split search.

Every expected value can be worked out on paper. They run on GitHub Actions
at each push, on Python 3.9, 3.10 and 3.11.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "week-16-classical-ml-trees"))

from day105_exercise import X, best_split, entropy, gini, y  # noqa: E402


def test_gini_even_and_pure():
    assert np.isclose(gini(np.array([[4, 4]]))[0], 0.5)   # 1 - (0.5^2 + 0.5^2)
    assert gini(np.array([[8, 0]]))[0] == 0               # one class only


def test_entropy_even_and_pure():
    assert np.isclose(entropy(np.array([[4, 4]]))[0], 1.0)        # one fair coin: 1 bit
    assert np.isclose(entropy(np.array([[2, 2, 2, 2]]))[0], 2.0)  # four equal classes: 2 bits
    assert entropy(np.array([[8, 0]]))[0] == 0                    # 0 log 0 counts as 0


def test_best_split_finds_temperature():
    for impurity, removed in ((gini, 0.5), (entropy, 1.0)):
        f, t, g = best_split(X, y, 2, impurity)
        assert f == 1                                     # temperature, not age
        assert np.isclose(t, 37.75)                       # halfway between 37.0 and 38.5
        assert np.isclose(g, removed)                     # the split removes all impurity


def test_best_split_returns_none_when_no_split_helps():
    assert best_split(X[y == 1], y[y == 1], 2, gini) is None                     # pure node
    assert best_split(np.ones((6, 2)), np.array([0, 1] * 3), 2, gini) is None    # identical values
    assert best_split(X, y, 2, gini, min_leaf=5) is None                         # 8 rows cannot give 5 + 5
