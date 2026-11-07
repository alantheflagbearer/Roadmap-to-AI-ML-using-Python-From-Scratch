"""
Day 128 Exercise -- Anomaly Scores and Thresholds

The third of Week 19's parts.

    ae_score(net, X, acts, kind)    reconstruction error per example: the anomaly score
    pca_score(X, pca, k)            the same score for PCA with k components
    nn_score(X, ref)                distance to the nearest normal example
    threshold(normal_scores, fa)    the score only a share fa of normal examples exceed
    alarm_stats(scores, y, t)       recall, false-alarm rate and precision of a threshold

Checked by hand: a threshold on six normal scores, the three rates it gives, a
nearest-neighbour distance, and a PCA score of 0 for a point on PCA's own line.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import os                                              # os: paths
import sys                                             # sys: path, to import earlier parts
import contextlib                                      # contextlib: keep the imports quiet
import io                                              # io: where their output goes
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "week-18-unsupervised-learning-i"))
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: reconstruct
    import day123_exercise as d123                      # Day 123: pca_fit, transform, reconstruct
    import day119_exercise as d119                      # Day 119: sqdist


def ae_score(net, X, acts, kind="bce"):                 # NEW: Day 128's version, unchanged
    return ((d126.reconstruct(net, X, acts, kind) - X) ** 2).sum(1)


def pca_score(X, pca, k):                               # NEW: Day 128's version, unchanged
    mu, V = pca
    return ((d123.reconstruct(d123.transform(X, mu, V, k), mu, V, k) - X) ** 2).sum(1)


def nn_score(X, ref, chunk=2000):                       # NEW: Day 128's version, unchanged
    return np.concatenate([np.sqrt(np.maximum(d119.sqdist(X[s:s + chunk], ref), 0).min(1)) for s in range(0, len(X), chunk)])


def threshold(normal_scores, false_alarm):              # NEW: Day 128's version, unchanged
    return float(np.quantile(normal_scores, 1 - false_alarm))


def alarm_stats(scores, y, t):                          # NEW: Day 128's version, unchanged
    flag = scores > t; tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    return {"recall": tp / max((y == 1).sum(), 1), "false alarms": fp / max((y == 0).sum(), 1), "precision": tp / max(flag.sum(), 1)}


s = np.array([1.0, 2.0, 2.5, 3.0, 4.0, 9.0, 3.5, 8.0]); y = np.array([0, 0, 0, 0, 0, 1, 0, 1])
t = threshold(s[y == 0], 0.2); st = alarm_stats(s, y, t)
print(f"threshold for 20% false alarms on [1, 2, 2.5, 3, 4, 3.5]: {t:.2f} (by hand: the 80th percentile, 3.5)")
print(f"recall {st['recall']:.3f} (2 of 2), false alarms {st['false alarms']:.3f} (1 of 6), precision {st['precision']:.3f} (2 of 3)")
ref = np.array([[0.0, 0.0], [3.0, 4.0]]); print(f"distance from (3, 0) to the nearest of (0, 0) and (3, 4): {nn_score(np.array([[3.0, 0.0]]), ref)[0]:.4f} (by hand 3)")
t_ = np.linspace(-1, 1, 21); L = np.c_[t_, 2 * t_]; mu, vals, V = d123.pca_fit(L)
print(f"PCA score, 1 component, of a point on the line: {pca_score(np.array([[0.5, 1.0]]), (mu, V), 1)[0]:.2e}; of (1, 0): {pca_score(np.array([[1.0, 0.0]]), (mu, V), 1)[0]:.4f} (by hand 4/5 = 0.8)")
