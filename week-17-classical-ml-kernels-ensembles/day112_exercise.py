"""
Day 112 Exercise -- An SVM Solver That Only Sees a Kernel Matrix

The first of Week 17's parts for Day 118's build (choosing among the week's
models with cross-validation).

    smo(K, y, C)          the dual multipliers alpha, from a matrix of dot products K
    bias(a, y, K, C)      the bias b, from the points on the margin
    decision(a, y, b, K_new)   the SVM's output for new points, given K(train, new)

Because the solver only ever looks at K, Day 113 can give it other kernels.
Checked by hand on two points and on the edge cases: labels must balance
(sum a y = 0), multipliers stay inside [0, C], and a point that is far from the
boundary gets a multiplier of exactly 0.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def smo(K, y, C, tol=1e-3, max_iter=200000):            # NEW: Day 112's solver, unchanged
    n = len(y); a = np.zeros(n); f = np.zeros(n)
    for it in range(max_iter):
        G = y * f - 1
        up = ((a < C) & (y > 0)) | ((a > 0) & (y < 0))
        low = ((a < C) & (y < 0)) | ((a > 0) & (y > 0))
        v = -y * G
        i = np.where(up)[0][np.argmax(v[up])]; j = np.where(low)[0][np.argmin(v[low])]
        if v[i] - v[j] < tol:
            break
        eta = max(K[i, i] + K[j, j] - 2 * K[i, j], 1e-12)
        if y[i] != y[j]:
            L, H = max(0.0, a[j] - a[i]), min(C, C + a[j] - a[i])
        else:
            L, H = max(0.0, a[i] + a[j] - C), min(C, a[i] + a[j])
        Ei, Ej = f[i] - y[i], f[j] - y[j]
        aj = np.clip(a[j] + y[j] * (Ei - Ej) / eta, L, H)
        ai = a[i] + y[i] * y[j] * (a[j] - aj)
        ai, aj = [0.0 if v < 1e-12 * C else C if v > C * (1 - 1e-12) else v for v in (ai, aj)]  # snap rounding errors onto the bounds
        f += (ai - a[i]) * y[i] * K[:, i] + (aj - a[j]) * y[j] * K[:, j]
        a[i], a[j] = ai, aj
    return a, it + 1


def bias(a, y, K, C):                                   # NEW: b from the KKT conditions
    f = K @ (a * y); e = 1e-8 * max(C, 1)
    free = (a > e) & (a < C - e)
    if free.any():
        return float(np.mean(y[free] - f[free]))
    lo = (a <= e) & (y > 0) | (a >= C - e) & (y < 0)
    return float((np.max((y - f)[lo]) + np.min((y - f)[~lo])) / 2)


def decision(a, y, b, K_new):                           # NEW: sum_i a_i y_i K(x_i, x) + b, for each new point
    return K_new.T @ (a * y) + b


# two points of class -1 near x = 0, one of class +1 at x = 2, and one far away at x = 10 (class +1)
X = np.array([[0.0], [0.5], [2.0], [10.0]]); y = np.array([-1.0, -1.0, 1.0, 1.0])
K = X @ X.T                                             # the linear kernel
a, it = smo(K, y, C=100.0); b = bias(a, y, K, 100.0)
w = float((a * y) @ X[:, 0])
print(f"multipliers {a.round(4).tolist()} after {it} steps; sum a y = {(a * y).sum():.1e}")
print(f"w = {w:.4f}, b = {b:.4f}: the boundary is at x = {-b / w:.4f}, halfway between 0.5 and 2 (by hand: 1.25, w = 2 / 1.5 = {2 / 1.5:.4f})")
print(f"the point at x = 10 is far from the boundary, so its multiplier is exactly {a[3]}; the one at x = 0 too: {a[0]}")
print(f"decision values for x = 1 and x = 1.5: {decision(a, y, b, X @ np.array([[1.0, 1.5]])).round(4).tolist()}")
