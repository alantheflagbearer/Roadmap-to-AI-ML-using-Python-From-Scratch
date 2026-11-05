"""
Day 112: Support Vector Machines From Scratch -- Separable and Non-Separable Data

Week 17 continues classical machine learning. A support vector machine (SVM)
draws the straight boundary that leaves the widest possible gap, the margin,
between two classes. Only the points on the edge of that gap, the support
vectors, decide where it goes. When no straight line separates the classes, a
cost C says how much each violation of the margin should cost.

Everything is written from scratch in NumPy:
  * the dual problem, solved by SMO (sequential minimal optimisation): two
    multipliers at a time, choosing the pair that most violates optimality
  * the weights, the bias and the margin recovered from the multipliers
  * checks: the KKT conditions, the duality gap, scikit-learn's SVC as an
    outside check, and retraining on the support vectors alone

The data:
  * Iris (bundled with scikit-learn): setosa against versicolor is linearly
    separable; versicolor against virginica is not.
  * Breast Cancer Wisconsin: 30 measurements, not separable at a sensible C.

  1. By hand: two points.
  2. Correctness checks.
  3. Separable data: the widest gap, against the perceptron's arbitrary line.
  4. Only the support vectors matter.
  5. Non-separable data: what C does.
  6. Choosing C, scaling, and a comparison with logistic regression.
  7. One mislabelled point: hard margin against soft margin.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported exercise quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 99's bootstrap

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_breast_cancer, load_iris  # bundled datasets
from sklearn.svm import SVC                            # used ONLY as an outside check of the from-scratch solver
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, os.path.join(HERE, "..", "week-15-practical-nlp-pretrained"))  # Day 99's exercise
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day99_exercise as d99                        # Day 99: bootstrap_ci
CS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]               # the cost values tried
SPLITS = 20                                             # repeated splits in section 6
TOL = 1e-3                                              # SMO stops when the worst violation is below this


# =============================================================================
# THE SOLVER -- NEW
# =============================================================================
def smo(K, y, C, tol=TOL, max_iter=200000):             # NEW: the dual multipliers alpha, and the iterations used
    n = len(y); a = np.zeros(n); f = np.zeros(n)        # multipliers; f = sum_k a_k y_k K(x_k, x) without the bias
    for it in range(max_iter):
        G = y * f - 1                                     # gradient of the dual objective (to be minimised)
        up = ((a < C) & (y > 0)) | ((a > 0) & (y < 0))    # multipliers that may move "up" along y
        low = ((a < C) & (y < 0)) | ((a > 0) & (y > 0))   # and "down"
        v = -y * G                                        # how strongly each point wants to change
        i = np.where(up)[0][np.argmax(v[up])]; j = np.where(low)[0][np.argmin(v[low])]  # the most violating pair
        if v[i] - v[j] < tol:                             # optimal, to within tol
            break
        eta = max(K[i, i] + K[j, j] - 2 * K[i, j], 1e-12)  # curvature along the pair's direction
        if y[i] != y[j]:                                  # the box the pair must stay in, given sum a_k y_k = 0
            L, H = max(0.0, a[j] - a[i]), min(C, C + a[j] - a[i])
        else:
            L, H = max(0.0, a[i] + a[j] - C), min(C, a[i] + a[j])
        Ei, Ej = f[i] - y[i], f[j] - y[j]                 # errors (the bias cancels in their difference)
        aj = np.clip(a[j] + y[j] * (Ei - Ej) / eta, L, H)  # Platt's step for a_j, clipped to the box
        ai = a[i] + y[i] * y[j] * (a[j] - aj)             # a_i moves to keep sum a_k y_k = 0
        ai, aj = [0.0 if v < 1e-12 * C else C if v > C * (1 - 1e-12) else v for v in (ai, aj)]  # snap rounding errors onto the bounds
        f +=(ai - a[i]) * y[i] * K[:, i] + (aj - a[j]) * y[j] * K[:, j]  # update the outputs
        a[i], a[j] = ai, aj
    return a, it + 1


def bias(a, y, K, C):                                   # NEW: b from the KKT conditions
    f = K @ (a * y); e = 1e-8 * max(C, 1)                 # outputs without the bias; a tolerance for "on the bound"
    free = (a > e) & (a < C - e)                          # points exactly on the margin
    if free.any():
        return float(np.mean(y[free] - f[free]))
    lo = (a <= e) & (y > 0) | (a >= C - e) & (y < 0)      # these need b >= y - f
    return float((np.max((y - f)[lo]) + np.min((y - f)[~lo])) / 2)  # the middle of the allowed range


def fit_svm(X, y, C):                                   # NEW: a linear SVM: (w, b, a)
    K = X @ X.T                                           # the linear kernel: every dot product
    a, _ = smo(K, y, C)
    return (a * y) @ X, bias(a, y, K, C), a


def primal(w, b, X, y, C):                              # the primal objective: half |w|^2 + C * total hinge loss
    return 0.5 * w @ w + C * np.maximum(0, 1 - y * (X @ w + b)).sum()


def dual(a, y, K):                                      # the dual objective, to be maximised
    return a.sum() - 0.5 * (a * y) @ K @ (a * y)


def standardise(A, B):                                  # scale with the training part's statistics
    mu, sd = A.mean(0), A.std(0) + 1e-12
    return (A - mu) / sd, (B - mu) / sd


def stratified_split(y, test_frac, seed):               # Day 105's split, unchanged
    rng = np.random.RandomState(seed); tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); k = int(round(len(idx) * test_frac))
        te += list(idx[:k]); tr += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


def perceptron(X, y, seed, epochs=100):                 # Rosenblatt's perceptron: any separating line will do
    rng = np.random.RandomState(seed); w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        mistakes = 0
        for i in rng.permutation(len(y)):                 # a random order each pass
            if y[i] * (X[i] @ w + b) <= 0:                  # wrong side (or on the line): move towards this point
                w += y[i] * X[i]; b += y[i]; mistakes += 1
        if mistakes == 0:
            break
    return w, b


def logistic(X, y, lam, steps=3000, lr=0.5):            # L2-regularised logistic regression by gradient descent
    w = np.zeros(X.shape[1]); b = 0.0; t = (y > 0).astype(float)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - t) / len(y) + lam * w); b -= lr * np.mean(p - t)
    return w, b


def kfold_svm_accuracy(X, y, C, k=5, seed=0):           # mean validation accuracy over k folds, scaling inside each fold
    folds = np.array_split(np.random.RandomState(seed).permutation(len(y)), k); acc = []
    for i in range(k):
        va = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        A, B = standardise(X[tr], X[va]); w, b, _ = fit_svm(A, y[tr], C)
        acc.append(np.mean(np.sign(B @ w + b) == y[va]))
    return float(np.mean(acc))


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- two points")
    print("=" * 70)
    X = np.array([[1.0, 1.0], [3.0, 3.0]]); y = np.array([-1.0, 1.0])  # one point of each class
    w, b, a = fit_svm(X, y, C=1e6)
    print("points (1, 1) labelled -1 and (3, 3) labelled +1: the best line is halfway, at right angles to the segment between them")
    print(f"w = {w.round(4).tolist()}, b = {b:.4f}, multipliers {a.round(4).tolist()}")
    print(f"by hand: w = (0.5, 0.5), b = -2, margin 2 / |w| = {2 / np.linalg.norm([0.5, 0.5]):.4f} = the distance between the points; from the solver {2 / np.linalg.norm(w):.4f}")
    print(f"both points are support vectors: y (w.x + b) = {(y * (X @ w + b)).round(4).tolist()}")
    print("Actual output from this lesson's run.")


def demo_check(X, y):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS CHECKS -- breast cancer, standardised, C = 1")
    print("=" * 70)
    tr, te = stratified_split(y, 0.3, 0); A, B = standardise(X[tr], X[te]); yt = y[tr]
    K = A @ A.T; a, iters = smo(K, yt, 1.0); w = (a * yt) @ A; b = bias(a, yt, K, 1.0)
    m = yt * (A @ w + b); e = 1e-6                         # margins of the training points
    print(f"SMO finished in {iters} pair updates; sum of a_i y_i = {(a * yt).sum():.1e} (must be 0); every multiplier in [0, C]: {bool(np.all((a >= -1e-12) & (a <= 1 + 1e-12)))}")
    print(f"KKT: points with a = 0 should have margin >= 1: worst {m[a < e].min():.4f}; points with 0 < a < C should have margin 1: "
          f"range {m[(a > e) & (a < 1 - e)].min():.4f} to {m[(a > e) & (a < 1 - e)].max():.4f}; points with a = C should have margin <= 1: worst {m[a > 1 - e].max():.4f}")
    p, d = primal(w, b, A, yt, 1.0), dual(a, yt, K)
    print(f"primal {p:.4f}, dual {d:.4f}: the duality gap is {p - d:.4f}, {100 * (p - d) / p:.3f}% of the primal (0 at the exact optimum)")
    sk = SVC(kernel="linear", C=1.0, tol=1e-6).fit(A, yt)
    print(f"scikit-learn's SVC on the same data: cosine between the two w {w @ sk.coef_[0] / np.linalg.norm(w) / np.linalg.norm(sk.coef_[0]):.6f}; "
          f"b {b:.4f} against {sk.intercept_[0]:.4f}; support vectors {int((a > e).sum())} against {len(sk.support_)}")
    print(f"test predictions that agree with scikit-learn: {np.mean(np.sign(B @ w + b) == sk.predict(B)):.4f}")
    print("Actual output from this lesson's run.")


def demo_separable(Xi, yi):
    print("\n" + "=" * 70)
    print("3. SEPARABLE DATA -- iris setosa against versicolor, petal length and width")
    print("=" * 70)
    m = yi < 2; X = Xi[m][:, 2:4]; y = np.where(yi[m] == 0, -1.0, 1.0)  # 100 flowers, two measurements
    w, b, a = fit_svm(X, y, C=1e6)                        # a huge C: a hard margin
    sv = np.where(a > 1e-6)[0]
    print(f"{len(y)} flowers; SVM with C = 1e6 (a hard margin): {len(sv)} support vectors, margin width {2 / np.linalg.norm(w):.4f} cm, "
          f"training accuracy {np.mean(np.sign(X @ w + b) == y):.4f}")
    gaps = []
    for s in range(20):                                   # the perceptron, from 20 different random orders
        pw, pb = perceptron(X, y, s)
        gaps.append(np.min(y * (X @ pw + pb)) / np.linalg.norm(pw) * 2)  # its gap: twice the closest point's distance
    print(f"perceptron from 20 random orders: training accuracy 1.0000 every time: {all(np.all(np.sign(X @ perceptron(X, y, s)[0] + perceptron(X, y, s)[1]) == y) for s in range(20))}")
    print(f"  but its gap ranges from {min(gaps):.4f} to {max(gaps):.4f} cm (mean {np.mean(gaps):.4f}); the SVM's is the largest possible, {2 / np.linalg.norm(w):.4f}")
    print("Actual output from this lesson's run.")
    return X, y, w, b, sv, [perceptron(X, y, s) for s in range(5)]  # the plot uses them


def demo_support(X, y):
    print("\n" + "=" * 70)
    print("4. ONLY THE SUPPORT VECTORS MATTER")
    print("=" * 70)
    print(f"breast cancer, standardised; SMO stops at a tolerance of {TOL}, so changes of about that size are the solver's, not the data's")
    tr, _ = stratified_split(y, 0.3, 0); A, _ = standardise(X[tr], X[tr]); yt = y[tr]
    for C in (0.01, 1.0):
        w, b, a = fit_svm(A, yt, C); sv = a > 1e-8
        w2, b2, _ = fit_svm(A[sv], yt[sv], C)              # retrain on the support vectors alone
        keep = ~sv; rng = np.random.RandomState(0); drop = rng.choice(np.where(keep)[0], keep.sum() // 2, replace=False)
        rest = np.setdiff1d(np.arange(len(yt)), drop); w3, b3, _ = fit_svm(A[rest], yt[rest], C)  # remove half the others
        half = rng.choice(np.where(sv)[0], sv.sum() // 2, replace=False)  # and, for contrast, half of the support vectors
        rest = np.setdiff1d(np.arange(len(yt)), half); w4, b4, _ = fit_svm(A[rest], yt[rest], C)
        ch = lambda v: np.linalg.norm(v - w) / np.linalg.norm(w)  # relative change in w
        print(f"C = {C}: {int(sv.sum())} support vectors of {len(yt)}. Relative change in w: retrained on the support vectors alone {ch(w2):.1e}; "
              f"without half of the other points {ch(w3):.1e}; without half of the support vectors {ch(w4):.1e}")
    print("Actual output from this lesson's run.")


def demo_cost(X, y):
    print("\n" + "=" * 70)
    print("5. NON-SEPARABLE DATA -- what C does (breast cancer, standardised, one split)")
    print("=" * 70)
    tr, te = stratified_split(y, 0.3, 0); A, B = standardise(X[tr], X[te]); yt = y[tr]
    print(f"{'C':>8}{'margin':>9}{'support':>9}{'inside or wrong':>17}{'train acc':>11}{'test acc':>10}{'SMO steps':>11}")
    rows = []
    for C in CS:
        K = A @ A.T; a, it = smo(K, yt, C); w = (a * yt) @ A; b = bias(a, yt, K, C); m = yt * (A @ w + b)
        r = (C, 2 / np.linalg.norm(w), int((a > 1e-8 * max(C, 1)).sum()), int((m < 1 - 1e-2).sum()),
             np.mean(np.sign(A @ w + b) == yt), np.mean(np.sign(B @ w + b) == y[te]), it)
        rows.append(r)
        print(f"{C:>8}{r[1]:>9.4f}{r[2]:>9}{r[3]:>17}{r[4]:>11.4f}{r[5]:>10.4f}{r[6]:>11}")
    print("margin is 2/|w| in standardised units; 'inside or wrong' counts training points with y f(x) < 0.99 (points on the margin sit within the solver's tolerance of 1)")
    print("Actual output from this lesson's run.")
    return rows                                              # the plot uses them


def demo_compare(X, y):
    print("\n" + "=" * 70)
    print(f"6. CHOOSING C, SCALING, AND LOGISTIC REGRESSION -- breast cancer, {SPLITS} splits")
    print("=" * 70)
    rows, picks, stuck = [], [], 0
    for s in range(SPLITS):
        tr, te = stratified_split(y, 0.3, s)
        cv = [kfold_svm_accuracy(X[tr], y[tr], C, seed=s) for C in CS]; C = CS[int(np.argmax(cv))]; picks.append(C)  # C by 5-fold CV
        A, B = standardise(X[tr], X[te]); w, b, _ = fit_svm(A, y[tr], C); svm = np.mean(np.sign(B @ w + b) == y[te])
        K = X[tr] @ X[tr].T; ra, it = smo(K, y[tr], C); stuck += it == 200000  # the same C, unscaled: did SMO finish?
        raw_w, raw_b = (ra * y[tr]) @ X[tr], bias(ra, y[tr], K, C); raw = np.mean(np.sign(X[te] @ raw_w + raw_b) == y[te])
        lw, lb = logistic(A, y[tr], 1 / (C * len(tr))); lr = np.mean(np.sign(B @ lw + lb) == y[te])  # the same penalty strength
        rows.append((svm, raw, lr))
    r = np.array(rows)
    print(f"C chosen by 5-fold CV on each training part: {sorted({c: picks.count(c) for c in set(picks)}.items())}")
    print(f"linear SVM, standardised: {r[:, 0].mean():.4f} (std {r[:, 0].std():.4f})")
    print(f"linear SVM, raw measurements, same C: {r[:, 1].mean():.4f} (std {r[:, 1].std():.4f}); SMO hit its 200,000-step limit on {stuck} of {SPLITS} splits")
    print(f"logistic regression, standardised, same penalty: {r[:, 2].mean():.4f} (std {r[:, 2].std():.4f})")
    for i, n in ((1, "raw"), (2, "logistic")):
        d = r[:, 0] - r[:, i]; lo, hi = d99.bootstrap_ci(d)
        print(f"  SVM minus {n:<9} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; better on {int((d > 0).sum())}, worse on {int((d < 0).sum())} of {SPLITS}")
    print("the splits share rows, so these intervals are too narrow, a rough guide only")
    print("Actual output from this lesson's run.")


def demo_outlier(X, y):
    print("\n" + "=" * 70)
    print("7. ONE MISLABELLED POINT -- hard margin against soft margin")
    print("=" * 70)
    Z = X.copy(); t = y.copy(); k = int(np.argmin(np.where(y > 0, X[:, 0], np.inf)))  # the versicolor with the shortest petal
    t[k] = -1.0                                           # relabelled as setosa
    print(f"flower {k} (petal {Z[k, 0]:.1f} x {Z[k, 1]:.1f} cm, really versicolor) relabelled as setosa")
    for C in (1e6, 10, 1, 0.1):
        w, b, a = fit_svm(Z, t, C)
        ok = np.mean(np.sign(Z @ w + b)[np.arange(len(t)) != k] == y[np.arange(len(t)) != k])  # against the TRUE labels, other flowers
        print(f"  C = {C:<7}: margin {2 / np.linalg.norm(w):.4f} cm, {int((a > 1e-8 * max(C, 1)).sum())} support vectors, the other 99 flowers right: {ok:.4f}")
    print("Actual output from this lesson's run.")


def plot_results(X, y, w, b, sv, percs, rows):          # two panels: the separable picture | C against margin and accuracy
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))             # canvas
    a = ax[0]; xs = np.linspace(0.5, 5.5, 100)
    a.scatter(X[y < 0, 0], X[y < 0, 1], s=18, label="setosa"); a.scatter(X[y > 0, 0], X[y > 0, 1], s=18, label="versicolor")
    a.scatter(X[sv, 0], X[sv, 1], s=110, facecolors="none", edgecolors="k", label="support vectors")
    for k, ls in ((0, "-"), (1, "--"), (-1, "--")):          # the boundary and the two edges of the gap
        a.plot(xs, (k - b - w[0] * xs) / w[1], "k" + ls, linewidth=1.2 if k == 0 else 0.8)
    for pw, pb in percs:
        a.plot(xs, (-pb - pw[0] * xs) / pw[1], color="#d62728", alpha=0.5, linewidth=0.8)
    a.plot([], [], color="#d62728", alpha=0.5, label="perceptron, 5 random orders")
    a.set_xlim(0.5, 5.5); a.set_ylim(-0.5, 2.2); a.set_xlabel("petal length (cm)"); a.set_ylabel("petal width (cm)")
    a.set_title("Iris: the widest gap (black) and other separating lines (red)", fontsize=10); a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)
    a = ax[1]; c = [r[0] for r in rows]
    a.plot(c, [r[4] for r in rows], marker="o", label="training accuracy"); a.plot(c, [r[5] for r in rows], marker="o", label="test accuracy")
    a.set_xscale("log"); a.set_xlabel("C"); a.set_ylabel("accuracy"); a.grid(alpha=0.3)
    t = a.twinx(); t.plot(c, [r[1] for r in rows], marker="s", linestyle="--", color="#2ca02c", label="margin 2/|w|"); t.set_yscale("log"); t.set_ylabel("margin (log scale)")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = t.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    a.set_title("Breast cancer: C trades margin for training errors", fontsize=10)
    plt.tight_layout(); plt.savefig("svm_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved svm_results.png")


def draw_diagram():                                     # primal, dual, SMO, recovery
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 112: a support vector machine, from the problem to the line", ha="center", fontsize=12, fontweight="bold")
    box(1.7, 2.3, 2.9, 2.0, "primal: find w, b\nminimise |w|^2 / 2\n+ C x hinge losses", "#FAEEDA")
    box(5.0, 2.3, 2.9, 2.0, "dual: one multiplier\nper point, 0 <= a <= C,\nsum a y = 0;\nonly dot products", "#DCEBF7")
    box(8.3, 2.3, 2.9, 2.0, "SMO: move the pair\nthat most violates\noptimality, exactly,\nuntil none does", "#E5F2E1")
    box(11.5, 2.3, 2.6, 2.0, "w = sum a y x\nb from points on\nthe margin;\na > 0: support vectors", "#EEEDFE")
    for x0, x1 in ((3.19, 3.51), (6.49, 6.81), (9.79, 10.16)):
        ax.annotate("", xy=(x1, 2.3), xytext=(x0, 2.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("svm_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved svm_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 113 is the kernel trick: the dual only ever uses dot products, so replacing them with a kernel gives curved boundaries from the same solver.")


def main():                                             # runs every section in order
    ir, bc = load_iris(), load_breast_cancer()
    Xc, yc = bc.data, np.where(bc.target == 1, 1.0, -1.0)  # benign +1, malignant -1
    demo_hand()                                            # 1. by hand
    demo_check(Xc, yc)                                     # 2. correctness
    X, y, w, b, sv, percs = demo_separable(ir.data, ir.target)  # 3. separable data
    demo_support(Xc, yc)                                   # 4. support vectors
    rows = demo_cost(Xc, yc)                               # 5. what C does
    demo_compare(Xc, yc)                                   # 6. choosing C
    demo_outlier(X, y)                                     # 7. one mislabelled point
    plot_results(X, y, w, b, sv, percs, rows)              # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 113


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
