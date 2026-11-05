"""
Day 113: The Kernel Trick From Scratch -- a Real RBF-Kernel Demonstration

Day 112's solver never looked at the data itself, only at the table of dot
products K. Replace that table with another similarity, a kernel, and the same
solver draws curved boundaries: it behaves as if each point had been mapped
into many new features, without ever computing them.

Everything is written from scratch in NumPy:
  * kernels: linear, polynomial and RBF (Gaussian), as matrices
  * the explicit feature map that a degree-2 polynomial kernel stands for
  * Day 112's SMO, unchanged, with one-vs-rest for ten classes
  * random Fourier features: an explicit approximation of the RBF kernel

  1. The trick, checked: a kernel equals a dot product of mapped features.
  2. What makes a valid kernel.
  3. A real curved boundary: two breast-cancer measurements, gamma from small to large.
  4. Where a kernel helps: breast cancer and handwritten digits.
  5. gamma and C together.
  6. Random Fourier features: the RBF kernel without the n-by-n matrix.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 112
import time                                            # time: kernel against explicit features

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_breast_cancer, load_digits  # bundled datasets
from sklearn.svm import SVC                            # used ONLY as an outside check
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 112 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day112_support_vector_machines as d112       # Day 112: smo, bias, standardise, stratified_split
smo, bias, standardise, stratified_split = d112.smo, d112.bias, d112.standardise, d112.stratified_split  # unchanged
d99 = d112.d99                                          # Day 99's bootstrap, through Day 112
GAMMAS = [0.01, 0.1, 1, 10, 100]                        # section 3: RBF widths, on two standardised features
GRID_G = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]         # section 5: gamma on digits (pixels scaled to 0..1)
GRID_C = [0.1, 1, 10, 100]                              # section 5: C
DIMS = [10, 30, 100, 300, 1000, 3000]                   # section 6: random features
SPLITS = 5                                              # repeated splits in sections 4 and 6


# =============================================================================
# KERNELS -- NEW
# =============================================================================
def linear(A, B):                                       # NEW: plain dot products
    return A @ B.T


def poly(A, B, degree=2, c=1.0):                        # NEW: (x.z + c)^degree
    return (A @ B.T + c) ** degree


def sqdist(A, B):                                       # all squared distances at once
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def rbf(A, B, gamma):                                   # NEW: exp(-gamma |x - z|^2), a similarity between 0 and 1
    return np.exp(-gamma * np.maximum(sqdist(A, B), 0))   # clip tiny negatives from rounding


def phi_poly2(X):                                       # NEW: the features (x.z + 1)^2 secretly uses
    n, p = X.shape; cols = [np.ones(n)] + [np.sqrt(2) * X[:, i] for i in range(p)]  # 1, and sqrt(2) x_i
    cols += [X[:, i] ** 2 for i in range(p)]              # x_i^2
    cols += [np.sqrt(2) * X[:, i] * X[:, j] for i in range(p) for j in range(i + 1, p)]  # sqrt(2) x_i x_j
    return np.column_stack(cols)


def rff(X, D, gamma, seed):                             # NEW: random Fourier features, z(x).z(x') ~ exp(-gamma |x - x'|^2)
    rng = np.random.RandomState(seed)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(X.shape[1], D))  # frequencies from the kernel's own spectrum
    c = rng.uniform(0, 2 * np.pi, D)                      # random phases
    return lambda Z: np.sqrt(2.0 / D) * np.cos(Z @ W + c)  # the same map for training and test rows


# =============================================================================
# A KERNEL SVM -- NEW, from Day 112's parts
# =============================================================================
def fit_kernel_svm(K, y, C):                            # NEW: multipliers and bias, given a kernel matrix
    a, it = smo(K, y, C)
    return a, bias(a, y, K, C), it


def ovr_fit(K, y, C):                                   # NEW: one binary SVM per class, that class against the rest
    return [fit_kernel_svm(K, np.where(y == c, 1.0, -1.0), C)[:2] for c in range(int(y.max()) + 1)]


def ovr_predict(models, y, K_new):                      # NEW: the class whose SVM is most sure
    scores = [K_new.T @ (a * np.where(y == c, 1.0, -1.0)) + b for c, (a, b) in enumerate(models)]
    return np.argmax(np.stack(scores, 1), 1)


def digits():                                           # 1,797 images, pixels scaled to 0..1
    d = load_digits(); return d.data / 16.0, d.target


# =============================================================================
# DEMOS
# =============================================================================
def demo_trick():
    print("=" * 70)
    print("1. THE TRICK, CHECKED -- a kernel equals a dot product of mapped features")
    print("=" * 70)
    X, _ = digits(); A = X[:500]                           # 500 digit images, 64 pixels each
    t0 = time.perf_counter(); F = phi_poly2(A); Ke = F @ F.T; te = time.perf_counter() - t0  # the long way
    t0 = time.perf_counter(); Kk = poly(A, A); tk = time.perf_counter() - t0  # the kernel
    print(f"64 pixels -> {F.shape[1]} explicit features for (x.z + 1)^2; largest gap between the two 500 x 500 matrices {np.abs(Ke - Kk).max():.1e} "
          f"(values up to {Kk.max():.0f})")
    print(f"time: explicit features then dot products {te * 1000:.1f} ms; the kernel {tk * 1000:.1f} ms (one run, a rough comparison)")
    for deg in (2, 3, 5):
        from math import comb
        print(f"  degree {deg}: the explicit map needs {comb(64 + deg, deg):,} features; the kernel needs 64")
    print("the RBF kernel corresponds to infinitely many features, so only the kernel form can be computed exactly")
    print("Actual output from this lesson's run.")


def demo_valid():
    print("\n" + "=" * 70)
    print("2. WHAT MAKES A VALID KERNEL -- no negative eigenvalues")
    print("=" * 70)
    X, _ = digits(); A = X[:300]
    for name, K in (("linear", linear(A, A)), ("polynomial, degree 2", poly(A, A)), ("RBF, gamma 0.03", rbf(A, A, 0.03)),
                    ("tanh(x.z - 1), a popular non-kernel", np.tanh(linear(A, A) - 1)), ("minus the distance |x - z|", -np.sqrt(np.maximum(sqdist(A, A), 0)))):
        e = np.linalg.eigvalsh((K + K.T) / 2)               # eigenvalues of the symmetric matrix
        print(f"  {name:<38}: smallest eigenvalue {e.min():+.3e}, largest {e.max():+.3e}, negative ones {int((e < -1e-8 * e.max()).sum())}")
    print("a kernel matrix must be positive semi-definite (no negative eigenvalues): then it is a table of real dot products, and the dual problem has one best answer")
    print("Actual output from this lesson's run.")


def demo_picture(X, y):
    print("\n" + "=" * 70)
    print("3. A REAL CURVED BOUNDARY -- breast cancer, two measurements, gamma from small to large")
    print("=" * 70)
    names = list(load_breast_cancer().feature_names); f = [names.index("worst radius"), names.index("worst concave points")]
    tr, te = stratified_split(y, 0.3, 0); A, B = standardise(X[tr][:, f], X[te][:, f]); yt = y[tr]
    print(f"{len(tr)} training, {len(te)} test tumours; 'worst radius' and 'worst concave points', standardised; C = 1")
    print(f"{'kernel':<18}{'support':>9}{'train acc':>11}{'test acc':>10}")
    out = []
    for name, kf in [("linear", linear)] + [(f"RBF gamma {g}", lambda P, Q, g=g: rbf(P, Q, g)) for g in GAMMAS]:
        a, b, _ = fit_kernel_svm(kf(A, A), yt, 1.0)
        tr_acc = np.mean(np.sign(kf(A, A).T @ (a * yt) + b) == yt); te_acc = np.mean(np.sign(kf(A, B).T @ (a * yt) + b) == y[te])
        out.append((name, kf, a, b)); print(f"{name:<18}{int((a > 1e-8).sum()):>9}{tr_acc:>11.4f}{te_acc:>10.4f}")
    sk = SVC(kernel="rbf", gamma=1.0, C=1.0, tol=1e-6).fit(A, yt); a, b = out[3][2], out[3][3]
    print(f"outside check, RBF gamma 1: test predictions that agree with scikit-learn's SVC {np.mean(np.sign(rbf(A, B, 1.0).T @ (a * yt) + b) == sk.predict(B)):.4f}")
    print("Actual output from this lesson's run.")
    return A, yt, out                                        # the plot uses them


def demo_where(X, y):
    print("\n" + "=" * 70)
    print(f"4. WHERE A KERNEL HELPS -- breast cancer (30 features) and digits (64 pixels, 10 classes), {SPLITS} splits")
    print("=" * 70)
    print("C = 1; RBF gamma = 1 / (features x their variance), a common default; digits by one-vs-rest")
    Xd, yd = digits(); res = {}
    for name, (Z, t) in (("Breast cancer", (X, y)), ("Digits", (Xd, yd))):
        r = []
        for s in range(SPLITS):
            tr, te = stratified_split(t, 0.3, s)
            A, B = standardise(Z[tr], Z[te]) if name == "Breast cancer" else (Z[tr], Z[te])
            g = 1.0 / (A.shape[1] * A.var())                  # scikit-learn's 'scale' default
            row = []
            for kf in (linear, lambda P, Q: poly(P, Q, 3), lambda P, Q: rbf(P, Q, g)):
                if name == "Breast cancer":
                    a, b, _ = fit_kernel_svm(kf(A, A), t[tr], 1.0); row.append(np.mean(np.sign(kf(A, B).T @ (a * t[tr]) + b) == t[te]))
                else:
                    m = ovr_fit(kf(A, A), t[tr], 1.0); row.append(np.mean(ovr_predict(m, t[tr], kf(A, B)) == t[te]))
            r.append(row)
        r = np.array(r); res[name] = r
        print(f"\n{name}: linear {r[:, 0].mean():.4f}, polynomial degree 3 {r[:, 1].mean():.4f}, RBF {r[:, 2].mean():.4f}")
        d = r[:, 2] - r[:, 0]; print(f"  RBF minus linear {d.mean():+.4f}; better on {int((d > 0).sum())}, worse on {int((d < 0).sum())} of {SPLITS}")
    print("for scale: on Day 107 a random forest scored 0.9712 on digits and 0.9596 on breast cancer, over its own splits")
    print("Actual output from this lesson's run.")
    return res


def demo_grid():
    print("\n" + "=" * 70)
    print("5. GAMMA AND C TOGETHER -- digits, one split, 3-fold CV on the training part")
    print("=" * 70)
    X, y = digits(); tr, te = stratified_split(y, 0.3, 0); folds = np.array_split(np.random.RandomState(0).permutation(tr), 3)
    S = np.zeros((len(GRID_G), len(GRID_C))); SV = np.zeros_like(S)
    for i, g in enumerate(GRID_G):
        for j, C in enumerate(GRID_C):
            acc = []
            for k in range(3):
                va = folds[k]; ft = np.concatenate([folds[m] for m in range(3) if m != k])
                m = ovr_fit(rbf(X[ft], X[ft], g), y[ft], C); acc.append(np.mean(ovr_predict(m, y[ft], rbf(X[ft], X[va], g)) == y[va]))
                SV[i, j] += np.mean([np.mean(a > 1e-8) for a, _ in m]) / 3  # share of rows that are support vectors
            S[i, j] = np.mean(acc)
    print(f"{'CV accuracy':<14}" + "".join(f"{'C ' + str(c):>10}" for c in GRID_C))
    for i, g in enumerate(GRID_G):
        print(f"{'gamma ' + str(g):<14}" + "".join(f"{S[i, j]:>10.4f}" for j in range(len(GRID_C))))
    print(f"{'support share':<14}" + "".join(f"{'C ' + str(c):>10}" for c in GRID_C))
    for i, g in enumerate(GRID_G):
        print(f"{'gamma ' + str(g):<14}" + "".join(f"{SV[i, j]:>10.3f}" for j in range(len(GRID_C))))
    i, j = np.unravel_index(np.argmax(S), S.shape); g, C = GRID_G[i], GRID_C[j]
    m = ovr_fit(rbf(X[tr], X[tr], g), y[tr], C)
    print(f"best by CV: gamma {g}, C {C}; test accuracy {np.mean(ovr_predict(m, y[tr], rbf(X[tr], X[te], g)) == y[te]):.4f}")
    print("Actual output from this lesson's run.")
    return S                                                 # the plot uses it


def demo_rff():
    print("\n" + "=" * 70)
    print("6. RANDOM FOURIER FEATURES -- the RBF kernel without the n-by-n matrix")
    print("=" * 70)
    X, y = digits(); g = 0.03; tr, te = stratified_split(y, 0.3, 0)
    K = rbf(X[:400], X[:400], g)
    print(f"digits, gamma {g}; approximation error on a 400 x 400 kernel matrix, and test accuracy of a LINEAR SVM on the random features (C = 1, {SPLITS} splits)")
    exact = []
    for s in range(SPLITS):
        tr, te = stratified_split(y, 0.3, s); m = ovr_fit(rbf(X[tr], X[tr], g), y[tr], 1.0)
        exact.append(np.mean(ovr_predict(m, y[tr], rbf(X[tr], X[te], g)) == y[te]))
    print(f"{'random features':>16}{'mean abs error':>16}{'accuracy':>10}")
    out = []
    for D in DIMS:
        z = rff(X, D, g, 0); Z = z(X[:400]); err = np.abs(Z @ Z.T - K).mean()
        acc = []
        for s in range(SPLITS):
            tr, te = stratified_split(y, 0.3, s); z = rff(X, D, g, s); A, B = z(X[tr]), z(X[te])
            m = ovr_fit(linear(A, A), y[tr], 1.0); acc.append(np.mean(ovr_predict(m, y[tr], linear(A, B)) == y[te]))
        out.append((D, err, np.mean(acc)))
        print(f"{D:>16}{err:>16.4f}{np.mean(acc):>10.4f}")
    print(f"the exact RBF kernel on the same splits: {np.mean(exact):.4f}")
    print("Actual output from this lesson's run.")
    return out, float(np.mean(exact))                         # the plot uses them


def plot_results(A, y, fits, rff_rows, exact):          # the boundaries, and random features
    fig = plt.figure(figsize=(16, 7.6)); gx, gy = np.meshgrid(np.linspace(-2.5, 4, 220), np.linspace(-2.2, 3.2, 220))
    G = np.column_stack([gx.ravel(), gy.ravel()])        # a grid of points to colour
    for k, (name, kf, a, b) in enumerate(fits[:6]):
        ax = fig.add_subplot(2, 4, k + 1 if k < 3 else k + 2)
        f = (kf(A, G).T @ (a * y) + b).reshape(gx.shape)
        ax.contourf(gx, gy, f > 0, levels=[-0.5, 0.5, 1.5], colors=["#f4c7c3", "#c6dbef"], alpha=0.8)
        ax.contour(gx, gy, f, levels=[0], colors="k", linewidths=1)
        ax.scatter(A[y < 0, 0], A[y < 0, 1], s=5, c="#d62728"); ax.scatter(A[y > 0, 0], A[y > 0, 1], s=5, c="#1f77b4")
        ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 4, 4); d = [r[0] for r in rff_rows]
    ax.plot(d, [r[2] for r in rff_rows], marker="o", label="linear SVM on random features"); ax.axhline(exact, color="k", linestyle="--", label="exact RBF kernel")
    ax.set_xscale("log"); ax.set_xlabel("random features"); ax.set_ylabel("digits test accuracy"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Approximating the RBF kernel", fontsize=10)
    fig.text(0.37, 0.005, "breast cancer, worst radius (across) and worst concave points (up), standardised; red malignant, blue benign", ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig("kernel_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved kernel_results.png")


def draw_diagram():                                     # the trick in one picture
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 113: the kernel trick", ha="center", fontsize=12, fontweight="bold")
    box(1.8, 3.1, 3.0, 1.1, "the long way: map every point\nto many features phi(x)", "#FAEEDA")
    box(6.5, 3.1, 3.4, 1.1, "dot products phi(x).phi(z):\nn x n numbers", "#DCEBF7")
    box(1.8, 1.1, 3.0, 1.1, "the trick: compute\nK(x, z) directly", "#FAEEDA")
    box(6.5, 1.1, 3.4, 1.1, "the same n x n numbers,\nfrom the original features", "#DCEBF7")
    box(11.2, 2.1, 3.2, 1.6, "Day 112's SMO only\never reads this table:\ncurved boundaries,\nsame solver", "#E5F2E1")
    ax.annotate("", xy=(4.76, 3.1), xytext=(3.34, 3.1), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(4.76, 1.1), xytext=(3.34, 1.1), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(9.56, 2.4), xytext=(8.24, 3.0), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(9.56, 1.8), xytext=(8.24, 1.2), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 2.1, "equal (section 1)", ha="center", fontsize=8.5, style="italic")
    plt.savefig("kernel_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved kernel_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 114 is Naive Bayes from scratch, on real text: a model that counts words, assumes they are independent, and is still hard to beat on small data.")


def main():                                             # runs every section in order
    bc = load_breast_cancer(); X, y = bc.data, np.where(bc.target == 1, 1.0, -1.0)  # benign +1, malignant -1
    demo_trick()                                           # 1. the trick
    demo_valid()                                           # 2. valid kernels
    A, yt, fits = demo_picture(X, y)                       # 3. a curved boundary
    demo_where(X, y)                                       # 4. where kernels help
    demo_grid()                                            # 5. gamma and C
    rows, exact = demo_rff()                               # 6. random features
    plot_results(A, yt, fits, rows, exact)                 # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 7. Day 114


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
