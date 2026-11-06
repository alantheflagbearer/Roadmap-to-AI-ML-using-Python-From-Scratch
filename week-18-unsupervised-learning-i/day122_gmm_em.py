"""
Day 122: Gaussian Mixture Models and the EM Algorithm From Scratch

k-means gives every point to exactly one centre and assumes round clusters of
similar size. A Gaussian mixture model (GMM) says the data came from k bell-
shaped clouds, each with its own centre, shape (covariance) and share of the
points (weight). Every point gets a PROBABILITY of belonging to each cloud.
The parameters are fitted by expectation-maximisation (EM): guess the
memberships from the current clouds (E step), refit the clouds from the
memberships (M step), repeat. Each round can only raise the likelihood.

Everything is written from scratch in NumPy:
  * the Gaussian log-density through a Cholesky factor, and log-sum-exp
  * the E step and the M step, with full, diagonal or spherical covariances
  * EM started from Day 119's k-means, and the Bayesian information criterion (BIC)

The data, labels used only to judge: iris and wine (standardised), digits, and
shapes made on purpose so the true groups are known exactly.

  1. By hand: one EM step on six points.
  2. Correctness: the likelihood never falls, and scikit-learn agrees.
  3. Where a mixture beats k-means.
  4. Soft assignments: how sure is each point?
  5. Choosing k with BIC.
  6. Covariance shapes: full, diagonal, spherical.
  7. When a cloud collapses: digits and regularisation.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Days 119 and 121

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.mixture import GaussianMixture            # used ONLY as an outside check
from sklearn.datasets import load_digits, load_iris, load_wine  # bundled datasets
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import Ellipse, FancyBboxPatch  # ellipses for the clouds, rounded boxes for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Days 119 and 121 sit next to this file
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing
    import day119_k_means as d119                       # Day 119: kmeans, ari, standardise
    import day121_dbscan as d121                        # Day 121: the shapes
ari, standardise = d119.ari, d119.standardise           # unchanged
LOG2PI = np.log(2 * np.pi)                              # a constant of the Gaussian density


# =============================================================================
# GAUSSIAN MIXTURES -- NEW
# =============================================================================
def log_gauss(X, mu, S):                                # NEW: log N(x | mu, S) for every row, through the Cholesky factor S = L L^T
    L = np.linalg.cholesky(S)                             # fails if S is not positive definite (section 7)
    Z = np.linalg.solve(L, (X - mu).T)                    # L z = x - mu, so |z|^2 = (x - mu)^T S^-1 (x - mu)
    return -0.5 * (Z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * LOG2PI  # log det S = 2 sum log diag L


def logsumexp(A):                                       # NEW: log(sum(exp(A))) along each row, without overflow
    m = A.max(1, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(1, keepdims=True)))[:, 0]


def e_step(X, w, M, S):                                 # NEW: responsibilities R[i, j] = P(cloud j | x_i), and the log-likelihood
    logp = np.stack([np.log(w[j]) + log_gauss(X, M[j], S[j]) for j in range(len(w))], 1)  # log w_j N(x_i | j)
    ll = logsumexp(logp)                                  # log p(x_i), summed over clouds
    return np.exp(logp - ll[:, None]), float(ll.sum())


def m_step(X, R, kind="full", reg=1e-6):                # NEW: weights, means and covariances from the responsibilities
    Nk = R.sum(0) + 1e-300; w = Nk / len(X); M = (R.T @ X) / Nk[:, None]; d = X.shape[1]; S = []
    for j in range(R.shape[1]):
        D = X - M[j]; Sj = (R[:, j, None] * D).T @ D / Nk[j]  # the responsibility-weighted covariance
        if kind == "diag":
            Sj = np.diag(np.diag(Sj))                       # variances only: axis-aligned ellipses
        elif kind == "spherical":
            Sj = np.eye(d) * np.trace(Sj) / d               # one variance: circles, close to a soft k-means
        S.append(Sj + reg * np.eye(d))                      # a small ridge keeps S invertible (section 7)
    return w, M, np.array(S)


def gmm(X, k, seed, kind="full", reg=1e-6, iters=500, tol=1e-6):  # NEW: EM from Day 119's k-means labels
    lab = d119.kmeans(X, k, seed)[0]; R = np.eye(k)[lab]  # hard memberships to start
    w, M, S = m_step(X, R, kind, reg); history = []
    for _ in range(iters):
        R, ll = e_step(X, w, M, S); history.append(ll)
        if len(history) > 1 and history[-1] - history[-2] < tol * len(X):
            break                                           # the mean log-likelihood rose by less than tol
        w, M, S = m_step(X, R, kind, reg)
    return {"w": w, "M": M, "S": S, "R": R, "history": history, "init": lab}


def n_params(k, d, kind):                               # NEW: free numbers in the model
    cov = {"full": d * (d + 1) // 2, "diag": d, "spherical": 1}[kind]
    return (k - 1) + k * d + k * cov                      # weights (they sum to 1), means, covariances


def bic(X, model, kind):                                # NEW: -2 log-likelihood + parameters x log n; lower is better
    k, d = model["M"].shape
    return -2 * model["history"][-1] + n_params(k, d, kind) * np.log(len(X))


def datasets():                                         # three labelled datasets, as on Days 119 and 120
    ir, wi, dg = load_iris(), load_wine(), load_digits()
    return {"iris": (standardise(ir.data), ir.target), "wine": (standardise(wi.data), wi.target), "digits": (dg.data, dg.target)}


def match(lab, y):                                      # each cluster named by its most common true class: the share of points it gets right
    m = np.array([np.bincount(y[lab == c], minlength=y.max() + 1).argmax() if np.any(lab == c) else -1 for c in range(lab.max() + 1)])
    return m[lab] == y


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- one EM step on six points: 0, 1, 2, 6, 7, 8")
    print("=" * 70)
    X = np.array([[0.0], [1.0], [2.0], [6.0], [7.0], [8.0]])
    w, M, S = np.array([0.5, 0.5]), np.array([[1.0], [5.0]]), np.array([[[1.0]], [[1.0]]])  # a poor start: the right cloud at 5
    print("start: weights 0.5 and 0.5, means 1 and 5, variances 1 and 1")
    R, ll = e_step(X, w, M, S)
    print(f"E step: P(left cloud | x) = {np.round(R[:, 0], 4).tolist()}; log-likelihood {ll:.4f}")
    print(f"by hand for x = 2: exp(-(2-1)^2/2) against exp(-(2-5)^2/2), so 1 / (1 + exp(-4)) = {1 / (1 + np.exp(-4)):.4f}")
    w, M, S = m_step(X, R, reg=0.0)
    print(f"M step: weights {np.round(w, 4).tolist()}, means {np.round(M[:, 0], 4).tolist()}, variances {np.round(S[:, 0, 0], 4).tolist()}")
    R, ll2 = e_step(X, w, M, S)
    print(f"the next E step: log-likelihood {ll2:.4f}, up from {ll:.4f}; the right cloud moved from 5 to the middle of 6, 7 and 8")
    print("Actual output from this lesson's run.")


def demo_check(sets):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- the log-likelihood never falls, and scikit-learn agrees")
    print("=" * 70)
    falls = steps = 0
    for name in ("iris", "wine"):
        X, y = sets[name]
        for s in range(10):
            for kind in ("full", "diag", "spherical"):
                h = gmm(X, 3, s, kind, tol=0)["history"]; steps += len(h) - 1; falls += int(np.sum(np.diff(h) < -1e-9))
    print(f"iris and wine, 3 covariance types, 10 seeds each: {steps} EM steps; steps where the log-likelihood fell: {falls}")
    X, y = sets["iris"]; lab = d119.kmeans(X, 3, 0)[0]; w0, M0, S0 = m_step(X, np.eye(3)[lab])  # the same starting clouds for both
    ours = gmm(X, 3, 0, tol=1e-10, iters=2000)
    sk = GaussianMixture(3, covariance_type="full", reg_covar=1e-6, weights_init=w0, means_init=M0, precisions_init=np.linalg.inv(S0),
                         tol=1e-10, max_iter=2000).fit(X)
    print(f"iris, full covariances, the same start: mean log-likelihood {ours['history'][-1] / len(X):.6f} here, {sk.score(X):.6f} in scikit-learn; "
          f"largest gap between the means {np.abs(np.sort(ours['M'], 0) - np.sort(sk.means_, 0)).max():.1e}; "
          f"labels agree (ARI) {ari(ours['R'].argmax(1), sk.predict(X)):.4f}")
    print("Actual output from this lesson's run.")


def demo_vs_kmeans(sets):
    print("\n" + "=" * 70)
    print("3. WHERE A MIXTURE BEATS k-MEANS -- full covariances, k = the true number of groups")
    print("=" * 70)
    shp = d121.shapes(); rows = {}
    for name, (X, y) in [("iris", sets["iris"]), ("wine", sets["wine"]), ("wide + tight", shp["wide + tight"]), ("two moons", shp["two moons"])]:
        k = len(np.unique(y)); m = gmm(X, k, 0); km = d119.kmeans(X, k, 0)[0]; lab = m["R"].argmax(1)
        rows[name] = (ari(lab, y), ari(km, y))
        print(f"  {name:<13}: ARI mixture {ari(lab, y):.4f}, k-means {ari(km, y):.4f}; cluster sizes mixture {sorted(np.bincount(lab, minlength=k).tolist(), reverse=True)}, "
              f"k-means {sorted(np.bincount(km, minlength=k).tolist(), reverse=True)}")
    print("each mixture cloud has its own size and shape, so a wide group and a tight one can sit side by side; moons are not ellipses at all")
    print("Actual output from this lesson's run.")
    return rows, shp


def demo_soft(sets):
    print("\n" + "=" * 70)
    print("4. SOFT ASSIGNMENTS -- how sure is each point? iris, full covariances")
    print("=" * 70)
    X, y = sets["iris"]; m = gmm(X, 3, 0); p = m["R"].max(1); right = match(m["R"].argmax(1), y)
    for lo, hi in ((0.99, 1.01), (0.9, 0.99), (0.0, 0.9)):
        g = (p >= lo) & (p < hi)
        print(f"  largest membership in [{lo:.2f}, {min(hi, 1.0):.2f}{']' if hi > 1 else ')'}: {int(g.sum()):>3} flowers; named correctly {int(right[g].sum())} "
              f"({right[g].mean() if g.any() else 0:.3f})")
    print("the less sure the mixture is about a flower, the more often it is wrong: its memberships say which assignments to trust")
    print("Actual output from this lesson's run.")
    return m


def demo_bic(sets):
    print("\n" + "=" * 70)
    print("5. CHOOSING k WITH BIC -- k = 1 to 8; lower is better")
    print("=" * 70)
    print("BIC = -2 log-likelihood + (number of parameters) x log n: a better fit, minus a price for every extra parameter")
    out = {}
    for name in ("iris", "wine"):
        X, y = sets[name]
        for kind in ("full", "diag"):
            b = [bic(X, gmm(X, k, 0, kind), kind) for k in range(1, 9)]; out[(name, kind)] = b
            print(f"  {name:<5} {kind:<5}: best k {int(np.argmin(b)) + 1} (true {len(np.unique(y))}); BIC " + ", ".join(f"{v:.0f}" for v in b))
    print(f"iris, full covariances: {n_params(3, 4, 'full')} parameters at k = 3; wine: {n_params(3, 13, 'full')} (13 features, 178 rows)")
    print("Actual output from this lesson's run.")
    return out


def demo_shapes(sets):
    print("\n" + "=" * 70)
    print("6. COVARIANCE SHAPES -- full, diagonal, spherical (k = the true number of groups)")
    print("=" * 70)
    print(f"  {'data':<7}{'covariance':<12}{'ARI':>8}{'BIC':>10}{'parameters':>12}")
    for name in ("iris", "wine"):
        X, y = sets[name]; k = len(np.unique(y))
        for kind in ("full", "diag", "spherical"):
            m = gmm(X, k, 0, kind)
            print(f"  {name:<7}{kind:<12}{ari(m['R'].argmax(1), y):>8.4f}{bic(X, m, kind):>10.0f}{n_params(k, X.shape[1], kind):>12}")
    print(f"  Day 119's k-means: iris {ari(d119.kmeans(sets['iris'][0], 3, 0)[0], sets['iris'][1]):.4f}, wine {ari(d119.kmeans(sets['wine'][0], 3, 0)[0], sets['wine'][1]):.4f}")
    print("spherical clouds with soft memberships behave much like k-means; full ones can tilt and stretch")
    print("Actual output from this lesson's run.")


def demo_digits(sets):
    print("\n" + "=" * 70)
    print("7. WHEN A CLOUD COLLAPSES -- digits, 64 pixels, full covariances, k = 10")
    print("=" * 70)
    X, y = sets["digits"]; sd = X.std(0)
    print(f"{int((sd == 0).sum())} pixels are 0 in every image, so their variance is 0 and no covariance can be inverted")
    try:
        gmm(X, 10, 0, "full", reg=0.0)
    except np.linalg.LinAlgError as e:
        print(f"  ridge 0: the Cholesky factor fails ('{e}')")
    tr = np.random.RandomState(0).permutation(len(X)); te, tr = tr[:540], tr[540:]  # 30% held out to judge the ridge
    print(f"  {'ridge':>7}{'ARI':>8}{'held-out log-likelihood per image':>36}")
    rows = []
    for reg in (1e-6, 1e-2, 1.0, 10.0, 100.0):
        m = gmm(X, 10, 0, "full", reg); h = gmm(X[tr], 10, 0, "full", reg); ll = e_step(X[te], h["w"], h["M"], h["S"])[1] / len(te)
        rows.append((reg, ari(m["R"].argmax(1), y), ll))
        print(f"  {reg:>7g}{rows[-1][1]:>8.4f}{ll:>36.2f}")
    print(f"Day 119's k-means scored 0.6525; the ridge adds the same small variance to every pixel, a little like blurring each cloud")
    print("Actual output from this lesson's run.")
    return rows


def ellipse(ax, mu, S, color):                          # a 2-standard-deviation ellipse for a 2-D Gaussian
    vals, vecs = np.linalg.eigh(S); ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang, fill=False, color=color, linewidth=1.5))


def plot_results(sets, shp, soft, bics, rows):          # six panels
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.6))             # canvas
    X, y = shp["wide + tight"]; m = gmm(X, 2, 0); km = d119.kmeans(X, 2, 0)[0]
    for a, lab, title in ((ax[0, 0], km, "wide + tight: k-means"), (ax[0, 1], m["R"].argmax(1), "wide + tight: mixture, 2-sd ellipses")):
        a.scatter(X[:, 0], X[:, 1], c=lab, cmap="coolwarm", s=6); a.set_aspect("equal"); a.set_title(title, fontsize=10)
    for j, c in enumerate(("#3b4cc0", "#b40426")):
        ellipse(ax[0, 1], m["M"][j], m["S"][j], c)
    a = ax[0, 2]; Xi = sets["iris"][0]; p = soft["R"].max(1)
    sc = a.scatter(Xi[:, 2], Xi[:, 3], c=p, cmap="viridis", s=14); fig.colorbar(sc, ax=a, label="largest membership")
    a.set_xlabel("petal length (standardised)"); a.set_ylabel("petal width (standardised)"); a.set_title("Iris: how sure the mixture is", fontsize=10)
    a = ax[1, 0]
    for (name, kind), b in bics.items():
        a.plot(range(1, 9), b, marker="o", label=f"{name}, {kind}")
    a.set_xlabel("k"); a.set_ylabel("BIC (lower is better)"); a.legend(fontsize=8); a.grid(alpha=0.3); a.set_title("BIC against k", fontsize=10)
    a = ax[1, 1]; h = gmm(sets["iris"][0], 3, 0, tol=0, iters=60)["history"]
    a.plot(np.arange(1, len(h) + 1), np.array(h) / len(sets["iris"][0]), marker=".")
    a.set_xlabel("EM step"); a.set_ylabel("mean log-likelihood"); a.grid(alpha=0.3); a.set_title("Iris: EM never goes down", fontsize=10)
    a = ax[1, 2]; r = np.array(rows); a.semilogx(r[:, 0], r[:, 1], marker="o", label="ARI with the digits")
    a.axhline(0.6525, color="k", linestyle=":", linewidth=0.8, label="Day 119's k-means"); a.set_xlabel("ridge added to every variance")
    a.set_ylabel("ARI"); b = a.twinx(); b.semilogx(r[:, 0], r[:, 2], marker="s", color="#d62728", label="held-out log-likelihood")
    b.set_ylabel("held-out log-likelihood per image"); a.legend(loc="lower left", fontsize=8); b.legend(loc="lower right", fontsize=8)
    a.set_title("Digits: full covariances need a ridge", fontsize=10); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("gmm_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved gmm_results.png")


def draw_diagram():                                     # the EM loop
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 122: a Gaussian mixture, fitted by EM", ha="center", fontsize=12, fontweight="bold")
    box(1.8, 2.5, 2.8, 1.6, "start: k clouds\nfrom k-means\n(weight, mean, covariance)", "#FAEEDA")
    box(5.4, 2.5, 3.2, 1.6, "E step: each point's\nprobability of each cloud\n(responsibilities)", "#DCEBF7")
    box(9.0, 2.5, 3.0, 1.6, "M step: refit each cloud\nfrom the points,\nweighted by responsibility", "#E5F2E1")
    box(11.9, 2.5, 1.8, 1.6, "repeat until\nthe likelihood\nstops rising", "#EEEDFE")
    for x0, x1 in ((3.24, 3.76), (7.04, 7.46), (10.54, 10.96)):
        ax.annotate("", xy=(x1, 2.5), xytext=(x0, 2.5), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(5.4, 1.65), xytext=(9.0, 1.65), arrowprops=dict(arrowstyle="->", color="#333333", connectionstyle="arc3,rad=-0.3"))
    ax.text(6.5, 0.55, "each step can only raise the likelihood; k-means is the special case of round, equal clouds and hard memberships", ha="center", fontsize=8.5)
    plt.savefig("gmm_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved gmm_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 123 is principal component analysis from scratch: the directions of greatest variance, with a real variance-explained demonstration.")


def main():                                             # runs every section in order
    sets = datasets()
    demo_hand()                                            # 1. by hand
    demo_check(sets)                                       # 2. correctness
    rows3, shp = demo_vs_kmeans(sets)                      # 3. against k-means
    soft = demo_soft(sets)                                 # 4. soft assignments
    bics = demo_bic(sets)                                  # 5. BIC
    demo_shapes(sets)                                      # 6. covariance shapes
    rows = demo_digits(sets)                               # 7. digits and the ridge
    plot_results(sets, shp, soft, bics, rows)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
