"""
Day 133: Generative vs Discriminative Models -- a Real Conceptual Comparison

A DISCRIMINATIVE classifier learns p(class | x) directly: where the boundary
between the classes runs, and nothing else. A GENERATIVE classifier learns how
the data is made, p(x | class) and p(class), and gets p(class | x) from Bayes'
rule. Day 114 compared the two on text (naive Bayes against logistic
regression). Today they are compared on images, with Gaussian class models,
and on the things only a generative model can do at all.

Everything is written from scratch in NumPy:
  * Gaussian class models: naive Bayes (diagonal), LDA (one shared
    covariance) and QDA (one covariance per class), through Cholesky factors
  * the proof, run as code, that LDA's posterior is a linear softmax: the
    same form logistic regression fits directly (Day 116's fit_softmax)
  * learning curves: who wins with little data, and who wins with a lot
  * what only the generative model can do: draw new images, classify with
    half of each image missing, and say how ordinary an image is (log p(x))

Data: MNIST (cached by Day 126), 60,000 training and 10,000 test images; and
scikit-learn's bundled breast-cancer table (569 tumours, 30 measurements).

  1. By hand: two 1-D Gaussians give a logistic posterior.
  2. The same family, fitted two ways: LDA against logistic regression.
  3. Learning curves: little data against a lot.
  4. Generating: drawing digits from p(x | class).
  5. Missing pixels: classifying with half of each image hidden.
  6. Knowing what it does not know: log p(x) against the classifier's confidence.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier parts
import time                                            # time: what each method costs

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day123_exercise as d123                      # Day 123: pca_fit, transform
    import day117_exercise as d117                      # Day 117: roc_auc
    import day116_exercise as d116                      # Day 116: softmax, fit_softmax (softmax regression)


# =============================================================================
# GENERATIVE: A GAUSSIAN FOR EACH CLASS -- NEW
# =============================================================================
def fit_gaussians(X, y, k, kind="shared", reg=1e-3):    # NEW: p(x | class) as a Gaussian per class, and p(class)
    d = X.shape[1]; mu = np.array([X[y == c].mean(0) for c in range(k)])            # one mean per class
    prior = np.bincount(y, minlength=k) / len(y)                                      # how common each class is
    if kind == "shared":                                                              # LDA: one covariance for every class
        D = X - mu[y]; S = D.T @ D / len(X) + reg * np.eye(d)                         # spread around each point's own class mean
        cov = np.repeat(S[None], k, 0)                                                # the same matrix for every class
    elif kind == "diagonal":                                                          # naive Bayes: features independent within a class
        cov = np.array([np.diag(X[y == c].var(0) + reg) for c in range(k)])          # only each feature's own variance
    else:                                                                             # QDA: each class its own covariance
        cov = np.array([np.cov(X[y == c].T, bias=True).reshape(d, d) + reg * np.eye(d) for c in range(k)])
    return {"mu": mu, "cov": cov, "prior": prior}                                     # reg keeps every covariance invertible


def log_gauss(X, mu, S):                                # NEW: log N(x; mu, S) for every row, through a Cholesky factor
    L = np.linalg.cholesky(S); Z = np.linalg.solve(L, (X - mu).T)                     # L L^T = S, so |Z|^2 is the Mahalanobis distance
    return -0.5 * ((Z ** 2).sum(0) + 2 * np.log(np.diag(L)).sum() + len(mu) * np.log(2 * np.pi))  # log det S = 2 sum log diag L


def log_joint(model, X, keep=None):                     # NEW: log p(x, class) for each class; keep: only these features are seen
    idx = np.arange(X.shape[1]) if keep is None else keep                              # a Gaussian's marginal: just drop the rest
    return np.stack([log_gauss(X[:, idx], m[idx], S[np.ix_(idx, idx)]) + np.log(p)    # log p(x | c) + log p(c)
                     for m, S, p in zip(model["mu"], model["cov"], model["prior"])], 1)


def posterior(model, X, keep=None):                     # NEW: Bayes' rule, p(class | x) = p(x, class) / p(x)
    J = log_joint(model, X, keep); J -= J.max(1, keepdims=True); P = np.exp(J)        # subtract the largest first: no overflow
    return P / P.sum(1, keepdims=True)                                                # dividing by the sum is dividing by p(x)


def log_px(model, X):                                   # NEW: log p(x) = log sum over classes of p(x, class): how ordinary x is
    J = log_joint(model, X); m = J.max(1)                                             # log-sum-exp, kept stable
    return m + np.log(np.exp(J - m[:, None]).sum(1))


def lda_linear(model):                                  # NEW: a shared-covariance model's posterior is a softmax of linear scores
    Si = np.linalg.inv(model["cov"][0]); W = Si @ model["mu"].T                        # w_c = S^-1 mu_c
    b = -0.5 * np.einsum("cd,dc->c", model["mu"], W) + np.log(model["prior"])          # b_c = -mu_c S^-1 mu_c / 2 + log p(c)
    return W, b


def sample_class(model, c, n, rng):                     # NEW: new data, drawn from p(x | class c)
    L = np.linalg.cholesky(model["cov"][c])                                           # any matrix with L L^T = S will do
    return model["mu"][c] + rng.standard_normal((n, len(L))) @ L.T                     # x = mu + L eps has covariance L L^T = S


# =============================================================================
# DISCRIMINATIVE: p(class | x) DIRECTLY -- Day 116's softmax regression
# =============================================================================
def fit_logistic(X, y, k, lam=1e-3, steps=500):        # NEW: Day 116's fit_softmax, on standardised features
    mu, sd = X.mean(0), np.maximum(X.std(0), 0.05)                                    # scale each feature; the floor stops a pixel that never
                                                                                      # varied in training from being multiplied a million-fold later
    W, b = d116.fit_softmax((X - mu) / sd, y, k, lam=lam, steps=steps)                 # Adam on the cross-entropy, Day 116 unchanged
    return lambda Z: d116.softmax(((Z - mu) / sd) @ W + b)                             # a function: features in, probabilities out


def learning_curve(fits, X, y, Xte, yte, sizes, draws, seed=0):  # NEW: test error of each method against training-set size
    rng = np.random.RandomState(seed); k = int(y.max()) + 1                           # the same random subsets for every method
    err = {m: np.zeros((len(sizes), draws)) for m in fits}                            # method -> size x draw
    for i, n in enumerate(sizes):
        for r in range(draws):
            while True:                                                               # every class present at least twice
                tr = rng.choice(len(X), n, replace=False)
                if np.bincount(y[tr], minlength=k).min() >= 2:
                    break
            for m, fit in fits.items():                                               # each method on the same rows
                err[m][i, r] = np.mean(fit(X[tr], y[tr], k)(Xte).argmax(1) != yte)
    return err


# =============================================================================
# DEMOS
# =============================================================================
def gen_fit(kind, reg):                                 # a generative classifier in learning_curve's form: fit(X, y, k) -> proba
    return lambda X, y, k: (lambda m: (lambda Z: posterior(m, Z)))(fit_gaussians(X, y, k, kind, reg))


def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- two 1-D Gaussians with the same spread give a logistic posterior")
    print("=" * 70)
    X = np.array([[0.0], [1.0], [2.0], [4.0], [5.0], [6.0]]); y = np.array([0, 0, 0, 1, 1, 1])  # two classes on a line
    m = fit_gaussians(X, y, 2, "shared", reg=0.0)                                      # no regularisation: the exact estimates
    s2 = m["cov"][0][0, 0]; mu0, mu1 = m["mu"][:, 0]                                   # the shared variance and the two means
    print(f"means {mu0:.1f} and {mu1:.1f}; shared variance (1/6) x (1 + 0 + 1 + 1 + 0 + 1) = {s2:.4f}")
    w = (mu1 - mu0) / s2; b = (mu0 ** 2 - mu1 ** 2) / (2 * s2)                         # p(1 | x) = sigmoid(w x + b), with equal priors
    print(f"by hand: p(class 1 | x) = sigmoid(w x + b) with w = (mu1 - mu0) / s2 = {w:.4f}, b = (mu0^2 - mu1^2) / (2 s2) = {b:.4f}")
    xs = np.array([[1.0], [3.0], [3.5]]); P = posterior(m, xs)[:, 1]                   # Bayes' rule, through the Gaussians
    print(f"at x = 1, 3, 3.5: Bayes' rule gives {', '.join(f'{p:.4f}' for p in P)}; the sigmoid gives "
          f"{', '.join(f'{1 / (1 + np.exp(-(w * x + b))):.4f}' for x in xs[:, 0])}")
    print("the boundary (probability 0.5) is halfway between the means, at x = 3: a straight line, as in logistic regression")
    print("Actual output from this lesson's run.")


def demo_same_family(X, y):
    print("\n" + "=" * 70)
    print("2. THE SAME FAMILY, FITTED TWO WAYS -- LDA against logistic regression, on all 784 pixels")
    print("=" * 70)
    tr, te = np.arange(60000), np.arange(60000, 70000)                                # the standard split
    t0 = time.perf_counter(); lda = fit_gaussians(X[tr], y[tr], 10, "shared", reg=0.05); t_lda = time.perf_counter() - t0  # means, one covariance, priors
    P_lda = posterior(lda, X[te]); W, b = lda_linear(lda); lin = X[te] @ W + b         # Bayes' rule, and the same model as linear scores
    S = np.exp(lin - lin.max(1, keepdims=True)); S /= S.sum(1, keepdims=True)          # a softmax of the linear scores
    print(f"LDA fitted in {t_lda:.1f} s (means and one covariance: no iterations); test accuracy {np.mean(P_lda.argmax(1) == y[te]):.4f}")
    print(f"LDA's posterior as a softmax of linear scores x W + b: largest difference from Bayes' rule {np.abs(S - P_lda).max():.1e}")
    t0 = time.perf_counter(); lr = fit_logistic(X[tr], y[tr], 10); t_lr = time.perf_counter() - t0  # the same form, fitted directly
    P_lr = lr(X[te])
    print(f"logistic regression fitted in {t_lr:.1f} s (500 Adam steps); test accuracy {np.mean(P_lr.argmax(1) == y[te]):.4f}")
    for name, P in (("LDA", P_lda), ("logistic", P_lr)):                               # how well each states its confidence
        nll = -np.mean(np.log(P[np.arange(len(te)), y[te]] + 1e-12)); conf = P.max(1)
        print(f"   {name:9s} test log-loss {nll:.4f}; average confidence {conf.mean():.4f}; share of predictions above 0.99 confident {np.mean(conf > 0.99):.4f}")
    print(f"they disagree on {np.mean(P_lda.argmax(1) != P_lr.argmax(1)):.4f} of test images")
    print("Actual output from this lesson's run.")
    return lda, lr


def demo_curves(Z, y, Xb, yb):
    print("\n" + "=" * 70)
    print("3. LEARNING CURVES -- test error against training-set size")
    print("=" * 70)
    fits = {"naive Bayes": gen_fit("diagonal", 0.1), "LDA": gen_fit("shared", 0.1), "QDA": gen_fit("full", 1.0), "logistic": fit_logistic}
    sizes = [30, 100, 300, 1000, 3000, 10000, 60000]
    print("MNIST, 50 PCA numbers per image (PCA fitted on the 60,000 training images, without labels); 5 random subsets per size")
    t0 = time.perf_counter(); err = learning_curve(fits, Z[:60000], y[:60000], Z[60000:], y[60000:], sizes, 5)
    print(f"   {'training images':>15}" + "".join(f"{m:>13}" for m in fits) + "   LDA - logistic")
    for i, n in enumerate(sizes):
        print(f"   {n:>15,}" + "".join(f"{err[m][i].mean():>13.4f}" for m in fits) + f"   {err['LDA'][i].mean() - err['logistic'][i].mean():+.4f}")
    print(f"({time.perf_counter() - t0:.0f} s; the 60,000 row is one fit, repeated)")
    sizes_b = [10, 20, 40, 80, 160, 369]                                              # the breast-cancer table is small
    print("breast cancer: 30 measurements (log, then standardised by the training rows); 369 training and 200 test tumours; 20 random subsets per size")
    errb = learning_curve(fits, Xb[:369], yb[:369], Xb[369:], yb[369:], sizes_b, 20)
    print(f"   {'training rows':>15}" + "".join(f"{m:>13}" for m in fits) + "   NB - logistic")
    for i, n in enumerate(sizes_b):
        print(f"   {n:>15}" + "".join(f"{errb[m][i].mean():>13.4f}" for m in fits) + f"   {errb['naive Bayes'][i].mean() - errb['logistic'][i].mean():+.4f}")
    print("one test error on 200 tumours is 0.005; settings fixed at every size (generative reg 0.1, QDA 1.0; logistic lambda 0.001), not re-tuned")
    print("Actual output from this lesson's run.")
    return sizes, err, sizes_b, errb


def demo_generate(Z, y, mu, V, lr):
    print("\n" + "=" * 70)
    print("4. GENERATING -- drawing new digits from p(x | digit), in 50 PCA numbers, then back to pixels")
    print("=" * 70)
    want = np.repeat(np.arange(10), 100)                                              # 100 new images per digit
    for reg in (1.0, 0.001):                                                          # section 3's QDA setting, then a small one
        qda = fit_gaussians(Z[:60000], y[:60000], 10, "full", reg); rng = np.random.RandomState(0)  # one Gaussian per digit
        G = np.vstack([sample_class(qda, c, 100, rng) for c in range(10)])             # draw from p(x | digit)
        I = np.clip(d123.reconstruct(G, mu, V, 50), 0, 1)                              # back to 784 pixels (Day 123)
        got = lr(I).argmax(1)                                                         # does section 2's classifier agree?
        print(f"QDA with reg {reg}: 1,000 sampled images; section 2's logistic regression names the intended digit for {np.mean(got == want):.4f}")
        print("   per digit: " + ", ".join(f"{c} {np.mean(got[want == c] == c):.2f}" for c in range(10)))
    tv = [np.trace(np.cov(Z[:60000][y[:60000] == c].T)) for c in range(10)]          # each digit's own total variance in the 50 numbers
    print(f"each digit's own total variance in the 50 numbers: {', '.join(f'{c} {v:.1f}' for c, v in enumerate(tv))}; reg 1.0 adds 50 x 1.0 = 50.0")
    R = np.clip(d123.reconstruct(Z[60000:61000], mu, V, 50), 0, 1)                     # real test digits, through the same 50 numbers
    print(f"for comparison, 1,000 real test digits squeezed through the same 50 PCA numbers: {np.mean(lr(R).argmax(1) == y[60000:61000]):.4f}")
    print("a logistic regression has no p(x | digit) to draw from: it can only label images it is given")
    print("Actual output from this lesson's run.")
    return I


def demo_missing(X, y, lda, lr):
    print("\n" + "=" * 70)
    print("5. MISSING PIXELS -- classifying with half of every test image hidden")
    print("=" * 70)
    te = np.arange(60000, 70000); g = np.arange(784).reshape(28, 28)                  # pixel numbers as a 28 x 28 grid
    halves = {"top half seen": g[:14].ravel(), "bottom half seen": g[14:].ravel(), "left half seen": g[:, :14].ravel(), "right half seen": g[:, 14:].ravel()}
    mean = X[:60000].mean(0)                                                          # the average training image
    print(f"   {'pattern':18s}  LDA, hidden pixels left out  logistic, hidden = 0  logistic, hidden = average  logistic retrained on that half")
    rows = {}
    for name, keep in halves.items():
        a = np.mean(posterior(lda, X[te], keep).argmax(1) == y[te])                     # the Gaussian's marginal: no guessing needed
        Z0 = X[te].copy(); hid = np.setdiff1d(np.arange(784), keep); Z0[:, hid] = 0    # fill hidden pixels with black
        Zm = X[te].copy(); Zm[:, hid] = mean[hid]                                     # or with the average image
        b, c = np.mean(lr(Z0).argmax(1) == y[te]), np.mean(lr(Zm).argmax(1) == y[te])
        d = np.mean(fit_logistic(X[:60000][:, keep], y[:60000], 10)(X[te][:, keep]).argmax(1) == y[te])  # knowing the pattern in advance
        rows[name] = (a, b, c, d); print(f"   {name:18s}  {a:27.4f}  {b:20.4f}  {c:26.4f}  {d:31.4f}")
    print("the retrained column needs a new model for every pattern of missing pixels; LDA uses the one model it already has")
    print("Actual output from this lesson's run.")
    return rows


def demo_novelty(X, y, lda, lr):
    print("\n" + "=" * 70)
    print("6. KNOWING WHAT IT DOES NOT KNOW -- log p(x) against the classifier's confidence")
    print("=" * 70)
    rng = np.random.RandomState(1); te = X[60000:62000]                               # 2,000 normal test digits
    odd = {"digits upside down": te[:, ::-1].copy(), "random noise": rng.uniform(0, 1, te.shape).astype(np.float32),
           "two digits overlaid": np.maximum(te, te[::-1])}                          # three kinds of image that are not one digit
    lp_n = log_px(lda, te); conf_n = lr(te).max(1)                                    # the generative and discriminative signals
    print("score: generative, -log p(x) under section 2's LDA; discriminative, 1 - the logistic regression's top probability")
    print(f"   {'unusual images':20s}  AUC -log p(x)  AUC 1 - confidence  logistic's average confidence on them")
    out = {}
    for name, O in odd.items():                                                       # normal 0, unusual 1
        lab = np.r_[np.zeros(len(te)), np.ones(len(O))].astype(int)
        a = d117.roc_auc(lab, -np.r_[lp_n, log_px(lda, O)]); c_o = lr(O).max(1)
        b = d117.roc_auc(lab, 1 - np.r_[conf_n, c_o]); out[name] = (a, b, float(c_o.mean()))
        print(f"   {name:20s}  {a:13.4f}  {b:18.4f}  {c_o.mean():.4f}")
    print(f"for comparison, its average confidence on normal test digits: {conf_n.mean():.4f}")
    print("Actual output from this lesson's run.")
    return out


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(sizes, err, sizes_b, errb, I, rows, nov, conf_normal, out):  # six panels
    fig = plt.figure(figsize=(15, 9))
    for j, (sz, e, title) in enumerate(((sizes, err, "MNIST (50 PCA numbers)"), (sizes_b, errb, "breast cancer (30 measurements)"))):
        ax = fig.add_subplot(2, 3, j + 1)                                             # one learning-curve panel per data set
        for m, st in zip(e, ("o--", "s--", "^--", "D-")):
            ax.plot(sz, e[m].mean(1), st, label=m)                                    # dashed: generative; solid: discriminative
        ax.set_xscale("log"); ax.set_xlabel("training examples"); ax.set_ylabel("test error"); ax.set_title(title); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 3); grid = np.ones((10 * 29, 8 * 29))                  # sampled digits: 8 per class
    for c in range(10):
        for i in range(8):
            grid[c * 29:c * 29 + 28, i * 29:i * 29 + 28] = 1 - I[c * 100 + i].reshape(28, 28)
    ax.imshow(grid, cmap="gray"); ax.set_yticks([14 + 29 * c for c in range(10)]); ax.set_yticklabels(range(10)); ax.set_xticks([])
    ax.set_title("drawn from p(x | digit), QDA reg 0.001")
    ax = fig.add_subplot(2, 3, 4); names = list(rows); x = np.arange(len(names))     # missing pixels
    labels = ["LDA, marginal", "logistic, hidden = 0", "logistic, hidden = average", "logistic, retrained"]
    for k in range(4):
        ax.bar(x + (k - 1.5) * 0.2, [rows[n][k] for n in names], 0.2, label=labels[k])
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" seen", "") for n in names], fontsize=8); ax.set_ylim(0, 1.25); ax.set_ylabel("test accuracy")
    ax.set_title("half of every test image hidden"); ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax = fig.add_subplot(2, 3, 5); names = list(nov); x = np.arange(len(names))      # spotting unusual images
    ax.bar(x - 0.2, [nov[n][0] for n in names], 0.4, label="-log p(x), LDA"); ax.bar(x + 0.2, [nov[n][1] for n in names], 0.4, label="1 - confidence, logistic")
    ax.axhline(0.5, color="gray", ls=":"); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0, 1.05); ax.set_ylabel("AUC")
    ax.set_title("telling unusual images from digits"); ax.legend(fontsize=8, loc="lower right")
    ax = fig.add_subplot(2, 3, 6); ax.bar(names + ["normal digits"], [nov[n][2] for n in names] + [conf_normal], color="#C44E52")
    ax.set_ylim(0, 1.05); ax.set_ylabel("average top probability"); ax.set_title("the logistic regression's confidence")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def draw_diagram():                                     # two routes to p(class | x)
    fig, ax = plt.subplots(figsize=(12, 5)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9):                                           # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
    box(1.3, 2.5, 2.0, 1.0, "labelled data\n(x, class)", "#DDEEFF")
    box(4.6, 3.8, 3.0, 1.0, "generative: learn\np(x | class) and p(class)", "#FFF2CC")
    box(4.6, 1.2, 3.0, 1.0, "discriminative: learn\np(class | x) directly", "#E2F0D9")
    box(8.4, 3.8, 2.6, 1.0, "Bayes' rule:\np(class | x)", "#FFF2CC")
    box(8.4, 1.2, 2.6, 1.0, "p(class | x)", "#E2F0D9")
    box(10.9, 3.8, 1.8, 1.6, "also:\nsample new x,\nhandle missing x,\nscore p(x)", "#F8CBAD", 8.5)
    arrow(2.35, 2.75, 3.05, 3.7); arrow(2.35, 2.25, 3.05, 1.3); arrow(6.15, 3.8, 7.05, 3.8); arrow(6.15, 1.2, 7.05, 1.2); arrow(9.75, 3.8, 9.95, 3.8)
    ax.text(6.4, 2.5, "same answer, different route: with one shared covariance (LDA)\nboth are a softmax of linear scores", ha="center", fontsize=9, style="italic")
    ax.set_title("Day 133: generative vs discriminative classifiers", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "generative_discriminative_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("A Gaussian per class generates blurry, average-looking digits. Day 134 builds a GAN from scratch: a generator and a")
    print("discriminator trained against each other, with no formula for p(x) at all.")


def breast_cancer():                                    # scikit-learn's bundled table: 569 tumours, 30 measurements, no download
    from sklearn.datasets import load_breast_cancer     # only the data; every model below is written here
    B = load_breast_cancer(); o = np.random.RandomState(0).permutation(len(B.target))  # one fixed shuffle
    Xb, yb = np.log1p(B.data[o]), B.target[o]                                          # logs tame the long right tails of sizes and areas
    return (Xb - Xb[:369].mean(0)) / Xb[:369].std(0), yb                               # standardised by the 369 training rows only


def main():                                             # runs every section in order
    demo_hand()                                                                       # 1. the logistic posterior, by hand
    X, y = d126m.mnist(); X = X.astype(np.float64)                                    # 70,000 images; float64 for the covariances
    lda, lr = demo_same_family(X, y)                                                  # 2. LDA against logistic regression
    mu, vals, V = d123.pca_fit(X[:60000]); Z = d123.transform(X, mu, V, 50)           # 50 PCA numbers per image (Day 123), no labels used
    Xb, yb = breast_cancer()
    sizes, err, sizes_b, errb = demo_curves(Z, y, Xb, yb)                              # 3. learning curves
    I = demo_generate(Z, y, mu, V, lr)                                                # 4. sampling
    rows = demo_missing(X, y, lda, lr)                                                # 5. missing pixels
    nov = demo_novelty(X, y, lda, lr)                                                 # 6. log p(x)
    conf = float(lr(X[60000:62000]).max(1).mean())                                    # the normal digits' confidence, for the plot
    plot_results(sizes, err, sizes_b, errb, I, rows, nov, conf, os.path.join(HERE, "generative_discriminative_results.png"))
    draw_diagram()                                                                    # the picture of the two routes
    print("\nsaved generative_discriminative_results.png\nsaved generative_discriminative_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
