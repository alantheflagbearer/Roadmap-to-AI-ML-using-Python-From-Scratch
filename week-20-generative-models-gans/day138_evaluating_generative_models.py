"""
Day 138: Evaluating Generative Models -- Real FID/IS-Style Metric Computation

Days 134-137 judged their GANs with hand-made measures. The field uses a few
standard scores instead, all built on a classifier's view of the images:
  * the INCEPTION SCORE (IS): are the images confidently classified, and are
    the classes spread out? exp(E_x KL(p(y|x) || p(y))), from 1 to the number
    of classes
  * the FRECHET DISTANCE (FID): fit a Gaussian to the classifier's features
    of real images and another to those of generated ones, and measure how
    far apart the two Gaussians are
  * PRECISION and RECALL: what share of generated images look like some real
    image, and what share of real images have some generated image near them

The originals use Google's Inception network on photographs; here the
classifier is a small convolutional network trained on MNIST, so the numbers
are "FID/IS-style": the same formulas with a different feature extractor, and
not comparable with published values.

Everything is written from scratch in NumPy, on Day 136's layers and Day
134's AdamGAN, imported unchanged:
  * the feature network, trained on MNIST's labels
  * the matrix square root (by eigen-decomposition), the Frechet distance,
    the Inception Score, and k-nearest-neighbour precision and recall

  1. By hand: the Frechet distance and the Inception Score on cases whose
     answers are known.
  2. The feature network.
  3. Real against real: what a perfect generator scores, and how the score
     depends on the number of samples, and on which samples.
  4. Damage done on purpose: noise, blur, missing digits, collapse within a
     digit, and copies of the training set. Which score sees which?
  5. Scoring real GANs during training: Day 134's dense GAN (both settings of
     Adam) and Day 136's DCGAN.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier parts
import time                                            # time: how long training takes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles", "week-07-cnns-i-and-ii"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day119_exercise as d119                      # Day 119: sqdist
    import day116_exercise as d116                      # Day 116: softmax
    import day134_exercise as d134                      # Day 134: AdamGAN
    import day136_exercise as d136                      # Day 136: Conv, Dense, Act, Net, net_gan_step, net_generate, dcgan_players
    import day136_dcgan as d136m                        # Day 136: dense_players (Day 134's dense GAN as a layer stack)


# =============================================================================
# THE SCORES -- NEW
# =============================================================================
def sqrtm_psd(S):                                       # NEW: the square root of a symmetric matrix with no negative eigenvalues
    w, V = np.linalg.eigh(S)                                                          # S = V diag(w) V^T
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T                                   # V diag(sqrt w) V^T; tiny negative w from rounding set to 0


def frechet(mu1, S1, mu2, S2):                          # NEW: the Frechet distance between N(mu1, S1) and N(mu2, S2)
    A = sqrtm_psd(S1); M = A @ S2 @ A                                                  # S1^(1/2) S2 S1^(1/2): symmetric, same eigenvalues as S1 S2
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((M + M.T) / 2), 0, None)).sum()       # Tr (S1 S2)^(1/2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * tr_sqrt)


def gauss(F):                                           # NEW: the mean and covariance of a set of feature vectors
    F = F.astype(np.float64); return F.mean(0), np.cov(F, rowvar=False)


def inception_score(P):                                 # NEW: exp of the average KL divergence from p(y|x) to p(y)
    P = np.clip(P.astype(np.float64), 1e-12, 1); py = P.mean(0)                        # p(y): the classes over all images
    return float(np.exp((P * (np.log(P) - np.log(py))).sum(1).mean()))


def precision_recall(R, F, k=3):                        # NEW: k-nearest-neighbour precision and recall (Kynkaanniemi et al., 2019)
    DRR, DFF, DRF = d119.sqdist(R, R), d119.sqdist(F, F), d119.sqdist(R, F)          # every squared distance, in feature space
    rR = np.sort(DRR, 1)[:, k]; rF = np.sort(DFF, 1)[:, k]                            # each point's k-th neighbour in its own set (column 0 is itself)
    precision = float(np.mean((DRF <= rR[:, None]).any(0)))                            # a fake counts if it falls inside some real point's ball
    recall = float(np.mean((DRF <= rF[None, :]).any(1)))                               # a real counts if it falls inside some fake point's ball
    return precision, recall


# =============================================================================
# THE FEATURE NETWORK -- NEW (a small convolutional classifier, from Day 136's layers)
# =============================================================================
def feature_net(seed=0):                                # NEW: 28 x 28 -> 16 x 14 x 14 -> 32 x 7 x 7 -> 64 features -> 10 classes
    r = np.random.RandomState(seed)
    body = d136.Net([d136.Act((-1, 1, 28, 28)), d136.Conv(1, 16, 4, 2, 1, r), d136.Act("lrelu"), d136.Conv(16, 32, 4, 2, 1, r),
                     d136.Act("lrelu"), d136.Act((-1, 32 * 49)), d136.Dense(32 * 49, 64, r), d136.Act("relu")])  # the features
    head = d136.Net([d136.Dense(64, 10, r)])                                         # the class scores
    return body, head


def train_classifier(X, y, epochs=3, batch=128, seed=0):  # NEW: softmax cross-entropy, Adam at 0.001
    body, head = feature_net(seed); ob, oh = d134.AdamGAN(body.net, 1e-3, 0.9), d134.AdamGAN(head.net, 1e-3, 0.9)  # b1 0.9: Day 126's plain Adam
    rng = np.random.RandomState(seed); Y = np.eye(10, dtype=np.float32)[y]
    for ep in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(X) - batch + 1, batch):
            b = o[s:s + batch]; L = head.forward(body.forward(X[b]))                    # scores for one batch
            dL = (d116.softmax(L) - Y[b]) / batch                                     # d cross-entropy / d scores, averaged
            gh, dF = head.backward(dL); gb, _ = body.backward(dF)
            oh.step(head.net, gh); ob.step(body.net, gb)
    return body, head


class Scorer:                                           # NEW: every score of one set of images, against fixed real references
    def __init__(self, body, head, Xref, Xnear):
        self.body, self.head = body, head
        self.Fref = self.feats(Xref); self.mu, self.S = gauss(self.Fref)              # the real Gaussian, from 10,000 training images
        self.Fpr = self.Fref[:2000]; self.Xnear = Xnear                               # 2,000 of them for precision/recall; pixels for the nearest-image test

    def feats(self, X):                                                               # features, in batches
        return np.concatenate([self.body.forward(X[i:i + 1000]) for i in range(0, len(X), 1000)])

    def __call__(self, X):                                                            # IS, FID, precision, recall, nearest training image
        F = self.feats(X); P = d116.softmax(self.head.forward(F).astype(np.float64)); mu, S = gauss(F)
        p, r = precision_recall(self.Fpr, F[:2000])
        near = float(np.sqrt(np.maximum(d119.sqdist(X[:500], self.Xnear).min(1), 0)).mean())  # pixel distance to the nearest training image
        return inception_score(P), frechet(mu, S, self.mu, self.S), p, r, near


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- the Frechet distance and the Inception Score where the answer is known")
    print("=" * 70)
    fd = frechet(np.zeros(1), np.eye(1) * 1.0, np.full(1, 3.0), np.eye(1) * 4.0)
    print(f"1-D: N(0, 1) against N(3, 4): formula (mu1 - mu2)^2 + (sd1 - sd2)^2 = 9 + 1 = 10; frechet() gives {fd:.6f}")
    rng = np.random.RandomState(0); A = rng.randn(5, 5); S1 = A @ A.T; B = rng.randn(5, 5); S2 = B @ B.T
    ev = np.linalg.eigvals(S1 @ S2); other = np.sqrt(ev.real.clip(0)).sum()            # an independent route: eigenvalues of S1 S2 itself
    A2 = sqrtm_psd(S1); print(f"square root check: |sqrtm(S1)^2 - S1| = {np.abs(A2 @ A2 - S1).max():.1e};  "
                              f"Tr (S1 S2)^(1/2) by two routes: {np.sqrt(np.linalg.eigvalsh(A2 @ S2 @ A2).clip(0)).sum():.8f} and {other:.8f}")
    X = rng.randn(200000, 3) @ np.diag([1, 2, 3]); m, S = gauss(X)
    print(f"200,000 samples of N(0, diag(1, 4, 9)) against the true Gaussian: {frechet(m, S, np.zeros(3), np.diag([1.0, 4, 9])):.5f} (0 in the limit)")
    eye = np.eye(10); print(f"Inception Score: every image a confident, different class, evenly spread: {inception_score(np.tile(eye, (100, 1))):.4f} (10 = the number of classes)")
    print(f"   every image a confident 3:                       {inception_score(np.tile(eye[3], (1000, 1))):.4f} (1: no spread)")
    print(f"   every image unsure, 0.1 on each class:           {inception_score(np.full((1000, 10), 0.1)):.4f} (1: no confidence)")
    print(f"   confident, but only digits 0-4:                  {inception_score(np.tile(eye[:5], (200, 1))):.4f} (5 = the classes used)")
    print("Actual output from this lesson's run.")


def demo_features(X, y):
    print("\n" + "=" * 70)
    print("2. THE FEATURE NETWORK -- a small convolutional classifier trained on MNIST's labels")
    print("=" * 70)
    t0 = time.perf_counter(); body, head = train_classifier(X[:60000], y[:60000])
    acc = np.mean(np.concatenate([head.forward(body.forward(X[i:i + 1000])).argmax(1) for i in range(60000, 70000, 1000)]) == y[60000:])
    print(f"28 x 28 -> conv 4 x 4 stride 2 -> 16 x 14 x 14 -> conv -> 32 x 7 x 7 -> dense -> 64 features -> 10 classes")
    print(f"trained 3 epochs on the 60,000 training images ({time.perf_counter() - t0:.0f} s): test accuracy {acc:.4f}")
    print("Actual output from this lesson's run.")
    return body, head, acc


def fid_only(sc, S):                                    # NEW: the Frechet distance alone, without the other scores
    mu, C = gauss(sc.feats(S)); return frechet(mu, C, sc.mu, sc.S)


def demo_real(X, y, body, head):
    print("\n" + "=" * 70)
    print("3. REAL AGAINST REAL -- what a perfect generator would score, and the effect of sample size")
    print("=" * 70)
    sc = Scorer(body, head, X[:10000], X[10000:30000])                                # references: training images 0-9,999; nearest-image pool 10,000-29,999
    print("references: the real Gaussian from training images 0-9,999 (FID); 2,000 of them for precision and recall (k = 3);")
    print("the nearest-image test searches training images 10,000-29,999, in pixels")
    perm = 60000 + np.random.RandomState(0).permutation(10000)                        # the 10,000 test images in a random order
    rows, sizes = [], (500, 1000, 2000, 5000, 10000)
    print(f"{'set scored':>40} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'nearest':>8}")
    for n in sizes:
        s = sc(X[perm[:n]]); rows.append((n, s)); print(f"{f'{n:,} real test images, random order':>40} {s[0]:7.3f} {s[1]:8.3f} {s[2]:10.3f} {s[3]:7.3f} {s[4]:8.2f}")
    print("FID falls as the scored set grows, with nothing about the images changing: a fitted covariance from few samples is noisy,")
    print("and the noise adds to the distance. Scores are only comparable at the same sample size; 2,000 is used below")
    pre = [fid_only(sc, X[60000:60000 + n]) for n in sizes]                           # the first n test images, in MNIST's stored order
    print(f"the first draft scored the first n test images in MNIST's stored order instead; FID for the same sizes: {' '.join(f'{v:.3f}' for v in pre)}")
    acc = [np.mean(np.concatenate([sc.head.forward(sc.feats(X[a + i:a + i + 1000])).argmax(1) for i in range(0, 5000, 1000)]) == y[a:a + 5000]) for a in (60000, 65000)]
    print(f"the stored order is not random: test images 0-4,999 against 5,000-9,999: FID {fid_only(sc, X[60000:65000]):.3f} and "
          f"{fid_only(sc, X[65000:70000]):.3f} (5,000 in random order: {rows[3][1][1]:.3f}),")
    print(f"named correctly {acc[0]:.4f} and {acc[1]:.4f}, average ink {X[60000:65000].mean():.4f} and {X[65000:70000].mean():.4f}")
    print("Actual output from this lesson's run.")
    return sc, rows, perm, pre


def blur(X):                                            # NEW: a 3 x 3 box blur of every image
    I = np.pad(X.reshape(-1, 28, 28), ((0, 0), (1, 1), (1, 1)), mode="edge")
    return sum(I[:, i:i + 28, j:j + 28] for i in range(3) for j in range(3)).reshape(-1, 784) / 9


def demo_damage(X, y, sc, perm):
    print("\n" + "=" * 70)
    print("4. DAMAGE DONE ON PURPOSE -- which score sees which failure?")
    print("=" * 70)
    rng = np.random.RandomState(0); T, yt = X[perm], y[perm]; base = T[:2000]           # the test set in section 3's random order
    low = T[yt < 5][:2000]                                                            # digits 0-4 only: half the modes gone
    one = np.concatenate([T[yt == d][:1].repeat(200, 0) for d in range(10)])            # one image per digit, 200 copies of each
    sets = [("real test images", base), ("+ noise, sd 0.1", np.clip(base + 0.1 * rng.randn(*base.shape), 0, 1).astype(np.float32)),
            ("+ noise, sd 0.3", np.clip(base + 0.3 * rng.randn(*base.shape), 0, 1).astype(np.float32)), ("3 x 3 blur", blur(base).astype(np.float32)),
            ("digits 0-4 only", low), ("one image per digit, x 200", one), ("copies of training images", X[10000:12000])]
    print("2,000 images in each set, from the test set in random order; the nearest-image test looks in training images 10,000-29,999,")
    print("which the last set was copied from")
    print(f"{'set scored':>30} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'nearest':>8}")
    res = {}
    for name, S in sets:
        s = sc(S); res[name] = s; print(f"{name:>30} {s[0]:7.3f} {s[1]:8.3f} {s[2]:10.3f} {s[3]:7.3f} {s[4]:8.2f}")
    print("Actual output from this lesson's run.")
    return res, one


def train_scored(G, D, X, sc, b1, epochs, seed, at, k=64, batch=128):  # NEW: Day 134's loop (Day 136's net_gan_step), scored at chosen epochs
    oG, oD = d134.AdamGAN(G.net, 2e-4, b1), d134.AdamGAN(D.net, 2e-4, b1)
    rng = np.random.RandomState(seed); Zfix = np.random.RandomState(9).standard_normal((2000, k)).astype(np.float32)
    log, t0 = [], time.perf_counter()
    for ep in range(1, epochs + 1):
        o = rng.permutation(len(X))
        for s in range(0, len(X) - batch + 1, batch):                                 # one pass over the training images
            d136.net_gan_step(G, D, oG, oD, X[o[s:s + batch]], rng, k)
        if ep in at:
            F = d136.net_generate(G, Zfix); log.append((ep, sc(F), F[:16].copy(), time.perf_counter() - t0))  # scored on 2,000 fixed noise vectors
    return log


def demo_gans(X, sc, epochs=20):
    print("\n" + "=" * 70)
    print("5. SCORING REAL GANS DURING TRAINING -- Day 134's dense GAN (two settings of Adam) and Day 136's DCGAN")
    print("=" * 70)
    print("each trained as before: 20,000 training images (0-19,999), batches of 128, learning rate 0.0002, seed 0; scored on")
    print("2,000 images from fixed noise. Note: training images 10,000-19,999 are both in the GANs' training set and in the")
    print("nearest-image pool, so 'nearest' measures closeness to images each GAN saw")
    at = (1, 2, 3, 5, 10, 15, 20); runs = {}
    for name, b1, maker in (("dense, b1 0.9", 0.9, d136m.dense_players), ("dense, b1 0.5", 0.5, d136m.dense_players), ("DCGAN, b1 0.5", 0.5, d136.dcgan_players)):
        G, D = maker(0); log = train_scored(G, D, X[:20000], sc, b1, epochs, 0, at); runs[name] = log
        print(f"{name}:  {'epoch':>5} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'nearest':>8}")
        for ep, s, _, _ in log:
            print(f"{'':{len(name) + 1}s}  {ep:5d} {s[0]:7.3f} {s[1]:8.3f} {s[2]:10.3f} {s[3]:7.3f} {s[4]:8.2f}")
        print(f"{'':{len(name) + 1}s}  ({log[-1][-1]:.0f} s)")
    print("Actual output from this lesson's run.")
    return runs


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(rows, pre, res, one, runs, out):       # seven panels
    fig = plt.figure(figsize=(17, 11))
    ax = fig.add_subplot(2, 4, 1); n = [r[0] for r in rows]; ax.plot(n, [r[1][1] for r in rows], marker="o", label="random order")
    ax.plot(n, pre, marker="s", label="MNIST's stored order (first n)"); ax.legend(fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("real test images scored"); ax.set_ylabel("FID against training images"); ax.set_title("real against real: FID shrinks with n", fontsize=10)
    names = list(res); short = ["real", "noise 0.1", "noise 0.3", "blur", "0-4 only", "1 per digit", "copies"]
    for j, (i, lab) in enumerate(((0, "Inception Score"), (1, "FID"), (3, "recall"))):
        ax = fig.add_subplot(2, 4, 2 + j); v = [res[k][i] for k in names]
        ax.bar(range(len(v)), v, color=["#4C72B0"] + ["#C44E52"] * (len(v) - 2) + ["#55A868"])
        ax.set_xticks(range(len(v))); ax.set_xticklabels(short, rotation=40, fontsize=8, ha="right"); ax.set_title(f"damage: {lab}", fontsize=10)
        if i == 1:
            ax.set_yscale("log")
    ax = fig.add_subplot(2, 4, 5); ax.imshow(np.hstack([one[200 * d].reshape(28, 28) for d in range(10)]), cmap="gray_r"); ax.axis("off")
    ax.set_title("'one image per digit': the 10 images, each repeated 200 times", fontsize=9)
    for j, (i, lab) in enumerate(((1, "FID"), (3, "recall"))):
        ax = fig.add_subplot(2, 4, 6 + j)
        for name, log in runs.items():
            ax.plot([r[0] for r in log], [r[1][i] for r in log], marker="o", label=name)
        ax.axhline(res["real test images"][i], color="gray", lw=1, label="real test images"); ax.set_xlabel("epoch"); ax.set_title(f"GANs during training: {lab}", fontsize=10)
        if i == 1:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 4, 8)
    ax.imshow(np.vstack([np.hstack([log[-1][2][c].reshape(28, 28) for c in range(8)]) for log in runs.values()]), cmap="gray_r", vmin=0, vmax=1); ax.axis("off")
    ax.set_title("epoch 20, rows: " + "; ".join(runs), fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # how the scores are computed
    fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.3); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9):                                           # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
    box(1.3, 3.6, 2.2, 1.0, "generated images\n(2,000)", "#F8CBAD"); box(1.3, 1.4, 2.2, 1.0, "real images\n(training set)", "#E2F0D9")
    box(4.5, 2.5, 2.4, 1.6, "feature network\n(a trained digit\nclassifier)", "#DDEEFF")
    box(8.2, 4.1, 3.4, 1.0, "class probabilities -> Inception\nScore: confident AND spread?", "#FFF2CC", 8.5)
    box(8.2, 2.5, 3.4, 1.0, "64 features -> fit two Gaussians\n-> Frechet distance (FID)", "#FFF2CC", 8.5)
    box(8.2, 0.9, 3.4, 1.0, "64 features -> nearest neighbours\n-> precision and recall", "#FFF2CC", 8.5)
    box(11.8, 2.5, 1.8, 2.6, "pixels ->\nnearest\ntraining\nimage:\ncopied?", "#EEEEEE", 8.5)
    arrow(2.45, 3.4, 3.25, 2.9); arrow(2.45, 1.6, 3.25, 2.1)
    arrow(5.75, 2.9, 6.45, 4.0); arrow(5.75, 2.5, 6.45, 2.5); arrow(5.75, 2.1, 6.45, 1.0)
    ax.plot([1.3, 1.3, 11.8], [4.15, 4.95, 4.95], ls="--", lw=1, color="gray")       # the pixels go straight to the nearest-image test
    ax.annotate("", xy=(11.8, 3.85), xytext=(11.8, 4.95), arrowprops=dict(arrowstyle="->", lw=1, ls="--", color="gray"))
    ax.text(6.5, 5.05, "generated pixels, compared directly (no feature network)", ha="center", fontsize=8.5, color="dimgray")
    ax.set_title("Day 138: scoring a generative model through a classifier's eyes", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "generative_metrics_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 139 builds the week: Day 137's conditional DCGAN from the exercise parts, scored with today's metrics against")
    print("Day 133's Gaussian class models, and then the first half of a different idea, diffusion: noise added to an image")
    print("step by step until nothing of it is left.")


def main():                                             # runs every section in order
    demo_hand()                                                                       # 1. the formulas, checked
    X, y = d126m.mnist()                                                              # 70,000 images, pixels in [0, 1], float32
    body, head, acc = demo_features(X, y)                                             # 2. the feature network
    sc, rows, perm, pre = demo_real(X, y, body, head)                                 # 3. real against real
    res, one = demo_damage(X, y, sc, perm)                                            # 4. damage done on purpose
    runs = demo_gans(X, sc)                                                           # 5. real GANs, scored
    plot_results(rows, pre, res, one, runs, os.path.join(HERE, "generative_metrics_results.png"))
    draw_diagram()                                                                    # the picture of the scores
    print("\nsaved generative_metrics_results.png\nsaved generative_metrics_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
