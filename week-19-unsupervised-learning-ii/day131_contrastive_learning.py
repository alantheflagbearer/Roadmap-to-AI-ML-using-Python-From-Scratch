"""
Day 131: Contrastive Learning Basics -- a Real Positive/Negative Pair Demonstration

The autoencoders of Days 126-130 learned by rebuilding pixels. Contrastive
learning never rebuilds anything. It makes two random VIEWS of each image
(shifted, rotated, partly erased, noisy) and trains a network so that the two
views of the same image land close together (a POSITIVE pair), while views of
different images are pushed apart (NEGATIVES). What survives every view is
what the image is, not the exact pixels.

Everything is written from scratch in NumPy, on Day 126's network parts:
  * the views: random rotation, shift, erasing and noise, for a whole batch at once
  * the NT-Xent loss of SimCLR (normalised temperature-scaled cross-entropy),
    with its gradient derived by hand and checked
  * linear probes with few labels, using Day 116's softmax regression

The data: MNIST (cached by Day 126); labels are used only to judge, never to train
the representation.

  1. By hand: the loss for two images.
  2. Correctness: gradient checks.
  3. The views: what a positive pair looks like.
  4. Training: positives pulled together, negatives pushed apart.
  5. What it learned: probes with 10 and 100 labels per digit.
  6. Temperature.
  7. Which views matter.

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
for w in ("week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' exercise parts
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam, train_ae
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day123_exercise as d123                      # Day 123: pca_fit, transform
    import day119_exercise as d119                      # Day 119: sqdist, kmeans, ari
    import day116_exercise as d116                      # Day 116: fit_softmax (softmax regression)
SIZES = [784, 256, 128, 64]                             # encoder 784 -> 256 -> 128 (the representation), projection head 128 -> 64
TAU = 0.1                                               # the temperature (section 6 tries others)


# =============================================================================
# VIEWS AND THE CONTRASTIVE LOSS -- NEW
# =============================================================================
def augment(X, rng, shift=3, angle=15, erase=8, noise=0.1):  # NEW: a random view of each 28 x 28 image
    n = len(X); a = np.radians(rng.uniform(-angle, angle, n)); dx, dy = rng.randint(-shift, shift + 1, (2, n))
    r, c = np.mgrid[0:28, 0:28].reshape(2, -1) - 13.5      # pixel coordinates around the centre
    cos, sin = np.cos(a)[:, None], np.sin(a)[:, None]
    src_r = np.rint(cos * r + sin * c + 13.5 - dy[:, None]).astype(int)  # for each output pixel, where it comes from:
    src_c = np.rint(-sin * r + cos * c + 13.5 - dx[:, None]).astype(int)  # rotated and shifted back (nearest pixel)
    ok = (src_r >= 0) & (src_r < 28) & (src_c >= 0) & (src_c < 28)
    V = np.where(ok, X[np.arange(n)[:, None], np.clip(src_r, 0, 27) * 28 + np.clip(src_c, 0, 27)], 0).astype(np.float32)
    if erase:                                             # blank a random erase x erase patch in half of the images
        r0, c0 = rng.randint(0, 28 - erase, (2, n)); hit = rng.uniform(size=n) < 0.5
        V = V.reshape(n, 28, 28)
        for i in np.where(hit)[0]:
            V[i, r0[i]:r0[i] + erase, c0[i]:c0[i] + erase] = 0
        V = V.reshape(n, -1)
    return np.clip(V + rng.normal(0, noise, V.shape).astype(np.float32), 0, 1)


def nt_xent(Z, tau):                                    # NEW: the contrastive loss for 2N projections (rows i and i + N are a pair), and its gradient
    n2 = len(Z); N = n2 // 2; norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12; U = Z / norm  # unit vectors
    S = U @ U.T / tau; np.fill_diagonal(S, -np.inf)       # cosine similarities over the temperature; never compare a row with itself
    pos = np.r_[np.arange(N, n2), np.arange(N)]           # each row's partner: the other view of the same image
    m = S.max(1, keepdims=True); E = np.exp(S - m); P = E / E.sum(1, keepdims=True)  # softmax over all other rows
    loss = float(np.mean(-np.log(P[np.arange(n2), pos] + 1e-30)))
    dS = P.copy(); dS[np.arange(n2), pos] -= 1; dS /= n2   # the softmax gradient: probability minus target
    dU = (dS + dS.T) @ U / tau                            # S is symmetric in U, so both halves contribute
    dZ = (dU - U * (U * dU).sum(1, keepdims=True)) / norm  # through the normalisation: remove the part along U
    return loss, dZ, P[np.arange(n2), pos]


def train_contrastive(X, sizes, tau=0.5, epochs=20, batch=256, lr=1e-3, seed=0, aug=None):  # NEW: two views per image, pulled together
    acts = ["relu"] * (len(sizes) - 2) + ["linear"]; net = d126.init_net(sizes, seed); opt = d126.Adam(net, lr)
    rng = np.random.RandomState(seed + 1); aug = aug or augment; hist = []
    for _ in range(epochs):
        o = rng.permutation(len(X)); losses = []
        for s in range(0, len(X) - batch + 1, batch):       # full batches only: every batch has the same number of negatives
            xb = X[o[s:s + batch]]; V = np.vstack([aug(xb, rng), aug(xb, rng)])
            A = d126.forward(net, V, acts); loss, dZ, _ = nt_xent(A[-1], tau)
            opt.step(net, d126.backward(net, A, acts, dZ)[0]); losses.append(loss)
        hist.append(float(np.mean(losses)))
    return net, hist


def represent(net, X, layer=-2):                        # NEW: the representation used afterwards: the layer before the projection head
    acts = ["relu"] * (len(net) - 1) + ["linear"]
    return d126.forward(net, X, acts)[layer]


# =============================================================================
# JUDGING A REPRESENTATION -- labels used only here
# =============================================================================
def knn_acc(R, y, tr, te, k=3):                         # kNN accuracy with the given reference and test rows
    nn = np.argsort(d119.sqdist(R[te].astype(np.float64), R[tr].astype(np.float64)), 1)[:, :k]
    return float(np.mean(np.array([np.bincount(r, minlength=10).argmax() for r in y[tr][nn]]) == y[te]))


def probe(R, y, n_per, seed=0):                         # a linear classifier (Day 116) trained on n_per labelled images per digit, tested on 10,000
    rng = np.random.RandomState(seed)
    tr = np.concatenate([rng.choice(np.where(y[:60000] == c)[0], n_per, replace=False) for c in range(10)]); te = np.arange(60000, 70000)
    mu, sd = R[tr].mean(0), R[tr].std(0) + 1e-6
    W, b = d116.fit_softmax(((R[tr] - mu) / sd).astype(np.float64), y[tr], 10, lam=1e-2, steps=500)
    return float(np.mean((((R[te] - mu) / sd) @ W + b).argmax(1) == y[te]))


def cosines(R, idx_a, idx_b):                           # cosine similarity between rows idx_a[i] and idx_b[i]
    A, B = R[idx_a], R[idx_b]
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- the loss for two images, four views, temperature 0.5")
    print("=" * 70)
    Z = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, -0.8]])  # rows 0, 1 are the first views; rows 2, 3 their partners
    loss, _, p = nt_xent(Z, 0.5)
    s = Z @ Z.T / 0.5
    by_hand = -np.log(np.exp(s[0, 2]) / (np.exp(s[0, 1]) + np.exp(s[0, 2]) + np.exp(s[0, 3])))
    print(f"row 0's similarities to rows 1, 2, 3 (cosine / 0.5): {s[0, 1]:.1f}, {s[0, 2]:.1f}, {s[0, 3]:.1f}; its partner is row 2")
    print(f"by hand: -log(e^{s[0, 2]:.1f} / (e^{s[0, 1]:.1f} + e^{s[0, 2]:.1f} + e^{s[0, 3]:.1f})) = {by_hand:.4f}")
    print(f"each row's probability of picking its partner: {np.round(p, 4).tolist()}; the loss averages -log of these: {loss:.4f}")
    print("Actual output from this lesson's run.")


def demo_gradcheck():
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- the hand-derived gradient against finite differences")
    print("=" * 70)
    rng = np.random.RandomState(0); Z = rng.randn(8, 3); _, dZ, _ = nt_xent(Z, 0.5); num = np.zeros_like(Z)
    for idx in np.ndindex(Z.shape):
        o = Z[idx]; Z[idx] = o + 1e-6; lp = nt_xent(Z, 0.5)[0]; Z[idx] = o - 1e-6; lm = nt_xent(Z, 0.5)[0]; Z[idx] = o
        num[idx] = (lp - lm) / 2e-6
    print(f"the loss alone, 8 projections of 3 numbers: largest gap {np.abs(num - dZ).max():.1e}")
    X = rng.rand(6, 10); net = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([10, 7, 5, 4], 0)]; acts = ["relu", "relu", "linear"]
    A = d126.forward(net, X, acts); _, d, _ = nt_xent(A[-1], 0.5); g, _ = d126.backward(net, A, acts, d); worst = 0.0
    for li, (W, b) in enumerate(net):
        for P, G in ((W, g[li][0]), (b, g[li][1])):
            for idx in np.ndindex(P.shape):
                o = P[idx]; P[idx] = o + 1e-6; lp = nt_xent(d126.forward(net, X, acts)[-1], 0.5)[0]
                P[idx] = o - 1e-6; lm = nt_xent(d126.forward(net, X, acts)[-1], 0.5)[0]; P[idx] = o
                worst = max(worst, abs((lp - lm) / 2e-6 - G[idx]))
    print(f"through a whole network (10-7-5-4, 3 images, 2 views each): largest gap {worst:.1e}")
    print("Actual output from this lesson's run.")


def demo_views(X):
    print("\n" + "=" * 70)
    print("3. THE VIEWS -- what a positive pair looks like, in raw pixels")
    print("=" * 70)
    rng = np.random.RandomState(1); x = X[:2000]; V1, V2 = augment(x, rng), augment(x, rng); i = np.arange(2000)
    same = cosines(np.vstack([V1, V2]), i, i + 2000); other = cosines(np.vstack([V1, V2]), i, (i + 1) % 2000 + 2000)
    print(f"cosine similarity in raw pixels: two views of the same image {same.mean():.4f} on average; views of different images {other.mean():.4f}")
    print(f"share of positive pairs less similar than the average negative pair: {np.mean(same < other.mean()):.4f}")
    print("in pixel space many positive pairs look no more alike than strangers: the network has to learn what stays the same")
    print("Actual output from this lesson's run.")
    return V1[:8], V2[:8]


def demo_train(X):
    print("\n" + "=" * 70)
    print(f"4. TRAINING -- 784 -> 256 -> 128 (the representation) -> 64, temperature {TAU}, 30 epochs on 20,000 images, batches of 256")
    print("=" * 70)
    t0 = time.perf_counter(); net, hist = train_contrastive(X[:20000], SIZES, TAU, epochs=30); t = time.perf_counter() - t0
    print(f"loss every 5 epochs: {[round(h, 3) for h in hist[::5] + hist[-1:]]}; trained in {t:.0f} s")
    print(f"for comparison: a network that could not tell views apart would score log(511) = {np.log(511):.3f}")
    rng = np.random.RandomState(2); x = X[60000:62000]; V = np.vstack([augment(x, rng), augment(x, rng)]); i = np.arange(2000)
    Z = d126.forward(net, V, ["relu", "relu", "linear"])[-1]
    print(f"on 2,000 test images: cosine of positive pairs {cosines(Z, i, i + 2000).mean():.4f}, of negative pairs {cosines(Z, i, (i + 1) % 2000 + 2000).mean():.4f}")
    print("Actual output from this lesson's run.")
    return net, hist


def demo_learned(X, y, net):
    print("\n" + "=" * 70)
    print("5. WHAT IT LEARNED -- 128 numbers per image; labels used only to judge")
    print("=" * 70)
    mu, vals, V = d123.pca_fit(X[:20000].astype(np.float64))
    t0 = time.perf_counter(); ae, _ = d126.train_ae(X[:20000], [784, 256, 128, 256, 784], ["relu", "linear", "relu", "linear"], "bce", epochs=20, seed=0)
    print(f"for comparison, Day 126's autoencoder with a 128-number code, 20 epochs ({time.perf_counter() - t0:.0f} s)")
    reps = {"raw pixels (784)": X, "PCA (128)": d123.transform(X.astype(np.float64), mu, V, 128).astype(np.float32),
            "autoencoder code (128)": d126.forward(ae, X, ["relu", "linear", "relu", "linear"])[2], "contrastive (128)": represent(net, X)}
    tr, te = np.arange(20000, 25000), np.arange(60000, 62000); out = {}
    print(f"  {'representation':<24}{'kNN (5,000 labelled)':>22}{'probe, 10 per digit':>21}{'probe, 100 per digit':>22}")
    for name, R in reps.items():
        out[name] = (knn_acc(R, y, tr, te), np.mean([probe(R, y, 10, s) for s in range(3)]), probe(R, y, 100))
        print(f"  {name:<24}{out[name][0]:>22.4f}{out[name][1]:>21.4f}{out[name][2]:>22.4f}")
    print("probes: Day 116's softmax regression on standardised features, tested on all 10,000 test images; 10 per digit averaged over 3 draws")
    print("Actual output from this lesson's run.")
    return out


def demo_tau(X, y, net01, hist01):
    print("\n" + "=" * 70)
    print("6. TEMPERATURE -- the same network, 30 epochs each (0.1 is section 4's network)")
    print("=" * 70)
    rows = []
    print(f"  {'temperature':>12}{'final loss':>12}{'probe, 10 per digit':>21}{'probe, 100 per digit':>22}")
    for tau in (0.1, 0.5, 1.0):
        net, hist = (net01, hist01) if tau == TAU else train_contrastive(X[:20000], SIZES, tau, epochs=30); R = represent(net, X)
        rows.append((tau, hist[-1], np.mean([probe(R, y, 10, s) for s in range(3)]), probe(R, y, 100)))
        print(f"  {tau:>12}{rows[-1][1]:>12.3f}{rows[-1][2]:>21.4f}{rows[-1][3]:>22.4f}")
    print("(losses at different temperatures are on different scales and cannot be compared with each other)")
    print("Actual output from this lesson's run.")
    return rows


def demo_views_matter(X, y):
    print("\n" + "=" * 70)
    print("7. WHICH VIEWS MATTER -- the same training with only some of the changes")
    print("=" * 70)
    variants = {"noise": dict(shift=0, angle=0, erase=0), "noise + shift": dict(angle=0, erase=0),
                "noise + shift + rotation": dict(erase=0)}                             # the noise (0.1) stays on in every variant
    rows = []
    for name, kw in variants.items():
        net, _ = train_contrastive(X[:20000], SIZES, TAU, epochs=30, aug=lambda Xb, rng, kw=kw: augment(Xb, rng, **kw)); R = represent(net, X)
        rows.append((name, np.mean([probe(R, y, 10, s) for s in range(3)]), probe(R, y, 100)))
        print(f"  {name:<26}: probe, 10 per digit {rows[-1][1]:.4f}; 100 per digit {rows[-1][2]:.4f}")
    print("  (noise + shift + rotation + erasing: section 5's contrastive row)")
    print("Actual output from this lesson's run.")
    return rows


def plot_results(X, V1, V2, hist, net, y, rows_tau, out):  # five panels
    fig = plt.figure(figsize=(16, 9.6)); G = fig.add_gridspec(2, 3)
    sub = G[0, :2].subgridspec(3, 8)
    for c in range(8):
        for r, (imgs, name) in enumerate(((X[:8], "original"), (V1, "view 1"), (V2, "view 2"))):
            a = fig.add_subplot(sub[r, c]); a.imshow(imgs[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_ylabel(name, fontsize=8)
    a = fig.add_subplot(G[0, 2]); a.plot(np.arange(1, len(hist) + 1), hist, marker="."); a.axhline(np.log(511), color="k", linestyle=":", linewidth=0.8, label="log 511: no idea which view is the partner")
    a.set_xlabel("epoch"); a.set_ylabel("NT-Xent loss"); a.set_title("Training", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = fig.add_subplot(G[1, 0]); names = list(out); x = np.arange(len(names)); w = 0.38
    a.bar(x - w / 2, [out[n][1] for n in names], w, label="10 labels per digit"); a.bar(x + w / 2, [out[n][2] for n in names], w, label="100 labels per digit")
    a.set_xticks(x); a.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8); a.set_ylim(0.5, 1.0); a.set_ylabel("linear probe accuracy")
    a.set_title("A linear classifier on each representation", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    R2 = represent(net, X[60000:62000]).astype(np.float64); m, _, Vr = d123.pca_fit(R2); C = d123.transform(R2, m, Vr, 2)  # the representation's top 2 directions
    a = fig.add_subplot(G[1, 1]); s = a.scatter(C[:, 0], C[:, 1], c=y[60000:62000], cmap="tab10", s=4); fig.colorbar(s, ax=a, label="digit")
    a.set_title("Contrastive representation, first 2 PCA directions", fontsize=10); a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(G[1, 2]); r = np.array(rows_tau); a.plot(r[:, 0], r[:, 2], marker="o", label="10 per digit"); a.plot(r[:, 0], r[:, 3], marker="s", label="100 per digit")
    a.set_xscale("log"); a.minorticks_off(); a.set_xticks(r[:, 0]); a.set_xticklabels([str(t) for t in r[:, 0]]); a.set_xlabel("temperature")
    a.set_ylabel("linear probe accuracy"); a.set_title("Temperature", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("contrastive_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved contrastive_results.png")


def draw_diagram():                                     # two views in, one loss
    fig, ax = plt.subplots(figsize=(13, 4.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.35, "Day 131: contrastive learning (SimCLR-style)", ha="center", fontsize=12, fontweight="bold")
    box(1.2, 2.3, 1.6, 1.4, "image", "#FAEEDA")
    box(3.3, 3.2, 1.9, 0.9, "view 1: rotate,\nshift, erase, noise", "#F7DCDC", fs=8); box(3.3, 1.4, 1.9, 0.9, "view 2: a different\nrandom change", "#F7DCDC", fs=8)
    box(5.9, 3.2, 2.2, 0.9, "encoder ->\nrepresentation (128)", "#DCEBF7", fs=8); box(5.9, 1.4, 2.2, 0.9, "the same encoder\n(shared weights)", "#DCEBF7", fs=8)
    box(8.4, 3.2, 1.8, 0.9, "head -> z (64)", "#E5F2E1", fs=8); box(8.4, 1.4, 1.8, 0.9, "head -> z (64)", "#E5F2E1", fs=8)
    box(11.4, 2.3, 2.6, 2.4, "NT-Xent: the two z of\none image together;\nevery other image in\nthe batch apart", "#EEEDFE")
    for (x0, y0), (x1, y1) in (((2.04, 2.6), (2.34, 3.0)), ((2.04, 2.0), (2.34, 1.6)), ((4.26, 3.2), (4.79, 3.2)), ((4.26, 1.4), (4.79, 1.4)),
                               ((7.01, 3.2), (7.49, 3.2)), ((7.01, 1.4), (7.49, 1.4)), ((9.31, 3.2), (10.09, 2.7)), ((9.31, 1.4), (10.09, 1.9))):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 0.35, "no labels and no reconstruction; afterwards the head is dropped and the 128-number representation is used", ha="center", fontsize=8.5)
    plt.savefig("contrastive_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved contrastive_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 132 is the Week 19 build: a self-supervised pretext task (predicting rotations), compared with every representation of the week.")


def main():                                             # runs every section in order
    X, y = d126m.mnist()
    demo_hand()                                            # 1. by hand
    demo_gradcheck()                                       # 2. gradient checks
    V1, V2 = demo_views(X)                                 # 3. the views
    net, hist = demo_train(X)                              # 4. training
    out = demo_learned(X, y, net)                          # 5. what it learned
    rows_tau = demo_tau(X, y, net, hist)                   # 6. temperature
    demo_views_matter(X, y)                                # 7. which views matter
    plot_results(X, V1, V2, hist, net, y, rows_tau, out)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
