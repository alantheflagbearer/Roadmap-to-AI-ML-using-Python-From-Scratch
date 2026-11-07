"""
Day 126: Autoencoders From Scratch -- a Real Reconstruction Demonstration

An autoencoder is a network trained to copy its input to its output through a
narrow middle layer, the CODE. Squeezed through a few dozen numbers, it must
keep what matters about each image and drop the rest. The first half (the
ENCODER) compresses; the second (the DECODER) rebuilds. With straight-line
layers it can do no better than PCA (Day 123); with ReLU layers it can.

Everything is written from scratch in NumPy:
  * dense layers, ReLU, backpropagation through the whole stack, Adam
  * reconstruction losses: squared error, and binary cross-entropy on logits
  * minibatch training, with held-out reconstruction error each epoch

The data: MNIST, 70,000 handwritten digits of 28 x 28 pixels, fetched once from
openml.org through scikit-learn and cached outside the repository. Labels are
used only to judge what the codes hold.

  1. By hand: a 2-1-2 autoencoder on points along a line.
  2. Correctness: a gradient check through the whole network.
  3. A linear autoencoder rediscovers PCA.
  4. Nonlinear against PCA, at code sizes 2, 8 and 32.
  5. How much data: 1,000 against 20,000 training images.
  6. What the code holds: neighbours and clusters in code space.
  7. Decoding codes it has never seen: the gaps in code space.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import Week 18's parts
import time                                            # time: how long training takes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, os.path.join(HERE, "..", "week-18-unsupervised-learning-i"))  # Week 18's exercise parts
with contextlib.redirect_stdout(io.StringIO()):         # their imports print a few checks; keep them quiet
    import day119_exercise as d119                      # Day 119: kmeans, ari
    import day123_exercise as d123                      # Day 123: pca_fit, transform, reconstruct
CODE_ACTS = ["relu", "linear", "relu", "linear"]        # encoder 784 -> 256 -> code, decoder code -> 256 -> 784


def mnist():                                            # NEW: MNIST as floats in [0, 1], cached as a small file after the first download
    p = os.path.join(os.path.expanduser("~"), "scikit_learn_data", "mnist_784_uint8.npz")
    if not os.path.exists(p):
        from sklearn.datasets import fetch_openml       # only needed once
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")  # about a minute the first time
        np.savez_compressed(p, X=d.data.astype(np.uint8), y=d.target.astype(np.uint8))
    z = np.load(p)
    return (z["X"] / 255.0).astype(np.float32), z["y"].astype(int)  # float32: about 3 times faster here than float64


# =============================================================================
# THE NETWORK -- NEW (a stack of dense layers, trained by Adam)
# =============================================================================
def init_net(sizes, seed):                              # NEW: [(W, b), ...] for layer sizes like [784, 256, 32, 256, 784]
    rng = np.random.RandomState(seed)
    return [((rng.randn(a, b) * np.sqrt(2.0 / a)).astype(np.float32), np.zeros(b, np.float32)) for a, b in zip(sizes[:-1], sizes[1:])]


def forward(net, X, acts):                              # NEW: every layer's output; acts names each layer's activation
    A = [X]
    for (W, b), act in zip(net, acts):
        Z = A[-1] @ W + b
        A.append(np.maximum(Z, 0) if act == "relu" else Z)  # "linear" keeps Z: the last layer gives logits or values
    return A


def backward(net, A, acts, dout):                       # NEW: gradients of every W and b, given d(loss)/d(last output)
    grads = [None] * len(net)
    for i in range(len(net) - 1, -1, -1):
        dZ = dout * (A[i + 1] > 0) if acts[i] == "relu" else dout  # through ReLU: zero where the unit was off
        grads[i] = (A[i].T @ dZ, dZ.sum(0))
        dout = dZ @ net[i][0].T                             # back through the weights to the layer below
    return grads, dout


def sigmoid(Z):                                         # 1 / (1 + e^-z), in a form that cannot overflow
    return 0.5 * (1 + np.tanh(0.5 * Z))


def recon_loss(L, X, kind):                             # NEW: loss per image and its gradient at the last layer
    n = len(X)
    if kind == "bce":                                     # pixels in [0, 1] as probabilities; L are logits
        return float((np.logaddexp(0, L) - X * L).sum() / n), (sigmoid(L) - X) / n
    return float(((L - X) ** 2).sum() / n), 2 * (L - X) / n  # squared error, summed over pixels


class Adam:                                             # NEW: Day 37's Adam, for a list of (W, b) pairs
    def __init__(self, net, lr=1e-3):
        self.lr, self.t = lr, 0
        self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in net]; self.v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in net]

    def step(self, net, grads):
        self.t += 1; c1, c2 = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for (W, b), (gW, gb), m, v in zip(net, grads, self.m, self.v):
            for p, g, mm, vv in ((W, gW, m[0], v[0]), (b, gb, m[1], v[1])):
                mm *= 0.9; mm += 0.1 * g; vv *= 0.999; vv += 0.001 * g * g
                p -= self.lr * (mm / c1) / (np.sqrt(vv / c2) + 1e-8)  # in place, so the network itself changes


def train_ae(X, sizes, acts, kind="bce", epochs=20, batch=128, lr=1e-3, seed=0, Xval=None, noise=None):  # NEW: minibatch training
    net = init_net(sizes, seed); opt = Adam(net, lr); rng = np.random.RandomState(seed + 1); hist = []
    for _ in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            xb = X[o[s:s + batch]]; xin = xb if noise is None else noise(xb, rng)  # noise: Day 127's corruption, if any
            A = forward(net, xin, acts); _, d = recon_loss(A[-1], xb, kind)
            opt.step(net, backward(net, A, acts, d)[0])
        hist.append((recon_loss(forward(net, X[:2000], acts)[-1], X[:2000], kind)[0],
                     None if Xval is None else recon_loss(forward(net, Xval, acts)[-1], Xval, kind)[0]))
    return net, hist


def reconstruct(net, X, acts, kind="bce"):              # NEW: the network's output as an image
    L = forward(net, X, acts)[-1]
    return sigmoid(L) if kind == "bce" else L


# =============================================================================
# DEMOS
# =============================================================================
def mse(R, X):                                          # mean squared error per pixel
    return float(((R - X) ** 2).mean())


def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- a 2-1-2 linear autoencoder on points along the line y = 2x")
    print("=" * 70)
    t = np.linspace(-1, 1, 41, dtype=np.float32); X = np.c_[t, 2 * t] + np.random.RandomState(0).normal(0, 0.05, (41, 2)).astype(np.float32)
    net, hist = train_ae(X, [2, 1, 2], ["linear", "linear"], "mse", epochs=2000, batch=41, lr=0.01, seed=0)
    w = net[1][0][0]; w = w / np.linalg.norm(w) * np.sign(w[0])  # the decoder's one direction, as a unit vector
    _, _, V = d123.pca_fit(X.astype(np.float64)); v = V[:, 0] * np.sign(V[0, 0])  # Day 123's first component of the same points
    print(f"squared error per point: {hist[0][0]:.4f} after the first epoch, {hist[-1][0]:.4f} after 2,000")
    print(f"the decoder's direction ({w[0]:.4f}, {w[1]:.4f}); PCA's first component of the same noisy points ({v[0]:.4f}, {v[1]:.4f}); "
          f"the line without noise (1, 2)/sqrt(5) = ({1 / np.sqrt(5):.4f}, {2 / np.sqrt(5):.4f})")
    print("one number per point (its position along the line) is all the code can hold, so the network learns the line")
    print("Actual output from this lesson's run.")


def demo_gradcheck():
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- backpropagation against finite differences, through the whole network")
    print("=" * 70)
    rng = np.random.RandomState(0); xs = rng.rand(5, 6); acts = ["relu", "relu", "linear"]
    for kind in ("bce", "mse"):
        net = [(W.astype(np.float64), b.astype(np.float64)) for W, b in init_net([6, 5, 3, 6], 0)]  # float64 for the check
        A = forward(net, xs, acts); _, d = recon_loss(A[-1], xs, kind); g, _ = backward(net, A, acts, d); worst = 0.0
        for li, (W, b) in enumerate(net):
            for P, G in ((W, g[li][0]), (b, g[li][1])):
                for idx in np.ndindex(P.shape):           # nudge every weight and bias both ways
                    o = P[idx]; P[idx] = o + 1e-6; lp = recon_loss(forward(net, xs, acts)[-1], xs, kind)[0]
                    P[idx] = o - 1e-6; lm = recon_loss(forward(net, xs, acts)[-1], xs, kind)[0]; P[idx] = o
                    worst = max(worst, abs((lp - lm) / 2e-6 - G[idx]))
        print(f"  {kind}: largest gap over all {sum(W.size + b.size for W, b in net)} parameters {worst:.1e}")
    print("Actual output from this lesson's run.")


def demo_linear(Xtr, Xte):
    print("\n" + "=" * 70)
    print("3. A LINEAR AUTOENCODER REDISCOVERS PCA -- 784 -> 32 -> 784, no ReLU, squared error, 20,000 training images")
    print("=" * 70)
    t0 = time.perf_counter(); net, hist = train_ae(Xtr, [784, 32, 784], ["linear", "linear"], "mse", epochs=40, lr=3e-3, seed=0, Xval=Xte)
    R = reconstruct(net, Xte, ["linear", "linear"], "mse"); mu, vals, V = d123.pca_fit(Xtr.astype(np.float64))
    Rp = d123.reconstruct(d123.transform(Xte, mu, V, 32), mu, V, 32)
    Q, _ = np.linalg.qr(net[1][0].T.astype(np.float64))   # an orthonormal basis for the decoder's 32 directions
    cos = np.linalg.svd(Q.T @ V[:, :32], compute_uv=False)  # cosines of the angles between the two 32-dimensional subspaces
    print(f"test error per pixel: linear autoencoder {mse(R, Xte):.5f} after 40 epochs ({time.perf_counter() - t0:.0f} s); PCA with 32 components {mse(Rp, Xte):.5f}")
    print(f"held-out error every 10 epochs: {[round(h[1], 3) for h in hist[9::10]]} (squared error summed per image)")
    print(f"the decoder's subspace against PCA's: cosines of the principal angles from {cos.max():.4f} down to {cos.min():.4f} (1 = the same direction)")
    print("a linear autoencoder can only learn PCA's subspace; it may not find all of it, and gradient descent gets there slowly")
    print("Actual output from this lesson's run.")


def demo_sizes(Xtr, Xte):
    print("\n" + "=" * 70)
    print("4. NONLINEAR AGAINST PCA -- 784 -> 256 -> k -> 256 -> 784 with ReLU, cross-entropy, 20 epochs")
    print("=" * 70)
    mu, vals, V = d123.pca_fit(Xtr.astype(np.float64)); out = {}
    print(f"  {'code size':>10}{'PCA':>10}{'autoencoder':>13}{'seconds':>9}")
    for k in (2, 8, 32):
        t0 = time.perf_counter(); net, hist = train_ae(Xtr, [784, 256, k, 256, 784], CODE_ACTS, "bce", epochs=20, seed=0, Xval=Xte)
        R = reconstruct(net, Xte, CODE_ACTS); Rp = d123.reconstruct(d123.transform(Xte, mu, V, k), mu, V, k)
        out[k] = (net, hist, mse(Rp, Xte), mse(R, Xte))
        print(f"  {k:>10}{mse(Rp, Xte):>10.5f}{mse(R, Xte):>13.5f}{time.perf_counter() - t0:>9.0f}")
    print("test error per pixel (pixels from 0 to 1); the same 10,000 test images for both")
    print("Actual output from this lesson's run.")
    return out, (mu, V)


def demo_data(X, Xte):
    print("\n" + "=" * 70)
    print("5. HOW MUCH DATA -- the same 32-code network on 1,000 and on 20,000 training images, equal numbers of updates")
    print("=" * 70)
    rows = []
    for n, ep in ((1000, 400), (20000, 20)):                # 400 x 1,000 = 20 x 20,000 images seen
        net, hist = train_ae(X[:n], [784, 256, 32, 256, 784], CODE_ACTS, "bce", epochs=ep, seed=0, Xval=Xte[:2000])  # curves on 2,000 test images
        tr = mse(reconstruct(net, X[:min(n, 2000)], CODE_ACTS), X[:min(n, 2000)]); te = mse(reconstruct(net, Xte, CODE_ACTS), Xte); rows.append((n, hist))
        print(f"  {n:>6} training images: error per pixel on them {tr:.5f}, on the test images {te:.5f}; gap {te - tr:+.5f}")
    print("Actual output from this lesson's run.")
    return rows


def demo_codes(X, y, net32, pca):
    print("\n" + "=" * 70)
    print("6. WHAT THE CODE HOLDS -- 32 numbers per image; labels used only to judge")
    print("=" * 70)
    mu, V = pca; tr, te = np.arange(5000), np.arange(60000, 62000)  # 5,000 training images as the reference set, 2,000 test images
    reps = {"raw pixels (784)": X, "PCA (32)": d123.transform(X.astype(np.float64), mu, V, 32), "autoencoder code (32)": forward(net32, X, CODE_ACTS)[2]}
    print(f"  {'representation':<24}{'kNN accuracy (k = 3)':>22}{'k-means ARI (k = 10)':>22}")
    for name, Z in reps.items():
        Z = np.asarray(Z, np.float64); D = d119.sqdist(Z[te], Z[tr]); nn = np.argsort(D, 1)[:, :3]  # the 3 nearest reference images
        pred = np.array([np.bincount(row, minlength=10).argmax() for row in y[tr][nn]])
        a = d119.ari(d119.kmeans(Z[te], 10, 0, restarts=3)[0], y[te])
        print(f"  {name:<24}{np.mean(pred == y[te]):>22.4f}{a:>22.4f}")
    print("Actual output from this lesson's run.")


def demo_gaps(X, net32):
    print("\n" + "=" * 70)
    print("7. DECODING CODES IT HAS NEVER SEEN -- the gaps in code space")
    print("=" * 70)
    C = forward(net32, X[:20000], CODE_ACTS)[2]; mu, sd = C.mean(0), C.std(0); rng = np.random.RandomState(0)
    real = sigmoid(forward(net32[2:], C[:500], CODE_ACTS[2:])[-1])  # decoding the codes of real images
    fake = sigmoid(forward(net32[2:], (mu + sd * rng.randn(500, 32)).astype(np.float32), CODE_ACTS[2:])[-1])  # codes drawn at random, same mean and spread
    ref = X[20000:30000]
    near = lambda R: np.sqrt(np.maximum(d119.sqdist(R.astype(np.float64), ref.astype(np.float64)), 0).min(1) / 784)  # distance to the nearest real image, per pixel
    print(f"distance to the nearest of 10,000 other real images (root mean square per pixel): decoded real codes {near(real).mean():.4f}, "
          f"decoded random codes {near(fake).mean():.4f}")
    print("random codes land in regions no training image used, and decode to blurs: the code space has gaps (Day 129's VAE closes them)")
    print("Actual output from this lesson's run.")
    return real, fake


def plot_results(Xte, sizes, data_rows, real, fake):    # five panels: reconstructions | errors | learning curves | decoded codes
    fig = plt.figure(figsize=(16, 9.6)); G = fig.add_gridspec(2, 3)  # a 2 x 3 grid of panels
    sub = G[0, :2].subgridspec(4, 10)                       # rows: originals, PCA 32, autoencoder 2, autoencoder 32
    mu, V = sizes["pca"]; R32 = reconstruct(sizes[32][0], Xte[:10], CODE_ACTS); R2 = reconstruct(sizes[2][0], Xte[:10], CODE_ACTS)
    Rp = d123.reconstruct(d123.transform(Xte[:10].astype(np.float64), mu, V, 32), mu, V, 32)
    for r, (imgs, name) in enumerate(((Xte[:10], "original"), (Rp, "PCA, 32"), (R2, "autoencoder, 2"), (R32, "autoencoder, 32"))):
        for c in range(10):
            a = fig.add_subplot(sub[r, c]); a.imshow(np.clip(imgs[c], 0, 1).reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_ylabel(name, fontsize=8)
    a = fig.add_subplot(G[0, 2]); ks = [2, 8, 32]
    a.plot(ks, [sizes[k][2] for k in ks], marker="o", label="PCA"); a.plot(ks, [sizes[k][3] for k in ks], marker="s", label="autoencoder")
    a.set_xscale("log", base=2); a.set_xticks(ks); a.set_xticklabels([str(k) for k in ks]); a.minorticks_off()
    a.set_xlabel("code size"); a.set_ylabel("test error per pixel"); a.set_title("Reconstruction error", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = fig.add_subplot(G[1, 0])
    for n, hist in data_rows:
        h = np.array([[p[0], p[1]] for p in hist], float); x = np.linspace(0, 1, len(h))
        a.plot(x, h[:, 0], label=f"{n:,} images: training"); a.plot(x, h[:, 1], linestyle="--", label=f"{n:,} images: test")
    a.set_xlabel("share of training done"); a.set_ylabel("cross-entropy per image"); a.set_ylim(55, 140); a.legend(fontsize=8); a.grid(alpha=0.3)
    a.set_title("Learning curves", fontsize=10)
    for g, imgs, title in ((G[1, 1], real, "Decoded codes of real images"), (G[1, 2], fake, "Decoded random codes")):
        sub = g.subgridspec(4, 6)
        for j in range(24):
            b = fig.add_subplot(sub[j // 6, j % 6]); b.imshow(imgs[j].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); b.set_xticks([]); b.set_yticks([])
            if j == 2:
                b.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig("autoencoder_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved autoencoder_results.png")


def draw_diagram():                                     # encoder, code, decoder
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 126: an autoencoder", ha="center", fontsize=12, fontweight="bold")
    box(1.2, 2.3, 1.8, 2.8, "image\n784 pixels", "#FAEEDA")
    box(3.6, 2.3, 2.0, 2.2, "encoder\n784 -> 256\n(ReLU)", "#DCEBF7")
    box(6.5, 2.3, 1.8, 1.0, "code: k\nnumbers", "#E5F2E1")
    box(9.4, 2.3, 2.0, 2.2, "decoder\nk -> 256 -> 784\n(ReLU)", "#DCEBF7")
    box(11.8, 2.3, 1.8, 2.8, "rebuilt\nimage", "#FAEEDA")
    for x0, x1 in ((2.14, 2.56), (4.64, 5.56), (7.44, 8.36), (10.44, 10.86)):
        ax.annotate("", xy=(x1, 2.3), xytext=(x0, 2.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 0.45, "trained only to make the output match the input: no labels; the narrow code forces it to keep what matters", ha="center", fontsize=8.5)
    plt.savefig("autoencoder_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved autoencoder_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 127 is denoising autoencoders: trained to rebuild clean images from damaged ones, with a real noise-removal demonstration.")


def main():                                             # runs every section in order
    X, y = mnist(); Xtr, Xte = X[:20000], X[60000:]        # the first 20,000 training images; the official 10,000 test images
    demo_hand()                                            # 1. by hand
    demo_gradcheck()                                       # 2. gradient check
    demo_linear(Xtr, Xte)                                  # 3. linear autoencoder and PCA
    sizes, pca = demo_sizes(Xtr, Xte); sizes["pca"] = pca  # 4. nonlinear against PCA
    data_rows = demo_data(X, Xte)                          # 5. how much data
    demo_codes(X, y, sizes[32][0], pca)                    # 6. what the code holds
    real, fake = demo_gaps(X, sizes[32][0])                # 7. gaps in code space
    plot_results(Xte, sizes, data_rows, real, fake)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
