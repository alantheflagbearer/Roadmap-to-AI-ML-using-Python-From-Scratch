"""
Day 127: Denoising Autoencoders -- a Real Noise-Removal Demonstration

A plain autoencoder (Day 126) learns to copy clean images. A denoising
autoencoder is shown a DAMAGED image and asked for the CLEAN one. It cannot
just copy, so it has to learn what handwritten digits look like, and use that
knowledge to undo the damage.

Everything is written from scratch in NumPy, on Day 126's network parts:
  * three kinds of damage: Gaussian noise, masking (pixels set to 0), and a
    hidden square
  * peak signal-to-noise ratio (PSNR), the usual score for image restoration
  * two baselines: PCA projection (Day 123) and a 3 x 3 median filter

The data: MNIST (cached by Day 126), damaged on purpose so the clean answer is
known exactly.

  1. By hand: PSNR, and what each kind of damage does.
  2. Plain against denoising, against the baselines.
  3. Training noise against test noise.
  4. The wrong kind of noise.
  5. What the code holds, when the input is damaged.
  6. Filling in a hidden square.

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
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "week-18-unsupervised-learning-i"))
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: train_ae, forward, reconstruct
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day123_exercise as d123                      # Day 123: pca_fit, transform, reconstruct
    import day119_exercise as d119                      # Day 119: sqdist
ACTS = ["relu", "linear", "relu", "linear"]             # Day 126's 784 -> 256 -> 32 -> 256 -> 784 network
SIZES = [784, 256, 32, 256, 784]


# =============================================================================
# DAMAGE, AND HOW TO SCORE A REPAIR -- NEW
# =============================================================================
def gaussian(X, sigma, rng):                            # NEW: add N(0, sigma^2) to every pixel, kept inside [0, 1]
    return np.clip(X + rng.normal(0, sigma, X.shape).astype(np.float32), 0, 1)


def masking(X, p, rng):                                 # NEW: set each pixel to 0 with probability p
    return X * (rng.uniform(size=X.shape) >= p).astype(np.float32)


def hide_square(X, size=10, rng=None):                  # NEW: blank a size x size square at a random place in each image
    rng = rng or np.random.RandomState(0); Y = X.reshape(-1, 28, 28).copy(); M = np.zeros_like(Y, bool)
    for i, (r, c) in enumerate(rng.randint(4, 24 - size, size=(len(Y), 2))):  # squares stay away from the empty border
        Y[i, r:r + size, c:c + size] = 0; M[i, r:r + size, c:c + size] = True
    return Y.reshape(len(X), -1), M.reshape(len(X), -1)


def psnr(R, X):                                         # NEW: peak signal-to-noise ratio in decibels, pixels in [0, 1]; higher is better
    return float(10 * np.log10(1.0 / np.mean((R - X) ** 2)))


def median3(X):                                         # NEW: each pixel replaced by the median of its 3 x 3 neighbourhood
    P = np.pad(X.reshape(-1, 28, 28), ((0, 0), (1, 1), (1, 1)))
    stack = np.stack([P[:, i:i + 28, j:j + 28] for i in range(3) for j in range(3)])  # nine shifted copies
    return np.median(stack, 0).reshape(len(X), -1)


def pca_denoise(Xn, pca, k=32):                         # NEW: project the damaged images onto PCA's top k directions of clean images
    mu, V = pca
    return np.clip(d123.reconstruct(d123.transform(Xn.astype(np.float64), mu, V, k), mu, V, k), 0, 1)


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand(Xte):
    print("=" * 70)
    print("1. BY HAND -- PSNR, and what each kind of damage does")
    print("=" * 70)
    x = Xte[:1]; rng = np.random.RandomState(0); n = gaussian(x, 0.3, rng)
    m = float(np.mean((n - x) ** 2))
    print(f"one test image with Gaussian noise (sigma 0.3): mean squared error {m:.5f}; by hand 10 log10(1 / {m:.5f}) = {10 * np.log10(1 / m):.2f} dB; psnr() {psnr(n, x):.2f} dB")
    print("(sigma^2 = 0.09 would be the error without clipping; clipping to [0, 1] removes part of it, because most pixels are 0)")
    for name, f in (("Gaussian, sigma 0.3", lambda X: gaussian(X, 0.3, np.random.RandomState(1))), ("masking, half the pixels", lambda X: masking(X, 0.5, np.random.RandomState(1))),
                    ("a hidden 10 x 10 square", lambda X: hide_square(X)[0])):
        D = f(Xte); print(f"  {name:<26}: PSNR of the damaged test images {psnr(D, Xte):.2f} dB")
    print("Actual output from this lesson's run.")


def train(Xtr, Xte, noise, seed=0):                     # Day 126's training, with or without damage on the way in
    t0 = time.perf_counter(); net, hist = d126.train_ae(Xtr, SIZES, ACTS, "bce", epochs=20, seed=seed, Xval=Xte[:2000], noise=noise)
    return net, time.perf_counter() - t0


def demo_compare(Xtr, Xte, pca):
    print("\n" + "=" * 70)
    print("2. PLAIN AGAINST DENOISING, AGAINST THE BASELINES -- PSNR on 10,000 damaged test images (dB, higher is better)")
    print("=" * 70)
    plain, t1 = train(Xtr, Xte, None)
    dae_g, t2 = train(Xtr, Xte, lambda X, rng: gaussian(X, 0.3, rng))
    dae_m, t3 = train(Xtr, Xte, lambda X, rng: masking(X, 0.5, rng))
    print(f"three networks trained, 20 epochs each on 20,000 images: {t1:.0f}, {t2:.0f} and {t3:.0f} s")
    tests = {"Gaussian 0.3": gaussian(Xte, 0.3, np.random.RandomState(1)), "masking 0.5": masking(Xte, 0.5, np.random.RandomState(1))}
    print(f"  {'method':<36}" + "".join(f"{t:>14}" for t in tests))
    rows = {"no repair (the damaged image)": lambda D: D, "3 x 3 median filter": median3, "PCA, 32 components": lambda D: pca_denoise(D, pca),
            "plain autoencoder (Day 126)": lambda D: d126.reconstruct(plain, D, ACTS), "denoising, trained on Gaussian 0.3": lambda D: d126.reconstruct(dae_g, D, ACTS),
            "denoising, trained on masking 0.5": lambda D: d126.reconstruct(dae_m, D, ACTS)}
    out = {}
    for name, f in rows.items():
        out[name] = [psnr(f(D), Xte) for D in tests.values()]
        print(f"  {name:<36}" + "".join(f"{v:>14.2f}" for v in out[name]))
    print("Actual output from this lesson's run.")
    return plain, dae_g, dae_m, tests


def demo_levels(Xtr, Xte, dae_g):
    print("\n" + "=" * 70)
    print("3. TRAINING NOISE AGAINST TEST NOISE -- Gaussian noise; PSNR (dB) of the repaired test images")
    print("=" * 70)
    nets = {0.3: dae_g}
    for s in (0.1, 0.5):
        nets[s], _ = train(Xtr, Xte, lambda X, rng, s=s: gaussian(X, s, rng))
    levels = (0.1, 0.3, 0.5); grid = np.zeros((3, 3))
    print("  trained on    " + "".join(f"{'tested on ' + str(t):>16}" for t in levels))
    for i, s in enumerate(levels):
        for j, t in enumerate(levels):
            grid[i, j] = psnr(d126.reconstruct(nets[s], gaussian(Xte, t, np.random.RandomState(2)), ACTS), Xte)
        print(f"  sigma {s:<7}" + "".join(f"{v:>16.2f}" for v in grid[i]))
    print("  no repair    " + "".join(f"{psnr(gaussian(Xte, t, np.random.RandomState(2)), Xte):>16.2f}" for t in levels))
    print("Actual output from this lesson's run.")
    return nets, grid


def demo_mismatch(Xte, dae_g, dae_m):
    print("\n" + "=" * 70)
    print("4. THE WRONG KIND OF NOISE -- each denoiser on the other's damage, and on salt-and-pepper it never saw")
    print("=" * 70)
    rng = np.random.RandomState(3); sp = Xte.copy(); u = rng.uniform(size=Xte.shape); sp[u < 0.1] = 0.0; sp[u > 0.9] = 1.0  # 20% of pixels forced to black or white
    tests = {"Gaussian 0.3": gaussian(Xte, 0.3, np.random.RandomState(1)), "masking 0.5": masking(Xte, 0.5, np.random.RandomState(1)), "salt and pepper 20%": sp}
    print(f"  {'trained on':<18}" + "".join(f"{t:>22}" for t in tests))
    for name, net in (("Gaussian 0.3", dae_g), ("masking 0.5", dae_m)):
        print(f"  {name:<18}" + "".join(f"{psnr(d126.reconstruct(net, D, ACTS), Xte):>22.2f}" for D in tests.values()))
    print(f"  {'no repair':<18}" + "".join(f"{psnr(D, Xte):>22.2f}" for D in tests.values()))
    print(f"  {'3 x 3 median':<18}" + "".join(f"{psnr(median3(D), Xte):>22.2f}" for D in tests.values()))
    print("Actual output from this lesson's run.")


def demo_codes(X, y, plain, dae_g):
    print("\n" + "=" * 70)
    print("5. WHAT THE CODE HOLDS WHEN THE INPUT IS DAMAGED -- kNN (k = 3) on the 32-number codes, labels used only to judge")
    print("=" * 70)
    tr, te = np.arange(5000), np.arange(60000, 62000); rng = np.random.RandomState(4)
    print(f"  {'network':<22}{'clean test images':>20}{'Gaussian 0.3':>16}{'Gaussian 0.5':>16}")
    for name, net in (("plain", plain), ("denoising (0.3)", dae_g)):
        ref = d126.forward(net, X[tr], ACTS)[2]; accs = []
        for D in (X[te], gaussian(X[te], 0.3, rng), gaussian(X[te], 0.5, rng)):
            C = d126.forward(net, D, ACTS)[2]; nn = np.argsort(d119.sqdist(C.astype(np.float64), ref.astype(np.float64)), 1)[:, :3]
            accs.append(np.mean(np.array([np.bincount(r, minlength=10).argmax() for r in y[tr][nn]]) == y[te]))
        print(f"  {name:<22}" + "".join(f"{a:>{w}.4f}" for a, w in zip(accs, (20, 16, 16))))
    print("the reference codes come from clean training images; only the test images are damaged")
    print("Actual output from this lesson's run.")


def demo_square(Xte, plain, dae_m, pca):
    print("\n" + "=" * 70)
    print("6. FILLING IN A HIDDEN SQUARE -- 10 x 10 pixels blanked at a random place in each test image")
    print("=" * 70)
    D, M = hide_square(Xte); err = lambda R: float(np.sqrt(np.mean(((R - Xte)[M]) ** 2)))  # error only inside the hidden squares
    for name, R in (("left blank", D), ("PCA, 32 components", pca_denoise(D, pca)), ("plain autoencoder", d126.reconstruct(plain, D, ACTS)),
                    ("denoising, trained on masking 0.5", d126.reconstruct(dae_m, D, ACTS))):
        print(f"  {name:<34}: error inside the square (root mean square) {err(R):.4f}; PSNR of the whole image {psnr(R, Xte):.2f} dB")
    print("masking taught the network to guess missing pixels from the rest of the digit, though never a whole square at once")
    print("Actual output from this lesson's run.")
    return D


def plot_results(Xte, tests, plain, dae_g, dae_m, pca, grid, D):  # five panels
    fig = plt.figure(figsize=(16, 9.6)); G = fig.add_gridspec(2, 3)
    sub = G[0, :2].subgridspec(5, 8); Dg = tests["Gaussian 0.3"][:8]
    rows = ((Xte[:8], "clean"), (Dg, "Gaussian 0.3"), (median3(Dg), "median 3 x 3"), (d126.reconstruct(plain, Dg, ACTS), "plain AE"),
            (d126.reconstruct(dae_g, Dg, ACTS), "denoising AE"))
    for r, (imgs, name) in enumerate(rows):
        for c in range(8):
            a = fig.add_subplot(sub[r, c]); a.imshow(imgs[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_ylabel(name, fontsize=8)
    a = fig.add_subplot(G[0, 2]); im = a.imshow(grid, cmap="viridis"); fig.colorbar(im, ax=a, label="PSNR (dB)")
    a.set_xticks(range(3)); a.set_xticklabels(["0.1", "0.3", "0.5"]); a.set_yticks(range(3)); a.set_yticklabels(["0.1", "0.3", "0.5"])
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="w", fontsize=9)
    a.set_xlabel("test noise (sigma)"); a.set_ylabel("training noise (sigma)"); a.set_title("Training against test noise", fontsize=10)
    sub = G[1, :2].subgridspec(4, 8); Dm = tests["masking 0.5"][:8]
    rows = ((Dm, "masking 0.5"), (d126.reconstruct(dae_m, Dm, ACTS), "denoising AE"), (D[:8], "hidden square"), (d126.reconstruct(dae_m, D[:8], ACTS), "filled in"))
    for r, (imgs, name) in enumerate(rows):
        for c in range(8):
            a = fig.add_subplot(sub[r, c]); a.imshow(imgs[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_ylabel(name, fontsize=8)
    a = fig.add_subplot(G[1, 2]); W = dae_g[0][0][:, :16].T  # the first 16 encoder filters
    a.imshow(np.vstack([np.hstack([W[r * 4 + c].reshape(28, 28) for c in range(4)]) for r in range(4)]), cmap="RdBu_r"); a.set_xticks([]); a.set_yticks([])
    a.set_title("16 first-layer filters of the denoising AE", fontsize=10)
    plt.tight_layout(); plt.savefig("denoising_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved denoising_results.png")


def draw_diagram():                                     # damage in, clean out
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 127: a denoising autoencoder", ha="center", fontsize=12, fontweight="bold")
    box(1.2, 2.3, 1.8, 1.6, "clean image\nx", "#FAEEDA")
    box(3.6, 2.3, 2.0, 1.6, "damage it:\nnoise, masking", "#F7DCDC")
    box(6.5, 2.3, 2.6, 1.6, "Day 126's network\n784 -> 256 -> 32\n-> 256 -> 784", "#DCEBF7")
    box(9.4, 2.3, 1.8, 1.6, "repaired\nimage", "#E5F2E1")
    box(11.8, 2.3, 1.8, 1.6, "compare with\nthe CLEAN x", "#EEEDFE")
    for x0, x1 in ((2.14, 2.56), (4.64, 5.16), (7.84, 8.46), (10.34, 10.86)):
        ax.annotate("", xy=(x1, 2.3), xytext=(x0, 2.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(11.8, 3.15), xytext=(1.2, 3.15), arrowprops=dict(arrowstyle="->", color="#888888", connectionstyle="arc3,rad=-0.15"))
    ax.text(6.5, 0.6, "the only change from Day 126: the input is damaged, the target is not", ha="center", fontsize=8.5)
    plt.savefig("denoising_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved denoising_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 128 is anomaly detection with autoencoders: flagging what the network cannot rebuild, with real thresholds.")


def main():                                             # runs every section in order
    X, y = d126m.mnist(); Xtr, Xte = X[:20000], X[60000:]
    mu, vals, V = d123.pca_fit(Xtr.astype(np.float64)); pca = (mu, V)  # Day 123's PCA of clean training images
    demo_hand(Xte)                                         # 1. PSNR by hand
    plain, dae_g, dae_m, tests = demo_compare(Xtr, Xte, pca)  # 2. the comparison
    nets, grid = demo_levels(Xtr, Xte, dae_g)              # 3. training against test noise
    demo_mismatch(Xte, dae_g, dae_m)                       # 4. the wrong noise
    demo_codes(X, y, plain, dae_g)                         # 5. codes of damaged images
    D = demo_square(Xte, plain, dae_m, pca)                # 6. a hidden square
    plot_results(Xte, tests, plain, dae_g, dae_m, pca, grid, D)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
