"""
Day 130: Training a VAE, and a Real Latent-Space Visualisation

Day 129 built the variational autoencoder and its reparameterization trick.
Today it is trained properly and examined: its two-dimensional code space is
drawn, new digits are generated from codes nobody encoded, and it is compared
with Day 126's plain autoencoder, whose code space has gaps.

Everything is written from scratch in NumPy, on Days 126 and 129's parts:
  * VAEs with 2 and 16 code dimensions, and a plain autoencoder for comparison
  * sampling from the prior, decoding a grid of codes, and interpolation
  * beta, the weight on the KL term: how much the codes are pulled to N(0, 1)

The data: MNIST (cached by Day 126); labels used only to colour and judge.

  1. Training: a 16-code VAE against a 16-code plain autoencoder.
  2. By hand: the loss of one image, term by term.
  3. The two-dimensional code space: a real map of the digits.
  4. Generating digits: codes drawn at random, VAE against plain autoencoder.
  5. Walking between two digits: interpolation in code space.
  6. Beta: trading reconstruction against a well-filled code space.

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
    import day126_exercise as d126                      # Day 126: forward, train_ae, sigmoid
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day129_exercise as d129                      # Day 129: kl_normal, vae_loss, train_vae
    import day119_exercise as d119                      # Day 119: sqdist
ENC = ["relu", "linear"]; DEC = ["relu", "linear"]      # the VAE's encoder and decoder layers (Day 129)
AE = ["relu", "linear", "relu", "linear"]               # the plain autoencoder (Day 126)


# =============================================================================
# USING A TRAINED VAE -- NEW
# =============================================================================
def encode(enc, X):                                     # NEW: the code cloud of each image: (mean, log-variance)
    H = d126.forward(enc, X, ENC)[-1]; k = H.shape[1] // 2
    return H[:, :k], H[:, k:]


def decode(dec, Z):                                     # NEW: images from codes
    return d126.sigmoid(d126.forward(dec, Z.astype(np.float32), DEC)[-1])


def sample_prior(dec, n, k, rng):                       # NEW: new images from codes drawn from N(0, 1)
    return decode(dec, rng.standard_normal((n, k)))


def interpolate(z0, z1, steps):                         # NEW: evenly spaced codes on the straight line from z0 to z1
    t = np.linspace(0, 1, steps)[:, None]
    return (1 - t) * z0 + t * z1


def nearest_real(R, ref, chunk=500):                    # NEW: root-mean-square distance per pixel to the nearest real image
    return np.concatenate([np.sqrt(np.maximum(d119.sqdist(R[s:s + chunk].astype(np.float64), ref), 0).min(1) / R.shape[1])
                           for s in range(0, len(R), chunk)])


def active_units(enc, X, thresh=0.1):                   # NEW: code dimensions whose KL averages above thresh nats (the others are unused)
    mu, lv = encode(enc, X); kl = (0.5 * (np.exp(lv) + mu ** 2 - 1 - lv)).mean(0)
    return int((kl > thresh).sum()), kl


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand(enc16, dec16, X):
    print("\n" + "=" * 70)
    print("2. BY HAND -- the loss of one test image, term by term (16-code VAE)")
    print("=" * 70)
    x = X[60000:60001]; mu, lv = encode(enc16, x)
    kl_hand = float(sum(0.5 * (np.exp(v) + m ** 2 - 1 - v) for m, v in zip(mu[0], lv[0])))
    _, rec, kl, _, _ = d129.vae_loss(enc16, dec16, x, np.random.RandomState(0))
    print(f"its 16 code means range from {mu.min():.3f} to {mu.max():.3f}; its spreads (sigma) from {np.exp(0.5 * lv).min():.3f} to {np.exp(0.5 * lv).max():.3f}")
    print(f"KL by hand, summing 0.5 (sigma^2 + mu^2 - 1 - log sigma^2) over the 16 dimensions: {kl_hand:.3f} nats; vae_loss: {kl:.3f}")
    print(f"reconstruction (cross-entropy, one draw of z): {rec:.3f} nats; loss = {rec:.3f} + {kl:.3f} = {rec + kl:.3f}, the negative ELBO")
    print("Actual output from this lesson's run.")


def demo_train(X):
    print("=" * 70)
    print("1. TRAINING -- a 16-code VAE against a 16-code plain autoencoder, 20,000 images, 30 epochs each")
    print("=" * 70)
    Xtr, Xte = X[:20000], X[60000:]
    t0 = time.perf_counter(); enc16, dec16, hist = d129.train_vae(Xtr, 16, epochs=30, Xval=Xte[:2000]); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); ae, _ = d126.train_ae(Xtr, [784, 256, 16, 256, 784], AE, "bce", epochs=30, seed=0); t2 = time.perf_counter() - t0
    loss, rec, kl = d129.vae_loss(enc16, dec16, Xte, np.random.RandomState(1))[:3]
    mu, _ = encode(enc16, Xte); Rv = decode(dec16, mu); Ra = d126.reconstruct(ae, Xte, AE)
    print(f"VAE: test loss {loss:.2f} nats per image = reconstruction {rec:.2f} + KL {kl:.2f}; trained in {t1:.0f} s")
    print(f"error per pixel, rebuilding from the code mean: VAE {np.mean((Rv - Xte) ** 2):.5f}; plain autoencoder {np.mean((Ra - Xte) ** 2):.5f} (trained in {t2:.0f} s)")
    print(f"held-out loss every 10 epochs: {[round(h[0], 2) for h in hist[9::10]]}")
    print("the VAE rebuilds a little worse: the KL term spends some of its capacity keeping the codes near N(0, 1)")
    print("Actual output from this lesson's run.")
    return enc16, dec16, ae, hist


def demo_map(X, y):
    print("\n" + "=" * 70)
    print("3. THE TWO-DIMENSIONAL CODE SPACE -- a 2-code VAE, 30 epochs")
    print("=" * 70)
    enc2, dec2, hist = d129.train_vae(X[:20000], 2, epochs=30, Xval=X[60000:62000]); te = np.arange(60000, 70000)
    mu, lv = encode(enc2, X[te]); ref, _ = encode(enc2, X[:5000])
    nn = np.argsort(d119.sqdist(mu.astype(np.float64), ref.astype(np.float64)), 1)[:, :3]
    acc = np.mean(np.array([np.bincount(r, minlength=10).argmax() for r in y[:5000][nn]]) == y[te])
    print(f"test loss {hist[-1][0]:.2f} nats (reconstruction {hist[-1][1]:.2f}, KL {hist[-1][2]:.2f}) with only 2 numbers per image")
    print(f"code means of the 10,000 test images: mean ({mu[:, 0].mean():.3f}, {mu[:, 1].mean():.3f}), spread ({mu[:, 0].std():.3f}, {mu[:, 1].std():.3f}); "
          f"N(0, 1) would give (0, 0) and (1, 1)")
    print(f"kNN (k = 3) on the 2 code numbers, labels used only to judge: {acc:.4f}")
    print("Actual output from this lesson's run.")
    return enc2, dec2, mu


def demo_generate(X, dec16, ae):
    print("\n" + "=" * 70)
    print("4. GENERATING DIGITS -- 1,000 codes drawn at random, each model's own way")
    print("=" * 70)
    ref = X[20000:30000].astype(np.float64); rng = np.random.RandomState(2)  # 10,000 real images neither model trained on
    C = d126.forward(ae, X[:20000], AE)[2]; m, s = C.mean(0), C.std(0)
    fake_ae = d126.sigmoid(d126.forward(ae[2:], (m + s * rng.standard_normal((1000, 16))).astype(np.float32), AE[2:])[-1])  # Day 126's way: match the codes' mean and spread
    fake_vae = sample_prior(dec16, 1000, 16, rng)
    real = X[30000:31000]                                 # real images, as a yardstick
    for name, R in (("real images (yardstick)", real), ("plain autoencoder, random codes", fake_ae), ("VAE, codes from N(0, 1)", fake_vae)):
        d = nearest_real(R, ref); print(f"  {name:<34}: distance to the nearest real image, per pixel: median {np.median(d):.4f}, 90th percentile {np.quantile(d, 0.9):.4f}")
    print("Actual output from this lesson's run.")
    return fake_ae, fake_vae


def demo_interp(X, enc16, dec16, ae):
    print("\n" + "=" * 70)
    print("5. WALKING BETWEEN TWO DIGITS -- 9 evenly spaced codes between two test images, 200 pairs")
    print("=" * 70)
    ref = X[20000:30000].astype(np.float64); rng = np.random.RandomState(3); pairs = rng.randint(60000, 70000, size=(200, 2))
    mids = {"plain autoencoder": [], "VAE": [], "pixels (no model)": []}
    for i, j in pairs:
        za, zb = d126.forward(ae, X[[i, j]], AE)[2]; mids["plain autoencoder"].append(d126.sigmoid(d126.forward(ae[2:], interpolate(za, zb, 9)[1:-1].astype(np.float32), AE[2:])[-1]))
        mu, _ = encode(enc16, X[[i, j]]); mids["VAE"].append(decode(dec16, interpolate(mu[0], mu[1], 9)[1:-1]))
        mids["pixels (no model)"].append(interpolate(X[i], X[j], 9)[1:-1])
    for name, M in mids.items():
        d = nearest_real(np.vstack(M), ref); print(f"  {name:<20}: the 7 in-between images, distance to the nearest real image per pixel: median {np.median(d):.4f}")
    print("Actual output from this lesson's run.")
    return pairs[0]


def demo_beta(X):
    print("\n" + "=" * 70)
    print("6. BETA -- the weight on the KL term; 16-code VAEs, 20 epochs each")
    print("=" * 70)
    ref = X[20000:30000].astype(np.float64); rows = []
    print(f"  {'beta':>5}{'reconstruction':>16}{'KL':>8}{'active dims':>13}{'samples: median distance to real':>35}")
    for beta in (0.5, 1.0, 4.0):
        enc, dec, hist = d129.train_vae(X[:20000], 16, epochs=20, beta=beta, Xval=X[60000:62000])
        _, rec, kl = d129.vae_loss(enc, dec, X[60000:], np.random.RandomState(1), beta)[:3]; n_act, _ = active_units(enc, X[60000:62000])
        d = np.median(nearest_real(sample_prior(dec, 1000, 16, np.random.RandomState(4)), ref)); rows.append((beta, rec, kl, n_act, d))
        print(f"  {beta:>5}{rec:>16.2f}{kl:>8.2f}{n_act:>13}{d:>35.4f}")
    print("Actual output from this lesson's run.")
    return rows


def plot_results(X, y, mu2, dec2, fake_ae, fake_vae, enc16, dec16, ae, pair, rows):  # six panels
    fig = plt.figure(figsize=(16, 10.5)); G = fig.add_gridspec(2, 3)
    a = fig.add_subplot(G[0, 0]); s = a.scatter(mu2[:, 0], mu2[:, 1], c=y[60000:], cmap="tab10", s=3); fig.colorbar(s, ax=a, label="digit")
    a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_title("2-code VAE: test images' code means", fontsize=10); a.set_xlabel("code 1"); a.set_ylabel("code 2")
    a = fig.add_subplot(G[0, 1]); g = np.linspace(-2.5, 2.5, 15); Z = np.array([(u, v) for v in g[::-1] for u in g])
    I = decode(dec2, Z).reshape(15, 15, 28, 28).transpose(0, 2, 1, 3).reshape(15 * 28, 15 * 28)
    a.imshow(I, cmap="gray_r"); a.set_xticks([]); a.set_yticks([]); a.set_title("Decoding a grid of codes from -2.5 to 2.5", fontsize=10)
    for g_, imgs, title in ((G[0, 2], fake_vae, "VAE: codes from N(0, 1)"), (G[1, 2], fake_ae, "Plain AE: random codes")):
        sub = g_.subgridspec(5, 6)
        for j in range(30):
            b = fig.add_subplot(sub[j // 6, j % 6]); b.imshow(imgs[j].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); b.set_xticks([]); b.set_yticks([])
            if j == 2:
                b.set_title(title, fontsize=10)
    sub = G[1, 0:2].subgridspec(4, 9); i, j = pair
    za, zb = d126.forward(ae, X[[i, j]], AE)[2]; mu, _ = encode(enc16, X[[i, j]])
    rows_img = ((interpolate(X[i], X[j], 9), "pixels"), (d126.sigmoid(d126.forward(ae[2:], interpolate(za, zb, 9).astype(np.float32), AE[2:])[-1]), "plain AE"),
                (decode(dec16, interpolate(mu[0], mu[1], 9)), "VAE"))
    for r, (imgs, name) in enumerate(rows_img):
        for c in range(9):
            b = fig.add_subplot(sub[r, c]); b.imshow(imgs[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); b.set_xticks([]); b.set_yticks([])
            if c == 0:
                b.set_ylabel(name, fontsize=8)
    b = fig.add_subplot(sub[3, :]); r = np.array(rows); b.plot(r[:, 2], r[:, 1], marker="o")
    for row in rows:
        b.annotate(f"beta {row[0]}", (row[2], row[1]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    b.set_xlabel("KL (nats)"); b.set_ylabel("reconstruction"); b.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("vae_latent_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved vae_latent_results.png")


def draw_diagram():                                     # what the KL term does to the code space
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 130: a trained VAE, used three ways", ha="center", fontsize=12, fontweight="bold")
    box(1.8, 2.4, 2.8, 1.8, "train: reconstruction\n+ beta x KL to N(0, 1)", "#DCEBF7")
    box(5.6, 3.25, 3.0, 0.9, "encode: a map of the data", "#E5F2E1")
    box(5.6, 2.15, 3.0, 0.9, "sample N(0, 1), decode:\nnew digits", "#E5F2E1")
    box(5.6, 1.05, 3.0, 0.9, "interpolate codes:\nsmooth in-betweens", "#E5F2E1")
    box(10.4, 2.15, 4.0, 2.4, "judge: distance from each\ngenerated image to the nearest\nreal one; compare with Day 126's\nplain autoencoder", "#FAEEDA")
    for y0 in (3.25, 2.15, 1.05):
        ax.annotate("", xy=(4.06, y0), xytext=(3.24, 2.4), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(8.36, 2.15), xytext=(7.14, y0), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("vae_latent_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved vae_latent_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 131 is contrastive learning: pulling two views of the same image together and pushing different images apart.")


def main():                                             # runs every section in order
    X, y = d126m.mnist()
    enc16, dec16, ae, hist = demo_train(X)                 # 1. training
    demo_hand(enc16, dec16, X)                             # 2. by hand, on the trained VAE
    enc2, dec2, mu2 = demo_map(X, y)                       # 3. the 2-D map
    fake_ae, fake_vae = demo_generate(X, dec16, ae)        # 4. generation
    pair = demo_interp(X, enc16, dec16, ae)                # 5. interpolation
    rows = demo_beta(X)                                    # 6. beta
    plot_results(X, y, mu2, dec2, fake_ae, fake_vae, enc16, dec16, ae, pair, rows)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
