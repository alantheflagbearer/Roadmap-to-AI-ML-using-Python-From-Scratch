"""
Day 137: Conditional GANs -- Real Class-Conditional Generation

Day 136's DCGAN drew whatever digit its noise happened to land on: nobody
could ask it for a 7. A CONDITIONAL GAN (Mirza and Osindero, 2014) gives the
label to both players. The generator gets noise AND a label, and must draw
that digit; the discriminator gets an image AND a label, and must say whether
this is a real image of this digit. A perfect 3 handed in with the label 7 is
now a fake, so the generator is pushed to obey the label.

Everything is written from scratch in NumPy, on Day 136's convolutional
players (Conv, ConvT, BN, the layer stack) and Day 134's losses and AdamGAN,
imported unchanged:
  * the label as a one-hot vector joined to the generator's noise
  * the label as ten extra image channels (one lit plane) for the
    discriminator
  * the conditional round of the game, and the gradient split back off the
    joined input

Data: MNIST (cached by Day 126); a softmax-regression judge (Day 134's)
says which digit a generated image shows.

  1. By hand: how the label enters each player, and the gradient checked.
  2. A conditional DCGAN on MNIST: does it draw the digit it is asked for?
  3. The label given to the generator only: what happens when the
     discriminator cannot check it?
  4. Same noise, ten labels: does the noise carry the style?

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
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day119_exercise as d119                      # Day 119: sqdist
    import day116_exercise as d116                      # Day 116: softmax, fit_softmax (the judge of generated digits)
    import day134_exercise as d134                      # Day 134: AdamGAN, d_loss, g_loss
    import day136_exercise as d136                      # Day 136: dcgan_players, net_generate, the layers


# =============================================================================
# THE LABEL, GIVEN TO BOTH PLAYERS -- NEW
# =============================================================================
def onehot(y, n=10):                                    # NEW: labels 0-9 as rows of ten numbers, one of them 1
    E = np.zeros((len(y), n), np.float32); E[np.arange(len(y)), y] = 1                # fancy indexing: row i, column y[i]
    return E


def g_input(Z, y):                                      # NEW: the generator's input: noise and the one-hot label side by side
    return np.concatenate([Z, onehot(y)], axis=1)


def d_input(X, y):                                      # NEW: the discriminator's input: the image and ten label planes, flattened
    P = np.zeros((len(X), 10, 784), np.float32); P[np.arange(len(X)), y] = 1          # plane y[i] all ones, the other nine all zeros
    return np.concatenate([X, P.reshape(len(X), -1)], axis=1)                          # reshaped by D's first layer to 11 x 28 x 28


def cgan_players(seed, k=64, c=32, d_label=True):       # NEW: Day 136's players with room for the label
    G = d136.dcgan_players(seed, k=k + 10, c=c)[0]                                    # the generator reads 64 + 10 numbers
    D = d136.dcgan_players(seed, c=c, ch=11 if d_label else 1)[1]                     # the discriminator reads 11 channels, or only the image
    return G, D


def cgan_step(G, D, oG, oD, X, y, rng, k=64, d_label=True):  # NEW: one round of the conditional game
    n = len(X); join = (lambda I, l: d_input(I, l)) if d_label else (lambda I, l: I)  # without d_label, D sees the image only
    yf = rng.randint(0, 10, n); F = G.forward(g_input(rng.standard_normal((n, k)).astype(np.float32), yf))  # 1. fakes of random digits
    Af = D.forward(join(F, yf)); Ar = D.forward(join(X, y))                           # D's verdicts: fakes with their asked-for labels, reals with their true ones
    ld, gr, gf = d134.d_loss(Ar, Af)                                                  # Day 134's loss and its two gradients
    g1, _ = D.backward(gr); D.forward(join(F, yf)); g2, _ = D.backward(gf)            # the real half, then the fake half (re-run to refill the caches)
    oD.step(D.net, [(a[0] + b[0], a[1] + b[1]) for a, b in zip(g1, g2)])              # one Adam step on their sum
    yf = rng.randint(0, 10, n); F = G.forward(g_input(rng.standard_normal((n, k)).astype(np.float32), yf))  # 2. the generator's turn
    lf = D.forward(join(F, yf)); lg, gl = d134.g_loss(lf)                             # the non-saturating loss
    _, dJ = D.backward(gl); gG, _ = G.backward(dJ[:, :784])                            # keep only the image's part of D's input gradient
    oG.step(G.net, gG)                                                                # only the generator moves
    return ld, lg, float(d134.d126.sigmoid(Ar).mean()), float(d134.d126.sigmoid(lf).mean())  # losses and D's average verdicts


def spread(F):                                          # NEW: average distance between pairs of images (how varied a set is)
    D2 = d119.sqdist(F, F); return float(np.sqrt(np.maximum(D2[np.triu_indices(len(F), 1)], 0)).mean())


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- how the label enters each player, and the gradient checked")
    print("=" * 70)
    y = np.array([3, 7]); Z = np.zeros((2, 64), np.float32); X = np.zeros((2, 784), np.float32)
    print(f"one-hot of [3, 7]:\n{onehot(y).astype(int)}")
    gi, di = g_input(Z, y), d_input(X, y); planes = di[:, 784:].reshape(2, 10, 28, 28)
    print(f"generator input {gi.shape}: 64 noise numbers then the 10 label numbers")
    print(f"discriminator input {di.shape} = 784 x 11: the image, then ten 28 x 28 planes; plane sums for the first image: "
          f"{planes[0].sum((1, 2)).astype(int).tolist()}")
    G, D = cgan_players(0, k=8, c=4)                                                  # small, for finite differences
    for N in (G, D):                                                                  # float64, so the check is not limited by rounding
        for L in N.layers:
            if L.params():
                L.W, L.b = L.W.astype(np.float64), L.b.astype(np.float64)
        N.net = [pr for L in N.layers for pr in L.params()]
    rng = np.random.RandomState(1); Z = rng.randn(6, 8); yy = rng.randint(0, 10, 6)
    loss = lambda: d134.g_loss(D.forward(d_input(G.forward(g_input(Z, yy)), yy)))[0]  # noise + label -> image -> image + label -> verdict -> loss
    _, gl = d134.g_loss(D.forward(d_input(G.forward(g_input(Z, yy)), yy))); _, dJ = D.backward(gl); gG, _ = G.backward(dJ[:, :784])
    worst = 0
    for li in (0, 2, 4):                                                              # the Dense and both ConvT weights
        W = G.net[li][0]
        for _ in range(6):
            idx = tuple(rng.randint(s) for s in W.shape); old = W[idx]
            W[idx] = old + 1e-5; up = loss(); W[idx] = old - 1e-5; dn = loss(); W[idx] = old
            f, g = (up - dn) / 2e-5, gG[li][0][idx]; worst = max(worst, abs(f - g) / max(1e-12, abs(f) + abs(g)))
    print(f"generator weights (18 entries, through the joined input and back off it): worst relative error {worst:.1e}")
    print(f"the label weights of the first layer (rows 8-17 of its weight matrix) get gradient too: "
          f"largest {np.abs(gG[0][0][8:]).max():.2e}")
    print("Actual output from this lesson's run.")
    return worst


def make_judge(X, y, Xt):                               # Day 134's judge (Day 116's softmax regression), with the asked-for labels
    jW, jb = d116.fit_softmax(X[:60000].astype(np.float64), y[:60000], 10, lam=1e-3, steps=300)
    P = lambda I: d116.softmax(I.astype(np.float64) @ jW + jb)

    def judge(F, asked):                                                              # does each image show the digit asked for?
        p = P(F); said = p.argmax(1); per = np.array([np.mean(said[asked == d] == d) for d in range(10)])
        near = float(np.sqrt(np.maximum(d119.sqdist(F[:500], Xt).min(1), 0)).mean())  # distance to the nearest training image
        within = np.mean([spread(F[asked == d][:100]) for d in range(10)])            # how varied each digit's images are
        return float(np.mean(said == asked)), float(per.min()), float(p.max(1).mean()), near, float(within), per, said
    return judge, P


def train_cgan(X, y, judge, epochs, seed, d_label=True, k=64, batch=128, keep=()):  # NEW: one conditional GAN, watched every epoch
    G, D = cgan_players(seed, k, d_label=d_label)
    oG, oD = d134.AdamGAN(G.net, 2e-4, 0.5), d134.AdamGAN(D.net, 2e-4, 0.5)           # Day 134's settings: lr 0.0002, b1 0.5
    rng = np.random.RandomState(seed); n = len(X)
    Zfix = np.random.RandomState(9).standard_normal((2000, k)).astype(np.float32); yfix = np.tile(np.arange(10), 200)  # 200 of each digit, interleaved
    log, snaps, t0 = [], {}, time.perf_counter()
    for ep in range(1, epochs + 1):
        o = rng.permutation(n); hs = []
        for s in range(0, n - batch + 1, batch):                                      # one pass over the training images
            b = o[s:s + batch]; hs.append(cgan_step(G, D, oG, oD, X[b], y[b], rng, k, d_label))
        F = d136.net_generate(G, g_input(Zfix, yfix)); m = np.mean(hs, 0)                  # the same 2,000 noise vectors and labels each time
        log.append((ep, m[2], m[3], judge(F, yfix), time.perf_counter() - t0))
        if ep in keep:
            snaps[ep] = F.copy()                                                      # all 2,000, for the figure and section 4
    return G, D, log, snaps


def print_log(name, log, epochs):                       # one table per run
    print(f"{name}  {'epoch':>5} {'D(real)':>8} {'D(fake)':>8} {'asked digit drawn':>18} {'worst digit':>12} {'confidence':>11} {'nearest':>8} {'variety':>8}")
    for ep, dr, df, j, _ in log:
        if ep in (1, 2, 3, 5, 10, 15, epochs):
            print(f"{'':{len(name)}s}  {ep:5d} {dr:8.3f} {df:8.3f} {j[0]:18.3f} {j[1]:12.3f} {j[2]:11.4f} {j[3]:8.2f} {j[4]:8.2f}")
    print(f"{'':{len(name)}s}  ({log[-1][-1]:.0f} s); per digit 0-9: {' '.join(f'{v:.2f}' for v in log[-1][3][5])}")


def demo_cgan(X, y, epochs=20):
    print("\n" + "=" * 70)
    print("2. A CONDITIONAL DCGAN ON MNIST -- does it draw the digit it is asked for?")
    print("=" * 70)
    Xt, yt = X[:20000], y[:20000]; judge, P = make_judge(X, y, Xt)
    real = judge(X[60000:62000], y[60000:62000])
    print(f"the judge on 2,000 real test digits: names the right digit {real[0]:.3f} (worst digit {real[1]:.3f}), confidence {real[2]:.4f},")
    print(f"nearest training image {real[3]:.2f}, variety within a digit (average distance between two images of it) {real[4]:.2f}")
    G, D = cgan_players(0)
    print(f"generator: 64 noise + 10 label numbers -> Day 136's DCGAN generator ({sum(W.size + b.size for W, b in G.net):,} numbers)")
    print(f"discriminator: 11 channels (image + ten label planes) -> Day 136's DCGAN discriminator ({sum(W.size + b.size for W, b in D.net):,} numbers)")
    print(f"20,000 training images with their labels, batches of 128, {epochs} epochs, AdamGAN (lr 0.0002, b1 0.5); fakes get random labels.")
    print("each epoch, 2,000 fixed noise vectors, 200 asked for each digit: the share the judge names as the asked-for digit,")
    print("the worst digit's share, the judge's confidence, the distance to the nearest training image, and the variety within a digit")
    runs = {}
    for seed in (0, 1):
        G, D, log, snaps = train_cgan(Xt, yt, judge, epochs, seed, keep=(1, 5, epochs))
        runs[seed] = (G, D, log, snaps); print_log(f"conditional, seed {seed}:", log, epochs)
    G, D = runs[0][:2]; Xr, yr = X[60000:62000], y[60000:62000]; wrong = (yr + 1 + np.random.RandomState(3).randint(0, 9, len(yr))) % 10
    dr = d134.d126.sigmoid(D.forward(d_input(Xr, yr))).mean(); dw = d134.d126.sigmoid(D.forward(d_input(Xr, wrong))).mean()
    print(f"seed 0's discriminator on 2,000 real test digits: D = {dr:.3f} with the true label, {dw:.3f} with a wrong one")
    said = runs[0][2][-1][3][6]; asked = np.tile(np.arange(10), 200); C = np.zeros((10, 10), int)
    for a, s in zip(asked, said):
        C[a, s] += 1
    print("seed 0, epoch 20: rows = digit asked for, columns = digit the judge sees (of 200 each)")
    print("        " + " ".join(f"{d:4d}" for d in range(10)))
    for a in range(10):
        print(f"   {a:4d} " + " ".join(f"{v:4d}" for v in C[a]))
    print("Actual output from this lesson's run.")
    return runs, real, judge, P, Xt, yt, C, (dr, dw)


def demo_g_only(Xt, yt, judge, epochs=20):
    print("\n" + "=" * 70)
    print("3. THE LABEL GIVEN TO THE GENERATOR ONLY -- the discriminator sees the image, never the label")
    print("=" * 70)
    G, D, log, snaps = train_cgan(Xt, yt, judge, epochs, 0, d_label=False, keep=(epochs,))
    print_log("generator-only label, seed 0:", log, epochs)
    W = G.net[0][0]; print(f"size of the first layer's weights: noise rows {np.abs(W[:64]).mean():.4f} on average, label rows {np.abs(W[64:]).mean():.4f}")
    print("Actual output from this lesson's run.")
    return log, snaps, G


def demo_style(runs, P):
    print("\n" + "=" * 70)
    print("4. SAME NOISE, TEN LABELS -- does the noise carry the style?")
    print("=" * 70)
    G = runs[0][0]; Z = np.random.RandomState(11).standard_normal((500, 64)).astype(np.float32)
    lab = np.tile(np.arange(10), 500); F = d136.net_generate(G, g_input(np.repeat(Z, 10, axis=0), lab))  # every noise vector as all ten digits, digits mixed in every batch
    I = F.reshape(500, 10, 28, 28).transpose(1, 0, 2, 3); P_ = P(F).argmax(1).reshape(500, 10).T  # [digit, noise vector, ...]; what the judge sees
    imgs = {d: I[d, :8].reshape(8, 784) for d in range(10)}                           # 8 noise vectors of each digit, for the figure
    ink = I.mean((2, 3))                                                              # how much ink each image uses
    r = np.arange(28)[None, None, :, None] - 13.5; c = np.arange(28)[None, None, None, :] - 13.5; m = I.sum((2, 3)) + 1e-8
    rbar, cbar = (I * r).sum((2, 3)) / m, (I * c).sum((2, 3)) / m                     # the ink's centre
    cov = (I * (r - rbar[..., None, None]) * (c - cbar[..., None, None])).sum((2, 3)) / m  # how row and column vary together, weighted by ink
    slant = cov / ((I * (r - rbar[..., None, None]) ** 2).sum((2, 3)) / m)              # slant: how far the ink's column moves per row, around its centre
    cor = lambda A: np.mean([np.corrcoef(A[a], A[b])[0, 1] for a in range(10) for b in range(a + 1, 10)])  # averaged over the 45 pairs of digits
    sh = np.random.RandomState(4).permutation(500)                                    # a fixed shuffle: pairs a digit with a DIFFERENT noise vector
    print("500 noise vectors, each drawn as all ten digits (seed 0's generator at epoch 20);")
    print("correlation, across the 500 noise vectors, of a measure on digit a with the same measure on digit b, averaged over the 45 pairs:")
    print(f"   ink (average pixel):            same noise {cor(ink):.3f};  noise shuffled for one digit {np.mean([np.corrcoef(ink[a], ink[b][sh])[0, 1] for a in range(10) for b in range(a + 1, 10)]):.3f}")
    print(f"   slant (column change per row):  same noise {cor(slant):.3f};  noise shuffled for one digit {np.mean([np.corrcoef(slant[a], slant[b][sh])[0, 1] for a in range(10) for b in range(a + 1, 10)]):.3f}")
    acc = np.mean(P_ == np.arange(10)[:, None])
    print(f"the same 500 noise vectors asked for each digit in turn: the judge names the asked-for digit {acc:.3f} of the time")
    print("Actual output from this lesson's run.")
    return imgs, cor(ink), cor(slant)


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(runs, real, gonly, imgs, C, dd, out):  # nine panels
    fig = plt.figure(figsize=(17, 12))
    ax = fig.add_subplot(2, 4, (1, 2))                                                # 8 noise vectors (rows) x 10 asked digits (columns)
    big = np.vstack([np.hstack([imgs[d][r].reshape(28, 28) for d in range(10)]) for r in range(8)])
    ax.imshow(big, cmap="gray_r", vmin=0, vmax=1); ax.axis("off")
    ax.set_title("seed 0, epoch 20: each row one noise vector, each column the digit asked for (0-9)", fontsize=10)
    ax = fig.add_subplot(2, 4, 3)                                                     # the generator-only run
    asked = np.tile(np.arange(10), 200); F = gonly[1][max(gonly[1])]; ax.imshow(np.vstack([np.hstack([F[asked == d][r].reshape(28, 28) for d in range(10)]) for r in range(4)]), cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off"); ax.set_title("label given to the generator only:\ncolumns asked for 0-9", fontsize=10)
    ax = fig.add_subplot(2, 4, 4); im = ax.imshow(C, cmap="Blues"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("digit the judge sees"); ax.set_ylabel("digit asked for"); ax.set_title("seed 0, epoch 20 (200 per row)", fontsize=10)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax = fig.add_subplot(2, 4, 5)
    for lab, log in [(f"conditional, seed {s}", runs[s][2]) for s in runs] + [("generator-only label", gonly[0])]:
        ax.plot([r[0] for r in log], [r[3][0] for r in log], marker=".", label=lab)
    ax.axhline(real[0], color="gray", lw=1, label="real test digits"); ax.axhline(0.1, color="gray", ls=":", lw=1, label="chance")
    ax.set_ylim(0, 1); ax.set_xlabel("epoch"); ax.set_ylabel("share drawn as asked"); ax.set_title("does it draw the asked-for digit?", fontsize=10); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 4, 6); w = 0.27
    for j, (lab, per) in enumerate([("conditional, seed 0", runs[0][2][-1][3][5]), ("conditional, seed 1", runs[1][2][-1][3][5]), ("real test digits", real[5])]):
        ax.bar(np.arange(10) + (j - 1) * w, per, w, label=lab)
    ax.set_xticks(range(10)); ax.set_xlabel("digit"); ax.set_ylabel("share named correctly"); ax.set_ylim(0, 1.15); ax.legend(fontsize=7, ncol=2, loc="upper center")
    ax.set_title("per digit, epoch 20", fontsize=10)
    ax = fig.add_subplot(2, 4, 7)
    for lab, log in [(f"conditional, seed {s}", runs[s][2]) for s in runs] + [("generator-only label", gonly[0])]:
        ax.plot([r[0] for r in log], [r[3][4] for r in log], marker=".", label=lab)
    ax.axhline(real[4], color="gray", lw=1, label="real test digits"); ax.set_xlabel("epoch"); ax.set_ylabel("average distance")
    ax.set_title("variety within a digit", fontsize=10); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 4, 8); ax.bar(["true label", "wrong label"], dd, color=["#4C72B0", "#C44E52"]); ax.set_ylim(0, 1)
    for i, v in enumerate(dd):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_ylabel("average D"); ax.set_title("seed 0's discriminator on real test digits", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the label goes to both players
    fig, ax = plt.subplots(figsize=(13.4, 5)); ax.set_xlim(0, 13.4); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9):                                           # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
    box(1.2, 4.0, 2.0, 0.9, "noise z\n64 numbers", "#EEEEEE"); box(1.2, 2.5, 2.0, 0.9, "label y\none-hot, 10", "#FFF2CC")
    box(4.2, 3.3, 2.4, 1.2, "generator\n(Day 136's DCGAN)\nreads 74 numbers", "#DDEEFF")
    box(7.3, 3.3, 2.0, 1.0, "image of\ndigit y", "#F8CBAD")
    box(7.3, 1.2, 2.0, 1.0, "real image\n+ its true label", "#E2F0D9")
    box(10.2, 2.3, 2.4, 1.6, "discriminator\nimage + 10 label\nplanes = 11 channels", "#E2F0D9")
    box(12.55, 2.3, 1.3, 1.1, "real\nAND\nmatching?", "#FFF2CC", 8.5)
    arrow(2.2, 4.0, 3.0, 3.5); arrow(2.2, 2.5, 3.0, 3.1); arrow(5.4, 3.3, 6.3, 3.3)
    arrow(8.3, 3.2, 9.0, 2.6); arrow(8.3, 1.3, 9.0, 2.0); arrow(11.45, 2.3, 11.85, 2.3)
    ax.annotate("", xy=(8.95, 2.25), xytext=(2.25, 2.25), arrowprops=dict(arrowstyle="->", lw=1, ls="--", color="#B8860B"))  # the label's second route
    ax.text(6.5, 2.35, "the same label y goes to the discriminator", fontsize=8.5, color="#8B6914", ha="center")
    ax.set_title("Day 137: a conditional GAN -- the label goes to both players", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "conditional_gan_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("So far each GAN was judged with a few hand-made measures: the judge's digit shares, the nearest training image, the")
    print("variety within a digit. Day 138 builds the standard scores, a Frechet distance and an Inception-style score, and")
    print("tests what each one can and cannot see.")


def main():                                             # runs every section in order
    demo_hand()                                                                       # 1. the label's way in, and the gradient
    X, y = d126m.mnist()                                                              # 70,000 images, pixels in [0, 1], float32
    runs, real, judge, P, Xt, yt, C, dd = demo_cgan(X, y)                                # 2. the conditional DCGAN
    gonly = demo_g_only(Xt, yt, judge)                                                # 3. the label for the generator only
    imgs, ci, cs = demo_style(runs, P)                          # 4. same noise, ten labels
    plot_results(runs, real, gonly, imgs, C, dd, os.path.join(HERE, "conditional_gan_results.png"))
    draw_diagram()                                                                    # the picture of the label's two routes
    print("\nsaved conditional_gan_results.png\nsaved conditional_gan_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
