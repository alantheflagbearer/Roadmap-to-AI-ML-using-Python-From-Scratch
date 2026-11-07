"""
Day 134: A GAN From Scratch -- Generator and Discriminator, a Real Minimax Training Loop

A generative adversarial network (GAN) has two networks. The GENERATOR turns
random noise into data. The DISCRIMINATOR looks at a sample and says how likely
it is to be real. They are trained against each other: the discriminator to
tell real from generated, the generator to fool it. Nobody ever writes down
p(x); the generator learns to make data only through the discriminator's
gradients.

Everything is written from scratch in NumPy, on Day 126's network parts
(init_net, forward, backward, Adam), imported unchanged:
  * both losses and their gradients, and the gradient that flows from the
    discriminator's verdict back into the generator's weights
  * one minimax step: the discriminator moves, then the generator
  * Day 126's Adam with its momentum b1 as a setting (GANs usually use 0.5)
  * checks: the losses by hand, the generator's gradient against finite
    differences, and the optimal discriminator D*(x) = p_data / (p_data + p_g)

Data: a 1-D two-peaked distribution whose density is known, and MNIST (cached
by Day 126); labels are used only to judge the generated digits.

  1. By hand: the losses when the discriminator cannot tell.
  2. Correctness: the generator's gradient, through the discriminator.
  3. The optimal discriminator, and what the game minimises.
  4. A 1-D game: watching the generator find the data.
  5. A GAN on MNIST, watched during training: Adam's b1 = 0.9 against 0.5.

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
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, sigmoid, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day119_exercise as d119                      # Day 119: sqdist
    import day116_exercise as d116                      # Day 116: softmax, fit_softmax (the judge of generated digits)


# =============================================================================
# THE TWO PLAYERS AND THEIR LOSSES -- NEW
# =============================================================================
class AdamGAN(d126.Adam):                                 # NEW: Day 126's Adam with the momentum b1 as a setting (GANs often use 0.5)
    def __init__(self, net, lr=2e-4, b1=0.5):
        super().__init__(net, lr); self.b1 = b1                                       # Day 126's constructor, plus b1

    def step(self, net, grads):                                                       # Day 126's step, with 0.9 replaced by b1
        self.t += 1; c1, c2 = 1 - self.b1 ** self.t, 1 - 0.999 ** self.t
        for (W, b), (gW, gb), m, v in zip(net, grads, self.m, self.v):
            for p, g, mm, vv in ((W, gW, m[0], v[0]), (b, gb, m[1], v[1])):
                mm *= self.b1; mm += (1 - self.b1) * g; vv *= 0.999; vv += 0.001 * g * g
                p -= self.lr * (mm / c1) / (np.sqrt(vv / c2) + 1e-8)


def softplus(Z):                                        # NEW: log(1 + e^z), in a form that cannot overflow
    return np.maximum(Z, 0) + np.log1p(np.exp(-np.abs(Z)))


def d_loss(lr, lf):                                     # NEW: the discriminator's loss on real logits lr and fake logits lf, and its gradients
    loss = softplus(-lr).mean() + softplus(lf).mean()                                  # -log D(real) - log(1 - D(fake)), D = sigmoid(logit)
    return float(loss), (d126.sigmoid(lr) - 1) / len(lr), d126.sigmoid(lf) / len(lf)    # d/dlogit of each term, averaged over the batch


def g_loss(lf, kind="non-saturating"):                  # NEW: the generator's loss on fake logits, and its gradient
    if kind == "non-saturating":                                                      # -log D(G(z)): strong gradients while D is winning
        return float(softplus(-lf).mean()), (d126.sigmoid(lf) - 1) / len(lf)
    return float(-softplus(lf).mean()), -d126.sigmoid(lf) / len(lf)                     # the minimax form, log(1 - D(G(z))), to be minimised


def generate(G, Z, out="sigmoid"):                      # NEW: noise in, data out (and the layers, for the backward pass)
    A = d126.forward(G, Z, ["relu"] * (len(G) - 1) + ["linear"])                       # Day 126's forward pass
    return (d126.sigmoid(A[-1]) if out == "sigmoid" else A[-1]), A                     # pixels need [0, 1]; 1-D or 2-D data does not


def g_grads(G, D, Z, out="sigmoid", kind="non-saturating"):  # NEW: the generator's loss and gradient, through the discriminator
    dacts = ["relu"] * (len(D) - 1) + ["linear"]; gacts = ["relu"] * (len(G) - 1) + ["linear"]
    F, AG = generate(G, Z, out); Af = d126.forward(D, F, dacts); lg, gf = g_loss(Af[-1], kind)  # noise -> fake -> verdict -> loss
    _, dF = d126.backward(D, Af, dacts, gf)                                           # backward through D to its input: d loss / d fake
    dL = dF * F * (1 - F) if out == "sigmoid" else dF                                 # back through the output sigmoid
    return lg, d126.backward(G, AG, gacts, dL)[0], Af                                 # D's own gradients are never computed here


def gan_step(G, D, oG, oD, X, rng, k, out="sigmoid", kind="non-saturating", d_steps=1):  # NEW: one round of the minimax game
    dacts = ["relu"] * (len(D) - 1) + ["linear"]; n = len(X)                          # D's activations; the batch size
    for _ in range(d_steps):                                                          # 1. the discriminator: real up, fake down
        F, _ = generate(G, rng.standard_normal((n, k)).astype(np.float32), out)       # a fresh batch of fakes
        Ar, Af = d126.forward(D, X, dacts), d126.forward(D, F, dacts)                  # its verdicts on both
        ld, gr, gf = d_loss(Ar[-1], Af[-1])
        g1, _ = d126.backward(D, Ar, dacts, gr); g2, _ = d126.backward(D, Af, dacts, gf)  # the two halves of its gradient
        oD.step(D, [(a[0] + b[0], a[1] + b[1]) for a, b in zip(g1, g2)])              # one Adam step on their sum
    lg, grads, Af = g_grads(G, D, rng.standard_normal((n, k)).astype(np.float32), out, kind)  # 2. the generator: fool the updated discriminator
    oG.step(G, grads)                                                                 # only the generator moves
    return ld, lg, float(d126.sigmoid(Ar[-1]).mean()), float(d126.sigmoid(Af[-1]).mean())  # losses, and D's average verdicts


# =============================================================================
# A 1-D DATA SET WHOSE DENSITY IS KNOWN -- NEW
# =============================================================================
def real1d(n, rng):                                     # NEW: half the samples near -2, half near +2, spread 0.5
    return (np.where(rng.uniform(size=n) < 0.5, -2.0, 2.0) + 0.5 * rng.standard_normal(n))[:, None].astype(np.float32)


def p_real1d(x):                                        # NEW: its exact density
    return 0.5 * (np.exp(-0.5 * ((x + 2) / 0.5) ** 2) + np.exp(-0.5 * ((x - 2) / 0.5) ** 2)) / (0.5 * np.sqrt(2 * np.pi))


# =============================================================================
# DEMOS
# =============================================================================
def w1(A, B):                                            # 1-D Wasserstein distance between two equal-sized samples
    return float(np.mean(np.abs(np.sort(A.ravel()) - np.sort(B.ravel()))))            # sort both, pair them up, average the gaps


def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- the losses when the discriminator cannot tell, and when it is sure")
    print("=" * 70)
    z = np.zeros((4, 1), np.float32)                                                  # logits 0: D says 0.5 to everything
    print(f"D = sigmoid(0) = {float(d126.sigmoid(z)[0, 0]):.4f} on every real and every fake sample")
    print(f"discriminator loss -log D(real) - log(1 - D(fake)) = {d_loss(z, z)[0]:.4f}; by hand 2 log 2 = {2 * np.log(2):.4f}")
    print(f"generator loss, non-saturating -log D(fake) = {g_loss(z)[0]:.4f}; by hand log 2 = {np.log(2):.4f}")
    print(f"generator loss, minimax log(1 - D(fake)) = {g_loss(z, 'minimax')[0]:.4f}; by hand -log 2 = {-np.log(2):.4f}")
    print("when the discriminator is sure a fake is fake (logit -6, D(fake) = "
          f"{float(d126.sigmoid(np.float32(-6))):.4f}), the gradient each fake sends back to the generator:")
    lf = np.full((1, 1), -6.0, np.float32)                                            # one fake, confidently rejected
    for kind in ("non-saturating", "minimax"):
        print(f"   {kind:15s} d loss / d logit = {float(g_loss(lf, kind)[1][0, 0]):+.4f}")
    print("the minimax form gives almost no gradient exactly when the generator is losing; the non-saturating form gives almost the most")
    print("Actual output from this lesson's run.")


def demo_gradcheck():
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- the generator's gradient, through the discriminator, against finite differences")
    print("=" * 70)
    rng = np.random.RandomState(0); Z = rng.standard_normal((16, 3))                   # 16 noise vectors of length 3
    for out, kind in (("linear", "non-saturating"), ("sigmoid", "non-saturating"), ("sigmoid", "minimax")):
        G = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([3, 6, 4], 1)]  # float64 for the check
        D = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([4, 5, 1], 2)]
        _, grads, _ = g_grads(G, D, Z, out, kind); worst = 0.0                        # the analytic gradient
        for li, (W, _) in enumerate(G):                                               # every weight of the generator
            for idx in np.ndindex(W.shape):
                old = W[idx]; W[idx] = old + 1e-6; up = g_grads(G, D, Z, out, kind)[0]
                W[idx] = old - 1e-6; dn = g_grads(G, D, Z, out, kind)[0]; W[idx] = old  # nudge it both ways
                num = (up - dn) / 2e-6; worst = max(worst, abs(num - grads[li][0][idx]) / max(1e-8, abs(num) + abs(grads[li][0][idx])))
        print(f"   generator output {out:7s}, {kind:14s} loss: largest relative error over {sum(W.size for W, _ in G)} weights {worst:.1e}")
    print("the discriminator's weights were held fixed: the generator learns only through the discriminator's verdict")
    print("Actual output from this lesson's run.")


def demo_optimal_d():
    print("\n" + "=" * 70)
    print("3. THE OPTIMAL DISCRIMINATOR -- D*(x) = p_data(x) / (p_data(x) + p_g(x)), and what the game minimises")
    print("=" * 70)
    p_g = lambda x: np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)                         # a fixed "generator": plain N(0, 1)
    rng = np.random.RandomState(3); D = d126.init_net([1, 32, 32, 1], 3); oD = d126.Adam(D, 1e-3)
    acts = ["relu", "relu", "linear"]
    for _ in range(4000):                                                              # train only the discriminator
        Xr, Xf = real1d(512, rng), rng.standard_normal((512, 1)).astype(np.float32)
        Ar, Af = d126.forward(D, Xr, acts), d126.forward(D, Xf, acts); _, gr, gf = d_loss(Ar[-1], Af[-1])
        g1, _ = d126.backward(D, Ar, acts, gr); g2, _ = d126.backward(D, Af, acts, gf)
        oD.step(D, [(a[0] + b[0], a[1] + b[1]) for a, b in zip(g1, g2)])
    x = np.linspace(-4, 4, 801)                                                        # a fine grid
    d_star = p_real1d(x) / (p_real1d(x) + p_g(x)); d_net = d126.sigmoid(d126.forward(D, x[:, None].astype(np.float32), acts)[-1])[:, 0]
    busy = (p_real1d(x) + p_g(x)) > 0.05                                               # where either distribution has real mass
    print(f"after 4,000 steps against N(0, 1): largest |D - D*| where either density is above 0.05: {np.abs(d_net - d_star)[busy].max():.4f}; "
          f"average {np.abs(d_net - d_star)[busy].mean():.4f}")
    print("   x:      " + "  ".join(f"{v:6.1f}" for v in (-3, -2, -1, 0, 1, 2, 3)))
    pick = [np.argmin(np.abs(x - v)) for v in (-3, -2, -1, 0, 1, 2, 3)]
    print("   D*(x):  " + "  ".join(f"{d_star[i]:6.3f}" for i in pick))
    print("   D(x):   " + "  ".join(f"{d_net[i]:6.3f}" for i in pick))
    xs = np.linspace(-12, 12, 200001); dx = xs[1] - xs[0]; pd, pg = p_real1d(xs), p_g(xs); m = 0.5 * (pd + pg)  # numerical integration
    kl = lambda p, q: float(np.sum(p * np.log((p + 1e-300) / (q + 1e-300))) * dx)
    jsd = 0.5 * kl(pd, m) + 0.5 * kl(pg, m)
    v_star = float(np.sum(pd * np.log(pd / (pd + pg) + 1e-300) + pg * np.log(pg / (pd + pg) + 1e-300)) * dx)
    Xr, Xf = real1d(200000, np.random.RandomState(4)), np.random.RandomState(5).standard_normal((200000, 1)).astype(np.float32)
    v_net = -d_loss(d126.forward(D, Xr, acts)[-1], d126.forward(D, Xf, acts)[-1])[0]  # V(D) = E log D(real) + E log(1 - D(fake))
    print(f"value of the game at D*: V = {v_star:.4f}; -log 4 + 2 JSD(p_data, p_g) = {-np.log(4):.4f} + 2 x {jsd:.4f} = {-np.log(4) + 2 * jsd:.4f}")
    print(f"the trained discriminator reaches V = {v_net:.4f} on 200,000 fresh samples of each (at most V(D*) = {v_star:.4f})")
    print(f"if the generator matched the data exactly, D* = 0.5 everywhere and V = -log 4 = {-np.log(4):.4f}: the generator's best possible score")
    print("Actual output from this lesson's run.")
    return x, d_star, d_net


def demo_1d(steps=5000):
    print("\n" + "=" * 70)
    print("4. A 1-D GAME -- watching the generator find the data (two peaks, at -2 and +2)")
    print("=" * 70)
    ref = real1d(5000, np.random.RandomState(99)); Zfix = np.random.RandomState(7).standard_normal((5000, 4)).astype(np.float32)
    floor = w1(ref, real1d(5000, np.random.RandomState(98)))                          # two real samples: the distance from noise alone
    print(f"generator [4, 32, 32, 1], discriminator [1, 32, 32, 1], Adam 0.001 for both, batches of 256, {steps:,} steps, 5 seeds")
    print(f"distance to the data: 1-D Wasserstein distance between 5,000 generated and 5,000 real samples; two real samples give {floor:.3f}")
    marks = (250, 500, 1000, 2000, 3000, 4000, 5000); hist, snaps = {}, {}
    print("   seed  " + "".join(f"{m:>8,}" for m in marks) + "   share of fakes above 0 at each mark")
    for seed in range(5):
        G = d126.init_net([4, 32, 32, 1], seed); D = d126.init_net([1, 32, 32, 1], seed + 100)
        oG, oD = d126.Adam(G, 1e-3), d126.Adam(D, 1e-3); rng = np.random.RandomState(seed); hist[seed] = []; pos = []
        for it in range(1, steps + 1):
            gan_step(G, D, oG, oD, real1d(256, rng), rng, 4, out="linear")
            if it % 250 == 0:                                                         # watch it during training, not only at the end
                F, _ = generate(G, Zfix, "linear"); hist[seed].append((it, w1(F, ref), float(np.mean(F > 0))))
                if seed == 1 and it in (500, 3000, 4000, 5000):                       # keep a few pictures of one run
                    g = np.linspace(-5, 5, 201)[:, None].astype(np.float32)
                    snaps[it] = (F[:, 0].copy(), d126.sigmoid(d126.forward(D, g, ["relu", "relu", "linear"])[-1])[:, 0])
        at = {i: (d, s) for i, d, s in hist[seed]}
        print(f"   {seed:4d}  " + "".join(f"{at[m][0]:8.3f}" for m in marks) + "   " + " ".join(f"{at[m][1]:.2f}" for m in marks))
    last = np.array([[d for _, d, _ in hist[s][-8:]] for s in range(5)])               # the last 2,000 steps, every 250
    print(f"over the last 2,000 steps the distance moves between {last.min():.3f} and {last.max():.3f}: "
          f"it keeps changing instead of settling (the data's own floor is {floor:.3f})")
    print("Actual output from this lesson's run.")
    return hist, snaps, floor


def train_mnist_gan(X, judge, b1, seed, epochs, n=20000, k=64, keep=()):  # one GAN on MNIST, watched every epoch
    G = d126.init_net([k, 256, 784], seed); D = d126.init_net([784, 256, 1], seed + 1)  # one hidden layer of 256 in each player
    oG, oD = AdamGAN(G, 2e-4, b1), AdamGAN(D, 2e-4, b1)                               # the only difference between the runs: b1
    rng = np.random.RandomState(seed); Xt = X[:n]; Zfix = np.random.RandomState(9).standard_normal((2000, k)).astype(np.float32)
    log, snaps = [], {}
    for ep in range(1, epochs + 1):
        o = rng.permutation(n); hs = []
        for s in range(0, n - 127, 128):                                              # one pass over the training images
            hs.append(gan_step(G, D, oG, oD, Xt[o[s:s + 128]], rng, k))
        F, _ = generate(G, Zfix); P = judge(F); h = np.bincount(P.argmax(1), minlength=10) / len(F)  # which digits it is making
        near = float(np.sqrt(d119.sqdist(F[:500], Xt).min(1)).mean()); m = np.mean(hs, 0)  # distance to the nearest training image
        log.append((ep, m[2], m[3], h.min(), h.max(), float(P.max(1).mean()), near, h, float(F.mean())))
        if ep in keep:
            snaps[ep] = F[:16].copy()                                                 # the same 16 noise vectors each time
    return log, snaps


def demo_mnist(X, y, epochs=20):
    print("\n" + "=" * 70)
    print("5. A GAN ON MNIST -- watched during training")
    print("=" * 70)
    judge_W, judge_b = d116.fit_softmax(X[:60000].astype(np.float64), y[:60000], 10, lam=1e-3, steps=300)  # Day 116's softmax regression
    judge = lambda I: d116.softmax(I.astype(np.float64) @ judge_W + judge_b)          # names the digit in an image
    Pt = judge(X[60000:]); near_real = float(np.sqrt(d119.sqdist(X[60000:60500], X[:20000]).min(1)).mean())
    print(f"the judge (softmax regression on pixels, never shown a generated image): test accuracy {np.mean(Pt.argmax(1) == y[60000:]):.4f}, "
          f"average top probability on real digits {Pt.max(1).mean():.4f}")
    print(f"real test digits: ink (average pixel) {X[60000:].mean():.3f}; distance to the nearest of the 20,000 training images {near_real:.2f}")
    print("generator [64, 256, 784] with sigmoid pixels, discriminator [784, 256, 1]; Adam, learning rate 0.0002 for both;")
    print(f"20,000 training images, batches of 128, {epochs} epochs. Each epoch, on 2,000 fixed noise vectors: the judge's share of the rarest")
    print("and commonest digit (an even spread is 0.100 each), its confidence, ink, and the distance to the nearest training image")
    runs, t0 = {}, time.perf_counter()
    for b1 in (0.9, 0.5):                                                             # Day 126's momentum, then the usual GAN setting
        for seed in (0, 1):
            log, snaps = train_mnist_gan(X, judge, b1, seed, epochs, keep=(1, 5, 10, epochs) if seed == 0 else (epochs,))
            runs[(b1, seed)] = (log, snaps)
            print(f"Adam b1 = {b1}, seed {seed}:   {'epoch':>5} {'D(real)':>8} {'D(fake)':>8} {'rarest':>7} {'commonest':>10} {'confidence':>11} {'ink':>6} {'nearest':>8}")
            for r in log:
                if r[0] in (1, 3, 5, 10, 15, epochs):
                    print(f"{'':27s}{r[0]:5d} {r[1]:8.3f} {r[2]:8.3f} {r[3]:7.3f} {r[4]:10.3f} {r[5]:11.4f} {r[8]:6.3f} {r[6]:8.2f}")
    print(f"({time.perf_counter() - t0:.0f} s for the four runs)")
    for (b1, seed), (log, _) in runs.items():
        print(f"b1 = {b1}, seed {seed}, final spread of digits: {' '.join(f'{v:.2f}' for v in log[-1][7])}")
    print("Actual output from this lesson's run.")
    return runs, near_real


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(opt, hist, snaps, floor, runs, out):  # eight panels
    fig = plt.figure(figsize=(16, 10)); g = np.linspace(-5, 5, 201); xs = np.linspace(-5, 5, 401)
    for j, it in enumerate(sorted(snaps)):                                             # one run of the 1-D game, four moments
        ax = fig.add_subplot(3, 4, j + 1); F, Dg = snaps[it]
        ax.hist(np.clip(F, -5, 5), bins=60, range=(-5, 5), density=True, alpha=0.6, label="generated")
        ax.plot(xs, p_real1d(xs), "k", lw=1.2, label="real density"); ax.plot(g, Dg, "r--", lw=1.2, label="D(x)")
        ax.set_ylim(0, 1.05); ax.set_title(f"seed 1, step {it:,}", fontsize=10)
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")
    ax = fig.add_subplot(3, 4, 5)                                                     # distance to the data, every seed
    for s, h in hist.items():
        ax.plot([i for i, _, _ in h], [d for _, d, _ in h], label=f"seed {s}", lw=1)
    ax.axhline(floor, color="gray", ls=":", label="two real samples"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("Wasserstein distance"); ax.set_title("1-D: distance to the data", fontsize=10); ax.legend(fontsize=7)
    ax = fig.add_subplot(3, 4, 6); x, d_star, d_net = opt                              # the optimal discriminator
    ax.plot(x, d_star, "k", label="D* = p_data / (p_data + p_g)"); ax.plot(x, d_net, "r--", label="trained D")
    ax.plot(x, p_real1d(x), color="#4C72B0", alpha=0.5, label="p_data"); ax.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), color="#DD8452", alpha=0.5, label="p_g = N(0, 1)")
    ax.set_ylim(0, 1.25); ax.set_title("the optimal discriminator", fontsize=10); ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax = fig.add_subplot(3, 4, 7)                                                     # the MNIST game, epoch by epoch
    for b1, st in ((0.9, "--"), (0.5, "-")):
        log = runs[(b1, 0)][0]; ep = [r[0] for r in log]
        ax.plot(ep, [r[1] for r in log], "C0" + st, label=f"D(real), b1 {b1}"); ax.plot(ep, [r[2] for r in log], "C3" + st, label=f"D(fake), b1 {b1}")
    ax.set_ylim(0, 1.3); ax.set_xlabel("epoch"); ax.set_title("MNIST: the discriminator's verdicts (seed 0)", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper center")
    ax = fig.add_subplot(3, 4, 8)                                                     # how unevenly it spreads over the digits
    for (b1, seed), (log, _) in runs.items():
        ax.plot([r[0] for r in log], [r[4] for r in log], "--" if b1 == 0.9 else "-", label=f"b1 {b1}, seed {seed}")
    ax.axhline(0.1, color="gray", ls=":"); ax.set_ylim(0, 1.05); ax.set_xlabel("epoch"); ax.set_ylabel("share of the commonest digit")
    ax.set_title("MNIST: one digit or all ten?", fontsize=10); ax.legend(fontsize=7)
    last = max(runs[(0.5, 0)][1])                                                     # four sample grids: b1 0.9 at the end, b1 0.5 over time
    for j, (b1, e) in enumerate(((0.9, last), (0.5, 1), (0.5, 5), (0.5, last))):
        ax = fig.add_subplot(3, 4, 9 + j); grid = np.ones((4 * 29, 4 * 29)); Fs = runs[(b1, 0)][1][e]
        for i in range(16):
            grid[(i // 4) * 29:(i // 4) * 29 + 28, (i % 4) * 29:(i % 4) * 29 + 28] = 1 - Fs[i].reshape(28, 28)
        ax.imshow(grid, cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(f"MNIST, b1 {b1}, epoch {e}", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the two players and where the gradients go
    fig, ax = plt.subplots(figsize=(12, 5)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9):                                           # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, color="black", ls="-"):                                 # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2, color=color, ls=ls))
    box(1.1, 1.3, 1.6, 0.9, "noise z\nN(0, I)", "#DDEEFF")
    box(3.6, 1.3, 2.2, 0.9, "generator G\nz -> fake x", "#FFF2CC")
    box(3.6, 3.7, 2.2, 0.9, "real data x", "#DDEEFF")
    box(7.0, 2.5, 2.4, 1.1, "discriminator D\nx -> P(real)", "#E2F0D9")
    box(10.4, 3.5, 2.6, 1.0, "D's loss:\n-log D(real) - log(1 - D(fake))", "#E2F0D9", 8.5)
    box(10.4, 1.4, 2.6, 1.0, "G's loss:\n-log D(fake)", "#FFF2CC", 8.5)
    arrow(1.9, 1.3, 2.5, 1.3); arrow(4.7, 1.45, 5.8, 2.25); arrow(4.7, 3.55, 5.8, 2.75)
    arrow(8.2, 2.75, 9.1, 3.35); arrow(8.2, 2.25, 9.1, 1.55)
    arrow(9.1, 1.2, 4.7, 1.0, "#C00000", "--")                                        # G's gradient, back through D
    ax.text(6.9, 0.55, "G's gradient flows back through D (D's weights held still)", ha="center", fontsize=8.5, color="#C00000")
    ax.text(6.3, 4.55, "step 1: D learns to tell real from fake;  step 2: G learns to fool the new D", ha="center", fontsize=9, style="italic")
    ax.set_title("Day 134: a GAN, two networks in one minimax game", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "gan_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Some 1-D runs spent thousands of steps on one peak, none of them settled, and on MNIST one setting of Adam made a")
    print("single image for every input. Day 135 studies that instability on eight peaks at once: mode collapse, the minimax")
    print("loss, and the balance between the two players.")


def main():                                             # runs every section in order
    demo_hand()                                                                       # 1. the losses, by hand
    demo_gradcheck()                                                                  # 2. the generator's gradient, checked
    opt = demo_optimal_d()                                                            # 3. D* and the Jensen-Shannon divergence
    hist, snaps, floor = demo_1d()                                                    # 4. the 1-D game
    X, y = d126m.mnist()                                                              # 70,000 images, pixels in [0, 1], float32
    runs, near_real = demo_mnist(X, y)                                                # 5. a GAN on MNIST, two settings of Adam
    plot_results(opt, hist, snaps, floor, runs, os.path.join(HERE, "gan_results.png"))
    draw_diagram()                                                                    # the picture of the game
    print("\nsaved gan_results.png\nsaved gan_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
