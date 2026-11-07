"""
Day 129: Variational Autoencoders -- the Reparameterization Trick From Scratch

Day 126's autoencoder maps each image to ONE code, and codes nobody used decode
to blurs. A variational autoencoder (VAE) maps each image to a small Gaussian
cloud of codes, mean mu and spread sigma, and is pushed to keep those clouds
close to a standard normal. Then any code drawn from N(0, 1) decodes to
something sensible (Day 130 shows this).

But a random draw blocks the gradient: how do you differentiate through
'pick a random z'? The reparameterization trick writes the draw as
z = mu + sigma * eps, with eps ~ N(0, 1) drawn independently. The randomness
moves into eps, and z becomes an ordinary function of mu and sigma.

Everything is written from scratch in NumPy, on Day 126's network parts:
  * the trick, the Gaussian KL divergence, and their gradients
  * the score-function (REINFORCE) estimator, the alternative without the trick
  * the VAE loss (the negative ELBO), with backpropagation through the draw

  1. By hand: one draw and one KL divergence.
  2. The KL formula against a Monte Carlo estimate.
  3. Two gradient estimators: both right on average, one far noisier.
  4. Learning a Gaussian with each estimator.
  5. Correctness: a gradient check through a whole VAE.
  6. The trick in a real network: a few epochs on MNIST.

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
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, recon_loss, sigmoid, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)


# =============================================================================
# THE TRICK, THE KL TERM AND THE VAE LOSS -- NEW
# =============================================================================
def reparam(mu, logvar, rng):                           # NEW: z = mu + sigma * eps; eps carries all the randomness
    eps = rng.standard_normal(mu.shape).astype(mu.dtype)
    return mu + np.exp(0.5 * logvar) * eps, eps


def kl_normal(mu, logvar):                              # NEW: KL( N(mu, sigma^2) || N(0, 1) ), summed over code dimensions, per example
    return 0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar).sum(1)


def vae_loss(enc, dec, X, rng, beta=1.0, acts_e=("relu", "linear"), acts_d=("relu", "linear"), eps=None):  # NEW: loss and every gradient
    n = len(X); H = d126.forward(enc, X, list(acts_e)); out = H[-1]; k = out.shape[1] // 2
    mu, logvar = out[:, :k], out[:, k:]                    # the encoder's last layer gives both halves
    if eps is None:
        z, eps = reparam(mu, logvar, rng)
    else:
        z = mu + np.exp(0.5 * logvar) * eps                 # a fixed eps, for the gradient check
    A = d126.forward(dec, z, list(acts_d)); rec, dL = d126.recon_loss(A[-1], X, "bce")  # reconstruction: cross-entropy per image
    kl = float(kl_normal(mu, logvar).mean())
    gdec, dz = d126.backward(dec, A, list(acts_d), dL)    # back through the decoder, down to z
    dmu = dz + beta * mu / n                               # z = mu + sigma eps, so dz/dmu = 1; plus the KL term's own gradient
    dlogvar = dz * eps * 0.5 * np.exp(0.5 * logvar) + beta * 0.5 * (np.exp(logvar) - 1) / n  # dz/dlogvar = eps sigma / 2
    genc, _ = d126.backward(enc, H, list(acts_e), np.hstack([dmu, dlogvar]))
    return rec + beta * kl, rec, kl, genc, gdec


def train_vae(X, k, hidden=256, epochs=10, batch=128, lr=1e-3, beta=1.0, seed=0, Xval=None):  # NEW: minibatch training of encoder and decoder
    enc = d126.init_net([X.shape[1], hidden, 2 * k], seed); dec = d126.init_net([k, hidden, X.shape[1]], seed + 1)
    oe, od = d126.Adam(enc, lr), d126.Adam(dec, lr); rng = np.random.RandomState(seed + 2); hist = []
    for _ in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            _, _, _, ge, gd = vae_loss(enc, dec, X[o[s:s + batch]], rng, beta)
            oe.step(enc, ge); od.step(dec, gd)
        ev = X[:2000] if Xval is None else Xval
        hist.append(vae_loss(enc, dec, ev, np.random.RandomState(99), beta)[:3])  # (loss, reconstruction, KL) on held-out images
    return enc, dec, hist


def score_grad(f, mu, sigma, rng, n):                   # NEW: n single-draw score-function estimates of d E[f(z)] / d(mu, sigma)
    z = mu + sigma * rng.standard_normal(n)
    return f(z) * (z - mu) / sigma ** 2, f(z) * ((z - mu) ** 2 / sigma ** 3 - 1 / sigma)  # f(z) times d log q(z) / d parameter


def reparam_grad(df, mu, sigma, rng, n):                # NEW: n single-draw reparameterization estimates of the same gradient
    eps = rng.standard_normal(n); z = mu + sigma * eps
    return df(z), df(z) * eps                              # chain rule through z = mu + sigma eps


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- one draw, and one KL divergence")
    print("=" * 70)
    mu, logvar = np.array([[1.0]]), np.array([[np.log(4.0)]]); z = mu + np.exp(0.5 * logvar) * 0.5
    print(f"mu 1, sigma 2 (logvar = log 4), eps 0.5: z = 1 + 2 x 0.5 = {z[0, 0]:.4f}")
    print(f"KL(N(1, 4) || N(0, 1)) by hand: 0.5 (sigma^2 + mu^2 - 1 - log sigma^2) = 0.5 (4 + 1 - 1 - {np.log(4):.4f}) = {0.5 * (4 + 1 - 1 - np.log(4)):.4f}; "
          f"kl_normal: {kl_normal(mu, logvar)[0]:.4f}")
    print(f"KL of N(0, 1) from itself: {kl_normal(np.zeros((1, 1)), np.zeros((1, 1)))[0]:.4f}")
    print("Actual output from this lesson's run.")


def demo_kl_mc():
    print("\n" + "=" * 70)
    print("2. THE KL FORMULA AGAINST A MONTE CARLO ESTIMATE -- 1,000,000 draws each")
    print("=" * 70)
    rng = np.random.RandomState(0)
    print(f"  {'mu':>6}{'sigma':>7}{'formula':>10}{'Monte Carlo':>13}{'standard error':>16}")
    for mu, s in ((0.0, 1.0), (1.0, 2.0), (-2.0, 0.5), (0.5, 0.1)):
        z = mu + s * rng.standard_normal(1_000_000)
        r = (-0.5 * ((z - mu) / s) ** 2 - np.log(s)) - (-0.5 * z ** 2)  # log q(z) - log p(z); the 2 pi terms cancel
        print(f"  {mu:>6}{s:>7}{kl_normal(np.array([[mu]]), np.array([[2 * np.log(s)]]))[0]:>10.4f}{r.mean():>13.4f}{r.std() / 1000:>16.4f}")
    print("Actual output from this lesson's run.")


def demo_estimators():
    print("\n" + "=" * 70)
    print("3. TWO GRADIENT ESTIMATORS -- d E[f(z)] / d mu and d sigma, z ~ N(mu, sigma^2), 100,000 single-draw estimates each")
    print("=" * 70)
    rng = np.random.RandomState(1); n = 100_000; out = {}
    cases = (("(z - 3)^2 at mu 0, sigma 1", lambda z: (z - 3) ** 2, lambda z: 2 * (z - 3), 0.0, 1.0, (2 * (0 - 3), 2 * 1.0)),
             ("(z - 3)^2 at mu 2.9, sigma 0.1", lambda z: (z - 3) ** 2, lambda z: 2 * (z - 3), 2.9, 0.1, (2 * (2.9 - 3), 2 * 0.1)))
    print(f"  {'f and where':<32}{'exact':>16}{'estimator':>16}{'mean':>18}{'spread (std)':>18}")
    for name, f, df, mu, s, exact in cases:
        for est, (gm, gs) in (("score function", score_grad(f, mu, s, rng, n)), ("reparameterized", reparam_grad(df, mu, s, rng, n))):
            out[(name, est)] = (gm.std(), gs.std())
            print(f"  {name:<32}{f'({exact[0]:.2f}, {exact[1]:.2f})':>16}{est:>16}{f'({gm.mean():.2f}, {gs.mean():.2f})':>18}{f'({gm.std():.2f}, {gs.std():.2f})':>18}")
    r = [out[(n, "score function")][0] / out[(n, "reparameterized")][0] for n, *_ in cases]
    print(f"both are right on average; the score function's d/d mu estimates spread {r[0]:.1f} and {r[1]:.1f} times wider than the trick's in the two cases")
    print("Actual output from this lesson's run.")
    return out


def demo_fit():
    print("\n" + "=" * 70)
    print("4. LEARNING A GAUSSIAN WITH EACH ESTIMATOR -- minimise E[(z - 3)^2 / (2 x 0.25)] + KL(q || N(0, 1)), one draw per step")
    print("=" * 70)
    print("by hand, the best q is the Bayesian posterior: precision 1 + 1/0.25 = 5, so sigma = sqrt(0.2), mu = 3 x 4/5 = 2.4")
    target = np.array([2.4, np.sqrt(0.2)]); f = lambda z: (z - 3) ** 2 / 0.5; df = lambda z: 2 * (z - 3) / 0.5
    paths = {}
    print(f"  {'estimator':<16}{'step size':>10}{'blew up':>10}{'mu (mean)':>11}{'sigma (mean)':>14}{'distance to the best':>24}")
    for est, lr in (("score function", 0.01), ("score function", 0.001), ("reparameterized", 0.01), ("reparameterized", 0.001)):
        final = []
        for seed in range(20):
            rng = np.random.RandomState(seed); mu, ls = 0.0, 0.0; path = []   # ls = log sigma, so sigma stays positive
            with np.errstate(all="ignore"):                      # a run that blows up is counted below, not warned about
                for t in range(2000):
                    s = np.exp(ls)
                    gm, gs = (score_grad(f, mu, s, rng, 1) if est == "score function" else reparam_grad(df, mu, s, rng, 1))
                    gm, gs = gm[0] + mu, gs[0] * s + (s * s - 1)          # add the KL term's gradients; gs is now with respect to log sigma
                    mu -= lr * gm; ls -= lr * gs; path.append((mu, np.exp(ls)))
            final.append(path[-1])
            if seed == 0 and lr == 0.001:
                paths[est] = path                           # keep one path per estimator for the figure
        F = np.array(final); ok = np.all(np.isfinite(F), 1); err = np.abs(F[ok] - target)
        print(f"  {est:<16}{lr:>10}{f'{int((~ok).sum())} of 20':>10}" +
              (f"{F[ok, 0].mean():>11.3f}{F[ok, 1].mean():>14.3f}{f'({err[:, 0].mean():.3f}, {err[:, 1].mean():.3f})':>24}" if ok.any() else f"{'-':>11}{'-':>14}{'-':>24}"))
    print("(the means and distances are over the runs that did not blow up)")
    print("Actual output from this lesson's run.")
    return paths, target


def demo_gradcheck():
    print("\n" + "=" * 70)
    print("5. CORRECTNESS -- a gradient check through a whole VAE (encoder 6-5-4, two codes, decoder 2-5-6), eps held fixed")
    print("=" * 70)
    rng = np.random.RandomState(0); X = rng.rand(4, 6); eps = rng.standard_normal((4, 2))
    enc = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([6, 5, 4], 0)]
    dec = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([2, 5, 6], 1)]
    _, _, _, ge, gd = vae_loss(enc, dec, X, None, eps=eps); worst = 0.0
    for net, g in ((enc, ge), (dec, gd)):
        for li, (W, b) in enumerate(net):
            for P, G in ((W, g[li][0]), (b, g[li][1])):
                for idx in np.ndindex(P.shape):
                    o = P[idx]; P[idx] = o + 1e-6; lp = vae_loss(enc, dec, X, None, eps=eps)[0]
                    P[idx] = o - 1e-6; lm = vae_loss(enc, dec, X, None, eps=eps)[0]; P[idx] = o
                    worst = max(worst, abs((lp - lm) / 2e-6 - G[idx]))
    print(f"all {sum(W.size + b.size for net in (enc, dec) for W, b in net)} parameters: largest gap between backpropagation and finite differences {worst:.1e}")
    print("with eps fixed, z is an ordinary function of the encoder's outputs, so ordinary backpropagation applies: that is the trick")
    print("Actual output from this lesson's run.")


def demo_mnist(X):
    print("\n" + "=" * 70)
    print("6. THE TRICK IN A REAL NETWORK -- a VAE on 20,000 MNIST images: 784 -> 256 -> (16 means, 16 log-variances), 16 -> 256 -> 784")
    print("=" * 70)
    t0 = time.perf_counter(); enc, dec, hist = train_vae(X[:20000], 16, epochs=10, Xval=X[60000:62000]); t = time.perf_counter() - t0
    print(f"  {'epoch':>6}{'loss (negative ELBO)':>22}{'reconstruction':>16}{'KL':>8}   per test image, in nats")
    for e in (0, 1, 2, 4, 9):
        print(f"  {e + 1:>6}{hist[e][0]:>22.2f}{hist[e][1]:>16.2f}{hist[e][2]:>8.2f}")
    H = d126.forward(enc, X[60000:62000], ["relu", "linear"])[-1]; mu, logvar = H[:, :16], H[:, 16:]
    klj = (0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar)).mean(0)  # KL per code dimension
    print(f"trained in {t:.0f} s; KL per code dimension, largest first: {[round(float(v), 2) for v in np.sort(klj)[::-1]]}")
    print(f"dimensions carrying more than 0.1 nats: {int((klj > 0.1).sum())} of 16 (a dimension below that stays at the prior and carries nothing)")
    print("Actual output from this lesson's run.")
    return enc, dec, hist, klj


def plot_results(est, paths, target, hist, klj, dec):   # five panels
    fig = plt.figure(figsize=(16, 9.6)); G = fig.add_gridspec(2, 3)
    a = fig.add_subplot(G[0, 0]); names = list(dict.fromkeys(k[0] for k in est)); x = np.arange(len(names)); w = 0.38
    for i, e in enumerate(("score function", "reparameterized")):
        a.bar(x + (i - 0.5) * w, [est[(n, e)][0] for n in names], w, label=e)
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(["mu 0, sigma 1", "mu 2.9, sigma 0.1"]); a.set_ylabel("spread of one-draw estimates (std, log scale)")
    a.set_title("Gradient noise, d/d mu", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    a = fig.add_subplot(G[0, 1])
    for e, p in paths.items():
        p = np.array(p); a.plot(p[:, 0], p[:, 1], linewidth=0.8, label=e)
    a.plot(*target, "k*", markersize=12, label="the best q"); a.set_xlabel("mu"); a.set_ylabel("sigma"); a.set_title("Learning a Gaussian, seed 0, step 0.001", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = fig.add_subplot(G[0, 2]); h = np.array(hist); e = np.arange(1, len(h) + 1)
    a.plot(e, h[:, 1], marker="o", label="reconstruction"); a.plot(e, h[:, 2], marker="s", label="KL")
    a.set_xlabel("epoch"); a.set_ylabel("nats per test image"); a.set_title("MNIST VAE: the two parts of the loss", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = fig.add_subplot(G[1, 0]); a.bar(range(16), np.sort(klj)[::-1]); a.set_xlabel("code dimension (sorted)"); a.set_ylabel("KL (nats)")
    a.set_title("Which code dimensions are used", fontsize=10); a.grid(alpha=0.3, axis="y")
    sub = G[1, 1:].subgridspec(3, 12); rng = np.random.RandomState(5)
    S = d126.sigmoid(d126.forward(dec, rng.standard_normal((36, 16)).astype(np.float32), ["relu", "linear"])[-1])  # codes drawn from N(0, 1)
    for j in range(36):
        b = fig.add_subplot(sub[j // 12, j % 12]); b.imshow(S[j].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); b.set_xticks([]); b.set_yticks([])
        if j == 5:
            b.set_title("Decoded codes drawn from N(0, 1), after 10 epochs (Day 130 trains properly)", fontsize=10)
    plt.tight_layout(); plt.savefig("vae_trick_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved vae_trick_results.png")


def draw_diagram():                                     # the trick
    fig, ax = plt.subplots(figsize=(13, 4.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.35, "Day 129: the reparameterization trick", ha="center", fontsize=12, fontweight="bold")
    box(1.2, 2.5, 1.6, 1.4, "image x", "#FAEEDA")
    box(3.4, 2.5, 1.8, 1.4, "encoder", "#DCEBF7")
    box(5.6, 3.3, 1.6, 0.8, "mu", "#E5F2E1"); box(5.6, 1.7, 1.6, 0.8, "sigma", "#E5F2E1")
    box(7.9, 1.05, 2.0, 0.7, "eps ~ N(0, 1)\n(no parameters)", "#F7DCDC", fs=8)
    box(7.9, 2.5, 2.0, 1.1, "z = mu + sigma x eps", "#EEEDFE")
    box(10.3, 2.5, 1.6, 1.4, "decoder", "#DCEBF7"); box(12.2, 2.5, 1.3, 1.4, "rebuilt x", "#FAEEDA")
    for (x0, y0), (x1, y1) in (((2.04, 2.5), (2.46, 2.5)), ((4.34, 2.8), (4.76, 3.2)), ((4.34, 2.2), (4.76, 1.8)), ((6.44, 3.2), (6.86, 2.8)),
                               ((6.44, 1.8), (6.86, 2.2)), ((7.9, 1.42), (7.9, 1.91)), ((8.94, 2.5), (9.46, 2.5)), ((11.14, 2.5), (11.51, 2.5))):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 0.2, "loss = reconstruction + KL(N(mu, sigma^2) || N(0, 1)); gradients flow through mu and sigma, not through eps", ha="center", fontsize=8.5)
    plt.savefig("vae_trick_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved vae_trick_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 130 trains VAEs properly, maps their latent space in two dimensions, and generates new digits from it.")


def main():                                             # runs every section in order
    X, y = d126m.mnist()
    demo_hand()                                            # 1. by hand
    demo_kl_mc()                                           # 2. KL formula
    est = demo_estimators()                                # 3. two estimators
    paths, target = demo_fit()                             # 4. learning a Gaussian
    demo_gradcheck()                                       # 5. gradient check
    enc, dec, hist, klj = demo_mnist(X)                    # 6. MNIST
    plot_results(est, paths, target, hist, klj, dec)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
