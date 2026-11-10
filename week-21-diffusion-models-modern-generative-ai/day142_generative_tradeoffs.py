"""
Day 142: GANs vs VAEs vs Diffusion -- Real, Honest Quality/Speed Tradeoffs

Three families of generative model have now been built from scratch: Week
19's variational autoencoder, Week 20's GAN and Day 141's diffusion model.
Today all three are trained on the same 60,000 MNIST digits, scored by the
same scorer (Day 138's), and timed on the same machine, so their tradeoffs
are measured rather than recited.

Everything is written from scratch in NumPy, on Day 126's dense networks and
Adam, Day 129's VAE, Day 130's prior sampler, Day 134's GAN step, Day 138's
scorer and Day 141's diffusion model, all imported unchanged:
  * an averaged copy of the diffusion model's weights (an EMA)
  * DDIM (Song et al., 2021): the same trained model sampled in fewer,
    larger steps, with or without fresh noise
  * a direct measure of blur, which the scorer does not see

  1. By hand: DDIM's step, checked against Day 140's sampler.
  2. Three models, one scorer: a VAE, a GAN and a diffusion model.
  3. Raw weights against an averaged copy, while the diffusion model trains.
  4. What the scores miss: blur.
  5. Fewer steps: DDIM from 1,000 steps down to 5.
  6. The tradeoffs in one table.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier parts
import time                                            # time: how long training and sampling take

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts (Days 140-141)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day129_exercise as d129                      # Day 129: train_vae
    import day130_exercise as d130                      # Day 130: sample_prior
    import day134_exercise as d134                      # Day 134: AdamGAN, gan_step, generate
    import day139_gan_build_and_forward_diffusion as d139  # Day 139: train_classifier, Scorer (Day 138's)
    import day140_exercise as d140                      # Day 140: ABAR, reverse_sample, posterior
    import day141_exercise as d141                      # Day 141: guess, train_image_denoiser, sample_scores, to_pixels

ABAR = d140.ABAR                                        # Day 139's linear schedule: alpha-bar after each of the 1,000 steps
STEPS = 30000                                           # the diffusion model's training steps, as on Day 141
DECAY = 0.999                                           # the averaged copy keeps 99.9% of itself each step: it remembers about 1,000 steps
CHECKS = (5000, 10000, 20000, 30000)                    # diffusion steps at which both copies are scored
GAN_EPOCHS = (1, 5, 10, 20, 30)                         # GAN epochs at which it is scored
VAE_EPOCHS = 30                                         # VAE epochs, as on Day 130


# =============================================================================
# THE NEW PARTS
# =============================================================================
def ema_update(ema, net, decay=DECAY):                  # NEW: move the averaged copy a little towards the current weights, in place
    for (We, be), (W, b) in zip(ema, net):
        We *= decay; We += (1 - decay) * W                                            # W_avg = decay W_avg + (1 - decay) W
        be *= decay; be += (1 - decay) * b                                            # the same for the biases


def ddim_sample(x0_fn, n, rng, steps, eta=0.0, d=784):  # NEW: DDIM (Song et al., 2021): `steps` jumps from t = 1,000 down to 0
    ts = np.linspace(len(ABAR) - 1, 0, steps + 1).round().astype(int)                  # e.g. 1000, 950, ..., 0 for 20 steps
    x = rng.standard_normal((n, d))                                                   # pure noise, as in Day 140's sampler
    for t, s in zip(ts[:-1], ts[1:]):                                                 # one jump from t to the earlier step s
        x0 = x0_fn(x, t); e = (x - np.sqrt(ABAR[t]) * x0) / np.sqrt(1 - ABAR[t])       # the guess of x_0, and the noise it implies
        sig = eta * np.sqrt((1 - ABAR[s]) / (1 - ABAR[t]) * (1 - ABAR[t] / ABAR[s]))   # fresh noise: none for eta = 0; Day 140's beta-tilde for eta = 1, stride 1
        x = np.sqrt(ABAR[s]) * x0 + np.sqrt(np.maximum(1 - ABAR[s] - sig ** 2, 0)) * e  # put x_0 back at step s's level, with the same noise direction
        if s > 0:
            x = x + sig * rng.standard_normal(x.shape)                                # plus fresh noise, except on the very last jump
    return x


def net_x0_fn(net):                                     # NEW: a Day 141 model as a function (x_t, t) -> x_0 guess, one t for the batch
    return lambda x, t: d141.guess(net, x, np.full(len(x), t))[2]


def grey_share(F, lo=0.25, hi=0.75):                    # NEW: the share of pixels that are neither background nor ink: a measure of blur
    return float(((F > lo) & (F < hi)).mean())


def n_params(*nets):                                    # NEW: how many numbers the networks hold
    return int(sum(W.size + b.size for net in nets for W, b in net))


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- DDIM's step, checked against Day 140's sampler")
    print("=" * 70)
    print("DDIM's fresh noise at a jump from t to s: sigma^2 = eta^2 (1 - abar_s) / (1 - abar_t) (1 - abar_t / abar_s);")
    print("with eta = 1 and one step at a time (s = t - 1) it should equal Day 140's beta-tilde_t")
    print(f"{'t':>6} {'DDIM sigma^2, eta 1':>20} {'beta-tilde_t':>13}")
    for t in (2, 10, 100, 500, 1000):
        s = t - 1; sig2 = (1 - ABAR[s]) / (1 - ABAR[t]) * (1 - ABAR[t] / ABAR[s])
        print(f"{t:6d} {sig2:20.10f} {d140.posterior(t)[2]:13.10f}")
    f = lambda x, t: np.tanh(x)                                                       # any x_0 guesser at all: the identities below do not depend on it
    A = d140.reverse_sample(lambda x, t: (x - np.sqrt(ABAR[t]) * f(x, t)) / np.sqrt(1 - ABAR[t]), 100, np.random.RandomState(4), "posterior", d=784)
    B = ddim_sample(f, 100, np.random.RandomState(4), 1000, eta=1.0)
    print(f"Day 140's sampler against DDIM with eta 1 and all 1,000 steps, same guesser, same random numbers: "
          f"largest difference {np.abs(A - B).max():.2e}")
    B2 = ddim_sample(f, 100, np.random.RandomState(4), 20, eta=0.0); B3 = ddim_sample(f, 100, np.random.RandomState(9), 20, eta=0.0)
    B4 = ddim_sample(f, 100, np.random.RandomState(4), 20, eta=0.0)
    print(f"DDIM with eta 0 uses randomness only for the start: same start, largest difference {np.abs(B2 - B4).max():.2e}; "
          f"different start, largest difference {np.abs(B2 - B3).max():.2f}")
    rng = np.random.RandomState(5); x0 = rng.uniform(-1, 1, (5, 784)); e = rng.standard_normal(x0.shape)
    xt = np.sqrt(ABAR[700]) * x0 + np.sqrt(1 - ABAR[700]) * e
    jump = np.sqrt(ABAR[300]) * x0 + np.sqrt(1 - ABAR[300]) * ((xt - np.sqrt(ABAR[700]) * x0) / np.sqrt(1 - ABAR[700]))  # DDIM's eta-0 jump with the true x_0
    print(f"one eta-0 jump from t = 700 to 300 with the true x_0 lands on sqrt(abar_300) x_0 + sqrt(1 - abar_300) eps, the same eps: "
          f"largest difference {np.abs(jump - (np.sqrt(ABAR[300]) * x0 + np.sqrt(1 - ABAR[300]) * e)).max():.2e}")
    print("Actual output from this lesson's run.")


def demo_vae(X, sc):
    t0 = time.perf_counter(); enc, dec, _ = d129.train_vae(X[:60000], 16, epochs=VAE_EPOCHS); took = time.perf_counter() - t0  # Day 129's VAE, 16 codes
    t0 = time.perf_counter(); F = d130.sample_prior(dec, 2000, 16, np.random.RandomState(1)).astype(np.float32); ts = time.perf_counter() - t0
    return dict(name="VAE", F=F, train=took, sample=ts, calls=1, scores=sc(F), params=n_params(enc, dec), gen=n_params(dec))


def demo_gan(X, sc):
    G = d126.init_net([64, 256, 784], 0); D = d126.init_net([784, 256, 1], 1)          # Day 134's dense players
    oG, oD = d134.AdamGAN(G, 2e-4, 0.5), d134.AdamGAN(D, 2e-4, 0.5); rng = np.random.RandomState(0)  # lr 0.0002, b1 0.5 (Day 138's better setting)
    Z = np.random.RandomState(1).standard_normal((2000, 64)).astype(np.float32); log = []; took = 0.0
    for ep in range(1, max(GAN_EPOCHS) + 1):
        t0 = time.perf_counter(); o = rng.permutation(60000)
        for s in range(0, 60000 - 127, 128):                                           # one pass over the 60,000 training digits
            d134.gan_step(G, D, oG, oD, X[o[s:s + 128]], rng, 64)
        took += time.perf_counter() - t0                                              # training time only, not scoring
        if ep in GAN_EPOCHS:
            t0 = time.perf_counter(); F, _ = d134.generate(G, Z); ts = time.perf_counter() - t0
            F = F.astype(np.float32); log.append((ep, *sc(F)))
    return dict(name="GAN", F=F, train=took, sample=ts, calls=1, scores=sc(F), params=n_params(G, D), gen=n_params(G), log=log)


def demo_diffusion(X2, sc):
    ema, log = [None], []; t_sample = [0.0]

    def hook(s, net):                                                                 # after every training step
        if ema[0] is None:
            ema[0] = [(W.copy(), b.copy()) for W, b in net]                           # the averaged copy starts as the first weights
        else:
            ema_update(ema[0], net)
        if s in CHECKS:
            for name, n in (("raw", net), ("averaged", ema[0])):
                F, took, r = d141.sample_scores(n, sc); t_sample[0] += took; log.append((s, name, *r, grey_share(F)))
    t0 = time.perf_counter(); net, _ = d141.train_image_denoiser(X2[:60000], STEPS, hook=hook); total = time.perf_counter() - t0
    F, ts, r = d141.sample_scores(ema[0], sc)                                         # the final averaged model, sampled once more, timed alone
    return dict(name="diffusion", F=F, train=total - t_sample[0], sample=ts, calls=1000, scores=r, params=n_params(net), gen=n_params(net),
                log=log, net=net, ema=ema[0])


def demo_three(X, X2, sc):
    print("\n" + "=" * 70)
    print("2. THREE MODELS, ONE SCORER -- each trained on the 60,000 training digits, 2,000 samples each from seed 1")
    print("=" * 70)
    print(f"VAE: Day 129's, 16 codes, hidden layers of 256, {VAE_EPOCHS} epochs, samples decoded from N(0, I) codes (Day 130)")
    print(f"GAN: Day 134's dense players [64, 256, 784] and [784, 256, 1], AdamGAN lr 0.0002 b1 0.5, {max(GAN_EPOCHS)} epochs")
    print(f"diffusion: Day 141's model (output read as x_0, zero-started), {STEPS:,} steps, sampled from an averaged copy of its weights")
    print("(section 3), Day 140's 1,000-step sampler with fresh noise beta-tilde_t")
    vae = demo_vae(X, sc); gan = demo_gan(X, sc); diff = demo_diffusion(X2, sc)
    print("GAN while training:")
    print(f"{'epoch':>8} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'nearest':>8}")
    for ep, i, f, p, r, n in gan["log"]:
        print(f"{ep:8d} {i:7.3f} {f:8.3f} {p:10.3f} {r:7.3f} {n:8.2f}")
    print(f"{'model':>10} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'nearest':>8}")
    for m in (vae, gan, diff):
        print(f"{m['name']:>10}", " ".join(f"{v:{w}.{k}f}" for v, w, k in zip(m["scores"], (7, 8, 10, 7, 8), (3, 3, 3, 3, 2))))
    print("Actual output from this lesson's run.")
    return vae, gan, diff


def demo_ema(diff):
    print("\n" + "=" * 70)
    print(f"3. RAW WEIGHTS AGAINST AN AVERAGED COPY -- the diffusion model while it trains (decay {DECAY})")
    print("=" * 70)
    print("the averaged copy: after every Adam step, W_avg = 0.999 W_avg + 0.001 W, so it is an average over roughly the last 1,000 steps;")
    print("it is never trained itself. Day 141 saw the raw model's samples get worse after 10,000 steps while its loss kept falling")
    print(f"{'step':>8} {'weights':>9} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'nearest':>8} {'grey share':>11}")
    for s, name, i, f, p, r, n, g in diff["log"]:
        print(f"{s:8,d} {name:>9} {i:7.3f} {f:8.3f} {p:10.3f} {r:7.3f} {n:8.2f} {g:11.4f}")
    print("Actual output from this lesson's run.")


def demo_blur(X, vae, gan, diff):
    print("\n" + "=" * 70)
    print("4. WHAT THE SCORES MISS -- blur")
    print("=" * 70)
    T = X[60000 + np.random.RandomState(0).permutation(10000)[:2000]]                  # 2,000 real test digits, in random order
    print("grey share: the share of all pixels between 0.25 and 0.75, neither background nor ink; real digits have a little, at")
    print("the edges of strokes; a blurred digit has much more. Ink: the average pixel.")
    print(f"{'images':>18} {'grey share':>11} {'ink':>7} {'FID':>8}")
    print(f"{'real test digits':>18} {grey_share(T):11.4f} {T.mean():7.3f} {'2.996':>8}")
    for m in (vae, gan, diff):
        print(f"{m['name']:>18} {grey_share(m['F']):11.4f} {m['F'].mean():7.3f} {m['scores'][1]:8.3f}")
    print("Actual output from this lesson's run.")
    return T


def demo_ddim(diff, sc):
    print("\n" + "=" * 70)
    print("5. FEWER STEPS -- DDIM, the same trained model (the averaged copy), from 1,000 steps down to 5")
    print("=" * 70)
    print("2,000 samples each from the same starting noise (seed 1); eta 0: no fresh noise after the start; eta 1: fresh noise as in DDPM")
    print(f"{'steps':>6} {'eta':>4} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'grey share':>11} {'time':>7}")
    f = net_x0_fn(diff["ema"]); out = []; keep = {}
    for eta in (0.0, 1.0):
        for steps in (1000, 250, 100, 50, 20, 10, 5):
            t0 = time.perf_counter(); F = d141.to_pixels(ddim_sample(f, 2000, np.random.RandomState(1), steps, eta)); took = time.perf_counter() - t0
            i, fid, p, r, n = sc(F); out.append((steps, eta, fid, took)); keep[(steps, eta)] = F[:16]
            print(f"{steps:6d} {eta:4.0f} {i:7.3f} {fid:8.3f} {p:10.3f} {r:7.3f} {grey_share(F):11.4f} {took:6.1f}s")
    print("Actual output from this lesson's run.")
    return out, keep


def demo_table(vae, gan, diff, ddim):
    print("\n" + "=" * 70)
    print("6. THE TRADEOFFS IN ONE TABLE -- all on this machine, 4 threads, 2,000 samples")
    print("=" * 70)
    best20 = next(r for r in ddim if r[0] == 20 and r[1] == 0.0)                      # DDIM, 20 steps, no fresh noise
    rows = [("VAE", vae["train"], vae["sample"], 1, vae["scores"], grey_share(vae["F"]), vae["params"], vae["gen"]),
            ("GAN", gan["train"], gan["sample"], 1, gan["scores"], grey_share(gan["F"]), gan["params"], gan["gen"]),
            ("diffusion, 1,000 steps", diff["train"], diff["sample"], 1000, diff["scores"], grey_share(diff["F"]), diff["params"], diff["gen"])]
    print(f"{'model':>22} {'train':>7} {'sample 2,000':>13} {'net calls':>10} {'FID':>8} {'recall':>7} {'grey':>7} {'numbers':>9} {'sampler':>9}")
    for name, tr, ts, calls, s, g, p, gp in rows:
        print(f"{name:>22} {tr:6.0f}s {ts:12.2f}s {calls:10,d} {s[1]:8.3f} {s[3]:7.3f} {g:7.4f} {p:9,d} {gp:9,d}")
    print(f"{'diffusion, DDIM 20':>22} {diff['train']:6.0f}s {best20[3]:12.2f}s {20:10,d} {best20[2]:8.3f} {'':>7} {'':>7} {diff['params']:9,d} {diff['gen']:9,d}")
    print("'numbers': every weight trained (VAE: encoder + decoder; GAN: both players); 'sampler': the part needed to generate")
    print("Day 139's conditional DCGAN, for reference (not rerun): 6,387 s for 10 epochs, FID 20.200, recall 0.573")
    print("Actual output from this lesson's run.")


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(vae, gan, diff, ddim, keep, T, out):  # eight panels
    fig = plt.figure(figsize=(17, 12)); gs = fig.add_gridspec(3, 4)

    def grid(ax, F, r, c, title):                                                     # r x c images tiled into one picture
        ax.imshow(np.vstack([np.hstack([F[i * c + j].reshape(28, 28) for j in range(c)]) for i in range(r)]), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10); ax.axis("off")
    grid(fig.add_subplot(gs[0, 0]), T, 5, 5, "real test digits")
    for j, m in enumerate((vae, gan, diff)):
        grid(fig.add_subplot(gs[0, 1 + j]), m["F"], 5, 5, f"{m['name']}: FID {m['scores'][1]:.1f}, grey share {grey_share(m['F']):.3f}")
    ax = fig.add_subplot(gs[1, 0:2]); L = diff["log"]                                 # raw against averaged
    for name, mk in (("raw", "o"), ("averaged", "s")):
        ax.plot([r[0] for r in L if r[1] == name], [r[3] for r in L if r[1] == name], marker=mk, label=f"diffusion, {name} weights")
    ax.plot([r[0] * 468 for r in gan["log"]], [r[2] for r in gan["log"]], marker="^", ls="--", label="GAN (x axis: its own steps)")
    ax.axhline(vae["scores"][1], color="gray", ls=":", label=f"VAE after {VAE_EPOCHS} epochs")
    ax.set_xlabel("training step (batches of 128)"); ax.set_ylabel("FID (log scale)"); ax.set_yscale("log"); ax.set_xlim(0, 32000)
    ax.set_title("quality while training", fontsize=10); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 2:4])                                                  # DDIM: quality against time
    for eta, mk in ((0.0, "o"), (1.0, "s")):
        r = [x for x in ddim if x[1] == eta]
        ax.plot([x[3] for x in r], [x[2] for x in r], marker=mk, label=f"DDIM, eta {eta:.0f}")
        for x in r:
            ax.annotate(str(x[0]), (x[3], x[2]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("seconds to sample 2,000 digits (log scale)"); ax.set_ylabel("FID (log scale)")
    ax.set_title("fewer steps: faster, and worse (labels: number of steps)", fontsize=10); ax.legend(fontsize=8)
    for j, (k, title) in enumerate((((1000, 0.0), "DDIM 1,000 steps, eta 0"), ((50, 0.0), "DDIM 50 steps, eta 0"),
                                    ((10, 0.0), "DDIM 10 steps, eta 0"), ((10, 1.0), "DDIM 10 steps, eta 1"))):
        grid(fig.add_subplot(gs[2, j]), keep[k], 4, 4, title)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # what each family pays to generate one image
    fig, ax = plt.subplots(figsize=(13, 4.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, c="black"):                                             # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3, color=c))
    for y, name, mid, note, col in ((3.4, "VAE", "decoder", "1 network call; trained on reconstruction + KL", "#DDEBF7"),
                                    (2.2, "GAN", "generator", "1 network call; trained against a discriminator", "#F4CCCC"),
                                    (1.0, "diffusion", "denoiser x 1,000\n(DDIM: x 20)", "1,000 calls (or fewer with DDIM); trained by regression", "#FFF2CC")):
        ax.text(0.3, y, name, fontsize=11, weight="bold", va="center")
        box(2.9, y, 1.6, 0.75, "noise\nN(0, I)", "#EEEEEE"); box(5.5, y, 2.2, 0.75, mid, col); box(8.0, y, 1.4, 0.75, "a digit", "#E2F0D9")
        arrow(3.75, y, 4.35, y); arrow(6.65, y, 7.25, y); ax.text(8.9, y, note, fontsize=8.5, va="center")
    ax.set_title("Day 142: three ways from noise to a digit, and what each costs at generation time", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "generative_tradeoffs_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Every model so far generates digits it chooses itself. Day 143 builds the piece text-to-image models start from: a")
    print("CLIP-style pair of encoders that puts images and short captions in one shared space.")


def main():                                             # runs every section in order
    X, y = d126m.mnist(); X2 = (X * 2 - 1).astype(np.float32)                         # Day 126's 70,000 digits; [-1, 1] for the diffusion model
    demo_hand()                                                                       # 1. DDIM checked by hand
    t0 = time.perf_counter(); body, head = d139.train_classifier(X[:60000], y[:60000])  # Day 138's feature network, for every score below
    sc = d139.Scorer(body, head, X[:10000], X[10000:30000]); print(f"\n(feature network for the scores: {time.perf_counter() - t0:.0f} s)")
    vae, gan, diff = demo_three(X, X2, sc)                                            # 2. three models
    demo_ema(diff)                                                                    # 3. raw against averaged weights
    T = demo_blur(X, vae, gan, diff)                                                  # 4. blur
    ddim, keep = demo_ddim(diff, sc)                                                  # 5. DDIM
    demo_table(vae, gan, diff, ddim)                                                  # 6. the table
    plot_results(vae, gan, diff, ddim, keep, T, os.path.join(HERE, "generative_tradeoffs_results.png"))
    draw_diagram()                                                                    # the three routes
    print("\nsaved generative_tradeoffs_results.png\nsaved generative_tradeoffs_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
