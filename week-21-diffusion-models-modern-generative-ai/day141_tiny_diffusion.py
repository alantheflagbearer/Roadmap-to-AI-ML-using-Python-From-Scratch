"""
Day 141: Training a Tiny Diffusion Model -- Real Toy Image Generation

Day 140 ran the reverse process on Day 135's ring, where the best possible
denoiser could be written down. On images it cannot: nobody knows E[x_0 | x_t]
for handwritten digits. So today a network learns it, from the 60,000 MNIST
training digits, and then generates new digits from pure noise, 1,000 steps
each. Its samples are scored with Day 138's metrics, the same ones that
judged Day 139's conditional GAN.

Everything is written from scratch in NumPy, on Day 126's dense networks and
Adam, Day 139's schedule and scorer, and Day 140's reverse process, all
imported unchanged:
  * the best straight-line noise guess, eps ~ c(t) x_t, in closed form
  * one network, three ways to read its output: as the noise eps, as the
    noise on top of c(t) x_t, or as the clean image x_0
  * the training loop of Ho et al. (2020): a random image, a random t, a
    squared-error loss
  * generation with Day 140's reverse_sample, and checks of what comes out

  1. By hand: the straight-line guess, and how eps and x_0 trade errors.
  2. First drafts that failed, and what fixed them.
  3. Training the model, with its samples scored as it trains.
  4. What the trained model knows at each noise level.
  5. Generating digits: fresh noise per step, and clipping the guess.
  6. Has it copied its training images?

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
sys.path.insert(0, HERE)                                # this week's parts (Day 140's)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day139_gan_build_and_forward_diffusion as d139  # Day 139: train_classifier, Scorer (Day 138's)
    import day140_exercise as d140                      # Day 140: BETAS, ABAR, reverse_sample, time_embed

BETAS, ABAR = d140.BETAS, d140.ABAR                     # Day 139's linear schedule: 1,000 steps, beta from 0.0001 to 0.02
SIZES = [816, 512, 512, 784]                            # 784 pixels + 32 time numbers in, two hidden layers of 512, 784 numbers out
ACTS = ["relu", "relu", "linear"]                       # ReLU hidden layers, a plain linear output
STEPS = 30000                                           # training steps for the model of sections 3-6
CHECKS = (2500, 5000, 10000, 20000, 30000)              # steps at which its samples are scored while it trains


# =============================================================================
# THE MODEL -- NEW
# =============================================================================
def skip_coef(X2):                                      # NEW: the best c(t) for the guess eps ~ c(t) x_t, every t at once
    m2 = float((X2.astype(np.float64) ** 2).mean())                                   # E[x_0^2], the average squared pixel
    return np.sqrt(1 - ABAR) / (ABAR * m2 + 1 - ABAR)                                 # E[x_t eps] / E[x_t^2], with x_0 and eps independent


def guess(net, xt, t, kind="x0", C=None):               # NEW: the network's output read as a guess; t is one step per row
    A = d126.forward(net, np.concatenate([xt, d140.time_embed(t)], 1).astype(np.float32), ACTS)  # Day 126's forward pass
    o = A[-1].astype(np.float64); sa, sb = np.sqrt(ABAR[t])[:, None], np.sqrt(1 - ABAR[t])[:, None]  # how much signal and noise x_t holds
    if kind == "x0":                                                                  # the output is the clean image ...
        x0 = o; e = (xt - sa * x0) / sb                                               # ... and the noise is whatever is left over
    else:                                                                             # the output is the noise (on top of c(t) x_t for "eps+line") ...
        e = o + (C[t][:, None] * xt if kind == "eps+line" else 0.0); x0 = (xt - sb * e) / sa  # ... and the clean image follows
    return A, e, x0                                                                   # every layer's output, the noise guess, the clean-image guess


def sampler_fn(net, kind="x0", C=None, clip=False):     # NEW: the model as Day 140's eps_fn(x, t), with one t for the whole batch
    def eps_fn(x, t):
        _, e, x0 = guess(net, x, np.full(len(x), t), kind, C)
        if clip:                                                                      # optionally force the clean-image guess into [-1, 1]
            x0 = np.clip(x0, -1, 1); e = (x - np.sqrt(ABAR[t]) * x0) / np.sqrt(1 - ABAR[t])  # and recompute the noise it implies
        return e
    return eps_fn


def to_pixels(S):                                       # NEW: samples in [-1, 1] back to pixels in [0, 1], for the scorer and pictures
    return np.clip((S + 1) / 2, 0, 1).astype(np.float32)


def train_image_denoiser(X2, steps, kind="x0", C=None, sizes=SIZES, zero_out=True, seed=0, lr=1e-3, batch=128,
                         every=1000, Xev=None, hook=None):  # NEW: Ho et al.'s training loop on images
    net = d126.init_net(sizes, seed); opt = d126.Adam(net, lr)                        # Day 126's He-initialised network and Adam
    if zero_out:
        net[-1] = (net[-1][0] * 0, net[-1][1])                                        # the output layer starts at zero (section 2 shows why)
    rng = np.random.RandomState(seed); log = []; run = np.zeros(3)                    # running sums: loss, ReLUs on in layers 1 and 2
    for s in range(1, steps + 1):
        x0 = X2[rng.randint(0, len(X2), batch)].astype(np.float64); t = rng.randint(1, len(BETAS) + 1, batch)  # random digits, a random t each
        e = rng.standard_normal(x0.shape); xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e  # Day 139's shortcut
        A, eh, x0h = guess(net, xt, t, kind, C)                                       # the guesses
        miss = x0h - x0 if kind == "x0" else eh - e                                   # the error in what the network was asked for
        grads, _ = d126.backward(net, A, ACTS, (miss * 2 / miss.size).astype(np.float32))  # d mean squared error / d output (c(t) x_t has no parameters)
        opt.step(net, grads)
        run += ((miss ** 2).mean(), (A[1] > 0).mean(), (A[2] > 0).mean())             # loss, and the share of ReLUs that are on
        if s % every == 0:
            ev = eval_net(net, *Xev, kind=kind, C=C)[0 if kind == "x0" else 1] if Xev is not None else np.nan  # the same error, on test triples
            log.append((s, *(run / every), ev)); run[:] = 0                           # (step, train loss, layer 1 on, layer 2 on, test error)
        if hook is not None:
            hook(s, net)                                                              # e.g. score samples at chosen steps
    return net, log


def eval_net(net, x0, t, e, kind="x0", C=None):          # NEW: mean squared error of both guesses on fixed (x_0, t, eps) triples
    xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e; ex = ee = 0.0
    for i in range(0, len(t), 500):                                                   # in batches of 500
        _, eh, x0h = guess(net, xt[i:i + 500], t[i:i + 500], kind, C)
        ex += ((x0h - x0[i:i + 500]) ** 2).sum(); ee += ((eh - e[i:i + 500]) ** 2).sum()
    return float(ex / x0.size), float(ee / e.size)                                    # (x_0 error, eps error)


def nearest_train(F, Xtr, chunk=250):                   # NEW: each image's pixel distance to its nearest training image, and which one
    n2 = (Xtr.astype(np.float32) ** 2).sum(1); dist, idx = [], []
    for i in range(0, len(F), chunk):
        D = (F[i:i + chunk] ** 2).sum(1)[:, None] + n2[None] - 2 * F[i:i + chunk] @ Xtr.T  # squared distances, by ||a||^2 + ||b||^2 - 2 a.b
        j = D.argmin(1); idx.append(j); dist.append(np.sqrt(np.maximum(D[np.arange(len(j)), j], 0)))
    return np.concatenate(dist), np.concatenate(idx)


def sample_scores(net, sc, kind="x0", C=None, var="posterior", clip=False, n=2000):  # NEW: n samples from seed 1, timed and scored
    t0 = time.perf_counter(); F = to_pixels(d140.reverse_sample(sampler_fn(net, kind, C, clip), n, np.random.RandomState(1), var, d=784))
    return F, time.perf_counter() - t0, sc(F)                                         # pixels, seconds, (IS, FID, precision, recall, nearest)


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand(X2):
    print("=" * 70)
    print("1. BY HAND -- the best straight-line noise guess, and how eps and x_0 trade errors")
    print("=" * 70)
    C = skip_coef(X2[:60000]); m2 = float((X2[:60000].astype(np.float64) ** 2).mean())
    print(f"pixels scaled to [-1, 1]; E[x_0^2] over the 60,000 training digits = {m2:.4f} (most pixels are background, -1)")
    print("c(t) = sqrt(1 - abar_t) / (abar_t E[x_0^2] + 1 - abar_t), against c fitted by least squares on 1,000 test digits noised to t;")
    print("since x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, a miss of d in x_0 is a miss of sqrt(abar_t / (1 - abar_t)) d in eps")
    print(f"{'t':>6} {'sqrt(1-abar)':>13} {'c(t)':>8} {'fitted c':>9} {'eps error of c(t) x_t':>22} {'x_0 miss -> eps miss':>21}")
    rng = np.random.RandomState(3); x0 = X2[60000 + rng.permutation(10000)[:1000]].astype(np.float64)
    for t in (1, 10, 50, 100, 200, 300, 500, 700, 1000):
        e = rng.standard_normal(x0.shape); xt = np.sqrt(ABAR[t]) * x0 + np.sqrt(1 - ABAR[t]) * e
        fit = (xt * e).sum() / (xt * xt).sum(); err = ((C[t] * xt - e) ** 2).mean()  # least squares through the origin, and the line's error
        print(f"{t:6d} {np.sqrt(1 - ABAR[t]):13.4f} {C[t]:8.4f} {fit:9.4f} {err:22.4f} {np.sqrt(ABAR[t] / (1 - ABAR[t])):21.4f}")
    print("at large t, x_t is nearly all noise, so c(t) x_t is nearly the right noise guess; at small t the noise is a sliver of x_t")
    print("and a straight line cannot pull it out. The last column is why the choice of target matters: at t = 1 a tiny miss in x_0")
    print("is a 100-times-larger miss in eps, at t = 1,000 a large miss in x_0 barely moves eps")
    print("Actual output from this lesson's run.")
    return C


def demo_first_drafts(X2, C, sc):
    print("\n" + "=" * 70)
    print("2. FIRST DRAFTS THAT FAILED, and what fixed them")
    print("=" * 70)
    print(f"network {SIZES[0]} -> {SIZES[1]} -> {SIZES[2]} -> {SIZES[3]}, ReLU, Adam lr 0.001, batches of 128, t uniform on 1-1,000")
    print("(a) the first draft: the output read as eps, Day 126's He initialisation everywhere. Training loss, and the share of ReLUs")
    print("that are on, each averaged over the last 50 steps:")
    print(f"{'output layer':>14} {'step':>6} {'loss':>7} {'layer 1 on':>11} {'layer 2 on':>11}")
    for zero in (False, True):
        _, log = train_image_denoiser(X2[:60000], 400, "eps", zero_out=zero, every=50)
        for s, loss, a1, a2, _ in log:
            if s in (50, 100, 200, 400):
                print(f"{'zero' if zero else 'He':>14} {s:6d} {loss:7.3f} {a1:11.2f} {a2:11.2f}")
    rng = np.random.RandomState(4); x0 = X2[60000 + rng.permutation(10000)[:1000]].astype(np.float64); e = rng.standard_normal(x0.shape)
    t = np.full(1000, 500); xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e
    print(f"before any training, the He-initialised output's spread (sd) at t = 500: {guess(d126.init_net(SIZES, 0), xt, t, 'eps')[1].std():.3f}; "
          f"the target eps's: {e.std():.3f}")
    print("A likely reason for the stall, consistent with these numbers but not proved here: the first large steps that shrink an")
    print("oversized output switch the second layer's ReLUs off; a switched-off ReLU passes no gradient, so they stay off, and the")
    print("network outputs a constant (loss near 1.0, the error of guessing eps = 0). A zero output layer starts every guess at 0.")
    print("\n(b) with the zero start, 3,000 steps each, then 2,000 samples each from the same noise, scored")
    ink = float(((X2[60000:] + 1) / 2).mean())
    print(f"('ink' is the average pixel, 0-1; the real test digits' is {ink:.3f}; Day 138's scores for them: IS 9.407, FID 2.996)")
    print(f"{'what the output is read as':>34} {'IS':>6} {'FID':>8} {'precision':>10} {'recall':>7} {'ink':>6} {'train':>6}")
    out = {}
    for name, kind, sizes in (("eps, 512 wide", "eps", SIZES), ("eps, 1,024 wide", "eps", [816, 1024, 1024, 784]),
                              ("eps on top of c(t) x_t, 512 wide", "eps+line", SIZES), ("x_0 (the clean digit), 512 wide", "x0", SIZES)):
        t0 = time.perf_counter(); net, _ = train_image_denoiser(X2[:60000], 3000, kind, C, sizes=sizes, every=3000); took = time.perf_counter() - t0
        F, _, (i, f, p, r, n) = sample_scores(net, sc, kind, C); out[name] = F[:8]
        print(f"{name:>34} {i:6.3f} {f:8.2f} {p:10.3f} {r:7.3f} {F.mean():6.3f} {took:5.0f}s")
    print("the eps readings leave speckled noise in their samples (too much ink); widening to 1,024 does not cure it; reading the")
    print("same network's output as x_0 does. A likely reason, not tested further here: to output eps the network must carry 784")
    print("independent noise values through 512 (or 1,024) ReLUs, while x_0 is a digit, far simpler to describe; reading the")
    print("output as x_0 hands the noise part, x_t itself, to the formula eps = (x_t - sqrt(abar) x_0) / sqrt(1 - abar) for free")
    print("Actual output from this lesson's run.")
    return out, ink


def demo_train(X2, Xev, sc):
    print("\n" + "=" * 70)
    print("3. TRAINING THE MODEL -- the output read as x_0, zero-started, with its samples scored as it trains")
    print("=" * 70)
    print(f"{SIZES[0]} -> {SIZES[1]} -> {SIZES[2]} -> {SIZES[3]} (ReLU); Adam lr 0.001; batches of 128 training digits; {STEPS:,} steps;")
    print("loss: mean squared error of the x_0 guess (pixels on the [-1, 1] scale), t uniform on 1-1,000")
    mean_digit = X2[:60000].astype(np.float64).mean(0); base = float(((mean_digit - Xev[0]) ** 2).mean())  # the best guess that ignores x_t
    print(f"test error: 1,000 test digits, each with its own fixed random t and eps; guessing the average training digit scores {base:.4f}")
    print("scores: 2,000 samples from the same noise (seed 1), Day 140's reverse_sample with fresh noise beta-tilde_t, scored by Day 138's")
    print("Scorer (the real test digits: IS 9.407, FID 2.996, precision 0.855, recall 0.884, nearest 4.49)")
    scores = []; t_sample = [0.0]

    def hook(s, net):                                                                 # scores the samples at the chosen steps
        if s in CHECKS:
            _, took, r = sample_scores(net, sc); t_sample[0] += took; scores.append((s, *r))
    t0 = time.perf_counter(); net, log = train_image_denoiser(X2[:60000], STEPS, Xev=Xev, hook=hook); total = time.perf_counter() - t0
    print(f"{'step':>8} {'train loss':>11} {'test error':>11}")
    for s, loss, _, _, ev in log:
        if s in (1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000):
            print(f"{s:8,d} {loss:11.4f} {ev:11.4f}")
    print(f"{'step':>8} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'nearest':>8}")
    for s, i, f, p, r, n in scores:
        print(f"{s:8,d} {i:7.3f} {f:8.3f} {p:10.3f} {r:7.3f} {n:8.2f}")
    print(f"training {total - t_sample[0]:.0f} s; sampling 2,000 digits, 1,000 network calls each, {t_sample[0] / len(CHECKS):.0f} s a time")
    print("Actual output from this lesson's run.")
    return net, log, scores, base, mean_digit


def demo_by_t(net, Xev, mean_digit):
    print("\n" + "=" * 70)
    print("4. WHAT THE TRAINED MODEL KNOWS AT EACH NOISE LEVEL")
    print("=" * 70)
    x0, e = Xev[0], Xev[2]; rows = []
    print("1,000 test digits at each t. x_0 errors are pixel RMSEs on the 0-1 scale (0.1: a typical pixel off by a tenth of")
    print("black-to-white). Two guesses that need no training: undo the shrink, x_t / sqrt(abar_t); and the average training digit")
    print(f"{'t':>6} {'x_0 RMSE, model':>16} {'x_t / sqrt(abar)':>17} {'average digit':>14} {'eps error, model':>17}")
    for t in (1, 10, 50, 100, 200, 300, 500, 700, 1000):
        tt = np.full(len(x0), t); xt = np.sqrt(ABAR[t]) * x0 + np.sqrt(1 - ABAR[t]) * e
        _, eh, x0h = guess(net, xt, tt)
        r = (t, np.sqrt(((x0h - x0) ** 2).mean()) / 2, np.sqrt(((xt / np.sqrt(ABAR[t]) - x0) ** 2).mean()) / 2,
             np.sqrt(((mean_digit - x0) ** 2).mean()) / 2, ((eh - e) ** 2).mean())   # /2: from the [-1, 1] scale to the 0-1 scale
        rows.append(r); print(f"{r[0]:6d} {r[1]:16.4f} {r[2]:17.4f} {r[3]:14.4f} {r[4]:17.4f}")
    print("the eps error at small t is large even where the x_0 guess is good: section 1's last column, at work")
    print("Actual output from this lesson's run.")
    pics = []                                                                         # one test digit, noised and denoised, for the figure
    for t in (100, 300, 500, 700, 900):
        z = np.random.RandomState(t).standard_normal((1, 784)); xt = np.sqrt(ABAR[t]) * x0[:1] + np.sqrt(1 - ABAR[t]) * z
        pics.append((t, to_pixels(xt)[0], to_pixels(guess(net, xt, np.array([t]))[2])[0]))
    return rows, pics


def demo_sampling(net, sc, ink):
    print("\n" + "=" * 70)
    print("5. GENERATING DIGITS -- fresh noise per step, and clipping the x_0 guess")
    print("=" * 70)
    print("2,000 samples each, from the same starting noise (seed 1); fresh noise per step: beta-tilde_t (the posterior's), beta_t, or none;")
    print("'clip': the x_0 guess forced into [-1, 1] at every step before the noise guess is worked out from it")
    print(f"'digits': how many samples Day 138's feature network names as each digit 0-9 (the real test set has about 200 each); real ink {ink:.3f}")
    print(f"{'noise':>10} {'clip':>5} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'ink':>6} {'time':>6}   digits 0-9")
    out = {}
    for clip in (False, True):
        for var in ("posterior", "beta", "none"):
            F, took, (i, f, p, r, n) = sample_scores(net, sc, var=var, clip=clip); out[(var, clip)] = F
            said = np.concatenate([sc.head.forward(sc.feats(F[k:k + 1000])).argmax(1) for k in range(0, 2000, 1000)])  # which digit each one shows
            print(f"{var:>10} {'yes' if clip else 'no':>5} {i:7.3f} {f:8.3f} {p:10.3f} {r:7.3f} {F.mean():6.3f} {took:5.0f}s   "
                  f"{' '.join(str(c) for c in np.bincount(said, minlength=10))}")
    print("Day 139's conditional GAN after 10 epochs, for comparison: IS 7.580, FID 20.200, precision 0.442, recall 0.573")
    print("Actual output from this lesson's run.")
    return out


def demo_copying(F, X):
    print("\n" + "=" * 70)
    print("6. HAS IT COPIED ITS TRAINING IMAGES?")
    print("=" * 70)
    Xtr = X[:60000]; T = X[60000 + np.random.RandomState(0).permutation(10000)[:2000]]   # 2,000 test digits, at random (the stored order is not)
    ds, js = nearest_train(F, Xtr); dt, _ = nearest_train(T, Xtr)                     # each one's nearest training digit
    print("pixel distance (0-1 pixels, 784 of them) from each image to its nearest of the 60,000 training digits")
    print(f"{'images':>22} {'min':>6} {'1%':>6} {'10%':>6} {'median':>7} {'mean':>6}")
    for name, d in (("2,000 generated", ds), ("2,000 real test digits", dt)):
        print(f"{name:>22} {d.min():6.2f} {np.percentile(d, 1):6.2f} {np.percentile(d, 10):6.2f} {np.median(d):7.2f} {d.mean():6.2f}")
    print(f"generated digits closer to a training digit than the closest real test digit is ({dt.min():.2f}): {int((ds < dt.min()).sum())}")
    print(f"distinct nearest training digits among the 2,000 samples: {len(np.unique(js))}")
    print("a test digit was never trained on, so its distances show how close an honest new digit lands by chance")
    print("Actual output from this lesson's run.")
    return ds, dt, js


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(log, base, scores, drafts, rows, pics, samp, near, X, out):  # nine panels
    fig = plt.figure(figsize=(17, 13)); gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[0, 0]); s = [r[0] for r in log]                          # the loss curves
    ax.plot(s, [r[1] for r in log], label="training loss"); ax.plot(s, [r[4] for r in log], label="test error")
    ax.axhline(base, color="gray", ls="--", label=f"guess the average digit, {base:.3f}")
    ax.set_ylim(0, base * 1.15); ax.set_xlabel("training step"); ax.set_ylabel("mean squared error of x_0 ([-1, 1] scale)")
    ax.set_title("learning to guess the clean digit", fontsize=10); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 1]); ax.plot([r[0] for r in scores], [r[2] for r in scores], marker="o", label="diffusion model")
    ax.axhline(20.200, color="#C44E52", ls="--", label="Day 139's conditional GAN, 10 epochs"); ax.axhline(2.996, color="gray", ls=":", label="real test digits")
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("FID (log scale)"); ax.set_title("sample quality while training", fontsize=10); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 2]); ts = [r[0] for r in rows]                         # error by t
    ax.plot(ts, [r[1] for r in rows], marker="o", label="model"); ax.plot(ts, [r[2] for r in rows], marker="s", label="x_t / sqrt(abar_t)")
    ax.plot(ts, [r[3] for r in rows], ls="--", color="gray", label="average digit"); ax.set_ylim(0, 0.5)
    ax.set_xscale("log"); ax.set_xlabel("t (log scale)"); ax.set_ylabel("RMSE of the x_0 guess (0-1 pixels)")
    ax.set_title("how well it recovers the clean digit", fontsize=10); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 3]); ax.hist(near[1], bins=40, alpha=0.6, label="real test digits"); ax.hist(near[0], bins=40, alpha=0.6, label="generated digits")
    ax.set_xlabel("pixel distance to the nearest training digit"); ax.set_ylabel("images"); ax.set_title("copying check", fontsize=10); ax.legend(fontsize=8)

    def grid(ax, F, r, c, title):                                                     # r x c images tiled into one picture
        ax.imshow(np.vstack([np.hstack([F[i * c + j].reshape(28, 28) for j in range(c)]) for i in range(r)]), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10); ax.axis("off")
    grid(fig.add_subplot(gs[1, 0:2]), samp[("posterior", False)], 4, 12, f"48 generated digits (fresh noise beta-tilde_t), after {STEPS:,} steps")
    grid(fig.add_subplot(gs[1, 2]), np.concatenate(list(drafts.values())), 4, 8, "section 2, 3,000 steps each, rows:\n"
         "eps 512; eps 1,024; eps on c(t) x_t; x_0")
    grid(fig.add_subplot(gs[1, 3]), samp[("posterior", True)], 4, 4, "the same noise, x_0 guess clipped")
    ax = fig.add_subplot(gs[2, 0:2])                                                  # one digit noised and denoised in one step
    ax.imshow(np.vstack([np.hstack([p[1].reshape(28, 28) for p in pics]), np.hstack([p[2].reshape(28, 28) for p in pics])]), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("top: a test digit noised to t = " + ", ".join(str(p[0]) for p in pics) + "\nbottom: the model's one-step guess of x_0", fontsize=10); ax.axis("off")
    ax = fig.add_subplot(gs[2, 2:4]); o = np.argsort(near[0])[:8]                     # the closest samples and their nearest training digits
    top = np.hstack([samp[("posterior", False)][i].reshape(28, 28) for i in o]); bot = np.hstack([X[near[2][i]].reshape(28, 28) for i in o])
    ax.imshow(np.vstack([top, bot]), cmap="gray_r", vmin=0, vmax=1); ax.axis("off")
    ax.set_title("top: the 8 generated digits closest to any training digit\nbottom: that nearest training digit", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # training and sampling
    fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, c="black"):                                             # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3, color=c))
    ax.text(0.2, 4.05, "TRAINING (one step)", fontsize=10, weight="bold")
    box(1.2, 3.2, 1.9, 0.8, "a training digit x_0\nrandom t, eps", "#E2F0D9")
    box(3.9, 3.2, 2.3, 0.8, "x_t = sqrt(abar_t) x_0\n+ sqrt(1 - abar_t) eps", "#EEEEEE")
    box(7.0, 3.2, 2.8, 0.8, "MLP(x_t, time_embed(t))\n= x_0-hat", "#FFF2CC")
    box(10.6, 3.2, 2.6, 0.8, "loss = mean (x_0-hat - x_0)^2\nDay 126 backward, Adam", "#F4CCCC")
    arrow(2.2, 3.2, 2.7, 3.2); arrow(5.1, 3.2, 5.55, 3.2); arrow(8.45, 3.2, 9.25, 3.2)
    ax.text(0.2, 1.95, "SAMPLING (Day 140's reverse_sample, 1,000 steps)", fontsize=10, weight="bold")
    box(1.2, 1.1, 1.9, 0.8, "x_1000\n~ N(0, I)", "#EEEEEE")
    box(4.6, 1.1, 3.4, 0.8, "x_0-hat = MLP(x_t, t)\neps-hat = (x_t - sqrt(abar_t) x_0-hat) / sqrt(1 - abar_t)", "#FFF2CC", 7.5)
    box(9.0, 1.1, 4.0, 0.8, "x_(t-1) = (x_t - beta_t / sqrt(1 - abar_t) eps-hat)\n/ sqrt(1 - beta_t) + sigma_t z", "#DDEBF7")
    box(12.2, 1.1, 1.1, 0.8, "x_0:\na digit", "#E2F0D9")
    arrow(2.2, 1.1, 2.85, 1.1); arrow(6.35, 1.1, 6.95, 1.1); arrow(11.05, 1.1, 11.6, 1.1)
    ax.plot([9.0, 9.0, 4.6], [0.67, 0.35, 0.35], color="#8B1A1A", lw=1.3); arrow(4.6, 0.35, 4.6, 0.67, "#8B1A1A")  # the loop back: one path, one arrowhead
    ax.text(6.8, 0.08, "t -> t - 1, back to the model, 1,000 times", ha="center", fontsize=9, color="#8B1A1A")
    ax.set_title("Day 141: a tiny diffusion model on MNIST digits", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "tiny_diffusion_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 142 puts this model next to Week 19's VAE and Week 20's GAN on the same scorer, counts what each costs to train and")
    print("to sample, and makes sampling faster by taking fewer, larger reverse steps (DDIM).")


def main():                                             # runs every section in order
    X, y = d126m.mnist(); X2 = (X * 2 - 1).astype(np.float32)                         # Day 126's 70,000 digits; pixels to [-1, 1]
    rng = np.random.RandomState(5); ex0 = X2[60000 + rng.permutation(10000)[:1000]].astype(np.float64)  # 1,000 fixed test digits
    Xev = (ex0, rng.randint(1, 1001, 1000), rng.standard_normal(ex0.shape))           # each with its own t and eps
    C = demo_hand(X2)                                                                 # 1. the straight-line guess
    t0 = time.perf_counter(); body, head = d139.train_classifier(X[:60000], y[:60000])  # Day 138's feature network, for every score below
    sc = d139.Scorer(body, head, X[:10000], X[10000:30000]); print(f"\n(feature network for the scores: {time.perf_counter() - t0:.0f} s)")
    drafts, ink = demo_first_drafts(X2, C, sc)                                        # 2. the first drafts
    net, log, scores, base, mean_digit = demo_train(X2, Xev, sc)                      # 3. the real training run
    rows, pics = demo_by_t(net, Xev, mean_digit)                                      # 4. its error at each t
    samp = demo_sampling(net, sc, ink)                                                # 5. generating digits
    near = demo_copying(samp[("posterior", False)], X)                                # 6. the copying check
    plot_results(log, base, scores, drafts, rows, pics, samp, near, X, os.path.join(HERE, "tiny_diffusion_results.png"))
    draw_diagram()                                                                    # training and sampling
    print("\nsaved tiny_diffusion_results.png\nsaved tiny_diffusion_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
