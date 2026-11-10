"""
Day 141 Exercise -- A Tiny Diffusion Model on Digits

The second of Week 21's parts for Day 146's capstone.

    skip_coef(X2)                          the best straight-line noise guess c(t), for every t at once
    guess(net, xt, t, kind, C)             the network's output read as x_0 ("x0"), as eps ("eps"), or as eps on c(t) x_t
    sampler_fn(net, kind, C, clip)         the model as Day 140's eps_fn, for reverse_sample
    to_pixels(S)                           [-1, 1] back to [0, 1] pixels
    train_image_denoiser(X2, steps, kind)  Ho et al.'s training loop, output layer started at zero
    eval_net(net, x0, t, e, kind)          its x_0 and eps errors on fixed (x_0, t, eps) triples
    nearest_train(F, Xtr)                  each image's distance to its nearest training image
    sample_scores(net, sc, ...)            2,000 samples from seed 1, timed and scored by Day 138's Scorer

Built on Day 126's networks and Adam and Day 140's schedule, time embedding and
reverse process, imported unchanged. Checked by hand: c(t) matches a least-squares
fit, the zero-started output layer keeps the second layer's ReLUs alive where He
initialisation does not, and a short x_0 run already beats both untrained guesses.
"""
import os                                              # os: the paths to earlier weeks' folders
import sys                                             # sys: path, to import them
import time                                            # time: sample_scores times its sampling
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
sys.path.insert(0, HERE)                                # this week's parts (Day 140's)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day140_exercise as d140                      # Day 140: BETAS, ABAR, reverse_sample, time_embed
BETAS, ABAR = d140.BETAS, d140.ABAR                     # Day 139's linear schedule, as in the lesson
SIZES = [816, 512, 512, 784]                            # as in the lesson
ACTS = ["relu", "relu", "linear"]                       # as in the lesson


def skip_coef(X2):                                      # NEW: Day 141's version, unchanged
    m2 = float((X2.astype(np.float64) ** 2).mean())                                   # E[x_0^2], the average squared pixel
    return np.sqrt(1 - ABAR) / (ABAR * m2 + 1 - ABAR)                                 # E[x_t eps] / E[x_t^2], with x_0 and eps independent


def guess(net, xt, t, kind="x0", C=None):               # NEW: Day 141's version, unchanged
    A = d126.forward(net, np.concatenate([xt, d140.time_embed(t)], 1).astype(np.float32), ACTS)  # Day 126's forward pass
    o = A[-1].astype(np.float64); sa, sb = np.sqrt(ABAR[t])[:, None], np.sqrt(1 - ABAR[t])[:, None]  # how much signal and noise x_t holds
    if kind == "x0":                                                                  # the output is the clean image ...
        x0 = o; e = (xt - sa * x0) / sb                                               # ... and the noise is whatever is left over
    else:                                                                             # the output is the noise (on top of c(t) x_t for "eps+line") ...
        e = o + (C[t][:, None] * xt if kind == "eps+line" else 0.0); x0 = (xt - sb * e) / sa  # ... and the clean image follows
    return A, e, x0                                                                   # every layer's output, the noise guess, the clean-image guess


def sampler_fn(net, kind="x0", C=None, clip=False):     # NEW: Day 141's version, unchanged
    def eps_fn(x, t):
        _, e, x0 = guess(net, x, np.full(len(x), t), kind, C)
        if clip:                                                                      # optionally force the clean-image guess into [-1, 1]
            x0 = np.clip(x0, -1, 1); e = (x - np.sqrt(ABAR[t]) * x0) / np.sqrt(1 - ABAR[t])  # and recompute the noise it implies
        return e
    return eps_fn


def to_pixels(S):                                       # NEW: Day 141's version, unchanged
    return np.clip((S + 1) / 2, 0, 1).astype(np.float32)


def train_image_denoiser(X2, steps, kind="x0", C=None, sizes=SIZES, zero_out=True, seed=0, lr=1e-3, batch=128,
                         every=1000, Xev=None, hook=None):  # NEW: Day 141's version, unchanged
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


def eval_net(net, x0, t, e, kind="x0", C=None):          # NEW: Day 141's version, unchanged
    xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e; ex = ee = 0.0
    for i in range(0, len(t), 500):                                                   # in batches of 500
        _, eh, x0h = guess(net, xt[i:i + 500], t[i:i + 500], kind, C)
        ex += ((x0h - x0[i:i + 500]) ** 2).sum(); ee += ((eh - e[i:i + 500]) ** 2).sum()
    return float(ex / x0.size), float(ee / e.size)                                    # (x_0 error, eps error)


def nearest_train(F, Xtr, chunk=250):                   # NEW: Day 141's version, unchanged
    n2 = (Xtr.astype(np.float32) ** 2).sum(1); dist, idx = [], []
    for i in range(0, len(F), chunk):
        D = (F[i:i + chunk] ** 2).sum(1)[:, None] + n2[None] - 2 * F[i:i + chunk] @ Xtr.T  # squared distances, by ||a||^2 + ||b||^2 - 2 a.b
        j = D.argmin(1); idx.append(j); dist.append(np.sqrt(np.maximum(D[np.arange(len(j)), j], 0)))
    return np.concatenate(dist), np.concatenate(idx)


def sample_scores(net, sc, kind="x0", C=None, var="posterior", clip=False, n=2000):  # NEW: Day 141's version, unchanged
    t0 = time.perf_counter(); F = to_pixels(d140.reverse_sample(sampler_fn(net, kind, C, clip), n, np.random.RandomState(1), var, d=784))
    return F, time.perf_counter() - t0, sc(F)                                         # pixels, seconds, (IS, FID, precision, recall, nearest)


X, _ = d126m.mnist(); X2 = (X * 2 - 1).astype(np.float32); C = skip_coef(X2[:60000])   # the lesson's data and c(t)
rng = np.random.RandomState(5); x0 = X2[60000 + rng.permutation(10000)[:1000]].astype(np.float64); e = rng.standard_normal(x0.shape)
xt = np.sqrt(ABAR[300]) * x0 + np.sqrt(1 - ABAR[300]) * e
print(f"c(300) = {C[300]:.4f}; least-squares fit on 1,000 test digits: {(xt * e).sum() / (xt * xt).sum():.4f}")
for zero in (False, True):
    _, log = train_image_denoiser(X2[:60000], 300, "eps", zero_out=zero, every=300)
    print(f"300 steps, output read as eps, output layer {'zero' if zero else 'He'}: loss {log[0][1]:.3f}, layer-2 ReLUs on {log[0][3]:.2f} (averages)")
net, _ = train_image_denoiser(X2[:60000], 1000)
t = np.full(1000, 300); ex, ee = eval_net(net, x0, t, e)
print(f"1,000 steps, output read as x_0, at t = 300: x_0 error {ex:.4f}; x_t / sqrt(abar) {((xt / np.sqrt(ABAR[300]) - x0) ** 2).mean():.4f}; "
      f"the average training digit {((X2[:60000].astype(np.float64).mean(0) - x0) ** 2).mean():.4f}")
F = to_pixels(d140.reverse_sample(sampler_fn(net), 100, np.random.RandomState(1), "posterior", d=784))
d, _ = nearest_train(F, X[:60000]); print(f"100 samples after 1,000 steps: ink {F.mean():.3f}, mean distance to the nearest training digit {d.mean():.2f}")
