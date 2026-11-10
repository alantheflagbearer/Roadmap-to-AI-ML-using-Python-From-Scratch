"""
Day 144 Exercise -- Prompts That Steer a Diffusion Model

The fifth of Week 21's parts for Day 146's capstone.

    guess_c(net, xt, t, c)                  the x_0 guess, given a caption embedding c as an extra input
    train_cond(X2, y, thick, lean, txt)     Day 141's model trained on captioned digits, 15% of captions dropped
    cfg_sample(net, cpos, n, w, cneg)       classifier-free guidance inside DDIM, with an optional negative prompt
    ink_lean(F)                             Day 143's thickness and lean measurements for any images
    said(sc, F)                             the digit Day 138's classifier sees in each image

Built on Days 126, 138, 140, 141, 142 and 143's parts, imported unchanged.
Checked by hand: a guided guess with w = 1 and the 'no caption' guess spelled out
equals the plain captioned guess, the measurements of a made-up stroke, and a
briefly trained model that already draws thicker digits for 'zero' than for 'one'.
"""
import os                                              # os: the paths to earlier weeks' folders
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # as in the lesson
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
sys.path.insert(0, HERE)                                # this week's parts (Days 140-143)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day140_exercise as d140                      # Day 140: ABAR, time_embed
    import day141_exercise as d141                      # Day 141: to_pixels
    import day142_exercise as d142                      # Day 142: ema_update, ddim_sample
    import day143_exercise as d143                      # Day 143: measure, caption, train_clip, embed_texts
ABAR = d140.ABAR                                        # as in the lesson
ACTS = ["relu", "relu", "linear"]
SIZES = [784 + 32 + 64, 512, 512, 784]
STEPS = 20000
DROP = 0.15
HOLD = (7, 1)
SAMPLE_STEPS = 100


def guess_c(net, xt, t, c):                             # NEW: Day 144's version, unchanged
    A = d126.forward(net, np.concatenate([xt, d140.time_embed(t), c], 1).astype(np.float32), ACTS)  # 784 + 32 + 64 inputs per row
    return A, A[-1].astype(np.float64)                                                # every layer's output (for backward), and the guess


def train_cond(X2, y, thick, lean, txt, steps=STEPS, drop=DROP, hold=HOLD, seed=0, batch=128, lr=1e-3, every=2000):  # NEW: Day 144's version, unchanged
    net = d126.init_net(SIZES, seed); net[-1] = (net[-1][0] * 0, net[-1][1])          # Day 126's network, output layer started at zero (Day 141)
    opt = d126.Adam(net, lr); ema = [(W.copy(), b.copy()) for W, b in net]            # Adam, and Day 142's averaged copy
    rng = np.random.RandomState(seed); log = []; run = 0.0
    for s in range(1, steps + 1):
        i = rng.randint(0, len(X2), batch); x0 = X2[i].astype(np.float64); t = rng.randint(1, len(ABAR), batch)  # random digits, a random t each
        e = rng.standard_normal(x0.shape); xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e  # Day 139's shortcut
        c = d143.embed_texts(txt, [d143.caption(y[k], thick[k], lean[k], rng, hold) for k in i])  # a fresh random caption for each, embedded
        c[rng.rand(batch) < drop] = 0.0                                               # some captions dropped: the same network learns 'no caption'
        A, x0h = guess_c(net, xt, t, c); miss = x0h - x0                              # Day 141's x_0 loss
        grads, _ = d126.backward(net, A, ACTS, (miss * 2 / miss.size).astype(np.float32))
        opt.step(net, grads); d142.ema_update(ema, net); run += (miss ** 2).mean()     # one Adam step, then the averaged copy follows
        if s % every == 0:
            log.append((s, run / every)); run = 0.0                                   # the average loss over the last `every` steps
    return net, ema, log


def cfg_sample(net, cpos, n, w, cneg=None, steps=SAMPLE_STEPS, seed=1):  # NEW: Day 144's version, unchanged
    cp = np.repeat(cpos[None], n, 0)                                                  # the prompt's embedding, one row per sample
    cn = np.zeros_like(cp) if cneg is None else np.repeat(cneg[None], n, 0)           # 'no caption', or a negative prompt in its place

    def x0_fn(x, t):                                                                  # the guided x_0 guess at step t
        tt = np.full(len(x), t); a = guess_c(net, x, tt, cp)[1]                        # with the prompt
        if w == 1.0 and cneg is None:
            return a                                                                  # w = 1 is plain conditioning: one network call is enough
        b = guess_c(net, x, tt, cn)[1]                                                # without it (or with the negative prompt)
        return b + w * (a - b)                                                        # w = 0: ignore the prompt; w > 1: push past it
    return d141.to_pixels(d142.ddim_sample(x0_fn, n, np.random.RandomState(seed), steps, 0.0))


def ink_lean(F):                                        # NEW: Day 144's version, unchanged
    R = np.repeat(np.arange(28.0), 28)[None]; Cc = np.tile(np.arange(28.0), 28)[None]  # each pixel's row and column
    m = F.sum(1, keepdims=True) + 1e-9; mr = (F * R).sum(1, keepdims=True) / m; mc = (F * Cc).sum(1, keepdims=True) / m  # the ink's centre
    return F.mean(1), -(F * (R - mr) * (Cc - mc)).sum(1) / (F * (R - mr) ** 2).sum(1)  # thickness (average pixel) and lean


def said(sc, F):                                        # NEW: Day 144's version, unchanged
    return np.concatenate([sc.head.forward(sc.feats(F[i:i + 1000])).argmax(1) for i in range(0, len(F), 1000)])


X, y = d126m.mnist(); X2 = (X * 2 - 1).astype(np.float32); _, _, thick, lean = d143.measure(X, y)
_, txt, _ = d143.train_clip(X[:20000], y[:20000], thick[:20000], lean[:20000], epochs=2, hold=HOLD)   # a quickly trained text encoder
net, ema, log = train_cond(X2[:60000], y[:60000], thick[:60000], lean[:60000], txt, steps=1500, every=500)
c7 = d143.embed_texts(txt, [["seven"]])[0]
A = cfg_sample(net, c7, 20, 1.0); B = cfg_sample(net, c7, 20, 1.0, cneg=np.zeros(64))       # the second spells out 'no caption' as a negative prompt
print(f"w = 1 with 'no caption' passed explicitly against the one-call shortcut: largest difference {np.abs(A - B).max():.2e}")
F = np.zeros((1, 784)); F[0, [r * 28 + 14 + (13 - r) // 3 for r in range(28)]] = 1.0       # a one-pixel stroke leaning right, 28 pixels of ink
i, l = ink_lean(F); print(f"a made-up stroke: ink {i[0]:.4f} (28 / 784 = {28 / 784:.4f}), lean {l[0]:.3f}")
print(f"1,500 steps of training: loss {log[0][1]:.4f} -> {log[-1][1]:.4f}")
i0, _ = ink_lean(cfg_sample(net, d143.embed_texts(txt, [["zero"]])[0], 100, 2.0)); i1, _ = ink_lean(cfg_sample(net, d143.embed_texts(txt, [["one"]])[0], 100, 2.0))
print(f"guided samples, w = 2: 'zero' ink {i0.mean():.4f}, 'one' ink {i1.mean():.4f} (real training digits: {X[:60000][y[:60000] == 0].mean():.4f} and {X[:60000][y[:60000] == 1].mean():.4f})")
