"""
Day 142 Exercise -- Faster Sampling and an Averaged Model

The third of Week 21's parts for Day 146's capstone.

    ema_update(ema, net, decay)            move an averaged copy of the weights towards the current ones
    ddim_sample(x0_fn, n, rng, steps, eta) DDIM: sample in `steps` jumps instead of 1,000 (eta 0: no fresh noise)
    net_x0_fn(net)                         a Day 141 model as a function (x_t, t) -> x_0 guess
    grey_share(F)                          the share of pixels between 0.25 and 0.75: a measure of blur
    n_params(*nets)                        how many numbers the networks hold

Built on Day 140's schedule and sampler and Day 141's model, imported unchanged.
Checked by hand: one averaging step by hand, DDIM with eta 1 and 1,000 steps
reproducing Day 140's sampler, the grey share of a made-up image, and a short-trained
model sampled in 20 DDIM steps.
"""
import os                                              # os: the paths to earlier weeks' folders
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
sys.path.insert(0, HERE)                                # this week's parts (Days 140-141)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day140_exercise as d140                      # Day 140: ABAR, reverse_sample
    import day141_exercise as d141                      # Day 141: guess, train_image_denoiser, to_pixels
ABAR = d140.ABAR                                        # Day 139's linear schedule, as in the lesson
DECAY = 0.999                                           # as in the lesson


def ema_update(ema, net, decay=DECAY):                  # NEW: Day 142's version, unchanged
    for (We, be), (W, b) in zip(ema, net):
        We *= decay; We += (1 - decay) * W                                            # W_avg = decay W_avg + (1 - decay) W
        be *= decay; be += (1 - decay) * b                                            # the same for the biases


def ddim_sample(x0_fn, n, rng, steps, eta=0.0, d=784):  # NEW: Day 142's version, unchanged
    ts = np.linspace(len(ABAR) - 1, 0, steps + 1).round().astype(int)                  # e.g. 1000, 950, ..., 0 for 20 steps
    x = rng.standard_normal((n, d))                                                   # pure noise, as in Day 140's sampler
    for t, s in zip(ts[:-1], ts[1:]):                                                 # one jump from t to the earlier step s
        x0 = x0_fn(x, t); e = (x - np.sqrt(ABAR[t]) * x0) / np.sqrt(1 - ABAR[t])       # the guess of x_0, and the noise it implies
        sig = eta * np.sqrt((1 - ABAR[s]) / (1 - ABAR[t]) * (1 - ABAR[t] / ABAR[s]))   # fresh noise: none for eta = 0; Day 140's beta-tilde for eta = 1, stride 1
        x = np.sqrt(ABAR[s]) * x0 + np.sqrt(np.maximum(1 - ABAR[s] - sig ** 2, 0)) * e  # put x_0 back at step s's level, with the same noise direction
        if s > 0:
            x = x + sig * rng.standard_normal(x.shape)                                # plus fresh noise, except on the very last jump
    return x


def net_x0_fn(net):                                     # NEW: Day 142's version, unchanged
    return lambda x, t: d141.guess(net, x, np.full(len(x), t))[2]


def grey_share(F, lo=0.25, hi=0.75):                    # NEW: Day 142's version, unchanged
    return float(((F > lo) & (F < hi)).mean())


def n_params(*nets):                                    # NEW: Day 142's version, unchanged
    return int(sum(W.size + b.size for net in nets for W, b in net))


net = [(np.ones((2, 2)), np.zeros(2))]; ema = [(np.zeros((2, 2)), np.ones(2))]; ema_update(ema, net, 0.9)
print(f"one averaging step, decay 0.9: weights 0 -> {ema[0][0][0, 0]:.2f} (0.9 x 0 + 0.1 x 1), biases 1 -> {ema[0][1][0]:.2f} (0.9 x 1 + 0.1 x 0)")
f = lambda x, t: np.tanh(x)                                                           # any x_0 guesser
A = d140.reverse_sample(lambda x, t: (x - np.sqrt(ABAR[t]) * f(x, t)) / np.sqrt(1 - ABAR[t]), 20, np.random.RandomState(4), "posterior", d=784)
print(f"DDIM, eta 1, 1,000 steps against Day 140's sampler: largest difference {np.abs(A - ddim_sample(f, 20, np.random.RandomState(4), 1000, 1.0)).max():.1e}")
F = np.zeros((1, 784)); F[0, :100] = 1.0; F[0, 100:150] = 0.5
print(f"an image with 100 ink pixels and 50 at 0.5: grey share {grey_share(F):.4f} (50 / 784 = {50 / 784:.4f})")
X, _ = d126m.mnist(); X2 = (X * 2 - 1).astype(np.float32); net, _ = d141.train_image_denoiser(X2[:60000], 1000)
S = d141.to_pixels(ddim_sample(net_x0_fn(net), 100, np.random.RandomState(1), 20))
print(f"a model trained 1,000 steps ({n_params(net):,} numbers), 100 samples in 20 DDIM steps: ink {S.mean():.3f}, grey share {grey_share(S):.4f}")
