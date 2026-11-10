"""
Day 140 Exercise -- One Reverse Step, and the Whole Way Back

The first of Week 21's parts for Day 146's capstone.

    posterior(t)                         q(x_(t-1) | x_t, x_0): the coefficients of x_0 and x_t, and the variance
    exact_x0(xt, t)                      E[x_0 | x_t] when the data is Day 135's ring of eight Gaussians
    eps_from_x0(xt, x0, t)               the noise implied by a guess of x_0
    reverse_sample(eps_fn, n, rng, var)  1,000 steps from N(0, I) back to data
    time_embed(t)                        a step number as 32 sines and cosines
    eps_net_fn(net), train_denoiser(...) a small network that learns to guess the noise
    eval_eps(eps_fn, x0, t, e)           its mean squared error on fixed triples
    mode_spread(S)                       how spread out samples are around their nearest centre

Built on Day 126's networks and Adam and Day 139's noise schedule, imported
unchanged. Checked by hand: the posterior at t = 1 has no variance left, one
exact denoising step at small t lands on the ring, and the whole reverse
process with the exact denoiser returns all eight modes.
"""
import os                                              # os: the paths to earlier weeks' folders
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day139_gan_build_and_forward_diffusion as d139  # Day 139: schedule, q_sample, ring, mode_stats
BETAS, ABAR = d139.schedule("linear")                   # Day 139's linear schedule, as in the lesson


def posterior(t):                                       # NEW: Day 140's version, unchanged
    b, ab, ab1 = BETAS[t - 1], ABAR[t], ABAR[t - 1]                                    # this step's beta, alpha-bar after it and before it
    c0 = np.sqrt(ab1) * b / (1 - ab)                                                  # how much of the clean x_0 to put back
    ct = np.sqrt(1 - b) * (1 - ab1) / (1 - ab)                                         # how much of the current x_t to keep
    return c0, ct, (1 - ab1) / (1 - ab) * b                                           # and the variance left: beta-tilde_t


def exact_x0(xt, t, radius=2.0, sd=0.05, k=8):          # NEW: Day 140's version, unchanged
    C = np.c_[np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)] * radius  # the true centres
    a, v = np.sqrt(ABAR[t]), ABAR[t] * sd ** 2 + 1 - ABAR[t]                          # given mode j: x_t ~ N(a C_j, v I)
    L = -((xt[:, None, :] - a * C[None]) ** 2).sum(-1) / (2 * v)                       # log of each mode's likelihood, up to a constant
    w = np.exp(L - L.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)              # Bayes' rule: p(mode j | x_t), all priors equal
    m = C[None] + (a * sd ** 2 / v) * (xt[:, None, :] - a * C[None])                   # E[x_0 | x_t, mode j]: a Gaussian conditional mean
    return (w[..., None] * m).sum(1)                                                  # averaged over the modes, weighted by how likely each is


def eps_from_x0(xt, x0, t):                             # NEW: Day 140's version, unchanged
    return (xt - np.sqrt(ABAR[t]) * x0) / np.sqrt(1 - ABAR[t])


def reverse_sample(eps_fn, n, rng, var="posterior", d=2):  # NEW: Day 140's version, unchanged
    x = rng.standard_normal((n, d))                                                   # x_1000: pure noise
    for t in range(len(BETAS), 0, -1):
        b = BETAS[t - 1]; e = eps_fn(x, t)                                            # the denoiser's guess of the noise inside x_t
        x = (x - b / np.sqrt(1 - ABAR[t]) * e) / np.sqrt(1 - b)                       # the mean of x_{t-1} (Ho et al. 2020, eq. 11)
        if t > 1:                                                                     # no noise is added on the very last step
            s2 = posterior(t)[2] if var == "posterior" else (b if var == "beta" else 0.0)  # how much fresh noise to add back
            x = x + np.sqrt(s2) * rng.standard_normal(x.shape)
    return x


def time_embed(t, n=16):                                # NEW: Day 140's version, unchanged
    f = 10000.0 ** (-np.arange(n) / n); a = np.asarray(t, np.float64)[:, None] * f[None]  # from 1 down to 1/10,000 radians a step
    return np.concatenate([np.sin(a), np.cos(a)], 1)


def eps_net_fn(net):                                    # NEW: Day 140's version, unchanged
    acts = ["relu"] * (len(net) - 1) + ["linear"]
    return lambda x, t: d126.forward(net, np.concatenate([x, time_embed(np.full(len(x), t))], 1).astype(np.float32), acts)[-1].astype(np.float64)


def train_denoiser(steps, seed=0, batch=512, lr=1e-3, every=2000, Xev=None):  # NEW: Day 140's version, unchanged
    net = d126.init_net([34, 256, 256, 2], seed); opt = d126.Adam(net, lr); acts = ["relu", "relu", "linear"]  # 2 + 32 inputs
    rng = np.random.RandomState(seed); log = []
    for s in range(1, steps + 1):
        x0 = d139.ring(batch, rng).astype(np.float64); t = rng.randint(1, len(BETAS) + 1, batch)  # clean points, and a random step for each
        e = rng.standard_normal(x0.shape); xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e  # Day 139's shortcut
        A = d126.forward(net, np.concatenate([xt, time_embed(t)], 1).astype(np.float32), acts)  # the network's guess of e
        d = (A[-1] - e) * 2 / e.size; grads, _ = d126.backward(net, A, acts, d.astype(np.float32))  # d mean squared error / d output
        opt.step(net, grads)
        if s % every == 0 and Xev is not None:
            log.append((s, eval_eps(eps_net_fn(net), *Xev)))                         # the error on fixed evaluation points
    return net, log


def eval_eps(eps_fn, x0, t, e):                         # NEW: Day 140's version, unchanged
    xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e; total = 0.0  # Day 139's shortcut, every triple at its own t
    for s in np.unique(t):                                                            # one call per t level (eps_fn takes one t at a time)
        k = t == s; total += ((eps_fn(xt[k], s) - e[k]) ** 2).sum()
    return float(total / e.size)


def mode_spread(S, radius=2.0, k=8):                     # NEW: Day 140's version, unchanged
    C = np.c_[np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)] * radius
    near = ((S[:, None] - C[None]) ** 2).sum(-1).argmin(1)                            # each sample's nearest centre
    return float((S - C[near]).std(0).mean())                                         # the standard deviation of the offsets, both coordinates averaged


c0, ct, var = posterior(1); print(f"posterior at t = 1: x_0 weight {c0:.4f}, x_t weight {ct:.4f}, variance {var:.6f} (step 1 starts from x_0 itself)")
c0, ct, var = posterior(1000); print(f"posterior at t = 1,000: x_0 weight {c0:.6f}, x_t weight {ct:.4f}, variance {var:.6f}")
rng = np.random.RandomState(0); x0 = d139.ring(2000, rng).astype(np.float64); xt = d139.q_sample(x0, 20, ABAR, rng)
g = exact_x0(xt, 20); print(f"one exact step from t = 20: distance to the true x_0 {np.sqrt(((g - x0) ** 2).sum(1)).mean():.4f}; modes {d139.mode_stats(g)[0]}")
S = reverse_sample(lambda x, t: eps_from_x0(x, exact_x0(x, t), t), 2000, np.random.RandomState(1), "beta")
m, gs, c = d139.mode_stats(S); print(f"reverse process, exact denoiser, noise beta_t: {m} modes, good share {gs:.3f}, spread {mode_spread(S):.4f} (real 0.05)")
net, _ = train_denoiser(3000); S = reverse_sample(eps_net_fn(net), 2000, np.random.RandomState(1), "beta")
m, gs, c = d139.mode_stats(S); print(f"after 3,000 training steps, the learned denoiser: {m} modes, good share {gs:.3f}, spread {mode_spread(S):.4f}")
