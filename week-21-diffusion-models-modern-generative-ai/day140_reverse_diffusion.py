"""
Day 140: Reverse Diffusion From Scratch -- a Real Denoising Step

Day 139 destroyed data on purpose: 1,000 small steps of Gaussian noise turn
any image, or any point of Day 135's ring, into pure N(0, I) noise. A
DIFFUSION MODEL generates by running that process backwards: start from pure
noise and take 1,000 small DENOISING steps. Each step needs one thing: a guess
of the noise inside the current x_t (or, equivalently, of the clean x_0).

On the ring that guess can be computed EXACTLY, because the data is a known
mixture of eight Gaussians. So today separates the two questions every
diffusion model mixes together:
  * is the reverse process itself right? (run it with the exact denoiser)
  * how good is a learned denoiser? (train one, and compare it with the
    exact one, whose error is the lowest any network could reach)

Everything is written from scratch in NumPy, on Day 126's dense networks and
Adam, Day 135's ring and mode count, and Day 139's noise schedule:
  * the posterior q(x_{t-1} | x_t, x_0) of the forward process
  * the exact denoiser E[x_0 | x_t] for a mixture of Gaussians
  * the reverse (ancestral) sampling step of Ho et al. (2020)
  * a small network eps_theta(x_t, t), trained on the noise-prediction loss

  1. By hand: one reverse step's formula, checked against simulation.
  2. One denoising step: what E[x_0 | x_t] looks like at each t.
  3. The whole reverse process with the exact denoiser: eight modes back?
  4. A learned denoiser: its error against the lowest possible error.
  5. Sampling with the learned denoiser, and the step's noise choices.

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
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day139_gan_build_and_forward_diffusion as d139  # Day 139: schedule, q_step, q_sample, and Day 135's ring and mode_stats


# =============================================================================
# THE REVERSE PROCESS -- NEW
# =============================================================================
BETAS, ABAR = d139.schedule("linear")                   # Day 139's linear schedule: 1,000 steps, beta from 0.0001 to 0.02


def posterior(t):                                       # NEW: q(x_{t-1} | x_t, x_0) = N(c0 x_0 + ct x_t, var), for t >= 1
    b, ab, ab1 = BETAS[t - 1], ABAR[t], ABAR[t - 1]                                    # this step's beta, alpha-bar after it and before it
    c0 = np.sqrt(ab1) * b / (1 - ab)                                                  # how much of the clean x_0 to put back
    ct = np.sqrt(1 - b) * (1 - ab1) / (1 - ab)                                         # how much of the current x_t to keep
    return c0, ct, (1 - ab1) / (1 - ab) * b                                           # and the variance left: beta-tilde_t


def exact_x0(xt, t, radius=2.0, sd=0.05, k=8):          # NEW: E[x_0 | x_t] when x_0 is the ring's mixture of k Gaussians
    C = np.c_[np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)] * radius  # the true centres
    a, v = np.sqrt(ABAR[t]), ABAR[t] * sd ** 2 + 1 - ABAR[t]                          # given mode j: x_t ~ N(a C_j, v I)
    L = -((xt[:, None, :] - a * C[None]) ** 2).sum(-1) / (2 * v)                       # log of each mode's likelihood, up to a constant
    w = np.exp(L - L.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)              # Bayes' rule: p(mode j | x_t), all priors equal
    m = C[None] + (a * sd ** 2 / v) * (xt[:, None, :] - a * C[None])                   # E[x_0 | x_t, mode j]: a Gaussian conditional mean
    return (w[..., None] * m).sum(1)                                                  # averaged over the modes, weighted by how likely each is


def eps_from_x0(xt, x0, t):                             # NEW: the noise implied by a guess of x_0: x_t = sqrt(abar) x_0 + sqrt(1 - abar) eps
    return (xt - np.sqrt(ABAR[t]) * x0) / np.sqrt(1 - ABAR[t])


def reverse_sample(eps_fn, n, rng, var="posterior", d=2):  # NEW: start from N(0, I) and take the 1,000 steps back
    x = rng.standard_normal((n, d))                                                   # x_1000: pure noise
    for t in range(len(BETAS), 0, -1):
        b = BETAS[t - 1]; e = eps_fn(x, t)                                            # the denoiser's guess of the noise inside x_t
        x = (x - b / np.sqrt(1 - ABAR[t]) * e) / np.sqrt(1 - b)                       # the mean of x_{t-1} (Ho et al. 2020, eq. 11)
        if t > 1:                                                                     # no noise is added on the very last step
            s2 = posterior(t)[2] if var == "posterior" else (b if var == "beta" else 0.0)  # how much fresh noise to add back
            x = x + np.sqrt(s2) * rng.standard_normal(x.shape)
    return x


def time_embed(t, n=16):                                # NEW: t as 2n numbers, sines and cosines of geometric frequencies
    f = 10000.0 ** (-np.arange(n) / n); a = np.asarray(t, np.float64)[:, None] * f[None]  # from 1 down to 1/10,000 radians a step
    return np.concatenate([np.sin(a), np.cos(a)], 1)


def eps_net_fn(net):                                    # NEW: a trained network as an eps_fn for reverse_sample
    acts = ["relu"] * (len(net) - 1) + ["linear"]
    return lambda x, t: d126.forward(net, np.concatenate([x, time_embed(np.full(len(x), t))], 1).astype(np.float32), acts)[-1].astype(np.float64)


def train_denoiser(steps, seed=0, batch=512, lr=1e-3, every=2000, Xev=None):  # NEW: learn eps_theta(x_t, t) with the noise-prediction loss
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


def eval_eps(eps_fn, x0, t, e):                         # NEW: mean squared error of an eps_fn on fixed (x_0, t, eps) triples
    xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e; total = 0.0  # Day 139's shortcut, every triple at its own t
    for s in np.unique(t):                                                            # one call per t level (eps_fn takes one t at a time)
        k = t == s; total += ((eps_fn(xt[k], s) - e[k]) ** 2).sum()
    return float(total / e.size)


def mode_spread(S, radius=2.0, k=8):                     # NEW: how spread out samples are around their nearest centre (the ring's own sd is 0.05)
    C = np.c_[np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)] * radius
    near = ((S[:, None] - C[None]) ** 2).sum(-1).argmin(1)                            # each sample's nearest centre
    return float((S - C[near]).std(0).mean())                                         # the standard deviation of the offsets, both coordinates averaged


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- one reverse step's formula, against simulation")
    print("=" * 70)
    rng = np.random.RandomState(0); x0 = 1.0; n = 1000000                              # one clean 1-D value, a million forward runs
    print(f"x_0 = {x0}; a million runs of the forward process to step t - 1, then one more step; x_(t-1) regressed on x_t")
    print(f"{'t':>6} {'c0 x_0':>9} {'simulated':>10} {'ct':>9} {'simulated':>10} {'beta-tilde':>11} {'simulated':>10}")
    for t in (2, 50, 300, 900):
        xp = d139.q_sample(np.full(n, x0), t - 1, ABAR, rng); xt = d139.q_step(xp, BETAS[t - 1], rng)  # pairs (x_(t-1), x_t)
        A = np.c_[np.ones(n), xt]; coef, *_ = np.linalg.lstsq(A, xp, rcond=None); res = (xp - A @ coef).var()  # x_(t-1) = intercept + slope x_t + leftover
        c0, ct, var = posterior(t)
        print(f"{t:6d} {c0 * x0:9.5f} {coef[0]:10.5f} {ct:9.5f} {coef[1]:10.5f} {var:11.6f} {res:10.6f}")
    print("given x_0, x_(t-1) and x_t are jointly Gaussian, so a straight-line fit recovers the posterior exactly (up to sampling error)")
    print("Actual output from this lesson's run.")


def demo_one_step():
    print("\n" + "=" * 70)
    print("2. ONE DENOISING STEP -- the exact E[x_0 | x_t] on Day 135's ring, at each t")
    print("=" * 70)
    rng = np.random.RandomState(1); x0 = d139.ring(2000, rng).astype(np.float64); rows = []
    print("2,000 ring points noised to step t, then the exact guess of x_0 from x_t alone; Day 135's mode_stats on the guesses")
    print(f"{'t':>6} {'noise sd':>9} {'modes':>6} {'good share':>11} {'guess radius':>13} {'|guess - x_0|':>14} {'right mode':>11}")
    C = np.c_[np.cos(2 * np.pi * np.arange(8) / 8), np.sin(2 * np.pi * np.arange(8) / 8)] * 2
    true_mode = np.argmin(((x0[:, None] - C[None]) ** 2).sum(-1), 1)
    for t in (1, 10, 50, 100, 200, 300, 500, 700, 1000):
        xt = d139.q_sample(x0, t, ABAR, rng); g = exact_x0(xt, t); m, gs, _ = d139.mode_stats(g)
        right = np.mean(np.argmin(((g[:, None] - C[None]) ** 2).sum(-1), 1) == true_mode)
        rows.append((t, g)); print(f"{t:6d} {np.sqrt(1 - ABAR[t]):9.4f} {m:6d} {gs:11.3f} {np.sqrt((g ** 2).sum(1)).mean():13.4f} "
                                   f"{np.sqrt(((g - x0) ** 2).sum(1)).mean():14.4f} {right:11.3f}")
    print("Actual output from this lesson's run.")
    return rows


def demo_exact():
    print("\n" + "=" * 70)
    print("3. THE WHOLE REVERSE PROCESS WITH THE EXACT DENOISER -- are all eight modes recovered?")
    print("=" * 70)
    R = d139.ring(2000, np.random.RandomState(5)); real = d139.mode_stats(R)
    print(f"real ring, 2,000 points: {real[0]} modes, good share {real[1]:.3f}, spread around its centre {mode_spread(R):.4f}, "
          f"counts {' '.join(str(c) for c in real[2])}")
    print("2,000 samples, 1,000 steps from N(0, I), the exact eps at every step, fresh noise of variance beta-tilde_t:")
    out = {}
    for seed in (0, 1, 2):
        t0 = time.perf_counter(); S = reverse_sample(lambda x, t: eps_from_x0(x, exact_x0(x, t), t), 2000, np.random.RandomState(seed))
        m, g, c = d139.mode_stats(S); out[seed] = S
        print(f"   seed {seed}: {m} modes, good share {g:.3f}, spread {mode_spread(S):.4f}, counts {' '.join(str(v) for v in c)} ({time.perf_counter() - t0:.1f} s)")
    print("Actual output from this lesson's run.")
    return out, real


def demo_learned(steps=30000):
    print("\n" + "=" * 70)
    print("4. A LEARNED DENOISER -- its error against the lowest possible error")
    print("=" * 70)
    rng = np.random.RandomState(7); x0 = d139.ring(20000, rng).astype(np.float64)     # fixed evaluation points
    t = np.repeat(np.array([1, 10, 50, 100, 200, 300, 500, 700, 1000]), 20000 // 9 + 1)[:20000]; e = rng.standard_normal(x0.shape)
    exact = lambda x, s: eps_from_x0(x, exact_x0(x, s), s)                           # the best possible eps guess: the exact posterior mean
    floor = eval_eps(exact, x0, t, e)
    print(f"network: [x_t (2 numbers), time embedding (32)] -> 256 -> 256 -> 2 (ReLU), Day 126's init_net, forward, backward and Adam;")
    print(f"loss: mean squared error between the true and the guessed eps; batches of 512 fresh ring points, t uniform on 1-1,000")
    print(f"evaluation: 20,000 fixed (x_0, t, eps) triples, t in 9 levels; the exact denoiser's error there is {floor:.4f}, the floor")
    t0 = time.perf_counter(); net, log = train_denoiser(steps, Xev=(x0, t, e))
    print(f"{'step':>8} {'error':>8} {'above the floor':>16}")
    for s, v in log:
        if s in (2000, 4000, 8000, 12000, 16000, 20000, 25000, 30000):
            print(f"{s:8,d} {v:8.4f} {v - floor:16.4f}")
    print(f"({time.perf_counter() - t0:.0f} s)")
    learned = eps_net_fn(net)
    print("by t: error of the learned denoiser, and of the exact one (the floor), on the same triples")
    print(f"{'t':>6} {'learned':>9} {'exact':>9}")
    for s in np.unique(t):
        xt = np.sqrt(ABAR[s]) * x0[t == s] + np.sqrt(1 - ABAR[s]) * e[t == s]
        print(f"{s:6d} {((learned(xt, s) - e[t == s]) ** 2).mean():9.4f} {((exact(xt, s) - e[t == s]) ** 2).mean():9.4f}")
    print("Actual output from this lesson's run.")
    return net, log, floor


def demo_sampling(net):
    print("\n" + "=" * 70)
    print("5. SAMPLING WITH THE LEARNED DENOISER, and how much fresh noise each step adds")
    print("=" * 70)
    learned = eps_net_fn(net); exact = lambda x, s: eps_from_x0(x, exact_x0(x, s), s); out = {}
    print("2,000 samples each, seeds 0-2; fresh noise per step: beta-tilde_t (the posterior's own), beta_t (a little more), or none")
    print(f"{'denoiser':>9} {'noise':>10} {'seed':>5} {'modes':>6} {'good share':>11} {'spread':>7}   counts (the real ring's spread is 0.05)")
    for name, fn in (("learned", learned), ("exact", exact)):
        for var in ("posterior", "beta", "none"):
            for seed in (0, 1, 2):
                S = reverse_sample(fn, 2000, np.random.RandomState(seed), var); m, g, c = d139.mode_stats(S); out[(name, var, seed)] = S
                print(f"{name:>9} {var:>10} {seed:5d} {m:6d} {g:11.3f} {mode_spread(S):7.4f}   {' '.join(str(v) for v in c)}")
    print("Actual output from this lesson's run.")
    return out


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(one, exact_runs, real, log, floor, samp, out):  # eight panels
    fig = plt.figure(figsize=(17, 9)); ring = d139.ring(2000, np.random.RandomState(5))
    for j, (t, g) in enumerate([r for r in one if r[0] in (50, 300, 700, 1000)]):     # one denoising step at four noise levels
        ax = fig.add_subplot(2, 4, 1 + j); ax.scatter(ring[:, 0], ring[:, 1], s=2, c="lightgray", label="real")
        ax.scatter(g[:, 0], g[:, 1], s=2, c="#4C72B0", alpha=0.5, label="E[x_0 | x_t]"); ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal"); ax.set_title(f"one exact step from t = {t}", fontsize=10)
        if j == 0:
            ax.legend(fontsize=7, loc="center", markerscale=4)
    for j, (title, S) in enumerate((("reverse process, exact denoiser, seed 0", exact_runs[0]),
                                    ("reverse process, learned denoiser, seed 0", samp[("learned", "posterior", 0)]),
                                    ("learned denoiser, no fresh noise, seed 0", samp[("learned", "none", 0)]))):
        ax = fig.add_subplot(2, 4, 5 + j); m, g, _ = d139.mode_stats(S)
        ax.scatter(ring[:, 0], ring[:, 1], s=2, c="lightgray"); ax.scatter(S[:, 0], S[:, 1], s=2, c="#C44E52", alpha=0.5)
        ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8); ax.set_aspect("equal"); ax.set_title(f"{title}\n{m} modes, good share {g:.2f}", fontsize=9)
    ax = fig.add_subplot(2, 4, 8); ax.plot([s for s, _ in log], [v for _, v in log], marker="o", label="learned denoiser")
    ax.axhline(floor, color="gray", ls="--", label=f"exact denoiser (the floor), {floor:.4f}")
    ax.set_xlabel("training step"); ax.set_ylabel("mean squared error of eps"); ax.set_title("learning to denoise", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # forward and reverse
    fig, ax = plt.subplots(figsize=(13, 3.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, c="black"):                                             # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3, color=c))
    xs = (1.2, 4.4, 8.6, 11.8)
    for x, t in zip(xs, ("x_0\nclean data", "x_(t-1)", "x_t", "x_1000\npure noise N(0, I)")):
        box(x, 2.3, 2.0, 1.0, t, "#E2F0D9" if x < 2 else "#EEEEEE")
    arrow(2.3, 2.75, 3.3, 2.75); arrow(5.5, 2.75, 7.5, 2.75); arrow(9.7, 2.75, 10.7, 2.75)
    ax.text(6.5, 3.05, "forward (Day 139): add noise, beta_t", ha="center", fontsize=9)
    arrow(10.7, 1.85, 9.7, 1.85, "#C44E52"); arrow(7.5, 1.85, 5.5, 1.85, "#C44E52"); arrow(3.3, 1.85, 2.3, 1.85, "#C44E52")
    ax.text(6.5, 1.35, "reverse (today): x_(t-1) = (x_t - beta_t / sqrt(1 - abar_t) eps_hat) / sqrt(1 - beta_t) + sigma_t z", ha="center", fontsize=9, color="#8B1A1A")
    box(6.5, 0.45, 7.6, 0.6, "eps_hat: the exact E[eps | x_t] on the ring, or a network eps_theta(x_t, t) trained on noise prediction", "#FFF2CC", 8.5)
    ax.set_title("Day 140: generating by running the noising process backwards", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "reverse_diffusion_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("The ring has two numbers per point and a denoiser that can be written down. Day 141 trains a tiny diffusion model on")
    print("images, where no exact denoiser exists, and generates new ones.")


def main():                                             # runs every section in order
    demo_hand()                                                                       # 1. one reverse step, checked
    one = demo_one_step()                                                             # 2. the exact denoiser's single guess
    exact_runs, real = demo_exact()                                                   # 3. the whole reverse process, exact
    net, log, floor = demo_learned()                                                  # 4. a learned denoiser
    samp = demo_sampling(net)                                                         # 5. sampling with it
    plot_results(one, exact_runs, real, log, floor, samp, os.path.join(HERE, "reverse_diffusion_results.png"))
    draw_diagram()                                                                    # the two directions
    print("\nsaved reverse_diffusion_results.png\nsaved reverse_diffusion_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
