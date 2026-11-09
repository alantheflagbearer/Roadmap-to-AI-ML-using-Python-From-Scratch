"""
Day 135 Exercise -- Measuring Mode Collapse

The third of Week 20's parts for Day 139's build.

    ring(n, rng, k, radius, sd)          k tight Gaussians on a circle: data whose modes are known
    mode_stats(F, k, radius, sd)         modes covered (at least 1% of samples), good share, good samples per mode
    train_ring(seed, steps, kind, ...)   a GAN on the ring (Day 134's parts), watched every `every` steps

Checked by hand: real ring samples cover all 8 modes, and the share within
3 spreads of a centre matches 1 - e^(-4.5) for a 2-D Gaussian; samples from
3 modes count as 3; and a short GAN run is watched as it trains.
"""
import os                                              # os: the paths to this week's and Day 126's folders
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
sys.path.insert(0, HERE)                                # Day 134's parts
sys.path.insert(0, os.path.join(HERE, "..", "week-19-unsupervised-learning-ii"))  # Day 126's parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, Adam
    import day134_exercise as d134                      # Day 134: gan_step, generate


def ring(n, rng, k=8, radius=2.0, sd=0.05):             # NEW: Day 135's version, unchanged
    a = 2 * np.pi * rng.randint(0, k, n) / k                                          # which mode, as an angle
    return (np.c_[np.cos(a), np.sin(a)] * radius + sd * rng.standard_normal((n, 2))).astype(np.float32)


def mode_stats(F, k=8, radius=2.0, sd=0.05):            # NEW: Day 135's version, unchanged
    C = np.c_[np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)] * radius  # the true centres
    D = np.sqrt(((F[:, None, :] - C[None]) ** 2).sum(-1)); near = D.argmin(1)                    # each sample's nearest centre
    good = D[np.arange(len(F)), near] < 3 * sd                                        # within 3 standard deviations: a real-looking sample
    counts = np.bincount(near[good], minlength=k)                                     # good samples per mode
    return int((counts >= 0.01 * len(F)).sum()), float(good.mean()), counts           # a mode counts if it gets 1% of the samples


def train_ring(seed, steps=6000, kind="non-saturating", lr_g=1e-3, lr_d=1e-3, d_steps=1, every=250, k=8, z=2, head=0):  # NEW: Day 135's version, unchanged
    rng = np.random.RandomState(seed); G = d126.init_net([z, 128, 128, 2], seed); D = d126.init_net([2, 128, 128, 1], seed + 100)
    oG, oD = d126.Adam(G, lr_g), d126.Adam(D, lr_d); track = []                         # (step, modes, good share, counts, D(real), D(fake))
    ev = np.random.RandomState(999).standard_normal((2000, z)).astype(np.float32)      # the same noise every time we look
    for t in range(1, steps + 1):
        oG.lr = 0.0 if t <= head else lr_g                                            # a head start: only the discriminator learns at first
        _, _, dr, df = d134.gan_step(G, D, oG, oD, ring(256, rng, k), rng, z, out="linear", kind=kind, d_steps=d_steps)
        if t % every == 0:                                                            # look during training, not only at the end
            m, g, c = mode_stats(d134.generate(G, ev, "linear")[0], k); track.append((t, m, g, c, dr, df))
    return G, track


rng = np.random.RandomState(0); R = ring(100000, rng)
m, g, c = mode_stats(R)
print(f"100,000 real samples: {m} of 8 modes covered; good share {g:.4f}; by hand 1 - e^(-4.5) = {1 - np.exp(-4.5):.4f}")
print(f"radius of the real samples: average {np.sqrt((R ** 2).sum(1)).mean():.4f} (centres at 2.0)")
a = 2 * np.pi * rng.randint(0, 3, 3000) / 8; P = (np.c_[np.cos(a), np.sin(a)] * 2 + 0.05 * rng.standard_normal((3000, 2))).astype(np.float32)
print(f"3,000 samples from modes 0, 1 and 2 only: {mode_stats(P)[0]} modes covered; counts {mode_stats(P)[2].tolist()}")
G, tr = train_ring(0, steps=2000, every=500)
print("a 2,000-step GAN, watched: " + "; ".join(f"step {t}: {m} modes, good {g:.2f}" for t, m, g, _, _, _ in tr))
