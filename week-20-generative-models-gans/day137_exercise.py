"""
Day 137 Exercise -- The Label, Given to Both Players

The fifth of Week 20's parts for Day 139's build.

    onehot(y)                           labels 0-9 as one-hot rows
    g_input(Z, y)                       the generator's input: noise and the one-hot label side by side
    d_input(X, y)                       the discriminator's input: the image and ten label planes (11 channels)
    cgan_players(seed, k, c, d_label)   Day 136's players with room for the label
    cgan_step(G, D, oG, oD, X, y, ...)  one round of the conditional game
    spread(F)                           average distance between pairs of images

Built on Day 136's players and Day 134's losses, imported unchanged. Checked
by hand: the shapes, a gradient through the joined input, and 300 rounds on
MNIST, after which the discriminator is asked about real digits with their
true and with wrong labels.
"""
import os                                              # os: the paths to this week's and earlier folders
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
sys.path.insert(0, HERE)                                # Days 134 and 136's parts
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # Day 126's and Day 119's parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day134_exercise as d134                      # Day 134: AdamGAN, d_loss, g_loss
    import day136_exercise as d136                      # Day 136: dcgan_players
    import day119_exercise as d119                      # Day 119: sqdist
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)


def onehot(y, n=10):                                    # NEW: Day 137's version, unchanged
    E = np.zeros((len(y), n), np.float32); E[np.arange(len(y)), y] = 1                # fancy indexing: row i, column y[i]
    return E


def g_input(Z, y):                                      # NEW: Day 137's version, unchanged
    return np.concatenate([Z, onehot(y)], axis=1)


def d_input(X, y):                                      # NEW: Day 137's version, unchanged
    P = np.zeros((len(X), 10, 784), np.float32); P[np.arange(len(X)), y] = 1          # plane y[i] all ones, the other nine all zeros
    return np.concatenate([X, P.reshape(len(X), -1)], axis=1)                          # reshaped by D's first layer to 11 x 28 x 28


def cgan_players(seed, k=64, c=32, d_label=True):       # NEW: Day 137's version, unchanged
    G = d136.dcgan_players(seed, k=k + 10, c=c)[0]                                    # the generator reads 64 + 10 numbers
    D = d136.dcgan_players(seed, c=c, ch=11 if d_label else 1)[1]                     # the discriminator reads 11 channels, or only the image
    return G, D


def cgan_step(G, D, oG, oD, X, y, rng, k=64, d_label=True):  # NEW: Day 137's version, unchanged
    n = len(X); join = (lambda I, l: d_input(I, l)) if d_label else (lambda I, l: I)  # without d_label, D sees the image only
    yf = rng.randint(0, 10, n); F = G.forward(g_input(rng.standard_normal((n, k)).astype(np.float32), yf))  # 1. fakes of random digits
    Af = D.forward(join(F, yf)); Ar = D.forward(join(X, y))                           # D's verdicts: fakes with their asked-for labels, reals with their true ones
    ld, gr, gf = d134.d_loss(Ar, Af)                                                  # Day 134's loss and its two gradients
    g1, _ = D.backward(gr); D.forward(join(F, yf)); g2, _ = D.backward(gf)            # the real half, then the fake half (re-run to refill the caches)
    oD.step(D.net, [(a[0] + b[0], a[1] + b[1]) for a, b in zip(g1, g2)])              # one Adam step on their sum
    yf = rng.randint(0, 10, n); F = G.forward(g_input(rng.standard_normal((n, k)).astype(np.float32), yf))  # 2. the generator's turn
    lf = D.forward(join(F, yf)); lg, gl = d134.g_loss(lf)                             # the non-saturating loss
    _, dJ = D.backward(gl); gG, _ = G.backward(dJ[:, :784])                            # keep only the image's part of D's input gradient
    oG.step(G.net, gG)                                                                # only the generator moves
    return ld, lg, float(d134.d126.sigmoid(Ar).mean()), float(d134.d126.sigmoid(lf).mean())  # losses and D's average verdicts


def spread(F):                                          # NEW: Day 137's version, unchanged
    D2 = d119.sqdist(F, F); return float(np.sqrt(np.maximum(D2[np.triu_indices(len(F), 1)], 0)).mean())


y = np.array([3, 7]); print(f"onehot([3, 7]): the 1s sit in columns {onehot(y).argmax(1).tolist()}; row sums {onehot(y).sum(1).tolist()}")
print(f"g_input: {g_input(np.zeros((2, 64), np.float32), y).shape}; d_input: {d_input(np.zeros((2, 784), np.float32), y).shape} = 11 x 784")
G, D = cgan_players(0, k=8, c=4)
for N in (G, D):
    for L in N.layers:
        if L.params():
            L.W, L.b = L.W.astype(np.float64), L.b.astype(np.float64)
    N.net = [pr for L in N.layers for pr in L.params()]
rng = np.random.RandomState(1); Z = rng.randn(6, 8); yy = rng.randint(0, 10, 6)
loss = lambda: d134.g_loss(D.forward(d_input(G.forward(g_input(Z, yy)), yy)))[0]
_, gl = d134.g_loss(D.forward(d_input(G.forward(g_input(Z, yy)), yy))); _, dJ = D.backward(gl); gG, _ = G.backward(dJ[:, :784])
W = G.net[0][0]; r = 8 + yy[0]; old = W[r, 5]; W[r, 5] = old + 1e-5; up = loss(); W[r, 5] = old - 1e-5; dn = loss(); W[r, 5] = old
print(f"d loss / d (weight from label {yy[0]}'s input to unit 5): backward {gG[0][0][r, 5]:.10f}; finite difference {(up - dn) / 2e-5:.10f}")
X, Y = d126m.mnist(); Xs, ys = X[:5000], Y[:5000]; G, D = cgan_players(0)
oG, oD = d134.AdamGAN(G.net, 2e-4, 0.5), d134.AdamGAN(D.net, 2e-4, 0.5); rng = np.random.RandomState(0)
for s in range(300):
    b = rng.randint(0, 5000, 128); cgan_step(G, D, oG, oD, Xs[b], ys[b], rng)
Xr, yr = X[60000:61000], Y[60000:61000]
print(f"after 300 rounds, D on 1,000 real test digits: {d134.d126.sigmoid(D.forward(d_input(Xr, yr))).mean():.3f} with the true label, "
      f"{d134.d126.sigmoid(D.forward(d_input(Xr, (yr + 1) % 10))).mean():.3f} with the label plus 1")
print(f"spread of 200 real 1s {spread(X[Y == 1][:200]):.2f}; of 200 real 8s {spread(X[Y == 8][:200]):.2f}")
