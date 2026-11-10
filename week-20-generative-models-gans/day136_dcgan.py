"""
Day 136: A DCGAN-Style Convolutional GAN -- Real Images, on MNIST and the Junction Dataset

Day 134's GAN used dense layers: every generated pixel had its own weights,
and nothing in the network knew that neighbouring pixels belong together.
A DCGAN (deep convolutional GAN, Radford et al. 2015) builds both players
from convolutions. The discriminator shrinks the image with STRIDED
convolutions (stride 2 halves the height and width); the generator grows
an image from noise with TRANSPOSED convolutions (stride 2 doubles them),
with batch normalisation between layers.

Everything is written from scratch in NumPy:
  * a strided convolution, vectorised by looping over the kernel offsets
    (Day 53's idea, now with a stride), and its backward pass
  * the transposed convolution: the strided convolution's input-gradient,
    used as a layer of its own
  * batch normalisation (Day 54), a small layer stack, and both players
  * Day 134's losses and AdamGAN (b1 0.5), imported unchanged

Data: MNIST (cached by Day 126), and Week 7's junction dataset (images of
1 to 4 straight arms meeting at a centre), where whether a generated image is
a valid junction can be checked exactly.

  1. By hand: the shapes, the transposed convolution as an adjoint, and
     how its kernel overlaps.
  2. Correctness: the generator's gradient through both convolutional
     players, against finite differences.
  3. MNIST: Day 134's dense GAN against the DCGAN, watched every epoch.
  4. The DCGAN without batch normalisation.
  5. Junctions: can each GAN draw straight arms of the right length?

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
sys.path.insert(0, HERE)                                # this week's parts
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles", "week-07-cnns-i-and-ii"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day119_exercise as d119                      # Day 119: sqdist
    import day116_exercise as d116                      # Day 116: softmax, fit_softmax (the judge of generated digits)
    import day134_exercise as d134                      # Day 134: AdamGAN, d_loss, g_loss
    import day47_lenet5_style_cnn as d47                # Day 47: make_junction_dataset_rgb (Week 7's junction images)


# =============================================================================
# CONVOLUTIONS WITH A STRIDE -- NEW
# =============================================================================
def pad2(X, p):                                         # NEW: zero-pad the two spatial axes of (m, C, H, W)
    return np.pad(X, ((0, 0), (0, 0), (p, p), (p, p))) if p else X                     # batch and channel axes untouched


def im2col(Xp, k, s, oh, ow):                           # NEW: every k x k patch as a row, looping over the k*k kernel offsets
    m, C = Xp.shape[:2]; cols = np.empty((m, oh, ow, C, k, k), Xp.dtype)               # one row per (image, output position)
    for i in range(k):                                                                # kernel row offset
        for j in range(k):                                                            # kernel column offset
            cols[:, :, :, :, i, j] = Xp[:, :, i:i + s * oh:s, j:j + s * ow:s].transpose(0, 2, 3, 1)  # every position's pixel at this offset, stepping by s
    return cols.reshape(m * oh * ow, C * k * k)                                       # (positions, patch values)


def col2im(dcols, shape, k, s, oh, ow):                 # NEW: the reverse: add every patch entry back onto the pixel it came from
    dX = np.zeros(shape, dcols.dtype); d = dcols.reshape(shape[0], oh, ow, shape[1], k, k)  # the image to fill, and the patches unflattened
    for i in range(k):                                                                # kernel row offset
        for j in range(k):                                                            # kernel column offset
            dX[:, :, i:i + s * oh:s, j:j + s * ow:s] += d[:, :, :, :, i, j].transpose(0, 3, 1, 2)  # overlapping patches add up
    return dX


class Conv:                                             # NEW: a strided convolution, (m, Cin, H, W) -> (m, Cout, H', W')
    def __init__(self, cin, cout, k, s, p, rng):
        self.W = (rng.randn(cout, cin, k, k) * np.sqrt(2.0 / (cin * k * k))).astype(np.float32)  # He scale: fan-in is cin*k*k
        self.b = np.zeros(cout, np.float32); self.k, self.s, self.p = k, s, p           # one bias per output channel

    def params(self):                                                                 # the (W, b) pairs Day 134's AdamGAN updates
        return [(self.W, self.b)]

    def forward(self, X):
        Xp = pad2(X, self.p); H, W = Xp.shape[2:]                                     # pad, then the padded size
        oh, ow = (H - self.k) // self.s + 1, (W - self.k) // self.s + 1               # output size: how many stride-s steps fit
        cols = im2col(Xp, self.k, self.s, oh, ow); self.cache = (X.shape, Xp.shape, cols, oh, ow)  # kept for the backward pass
        Z = cols @ self.W.reshape(len(self.W), -1).T + self.b                         # one matrix product does every patch at once
        return Z.reshape(len(X), oh, ow, -1).transpose(0, 3, 1, 2)                    # back to (m, Cout, oh, ow)

    def backward(self, dZ):                                                           # dZ arrives already averaged over the batch
        xs, ps, cols, oh, ow = self.cache; F = len(self.W)
        dZm = dZ.transpose(0, 2, 3, 1).reshape(-1, F)                                 # (positions, Cout), matching cols' rows
        self.grads = [((dZm.T @ cols).reshape(self.W.shape), dZm.sum(0))]             # dW and db
        dX = col2im(dZm @ self.W.reshape(F, -1), ps, self.k, self.s, oh, ow)          # d loss / d (padded) input
        return dX[:, :, self.p:self.p + xs[2], self.p:self.p + xs[3]] if self.p else dX  # drop the padding


class ConvT:                                            # NEW: a transposed convolution: a Conv's input-gradient, used as a layer
    def __init__(self, cin, cout, k, s, p, rng):
        self.W = (rng.randn(cin, cout, k, k) * np.sqrt(2.0 / (cin * k * k / (s * s)))).astype(np.float32)  # each output pixel hears about cin*k*k/s^2 inputs
        self.b = np.zeros(cout, np.float32); self.k, self.s, self.p = k, s, p

    def params(self):
        return [(self.W, self.b)]

    def forward(self, X):
        m, C, H, W = X.shape; k, s, p = self.k, self.s, self.p
        Ho, Wo = (H - 1) * s + k, (W - 1) * s + k                                     # size before cropping the padding
        Xm = X.transpose(0, 2, 3, 1).reshape(-1, C); self.cache = (Xm, H, W)          # one row per input pixel
        Y = col2im(Xm @ self.W.reshape(C, -1), (m, self.W.shape[1], Ho, Wo), k, s, H, W)  # each input pixel paints a k x k patch, stepping by s
        return Y[:, :, p:Ho - p, p:Wo - p] + self.b[None, :, None, None]              # crop p from each side: (H - 1)s - 2p + k

    def backward(self, dY):
        Xm, H, W = self.cache; C = len(self.W)
        cols = im2col(pad2(dY, self.p), self.k, self.s, H, W)                         # the patch each input pixel painted
        self.grads = [((Xm.T @ cols).reshape(self.W.shape), dY.sum((0, 2, 3)))]       # dW and db
        return (cols @ self.W.reshape(C, -1).T).reshape(len(dY), H, W, C).transpose(0, 3, 1, 2)  # a strided convolution of dY


class BN:                                               # NEW: batch normalisation (Day 54's idea), per channel
    def __init__(self, c):
        self.W = np.ones(c, np.float32); self.b = np.zeros(c, np.float32)             # scale gamma and shift beta, stored as (W, b)

    def params(self):
        return [(self.W, self.b)]

    def forward(self, X):
        ax = (0, 2, 3) if X.ndim == 4 else (0,); sh = (1, -1, 1, 1) if X.ndim == 4 else (1, -1)  # images: over batch and positions
        mu = X.mean(ax, keepdims=True); inv = 1 / np.sqrt(X.var(ax, keepdims=True) + 1e-5)  # the batch's own mean and spread
        Xh = (X - mu) * inv; self.cache = (Xh, inv, ax, sh)                           # normalised to mean 0, spread 1
        return Xh * self.W.reshape(sh) + self.b.reshape(sh)                           # then scaled and shifted by learned values

    def backward(self, dY):
        Xh, inv, ax, sh = self.cache; self.grads = [((dY * Xh).sum(ax), dY.sum(ax))]   # d gamma, d beta
        dXh = dY * self.W.reshape(sh)                                                 # back through the scale
        return inv * (dXh - dXh.mean(ax, keepdims=True) - Xh * (dXh * Xh).mean(ax, keepdims=True))  # back through mean and spread


class Dense:                                            # NEW: X @ W + b, drawing its weights exactly as Day 126's init_net does
    def __init__(self, a, b, rng):
        self.W = (rng.randn(a, b) * np.sqrt(2.0 / a)).astype(np.float32); self.b = np.zeros(b, np.float32)

    def params(self):
        return [(self.W, self.b)]

    def forward(self, X):
        self.X = X; return X @ self.W + self.b

    def backward(self, d):
        self.grads = [(self.X.T @ d, d.sum(0))]; return d @ self.W.T


class Act:                                              # NEW: relu, leaky relu (slope 0.2), sigmoid, or a reshape
    def __init__(self, kind):
        self.kind = kind                                                              # a name, or a tuple: the shape to reshape to

    def params(self):                                                                 # nothing to learn
        return []

    def forward(self, X):
        if self.kind == "relu":
            self.g = (X > 0).astype(X.dtype)                                          # 1 where on, 0 where off
        elif self.kind == "lrelu":
            self.g = np.where(X > 0, 1, 0.2).astype(X.dtype)                          # negative inputs keep a fifth
        elif self.kind == "sigmoid":
            self.y = 0.5 * (1 + np.tanh(0.5 * X)); return self.y                      # pixels in [0, 1], without overflow
        else:
            self.shape = X.shape; return X.reshape(self.kind)                         # a reshape: flat <-> (channels, height, width)
        return X * self.g

    def backward(self, d):
        if self.kind == "sigmoid":
            return d * self.y * (1 - self.y)
        if isinstance(self.kind, tuple):
            return d.reshape(self.shape)
        return d * self.g


class Net:                                              # NEW: a stack of layers, with one list of (W, b) pairs for Adam
    def __init__(self, layers):
        self.layers = layers; self.net = [pr for L in layers for pr in L.params()]     # the same arrays, so Adam's in-place steps reach the layers

    def forward(self, X):
        for L in self.layers:
            X = L.forward(X)
        return X

    def backward(self, d):                                                            # returns the parameter gradients and d loss / d input
        for L in reversed(self.layers):
            d = L.backward(d)
        return [g for L in self.layers if L.params() for g in L.grads], d


def dense_players(seed, k=64, d=784, out="sigmoid"):    # NEW: Day 134's two dense players, initialised exactly as d126.init_net does
    rg, rd = np.random.RandomState(seed), np.random.RandomState(seed + 1)
    G = Net([Dense(k, 256, rg), Act("relu"), Dense(256, d, rg)] + ([Act("sigmoid")] if out == "sigmoid" else []))
    return G, Net([Dense(d, 256, rd), Act("relu"), Dense(256, 1, rd)])


def dcgan_players(seed, k=64, c=32, ch=1, bn=True, out="sigmoid"):  # NEW: the convolutional players
    rg, rd = np.random.RandomState(seed), np.random.RandomState(seed + 1)
    norm = (lambda n: [BN(n)]) if bn else (lambda n: [])                              # batch normalisation, or nothing
    G = Net([Dense(k, 2 * c * 49, rg)] + norm(2 * c * 49) + [Act("relu"), Act((-1, 2 * c, 7, 7)),    # noise -> 2c maps of 7 x 7
             ConvT(2 * c, c, 4, 2, 1, rg)] + norm(c) + [Act("relu"),                                 # -> c maps of 14 x 14
            ConvT(c, ch, 4, 2, 1, rg)] + ([Act("sigmoid")] if out == "sigmoid" else []) + [Act((-1, ch * 784))])  # -> ch x 28 x 28, flattened
    D = Net([Act((-1, ch, 28, 28)), Conv(ch, c, 4, 2, 1, rd), Act("lrelu"),                         # 28 x 28 -> c maps of 14 x 14
             Conv(c, 2 * c, 4, 2, 1, rd), Act("lrelu"), Act((-1, 2 * c * 49)), Dense(2 * c * 49, 1, rd)])  # -> 2c of 7 x 7 -> one logit
    return G, D


def n_params(N):                                        # NEW: how many numbers a network learns
    return sum(W.size + b.size for W, b in N.net)


def net_gan_step(G, D, oG, oD, X, rng, k):              # NEW: Day 134's gan_step, for layer stacks (same order of noise draws)
    n = len(X); F = G.forward(rng.standard_normal((n, k)).astype(np.float32))         # 1. a fresh batch of fakes
    Af = D.forward(F); Ar = D.forward(X)                                              # D's verdicts on both (its caches now hold the real batch)
    ld, gr, gf = d134.d_loss(Ar, Af)                                                  # Day 134's loss and its two gradients
    g1, _ = D.backward(gr); D.forward(F); g2, _ = D.backward(gf)                       # the real half, then the fake half (re-run to refill the caches)
    oD.step(D.net, [(a[0] + b[0], a[1] + b[1]) for a, b in zip(g1, g2)])              # one Adam step on their sum
    F = G.forward(rng.standard_normal((n, k)).astype(np.float32))                     # 2. the generator: fool the updated D
    lf = D.forward(F); lg, gl = d134.g_loss(lf)                                       # the non-saturating loss
    _, dF = D.backward(gl); gG, _ = G.backward(dF)                                    # back through D to the image, then through G
    oG.step(G.net, gG)                                                                # only the generator moves
    return ld, lg, float(d134.d126.sigmoid(Ar).mean()), float(d134.d126.sigmoid(lf).mean())  # losses and D's average verdicts


def smoothness(F, ch=1):                                # NEW: average jump between neighbouring pixels (lower = smoother strokes)
    I = F.reshape(len(F), ch, 28, 28)
    return float((np.abs(np.diff(I, axis=2)).mean() + np.abs(np.diff(I, axis=3)).mean()) / 2)


def net_generate(G, Z, batch=500):                      # NEW: noise in, images out, in batches (batch normalisation uses each batch's statistics)
    return np.concatenate([G.forward(Z[i:i + batch]) for i in range(0, len(Z), batch)])


# =============================================================================
# JUNCTIONS: AN EXACT CHECK OF A GENERATED IMAGE -- NEW
# =============================================================================
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))                 # up, down, left, right: Day 43's four arm directions


def junction_templates(arm=8):                          # NEW: every perfect junction: 25 centres x 15 sets of arms, noise-free
    T, cross, info = [], [], []
    for r in range(12, 17):                                                           # the centre can sit up to 2 pixels from the middle
        for c in range(12, 17):
            X = np.zeros((28, 28), np.float32); X[r, c] = 1                           # this centre's cross: centre plus all four arms
            for dr, dc in DIRS:
                X[r + dr * np.arange(1, arm + 1), c + dc * np.arange(1, arm + 1)] = 1
            for m in range(1, 16):                                                    # the 15 non-empty sets of arms, as 4 bits
                t = np.zeros((28, 28), np.float32); t[r, c] = 2.0                     # Day 43 adds 2.0 at the centre...
                for j, (dr, dc) in enumerate(DIRS):
                    if m >> j & 1:
                        t[r + dr * np.arange(1, arm + 1), c + dc * np.arange(1, arm + 1)] += 2.0  # ...and 2.0 along each arm
                T.append(t.ravel()); cross.append(X.ravel()); info.append(bin(m).count("1"))
    return np.array(T), np.array(cross), np.array(info)


TEMPL, CROSS, ARMS = junction_templates()               # 375 templates, built once


def read_junctions(F):                                  # NEW: the closest perfect junction to each image, and how far it is
    I = F.reshape(len(F), 3, 784).mean(1)                                             # the three channels averaged
    best = np.argmin((TEMPL ** 2).sum(1)[None, :] - 2 * I @ TEMPL.T, axis=1)            # least squared error over the whole image (|I|^2 is the same for all)
    misfit = (((I - TEMPL[best]) ** 2) * CROSS[best]).sum(1) / CROSS[best].sum(1)       # mean squared error on the 33 pixels of the cross
    return misfit, ARMS[best]                                                         # (misfit, number of arms)


def junction_stats(F, cut):                             # NEW: share of valid junctions, and how they split by number of arms
    err, arms = read_junctions(F); ok = err < cut                                     # valid: as close to a perfect junction as real ones are
    h = np.bincount(arms[ok], minlength=5)[1:] / max(1, ok.sum())
    return float(ok.mean()), h, float(np.median(err))                                 # (valid share, share of 1, 2, 3, 4 arms among the valid, median misfit)


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- the shapes, the transposed convolution as an adjoint, and its overlaps")
    print("=" * 70)
    rng = np.random.RandomState(0); x = rng.randn(2, 1, 28, 28).astype(np.float32)
    c1, c2 = Conv(1, 32, 4, 2, 1, rng), Conv(32, 64, 4, 2, 1, rng); h1 = c1.forward(x); h2 = c2.forward(h1)
    t1, t2 = ConvT(64, 32, 4, 2, 1, rng), ConvT(32, 1, 4, 2, 1, rng); u1 = t1.forward(h2); u2 = t2.forward(u1)
    print(f"discriminator: {x.shape[1:]} -> {h1.shape[1:]} -> {h2.shape[1:]}   strided convolution: (H + 2p - k) / s + 1 = (28 + 2 - 4) / 2 + 1 = 14")
    print(f"generator:     {h2.shape[1:]} -> {u1.shape[1:]} -> {u2.shape[1:]}   transposed: (H - 1) s - 2p + k = 6 x 2 - 2 + 4 = 14")
    c = Conv(3, 5, 4, 2, 1, rng); t = ConvT(5, 3, 4, 2, 1, rng); c.W = c.W.astype(np.float64); t.W = c.W.copy()  # the same weights
    c.b[:] = 0; t.b[:] = 0; X = rng.randn(2, 3, 14, 14); Y = rng.randn(2, 5, 7, 7)
    a, b = float((c.forward(X) * Y).sum()), float((X * t.forward(Y)).sum())
    print(f"adjoint check, same weights: <conv(X), Y> = {a:.10f};  <X, convT(Y)> = {b:.10f}  (relative gap {abs(a - b) / abs(a):.1e})")
    counts = {}
    for k in (4, 3):                                                                  # how many input pixels paint each output pixel
        t = ConvT(1, 1, k, 2, 1, rng); t.W[...] = 1; t.b[...] = 0
        counts[k] = t.forward(np.ones((1, 1, 7, 7), np.float32))[0, 0]
        print(f"kernel {k}, stride 2: output {counts[k].shape}; contributions per pixel, top-left 4 x 4:")
        for row in counts[k][:4, :4]:
            print("      " + " ".join(f"{v:2.0f}" for v in row))
        inner = counts[k][2:-2, 2:-2]; print(f"   away from the edges: from {inner.min():.0f} to {inner.max():.0f} contributions")
    print("kernel 4 with stride 2 gives every inner pixel the same number of contributions; kernel 3 alternates 1, 2, 4:")
    print("an uneven grid built into the layer itself, the source of the 'checkerboard' pattern")
    print("Actual output from this lesson's run.")
    return counts


def demo_gradcheck():
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- the generator's gradient through both convolutional players")
    print("=" * 70)
    G, D = dcgan_players(0, k=8, c=4)                                                 # small, so finite differences are quick
    for N in (G, D):                                                                  # float64, so the check is not limited by rounding
        for L in N.layers:
            if L.params():
                L.W, L.b = L.W.astype(np.float64), L.b.astype(np.float64)
        N.net = [pr for L in N.layers for pr in L.params()]
    rng = np.random.RandomState(1); Z = rng.randn(6, 8)
    loss = lambda: d134.g_loss(D.forward(G.forward(Z)))[0]                            # noise -> image -> verdict -> loss
    lg, gl = d134.g_loss(D.forward(G.forward(Z))); _, dF = D.backward(gl); gG, _ = G.backward(dF)
    names = [type(L).__name__ for L in G.layers if L.params()]
    print(f"{'layer':>8} {'entries checked':>16} {'worst relative error':>21} {'largest |bias gradient|':>24}")
    rel = lambda f, g: abs(f - g) / max(1e-12, abs(f) + abs(g))                       # relative error, safe at zero
    worst_all = 0
    for name, (W, b), (gW, gb) in zip(names, G.net, gG):
        worst, n = 0, 0
        for arr, g in ((W, gW), (b, gb)):
            for _ in range(6):                                                        # six random entries of each array
                idx = tuple(rng.randint(s) for s in arr.shape); old = arr[idx]
                arr[idx] = old + 1e-5; up = loss(); arr[idx] = old - 1e-5; dn = loss(); arr[idx] = old
                f = (up - dn) / 2e-5
                if max(abs(f), abs(g[idx])) > 1e-10:                                  # skip entries whose gradient is zero: both sides are rounding noise
                    worst = max(worst, rel(f, g[idx])); n += 1
        worst_all = max(worst_all, worst); print(f"{name:>8} {n:16d} {worst:21.1e} {np.abs(gb).max():24.1e}")
    print(f"worst over the generator: {worst_all:.1e} (the gradient passes back through D's two strided convolutions first)")
    print("the Dense and first ConvT biases have zero gradient: the batch normalisation right after them subtracts the")
    print("batch mean, so a constant added before it changes nothing (those entries are left out of the relative error)")
    print("Actual output from this lesson's run.")
    return worst_all


def train_watch(G, D, X, judge, epochs, seed, k=64, batch=128, ch=1, every=1, keep=()):  # NEW: one GAN, watched during training
    oG, oD = d134.AdamGAN(G.net, 2e-4, 0.5), d134.AdamGAN(D.net, 2e-4, 0.5)           # Day 134's settings: lr 0.0002, b1 0.5
    rng = np.random.RandomState(seed); n = len(X); Zfix = np.random.RandomState(9).standard_normal((2000, k)).astype(np.float32)
    log, snaps, t0 = [], {}, time.perf_counter()
    for ep in range(1, epochs + 1):
        o = rng.permutation(n); hs = []
        for s in range(0, n - batch + 1, batch):                                      # one pass over the training images
            hs.append(net_gan_step(G, D, oG, oD, X[o[s:s + batch]], rng, k))
        if ep % every == 0 or ep == epochs:
            F = net_generate(G, Zfix); m = np.mean(hs, 0)                                 # the same 2,000 noise vectors each time
            log.append((ep, m[2], m[3], judge(F), smoothness(F, ch), time.perf_counter() - t0))  # judge returns this dataset's own measures
            if ep in keep:
                snaps[ep] = F[:16].copy()                                             # 16 of them kept as pictures
    return log, snaps


def demo_mnist(X, y, epochs=20):
    print("\n" + "=" * 70)
    print("3. MNIST -- Day 134's dense GAN against the DCGAN, watched every epoch")
    print("=" * 70)
    jW, jb = d116.fit_softmax(X[:60000].astype(np.float64), y[:60000], 10, lam=1e-3, steps=300)  # Day 134's judge, refit the same way
    P = lambda I: d116.softmax(I.astype(np.float64) @ jW + jb)
    Xt = X[:20000]

    def judge(F):                                                                     # which digits, how sure, and how far from the training set
        p = P(F); h = np.bincount(p.argmax(1), minlength=10) / len(F)
        return h.min(), h.max(), float(p.max(1).mean()), float(np.sqrt(d119.sqdist(F[:500], Xt).min(1)).mean()), h
    real = judge(X[60000:62000])
    print(f"real test digits: rarest {real[0]:.3f}, commonest {real[1]:.3f}, judge's confidence {real[2]:.4f}, "
          f"nearest training image {real[3]:.2f}, pixel-to-pixel jump {smoothness(X[60000:62000]):.4f}")
    Gd, Dd = dense_players(0); Gc, Dc = dcgan_players(0)
    print(f"dense: generator {n_params(Gd):,} numbers, discriminator {n_params(Dd):,}")
    print(f"DCGAN: generator {n_params(Gc):,} numbers (64 -> 64 x 7 x 7 -> 32 x 14 x 14 -> 1 x 28 x 28), discriminator {n_params(Dc):,}")
    print(f"both: 20,000 training images, batches of 128, {epochs} epochs, AdamGAN (lr 0.0002, b1 0.5), the non-saturating loss;")
    print("each epoch, on 2,000 fixed noise vectors: the judge's rarest and commonest digit share (0.100 each if even),")
    print("its confidence, the distance to the nearest training image, and the average jump between neighbouring pixels")
    runs = {}
    for name in ("dense", "DCGAN"):
        for seed in (0, 1):
            G, D = (dense_players if name == "dense" else dcgan_players)(seed)
            log, snaps = train_watch(G, D, Xt, judge, epochs, seed, keep=(1, 5, epochs))
            runs[(name, seed)] = (log, snaps)
            print(f"{name}, seed {seed}:  {'epoch':>5} {'D(real)':>8} {'D(fake)':>8} {'rarest':>7} {'commonest':>10} {'confidence':>11} {'nearest':>8} {'jump':>7}")
            for ep, dr, df, j, sm, _ in log:
                if ep in (1, 2, 3, 5, 10, 15, epochs):
                    print(f"{'':16s}{ep:5d} {dr:8.3f} {df:8.3f} {j[0]:7.3f} {j[1]:10.3f} {j[2]:11.4f} {j[3]:8.2f} {sm:7.4f}")
            print(f"{'':16s}({log[-1][-1]:.0f} s)")
    for (name, seed), (log, _) in runs.items():
        print(f"{name}, seed {seed}, final spread of digits: {' '.join(f'{v:.2f}' for v in log[-1][3][4])}")
    print("Actual output from this lesson's run.")
    return runs, real, judge, Xt


def demo_no_bn(Xt, judge, epochs=20):
    print("\n" + "=" * 70)
    print("4. THE DCGAN WITHOUT BATCH NORMALISATION -- seeds 0 and 1, everything else the same")
    print("=" * 70)
    runs = {}
    for seed in (0, 1):
        G, D = dcgan_players(seed, bn=False); log, snaps = train_watch(G, D, Xt, judge, epochs, seed, keep=(epochs,))
        runs[seed] = (log, snaps)
        print(f"no batchnorm, seed {seed}:  {'epoch':>5} {'D(real)':>8} {'D(fake)':>8} {'rarest':>7} {'commonest':>10} {'confidence':>11} {'nearest':>8} {'jump':>7}")
        for ep, dr, df, j, sm, _ in log:
            if ep in (1, 2, 3, 5, 10, 15, epochs):
                print(f"{'':23s}{ep:5d} {dr:8.3f} {df:8.3f} {j[0]:7.3f} {j[1]:10.3f} {j[2]:11.4f} {j[3]:8.2f} {sm:7.4f}")
        print(f"{'':23s}({log[-1][-1]:.0f} s); final spread of digits: {' '.join(f'{v:.2f}' for v in log[-1][3][4])}")
    print("Actual output from this lesson's run.")
    return runs


def demo_junctions(epochs=100):
    print("\n" + "=" * 70)
    print("5. JUNCTIONS -- can each GAN draw straight arms of the right length?")
    print("=" * 70)
    Xj, yj = d47.make_junction_dataset_rgb(n_per_class=1000, size=28, seed=0)         # Week 7's generator, unchanged, at 28 x 28
    Xj = Xj.reshape(len(Xj), -1).astype(np.float32)                                   # 4,000 images, 3 x 28 x 28, flattened
    err, arms = read_junctions(Xj)                                                    # the checker, run on the real images first
    cut = float(np.quantile(err, 0.99))                                               # valid: a misfit no worse than 99% of real images
    print(f"data: 4,000 junctions (1,000 each of 1, 2, 3 and 4 arms of length 8), 3 x 28 x 28, noise 0.4; pixel values from {Xj.min():.2f} to {Xj.max():.2f}")
    print(f"the arm counter on the real images: arms read correctly {np.mean(arms == yj + 1):.4f}; misfit median {np.median(err):.3f}, "
          f"99th percentile {np.quantile(err, 0.99):.3f}; valid if misfit < {cut:.3f}")
    v, h, _ = junction_stats(Xj[:2000], cut); print(f"real images: valid share {v:.3f}; split by arms 1-4: {' '.join(f'{x:.3f}' for x in h)}")
    judge = lambda F: junction_stats(F[:1000], cut)                                   # 1,000 of the 2,000 fixed samples, for speed
    runs = {}
    for name in ("dense", "DCGAN"):
        G, D = dense_players(0, d=2352, out="linear") if name == "dense" else dcgan_players(0, ch=3, out="linear")
        log, snaps = train_watch(G, D, Xj, judge, epochs, 0, ch=3, every=10, keep=(epochs,))
        runs[name] = (log, snaps)
        print(f"{name} (generator {n_params(G):,}, discriminator {n_params(D):,} numbers), seed 0, linear output, {epochs} epochs:")
        print(f"{'':6s}{'epoch':>5} {'D(real)':>8} {'D(fake)':>8} {'valid':>6} {'median misfit':>14}   arms 1-4 among the valid   {'jump':>6}")
        for ep, dr, df, (v, h, mm), sm, _ in log:
            print(f"{'':6s}{ep:5d} {dr:8.3f} {df:8.3f} {v:6.3f} {mm:14.3f}   {' '.join(f'{x:.3f}' for x in h):23s}   {sm:6.3f}")
        print(f"{'':6s}({log[-1][-1]:.0f} s)")
    print(f"real images' jump: {smoothness(Xj[:2000], 3):.3f}")
    print("Actual output from this lesson's run.")
    return runs, Xj, cut


# =============================================================================
# FIGURES
# =============================================================================
def grid(F, ch=1, n=4):                                 # 16 images as one 4 x 4 picture
    I = F[:n * n].reshape(n * n, ch, 28, 28).transpose(0, 2, 3, 1).mean(-1) if ch == 3 else F[:n * n].reshape(n * n, 28, 28)
    return np.vstack([np.hstack(list(I[r * n:(r + 1) * n])) for r in range(n)])


def plot_results(counts, mn, real, nobn, jn, Xj, out):  # ten panels
    fig = plt.figure(figsize=(17, 13))
    for j, k in enumerate((4, 3)):                                                    # the overlap counts
        ax = fig.add_subplot(3, 4, 1 + j); im = ax.imshow(counts[k], cmap="viridis"); plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"transposed conv, kernel {k}, stride 2:\ninputs painting each output pixel", fontsize=9); ax.axis("off")
    for j, (name, seed) in enumerate((("dense", 0), ("DCGAN", 0))):                   # samples at the last epoch
        log, snaps = mn[(name, seed)]; ep = max(snaps)
        ax = fig.add_subplot(3, 4, 3 + j); ax.imshow(grid(snaps[ep]), cmap="gray_r", vmin=0, vmax=1); ax.axis("off")
        ax.set_title(f"{name}, seed {seed}, epoch {ep}", fontsize=10)
    ax = fig.add_subplot(3, 4, 5)                                                     # the DCGAN's first epochs
    ax.imshow(np.hstack([grid(mn[("DCGAN", 0)][1][e], n=2) for e in (1, 5)]), cmap="gray_r", vmin=0, vmax=1); ax.axis("off")
    ax.set_title("DCGAN, seed 0: epoch 1 (left), epoch 5 (right)", fontsize=10)
    series = [(f"{n}, seed {s}", mn[(n, s)][0]) for n, s in mn] + [(f"no batchnorm, seed {s}", nobn[s][0]) for s in nobn]
    for j, (lab, f, yl, ref) in enumerate((("commonest digit share", lambda r: r[3][1], "share", 0.1),
                                           ("distance to nearest training image", lambda r: r[3][3], "distance", real[3]),
                                           ("average jump between neighbouring pixels", lambda r: r[4], "jump", real[5]))):
        ax = fig.add_subplot(3, 4, 6 + j)
        for name, log in series:
            ax.plot([r[0] for r in log], [f(r) for r in log], lw=1.3, ls="--" if "dense" in name else ("-" if "DCGAN" in name else ":"), label=name)
        ax.axhline(ref, color="gray", lw=1); ax.set_xlabel("epoch"); ax.set_ylabel(yl); ax.set_title(f"{lab} (gray: real)", fontsize=9)
        if j == 0:
            ax.legend(fontsize=7)
    ax = fig.add_subplot(3, 4, 9); ax.imshow(np.vstack([grid(Xj[:8], 3, 2), grid(Xj[8:16], 3, 2)]), cmap="gray_r"); ax.axis("off")
    ax.set_title("real junctions (channels averaged)", fontsize=10)
    for j, name in enumerate(("dense", "DCGAN")):
        log, snaps = jn[name]; ep = max(snaps)
        ax = fig.add_subplot(3, 4, 10 + j); ax.imshow(np.vstack([grid(snaps[ep][:8], 3, 2), grid(snaps[ep][8:16], 3, 2)]), cmap="gray_r"); ax.axis("off")
        ax.set_title(f"{name} junctions, epoch {ep}, valid {log[-1][3][0]:.2f}", fontsize=10)
    ax = fig.add_subplot(3, 4, 12)
    for name in ("dense", "DCGAN"):
        log = jn[name][0]; ax.plot([r[0] for r in log], [r[3][0] for r in log], marker="o", label=name)
    top = max(r[3][0] for n in ("dense", "DCGAN") for r in jn[n][0]); ax.set_ylim(0, max(0.1, 1.3 * top))  # the real images score 0.99, far above
    ax.set_xlabel("epoch"); ax.set_ylabel("valid share"); ax.set_title("generated junctions that are valid (real: 0.99)", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the two convolutional players
    fig, ax = plt.subplots(figsize=(13.4, 5)); ax.set_xlim(-0.3, 13.1); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9):                                           # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
    ys = 3.6; xs = (0.9, 3.1, 5.5, 7.9, 10.3)
    for x, t, c in zip(xs, ("noise z\n64 numbers", "dense + BN\n+ ReLU\n64 x 7 x 7", "transposed conv\nk 4, stride 2\n+ BN + ReLU\n32 x 14 x 14",
                            "transposed conv\nk 4, stride 2\n+ sigmoid\n1 x 28 x 28", "fake image"), ("#EEEEEE", "#DDEEFF", "#DDEEFF", "#DDEEFF", "#F8CBAD")):
        box(x, ys, 1.9, 1.3, t, c, 8.5)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + 1.0, ys, b - 1.0, ys)
    yd = 1.2; xd = (10.3, 7.9, 5.5, 3.1, 0.9)
    for x, t, c in zip(xd, ("real or fake\n1 x 28 x 28", "strided conv\nk 4, stride 2\n+ leaky ReLU\n32 x 14 x 14",
                            "strided conv\nk 4, stride 2\n+ leaky ReLU\n64 x 7 x 7", "dense\n3,136 -> 1", "D: one logit\nreal or fake?"),
                        ("#F8CBAD", "#E2F0D9", "#E2F0D9", "#E2F0D9", "#FFF2CC")):
        box(x, yd, 1.9, 1.3, t, c, 8.5)
    for a, b in zip(xd[:-1], xd[1:]):
        arrow(a - 1.0, yd, b + 1.0, yd)
    arrow(10.3, ys - 0.7, 10.3, yd + 0.7)
    ax.text(12.1, 2.4, "generator (top):\nstride 2 doubles\nthe size\n\ndiscriminator\n(bottom): stride 2\nhalves it", fontsize=8.5, ha="center", va="center")
    ax.set_title("Day 136: a DCGAN -- both players built from convolutions", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "dcgan_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("The generator above draws whatever digit its noise happens to land on. Day 137 gives both players the label as")
    print("well (a conditional GAN), so it can be asked for a 7 and must draw a 7.")


def main():                                             # runs every section in order
    counts = demo_hand()                                                              # 1. shapes, adjoint, overlaps
    demo_gradcheck()                                                                  # 2. the gradient, checked
    X, y = d126m.mnist()                                                              # 70,000 images, pixels in [0, 1], float32
    mn, real, judge, Xt = demo_mnist(X, y)                                            # 3. dense against DCGAN
    real = real[:4] + (real[4], smoothness(X[60000:62000]))                           # the real digits' jump, for the plot
    nobn = demo_no_bn(Xt, judge)                                                      # 4. without batch normalisation
    jn, Xj, cut = demo_junctions()                                                    # 5. junctions
    plot_results(counts, mn, real, nobn, jn, Xj, os.path.join(HERE, "dcgan_results.png"))
    draw_diagram()                                                                    # the picture of the two players
    print("\nsaved dcgan_results.png\nsaved dcgan_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
