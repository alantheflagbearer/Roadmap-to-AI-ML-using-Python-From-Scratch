"""
Day 136 Exercise -- The Convolutional Players

The fourth of Week 20's parts for Day 139's build.

    pad2, im2col, col2im         patches of an image as rows, and back (looping over kernel offsets)
    Conv(cin, cout, k, s, p)     a strided convolution: stride 2 halves the height and width
    ConvT(cin, cout, k, s, p)    a transposed convolution: the Conv's input-gradient as a layer; stride 2 doubles
    BN(c)                        batch normalisation per channel
    Dense, Act, Net              a dense layer, activations and reshapes, and a stack of layers
    dcgan_players(seed, ...)     the DCGAN generator and discriminator
    net_gan_step(G, D, oG, oD, ...)  Day 134's round of the game, for layer stacks
    net_generate(G, Z)               noise in, images out, in batches

Checked by hand: the shapes, the transposed convolution as the adjoint of the
convolution, its overlap counts, one gradient against finite differences, and
200 rounds of the game on MNIST.
"""
import os                                              # os: the paths to this week's and earlier folders
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
sys.path.insert(0, HERE)                                # Day 134's parts
sys.path.insert(0, os.path.join(HERE, "..", "week-19-unsupervised-learning-ii"))  # Day 126's parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day134_exercise as d134                      # Day 134: AdamGAN, d_loss, g_loss
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)


def pad2(X, p):                                         # NEW: Day 136's version, unchanged
    return np.pad(X, ((0, 0), (0, 0), (p, p), (p, p))) if p else X                     # batch and channel axes untouched


def im2col(Xp, k, s, oh, ow):                           # NEW: Day 136's version, unchanged
    m, C = Xp.shape[:2]; cols = np.empty((m, oh, ow, C, k, k), Xp.dtype)               # one row per (image, output position)
    for i in range(k):                                                                # kernel row offset
        for j in range(k):                                                            # kernel column offset
            cols[:, :, :, :, i, j] = Xp[:, :, i:i + s * oh:s, j:j + s * ow:s].transpose(0, 2, 3, 1)  # every position's pixel at this offset, stepping by s
    return cols.reshape(m * oh * ow, C * k * k)                                       # (positions, patch values)


def col2im(dcols, shape, k, s, oh, ow):                 # NEW: Day 136's version, unchanged
    dX = np.zeros(shape, dcols.dtype); d = dcols.reshape(shape[0], oh, ow, shape[1], k, k)  # the image to fill, and the patches unflattened
    for i in range(k):                                                                # kernel row offset
        for j in range(k):                                                            # kernel column offset
            dX[:, :, i:i + s * oh:s, j:j + s * ow:s] += d[:, :, :, :, i, j].transpose(0, 3, 1, 2)  # overlapping patches add up
    return dX


class Conv:                                             # NEW: Day 136's version, unchanged
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


class ConvT:                                            # NEW: Day 136's version, unchanged
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


class BN:                                               # NEW: Day 136's version, unchanged
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


class Dense:                                            # NEW: Day 136's version, unchanged
    def __init__(self, a, b, rng):
        self.W = (rng.randn(a, b) * np.sqrt(2.0 / a)).astype(np.float32); self.b = np.zeros(b, np.float32)

    def params(self):
        return [(self.W, self.b)]

    def forward(self, X):
        self.X = X; return X @ self.W + self.b

    def backward(self, d):
        self.grads = [(self.X.T @ d, d.sum(0))]; return d @ self.W.T


class Act:                                              # NEW: Day 136's version, unchanged
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


class Net:                                              # NEW: Day 136's version, unchanged
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


def dcgan_players(seed, k=64, c=32, ch=1, bn=True, out="sigmoid"):  # NEW: Day 136's version, unchanged
    rg, rd = np.random.RandomState(seed), np.random.RandomState(seed + 1)
    norm = (lambda n: [BN(n)]) if bn else (lambda n: [])                              # batch normalisation, or nothing
    G = Net([Dense(k, 2 * c * 49, rg)] + norm(2 * c * 49) + [Act("relu"), Act((-1, 2 * c, 7, 7)),    # noise -> 2c maps of 7 x 7
             ConvT(2 * c, c, 4, 2, 1, rg)] + norm(c) + [Act("relu"),                                 # -> c maps of 14 x 14
            ConvT(c, ch, 4, 2, 1, rg)] + ([Act("sigmoid")] if out == "sigmoid" else []) + [Act((-1, ch * 784))])  # -> ch x 28 x 28, flattened
    D = Net([Act((-1, ch, 28, 28)), Conv(ch, c, 4, 2, 1, rd), Act("lrelu"),                         # 28 x 28 -> c maps of 14 x 14
             Conv(c, 2 * c, 4, 2, 1, rd), Act("lrelu"), Act((-1, 2 * c * 49)), Dense(2 * c * 49, 1, rd)])  # -> 2c of 7 x 7 -> one logit
    return G, D


def net_gan_step(G, D, oG, oD, X, rng, k):              # NEW: Day 136's version, unchanged
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


def net_generate(G, Z, batch=500):                      # NEW: Day 136's version, unchanged
    return np.concatenate([G.forward(Z[i:i + batch]) for i in range(0, len(Z), batch)])


rng = np.random.RandomState(0); x = rng.randn(2, 1, 28, 28).astype(np.float32)
h = Conv(1, 4, 4, 2, 1, rng).forward(x); u = ConvT(4, 1, 4, 2, 1, rng).forward(h)
print(f"strided convolution {x.shape} -> {h.shape}; transposed convolution back to {u.shape}")
c = Conv(2, 3, 4, 2, 1, rng); t = ConvT(3, 2, 4, 2, 1, rng); c.W = c.W.astype(np.float64); t.W = c.W.copy(); c.b[:] = 0; t.b[:] = 0
X = rng.randn(1, 2, 8, 8); Y = rng.randn(1, 3, 4, 4)
print(f"<conv(X), Y> = {(c.forward(X) * Y).sum():.10f}; <X, convT(Y)> = {(X * t.forward(Y)).sum():.10f} (the same: one is the other's adjoint)")
t = ConvT(1, 1, 4, 2, 1, rng); t.W[...] = 1; t.b[...] = 0
print("kernel 4, stride 2, all-ones weights on a 3 x 3 input of ones; each output pixel counts its contributions:")
print(t.forward(np.ones((1, 1, 3, 3), np.float32))[0, 0].astype(int))
G, D = dcgan_players(0, k=8, c=4)
for N in (G, D):
    for L in N.layers:
        if L.params():
            L.W, L.b = L.W.astype(np.float64), L.b.astype(np.float64)
    N.net = [pr for L in N.layers for pr in L.params()]
Z = rng.randn(6, 8); loss = lambda: d134.g_loss(D.forward(G.forward(Z)))[0]
_, gl = d134.g_loss(D.forward(G.forward(Z))); _, dF = D.backward(gl); gG, _ = G.backward(dF)
W = G.net[2][0]; old = W[1, 2, 3, 0]; W[1, 2, 3, 0] = old + 1e-5; up = loss(); W[1, 2, 3, 0] = old - 1e-5; dn = loss(); W[1, 2, 3, 0] = old
print(f"d loss / d ConvT weight [1, 2, 3, 0]: backward {gG[2][0][1, 2, 3, 0]:.10f}; finite difference {(up - dn) / 2e-5:.10f}")
Xm = d126m.mnist()[0][:5000]; G, D = dcgan_players(0); oG, oD = d134.AdamGAN(G.net, 2e-4, 0.5), d134.AdamGAN(D.net, 2e-4, 0.5)
rng = np.random.RandomState(0); hist = []
for s in range(200):
    hist.append(net_gan_step(G, D, oG, oD, Xm[rng.randint(0, 5000, 128)], rng, 64))
for a in (0, 50, 100, 150):
    m = np.mean(hist[a:a + 50], 0); print(f"rounds {a + 1}-{a + 50}: D(real) {m[2]:.3f}, D(fake) {m[3]:.3f}")
F = net_generate(G, np.random.RandomState(9).standard_normal((500, 64)).astype(np.float32))
print(f"500 generated images after 200 rounds: shape {F.shape}, ink (average pixel) {F.mean():.3f}; real digits {Xm.mean():.3f}")
