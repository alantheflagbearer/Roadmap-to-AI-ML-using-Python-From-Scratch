"""
Day 139: The Week 20 Build -- a Conditional DCGAN, Scored, and the Forward Diffusion Process

Two halves. The first finishes the week: its six exercise parts are copied in
verbatim (one banner per file) and put together into one model, a
conditional DCGAN trained on all 60,000 MNIST training images, scored with
Day 138's metrics against the simplest generative model of the week, Day
133's Gaussian class models.

The second half starts the next idea. A DIFFUSION model learns to undo a
fixed process that destroys data: add a little Gaussian noise, many times,
until nothing of the image is left. Today builds that forward process from
scratch and measures what it destroys and how fast; Day 140 learns to run it
backwards.

  1. Checks: the parts are verbatim, and the forward process's shortcut
     formula matches 1,000 small steps.
  2. The week's model: a conditional DCGAN on all 60,000 training images.
  3. Against Day 133's Gaussian class models, scored the same way.
  4. The forward process on eight known modes (Day 135's ring).
  5. The forward process on MNIST: when does a digit stop being a digit?
     A linear against a cosine noise schedule.

Data: MNIST (cached by Day 126), and Day 135's ring of eight Gaussians.

Every line written for today carries its own comment, continuing Day 52's
convention; the copied parts keep their own.
"""
import ast                                             # ast: reading function sources out of the exercise files
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier weeks' parts
import time                                            # time: how long training takes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' exercise parts
with contextlib.redirect_stdout(io.StringIO()):         # their imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, sigmoid, Adam (Days 134-135's parts use them)
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day119_exercise as d119                      # Day 119: sqdist (Days 137-138's parts use it)
    import day116_exercise as d116                      # Day 116: softmax (Day 138's part uses it)
d134 = d136 = sys.modules[__name__]                     # the copied parts call each other as d134.gan_step, d136.Conv etc.: here they all live in this file


# =============================================================================
# from day133_exercise.py, verbatim: fit_gaussians, log_gauss, log_joint, posterior, log_px, lda_linear, sample_class
# =============================================================================
def fit_gaussians(X, y, k, kind="shared", reg=1e-3):    # NEW: Day 133's version, unchanged
    d = X.shape[1]; mu = np.array([X[y == c].mean(0) for c in range(k)])            # one mean per class
    prior = np.bincount(y, minlength=k) / len(y)                                      # how common each class is
    if kind == "shared":                                                              # LDA: one covariance for every class
        D = X - mu[y]; S = D.T @ D / len(X) + reg * np.eye(d)                         # spread around each point's own class mean
        cov = np.repeat(S[None], k, 0)                                                # the same matrix for every class
    elif kind == "diagonal":                                                          # naive Bayes: features independent within a class
        cov = np.array([np.diag(X[y == c].var(0) + reg) for c in range(k)])          # only each feature's own variance
    else:                                                                             # QDA: each class its own covariance
        cov = np.array([np.cov(X[y == c].T, bias=True).reshape(d, d) + reg * np.eye(d) for c in range(k)])
    return {"mu": mu, "cov": cov, "prior": prior}                                     # reg keeps every covariance invertible


def log_gauss(X, mu, S):                                # NEW: Day 133's version, unchanged
    L = np.linalg.cholesky(S); Z = np.linalg.solve(L, (X - mu).T)                     # L L^T = S, so |Z|^2 is the Mahalanobis distance
    return -0.5 * ((Z ** 2).sum(0) + 2 * np.log(np.diag(L)).sum() + len(mu) * np.log(2 * np.pi))  # log det S = 2 sum log diag L


def log_joint(model, X, keep=None):                     # NEW: Day 133's version, unchanged
    idx = np.arange(X.shape[1]) if keep is None else keep                              # a Gaussian's marginal: just drop the rest
    return np.stack([log_gauss(X[:, idx], m[idx], S[np.ix_(idx, idx)]) + np.log(p)    # log p(x | c) + log p(c)
                     for m, S, p in zip(model["mu"], model["cov"], model["prior"])], 1)


def posterior(model, X, keep=None):                     # NEW: Day 133's version, unchanged
    J = log_joint(model, X, keep); J -= J.max(1, keepdims=True); P = np.exp(J)        # subtract the largest first: no overflow
    return P / P.sum(1, keepdims=True)                                                # dividing by the sum is dividing by p(x)


def log_px(model, X):                                   # NEW: Day 133's version, unchanged
    J = log_joint(model, X); m = J.max(1)                                             # log-sum-exp, kept stable
    return m + np.log(np.exp(J - m[:, None]).sum(1))


def lda_linear(model):                                  # NEW: Day 133's version, unchanged
    Si = np.linalg.inv(model["cov"][0]); W = Si @ model["mu"].T                        # w_c = S^-1 mu_c
    b = -0.5 * np.einsum("cd,dc->c", model["mu"], W) + np.log(model["prior"])          # b_c = -mu_c S^-1 mu_c / 2 + log p(c)
    return W, b


def sample_class(model, c, n, rng):                     # NEW: Day 133's version, unchanged
    L = np.linalg.cholesky(model["cov"][c])                                           # any matrix with L L^T = S will do
    return model["mu"][c] + rng.standard_normal((n, len(L))) @ L.T                     # x = mu + L eps has covariance L L^T = S


# =============================================================================
# from day134_exercise.py, verbatim: AdamGAN, softplus, d_loss, g_loss, generate, g_grads, gan_step
# =============================================================================
class AdamGAN(d126.Adam):                                 # NEW: Day 134's version, unchanged
    def __init__(self, net, lr=2e-4, b1=0.5):
        super().__init__(net, lr); self.b1 = b1                                       # Day 126's constructor, plus b1

    def step(self, net, grads):                                                       # Day 126's step, with 0.9 replaced by b1
        self.t += 1; c1, c2 = 1 - self.b1 ** self.t, 1 - 0.999 ** self.t
        for (W, b), (gW, gb), m, v in zip(net, grads, self.m, self.v):
            for p, g, mm, vv in ((W, gW, m[0], v[0]), (b, gb, m[1], v[1])):
                mm *= self.b1; mm += (1 - self.b1) * g; vv *= 0.999; vv += 0.001 * g * g
                p -= self.lr * (mm / c1) / (np.sqrt(vv / c2) + 1e-8)


def softplus(Z):                                        # NEW: Day 134's version, unchanged
    return np.maximum(Z, 0) + np.log1p(np.exp(-np.abs(Z)))


def d_loss(lr, lf):                                     # NEW: Day 134's version, unchanged
    loss = softplus(-lr).mean() + softplus(lf).mean()                                  # -log D(real) - log(1 - D(fake)), D = sigmoid(logit)
    return float(loss), (d126.sigmoid(lr) - 1) / len(lr), d126.sigmoid(lf) / len(lf)    # d/dlogit of each term, averaged over the batch


def g_loss(lf, kind="non-saturating"):                  # NEW: Day 134's version, unchanged
    if kind == "non-saturating":                                                      # -log D(G(z)): strong gradients while D is winning
        return float(softplus(-lf).mean()), (d126.sigmoid(lf) - 1) / len(lf)
    return float(-softplus(lf).mean()), -d126.sigmoid(lf) / len(lf)                     # the minimax form, log(1 - D(G(z))), to be minimised


def generate(G, Z, out="sigmoid"):                      # NEW: Day 134's version, unchanged
    A = d126.forward(G, Z, ["relu"] * (len(G) - 1) + ["linear"])                       # Day 126's forward pass
    return (d126.sigmoid(A[-1]) if out == "sigmoid" else A[-1]), A                     # pixels need [0, 1]; 1-D or 2-D data does not


def g_grads(G, D, Z, out="sigmoid", kind="non-saturating"):  # NEW: Day 134's version, unchanged
    dacts = ["relu"] * (len(D) - 1) + ["linear"]; gacts = ["relu"] * (len(G) - 1) + ["linear"]
    F, AG = generate(G, Z, out); Af = d126.forward(D, F, dacts); lg, gf = g_loss(Af[-1], kind)  # noise -> fake -> verdict -> loss
    _, dF = d126.backward(D, Af, dacts, gf)                                           # backward through D to its input: d loss / d fake
    dL = dF * F * (1 - F) if out == "sigmoid" else dF                                 # back through the output sigmoid
    return lg, d126.backward(G, AG, gacts, dL)[0], Af                                 # D's own gradients are never computed here


def gan_step(G, D, oG, oD, X, rng, k, out="sigmoid", kind="non-saturating", d_steps=1):  # NEW: Day 134's version, unchanged
    dacts = ["relu"] * (len(D) - 1) + ["linear"]; n = len(X)                          # D's activations; the batch size
    for _ in range(d_steps):                                                          # 1. the discriminator: real up, fake down
        F, _ = generate(G, rng.standard_normal((n, k)).astype(np.float32), out)       # a fresh batch of fakes
        Ar, Af = d126.forward(D, X, dacts), d126.forward(D, F, dacts)                  # its verdicts on both
        ld, gr, gf = d_loss(Ar[-1], Af[-1])
        g1, _ = d126.backward(D, Ar, dacts, gr); g2, _ = d126.backward(D, Af, dacts, gf)  # the two halves of its gradient
        oD.step(D, [(a[0] + b[0], a[1] + b[1]) for a, b in zip(g1, g2)])              # one Adam step on their sum
    lg, grads, Af = g_grads(G, D, rng.standard_normal((n, k)).astype(np.float32), out, kind)  # 2. the generator: fool the updated discriminator
    oG.step(G, grads)                                                                 # only the generator moves
    return ld, lg, float(d126.sigmoid(Ar[-1]).mean()), float(d126.sigmoid(Af[-1]).mean())  # losses, and D's average verdicts


# =============================================================================
# from day135_exercise.py, verbatim: ring, mode_stats, train_ring
# =============================================================================
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


# =============================================================================
# from day136_exercise.py, verbatim: pad2, im2col, col2im, Conv, ConvT, BN, Dense, Act, Net, dcgan_players, net_gan_step, net_generate
# =============================================================================
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


# =============================================================================
# from day137_exercise.py, verbatim: onehot, g_input, d_input, cgan_players, cgan_step, spread
# =============================================================================
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


# =============================================================================
# from day138_exercise.py, verbatim: sqrtm_psd, frechet, gauss, inception_score, precision_recall, feature_net, train_classifier, Scorer
# =============================================================================
def sqrtm_psd(S):                                       # NEW: Day 138's version, unchanged
    w, V = np.linalg.eigh(S)                                                          # S = V diag(w) V^T
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T                                   # V diag(sqrt w) V^T; tiny negative w from rounding set to 0


def frechet(mu1, S1, mu2, S2):                          # NEW: Day 138's version, unchanged
    A = sqrtm_psd(S1); M = A @ S2 @ A                                                  # S1^(1/2) S2 S1^(1/2): symmetric, same eigenvalues as S1 S2
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((M + M.T) / 2), 0, None)).sum()       # Tr (S1 S2)^(1/2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * tr_sqrt)


def gauss(F):                                           # NEW: Day 138's version, unchanged
    F = F.astype(np.float64); return F.mean(0), np.cov(F, rowvar=False)


def inception_score(P):                                 # NEW: Day 138's version, unchanged
    P = np.clip(P.astype(np.float64), 1e-12, 1); py = P.mean(0)                        # p(y): the classes over all images
    return float(np.exp((P * (np.log(P) - np.log(py))).sum(1).mean()))


def precision_recall(R, F, k=3):                        # NEW: Day 138's version, unchanged
    DRR, DFF, DRF = d119.sqdist(R, R), d119.sqdist(F, F), d119.sqdist(R, F)          # every squared distance, in feature space
    rR = np.sort(DRR, 1)[:, k]; rF = np.sort(DFF, 1)[:, k]                            # each point's k-th neighbour in its own set (column 0 is itself)
    precision = float(np.mean((DRF <= rR[:, None]).any(0)))                            # a fake counts if it falls inside some real point's ball
    recall = float(np.mean((DRF <= rF[None, :]).any(1)))                               # a real counts if it falls inside some fake point's ball
    return precision, recall


def feature_net(seed=0):                                # NEW: Day 138's version, unchanged
    r = np.random.RandomState(seed)
    body = d136.Net([d136.Act((-1, 1, 28, 28)), d136.Conv(1, 16, 4, 2, 1, r), d136.Act("lrelu"), d136.Conv(16, 32, 4, 2, 1, r),
                     d136.Act("lrelu"), d136.Act((-1, 32 * 49)), d136.Dense(32 * 49, 64, r), d136.Act("relu")])  # the features
    head = d136.Net([d136.Dense(64, 10, r)])                                         # the class scores
    return body, head


def train_classifier(X, y, epochs=3, batch=128, seed=0):  # NEW: Day 138's version, unchanged
    body, head = feature_net(seed); ob, oh = d134.AdamGAN(body.net, 1e-3, 0.9), d134.AdamGAN(head.net, 1e-3, 0.9)  # b1 0.9: Day 126's plain Adam
    rng = np.random.RandomState(seed); Y = np.eye(10, dtype=np.float32)[y]
    for ep in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(X) - batch + 1, batch):
            b = o[s:s + batch]; L = head.forward(body.forward(X[b]))                    # scores for one batch
            dL = (d116.softmax(L) - Y[b]) / batch                                     # d cross-entropy / d scores, averaged
            gh, dF = head.backward(dL); gb, _ = body.backward(dF)
            oh.step(head.net, gh); ob.step(body.net, gb)
    return body, head


class Scorer:                                           # NEW: Day 138's version, unchanged
    def __init__(self, body, head, Xref, Xnear):
        self.body, self.head = body, head
        self.Fref = self.feats(Xref); self.mu, self.S = gauss(self.Fref)              # the real Gaussian, from 10,000 training images
        self.Fpr = self.Fref[:2000]; self.Xnear = Xnear                               # 2,000 of them for precision/recall; pixels for the nearest-image test

    def feats(self, X):                                                               # features, in batches
        return np.concatenate([self.body.forward(X[i:i + 1000]) for i in range(0, len(X), 1000)])

    def __call__(self, X):                                                            # IS, FID, precision, recall, nearest training image
        F = self.feats(X); P = d116.softmax(self.head.forward(F).astype(np.float64)); mu, S = gauss(F)
        p, r = precision_recall(self.Fpr, F[:2000])
        near = float(np.sqrt(np.maximum(d119.sqdist(X[:500], self.Xnear).min(1), 0)).mean())  # pixel distance to the nearest training image
        return inception_score(P), frechet(mu, S, self.mu, self.S), p, r, near


# =============================================================================
# THE FORWARD DIFFUSION PROCESS -- NEW
# =============================================================================
def schedule(kind="linear", T=1000):                    # NEW: how much noise each of the T steps adds
    if kind == "linear":                                                              # Ho et al. (2020): beta from 0.0001 to 0.02
        betas = np.linspace(1e-4, 0.02, T)
    else:                                                                             # Nichol and Dhariwal (2021): alpha-bar follows a squared cosine
        f = lambda t: np.cos((t / T + 0.008) / 1.008 * np.pi / 2) ** 2
        abar = f(np.arange(T + 1)) / f(0); betas = np.clip(1 - abar[1:] / abar[:-1], 0, 0.999)  # beta_t from the ratio of neighbours
    abar = np.concatenate([[1.0], np.cumprod(1 - betas)])                             # alpha-bar_t = prod (1 - beta_s); alpha-bar_0 = 1
    return betas, abar                                                                # betas[t - 1] is step t's beta; abar[t] after t steps


def q_step(x, beta, rng):                               # NEW: one small step: shrink the signal a little, add a little noise
    return np.sqrt(1 - beta) * x + np.sqrt(beta) * rng.standard_normal(x.shape)


def q_sample(x0, t, abar, rng):                         # NEW: jump straight to step t: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * rng.standard_normal(x0.shape)


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_checks():
    print("=" * 70)
    print("1. CHECKS -- the parts are verbatim, and the forward process's shortcut matches 1,000 small steps")
    print("=" * 70)
    me = open(os.path.abspath(__file__), encoding="utf-8").read(); found, total = 0, 0  # this file's own text
    for d in range(133, 139):                                                        # the week's six exercise files
        src = open(os.path.join(HERE, f"day{d}_exercise.py"), encoding="utf-8").read(); lines = src.split("\n")
        for node in ast.parse(src).body:                                             # every function and class at the top level
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total += 1; found += "\n".join(lines[node.lineno - 1:node.end_lineno]) in me  # its exact text, found in this file?
    print(f"functions and classes from day133_exercise.py to day138_exercise.py found here, character for character: {found} of {total}")
    betas, abar = schedule("linear")
    print(f"linear schedule: beta from {betas[0]:.4f} to {betas[-1]:.4f} over {len(betas)} steps; alpha-bar_1000 = {abar[-1]:.2e}, "
          f"so sqrt(alpha-bar) = {np.sqrt(abar[-1]):.4f} of the signal is left")
    rng = np.random.RandomState(0); x0 = np.where(rng.rand(200000) < 0.5, -1.0, 1.0)  # 200,000 1-D points, half at -1 and half at +1
    x = x0.copy(); print(f"{'t':>6} {'1,000-step route: mean, variance, E[x x_0]':>44} {'shortcut: mean, variance, E[x x_0]':>37} {'formula E[x x_0] = sqrt(abar)':>30}")
    for t in range(1, 1001):
        x = q_step(x, betas[t - 1], rng)                                              # one small step at a time
        if t in (10, 100, 300, 1000):
            j = q_sample(x0, t, abar, rng)                                            # the same t in one jump
            print(f"{t:6d} {x.mean():14.4f} {x.var():9.4f} {(x * x0).mean():12.4f}       {j.mean():14.4f} {j.var():9.4f} {(j * x0).mean():10.4f} {np.sqrt(abar[t]):18.4f}")
    print("the data has variance 1, and so does every x_t: each step takes away exactly as much signal variance as it adds noise")
    print("Actual output from this lesson's run.")
    return found, total


def train_week(X, y, sc, epochs, seed, at, k=64, batch=128):  # NEW: the conditional DCGAN of Days 136-137, scored with Day 138's Scorer
    G, D = cgan_players(seed, k)
    oG, oD = AdamGAN(G.net, 2e-4, 0.5), AdamGAN(D.net, 2e-4, 0.5)                    # Day 134's settings: lr 0.0002, b1 0.5
    rng = np.random.RandomState(seed); Zfix = np.random.RandomState(9).standard_normal((2000, k)).astype(np.float32)
    yfix = np.tile(np.arange(10), 200); log, t0 = [], time.perf_counter()            # 200 of each digit, interleaved
    for ep in range(1, epochs + 1):
        o = rng.permutation(len(X))
        for s in range(0, len(X) - batch + 1, batch):                                 # one pass over all the training images
            b = o[s:s + batch]; cgan_step(G, D, oG, oD, X[b], y[b], rng, k)
        if ep in at:
            F = net_generate(G, g_input(Zfix, yfix)); log.append((ep, sc(F), asked(sc, F, yfix), time.perf_counter() - t0))
    return G, log, F, yfix


def asked(sc, F, labels):                               # NEW: the feature network's verdict: does each image show the digit asked for?
    said = np.concatenate([sc.head.forward(sc.feats(F[i:i + 1000])).argmax(1) for i in range(0, len(F), 1000)])
    return float(np.mean(said == labels)), np.array([np.mean(said[labels == d] == d) for d in range(10)])


def per_digit_fid(sc, F, labels, X, y):                 # NEW: FID of each digit's 200 images against that digit's real images
    Fr = sc.Fref; yr = y[:len(Fr)]; out = []                                          # the reference features are training images 0-9,999
    for d in range(10):
        mu, S = gauss(Fr[yr == d]); out.append(frechet(*gauss(sc.feats(F[labels == d])), mu, S))
    return np.array(out)


def demo_week(X, y, epochs=10):
    print("\n" + "=" * 70)
    print("2. THE WEEK'S MODEL -- a conditional DCGAN (Days 134, 136, 137) on all 60,000 training images, scored (Day 138)")
    print("=" * 70)
    t0 = time.perf_counter(); body, head = train_classifier(X[:60000], y[:60000])     # Day 138's feature network, 3 epochs
    acc = np.mean(np.concatenate([head.forward(body.forward(X[i:i + 1000])).argmax(1) for i in range(60000, 70000, 1000)]) == y[60000:])
    sc = Scorer(body, head, X[:10000], X[10000:30000])
    print(f"feature network (Day 138's part, 3 epochs, {time.perf_counter() - t0:.0f} s): test accuracy {acc:.4f}")
    pt = 60000 + np.random.RandomState(0).permutation(10000)[:2000]                   # 2,000 test digits in random order (Day 138: the stored order is not random)
    real = sc(X[pt]); ra = asked(sc, X[pt], y[pt]); rpd = per_digit_fid(sc, X[pt], y[pt], X, y)
    print(f"real test digits (2,000): IS {real[0]:.3f}, FID {real[1]:.3f}, precision {real[2]:.3f}, recall {real[3]:.3f}, nearest {real[4]:.2f}, "
          f"named correctly {ra[0]:.3f}")
    print(f"generator 74 -> 64 x 7 x 7 -> 32 x 14 x 14 -> 1 x 28 x 28; discriminator 11 channels in; AdamGAN (lr 0.0002, b1 0.5);")
    print(f"60,000 images, batches of 128, {epochs} epochs (468 rounds each); scored on 2,000 fixed noise vectors, 200 asked for each digit")
    G, log, F, yfix = train_week(X[:60000], y[:60000], sc, epochs, 0, (1, 2, 5, epochs))
    print(f"{'epoch':>7} {'IS':>7} {'FID':>8} {'precision':>10} {'recall':>7} {'nearest':>8} {'asked digit drawn':>18}")
    for ep, s, a, _ in log:
        print(f"{ep:7d} {s[0]:7.3f} {s[1]:8.3f} {s[2]:10.3f} {s[3]:7.3f} {s[4]:8.2f} {a[0]:18.3f}")
    print(f"({log[-1][-1]:.0f} s)")
    gpd = per_digit_fid(sc, F, yfix, X, y)
    print("per digit at the end, 200 images each (FID with only 200 images is biased upwards, so real test digits are shown too):")
    print("   digit         " + " ".join(f"{d:6d}" for d in range(10)))
    print("   asked, drawn  " + " ".join(f"{v:6.2f}" for v in log[-1][2][1]))
    print("   FID, GAN      " + " ".join(f"{v:6.1f}" for v in gpd))
    print("   FID, real     " + " ".join(f"{v:6.1f}" for v in rpd))
    print("Actual output from this lesson's run.")
    return G, log, F, yfix, sc, real, ra, gpd, rpd


def demo_gaussians(X, y, sc):
    print("\n" + "=" * 70)
    print("3. AGAINST DAY 133'S GAUSSIAN CLASS MODELS -- the simplest generative model of the week, scored the same way")
    print("=" * 70)
    labels = np.tile(np.arange(10), 200); out = {}
    for kind, reg in (("shared", 1e-3), ("full", 1e-3), ("full", 1e-2)):
        t0 = time.perf_counter(); m = fit_gaussians(X[:60000].astype(np.float64), y[:60000], 10, kind, reg); rng = np.random.RandomState(0)
        F = np.empty((2000, 784), np.float32)
        for d in range(10):                                                           # 200 images of each digit, from p(x | digit)
            F[labels == d] = sample_class(m, d, 200, rng)
        F = np.clip(F, 0, 1); s = sc(F); a = asked(sc, F, labels); out[(kind, reg)] = (s, a, F)  # pixels clipped to [0, 1]
        print(f"{kind:>6} covariance, reg {reg:g} ({time.perf_counter() - t0:.0f} s): IS {s[0]:.3f}, FID {s[1]:.3f}, precision {s[2]:.3f}, "
              f"recall {s[3]:.3f}, nearest {s[4]:.2f}, asked digit drawn {a[0]:.3f}")
    print("Actual output from this lesson's run.")
    return out


def demo_ring():
    print("\n" + "=" * 70)
    print("4. THE FORWARD PROCESS ON EIGHT KNOWN MODES -- Day 135's ring, noised step by step")
    print("=" * 70)
    rng = np.random.RandomState(0); x0 = ring(20000, rng).astype(np.float64)        # 20,000 points: 8 modes of spread 0.05 on a circle of radius 2
    betas, abar = schedule("linear"); rows = []
    print("Day 135's mode_stats on x_t (a sample is good within 0.15 of a centre); the Frechet distance (Day 138) between")
    print("x_t's fitted Gaussian and N(0, I), the end point of the process")
    print(f"{'t':>6} {'sqrt(abar)':>11} {'noise sd':>9} {'modes':>6} {'good share':>11} {'distance from N(0, I)':>22}")
    for t in (0, 1, 5, 10, 25, 50, 100, 200, 400, 700, 1000):
        xt = q_sample(x0, t, abar, rng); m, g, _ = mode_stats(xt); fd = frechet(*gauss(xt), np.zeros(2), np.eye(2))
        rows.append((t, m, g, fd, xt[:2000])); print(f"{t:6d} {np.sqrt(abar[t]):11.4f} {np.sqrt(1 - abar[t]):9.4f} {m:6d} {g:11.3f} {fd:22.4f}")
    print(f"for comparison, 20,000 exact N(0, I) samples: distance {frechet(*gauss(rng.standard_normal((20000, 2))), np.zeros(2), np.eye(2)):.4f}")
    print("Actual output from this lesson's run.")
    return rows


def demo_mnist_noise(X, y, sc):
    print("\n" + "=" * 70)
    print("5. THE FORWARD PROCESS ON MNIST -- when does a digit stop being a digit? Linear against cosine schedule")
    print("=" * 70)
    pt = 60000 + np.random.RandomState(0).permutation(10000)[:2000]                   # the same 2,000 test digits as section 2, in random order
    x0 = 2 * X[pt].astype(np.float64) - 1; yt = y[pt]                                 # pixels moved to [-1, 1]
    xtr = 2 * X[:10000].astype(np.float64) - 1; ytr = y[:10000]                       # 10,000 training digits, for classifiers trained on noise
    print("pixels scaled to [-1, 1] (variance 1 would be ideal; here it is", f"{x0.var():.3f});",
          "each x_t is mapped back by (x_t + 1) / 2 for the feature network")
    print("columns: signal left sqrt(abar); the feature network's accuracy (trained on clean digits); the accuracy of Day 116's softmax")
    print("regression trained on 10,000 digits noised to the same t (an estimate of what is left to find); FID against clean digits")
    res = {}
    for kind in ("linear", "cosine"):
        betas, abar = schedule(kind); rng = np.random.RandomState(1); rows = []
        print(f"{kind} schedule:\n{'t':>8} {'sqrt(abar)':>11} {'clean-trained':>14} {'noise-trained':>14} {'FID':>9}")
        for t in (0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000):
            xt = q_sample(x0, t, abar, rng); img = ((xt + 1) / 2).astype(np.float32)
            ca = asked(sc, img, yt)[0]                                                # the classifier that has only seen clean digits
            W, b = d116.fit_softmax(q_sample(xtr, t, abar, rng), ytr, 10, lam=1e-3, steps=200, lr=0.05)  # one trained on this much noise (lr: see the note printed below)
            na = float(np.mean(d116.softmax(xt @ W + b).argmax(1) == yt)); s = sc(img)
            rows.append((t, np.sqrt(abar[t]), ca, na, s[1], img[:1].copy())); print(f"{t:8d} {np.sqrt(abar[t]):11.4f} {ca:14.3f} {na:14.3f} {s[1]:9.2f}")
        res[kind] = rows
        half = next((t for t, _, _, na, _, _ in rows if na < 0.5), None); print(f"   the noise-trained accuracy first falls below 0.5 at t = {half}")
    print("note: Day 116's default learning rate, 0.5, was unstable on these inputs in a first draft (linear schedule, t = 0, 50, 100,")
    print("200, 300, 500: 0.883 0.889 0.545 0.808 0.729 0.373, rising again after t = 100); 0.05 for 200 steps, 0.01 for 500 and 0.05")
    print("for 500 all fell steadily and agreed within about 0.01, so 0.05 for 200 steps is used")
    print("Actual output from this lesson's run.")
    return res


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(week, gs, rows, res, out):             # nine panels
    G, log, F, yfix, sc, real, ra, gpd, rpd = week; fig = plt.figure(figsize=(17, 13))
    ax = fig.add_subplot(3, 3, 1)                                                     # the week's model: 6 images of each digit
    ax.imshow(np.vstack([np.hstack([F[yfix == d][r].reshape(28, 28) for d in range(10)]) for r in range(6)]), cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off"); ax.set_title(f"conditional DCGAN, epoch {log[-1][0]}: columns asked for 0-9", fontsize=10)
    ax = fig.add_subplot(3, 3, 2); Fg = gs[("full", 1e-3)][2]; lab = np.tile(np.arange(10), 200)
    ax.imshow(np.vstack([np.hstack([Fg[lab == d][r].reshape(28, 28) for d in range(10)]) for r in range(6)]), cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off"); ax.set_title("Day 133's Gaussian per digit (full covariance, reg 0.001)", fontsize=10)
    ax = fig.add_subplot(3, 3, 3); names = ["real test", "cDCGAN"] + [f"{k}, {r:g}" for k, r in gs]
    vals = [real[1], log[-1][1][1]] + [gs[k][0][1] for k in gs]; ax.bar(range(len(vals)), vals, color=["gray", "#4C72B0"] + ["#C44E52"] * len(gs))
    ax.set_yscale("log"); ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=8, rotation=20); ax.set_ylabel("FID (log scale)")
    ax.set_title("FID, 2,000 images each", fontsize=10)
    ax = fig.add_subplot(3, 3, 4); w = 0.4
    ax.bar(np.arange(10) - w / 2, gpd, w, label="cDCGAN"); ax.bar(np.arange(10) + w / 2, rpd, w, label="real test digits", color="gray")
    ax.set_xticks(range(10)); ax.set_xlabel("digit"); ax.set_ylabel("FID, 200 images"); ax.set_title("per-digit FID", fontsize=10); ax.legend(fontsize=8)
    ax = fig.add_subplot(3, 3, 5)                                                     # the ring, noised
    for j, (t, m, g, fd, xt) in enumerate([r for r in rows if r[0] in (0, 50, 200, 1000)]):
        ax.scatter(xt[:, 0] + 7 * j, xt[:, 1], s=1, alpha=0.4)
        ax.text(7 * j, 4.2, f"t = {t}\n{m} modes, good {g:.2f}", ha="center", fontsize=8)
    ax.set_xlim(-4, 25); ax.set_ylim(-4, 5.5); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("the ring, noised (linear schedule)", fontsize=10)
    ax = fig.add_subplot(3, 3, 6); ts = np.arange(1001)
    for kind in ("linear", "cosine"):
        ax.plot(ts, np.sqrt(schedule(kind)[1]), label=f"{kind}: signal sqrt(abar)")
    ax.set_xlabel("t"); ax.set_ylabel("share of the signal left"); ax.set_title("two noise schedules", fontsize=10); ax.legend(fontsize=8)
    ax = fig.add_subplot(3, 3, 7)                                                     # one digit through the process, both schedules
    ax.imshow(np.vstack([np.hstack([r[5].reshape(28, 28) for r in res[k]]) for k in ("linear", "cosine")]), cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off"); ax.set_title("one test digit at t = " + ", ".join(str(r[0]) for r in res["linear"]) + "\n(top: linear, bottom: cosine)", fontsize=8)
    ax = fig.add_subplot(3, 3, 8)
    for kind, ls in (("linear", "-"), ("cosine", "--")):
        r = res[kind]; ax.plot([x[0] for x in r], [x[2] for x in r], ls=ls, marker="o", label=f"{kind}, clean-trained")
        ax.plot([x[0] for x in r], [x[3] for x in r], ls=ls, marker="s", label=f"{kind}, noise-trained")
    ax.axhline(0.1, color="gray", ls=":", lw=1); ax.set_xlabel("t"); ax.set_ylabel("accuracy"); ax.set_title("is it still a digit?", fontsize=10); ax.legend(fontsize=7)
    ax = fig.add_subplot(3, 3, 9)
    for kind in ("linear", "cosine"):
        r = res[kind]; ax.plot([x[0] for x in r], [x[4] for x in r], marker="o", label=kind)
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("FID against clean digits"); ax.set_title("FID of noised test digits", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the week's parts, and the forward process
    fig, ax = plt.subplots(figsize=(13, 5.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
    parts = ["133: Gaussian\nclass models", "134: losses,\nAdamGAN", "135: ring,\nmode_stats", "136: Conv,\nConvT, BN", "137: label to\nboth players", "138: IS, FID,\nprecision/recall"]
    for i, p in enumerate(parts):
        box(1.1 + 2.15 * i, 4.6, 1.85, 0.9, p, "#DDEEFF")
    box(4.3, 2.9, 3.4, 1.0, "conditional DCGAN\n(134 + 136 + 137), 60,000 images", "#F8CBAD")
    box(8.9, 2.9, 3.4, 1.0, "scored with 138's metrics,\nagainst 133's Gaussians", "#FFF2CC")
    for x in (3.25, 7.55, 9.7):
        arrow(x, 4.12, 4.3 if x < 5 else 8.9, 3.42)
    arrow(1.1, 4.12, 7.8, 3.42); arrow(12.0, 4.12, 9.4, 3.42); arrow(6.0, 2.9, 7.2, 2.9)
    box(1.6, 1.0, 2.4, 0.9, "x_0: a digit,\nor the ring", "#E2F0D9")
    box(5.3, 1.0, 3.8, 0.9, "x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps\n(1,000 small steps, or one jump)", "#E2F0D9")
    box(9.5, 1.0, 2.4, 0.9, "x_1000: almost\npure N(0, I)", "#E2F0D9")
    box(12.0, 1.0, 1.6, 0.9, "Day 140:\nrun it back", "#EEEEEE")
    arrow(2.8, 1.0, 3.4, 1.0); arrow(7.2, 1.0, 8.3, 1.0); arrow(10.7, 1.0, 11.2, 1.0)
    ax.text(0.2, 1.75, "the forward process (today's second half)", fontsize=9, style="italic")
    ax.set_title("Day 139: the week's parts as one model, and the start of diffusion", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "week20_build_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 140 (Week 21) runs the process backwards: starting from pure noise, remove a little noise at a time. With the")
    print("ring, where the best possible denoiser can be written down exactly, and then with one learned from data.")


def main():                                             # runs every section in order
    demo_checks()                                                                     # 1. verbatim parts; the shortcut formula
    X, y = d126m.mnist()                                                              # 70,000 images, pixels in [0, 1], float32
    week = demo_week(X, y)                                                            # 2. the week's model
    gs = demo_gaussians(X, y, week[4])                                                # 3. against Day 133's Gaussians
    rows = demo_ring()                                                                # 4. the ring, noised
    res = demo_mnist_noise(X, y, week[4])                                             # 5. digits, noised
    plot_results(week, gs, rows, res, os.path.join(HERE, "week20_build_results.png"))
    draw_diagram()                                                                    # the week, and the forward process
    print("\nsaved week20_build_results.png\nsaved week20_build_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
