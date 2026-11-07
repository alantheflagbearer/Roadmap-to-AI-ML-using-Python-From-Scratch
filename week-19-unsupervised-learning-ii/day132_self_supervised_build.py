"""
Day 132: The Week 19 Build -- a Self-Supervised Pretext Task, and Every Representation of the Week

Self-supervised learning invents a task whose answers come free with the data,
and hopes that solving it teaches something useful. Today's pretext task: turn
each digit by 0, 90, 180 or 270 degrees and ask the network which. To answer,
it must know which way up a digit goes, and so what digits look like.

The week's six exercise parts are copied in verbatim (one banner per file):
the network and its training (Day 126), damage (127), anomaly scores (128),
the VAE (129, 130) and contrastive learning (131). Together with today's
rotation task they give five ways to learn a representation without labels,
compared here on the same footing: a linear classifier given only a few
labelled digits.

  1. Checks: the parts are verbatim.
  2. The pretext task: predicting rotations, and the digits that defeat it.
  3. Five self-supervised representations, judged with few labels.
  4. Robustness: the same classifiers on damaged test images.
  5. Anomalies: which representation notices a digit turned upside down.

Data: MNIST (cached by Day 126); labels are used only to judge.

Every line written for today carries its own comment, continuing Day 52's
convention; the copied parts keep their own.
"""
import ast                                             # ast: reading function sources out of the exercise files
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier weeks' parts
import time                                            # time: what each method costs

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
for w in ("week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' exercise parts
with contextlib.redirect_stdout(io.StringIO()):         # their imports print their own checks; keep them quiet
    import day123_exercise as d123                      # Day 123: pca_fit, transform, reconstruct (Day 128's part uses it)
    import day119_exercise as d119                      # Day 119: sqdist, kmeans, ari (Days 128 and 130's parts use it)
    import day117_exercise as d117                      # Day 117: roc_auc
    import day116_exercise as d116                      # Day 116: fit_softmax
d126 = sys.modules[__name__]                            # the copied parts call Day 126's functions as d126.forward etc.: here they live in this file
ENC = ["relu", "linear"]; DEC = ["relu", "linear"]      # Day 130's part needs these, as on Day 130
SIZES_AE = [784, 256, 128, 256, 784]; ACTS_AE = ["relu", "linear", "relu", "linear"]  # every representation has 128 numbers


# =============================================================================
# from day126_exercise.py, verbatim: init_net, forward, backward, sigmoid, recon_loss, Adam, train_ae, reconstruct
# =============================================================================
def init_net(sizes, seed):                              # NEW: Day 126's version, unchanged
    rng = np.random.RandomState(seed)
    return [((rng.randn(a, b) * np.sqrt(2.0 / a)).astype(np.float32), np.zeros(b, np.float32)) for a, b in zip(sizes[:-1], sizes[1:])]


def forward(net, X, acts):                              # NEW: Day 126's version, unchanged
    A = [X]
    for (W, b), act in zip(net, acts):
        Z = A[-1] @ W + b
        A.append(np.maximum(Z, 0) if act == "relu" else Z)  # "linear" keeps Z: the last layer gives logits or values
    return A


def backward(net, A, acts, dout):                       # NEW: Day 126's version, unchanged
    grads = [None] * len(net)
    for i in range(len(net) - 1, -1, -1):
        dZ = dout * (A[i + 1] > 0) if acts[i] == "relu" else dout  # through ReLU: zero where the unit was off
        grads[i] = (A[i].T @ dZ, dZ.sum(0))
        dout = dZ @ net[i][0].T                             # back through the weights to the layer below
    return grads, dout


def sigmoid(Z):                                         # 1 / (1 + e^-z), in a form that cannot overflow
    return 0.5 * (1 + np.tanh(0.5 * Z))


def recon_loss(L, X, kind):                             # NEW: Day 126's version, unchanged
    n = len(X)
    if kind == "bce":                                     # pixels in [0, 1] as probabilities; L are logits
        return float((np.logaddexp(0, L) - X * L).sum() / n), (sigmoid(L) - X) / n
    return float(((L - X) ** 2).sum() / n), 2 * (L - X) / n  # squared error, summed over pixels


class Adam:                                             # NEW: Day 126's version, unchanged
    def __init__(self, net, lr=1e-3):
        self.lr, self.t = lr, 0
        self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in net]; self.v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in net]

    def step(self, net, grads):
        self.t += 1; c1, c2 = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for (W, b), (gW, gb), m, v in zip(net, grads, self.m, self.v):
            for p, g, mm, vv in ((W, gW, m[0], v[0]), (b, gb, m[1], v[1])):
                mm *= 0.9; mm += 0.1 * g; vv *= 0.999; vv += 0.001 * g * g
                p -= self.lr * (mm / c1) / (np.sqrt(vv / c2) + 1e-8)  # in place, so the network itself changes


def train_ae(X, sizes, acts, kind="bce", epochs=20, batch=128, lr=1e-3, seed=0, Xval=None, noise=None):  # NEW: Day 126's version, unchanged
    net = init_net(sizes, seed); opt = Adam(net, lr); rng = np.random.RandomState(seed + 1); hist = []
    for _ in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            xb = X[o[s:s + batch]]; xin = xb if noise is None else noise(xb, rng)  # noise: Day 127's corruption, if any
            A = forward(net, xin, acts); _, d = recon_loss(A[-1], xb, kind)
            opt.step(net, backward(net, A, acts, d)[0])
        hist.append((recon_loss(forward(net, X[:2000], acts)[-1], X[:2000], kind)[0],
                     None if Xval is None else recon_loss(forward(net, Xval, acts)[-1], Xval, kind)[0]))
    return net, hist


def reconstruct(net, X, acts, kind="bce"):              # NEW: Day 126's version, unchanged
    L = forward(net, X, acts)[-1]
    return sigmoid(L) if kind == "bce" else L


# =============================================================================
# from day127_exercise.py, verbatim: gaussian, masking, hide_square, psnr, median3
# =============================================================================
def gaussian(X, sigma, rng):                            # NEW: Day 127's version, unchanged
    return np.clip(X + rng.normal(0, sigma, X.shape).astype(np.float32), 0, 1)


def masking(X, p, rng):                                 # NEW: Day 127's version, unchanged
    return X * (rng.uniform(size=X.shape) >= p).astype(np.float32)


def hide_square(X, size=10, rng=None):                  # NEW: Day 127's version, unchanged
    rng = rng or np.random.RandomState(0); Y = X.reshape(-1, 28, 28).copy(); M = np.zeros_like(Y, bool)
    for i, (r, c) in enumerate(rng.randint(4, 24 - size, size=(len(Y), 2))):  # squares stay away from the empty border
        Y[i, r:r + size, c:c + size] = 0; M[i, r:r + size, c:c + size] = True
    return Y.reshape(len(X), -1), M.reshape(len(X), -1)


def psnr(R, X):                                         # NEW: Day 127's version, unchanged
    return float(10 * np.log10(1.0 / np.mean((R - X) ** 2)))


def median3(X):                                         # NEW: Day 127's version, unchanged
    P = np.pad(X.reshape(-1, 28, 28), ((0, 0), (1, 1), (1, 1)))
    stack = np.stack([P[:, i:i + 28, j:j + 28] for i in range(3) for j in range(3)])  # nine shifted copies
    return np.median(stack, 0).reshape(len(X), -1)


# =============================================================================
# from day128_exercise.py, verbatim: ae_score, pca_score, nn_score, threshold, alarm_stats
# =============================================================================
def ae_score(net, X, acts, kind="bce"):                 # NEW: Day 128's version, unchanged
    return ((d126.reconstruct(net, X, acts, kind) - X) ** 2).sum(1)


def pca_score(X, pca, k):                               # NEW: Day 128's version, unchanged
    mu, V = pca
    return ((d123.reconstruct(d123.transform(X, mu, V, k), mu, V, k) - X) ** 2).sum(1)


def nn_score(X, ref, chunk=2000):                       # NEW: Day 128's version, unchanged
    return np.concatenate([np.sqrt(np.maximum(d119.sqdist(X[s:s + chunk], ref), 0).min(1)) for s in range(0, len(X), chunk)])


def threshold(normal_scores, false_alarm):              # NEW: Day 128's version, unchanged
    return float(np.quantile(normal_scores, 1 - false_alarm))


def alarm_stats(scores, y, t):                          # NEW: Day 128's version, unchanged
    flag = scores > t; tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    return {"recall": tp / max((y == 1).sum(), 1), "false alarms": fp / max((y == 0).sum(), 1), "precision": tp / max(flag.sum(), 1)}


# =============================================================================
# from day129_exercise.py, verbatim: reparam, kl_normal, vae_loss, train_vae, score_grad, reparam_grad
# =============================================================================
def reparam(mu, logvar, rng):                           # NEW: Day 129's version, unchanged
    eps = rng.standard_normal(mu.shape).astype(mu.dtype)
    return mu + np.exp(0.5 * logvar) * eps, eps


def kl_normal(mu, logvar):                              # NEW: Day 129's version, unchanged
    return 0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar).sum(1)


def vae_loss(enc, dec, X, rng, beta=1.0, acts_e=("relu", "linear"), acts_d=("relu", "linear"), eps=None):  # NEW: Day 129's version, unchanged
    n = len(X); H = d126.forward(enc, X, list(acts_e)); out = H[-1]; k = out.shape[1] // 2
    mu, logvar = out[:, :k], out[:, k:]                    # the encoder's last layer gives both halves
    if eps is None:
        z, eps = reparam(mu, logvar, rng)
    else:
        z = mu + np.exp(0.5 * logvar) * eps                 # a fixed eps, for the gradient check
    A = d126.forward(dec, z, list(acts_d)); rec, dL = d126.recon_loss(A[-1], X, "bce")  # reconstruction: cross-entropy per image
    kl = float(kl_normal(mu, logvar).mean())
    gdec, dz = d126.backward(dec, A, list(acts_d), dL)    # back through the decoder, down to z
    dmu = dz + beta * mu / n                               # z = mu + sigma eps, so dz/dmu = 1; plus the KL term's own gradient
    dlogvar = dz * eps * 0.5 * np.exp(0.5 * logvar) + beta * 0.5 * (np.exp(logvar) - 1) / n  # dz/dlogvar = eps sigma / 2
    genc, _ = d126.backward(enc, H, list(acts_e), np.hstack([dmu, dlogvar]))
    return rec + beta * kl, rec, kl, genc, gdec


def train_vae(X, k, hidden=256, epochs=10, batch=128, lr=1e-3, beta=1.0, seed=0, Xval=None):  # NEW: Day 129's version, unchanged
    enc = d126.init_net([X.shape[1], hidden, 2 * k], seed); dec = d126.init_net([k, hidden, X.shape[1]], seed + 1)
    oe, od = d126.Adam(enc, lr), d126.Adam(dec, lr); rng = np.random.RandomState(seed + 2); hist = []
    for _ in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            _, _, _, ge, gd = vae_loss(enc, dec, X[o[s:s + batch]], rng, beta)
            oe.step(enc, ge); od.step(dec, gd)
        ev = X[:2000] if Xval is None else Xval
        hist.append(vae_loss(enc, dec, ev, np.random.RandomState(99), beta)[:3])  # (loss, reconstruction, KL) on held-out images
    return enc, dec, hist


def score_grad(f, mu, sigma, rng, n):                   # NEW: Day 129's version, unchanged
    z = mu + sigma * rng.standard_normal(n)
    return f(z) * (z - mu) / sigma ** 2, f(z) * ((z - mu) ** 2 / sigma ** 3 - 1 / sigma)  # f(z) times d log q(z) / d parameter


def reparam_grad(df, mu, sigma, rng, n):                # NEW: Day 129's version, unchanged
    eps = rng.standard_normal(n); z = mu + sigma * eps
    return df(z), df(z) * eps                              # chain rule through z = mu + sigma eps


# =============================================================================
# from day130_exercise.py, verbatim: encode, decode, sample_prior, interpolate, nearest_real, active_units
# =============================================================================
def encode(enc, X):                                     # NEW: Day 130's version, unchanged
    H = d126.forward(enc, X, ENC)[-1]; k = H.shape[1] // 2
    return H[:, :k], H[:, k:]


def decode(dec, Z):                                     # NEW: Day 130's version, unchanged
    return d126.sigmoid(d126.forward(dec, Z.astype(np.float32), DEC)[-1])


def sample_prior(dec, n, k, rng):                       # NEW: Day 130's version, unchanged
    return decode(dec, rng.standard_normal((n, k)))


def interpolate(z0, z1, steps):                         # NEW: Day 130's version, unchanged
    t = np.linspace(0, 1, steps)[:, None]
    return (1 - t) * z0 + t * z1


def nearest_real(R, ref, chunk=500):                    # NEW: Day 130's version, unchanged
    return np.concatenate([np.sqrt(np.maximum(d119.sqdist(R[s:s + chunk].astype(np.float64), ref), 0).min(1) / R.shape[1])
                           for s in range(0, len(R), chunk)])


def active_units(enc, X, thresh=0.1):                   # NEW: Day 130's version, unchanged
    mu, lv = encode(enc, X); kl = (0.5 * (np.exp(lv) + mu ** 2 - 1 - lv)).mean(0)
    return int((kl > thresh).sum()), kl


# =============================================================================
# from day131_exercise.py, verbatim: augment, nt_xent, train_contrastive, represent
# =============================================================================
def augment(X, rng, shift=3, angle=15, erase=8, noise=0.1):  # NEW: Day 131's version, unchanged
    n = len(X); a = np.radians(rng.uniform(-angle, angle, n)); dx, dy = rng.randint(-shift, shift + 1, (2, n))
    r, c = np.mgrid[0:28, 0:28].reshape(2, -1) - 13.5      # pixel coordinates around the centre
    cos, sin = np.cos(a)[:, None], np.sin(a)[:, None]
    src_r = np.rint(cos * r + sin * c + 13.5 - dy[:, None]).astype(int)  # for each output pixel, where it comes from:
    src_c = np.rint(-sin * r + cos * c + 13.5 - dx[:, None]).astype(int)  # rotated and shifted back (nearest pixel)
    ok = (src_r >= 0) & (src_r < 28) & (src_c >= 0) & (src_c < 28)
    V = np.where(ok, X[np.arange(n)[:, None], np.clip(src_r, 0, 27) * 28 + np.clip(src_c, 0, 27)], 0).astype(np.float32)
    if erase:                                             # blank a random erase x erase patch in half of the images
        r0, c0 = rng.randint(0, 28 - erase, (2, n)); hit = rng.uniform(size=n) < 0.5
        V = V.reshape(n, 28, 28)
        for i in np.where(hit)[0]:
            V[i, r0[i]:r0[i] + erase, c0[i]:c0[i] + erase] = 0
        V = V.reshape(n, -1)
    return np.clip(V + rng.normal(0, noise, V.shape).astype(np.float32), 0, 1)


def nt_xent(Z, tau):                                    # NEW: Day 131's version, unchanged
    n2 = len(Z); N = n2 // 2; norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12; U = Z / norm  # unit vectors
    S = U @ U.T / tau; np.fill_diagonal(S, -np.inf)       # cosine similarities over the temperature; never compare a row with itself
    pos = np.r_[np.arange(N, n2), np.arange(N)]           # each row's partner: the other view of the same image
    m = S.max(1, keepdims=True); E = np.exp(S - m); P = E / E.sum(1, keepdims=True)  # softmax over all other rows
    loss = float(np.mean(-np.log(P[np.arange(n2), pos] + 1e-30)))
    dS = P.copy(); dS[np.arange(n2), pos] -= 1; dS /= n2   # the softmax gradient: probability minus target
    dU = (dS + dS.T) @ U / tau                            # S is symmetric in U, so both halves contribute
    dZ = (dU - U * (U * dU).sum(1, keepdims=True)) / norm  # through the normalisation: remove the part along U
    return loss, dZ, P[np.arange(n2), pos]


def train_contrastive(X, sizes, tau=0.5, epochs=20, batch=256, lr=1e-3, seed=0, aug=None):  # NEW: Day 131's version, unchanged
    acts = ["relu"] * (len(sizes) - 2) + ["linear"]; net = d126.init_net(sizes, seed); opt = d126.Adam(net, lr)
    rng = np.random.RandomState(seed + 1); aug = aug or augment; hist = []
    for _ in range(epochs):
        o = rng.permutation(len(X)); losses = []
        for s in range(0, len(X) - batch + 1, batch):       # full batches only: every batch has the same number of negatives
            xb = X[o[s:s + batch]]; V = np.vstack([aug(xb, rng), aug(xb, rng)])
            A = d126.forward(net, V, acts); loss, dZ, _ = nt_xent(A[-1], tau)
            opt.step(net, d126.backward(net, A, acts, dZ)[0]); losses.append(loss)
        hist.append(float(np.mean(losses)))
    return net, hist


def represent(net, X, layer=-2):                        # NEW: Day 131's version, unchanged
    acts = ["relu"] * (len(net) - 1) + ["linear"]
    return d126.forward(net, X, acts)[layer]


# =============================================================================
# DATA
# =============================================================================
def mnist():                                            # Day 126's cached MNIST loader, unchanged (it lives in Day 126's main script)
    p = os.path.join(os.path.expanduser("~"), "scikit_learn_data", "mnist_784_uint8.npz")  # the cache, outside the repository
    if not os.path.exists(p):                           # the first run downloads
        from sklearn.datasets import fetch_openml       # only needed once
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")  # about a minute the first time
        np.savez_compressed(p, X=d.data.astype(np.uint8), y=d.target.astype(np.uint8))  # small, fast to load next time
    z = np.load(p)                                      # 70,000 images and their digits
    return (z["X"] / 255.0).astype(np.float32), z["y"].astype(int)  # float32: about 3 times faster here than float64


# =============================================================================
# THE PRETEXT TASK -- NEW
# =============================================================================
def rotate4(X):                                         # NEW: every image turned by 0, 90, 180 and 270 degrees, and which turn it was
    I = X.reshape(-1, 28, 28)                           # back to 28 x 28 pictures
    return np.vstack([np.rot90(I, k, axes=(1, 2)).reshape(len(X), -1) for k in range(4)]), np.repeat(np.arange(4), len(X))


def softmax_ce(L, y):                                   # NEW: cross-entropy for class scores, and its gradient
    m = L.max(1, keepdims=True); P = np.exp(L - m); P /= P.sum(1, keepdims=True); n = len(y)  # probabilities, without overflow
    return float(-np.log(P[np.arange(n), y] + 1e-30).mean()), (P - np.eye(L.shape[1], dtype=L.dtype)[y]) / n  # gradient: P minus the answer


def train_rotation(X, sizes, epochs=10, batch=128, lr=1e-3, seed=0):  # NEW: Day 126's network and Adam, trained to name the turn
    acts = ["relu"] * (len(sizes) - 2) + ["linear"]; net = init_net(sizes, seed); opt = Adam(net, lr)  # ReLU layers, then 4 scores
    R, r = rotate4(X); rng = np.random.RandomState(seed + 1); hist = []  # every image in all four turns: the labels come free
    for _ in range(epochs):
        o = rng.permutation(len(R)); losses = []          # a fresh order each pass
        for s in range(0, len(R), batch):
            A = forward(net, R[o[s:s + batch]], acts); loss, d = softmax_ce(A[-1], r[o[s:s + batch]])  # how wrong about the turn
            opt.step(net, backward(net, A, acts, d)[0]); losses.append(loss)  # Day 126's backward pass and Adam
        hist.append(float(np.mean(losses)))               # the average training loss this epoch
    return net, hist


# =============================================================================
# JUDGING A REPRESENTATION WITH FEW LABELS -- NEW
# =============================================================================
def probe_fit(R, y, n_per, seed=0):                     # NEW: Day 116's softmax regression on n_per labelled images per digit
    rng = np.random.RandomState(seed)                   # which labelled images: a fixed random draw
    tr = np.concatenate([rng.choice(np.where(y[:60000] == c)[0], n_per, replace=False) for c in range(10)])  # n_per of each digit
    mu, sd = R[tr].mean(0), R[tr].std(0) + 1e-6        # scale each feature, from the labelled images alone
    W, b = d116.fit_softmax(((R[tr] - mu) / sd).astype(np.float64), y[tr], 10, lam=1e-2, steps=500)  # Day 116, unchanged
    return lambda Z: (((Z - mu) / sd) @ W + b).argmax(1)  # a classifier: images' representations in, digits out


def knn_acc(R, y, tr, te, k=3):                         # kNN accuracy with labelled reference rows tr
    nn = np.argsort(d119.sqdist(R[te].astype(np.float64), R[tr].astype(np.float64)), 1)[:, :k]  # the k closest labelled rows
    return float(np.mean(np.array([np.bincount(r, minlength=10).argmax() for r in y[tr][nn]]) == y[te]))  # majority vote, scored


# =============================================================================
# DEMOS
# =============================================================================
def demo_checks():
    print("=" * 70)
    print("1. CHECKS -- the parts are verbatim, and the new pieces are right")
    print("=" * 70)
    me = open(os.path.abspath(__file__), encoding="utf-8").read(); found, total = 0, 0  # this file's own text
    for d in range(126, 132):                                                        # the week's six exercise files
        src = open(os.path.join(HERE, f"day{d}_exercise.py"), encoding="utf-8").read(); lines = src.split("\n")
        for node in ast.parse(src).body:                                             # every function and class at the top level
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total += 1; found += "\n".join(lines[node.lineno - 1:node.end_lineno]) in me  # its exact text, found in this file?
    print(f"functions and classes from day126_exercise.py to day131_exercise.py found here, character for character: {found} of {total}")
    img = np.arange(784, dtype=np.float32).reshape(1, 784); R, r = rotate4(img)       # one image, four turns
    back = rotate4(R[3:4])[0][1]                                                      # the 270-degree copy, turned 90 degrees more
    print(f"rotate4: labels {r.tolist()}; turning the 270-degree copy a further 90 degrees gives the original back: {bool(np.array_equal(back, img[0]))}")
    L = np.array([[2.0, 0.0, 0.0, 0.0]]); loss, d = softmax_ce(L, np.array([0]))      # one row of scores, the answer is class 0
    print(f"softmax_ce for scores (2, 0, 0, 0), answer 0: {loss:.4f}; by hand -log(e^2 / (e^2 + 3)) = {-np.log(np.exp(2) / (np.exp(2) + 3)):.4f}")
    rng = np.random.RandomState(0); Ls = rng.randn(5, 4); ys = rng.randint(0, 4, 5)   # random scores and answers
    _, g = softmax_ce(Ls, ys); num = np.zeros_like(Ls)                                # the formula's gradient, and room for the numerical one
    for idx in np.ndindex(Ls.shape):                                                  # nudge every score both ways
        o = Ls[idx]; Ls[idx] = o + 1e-6; lp = softmax_ce(Ls, ys)[0]                    # up
        Ls[idx] = o - 1e-6; lm = softmax_ce(Ls, ys)[0]; Ls[idx] = o                    # down, then put it back
        num[idx] = (lp - lm) / 2e-6                                                   # the slope between them
    print(f"its gradient against finite differences: largest gap {np.abs(num - g).max():.1e}")
    print("Actual output from this lesson's run.")


def demo_pretext(X, y):
    print("\n" + "=" * 70)
    print("2. THE PRETEXT TASK -- which way was this digit turned?")
    print("=" * 70)
    t0 = time.perf_counter(); net, hist = train_rotation(X[:20000], [784, 256, 128, 4], epochs=10); t = time.perf_counter() - t0  # 80,000 turned images a pass
    print(f"network 784 -> 256 -> 128 -> 4, trained on 20,000 images x 4 turns, 10 epochs ({t:.0f} s); no digit labels used")
    print(f"training loss by epoch: {', '.join(f'{h:.3f}' for h in hist)} (guessing among 4 turns: {np.log(4):.3f})")
    te = np.arange(60000, 70000); R, r = rotate4(X[te])                                # 40,000 turned test images
    pred = forward(net, R, ["relu", "relu", "linear"])[-1].argmax(1)                   # the turn the network names
    yy = np.tile(y[te], 4); ok = pred == r                                            # each turned copy keeps its digit's label
    print(f"10,000 test digits x 4 turns: the turn named correctly {ok.mean():.4f} of the time")
    print("   digit  correct  most common mistake")
    rows = []                                                                         # (digit, accuracy)
    for c in range(10):                                                               # one row per digit
        m = yy == c; wrong = (~ok) & m                                                # this digit's copies, and its mistakes
        pairs = np.bincount(r[wrong] * 4 + pred[wrong], minlength=16).reshape(4, 4)   # true turn x named turn, mistakes only
        a, b = np.unravel_index(pairs.argmax(), (4, 4)); rows.append((c, float(ok[m].mean())))  # the commonest mistake
        print(f"   {c:5d}  {ok[m].mean():.4f}   {90 * a:3d} called {90 * b:3d} ({pairs[a, b] / m.sum():.3f} of this digit's turned copies)")
    worst = sorted(rows, key=lambda t: t[1])[:3]                                      # the three hardest digits
    print(f"hardest digits: {', '.join(f'{c} ({a:.3f})' for c, a in worst)}")
    s69, s96 = (yy == 6) & (r == 2), (yy == 9) & (r == 2)                             # sixes and nines turned 180 degrees
    print(f"a 6 turned 180 degrees looks like an upright 9: named 'no turn' {np.mean(pred[s69] == 0):.3f} of the time; a 9 turned 180, {np.mean(pred[s96] == 0):.3f}")
    print("Actual output from this lesson's run.")
    return net, hist, rows


def train_all(X, rot_net):                              # every representation, learned from the same 20,000 unlabelled images
    reps, secs = {}, {}                                 # name -> function from images to representation; name -> training seconds
    mu, vals, V = d123.pca_fit(X[:20000].astype(np.float64))                           # Day 123's PCA
    for k in (16, 128):                                                               # PCA at the VAE's size and at everyone else's
        reps[f"PCA {k}"] = lambda Z, k=k: d123.transform(Z.astype(np.float64), mu, V, k); secs[f"PCA {k}"] = None
    t0 = time.perf_counter(); ae, _ = train_ae(X[:20000], SIZES_AE, ACTS_AE, "bce", epochs=20, seed=0)  # Day 126's autoencoder
    secs["autoencoder 128"] = time.perf_counter() - t0; reps["autoencoder 128"] = lambda Z: forward(ae, Z, ACTS_AE)[2]  # the middle layer
    t0 = time.perf_counter()                                                          # Day 127: trained to rebuild from images with half the pixels dropped
    dae, _ = train_ae(X[:20000], SIZES_AE, ACTS_AE, "bce", epochs=20, seed=0, noise=lambda xb, rng: masking(xb, 0.5, rng))
    secs["denoising AE 128"] = time.perf_counter() - t0; reps["denoising AE 128"] = lambda Z: forward(dae, Z, ACTS_AE)[2]
    t0 = time.perf_counter(); enc, dec, _ = train_vae(X[:20000], 16, epochs=30, Xval=X[60000:62000])  # Day 130's VAE, 16 numbers
    secs["VAE means 16"] = time.perf_counter() - t0; reps["VAE means 16"] = lambda Z: encode(enc, Z)[0]  # the code means
    t0 = time.perf_counter(); con, _ = train_contrastive(X[:20000], [784, 256, 128, 64], 0.1, epochs=30)  # Day 131's settings
    secs["contrastive 128"] = time.perf_counter() - t0; reps["contrastive 128"] = lambda Z: represent(con, Z)  # before the projection head
    reps["rotation 128"] = lambda Z: represent(rot_net, Z); secs["rotation 128"] = "sec. 2"  # section 2's network, before its 4-way head
    reps["rotation, layer 1"] = lambda Z: forward(rot_net, Z, ["relu"])[1]; secs["rotation, layer 1"] = "sec. 2"  # its first layer: 256 numbers
    t0 = time.perf_counter(); rot1, _ = train_rotation(X[:20000], [784, 256, 128, 4], epochs=1)  # the same task, stopped after 1 epoch
    secs["rotation, 1 epoch"] = time.perf_counter() - t0; reps["rotation, 1 epoch"] = lambda Z: represent(rot1, Z)
    rnd = init_net([784, 256, 128, 4], 0); reps["untrained net 128"] = lambda Z: represent(rnd, Z)  # the rotation network's starting weights
    secs["untrained net 128"] = None                                                  # a baseline: what the shape alone gives
    return reps, secs, ae


def demo_compare(X, y, reps, secs):
    print("\n" + "=" * 70)
    print("3. FIVE SELF-SUPERVISED REPRESENTATIONS, JUDGED WITH FEW LABELS")
    print("=" * 70)
    print("all learned from the same 20,000 unlabelled images; tests use images 60,000-69,999, never trained on")
    print("kNN: 3 neighbours among 5,000 labelled images (20,000-24,999), 2,000 test images")
    print("probes: a linear classifier (Day 116) on 10 or 100 labelled images per digit, tested on 10,000")
    print("k-means: 10 clusters of the 2,000 test images, ARI with the true digits (Day 119)")
    print("baselines: raw pixels, PCA, and an untrained network of the rotation network's shape (its starting weights)")
    allr = {"raw pixels 784": lambda Z: Z}; allr.update(reps)                          # the baseline first
    te, tr, tk = np.arange(60000, 70000), np.arange(20000, 25000), np.arange(60000, 62000)  # probe test, kNN references, kNN/k-means test
    print(f"   {'representation':20s}  kNN     probe 10/digit (3 draws)  probe 100  k-means ARI  train s")
    out = {}                                                                          # name -> (kNN, probe 10, probe 100, ARI)
    for name, f in allr.items():                                                      # one row per representation
        R = np.asarray(f(X)).astype(np.float32)                                       # every image's representation
        kn = knn_acc(R, y, tr, tk)                                                    # nearest neighbours
        p10 = [float(np.mean(probe_fit(R, y, 10, s)(R[te]) == y[te])) for s in range(3)]  # three random draws of 10 per digit
        p100 = float(np.mean(probe_fit(R, y, 100)(R[te]) == y[te]))                   # 100 per digit
        a = d119.ari(d119.kmeans(R[tk].astype(np.float64), 10, 0)[0], y[tk])          # clusters against the digits
        out[name] = (kn, float(np.mean(p10)), p100, a); s = secs.get(name)            # keep the row
        st = "-" if s is None else (s if isinstance(s, str) else f"{s:.0f}")          # PCA needs no training loop; rotation was timed in section 2
        print(f"   {name:20s}  {kn:.4f}  {np.mean(p10):.4f} ({min(p10):.3f}-{max(p10):.3f})     {p100:.4f}     {a:7.4f}      {st}")
    for j, what in ((1, "10 labels per digit"), (2, "100 labels per digit"), (0, "kNN"), (3, "k-means ARI")):  # the winner of each column
        best = max(out, key=lambda n: out[n][j])
        print(f"best by {what}: {best} ({out[best][j]:.4f}); raw pixels {out['raw pixels 784'][j]:.4f}")
    print("Actual output from this lesson's run.")
    return out


def demo_robust(X, y, reps):
    print("\n" + "=" * 70)
    print("4. ROBUSTNESS -- the 100-per-digit probes, tested on damaged images (Day 127's damage)")
    print("=" * 70)
    te = np.arange(60000, 70000); rng = np.random.RandomState(5); C = X[te]          # the clean test images
    dmg = {"clean": C, "noise 0.3": gaussian(C, 0.3, rng), "half pixels gone": masking(C, 0.5, rng),
           "10x10 square hidden": hide_square(C, 10, rng)[0]}                         # the same damage for every representation
    allr = {"raw pixels 784": lambda Z: Z}; allr.update(reps); out = {}               # name -> accuracy under each damage
    print(f"   {'representation':20s}" + "".join(f"  {k:>19s}" for k in dmg))
    for name, f in allr.items():                                                      # one row per representation
        clf = probe_fit(np.asarray(f(X)).astype(np.float32), y, 100)                  # trained on clean labelled images only
        acc = [float(np.mean(clf(np.asarray(f(D)).astype(np.float32)) == y[te])) for D in dmg.values()]  # then tested on each damage
        out[name] = acc; print(f"   {name:20s}" + "".join(f"  {a:19.4f}" for a in acc))
    for j, k in enumerate(list(dmg)[1:], 1):                                          # the most robust under each damage
        best = max(out, key=lambda n: out[n][j]); print(f"best under {k}: {best} ({out[best][j]:.4f})")
    print("caution: the contrastive network trained on views with noise and erased patches (Day 131's augment), and the")
    print("denoising autoencoder on dropped pixels; the others never saw damage")
    print("Actual output from this lesson's run.")
    return list(dmg), out


def demo_upside_down(X, y, reps, ae, rot_net):
    print("\n" + "=" * 70)
    print("5. ANOMALIES -- which representation notices a digit turned upside down?")
    print("=" * 70)
    te = np.arange(60000, 62000); up = X[te]; down = rotate4(up)[0][4000:6000]       # the 180-degree copies of the same 2,000 digits
    Q = np.vstack([up, down]); lab = np.r_[np.zeros(2000), np.ones(2000)].astype(int)  # normal 0, anomaly 1
    keep = ~np.isin(y[te], (0, 1, 8)); kk = np.r_[keep, keep]                          # 0, 1 and 8 are close to symmetric under a half turn, so expected hardest
    ref = np.arange(20000, 25000)                                                     # 5,000 upright training images
    print("normal: 2,000 upright test digits; anomalies: the same digits turned 180 degrees")
    print("score: distance to the nearest of 5,000 upright training images, in each representation (Day 128's nn_score)")
    print(f"   {'score':22s}  AUC all digits  AUC without 0, 1, 8")
    allr = {"raw pixels 784": lambda Z: Z}; allr.update(reps); out = {}
    for name, f in allr.items():                                                      # one row per representation
        s = nn_score(np.asarray(f(Q)).astype(np.float64), np.asarray(f(X[ref])).astype(np.float64))  # far from every upright image: odd
        out[name] = (d117.roc_auc(lab, s), d117.roc_auc(lab[kk], s[kk])); print(f"   {name:22s}  {out[name][0]:.4f}          {out[name][1]:.4f}")
    s = ae_score(ae, Q, ACTS_AE); out["AE rebuild error"] = (d117.roc_auc(lab, s), d117.roc_auc(lab[kk], s[kk]))  # Day 128's score
    print(f"   {'AE rebuild error':22s}  {out['AE rebuild error'][0]:.4f}          {out['AE rebuild error'][1]:.4f}")
    L = forward(rot_net, Q, ["relu", "relu", "linear"])[-1]; P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    s = 1 - P[:, 0]; out["rotation head"] = (d117.roc_auc(lab, s), d117.roc_auc(lab[kk], s[kk]))  # its own doubt that the image is upright
    print(f"   {'rotation head 1-P(0)':22s}  {out['rotation head'][0]:.4f}          {out['rotation head'][1]:.4f}")
    print("Actual output from this lesson's run.")
    return out


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(X, rows, hist, comp, dmg_names, rob, anom, out):  # five panels
    fig = plt.figure(figsize=(15, 9.5))
    ax = fig.add_subplot(2, 3, 1); R, _ = rotate4(X[60000:60006])                     # six test digits, four turns each
    grid = np.ones((4 * 29, 6 * 29))                                                  # white gaps between images
    for k in range(4):
        for i in range(6):
            grid[k * 29:k * 29 + 28, i * 29:i * 29 + 28] = 1 - R[k * 6 + i].reshape(28, 28)  # dark ink on white
    ax.imshow(grid, cmap="gray"); ax.set_yticks([14 + 29 * k for k in range(4)]); ax.set_yticklabels(["0", "90", "180", "270"])
    ax.set_xticks([]); ax.set_ylabel("turn (degrees)"); ax.set_title("the pretext task: name the turn")
    ax = fig.add_subplot(2, 3, 2); ax.bar([str(c) for c, _ in rows], [a for _, a in rows], color="#4C72B0")  # accuracy per digit
    ax.set_ylim(0, 1.05); ax.set_xlabel("digit"); ax.set_ylabel("turn named correctly"); ax.set_title("rotation accuracy by digit (test)")
    for c, a in rows:
        ax.text(c, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax = fig.add_subplot(2, 3, 3); names = list(comp); ypos = np.arange(len(names))   # probe accuracy, 10 and 100 labels per digit
    ax.barh(ypos - 0.2, [comp[n][1] for n in names], 0.4, label="10 per digit", color="#DD8452")
    ax.barh(ypos + 0.2, [comp[n][2] for n in names], 0.4, label="100 per digit", color="#4C72B0")
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("test accuracy of a linear probe"); ax.set_title("few labels: which representation helps?"); ax.legend(fontsize=8, loc="lower right")
    ax = fig.add_subplot(2, 3, 4); show = ["raw pixels 784", "autoencoder 128", "denoising AE 128", "contrastive 128", "rotation 128"]  # robustness
    for n in show:
        ax.plot(range(len(dmg_names)), rob[n], "o-", label=n)
    ax.set_xticks(range(len(dmg_names))); ax.set_xticklabels(dmg_names, fontsize=8); ax.set_ylabel("probe accuracy (100 per digit)")
    ax.set_title("damaged test images"); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 3, 5); an = list(anom); ypos = np.arange(len(an))         # upside-down detection
    ax.barh(ypos, [anom[n][1] for n in an], color="#55A868"); ax.set_yticks(ypos); ax.set_yticklabels(an, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0.4, 1.02); ax.axvline(0.5, color="gray", ls=":"); ax.set_xlabel("AUC, digits other than 0, 1, 8"); ax.set_title("spotting upside-down digits")
    ax = fig.add_subplot(2, 3, 6); ax.plot(range(1, len(hist) + 1), hist, "o-", color="#C44E52")  # pretext training loss
    ax.axhline(np.log(4), color="gray", ls=":", label="guessing (log 4)"); ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy")
    ax.set_title("rotation network training loss"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def draw_diagram():                                     # one pretext task, five representations, the same judges
    fig, ax = plt.subplots(figsize=(13, 5.4)); ax.set_xlim(0, 13); ax.set_ylim(-0.1, 5.1); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
    box(1.3, 2.6, 2.0, 1.0, "20,000 MNIST images\nno labels", "#DDEEFF")
    methods = [("autoencoder: rebuild it (126)", 4.5), ("denoising AE: rebuild\nfrom damage (127)", 3.5), ("VAE: rebuild through\na noisy code (129-130)", 2.5),
               ("contrastive: match two\nviews (131)", 1.5), ("rotation: name the\nturn (today)", 0.5)]
    for text, yy in methods:                                                          # five ways to learn without labels
        box(4.6, yy, 2.9, 0.7, text, "#FFF2CC", 8); arrow(2.35, 2.6, 3.1, yy)
    box(8.0, 2.6, 2.0, 1.2, "a representation\n(16 or 128 numbers\nper image)", "#E2F0D9"); [arrow(6.1, yy, 6.95, 2.6) for _, yy in methods]
    judges = [("probe: 10 or 100 labels\nper digit (116)", 4.3), ("kNN and k-means\n(119)", 3.15), ("damaged test images\n(127)", 2.0), ("upside-down digits\nas anomalies (128)", 0.85)]
    for text, yy in judges:                                                           # the same judges for every method
        box(11.4, yy, 2.6, 0.8, text, "#F8CBAD", 8); arrow(9.05, 2.6, 10.05, yy)
    ax.set_title("Week 19 build: five self-supervised representations, one set of judges", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "self_supervised_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Week 19 learned representations; Week 20 turns to generating new data.")
    print("Day 133 is generative versus discriminative models: modelling how the data is made, against drawing the boundary directly.")


def main():                                             # runs every section in order
    demo_checks()                                                                     # 1. verbatim parts and the new pieces
    X, y = mnist()                                                                    # 70,000 images, pixels in [0, 1]
    rot_net, hist, rows = demo_pretext(X, y)                                          # 2. the rotation task
    reps, secs, ae = train_all(X, rot_net)                                            # the week's representations
    comp = demo_compare(X, y, reps, secs)                                             # 3. few labels
    dmg_names, rob = demo_robust(X, y, reps)                                          # 4. damage
    anom = demo_upside_down(X, y, reps, ae, rot_net)                                  # 5. anomalies
    plot_results(X, rows, hist, comp, dmg_names, rob, anom, os.path.join(HERE, "self_supervised_results.png"))
    draw_diagram()                                                                    # the picture of the build
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
