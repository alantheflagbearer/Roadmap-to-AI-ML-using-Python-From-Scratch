"""
Day 131 Exercise -- Views and the Contrastive Loss

The sixth of Week 19's parts.

    augment(X, rng, ...)              a random view of each 28 x 28 image: rotate, shift, erase, noise
    nt_xent(Z, tau)                   SimCLR's contrastive loss for 2N projections, and its gradient
    train_contrastive(X, sizes, ...)  two views per image, pulled together, every other image pushed away
    represent(net, X)                 the representation: the layer before the projection head

Checked by hand: a view with no changes is the image itself and keeps a single
pixel where it was, the loss for two images by hand, the gradient
against finite differences, and a loss that falls in training.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import os                                              # os: paths
import sys                                             # sys: path, to import Day 126's parts
import contextlib                                      # contextlib: keep the import quiet
import io                                              # io: where its output goes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam


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


rng = np.random.RandomState(0); X = rng.rand(4, 784).astype(np.float32)
print(f"a view with every change switched off equals the image: {bool(np.allclose(augment(X, rng, shift=0, angle=0, erase=0, noise=0), X))}")
img = np.zeros((1, 784), np.float32); img[0, 3 * 28 + 13] = 1.0          # one pixel near the top, in column 13
V = augment(img, rng, shift=0, angle=0, erase=0, noise=0); print(f"no change keeps the pixel at row 3, column 13: {bool(V[0, 3 * 28 + 13] == 1)}")
Z = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, -0.8]]); loss, dZ, p = nt_xent(Z, 0.5)
print(f"two images, four views: loss {loss:.4f}; row 0 picks its partner with probability {p[0]:.4f}; "
      f"by hand e^1.6 / (e^0 + e^1.6 + e^1.2) = {np.exp(1.6) / (1 + np.exp(1.6) + np.exp(1.2)):.4f}")
Zr = rng.randn(6, 3); _, dZ, _ = nt_xent(Zr, 0.5); num = np.zeros_like(Zr)
for idx in np.ndindex(Zr.shape):
    o = Zr[idx]; Zr[idx] = o + 1e-6; lp = nt_xent(Zr, 0.5)[0]; Zr[idx] = o - 1e-6; lm = nt_xent(Zr, 0.5)[0]; Zr[idx] = o; num[idx] = (lp - lm) / 2e-6
print(f"gradient against finite differences: largest gap {np.abs(num - dZ).max():.1e}")
D = (rng.rand(64, 784) < 0.2).astype(np.float32)
net, hist = train_contrastive(D, [784, 32, 16], tau=0.5, epochs=8, batch=32, seed=0)
print(f"64 random images, a small network: loss {hist[0]:.3f} after epoch 1, {hist[-1]:.3f} after 8; representation shape {represent(net, D).shape}")
