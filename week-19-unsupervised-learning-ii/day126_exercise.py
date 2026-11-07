"""
Day 126 Exercise -- An Autoencoder's Machinery

The first of Week 19's parts.

    init_net(sizes, seed)            dense layers with He initialisation
    forward(net, X, acts)            every layer's output
    backward(net, A, acts, dout)     gradients of every weight and bias
    sigmoid(Z), recon_loss(L, X, kind)   cross-entropy on logits, or squared error, with its gradient
    Adam(net, lr)                    Day 37's optimiser for a list of layers
    train_ae(X, sizes, acts, ...)    minibatch training; noise= for Day 127's corruption
    reconstruct(net, X, acts, kind)  the network's output as an image

Checked by hand: backpropagation against finite differences for both losses,
a 2-1-2 linear autoencoder that finds the line its points lie on, and the loss
falling while a 6-4-6 network learns six-pixel patterns.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


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


rng = np.random.RandomState(0); xs = rng.rand(5, 6); acts = ["relu", "relu", "linear"]
for kind in ("bce", "mse"):
    net = [(W.astype(np.float64), b.astype(np.float64)) for W, b in init_net([6, 5, 3, 6], 0)]
    A = forward(net, xs, acts); _, d = recon_loss(A[-1], xs, kind); g, _ = backward(net, A, acts, d); worst = 0.0
    for li, (W, b) in enumerate(net):
        for P, G in ((W, g[li][0]), (b, g[li][1])):
            for idx in np.ndindex(P.shape):
                o = P[idx]; P[idx] = o + 1e-6; lp = recon_loss(forward(net, xs, acts)[-1], xs, kind)[0]
                P[idx] = o - 1e-6; lm = recon_loss(forward(net, xs, acts)[-1], xs, kind)[0]; P[idx] = o
                worst = max(worst, abs((lp - lm) / 2e-6 - G[idx]))
    print(f"gradient check, {kind}: largest gap {worst:.1e}")
t = np.linspace(-1, 1, 41, dtype=np.float32); X = np.c_[t, 2 * t]
net, hist = train_ae(X, [2, 1, 2], ["linear", "linear"], "mse", epochs=2000, batch=41, lr=0.01, seed=0)
w = net[1][0][0]; w = w / np.linalg.norm(w) * np.sign(w[0])
print(f"points on y = 2x: decoder direction ({w[0]:.4f}, {w[1]:.4f}); by hand (1, 2)/sqrt(5) = ({1 / np.sqrt(5):.4f}, {2 / np.sqrt(5):.4f}); error {hist[-1][0]:.2e}")
P = (rng.rand(300, 6) < np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1])).astype(np.float32)  # six-pixel images, each pixel on with its own chance
net, hist = train_ae(P, [6, 4, 6], ["relu", "linear"], "bce", epochs=60, batch=30, lr=0.01, seed=0)
print(f"6-4-6 network, cross-entropy per image: {hist[0][0]:.3f} after epoch 1, {hist[-1][0]:.3f} after 60; "
      f"falls every 10 epochs: {bool(np.all(np.diff([h[0] for h in hist[::10]]) < 0))}")
