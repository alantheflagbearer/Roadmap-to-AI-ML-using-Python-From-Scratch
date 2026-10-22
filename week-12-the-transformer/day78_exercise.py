"""
Day 78 Exercise -- One Encoder Block, Forward and Backward

The second of Week 12's parts for Day 83's Transformer: one post-LN
encoder block on its own, with no tagger around it:

    h = LayerNorm( x + MultiHeadAttention(x) )
    y = LayerNorm( h + FFN(h) ),     FFN(h) = ReLU(h W1 + c1) W2 + c2

Multi-head attention is Day 76's (Day 75's function on each head); the
positional encoding is left to the model that uses the block. The
exercise checks every gradient numerically, for the input and for every
weight, and checks that the block's output does not depend on padded
positions.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def attention(Q, K, V, mask):                           # day75_exercise.py: scaled dot-product attention with a key mask
    S = np.where(mask, np.einsum("mid,mjd->mij", Q, K) / np.sqrt(Q.shape[-1]), -np.inf)  # scaled, masked scores
    A = np.exp(S - S.max(axis=-1, keepdims=True)); A /= A.sum(axis=-1, keepdims=True)  # softmax over the keys
    return np.einsum("mij,mjd->mid", A, V), A             # outputs and weights


def split(T, h):                                        # Day 76: (m, n, d) -> (m * h, n, d / h)
    m, n, d = T.shape                                     # sizes
    return T.reshape(m, n, h, d // h).transpose(0, 2, 1, 3).reshape(m * h, n, d // h)  # heads become batch entries


def merge(T, h):                                        # Day 76: the inverse
    mh, n, dk = T.shape                                   # sizes
    return T.reshape(mh // h, h, n, dk).transpose(0, 2, 1, 3).reshape(mh // h, n, h * dk)  # heads side by side


def layer_norm(x, g, b):                                # Day 78: each position to mean 0, spread 1, then g and b
    sd = np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)    # spread of each position's vector
    xhat = (x - x.mean(axis=-1, keepdims=True)) / sd      # normalised
    return xhat * g + b, (xhat, sd)                       # output and cache


def layer_norm_backward(dy, g, c):                      # Day 78: gradients for x, g, b
    xhat, sd = c                                          # cache
    dh = dy * g                                           # through the scale
    dx = (dh - dh.mean(axis=-1, keepdims=True) - xhat * (dh * xhat).mean(axis=-1, keepdims=True)) / sd  # through mean and spread
    return dx, (dy * xhat).reshape(-1, len(g)).sum(axis=0), dy.reshape(-1, len(g)).sum(axis=0)  # dx, dg, db


def encoder_block(x, P, h, mask):                       # NEW: x (m, n, d) -> y (m, n, d), and a cache
    km = np.repeat(mask[:, None, None, :], h, axis=1).reshape(-1, 1, x.shape[1])  # padded keys, per head
    Q, K, V = (split(x @ P[w], h) for w in ("Wq", "Wk", "Wv"))  # Day 74's projections, cut into heads
    Zh, A = attention(Q, K, V, km)                        # every head at once
    Z = merge(Zh, h)                                      # heads side by side
    h1, n1 = layer_norm(x + Z @ P["Wo"], P["g1"], P["b1"])  # add & norm
    u = h1 @ P["W1"] + P["c1"]; r = np.maximum(u, 0)      # FFN: widen, ReLU
    y, n2 = layer_norm(h1 + r @ P["W2"] + P["c2"], P["g2"], P["b2"])  # FFN back to d, add & norm
    return y, (x, Q, K, V, A, Z, h1, n1, u, r, n2)        # output, and everything the backward pass needs


def encoder_block_backward(dy, P, h, cache):            # NEW: gradients for x and every weight
    x, Q, K, V, A, Z, h1, n1, u, r, n2 = cache            # what the forward pass computed
    f = lambda T: T.reshape(-1, T.shape[-1])              # every position as one row
    g = {}                                                # weight gradients
    ds2, g["g2"], g["b2"] = layer_norm_backward(dy, P["g2"], n2)  # second norm
    g["W2"], g["c2"] = f(r).T @ f(ds2), f(ds2).sum(axis=0)  # FFN, second layer
    du = (ds2 @ P["W2"].T) * (u > 0)                      # through W2 and the ReLU
    g["W1"], g["c1"] = f(h1).T @ f(du), f(du).sum(axis=0)  # FFN, first layer
    dh1 = ds2 + du @ P["W1"].T                            # the residual and the FFN both read h1
    ds1, g["g1"], g["b1"] = layer_norm_backward(dh1, P["g1"], n1)  # first norm
    g["Wo"] = f(Z).T @ f(ds1)                             # output projection
    dZh = split(ds1 @ P["Wo"].T, h)                       # back into the heads
    dA = np.einsum("mid,mjd->mij", dZh, V); dV = np.einsum("mij,mid->mjd", A, dZh)  # Z = A V
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True)) / np.sqrt(Q.shape[-1])  # softmax and scaling
    dQ, dK = merge(np.einsum("mij,mjd->mid", dS, K), h), merge(np.einsum("mij,mid->mjd", dS, Q), h)  # S = Q K^T
    dV = merge(dV, h)                                     # heads side by side
    g["Wq"], g["Wk"], g["Wv"] = f(x).T @ f(dQ), f(x).T @ f(dK), f(x).T @ f(dV)  # projections
    dx = ds1 + dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T  # the residual and the three projections
    return dx, g                                          # input gradient, weight gradients


rng = np.random.RandomState(0)                          # fixed rng
m, n, d, ff, h = 2, 5, 8, 16, 2                           # 2 sequences, 5 positions, size 8, FFN 16, 2 heads
s = lambda a, b: rng.randn(a, b) / np.sqrt(a)             # a weight matrix
P = dict(Wq=s(d, d), Wk=s(d, d), Wv=s(d, d), Wo=s(d, d), g1=1 + 0.1 * rng.randn(d), b1=0.1 * rng.randn(d),  # attention and the first norm
         W1=s(d, ff), c1=0.1 * rng.randn(ff), W2=s(ff, d), c2=0.1 * rng.randn(d), g2=1 + 0.1 * rng.randn(d), b2=0.1 * rng.randn(d))  # FFN, second norm (g, b moved off 1, 0 so they are tested)
x = rng.randn(m, n, d)                                    # inputs
mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], bool)  # sequence 2 has two padded positions
U = rng.randn(m, n, d) * mask[:, :, None]                 # L = sum(U * y) over the real positions only

y, cache = encoder_block(x, P, h, mask)                   # forward
dx, g = encoder_block_backward(U, P, h, cache)            # analytic gradients
worst = 0.0                                               # largest |numeric - analytic|
for name, A_ in dict(x=x, **P).items():                   # the input and every weight
    G = dx if name == "x" else g[name]                      # its analytic gradient
    for pos in np.ndindex(A_.shape):                        # every entry
        old = A_[pos]                                         # remember it
        A_[pos] = old + 1e-6; up = (U * encoder_block(x, P, h, mask)[0]).sum()  # nudge up
        A_[pos] = old - 1e-6; dn = (U * encoder_block(x, P, h, mask)[0]).sum()  # nudge down
        A_[pos] = old                                         # restore
        worst = max(worst, abs((up - dn) / 2e-6 - G[pos]))    # largest difference so far
print(f"gradient check (x and all 12 weight arrays): largest |numeric - analytic| = {worst:.2e}")

x2 = x.copy(); x2[1, 3:] = rng.randn(2, d) * 100          # change sequence 2's padded positions wildly
y2, _ = encoder_block(x2, P, h, mask)                     # forward again
print(f"padded inputs changed: largest change at a real position = {np.abs(y2 - y)[mask].max():.2e}")
