"""
Day 79 Exercise -- LayerNorm as One Reusable Function, Next to BatchNorm

The third of Week 12's parts for Day 83's Transformer: layer
normalisation on its own, forward and backward,

    y = (x - mean(x)) / sqrt(var(x) + eps) * g + b      over the LAST axis (one position's features)

gradient-checked, and set beside batch normalisation (the same formula
over the batch axis) to show the property the lesson relied on: a
position's LayerNorm output depends on that position alone, while its
BatchNorm output changes when the rest of the batch changes.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def layer_norm(x, g, b, eps=1e-5):                      # NEW: x (..., d) -> same shape; statistics over the last axis
    sd = np.sqrt(x.var(axis=-1, keepdims=True) + eps)     # each position's spread
    xhat = (x - x.mean(axis=-1, keepdims=True)) / sd      # mean 0, spread 1 per position
    return xhat * g + b, (xhat, sd)                       # output and cache


def layer_norm_backward(dy, g, cache):                  # NEW: gradients for x, g and b
    xhat, sd = cache                                      # what the forward pass computed
    dh = dy * g                                           # through the scale
    dx = (dh - dh.mean(axis=-1, keepdims=True) - xhat * (dh * xhat).mean(axis=-1, keepdims=True)) / sd  # through mean and spread
    flat = lambda T: T.reshape(-1, T.shape[-1])           # every position as a row
    return dx, flat(dy * xhat).sum(axis=0), flat(dy).sum(axis=0)  # dx, dg, db


def batch_norm(x, g, b, eps=1e-5):                      # for comparison: statistics over every position of the batch, per feature
    flat = x.reshape(-1, x.shape[-1])                     # all positions as rows
    return (x - flat.mean(axis=0)) / np.sqrt(flat.var(axis=0) + eps) * g + b  # normalised per feature


rng = np.random.RandomState(0)                          # fixed rng
x = rng.randn(3, 4, 6) * 3 + 1                            # 3 sequences, 4 positions, 6 features; mean 1, spread 3
g, b = 1 + 0.1 * rng.randn(6), 0.1 * rng.randn(6)         # a scale and shift away from 1 and 0, so they are tested
U = rng.randn(3, 4, 6)                                    # L = sum(U * y)

y, cache = layer_norm(x, g, b)                            # forward
dx, dg, db = layer_norm_backward(U, g, cache)             # analytic gradients
worst = 0.0                                               # largest |numeric - analytic|
for A, G in [(x, dx), (g, dg), (b, db)]:                  # the input and both weights
    for pos in np.ndindex(A.shape):                         # every entry
        old = A[pos]                                          # remember it
        A[pos] = old + 1e-6; up = (U * layer_norm(x, g, b)[0]).sum()  # nudge up
        A[pos] = old - 1e-6; dn = (U * layer_norm(x, g, b)[0]).sum()  # nudge down
        A[pos] = old                                          # restore
        worst = max(worst, abs((up - dn) / 2e-6 - G[pos]))    # largest difference so far
print(f"gradient check (x, g, b): largest |numeric - analytic| = {worst:.2e}")

xhat = (y - b) / g                                        # undo the scale and shift
print(f"each position before g and b: largest |mean| = {np.abs(xhat.mean(axis=-1)).max():.1e}, spreads {xhat.std(axis=-1).min():.4f} .. {xhat.std(axis=-1).max():.4f}")

other = x.copy(); other[1:] = rng.randn(2, 4, 6) * 10     # keep sequence 0, replace the rest of the batch
print(f"sequence 0's output when the rest of the batch changes: LayerNorm {np.abs(layer_norm(other, g, b)[0][0] - y[0]).max():.2e}, "
      f"BatchNorm {np.abs(batch_norm(other, g, b)[0] - batch_norm(x, g, b)[0]).max():.2e}")
