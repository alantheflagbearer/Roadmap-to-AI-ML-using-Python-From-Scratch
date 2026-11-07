"""
Day 127 Exercise -- Damage, and How to Score a Repair

The second of Week 19's parts.

    gaussian(X, sigma, rng)       add N(0, sigma^2) noise, kept inside [0, 1]
    masking(X, p, rng)            set each pixel to 0 with probability p
    hide_square(X, size, rng)     blank a square in each 28 x 28 image, and say where
    psnr(R, X)                    peak signal-to-noise ratio in decibels
    median3(X)                    the 3 x 3 median filter, the classic baseline

Checked by hand: PSNR of a known error, the share of pixels masking removes,
a hidden square of exactly size x size pixels, and a median filter that removes
a single bright pixel but keeps a solid block.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


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


rng = np.random.RandomState(0); X = np.full((1, 784), 0.5, np.float32)
print(f"PSNR of an image off by 0.1 everywhere: {psnr(X + 0.1, X):.2f} dB; by hand 10 log10(1 / 0.01) = 20.00 dB")
M = masking(np.ones((200, 784), np.float32), 0.3, rng); print(f"masking 0.3 keeps {M.mean():.4f} of the pixels (by hand about 0.7)")
G = gaussian(np.zeros((200, 784), np.float32), 0.3, rng); print(f"Gaussian noise on black pixels, clipped: none below 0: {bool(G.min() >= 0)}; about half stay 0: {np.mean(G == 0):.4f}")
D, Mk = hide_square(np.ones((5, 784), np.float32), 10, rng); print(f"hidden squares: {Mk.sum(1).tolist()} pixels each (by hand 10 x 10 = 100), all set to 0: {bool(np.all(D[Mk] == 0))}")
img = np.zeros((1, 28, 28), np.float32); img[0, 5, 5] = 1.0; img[0, 15:20, 15:20] = 1.0; F = median3(img.reshape(1, -1)).reshape(28, 28)
print(f"median 3 x 3: the lone bright pixel becomes {F[5, 5]:.0f}; the middle of the 5 x 5 block stays {F[17, 17]:.0f}")
