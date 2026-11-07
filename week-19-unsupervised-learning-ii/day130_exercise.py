"""
Day 130 Exercise -- Using a Trained VAE

The fifth of Week 19's parts.

    encode(enc, X)                  each image's code cloud: (mean, log-variance)
    decode(dec, Z)                  images from codes
    sample_prior(dec, n, k, rng)    new images from codes drawn from N(0, 1)
    interpolate(z0, z1, steps)      evenly spaced codes on a straight line
    nearest_real(R, ref)            distance per pixel to the nearest real image
    active_units(enc, X, thresh)    code dimensions that carry information

Checked by hand on a tiny untrained VAE: shapes, a straight-line walk, a
distance of exactly 0 for an image in the reference set, and a dimension
that the KL term counts as unused.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import os                                              # os: paths
import sys                                             # sys: path, to import earlier parts
import contextlib                                      # contextlib: keep the imports quiet
import io                                              # io: where their output goes
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "week-18-unsupervised-learning-i"))
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, sigmoid
    import day119_exercise as d119                      # Day 119: sqdist
ENC = ["relu", "linear"]; DEC = ["relu", "linear"]      # Day 129's encoder and decoder layers


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


rng = np.random.RandomState(0); X = rng.rand(6, 10).astype(np.float32)
enc = d126.init_net([10, 8, 6], 0); dec = d126.init_net([3, 8, 10], 1)
mu, lv = encode(enc, X); S = sample_prior(dec, 4, 3, rng)
print(f"encode: means {mu.shape}, log-variances {lv.shape}; sample_prior: {S.shape}, all pixels in (0, 1): {bool(np.all((S > 0) & (S < 1)))}")
L = interpolate(np.zeros(3), np.array([2.0, 4.0, 6.0]), 5); print(f"interpolate from 0 to (2, 4, 6) in 5 steps, first coordinate: {L[:, 0].tolist()} (by hand 0, 0.5, 1, 1.5, 2)")
print(f"nearest_real: an image in the reference set is {nearest_real(X[:1], X.astype(np.float64))[0]:.4f} from it; "
      f"an all-0.5 image from an all-0 reference: {nearest_real(np.full((1, 4), 0.5), np.zeros((1, 4)))[0]:.4f} (by hand 0.5)")
enc[1] = (enc[1][0] * 0, np.zeros(6, np.float32))      # an encoder whose outputs are all 0: every cloud is exactly N(0, 1)
n_act, kl = active_units(enc, X); print(f"an encoder that always outputs N(0, 1): active dimensions {n_act}, KL per dimension {kl.round(4).tolist()}")
