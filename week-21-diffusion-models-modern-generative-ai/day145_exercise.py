"""
Day 145 Exercise -- Measuring the Risks of a Generative Model

The sixth of Week 21's parts for Day 146's capstone.

    skew_ones(y, lean, share)               training indices with most ones not leaning right removed (a deliberate bias)
    train_detector(R, G, epochs)            a small network trained to tell real images (0) from generated ones (1)
    p_fake(net, F)                          that detector's probability that each image is generated
    blur(F)                                 each pixel replaced by the average of its 3 x 3 neighbourhood (an edit)
    flagged_precision(tpr, fpr, prevalence) of the images a detector flags, the share that really are generated

Built on Day 126's networks and Day 143's measurements, imported unchanged.
Checked by hand: Bayes' rule at 1% prevalence, the skew on twelve made-up ones,
the blur of a single pixel, and a detector that learns to spot blurred digits.
"""
import os                                              # os: the paths to earlier weeks' folders
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # as in the lesson
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
sys.path.insert(0, HERE)                                # this week's parts (Day 143)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day143_exercise as d143                      # Day 143: measure (thickness and lean classes)
SKEW = 0.75                                             # as in the lesson


def skew_ones(y, lean, share=SKEW, seed=0):              # NEW: Day 145's version, unchanged
    rng = np.random.RandomState(seed); idx = np.arange(len(y))
    one = y == 1; right = one & (lean == 1); other = one & (lean != 1)                # ones leaning right, and every other one
    keep_other = int(right.sum() * (1 - share) / share)                               # how many other ones leave `share` of the ones leaning right
    drop = rng.permutation(idx[other])[keep_other:]                                   # the rest of them are removed
    return np.setdiff1d(idx, drop)                                                    # every other digit is kept as it was


def train_detector(R, G, epochs=10, seed=0, batch=128, lr=1e-3):  # NEW: Day 145's version, unchanged
    net = d126.init_net([784, 128, 1], seed); opt = d126.Adam(net, lr); rng = np.random.RandomState(seed)
    Xd = np.vstack([R, G]).astype(np.float32); yd = np.r_[np.zeros(len(R)), np.ones(len(G))]  # images and their labels
    for _ in range(epochs):
        o = rng.permutation(len(Xd))
        for s in range(0, len(Xd) - batch + 1, batch):
            b = o[s:s + batch]; A = d126.forward(net, Xd[b], ["relu", "linear"])      # one score per image
            p = 1 / (1 + np.exp(-A[-1][:, 0].astype(np.float64)))                    # the score as a probability of 'generated'
            grads, _ = d126.backward(net, A, ["relu", "linear"], ((p - yd[b]) / len(b))[:, None].astype(np.float32))  # logistic loss's gradient
            opt.step(net, grads)
    return net


def p_fake(net, F):                                     # NEW: Day 145's version, unchanged
    return 1 / (1 + np.exp(-d126.forward(net, F.astype(np.float32), ["relu", "linear"])[-1][:, 0].astype(np.float64)))


def blur(F):                                            # NEW: Day 145's version, unchanged
    P = np.pad(F.reshape(-1, 28, 28), ((0, 0), (1, 1), (1, 1)))                       # a border of zeros, so edge pixels have 9 neighbours
    return (sum(P[:, i:i + 28, j:j + 28] for i in range(3) for j in range(3)) / 9).reshape(-1, 784)


def flagged_precision(tpr, fpr, prevalence):            # NEW: Day 145's version, unchanged
    return tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence))             # Bayes' rule


p = flagged_precision(0.95, 0.05, 0.01)                                              # Bayes' rule, against the hand calculation
print(f"flagged precision at 1% prevalence: {p:.4f} (by hand: 0.0095 / (0.0095 + 0.0495) = {0.0095 / (0.0095 + 0.0495):.4f})")
yy = np.r_[np.ones(12, int), np.full(5, 3)]; ll = np.r_[[1, 1, 1], np.zeros(9, int), np.zeros(5, int)]  # 12 ones (3 leaning right), 5 threes
k = skew_ones(yy, ll, 0.75); o = k[yy[k] == 1]
print(f"skew on 12 made-up ones (3 leaning right): {len(o)} ones kept, {int((ll[o] == 1).sum())} leaning right "
      f"(share {np.mean(ll[o] == 1):.2f}); threes kept {int((yy[k] == 3).sum())} of 5")
F = np.zeros((2, 784)); F[0, 14 * 28 + 14] = 1.0; F[1, 0] = 1.0                       # one pixel in the middle, one in the corner
B = blur(F)
print(f"blur of one middle pixel: {int((B[0] > 0).sum())} pixels of {B[0].max():.4f} (1/9 = {1 / 9:.4f}), total ink {B[0].sum():.4f}; "
      f"corner pixel: {int((B[1] > 0).sum())} pixels, total {B[1].sum():.4f} (4/9 = {4 / 9:.4f})")
X, y = d126m.mnist(); o = np.random.RandomState(0).permutation(60000)                 # the training digits, in random order
R, Gs, Rt, Gt = X[o[:3000]], blur(X[o[3000:6000]]), X[o[6000:7000]], blur(X[o[7000:8000]])  # real, and 'generated' = blurred digits
net = train_detector(R, Gs, epochs=3)
pr, pg = p_fake(net, Rt), p_fake(net, Gt)
print(f"detector, real against blurred digits, 3 epochs: held-out real called generated {np.mean(pr > 0.5):.3f}, "
      f"blurred called generated {np.mean(pg > 0.5):.3f}")
