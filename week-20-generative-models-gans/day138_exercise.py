"""
Day 138 Exercise -- Scoring Generated Images

The sixth of Week 20's parts for Day 139's build.

    sqrtm_psd(S)                       the square root of a symmetric positive semi-definite matrix
    frechet(mu1, S1, mu2, S2)          the Frechet distance between two Gaussians
    gauss(F)                           mean and covariance of feature vectors
    inception_score(P)                 exp of the average KL(p(y|x) || p(y))
    precision_recall(R, F, k)          k-nearest-neighbour precision and recall
    feature_net, train_classifier      a small convolutional digit classifier (Day 136's layers)
    Scorer(body, head, Xref, Xnear)    IS, FID, precision, recall and nearest training image of a set of images

Checked by hand: the Frechet distance of two 1-D Gaussians, the Inception
Score of hand-made probability tables, and real test digits against noisy
ones.
"""
import os                                              # os: the paths to this week's and earlier folders
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
sys.path.insert(0, HERE)                                # Days 134 and 136's parts
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day134_exercise as d134                      # Day 134: AdamGAN
    import day136_exercise as d136                      # Day 136: the layers
    import day119_exercise as d119                      # Day 119: sqdist
    import day116_exercise as d116                      # Day 116: softmax
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)


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


print(f"N(0, 1) against N(3, 4): Frechet distance {frechet(np.zeros(1), np.eye(1), np.full(1, 3.0), np.eye(1) * 4):.6f}; by hand 3^2 + (1 - 2)^2 = 10")
E = np.eye(10)
print(f"Inception Score: ten confident, evenly spread classes {inception_score(np.tile(E, (50, 1))):.4f}; always a confident 3 "
      f"{inception_score(np.tile(E[3], (500, 1))):.4f}; always unsure {inception_score(np.full((500, 10), 0.1)):.4f}")
X, y = d126m.mnist(); body, head = train_classifier(X[:10000], y[:10000], epochs=1)
acc = np.mean(head.forward(body.forward(X[60000:62000])).argmax(1) == y[60000:62000])
print(f"feature network, 1 epoch on 10,000 images: test accuracy {acc:.4f}")
sc = Scorer(body, head, X[:10000], X[10000:20000]); rng = np.random.RandomState(0); T = X[60000:62000]
for name, S in (("real test digits", T), ("+ noise, sd 0.3", np.clip(T + 0.3 * rng.randn(*T.shape), 0, 1).astype(np.float32))):
    s = sc(S); print(f"{name:>18}: IS {s[0]:.3f}, FID {s[1]:.3f}, precision {s[2]:.3f}, recall {s[3]:.3f}, nearest {s[4]:.2f}")
