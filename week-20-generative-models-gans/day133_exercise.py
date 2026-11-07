"""
Day 133 Exercise -- Gaussian Class Models: Classify, Sample, Score

The first of Week 20's parts for Day 139's build.

    fit_gaussians(X, y, k, kind, reg)   p(x | class) as a Gaussian per class: "diagonal", "shared" (LDA) or "full" (QDA)
    log_gauss(X, mu, S)                 log N(x; mu, S) for every row, through a Cholesky factor
    log_joint(model, X, keep)           log p(x, class); keep = the features that were seen
    posterior(model, X, keep)           Bayes' rule: p(class | x)
    log_px(model, X)                    log p(x): how ordinary each x is
    lda_linear(model)                   a shared-covariance model as linear scores x W + b
    sample_class(model, c, n, rng)      new data drawn from p(x | class c)

Checked by hand: log N(0; 0, 1) = -log(2 pi) / 2, the posterior of two 1-D
classes equals a sigmoid, LDA's linear form equals Bayes' rule, a marginal
equals the Gaussian of the kept features, and samples have the right mean
and covariance.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


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


print(f"log N(0; 0, 1): {log_gauss(np.zeros((1, 1)), np.zeros(1), np.eye(1))[0]:.6f}; by hand -log(2 pi) / 2 = {-0.5 * np.log(2 * np.pi):.6f}")
X = np.array([[0.0], [1.0], [2.0], [4.0], [5.0], [6.0]]); y = np.array([0, 0, 0, 1, 1, 1]); m = fit_gaussians(X, y, 2, "shared", 0.0)
print(f"two 1-D classes: p(1 | x = 3.5) = {posterior(m, np.array([[3.5]]))[0, 1]:.6f}; by hand sigmoid(6 x 3.5 - 18) = {1 / (1 + np.exp(-3.0)):.6f}")
rng = np.random.RandomState(0); A = rng.randn(300, 3) @ np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]]); b = rng.randint(0, 3, 300)
A[b == 1] += 2.0; m = fit_gaussians(A, b, 3, "shared", 1e-3); W, c = lda_linear(m); L = A @ W + c
S = np.exp(L - L.max(1, keepdims=True)); S /= S.sum(1, keepdims=True)
print(f"LDA as linear scores against Bayes' rule: largest gap {np.abs(S - posterior(m, A)).max():.1e}")
m2 = fit_gaussians(A[:, :2], b, 3, "shared", 1e-3); m3 = fit_gaussians(A, b, 3, "shared", 1e-3)
print(f"marginal (features 0 and 1 of a 3-feature model) against a model of those 2 features: largest gap "
      f"{np.abs(log_joint(m3, A, np.array([0, 1])) - log_joint(m2, A[:, :2])).max():.1e}")
m = fit_gaussians(A, b, 3, "full", 1e-3); D = sample_class(m, 1, 200000, rng)
print(f"200,000 samples of class 1: mean off by {np.abs(D.mean(0) - m['mu'][1]).max():.4f}, covariance off by {np.abs(np.cov(D.T) - m['cov'][1]).max():.4f}")
print(f"log p(x) of a typical point {log_px(m, m['mu'][:1])[0]:.3f}; of a far-away point {log_px(m, np.full((1, 3), 20.0))[0]:.3f}")
