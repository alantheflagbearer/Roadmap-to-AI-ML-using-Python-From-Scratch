"""
Day 122 Exercise -- The Pieces of EM for a Gaussian Mixture

The fourth of Week 18's parts.

    log_gauss(X, mu, S)       log N(x | mu, S) for every row, through a Cholesky factor
    logsumexp(A)              log(sum(exp(A))) along each row, without overflow
    e_step(X, w, M, S)        responsibilities, and the log-likelihood
    m_step(X, R, kind, reg)   weights, means and covariances from the responsibilities

Checked by hand: the density of a 2-D Gaussian against the textbook formula,
log-sum-exp on numbers too large for exp, responsibilities that sum to 1, an
M step with hard memberships that gives each group's own mean and covariance,
and EM steps that never lower the likelihood.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this

LOG2PI = np.log(2 * np.pi)                              # a constant of the Gaussian density


def log_gauss(X, mu, S):                                # NEW: Day 122's version, unchanged
    L = np.linalg.cholesky(S)
    Z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (Z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * LOG2PI


def logsumexp(A):                                       # NEW: Day 122's version, unchanged
    m = A.max(1, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(1, keepdims=True)))[:, 0]


def e_step(X, w, M, S):                                 # NEW: Day 122's version, unchanged
    logp = np.stack([np.log(w[j]) + log_gauss(X, M[j], S[j]) for j in range(len(w))], 1)
    ll = logsumexp(logp)
    return np.exp(logp - ll[:, None]), float(ll.sum())


def m_step(X, R, kind="full", reg=1e-6):                # NEW: Day 122's version, unchanged
    Nk = R.sum(0) + 1e-300; w = Nk / len(X); M = (R.T @ X) / Nk[:, None]; d = X.shape[1]; S = []
    for j in range(R.shape[1]):
        D = X - M[j]; Sj = (R[:, j, None] * D).T @ D / Nk[j]
        if kind == "diag":
            Sj = np.diag(np.diag(Sj))
        elif kind == "spherical":
            Sj = np.eye(d) * np.trace(Sj) / d
        S.append(Sj + reg * np.eye(d))
    return w, M, np.array(S)


S = np.array([[2.0, 0.6], [0.6, 1.0]]); mu = np.array([1.0, -1.0]); x = np.array([[2.0, 0.0]])
d = x[0] - mu; by_hand = -0.5 * d @ np.linalg.inv(S) @ d - 0.5 * np.log(np.linalg.det(S)) - LOG2PI
print(f"log N((2, 0) | (1, -1), S): {log_gauss(x, mu, S)[0]:.6f}; textbook formula with the inverse and determinant: {by_hand:.6f}")
print(f"logsumexp of [1000, 1000]: {logsumexp(np.array([[1000.0, 1000.0]]))[0]:.6f} (exp(1000) overflows; the answer is 1000 + log 2 = {1000 + np.log(2):.6f})")
rng = np.random.RandomState(0); X = np.r_[rng.normal(0, 1, (40, 2)), rng.normal(4, 0.5, (20, 2))]; lab = np.r_[np.zeros(40, int), np.ones(20, int)]
w, M, Sg = m_step(X, np.eye(2)[lab], reg=0.0)
print(f"hard memberships: weights {np.round(w, 4).tolist()} (40 and 20 of 60); mean of group 2 equals its own mean {bool(np.allclose(M[1], X[40:].mean(0)))}; "
      f"its covariance equals np.cov(bias=True) {bool(np.allclose(Sg[1], np.cov(X[40:].T, bias=True)))}")
R, ll = e_step(X, w, M, Sg); h = [ll]
for _ in range(5):
    w, M, Sg = m_step(X, R); R, ll = e_step(X, w, M, Sg); h.append(ll)
print(f"responsibilities sum to 1 on every row: {bool(np.allclose(R.sum(1), 1))}; log-likelihood over 5 EM steps never falls: {bool(np.all(np.diff(h) >= -1e-9))}")
