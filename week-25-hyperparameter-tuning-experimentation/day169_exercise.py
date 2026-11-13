"""
Day 169 Exercise -- Bayesian Optimisation

The second of Week 25's parts for Day 174's reproducible tuning study.

    normal_cdf(z), normal_pdf(z)     the standard normal distribution, without SciPy
    matern52(A, B, ell)              how similar every row of A is to every row of B
    gp_fit(X, y, ell)                a Gaussian process conditioned on the trials so far
    gp_predict(g, Xs)                its mean and standard deviation at new points
    fit_gp(X, y)                     the length scale with the highest marginal likelihood
    expected_improvement(mu, sd, best)   how far below the best score a point is expected to land
    propose(X, y, C)                 the candidate with the highest expected improvement

Checked by hand: normal values, the kernel at three distances, a GP on two
trials, expected improvement at two beliefs, and which candidate is proposed.
"""
import math                                            # math: erf, for the normal distribution

import numpy as np                                     # numpy: every array operation below is built on this

ELLS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0)        # length scales the GP may choose from (the unit cube is 1 wide)
NOISE = 1e-4                                            # a small noise term on the standardised scores, for numerical safety
_erf = np.vectorize(math.erf)                           # math.erf, one value at a time, over arrays


def normal_cdf(z):                                      # P(Z <= z) for a standard normal Z
    return 0.5 * (1 + _erf(np.asarray(z, float) / np.sqrt(2)))                       # through the error function


def normal_pdf(z):                                      # the standard normal density
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)             # the bell curve


def matern52(A, B, ell):                                # similarity of every row of A to every row of B (1 = same point)
    d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0)) / ell  # distances, in length scales
    return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(-np.sqrt(5) * d)          # Matern 5/2: smooth, but less than Gaussian


def gp_fit(X, y, ell, noise=NOISE):                     # condition a GP on trials X with scores y
    m = y.mean(); sd = y.std() if y.std() > 0 else 1.0                                # standardise the scores
    z = (y - m) / sd                                                                  # mean 0, spread 1
    K = matern52(X, X, ell) + noise * np.eye(len(X))                                  # the trials' covariance
    L = np.linalg.cholesky(K)                                                         # K = L L^T
    a = np.linalg.solve(L.T, np.linalg.solve(L, z))                                   # K^-1 z, by two triangular solves
    lml = -0.5 * z @ a - np.log(np.diag(L)).sum() - 0.5 * len(X) * np.log(2 * np.pi)  # log marginal likelihood
    return dict(X=X, L=L, a=a, m=m, sd=sd, ell=ell, lml=lml)                          # everything prediction needs


def gp_predict(g, Xs):                                  # the GP's mean and standard deviation at new points
    Ks = matern52(Xs, g["X"], g["ell"])                                               # similarity of each new point to each trial
    mu = Ks @ g["a"]                                                                  # posterior mean (standardised)
    v = np.linalg.solve(g["L"], Ks.T)                                                 # L^-1 k for every new point
    var = np.maximum(1 - (v * v).sum(0), 1e-12)                                       # posterior variance: prior 1 minus what the trials explain
    return g["m"] + g["sd"] * mu, g["sd"] * np.sqrt(var)                              # back on the scores' scale


def fit_gp(X, y, ells=ELLS):                            # choose the length scale with the highest marginal likelihood
    return max((gp_fit(X, y, e) for e in ells), key=lambda g: g["lml"])               # one fit per candidate length scale


def expected_improvement(mu, sd, best):                 # how much below `best` the score is expected to fall (lower is better)
    z = (best - mu) / sd                                                              # standardised distance below the best so far
    return (best - mu) * normal_cdf(z) + sd * normal_pdf(z)                           # closed form for a normal belief


def propose(X, y, C):                                   # the candidate in C with the highest expected improvement
    g = fit_gp(X, y)                                                                  # the GP given the trials so far
    mu, sd = gp_predict(g, C)                                                         # its belief at every candidate
    return int(np.argmax(expected_improvement(mu, sd, y.min()))), g                   # the best candidate's index, and the GP


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    print(f"normal_cdf 0, 1.96: {normal_cdf([0, 1.96]).round(4).tolist()} (by hand 0.5, 0.975); normal_pdf(0) {float(normal_pdf(0)):.5f} "
          f"(1 / sqrt(2 pi) = 0.39894)")
    k = matern52(np.array([[0.0]]), np.array([[0.0], [0.5], [1.0]]), 1.0)[0]  # distances 0, 0.5 and 1
    print(f"matern52 at distances 0, 0.5, 1 (length scale 1): {k.round(4).tolist()} (by hand 1, (1 + 1.1180 + 0.4167) e^-1.1180 = 0.82865, "
          f"(1 + 2.2361 + 1.6667) e^-2.2361 = 0.5240)")
    g = gp_fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), 1.0)  # two trials
    mu, sd = gp_predict(g, np.array([[0.0], [0.5], [1.0]]))  # at the trials and between
    print(f"GP on trials 0 -> 0 and 1 -> 1: mean {mu.round(3).tolist()} (by hand 0, 0.5, 1), sd at 0.5 {sd[1]:.3f} "
          f"(0.5 x sqrt(1 - 2 x 0.8287^2 / 1.5240) = 0.157)")
    X = np.linspace(0, 1, 8)[:, None]; y = np.sin(3 * X[:, 0])  # a smooth curve
    g = fit_gp(X, y); best = max(gp_fit(X, y, e)["lml"] for e in ELLS)  # the chosen GP, and the best likelihood by brute force
    print(f"fit_gp on 8 points of sin(3x): length scale {g['ell']}, its log marginal likelihood equals the largest of all {len(ELLS)}: {g['lml'] == best}")
    print(f"expected_improvement(mean 0, sd 1, best 0) {float(expected_improvement(0.0, 1.0, 0.0)):.5f} (by hand 0.39894); "
          f"(mean -1, sd 0.001, best 0) {float(expected_improvement(-1.0, 0.001, 0.0)):.4f} (by hand 1: a sure improvement of 1)")
    i, _ = propose(np.array([[0.0], [1.0]]), np.array([1.0, 0.0]), np.array([[0.05], [0.95]]))  # trials: 0 scores 1, 1 scores 0
    print(f"propose with trials 0 -> 1 and 1 -> 0, candidates 0.05 and 0.95: index {i} (by hand 1: next to the better trial)")
