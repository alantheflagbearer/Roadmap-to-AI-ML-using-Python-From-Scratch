"""
Day 157 Exercise -- Hybrid Recommenders

The fourth of Week 23's parts for Day 160's neural recommender.

    stack_fit(P, y)                        blend weights by least squares, intercept first
    stack_predict(P, w)                    the blend's prediction
    fw_design(P, n)                        feature-weighted stacking: weights that move with log(1 + count)
    switch_predict(P_rich, P_poor, n, t)   the content model for films with fewer than t ratings
    hybrid_mf(u, i, r, X, nu, ni, k, lam)  factors with content inside, by SGD
    hybrid_predict(F, X, u, i)             its prediction; a new film uses its content alone

Checked by hand: a blend recovering known weights, a prediction worked on paper,
the feature-weighted columns of a film with no ratings, a switch, and the
in-model hybrid predicting films it never saw from their genre alone.
"""
import numpy as np                                     # numpy: every array operation below is built on this


def stack_fit(P, y):                                    # NEW: Day 157's version, unchanged
    A = np.c_[np.ones(len(y)), P]                                                     # a column of 1s for the intercept
    return np.linalg.lstsq(A, y, rcond=None)[0]                                       # intercept first, then one weight per model


def stack_predict(P, w):                                # NEW: Day 157's version, unchanged
    return w[0] + P @ w[1:]                                                           # intercept + weighted sum of the models


def fw_design(P, n):                                    # NEW: Day 157's version, unchanged
    g = np.log1p(n)[:, None]                                                          # how much is known about the film
    return np.c_[g, P, P * g]                                                         # (intercept added by stack_fit) + slope, weights, weight slopes


def switch_predict(P_rich, P_poor, n, threshold):       # NEW: Day 157's version, unchanged
    return np.where(n < threshold, P_poor, P_rich)                                    # few ratings: content; enough: the ratings-based blend


def hybrid_mf(u, i, r, X, nu, ni, k, lam, lr=0.01, lr_s=0.1, lam_s=0.001, epochs=60, batch=128, seed=0, track=None):  # NEW: Day 157's version, unchanged
    rng = np.random.RandomState(seed); f = X.shape[1]                                 # seeded, so a run repeats
    seen = np.bincount(i, minlength=ni) > 0                                           # films with a training rating
    F = dict(mu=r.mean(), P=0.1 * rng.randn(nu, k), Q=0.1 * rng.randn(ni, k) * seen[:, None],  # unseen films' own factors stay 0
             bu=np.zeros(nu), bi=np.zeros(ni), a=np.zeros(f), Z=0.1 * rng.randn(f, k) / np.sqrt(f))  # a: content bias; Z: content to factors
    hist = []                                                                         # measures after each epoch, if asked for
    for _ in range(epochs):                                                           # one pass over the ratings
        o = rng.permutation(len(r))                                                   # a fresh random order
        for s in range(0, len(o), batch):                                             # small batches
            b = o[s:s + batch]; a, c = u[b], i[b]; xc = X[c]                          # the batch's users, films and features
            qe = F["Q"][c] + xc @ F["Z"]                                              # each film's factors: its own + what its content says
            e = r[b] - (F["mu"] + F["bu"][a] + F["bi"][c] + xc @ F["a"] + (F["P"][a] * qe).sum(1))  # errors
            gp = e[:, None] * qe - lam * F["P"][a]                                    # downhill for the user factors
            gq = e[:, None] * F["P"][a] - lam * F["Q"][c]                             # and the films' own factors
            gz = xc.T @ (e[:, None] * F["P"][a]) / len(b) - lam_s * F["Z"]            # shared: averaged over the batch
            ga = xc.T @ e / len(b) - lam_s * F["a"]                                   # the content bias, the same way
            np.add.at(F["bu"], a, lr * (e - lam * F["bu"][a]))                        # user biases (repeats counted)
            np.add.at(F["bi"], c, lr * (e - lam * F["bi"][c]))                        # film biases
            np.add.at(F["P"], a, lr * gp); np.add.at(F["Q"], c, lr * gq)              # and the factors
            F["Z"] += lr_s * gz; F["a"] += lr_s * ga                                  # the shared content weights
        hist.append(track(F) if track else ())                                        # measures after this epoch
    return F, hist                                                                    # the model, and one record per epoch


def hybrid_predict(F, X, u, i):                         # NEW: Day 157's version, unchanged
    q = F["Q"][i] + X[i] @ F["Z"]                                                     # a new film's own part is 0: content alone
    return F["mu"] + F["bu"][u] + F["bi"][i] + X[i] @ F["a"] + (F["P"][u] * q).sum(1)  # one number per pair


rng = np.random.RandomState(0); P = rng.randn(50, 2); y = 1.0 + 2.0 * P[:, 0] + 3.0 * P[:, 1]
print(f"y = 1 + 2 a + 3 b exactly: stack_fit gives {np.round(stack_fit(P, y), 6).tolist()}")
print(f"blend of predictions 4 and 3 with weights (0.5; 0.6, 0.3): 0.5 + 2.4 + 0.9 = 3.8; stack_predict gives "
      f"{stack_predict(np.array([[4.0, 3.0]]), np.array([0.5, 0.6, 0.3]))[0]:.4f}")
print(f"fw_design for a film with 0 ratings (log 1 = 0) and predictions 4, 3: {fw_design(np.array([[4.0, 3.0]]), np.array([0]))[0].tolist()} "
      "(0, the two predictions, then 0, 0)")
print(f"switch at 10 ratings, films with 0, 5 and 20: rich 4, poor 3 -> 3, 3, 4; switch_predict gives "
      f"{switch_predict(np.full(3, 4.0), np.full(3, 3.0), np.array([0, 5, 20]), 10).tolist()}")
X = np.array([[1.0, 0.0], [0.0, 1.0]] * 3); s = np.array([1.0, -1.0] * 4); g = np.array([1.0, -1.0] * 3)  # 6 films, genre A or B; 8 users
uu, ii = np.meshgrid(np.arange(8), np.arange(6), indexing="ij"); uu, ii = uu.ravel(), ii.ravel()
rr = 3.0 + s[uu] * g[ii]                                                              # half the users like A (4 stars), half like B
tr = ii < 4                                                                           # films 4 and 5 are never seen in training
for lam in (0.0, 0.1):                                                                # without and with a penalty on the films' own factors
    F, _ = hybrid_mf(uu[tr], ii[tr], rr[tr], X, 8, 6, 2, lam, lr=0.05, lr_s=0.2, lam_s=0.0, epochs=800, batch=8)
    print(f"in-model hybrid, 8 users x 6 films, films 4-5 never seen, penalty {lam}: their own factors stay {np.abs(F['Q'][4:]).max():.1f}; "
          f"largest error on seen films {np.abs(hybrid_predict(F, X, uu[tr], ii[tr]) - rr[tr]).max():.3f}, "
          f"on unseen films, from genre alone {np.abs(hybrid_predict(F, X, uu[~tr], ii[~tr]) - rr[~tr]).max():.3f}")
