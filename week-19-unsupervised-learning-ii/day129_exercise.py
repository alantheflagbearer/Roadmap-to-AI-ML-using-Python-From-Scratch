"""
Day 129 Exercise -- the Reparameterization Trick and the VAE Loss

The fourth of Week 19's parts.

    reparam(mu, logvar, rng)        z = mu + sigma * eps, eps ~ N(0, 1)
    kl_normal(mu, logvar)           KL( N(mu, sigma^2) || N(0, 1) ) per example
    vae_loss(enc, dec, X, rng, ...)  reconstruction + beta x KL, with every gradient
    train_vae(X, k, ...)            minibatch training of encoder and decoder
    score_grad, reparam_grad        two estimators of d E[f(z)] / d(mu, sigma)

Checked by hand: one KL value, the mean and spread of reparameterized draws,
both gradient estimators against the exact answer, and a gradient check
through a small VAE with eps held fixed.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import os                                              # os: paths
import sys                                             # sys: path, to import Day 126's parts
import contextlib                                      # contextlib: keep the import quiet
import io                                              # io: where its output goes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, recon_loss, Adam


def reparam(mu, logvar, rng):                           # NEW: Day 129's version, unchanged
    eps = rng.standard_normal(mu.shape).astype(mu.dtype)
    return mu + np.exp(0.5 * logvar) * eps, eps


def kl_normal(mu, logvar):                              # NEW: Day 129's version, unchanged
    return 0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar).sum(1)


def vae_loss(enc, dec, X, rng, beta=1.0, acts_e=("relu", "linear"), acts_d=("relu", "linear"), eps=None):  # NEW: Day 129's version, unchanged
    n = len(X); H = d126.forward(enc, X, list(acts_e)); out = H[-1]; k = out.shape[1] // 2
    mu, logvar = out[:, :k], out[:, k:]                    # the encoder's last layer gives both halves
    if eps is None:
        z, eps = reparam(mu, logvar, rng)
    else:
        z = mu + np.exp(0.5 * logvar) * eps                 # a fixed eps, for the gradient check
    A = d126.forward(dec, z, list(acts_d)); rec, dL = d126.recon_loss(A[-1], X, "bce")  # reconstruction: cross-entropy per image
    kl = float(kl_normal(mu, logvar).mean())
    gdec, dz = d126.backward(dec, A, list(acts_d), dL)    # back through the decoder, down to z
    dmu = dz + beta * mu / n                               # z = mu + sigma eps, so dz/dmu = 1; plus the KL term's own gradient
    dlogvar = dz * eps * 0.5 * np.exp(0.5 * logvar) + beta * 0.5 * (np.exp(logvar) - 1) / n  # dz/dlogvar = eps sigma / 2
    genc, _ = d126.backward(enc, H, list(acts_e), np.hstack([dmu, dlogvar]))
    return rec + beta * kl, rec, kl, genc, gdec


def train_vae(X, k, hidden=256, epochs=10, batch=128, lr=1e-3, beta=1.0, seed=0, Xval=None):  # NEW: Day 129's version, unchanged
    enc = d126.init_net([X.shape[1], hidden, 2 * k], seed); dec = d126.init_net([k, hidden, X.shape[1]], seed + 1)
    oe, od = d126.Adam(enc, lr), d126.Adam(dec, lr); rng = np.random.RandomState(seed + 2); hist = []
    for _ in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            _, _, _, ge, gd = vae_loss(enc, dec, X[o[s:s + batch]], rng, beta)
            oe.step(enc, ge); od.step(dec, gd)
        ev = X[:2000] if Xval is None else Xval
        hist.append(vae_loss(enc, dec, ev, np.random.RandomState(99), beta)[:3])  # (loss, reconstruction, KL) on held-out images
    return enc, dec, hist


def score_grad(f, mu, sigma, rng, n):                   # NEW: Day 129's version, unchanged
    z = mu + sigma * rng.standard_normal(n)
    return f(z) * (z - mu) / sigma ** 2, f(z) * ((z - mu) ** 2 / sigma ** 3 - 1 / sigma)  # f(z) times d log q(z) / d parameter


def reparam_grad(df, mu, sigma, rng, n):                # NEW: Day 129's version, unchanged
    eps = rng.standard_normal(n); z = mu + sigma * eps
    return df(z), df(z) * eps                              # chain rule through z = mu + sigma eps


print(f"KL(N(1, 4) || N(0, 1)): {kl_normal(np.array([[1.0]]), np.array([[np.log(4.0)]]))[0]:.4f}; by hand 0.5 (4 + 1 - 1 - log 4) = {0.5 * (4 - np.log(4)):.4f}")
rng = np.random.RandomState(0); z, _ = reparam(np.full((200000, 1), 2.0), np.full((200000, 1), np.log(9.0)), rng)
print(f"200,000 draws with mu 2, sigma 3: mean {z.mean():.3f}, spread {z.std():.3f} (by hand 2 and 3)")
gm, gs = score_grad(lambda z: z ** 2, 1.0, 2.0, rng, 200000); rm, rs = reparam_grad(lambda z: 2 * z, 1.0, 2.0, rng, 200000)
print(f"d E[z^2] / d(mu, sigma) at (1, 2): exact (2, 4); score function ({gm.mean():.2f}, {gs.mean():.2f}), spread ({gm.std():.1f}, {gs.std():.1f}); "
      f"reparameterized ({rm.mean():.2f}, {rs.mean():.2f}), spread ({rm.std():.1f}, {rs.std():.1f})")
X = rng.rand(4, 6); eps = rng.standard_normal((4, 2))
enc = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([6, 5, 4], 0)]
dec = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([2, 5, 6], 1)]
_, _, _, ge, gd = vae_loss(enc, dec, X, None, eps=eps); worst = 0.0
for net, g in ((enc, ge), (dec, gd)):
    for li, (W, b) in enumerate(net):
        for P, G in ((W, g[li][0]), (b, g[li][1])):
            for idx in np.ndindex(P.shape):
                o = P[idx]; P[idx] = o + 1e-6; lp = vae_loss(enc, dec, X, None, eps=eps)[0]
                P[idx] = o - 1e-6; lm = vae_loss(enc, dec, X, None, eps=eps)[0]; P[idx] = o
                worst = max(worst, abs((lp - lm) / 2e-6 - G[idx]))
print(f"gradient check through a small VAE, eps fixed: largest gap {worst:.1e}")
