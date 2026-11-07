"""
Day 134 Exercise -- The Two Players of a GAN

The second of Week 20's parts for Day 139's build.

    AdamGAN(net, lr, b1)                   Day 126's Adam with its momentum b1 as a setting
    softplus(Z)                            log(1 + e^z), without overflow
    d_loss(lr, lf)                         the discriminator's loss on real and fake logits, and its gradients
    g_loss(lf, kind)                       the generator's loss: "non-saturating" -log D(G(z)) or "minimax" log(1 - D(G(z)))
    generate(G, Z, out)                    noise in, data out ("sigmoid" pixels or "linear" values)
    g_grads(G, D, Z, out, kind)            the generator's loss and gradient, flowing back through the discriminator
    gan_step(G, D, oG, oD, X, rng, k, ...) one round of the game: the discriminator moves, then the generator

Built on Day 126's init_net, forward, backward, sigmoid and Adam, imported
unchanged. Checked by hand: the losses at D = 0.5, the generator's gradient
against finite differences, and a short 1-D game.
"""
import os                                              # os: the path to Day 126's folder
import sys                                             # sys: path, to import it
import numpy as np                                    # numpy: every array/matrix operation below is built on this
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week-19-unsupervised-learning-ii"))  # Day 126's folder
import contextlib                                      # contextlib: redirect_stdout, to keep Day 126's checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, sigmoid, Adam


class AdamGAN(d126.Adam):                                 # NEW: Day 134's version, unchanged
    def __init__(self, net, lr=2e-4, b1=0.5):
        super().__init__(net, lr); self.b1 = b1                                       # Day 126's constructor, plus b1

    def step(self, net, grads):                                                       # Day 126's step, with 0.9 replaced by b1
        self.t += 1; c1, c2 = 1 - self.b1 ** self.t, 1 - 0.999 ** self.t
        for (W, b), (gW, gb), m, v in zip(net, grads, self.m, self.v):
            for p, g, mm, vv in ((W, gW, m[0], v[0]), (b, gb, m[1], v[1])):
                mm *= self.b1; mm += (1 - self.b1) * g; vv *= 0.999; vv += 0.001 * g * g
                p -= self.lr * (mm / c1) / (np.sqrt(vv / c2) + 1e-8)


def softplus(Z):                                        # NEW: Day 134's version, unchanged
    return np.maximum(Z, 0) + np.log1p(np.exp(-np.abs(Z)))


def d_loss(lr, lf):                                     # NEW: Day 134's version, unchanged
    loss = softplus(-lr).mean() + softplus(lf).mean()                                  # -log D(real) - log(1 - D(fake)), D = sigmoid(logit)
    return float(loss), (d126.sigmoid(lr) - 1) / len(lr), d126.sigmoid(lf) / len(lf)    # d/dlogit of each term, averaged over the batch


def g_loss(lf, kind="non-saturating"):                  # NEW: Day 134's version, unchanged
    if kind == "non-saturating":                                                      # -log D(G(z)): strong gradients while D is winning
        return float(softplus(-lf).mean()), (d126.sigmoid(lf) - 1) / len(lf)
    return float(-softplus(lf).mean()), -d126.sigmoid(lf) / len(lf)                     # the minimax form, log(1 - D(G(z))), to be minimised


def generate(G, Z, out="sigmoid"):                      # NEW: Day 134's version, unchanged
    A = d126.forward(G, Z, ["relu"] * (len(G) - 1) + ["linear"])                       # Day 126's forward pass
    return (d126.sigmoid(A[-1]) if out == "sigmoid" else A[-1]), A                     # pixels need [0, 1]; 1-D or 2-D data does not


def g_grads(G, D, Z, out="sigmoid", kind="non-saturating"):  # NEW: Day 134's version, unchanged
    dacts = ["relu"] * (len(D) - 1) + ["linear"]; gacts = ["relu"] * (len(G) - 1) + ["linear"]
    F, AG = generate(G, Z, out); Af = d126.forward(D, F, dacts); lg, gf = g_loss(Af[-1], kind)  # noise -> fake -> verdict -> loss
    _, dF = d126.backward(D, Af, dacts, gf)                                           # backward through D to its input: d loss / d fake
    dL = dF * F * (1 - F) if out == "sigmoid" else dF                                 # back through the output sigmoid
    return lg, d126.backward(G, AG, gacts, dL)[0], Af                                 # D's own gradients are never computed here


def gan_step(G, D, oG, oD, X, rng, k, out="sigmoid", kind="non-saturating", d_steps=1):  # NEW: Day 134's version, unchanged
    dacts = ["relu"] * (len(D) - 1) + ["linear"]; n = len(X)                          # D's activations; the batch size
    for _ in range(d_steps):                                                          # 1. the discriminator: real up, fake down
        F, _ = generate(G, rng.standard_normal((n, k)).astype(np.float32), out)       # a fresh batch of fakes
        Ar, Af = d126.forward(D, X, dacts), d126.forward(D, F, dacts)                  # its verdicts on both
        ld, gr, gf = d_loss(Ar[-1], Af[-1])
        g1, _ = d126.backward(D, Ar, dacts, gr); g2, _ = d126.backward(D, Af, dacts, gf)  # the two halves of its gradient
        oD.step(D, [(a[0] + b[0], a[1] + b[1]) for a, b in zip(g1, g2)])              # one Adam step on their sum
    lg, grads, Af = g_grads(G, D, rng.standard_normal((n, k)).astype(np.float32), out, kind)  # 2. the generator: fool the updated discriminator
    oG.step(G, grads)                                                                 # only the generator moves
    return ld, lg, float(d126.sigmoid(Ar[-1]).mean()), float(d126.sigmoid(Af[-1]).mean())  # losses, and D's average verdicts


z = np.zeros((4, 1))
print(f"at D = 0.5: discriminator loss {d_loss(z, z)[0]:.6f} (2 log 2 = {2 * np.log(2):.6f}); generator loss {g_loss(z)[0]:.6f} (log 2 = {np.log(2):.6f})")
print(f"softplus(1000) = {float(softplus(np.array(1000.0))):.1f}; softplus(-1000) = {float(softplus(np.array(-1000.0))):.1f} (no overflow)")
rng = np.random.RandomState(0); Z = rng.standard_normal((8, 2))
G = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([2, 4, 3], 1)]
D = [(W.astype(np.float64), b.astype(np.float64)) for W, b in d126.init_net([3, 4, 1], 2)]
_, grads, _ = g_grads(G, D, Z); W = G[0][0]; old = W[0, 0]
W[0, 0] = old + 1e-6; up = g_grads(G, D, Z)[0]; W[0, 0] = old - 1e-6; dn = g_grads(G, D, Z)[0]; W[0, 0] = old
print(f"d loss / d G[0][0][0, 0]: backward {grads[0][0][0, 0]:.8f}; finite difference {(up - dn) / 2e-6:.8f}")
for gs, ds in (([2, 16, 1], [1, 16, 1]), ([2, 32, 32, 1], [1, 32, 32, 1])):  # a small pair, then the pair used in the lesson
    rng = np.random.RandomState(0); G = d126.init_net(gs, 0); D = d126.init_net(ds, 1); oG, oD = d126.Adam(G, 1e-3), d126.Adam(D, 1e-3)
    for _ in range(3000):                                # real data: N(3, 0.5)
        gan_step(G, D, oG, oD, (3 + 0.5 * rng.standard_normal((128, 1))).astype(np.float32), rng, 2, out="linear")
    F, _ = generate(G, np.random.RandomState(5).standard_normal((5000, 2)).astype(np.float32), "linear")
    x = np.array([[1.0], [3.0], [7.0]], np.float32); Dx = d126.sigmoid(d126.forward(D, x, ["relu"] * (len(D) - 1) + ["linear"])[-1])[:, 0]
    print(f"G {gs}, D {ds}, 3,000 steps on N(3, 0.5): generated mean {F.mean():.3f}, spread {F.std():.3f}; "
          f"D at x = 1, 3, 7: {Dx[0]:.2f} {Dx[1]:.2f} {Dx[2]:.2f}")
