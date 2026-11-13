"""
Day 170 Exercise -- A Learning-Rate Finder

The third of Week 25's parts for Day 174's reproducible tuning study.

    lr_range(lo, hi, steps)          learning rates rising by the same factor every step
    lr_find(step, lrs)               one finder run; step(lr) trains one step and returns the loss
    suggest(lrs, sm)                 one tenth of the lowest loss's rate, and the steepest point
    SGD(net, lr)                     plain gradient descent with the interface of Day 126's Adam
    bce_logits(L, y)                 log loss on logits, and its gradient
    train_mlp(Xa, ya, Xv, yv, ...)   one real training run of Day 126's network; validation log loss
    mlp_stepper(Xa, ya, opt_cls, seed)   a step(lr) function for lr_find

Built on Day 126's init_net, forward, backward, sigmoid and Adam, imported
unchanged. Checked by hand: the rates, a finder that must stop at a known
step, both suggestions on a curve with a known minimum, one SGD step, the
log loss at a logit of 0, and a finder and a training run on a toy problem.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep Day 126's checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: the path to Day 126's folder
import sys                                             # sys: path, to import it

import numpy as np                                     # numpy: every array operation below is built on this

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "week-19-unsupervised-learning-ii"))  # Day 126's folder
with contextlib.redirect_stdout(io.StringIO()):         # Day 126's exercise prints its checks when imported
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, sigmoid, Adam

HIDDEN = (64, 32)                                       # the network's two hidden layers
EPOCHS = 5                                              # passes over the training rows in each real run
BATCH = 128                                             # rows per step
LO, HI = 1e-6, 10.0                                     # the finder's range of learning rates


def lr_range(lo, hi, steps):                            # learning rates rising by the same factor every step
    return lo * (hi / lo) ** (np.arange(steps) / (steps - 1))                         # lo, ..., hi, evenly spaced on a log scale


def lr_find(step, lrs, beta=0.98, blowup=4.0):          # one finder run: step(lr) trains one step and returns that batch's loss
    raw, sm, avg, best = [], [], 0.0, np.inf                                          # losses as measured; smoothed; running average; best smoothed
    for t, lr in enumerate(lrs):                                                      # rising learning rates
        loss = step(lr); avg = beta * avg + (1 - beta) * loss                         # one step; exponential moving average
        s = avg / (1 - beta ** (t + 1))                                               # corrected for starting the average at 0
        raw.append(loss); sm.append(s)                                                # record both
        if not np.isfinite(s) or s > blowup * best:                                   # the loss has blown up
            break                                                                     # stop: higher rates only get worse
        best = min(best, s)                                                           # the best smoothed loss so far
    return np.asarray(lrs[:len(sm)]), np.array(raw), np.array(sm)                     # the rates used and both loss curves


def suggest(lrs, sm, skip=10):                          # two common ways to read the curve
    g = np.gradient(sm, np.log(lrs))                                                  # slope of the smoothed loss against log learning rate
    steep = lrs[skip + int(np.argmin(g[skip:-5]))] if len(sm) > skip + 6 else lrs[0]  # where it falls fastest (ignoring both ends)
    return dict(min_over_10=float(lrs[int(np.argmin(sm))] / 10), steepest=float(steep))  # one tenth of the minimum's rate; the steepest point


class SGD:                                              # plain stochastic gradient descent, with the same interface as Day 126's Adam
    def __init__(self, net, lr=1e-2):                   # net is not needed: SGD keeps no memory
        self.lr = lr                                                                  # the learning rate, which the finder changes each step

    def step(self, net, grads):                         # one step against the gradient
        for (W, b), (gW, gb) in zip(net, grads):                                      # each layer
            W -= self.lr * gW; b -= self.lr * gb                                      # in place, so the network itself changes


def bce_logits(L, y):                                   # log loss on a network's raw outputs (logits), and its gradient
    z = L[:, 0]                                                                       # one logit per row
    loss = float(np.mean(np.logaddexp(0, z) - y * z))                                 # log(1 + e^z) - y z: log loss without overflow
    return loss, ((d126.sigmoid(z) - y) / len(y))[:, None].astype(np.float32)         # gradient: (p - y) / n


def train_mlp(Xa, ya, Xv, yv, opt_cls, lr, seed, epochs=EPOCHS, batch=BATCH):  # one real training run; returns validation log loss
    sizes = [Xa.shape[1], *HIDDEN, 1]; acts = ["relu"] * len(HIDDEN) + ["linear"]    # the network's shape
    net = d126.init_net(sizes, seed); opt = opt_cls(net, lr); rng = np.random.RandomState(seed + 1)  # network, optimiser, batch order
    with np.errstate(all="ignore"):                                                   # a diverging run overflows; its loss shows that
        for _ in range(epochs):                                                       # passes over the data
            o = rng.permutation(len(Xa))                                              # a new order each pass
            for s in range(0, len(Xa), batch):                                        # each batch
                A = d126.forward(net, Xa[o[s:s + batch]], acts)                       # forward
                opt.step(net, d126.backward(net, A, acts, bce_logits(A[-1], ya[o[s:s + batch]])[1])[0])  # backward and step
        v = bce_logits(d126.forward(net, Xv, acts)[-1], yv)[0]                        # validation log loss
    return v if np.isfinite(v) else np.inf                                            # a diverged run scores infinity


def mlp_stepper(Xa, ya, opt_cls, seed, batch=BATCH):    # a step(lr) function for lr_find: one batch of a fresh network
    sizes = [Xa.shape[1], *HIDDEN, 1]; acts = ["relu"] * len(HIDDEN) + ["linear"]    # the same network as train_mlp
    net = d126.init_net(sizes, seed); opt = opt_cls(net, LO); rng = np.random.RandomState(seed + 1)  # fresh network and optimiser
    state = dict(o=rng.permutation(len(Xa)), pos=0)                                   # where we are in the shuffled rows

    def step(lr):                                       # one finder step at learning rate lr
        if state["pos"] + batch > len(Xa):                                            # end of the rows: reshuffle
            state["o"], state["pos"] = rng.permutation(len(Xa)), 0                    # a new order
        b = state["o"][state["pos"]:state["pos"] + batch]; state["pos"] += batch      # the next batch
        A = d126.forward(net, Xa[b], acts); loss, d = bce_logits(A[-1], ya[b])        # its loss before the step
        opt.lr = lr; opt.step(net, d126.backward(net, A, acts, d)[0])                 # the step, at this rate
        return loss                                                                   # the loss the finder records
    return step                                                                       # a closure: it remembers net, opt and state


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    print(f"lr_range(0.001, 1, 4): {lr_range(0.001, 1, 4).round(6).tolist()} (by hand 0.001, 0.01, 0.1, 1)")
    lrs, raw, sm = lr_find(lambda lr: 1.0 if lr < 0.05 else 10.0, lr_range(0.001, 1, 4), beta=0.0)  # the loss jumps to 10 above rate 0.05
    print(f"lr_find, loss 1 below rate 0.05 and 10 above it, no smoothing: stopped after {len(sm)} rates at {lrs[-1]:.3g} "
          f"(by hand 3, at 0.1: there 10 > 4 x 1; the step that blew up is kept as the last)")
    lrs = lr_range(1e-4, 1, 41); sm = (np.log10(lrs) + 2) ** 2  # a bowl with its lowest point at rate 0.01
    s = suggest(lrs, sm)  # both rules
    print(f"suggest on a bowl lowest at 0.01: min / 10 {s['min_over_10']:.4g} (by hand 0.001); steepest {s['steepest']:.4g} "
          f"(by hand 0.001: the slope is steepest at the left end, and the first 10 rates are skipped)")
    net = [(np.array([[1.0]]), np.array([0.0]))]; SGD(net, 0.1).step(net, [(np.array([[0.5]]), np.array([1.0]))])  # one step
    print(f"SGD, weight 1, gradient 0.5, rate 0.1: weight {net[0][0][0, 0]:.3f}, bias {net[0][1][0]:.3f} (by hand 0.95 and -0.1)")
    loss, g = bce_logits(np.array([[0.0]]), np.array([1.0]))  # a logit of 0 means probability 0.5
    print(f"bce_logits at logit 0, label 1: loss {loss:.4f} (by hand ln 2 = 0.6931), gradient {g[0, 0]:.2f} (by hand 0.5 - 1 = -0.5)")
    rng = np.random.RandomState(0); X = rng.randn(400, 2).astype(np.float32); y = (X[:, 0] > 0).astype(np.float32)  # a toy problem: the sign of x0
    lrs, raw, sm = lr_find(mlp_stepper(X[:300], y[:300], d126.Adam, 0, batch=32), lr_range(1e-5, 10, 150))  # a finder run on it
    v = train_mlp(X[:300], y[:300], X[300:], y[300:], d126.Adam, suggest(lrs, sm)["min_over_10"], 0, epochs=20, batch=32)  # train there
    print(f"toy problem (y = 1 when x0 > 0): finder stopped at {lrs[-1]:.3g}, suggests {suggest(lrs, sm)['min_over_10']:.3g}; "
          f"20 epochs there give validation log loss {v:.3f} (well below ln 2 = 0.693)")
