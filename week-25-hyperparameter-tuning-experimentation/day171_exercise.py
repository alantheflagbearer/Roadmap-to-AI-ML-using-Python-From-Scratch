"""
Day 171 Exercise -- Early Stopping

The fourth of Week 25's parts for Day 174's reproducible tuning study.

    copy_net(net)                    an independent copy of a network's weights
    EarlyStopper(patience, min_delta)    update(epoch, score, net) returns True when to stop
    stop_at(curve, patience)         the same rule replayed on a recorded curve: (stop, best)
    gl_stop_at(curve, alpha)         Prechelt's generalisation-loss rule on a curve
    row_losses(net, acts, X, y)      each row's log loss
    train_epoch(net, opt, acts, ...) one pass over the training rows in a new order
    train_es(Xa, ya, Xs, ys, ...)    train Day 170's network with early stopping;
                                     returns the best epoch's weights and the curve

Built on Day 170's exercise (bce_logits, HIDDEN) and Day 126's network,
imported unchanged. Checked by hand: a copy that does not change, a stopper
that must stop at a known epoch, min_delta, the GL rule on a short curve,
the log loss at a logit of 0, and a run whose restored weights must give
exactly the lowest loss on its curve.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep Day 170's checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: this file's folder
import sys                                             # sys: path, to import Day 170's exercise

import numpy as np                                     # numpy: every array operation below is built on this

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this week's folder, where Day 170's exercise lives
with contextlib.redirect_stdout(io.StringIO()):         # Day 170's exercise prints its checks when imported
    import day170_exercise as d170                      # Day 170: bce_logits, HIDDEN, and Day 126 through it
d126 = d170.d126                                        # Day 126: init_net, forward, backward, Adam

MAX_EPOCHS = 100                                        # the longest any run trains


def copy_net(net):                                      # an independent copy of a network's weights
    return [(W.copy(), b.copy()) for W, b in net]                                     # new arrays: later steps do not change the copy


class EarlyStopper:                                     # stop when the score has not improved for `patience` epochs
    def __init__(self, patience=5, min_delta=0.0):      # min_delta: how much better counts as better
        self.patience, self.min_delta = patience, min_delta                           # the rule
        self.best, self.best_epoch, self.best_net, self.wait = np.inf, -1, None, 0    # nothing seen yet

    def update(self, epoch, score, net=None):           # call once per epoch; returns True when training should stop
        if score < self.best - self.min_delta:                                        # a real improvement
            self.best, self.best_epoch, self.wait = score, epoch, 0                   # remember it; reset the count
            self.best_net = copy_net(net) if net is not None else None                # and the weights that gave it
        else:                                                                         # no improvement
            self.wait += 1                                                            # one more epoch without one
        return self.wait >= self.patience                                             # out of patience


def stop_at(curve, patience, min_delta=0.0):            # replay EarlyStopper on a recorded curve: (stop epoch, best epoch)
    es = EarlyStopper(patience, min_delta)                                            # the same class, so the same rule
    for e, v in enumerate(curve):                                                     # epoch by epoch
        if es.update(e, v):                                                           # it would have stopped here
            return e, es.best_epoch                                                   # where it stopped; the weights it restores
    return len(curve) - 1, es.best_epoch                                              # never stopped: ran to the end


def gl_stop_at(curve, alpha):                           # Prechelt (1998): stop once the loss is alpha percent above its best so far
    best = np.minimum.accumulate(np.asarray(curve, float))                            # the best loss up to each epoch
    gl = 100 * (np.asarray(curve, float) / best - 1)                                  # generalisation loss, in percent
    hit = np.flatnonzero(gl > alpha)                                                  # epochs where the rule fires
    e = int(hit[0]) if len(hit) else len(curve) - 1                                   # the first one, or the end
    return e, int(np.argmin(curve[:e + 1]))                                           # where it stopped; the best epoch up to then


def row_losses(net, acts, X, y):                        # each row's log loss (float64, so every caller averages it identically)
    z = d126.forward(net, X, acts)[-1][:, 0].astype(float)                            # one logit per row
    return np.logaddexp(0, z) - y * z                                                 # log(1 + e^z) - y z, without overflow


def train_epoch(net, opt, acts, Xa, ya, rng, batch=128):  # one pass over the training rows in a new random order
    o = rng.permutation(len(Xa))                                                      # the order
    for s in range(0, len(Xa), batch):                                                # each batch
        A = d126.forward(net, Xa[o[s:s + batch]], acts)                               # forward
        opt.step(net, d126.backward(net, A, acts, d170.bce_logits(A[-1], ya[o[s:s + batch]])[1])[0])  # backward and step


def train_es(Xa, ya, Xs, ys, opt_cls, lr, seed, stopper, max_epochs=MAX_EPOCHS, batch=128):  # train with early stopping
    sizes = [Xa.shape[1], *d170.HIDDEN, 1]; acts = ["relu"] * len(d170.HIDDEN) + ["linear"]  # Day 170's network
    net = d126.init_net(sizes, seed); opt = opt_cls(net, lr); rng = np.random.RandomState(seed + 1)  # as in Day 170's train_mlp
    curve = []                                                                        # the stop set's loss after each epoch
    for e in range(max_epochs):                                                       # up to the limit
        train_epoch(net, opt, acts, Xa, ya, rng, batch)                               # one epoch
        curve.append(row_losses(net, acts, Xs, ys).mean())                            # score it on the stop set
        if stopper.update(e, curve[-1], net):                                         # out of patience
            break                                                                     # stop
    return stopper.best_net, acts, np.array(curve)                                    # the best epoch's weights, not the last


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    net = [(np.ones((2, 1)), np.zeros(1))]; cp = copy_net(net); net[0][0][0, 0] = 9.0  # change the original after copying
    print(f"copy_net: original weight now {net[0][0][0, 0]:.0f}, copy {cp[0][0][0, 0]:.0f} (by hand 9 and 1)")
    print(f"stop_at([5, 4, 4.5, 4.2, 4.1, 3.9], patience 2): {stop_at([5, 4, 4.5, 4.2, 4.1, 3.9], 2)} "
          f"(by hand (3, 1): best at epoch 1, then epochs 2 and 3 are no better, so it stops at 3 and never sees 3.9)")
    print(f"stop_at([1.0, 0.995, 0.99, 0.985], patience 2, min_delta 0.01): {stop_at([1.0, 0.995, 0.99, 0.985], 2, 0.01)} "
          f"(by hand (2, 0): neither 0.995 nor 0.99 is 0.01 better than 1.0)")
    print(f"gl_stop_at([1.0, 0.8, 0.82, 0.85], alpha 5): {gl_stop_at([1.0, 0.8, 0.82, 0.85], 5)} "
          f"(by hand (3, 1): 0.82 is 2.5% above the best 0.8, 0.85 is 6.25%)")
    z = [(np.zeros((2, 1)), np.zeros(1))]                                            # a network whose logit is always 0
    print(f"row_losses at logit 0, labels 1 and 0: {np.round(row_losses(z, ['linear'], np.ones((2, 2)), np.array([1.0, 0.0])), 4).tolist()} "
          f"(by hand ln 2 = 0.6931 for both)")
    rng = np.random.RandomState(0); X = rng.randn(800, 5).astype(np.float32)        # a toy problem: the sign of x0, with label noise
    y = ((X[:, 0] > 0) ^ (rng.rand(800) < 0.2)).astype(np.float32)                   # 20% of labels flipped, so the network can over-fit
    es = EarlyStopper(5); best, acts, curve = train_es(X[:200], y[:200], X[200:], y[200:], d126.Adam, 1e-3, 0, es, max_epochs=300, batch=32)  # stop, then restore the best epoch's weights
    print(f"toy problem (200 training rows, 20% noisy labels): stopped after {len(curve)} of 300 epochs, best epoch {es.best_epoch + 1}; "
          f"restored weights score {row_losses(best, acts, X[200:], y[200:]).mean():.6f}, the curve's lowest {curve.min():.6f} (must be equal; "
          f"no model can beat the noise floor -0.2 ln 0.2 - 0.8 ln 0.8 = 0.5004)")
