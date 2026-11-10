"""
Day 152 Exercise -- An LSTM Forecaster from Week 10's Parts

The sixth of Week 22's parts for Day 153's Prophet-style forecaster.

    windows(z, L)                       inputs of L values, and the value after each
    init_params(H, seed)                LSTM weights (forget bias 1) and a linear output
    loss_and_grads(P, X, Y)             Day 65's LSTM -> last hidden state -> one number; mean squared error
    predict(P, X)                       the network's output for each window
    adam_train(X, Y, Xv, Yv, H, seed)   Adam; keeps the epoch best on the validation windows
    recursive(P, z, L, h)               feed each prediction back in, h times
    fit_lstm(y)                         differences, scaling, windows, one network per seed
    lstm_forecast(F, y, h)              every seed's forecast on the original scale
    seasonal_drift(y, start, h)         the control: each calendar month's average change, added up

Checked by hand: the windows of 1..6, the starting biases, the loss and one
gradient of a network whose weights are all zero, a numerical gradient
check, training on a constant series, the recursion and the adding-up of
changes for networks that output a constant, and the control on a series
that rises by 1 every month.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported part quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: this file's folder
import sys                                             # sys: path, to import Week 10's part
import numpy as np                                     # numpy: every array operation below is built on this
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week-10-lstm-gru"))  # Week 10's parts
with contextlib.redirect_stdout(io.StringIO()):         # Day 65's exercise prints its own gradient check when imported
    import day65_exercise as d65                        # Day 65: lstm_forward, lstm_backward
L = 24                                                  # window: the network reads the last 24 months
HID = 16                                                # hidden units in the LSTM
SEEDS = (0, 1, 2, 3, 4)                                 # five random starts for every network
EPOCHS = 300                                            # passes over the training windows
NVAL = 60                                               # the last 60 training windows are the validation block


def windows(z, L):                                      # NEW: Day 152's version, unchanged
    X = np.lib.stride_tricks.sliding_window_view(z[:-1], L)[:, :, None].copy()       # (windows, L steps, 1 input)
    return X, z[L:]                                                                   # each window's next value


def init_params(H, seed, D=1):                          # NEW: Day 152's version, unchanged
    r = np.random.RandomState(seed)                                                   # reproducible
    P = dict(W=r.randn(D + H, 4 * H) / np.sqrt(D + H), b=np.zeros(4 * H), Wy=r.randn(H) * 0.1, by=np.zeros(1))
    P["b"][:H] = 1.0                                                                  # forget gates start mostly open
    return P


def loss_and_grads(P, X, Y):                            # NEW: Day 152's version, unchanged
    m, H = len(X), P["Wy"].shape[0]                                                   # windows, hidden size
    z0 = np.zeros((m, H))                                                             # both states start at zero
    Hs, Cs, gates = d65.lstm_forward(X, z0, z0, P["W"], P["b"])                       # Day 65's loop over the window
    hT = Hs[:, -1]                                                                    # the last hidden state
    r = hT @ P["Wy"] + P["by"] - Y                                                    # prediction errors
    dy = 2 * r / m                                                                    # d(mean squared error) / d(prediction)
    dW, db = d65.lstm_backward(np.outer(dy, P["Wy"]), X, z0, z0, P["W"], Hs, Cs, gates)  # Day 65's backward pass, from the last h
    return np.mean(r ** 2), dict(W=dW, b=db, Wy=hT.T @ dy, by=np.array([dy.sum()]))


def predict(P, X):                                      # NEW: Day 152's version, unchanged
    z0 = np.zeros((len(X), P["Wy"].shape[0]))                                         # zero starting states
    return d65.lstm_forward(X, z0, z0, P["W"], P["b"])[0][:, -1] @ P["Wy"] + P["by"]


def adam_train(X, Y, Xv, Yv, H, seed, epochs=EPOCHS, lr=0.01, bs=32):  # NEW: Day 152's version, unchanged
    P = init_params(H, seed); m1 = {k: 0 * v for k, v in P.items()}; m2 = {k: 0 * v for k, v in P.items()}  # weights, Adam's averages
    r, k = np.random.RandomState(seed + 100), 0                                       # batch order, update count
    best, hist = (np.inf, None, 0), []                                                # (validation loss, weights, epoch), the curves
    for ep in range(1, epochs + 1):
        idx = r.permutation(len(X)); tl = 0.0                                         # a new order of the training windows
        for s in range(0, len(X), bs):                                                # one mini-batch at a time
            j = idx[s:s + bs]; loss, g = loss_and_grads(P, X[j], Y[j]); k += 1; tl += loss * len(j)
            for q in P:                                                               # Adam: averaged gradient over its typical size
                m1[q] = 0.9 * m1[q] + 0.1 * g[q]; m2[q] = 0.999 * m2[q] + 0.001 * g[q] ** 2
                P[q] -= lr * (m1[q] / (1 - 0.9 ** k)) / (np.sqrt(m2[q] / (1 - 0.999 ** k)) + 1e-8)
        vl = np.mean((predict(P, Xv) - Yv) ** 2); hist.append((tl / len(X), vl))      # training and validation loss this epoch
        if vl < best[0]:                                                              # the best epoch so far on validation
            best = (vl, {q: w.copy() for q, w in P.items()}, ep)
    return best[1], best[2], np.array(hist)


def recursive(P, z, L, h):                              # NEW: Day 152's version, unchanged
    zz = list(z[-L:])                                                                 # the last window of real values
    for _ in range(h):
        zz.append(predict(P, np.array(zz[-L:])[None, :, None])[0])                    # the forecast becomes the newest value
    return np.array(zz[L:])


def fit_lstm(y, L=L, H=HID, seeds=SEEDS, epochs=EPOCHS, nval=NVAL, diff=True):  # NEW: Day 152's version, unchanged
    d = np.diff(y) if diff else y                                                     # month-to-month changes, or the levels
    mu, sd = d.mean(), d.std(); z = (d - mu) / sd                                     # scaled with these training months only
    X, Y = windows(z, L)                                                              # every window and its next value
    fits = [adam_train(X[:-nval], Y[:-nval], X[-nval:], Y[-nval:], H, s, epochs) for s in seeds]  # validation = the last windows
    return dict(models=[f[0] for f in fits], epochs=[f[1] for f in fits], hist=[f[2] for f in fits], mu=mu, sd=sd, L=L, diff=diff)


def lstm_forecast(F, y, h):                             # NEW: Day 152's version, unchanged
    d = np.diff(y) if F["diff"] else y                                                # the history the networks read
    z = (d - F["mu"]) / F["sd"]                                                       # scaled as in training
    out = []
    for P in F["models"]:
        f = recursive(P, z, F["L"], h) * F["sd"] + F["mu"]                            # forecast changes (or levels), unscaled
        out.append(y[-1] + np.cumsum(f) if F["diff"] else f)                          # changes add up from the last level
    return np.array(out)                                                              # (seeds, h)


def seasonal_drift(y, start, h, years=None):            # NEW: Day 152's version, unchanged
    d = np.diff(y)                                                                    # month-to-month changes
    mo = (np.datetime64(start, "M").astype(int) + np.arange(1, len(y))) % 12          # the calendar month of each change
    if years:                                                                         # optionally only the last few years
        d, mo = d[-12 * years:], mo[-12 * years:]
    avg = np.array([d[mo == k].mean() for k in range(12)])                            # the average change into each calendar month
    fm = (np.datetime64(start, "M").astype(int) + len(y) + np.arange(h)) % 12         # the calendar months being forecast
    return y[-1] + np.cumsum(avg[fm])                                                 # added up from the last real month


X, Y = windows(np.arange(1.0, 7.0), 3)
print(f"windows of 1..6, length 3: inputs [1,2,3], [2,3,4], [3,4,5], targets 4, 5, 6; gives {X[:, :, 0].tolist()}, {Y.tolist()}")
P = init_params(2, 0)
print(f"starting biases, 2 hidden units: forget gates 1, the rest 0; gives {P['b'].tolist()}")
Z = {k: np.zeros_like(v) for k, v in P.items()}                                       # every weight zero: gates 0.5, candidate 0, h = 0
loss, g = loss_and_grads(Z, np.ones((2, 3, 1)), np.array([1.0, 3.0]))
print(f"all-zero network, targets 1 and 3: output 0, loss (1 + 9) / 2 = 5, gradient for by = 2 x mean(0 - target) = -4; "
      f"gives {loss:.1f}, {g['by'][0]:.1f}")
Xs, Ys = np.random.RandomState(1).randn(4, 5, 1), np.random.RandomState(2).randn(4); Pg = init_params(3, 0)
_, ga = loss_and_grads(Pg, Xs, Ys); worst = 0.0
for q in Pg:
    for i in np.ndindex(Pg[q].shape):
        o = Pg[q][i]; Pg[q][i] = o + 1e-5; lp = loss_and_grads(Pg, Xs, Ys)[0]; Pg[q][i] = o - 1e-5; lm = loss_and_grads(Pg, Xs, Ys)[0]; Pg[q][i] = o
        worst = max(worst, abs((lp - lm) / 2e-5 - ga[q][i]))
print(f"numerical gradient check, every weight of a 3-unit network: largest |numeric - analytic| {worst:.1e}")
Xc, Yc = windows(np.full(40, 0.5), 4)
Pc, ep, hist = adam_train(Xc[:-5], Yc[:-5], Xc[-5:], Yc[-5:], 4, 0, epochs=60)
print(f"trained on a series that is always 0.5: predicts {predict(Pc, Xc[:1])[0]:.3f} (validation loss {hist[ep - 1, 1]:.1e} at epoch {ep})")
C = {k: np.zeros_like(v) for k, v in P.items()}; C["by"] = np.array([0.7])           # a network that always outputs 0.7
print(f"recursion with a network that always outputs 0.7: 0.7, 0.7, 0.7; gives {np.round(recursive(C, np.zeros(5), 3, 3), 10).tolist()}")
F = dict(models=[Z], mu=2.0, sd=1.0, L=3, diff=True)                                  # outputs 0 scaled change = a change of 2
print(f"changes of 2 added up from 16 (history 10, 12, 14, 16): 18, 20, 22; lstm_forecast gives {lstm_forecast(F, np.array([10.0, 12, 14, 16]), 3)[0].tolist()}")
print(f"control on 0, 1, ..., 24 (every change 1): 25, 26, 27; seasonal_drift gives {seasonal_drift(np.arange(25.0), np.datetime64('2000-01'), 3).tolist()}")
