"""
Day 152: Recurrent Networks for Forecasting -- Week 10's LSTM against ARIMA and Regression

Week 10 built an LSTM from scratch: a cell that carries a memory along a
sequence (Day 63), the loop over time (Day 64) and backpropagation through it
(Day 65). There it learned toy token tasks. Today the same parts, imported
unchanged from Day 65's exercise, read a window of past months and predict the
next one; forecasts further ahead feed each prediction back in, as Day 151's
recursive regression did.

A network is not a time-series model by birth. It has no notion of a trend
it can extend: its hidden state is squashed between -1 and 1, so its output
is bounded, and section 2 shows networks trained on CO2's rising level
staying below the highest level they were trained on. Trained on
month-to-month changes instead, they can follow the rise.

A network that wins needs a control: section 3 compares it with a forecast
that has no network at all -- the average change of each calendar month,
added up from the last month.

Everything is written from scratch in NumPy (the optimiser is Adam). Every
setting -- window length, network size, epochs -- is fixed in advance or
chosen on a time-ordered validation block; the test months (2000-2001) are
scored once.

  1. The forecaster, and a gradient check through the whole pipeline.
  2. Levels against changes: the ceiling on what a network can output.
  3. Forecasting CO2 in 2000-2001: five seeds, the training curves, ARIMA, regression and a control.
  4. Rolling origins with models fitted once: 97 forecasts, against ARIMA and regression.
  5. Electrical-equipment orders: the crash and the last two years.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: large matrix products were very slow with this machine's default
import contextlib                                      # contextlib: redirect_stdout, to keep the imported parts quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import Week 10's and this week's parts
import time                                            # time: how long the training takes

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
import matplotlib.dates as mdates                       # year ticks on date axes
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts
sys.path.insert(0, os.path.join(HERE, "..", "week-10-lstm-gru"))  # Week 10's LSTM parts
with contextlib.redirect_stdout(io.StringIO()):         # the exercises print their own checks when imported; keep them quiet
    import day65_exercise as d65                        # Day 65: lstm_forward, lstm_backward
    import day147_exercise as d147                      # Day 147: load_co2, load_elec
    import day149_exercise as d149                      # Day 149: the ARIMA parts
    import day151_exercise as d151                      # Day 151: make_features, fit_ols, recursive_forecast
L = 24                                                  # window: the network reads the last 24 months
HID = 16                                                # hidden units in the LSTM
SEEDS = (0, 1, 2, 3, 4)                                 # five random starts for every network
EPOCHS = 300                                            # passes over the training windows
NVAL = 60                                               # the last 60 training windows are the validation block
AIRLINE = (0, 1, 0, 1, 12)                              # Day 149's airline model


# =============================================================================
# THE PARTS -- NEW (copied unchanged into day152_exercise.py)
# =============================================================================
def windows(z, L):                                      # NEW: inputs = L consecutive values, target = the value after them
    X = np.lib.stride_tricks.sliding_window_view(z[:-1], L)[:, :, None].copy()       # (windows, L steps, 1 input)
    return X, z[L:]                                                                   # each window's next value


def init_params(H, seed, D=1):                          # NEW: random LSTM weights (forget bias 1, as in Week 10) and a linear output
    r = np.random.RandomState(seed)                                                   # reproducible
    P = dict(W=r.randn(D + H, 4 * H) / np.sqrt(D + H), b=np.zeros(4 * H), Wy=r.randn(H) * 0.1, by=np.zeros(1))
    P["b"][:H] = 1.0                                                                  # forget gates start mostly open
    return P


def loss_and_grads(P, X, Y):                            # NEW: Day 65's LSTM, the last hidden state -> one number, mean squared error
    m, H = len(X), P["Wy"].shape[0]                                                   # windows, hidden size
    z0 = np.zeros((m, H))                                                             # both states start at zero
    Hs, Cs, gates = d65.lstm_forward(X, z0, z0, P["W"], P["b"])                       # Day 65's loop over the window
    hT = Hs[:, -1]                                                                    # the last hidden state
    r = hT @ P["Wy"] + P["by"] - Y                                                    # prediction errors
    dy = 2 * r / m                                                                    # d(mean squared error) / d(prediction)
    dW, db = d65.lstm_backward(np.outer(dy, P["Wy"]), X, z0, z0, P["W"], Hs, Cs, gates)  # Day 65's backward pass, from the last h
    return np.mean(r ** 2), dict(W=dW, b=db, Wy=hT.T @ dy, by=np.array([dy.sum()]))


def predict(P, X):                                      # NEW: the network's output for each window
    z0 = np.zeros((len(X), P["Wy"].shape[0]))                                         # zero starting states
    return d65.lstm_forward(X, z0, z0, P["W"], P["b"])[0][:, -1] @ P["Wy"] + P["by"]


def adam_train(X, Y, Xv, Yv, H, seed, epochs=EPOCHS, lr=0.01, bs=32):  # NEW: Adam on mini-batches; keep the epoch best on validation
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


def recursive(P, z, L, h):                              # NEW: predict the next value, append it, repeat h times
    zz = list(z[-L:])                                                                 # the last window of real values
    for _ in range(h):
        zz.append(predict(P, np.array(zz[-L:])[None, :, None])[0])                    # the forecast becomes the newest value
    return np.array(zz[L:])


def fit_lstm(y, L=L, H=HID, seeds=SEEDS, epochs=EPOCHS, nval=NVAL, diff=True):  # NEW: scale, window, train one network per seed
    d = np.diff(y) if diff else y                                                     # month-to-month changes, or the levels
    mu, sd = d.mean(), d.std(); z = (d - mu) / sd                                     # scaled with these training months only
    X, Y = windows(z, L)                                                              # every window and its next value
    fits = [adam_train(X[:-nval], Y[:-nval], X[-nval:], Y[-nval:], H, s, epochs) for s in seeds]  # validation = the last windows
    return dict(models=[f[0] for f in fits], epochs=[f[1] for f in fits], hist=[f[2] for f in fits], mu=mu, sd=sd, L=L, diff=diff)


def lstm_forecast(F, y, h):                             # NEW: each seed's h-month forecast from the end of y, on the original scale
    d = np.diff(y) if F["diff"] else y                                                # the history the networks read
    z = (d - F["mu"]) / F["sd"]                                                       # scaled as in training
    out = []
    for P in F["models"]:
        f = recursive(P, z, F["L"], h) * F["sd"] + F["mu"]                            # forecast changes (or levels), unscaled
        out.append(y[-1] + np.cumsum(f) if F["diff"] else f)                          # changes add up from the last level
    return np.array(out)                                                              # (seeds, h)


def seasonal_drift(y, start, h, years=None):            # NEW: the control -- each calendar month's average change, added up
    d = np.diff(y)                                                                    # month-to-month changes
    mo = (np.datetime64(start, "M").astype(int) + np.arange(1, len(y))) % 12          # the calendar month of each change
    if years:                                                                         # optionally only the last few years
        d, mo = d[-12 * years:], mo[-12 * years:]
    avg = np.array([d[mo == k].mean() for k in range(12)])                            # the average change into each calendar month
    fm = (np.datetime64(start, "M").astype(int) + len(y) + np.arange(h)) % 12         # the calendar months being forecast
    return y[-1] + np.cumsum(avg[fm])                                                 # added up from the last real month


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def split(tc):                                          # Day 147's split: train up to 1999-12, forecast the last 24 months
    return int(np.flatnonzero(tc == np.datetime64("2000-01"))[0])


def airline_fit(y):                                     # Day 149's airline model fitted by CSS: the polynomials
    w = np.diff(y[12:] - y[:-12]); th, s2, _ = d149.fit_css(w, AIRLINE); a, b = d149.sarima_polys(th, AIRLINE)
    return d149.full_ar(a), a, b


def airline_forecast(fit, y, h):                        # forecast h months from the end of y with fixed coefficients
    A, a, b = fit; w = np.diff(y[12:] - y[:-12])                                      # the differenced history
    return d149.forecast(y, A, b, d149.css_residuals(w, a, b), h)                     # errors recomputed on this history


def demo_check():
    print("\n" + "=" * 70)
    print("1. THE FORECASTER -- Day 65's LSTM, a linear output, and a gradient check through all of it")
    print("=" * 70)
    P = init_params(4, 0); X = np.random.RandomState(1).randn(5, 6, 1); Y = np.random.RandomState(2).randn(5)  # a tiny problem
    _, g = loss_and_grads(P, X, Y); rows = []                                         # analytic gradients
    for eps in (1e-6, 1e-5):                                                          # two nudge sizes
        worst = (0.0, 0.0, 0.0)                                                       # (relative error, absolute error, |gradient|)
        for q in P:
            for i in np.ndindex(P[q].shape):                                          # every weight
                o = P[q][i]; P[q][i] = o + eps; lp = loss_and_grads(P, X, Y)[0]; P[q][i] = o - eps; lm = loss_and_grads(P, X, Y)[0]; P[q][i] = o
                num = (lp - lm) / (2 * eps); a = abs(num - g[q][i])                   # central difference against analytic
                worst = max(worst, (a / max(abs(num) + abs(g[q][i]), 1e-8), a, abs(g[q][i])))
        rows.append(worst)
        print(f"nudge {eps:.0e}: worst relative error {worst[0]:.1e} (absolute {worst[1]:.1e}, on a weight whose gradient is {worst[2]:.1e})")
    print(f"first draft used only the 1e-06 nudge and saw {rows[0][0]:.1e}: the worst weight's gradient is tiny, and rounding in the")
    print("two losses swamps the difference; the larger nudge shows the gradients are right")
    print(f"network: a window of {L} months -> LSTM with {HID} hidden units -> one number; {4 * HID * (1 + HID) + 4 * HID + HID + 1} weights")
    print("Actual output from this lesson's run.")


def demo_levels_vs_changes(tc, yc):
    print("\n" + "=" * 70)
    print("2. LEVELS AGAINST CHANGES -- how far a network's forecasts can go")
    print("=" * 70)
    n = split(tc); t0 = time.perf_counter()
    lev = fit_lstm(yc[:n], diff=False); chg = fit_lstm(yc[:n])                        # 5 seeds each, trained up to 1999-12
    print(f"10 networks trained ({time.perf_counter() - t0:.1f} s)")
    fl, fc = lstm_forecast(lev, yc[:n], 24), lstm_forecast(chg, yc[:n], 24)           # 24 months from 1999-12
    cap = [(np.abs(P["Wy"]).sum() + P["by"][0]) * lev["sd"] + lev["mu"] for P in lev["models"]]  # |h| < 1, so output < sum|Wy| + by
    print(f"trained on levels: test MAE by seed {', '.join(f'{np.mean(np.abs(f - yc[n:])):.2f}' for f in fl)} ppm")
    print(f"  the training months reached {yc[:n].max():.1f} ppm and the test months reach {yc[n:].max():.1f}; the networks' forecasts "
          f"reach at most {fl.max():.1f}, below the training maximum")
    print(f"  the hard bound (|h| < 1, so output < sum|Wy| + by) is far higher: {', '.join(f'{c:.1f}' for c in cap)} -- it is not what held them back")
    print(f"trained on month-to-month changes: test MAE by seed {', '.join(f'{np.mean(np.abs(f - yc[n:])):.3f}' for f in fc)} ppm")
    print("a forecast change is added to the last real level, so the forecast can rise past anything in training")
    print("Actual output from this lesson's run.")
    return lev, chg, fl, fc, cap


def demo_forecast(tc, yc, chg, fc):
    print("\n" + "=" * 70)
    print("3. FORECASTING CO2 IN 2000-2001 -- five seeds, against ARIMA, regression and a control")
    print("=" * 70)
    n = split(tc); act = yc[n:]
    print(f"best epoch on validation, by seed: {chg['epochs']} of {EPOCHS}; validation loss at the best epoch "
          f"{', '.join(f'{h[e - 1, 1]:.4f}' for h, e in zip(chg['hist'], chg['epochs']))} (scaled changes; 1 = their variance)")
    print(f"training loss at the last epoch {', '.join(f'{h[-1, 0]:.4f}' for h in chg['hist'])}: below validation, as expected")
    ens = fc.mean(axis=0); ar = airline_forecast(airline_fit(yc[:n]), yc[:n], 24)    # the five networks averaged; Day 149's model
    spec = dict(lags=tuple(range(1, 14))); X, _ = d151.make_features(yc[:n], tc[0], spec)  # Day 151's chosen regression
    rg = d151.recursive_forecast(yc[:n], tc[0], spec, d151.fit_ols(X, yc[:n]), 24)
    sn = yc[n - 12:n][np.arange(24) % 12]                                             # same month last year
    ctl = {yrs: seasonal_drift(yc[:n], tc[0], 24, yrs) for yrs in (None, 10, 5)}      # the control, from all, 10 or 5 years
    mae = lambda f: np.mean(np.abs(f - act))
    print(f"2000-01 to 2001-12, MAE (ppm): LSTM seeds {', '.join(f'{mae(f):.3f}' for f in fc)} (range {min(map(mae, fc)):.3f} to "
          f"{max(map(mae, fc)):.3f}); average of the five {mae(ens):.3f}")
    print(f"  airline ARIMA {mae(ar):.3f}; regression on lags 1-13 (Day 151) {mae(rg):.3f}; same month last year {mae(sn):.3f}")
    print(f"the seed alone moves the error by {max(map(mae, fc)) - min(map(mae, fc)):.3f} ppm; one network's score is not the method's score")
    print(f"the control, no network (each calendar month's average change over all training years, added up): {mae(ctl[None]):.3f}")
    print(f"  the same control from the last 10 years {mae(ctl[10]):.3f}, the last 5 {mae(ctl[5]):.3f}: the window matters, and picking it "
          "now would be choosing on the test")
    bias = lambda f: np.mean(f - act)
    print(f"bias (forecast - actual, mean): LSTM average {bias(ens):+.3f}, airline ARIMA {bias(ar):+.3f}, control {bias(ctl[None]):+.3f}")
    print("Actual output from this lesson's run.")
    return ens, ar, rg, ctl[None]


def demo_rolling(tc, yc):
    print("\n" + "=" * 70)
    print("4. ROLLING ORIGINS -- every model fitted once on data up to 1989-12, then 97 forecasts from 1990-1998")
    print("=" * 70)
    n = split(tc); first = int(np.flatnonzero(tc == np.datetime64("1990-01"))[0]); t0 = time.perf_counter()
    F = fit_lstm(yc[:first]); ar = airline_fit(yc[:first])                            # trained once, on the months before 1990
    spec = dict(lags=tuple(range(1, 14))); X, _ = d151.make_features(yc[:first], tc[0], spec); b = d151.fit_ols(X, yc[:first])
    E = {"LSTM, average of 5": [], "LSTM, single seeds": [], "airline ARIMA": [], "regression, lags 1-13": [], "same month last year": [],
         "control, all years": [], "control, last 10 years": []}
    mo = (tc[:first].astype("datetime64[M]").astype(int) % 12)[1:]; dd = np.diff(yc[:first])  # the control's averages, fixed at 1989-12
    avg = {k: np.array([v[0][v[1] == j].mean() for j in range(12)]) for k, v in
           (("control, all years", (dd, mo)), ("control, last 10 years", (dd[-120:], mo[-120:])))}
    for o in range(first, n - 24 + 1):                                                # each origin: the same models, more history
        act = yc[o:o + 24]; f = lstm_forecast(F, yc[:o], 24)
        E["LSTM, average of 5"].append(np.abs(f.mean(axis=0) - act)); E["LSTM, single seeds"].append(np.abs(f - act).mean(axis=0))
        E["airline ARIMA"].append(np.abs(airline_forecast(ar, yc[:o], 24) - act))
        E["regression, lags 1-13"].append(np.abs(d151.recursive_forecast(yc[:o], tc[0], spec, b, 24) - act))
        E["same month last year"].append(np.abs(yc[o - 12:o][np.arange(24) % 12] - act))
        fm = (tc[o - 1].astype("datetime64[M]").astype(int) + 1 + np.arange(24)) % 12  # the months being forecast
        for k in avg:
            E[k].append(np.abs(yc[o - 1] + np.cumsum(avg[k][fm]) - act))
    E = {k: np.array(v) for k, v in E.items()}
    print(f"{len(E['airline ARIMA'])} origins ({time.perf_counter() - t0:.1f} s, including training)")
    hs = (1, 6, 12, 24)
    print(f"{'MAE (ppm), months ahead':>26} " + " ".join(f"{k:>7}" for k in hs))
    for k, v in E.items():
        print(f"{k:>26} " + " ".join(f"{v[:, j - 1].mean():7.3f}" for j in hs))
    ls, a = E["LSTM, average of 5"].mean(axis=0), E["airline ARIMA"].mean(axis=0)
    print(f"the averaged LSTM beats the airline model at {np.sum(ls < a)} of 24 horizons; averaging the seeds beats the typical single seed "
          f"at {np.sum(ls < E['LSTM, single seeds'].mean(axis=0))} of 24")
    print(f"against the control: the averaged LSTM is better than all years' averages at {np.sum(ls < E['control, all years'].mean(axis=0))} "
          f"of 24 horizons, and than the last 10 years' at {np.sum(ls < E['control, last 10 years'].mean(axis=0))} of 24")
    print("Actual output from this lesson's run.")
    return E, F


def demo_orders(te, ye):
    print("\n" + "=" * 70)
    print("5. ELECTRICAL-EQUIPMENT ORDERS -- log scale; the crash and the last two years")
    print("=" * 70)
    le = np.log(ye); keep = {}                                                        # the season adds on the log scale
    mape = lambda f, act: 100 * np.mean(np.abs(np.exp(f - act) - 1))                  # mean absolute % error
    for name, cut in (("the crash", int(np.flatnonzero(te == np.datetime64("2008-01"))[0])), ("the last 24 months", len(le) - 24)):
        F = fit_lstm(le[:cut], nval=24); act = le[cut:cut + 24]                       # a shorter validation block: less data here
        f = lstm_forecast(F, le[:cut], 24); ar = airline_forecast(airline_fit(le[:cut]), le[:cut], 24); sn = le[cut - 12:cut][np.arange(24) % 12]
        ct = seasonal_drift(le[:cut], te[0], 24)                                      # the control, all training years
        print(f"{name}: train up to {te[cut - 1]} ({cut - 1 - L} windows, the last 24 for validation); 24 months, MAE:")
        print(f"  LSTM seeds {', '.join(f'{mape(g, act):.2f}%' for g in f)}; average {mape(f.mean(axis=0), act):.2f}%; "
              f"airline ARIMA {mape(ar, act):.2f}%; same month last year {mape(sn, act):.2f}%; control {mape(ct, act):.2f}%")
        keep[name] = (cut, f.mean(axis=0), ar)
    print("Actual output from this lesson's run.")
    return keep


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(tc, yc, te, ye, lev, chg, fl, fc, cap, fore, roll, orders, out):
    fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 3); cols = ("#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974")
    ax = fig.add_subplot(gs[0, 0])                                                    # the training curves
    for h, e, c in zip(chg["hist"], chg["epochs"], cols):
        ax.plot(h[:, 0], color=c, lw=0.8, alpha=0.6); ax.plot(h[:, 1], color=c, lw=1.3); ax.plot(e - 1, h[e - 1, 1], "o", color=c, ms=5)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("mean squared error (scaled changes)")
    ax.plot([], [], color="gray", lw=0.8, alpha=0.6, label="training (thin)"); ax.plot([], [], color="gray", lw=1.3, label="validation (thick)")
    ax.plot([], [], "o", color="gray", label="best epoch, kept"); ax.legend(fontsize=8); ax.set_title("five seeds: training and validation loss", fontsize=10)
    n = split(tc); ax = fig.add_subplot(gs[0, 1]); x = tc.astype("datetime64[D]"); s = n - 60
    ax.plot(x[s:], yc[s:], color="k", lw=1.2, label="actual")
    ax.plot(x[n:], fl.mean(axis=0), color="#C44E52", lw=1.2, label="LSTM on levels (average)")
    ax.plot(x[n:], fc.mean(axis=0), color="#4C72B0", lw=1.2, label="LSTM on changes (average)")
    ax.axhline(yc[:n].max(), color="#C44E52", ls="--", lw=0.8, label="highest training month")
    ax.axvline(x[n], color="gray", ls=":"); ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("trained on levels, forecasts stay below the training maximum", fontsize=10)
    ens, ar, rg, ctl = fore; ax = fig.add_subplot(gs[0, 2]); act = yc[n:]
    ax.plot(x[n - 12:], yc[n - 12:], color="k", lw=1.2, label="actual")
    for f, c in zip(fc, cols):
        ax.plot(x[n:], f, color=c, lw=0.7, alpha=0.7)
    ax.plot(x[n:], ens, color="#4C72B0", lw=2, label="LSTM, average of 5 (thin: each seed)")
    ax.plot(x[n:], ar, color="#8172B2", lw=1.2, ls="--", label="airline ARIMA (Day 149)")
    ax.plot(x[n:], ctl, color="#DD8452", lw=1.2, ls=":", label="control: average change per month")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # one tick per year
    ax.axvline(x[n], color="gray", ls=":"); ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("2000-2001 forecasts from data up to 1999-12", fontsize=10)
    E, _ = roll; ax = fig.add_subplot(gs[1, 0]); hz = np.arange(1, 25)
    for (name, v), c in zip(E.items(), ("#4C72B0", "#9DB7DD", "#8172B2", "#55A868", "#DD8452", "#937860", "#DA8BC3")):
        if name != "same month last year":                                            # off the scale; its numbers are printed
            ax.plot(hz, v.mean(axis=0), marker="o", ms=3, color=c, label=name)
    ax.set_xlabel("months ahead"); ax.set_ylabel("MAE (ppm)"); ax.legend(fontsize=7)
    ax.set_title("97 origins, models fitted once on data to 1989-12", fontsize=10)
    for k, (name, title) in enumerate((("the crash", "the 2008-09 crash"), ("the last 24 months", "the last 24 months"))):
        cut, f, a = orders[name]; ax = fig.add_subplot(gs[1, 1 + k]); x = te.astype("datetime64[D]"); s = cut - 48
        ax.plot(x[s:cut + 24], ye[s:cut + 24], color="k", lw=1.2, label="actual orders")
        ax.plot(x[cut:cut + 24], np.exp(f), color="#4C72B0", label="LSTM, average of 5")
        ax.plot(x[cut:cut + 24], np.exp(a), color="#8172B2", ls="--", label="airline ARIMA")
        ax.axvline(x[cut], color="gray", ls=":"); ax.set_ylabel("orders (index)"); ax.legend(fontsize=8, loc="lower left"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the forecaster, from the series to a 24-month forecast
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 15); ax.set_ylim(0.3, 5.5); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.3, 3.6, 2.1, 1.0, "monthly series\ny (CO2, ppm)", "#DDEBF7")
    box(3.9, 3.6, 2.3, 1.0, "month-to-month\nchanges, scaled\n(training mean, sd)", "#DDEBF7")
    box(6.6, 3.6, 2.3, 1.0, "the last 24\nchanges", "#FFF2CC")
    box(9.3, 3.6, 2.3, 1.0, "LSTM, 16 units\n(Day 65's forward\nand backward)", "#E2F0D9")
    box(12.2, 3.6, 2.6, 1.0, "last hidden state x Wy + by\n= next month's change", "#E2F0D9")
    box(12.2, 1.4, 2.6, 1.0, "unscale; add up the changes\nfrom the last real month", "#FCE4D6")
    box(7.6, 1.4, 4.2, 1.0, "forecast for 24 months; five seeds, averaged", "#FCE4D6")
    arrow(2.35, 3.6, 2.75, 3.6); arrow(5.05, 3.6, 5.45, 3.6); arrow(7.75, 3.6, 8.15, 3.6); arrow(10.45, 3.6, 10.9, 3.6)
    arrow(12.2, 3.1, 12.2, 1.9); arrow(10.9, 1.4, 9.7, 1.4)
    ax.annotate("", xy=(6.6, 4.12), xytext=(12.2, 4.12), arrowprops=dict(arrowstyle="->", lw=1.3, connectionstyle="arc3,rad=0.3"))
    ax.text(9.4, 4.4, "recursive: the forecast change joins the window,\nand the next month is predicted", ha="center", va="bottom", fontsize=8.5)
    ax.text(7.5, 0.55, "training: Adam on mini-batches of windows; the epoch with the lowest loss on the last 60 training windows is kept",
            ha="center", fontsize=8.5, style="italic")
    ax.set_title("Day 152: an LSTM forecaster built from Week 10's parts", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "lstm_forecasting_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 153: the Week 22 build -- a Prophet-style forecaster from this week's parts: a trend that may bend at")
    print("changepoints, a season made of Fourier waves, fitted by least squares, and compared with every model so far.")


def main():                                             # runs every section in order
    tc, yc, _ = d147.load_co2(); te, ye = d147.load_elec()                           # Day 147's two series
    demo_check()                                                                      # 1. the forecaster, gradient-checked
    lev, chg, fl, fc, cap = demo_levels_vs_changes(tc, yc)                            # 2. levels against changes
    fore = demo_forecast(tc, yc, chg, fc)                                             # 3. 2000-2001
    roll = demo_rolling(tc, yc)                                                       # 4. rolling origins
    orders = demo_orders(te, ye)                                                      # 5. the orders
    plot_results(tc, yc, te, ye, lev, chg, fl, fc, cap, fore, roll, orders, os.path.join(HERE, "lstm_forecasting_results.png"))
    draw_diagram()                                                                    # the forecaster
    print("\nsaved lstm_forecasting_results.png\nsaved lstm_forecasting_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
