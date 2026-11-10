"""
Day 151: Feature Engineering for Time Series -- Lags, Rolling Statistics, Calendar Features, and Leakage

Days 149 and 150 forecast with models built for time series. Another route is
to turn forecasting into ordinary regression: for every month, write down
features known before that month (its value 1 and 12 months earlier, the mean
of the last year, which calendar month it is) and fit any regression model
from features to value.

The danger is in the words 'known before that month'. A feature that quietly
contains the month itself, or a later month, makes a model look far better
than it can ever be in use. Such leaks are easy to write and hard to see,
because nothing fails: the numbers just look good.

Everything is written from scratch in NumPy; pandas and statsmodels are used
only to check. Every setting is chosen on earlier data only; the test months
(2000-2001, as on Days 147-150) are used once, at the end.

  1. Building the features, checked against pandas and statsmodels.
  2. Four leaks, and what each does to the measured error.
  3. Cross-validation: shuffled folds against time-ordered folds.
  4. Choosing the features for CO2 by time-ordered validation; recursive and direct forecasts.
  5. Rolling origins: 97 forecasts, against Day 149's ARIMA.
  6. Electrical-equipment orders: an ordinary period and the crash.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: large matrix products were very slow with this machine's default
import contextlib                                      # contextlib: redirect_stdout, to keep the imported parts quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import this week's parts
import time                                            # time: how long the rolling forecasts take
import warnings                                        # warnings: statsmodels' notices are not needed here

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import Rectangle               # month cells, for the diagram
import pandas as pd                                    # pandas: only to check our features
from statsmodels.tsa.ar_model import AutoReg           # statsmodels' autoregression: only to check our least squares
warnings.filterwarnings("ignore")                       # keep the output to this lesson's own lines

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts
with contextlib.redirect_stdout(io.StringIO()):         # the exercises print their own checks when imported; keep them quiet
    import day147_exercise as d147                      # Day 147: load_co2, load_elec, centred_ma
    import day149_exercise as d149                      # Day 149: the ARIMA parts, for the comparison
M = 12                                                  # months in a season
SPECS = {                                               # the candidate feature sets: lags, rolling statistics, trend, calendar
    "lags 1-13": dict(lags=tuple(range(1, 14))),
    "lags 1, 12, 13": dict(lags=(1, 12, 13)),
    "trend + months": dict(trend=True, calendar="months"),
    "lags 1, 12, 13 + trend + months": dict(lags=(1, 12, 13), trend=True, calendar="months"),
    "lags 1, 12, 13 + trend + 2 waves": dict(lags=(1, 12, 13), trend=True, calendar=2),
    "lags 1, 12, 13 + trend + months + 12-month mean, sd": dict(lags=(1, 12, 13), trend=True, calendar="months",
                                                               rolling=((12, "mean"), (12, "sd"))),
}


# =============================================================================
# THE PARTS -- NEW (copied unchanged into day151_exercise.py)
# =============================================================================
def lag(y, k):                                          # NEW: the value k months earlier; unknown (NaN) for the first k months
    out = np.full(len(y), np.nan)                                                     # unknown until filled
    out[k:] = y[:len(y) - k]                                                          # month t gets month t - k's value
    return out


def rolling(y, w, stat, shift=1):                       # NEW: a statistic of the w months ending `shift` months before t
    out = np.full(len(y), np.nan)                                                     # unknown until a full window exists
    win = np.lib.stride_tricks.sliding_window_view(y, w)                              # row i holds y[i], ..., y[i + w - 1]
    first = w - 1 + shift                                                             # the first month with a full window behind it
    out[first:] = stat(win, axis=1)[:len(y) - first]                                  # month t uses the window ending at t - shift
    return out


def month_dummies(months):                              # NEW: 11 columns, February to December; January is the baseline
    mo = months.astype("datetime64[M]").astype(int) % M                               # 0 = January, ..., 11 = December
    return (mo[:, None] == np.arange(1, M)).astype(float)                             # 1 in the column of that month, else 0


def fourier(months, K):                                 # NEW: the season as K pairs of sine and cosine waves (periods 12, 6, ...)
    mo = months.astype("datetime64[M]").astype(int) % M                               # 0 = January, ..., 11 = December
    k = np.arange(1, K + 1)                                                           # wave 1 repeats once a year, wave 2 twice, ...
    return np.column_stack([np.sin(2 * np.pi * mo[:, None] * k / M), np.cos(2 * np.pi * mo[:, None] * k / M)])


def make_features(y, start, spec, h=1):                 # NEW: the design matrix; each feature uses only months at least h before the target
    months = np.datetime64(start, "M") + np.arange(len(y))                            # the calendar month of every row
    cols, names = [np.ones(len(y))], ["intercept"]                                    # the constant first
    for k in spec.get("lags", ()):                                                    # lag k for a one-step model is lag k + h - 1 for h steps
        cols.append(lag(y, k + h - 1)); names.append(f"lag {k + h - 1}")
    for w, stat in spec.get("rolling", ()):                                           # a window that ends h months before the target
        cols.append(rolling(y, w, np.mean if stat == "mean" else np.std, shift=h)); names.append(f"{stat} of {w}, ending {h} back")
    if spec.get("trend"):                                                             # time in decades since the first month
        cols.append(np.arange(len(y)) / 120.0); names.append("trend")
    cal = spec.get("calendar")                                                        # "months", a number of waves, or nothing
    if cal == "months":
        D = month_dummies(months); cols += list(D.T); names += [f"month {j + 1}" for j in range(1, M)]
    elif cal:
        D = fourier(months, cal); cols += list(D.T); names += [f"wave {j}" for j in range(D.shape[1])]
    return np.column_stack(cols), names


def fit_ols(X, y):                                      # NEW: least squares on the rows where the target and every feature are known
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)                                      # rows with nothing missing
    return np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]                               # the coefficients


def recursive_forecast(y, start, spec, b, h):           # NEW: one month at a time; each forecast is fed back in as if it were data
    yy = np.r_[y, np.full(h, np.nan)]                                                 # room for the h forecasts
    for i in range(len(y), len(y) + h):                                               # the next unknown month
        X, _ = make_features(yy[:i + 1], start, spec)                                 # its features, from data and earlier forecasts
        yy[i] = X[-1] @ b                                                             # the model's value becomes that month's 'data'
    return yy[len(y):]


def direct_forecast(y, start, spec, h):                 # NEW: one model per horizon j, using only months at least j before its target
    out = np.empty(h)                                                                 # the h forecasts
    for j in range(1, h + 1):                                                         # horizon j: its own features and coefficients
        X, _ = make_features(y, start, spec, h=j); b = fit_ols(X, y)                  # trained on targets j months after their features
        Xf, _ = make_features(np.r_[y, np.full(j, np.nan)], start, spec, h=j)         # the row for the month j after the last
        out[j - 1] = Xf[-1] @ b
    return out


def ols_fit_predict(Xtr, ytr, Xte):                     # NEW: least squares, as a fit-and-predict function
    return Xte @ np.linalg.lstsq(Xtr, ytr, rcond=None)[0]


def knn_fit_predict(Xtr, ytr, Xte, k=5):                # NEW: the mean of the k training rows nearest in (scaled) features
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12                               # scale with the training rows only
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd                                           # every feature on the same scale
    d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(axis=2)                            # squared distance, test row by training row
    return ytr[np.argsort(d, axis=1)[:, :k]].mean(axis=1)                             # average the k nearest targets


def cv_random(X, y, fit_predict, folds=5, seed=0):      # NEW: K-fold with the months shuffled: time order ignored
    idx = np.random.RandomState(seed).permutation(len(y)); err = []                   # a random order of the rows
    for f in range(folds):                                                            # every fifth shuffled row is held out
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        err.append(np.abs(fit_predict(X[tr], y[tr], X[te]) - y[te]))
    return np.mean(np.concatenate(err))


def cv_time(X, y, fit_predict, folds=5):                # NEW: time-ordered: train on everything before a block, test on the block
    edges = np.linspace(len(y) // 2, len(y), folds + 1).astype(int); err = []          # blocks cover the second half of the rows
    for a, b in zip(edges[:-1], edges[1:]):                                           # each block is tested by a model of the past only
        err.append(np.abs(fit_predict(X[:a], y[:a], X[a:b]) - y[a:b]))
    return np.mean(np.concatenate(err))


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def split(tc):                                          # Day 147's split: train up to 1999-12, forecast the last 24 months
    return int(np.flatnonzero(tc == np.datetime64("2000-01"))[0])


def airline(y, h):                                      # Day 150's helper, unchanged: Day 149's airline model (CSS), forecast and sd
    w = np.diff(y[12:] - y[:-12]); th, s2, _ = d149.fit_css(w, (0, 1, 0, 1, 12)); a, b = d149.sarima_polys(th, (0, 1, 0, 1, 12))
    A = d149.full_ar(a); return d149.forecast(y, A, b, d149.css_residuals(w, a, b), h), np.sqrt(s2 * np.cumsum(d149.psi_weights(A, b, h) ** 2))


def one_step_rows(X, y, n):                             # the training rows with nothing missing, and the 24 test rows
    ok = ~np.isnan(X).any(axis=1); idx = np.arange(len(y))
    return ok & (idx < n), (idx >= n) & ok


def demo_build(tc, yc):
    print("\n" + "=" * 70)
    print("1. BUILDING THE FEATURES -- checked against pandas and statsmodels")
    print("=" * 70)
    s = pd.Series(yc); n = split(tc)                                                  # pandas' version of the same series
    gap = lambda a, b: np.nanmax(np.abs(a - b)) if np.array_equal(np.isnan(a), np.isnan(b)) else np.inf  # inf if the gaps differ
    print(f"lag 12 against shift(12): largest difference {gap(lag(yc, 12), s.shift(12).to_numpy()):.1e}")
    print(f"mean of the 12 months before against rolling(12).mean().shift(1): {gap(rolling(yc, 12, np.mean), s.rolling(12).mean().shift(1).to_numpy()):.1e}")
    print(f"sd of the 12 months before against rolling(12).std(ddof=0).shift(1): "
          f"{gap(rolling(yc, 12, np.std), s.rolling(12).std(ddof=0).shift(1).to_numpy()):.1e}")
    ref = pd.get_dummies(pd.DatetimeIndex(tc.astype("datetime64[D]")).month).to_numpy(float)[:, 1:]  # January's column dropped
    print(f"month columns against get_dummies: {np.abs(month_dummies(tc) - ref).max():.1e} (11 columns; January is all zeros)")
    X, names = make_features(yc[:n], tc[0], SPECS["lags 1-13"]); b = fit_ols(X, yc[:n])  # lags 1-13 and a constant
    ar = AutoReg(yc[:n], 13, trend="c").fit()                                          # statsmodels' least-squares autoregression
    print(f"least squares on lags 1-13 against AutoReg(13): largest coefficient difference {np.abs(b - ar.params).max():.1e}")
    print(f"the first rows: {int(np.isnan(X).any(axis=1).sum())} months have a missing lag and are left out of the fit "
          f"({len(names)} columns: {names[0]}, {names[1]}, ..., {names[-1]})")
    print("Actual output from this lesson's run.")


def demo_leaks(tc, yc):
    print("\n" + "=" * 70)
    print("2. FOUR LEAKS -- each looks like better accuracy; none is available when forecasting")
    print("=" * 70)
    n = split(tc); spec = dict(lags=(1, 2, 12, 13)); X, _ = make_features(yc, tc[0], spec)  # an honest base: lags 1, 2, 12, 13
    rows = []                                                                         # (name, train MAE, test MAE, number of test months)

    def score(name, Xf, keep=None):                                                   # fit on training rows, score one step ahead
        tr, te = one_step_rows(Xf, yc, n)
        if keep is not None: te = te & keep                                           # only test months where the feature exists
        b = fit_ols(Xf[tr], yc[tr]); rows.append((name, np.mean(np.abs(Xf[tr] @ b - yc[tr])), np.mean(np.abs(Xf[te] @ b - yc[te])), int(te.sum())))
        return b
    score("honest: lags 1, 2, 12, 13", X)
    b = score("A: + 3-month mean that includes this month", np.c_[X, rolling(yc, 3, np.mean, shift=0)])
    T = d147.centred_ma(yc); has = ~np.isnan(T)                                        # Day 147's centred trend: 6 months either side
    score("honest: lags, on the 18 months B can reach", X, has)
    score("B: lags + centred 12-month trend (Day 147)", np.c_[X, T], has)
    C, _ = make_features(yc, tc[0], dict(calendar="months"))                          # a constant and the 11 month columns
    score("honest: straight trend line + months", np.c_[C, np.arange(len(yc)) / 120.0], has)
    score("B: centred 12-month trend + months", np.c_[C, T], has)
    print(f"{'features':>45} {'train MAE':>10} {'test MAE':>9} {'months':>7}")
    for name, a, c, m in rows:
        print(f"{name:>45} {a:10.3f} {c:9.3f} {m:7d}")
    print(f"B reads 6 months past the target, so it does not exist for the last 6 test months; added to the lags it changes little "
          f"({rows[2][2]:.3f} to {rows[3][2]:.3f}), but in place of a trend line it looks {rows[4][2] / rows[5][2]:.0f} times better")
    print(f"A's coefficients on lag 1, lag 2 and the 3-month mean: {b[1]:.3f}, {b[2]:.3f}, {b[5]:.3f} -- "
          f"3 x mean - lag 1 - lag 2 is this month itself")
    tr, te = one_step_rows(X, yc, n); b = fit_ols(X[tr], yc[tr])                       # C: the honest model, scored two ways
    one = np.mean(np.abs(X[te] @ b - yc[te])); rec = np.mean(np.abs(recursive_forecast(yc[:n], tc[0], spec, b, 24) - yc[n:]))
    print(f"C: the honest model's 2000-2001 error one step ahead (true lags every month) {one:.3f}; forecast from 1999-12 "
          f"for 24 months {rec:.3f} -- a 24-month forecast must not be scored with lags it could not have known")
    mu_all, sd_all = X[~np.isnan(X).any(axis=1)][:, 1:].mean(0), X[~np.isnan(X).any(axis=1)][:, 1:].std(0)  # D: scaled with every month
    mu_tr, sd_tr = X[tr][:, 1:].mean(0), X[tr][:, 1:].std(0)                           # scaled with the training months only
    preds = {}
    for name, mu, sd in (("all", mu_all, sd_all), ("train", mu_tr, sd_tr)):
        Z = np.c_[np.ones(len(yc)), (X[:, 1:] - mu) / sd]; preds[name] = Z[te] @ fit_ols(Z[tr], yc[tr])
    print(f"D: scaling the features with all months (test included) instead of the training months: least-squares forecasts "
          f"change by at most {np.abs(preds['all'] - preds['train']).max():.1e}")
    print(f"   (least squares with a constant gives the same fit for any shift and scaling of the features, so this leak cannot matter here)")
    print("Actual output from this lesson's run.")
    return rows


def demo_cv(tc, yc):
    print("\n" + "=" * 70)
    print("3. CROSS-VALIDATION -- shuffled folds against time-ordered folds, on the training years only")
    print("=" * 70)
    n = split(tc); d = np.r_[np.nan, np.diff(yc)]; out = {}                           # levels, and month-to-month changes
    knn = lambda a, b, c: knn_fit_predict(a, b, c, 5)                                 # 5 nearest neighbours
    print(f"{'target, lags, model':>32} {'shuffled CV':>12} {'time CV':>8} {'test (2000-01)':>15}")
    for target, z, ps in (("CO2 level", yc, (3, 12, 24)), ("monthly change", d, (12,))):
        for p in ps:
            X, _ = make_features(z, tc[0], dict(lags=tuple(range(1, p + 1))))         # lags 1 to p
            tr, te = one_step_rows(X, z, n); tr &= ~np.isnan(z)                       # the first change is unknown
            for mname, fp in (("least squares", ols_fit_predict), ("5-NN", knn)):
                r, t = cv_random(X[tr], z[tr], fp), cv_time(X[tr], z[tr], fp)
                h = np.mean(np.abs(fp(X[tr], z[tr], X[te]) - z[te])); out[(target, p, mname)] = (r, t, h)
                print(f"{target + ', ' + str(p) + ', ' + mname:>32} {r:12.3f} {t:8.3f} {h:15.3f}")
    r, t, h = out[("CO2 level", 12, "5-NN")]
    print(f"5-NN on levels: shuffled CV promises {r:.3f}, the test gives {h:.3f} ({h / r:.1f} times worse); time-ordered CV warned ({t:.3f})")
    X, _ = make_features(yc, tc[0], dict(lags=tuple(range(1, 13)))); tr, _ = one_step_rows(X, yc, n)  # levels, 12 lags, again
    Z, pos = X[tr], np.flatnonzero(tr); idx = np.random.RandomState(0).permutation(len(Z)); gaps = []  # the same shuffle as cv_random
    for f in range(5):                                                                # for every held-out month: how far away in time
        te_ = idx[f::5]; tr_ = np.setdiff1d(idx, te_); mu, sd = Z[tr_].mean(0), Z[tr_].std(0) + 1e-12   # is its nearest training month?
        dist = ((((Z[te_] - mu) / sd)[:, None, :] - ((Z[tr_] - mu) / sd)[None, :, :]) ** 2).sum(axis=2)
        gaps.append(np.abs(pos[tr_][np.argmin(dist, axis=1)] - pos[te_]))
    g = np.concatenate(gaps)
    print(f"why: in the shuffled folds, a held-out month's nearest training month (in features) is a median {np.median(g):.0f} month(s) "
          f"away in time, and within 2 months for {100 * np.mean(g <= 2):.0f}% of them")
    blk = (int(tr.sum()) - int(tr.sum()) // 2) // 5                                    # rows per time-ordered block (last target used)
    print(f"each time-ordered block is about {blk} months long, against the test's 24, so it reaches further past its training rows:"
          " a likely reason it is more pessimistic than the test (not tested here)")
    print("for least squares, and for 5-NN on the monthly changes, the shuffled estimate stays close to the test")
    print("Actual output from this lesson's run.")
    return out


def validate(y, start, spec, origins, h):               # mean error of h-month recursive forecasts from each origin
    err = []
    for o in origins:
        X, _ = make_features(y[:o], start, spec); b = fit_ols(X, y[:o])               # fitted on the months before the origin
        err.append(np.abs(recursive_forecast(y[:o], start, spec, b, h) - y[o:o + h]))
    return np.mean(err)


def demo_choose(tc, yc):
    print("\n" + "=" * 70)
    print("4. CHOOSING THE FEATURES FOR CO2 -- time-ordered validation before 1990, then the test")
    print("=" * 70)
    n = split(tc); lo, hi = (int(np.flatnonzero(tc == np.datetime64(m))[0]) for m in ("1975-01", "1988-01"))
    origins = range(lo, hi + 1)                                                       # 157 origins; every forecast ends by 1989-12
    print(f"validation: {len(origins)} origins, {tc[lo]} to {tc[hi]}, each forecasting 24 months (forecasts end by {tc[hi + 23]})")
    print(f"{'features':>52} {'validation':>11} {'test 2000-01':>13}")
    val, test = {}, {}
    for name, spec in SPECS.items():
        val[name] = validate(yc, tc[0], spec, origins, 24)                            # the choice is made on this column
        X, _ = make_features(yc[:n], tc[0], spec); test[name] = recursive_forecast(yc[:n], tc[0], spec, fit_ols(X, yc[:n]), 24)
        print(f"{name:>52} {val[name]:11.3f} {np.mean(np.abs(test[name] - yc[n:])):13.3f}")
    best = min(val, key=val.get)                                                      # lowest validation error
    print(f"validation chooses: {best}")
    direct = direct_forecast(yc[:n], tc[0], SPECS[best], 24); ar = airline(yc[:n], 24)[0]  # the same features, one model per horizon
    print(f"2000-01 to 2001-12, MAE (ppm): {best}, recursive {np.mean(np.abs(test[best] - yc[n:])):.3f}, direct "
          f"{np.mean(np.abs(direct - yc[n:])):.3f}; airline ARIMA (Day 149) {np.mean(np.abs(ar - yc[n:])):.3f}")
    print("Actual output from this lesson's run.")
    return best, test, direct, ar, val


def demo_rolling(tc, yc, best, val):
    print("\n" + "=" * 70)
    print("5. ROLLING ORIGINS -- 97 forecasts, 24 months ahead, 1990-1998, against ARIMA")
    print("=" * 70)
    n = split(tc); first = int(np.flatnonzero(tc == np.datetime64("1990-01"))[0]); spec = SPECS[best]
    E = {"regression, recursive": [], "regression, direct": [], "airline ARIMA, CSS": [], "same month last year": []}
    t0 = time.perf_counter()
    for o in range(first, n - 24 + 1):                                                # each origin: fit on everything before it
        act = yc[o:o + 24]; X, _ = make_features(yc[:o], tc[0], spec)
        E["regression, recursive"].append(recursive_forecast(yc[:o], tc[0], spec, fit_ols(X, yc[:o]), 24) - act)
        E["regression, direct"].append(direct_forecast(yc[:o], tc[0], spec, 24) - act)
        E["airline ARIMA, CSS"].append(airline(yc[:o], 24)[0] - act); E["same month last year"].append(yc[o - 12:o][np.arange(24) % 12] - act)
    E = {k: np.abs(np.array(v)) for k, v in E.items()}
    print(f"{len(E['airline ARIMA, CSS'])} origins ({time.perf_counter() - t0:.1f} s); features: {best}")
    hs = (1, 6, 12, 24)
    print(f"{'MAE (ppm), months ahead':>28} " + " ".join(f"{k:>7}" for k in hs))
    for k, v in E.items():
        print(f"{k:>28} " + " ".join(f"{v[:, j - 1].mean():7.3f}" for j in hs))
    rec, dr, ar = (E[k].mean(axis=0) for k in ("regression, recursive", "regression, direct", "airline ARIMA, CSS"))
    print(f"recursive regression beats the airline model at {np.sum(rec < ar)} of 24 horizons; direct beats recursive at {np.sum(dr < rec)} of 24")
    later = {name: validate(yc, tc[0], sp, range(first, n - 24 + 1), 24) for name, sp in SPECS.items()}  # every candidate, same origins
    print("did the validation ranking (1975-1988 origins) hold in 1990-1998? mean error over 24 months, recursive:")
    for name in sorted(val, key=val.get):                                             # in validation order
        print(f"  {name:>52}  validation {val[name]:.3f}  1990-1998 {later[name]:.3f}")
    order = sorted(later, key=later.get)                                              # the ranking in the later period
    print(f"validation's choice ranked {order.index(best) + 1} of {len(order)} in 1990-1998; the best there: {order[0]}")
    print("Actual output from this lesson's run.")
    return E


def demo_orders(te, ye):
    print("\n" + "=" * 70)
    print("6. ELECTRICAL-EQUIPMENT ORDERS -- log scale; features chosen before 2001")
    print("=" * 70)
    le = np.log(ye); idx = lambda m: int(np.flatnonzero(te == np.datetime64(m))[0])  # position of a month
    origins = range(idx("1998-01"), idx("2000-01") + 1)                              # validation: forecasts end by 2000-12
    val = {name: validate(le, te[0], spec, origins, 12) for name, spec in SPECS.items()}
    best = min(val, key=val.get)
    print(f"validation ({len(origins)} origins, 12 months ahead, log scale) chooses: {best} ({val[best]:.4f}; "
          + ", ".join(f"{k} {v:.4f}" for k, v in val.items() if k != best) + ")")
    spec = SPECS[best]; first, last = idx("2001-01"), idx("2007-01")
    E = {"regression, recursive": [], "regression, direct": [], "airline ARIMA, CSS": [], "same month last year": []}
    for o in range(first, last + 1):                                                  # origins 2001-01 to 2007-01, 12 months ahead
        act = le[o:o + 12]; pe = lambda f: np.abs(np.exp(f - act) - 1)              # absolute percentage error
        X, _ = make_features(le[:o], te[0], spec)
        E["regression, recursive"].append(pe(recursive_forecast(le[:o], te[0], spec, fit_ols(X, le[:o]), 12)))
        E["regression, direct"].append(pe(direct_forecast(le[:o], te[0], spec, 12)))
        E["airline ARIMA, CSS"].append(pe(airline(le[:o], 12)[0])); E["same month last year"].append(pe(le[o - 12:o]))
    later = {}                                                                        # every candidate on the same origins, for comparison
    for name, sp in SPECS.items():
        later[name] = np.mean([np.abs(np.exp(recursive_forecast(le[:o], te[0], sp, fit_ols(make_features(le[:o], te[0], sp)[0], le[:o]), 12)
                                             - le[o:o + 12]) - 1) for o in range(first, last + 1)])
    print(f"rolling origins {te[first]} to {te[last]} ({last - first + 1}), mean absolute % error by months ahead:")
    print(f"{'':>24} {'1':>7} {'6':>7} {'12':>7}")
    for k, v in E.items():
        v = np.array(v); print(f"{k:>24} " + " ".join(f"{100 * v[:, j - 1].mean():6.2f}%" for j in (1, 6, 12)))
    print("did the validation ranking (1998-1999 origins) hold in 2001-2007? mean error over 12 months, recursive:")
    for name in sorted(val, key=val.get):
        print(f"  {name:>52}  validation {100 * val[name]:.2f} (log error x 100)  2001-2007 {100 * later[name]:.2f}%")
    order = sorted(later, key=later.get)                                              # the ranking in the later period
    top3 = sorted(val, key=val.get)[:3]                                               # validation's top three
    print(f"validation's choice ranked {order.index(best) + 1} of {len(order)} in 2001-2007; the best there: {order[0]}; "
          f"validation's top three (all with a trend line: {all(SPECS[k].get('trend') for k in top3)}) "
          f"ranked {', '.join(str(order.index(k) + 1) for k in top3)}")
    cut = idx("2008-01"); act = le[cut:cut + 24]; X, _ = make_features(le[:cut], te[0], spec)  # the crash: train up to 2007-12
    f = recursive_forecast(le[:cut], te[0], spec, fit_ols(X, le[:cut]), 24); ar = airline(le[:cut], 24)[0]; sn = le[cut - 12:cut][np.arange(24) % 12]
    mape = lambda g: 100 * np.mean(np.abs(np.exp(g - act) - 1))
    print(f"the crash (train up to {te[cut - 1]}, 24 months): regression {mape(f):.2f}%, airline ARIMA {mape(ar):.2f}%, same month last year {mape(sn):.2f}%")
    print("Actual output from this lesson's run.")
    return E, (cut, f, ar)


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(tc, yc, te, ye, leaks, cv, chosen, roll, orders, out):
    fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0]); x = np.arange(len(leaks))                          # the leaks: train and test error
    ax.bar(x - 0.2, [r[1] for r in leaks], 0.4, color="#4C72B0", label="training months")
    ax.bar(x + 0.2, [r[2] for r in leaks], 0.4, color="#DD8452", label="test months")
    ax.set_xticks(x); ax.set_xticklabels(["lags", "A: + mean\nincl. t", "lags\n(18 mo.)", "B: lags +\ncentred", "line +\nmonths", "B: centred\n+ months"], fontsize=7.5)
    ax.text(1, 0.06, "0.000", ha="center", fontsize=8)                                # A's error is too small to see as a bar
    ax.set_ylabel("one-step MAE (ppm)"); ax.legend(fontsize=8); ax.set_title("leaks A and B (B in place of a trend line) look far better", fontsize=10)
    ax = fig.add_subplot(gs[0, 1]); keys = [("CO2 level", 12, "least squares"), ("CO2 level", 12, "5-NN"), ("monthly change", 12, "5-NN")]
    x = np.arange(len(keys))
    for j, (lab, c) in enumerate((("shuffled CV", "#C44E52"), ("time-ordered CV", "#55A868"), ("test", "#333333"))):
        ax.bar(x + (j - 1) * 0.27, [cv[k][j] for k in keys], 0.27, color=c, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["levels,\nleast squares", "levels,\n5-NN", "changes,\n5-NN"], fontsize=8)
    ax.set_ylabel("one-step MAE (ppm)"); ax.legend(fontsize=8); ax.set_title("shuffled CV flatters 5-NN on levels (12 lags)", fontsize=10)
    best, test, direct, ar, _ = chosen; n = split(tc); ax = fig.add_subplot(gs[0, 2]); x = tc.astype("datetime64[D]"); s = n - 36
    ax.plot(x[s:], yc[s:], color="k", lw=1.2, label="actual")
    ax.plot(x[n:], test[best], color="#4C72B0", lw=1.2, label="regression, recursive")
    ax.plot(x[n:], direct, color="#55A868", lw=1.2, label="regression, direct")
    ax.plot(x[n:], ar, color="#8172B2", lw=1.2, label="airline ARIMA (Day 149)")
    ax.axvline(x[n], color="gray", ls=":"); ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("2000-2001 forecasts from data up to 1999-12", fontsize=10)
    ax = fig.add_subplot(gs[1, 0]); hz = np.arange(1, 25)
    for (name, v), c in zip(roll.items(), ("#4C72B0", "#55A868", "#8172B2", "#DD8452")):
        ax.plot(hz, v.mean(axis=0), marker="o", ms=3, color=c, label=name)
    ax.set_xlabel("months ahead"); ax.set_ylabel("MAE (ppm)"); ax.legend(fontsize=7); ax.set_title("CO2, 97 rolling origins: error by horizon", fontsize=10)
    E, (cut, f, arc) = orders; ax = fig.add_subplot(gs[1, 1]); hz = np.arange(1, 13)
    for (name, v), c in zip(E.items(), ("#4C72B0", "#55A868", "#8172B2", "#DD8452")):
        ax.plot(hz, 100 * np.array(v).mean(axis=0), marker="o", ms=3, color=c, label=name)
    ax.set_xlabel("months ahead"); ax.set_ylabel("mean absolute % error"); ax.legend(fontsize=7)
    ax.set_title("orders, 73 rolling origins 2001-2007", fontsize=10)
    ax = fig.add_subplot(gs[1, 2]); x = te.astype("datetime64[D]"); s = cut - 48
    ax.plot(x[s:cut + 30], ye[s:cut + 30], color="k", lw=1.2, label="actual orders")
    ax.plot(x[cut:cut + 24], np.exp(f), color="#4C72B0", label="regression from 2007-12")
    ax.plot(x[cut:cut + 24], np.exp(arc), color="#8172B2", label="airline ARIMA from 2007-12")
    ax.axvline(x[cut], color="gray", ls=":"); ax.set_ylabel("orders (index)"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("the 2008-09 crash", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # which months each feature reads, for a target month t
    fig, ax = plt.subplots(figsize=(15, 5.6)); ax.set_xlim(-8.5, 23.2); ax.set_ylim(0.7, 7.75); ax.axis("off")
    X0 = 15                                                                           # x position of the target month's cell

    def cell(k, y0, color, lw=0.4):                                                   # one month's cell, k months from the target
        ax.add_patch(Rectangle((X0 + k, y0), 0.9, 0.55, fc=color, ec="black", lw=lw))
    for k in range(-14, 8):                                                           # the month row along the top
        cell(k, 6.6, "#FFF2CC" if k == 0 else ("#EEEEEE" if k > 0 else "white"), 0.6)
        ax.text(X0 + k + 0.45, 6.87, "t" if k == 0 else f"{k:+d}", ha="center", va="center", fontsize=7.5)
    ax.text(0.6, 6.87, "month (t = the target)", ha="right", va="center", fontsize=9)
    ax.plot([X0 - 0.05, X0 - 0.05], [1.3, 7.3], color="#C44E52", lw=1.5, ls="--")     # the line between the past and the target
    ax.text(X0 - 0.05, 7.4, "known  |  not known when forecasting", ha="center", fontsize=8.5, color="#C44E52")

    def span(row, ks, label, color):                                                  # one feature: the months it reads
        y0 = 5.75 - row * 0.85
        for k in ks:
            cell(k, y0, color)
        ax.text(0.6, y0 + 0.27, label, ha="right", va="center", fontsize=8.5)
    span(0, (-1,), "lag 1", "#C6E0B4"); span(1, (-13, -12), "lags 12 and 13", "#C6E0B4")
    span(2, range(-12, 0), "mean of the 12 months before", "#C6E0B4")
    span(3, range(-2, 1), "A: 3-month mean including t (leak)", "#F4B183")
    span(4, range(-6, 7), "B: centred 12-month trend (leak)", "#F4B183")
    span(5, range(-14, -2), "3 ahead, direct: mean ending 3 back", "#BDD7EE")
    ax.text(7.4, 0.95, "green: honest one step ahead;  orange: reads month t or later;  blue: a direct model for 3 months ahead "
            "may read only up to t - 3", ha="center", fontsize=8.5, style="italic")
    ax.set_title("Day 151: which months each feature reads", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "feature_engineering_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 152: recurrent networks for forecasting -- Week 10's LSTM, trained on these lag windows, against the")
    print("classical models of Days 149-151.")


def main():                                             # runs every section in order
    tc, yc, _ = d147.load_co2(); te, ye = d147.load_elec()                           # Day 147's two series
    demo_build(tc, yc)                                                                # 1. the features, checked
    leaks = demo_leaks(tc, yc)                                                        # 2. the leaks
    cv = demo_cv(tc, yc)                                                              # 3. cross-validation
    chosen = demo_choose(tc, yc)                                                      # 4. choosing the features
    roll = demo_rolling(tc, yc, chosen[0], chosen[4])                                            # 5. rolling origins
    orders = demo_orders(te, ye)                                                      # 6. the orders
    plot_results(tc, yc, te, ye, leaks, cv, chosen, roll, orders, os.path.join(HERE, "feature_engineering_results.png"))
    draw_diagram()                                                                    # which months each feature reads
    print("\nsaved feature_engineering_results.png\nsaved feature_engineering_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
