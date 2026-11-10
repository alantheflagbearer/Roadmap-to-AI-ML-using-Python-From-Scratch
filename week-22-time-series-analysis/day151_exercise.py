"""
Day 151 Exercise -- Feature Engineering for Time Series

The fifth of Week 22's parts for Day 153's Prophet-style forecaster.

    lag(y, k)                          the value k months earlier
    rolling(y, w, stat, shift)         a statistic of the w months ending `shift` months back
    month_dummies(months)              11 calendar columns (January is the baseline)
    fourier(months, K)                 the season as K sine-cosine pairs
    make_features(y, start, spec, h)   the design matrix, every feature at least h months before its target
    fit_ols(X, y)                      least squares on the complete rows
    recursive_forecast(...)            one month at a time, feeding forecasts back in
    direct_forecast(...)               one model per horizon
    ols_fit_predict, knn_fit_predict   two fit-and-predict functions
    cv_random, cv_time                 shuffled and time-ordered cross-validation

Checked by hand: a lag, a rolling mean with and without the current month, the
month columns, one Fourier pair, the lag names for a 3-month direct model, a
straight line fitted exactly, and recursive and direct forecasts of a series
that halves every month.
"""
import numpy as np                                     # numpy: every array operation below is built on this
M = 12                                                  # months in a season


def lag(y, k):                                          # NEW: Day 151's version, unchanged
    out = np.full(len(y), np.nan)                                                     # unknown until filled
    out[k:] = y[:len(y) - k]                                                          # month t gets month t - k's value
    return out


def rolling(y, w, stat, shift=1):                       # NEW: Day 151's version, unchanged
    out = np.full(len(y), np.nan)                                                     # unknown until a full window exists
    win = np.lib.stride_tricks.sliding_window_view(y, w)                              # row i holds y[i], ..., y[i + w - 1]
    first = w - 1 + shift                                                             # the first month with a full window behind it
    out[first:] = stat(win, axis=1)[:len(y) - first]                                  # month t uses the window ending at t - shift
    return out


def month_dummies(months):                              # NEW: Day 151's version, unchanged
    mo = months.astype("datetime64[M]").astype(int) % M                               # 0 = January, ..., 11 = December
    return (mo[:, None] == np.arange(1, M)).astype(float)                             # 1 in the column of that month, else 0


def fourier(months, K):                                 # NEW: Day 151's version, unchanged
    mo = months.astype("datetime64[M]").astype(int) % M                               # 0 = January, ..., 11 = December
    k = np.arange(1, K + 1)                                                           # wave 1 repeats once a year, wave 2 twice, ...
    return np.column_stack([np.sin(2 * np.pi * mo[:, None] * k / M), np.cos(2 * np.pi * mo[:, None] * k / M)])


def make_features(y, start, spec, h=1):                 # NEW: Day 151's version, unchanged
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


def fit_ols(X, y):                                      # NEW: Day 151's version, unchanged
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)                                      # rows with nothing missing
    return np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]                               # the coefficients


def recursive_forecast(y, start, spec, b, h):           # NEW: Day 151's version, unchanged
    yy = np.r_[y, np.full(h, np.nan)]                                                 # room for the h forecasts
    for i in range(len(y), len(y) + h):                                               # the next unknown month
        X, _ = make_features(yy[:i + 1], start, spec)                                 # its features, from data and earlier forecasts
        yy[i] = X[-1] @ b                                                             # the model's value becomes that month's 'data'
    return yy[len(y):]


def direct_forecast(y, start, spec, h):                 # NEW: Day 151's version, unchanged
    out = np.empty(h)                                                                 # the h forecasts
    for j in range(1, h + 1):                                                         # horizon j: its own features and coefficients
        X, _ = make_features(y, start, spec, h=j); b = fit_ols(X, y)                  # trained on targets j months after their features
        Xf, _ = make_features(np.r_[y, np.full(j, np.nan)], start, spec, h=j)         # the row for the month j after the last
        out[j - 1] = Xf[-1] @ b
    return out


def ols_fit_predict(Xtr, ytr, Xte):                     # NEW: Day 151's version, unchanged
    return Xte @ np.linalg.lstsq(Xtr, ytr, rcond=None)[0]


def knn_fit_predict(Xtr, ytr, Xte, k=5):                # NEW: Day 151's version, unchanged
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12                               # scale with the training rows only
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd                                           # every feature on the same scale
    d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(axis=2)                            # squared distance, test row by training row
    return ytr[np.argsort(d, axis=1)[:, :k]].mean(axis=1)                             # average the k nearest targets


def cv_random(X, y, fit_predict, folds=5, seed=0):      # NEW: Day 151's version, unchanged
    idx = np.random.RandomState(seed).permutation(len(y)); err = []                   # a random order of the rows
    for f in range(folds):                                                            # every fifth shuffled row is held out
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        err.append(np.abs(fit_predict(X[tr], y[tr], X[te]) - y[te]))
    return np.mean(np.concatenate(err))


def cv_time(X, y, fit_predict, folds=5):                # NEW: Day 151's version, unchanged
    edges = np.linspace(len(y) // 2, len(y), folds + 1).astype(int); err = []          # blocks cover the second half of the rows
    for a, b in zip(edges[:-1], edges[1:]):                                           # each block is tested by a model of the past only
        err.append(np.abs(fit_predict(X[:a], y[:a], X[a:b]) - y[a:b]))
    return np.mean(np.concatenate(err))


print(f"lag 1 of 1, 2, 3, 4: nan, 1, 2, 3; lag gives {lag(np.array([1.0, 2, 3, 4]), 1).tolist()}")
y5 = np.array([1.0, 2, 3, 4, 5])
print(f"mean of the 2 months before, on 1..5: nan, nan, 1.5, 2.5, 3.5; rolling gives {rolling(y5, 2, np.mean).tolist()}")
print(f"the same window including this month (shift 0, a leak): nan, 1.5, 2.5, 3.5, 4.5; gives {rolling(y5, 2, np.mean, shift=0).tolist()}")
mo = np.array(["2001-01", "2001-02", "2001-12"], dtype="datetime64[M]")
D = month_dummies(mo)
print(f"month columns for January, February, December: all zeros, a 1 in column 1, a 1 in column 11; month_dummies gives "
      f"{int(D[0].sum())} ones, column {D[1].tolist().index(1.0) + 1}, column {D[2].tolist().index(1.0) + 1}")
print(f"one Fourier pair for April (month 3 from January): sin(2 pi 3/12) = 1, cos = 0; fourier gives {np.round(fourier(np.array(['2001-04'], dtype='datetime64[M]'), 1)[0], 12).tolist()}")
print(f"a 3-month direct model with lags 1 and 12 reads lags 3 and 14: {make_features(np.arange(20.0), '2001-01', dict(lags=(1, 12)), h=3)[1]}")
x = np.arange(6.0); X = np.c_[np.ones(6), x]
print(f"least squares on y = 2 + 3x, exactly: {np.round(fit_ols(X, 2 + 3 * x), 10).tolist()}; "
      f"cv_random {cv_random(X, 2 + 3 * x, ols_fit_predict, folds=3):.1e}, cv_time {cv_time(X, 2 + 3 * x, ols_fit_predict, folds=3):.1e}")
h = 64 * 0.5 ** np.arange(7)
print(f"a series that halves every month (64, ..., 1); with y = 0.5 x last month, the next three are 0.5, 0.25, 0.125: "
      f"recursive {recursive_forecast(h, '2001-01', dict(lags=(1,)), np.array([0.0, 0.5]), 3).tolist()}, "
      f"direct {np.round(direct_forecast(h, '2001-01', dict(lags=(1,)), 3), 10).tolist()}")
print(f"2 nearest neighbours of 1.2 among 0, 1, 2, 10 are 1 and 2: knn gives {knn_fit_predict(np.array([[0.0], [1], [2], [10]]), np.array([0.0, 1, 2, 10]), np.array([[1.2]]), 2).tolist()}")
