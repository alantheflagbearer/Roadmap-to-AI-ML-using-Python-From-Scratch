"""
Day 148 Exercise -- Differencing, and Testing for Stationarity

The second of Week 22's parts for Day 153's Prophet-style forecaster.

    acf(x, nlags)                      the autocorrelation function, lags 0 ... nlags
    difference(y, lag)                 y_t - y_(t - lag): lag 1 removes a trend, lag 12 a yearly season
    df_stats(Y, regression)            the Dickey-Fuller t statistic for every column of Y (no extra lags)
    df_table(n, regression, sims)      that statistic's distribution for random walks: a simulated table of critical values
    mc_pvalue(stat, table)             the share of the table at or below a statistic
    adf(y, regression, maxlag)         the augmented Dickey-Fuller statistic, lags chosen by AIC
    kpss(y, regression, nlags)         the KPSS statistic (null hypothesis: stationary)

Checked by hand: the autocorrelation of 1, 2, 3, 4, 5, two differences, the
Dickey-Fuller statistic against a plain least-squares fit, the ADF test with no
extra lags against it, a small simulated table, and KPSS on a stationary series
and a random walk.
"""
import numpy as np                                     # numpy: every array operation below is built on this
SIMS = 20000                                            # simulated random walks per table of critical values


def acf(x, nlags):                                      # NEW: Day 148's version, unchanged
    x = np.asarray(x, float) - np.mean(x); d = np.sum(x * x)                         # centred, and the sum of squares
    return np.array([1.0] + [np.sum(x[:-k] * x[k:]) / d for k in range(1, nlags + 1)])  # correlation of the series with itself k steps later


def difference(y, lag=1):                               # NEW: Day 148's version, unchanged
    return y[lag:] - y[:-lag]


def df_stats(Y, regression="c"):                        # NEW: Day 148's version, unchanged
    n = len(Y) - 1                                                                    # one row per difference
    D = np.diff(Y, axis=0); L = Y[:-1]                                                # the changes, and the level before each change
    det = np.ones((n, 1)) if regression == "c" else np.column_stack([np.ones(n), np.arange(1, n + 1)])  # a constant, or a constant and a trend
    Q, _ = np.linalg.qr(det)                                                          # an orthonormal basis for the deterministic part
    D = D - Q @ (Q.T @ D); L = L - Q @ (Q.T @ L)                                      # remove it from both (the regression's other terms)
    g = np.sum(L * D, axis=0) / np.sum(L * L, axis=0)                                 # gamma: how the change depends on the level
    s2 = np.sum((D - g * L) ** 2, axis=0) / (n - det.shape[1] - 1)                    # the residual variance
    return g / np.sqrt(s2 / np.sum(L * L, axis=0))                                    # gamma divided by its standard error


def df_table(n, regression="c", sims=SIMS, seed=0):     # NEW: Day 148's version, unchanged
    Y = np.cumsum(np.random.RandomState(seed).standard_normal((n, sims)), axis=0)    # sims random walks of n steps, one per column
    return np.sort(df_stats(Y, regression))                                           # sorted, so quantiles and p-values are look-ups


def mc_pvalue(stat, table):                             # NEW: Day 148's version, unchanged
    return np.searchsorted(table, stat, side="right") / len(table)


def adf(y, regression="c", maxlag=None):                # NEW: Day 148's version, unchanged
    y = np.asarray(y, float); n = len(y); k0 = 1 if regression == "c" else 2          # k0: how many deterministic terms
    if maxlag is None:
        maxlag = min(n // 2 - k0 - 1, int(np.ceil(12 * (n / 100) ** 0.25)))           # Schwert's rule, as statsmodels
    dy = np.diff(y)                                                                   # the changes

    def fit(p, start):                                                                # regress dy_t on y_t, p lagged changes, and the deterministic terms
        r = np.arange(start, len(dy))                                                 # the rows used
        det = [np.ones(len(r))] + ([np.arange(1, len(r) + 1)] if regression == "ct" else [])
        X = np.column_stack([y[r]] + [dy[r - i] for i in range(1, p + 1)] + det)      # level first, then lagged changes, then constant (and trend)
        b, ssr = np.linalg.lstsq(X, dy[r], rcond=None)[:2]; ssr = float(ssr[0]) if len(ssr) else float(np.sum((dy[r] - X @ b) ** 2))
        return b, ssr, X, len(r)
    aic = []
    for p in range(maxlag + 1):                                                       # every lag length, on the same rows so AIC is comparable
        _, ssr, X, m = fit(p, maxlag); aic.append((m * np.log(ssr / m) + 2 * X.shape[1], p))  # AIC up to a constant
    p = min(aic)[1]                                                                   # the smallest AIC (the shorter lag on a tie)
    b, ssr, X, m = fit(p, p)                                                          # refit with that lag, on every row it allows
    se = np.sqrt(ssr / (m - X.shape[1]) * np.linalg.inv(X.T @ X)[0, 0])               # the level coefficient's standard error
    return b[0] / se, p, m                                                            # statistic, lags used, rows used


def kpss(y, regression="c", nlags=None):                # NEW: Day 148's version, unchanged
    y = np.asarray(y, float); n = len(y)
    if nlags is None:
        nlags = min(int(np.ceil(12 * (n / 100) ** 0.25)), n - 1)                     # the same rule as statsmodels' 'legacy'
    if regression == "c":
        e = y - y.mean()                                                              # deviations from the mean
    else:
        X = np.column_stack([np.ones(n), np.arange(1, n + 1)]); e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]  # from a straight line
    s2 = (np.sum(e * e) + 2 * sum((1 - k / (nlags + 1)) * np.sum(e[k:] * e[:-k]) for k in range(1, nlags + 1))) / n  # long-run variance (Bartlett weights)
    return float(np.sum(np.cumsum(e) ** 2) / n ** 2 / s2)                             # the partial sums, relative to that variance


print(f"acf of 1, 2, 3, 4, 5: centred -2 -1 0 1 2, sum of squares 10; lag 1 = 4 / 10 = 0.4, lag 2 = -1 / 10 = -0.1; "
      f"acf() gives {acf([1, 2, 3, 4, 5], 2)[1]:.1f}, {acf([1, 2, 3, 4, 5], 2)[2]:.1f}")
print(f"difference of 1, 4, 9, 16: {difference(np.array([1, 4, 9, 16])).tolist()} (lag 1), {difference(np.array([1, 4, 9, 16]), 2).tolist()} (lag 2)")
rng = np.random.RandomState(0); y = np.cumsum(rng.standard_normal(200))                # one random walk
X = np.column_stack([y[:-1], np.ones(199)]); b = np.linalg.lstsq(X, np.diff(y), rcond=None)[0]  # the same regression, by plain least squares
r = np.diff(y) - X @ b; se = np.sqrt(r @ r / (199 - 2) * np.linalg.inv(X.T @ X)[0, 0])
print(f"Dickey-Fuller statistic of one random walk: plain least squares {b[0] / se:.6f}, df_stats {df_stats(y[:, None])[0]:.6f}, "
      f"adf with no extra lags {adf(y, maxlag=0)[0]:.6f}")
T = df_table(100, sims=5000)
print(f"simulated table, length 100, 5,000 walks: 5% point {np.quantile(T, 0.05):.3f} (MacKinnon's published value for 100: about -2.89)")
print(f"mc_pvalue(2.5, table 1 2 3 4) = {mc_pvalue(2.5, np.array([1.0, 2, 3, 4])):.2f} (two of four values are at or below 2.5)")
print(f"KPSS: white noise {kpss(rng.standard_normal(300)):.3f}, a random walk {kpss(np.cumsum(rng.standard_normal(300))):.3f} "
      f"(5% critical value 0.463: the walk is rejected as stationary)")
