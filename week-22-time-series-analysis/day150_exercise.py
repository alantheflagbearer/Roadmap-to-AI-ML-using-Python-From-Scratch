"""
Day 150 Exercise -- Exponential Smoothing from Scratch

The fourth of Week 22's parts for Day 153's Prophet-style forecaster.

    ets_filter(y, alpha, beta, gamma, phi, l0, b0, s0, trend, season)   the smoothing equations, run through y
    ets_init(y, m, season)             starting level, trend and season from the first two years
    fit_ets(y, trend, season, damped)  the weights by least squares (Day 149's solver), from four starts
    ets_forecast(F, h)                 h steps ahead from the final states
    ets_sd(F, h)                       the forecast standard deviation (additive model)

Checked by hand: three steps of simple smoothing, a Holt line that is followed
exactly, the starting values of a tiny seasonal series, the fitted alpha
against a brute-force search, and the interval formula for simple smoothing.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported part quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: this file's folder
import sys                                             # sys: path, to import Day 149's part
import numpy as np                                     # numpy: every array operation below is built on this
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this week's parts
with contextlib.redirect_stdout(io.StringIO()):         # Day 149's exercise prints its own checks when imported
    import day149_exercise as d149                      # Day 149: levenberg_marquardt
M = 12                                                  # months in a season
STARTS = ((0.3, 0.05, 0.1, 0.5), (0.8, 0.5, 0.3, 0.5), (0.1, 0.01, 0.01, 0.5), (0.9, 0.9, 0.3, 0.5))  # (alpha, beta, gamma, phi) starting points


def ets_filter(y, alpha, beta, gamma, phi, l0, b0, s0, trend, season, m=M):  # NEW: Day 150's version, unchanged
    l, b, s = l0, b0, np.array(s0, float).copy()                                      # level, trend, and one seasonal value per month
    fit = np.empty(len(y)); hist = np.empty(len(y))                                   # one-step forecasts, and each month's updated seasonal value
    for t, yt in enumerate(y):                                                        # in time order
        st = s[t % m]                                                                 # this month's seasonal value, last updated a year ago
        base = l + phi * b if trend else l                                            # level plus the (damped) trend: the deseasonalised forecast
        if season == "mul":
            fit[t] = base * st; ln = alpha * yt / st + (1 - alpha) * base             # forecast; new level from the deseasonalised value
            s[t % m] = gamma * yt / base + (1 - gamma) * st                           # new seasonal factor
        elif season == "add":
            fit[t] = base + st; ln = alpha * (yt - st) + (1 - alpha) * base           # forecast; new level, part of the way towards y - season
            s[t % m] = gamma * (yt - base) + (1 - gamma) * st                         # new seasonal value, part of the way towards y - level
        else:
            fit[t] = base; ln = alpha * yt + (1 - alpha) * base                       # no season
        if trend:
            b = beta * (ln - l) + (1 - beta) * phi * b                                # new trend, part of the way towards the level's change
        l = ln; hist[t] = s[t % m]                                                    # move on; record the season for the plot
    return fit, l, b, s, hist


def ets_init(y, m=M, season="add"):                     # NEW: Day 150's version, unchanged
    l0 = y[:m].mean(); b0 = (y[m:2 * m].mean() - l0) / m                              # first year's mean; the change between years, per month
    return l0, b0, (y[:m] - l0 if season == "add" else y[:m] / l0)                    # each first-year month against that mean


def fit_ets(y, trend, season, damped, m=M, starts=STARTS):  # NEW: Day 150's version, unchanged
    l0, b0, s0 = ets_init(y, m, season or "add")                                      # starting states, fixed (not fitted)
    use = [True, trend is not None, season is not None, damped]                      # which of alpha, beta, gamma, phi are free

    def unpack(u):                                                                    # unconstrained numbers -> weights between 0 and 1
        p = iter(1 / (1 + np.exp(-u)))                                                # the logistic function keeps each in (0, 1)
        a, b, g, f = (next(p) if k else d for k, d in zip(use, (0.0, 0.0, 0.0, 1.0)))  # take one for each free weight, else its default
        return a, b, g, (0.8 + 0.18 * f if damped else 1.0)                           # damping kept between 0.8 and 0.98

    def res(u):                                                                       # one-step errors for these weights
        return y - ets_filter(y, *unpack(u), l0, b0, s0, trend, season, m)[0]
    best = None
    for st in starts:                                                                 # several starting points: the fit can stop at a local minimum
        p0 = np.array([v for v, k in zip(st, use) if k]); u, ssr = d149.levenberg_marquardt(res, np.log(p0 / (1 - p0)))
        if best is None or ssr < best[1]:
            best = (u, ssr)                                                           # keep the smallest sum of squares
    a, b, g, f = unpack(best[0]); _, l, bb, s, hist = ets_filter(y, a, b, g, f, l0, b0, s0, trend, season, m)
    return dict(alpha=a, beta=b, gamma=g, phi=f, sse=best[1], n=len(y), k=sum(use), level=l, slope=bb, season=s, hist=hist,
                trend_kind=trend, season_kind=season, init=(l0, b0, s0))


def ets_forecast(F, h, m=M):                            # NEW: Day 150's version, unchanged
    steps = np.cumsum(F["phi"] ** np.arange(1, h + 1)) if F["trend_kind"] else np.zeros(h)  # phi + phi^2 + ...: the damped number of trend steps
    base = F["level"] + steps * F["slope"]                                            # level plus trend
    s = F["season"][(F["n"] + np.arange(h)) % m]                                      # each future month's latest seasonal value
    return base + s if F["season_kind"] == "add" else base * s if F["season_kind"] == "mul" else base


def ets_sd(F, h, m=M):                                  # NEW: Day 150's version, unchanged
    j = np.arange(1, h)                                                               # 1 ... h-1
    c = F["alpha"] * (1 + F["beta"] * j) + F["gamma"] * (j % m == 0)                  # how much one error moves the forecast j steps later
    return np.sqrt(F["sse"] / F["n"] * np.r_[1.0, 1 + np.cumsum(c ** 2)])             # sigma^2 (1 + c_1^2 + ... + c_(h-1)^2)


f, l, b, s, _ = ets_filter(np.array([10.0, 12, 11]), 0.5, 0, 0, 1, 10.0, 0, np.zeros(M), None, None)
print(f"SES, alpha 0.5, start 10, data 10, 12, 11: forecasts 10, 10, 0.5 x 12 + 0.5 x 10 = 11; final level 0.5 x 11 + 0.5 x 11 = 11; "
      f"ets_filter gives {f.tolist()}, level {l}")
F = dict(level=0.0, slope=0.0, phi=1.0, n=0, trend_kind="add", season_kind=None, season=np.zeros(M))
f, l, b, s, _ = ets_filter(np.array([1.0, 2, 3, 4]), 1, 1, 0, 1, 0.0, 1.0, np.zeros(M), "add", None)
F.update(level=l, slope=b, n=4)
print(f"Holt, alpha 1 and beta 1, on the line 1, 2, 3, 4 (start level 0, trend 1): forecasts {f.tolist()}, then {ets_forecast(F, 3).tolist()}")
print(f"ets_init on 11, 9, 13, 11 with a season of 2: level (11 + 9) / 2 = 10, trend (12 - 10) / 2 = 1, season 1 and -1; "
      f"gives {[float(v) for v in ets_init(np.array([11.0, 9, 13, 11]), 2)[:2]]}, {ets_init(np.array([11.0, 9, 13, 11]), 2)[2].tolist()}")
rng = np.random.RandomState(1); z = np.cumsum(rng.standard_normal(200)) + rng.standard_normal(200)
grid = np.linspace(0.001, 0.999, 999); l0 = z[:M].mean()
sse = [np.sum((z - ets_filter(z, a, 0, 0, 1, l0, 0, np.zeros(M), None, None)[0]) ** 2) for a in grid]
print(f"SES alpha on 200 simulated values: brute force over 999 values {grid[np.argmin(sse)]:.3f}, fit_ets {fit_ets(z, None, None, False)['alpha']:.3f}")
F = dict(alpha=0.5, beta=0.0, gamma=0.0, sse=100.0, n=100)
print(f"interval for SES, alpha 0.5, error variance 1: sd 3 steps ahead = sqrt(1 + 2 x 0.25) = {np.sqrt(1.5):.4f}; ets_sd gives {ets_sd(F, 3)[2]:.4f}")
