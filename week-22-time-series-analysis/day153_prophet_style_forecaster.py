"""
Day 153: Week 22 Build -- A Prophet-Style Forecaster from This Week's Parts

Prophet (Taylor and Letham, 2017) treats a series as a curve in time:

    y(t) = trend g(t) + season s(t) + error

The trend is a straight line whose slope may change at a set of candidate
months (changepoints); a penalty keeps most of those changes small. The
season is a sum of Fourier waves (Day 151's columns). Everything is linear in
its coefficients, so the whole model is one least-squares fit, and its
forecast is the same curve carried past the last month.

Prophet itself puts a Laplace prior on the slope changes and fits with Stan;
here the penalty is a squared one (ridge), so the fit stays a single call to
least squares. Its uncertainty interval follows Prophet's recipe: future
slope changes are simulated at the rate and size seen in the history, plus
noise.

Built from the week's parts, imported unchanged: Day 147's data and
decomposition, Day 148's autocorrelation and KPSS test, Day 149's airline
ARIMA and Ljung-Box test, Day 150's Holt-Winters, Day 151's Fourier columns
and regression, and Day 152's LSTM and control. Every setting is chosen on
time-ordered validation before the test months are touched.

  1. The parts, checked: known slopes recovered, and the fit against plain least squares.
  2. Choosing the number of waves and the penalty on validation.
  3. What the model says about CO2: changepoints, season, and what it leaves behind.
  4. Forecasting CO2 in 2000-2001: against every model of the week.
  5. Rolling origins and interval coverage.
  6. Electrical-equipment orders: the crash, the last two years, and where the slope changed.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: large matrix products were very slow with this machine's default
import contextlib                                      # contextlib: redirect_stdout, to keep the imported parts quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import the parts
import time                                            # time: how long the sections take

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
import matplotlib.dates as mdates                       # year ticks on date axes
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts
sys.path.insert(0, os.path.join(HERE, "..", "week-10-lstm-gru"))  # Week 10's LSTM, which Day 152's parts use
with contextlib.redirect_stdout(io.StringIO()):         # the exercises print their own checks when imported; keep them quiet
    import day147_exercise as d147                      # Day 147: load_co2, load_elec, classical_decompose, by_month
    import day148_exercise as d148                      # Day 148: acf, kpss
    import day149_exercise as d149                      # Day 149: the ARIMA parts, psi_weights, ljung_box
    import day150_exercise as d150                      # Day 150: fit_ets, ets_filter, ets_forecast
    import day151_exercise as d151                      # Day 151: fourier, make_features, fit_ols, recursive_forecast
    import day152_exercise as d152                      # Day 152: fit_lstm, lstm_forecast, seasonal_drift
AIRLINE = (0, 1, 0, 1, 12)                              # Day 149's airline model
KS = (2, 3, 4, 5)                                       # numbers of Fourier wave pairs tried
LAMS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)            # penalties on the slope changes tried
HS = (1, 6, 12, 24)                                     # horizons shown in the tables


# =============================================================================
# THE PARTS -- NEW (today is the week's build, so there is no exercise file)
# =============================================================================
def changepoints(n, count=25, frac=0.8):                # NEW: candidate months, evenly spread over the first 80% of the history
    return np.unique(np.linspace(0, int(frac * n), count + 1).round().astype(int)[1:])  # Prophet's defaults: 25, in the first 80%


def design(start, idx, cps, K):                         # NEW: the columns -- 1, t, one hinge per changepoint, Fourier waves
    t = idx / 12.0                                                                    # time in years since the first month
    H = np.maximum(t[:, None] - cps[None, :] / 12.0, 0.0)                             # hinge: 0 before the changepoint, then rising by 1 a year
    return np.column_stack([np.ones(len(idx)), t, H, d151.fourier(np.datetime64(start, "M") + idx, K)])


def fit_prophet(y, start, K=4, lam=100.0, count=25, frac=0.8):  # NEW: penalised least squares; only the slope changes are penalised
    n = len(y); cps = changepoints(n, count, frac) if count else np.array([], int)    # count 0: a straight line plus the season
    X = design(start, np.arange(n), cps, K)                                           # one row per month
    P = np.zeros((len(cps), X.shape[1])); P[:, 2:2 + len(cps)] = np.sqrt(lam) * np.eye(len(cps))  # one penalty row per slope change
    beta = np.linalg.lstsq(np.vstack([X, P]), np.r_[y, np.zeros(len(cps))], rcond=None)[0]  # minimise |y - X b|^2 + lam |delta|^2
    r = y - X @ beta                                                                  # what the curve leaves behind
    return dict(beta=beta, cps=cps, K=K, lam=lam, start=start, n=n, sigma=r.std(), resid=r, delta=beta[2:2 + len(cps)])


def components(F, idx):                                 # NEW: the trend and the season at months idx (past or future)
    X = design(F["start"], idx, F["cps"], F["K"]); m = 2 + len(F["cps"])              # the trend's columns come first
    return X[:, :m] @ F["beta"][:m], X[:, m:] @ F["beta"][m:]


def prophet_forecast(F, h):                             # NEW: the same curve, h months past the last month
    return sum(components(F, np.arange(F["n"], F["n"] + h)))                          # every hinge keeps rising: the last slope continues


def prophet_sims(F, h, sims=1000, seed=0):              # NEW: Prophet's interval -- simulated future slope changes, plus noise
    r = np.random.RandomState(seed)                                                   # reproducible
    rate = len(F["cps"]) / F["n"]                                                     # changepoints per month in the history
    scale = np.mean(np.abs(F["delta"])) if len(F["delta"]) else 0.0                   # their typical size (Laplace scale, as Prophet)
    size = r.laplace(0, scale, (sims, h)) * (r.rand(sims, h) < rate)                  # a slope change in some future months
    ramp = (np.cumsum(np.cumsum(size, axis=1), axis=1) - np.cumsum(size, axis=1)) / 12.0  # a change at month j adds size x (k - j) / 12 at month k
    return prophet_forecast(F, h) + ramp + r.randn(sims, h) * F["sigma"]              # (sims, h) possible futures


def ar1_correction(F, h):                               # NEW: the leftover error's memory -- AR(1) on the residuals, decaying ahead
    r = F["resid"]; phi = (r[:-1] @ r[1:]) / (r[:-1] @ r[:-1])                        # least squares of each residual on the one before
    return r[-1] * phi ** np.arange(1, h + 1), phi                                    # the last residual, shrinking by phi each month


def choose(y, start, origins, h=24, Ks=KS, lams=LAMS):  # NEW: mean h-month error from each validation origin, for every setting
    E = np.array([[np.mean([np.mean(np.abs(prophet_forecast(fit_prophet(y[:o], start, K, lam), h) - y[o:o + h])) for o in origins])
                   for lam in lams] for K in Ks])                                     # (waves, penalties)
    i, j = np.unravel_index(np.argmin(E), E.shape)                                    # the smallest mean error
    return Ks[i], lams[j], E


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def split(tc):                                          # Day 147's split: train up to 1999-12, forecast the last 24 months
    return int(np.flatnonzero(tc == np.datetime64("2000-01"))[0])


def at(t, month):                                       # the position of a month in a list of months
    return int(np.flatnonzero(t == np.datetime64(month))[0])


def airline_fit(y):                                     # Day 152's helper, unchanged: the airline polynomials (CSS)
    w = np.diff(y[12:] - y[:-12]); th, s2, _ = d149.fit_css(w, AIRLINE); a, b = d149.sarima_polys(th, AIRLINE)
    return d149.full_ar(a), a, b


def airline_forecast(fit, y, h):                        # Day 152's helper, unchanged: forecast with fixed coefficients
    A, a, b = fit; w = np.diff(y[12:] - y[:-12])                                      # the differenced history
    return d149.forecast(y, A, b, d149.css_residuals(w, a, b), h)                     # errors recomputed on this history


def fit_hw(y, season, damped):                         # Day 150's fit_ets, with its harmless overflow warning silenced
    with np.errstate(over="ignore"):                                                  # exp(-u) overflows for large u; 1 / (1 + inf) is 0, as intended
        return d150.fit_ets(y, "add", season, damped)


def hw_forecast(F, y, h):                               # Holt-Winters with fixed weights, run over y, then h months ahead
    _, l, b, s, _ = d150.ets_filter(y, F["alpha"], F["beta"], F["gamma"], F["phi"], *F["init"], F["trend_kind"], F["season_kind"])
    return d150.ets_forecast(dict(F, level=l, slope=b, season=s, n=len(y)), h)        # the final states replace the fitted ones


def regression(y, start, h):                            # Day 151's chosen regression (lags 1-13), fitted on y, recursive
    spec = dict(lags=tuple(range(1, 14))); X, _ = d151.make_features(y, start, spec)
    return d151.recursive_forecast(y, start, spec, d151.fit_ols(X, y), h)


def demo_check(tc, yc):
    print("\n" + "=" * 70)
    print("1. THE PARTS, CHECKED -- known slopes recovered, and the fit against plain least squares")
    print("=" * 70)
    t = np.arange(180); season = 2 * np.sin(2 * np.pi * (t % 12) / 12)                # 15 years; a season of one wave
    trend = np.where(t < 60, 1.0 * t / 12, np.where(t < 120, 5 + 3.0 * (t - 60) / 12, 20 + 0.5 * (t - 120) / 12))  # slopes 1, 3, 0.5 a year
    y = 10 + trend + season + np.random.RandomState(0).randn(180) * 0.3               # plus noise
    F = fit_prophet(y, "2000-01", K=1, lam=0.0, count=2, frac=2 / 3)                  # changepoints at months 60 and 120, no penalty
    sl = np.cumsum(np.r_[F["beta"][1], F["delta"]])                                   # the slope in each piece
    print(f"a line with slopes 1, 3, 0.5 a year (changes at months 60 and 120), a season and noise sd 0.3; changepoints placed at "
          f"{F['cps'].tolist()}")
    print(f"  fitted slopes {', '.join(f'{s:.3f}' for s in sl)} a year; noise sd left over {F['sigma']:.3f}")
    n = split(tc); F = fit_prophet(yc[:n], tc[0], K=4, lam=0.0); X = design(tc[0], np.arange(n), F["cps"], 4)
    print(f"CO2 to 1999-12, no penalty: against Day 151's fit_ols on the same columns, largest coefficient difference "
          f"{np.max(np.abs(F['beta'] - d151.fit_ols(X, yc[:n]))):.1e}")
    F = fit_prophet(yc[:n], tc[0], K=4, lam=100.0); m = 2 + len(F["cps"]); D = np.zeros(X.shape[1]); D[2:m] = 100.0
    b = np.linalg.solve(X.T @ X + np.diag(D), X.T @ yc[:n])                           # the closed form: (X'X + lam D) b = X'y
    print(f"penalty 100: stacked least squares against the closed form (X'X + lam D) b = X'y, largest difference {np.max(np.abs(F['beta'] - b)):.1e}")
    F0 = fit_prophet(yc[:n], tc[0], K=4, count=0); spec = dict(trend=True, calendar=4)  # no changepoints: a straight line plus waves
    Xs, _ = d151.make_features(yc[:n], tc[0], spec)                                   # Day 151's trend and Fourier columns
    print(f"no changepoints: fitted values against Day 151's regression with a trend and 4 waves, largest difference "
          f"{np.max(np.abs(sum(components(F0, np.arange(n))) - Xs @ d151.fit_ols(Xs, yc[:n]))):.1e}")
    f = prophet_forecast(F, 24); s = F["beta"][1] + F["delta"].sum()                  # the forecast, and the last piece's slope
    tr = components(F, np.arange(n - 1, n + 24))[0]                                   # the trend from the last month on
    print(f"the forecast's trend rises {np.mean(np.diff(tr)) * 12:.4f} ppm a year; the last piece's slope is {s:.4f}")
    sm = prophet_sims(F, 24); print(f"1000 simulated futures, 24 months each; their mean against the forecast, largest gap "
                                    f"{np.max(np.abs(sm.mean(axis=0) - f)):.3f} ppm (noise sd {F['sigma']:.3f})")
    print("Actual output from this lesson's run.")


def demo_choose(tc, yc):
    print("\n" + "=" * 70)
    print("2. CHOOSING THE WAVES AND THE PENALTY -- validation origins whose 24 months end by 1989-12")
    print("=" * 70)
    first = at(tc, "1990-01"); origins = range(first - 24 - 96, first - 24 + 1); t0 = time.perf_counter()
    K, lam, E = choose(yc, tc[0], origins)                                            # every setting, refitted at every origin
    line = np.mean([np.mean(np.abs(prophet_forecast(fit_prophet(yc[:o], tc[0], K, count=0), 24) - yc[o:o + 24])) for o in origins])
    print(f"{len(origins)} origins, {tc[origins[0]]} to {tc[origins[-1]]} ({time.perf_counter() - t0:.1f} s); mean MAE (ppm) over 24 months")
    print(f"{'waves':>8} " + " ".join(f"{'pen ' + format(v, 'g'):>9}" for v in LAMS))
    for k, row in zip(KS, E):
        print(f"{k:>8} " + " ".join(f"{v:9.3f}" for v in row))
    print(f"validation's choice: {K} waves, penalty {lam:g} ({E.min():.3f}); no changepoints at all (a straight line) {line:.3f}")
    n = split(tc); test = [np.mean(np.abs(prophet_forecast(fit_prophet(yc[:n], tc[0], K, l), 24) - yc[n:])) for l in LAMS]
    print(f"did the choice hold? test MAE (2000-2001) with {K} waves, by penalty: " + ", ".join(f"{l:g}: {v:.3f}" for l, v in zip(LAMS, test)))
    print(f"  the choice ranks {1 + np.sum(np.array(test) < test[LAMS.index(lam)])} of {len(LAMS)} on the test; "
          "the test is reported, not used to choose")
    print("Actual output from this lesson's run.")
    return K, lam, E, test


def demo_components(tc, yc, K, lam):
    print("\n" + "=" * 70)
    print("3. WHAT THE MODEL SAYS ABOUT CO2 -- fitted to 1999-12")
    print("=" * 70)
    n = split(tc); F = fit_prophet(yc[:n], tc[0], K, lam); sl = F["beta"][1] + np.r_[0, np.cumsum(F["delta"])]  # the slope in each piece
    print(f"{len(F['cps'])} changepoints, {tc[F['cps'][0]]} to {tc[F['cps'][-1]]}; the slope went from {sl[0]:.3f} ppm a year "
          f"(1958) to {sl[-1]:.3f} (after {tc[F['cps'][-1]]})")
    big = np.argsort(-np.abs(F["delta"]))[:3]                                         # the three largest slope changes
    print("  largest slope changes: " + ", ".join(f"{tc[F['cps'][j]]} {F['delta'][j]:+.3f}" for j in big))
    _, S = components(F, np.arange(n)); T7, S7, _ = d147.classical_decompose(yc[:n])  # ours, and Day 147's
    a, b = d147.by_month(S, tc[:n]), d147.by_month(S7, tc[:n])                        # each calendar month's seasonal value
    print(f"the season ({K} waves) against Day 147's classical decomposition: largest difference {np.max(np.abs(a - b)):.3f} ppm "
          f"(peak {a.max():.2f} against {b.max():.2f}, month {np.argmax(a) + 1} of 12)")
    r = F["resid"]; ar = airline_fit(yc[:n]); w = np.diff(yc[12:n] - yc[:n - 12]); e = d149.css_residuals(w, ar[1], ar[2])
    print(f"what is left: sd {F['sigma']:.3f} ppm; autocorrelation at lags 1, 2, 12: "
          f"{', '.join(f'{v:.3f}' for v in d148.acf(r, 12)[[1, 2, 12]])}")
    q, df = d149.ljung_box(r, 24); qa, dfa = d149.ljung_box(e[13:], 24, 2)            # the airline model's residuals, for comparison
    print(f"  Ljung-Box over 24 lags: Q = {q:.1f} (5% critical value about 36); the airline model's residuals Q = {qa:.1f}")
    print(f"  KPSS on what is left: {d148.kpss(r):.3f} (5% critical value 0.463): it wanders, but around a fixed level")
    print("the curve captures the trend and the season; the month-to-month memory is left in the errors, and it forecasts")
    print("as if it were not there")
    print("Actual output from this lesson's run.")
    return F


def demo_forecast(tc, yc, K, lam):
    print("\n" + "=" * 70)
    print("4. FORECASTING CO2 IN 2000-2001 -- against every model of the week")
    print("=" * 70)
    n = split(tc); y = yc[:n]; act = yc[n:]; t0 = time.perf_counter()
    F = fit_prophet(y, tc[0], K, lam); f = prophet_forecast(F, 24); c, phi = ar1_correction(F, 24)
    L = d152.fit_lstm(y); lstm = d152.lstm_forecast(L, y, 24).mean(axis=0)             # Day 152's five networks
    hw = fit_hw(y, "mul", False)                                                      # Day 150's AIC choice for CO2
    rows = [("Prophet-style (validation's choice)", f), ("Prophet-style + AR(1) on what is left", f + c),
            ("straight line + season (no changepoints)", prophet_forecast(fit_prophet(y, tc[0], K, count=0), 24)),
            ("airline ARIMA (Day 149)", airline_forecast(airline_fit(y), y, 24)), ("Holt-Winters, mult. (Day 150)", d150.ets_forecast(hw, 24)),
            ("regression, lags 1-13 (Day 151)", regression(y, tc[0], 24)), ("LSTM, average of 5 (Day 152)", lstm),
            ("control: average change (Day 152)", d152.seasonal_drift(y, tc[0], 24)), ("same month last year (Day 147)", y[-12:][np.arange(24) % 12])]
    print(f"all models fitted on data to 1999-12 ({time.perf_counter() - t0:.1f} s, mostly the networks); AR(1) on the residuals: phi = {phi:.3f}")
    print(f"{'2000-01 to 2001-12, 24 months':>42} {'MAE':>7} {'bias':>7}")
    for name, g in rows:
        print(f"{name:>42} {np.mean(np.abs(g - act)):7.3f} {np.mean(g - act):+7.3f}")
    sm = prophet_sims(F, 24); lo, hi = np.percentile(sm, [10, 90], axis=0)             # Prophet's 80% interval
    print(f"Prophet-style 80% interval: half-width {(hi[0] - lo[0]) / 2:.2f} ppm one month ahead, {(hi[-1] - lo[-1]) / 2:.2f} at 24; "
          f"{np.sum((act >= lo) & (act <= hi))} of 24 months inside")
    print("Actual output from this lesson's run.")
    return dict(rows), (lo, hi), F


def demo_rolling(tc, yc, K, lam):
    print("\n" + "=" * 70)
    print("5. ROLLING ORIGINS -- 97 forecasts from 1990-1998; how often the 80% intervals hold")
    print("=" * 70)
    n = split(tc); first = at(tc, "1990-01"); t0 = time.perf_counter()
    once = fit_prophet(yc[:first], tc[0], K, lam); ar = airline_fit(yc[:first]); hw = fit_hw(yc[:first], "mul", False)
    L = d152.fit_lstm(yc[:first]); spec = dict(lags=tuple(range(1, 14))); X, _ = d151.make_features(yc[:first], tc[0], spec)
    b = d151.fit_ols(X, yc[:first])                                                   # everything except the curve: fitted once
    w = np.diff(yc[12:first] - yc[:first - 12]); _, s2, _ = d149.fit_css(w, AIRLINE)   # the airline model's error variance
    sd = np.sqrt(s2 * np.cumsum(d149.psi_weights(ar[0], ar[2], 24) ** 2))             # its forecast sd, 1-24 months ahead
    E = {k: [] for k in ("Prophet-style, refitted", "Prophet-style, fitted once", "Prophet-style + AR(1)", "airline ARIMA",
                         "Holt-Winters, mult.", "regression, lags 1-13", "LSTM, average of 5", "control: average change")}
    C = {"Prophet-style": [], "airline ARIMA": []}                                    # inside the 80% interval or not
    for o in range(first, n - 24 + 1):                                                # each origin
        act = yc[o:o + 24]; F = fit_prophet(yc[:o], tc[0], K, lam); f = prophet_forecast(F, 24)  # the curve, refitted: one least squares
        E["Prophet-style, refitted"].append(np.abs(f - act))
        E["Prophet-style, fitted once"].append(np.abs(sum(components(once, np.arange(o, o + 24))) - act))  # the 1989 curve, read further on
        E["Prophet-style + AR(1)"].append(np.abs(f + ar1_correction(F, 24)[0] - act))
        a = airline_forecast(ar, yc[:o], 24); E["airline ARIMA"].append(np.abs(a - act))
        E["Holt-Winters, mult."].append(np.abs(hw_forecast(hw, yc[:o], 24) - act))
        E["regression, lags 1-13"].append(np.abs(d151.recursive_forecast(yc[:o], tc[0], spec, b, 24) - act))
        E["LSTM, average of 5"].append(np.abs(d152.lstm_forecast(L, yc[:o], 24).mean(axis=0) - act))
        E["control: average change"].append(np.abs(d152.seasonal_drift(yc[:o], tc[0], 24) - act))
        lo, hi = np.percentile(prophet_sims(F, 24, seed=o), [10, 90], axis=0)          # Prophet's 80% interval
        C["Prophet-style"].append((act >= lo) & (act <= hi)); C["airline ARIMA"].append(np.abs(a - act) <= 1.2816 * sd)
    E = {k: np.array(v) for k, v in E.items()}; C = {k: np.array(v) for k, v in C.items()}
    print(f"{len(E['airline ARIMA'])} origins ({time.perf_counter() - t0:.1f} s); the curve and the control recomputed at every origin, the rest fitted once on data to 1989-12")
    print(f"{'MAE (ppm), months ahead':>30} " + " ".join(f"{k:>7}" for k in HS))
    for k, v in E.items():
        print(f"{k:>30} " + " ".join(f"{v[:, j - 1].mean():7.3f}" for j in HS))
    m = {k: v.mean(axis=0) for k, v in E.items()}
    print(f"the refitted curve beats the airline model at {np.sum(m['Prophet-style, refitted'] < m['airline ARIMA'])} of 24 horizons; "
          f"with AR(1) added, at {np.sum(m['Prophet-style + AR(1)'] < m['airline ARIMA'])} of 24")
    print(f"{'inside the 80% interval':>30} " + " ".join(f"{k:>7}" for k in HS))
    for k, v in C.items():
        print(f"{k:>30} " + " ".join(f"{v[:, j - 1].mean():7.3f}" for j in HS))
    print("Actual output from this lesson's run.")
    return E, C


def demo_orders(te, ye):
    print("\n" + "=" * 70)
    print("6. ELECTRICAL-EQUIPMENT ORDERS -- log scale; settings chosen on validation before each cut")
    print("=" * 70)
    le = np.log(ye); keep = {}                                                        # the season adds on the log scale
    mape = lambda f, act: 100 * np.mean(np.abs(np.exp(f - act) - 1))                  # mean absolute % error
    for name, cut in (("the crash", at(te, "2008-01")), ("the last 24 months", len(le) - 24)):
        y = le[:cut]; act = le[cut:cut + 24]; origins = range(cut - 24 - 48, cut - 24 + 1)  # 49 validation origins, ending by the cut
        K, lam, E = choose(y, te[0], origins); F = fit_prophet(y, te[0], K, lam); f = prophet_forecast(F, 24)
        L = d152.fit_lstm(y, nval=24); hw = fit_hw(y, "add", True)                     # Day 152's networks; Day 150's AIC choice (damped)
        rows = {"Prophet-style": f, "+ AR(1)": f + ar1_correction(F, 24)[0], "line + season": prophet_forecast(fit_prophet(y, te[0], K, count=0), 24),
                "airline": airline_forecast(airline_fit(y), y, 24), "Holt-Winters": d150.ets_forecast(hw, 24),
                "regression": regression(y, te[0], 24), "LSTM avg": d152.lstm_forecast(L, y, 24).mean(axis=0),
                "control": d152.seasonal_drift(y, te[0], 24), "last year": y[-12:][np.arange(24) % 12]}
        print(f"{name}: train up to {te[cut - 1]}; validation chose {K} waves, penalty {lam:g}; 24 months, MAE:")
        print("  " + "; ".join(f"{k} {mape(g, act):.2f}%" for k, g in rows.items()))
        keep[name] = (cut, rows)
    F = fit_prophet(le, te[0], 4, 1.0); big = np.argsort(-np.abs(F["delta"]))[:3]      # the whole series, as a description, not a forecast
    print(f"the whole series (4 waves, penalty 1, changepoints to {te[F['cps'][-1]]}): largest slope changes "
          + ", ".join(f"{te[F['cps'][j]]} {F['delta'][j]:+.3f}" for j in big) + " (log units a year)")
    print("Actual output from this lesson's run.")
    return keep, F


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(tc, yc, te, ye, Fc, fore, roll, orders, out):
    fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 3); n = split(tc); x = tc.astype("datetime64[D]")
    ax = fig.add_subplot(gs[0, 0]); T, _ = components(Fc, np.arange(n))               # the trend, with its changepoints
    ax.plot(x[:n], yc[:n], color="#BBBBBB", lw=0.6, label="CO2, monthly"); ax.plot(x[:n], T, color="#4C72B0", lw=1.6, label="fitted trend")
    for c, d in zip(Fc["cps"], Fc["delta"]):
        ax.axvline(x[c], color="#C44E52" if d > 0 else "#55A868", lw=0.3 + 12 * abs(d), alpha=0.7)  # line width: the size of the change
    ax.plot([], [], color="#C44E52", label="changepoint, slope up"); ax.plot([], [], color="#55A868", label="changepoint, slope down")
    ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=8, loc="upper left"); ax.set_title("the trend bends at changepoints (to 1999-12)", fontsize=10)
    ax = fig.add_subplot(gs[0, 1]); _, S = components(Fc, np.arange(n)); _, S7, _ = d147.classical_decompose(yc[:n])
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.plot(names, d147.by_month(S, tc[:n]), "o-", color="#4C72B0", label=f"Fourier, {Fc['K']} waves (today)")
    ax.plot(names, d147.by_month(S7, tc[:n]), "s--", color="#DD8452", ms=4, label="classical decomposition (Day 147)")
    ax.axhline(0, color="gray", lw=0.6); ax.set_ylabel("seasonal effect (ppm)"); ax.legend(fontsize=8); ax.set_title("the season, two ways", fontsize=10)
    rows, (lo, hi), _ = fore; ax = fig.add_subplot(gs[0, 2])
    ax.plot(x[n - 24:], yc[n - 24:], color="k", lw=1.2, label="actual"); ax.fill_between(x[n:], lo, hi, color="#4C72B0", alpha=0.2, label="Prophet-style 80% interval")
    for name, c, ls in (("Prophet-style (validation's choice)", "#4C72B0", "-"), ("airline ARIMA (Day 149)", "#8172B2", "--"),
                        ("LSTM, average of 5 (Day 152)", "#55A868", "-."), ("control: average change (Day 152)", "#DD8452", ":")):
        ax.plot(x[n:], rows[name], color=c, ls=ls, lw=1.3, label=name)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axvline(x[n], color="gray", ls=":"); ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("2000-2001 from data up to 1999-12", fontsize=10)
    E, C = roll; ax = fig.add_subplot(gs[1, 0]); hz = np.arange(1, 25)
    for (name, v), c in zip(E.items(), ("#4C72B0", "#9DB7DD", "#1F3A68", "#8172B2", "#C44E52", "#937860", "#55A868", "#DD8452")):
        ax.plot(hz, v.mean(axis=0), marker="o", ms=3, color=c, label=name)
    ax.set_xlabel("months ahead"); ax.set_ylabel("MAE (ppm)"); ax.legend(fontsize=7); ax.set_title("97 origins, 1990-1998", fontsize=10)
    ax = fig.add_subplot(gs[1, 1])
    for (name, v), c in zip(C.items(), ("#4C72B0", "#8172B2")):
        ax.plot(hz, v.mean(axis=0), marker="o", ms=3, color=c, label=name)
    ax.axhline(0.8, color="k", ls="--", lw=0.8, label="promised: 0.8"); ax.set_ylim(0, 1); ax.set_xlabel("months ahead")
    ax.set_ylabel("share of actual months inside"); ax.legend(fontsize=8); ax.set_title("80% intervals: how often they held", fontsize=10)
    keep, Fe = orders; ax = fig.add_subplot(gs[1, 2]); xe = te.astype("datetime64[D]"); Te, Se = components(Fe, np.arange(len(ye)))
    ax.plot(xe, ye, color="#BBBBBB", lw=0.8, label="orders"); ax.plot(xe, np.exp(Te), color="#4C72B0", lw=1.6, label="fitted trend (whole series)")
    for j in np.argsort(-np.abs(Fe["delta"]))[:3]:
        ax.axvline(xe[Fe["cps"][j]], color="#C44E52" if Fe["delta"][j] > 0 else "#55A868", lw=1.2, ls="--")
    cut, r = keep["the crash"]; ax.plot(xe[cut:cut + 24], np.exp(r["Prophet-style"]), color="#8172B2", lw=1.3, label="Prophet-style from 2007-12")
    ax.plot([], [], color="#C44E52", ls="--", label="largest slope changes"); ax.set_ylabel("orders (index)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("orders: where the slope changed", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the forecaster, from the months to a forecast and its interval
    fig, ax = plt.subplots(figsize=(15, 5.6)); ax.set_xlim(0, 15); ax.set_ylim(0.2, 5.6); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.4, 3.0, 2.2, 1.0, "month index t\n(years since start)", "#DDEBF7")
    box(4.6, 4.5, 3.0, 0.9, "1, t, and a hinge (t - c)+\nat each of 25 changepoints", "#FFF2CC")
    box(4.6, 1.5, 3.0, 0.9, "Fourier waves (Day 151):\nsin and cos of 2 pi k month / 12", "#FFF2CC")
    box(8.2, 3.0, 2.8, 1.3, "least squares, with a penalty\nlam x (slope changes)^2\nsettings chosen on validation", "#E2F0D9")
    box(11.6, 4.5, 2.6, 0.9, "trend g(t): a line that\nbends at changepoints", "#E2F0D9")
    box(11.6, 1.5, 2.6, 0.9, "season s(t): the same\nshape every year", "#E2F0D9")
    box(13.9, 3.0, 1.9, 1.3, "forecast:\ng(t) + s(t)\npast the end", "#FCE4D6")
    arrow(2.5, 3.2, 3.1, 4.4); arrow(2.5, 2.8, 3.1, 1.6); arrow(6.1, 4.4, 6.8, 3.4); arrow(6.1, 1.6, 6.8, 2.6)
    arrow(9.6, 3.4, 10.3, 4.4); arrow(9.6, 2.6, 10.3, 1.6); arrow(12.9, 4.3, 13.2, 3.65); arrow(12.9, 1.7, 13.2, 2.35)
    ax.text(7.5, 0.45, "interval: simulate future slope changes at the history's rate and typical size, add noise, take the 10th and 90th "
            "percentiles", ha="center", fontsize=8.5, style="italic")
    ax.set_title("Day 153: a Prophet-style forecaster from Week 22's parts", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "prophet_style_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Week 23: recommender systems -- Day 154 starts with collaborative filtering, user-based and item-based,")
    print("on a real ratings matrix.")


def main():                                             # runs every section in order
    tc, yc, _ = d147.load_co2(); te, ye = d147.load_elec()                           # Day 147's two series
    demo_check(tc, yc)                                                                # 1. the parts, checked
    K, lam, _, _ = demo_choose(tc, yc)                                                # 2. settings, on validation
    Fc = demo_components(tc, yc, K, lam)                                              # 3. what the model says
    fore = demo_forecast(tc, yc, K, lam)                                              # 4. 2000-2001
    roll = demo_rolling(tc, yc, K, lam)                                               # 5. rolling origins
    orders = demo_orders(te, ye)                                                      # 6. the orders
    plot_results(tc, yc, te, ye, Fc, fore, roll, orders, os.path.join(HERE, "prophet_style_results.png"))
    draw_diagram()                                                                    # the forecaster
    print("\nsaved prophet_style_results.png\nsaved prophet_style_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
