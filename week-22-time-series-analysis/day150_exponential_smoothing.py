"""
Day 150: Exponential Smoothing from Scratch -- Simple, Holt and Holt-Winters, against ARIMA

Day 149 forecast CO2 with ARIMA: a model of the differenced series' memory.
Exponential smoothing takes a different route to forecasts. It keeps a running
estimate of the series' level (and its trend, and its season), and after each
new month it nudges every estimate a fraction of the way towards what that
month showed. Recent months therefore count most, and older months' weight
decays geometrically -- hence 'exponential'.

Three members of the family, each adding one part:
  simple (SES)   a level only: the forecast is flat;
  Holt           a level and a trend: the forecast is a straight line;
  Holt-Winters   a level, a trend and a season, added to or multiplying the level.

Everything is written from scratch in NumPy, including the fit (Day 149's
Levenberg-Marquardt solver, with every smoothing weight kept between 0 and 1);
statsmodels' ExponentialSmoothing is the reference. Every setting -- which
variant, the weights -- is chosen on earlier data only; the test months are
used once, at the end.

  1. Simple exponential smoothing: a weighted average, and the same model as ARIMA(0,1,1).
  2. Checked against statsmodels -- and the two steps where its forecasts differ.
  3. Choosing the variant for CO2 by AIC on the training years.
  4. Forecasting 2000-2001, against Day 147's forecasts and Day 149's ARIMA.
  5. Rolling origins: 97 forecasts, accuracy and interval coverage, against ARIMA.
  6. Electrical-equipment orders: an ordinary period, the crash, the last two years.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: large matrix products were very slow with this machine's default
import contextlib                                      # contextlib: redirect_stdout, to keep the imported parts quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import this week's parts
import time                                            # time: how long the fits take
import warnings                                        # warnings: statsmodels' notices are not needed here

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram
import statsmodels                                     # statsmodels: only its version number is printed
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # the library's version, to check ours
from statsmodels.tsa.statespace.sarimax import SARIMAX  # the library's seasonal ARIMA (Day 149's reference)
warnings.filterwarnings("ignore")                       # keep the output to this lesson's own lines

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts
with contextlib.redirect_stdout(io.StringIO()):         # the exercises print their own checks when imported; keep them quiet
    import day147_exercise as d147                      # Day 147: load_co2, load_elec, classical_decompose
    import day149_exercise as d149                      # Day 149: levenberg_marquardt, the ARIMA parts
M = 12                                                  # months in a season
STARTS = ((0.3, 0.05, 0.1, 0.5), (0.8, 0.5, 0.3, 0.5), (0.1, 0.01, 0.01, 0.5), (0.9, 0.9, 0.3, 0.5))  # (alpha, beta, gamma, phi) starting points
VARIANTS = (("SES", None, None, False), ("Holt", "add", None, False), ("Holt, damped", "add", None, True),
            ("Holt-Winters, additive", "add", "add", False), ("Holt-Winters, multiplicative", "add", "mul", False),
            ("HW additive, damped", "add", "add", True), ("HW multiplicative, damped", "add", "mul", True))  # (name, trend, season, damped)


# =============================================================================
# THE PARTS -- NEW (copied unchanged into day150_exercise.py)
# =============================================================================
def ets_filter(y, alpha, beta, gamma, phi, l0, b0, s0, trend, season, m=M):  # NEW: run the smoothing equations through y
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


def ets_init(y, m=M, season="add"):                     # NEW: starting values from the first two years
    l0 = y[:m].mean(); b0 = (y[m:2 * m].mean() - l0) / m                              # first year's mean; the change between years, per month
    return l0, b0, (y[:m] - l0 if season == "add" else y[:m] / l0)                    # each first-year month against that mean


def fit_ets(y, trend, season, damped, m=M, starts=STARTS):  # NEW: least squares from several starts; keep the best
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


def ets_forecast(F, h, m=M):                            # NEW: h months ahead from the final states
    steps = np.cumsum(F["phi"] ** np.arange(1, h + 1)) if F["trend_kind"] else np.zeros(h)  # phi + phi^2 + ...: the damped number of trend steps
    base = F["level"] + steps * F["slope"]                                            # level plus trend
    s = F["season"][(F["n"] + np.arange(h)) % m]                                      # each future month's latest seasonal value
    return base + s if F["season_kind"] == "add" else base * s if F["season_kind"] == "mul" else base


def ets_sd(F, h, m=M):                                  # NEW: forecast standard deviation for the additive model (undamped)
    j = np.arange(1, h)                                                               # 1 ... h-1
    c = F["alpha"] * (1 + F["beta"] * j) + F["gamma"] * (j % m == 0)                  # how much one error moves the forecast j steps later
    return np.sqrt(F["sse"] / F["n"] * np.r_[1.0, 1 + np.cumsum(c ** 2)])             # sigma^2 (1 + c_1^2 + ... + c_(h-1)^2)


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def aic(F):                                             # AIC from the sum of squared one-step errors
    return F["n"] * np.log(F["sse"] / F["n"]) + 2 * F["k"]


def split(tc):                                          # Day 147's split: train up to 1999-12, forecast the last 24 months
    return int(np.flatnonzero(tc == np.datetime64("2000-01"))[0])


def airline(y, h):                                      # Day 149's airline model (CSS): forecast and standard deviation
    w = np.diff(y[12:] - y[:-12]); th, s2, _ = d149.fit_css(w, (0, 1, 0, 1, 12)); a, b = d149.sarima_polys(th, (0, 1, 0, 1, 12))
    A = d149.full_ar(a); return d149.forecast(y, A, b, d149.css_residuals(w, a, b), h), np.sqrt(s2 * np.cumsum(d149.psi_weights(A, b, h) ** 2))


def demo_ses():
    print("=" * 70)
    print("1. SIMPLE EXPONENTIAL SMOOTHING -- a weighted average, and the same model as ARIMA(0,1,1)")
    print("=" * 70)
    print("the level after each month: level = alpha x this month + (1 - alpha) x the old level; unrolled, month k back gets")
    print("weight alpha (1 - alpha)^k")
    for a in (0.2, 0.5, 0.8):
        w = a * (1 - a) ** np.arange(12)                                              # the weights of the last 12 months
        print(f"  alpha {a}: weights of this month and the 3 before {np.round(w[:4], 3).tolist()}; last 12 months together {w.sum():.3f}")
    rng = np.random.RandomState(0); z = np.cumsum(rng.standard_normal(300)) + 2 * rng.standard_normal(300)  # a wandering level plus noise
    F = fit_ets(z, None, None, False); a = F["alpha"]                                 # SES fitted by least squares
    f = ets_filter(z, a, 0, 0, 1, z[0], 0, np.zeros(M), None, None)[0]                # SES started at the first value
    e = d149.css_residuals(np.diff(z), np.array([1.0]), np.array([1.0, a - 1]))      # ARIMA(0,1,1) with theta = alpha - 1, CSS errors
    print(f"a wandering level plus noise (300 values): fitted alpha {a:.4f}; SES's one-step errors against ARIMA(0,1,1) with "
          f"theta = alpha - 1 = {a - 1:.4f}: largest difference {np.max(np.abs((z - f)[1:] - e)):.1e}")
    print("so SES is a special case of Day 149's model: both say the series is a random walk seen through noise")
    print("Actual output from this lesson's run.")


def demo_check(tc, yc):
    print("\n" + "=" * 70)
    print("2. CHECKED AGAINST STATSMODELS -- and the two steps where its forecasts differ")
    print("=" * 70)
    n = split(tc); y = yc[:n]
    for season, damped in (("add", False), ("mul", False), ("add", True)):
        l0, b0, s0 = ets_init(y, M, season); phi = 0.9 if damped else 1.0             # the same fixed weights and starting values for both
        ours = ets_filter(y, 0.5, 0.1, 0.2, phi, l0, b0, s0, "add", season)[0]
        ref = ExponentialSmoothing(y, trend="add", damped_trend=damped, seasonal=season, seasonal_periods=M, initialization_method="known",
                                   initial_level=l0, initial_trend=b0, initial_seasonal=s0).fit(
            smoothing_level=0.5, smoothing_trend=0.1, smoothing_seasonal=0.2, damping_trend=phi if damped else None, optimized=False)
        print(f"  fixed weights, {season + (', damped' if damped else ''):<13} one-step forecasts: largest difference {np.max(np.abs(ours - ref.fittedvalues)):.1e}")
    for season in ("add", "mul"):
        F = fit_ets(y, "add", season, False); l0, b0, s0 = F["init"]
        ref = ExponentialSmoothing(y, trend="add", seasonal=season, seasonal_periods=M, initialization_method="known",
                                   initial_level=l0, initial_trend=b0, initial_seasonal=s0).fit()  # statsmodels' own optimiser, same starting states
        print(f"  fitted, {season}: ours alpha {F['alpha']:.4f} beta {F['beta']:.4f} gamma {F['gamma']:.4f}, sum of squares {F['sse']:.4f}; "
              f"statsmodels {ref.params['smoothing_level']:.4f} {ref.params['smoothing_trend']:.4f} {ref.params['smoothing_seasonal']:.4f}, {ref.sse:.4f}")
    F = fit_ets(y, "add", "add", False); l0, b0, s0 = F["init"]
    ref = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=M, initialization_method="known", initial_level=l0,
                               initial_trend=b0, initial_seasonal=s0).fit(smoothing_level=F["alpha"], smoothing_trend=F["beta"],
                                                                          smoothing_seasonal=F["gamma"], optimized=False)
    d = ets_forecast(F, 24) - ref.forecast(24); bad = np.flatnonzero(np.abs(d) > 1e-9) + 1
    print(f"  forecasts with the same weights: equal at {24 - len(bad)} of 24 steps; steps {bad.tolist()} differ by {d[bad[0] - 1]:+.4f}")
    k = (n - 1) % M; _, _, _, s_prev, _ = ets_filter(y[:-1], F["alpha"], F["beta"], F["gamma"], 1.0, l0, b0, s0, "add", "add")
    print(f"  at step 12 statsmodels used {ref.forecast(24)[11] - F['level'] - 12 * F['slope']:.4f} as the seasonal value: the value before the "
          f"last update ({s_prev[k]:.4f}), not the latest ({F['season'][k]:.4f})")
    y2 = np.r_[y, ets_forecast(F, 12)]                                                # pretend the next 12 months came in exactly as forecast
    ours = ets_filter(y2, F["alpha"], F["beta"], F["gamma"], 1.0, l0, b0, s0, "add", "add")[0][n:]
    ref2 = ExponentialSmoothing(y2, trend="add", seasonal="add", seasonal_periods=M, initialization_method="known", initial_level=l0,
                                initial_trend=b0, initial_seasonal=s0).fit(smoothing_level=F["alpha"], smoothing_trend=F["beta"],
                                                                           smoothing_seasonal=F["gamma"], optimized=False)
    print(f"  self-check: feed the 12 forecasts back in as data (every error 0); each one-step forecast should equal the forecast:")
    print(f"    ours {np.max(np.abs(ours - ets_forecast(F, 12))):.1e}; statsmodels' {np.max(np.abs(ref2.fittedvalues[n:] - ref.forecast(12))):.1e}"
          f" (statsmodels {statsmodels.__version__})")
    print("Actual output from this lesson's run.")


def demo_choose(tc, yc):
    print("\n" + "=" * 70)
    print("3. CHOOSING THE VARIANT FOR CO2 -- AIC on the training years only")
    print("=" * 70)
    n = split(tc); y = yc[:n]; fits = {}; t0 = time.perf_counter()
    print(f"{'variant':>30} {'alpha':>7} {'beta':>7} {'gamma':>7} {'phi':>6} {'sum of sq.':>11} {'AIC':>9}")
    for name, trend, season, damped in VARIANTS:
        F = fit_ets(y, trend, season, damped); fits[name] = F                         # three starts each
        print(f"{name:>30} {F['alpha']:7.4f} {F['beta']:7.4f} {F['gamma']:7.4f} {F['phi']:6.3f} {F['sse']:11.4f} {aic(F):9.2f}")
    best = min(fits, key=lambda k: aic(fits[k]))                                      # the smallest AIC
    print(f"({time.perf_counter() - t0:.1f} s) AIC's choice: {best}")
    three = fit_ets(y, "add", None, False, starts=STARTS[:3])                         # the first draft: three starting points
    print(f"first draft, three starts: Holt stopped at a sum of squares of {three['sse']:.4f} (alpha {three['alpha']:.4f}, beta {three['beta']:.4f});")
    print(f"the fourth start (alpha 0.9, beta 0.9) finds {fits['Holt']['sse']:.4f} (alpha {fits['Holt']['alpha']:.4f}, beta {fits['Holt']['beta']:.4f}):"
          " a local minimum, and several starts are no guarantee")
    print("Actual output from this lesson's run.")
    return fits, best


def demo_forecast(tc, yc, fits, best):
    print("\n" + "=" * 70)
    print("4. FORECASTING 2000-2001 -- against Day 147's forecasts and Day 149's ARIMA")
    print("=" * 70)
    n = split(tc); h = len(yc) - n; act = yc[n:]
    ar, ar_sd = airline(yc[:n], h)                                                     # Day 149's airline model
    ml = SARIMAX(yc[:n], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(h)
    rows = [("same month last year (Day 147)", yc[n - 12:n][np.arange(h) % 12]), ("airline ARIMA, CSS (Day 149)", ar),
            ("airline ARIMA, MLE (Day 149)", ml)] + [(name, ets_forecast(F, h)) for name, F in fits.items()]
    print(f"{tc[n]} to {tc[-1]}, {h} months              MAE (ppm)   bias")
    for name, f in rows:
        print(f"  {name + (' <- AIC' if name == best else ''):<38} {np.mean(np.abs(f - act)):8.3f} {np.mean(f - act):+7.3f}")
    F = fits["Holt-Winters, additive"]; sd = ets_sd(F, h); fa = ets_forecast(F, h)
    print(f"Holt-Winters additive 95% interval: half-width {1.96 * sd[0]:.2f} one month ahead, {1.96 * sd[-1]:.2f} at 24; "
          f"{np.sum(np.abs(fa - act) < 1.96 * sd)} of {h} months inside (the airline model's: {1.96 * ar_sd[-1]:.2f} at 24)")
    print("AIC's choice and the additive version are close on the training data; on these 24 test months the additive one did")
    print("better -- picking it now would be choosing on the test, so the comparison stands as AIC chose")
    print("Actual output from this lesson's run.")
    return rows


def demo_rolling(tc, yc, best):
    print("\n" + "=" * 70)
    print("5. ROLLING ORIGINS -- 97 forecasts, 24 months ahead, 1990-1998, against ARIMA")
    print("=" * 70)
    n = split(tc); first = int(np.flatnonzero(tc == np.datetime64("1990-01"))[0]); kinds = {"Holt-Winters, additive": "add", "Holt-Winters, multiplicative": "mul"}
    E = {best + " (AIC)": [], [k for k in kinds if k != best][0]: [], "airline ARIMA, CSS": [], "same month last year": []}; C = {"HW additive": [], "airline": []}
    t0 = time.perf_counter()
    for o in range(first, n - 24 + 1):                                                # each origin: fit on everything before it
        act = yc[o:o + 24]
        for name, kind in kinds.items():
            F = fit_ets(yc[:o], "add", kind, False); f = ets_forecast(F, 24)
            E[best + " (AIC)" if name == best else name].append(f - act)
            if kind == "add":
                C["HW additive"].append(np.abs(f - act) < 1.96 * ets_sd(F, 24))
        f, sd = airline(yc[:o], 24); E["airline ARIMA, CSS"].append(f - act); C["airline"].append(np.abs(f - act) < 1.96 * sd)
        E["same month last year"].append(yc[o - 12:o][np.arange(24) % 12] - act)
    E = {k: np.array(v) for k, v in E.items()}; C = {k: np.array(v) for k, v in C.items()}
    print(f"{len(E['airline ARIMA, CSS'])} origins ({time.perf_counter() - t0:.1f} s); all forecasts end by {tc[n - 1]}")
    hs = (1, 6, 12, 24)
    print(f"{'MAE (ppm), months ahead':>36} " + " ".join(f"{k:>7}" for k in hs))
    for k, v in E.items():
        print(f"{k:>36} " + " ".join(f"{np.mean(np.abs(v[:, j - 1])):7.3f}" for j in hs))
    for k, v in C.items():
        print(f"{'95% coverage, ' + k:>36} " + " ".join(f"{v[:, j - 1].mean():7.3f}" for j in hs) + f"   lowest {v.mean(axis=0).min():.3f}")
    hw, ar = np.mean(np.abs(E[best + " (AIC)"]), axis=0), np.mean(np.abs(E["airline ARIMA, CSS"]), axis=0)
    print(f"Holt-Winters ({best.split(', ')[1]}) is more accurate than the airline model at {np.sum(hw < ar)} of 24 horizons "
          f"(from {np.argmax(hw < ar) + 1} months ahead on); its intervals are too narrow, the airline model's are not")
    print("Actual output from this lesson's run.")
    return E, C


def demo_orders(te, ye):
    print("\n" + "=" * 70)
    print("6. ELECTRICAL-EQUIPMENT ORDERS -- log scale, so the season adds")
    print("=" * 70)
    le = np.log(ye); first, last = (int(np.flatnonzero(te == np.datetime64(m))[0]) for m in ("2001-01", "2007-01"))
    E = {"Holt-Winters, additive": [], "airline ARIMA, CSS": [], "same month last year": []}
    for o in range(first, last + 1):                                                  # origins 2001-01 to 2007-01, 12 months ahead
        act = le[o:o + 12]; pe = lambda f: np.abs(np.exp(f - act) - 1)              # absolute percentage error
        E["Holt-Winters, additive"].append(pe(ets_forecast(fit_ets(le[:o], "add", "add", False), 12)))
        E["airline ARIMA, CSS"].append(pe(airline(le[:o], 12)[0])); E["same month last year"].append(pe(le[o - 12:o]))
    print(f"rolling origins {te[first]} to {te[last]} ({last - first + 1}), mean absolute % error by months ahead:")
    print(f"{'':>24} {'1':>7} {'6':>7} {'12':>7}")
    for k, v in E.items():
        v = np.array(v); print(f"{k:>24} " + " ".join(f"{100 * v[:, j - 1].mean():6.2f}%" for j in (1, 6, 12)))
    keep = {}
    for name, cut in (("the crash", int(np.flatnonzero(te == np.datetime64("2008-01"))[0])), ("the last 24 months", len(le) - 24)):
        fits = {d: fit_ets(le[:cut], "add", "add", d) for d in (False, True)}          # additive, undamped and damped
        d = min(fits, key=lambda k: aic(fits[k])); F = fits[d]; act = le[cut:cut + 24]  # AIC's choice on the training months
        f = ets_forecast(F, 24); sn = le[cut - 12:cut][np.arange(24) % 12]; ar = airline(le[:cut], 24)[0]  # and Day 149's airline model
        print(f"{name}: train up to {te[cut - 1]}; AIC chose {'damped' if d else 'undamped'} (alpha {F['alpha']:.3f}, beta {F['beta']:.3f}, "
              f"gamma {F['gamma']:.3f}, phi {F['phi']:.3f})")
        line = (f"  MAE: Holt-Winters {100 * np.mean(np.abs(np.exp(f - act) - 1)):.2f}%, airline ARIMA {100 * np.mean(np.abs(np.exp(ar - act) - 1)):.2f}%, "
                f"same month last year {100 * np.mean(np.abs(np.exp(sn - act) - 1)):.2f}%")
        if not d:
            line += f"; outside the 95% interval: {np.sum(np.abs(f - act) > 1.96 * ets_sd(F, 24))} of 24"
        print(line); keep[name] = (cut, f)
    print("in the ordinary period the airline model beats Holt-Winters at every horizon shown; through the crash Holt-Winters (damped,")
    print("as AIC chose) was much closer than the airline model, but neither beat last year's value in the crash or in the last two years")
    print("Actual output from this lesson's run.")
    return keep


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(tc, yc, te, ye, fits, rows, roll, keep, out):
    fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0]); k = np.arange(24)                                  # SES weights
    for a, c in zip((0.2, 0.5, 0.8), ("#4C72B0", "#55A868", "#C44E52")):
        ax.bar(k + (a - 0.5) * 0.9, a * (1 - a) ** k, width=0.27, color=c, label=f"alpha {a}")
    ax.set_xlabel("months back"); ax.set_ylabel("weight in the level"); ax.legend(fontsize=8)
    ax.set_title("SES: each month's weight decays geometrically", fontsize=10)
    n = split(tc); ax = fig.add_subplot(gs[0, 1]); x = tc.astype("datetime64[D]"); s = n - 36
    ax.plot(x[s:], yc[s:], color="k", lw=1.2, label="actual")
    for name, c in (("airline ARIMA, CSS (Day 149)", "#8172B2"), ("Holt-Winters, additive", "#55A868"), ("Holt-Winters, multiplicative", "#4C72B0")):
        ax.plot(x[n:], dict(rows)[name], color=c, lw=1.2, label=name)
    ax.axvline(x[n], color="gray", ls=":"); ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("2000-2001 forecasts from data up to 1999-12", fontsize=10)
    ax = fig.add_subplot(gs[0, 2]); F = fits["Holt-Winters, additive"]; cal = tc[:n].astype(int) % 12  # the season as it evolved
    for mo, c in ((4, "#C44E52"), (9, "#4C72B0")):
        ax.plot(x[:n][cal == mo], F["hist"][cal == mo], color=c, label=("May" if mo == 4 else "October") + " seasonal value")
    ax.axhline(0, color="gray", lw=0.5); ax.set_ylabel("ppm above / below the level"); ax.legend(fontsize=8)
    ax.set_title("Holt-Winters lets the season drift (Day 147: it grew)", fontsize=10)
    E, C = roll; ax = fig.add_subplot(gs[1, 0]); hz = np.arange(1, 25)
    for (name, v), c in zip(E.items(), ("#4C72B0", "#55A868", "#8172B2", "#DD8452")):
        ax.plot(hz, np.mean(np.abs(v), axis=0), marker="o", ms=3, color=c, label=name)
    ax.set_xlabel("months ahead"); ax.set_ylabel("MAE (ppm)"); ax.legend(fontsize=7); ax.set_title("97 rolling origins: error by horizon", fontsize=10)
    ax = fig.add_subplot(gs[1, 1])
    for (name, v), c in zip(C.items(), ("#55A868", "#8172B2")):
        ax.plot(hz, v.mean(axis=0), marker="o", ms=3, color=c, label=name)
    ax.axhline(0.95, color="gray", ls="--"); ax.set_ylim(0.7, 1.02); ax.set_xlabel("months ahead"); ax.set_ylabel("share inside")
    ax.legend(fontsize=8); ax.set_title("95% intervals: how often the truth fell inside", fontsize=10)
    ax = fig.add_subplot(gs[1, 2]); x = te.astype("datetime64[D]"); cut, f = keep["the crash"]; s = cut - 48
    ax.plot(x[s:cut + 30], ye[s:cut + 30], color="k", lw=1.2, label="actual orders")
    ax.plot(x[cut:cut + 24], np.exp(f), color="#55A868", label="Holt-Winters from 2007-12")
    ax.axvline(x[cut], color="gray", ls=":"); ax.set_ylabel("orders (index)"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("the 2008-09 crash: smoothing does not foresee it either", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # one step of Holt-Winters
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 15); ax.set_ylim(0.2, 5.0); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.6, 3.7, 2.6, 1.1, "last month's states:\nlevel l, trend b,\nseason s (12 values)", "#DDEBF7")
    box(5.0, 3.7, 2.6, 1.1, "forecast this month:\nl + phi b + s\n(or (l + phi b) x s)", "#E2F0D9")
    box(8.4, 3.7, 2.4, 1.1, "this month arrives:\ny; error =\ny - forecast", "#FFF2CC")
    box(12.4, 4.2, 3.6, 0.7, "level: alpha of the way to y - s", "#FCE4D6")
    box(12.4, 3.2, 3.6, 0.7, "trend: beta of the way to the level's change", "#FCE4D6")
    box(12.4, 2.2, 3.6, 0.7, "season: gamma of the way to y - (l + phi b)", "#FCE4D6")
    box(7.0, 2.0, 5.4, 0.8, "after the last month: forecast h ahead = l + (phi + ... + phi^h) b + s for that month", "#E2F0D9")
    arrow(2.9, 3.7, 3.7, 3.7); arrow(6.3, 3.7, 7.2, 3.7); arrow(9.6, 3.9, 10.6, 4.2); arrow(9.6, 3.7, 10.6, 3.2); arrow(9.6, 3.5, 10.6, 2.3)
    ax.plot([12.4, 12.4, 1.6, 1.6], [1.85, 0.8, 0.8, 1.2], color="black", lw=1.3); ax.annotate("", xy=(1.6, 3.15), xytext=(1.6, 1.2), arrowprops=dict(arrowstyle="->", lw=1.3, shrinkA=0))  # back to the start
    ax.text(7.0, 0.9, "next month: the updated states become last month's states", ha="center", fontsize=8.5)
    arrow(2.4, 3.15, 4.3, 2.4)                                                        # after the last month, the states give the forecasts
    ax.text(7.5, 4.75, "alpha, beta, gamma (and phi) are fitted on the training months; small = slow to change, large = follows recent months",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Day 150: one month of Holt-Winters exponential smoothing", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "exponential_smoothing_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 151: feature engineering for time series -- lags, rolling statistics and calendar features turn forecasting")
    print("into ordinary regression, with a demonstration of how easily future information leaks into the features.")


def main():                                             # runs every section in order
    tc, yc, _ = d147.load_co2(); te, ye = d147.load_elec()                           # Day 147's two series
    demo_ses()                                                                        # 1. SES
    demo_check(tc, yc)                                                                # 2. against statsmodels
    fits, best = demo_choose(tc, yc)                                                  # 3. choosing the variant
    rows = demo_forecast(tc, yc, fits, best)                                          # 4. the 2000-2001 forecast
    roll = demo_rolling(tc, yc, best)                                                 # 5. rolling origins
    keep = demo_orders(te, ye)                                                        # 6. the orders
    plot_results(tc, yc, te, ye, fits, rows, roll, keep, os.path.join(HERE, "exponential_smoothing_results.png"))
    draw_diagram()                                                                    # one step
    print("\nsaved exponential_smoothing_results.png\nsaved exponential_smoothing_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
