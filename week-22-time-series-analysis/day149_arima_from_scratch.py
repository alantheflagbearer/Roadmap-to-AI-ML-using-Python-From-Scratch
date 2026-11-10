"""
Day 149: ARIMA from Scratch -- Fitting, Choosing and Forecasting on Real Data

Day 148 made CO2 stationary by differencing it twice: once at lag 1 (the trend)
and once at lag 12 (the season). The differenced series still has memory -- its
autocorrelation one month apart is -0.297 and twelve months apart -0.409 (Day
148, whole series) -- and
memory is what a forecast can use.

An ARMA model describes that memory with two kinds of term. Autoregressive (AR)
terms make each value depend on earlier values; moving-average (MA) terms make
it depend on earlier surprises (errors). ARIMA is the same model applied to a
differenced series, with the differences undone when forecasting; the seasonal
version adds terms twelve months apart.

Everything is written from scratch in NumPy: the partial autocorrelation, the
model's error recursion, a Levenberg-Marquardt fit by conditional least squares
(CSS), the forecasts, their intervals and the Ljung-Box check. statsmodels is
used only as a reference: its exact-likelihood fit (MLE) and its Kalman-filter
forecasts are what ours are checked against.

  1. AR and MA signatures: what the ACF and PACF show, on simulated series.
  2. Fitting by conditional least squares, on a simulated ARMA(1,1).
  3. Choosing the model for CO2: ACF, PACF, AIC on the training years, Ljung-Box.
  4. Forecasting 2000-2001, against Day 147's simple forecasts.
  5. CSS against exact likelihood: where the two fits differ, and why.
  6. Rolling origins: 97 forecasts, accuracy and interval coverage by horizon.
  7. Electrical-equipment orders: an ordinary period, and the 2008-09 crash.

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
from scipy.optimize import least_squares                # a library least-squares solver, to check ours
from scipy.signal import lfilter                        # a library recursive filter, to check the error recursion
from scipy.stats import chi2                            # the chi-squared tail, for the Ljung-Box p-value
from statsmodels.stats.diagnostic import acorr_ljungbox  # the library's Ljung-Box test, to check ours
from statsmodels.tsa.arima.model import ARIMA           # the library's ARIMA (exact likelihood), to check ours
from statsmodels.tsa.statespace.sarimax import SARIMAX  # the library's seasonal ARIMA (exact likelihood)
from statsmodels.tsa.stattools import pacf as sm_pacf   # the library's partial autocorrelation, to check ours
warnings.filterwarnings("ignore")                       # keep the output to this lesson's own lines

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts
with contextlib.redirect_stdout(io.StringIO()):         # the exercises print their own checks when imported; keep them quiet
    import day147_exercise as d147                      # Day 147: load_co2, load_elec, classical_decompose
    import day148_exercise as d148                      # Day 148: acf, difference
AIRLINE = (0, 1, 0, 1, 12)                              # (p, q, P, Q, s): one MA term at lag 1 and one at lag 12, the 'airline' model


# =============================================================================
# THE PARTS -- NEW (copied unchanged into day149_exercise.py)
# =============================================================================
def pacf(x, nlags):                                     # NEW: the partial autocorrelation, by the Durbin-Levinson recursion
    r = d148.acf(x, nlags); out = [1.0]; phi = np.zeros(0)                            # Day 148's autocorrelation; phi: the AR fit of the current order
    for k in range(1, nlags + 1):                                                     # one more lag each time
        a = (r[k] - phi @ r[k - 1:0:-1]) / (1 - phi @ r[1:k])                         # the new last coefficient: lag k's own contribution
        phi = np.r_[phi - a * phi[::-1], a]; out.append(a)                            # update the earlier coefficients, keep the new one
    return np.array(out)


def sarima_polys(params, order):                        # NEW: the AR and MA lag polynomials of a seasonal ARMA
    p, q, P, Q, s = order                                                             # orders, and the season length
    phi, theta = params[:p], params[p:p + q]                                          # the ordinary AR and MA coefficients
    PHI, THETA = params[p + q:p + q + P], params[p + q + P:p + q + P + Q]             # the seasonal ones
    sa = np.zeros(s * P + 1); sa[0] = 1; sa[s::s] = -PHI                              # 1 - PHI_1 B^s - ...
    sb = np.zeros(s * Q + 1); sb[0] = 1; sb[s::s] = THETA                             # 1 + THETA_1 B^s + ...
    return np.convolve(np.r_[1.0, -phi], sa), np.convolve(np.r_[1.0, theta], sb)     # multiplying polynomials = convolving coefficients


def css_residuals(w, a, b, start=None):                 # NEW: the one-step errors e_t, with the errors before the start set to 0
    r = len(a) - 1 if start is None else start                                        # the first row with enough history for the AR side
    u = np.convolve(w, a)[r:len(w)]                                                   # the AR side, a(B) w_t, for every row from r on
    q = len(b) - 1; lags = np.flatnonzero(b[1:]) + 1; bl = b[lags]                    # the MA lags that are not zero, and their coefficients
    e = np.zeros(q + len(u))                                                          # q zeros in front: the unknown errors before the start
    for t in range(len(u)):                                                           # in time order: each error needs the earlier ones
        e[q + t] = u[t] - bl @ e[q + t - lags]                                        # e_t = a(B) w_t - (b_1 e_(t-1) + ...)
    return e[q:]


def levenberg_marquardt(fun, x0, iters=200, tol=1e-12):  # NEW: minimise the sum of squares of fun(x), with numerical derivatives
    x = np.array(x0, float); r = fun(x); ssr = r @ r; lam = 1e-3                      # start; lam: how cautious the step is
    for _ in range(iters):
        J = np.empty((len(r), len(x)))                                                # the Jacobian: how each residual moves with each parameter
        for j in range(len(x)):
            dx = np.zeros(len(x)); dx[j] = 1e-6 * max(1.0, abs(x[j]))                 # a small nudge to one parameter
            J[:, j] = (fun(x + dx) - r) / dx[j]                                       # forward difference
        g, H = J.T @ r, J.T @ J                                                       # the gradient (up to 2) and the Gauss-Newton curvature
        while True:
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)        # Gauss-Newton when lam is small, short gradient steps when large
            r_new = fun(x + step); ssr_new = r_new @ r_new
            if ssr_new < ssr:                                                         # better: take the step
                break
            lam *= 10                                                                 # worse: be more cautious and try again
            if lam > 1e10:                                                            # no step helps: at a minimum
                return x, ssr
        x, r, lam, gain, ssr = x + step, r_new, lam / 10, ssr - ssr_new, ssr_new      # accept, and be bolder next time
        if gain < tol * ssr:                                                          # the improvement has become negligible
            break
    return x, ssr


def fit_css(w, order, start=None):                      # NEW: fit a seasonal ARMA to a stationary series by conditional least squares
    def res(th):                                                                      # the errors, for a given parameter vector
        return css_residuals(w, *sarima_polys(th, order), start=start)
    x, ssr = levenberg_marquardt(res, np.full(sum(order[:4]), 0.1))                   # every coefficient starts at 0.1
    m = len(res(x))                                                                   # the rows the sum covers
    return x, ssr / m, m                                                              # coefficients, error variance, rows


def full_ar(a, d=1, D=1, s=12):                         # NEW: the AR polynomial times the differences, so the model describes y itself
    for _ in range(d):
        a = np.convolve(a, [1.0, -1.0])                                               # times (1 - B): one ordinary difference
    for _ in range(D):
        a = np.convolve(a, np.r_[1.0, np.zeros(s - 1), -1.0])                         # times (1 - B^s): one seasonal difference
    return a


def forecast(y, A, b, e, h):                            # NEW: h steps ahead; future errors are 0, their expected value
    n = len(y); x = np.r_[y, np.zeros(h)]                                             # the history, with room for the forecasts
    E = np.r_[np.zeros(n - len(e)), e, np.zeros(h)]                                   # the errors, lined up with y (0 before they start, and ahead)
    for t in range(n, n + h):                                                         # one step at a time: each forecast feeds the next
        x[t] = -A[1:] @ x[t - 1:t - len(A):-1] + b[1:] @ E[t - 1:t - len(b):-1]       # A(B) y_t = b(B) e_t, solved for y_t
    return x[n:]


def psi_weights(A, b, h):                               # NEW: how much each future shock adds to the forecast error
    p = np.zeros(h)                                                                   # psi_0 = 1, psi_1, ...: b(B) / A(B) as a power series
    for j in range(h):
        k = min(j, len(A) - 1)                                                        # how many earlier weights enter
        p[j] = (b[j] if j < len(b) else 0.0) - A[1:k + 1] @ p[j - k:j][::-1]          # match the coefficients of B^j
    return p


def ljung_box(e, lags, fitted=0):                       # NEW: the Ljung-Box statistic: is there autocorrelation left?
    r = d148.acf(e, lags); n = len(e)                                                 # the residuals' autocorrelations
    return n * (n + 2) * np.sum(r[1:] ** 2 / (n - np.arange(1, lags + 1))), lags - fitted  # Q, and its degrees of freedom


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def simulate_arma(phi, theta, n=500, seed=0, burn=100):  # an ARMA series with known coefficients
    e = np.random.RandomState(seed).standard_normal(n + burn); x = np.zeros(n + burn)  # the shocks, and room for the series
    for t in range(max(len(phi), len(theta)), n + burn):                              # skip the first few, which lack history
        x[t] = (sum(f * x[t - i - 1] for i, f in enumerate(phi)) + e[t]
                + sum(g * e[t - j - 1] for j, g in enumerate(theta)))                 # AR terms + this shock + MA terms
    return x[burn:]                                                                   # the start-up stretch dropped


def train_split(tc):                                    # Day 147's split: train up to 1999-12, forecast the last 24 months
    return int(np.flatnonzero(tc == np.datetime64("2000-01"))[0])


def demo_signatures():
    print("=" * 70)
    print("1. AR AND MA SIGNATURES -- the ACF and the PACF of two simulated series")
    print("=" * 70)
    ar2 = simulate_arma([0.5, 0.3], [], seed=1); ma1 = simulate_arma([], [0.7], seed=2)  # AR(2) and MA(1), 500 steps each
    print("PACF at lag k: the correlation k steps apart after removing what lags 1 ... k-1 already explain")
    print(f"{'series':>34} {'lag 1':>7} {'lag 2':>7} {'lag 3':>7} {'lag 4':>7} {'lag 5':>7}")
    for name, x in (("AR(2): y = 0.5 y(-1) + 0.3 y(-2) + e", ar2), ("MA(1): y = e + 0.7 e(-1)", ma1)):
        for kind, f in (("ACF", d148.acf), ("PACF", pacf)):
            v = f(x, 5)
            print(f"{name + ', ' + kind:>34} " + " ".join(f"{z:7.3f}" for z in v[1:]))
    print(f"theory: AR(2) ACF lag 1 = 0.5 / (1 - 0.3) = {0.5 / 0.7:.3f}, PACF lag 2 = 0.3, PACF beyond 2 = 0;"
          f" MA(1) ACF lag 1 = 0.7 / (1 + 0.49) = {0.7 / 1.49:.3f}, ACF beyond 1 = 0")
    err = max(np.max(np.abs(pacf(x, 24) - sm_pacf(x, nlags=24, method="ldb"))) for x in (ar2, ma1))
    print(f"largest difference from statsmodels' pacf (Levinson-Durbin), lags 0-24: {err:.1e}")
    print(f"band for 'no correlation' at 500 values: +-{1.96 / np.sqrt(500):.3f}; the AR(2)'s PACF stops after lag 2 while its ACF fades;")
    print("the MA(1)'s ACF stops after lag 1 while its PACF fades with alternating signs -- the two signatures used to read real data")
    print("Actual output from this lesson's run.")
    return ar2, ma1


def demo_fit():
    print("\n" + "=" * 70)
    print("2. FITTING BY CONDITIONAL LEAST SQUARES -- a simulated ARMA(1,1)")
    print("=" * 70)
    x = simulate_arma([0.6], [0.3], seed=0); order = (1, 1, 0, 0, 12)                 # true phi 0.6, theta 0.3; no seasonal terms
    a, b = sarima_polys(np.array([0.6, 0.3]), order)                                  # the true model's polynomials
    ref = lfilter([1.0], b, np.convolve(x, a)[1:len(x)])                              # the same recursion by scipy's filter
    print(f"error recursion against scipy's lfilter, at the true coefficients: largest difference {np.max(np.abs(css_residuals(x, a, b) - ref)):.1e}")
    t0 = time.perf_counter(); th, s2, m = fit_css(x, order); dt = time.perf_counter() - t0  # ours
    ls = least_squares(lambda v: css_residuals(x, *sarima_polys(v, order)), [0.1, 0.1], xtol=1e-15, ftol=1e-15, gtol=1e-15).x  # scipy's solver, same errors
    ml = ARIMA(x, order=(1, 0, 1), trend="n").fit().params                            # statsmodels' exact likelihood
    print(f"{'':>34} {'phi':>8} {'theta':>8} {'error variance':>15}")
    print(f"{'truth':>34} {0.6:8.3f} {0.3:8.3f} {1.0:15.3f}")
    print(f"{'ours: CSS, Levenberg-Marquardt':>34} {th[0]:8.4f} {th[1]:8.4f} {s2:15.4f}   ({1000 * dt:.0f} ms)")
    print(f"{'scipy least_squares, same errors':>34} {ls[0]:8.4f} {ls[1]:8.4f}")
    print(f"{'statsmodels ARIMA, exact MLE':>34} {ml[0]:8.4f} {ml[1]:8.4f} {ml[2]:15.4f}")
    print(f"ours against scipy's solver: largest difference {np.max(np.abs(th - ls)):.1e}")
    print("Actual output from this lesson's run.")


def demo_choose(tc, yc):
    print("\n" + "=" * 70)
    print("3. CHOOSING THE MODEL FOR CO2 -- ACF, PACF and AIC on the training years only")
    print("=" * 70)
    n = train_split(tc); w = d148.difference(d148.difference(yc[:n], 12))             # d1 d12 of the training months
    r, pr = d148.acf(w, 36), pacf(w, 36); band = 1.96 / np.sqrt(len(w))               # and the rough 95% band for 'no correlation'
    print(f"CO2 d1 d12, training months up to {tc[n - 1]} ({len(w)} values); band for no correlation: +-{band:.3f}")
    ks = (1, 2, 3, 11, 12, 13, 24, 36)
    print(f"{'lag':>6} " + " ".join(f"{k:7d}" for k in ks))
    print(f"{'ACF':>6} " + " ".join(f"{r[k]:7.3f}" for k in ks))
    print(f"{'PACF':>6} " + " ".join(f"{pr[k]:7.3f}" for k in ks))
    t0 = time.perf_counter(); grid = []
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    _, s2, m = fit_css(w, (p, q, P, Q, 12), start=14)                 # every model on the same rows, so AIC compares fairly
                    grid.append((m * np.log(s2) + 2 * (p + q + P + Q), (p, q, P, Q), s2))
    grid.sort(key=lambda g: g[0])                                                     # smallest AIC first
    print(f"AIC = rows x log(error variance) + 2 x coefficients, {len(grid)} models (p, q 0-2; P, Q 0-1), "
          f"rows from 14 on ({time.perf_counter() - t0:.2f} s):")
    for aic, o, s2 in grid[:5]:
        print(f"  (p, q, P, Q) = {o}   AIC {aic:9.2f}   error variance {s2:.4f}")
    worst = grid[-1]; print(f"  ... worst: {worst[1]}   AIC {worst[0]:9.2f}   error variance {worst[2]:.4f}")
    th, s2, m = fit_css(w, AIRLINE); e = css_residuals(w, *sarima_polys(th, AIRLINE))  # the chosen model on every row
    Q, df = ljung_box(e, 24, fitted=2); ref = acorr_ljungbox(e, lags=[24], model_df=2)
    Q0, df0 = ljung_box(w, 24)                                                        # the differenced series itself, before any model
    print(f"the airline model itself implies ACF at lags 11 and 13 of theta THETA / ((1 + theta^2)(1 + THETA^2)) = "
          f"{th[0] * th[1] / ((1 + th[0] ** 2) * (1 + th[1] ** 2)):.3f} with the fitted values; lag 3 ({r[3]:.3f}) is outside the band and not explained by it")
    print(f"the next four models add a coefficient each and lower the error variance by at most {grid[0][2] - min(g[2] for g in grid[1:5]):.4f}")
    print(f"Ljung-Box, 24 lags: residuals Q {Q:.2f}, {df} df, p {chi2.sf(Q, df):.3f} (statsmodels: Q {ref['lb_stat'].iloc[0]:.2f}, "
          f"p {ref['lb_pvalue'].iloc[0]:.3f}); d1 d12 before modelling Q {Q0:.2f}, p {chi2.sf(Q0, df0):.1e}")
    print("Actual output from this lesson's run.")
    return w, r, pr, band


def demo_forecast(tc, yc):
    print("\n" + "=" * 70)
    print("4. FORECASTING 2000-2001 -- against Day 147's simple forecasts")
    print("=" * 70)
    n = train_split(tc); h = len(yc) - n; w = d148.difference(d148.difference(yc[:n], 12))
    th, s2, m = fit_css(w, AIRLINE); a, b = sarima_polys(th, AIRLINE); e = css_residuals(w, a, b)
    A = full_ar(a); f = forecast(yc[:n], A, b, e, h); sd = np.sqrt(s2 * np.cumsum(psi_weights(A, b, h) ** 2))
    print(f"airline model, CSS on {m} rows: theta {th[0]:.4f}, THETA (lag 12) {th[1]:.4f}, error variance {s2:.4f}")
    fw = forecast(w, a, b, e, h); z = np.r_[np.diff(yc[:n]), np.zeros(h)]             # forecast the differenced series instead ...
    for k in range(h):
        z[n - 1 + k] = fw[k] + z[n - 1 + k - 12]                                      # ... undo the seasonal difference ...
    print(f"checks: the model on y against forecasting d1 d12 and undoing both differences by hand: {np.max(np.abs(yc[n - 1] + np.cumsum(z[-h:]) - f)):.1e}")
    kal = SARIMAX(yc[:n], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).filter(np.r_[th, s2])  # statsmodels' Kalman filter, our coefficients
    print(f"        statsmodels' forecast with our coefficients: {np.max(np.abs(kal.forecast(h) - f)):.1e}; "
          f"interval widths: {np.max(np.abs(kal.get_forecast(h).se_mean - sd)):.1e}")
    ml = SARIMAX(yc[:n], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)  # statsmodels' own fit
    T, S, _ = d147.classical_decompose(yc[:n]); ok = np.flatnonzero(~np.isnan(T))[-120:]  # Day 147's best: a line through 120 trend values
    sl, ic = np.polyfit(ok, T[ok], 1); steps = np.arange(n, n + h)
    rows = (("last value (Day 147)", np.full(h, yc[n - 1])), ("same month last year (Day 147)", yc[n - 12:n][np.arange(h) % 12]),
            ("trend line, 120 months + season (Day 147)", sl * steps + ic + S[:12][steps % 12]),
            ("airline, ours (CSS)", f), ("airline, statsmodels (MLE)", ml.forecast(h)))
    print(f"{tc[n]} to {tc[-1]}, {h} months       MAE (ppm)   bias")
    for name, g in rows:
        print(f"  {name:<42} {np.mean(np.abs(g - yc[n:])):6.3f} {np.mean(g - yc[n:]):+7.3f}")
    inside = np.abs(f - yc[n:]) < 1.96 * sd
    print(f"our 95% interval: half-width {1.96 * sd[0]:.2f} ppm one month ahead, {1.96 * sd[11]:.2f} at 12, {1.96 * sd[-1]:.2f} at 24; "
          f"{inside.sum()} of {h} months inside")
    print(f"both airline fits beat Day 147's best; both overshoot: ours is above the actual value in {np.sum(f > yc[n:])} of {h} months")
    print(f"MLE coefficients: theta {ml.params[0]:.4f}, THETA {ml.params[1]:.4f}, error variance {ml.params[2]:.4f}")
    print("Actual output from this lesson's run.")
    return n, f, sd, rows


def demo_css_vs_mle(tc, yc):
    print("\n" + "=" * 70)
    print("5. CSS AGAINST EXACT LIKELIHOOD -- where the two fits differ")
    print("=" * 70)
    n = train_split(tc); w = d148.difference(d148.difference(yc[:n], 12))
    e = css_residuals(w, *sarima_polys(fit_css(w, AIRLINE)[0], AIRLINE))
    print(f"CSS starts with the unknown earlier errors set to 0; on CO2 the first 13 errors have mean square {np.mean(e[:13] ** 2):.4f}, "
          f"the rest {np.mean(e[13:] ** 2):.4f}")
    print(f"{'CO2, airline model':>36} {'theta':>8} {'THETA':>8}")
    for burn in (0, 13, 26, 39):                                                      # leave the first rows' errors out of the sum
        def res(v, burn=burn):
            return css_residuals(w, *sarima_polys(v, AIRLINE))[burn:]
        v = levenberg_marquardt(res, [0.1, 0.1])[0]
        if burn == 0:
            v0 = v                                                                    # keep the plain fit for the comparison below
        print(f"{'CSS, first ' + str(burn) + ' errors left out':>36} {v[0]:8.4f} {v[1]:8.4f}")
    ml = SARIMAX(yc[:n], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).params
    print(f"{'statsmodels, exact MLE':>36} {ml[0]:8.4f} {ml[1]:8.4f}")
    print(f"leaving out early errors moves THETA only to {v[1]:.4f}; the gap to MLE on CO2 is {v0[1] - ml[1]:.3f} with none left out")
    sims = []
    for seed in range(40):                                                            # simulated d1 d12 series with known coefficients
        z = np.random.RandomState(seed).standard_normal(len(w) + 13)
        sv = z[13:] - 0.35 * z[12:-1] - 0.85 * z[1:-12] + 0.35 * 0.85 * z[:-13]         # (1 - 0.35 B)(1 - 0.85 B^12) e
        mm = SARIMAX(sv, order=(0, 0, 1), seasonal_order=(0, 0, 1, 12), trend="n").fit(disp=False).params[:2]
        sims.append(np.r_[fit_css(sv, AIRLINE)[0], mm])
    sims = np.array(sims)
    print(f"40 simulated series of {len(w)} values, true theta -0.35 and THETA -0.85:")
    print(f"  mean CSS {sims[:, 0].mean():.4f}, {sims[:, 1].mean():.4f};  mean MLE {sims[:, 2].mean():.4f}, {sims[:, 3].mean():.4f};  "
          f"CSS's THETA closer to 0 than MLE's in {np.sum(sims[:, 1] > sims[:, 3])} of 40")
    print(f"so CSS does pull THETA towards 0, by {sims[:, 1].mean() - sims[:, 3].mean():.3f} on average here: less than CO2's gap; "
          "the rest is not explained here")
    print("Actual output from this lesson's run.")
    return sims


def demo_rolling(tc, yc):
    print("\n" + "=" * 70)
    print("6. ROLLING ORIGINS -- forecasting 24 months ahead from 97 starting months, 1990-1998")
    print("=" * 70)
    n = train_split(tc); first = int(np.flatnonzero(tc == np.datetime64("1990-01"))[0])
    origins = range(first, n - 24 + 1); E = {"airline, ours (CSS)": [], "airline, statsmodels (MLE)": [], "same month last year": []}; cover = []
    t0 = time.perf_counter()
    for o in origins:                                                                 # each origin: train on everything before it
        w = d148.difference(d148.difference(yc[:o], 12)); th, s2, _ = fit_css(w, AIRLINE); a, b = sarima_polys(th, AIRLINE)
        A = full_ar(a); f = forecast(yc[:o], A, b, css_residuals(w, a, b), 24); act = yc[o:o + 24]
        E["airline, ours (CSS)"].append(f - act); cover.append(np.abs(f - act) < 1.96 * np.sqrt(s2 * np.cumsum(psi_weights(A, b, 24) ** 2)))
        E["same month last year"].append(yc[o - 12:o][np.arange(24) % 12] - act)
    t1 = time.perf_counter()
    for o in origins:
        g = SARIMAX(yc[:o], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(24)
        E["airline, statsmodels (MLE)"].append(g - yc[o:o + 24])
    t2 = time.perf_counter(); E = {k: np.array(v) for k, v in E.items()}; cover = np.array(cover)
    print(f"{len(origins)} origins, {tc[first]} to {tc[n - 24]}; every forecast ends by {tc[n - 1]}, so the 2000-2001 test months are untouched")
    print(f"time: CSS {t1 - t0:.1f} s for all fits, statsmodels {t2 - t1:.1f} s")
    hs = (1, 6, 12, 24)
    print(f"{'MAE (ppm), months ahead':>30} " + " ".join(f"{k:>7}" for k in hs) + "   bias at 24")
    for k, v in E.items():
        print(f"{k:>30} " + " ".join(f"{np.mean(np.abs(v[:, j - 1])):7.3f}" for j in hs) + f"   {np.mean(v[:, -1]):+.3f}")
    print(f"{'95% interval coverage (ours)':>30} " + " ".join(f"{cover[:, j - 1].mean():7.3f}" for j in hs))
    lo = np.argmin(cover.mean(axis=0))                                                # the horizon where the intervals did worst
    print(f"lowest coverage over all 24 horizons: {cover[:, lo].mean():.3f}, {lo + 1} months ahead")
    better = np.sum(np.mean(np.abs(E["airline, statsmodels (MLE)"]), 0) < np.mean(np.abs(E["airline, ours (CSS)"]), 0))
    print(f"MLE is more accurate than CSS at {better} of 24 horizons, by little; both are far ahead of last year's value, which misses the rise")
    print("Actual output from this lesson's run.")
    return E, cover


def demo_orders(te, ye):
    print("\n" + "=" * 70)
    print("7. ELECTRICAL-EQUIPMENT ORDERS -- an ordinary period, and the 2008-09 crash")
    print("=" * 70)
    le = np.log(ye); D = d148.difference                                              # log scale: errors become percentages
    first, last = (int(np.flatnonzero(te == np.datetime64(m))[0]) for m in ("2001-01", "2007-01"))
    E = {"airline (CSS)": [], "same month last year": [], "last value": []}
    for o in range(first, last + 1):                                                  # origins 2001-01 to 2007-01, 12 months ahead
        w = D(D(le[:o], 12)); th, s2, _ = fit_css(w, AIRLINE); a, b = sarima_polys(th, AIRLINE)
        f = forecast(le[:o], full_ar(a), b, css_residuals(w, a, b), 12); act = le[o:o + 12]
        E["airline (CSS)"].append(np.abs(np.exp(f - act) - 1)); E["same month last year"].append(np.abs(np.exp(le[o - 12:o] - act) - 1))
        E["last value"].append(np.abs(np.exp(le[o - 1] - act) - 1))                   # absolute percentage errors
    print(f"rolling origins {te[first]} to {te[last]} ({last - first + 1}), log orders, mean absolute % error by months ahead:")
    print(f"{'':>22} {'1':>7} {'6':>7} {'12':>7}")
    for k, v in E.items():
        v = np.array(v); print(f"{k:>22} " + " ".join(f"{100 * v[:, j - 1].mean():6.2f}%" for j in (1, 6, 12)))
    print("the model wins one month ahead, where recent errors carry information; by 12 months ahead last year's value is better")
    keep = {}
    for name, cut in (("the crash", int(np.flatnonzero(te == np.datetime64("2008-01"))[0])), ("the last 24 months", len(le) - 24)):
        w = D(D(le[:cut], 12)); grid = []
        for p in range(3):
            for q in range(3):
                for P in range(2):
                    for Q in range(2):
                        _, s2, m = fit_css(w, (p, q, P, Q, 12), start=14); grid.append((m * np.log(s2) + 2 * (p + q + P + Q), (p, q, P, Q, 12)))
        best = min(grid, key=lambda g: g[0])[1]                                       # AIC's choice, on the training months only
        act = le[cut:cut + 24]; sn = le[cut - 12:cut][np.arange(24) % 12]
        print(f"{name}: train up to {te[cut - 1]}, forecast {te[cut]} to {te[cut + 23]}; same month last year: {100 * np.mean(np.abs(np.exp(sn - act) - 1)):.2f}%")
        for label, order in (("airline", AIRLINE), (f"AIC's choice {best[:4]}", best)):
            th, s2, _ = fit_css(w, order); a, b = sarima_polys(th, order); A = full_ar(a)
            f = forecast(le[:cut], A, b, css_residuals(w, a, b), 24); sd = np.sqrt(s2 * np.cumsum(psi_weights(A, b, 24) ** 2))
            out = np.abs(f - act) > 1.96 * sd; j = np.argmax(np.abs(f - act))
            print(f"  {label:<28} MAE {100 * np.mean(np.abs(np.exp(f - act) - 1)):5.2f}%   outside the 95% interval: {out.sum():2d} of 24   "
                  f"worst {te[cut + j]}: actual {100 * (np.exp(act[j] - f[j]) - 1):+.1f}% against the forecast")
            keep[(name, label[:7])] = (cut, f, sd)
    print("in the crash no method saw it coming and the intervals were far too narrow; in the quiet last two years last year's")
    print("value still beat both models, and AIC's larger model was no better than the airline model")
    print("Actual output from this lesson's run.")
    return keep


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(tc, yc, te, ye, ar2, ma1, chosen, fc, roll, keep, out):
    fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0]); k = np.arange(1, 11)                                # the two signatures
    for x, name, c in ((ar2, "AR(2)", "#4C72B0"), (ma1, "MA(1)", "#C44E52")):
        ax.plot(k, d148.acf(x, 10)[1:], marker="o", color=c, label=f"{name} ACF")
        ax.plot(k, pacf(x, 10)[1:], marker="s", ls="--", color=c, label=f"{name} PACF")
    ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("lag"); ax.legend(fontsize=8)
    ax.set_title("AR: PACF stops after p.  MA: ACF stops after q", fontsize=10)
    w, r, pr, band = chosen; ax = fig.add_subplot(gs[0, 1]); k = np.arange(1, 37)
    ax.bar(k - 0.2, r[1:], width=0.4, label="ACF"); ax.bar(k + 0.2, pr[1:], width=0.4, label="PACF")
    ax.axhspan(-band, band, color="gray", alpha=0.2); ax.set_xlabel("lag (months)"); ax.legend(fontsize=8)
    ax.set_title("CO2 d1 d12, training years: spikes at lags 1 and 12", fontsize=10)
    n, f, sd, rows = fc; ax = fig.add_subplot(gs[0, 2]); x = tc.astype("datetime64[D]"); s = n - 36  # the last three training years and the test
    ax.plot(x[s:], yc[s:], color="k", lw=1.2, label="actual")
    ax.fill_between(x[n:], f - 1.96 * sd, f + 1.96 * sd, color="#4C72B0", alpha=0.25, label="95% interval")
    for (name, g), c in zip(rows[1:4], ("#DD8452", "#55A868", "#4C72B0")):
        ax.plot(x[n:], g, color=c, lw=1.2, label=name.replace(" (Day 147)", ""))
    ax.axvline(x[n], color="gray", ls=":"); ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("2000-2001 forecasts from data up to 1999-12", fontsize=10)
    E, cover = roll; ax = fig.add_subplot(gs[1, 0]); hz = np.arange(1, 25)
    for (name, v), c in zip(E.items(), ("#4C72B0", "#8172B2", "#DD8452")):
        ax.plot(hz, np.mean(np.abs(v), axis=0), marker="o", ms=3, color=c, label=name)
    ax.set_xlabel("months ahead"); ax.set_ylabel("MAE (ppm)"); ax.legend(fontsize=8); ax.set_title("97 rolling origins, 1990-1998: error by horizon", fontsize=10)
    ax = fig.add_subplot(gs[1, 1]); ax.plot(hz, cover.mean(axis=0), marker="o", ms=3, color="#4C72B0")
    ax.axhline(0.95, color="gray", ls="--"); ax.set_ylim(0.7, 1.02); ax.set_xlabel("months ahead"); ax.set_ylabel("share inside")
    ax.set_title("our 95% intervals: how often the truth fell inside", fontsize=10)
    cut, f, sd = keep[("the crash", "airline")]; ax = fig.add_subplot(gs[1, 2]); x = te.astype("datetime64[D]"); s = cut - 48
    ax.plot(x[s:cut + 30], ye[s:cut + 30], color="k", lw=1.2, label="actual orders")
    ax.plot(x[cut:cut + 24], np.exp(f), color="#4C72B0", label="airline forecast from 2007-12")
    ax.fill_between(x[cut:cut + 24], np.exp(f - 1.96 * sd), np.exp(f + 1.96 * sd), color="#4C72B0", alpha=0.25, label="95% interval")
    ax.axvline(x[cut], color="gray", ls=":"); ax.set_ylabel("orders (index)"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("the 2008-09 crash: outside anything the past suggested", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the ARIMA workflow
    fig, ax = plt.subplots(figsize=(15, 5.6)); ax.set_xlim(0, 15); ax.set_ylim(0.2, 5.2); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, **kw):                                                  # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3, **kw))
    box(1.3, 3.8, 2.0, 1.0, "series y\n(training months only)", "#FFF2CC")
    box(4.2, 3.8, 2.4, 1.0, "difference until\nstationary: d1 d12\n(Day 148)", "#DDEBF7")
    box(7.3, 3.8, 2.4, 1.0, "read ACF and PACF:\nwhich lags have\nmemory?", "#DDEBF7")
    box(10.5, 3.8, 2.6, 1.0, "choose (p, q, P, Q)\nby AIC, all models\non the same rows", "#DDEBF7")
    box(13.5, 3.8, 2.2, 1.0, "fit by CSS:\nerror recursion +\nLevenberg-Marquardt", "#DDEBF7")
    box(13.5, 1.5, 2.2, 1.0, "Ljung-Box on\nthe errors: memory\nleft?", "#FCE4D6")
    box(9.9, 1.5, 3.0, 1.0, "forecast: multiply in\nthe differences, recurse\nwith future errors = 0", "#E2F0D9")
    box(5.9, 1.5, 3.0, 1.0, "intervals from the\npsi weights: they widen\nwith the horizon", "#E2F0D9")
    box(2.2, 1.5, 2.6, 1.0, "check on months the\nfit never saw\n(rolling origins)", "#E2F0D9")
    arrow(2.3, 3.8, 3.0, 3.8); arrow(5.4, 3.8, 6.1, 3.8); arrow(8.5, 3.8, 9.2, 3.8); arrow(11.8, 3.8, 12.4, 3.8)
    arrow(13.5, 3.3, 13.5, 2.0); arrow(12.4, 1.5, 11.4, 1.5); arrow(8.4, 1.5, 7.4, 1.5); arrow(4.4, 1.5, 3.5, 1.5)
    arrow(12.7, 2.0, 10.9, 3.3)
    ax.text(11.75, 1.6, "no", fontsize=8.5); ax.text(12.05, 2.75, "yes: try\nother orders", fontsize=8)
    ax.text(7.5, 4.85, "every choice (differences, orders) is made on the training months; the test months are used once, at the end",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Day 149: the ARIMA workflow", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "arima_from_scratch_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 150: exponential smoothing (simple, Holt, Holt-Winters) from scratch -- forecasts as weighted averages of the past,")
    print("compared with today's ARIMA on the same holdout and the same rolling origins.")


def main():                                             # runs every section in order
    tc, yc, _ = d147.load_co2(); te, ye = d147.load_elec()                           # Day 147's two series
    ar2, ma1 = demo_signatures()                                                      # 1. AR and MA signatures
    demo_fit()                                                                        # 2. fitting by CSS
    chosen = demo_choose(tc, yc)                                                      # 3. choosing the model
    fc = demo_forecast(tc, yc)                                                        # 4. the 2000-2001 forecast
    demo_css_vs_mle(tc, yc)                                                           # 5. CSS against MLE
    roll = demo_rolling(tc, yc)                                                       # 6. rolling origins
    keep = demo_orders(te, ye)                                                        # 7. the orders
    plot_results(tc, yc, te, ye, ar2, ma1, chosen, fc, roll, keep, os.path.join(HERE, "arima_from_scratch_results.png"))
    draw_diagram()                                                                    # the workflow
    print("\nsaved arima_from_scratch_results.png\nsaved arima_from_scratch_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
