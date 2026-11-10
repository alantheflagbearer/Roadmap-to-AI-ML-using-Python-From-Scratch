"""
Day 149 Exercise -- ARIMA from Scratch

The third of Week 22's parts for Day 153's Prophet-style forecaster.

    pacf(x, nlags)                     the partial autocorrelation (Durbin-Levinson)
    sarima_polys(params, order)        the AR and MA lag polynomials of a seasonal ARMA
    css_residuals(w, a, b, start)      the one-step errors, with the errors before the start set to 0
    levenberg_marquardt(fun, x0)       least squares for any residual function
    fit_css(w, order, start)           a seasonal ARMA fitted by conditional least squares
    full_ar(a, d, D, s)                the AR polynomial times the differences
    forecast(y, A, b, e, h)            h steps ahead, future errors set to 0
    psi_weights(A, b, h)               the weights that give the forecast intervals
    ljung_box(e, lags, fitted)         the Ljung-Box statistic and its degrees of freedom

Checked by hand: a partial autocorrelation, the airline model's MA polynomial,
three errors of an MA(1), a straight-line fit, the differencing polynomial, a
random walk's forecast and intervals, and a one-lag Ljung-Box statistic.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported part quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: this file's folder
import sys                                             # sys: path, to import Day 148's part
import numpy as np                                     # numpy: every array operation below is built on this
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this week's parts
with contextlib.redirect_stdout(io.StringIO()):         # Day 148's exercise prints its own checks when imported
    import day148_exercise as d148                      # Day 148: acf, difference


def pacf(x, nlags):                                     # NEW: Day 149's version, unchanged
    r = d148.acf(x, nlags); out = [1.0]; phi = np.zeros(0)                            # Day 148's autocorrelation; phi: the AR fit of the current order
    for k in range(1, nlags + 1):                                                     # one more lag each time
        a = (r[k] - phi @ r[k - 1:0:-1]) / (1 - phi @ r[1:k])                         # the new last coefficient: lag k's own contribution
        phi = np.r_[phi - a * phi[::-1], a]; out.append(a)                            # update the earlier coefficients, keep the new one
    return np.array(out)


def sarima_polys(params, order):                        # NEW: Day 149's version, unchanged
    p, q, P, Q, s = order                                                             # orders, and the season length
    phi, theta = params[:p], params[p:p + q]                                          # the ordinary AR and MA coefficients
    PHI, THETA = params[p + q:p + q + P], params[p + q + P:p + q + P + Q]             # the seasonal ones
    sa = np.zeros(s * P + 1); sa[0] = 1; sa[s::s] = -PHI                              # 1 - PHI_1 B^s - ...
    sb = np.zeros(s * Q + 1); sb[0] = 1; sb[s::s] = THETA                             # 1 + THETA_1 B^s + ...
    return np.convolve(np.r_[1.0, -phi], sa), np.convolve(np.r_[1.0, theta], sb)     # multiplying polynomials = convolving coefficients


def css_residuals(w, a, b, start=None):                 # NEW: Day 149's version, unchanged
    r = len(a) - 1 if start is None else start                                        # the first row with enough history for the AR side
    u = np.convolve(w, a)[r:len(w)]                                                   # the AR side, a(B) w_t, for every row from r on
    q = len(b) - 1; lags = np.flatnonzero(b[1:]) + 1; bl = b[lags]                    # the MA lags that are not zero, and their coefficients
    e = np.zeros(q + len(u))                                                          # q zeros in front: the unknown errors before the start
    for t in range(len(u)):                                                           # in time order: each error needs the earlier ones
        e[q + t] = u[t] - bl @ e[q + t - lags]                                        # e_t = a(B) w_t - (b_1 e_(t-1) + ...)
    return e[q:]


def levenberg_marquardt(fun, x0, iters=200, tol=1e-12):  # NEW: Day 149's version, unchanged
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


def fit_css(w, order, start=None):                      # NEW: Day 149's version, unchanged
    def res(th):                                                                      # the errors, for a given parameter vector
        return css_residuals(w, *sarima_polys(th, order), start=start)
    x, ssr = levenberg_marquardt(res, np.full(sum(order[:4]), 0.1))                   # every coefficient starts at 0.1
    m = len(res(x))                                                                   # the rows the sum covers
    return x, ssr / m, m                                                              # coefficients, error variance, rows


def full_ar(a, d=1, D=1, s=12):                         # NEW: Day 149's version, unchanged
    for _ in range(d):
        a = np.convolve(a, [1.0, -1.0])                                               # times (1 - B): one ordinary difference
    for _ in range(D):
        a = np.convolve(a, np.r_[1.0, np.zeros(s - 1), -1.0])                         # times (1 - B^s): one seasonal difference
    return a


def forecast(y, A, b, e, h):                            # NEW: Day 149's version, unchanged
    n = len(y); x = np.r_[y, np.zeros(h)]                                             # the history, with room for the forecasts
    E = np.r_[np.zeros(n - len(e)), e, np.zeros(h)]                                   # the errors, lined up with y (0 before they start, and ahead)
    for t in range(n, n + h):                                                         # one step at a time: each forecast feeds the next
        x[t] = -A[1:] @ x[t - 1:t - len(A):-1] + b[1:] @ E[t - 1:t - len(b):-1]       # A(B) y_t = b(B) e_t, solved for y_t
    return x[n:]


def psi_weights(A, b, h):                               # NEW: Day 149's version, unchanged
    p = np.zeros(h)                                                                   # psi_0 = 1, psi_1, ...: b(B) / A(B) as a power series
    for j in range(h):
        k = min(j, len(A) - 1)                                                        # how many earlier weights enter
        p[j] = (b[j] if j < len(b) else 0.0) - A[1:k + 1] @ p[j - k:j][::-1]          # match the coefficients of B^j
    return p


def ljung_box(e, lags, fitted=0):                       # NEW: Day 149's version, unchanged
    r = d148.acf(e, lags); n = len(e)                                                 # the residuals' autocorrelations
    return n * (n + 2) * np.sum(r[1:] ** 2 / (n - np.arange(1, lags + 1))), lags - fitted  # Q, and its degrees of freedom


x = [1, 2, 3, 4, 5]                                                                   # acf 0.4 at lag 1, -0.1 at lag 2 (Day 148's check)
print(f"pacf of 1, 2, 3, 4, 5 at lag 2: (r2 - r1^2) / (1 - r1^2) = (-0.1 - 0.16) / 0.84 = {(-0.1 - 0.16) / 0.84:.4f}; pacf() gives {pacf(x, 2)[2]:.4f}")
a, b = sarima_polys(np.array([-0.4, -0.6]), (0, 1, 0, 1, 12))
print(f"airline MA polynomial (1 - 0.4 B)(1 - 0.6 B^12): lags {np.flatnonzero(b).tolist()}, coefficients {np.round(b[b != 0], 4).tolist()} "
      f"(by hand: 1, -0.4, -0.6, +0.24)")
print(f"MA(1), theta 0.5, on 1, 2, 3: e = 1, 2 - 0.5 x 1 = 1.5, 3 - 0.5 x 1.5 = 2.25; css_residuals gives "
      f"{css_residuals(np.array([1.0, 2, 3]), np.array([1.0]), np.array([1.0, 0.5])).tolist()}")
t = np.arange(10.0); y = 3 + 2 * t + np.array([0.1, -0.2, 0.0, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.1])  # a line plus small noise
print(f"a straight line: levenberg_marquardt {np.round(levenberg_marquardt(lambda v: y - v[0] - v[1] * t, [0.0, 0.0])[0], 6).tolist()}, "
      f"np.polyfit {np.round(np.polyfit(t, y, 1)[::-1], 6).tolist()}")
print(f"(1 - B)(1 - B^12) = 1 - B - B^12 + B^13: full_ar gives lags {np.flatnonzero(full_ar(np.array([1.0]))).tolist()}, "
      f"coefficients {full_ar(np.array([1.0]))[full_ar(np.array([1.0])) != 0].tolist()}")
A = np.array([1.0, -1.0]); b = np.array([1.0])                                        # a random walk: (1 - B) y = e
print(f"random walk 1, 2, 5: forecast {forecast(np.array([1.0, 2, 5]), A, b, np.zeros(2), 3).tolist()} (the last value), "
      f"psi weights {psi_weights(A, b, 4).tolist()} (so the interval grows with sqrt(h))")
rng = np.random.RandomState(0); e = rng.standard_normal(200); r1 = d148.acf(e, 1)[1]
print(f"Ljung-Box, one lag, 200 values: n(n + 2) r1^2 / (n - 1) = {200 * 202 * r1 ** 2 / 199:.4f}; ljung_box gives {ljung_box(e, 1)[0]:.4f}")
