"""
Day 148: Stationarity and Differencing -- the Augmented Dickey-Fuller Test, from Scratch

A series is stationary when its behaviour does not drift: the same mean, the
same spread and the same pattern of correlation whatever stretch of time you
look at. Most classical forecasting models (ARIMA, Day 149) assume it. Day 147's
series are not stationary: CO2 rises, and both have a yearly season.

Differencing -- each month minus the month before, or minus the same month a
year earlier -- removes a trend or a season. Two tests then ask whether it
worked, from opposite sides: the augmented Dickey-Fuller (ADF) test assumes the
series has a unit root (it wanders like a random walk) and looks for evidence
against it; the KPSS test assumes the series is stationary and looks for
evidence against that.

Everything is written from scratch in NumPy, including the table of critical
values the ADF test needs, which is simulated here; statsmodels is used only as
a reference to check against.

  1. What stationary means: four simulated series and the two real ones.
  2. The autocorrelation function, checked against statsmodels.
  3. Why the Dickey-Fuller test needs its own table: a simulation.
  4. The ADF test from scratch, on the real series and their differences.
  5. Over-differencing: what one difference too many does.
  6. Two tests, opposite questions: KPSS beside ADF.
  7. How often the ADF test is right: its power near a unit root.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: large matrix products were very slow with this machine's default
import contextlib                                      # contextlib: redirect_stdout, to keep the imported part quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import Day 147's part
import time                                            # time: how long the simulations take
import warnings                                        # warnings: statsmodels' notices are not needed here

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp  # MacKinnon's published tables, to check the simulated ones
from statsmodels.tsa.stattools import acf as sm_acf, adfuller, kpss as sm_kpss  # the library's versions, to check against
warnings.filterwarnings("ignore")                       # keep the output to this lesson's own lines

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts
with contextlib.redirect_stdout(io.StringIO()):         # the exercise prints its own checks when imported; keep them quiet
    import day147_exercise as d147                      # Day 147: load_co2, load_elec
SIMS = 20000                                            # simulated random walks per table of critical values
POWER_SIMS = 2000                                       # simulated series per cell of the power table


# =============================================================================
# THE PARTS -- NEW (copied unchanged into day148_exercise.py)
# =============================================================================
def acf(x, nlags):                                      # NEW: the autocorrelation function, lags 0 ... nlags
    x = np.asarray(x, float) - np.mean(x); d = np.sum(x * x)                         # centred, and the sum of squares
    return np.array([1.0] + [np.sum(x[:-k] * x[k:]) / d for k in range(1, nlags + 1)])  # correlation of the series with itself k steps later


def difference(y, lag=1):                               # NEW: y_t - y_(t - lag); the result is lag values shorter
    return y[lag:] - y[:-lag]


def df_stats(Y, regression="c"):                        # NEW: the Dickey-Fuller t statistic for every column of Y (no extra lags)
    n = len(Y) - 1                                                                    # one row per difference
    D = np.diff(Y, axis=0); L = Y[:-1]                                                # the changes, and the level before each change
    det = np.ones((n, 1)) if regression == "c" else np.column_stack([np.ones(n), np.arange(1, n + 1)])  # a constant, or a constant and a trend
    Q, _ = np.linalg.qr(det)                                                          # an orthonormal basis for the deterministic part
    D = D - Q @ (Q.T @ D); L = L - Q @ (Q.T @ L)                                      # remove it from both (the regression's other terms)
    g = np.sum(L * D, axis=0) / np.sum(L * L, axis=0)                                 # gamma: how the change depends on the level
    s2 = np.sum((D - g * L) ** 2, axis=0) / (n - det.shape[1] - 1)                    # the residual variance
    return g / np.sqrt(s2 / np.sum(L * L, axis=0))                                    # gamma divided by its standard error


def df_table(n, regression="c", sims=SIMS, seed=0):     # NEW: the statistic's distribution when the series IS a random walk
    Y = np.cumsum(np.random.RandomState(seed).standard_normal((n, sims)), axis=0)    # sims random walks of n steps, one per column
    return np.sort(df_stats(Y, regression))                                           # sorted, so quantiles and p-values are look-ups


def mc_pvalue(stat, table):                             # NEW: the share of simulated statistics at or below the observed one
    return np.searchsorted(table, stat, side="right") / len(table)


def adf(y, regression="c", maxlag=None):                # NEW: augmented Dickey-Fuller statistic, lags chosen by AIC (as statsmodels)
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


def kpss(y, regression="c", nlags=None):                # NEW: the KPSS statistic (null hypothesis: stationary)
    y = np.asarray(y, float); n = len(y)
    if nlags is None:
        nlags = min(int(np.ceil(12 * (n / 100) ** 0.25)), n - 1)                     # the same rule as statsmodels' 'legacy'
    if regression == "c":
        e = y - y.mean()                                                              # deviations from the mean
    else:
        X = np.column_stack([np.ones(n), np.arange(1, n + 1)]); e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]  # from a straight line
    s2 = (np.sum(e * e) + 2 * sum((1 - k / (nlags + 1)) * np.sum(e[k:] * e[:-k]) for k in range(1, nlags + 1))) / n  # long-run variance (Bartlett weights)
    return float(np.sum(np.cumsum(e) ** 2) / n ** 2 / s2)                             # the partial sums, relative to that variance


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def simulate(n=500, seed=0):                            # four series with known behaviour
    rng = np.random.RandomState(seed); e = rng.standard_normal((4, n))                # the same kind of noise for all four
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + e[1, t]                                             # AR(1): pulled back towards 0 each step
    return {"white noise": e[0], "AR(1), phi 0.9": ar, "random walk": np.cumsum(e[2]),
            "trend + noise": 0.02 * np.arange(n) + e[3]}


def demo_meaning(sims, tc, yc, te, ye):
    print("=" * 70)
    print("1. WHAT STATIONARY MEANS -- four simulated series and the two real ones")
    print("=" * 70)
    print("a stationary series has the same mean and spread in every stretch; compare the first and second halves")
    print(f"{'series':>22} {'mean, 1st half':>15} {'2nd half':>9} {'std, 1st half':>14} {'2nd half':>9} {'acf lag 1':>10} {'lag 12':>7}")
    for name, y in list(sims.items()) + [("CO2", yc), ("log orders", np.log(ye))]:
        a, b = np.array_split(y, 2); r = acf(y, 12)                                   # the two halves, and the autocorrelation
        print(f"{name:>22} {a.mean():15.3f} {b.mean():9.3f} {a.std():14.3f} {b.std():9.3f} {r[1]:10.3f} {r[12]:7.3f}")
    print("white noise keeps its mean and spread; the AR(1) is stationary too, but with phi 0.9 it has long memory, so 250 steps")
    print("are a short sample and its halves still differ; the random walk, the trend and CO2 drift away")
    print("Actual output from this lesson's run.")


def demo_acf(sims, yc):
    print("\n" + "=" * 70)
    print("2. THE AUTOCORRELATION FUNCTION -- from scratch, checked against statsmodels")
    print("=" * 70)
    err = max(np.max(np.abs(acf(y, 40) - sm_acf(y, nlags=40, fft=False))) for y in list(sims.values()) + [yc])
    print(f"largest difference from statsmodels' acf, lags 0-40, five series: {err:.1e}")
    r = acf(sims["AR(1), phi 0.9"], 5)
    print("AR(1) with phi 0.9: lags 1-5 " + " ".join(f"{v:.3f}" for v in r[1:]) + "; theory phi^k: " + " ".join(f"{0.9 ** k:.3f}" for k in range(1, 6)))
    r = acf(sims["random walk"], 100)
    print(f"random walk: lag 1 {r[1]:.3f}, lag 25 {r[25]:.3f}, lag 100 {r[100]:.3f} -- it fades slowly, the mark of a unit root")
    print("Actual output from this lesson's run.")


def demo_table():
    print("\n" + "=" * 70)
    print("3. WHY THE DICKEY-FULLER TEST NEEDS ITS OWN TABLE -- a simulation")
    print("=" * 70)
    print(f"{SIMS:,} random walks of 500 steps; for each, regress the change on the previous level and take the t statistic")
    print(f"{'regression':>28} {'1%':>8} {'5%':>8} {'10%':>8}   share below -1.645 (a 5% one-sided normal test)")
    tables = {}; t0 = time.perf_counter()
    for reg, name in (("c", "constant"), ("ct", "constant + trend")):
        T = df_table(500, reg); tables[reg] = T; q = np.quantile(T, [0.01, 0.05, 0.10])  # simulated critical values
        m = mackinnoncrit(N=1, regression=reg, nobs=499)                              # MacKinnon's, for the same size
        print(f"{name + ', simulated':>28} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f}   {np.mean(T < -1.645):.3f}")
        print(f"{name + ', MacKinnon':>28} {m[0]:8.3f} {m[1]:8.3f} {m[2]:8.3f}")
    print(f"{'normal distribution':>28} {-2.326:8.3f} {-1.645:8.3f} {-1.282:8.3f}")
    print(f"(simulation: {time.perf_counter() - t0:.1f} s)")
    print("Actual output from this lesson's run.")
    return tables


def demo_adf(tc, yc, te, ye):
    print("\n" + "=" * 70)
    print("4. THE ADF TEST FROM SCRATCH -- on the real series and their differences")
    print("=" * 70)
    print("ADF: regress the change on the previous level, lagged changes (number chosen by AIC) and a constant; a very negative")
    print("statistic is evidence against a unit root. p: from this lesson's simulated table, and from MacKinnon's (statsmodels)")
    tab = {n: df_table(n, "c", seed=1) for n in (500, 240)}                           # tables near each series' length
    print(f"{'series':>20} {'ADF':>8} {'statsmodels':>12} {'lags':>5} {'p, simulated':>13} {'p, MacKinnon':>13} {'acf 1':>6} {'acf 12':>7} {'std':>8}")
    D = difference; le = np.log(ye); keep = {}; worst = 0.0
    for name, y in (("CO2", yc), ("CO2, d1", D(yc)), ("CO2, d12", D(yc, 12)), ("CO2, d1 d12", D(D(yc, 12))),
                    ("log orders", le), ("log orders, d1", D(le)), ("log orders, d12", D(le, 12)), ("log orders, d1 d12", D(D(le, 12)))):
        s, p, m = adf(y); ref = adfuller(y, regression="c", autolag="AIC")            # ours, and the library's
        worst = max(worst, abs(s - ref[0])); assert p == ref[2]                       # the same statistic and the same lag
        pm = mc_pvalue(s, tab[500 if len(y) > 400 else 240]); r = acf(y, 12); keep[name] = (y, s)
        ps = f"{pm:.4f}" if pm > 0 else f"< {1 / SIMS:.4f}"                           # the simulation cannot resolve below 1 in 20,000
        print(f"{name:>20} {s:8.3f} {ref[0]:12.3f} {p:5d} {ps:>13} {ref[1]:13.4f} {r[1]:6.3f} {r[12]:7.3f} {y.std():8.4f}")
    print(f"largest difference from statsmodels' adfuller: {worst:.1e}; lags chosen: the same in every row")
    print("d1: each month minus the one before; d12: minus the same month a year before; std: the transformed series' spread")
    print("Actual output from this lesson's run.")
    return keep


def demo_over(sims, yc):
    print("\n" + "=" * 70)
    print("5. OVER-DIFFERENCING -- what one difference too many does")
    print("=" * 70)
    w = sims["white noise"]; dw = difference(w)                                       # differencing a series that was already stationary
    print(f"white noise, differenced: std {w.std():.3f} -> {dw.std():.3f} (theory: x sqrt(2) = {w.std() * np.sqrt(2):.3f}); "
          f"acf lag 1 {acf(dw, 1)[1]:.3f} (theory -0.5)")
    for name, y in (("CO2, d1 d12", difference(difference(yc, 12))), ("CO2, d1 d1 d12", difference(difference(difference(yc, 12)))),
                    ("CO2, d1 d12 d12", difference(difference(difference(yc, 12)), 12))):
        r = acf(y, 12)
        print(f"{name:>16}: std {y.std():.4f}, acf lag 1 {r[1]:+.3f}, lag 12 {r[12]:+.3f}, ADF {adf(y)[0]:.2f}")
    print("each extra difference raises the spread and makes one lag's autocorrelation more negative; the ADF test still 'passes',")
    print("so it cannot warn about this; the spread and the acf can")
    print("Actual output from this lesson's run.")


def demo_kpss(tc, yc, te, ye):
    print("\n" + "=" * 70)
    print("6. TWO TESTS, OPPOSITE QUESTIONS -- KPSS beside ADF")
    print("=" * 70)
    print("ADF's null hypothesis: a unit root (reject when the statistic is below -2.87, 5%, MacKinnon);")
    print("KPSS's null hypothesis: stationary around a constant (reject when the statistic is above 0.463, 5%, KPSS's table)")
    D = difference; le = np.log(ye); worst = 0.0
    print(f"{'series':>20} {'ADF':>8} {'KPSS':>7} {'statsmodels':>12}   reading")
    for name, y in (("CO2", yc), ("CO2, d1", D(yc)), ("CO2, d12", D(yc, 12)), ("CO2, d1 d12", D(D(yc, 12))),
                    ("log orders", le), ("log orders, d12", D(le, 12)), ("log orders, d1 d12", D(D(le, 12)))):
        a = adf(y)[0]; k = kpss(y); ref = sm_kpss(y, regression="c", nlags="legacy")[0]; worst = max(worst, abs(k - ref))
        ra, rk = a < -2.87, k > 0.463                                                 # does each test reject its null hypothesis?
        read = {(True, False): "both say stationary", (False, True): "both say not stationary",
                (True, True): "conflict: both reject", (False, False): "conflict: neither rejects"}[(ra, rk)]
        print(f"{name:>20} {a:8.3f} {k:7.3f} {ref:12.3f}   {read}")
    print(f"largest difference from statsmodels' kpss: {worst:.1e}")
    print("Actual output from this lesson's run.")


def demo_power(tables):
    print("\n" + "=" * 70)
    print("7. HOW OFTEN THE ADF TEST IS RIGHT -- its power near a unit root")
    print("=" * 70)
    print(f"{POWER_SIMS:,} simulated AR(1) series per cell, y_t = phi y_(t-1) + noise, started from their long-run spread;")
    print("the share the Dickey-Fuller test (constant, no extra lags) calls stationary at 5%, using a simulated table for each length")
    phis = (0.8, 0.9, 0.95, 0.99, 1.0); ns = (100, 250, 500); rows = {}
    print(f"{'length':>7} " + " ".join(f"{'phi ' + str(p):>10}" for p in phis))
    for n in ns:
        crit = np.quantile(df_table(n, "c", seed=2), 0.05); rng = np.random.RandomState(3 + n); out = []  # this length's 5% critical value
        for phi in phis:
            e = rng.standard_normal((n, POWER_SIMS)); Y = np.zeros((n, POWER_SIMS))
            Y[0] = e[0] / np.sqrt(1 - phi ** 2) if phi < 1 else 0.0                  # start where the series would settle (a walk starts at 0)
            for t in range(1, n):
                Y[t] = phi * Y[t - 1] + e[t]                                          # the AR(1) recursion, all series at once
            out.append(np.mean(df_stats(Y) < crit))                                   # the share called stationary
        rows[n] = out
        print(f"{n:7d} " + " ".join(f"{v:10.3f}" for v in out))
    print("phi 1.0 is a random walk: there the share is the false-alarm rate, which should be 0.05")
    print("Actual output from this lesson's run.")
    return phis, rows


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(sims, tables, keep, phis, rows, out):
    fig = plt.figure(figsize=(16, 11)); gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[0, 0:2]); cols = ("#4C72B0", "#55A868", "#C44E52", "#8172B2")
    for (name, y), c, off in zip(sims.items(), cols, (0, 12, 24, 36)):
        ax.plot(y + off, lw=0.7, color=c, label=name)                                 # each series shifted up so they do not overlap
    ax.set_xlabel("step"); ax.set_yticks([]); ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.set_title("four simulated series (shifted apart)", fontsize=10)
    ax = fig.add_subplot(gs[0, 2:4])
    for (name, y), c in zip(sims.items(), cols):
        ax.plot(acf(y, 40), marker="o", ms=2.5, lw=1, color=c, label=name)
    ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("lag"); ax.set_ylabel("autocorrelation"); ax.legend(fontsize=8)
    ax.set_title("their autocorrelation functions", fontsize=10)
    ax = fig.add_subplot(gs[1, 0:2]); x = np.linspace(-6, 4, 300)
    ax.hist(tables["c"], bins=120, density=True, alpha=0.6, color="#4C72B0", label="Dickey-Fuller, constant (simulated)")
    ax.hist(tables["ct"], bins=120, density=True, alpha=0.5, color="#DD8452", label="Dickey-Fuller, constant + trend")
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color="k", label="normal")
    ax.axvline(np.quantile(tables["c"], 0.05), color="#4C72B0", ls="--"); ax.axvline(-1.645, color="k", ls="--")
    ax.set_xlabel("t statistic, when the series is a random walk"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title("the statistic under a unit root is not normal (dashed: 5% points)", fontsize=10)
    ax = fig.add_subplot(gs[1, 2:4])
    for n, c in zip(rows, cols):
        ax.plot(phis, rows[n], marker="o", color=c, label=f"length {n}")
    ax.axhline(0.05, color="gray", ls="--", lw=0.8); ax.set_xlabel("phi (1.0 = random walk)"); ax.set_ylabel("share called stationary")
    ax.legend(fontsize=8); ax.set_title("power of the Dickey-Fuller test at 5%", fontsize=10)
    for j, name in enumerate(("CO2", "CO2, d1", "CO2, d12", "CO2, d1 d12")):
        y, s = keep[name]; ax = fig.add_subplot(gs[2, j]); ax.plot(y, lw=0.6, color="#4C72B0")
        ax.set_title(f"{name}: ADF {s:.2f}", fontsize=10); ax.set_xlabel("month")
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the testing-and-differencing loop
    fig, ax = plt.subplots(figsize=(15, 5.2)); ax.set_xlim(0, 15); ax.set_ylim(0.2, 4.9); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.2, 3.0, 1.8, 1.0, "series y\n(log if the swing\ngrows with level)", "#FFF2CC")
    box(3.8, 3.0, 2.2, 1.0, "plot it, and its\nautocorrelation\n(Day 147: season?)", "#DDEBF7")
    box(6.6, 3.9, 2.4, 0.9, "ADF: null = unit root\nreject if below -2.87", "#DDEBF7")
    box(6.6, 2.1, 2.4, 0.9, "KPSS: null = stationary\nreject if above 0.463", "#DDEBF7")
    box(9.6, 3.0, 2.2, 1.1, "ADF rejects and\nKPSS does not?", "#FCE4D6")
    box(12.9, 3.0, 2.4, 1.0, "stationary enough:\nmodel it (ARIMA,\nDay 149)", "#E2F0D9")
    box(9.6, 0.8, 3.4, 0.9, "no: difference once more (d12 for a season,\nd1 for a trend), then test again", "#F4CCCC")
    arrow(2.1, 3.0, 2.7, 3.0); arrow(4.9, 3.2, 5.4, 3.8); arrow(4.9, 2.8, 5.4, 2.2); arrow(7.8, 3.8, 8.5, 3.2); arrow(7.8, 2.2, 8.5, 2.8)
    arrow(10.7, 3.0, 11.7, 3.0); arrow(9.6, 2.45, 9.6, 1.25)
    ax.annotate("", xy=(1.2, 2.5), xytext=(7.9, 0.8), arrowprops=dict(arrowstyle="->", lw=1.3, connectionstyle="arc3,rad=-0.25"))
    ax.text(11.2, 3.15, "yes", fontsize=8.5); ax.text(9.7, 1.85, "no", fontsize=8.5)
    ax.text(7.5, 4.65, "stop at the first difference that passes: one too many raises the spread and adds a spurious negative correlation",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Day 148: testing for stationarity, and differencing until it holds", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stationarity_differencing_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 149: ARIMA from scratch. The differenced CO2 series still has memory (its autocorrelation is not 0);")
    print("autoregressive and moving-average terms model that memory, and forecasts are built back up through the differences.")


def main():                                             # runs every section in order
    tc, yc, _ = d147.load_co2(); te, ye = d147.load_elec()                           # Day 147's two series
    sims = simulate()                                                                 # four series with known behaviour
    demo_meaning(sims, tc, yc, te, ye)                                                # 1. what stationary means
    demo_acf(sims, yc)                                                                # 2. the autocorrelation function
    tables = demo_table()                                                             # 3. the Dickey-Fuller table
    keep = demo_adf(tc, yc, te, ye)                                                   # 4. ADF on the real series
    demo_over(sims, yc)                                                               # 5. over-differencing
    demo_kpss(tc, yc, te, ye)                                                         # 6. KPSS beside ADF
    phis, rows = demo_power(tables)                                                   # 7. power
    plot_results(sims, tables, keep, phis, rows, os.path.join(HERE, "stationarity_differencing_results.png"))
    draw_diagram()                                                                    # the loop
    print("\nsaved stationarity_differencing_results.png\nsaved stationarity_differencing_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
