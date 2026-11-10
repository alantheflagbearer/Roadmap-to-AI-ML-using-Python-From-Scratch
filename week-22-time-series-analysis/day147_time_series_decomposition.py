"""
Day 147: Time-Series Decomposition -- Trend, Seasonality and Residual on Real Data

A time series is a list of measurements in time order. Many of them are made of
three parts: a slow movement (the trend), a pattern that repeats every year (the
season), and what is left over (the residual). Decomposition separates them.

Week 22 is about time series. Like Weeks 16-21, Days 147-152 each add a full
lesson and a short exercise file holding that day's parts, and Day 153 builds a
Prophet-style forecaster from those parts.

Two real monthly series, both shipped with statsmodels (no download):
  * carbon dioxide at Mauna Loa, Hawaii, 1958-2001 (smooth: a clear trend and season)
  * new orders for electrical equipment in the euro area, 1995-2016 (noisy, and
    hit by the 2008-09 financial crisis)

Everything is written from scratch in NumPy; statsmodels is used only for the
data and as a reference to check against (its seasonal_decompose and STL).

  1. The data.
  2. The trend: a centred moving average, checked by hand.
  3. Classical decomposition, checked against statsmodels.
  4. Additive or multiplicative?
  5. What a classical decomposition gets wrong: ends, a changing season, a crash.
  6. Using it: seasonal adjustment, and a first forecast.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: the folder this script lives in
import warnings                                        # warnings: statsmodels' notices about pandas are not needed here

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram
import statsmodels.api as sm                            # statsmodels: the CO2 data, and the reference decompositions
import statsmodels.datasets.elec_equip as elec_equip    # statsmodels: the electrical-equipment data
from statsmodels.tsa.seasonal import STL, seasonal_decompose  # the library's decompositions, to check against
warnings.filterwarnings("ignore")                       # keep the output to this lesson's own lines

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
M = 12                                                  # months in a year: the season's length
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")  # calendar month names


# =============================================================================
# THE PARTS -- NEW (copied unchanged into day147_exercise.py)
# =============================================================================
def load_co2():                                         # NEW: Mauna Loa CO2, monthly means of the weekly readings
    s = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()                # weekly readings -> one mean per month
    y = s.to_numpy().astype(float); bad = np.isnan(y)                                  # months with no reading at all
    y[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), y[~bad])            # filled on a straight line between neighbours
    t = s.index.to_numpy().astype("datetime64[M]")                                    # the months, as numpy dates
    return t, y, [str(m) for m in t[bad]]                                            # months, values, and which were filled


def load_elec():                                        # NEW: euro-area new orders for electrical equipment, an index
    s = elec_equip.load().data.iloc[:, 0]                                             # one column, one row per month
    return s.index.to_numpy().astype("datetime64[M]"), s.to_numpy().astype(float)     # months, values


def centred_ma(y, m=M):                                 # NEW: the centred moving average (2 x m when m is even)
    w = np.ones(m) / m if m % 2 else np.r_[0.5, np.ones(m - 1), 0.5] / m             # even m: half weight on the two end months
    h = len(w) // 2                                                                   # months lost at each end
    T = np.full(len(y), np.nan)                                                       # no value where the window does not fit
    T[h:len(y) - h] = np.convolve(y, w, mode="valid")                                 # a weighted average of the window around each month
    return T


def classical_decompose(y, m=M, kind="additive"):       # NEW: trend, seasonal and residual, the classical way
    T = centred_ma(y, m)                                                              # 1. the trend
    D = y - T if kind == "additive" else y / T                                        # 2. the series without its trend
    pos = np.arange(len(y)) % m                                                       # each month's place in the year (from the first month)
    S0 = np.array([np.nanmean(D[pos == k]) for k in range(m)])                        # 3. the average for each place in the year
    S0 = S0 - S0.mean() if kind == "additive" else S0 / S0.mean()                     # 4. centred: adds to 0 (or multiplies to about 1) over a year
    S = S0[pos]                                                                       # the same pattern, every year
    R = y - T - S if kind == "additive" else y / (T * S)                              # 5. what is left
    return T, S, R


def strength(T, S, R):                                  # NEW: how strong the trend and season are, 0 (none) to 1 (all)
    ok = ~np.isnan(T)                                                                 # only where all three exist
    ft = max(0.0, 1 - np.var(R[ok]) / np.var(T[ok] + R[ok]))                          # trend strength: the residual's share of trend + residual
    fs = max(0.0, 1 - np.var(R[ok]) / np.var(S[ok] + R[ok]))                          # seasonal strength: the same, for the season
    return ft, fs


def by_month(S, t):                                     # NEW: a seasonal pattern reordered January ... December
    cal = t.astype(int) % 12                                                          # numpy months count from January 1970: % 12 is the calendar month
    return np.array([S[cal == k][0] for k in range(12)])                              # the pattern's value in each calendar month


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def acf(x, lag):                                        # autocorrelation at one lag (Day 148 builds it properly)
    x = x[~np.isnan(x)] - np.nanmean(x)                                               # centred, gaps dropped
    return float(np.sum(x[:-lag] * x[lag:]) / np.sum(x * x))                          # the standard estimate


def year(t):                                            # the calendar year of each month
    return t.astype("datetime64[Y]").astype(int) + 1970


def demo_data(tc, yc, miss, te, ye):
    print("=" * 70)
    print("1. THE DATA -- two real monthly series")
    print("=" * 70)
    print(f"Mauna Loa CO2 (ppm, parts per million; weekly readings averaged by month): {len(yc)} months, {tc[0]} to {tc[-1]}")
    print(f"  {len(miss)} months had no reading and were filled on a straight line between neighbours: {', '.join(miss)}")
    print(f"  lowest {yc.min():.1f}, highest {yc.max():.1f}, first year's mean {yc[:12].mean():.1f}, last year's {yc[-12:].mean():.1f}")
    print(f"euro-area new orders for electrical equipment (an index): {len(ye)} months, {te[0]} to {te[-1]}, none missing")
    print(f"  lowest {ye.min():.1f} ({te[ye.argmin()]}), highest {ye.max():.1f} ({te[ye.argmax()]})")
    print("Actual output from this lesson's run.")


def demo_trend(tc, yc):
    print("\n" + "=" * 70)
    print("2. THE TREND -- a centred moving average, checked by hand")
    print("=" * 70)
    w = np.r_[0.5, np.ones(11), 0.5] / 12                                             # the weights centred_ma uses for m = 12
    print(f"weights of the 2 x 12 moving average: {w[0]:.5f} (= 1/24) at both ends, {w[1]:.5f} (= 1/12) for the 11 in between; sum {w.sum():.3f}")
    rng = np.random.RandomState(0); pat = rng.randn(12)                               # any 12-month pattern
    T = centred_ma(np.tile(pat, 6))                                                   # repeated for 6 years
    print(f"a made-up 12-month pattern repeated 6 years: every trend value equals the pattern's mean; largest difference "
          f"{np.nanmax(np.abs(T - pat.mean())):.1e}")
    line = 3.0 + 0.5 * np.arange(72); T = centred_ma(line)                            # a straight line
    print(f"a straight line passes through unchanged: largest difference {np.nanmax(np.abs(T - line)):.1e}")
    print(f"so the average removes the season and keeps a straight trend; it loses {np.isnan(T[:12]).sum()} months at each end")
    T = centred_ma(yc); yr = year(tc)                                                 # CO2's trend
    print("CO2's trend, average rise per year in each decade:")
    for a in (1960, 1970, 1980, 1990):
        i = np.flatnonzero((yr == a) & (tc.astype(int) % 12 == 0))[0]                 # January of the decade's first year
        print(f"  {a}s: {(T[i + 120] - T[i]) / 10:.2f} ppm per year ({T[i]:.1f} -> {T[i + 120]:.1f})")
    print("Actual output from this lesson's run.")


def demo_classical(tc, yc, te, ye):
    print("\n" + "=" * 70)
    print("3. CLASSICAL DECOMPOSITION -- from scratch, checked against statsmodels")
    print("=" * 70)
    print("largest difference from statsmodels' seasonal_decompose (period 12):")
    for name, y in (("CO2", yc), ("orders", ye)):
        for kind in ("additive", "multiplicative"):
            T, S, R = classical_decompose(y, kind=kind); ref = seasonal_decompose(y, model=kind, period=12)  # ours, and the library's
            d = [np.nanmax(np.abs(a - b)) for a, b in ((T, ref.trend), (S, ref.seasonal), (R, ref.resid))]
            print(f"  {name:>6} {kind:<14} trend {d[0]:.1e}  seasonal {d[1]:.1e}  residual {d[2]:.1e}")
    keep = {}
    for name, t, y in (("CO2", tc, yc), ("orders", te, ye)):
        T, S, R = classical_decompose(y); cal = by_month(S, t); keep[name] = (T, S, R)
        print(f"{name}, additive seasonal pattern by calendar month:")
        print("   " + " ".join(f"{m:>6}" for m in MONTHS))
        print("   " + " ".join(f"{v:6.2f}" for v in cal))
        ft, fs = strength(T, S, R)
        print(f"   peak {MONTHS[cal.argmax()]}, low {MONTHS[cal.argmin()]}, swing {cal.max() - cal.min():.2f}; residual std {np.nanstd(R):.3f}; "
              f"strength of trend {ft:.3f}, of season {fs:.3f}")
    print("strength: 1 - var(residual) / var(component + residual); near 1 means the residual is small beside that component")
    print("Actual output from this lesson's run.")
    return keep


def demo_kind(tc, yc, te, ye):
    print("\n" + "=" * 70)
    print("4. ADDITIVE OR MULTIPLICATIVE -- does the season grow with the level?")
    print("=" * 70)
    print("additive: y = T + S + R (the season adds the same amount every year); multiplicative: y = T x S x R (the same share)")
    print(f"{'series':>8} {'corr(swing, level)':>19} {'fit error: additive':>20} {'multiplicative':>15}")
    out = {}
    for name, t, y in (("CO2", tc, yc), ("orders", te, ye)):
        T, S, _ = classical_decompose(y); Tm, Sm, _ = classical_decompose(y, kind="multiplicative")
        yr = year(t); ok = ~np.isnan(T); full = [a for a in np.unique(yr) if np.sum(ok & (yr == a)) == 12]  # years with all 12 trend values
        swing = [np.ptp((y - T)[yr == a]) for a in full]; level = [y[yr == a].mean() for a in full]  # each year's swing, and its level
        ea = np.sqrt(np.mean((y - T - S)[ok] ** 2)); em = np.sqrt(np.mean((y - Tm * Sm)[ok] ** 2))  # both fits' errors, in the series' own units
        print(f"{name:>8} {np.corrcoef(swing, level)[0, 1]:19.3f} {ea:20.4f} {em:15.4f}")
        out[name] = (Tm, Sm)
    print("fit error: root mean squared difference between the series and trend + season (or trend x season), in its own units")
    print("orders, multiplicative seasonal factors by calendar month (1.000 = an ordinary month):")
    print("   " + " ".join(f"{m:>6}" for m in MONTHS))
    print("   " + " ".join(f"{v:6.3f}" for v in by_month(out["orders"][1], te)))
    print("Actual output from this lesson's run.")
    return out


def demo_limits(tc, yc, te, ye):
    print("\n" + "=" * 70)
    print("5. WHAT A CLASSICAL DECOMPOSITION GETS WRONG -- ends, a changing season, a crash")
    print("=" * 70)
    T, S, R = classical_decompose(yc)
    print(f"ends: no trend for the first 6 and last 6 months (CO2: before {tc[6]} and after {tc[-7]}); the latest months are the ones a forecast needs")
    yr = year(tc); res = {}
    for a, b in ((1959, 1968), (1992, 2001)):
        k = (yr >= a) & (yr <= b); _, Sk, _ = classical_decompose(yc[k]); cal = by_month(Sk, tc[k]); res[a] = cal  # each decade on its own
        print(f"CO2 season from {a}-{b} alone: peak {MONTHS[cal.argmax()]} {cal.max():+.2f}, low {MONTHS[cal.argmin()]} {cal.min():+.2f}, swing {np.ptp(cal):.2f}")
    print(f"one fixed pattern for all 44 years cannot follow that; what it misses stays in the residual, which is not random noise:")
    st = STL(yc, period=12, robust=True).fit()                                        # statsmodels' STL: the season may change slowly
    print(f"  residual autocorrelation, lag 1 and lag 12: classical {acf(R, 1):.3f}, {acf(R, 12):.3f}; STL {acf(st.resid, 1):.3f}, {acf(st.resid, 12):.3f}")
    first, last = by_month(st.seasonal[:120], tc[:120]), by_month(st.seasonal[-120:], tc[-120:])
    print(f"  STL's season, averaged over the first and last 10 years: swing {np.ptp(first):.2f} -> {np.ptp(last):.2f}")
    Te, Se, Re = classical_decompose(ye, kind="multiplicative"); ye_ = year(te)       # the orders, multiplicatively
    print("the 2008-09 crash in the orders:")
    print(f"  trend peak {Te[np.nanargmax(Te)]:.1f} ({te[np.nanargmax(Te)]}), lowest after it {np.nanmin(Te[te > te[np.nanargmax(Te)]]):.1f} "
          f"({te[te > te[np.nanargmax(Te)]][np.nanargmin(Te[te > te[np.nanargmax(Te)]])]})")
    big = np.argsort(-np.nan_to_num(np.abs(np.log(Re))))[:5]                          # the five largest residuals (as a share)
    print("  five largest residuals: " + ", ".join(f"{te[i]} {Re[i] - 1:+.1%}" for i in sorted(big)))
    keep = (ye_ < 2008) | (ye_ > 2010)                                                # drop 2008-2010
    Sk = Se.copy(); D = ye / Te; pos = np.arange(len(ye)) % 12                        # the seasonal factors again, without the crash years
    S0 = np.array([np.nanmean(D[(pos == k) & keep]) for k in range(12)]); S0 /= S0.mean(); Sk = S0[pos]
    print(f"  seasonal factors with 2008-2010 left out: largest change in any month {np.max(np.abs(by_month(Sk, te) - by_month(Se, te))):.3f}")
    ste = STL(np.log(ye), period=12, robust=True).fit()                               # STL on the log: multiplicative, the season may change
    print(f"  residual std (as a share): classical {np.nanstd(Re - 1):.4f}; STL on the log {np.std(ste.resid):.4f}")
    print("Actual output from this lesson's run.")
    return res, st, ste


def demo_use(tc, yc, te, ye):
    print("\n" + "=" * 70)
    print("6. USING IT -- seasonal adjustment, and a first forecast")
    print("=" * 70)
    _, Se, _ = classical_decompose(ye, kind="multiplicative"); adj = ye / Se         # seasonally adjusted orders
    dr, da = np.diff(ye), np.diff(adj)                                                # month-on-month changes, raw and adjusted
    print(f"orders, month-on-month changes: raw and seasonally adjusted point opposite ways in {np.mean(np.sign(dr) != np.sign(da)):.3f} of months")
    for i in (np.argmin(dr), np.argmax(dr)):
        print(f"  {te[i + 1]}: raw {dr[i] / ye[i]:+.1%}, seasonally adjusted {da[i] / adj[i]:+.1%}")
    i = np.flatnonzero(te == np.datetime64("2008-11"))[0]                              # a month inside the crash
    print(f"  {te[i]}: raw {dr[i - 1] / ye[i - 1]:+.1%}, seasonally adjusted {da[i - 1] / adj[i - 1]:+.1%}")
    n = np.flatnonzero(tc == np.datetime64("2000-01"))[0]; h = len(yc) - n            # train up to 1999-12, forecast the last 24 months
    T, S, _ = classical_decompose(yc[:n]); valid = np.flatnonzero(~np.isnan(T))     # the training trend: it ends 6 months before the data
    steps = np.arange(n, n + h); pos = steps % 12                                     # the months to forecast, and their place in the year
    naive = np.full(h, yc[n - 1]); snaive = yc[n - 12:n][np.arange(h) % 12]           # last value; same month last year
    print(f"CO2 forecast for {tc[n]} to {tc[-1]} ({h} months) from data up to {tc[n - 1]} (training trend ends {tc[valid[-1]]});")
    rises = [f"{tc[i - 12]} -> {tc[i]}: {T[i] - T[i - 12]:+.2f}" for i in valid[-1] - np.array([24, 12, 0])]  # the training trend's last three yearly rises
    print("training trend's rise over each of its last three years: " + "; ".join(rises))
    print("mean absolute error and mean error (forecast - actual), ppm:")
    for name, f in (("last value", naive), ("same month last year", snaive)):
        print(f"  {name:<37} {np.mean(np.abs(f - yc[n:])):6.3f} {np.mean(f - yc[n:]):+7.3f}")
    dec = {}
    for k in (24, 60, 120):
        ok = valid[-k:]; b, a = np.polyfit(ok, T[ok], 1)                              # a straight line through the last k trend values
        dec[k] = b * steps + a + S[:12][pos]                                          # the line, extended, plus the season for that month
        print(f"  {'trend line (last ' + str(k) + ' months) + season':<37} {np.mean(np.abs(dec[k] - yc[n:])):6.3f} {np.mean(dec[k] - yc[n:]):+7.3f}"
              f"   slope {12 * b:.2f} ppm per year")
    print("the line's window is a choice; picking it by this table would use the test months to choose it (later this week:")
    print("choosing on earlier data only)")
    print("Actual output from this lesson's run.")
    return adj, dec, snaive, n


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(tc, yc, te, ye, keep, res, st, ste, adj, dec, snaive, n, out):
    fig = plt.figure(figsize=(16, 11)); gs = fig.add_gridspec(3, 3)
    T, S, R = keep["CO2"]; x = tc.astype("datetime64[D]")                             # dates matplotlib can place
    ax = fig.add_subplot(gs[0, 0:2]); ax.plot(x, yc, lw=0.8, label="CO2, monthly"); ax.plot(x, T, color="k", lw=1.5, label="trend (2 x 12 moving average)")
    ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=8); ax.set_title("Mauna Loa CO2 and its trend", fontsize=10)
    ax = fig.add_subplot(gs[0, 2]); m = np.arange(12)
    for a, c in ((1959, "#4C72B0"), (1992, "#C44E52")):
        ax.plot(m, res[a], marker="o", color=c, label=f"{a}-{a + 9}, classical")
    ax.plot(m, by_month(st.seasonal[:120], tc[:120]), ls="--", color="#4C72B0", label="first 10 years, STL")
    ax.plot(m, by_month(st.seasonal[-120:], tc[-120:]), ls="--", color="#C44E52", label="last 10 years, STL")
    ax.set_xticks(m); ax.set_xticklabels([s[0] for s in MONTHS]); ax.set_ylabel("seasonal (ppm)"); ax.legend(fontsize=7)
    ax.set_title("CO2's season grew", fontsize=10); ax.axhline(0, color="gray", lw=0.5)
    xe = te.astype("datetime64[D]"); Te, Se, Re = classical_decompose(ye, kind="multiplicative")
    ax = fig.add_subplot(gs[1, 0:2]); ax.plot(xe, ye, lw=0.8, label="orders, monthly"); ax.plot(xe, Te, color="k", lw=1.5, label="trend, classical")
    ax.plot(xe, np.exp(ste.trend), color="#DD8452", lw=1.2, label="trend, STL on the log")
    ax.axvspan(np.datetime64("2008-01-01"), np.datetime64("2010-12-31"), color="gray", alpha=0.15, label="2008-2010")
    ax.set_ylabel("orders (index)"); ax.legend(fontsize=8, ncol=2); ax.set_title("Euro-area electrical equipment orders", fontsize=10)
    ax = fig.add_subplot(gs[1, 2]); ax.bar(m, by_month(Se, te) - 1, color="#4C72B0")
    ax.set_xticks(m); ax.set_xticklabels([s[0] for s in MONTHS]); ax.set_ylabel("seasonal factor - 1"); ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("orders: multiplicative season", fontsize=10)
    ax = fig.add_subplot(gs[2, 0]); ax.plot(x, R, lw=0.7, label="classical"); ax.plot(x, st.resid, lw=0.7, alpha=0.8, label="STL")
    ax.set_ylabel("residual (ppm)"); ax.legend(fontsize=8); ax.set_title("CO2 residuals", fontsize=10)
    ax = fig.add_subplot(gs[2, 1]); k = (te >= np.datetime64("2005-01")) & (te <= np.datetime64("2012-12"))
    ax.plot(xe[k], ye[k], lw=0.9, label="raw"); ax.plot(xe[k], adj[k], lw=1.4, color="k", label="seasonally adjusted")
    ax.set_ylabel("orders (index)"); ax.legend(fontsize=8); ax.set_title("orders 2005-2012: raw and adjusted", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax = fig.add_subplot(gs[2, 2]); xs = x[n - 24:]
    ax.plot(xs, yc[n - 24:], color="k", lw=1.2, label="actual"); ax.plot(x[n:], dec[24], color="#C44E52", label="trend line (24 months) + season"); ax.plot(x[n:], dec[120], color="#8172B2", label="trend line (120 months) + season")
    ax.plot(x[n:], snaive, color="#55A868", ls="--", label="same month last year"); ax.axvline(x[n], color="gray", lw=0.7)
    ax.set_ylabel("CO2 (ppm)"); ax.legend(fontsize=8); ax.set_title("forecasting 2000-2001 from data to 1999", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the classical decomposition, step by step
    fig, ax = plt.subplots(figsize=(15, 5.0)); ax.set_xlim(0, 15); ax.set_ylim(0.3, 4.8); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.2, 3.0, 1.8, 1.0, "series y\n(one value\nper month)", "#FFF2CC")
    box(3.9, 3.0, 2.3, 1.1, "centred 2 x 12\nmoving average\n(loses 6 months\nat each end)", "#DDEBF7")
    box(6.6, 3.0, 1.6, 0.8, "trend T", "#E2F0D9")
    box(3.9, 1.0, 2.3, 1.0, "y - T\n(or y / T)\nwithout the trend", "#DDEBF7")
    box(6.9, 1.0, 2.5, 1.0, "average each month\nof the year, then\ncentre over the year", "#DDEBF7")
    box(9.7, 1.0, 1.8, 0.8, "season S\n(same every year)", "#E2F0D9")
    box(10.4, 3.0, 2.4, 1.0, "residual R\n= y - T - S\n(or y / (T x S))", "#E2F0D9")
    box(13.3, 3.0, 2.4, 1.1, "checks: is R\nrandom? does the\nseason change?", "#FCE4D6")
    arrow(2.1, 3.0, 2.75, 3.0); arrow(5.05, 3.0, 5.8, 3.0); arrow(1.2, 2.5, 2.75, 1.2)
    arrow(5.05, 1.0, 5.65, 1.0); arrow(8.15, 1.0, 8.8, 1.0); arrow(7.4, 3.0, 9.2, 3.0); arrow(9.7, 1.4, 10.1, 2.5)
    arrow(11.6, 3.0, 12.1, 3.0); arrow(6.3, 2.6, 4.7, 1.5)                             # the trend is also what y - T takes away
    ax.text(7.5, 4.35, "additive: y = T + S + R (the season adds a fixed amount);  multiplicative: y = T x S x R (a fixed share)",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Day 147: classical decomposition, step by step", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "time_series_decomposition_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 148: stationarity. Many forecasting models need a series whose behaviour does not drift; differencing")
    print("(month minus the month before, or minus the same month last year) removes trend and season, and the")
    print("augmented Dickey-Fuller test, built from scratch, checks whether it worked.")


def main():                                             # runs every section in order
    tc, yc, miss = load_co2(); te, ye = load_elec()                                   # the two series
    demo_data(tc, yc, miss, te, ye)                                                   # 1. the data
    demo_trend(tc, yc)                                                                # 2. the trend
    keep = demo_classical(tc, yc, te, ye)                                             # 3. classical decomposition
    demo_kind(tc, yc, te, ye)                                                         # 4. additive or multiplicative
    res, st, ste = demo_limits(tc, yc, te, ye)                                        # 5. what it gets wrong
    adj, dec, snaive, n = demo_use(tc, yc, te, ye)                                    # 6. using it
    plot_results(tc, yc, te, ye, keep, res, st, ste, adj, dec, snaive, n, os.path.join(HERE, "time_series_decomposition_results.png"))
    draw_diagram()                                                                    # the method
    print("\nsaved time_series_decomposition_results.png\nsaved time_series_decomposition_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
