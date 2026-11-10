"""
Day 147 Exercise -- Decomposing a Series into Trend, Season and Residual

The first of Week 22's parts for Day 153's Prophet-style forecaster.

    load_co2()                         Mauna Loa CO2, monthly, 1958-2001 (months with no reading filled on a line)
    load_elec()                        euro-area electrical equipment orders, monthly, 1995-2016
    centred_ma(y, m)                   the centred moving average (2 x m when m is even): the trend
    classical_decompose(y, m, kind)    trend, seasonal and residual, additive or multiplicative
    strength(T, S, R)                  how strong the trend and the season are, from 0 to 1
    by_month(S, t)                     a seasonal pattern reordered January ... December

Checked by hand: two moving averages worked out on paper, a made-up series whose
trend and season are known, the strength of made-up components, and the CO2
season's peak and low.
"""
import numpy as np                                     # numpy: every array operation below is built on this
import statsmodels.api as sm                            # statsmodels: the CO2 data
import statsmodels.datasets.elec_equip as elec_equip    # statsmodels: the electrical-equipment data
M = 12                                                  # months in a year: the season's length


def load_co2():                                         # NEW: Day 147's version, unchanged
    s = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()                # weekly readings -> one mean per month
    y = s.to_numpy().astype(float); bad = np.isnan(y)                                  # months with no reading at all
    y[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), y[~bad])            # filled on a straight line between neighbours
    t = s.index.to_numpy().astype("datetime64[M]")                                    # the months, as numpy dates
    return t, y, [str(m) for m in t[bad]]                                            # months, values, and which were filled


def load_elec():                                        # NEW: Day 147's version, unchanged
    s = elec_equip.load().data.iloc[:, 0]                                             # one column, one row per month
    return s.index.to_numpy().astype("datetime64[M]"), s.to_numpy().astype(float)     # months, values


def centred_ma(y, m=M):                                 # NEW: Day 147's version, unchanged
    w = np.ones(m) / m if m % 2 else np.r_[0.5, np.ones(m - 1), 0.5] / m             # even m: half weight on the two end months
    h = len(w) // 2                                                                   # months lost at each end
    T = np.full(len(y), np.nan)                                                       # no value where the window does not fit
    T[h:len(y) - h] = np.convolve(y, w, mode="valid")                                 # a weighted average of the window around each month
    return T


def classical_decompose(y, m=M, kind="additive"):       # NEW: Day 147's version, unchanged
    T = centred_ma(y, m)                                                              # 1. the trend
    D = y - T if kind == "additive" else y / T                                        # 2. the series without its trend
    pos = np.arange(len(y)) % m                                                       # each month's place in the year (from the first month)
    S0 = np.array([np.nanmean(D[pos == k]) for k in range(m)])                        # 3. the average for each place in the year
    S0 = S0 - S0.mean() if kind == "additive" else S0 / S0.mean()                     # 4. centred: adds to 0 (or multiplies to about 1) over a year
    S = S0[pos]                                                                       # the same pattern, every year
    R = y - T - S if kind == "additive" else y / (T * S)                              # 5. what is left
    return T, S, R


def strength(T, S, R):                                  # NEW: Day 147's version, unchanged
    ok = ~np.isnan(T)                                                                 # only where all three exist
    ft = max(0.0, 1 - np.var(R[ok]) / np.var(T[ok] + R[ok]))                          # trend strength: the residual's share of trend + residual
    fs = max(0.0, 1 - np.var(R[ok]) / np.var(S[ok] + R[ok]))                          # seasonal strength: the same, for the season
    return ft, fs


def by_month(S, t):                                     # NEW: Day 147's version, unchanged
    cal = t.astype(int) % 12                                                          # numpy months count from January 1970: % 12 is the calendar month
    return np.array([S[cal == k][0] for k in range(12)])                              # the pattern's value in each calendar month


y = np.array([1.0, 5, 2, 8, 3, 7, 4])                                                 # seven made-up values
print(f"3-month average of {y.tolist()}: at the second value (1 + 5 + 2) / 3 = {(1 + 5 + 2) / 3:.4f}; centred_ma gives {centred_ma(y, 3)[1]:.4f}")
print(f"2 x 4 average at the third value: (0.5 x 1 + 5 + 2 + 8 + 0.5 x 3) / 4 = {(0.5 * 1 + 5 + 2 + 8 + 0.5 * 3) / 4:.4f}; centred_ma gives {centred_ma(y, 4)[2]:.4f}")
t = np.arange(60); season = np.array([3, 1, -2, -4, -1, 0, 2, 4, 1, -1, -2, -1.0])     # a season that adds to 0 over the year
y = 10 + 0.2 * t + season[t % 12]                                                     # a straight trend plus that season, no noise
T, S, R = classical_decompose(y)
print(f"trend 10 + 0.2 t plus a known season, 5 years: largest error in the season {np.max(np.abs(S - season[t % 12])):.1e}, "
      f"in the trend {np.nanmax(np.abs(T - (10 + 0.2 * t))):.1e}, largest residual {np.nanmax(np.abs(R)):.1e}")
T, S, R = np.array([1.0, 2, 3, 4]), np.array([1.0, -1, 1, -1]), np.array([0.1, -0.1, 0.1, -0.1])
print(f"strength of made-up parts: trend 1 - 0.01 / 1.16 = {1 - 0.01 / 1.16:.5f}, season 1 - 0.01 / 1.21 = {1 - 0.01 / 1.21:.5f}; "
      f"strength() gives {strength(T, S, R)[0]:.5f}, {strength(T, S, R)[1]:.5f}")
tc, yc, miss = load_co2(); te, ye = load_elec()
cal = by_month(classical_decompose(yc)[1], tc)                                        # CO2's season, January ... December
print(f"CO2: {len(yc)} months ({len(miss)} filled); its season peaks in month {cal.argmax() + 1} (May) at {cal.max():+.2f} ppm "
      f"and is lowest in month {cal.argmin() + 1} (October) at {cal.min():+.2f}")
print(f"orders: {len(ye)} months, {te[0]} to {te[-1]}")
