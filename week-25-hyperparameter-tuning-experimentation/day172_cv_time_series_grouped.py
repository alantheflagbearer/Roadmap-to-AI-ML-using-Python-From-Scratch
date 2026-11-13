"""
Day 172: Cross-Validation for Time Series and Grouped Data

Ordinary k-fold cross-validation shuffles rows into folds. That assumes rows
are independent. Two common kinds of data break it: a time series, where a
row's neighbours in time carry most of its information, and grouped data,
where many rows come from the same person. Shuffled folds then put a test
row's neighbours, or its owner's other rows, into training, and the score
flatters any model that can exploit them. Today builds the honest splitters
from scratch and measures the leak against a known truth:

  1. a simulated series whose true forecast error can be computed exactly
     (by simulating its future): shuffled, blocked, blocked with a gap, and
     forward-chaining folds, for three forecasters
  2. the same on two real series, Mauna Loa CO2 and euro-area electrical
     equipment orders, with the final five years held out as one check
  3. forward folds estimate a period: the electrical series' 2008 crash
  4. grouped data, MovieLens: predicting ratings for new users with
     boosted trees; shuffled rows against grouped folds against 300 users
     never used in any fold, and which tree depth each would choose
  5. what carried the leak: the timestamp, the users' own facts, or neither

Data: statsmodels' CO2 and electrical-equipment series (Week 22's loaders),
MovieLens 100K (Week 23's loaders), all already on disk. The boosted trees
are Day 109's. Every line of code below carries its own comment, continuing
Day 52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: what each method costs

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
for w in ("week-16-classical-ml-trees", "week-22-time-series-analysis", "week-23-recommender-systems"):  # earlier weeks' parts
    sys.path.insert(0, os.path.join(ROOT, w))                                         # importable by name
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day109_exercise as d109                      # Day 109: fit_boost, predict_boost
    import day147_exercise as d147                      # Day 147: load_co2, load_elec
    import day154_exercise as d154                      # Day 154: load_movielens
    import day156_exercise as d156                      # Day 156: load_years
    import day158_exercise as d158                      # Day 158: load_users

H = 12                                                  # forecast horizon: the next 12 months
LAGS = 12                                               # past values the lag model uses
K = 5                                                   # folds
HOLD_MONTHS = 60                                        # the real series' final five years, held out
SIMS, SIM_N, FUTURES = 50, 300, 400                     # simulated series, their length, futures simulated per series
DEPTHS = (1, 2, 4, 6, 8)                                # tree depths for the MovieLens choice
ROUNDS, RATE, MIN_LEAF = 50, 0.2, 20                    # boosting settings, fixed
NEW_USERS, DEV_USERS = 300, 400                         # users held out entirely; users the folds are built from
CV_SEEDS = 2                                            # fold shuffles per scheme


# =============================================================================
# THE PARTS (copied unchanged into day172_exercise.py, for Day 174's build)
# =============================================================================
def shuffled_folds(n, k, seed):                         # ordinary k-fold: rows dealt into k folds at random
    f = np.random.RandomState(seed).permutation(n) % k                               # each row's fold
    return [(np.flatnonzero(f != j), np.flatnonzero(f == j)) for j in range(k)]       # (training rows, test rows) per fold


def blocked_folds(n, k, gap=0):                         # k contiguous blocks; optionally drop `gap` rows either side of the test block
    b = np.arange(n) * k // n                                                         # each row's block, in time order
    out = []                                                                          # one split per block
    for j in range(k):                                                                # each block in turn is the test block
        te = np.flatnonzero(b == j); lo, hi = te[0] - gap, te[-1] + gap               # the test rows; the purged zone around them
        tr = np.flatnonzero((b != j) & ((np.arange(n) < lo) | (np.arange(n) > hi)))   # everything outside it trains
        out.append((tr, te))                                                          # one split
    return out                                                                        # k splits


def forward_folds(t, origins, horizon):                 # forward chaining: train on the past, test the next `horizon` steps
    t = np.asarray(t)                                                                 # each row's time
    return [(np.flatnonzero(t <= o), np.flatnonzero((t > o) & (t <= o + horizon))) for o in origins]  # one split per origin


def group_folds(groups, k, seed):                       # every row of a group in the same fold
    g = np.unique(groups); f = dict(zip(g, np.random.RandomState(seed).permutation(len(g)) % k))  # each group's fold
    fold = np.array([f[x] for x in groups])                                           # each row inherits its group's fold
    return [(np.flatnonzero(fold != j), np.flatnonzero(fold == j)) for j in range(k)]  # (training rows, test rows) per fold


def cross_val(folds, fit_score):                        # fit_score(train rows, test rows) -> one number; mean and per-fold scores
    s = np.array([fit_score(tr, te) for tr, te in folds])                             # one score per fold
    return s.mean(), s                                                                # the estimate and its pieces


# =============================================================================
# THREE FORECASTERS (rows are target months; X, y, t from design())
# =============================================================================
def design(y, kind):                                    # the rows each forecaster learns from
    n = len(y); t = np.arange(n)                                                      # months 0 .. n-1
    if kind == "lag":                                                                 # predict y[T] from y[T-12] ... y[T-23]
        T = np.arange(H + LAGS - 1, n)                                                # target months that have all their lags
        X = np.column_stack([y[T - H - j] for j in range(LAGS)] + [np.ones(len(T))])  # 12 lagged values and an intercept
        return X, y[T], T                                                             # features, targets, target months
    X = np.column_stack([t / 100, (t / 100) ** 2] + [(t % 12 == m).astype(float) for m in range(12)])  # trend and month
    return X, y.copy(), t                                                             # the time-index models use the month itself


def fit_predict(kind, X, y, t, tr, te):                 # fit on rows tr, predict rows te
    if kind == "knn":                                                                 # the mean of the 3 nearest training months
        nn = np.argsort(np.abs(t[te][:, None] - t[tr][None, :]), 1)[:, :3]           # nearest in time
        return y[tr][nn].mean(1)                                                      # an interpolator
    w = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]                                  # least squares
    return X[te] @ w                                                                  # trend + season, or the lag model


def mae_scorer(kind, X, y, t):                          # a fit_score function for cross_val
    return lambda tr, te: np.mean(np.abs(fit_predict(kind, X, y, t, tr, te) - y[te]))  # mean absolute error on the test rows


def schemes(n_rows, t, last):                           # the four ways to cut a series (rows up to month `last` only)
    origins = range(last - 5 * H, last - H + 1, H)                                    # five forward origins, a year apart, at the end
    return {"shuffled": shuffled_folds(n_rows, K, 0), "blocked": blocked_folds(n_rows, K),  # four splitters over the same rows
            f"blocked, gap {H + LAGS}": blocked_folds(n_rows, K, gap=H + LAGS),     # a gap wide enough that no lag reaches across
            "forward": forward_folds(t, origins, H)}                                  # train on the past only


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def head(title):                                        # a section banner
    print("\n" + "=" * 70); print(title); print("=" * 70)  # three lines


def simulate(rng, n):                                   # a trending, seasonal series with a known recipe
    level = np.cumsum(0.05 + 0.3 * rng.randn(n))                                     # a random walk with drift
    return level + 2 * np.sin(2 * np.pi * np.arange(n) / 12) + 0.5 * rng.randn(n), level[-1]  # plus season and noise; the last level


def true_error(kind, y, last_level, rng):               # expected error on the next 12 months, by simulating them
    n = len(y); X, Y, T = design(y, kind)                                              # the model is fitted on the whole series
    tf = np.arange(n, n + H)                                                          # the 12 future months
    steps = 0.05 + 0.3 * rng.randn(FUTURES, H)                                        # each future's level steps
    fut = last_level + np.cumsum(steps, 1) + 2 * np.sin(2 * np.pi * tf / 12) + 0.5 * rng.randn(FUTURES, H)  # FUTURES x 12 futures
    if kind == "lag":                                                                 # features of future targets are already known
        Xf = np.column_stack([y[tf - H - j] for j in range(LAGS)] + [np.ones(H)])    # y[T-12] ... y[T-23], all observed
        w = np.linalg.lstsq(X, Y, rcond=None)[0]; p = Xf @ w                          # one forecast per future month
    elif kind == "knn":                                                               # the 3 nearest months are the last 3
        p = np.full(H, y[-3:].mean())                                                 # a flat forecast
    else:                                                                             # trend and season, extrapolated
        Xf = np.column_stack([tf / 100, (tf / 100) ** 2] + [(tf % 12 == m).astype(float) for m in range(12)])  # trend and month, for the future months
        p = Xf @ np.linalg.lstsq(X, Y, rcond=None)[0]                                 # one forecast per future month
    return np.mean(np.abs(fut - p))                                                   # averaged over futures and months


KINDS = ("trend", "knn", "lag")                         # trend + season regression; 3 nearest months; lags 12 to 23


def demo_simulated():                                   # section 1
    head("1. A KNOWN TRUTH: SIMULATED SERIES, 12-MONTH FORECASTS (mean absolute error)")  # the banner
    rng = np.random.RandomState(0); out = {k: {} for k in KINDS}                      # results per forecaster and scheme
    for s in range(SIMS):                                                             # many series
        y, lv = simulate(rng, SIM_N)                                                  # one series and its last level
        for kind in KINDS:                                                            # each forecaster
            X, Y, T = design(y, kind)                                                 # its rows
            for name, folds in schemes(len(Y), T, SIM_N - 1).items():                 # each way of cutting
                out[kind].setdefault(name, []).append(cross_val(folds, mae_scorer(kind, X, Y, T))[0])  # its estimate
            out[kind].setdefault("truth", []).append(true_error(kind, y, lv, rng))   # the real error on the next 12 months
    print(f"  {SIMS} series of {SIM_N} months: random walk with drift + season + noise; truth from {FUTURES} simulated futures each")
    names = list(out[KINDS[0]])                                                       # scheme names, truth last
    print("  forecaster " + "".join(f"{n:>18s}" for n in names))
    for kind in KINDS:                                                                # one line per forecaster
        print(f"  {kind:10s} " + "".join(f"{np.mean(out[kind][n]):18.3f}" for n in names))
    for n in names:                                                                   # which forecaster each would choose
        print(f"  choice by {n:16s}: {min(KINDS, key=lambda k: np.mean(out[k][n]))}")
    print("Actual output from this lesson's run.")
    return {k: {n: float(np.mean(v)) for n, v in d.items()} for k, d in out.items()}  # for the figure


def demo_real():                                        # sections 2 and 3
    head("2. REAL SERIES: CV ESTIMATES AGAINST THE FINAL FIVE YEARS (mean absolute error)")  # the banner
    R = {}                                                                            # per series
    for name, (t, y) in (("CO2", d147.load_co2()[:2]), ("electrical equipment", d147.load_elec())):  # two real series
        n = len(y); last = n - HOLD_MONTHS - 1                                        # the last month the folds may use
        print(f"  {name}: {n} months, {t[0]} to {t[-1]}; folds use months up to {t[last]}; held out {t[last + 1]} to {t[-1]}")
        R[name] = {}                                                                  # per forecaster
        for kind in KINDS:                                                            # each forecaster
            X, Y, T = design(y, kind); dev = np.flatnonzero(T <= last)               # rows the folds may use
            est = {s: cross_val(f, mae_scorer(kind, X[dev], Y[dev], T[dev]))[0] for s, f in schemes(len(dev), T[dev], last).items()}  # every scheme's estimate
            hold = forward_folds(T, range(last, n - H, H), H)                        # five origins inside the held-out years
            truth, per = cross_val(hold, mae_scorer(kind, X, Y, T))                  # forecasting them for real, once
            fw = forward_folds(T[dev], range(last - 5 * H, last - H + 1, H), H)       # the forward folds again, to keep each fold
            R[name][kind] = dict(est, held_out=truth, fold=cross_val(fw, mae_scorer(kind, X[dev], Y[dev], T[dev]))[1], hold_fold=per)  # kept for the figure and section 3
            print(f"    {kind:6s} " + ", ".join(f"{s} {v:.3f}" for s, v in est.items()) + f"; held-out years {truth:.3f}")
        for s in ("shuffled", "forward", "held_out"):                                 # which forecaster each would choose
            print(f"    choice by {s:9s}: {min(KINDS, key=lambda k: R[name][k][s])}")
    print("Actual output from this lesson's run.")
    head("3. FORWARD FOLDS ESTIMATE A PERIOD: ONE ERROR PER ORIGIN (lag model)")          # the banner
    for name in R:                                                                    # both series
        t = (d147.load_co2()[0] if name == "CO2" else d147.load_elec()[0])            # the months, for the labels
        last = len(t) - HOLD_MONTHS - 1                                               # as above
        o = [str(t[x + 1]) for x in range(last - 5 * H, last - H + 1, H)]            # first forecast month of each fold
        print(f"  {name}: " + ", ".join(f"{a} {v:.2f}" for a, v in zip(o, R[name]["lag"]["fold"]))
              + "; held-out origins " + ", ".join(f"{v:.2f}" for v in R[name]["lag"]["hold_fold"]))
    print("Actual output from this lesson's run.")
    return R                                                                          # for the figure


def movielens():                                        # the rows, users and features for sections 4 and 5
    u, i, r, ts = d154.load_movielens(); age, fem, occ = d158.load_users()           # ratings; users' age, sex, occupation
    yr = d156.load_years().astype(float); yr[yr < 0] = np.median(yr[yr > 0])          # film years; the one missing one filled
    perm = np.random.RandomState(0).permutation(u.max() + 1)                          # users in a fixed random order
    new = np.flatnonzero(np.isin(u, perm[:NEW_USERS]))                               # 300 users never used in any fold
    dev = np.flatnonzero(np.isin(u, perm[NEW_USERS:NEW_USERS + DEV_USERS]))           # 400 users the folds are built from
    return dict(u=u, i=i, r=r, ts=ts / 86400.0, age=age, fem=fem, occ=occ, yr=yr, new=new, dev=dev)  # timestamps in days


def ml_features(M, rows, fit_rows, use_time=True, use_user=True):  # features for `rows`, film statistics from `fit_rows` only
    ni = M["i"].max() + 1; g = M["r"][fit_rows].mean()                                 # films; the overall mean rating
    s = np.bincount(M["i"][fit_rows], M["r"][fit_rows], ni); c = np.bincount(M["i"][fit_rows], minlength=ni)  # per film
    film = (s + 10 * g) / (c + 10)                                                    # each film's mean, shrunk towards g
    i, u = M["i"][rows], M["u"][rows]                                                 # the rows' films and users
    cols = [film[i], c[i], M["yr"][i]] + ([M["age"][u], M["fem"][u], M["occ"][u]] if use_user else [])  # film facts; user facts
    return np.column_stack(cols + ([M["ts"][rows]] if use_time else []))             # and when the rating was made


def ml_scorer(M, depth, use_time=True, use_user=True):  # a fit_score function: RMSE of boosted trees fitted on tr
    def fs(tr, te):                                     # tr, te: positions in M["dev"] or absolute rows (see callers)
        f = lambda rows: ml_features(M, rows, tr, use_time, use_user)                  # the chosen features, film facts from tr
        m, _ = d109.fit_boost(f(tr), M["r"][tr], "squared", ROUNDS, RATE, depth, min_leaf=MIN_LEAF)  # fit
        return np.sqrt(np.mean((d109.predict_boost(m, f(te)) - M["r"][te]) ** 2))    # score
    return fs                                                                         # for cross_val


def ml_estimates(M, depth, use_time=True, use_user=True):  # shuffled and grouped CV over the dev users, and the new users' truth
    dev = M["dev"]; fs = ml_scorer(M, depth, use_time, use_user)                     # rows and scorer
    absf = lambda folds: [(dev[a], dev[b]) for a, b in folds]                         # fold positions -> row numbers
    sh = np.mean([cross_val(absf(shuffled_folds(len(dev), K, s)), fs)[0] for s in range(CV_SEEDS)])  # rows dealt at random
    gr = np.mean([cross_val(absf(group_folds(M["u"][dev], K, s)), fs)[0] for s in range(CV_SEEDS)])  # users dealt at random
    return sh, gr, fs(dev, M["new"])                                                  # and the truth: all dev users -> new users


def demo_grouped(M):                                    # section 4
    head("4. GROUPED DATA: RATINGS FOR NEW USERS (MovieLens, boosted trees, RMSE)")       # the banner
    print(f"  {len(M['dev'])} ratings from {DEV_USERS} users for the folds; {len(M['new'])} ratings from {NEW_USERS} new users, never in a fold")
    out = {}; t0 = time.time()                                                        # per depth; a clock
    for d in DEPTHS:                                                                  # each tree depth
        out[d] = ml_estimates(M, d)                                                   # shuffled, grouped, truth
        print(f"  depth {d}: shuffled rows {out[d][0]:.4f}, grouped by user {out[d][1]:.4f}, new users {out[d][2]:.4f}")
    for j, s in enumerate(("shuffled rows", "grouped by user", "new users")):         # which depth each would choose
        print(f"  choice by {s:15s}: depth {min(DEPTHS, key=lambda d: out[d][j])}")
    print(f"  ({time.time() - t0:.0f} s)")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_feature(M):                                    # section 5
    head("5. WHAT CARRIED THE LEAK? DROPPING THE TIMESTAMP, THEN THE USERS' OWN FACTS")   # the banner
    out = {}                                                                          # per (depth, feature set)
    for d in (2, 6):                                                                  # a shallow and a deep model
        for name, ut, uu in (("no timestamp", False, True), ("film facts only", False, False)):  # two reduced feature sets
            out[d, name] = ml_estimates(M, d, use_time=ut, use_user=uu)               # shuffled, grouped, truth
            v = out[d, name]                                                          # for printing
            print(f"  depth {d}, {name:15s}: shuffled rows {v[0]:.4f}, grouped by user {v[1]:.4f}, new users {v[2]:.4f}; "
                  f"grouped minus shuffled {v[1] - v[0]:+.4f}")
    dev = M["dev"]; same = np.mean(np.diff(M["u"][dev][np.argsort(M["ts"][dev])]) == 0)  # neighbours in time from the same user
    print(f"  ratings sorted by time: the next rating is by the same user {same:.3f} of the time")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(R):                                    # four panels from plain arrays
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); ax = ax.ravel()                   # two rows of two, read left to right
    a = ax[0]; S = R["sim"]; names = [n for n in S["trend"] if n != "truth"]          # simulated: estimates against truth
    x = np.arange(len(KINDS)); w = 0.16                                               # bar positions
    for j, n in enumerate(names + ["truth"]):                                         # each scheme, truth last
        a.bar(x + (j - 2) * w, [S[k][n] for k in KINDS], w, label=n, color="black" if n == "truth" else None)  # one bar per forecaster
    a.set_xticks(x); a.set_xticklabels(KINDS); a.set_ylabel("mean absolute error")    # axes
    a.set_title("simulated series: CV estimates against the true error"); a.legend(fontsize=8)  # labels
    a = ax[1]; t, y = d147.load_elec(); last = len(y) - HOLD_MONTHS - 1               # the electrical series
    a.plot(t.astype("datetime64[D]"), y, color="0.3", lw=1)                           # the series
    for o in range(last - 5 * H, last - H + 1, H):                                    # the forward folds
        a.axvspan(t[o + 1].astype("datetime64[D]"), t[o + H].astype("datetime64[D]"), color="tab:orange", alpha=0.25)  # a test year
    a.axvspan(t[last + 1].astype("datetime64[D]"), t[-1].astype("datetime64[D]"), color="tab:green", alpha=0.15)  # held out
    a.set_title("electrical equipment: forward test years (orange), held-out years (green)"); a.set_ylabel("orders index")  # title and axis
    a = ax[2]; G = R["grouped"]                                                       # MovieLens by depth
    for j, (n, c) in enumerate((("shuffled rows", "tab:red"), ("grouped by user", "tab:blue"), ("new users (truth)", "black"))):  # three estimates
        a.plot(DEPTHS, [G[d][j] for d in DEPTHS], "-o", color=c, label=n)             # one line per estimate
    a.set_xlabel("tree depth"); a.set_ylabel("RMSE"); a.set_title("MovieLens: new users' ratings"); a.legend(fontsize=8)  # axes and labels
    a = ax[3]; F = R["feature"]; x = np.arange(2)                                    # the leak for each feature set, two depths
    sets = [("all features", lambda d: G[d]), ("no timestamp", lambda d: F[d, "no timestamp"]),  # three feature sets
            ("film facts only", lambda d: F[d, "film facts only"])]                 # three feature sets
    for j, (n, get) in enumerate(sets):                                               # one bar group per feature set
        a.bar(x + (j - 1) * 0.25, [get(d)[1] - get(d)[0] for d in (2, 6)], 0.23, label=n)  # grouped minus shuffled
    a.set_xticks(x); a.set_xticklabels(["depth 2", "depth 6"]); a.axhline(0, color="0.5", lw=0.8)  # axes
    a.set_ylabel("grouped minus shuffled RMSE (the leak)"); a.set_title("what lets the trees recognise a user")  # axis and title
    a.legend(fontsize=8)                                                              # labels
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "cv_leakage_results.png"), dpi=110); plt.close(fig)  # save
    print("saved cv_leakage_results.png")


def draw_diagram():                                     # how each scheme cuts 60 rows: training, test, unused
    fig, ax = plt.subplots(figsize=(11, 5.6)); n = 60                                 # one panel; 60 rows in time order
    rows = [("shuffled", shuffled_folds(n, K, 0)[1]), ("blocked", blocked_folds(n, K)[2]),  # one fold of each scheme
            ("blocked, gap", blocked_folds(n, K, gap=4)[2]), ("forward", forward_folds(np.arange(n), [35], 12)[0])]  # one fold each
    g = np.repeat(np.arange(12), 5); grp = group_folds(g, K, 0)[1]                    # 12 users of 5 rows each
    rows.append(("grouped (users)", grp))                                             # every row of a test user is in test
    for k, (name, (tr, te)) in enumerate(rows):                                       # one strip per scheme
        yk = len(rows) - 1 - k                                                        # top to bottom
        c = np.full(n, "0.9", dtype=object); c[tr] = "tab:blue"; c[te] = "tab:orange"  # unused, training, test
        ax.bar(np.arange(n), 0.7, bottom=yk, color=list(c), width=0.92)               # one cell per row
        ax.text(-1.5, yk + 0.35, name, ha="right", va="center", fontsize=10)          # the scheme's name
    for x in range(5, n, 5):                                                          # user boundaries on the grouped strip
        ax.plot([x - 0.5, x - 0.5], [-0.12, 0.82], color="black", lw=1.2)             # a black line between users
    ax.set_xlim(-14, n); ax.set_ylim(-0.6, len(rows)); ax.axis("off")                 # room for names
    ax.text(n / 2, -0.45, "rows in time order (grouped: 12 users of 5 rows each)", ha="center", fontsize=9)  # the axis note
    for x0, (lab, col) in zip((0, 16, 32), (("training", "tab:blue"), ("test", "tab:orange"), ("not used", "0.9"))):  # a legend
        ax.bar(x0 + 1, 0.3, bottom=len(rows) - 0.15, color=col, width=2); ax.text(x0 + 3, len(rows), lab, va="center", fontsize=9)  # a legend entry
    ax.set_title("Day 172: one fold of each scheme", pad=18)                          # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "cv_schemes_diagram.png"), dpi=110); plt.close(fig)  # save
    print("saved cv_schemes_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    sim = demo_simulated()                                                            # 1
    real = demo_real()                                                                # 2 and 3
    M = movielens()                                                                   # the MovieLens rows
    grouped = demo_grouped(M)                                                         # 4
    feature = demo_feature(M)                                                         # 5
    print()
    plot_results(dict(sim=sim, real=real, grouped=grouped, feature=feature))          # plain arrays and dictionaries only
    draw_diagram()                                                                    # the diagram
    print(f"total time {time.time() - t0:.0f} s")
    head("WHAT COMES NEXT")                                                               # the banner
    print("Day 173: experiment tracking. Every run's settings, data version, seed, code version and")
    print("scores written down automatically, so a result can be found, compared and trusted later.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
