"""
Day 168: Grid Search vs Random Search -- a Real Efficiency Comparison

Week 25 tunes models. Before anything clever, the two simplest ways to choose
hyperparameters: try every point of a grid, or try random points. Today
compares them honestly:

  1. a known-truth benchmark: functions where we choose how many settings
     matter, so the best possible answer is known exactly
  2. the real thing: Day 109's gradient boosting on UCI Adult, three
     hyperparameters, grids of 8, 27 and 64 points against random search with
     the same budgets
  3. which hyperparameters mattered, and how many distinct values of each a
     grid and a random search actually try
  4. what each search costs in time
  5. the winner's curse: the best configuration's score on the rows used to
     choose it, against its score on fresh rows

Data: Week 24's split of Adult's training file. From the 24,130 fitting rows,
6,000 train every configuration and 6,000 others score it; the 6,032
validation rows are kept fresh to check the winner. Adult's test file was used
once on Day 167 and is not touched.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import itertools                                       # itertools: product, every combination of grid values
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: how long each fit takes

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, os.path.join(ROOT, "week-16-classical-ml-trees"))                 # Day 109's parts
sys.path.insert(0, os.path.join(ROOT, "week-24-model-evaluation-statistics"))        # Days 161 and 165
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day109_exercise as d109                      # Day 109: gradient boosting
    import day161_exercise as d161                      # Day 161: the Adult data and Week 24's split
    import day165_exercise as d165                      # Day 165: log loss

SPACE = (("learning rate", "log", 0.01, 1.0),           # the search space: name, scale, lowest, highest
         ("depth", "int", 1, 8),                        # tree depth: whole numbers 1 to 8
         ("min leaf", "logint", 1, 300))                # fewest rows a leaf may hold: whole numbers on a log scale
ROUNDS = 100                                            # trees per model, fixed today (Day 171 tunes when to stop)
N_TUNE = 6000                                           # rows that train each configuration, and rows that score it
POOL = 200                                              # random configurations evaluated once and reused by every random search
BUDGETS = (8, 27, 64)                                   # trials per search: 2, 3 and 4 values per hyperparameter for the grid
SHIFTS = 5                                              # control: grids moved by random amounts within their slices
REPEATS = 1000                                          # random orders of the pool (real data) and repeats (benchmark)


# =============================================================================
# THE PARTS (copied unchanged into day168_exercise.py, for Day 174's build)
# =============================================================================
def to_value(u, kind, lo, hi):                          # a number in [0, 1) -> a hyperparameter value on its scale
    if kind == "log":                                                                 # spread evenly on a log scale
        return float(lo * (hi / lo) ** u)                                             # u = 0 gives lo, u -> 1 gives hi
    if kind == "int":                                                                 # whole numbers, each equally likely
        return int(min(hi, np.floor(lo + u * (hi - lo + 1))))                         # hi - lo + 1 equal slices of [0, 1)
    return int(min(hi, np.floor(np.exp(np.log(lo) + u * (np.log(hi + 1) - np.log(lo))))))  # logint: whole numbers, log-spaced


def grid_unit(k, d, shift=None):                        # a k x k x ... grid in the unit cube; shift moves each axis within its slices
    s = np.zeros(d) if shift is None else np.asarray(shift, float)                    # no shift: the centre of each of k slices
    axes = [(np.arange(k) + 0.5 + s[j]) / k for j in range(d)]                        # e.g. k = 3, no shift: 1/6, 1/2, 5/6
    return np.array(list(itertools.product(*axes)))                                   # every combination: k**d rows


def random_unit(n, d, rng):                             # n random points in the unit cube
    return rng.rand(n, d)                                                             # each coordinate uniform on [0, 1)


def decode(U, space=SPACE):                             # unit-cube points -> configurations (one dictionary each)
    return [{nm: to_value(u, kind, lo, hi) for u, (nm, kind, lo, hi) in zip(row, space)} for row in U]  # one value per hyperparameter


def best_so_far(scores):                                # the best (lowest) score after each trial
    return np.minimum.accumulate(np.asarray(scores, float))                           # running minimum


def importance(U, scores, bins=5):                      # how much of the scores' spread each hyperparameter explains on its own
    s = np.asarray(scores, float); tot = np.sum((s - s.mean()) ** 2)                  # the total spread
    out = []                                                                          # one share per hyperparameter
    for j in range(U.shape[1]):                                                       # each hyperparameter
        b = np.minimum((U[:, j] * bins).astype(int), bins - 1)                        # its slice of [0, 1)
        m = np.array([s[b == i].mean() if np.any(b == i) else 0.0 for i in range(bins)])  # mean score per slice
        n = np.bincount(b, minlength=bins)                                            # rows per slice
        out.append(float(np.sum(n * (m - s.mean()) ** 2) / tot))                      # between-slice spread / total spread (eta squared)
    return np.array(out)                                                              # 0 = no effect, 1 = explains everything


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_benchmark():                                   # section 1: known truth
    print("\n" + "=" * 70)
    print("1. A KNOWN-TRUTH BENCHMARK: HOW MANY SETTINGS MATTER?")
    print("=" * 70)
    rng = np.random.RandomState(0); out = {}                                          # a seeded generator; results for the figure
    for name, w in (("one of three matters", np.array([1.0, 0.01, 0.01])), ("all three matter equally", np.ones(3))):  # two kinds of problem
        for n in BUDGETS:                                                             # each budget
            k = round(n ** (1 / 3)); G = grid_unit(k, 3)                              # the grid with this many points
            rg, rr = [], []                                                           # best found, per repeat
            for _ in range(REPEATS):                                                  # a new problem each time
                c = rng.rand(3)                                                       # where the best setting is, unknown to the search
                f = lambda P: np.sum(w * (P - c) ** 2, 1)                             # the score: weighted squared distance from it (0 is perfect)
                rg.append(f(G).min()); rr.append(f(random_unit(n, 3, rng)).min())     # the best each search finds
            rg, rr = np.array(rg), np.array(rr); out[(name, n)] = (rg.mean(), rr.mean())  # averages, for the figure
            print(f"  {name:25s}, {n:2d} trials: mean shortfall grid {rg.mean():.5f}, random {rr.mean():.5f}; "
                  f"random better in {np.mean(rr < rg):.3f} of {REPEATS} problems")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def evaluate(configs, X, y, A, V, F):                   # train each configuration on rows A; score it on rows V and on fresh rows F
    res = []                                                                          # one result per configuration
    with np.errstate(over="ignore"):                                                  # a huge log-odds overflows exp; sigmoid's limit (0 or 1) is still right
        for c in configs:                                                                 # each configuration
            t0 = time.time()                                                          # a clock
            _, E = d109.fit_boost(X[A], y[A], "log", ROUNDS, c["learning rate"], c["depth"], evals=(X[V], X[F]), min_leaf=c["min leaf"])  # Day 109's boosting
            res.append((d165.log_loss(y[V], d109.sigmoid(E[0][-1])), d165.log_loss(y[F], d109.sigmoid(E[1][-1])), time.time() - t0))  # tuning score, fresh score, seconds
    return np.array(res)                                                              # configurations x (tuning loss, fresh loss, seconds)


def demo_real(D):                                       # section 2: boosting on Adult
    print("\n" + "=" * 70)
    print("2. THE REAL THING: GRID AGAINST RANDOM SEARCH FOR BOOSTING ON ADULT")
    print("=" * 70)
    X, y, A, V, F = D["X"], D["y"], D["A"], D["V"], D["F"]                            # the data and the three row sets
    print(f"  {len(A)} rows train each configuration, {len(V)} score it (log loss), {len(F)} fresh rows are kept for section 5; {ROUNDS} trees")
    t0 = time.time(); G = {}                                                          # the grids
    for n in BUDGETS:                                                                 # each budget
        U = grid_unit(round(n ** (1 / 3)), 3); G[n] = (U, decode(U), evaluate(decode(U), X, y, A, V, F))  # points, configurations, results
        vals = [sorted({c[nm] for c in G[n][1]}) for nm, _, _, _ in SPACE]            # the values each hyperparameter takes
        b = int(np.argmin(G[n][2][:, 0]))                                             # the grid's best point
        print(f"  grid of {n:2d}: values {vals}; best log loss {G[n][2][b, 0]:.4f} at {G[n][1][b]}")
    U = random_unit(POOL, 3, np.random.RandomState(1)); P = (U, decode(U), evaluate(decode(U), X, y, A, V, F))  # the random pool
    print(f"  ({POOL} random configurations and {sum(BUDGETS)} grid points evaluated in {time.time() - t0:.0f} s)")
    b = np.argsort(P[2][:, 0])                                                        # the pool, best first
    for i in b[:3]:                                                                   # its three best
        print(f"  pool's best: log loss {P[2][i, 0]:.4f} at " + ", ".join(f"{k} {v:.3g}" for k, v in P[1][i].items()))
    print(f"  pool's worst: log loss {P[2][b[-1], 0]:.4f} at " + ", ".join(f"{k} {v:.3g}" for k, v in P[1][b[-1]].items()))
    rng = np.random.RandomState(2); orders = [rng.permutation(POOL) for _ in range(REPEATS)]  # random searches: random orders of the pool
    S = {}                                                                            # best-so-far curves, for the figure
    for n in BUDGETS:                                                                 # each budget
        best = np.array([P[2][o[:n], 0].min() for o in orders])                       # each random search's best after n trials
        g = G[n][2][:, 0].min()                                                       # the grid's best
        print(f"  {n:2d} trials: grid {g:.4f}; random median {np.median(best):.4f} (2.5% to 97.5%: {np.quantile(best, 0.025):.4f} to "
              f"{np.quantile(best, 0.975):.4f}); random at least as good in {np.mean(best <= g):.3f} of {REPEATS} searches")
    S["random"] = np.array([best_so_far(P[2][o[:max(BUDGETS)], 0]) for o in orders])  # every search's curve up to 64 trials
    S["grid"] = np.array([best_so_far(G[max(BUDGETS)][2][o, 0]) for o in (rng.permutation(max(BUDGETS)) for _ in range(REPEATS))])  # the 64-point grid in random orders
    sh = np.random.RandomState(5).rand(SHIFTS, 3) - 0.5                              # control: grids of 27 moved by random amounts within their slices
    sg = [evaluate(decode(grid_unit(3, 3, s)), X, y, A, V, F)[:, 0].min() for s in sh]  # each moved grid's best
    best27 = np.array([P[2][o[:27], 0].min() for o in orders])                        # random searches of 27, as above
    print(f"  {SHIFTS} shifted grids of 27: best " + ", ".join(f"{v:.4f}" for v in sg) + f" (mean {np.mean(sg):.4f}; the centred grid {G[27][2][:, 0].min():.4f}); "
          f"random of 27 at least as good as a shifted grid in {np.mean([np.mean(best27 <= v) for v in sg]):.3f} of comparisons")
    print(f"  the best of all {POOL + sum(BUDGETS)} configurations: {min(P[2][:, 0].min(), *(G[n][2][:, 0].min() for n in BUDGETS)):.4f}")
    print("Actual output from this lesson's run.")
    return G, P, S                                                                    # for the next sections and the figure


def demo_importance(G, P):                              # section 3
    print("\n" + "=" * 70)
    print("3. WHICH HYPERPARAMETERS MATTERED, AND HOW MANY VALUES EACH SEARCH TRIED")
    print("=" * 70)
    imp = importance(P[0], P[2][:, 0])                                                # share of spread explained, from the random pool
    for (nm, _, _, _), v in zip(SPACE, imp):                                          # each hyperparameter
        print(f"  {nm:13s}: explains {v:.3f} of the spread in log loss on its own")
    s = P[2][:, 0]; c = np.minimum((P[0][:, 0] * 4).astype(int), 3) * 4 + np.minimum((P[0][:, 1] * 4).astype(int), 3)  # 4 x 4 cells of learning rate and depth
    m = np.array([s[c == i].mean() for i in range(16)]); n = np.bincount(c, minlength=16)  # mean score and count per cell
    print(f"  learning rate and depth together (4 x 4 cells): explain {np.sum(n * (m - s.mean()) ** 2) / np.sum((s - s.mean()) ** 2):.3f}")
    keep = P[2][:, 0] < np.quantile(P[2][:, 0], 0.5)                                  # the better half of the pool
    lr = np.array([c["learning rate"] for c in P[1]])                                 # each configuration's learning rate
    print(f"  in the better half of the pool, learning rate ran from {lr[keep].min():.3f} to {lr[keep].max():.3f}; "
          f"outside it from {lr[~keep].min():.3f} to {lr[~keep].max():.3f}")
    for n in BUDGETS:                                                                 # distinct values tried
        print(f"  {n:2d} trials: grid tries {round(n ** (1 / 3))} values of each hyperparameter; random tries "
              + ", ".join(f"{len({c[nm] for c in P[1][:n]})} {nm}" for nm, _, _, _ in SPACE) + " (first random order)")
    print("Actual output from this lesson's run.")
    return imp                                                                        # for the figure


def demo_cost(G, P):                                    # section 4
    print("\n" + "=" * 70)
    print("4. WHAT EACH SEARCH COSTS")
    print("=" * 70)
    sec = P[2][:, 2]; dep = np.array([c["depth"] for c in P[1]])                      # seconds per fit; depth per fit
    for d_ in (1, 4, 8):                                                              # a few depths
        print(f"  depth {d_}: {sec[dep == d_].mean():.2f} s per configuration on average ({int(np.sum(dep == d_))} in the pool)")
    for n in BUDGETS:                                                                 # each budget
        print(f"  {n:2d} trials: grid {G[n][2][:, 2].sum():.0f} s; random {sec[:n].sum():.0f} s (first random order)")
    print(f"  pool of {POOL}: {sec.sum():.0f} s in all")
    print("Actual output from this lesson's run.")


def demo_fresh(G, P):                                   # section 5
    print("\n" + "=" * 70)
    print("5. THE WINNER'S CURSE: THE BEST SCORE ON ITS OWN ROWS AGAINST FRESH ROWS")
    print("=" * 70)
    tune, fresh = P[2][:, 0], P[2][:, 1]                                              # every pool configuration's two scores
    print(f"  over all {POOL} configurations: tuning and fresh log loss correlate {np.corrcoef(tune, fresh)[0, 1]:.3f}; "
          f"mean fresh - tuning {np.mean(fresh - tune):+.4f}")
    rng = np.random.RandomState(3); out = {}                                          # random searches again; results
    for n in BUDGETS:                                                                 # each budget
        w = [o[np.argmin(tune[o])] for o in (rng.permutation(POOL)[:n] for _ in range(REPEATS))]  # the winner of each random search
        gap = fresh[w] - tune[w]; out[n] = gap                                        # how much worse it does on fresh rows
        g = int(np.argmin(G[n][2][:, 0]))                                             # the grid's winner
        print(f"  {n:2d} trials: random winner's fresh - tuning {np.mean(gap):+.4f} on average, {np.mean(gap) - np.mean(fresh - tune):+.4f} beyond the average "
              f"configuration's (fresh log loss {np.mean(fresh[w]):.4f}); "
              f"grid winner {G[n][2][g, 1] - G[n][2][g, 0]:+.4f} (fresh {G[n][2][g, 1]:.4f})")
    i = int(np.argmin(tune)); r = np.argsort(np.argsort(fresh))[i] + 1               # the pool's overall winner; its rank on fresh rows
    print(f"  the pool's winner on tuning rows ranks {r} of {POOL} on fresh rows (fresh {fresh[i]:.4f}; best on fresh {fresh.min():.4f})")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(R):                                    # six panels from plain arrays, so a wrapper can save and redraw them
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # a 2 x 3 grid
    for a, name in zip(ax[0, :2], ("one of three matters", "all three matter equally")):  # the benchmark, two kinds of problem
        a.plot(BUDGETS, [R["bench"][(name, n)][0] for n in BUDGETS], "s-", label="grid")  # grid's mean shortfall
        a.plot(BUDGETS, [R["bench"][(name, n)][1] for n in BUDGETS], "o-", label="random")  # random's
        a.set_xscale("log"); a.set_yscale("log"); a.minorticks_off(); a.set_xticks(BUDGETS); a.set_xticklabels(BUDGETS)  # log axes, budget ticks only
        a.set_xlabel("trials"); a.set_ylabel("mean shortfall from the best"); a.legend(); a.set_title(f"benchmark: {name}")  # labels
    a = ax[0, 2]; x = np.arange(1, R["curves_r"].shape[1] + 1)                        # best-so-far on Adult
    for k, c, lab in (("curves_r", "tab:orange", "random"), ("curves_g", "tab:blue", "grid of 64, random order")):  # both searches
        m, lo, hi = np.median(R[k], 0), np.quantile(R[k], 0.1, 0), np.quantile(R[k], 0.9, 0)  # median and middle 80%
        a.plot(x, m, color=c, label=lab); a.fill_between(x, lo, hi, color=c, alpha=0.2)  # curve and band
    for n in BUDGETS:                                                                 # each fixed grid
        a.plot(n, R["grid_best"][n], "k*", ms=11, label="grid of 8, 27, 64 (in full)" if n == BUDGETS[0] else None)  # its final best
    a.set_xscale("log"); a.set_xlabel("trials"); a.set_ylabel("best log loss so far"); a.legend(fontsize=9)  # labels
    a.set_title("boosting on Adult: best found so far (median, 10-90%)")              # title
    a = ax[1, 0]; U, s = R["pool_u"], R["pool_tune"]                                  # the pool: learning rate against score
    lr = [to_value(u, *SPACE[0][1:]) for u in U[:, 0]]                                # learning rates
    sc = a.scatter(lr, s, c=[to_value(u, *SPACE[1][1:]) for u in U[:, 1]], cmap="viridis", s=18)  # coloured by depth
    a.set_xscale("log"); a.set_xlabel("learning rate"); a.set_ylabel("tuning log loss"); fig.colorbar(sc, ax=a, label="depth")  # labels
    a.set_title(f"the random pool ({POOL} configurations)")                           # title
    a = ax[1, 1]; a.bar([nm for nm, _, _, _ in SPACE], R["imp"], color="tab:green")  # importance
    a.set_ylabel("share of spread explained alone"); a.set_ylim(0, 1); a.set_title("which hyperparameters mattered")  # labels
    a = ax[1, 2]; a.scatter(R["pool_tune"], R["pool_fresh"], s=14, color="tab:gray")  # tuning against fresh, every configuration
    lo, hi = min(R["pool_tune"].min(), R["pool_fresh"].min()), max(R["pool_tune"].max(), R["pool_fresh"].max())  # range of both
    a.plot([lo, hi], [lo, hi], "k:", lw=1); a.set_xscale("log"); a.set_yscale("log")  # the diagonal; log axes
    a.set_xlabel("log loss on tuning rows"); a.set_ylabel("log loss on fresh rows"); a.set_title("tuning score against fresh score")  # labels
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "grid_random_results.png"), dpi=110); plt.close(fig)  # save
    print("saved grid_random_results.png")


def draw_diagram():                                     # the same nine trials, as a grid and at random
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))                                   # two panels
    rng = np.random.RandomState(4)                                                    # a seeded generator
    for a, (title, P) in zip(ax, (("grid: 9 trials, 3 values of each setting", grid_unit(3, 2)),  # a 3 x 3 grid
                                  ("random: 9 trials, 9 values of each (one draw; note its gaps)", random_unit(9, 2, rng)))):  # 9 random points
        xs = np.linspace(0, 1, 200); a.plot(xs, -0.08 - 0.07 * np.exp(-((xs - 0.37) / 0.08) ** 2), color="tab:red")  # the score along the setting that matters
        a.scatter(P[:, 0], P[:, 1], s=60, color="tab:blue", zorder=3)                 # the trials
        for p in P:                                                                   # each trial's projection onto the axis that matters
            a.plot([p[0], p[0]], [p[1], -0.08], color="tab:blue", lw=0.6, alpha=0.5)  # a faint line down to it
        a.set_xlim(0, 1); a.set_ylim(-0.17, 1); a.set_yticks([0, 0.5, 1])             # the unit square, with room for the curve
        a.set_xlabel("the setting that matters"); a.set_ylabel("a setting that hardly matters")  # axis labels
        a.set_title(title, fontsize=11)                                               # title
    fig.suptitle("Day 168: when one setting matters, a grid tests only 3 of its values; random search tests 9", fontsize=12)  # overall title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "grid_random_diagram.png"), dpi=110); plt.close(fig)  # save
    print("saved grid_random_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    tr, _ = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Week 24's data and split (the test file is not used)
    ref = {k: v[fi] for k, v in tr.items()}                                           # category codes from the fitting rows
    p = np.random.RandomState(0).permutation(fi)                                      # the fitting rows, shuffled once
    D = dict(X=d161.tree_matrix(tr, ref), y=tr["y"], A=p[:N_TUNE], V=p[N_TUNE:2 * N_TUNE], F=va)  # matrix, targets, train / score / fresh rows
    bench = demo_benchmark()                                                          # 1
    G, P, S = demo_real(D)                                                            # 2
    imp = demo_importance(G, P)                                                       # 3
    demo_cost(G, P)                                                                   # 4
    demo_fresh(G, P)                                                                  # 5
    print()
    plot_results(dict(bench=bench, curves_r=S["random"], curves_g=S["grid"], grid_best={n: G[n][2][:, 0].min() for n in BUDGETS},  # plain arrays only
                      pool_u=P[0], pool_tune=P[2][:, 0], pool_fresh=P[2][:, 1], imp=imp))  # the pool's points and both scores, and the importances
    draw_diagram()                                                                    # the diagram
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 169: Bayesian optimisation from scratch. A Gaussian process learns the score from the trials")
    print("so far and picks the next one, against today's random search on the same budget.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
