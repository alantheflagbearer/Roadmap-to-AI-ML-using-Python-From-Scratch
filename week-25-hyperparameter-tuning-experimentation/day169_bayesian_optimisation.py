"""
Day 169: Bayesian Optimisation from Scratch (Simplified)

Day 168 compared grid and random search: both choose every trial without
looking at the scores so far. Bayesian optimisation looks. A Gaussian process
(GP) learns the score as a function of the settings from the trials so far,
says how sure it is everywhere, and expected improvement (EI) picks the next
trial where a better score is most worth hoping for.

  1. a Gaussian process from scratch, checked by hand and on a 1-D test
     function, with its length scale chosen by marginal likelihood
  2. expected improvement: the closed form checked against simulation
  3. a known-truth benchmark (Hartmann-3, minimum -3.86278): Bayesian
     optimisation against random search and against a greedy rule
  4. Day 168's 200 real boosting configurations on Adult, recomputed: Bayesian
     optimisation choosing among them, against Day 168's random searches, with
     a shuffled-score control and a check of what the GP actually predicts
  5. Bayesian optimisation proposing new configurations anywhere, fitted for
     real, its winners checked on fresh rows, and what the GP costs

Data: Day 168's rows. Of Week 24's 24,130 fitting rows of UCI Adult, 6,000
train each configuration and 6,000 others score it (log loss); the 6,032
validation rows are kept fresh. Adult's test file is not touched.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import math                                            # math: erf, for the normal distribution
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: how long each step takes

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, HERE)                                                             # Day 168, in this folder
sys.path.insert(0, os.path.join(ROOT, "week-16-classical-ml-trees"))                 # Day 109's parts
sys.path.insert(0, os.path.join(ROOT, "week-24-model-evaluation-statistics"))        # Days 161 and 165
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day168_exercise as d168                      # Day 168's parts: scales, grids, random points, running best
    import day168_grid_vs_random_search as L168         # Day 168's lesson: evaluate(), one fit per configuration
    import day161_exercise as d161                      # Day 161: the Adult data and Week 24's split

ELLS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0)        # length scales the GP may choose from (the unit cube is 1 wide)
NOISE = 1e-4                                            # a small noise term on the standardised scores, for numerical safety
N0 = 5                                                  # random trials before the GP takes over
POOL = 200                                              # Day 168's pool of random configurations
BUDGET = 64                                             # the largest budget, as on Day 168
REPEATS = 100                                           # Bayesian optimisation runs per variant on the pool
CONT_RUNS = 3                                           # Bayesian optimisation runs that fit new configurations for real
CONT_BUDGET = 30                                        # trials in each of those runs
HMIN = -3.86278                                         # Hartmann-3's known minimum


# =============================================================================
# THE PARTS (copied unchanged into day169_exercise.py, for Day 174's build)
# =============================================================================
_erf = np.vectorize(math.erf)                           # math.erf, one value at a time, over arrays


def normal_cdf(z):                                      # P(Z <= z) for a standard normal Z
    return 0.5 * (1 + _erf(np.asarray(z, float) / np.sqrt(2)))                       # through the error function


def normal_pdf(z):                                      # the standard normal density
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)             # the bell curve


def matern52(A, B, ell):                                # similarity of every row of A to every row of B (1 = same point)
    d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0)) / ell  # distances, in length scales
    return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(-np.sqrt(5) * d)          # Matern 5/2: smooth, but less than Gaussian


def gp_fit(X, y, ell, noise=NOISE):                     # condition a GP on trials X with scores y
    m = y.mean(); sd = y.std() if y.std() > 0 else 1.0                                # standardise the scores
    z = (y - m) / sd                                                                  # mean 0, spread 1
    K = matern52(X, X, ell) + noise * np.eye(len(X))                                  # the trials' covariance
    L = np.linalg.cholesky(K)                                                         # K = L L^T
    a = np.linalg.solve(L.T, np.linalg.solve(L, z))                                   # K^-1 z, by two triangular solves
    lml = -0.5 * z @ a - np.log(np.diag(L)).sum() - 0.5 * len(X) * np.log(2 * np.pi)  # log marginal likelihood
    return dict(X=X, L=L, a=a, m=m, sd=sd, ell=ell, lml=lml)                          # everything prediction needs


def gp_predict(g, Xs):                                  # the GP's mean and standard deviation at new points
    Ks = matern52(Xs, g["X"], g["ell"])                                               # similarity of each new point to each trial
    mu = Ks @ g["a"]                                                                  # posterior mean (standardised)
    v = np.linalg.solve(g["L"], Ks.T)                                                 # L^-1 k for every new point
    var = np.maximum(1 - (v * v).sum(0), 1e-12)                                       # posterior variance: prior 1 minus what the trials explain
    return g["m"] + g["sd"] * mu, g["sd"] * np.sqrt(var)                              # back on the scores' scale


def fit_gp(X, y, ells=ELLS):                            # choose the length scale with the highest marginal likelihood
    return max((gp_fit(X, y, e) for e in ells), key=lambda g: g["lml"])               # one fit per candidate length scale


def expected_improvement(mu, sd, best):                 # how much below `best` the score is expected to fall (lower is better)
    z = (best - mu) / sd                                                              # standardised distance below the best so far
    return (best - mu) * normal_cdf(z) + sd * normal_pdf(z)                           # closed form for a normal belief


def propose(X, y, C):                                   # the candidate in C with the highest expected improvement
    g = fit_gp(X, y)                                                                  # the GP given the trials so far
    mu, sd = gp_predict(g, C)                                                         # its belief at every candidate
    return int(np.argmax(expected_improvement(mu, sd, y.min()))), g                   # the best candidate's index, and the GP


# =============================================================================
# TEST FUNCTIONS WITH KNOWN ANSWERS
# =============================================================================
def forrester(x):                                       # a 1-D test function on [0, 1]: minimum about -6.02 near x = 0.757
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)                                      # Forrester, Sobester and Keane (2008)


H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])                                             # Hartmann-3's constants: weights
H_A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35.]])            # widths
H_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828.]])  # centres


def hartmann3(X):                                       # a 3-D test function on the unit cube with four dips; minimum -3.86278
    return -np.sum(H_ALPHA * np.exp(-np.sum(H_A * (X[:, None, :] - H_P) ** 2, -1)), -1)  # a sum of four Gaussian-like dips


def candidates(X, y, rng, n=2000, near=500):            # where to look: random points, plus points near the best so far
    return np.vstack([rng.rand(n, X.shape[1]), np.clip(X[np.argmin(y)] + 0.05 * rng.randn(near, X.shape[1]), 0, 1)])  # inside the cube


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_gp():                                          # section 1
    print("\n" + "=" * 70)
    print("1. A GAUSSIAN PROCESS FROM SCRATCH")
    print("=" * 70)
    g = gp_fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), 1.0)                   # two trials, length scale 1
    mu, sd = gp_predict(g, np.array([[0.0], [0.5], [1.0]]))                          # at both trials and between them
    print(f"  two trials (0 -> 0, 1 -> 1), length scale 1: mean {np.round(mu, 4).tolist()}, sd {np.round(sd, 4).tolist()} "
          f"(by hand: 0, 0.5, 1; sd at 0.5 = 0.5 x sqrt(1 - 2 x 0.8287^2 / 1.5240) = 0.157)")
    X = np.array([[0.0], [0.4], [0.6], [1.0]]); y = forrester(X[:, 0])               # four trials of the 1-D test function
    for e in ELLS:                                                                    # each length scale
        print(f"  Forrester, 4 trials: length scale {e:4.2f}: log marginal likelihood {gp_fit(X, y, e)['lml']:8.3f}")
    g = fit_gp(X, y); xs = np.linspace(0, 1, 401)[:, None]; mu, sd = gp_predict(g, xs)  # the chosen GP, on a fine line
    print(f"  chosen length scale {g['ell']}; mean at the trials {np.round(gp_predict(g, X)[0], 3).tolist()} against "
          f"{np.round(y, 3).tolist()} (it passes through them)")
    print(f"  largest uncertainty {sd.max():.3f} at x = {xs[np.argmax(sd), 0]:.3f}; true minimum "
          f"{forrester(xs[:, 0]).min():.3f} at x = {xs[np.argmin(forrester(xs[:, 0])), 0]:.3f}")
    X = np.array([[0.1], [0.45], [0.9]]); y = forrester(X[:, 0])                      # Bayesian optimisation from three trials
    for _ in range(12):                                                               # twelve more trials
        i, g = propose(X, y, xs); X = np.vstack([X, xs[i]]); y = np.r_[y, forrester(xs[i:i + 1, 0])]  # choose by EI, run it
    print("  Bayesian optimisation from trials at 0.1, 0.45 and 0.9; trials 4 to 15 at " + ", ".join(f"{v:.3f}" for v in X[3:, 0]))  # where it looked
    print("  best score after each of trials 4 to 15: " + ", ".join(f"{v:.2f}" for v in d168.best_so_far(y)[3:]))  # what it had found
    print("Actual output from this lesson's run.")


def demo_ei():                                          # section 2
    print("\n" + "=" * 70)
    print("2. EXPECTED IMPROVEMENT: THE FORMULA AGAINST SIMULATION")
    print("=" * 70)
    rng = np.random.RandomState(0)                                                    # seeded draws
    for mu, sd, best in ((0.0, 1.0, 0.0), (1.0, 0.5, 0.0), (-0.5, 0.2, 0.0), (0.3, 2.0, 0.0)):  # beliefs at a point, and the best so far
        draws = mu + sd * rng.randn(1_000_000)                                        # possible scores under that belief
        mc = np.mean(np.maximum(best - draws, 0))                                     # average improvement over the best (0 if none)
        print(f"  mean {mu:+.1f}, sd {sd:.1f}, best {best:.1f}: formula {expected_improvement(mu, sd, best):.5f}, "
              f"simulation {mc:.5f}")
    print("  (the first: 0.39894 = 1 / sqrt(2 pi), by hand)")
    print("Actual output from this lesson's run.")


def bo_continuous(f, n, seed, acq="ei"):                # Bayesian optimisation on a cheap test function; returns best-so-far
    rng = np.random.RandomState(seed); X = rng.rand(N0, 3); y = f(X)                 # random start
    while len(y) < n:                                                                 # one trial at a time
        C = candidates(X, y, rng)                                                     # where to look
        if acq == "ei":                                                               # expected improvement
            i, _ = propose(X, y, C)                                                   # the EI choice
        else:                                                                         # greedy: the lowest predicted mean
            i = int(np.argmin(gp_predict(fit_gp(X, y), C)[0]))                        # no reward for uncertainty
        X = np.vstack([X, C[i]]); y = np.r_[y, f(C[i][None])]                         # run the trial
    return d168.best_so_far(y)                                                        # the best after each trial


def demo_benchmark():                                   # section 3
    print("\n" + "=" * 70)
    print("3. A KNOWN-TRUTH BENCHMARK: HARTMANN-3")
    print("=" * 70)
    print(f"  check: the function at its known minimiser {hartmann3(np.array([[0.114614, 0.555649, 0.852547]]))[0]:.5f}")
    t0 = time.time()                                                                  # a clock
    B = {a: np.array([bo_continuous(hartmann3, 40, k, a) for k in range(50)]) for a in ("ei", "mean")}  # 50 runs each
    rr = np.random.RandomState(1)                                                     # random search
    B["random"] = np.array([d168.best_so_far(hartmann3(d168.random_unit(40, 3, rr))) for _ in range(1000)])  # 1,000 runs
    for n in (10, 20, 30, 40):                                                        # budgets
        print(f"  {n} trials: median gap from the minimum: EI {np.median(B['ei'][:, n - 1] - HMIN):.4f}, greedy "
              f"{np.median(B['mean'][:, n - 1] - HMIN):.4f}, random {np.median(B['random'][:, n - 1] - HMIN):.4f}")
    for a in ("ei", "mean", "random"):                                                # stuck runs
        print(f"  {a:6s}: runs still more than 0.1 from the minimum after 40 trials: {np.mean(B[a][:, -1] - HMIN > 0.1):.2f}")
    print(f"  ({time.time() - t0:.0f} s)")
    print("Actual output from this lesson's run.")
    return B                                                                          # for the figure


def bo_pool(U, s, n, seed, acq="ei"):                   # Bayesian optimisation choosing among the pool's configurations
    rng = np.random.RandomState(seed); idx = list(rng.choice(len(U), N0, replace=False))  # random start
    while len(idx) < n:                                                               # one trial at a time
        rest = np.setdiff1d(np.arange(len(U)), idx)                                   # configurations not yet tried
        if acq == "ei":                                                               # expected improvement
            i, _ = propose(U[idx], s[idx], U[rest])                                   # the EI choice
        else:                                                                         # greedy, or exploration only
            mu, sd = gp_predict(fit_gp(U[idx], s[idx]), U[rest])                      # the GP's belief
            i = int(np.argmin(mu)) if acq == "mean" else int(np.argmax(sd))           # lowest mean, or most uncertain
        idx.append(int(rest[i]))                                                      # run it (its score is already known)
    return d168.best_so_far(s[idx])                                                   # the best after each trial


def demo_pool(D):                                       # section 4
    print("\n" + "=" * 70)
    print("4. DAY 168'S 200 REAL CONFIGURATIONS: BAYESIAN OPTIMISATION AGAINST RANDOM SEARCH")
    print("=" * 70)
    t0 = time.time()                                                                  # a clock
    U = d168.random_unit(POOL, 3, np.random.RandomState(1))                           # Day 168's pool, from the same seed
    res = L168.evaluate(d168.decode(U), D["X"], D["y"], D["A"], D["V"], D["F"])       # fitted again
    s = res[:, 0]                                                                     # tuning log loss
    print(f"  Day 168's pool recomputed in {time.time() - t0:.0f} s: best {s.min():.4f}, worst {s.max():.4f} "
          f"(Day 168 printed 0.2918 and 0.7371)")
    rng = np.random.RandomState(2); orders = [rng.permutation(POOL) for _ in range(1000)]  # Day 168's 1,000 random searches
    C = {"random": np.array([d168.best_so_far(s[o[:BUDGET]]) for o in orders])}      # their best-so-far curves
    t1 = time.time()                                                                  # a clock for the GP runs
    for a in ("ei", "mean", "sd"):                                                    # three ways to choose the next trial
        C[a] = np.array([bo_pool(U, s, BUDGET, k, a) for k in range(REPEATS)])        # 100 runs each
    sh = np.random.RandomState(9)                                                     # control: scores shuffled among configurations
    C["shuffled"] = np.array([bo_pool(U, s[sh.permutation(POOL)], BUDGET, k) for k in range(REPEATS)])  # nothing to learn
    print(f"  ({4 * REPEATS} Bayesian optimisation runs of {BUDGET} trials in {time.time() - t1:.0f} s)")
    names = dict(random="random search (1000)", ei="BO, expected improvement", mean="BO, greedy (lowest mean)",  # labels for the printout
                 sd="BO, explore only (largest sd)", shuffled="BO on shuffled scores")  # labels
    for a in ("random", "ei", "mean", "sd", "shuffled"):                              # each method
        c = C[a]                                                                      # its curves
        print(f"  {names[a]:30s}: median best at 8 / 16 / 27 / 64 trials " + " / ".join(f"{np.median(c[:, n - 1]):.4f}" for n in (8, 16, 27, 64))
              + f"; found the pool's best by 27: {np.mean(c[:, 26] <= s.min()):.2f}, by 64: {np.mean(c[:, 63] <= s.min()):.2f}")
    reach = lambda c: int(np.argmax(c <= s.min() + 0.001)) + 1 if np.any(c <= s.min() + 0.001) else BUDGET + 1  # trials until within 0.001 of the pool's best (65 = never)
    first, firstr = [reach(c) for c in C["ei"]], [reach(c) for c in C["random"]]  # for EI and for random search
    print(f"  trials until within 0.001 of the pool's best: EI median {np.median(first):.0f}, random median {np.median(firstr):.0f}")
    r2 = np.random.RandomState(7); pred = []                                          # what the GP knows: fit on 30 random configurations
    for _ in range(20):                                                               # 20 different sets of 30
        tr = r2.choice(POOL, 30, replace=False); te = np.setdiff1d(np.arange(POOL), tr)  # 30 to learn from, 170 to predict
        g = fit_gp(U[tr], s[tr]); mu, _ = gp_predict(g, U[te])                         # the GP's predictions
        rk = lambda v: np.argsort(np.argsort(v))                                       # ranks
        pred.append((np.corrcoef(mu, s[te])[0, 1], np.corrcoef(rk(mu), rk(s[te]))[0, 1], g["ell"], s[te][np.argmin(mu)] <= np.quantile(s[te], 0.1)))  # accuracy
    pred = np.array(pred, float)                                                      # 20 x 4
    print(f"  the GP fitted on 30 configurations, predicting the other 170 (20 repeats): correlation {pred[:, 0].mean():.2f}, "
          f"rank correlation {pred[:, 1].mean():.2f}, length scale chosen {np.median(pred[:, 2])}; its predicted best is in the true "
          f"best 10% in {pred[:, 3].mean():.2f} of repeats")
    print("Actual output from this lesson's run.")
    return U, res, C                                                                  # for section 5 and the figure


def demo_real(D, U, res):                               # section 5
    print("\n" + "=" * 70)
    print("5. BAYESIAN OPTIMISATION PROPOSING NEW CONFIGURATIONS, FITTED FOR REAL")
    print("=" * 70)
    X, y, A, V, F = D["X"], D["y"], D["A"], D["V"], D["F"]                            # the data and the three row sets
    runs = []                                                                         # one record per run
    for k in range(CONT_RUNS):                                                        # independent runs
        rng = np.random.RandomState(100 + k); Ut = d168.random_unit(N0, 3, rng)       # a random start
        R = L168.evaluate(d168.decode(Ut), X, y, A, V, F); seen = {}                  # fit the start; configurations already fitted
        for u, r in zip(d168.decode(Ut), R):                                          # remember them
            seen[tuple(u.values())] = r                                               # keyed by the decoded values
        gp_s, dup = 0.0, 0                                                            # seconds in the GP; wasted trials
        while len(Ut) < CONT_BUDGET:                                                  # one trial at a time
            t0 = time.time(); Cn = candidates(Ut, R[:, 0], rng)                       # where to look
            i, g = propose(Ut, R[:, 0], Cn); gp_s += time.time() - t0                 # the EI choice
            c = d168.decode(Cn[i:i + 1])[0]; key = tuple(c.values())                  # the configuration it decodes to
            if key in seen:                                                           # already fitted: depth and min leaf are whole numbers
                r = seen[key]; dup += 1                                               # reuse the result; the trial is still spent
            else:                                                                     # new
                r = L168.evaluate([c], X, y, A, V, F)[0]; seen[key] = r               # fit it
            Ut = np.vstack([Ut, Cn[i]]); R = np.vstack([R, r])                        # record the trial
        b = int(np.argmin(R[:, 0])); cb = d168.decode(Ut[b:b + 1])[0]                 # the run's winner
        runs.append(d168.best_so_far(R[:, 0]))                                        # its curve, for the figure
        print(f"  run {k + 1}: best of its {N0} random starting trials {R[:N0, 0].min():.4f}; best tuning log loss {R[b, 0]:.4f} at trial {b + 1} (learning rate {cb['learning rate']:.3g}, depth "
              f"{cb['depth']}, min leaf {cb['min leaf']}); fresh rows {R[b, 1]:.4f}; repeated configurations {dup}; "
              f"GP time {gp_s:.1f} s against fitting {R[:, 2].sum():.0f} s")
    rng = np.random.RandomState(2); best = np.array([res[rng.permutation(POOL)[:CONT_BUDGET], 0].min() for _ in range(1000)])  # random searches of 30
    print(f"  random search of {CONT_BUDGET} trials (from Day 168's pool): median {np.median(best):.4f}; "
          f"10% of searches reach {np.quantile(best, 0.1):.4f}; the pool's best {res[:, 0].min():.4f}, its fresh score "
          f"{res[np.argmin(res[:, 0]), 1]:.4f}")
    print("Actual output from this lesson's run.")
    return np.array(runs)                                                             # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def draw_diagram():                                     # Bayesian optimisation on the 1-D test function, step by step
    X = np.array([[0.1], [0.45], [0.9]]); y = forrester(X[:, 0])                      # three starting trials
    xs = np.linspace(0, 1, 401)[:, None]                                              # a fine line
    fig, ax = plt.subplots(2, 4, figsize=(19, 6.6), gridspec_kw=dict(height_ratios=[2.2, 1]))  # GP above, EI below
    for col, n in enumerate((3, 6, 8, 11)):                                           # four moments of the search
        while len(y) < n:                                                             # run the search up to n trials
            i, _ = propose(X, y, xs); X = np.vstack([X, xs[i]]); y = np.r_[y, forrester(xs[i:i + 1, 0])]  # choose by EI, run it
        i, g = propose(X, y, xs); mu, sd = gp_predict(g, xs)                          # the GP and its choice
        e = expected_improvement(mu, sd, y.min())                                     # EI along the line
        a = ax[0, col]; a.plot(xs, forrester(xs[:, 0]), "k:", lw=1, label="true score (unknown)")  # the truth
        a.plot(xs, mu, color="tab:blue", label="GP mean"); a.fill_between(xs[:, 0], mu - 2 * sd, mu + 2 * sd, color="tab:blue", alpha=0.2, label="mean +/- 2 sd")  # belief
        a.plot(X[:, 0], y, "ko", label="trials so far"); a.axvline(xs[i, 0], color="tab:red", lw=1)  # trials and the next choice
        a.set_ylim(-12, 18); a.set_title(f"after {len(y)} trials (length scale {g['ell']})", fontsize=11)  # title
        if col == 0: a.legend(fontsize=8, loc="upper left")                           # one legend
        b = ax[1, col]; b.plot(xs, e, color="tab:red"); b.axvline(xs[i, 0], color="tab:red", lw=1)  # EI and its peak
        b.set_xlabel("the setting"); b.set_ylabel("expected improvement" if col == 0 else "")  # labels
        b.set_title(f"next trial at x = {xs[i, 0]:.3f}", fontsize=10)                 # where it goes
    fig.suptitle("Day 169: the GP learns from the trials; expected improvement chooses the next one", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "bayes_opt_diagram.png"), dpi=110); plt.close(fig)  # save
    print("saved bayes_opt_diagram.png")


def plot_results(R):                                    # three panels from plain arrays, so a wrapper can save and redraw them
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))                                   # one row
    a = ax[0]; x = np.arange(1, 41)                                                   # Hartmann-3
    for k, c, lab in (("random", "tab:orange", "random"), ("mean", "tab:green", "BO, greedy"), ("ei", "tab:blue", "BO, EI")):  # three methods
        a.plot(x, np.median(R["bench"][k], 0) - HMIN, color=c, label=lab)            # median gap
        a.fill_between(x, np.quantile(R["bench"][k], 0.1, 0) - HMIN, np.quantile(R["bench"][k], 0.9, 0) - HMIN, color=c, alpha=0.15)  # 10-90%
    a.set_yscale("log"); a.set_xlabel("trials"); a.set_ylabel("gap from the known minimum"); a.legend(); a.set_title("Hartmann-3 (median, 10-90%)")  # labels
    a = ax[1]; x = np.arange(1, BUDGET + 1)                                           # the pool
    for k, c, lab in (("random", "tab:orange", "random"), ("shuffled", "tab:gray", "BO, shuffled scores"), ("sd", "tab:purple", "BO, explore only"),  # the curves to draw
                      ("mean", "tab:green", "BO, greedy"), ("ei", "tab:blue", "BO, EI")):  # five curves
        a.plot(x, np.median(R["pool"][k], 0), color=c, label=lab)                     # median best so far
    a.axhline(R["pool_best"], color="k", lw=0.8, ls=":"); a.set_xscale("log")         # the pool's best
    a.set_ylim(R["pool_best"] - 0.001, R["pool_best"] + 0.02); a.set_xlabel("trials"); a.set_ylabel("best log loss so far (median)")  # labels
    a.legend(fontsize=9); a.set_title("Day 168's 200 configurations on Adult")       # title
    a = ax[2]; x = np.arange(1, CONT_BUDGET + 1)                                      # fitted for real
    for k, c in enumerate(R["runs"]):                                                 # each run
        a.plot(x, c, color="tab:blue", alpha=0.8, label="BO, EI, new configurations" if k == 0 else None)  # its best so far
    a.plot(x, np.median(R["pool"]["random"][:, :CONT_BUDGET], 0), color="tab:orange", label="random (median)")  # random search
    a.axhline(R["pool_best"], color="k", lw=0.8, ls=":", label="pool's best")         # the pool's best
    a.set_ylim(R["pool_best"] - 0.003, R["pool_best"] + 0.03); a.set_xlabel("trials"); a.set_ylabel("best tuning log loss so far")  # labels
    a.legend(fontsize=9); a.set_title("proposing anywhere, fitted for real")         # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "bayes_opt_results.png"), dpi=110); plt.close(fig)  # save
    print("saved bayes_opt_results.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    tr, _ = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Week 24's data and split (the test file is not used)
    ref = {k: v[fi] for k, v in tr.items()}                                           # category codes from the fitting rows
    p = np.random.RandomState(0).permutation(fi)                                      # the fitting rows, shuffled as on Day 168
    D = dict(X=d161.tree_matrix(tr, ref), y=tr["y"], A=p[:L168.N_TUNE], V=p[L168.N_TUNE:2 * L168.N_TUNE], F=va)  # Day 168's rows
    demo_gp()                                                                         # 1
    demo_ei()                                                                         # 2
    bench = demo_benchmark()                                                          # 3
    U, res, C = demo_pool(D)                                                          # 4
    runs = demo_real(D, U, res)                                                       # 5
    print()
    plot_results(dict(bench=bench, pool=C, pool_best=res[:, 0].min(), runs=runs))     # plain arrays only
    draw_diagram()                                                                    # the diagram
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 170: learning-rate finders. Raise the learning rate step by step during one short run and")
    print("read a good value off the loss curve, then check that choice against full training runs.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
