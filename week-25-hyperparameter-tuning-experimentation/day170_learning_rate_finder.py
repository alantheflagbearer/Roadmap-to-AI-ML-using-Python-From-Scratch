"""
Day 170: Learning-Rate Finders

The learning rate is usually the most important setting of a network trained
by gradient steps (Day 168 found the same for boosting). A learning-rate
finder (L. Smith, 2015; popularised by fast.ai) spends one short run on it:
start tiny, multiply the learning rate a little every step, record the loss,
stop when the loss blows up, and read a good value off the curve. Today builds
it from scratch and checks it against the real thing, full training runs:

  1. a known truth: on least squares, gradient descent diverges exactly above
     2 / (largest eigenvalue); where does the finder say it blows up?
  2. the finder on a neural network for UCI Adult, with Adam and with plain
     SGD, over five seeds, and two rules for reading the curve
  3. the check: 13 learning rates x 3 seeds of real training, scored on
     validation rows, and real training at each suggestion
  4. does the finder follow a change of batch size?
  5. what the finder costs against the sweep it replaces

Data: Week 24's split of UCI Adult: the 24,130 fitting rows train, the 6,032
validation rows score. Adult's test file is not touched. The network is Day
126's (two ReLU layers, He initialisation, Day 37's Adam), imported unchanged.

Every line of code below carries its own comment, continuing Day 52's
convention.
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
sys.path.insert(0, os.path.join(ROOT, "week-19-unsupervised-learning-ii"))           # Day 126's network
sys.path.insert(0, os.path.join(ROOT, "week-24-model-evaluation-statistics"))        # Day 161's data
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, sigmoid, Adam
    import day161_exercise as d161                      # Day 161: the Adult data and Week 24's split

HIDDEN = (64, 32)                                       # the network's two hidden layers
EPOCHS = 5                                              # passes over the training rows in each real run
BATCH = 128                                             # rows per step
FIND_STEPS = 300                                        # steps in one finder run
LO, HI = 1e-6, 10.0                                     # the finder's range of learning rates
SEEDS = 5                                               # finder runs per optimiser
GRID = 10.0 ** np.arange(-5, 1.01, 0.5)                 # 13 learning rates for the real check, 1e-5 to 10
SWEEP_SEEDS = 3                                         # real training runs per learning rate


# =============================================================================
# THE PARTS (copied unchanged into day170_exercise.py, for Day 174's build)
# =============================================================================
def lr_range(lo, hi, steps):                            # learning rates rising by the same factor every step
    return lo * (hi / lo) ** (np.arange(steps) / (steps - 1))                         # lo, ..., hi, evenly spaced on a log scale


def lr_find(step, lrs, beta=0.98, blowup=4.0):          # one finder run: step(lr) trains one step and returns that batch's loss
    raw, sm, avg, best = [], [], 0.0, np.inf                                          # losses as measured; smoothed; running average; best smoothed
    for t, lr in enumerate(lrs):                                                      # rising learning rates
        loss = step(lr); avg = beta * avg + (1 - beta) * loss                         # one step; exponential moving average
        s = avg / (1 - beta ** (t + 1))                                               # corrected for starting the average at 0
        raw.append(loss); sm.append(s)                                                # record both
        if not np.isfinite(s) or s > blowup * best:                                   # the loss has blown up
            break                                                                     # stop: higher rates only get worse
        best = min(best, s)                                                           # the best smoothed loss so far
    return np.asarray(lrs[:len(sm)]), np.array(raw), np.array(sm)                     # the rates used and both loss curves


def suggest(lrs, sm, skip=10):                          # two common ways to read the curve
    g = np.gradient(sm, np.log(lrs))                                                  # slope of the smoothed loss against log learning rate
    steep = lrs[skip + int(np.argmin(g[skip:-5]))] if len(sm) > skip + 6 else lrs[0]  # where it falls fastest (ignoring both ends)
    return dict(min_over_10=float(lrs[int(np.argmin(sm))] / 10), steepest=float(steep))  # one tenth of the minimum's rate; the steepest point


class SGD:                                              # plain stochastic gradient descent, with the same interface as Day 126's Adam
    def __init__(self, net, lr=1e-2):                   # net is not needed: SGD keeps no memory
        self.lr = lr                                                                  # the learning rate, which the finder changes each step

    def step(self, net, grads):                         # one step against the gradient
        for (W, b), (gW, gb) in zip(net, grads):                                      # each layer
            W -= self.lr * gW; b -= self.lr * gb                                      # in place, so the network itself changes


def bce_logits(L, y):                                   # log loss on a network's raw outputs (logits), and its gradient
    z = L[:, 0]                                                                       # one logit per row
    loss = float(np.mean(np.logaddexp(0, z) - y * z))                                 # log(1 + e^z) - y z: log loss without overflow
    return loss, ((d126.sigmoid(z) - y) / len(y))[:, None].astype(np.float32)         # gradient: (p - y) / n


def train_mlp(Xa, ya, Xv, yv, opt_cls, lr, seed, epochs=EPOCHS, batch=BATCH):  # one real training run; returns validation log loss
    sizes = [Xa.shape[1], *HIDDEN, 1]; acts = ["relu"] * len(HIDDEN) + ["linear"]    # the network's shape
    net = d126.init_net(sizes, seed); opt = opt_cls(net, lr); rng = np.random.RandomState(seed + 1)  # network, optimiser, batch order
    with np.errstate(all="ignore"):                                                   # a diverging run overflows; its loss shows that
        for _ in range(epochs):                                                       # passes over the data
            o = rng.permutation(len(Xa))                                              # a new order each pass
            for s in range(0, len(Xa), batch):                                        # each batch
                A = d126.forward(net, Xa[o[s:s + batch]], acts)                       # forward
                opt.step(net, d126.backward(net, A, acts, bce_logits(A[-1], ya[o[s:s + batch]])[1])[0])  # backward and step
        v = bce_logits(d126.forward(net, Xv, acts)[-1], yv)[0]                        # validation log loss
    return v if np.isfinite(v) else np.inf                                            # a diverged run scores infinity


def mlp_stepper(Xa, ya, opt_cls, seed, batch=BATCH):    # a step(lr) function for lr_find: one batch of a fresh network
    sizes = [Xa.shape[1], *HIDDEN, 1]; acts = ["relu"] * len(HIDDEN) + ["linear"]    # the same network as train_mlp
    net = d126.init_net(sizes, seed); opt = opt_cls(net, LO); rng = np.random.RandomState(seed + 1)  # fresh network and optimiser
    state = dict(o=rng.permutation(len(Xa)), pos=0)                                   # where we are in the shuffled rows

    def step(lr):                                       # one finder step at learning rate lr
        if state["pos"] + batch > len(Xa):                                            # end of the rows: reshuffle
            state["o"], state["pos"] = rng.permutation(len(Xa)), 0                    # a new order
        b = state["o"][state["pos"]:state["pos"] + batch]; state["pos"] += batch      # the next batch
        A = d126.forward(net, Xa[b], acts); loss, d = bce_logits(A[-1], ya[b])        # its loss before the step
        opt.lr = lr; opt.step(net, d126.backward(net, A, acts, d)[0])                 # the step, at this rate
        return loss                                                                   # the loss the finder records
    return step                                                                       # a closure: it remembers net, opt and state


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_known(D):                                      # section 1
    print("\n" + "=" * 70)
    print("1. A KNOWN TRUTH: LEAST SQUARES, WHERE THE LIMIT IS 2 / (LARGEST EIGENVALUE)")
    print("=" * 70)
    X, y = D["Xa"].astype(float), D["ya"].astype(float)                               # the fitting rows, predicting income > $50K by least squares
    lam = np.linalg.eigvalsh(X.T @ X / len(X)).max(); crit = 2 / lam                  # the curvature's largest eigenvalue; the critical rate
    print(f"  largest eigenvalue of X^T X / n: {lam:.4f}; gradient descent diverges above 2 / {lam:.4f} = {crit:.4f}")
    loss = lambda w: 0.5 * np.mean((X @ w - y) ** 2)                                  # half the mean squared error
    for f in (0.9, 0.99, 1.01, 1.1):                                                  # constant rates either side of the limit
        w = np.zeros(X.shape[1])                                                      # start at zero
        with np.errstate(all="ignore"):                                               # a diverging run overflows
            for _ in range(2000):                                                     # full-batch gradient descent
                w -= f * crit * X.T @ (X @ w - y) / len(X)                            # one step
            print(f"  constant rate {f:4.2f} x the limit, 2000 steps: loss {loss(w):.4g}")  # inside errstate: a diverged loss overflows
    w = np.zeros(X.shape[1]); out = {}                                                # the finder, full batch

    def step(lr):                                       # one gradient-descent step at rate lr; returns the loss before it
        nonlocal w                                                                    # w lives in demo_known
        l = loss(w); w = w - lr * X.T @ (X @ w - y) / len(X); return l                # loss, then step
    for beta in (0.0, 0.98):                                                          # no smoothing, and the usual smoothing
        w = np.zeros(X.shape[1])                                                      # start again
        with np.errstate(all="ignore"):                                               # the end of the run overflows
            lrs, raw, sm = lr_find(step, lr_range(crit / 1000, crit * 100, 200), beta=beta)  # the finder
        out[beta] = (lrs, sm)                                                         # for the figure
        print(f"  finder (smoothing {beta}): lowest loss at rate {lrs[np.argmin(sm)] / crit:.2f} x the limit; stopped at "
              f"{lrs[-1] / crit:.2f} x the limit; min / 10 suggests {suggest(lrs, sm)['min_over_10'] / crit:.3f} x the limit")
    print("Actual output from this lesson's run.")
    return crit, out                                                                  # for the figure


def demo_finder(D):                                     # section 2
    print("\n" + "=" * 70)
    print("2. THE FINDER ON A NEURAL NETWORK FOR ADULT")
    print("=" * 70)
    print(f"  network {D['Xa'].shape[1]} -> {HIDDEN[0]} -> {HIDDEN[1]} -> 1, batches of {BATCH}; {FIND_STEPS} steps from {LO:g} to {HI:g}")
    F = {}                                                                            # finder results, per optimiser
    for name, oc in (("Adam", d126.Adam), ("SGD", SGD)):                              # two optimisers
        F[name] = []                                                                  # one result per seed
        for seed in range(SEEDS):                                                     # five seeds
            t0 = time.time()                                                          # a clock
            with np.errstate(all="ignore"):                                           # the end of the run overflows
                lrs, raw, sm = lr_find(mlp_stepper(D["Xa"], D["ya"], oc, seed), lr_range(LO, HI, FIND_STEPS))  # one finder run
            s = suggest(lrs, sm); F[name].append(dict(lrs=lrs, raw=raw, sm=sm, sec=time.time() - t0, **s))  # its suggestions
            print(f"  {name:4s} seed {seed}: {len(sm)} steps, stopped at {lrs[-1]:.2e}; lowest smoothed loss {sm.min():.4f} at "
                  f"{lrs[np.argmin(sm)]:.2e}; min / 10 suggests {s['min_over_10']:.2e}; steepest {s['steepest']:.2e}")
        for k in ("min_over_10", "steepest"):                                         # how much each rule moves between seeds
            v = np.array([f[k] for f in F[name]])                                     # five suggestions
            print(f"  {name:4s} {k:11s}: suggestions span a factor of {v.max() / v.min():.1f} over {SEEDS} seeds")
    print("Actual output from this lesson's run.")
    return F                                                                          # for the next sections


def demo_check(D, F):                                   # section 3
    print("\n" + "=" * 70)
    print("3. THE CHECK: REAL TRAINING AT 13 LEARNING RATES, AND AT THE SUGGESTIONS")
    print("=" * 70)
    print(f"  each run: {EPOCHS} epochs on the {len(D['ya'])} fitting rows, log loss on the {len(D['yv'])} validation rows")
    S = {}; t0 = time.time()                                                          # the sweeps; a clock
    for name, oc in (("Adam", d126.Adam), ("SGD", SGD)):                              # both optimisers
        S[name] = np.array([[train_mlp(D["Xa"], D["ya"], D["Xv"], D["yv"], oc, lr, s) for s in range(SWEEP_SEEDS)] for lr in GRID])  # 13 x 3
        m = S[name].mean(1); b = int(np.argmin(m))                                    # mean over seeds; the best rate
        print(f"  {name}: " + ", ".join(f"{lr:.0e} {v:.4f}" if np.isfinite(v) else f"{lr:.0e} diverged" for lr, v in zip(GRID, m)))
        good = GRID[m <= m[b] + 0.005]                                                # rates within 0.005 of the best
        print(f"  {name}: best {GRID[b]:.0e} ({m[b]:.4f}); within 0.005 of it from {good.min():.0e} to {good.max():.0e}; "
              f"its {SWEEP_SEEDS} seeds there: " + ", ".join(f"{x:.4f}" for x in S[name][b]))  # how much a seed alone moves the score
        for k in ("min_over_10", "steepest"):                                         # train for real at each seed's suggestion
            v = [np.mean([train_mlp(D["Xa"], D["ya"], D["Xv"], D["yv"], oc, f[k], s) for s in range(SWEEP_SEEDS)]) for f in F[name]]  # each suggestion, the sweep's seeds
            print(f"  {name}: training at each finder run's {k:11s} suggestion ({SWEEP_SEEDS} seeds each): " + ", ".join(f"{x:.4f}" for x in v)
                  + f" (worst {max(v) - m[b]:+.4f} against the sweep's best)")
    sweep_s = time.time() - t0                                                        # what the sweeps and checks cost
    print(f"  ({sweep_s:.0f} s for the sweeps and the checks)")
    print("Actual output from this lesson's run.")
    return S, sweep_s                                                                 # for the figure and section 5


def demo_batch(D):                                      # section 4
    print("\n" + "=" * 70)
    print("4. DOES THE FINDER FOLLOW A CHANGE OF BATCH SIZE? (SGD)")
    print("=" * 70)
    out = {}                                                                          # per batch size
    for bs in (32, 512):                                                              # small and large batches
        with np.errstate(all="ignore"):                                               # the end of each finder run overflows
            f = [suggest(*lr_find(mlp_stepper(D["Xa"], D["ya"], SGD, s, batch=bs), lr_range(LO, HI, FIND_STEPS))[::2])["min_over_10"]  # each finder run's min / 10 suggestion
                 for s in range(3)]                                                   # three finder runs; [::2] keeps (rates, smoothed)
        m = np.array([np.mean([train_mlp(D["Xa"], D["ya"], D["Xv"], D["yv"], SGD, lr, s, batch=bs) for s in range(SWEEP_SEEDS)]) for lr in GRID])  # 13 x 3 real runs
        b = int(np.argmin(m)); out[bs] = (f, m); good = GRID[m <= m[b] + 0.01]         # the sweep's best, and rates within 0.01 of it
        at = [np.mean([train_mlp(D["Xa"], D["ya"], D["Xv"], D["yv"], SGD, v, s, batch=bs) for s in range(SWEEP_SEEDS)]) for v in f]  # each suggestion, same seeds
        print(f"  batch {bs:3d}: finder (min / 10) suggests " + ", ".join(f"{v:.2e}" for v in f) + f"; real training there "
              + ", ".join(f"{v:.4f}" for v in at) + f"; the sweep's best {GRID[b]:.0e} ({m[b]:.4f}), within 0.01 of it from {good.min():.0e} to {good.max():.0e}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_cost(D, F, sweep_s):                              # section 5
    print("\n" + "=" * 70)
    print("5. WHAT THE FINDER COSTS")
    print("=" * 70)
    steps = np.mean([len(f["sm"]) for fs in F.values() for f in fs]); sec = np.mean([f["sec"] for fs in F.values() for f in fs])  # per finder run
    epoch_steps = int(np.ceil(len(D["ya"]) / BATCH))                                         # steps in one epoch
    print(f"  one finder run: {steps:.0f} steps on average ({steps / epoch_steps:.2f} of an epoch), {sec:.2f} s")
    print(f"  the check in section 3: {2 * len(GRID) * SWEEP_SEEDS} runs of {EPOCHS} epochs, plus the suggestions, {sweep_s:.0f} s")
    print("Actual output from this lesson's run.")


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(R):                                    # four panels from plain arrays, so a wrapper can save and redraw them
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); ax = ax.ravel()                   # two rows of two, read left to right
    a = ax[0]                                                                         # the known truth
    for beta, c in ((0.0, "tab:gray"), (0.98, "tab:blue")):                           # both smoothing settings
        lrs, sm = R["known"][beta]; a.plot(lrs / R["crit"], sm, color=c, label=f"finder, smoothing {beta}")  # loss against rate / limit
    a.axvline(1, color="tab:red", ls="--", label="2 / largest eigenvalue"); a.set_xscale("log"); a.set_yscale("log")  # the limit
    a.set_xlabel("learning rate / the limit"); a.set_ylabel("loss"); a.legend(fontsize=8); a.set_title("least squares: known limit")  # labels
    for a, name in zip(ax[1:3], ("Adam", "SGD")):                                     # the finder on the network
        for k, f in enumerate(R["finder"][name]):                                     # each seed
            a.plot(f["lrs"], f["sm"], color="tab:blue", alpha=0.6, label="finder, smoothed (5 seeds)" if k == 0 else None)  # its curve
        f = R["finder"][name][0]; a.plot(f["lrs"], f["raw"], color="tab:gray", lw=0.5, alpha=0.6, label="seed 0, raw batch losses")  # raw
        m = R["sweep"][name].mean(1); a.plot(GRID, np.where(np.isfinite(m), m, np.nan), "ko-", label=f"real training, {EPOCHS} epochs (validation)")  # sweep
        for k, f in enumerate(R["finder"][name]):                                     # suggestions
            a.axvline(f["min_over_10"], color="tab:green", lw=0.8, label="min / 10 suggestions" if k == 0 else None)  # green
            a.axvline(f["steepest"], color="tab:orange", lw=0.8, ls=":", label="steepest suggestions" if k == 0 else None)  # orange
        a.set_xscale("log"); a.set_ylim(0.3, 0.8); a.set_xlabel("learning rate"); a.set_ylabel("log loss")  # labels
        a.legend(fontsize=7.5, loc="upper left"); a.set_title(f"{name}: the finder against real training")  # title
    a = ax[3]                                                                         # batch size
    for bs, c in ((32, "tab:purple"), (512, "tab:brown")):                            # both batch sizes
        f, m = R["batch"][bs]; a.plot(GRID, np.where(np.isfinite(m), m, np.nan), "o-", color=c, label=f"real training, batch {bs}")  # sweep
        for v in f:                                                                   # its finder suggestions
            a.axvline(v, color=c, lw=0.8, ls="--")                                    # dashed
    a.set_xscale("log"); a.set_ylim(0.3, 0.8); a.set_xlabel("learning rate"); a.set_ylabel("validation log loss")  # labels
    a.legend(fontsize=8); a.set_title("SGD: batch 32 and 512 (dashed: finder)")      # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "lr_finder_results.png"), dpi=110); plt.close(fig)  # save
    print("saved lr_finder_results.png")


def draw_diagram(f):                                    # how to read a finder curve, on a real one
    fig, ax = plt.subplots(figsize=(11, 5.2))                                         # one panel
    ax.plot(f["lrs"], f["raw"], color="tab:gray", lw=0.6, alpha=0.7, label="loss of each batch")  # raw
    ax.plot(f["lrs"], f["sm"], color="tab:blue", lw=2, label="smoothed (moving average)")  # smoothed
    i = int(np.argmin(f["sm"])); ax.plot(f["lrs"][i], f["sm"][i], "ko")              # the minimum
    ax.annotate("lowest loss", (f["lrs"][i], f["sm"][i]), (f["lrs"][i] * 3, f["sm"][i] - 0.08), arrowprops=dict(arrowstyle="->"))  # label
    ax.axvline(f["min_over_10"], color="tab:green", lw=1.5, label="suggestion: one tenth of that rate")  # min / 10
    ax.axvspan(f["lrs"][0], 1e-4, color="tab:gray", alpha=0.08); ax.text(2e-6, 0.75, "too small:\nloss barely moves", fontsize=9)  # left region
    ax.text(f["lrs"][i] * 2.5, 0.75, "too large:\nloss blows up", fontsize=9)          # right region
    ax.set_xscale("log"); ax.set_ylim(0.3, 0.85); ax.set_xlabel("learning rate (rises every step)"); ax.set_ylabel("training loss")  # labels
    ax.legend(loc="lower left", fontsize=9)                                           # legend
    ax.set_title("Day 170: one short run with a rising learning rate (Adam, seed 0, Adult)", fontsize=11)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "lr_finder_diagram.png"), dpi=110); plt.close(fig)  # save
    print("saved lr_finder_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    tr, _ = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Week 24's data and split (the test file is not used)
    ref = {k: v[fi] for k, v in tr.items()}                                           # category codes and scaling from the fitting rows
    X = d161.onehot_matrix(tr, ref).astype(np.float32); y = tr["y"].astype(np.float32)  # standardised numbers and one-hot categories
    D = dict(Xa=X[fi], ya=y[fi], Xv=X[va], yv=y[va])                                  # fitting rows train; validation rows score
    crit, known = demo_known(D)                                                       # 1
    F = demo_finder(D)                                                                # 2
    S, sweep_s = demo_check(D, F)                                                     # 3
    B = demo_batch(D)                                                                 # 4
    demo_cost(D, F, sweep_s)                                                             # 5
    print()
    plot_results(dict(crit=crit, known=known, finder=F, sweep=S, batch=B))            # plain arrays and dictionaries only
    draw_diagram(F["Adam"][0])                                                        # the diagram
    print(f"total time {time.time() - t0:.0f} s")
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 171: early stopping. Watch the validation loss while training and stop when it stops")
    print("improving: patience, restoring the best weights, and what each rule costs and saves.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
