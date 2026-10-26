"""
Day 95: Scaling-Laws Intuition -- a Real Experiment Varying Model Size and Data

Kaplan et al. (2020) and Hoffmann et al. (2022) found that a language
model's held-out loss falls smoothly, close to a power law, as the model
gets bigger or the data grows -- until something else runs out:

    L(N) = c + a N^(-alpha)      N = number of weights, c = what no model size removes

Today the same experiment at the smallest possible scale: Day 90's GPT on
Tiny Shakespeare, with

  * four model sizes (width 16, 32, 64, 128; two blocks each), all the data, and
  * four amounts of data (1/32, 1/8, 1/2 and all of the training text), width 64,

every run with the same 3,000 steps of Day 92's schedule. Seven runs, trained
at the same time in separate processes. With four points per curve, a fitted
exponent is a rough description, not a law; the lesson says how rough.

The model is Day 90's, imported from its script; the schedule and the
full-context measurement are Day 92's, imported from its script. Every line
of code below carries its own comment, continuing Day 52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import multiprocessing as mp                           # multiprocessing: independent runs in parallel processes
import os                                              # os: paths
import sys                                             # sys: path, to import Days 90 and 92
import time                                            # time: perf_counter, for timings

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 92 sits in the same folder
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing; kept quiet to be sure
    import day92_training_mini_gpt as d92               # Day 92: the schedule, the full-context measurement, Day 90's model (d92.d90)
d90 = d92.d90                                           # Day 90's script, as Day 92 imported it

T, B, H, LAYERS = 64, 32, 4, 2                          # Day 90's context, batch, heads and depth for every run
STEPS, PEAK = 3000, 3e-3                                # the same budget and peak learning rate for every run
WIDTHS = (16, 32, 64, 128)                              # model sizes: the width d (feed-forward 4d)
FRACTIONS = (1 / 32, 1 / 8, 1 / 2, 1.0)                 # amounts of training text, at width 64
CHECKS = (250, 500, 1000, 1500, 2000, 2500, 3000)       # when held-out loss is measured


# =============================================================================
# 1. COUNTING AND FITTING -- NEW
# =============================================================================
def count_weights(V, d, layers, t, ff=None):            # NEW: weights of Day 90's GPT, by formula
    ff = ff or 4 * d                                      # feed-forward width
    block = 4 * d * d + 2 * d * ff + ff + d + 4 * d       # Wq, Wk, Wv, Wo; W1, W2; c1, c2; two LayerNorms (gain and shift)
    return V * d + t * d + layers * block + d * V + V     # token table, position table, blocks, output layer and its bias


def fit_power_law(N, L):                                # NEW: L = c + a N^-alpha, by trying c on a grid and fitting a line in log-log space
    N, L = np.asarray(N, float), np.asarray(L, float)     # arrays
    best = None                                           # (squared error, c, a, alpha)
    for c in np.linspace(0, L.min() * 0.999, 400):        # the floor c must be below every loss
        slope, icpt = np.polyfit(np.log(N), np.log(L - c), 1)  # log(L - c) = log a - alpha log N
        err = np.sum((c + np.exp(icpt) * N ** slope - L) ** 2)  # squared error in the losses themselves
        if best is None or err < best[0]:                   # the best c so far
            best = (err, c, np.exp(icpt), -slope)
    return best[1:]                                       # c, a, alpha


# =============================================================================
# 2. ONE RUN -- in its own process
# =============================================================================
def run(job):                                           # NEW: train one configuration with Day 92's schedule; held-out loss at the checks
    name, d, frac = job                                   # width and share of the training text
    _, _, vocab, tr, va = d92.data()                      # Day 92's data (characters)
    tr = tr[:int(len(tr) * frac)]                         # the first part of the training text
    with threadpool_limits(1, "blas"):                    # one BLAS thread per process
        rng = np.random.RandomState(0)                      # seed 0 for every run
        P = d90.init_gpt(rng, len(vocab), d=d, ff=4 * d, layers=LAYERS, t=T)  # Day 90's model at this size
        state, curve, t0 = {"t": 0, "m": {}, "v": {}}, [], time.perf_counter()  # Adam; checks; clock
        for step in range(1, STEPS + 1):                    # the same budget for every run
            o = rng.randint(0, len(tr) - T - 1, B)            # random windows from the available text
            x = np.stack([tr[i:i + T] for i in o]); y = np.stack([tr[i + 1:i + T + 1] for i in o])  # inputs, next tokens
            _, G = d90.gpt_loss_grads(x, y, P, h=H)           # Day 90's forward and backward
            d90.adam_step(P, G, state, d92.lr_schedule(step - 1, PEAK, total=STEPS))  # Day 92's schedule, Day 37's Adam
            if step in CHECKS:                                # a check
                curve.append((step, d92.eval_bits(P, va, T, H), d92.eval_bits(P, tr[:len(va)], T, H)))  # held-out, training text
        n = sum(a.size for _, a in d90.w12.leaves(P))        # weights
    return dict(name=name, d=d, frac=frac, chars=len(tr), weights=n, curve=np.array(curve), minutes=(time.perf_counter() - t0) / 60)


# =============================================================================
# DEMOS
# =============================================================================
def demo_count(V):
    print("=" * 70)
    print("1. HOW MANY WEIGHTS -- counted by formula, checked against the model")
    print("=" * 70)
    print(f"Two blocks, context {T}, {V} characters; feed-forward width 4d.")
    print(f"\n{'width d':>8}{'by formula':>12}{'in the model':>14}{'in the blocks':>15}")
    for d in WIDTHS:                                      # each width
        P = d90.init_gpt(np.random.RandomState(0), V, d=d, ff=4 * d, layers=LAYERS, t=T)  # build it
        n = sum(a.size for _, a in d90.w12.leaves(P))       # count it
        nb = sum(a.size for k, a in d90.w12.leaves(P) if k.startswith("blocks"))  # the blocks alone
        print(f"{d:>8}{count_weights(V, d, LAYERS, T):>12}{n:>14}{nb:>15}")
    print("Actual output from this lesson's run.")


def demo_runs():
    print("\n" + "=" * 70)
    print("2. SEVEN RUNS -- four model sizes on all the data, four data sizes at width 64")
    print("=" * 70)
    print(f"Every run: {STEPS} steps, batches of {B} windows of {T} characters, Day 92's schedule with peak {PEAK}, seed 0; "
          "held-out bits per character with full context (Day 92's measurement). All seven train at the same time.")
    jobs = [(f"width {d}", d, 1.0) for d in WIDTHS] + [(f"width 64, {f:g} of the text", 64, f) for f in FRACTIONS if f < 1]
    t0 = time.perf_counter()                              # wall clock
    with mp.get_context("spawn").Pool(len(jobs)) as pool:  # one process per run
        res = pool.map(run, jobs)                           # train them all
    print(f"(wall time {(time.perf_counter() - t0) / 60:.0f} minutes for all seven)")
    full64 = next(r for r in res if r["d"] == 64 and r["frac"] == 1.0)  # shared by both experiments
    print(f"\n{'run':<30}{'weights':>9}{'chars':>9}" + "".join(f"{c:>7}" for c in CHECKS) + f"{'train text':>12}{'min':>6}")
    for r in res:                                         # one row per run
        print(f"{r['name']:<30}{r['weights']:>9}{r['chars']:>9}" + "".join(f"{v:>7.3f}" for v in r["curve"][:, 1])
              + f"{r['curve'][-1, 2]:>12.3f}{r['minutes']:>6.0f}")
    print("(columns: held-out bits per character after that many steps; train text: the same measured on training text at the end)")
    print("Actual output from this lesson's run.")
    sizes = [r for r in res if r["frac"] == 1.0]          # the model-size runs (width 64 included)
    datas = sorted([r for r in res if r["d"] == 64], key=lambda r: r["frac"])  # the data-size runs (full data included)
    return sizes, datas                                    # the next sections use them


def demo_fits(sizes, datas):
    print("\n" + "=" * 70)
    print("3. FITTING POWER LAWS -- and how much four points can say")
    print("=" * 70)
    print("L = c + a x^-alpha fitted to the held-out losses, x = weights (model size) or training characters (data). "
          "For data, twice: with the final losses, and with each run's best checkpoint (its lowest held-out loss at any "
          "check), because the small-data runs overfit within the fixed budget (section 4). Then the same fits with each "
          "point left out in turn: how much does alpha move?")
    out = {}                                              # name -> (x, L, fit)
    for name, rs, key, pick in (("model size", sizes, "weights", "final"), ("data", datas, "chars", "final"),
                                ("data, best checkpoint", datas, "chars", "best")):
        x = np.array([r[key] for r in rs])                  # weights or characters
        L = np.array([r["curve"][-1, 1] if pick == "final" else r["curve"][:, 1].min() for r in rs])  # final or best held-out loss
        c, a, al = fit_power_law(x, L)                      # the fit
        loo = [fit_power_law(np.delete(x, i), np.delete(L, i))[2] for i in range(len(x))]  # leave one out
        out[name] = (x, L, (c, a, al))
        print(f"\n{name}: x = {', '.join(str(int(v)) for v in x)}; held-out bits = {', '.join(f'{v:.3f}' for v in L)}")
        print(f"  fit: c = {c:.3f}, alpha = {al:.3f}; leaving one point out, alpha = " + ", ".join(f"{v:.2f}" for v in loo))
        print(f"  largest gap between the fit and a point: {np.abs(c + a * x ** -al - L).max():.3f} bits")
    print("Actual output from this lesson's run.")
    return out                                             # the plot shows the fits


def demo_overfit(datas):
    print("\n" + "=" * 70)
    print("4. WHERE DATA RUNS OUT -- training text against held-out text")
    print("=" * 70)
    print("The same width-64 model on less and less text, all for 3000 steps. Gap = held-out minus training-text bits.")
    print(f"\n{'training text':<16}{'characters':>11}{'passes over it':>15}{'train text':>12}{'held-out':>10}{'gap':>7}")
    for r in datas:                                       # least data first
        passes = STEPS * B * T / r["chars"]                 # how many times each character was seen, on average
        print(f"{r['frac']:<16g}{r['chars']:>11}{passes:>15.1f}{r['curve'][-1, 2]:>12.3f}{r['curve'][-1, 1]:>10.3f}"
              f"{r['curve'][-1, 1] - r['curve'][-1, 2]:>7.3f}")
    best = [CHECKS[int(np.argmin(r["curve"][:, 1]))] for r in datas]  # when each held-out curve was lowest
    print("step with the lowest held-out loss: " + ", ".join(f"{r['frac']:g}: {b}" for r, b in zip(datas, best)))
    print("Actual output from this lesson's run.")


def plot_results(sizes, datas, fits):                   # three panels: curves by size | loss against size | loss against data
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))          # canvas
    for r in sizes:                                          # learning curves per model size
        ax[0].plot(r["curve"][:, 0], r["curve"][:, 1], marker="o", markersize=3, label=f"width {r['d']} ({r['weights']:,} weights)")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("held-out bits per character"); ax[0].set_title("Bigger models learn faster per step", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for a, names, xlabel in ((ax[1], ["model size"], "weights"), (ax[2], ["data", "data, best checkpoint"], "training characters")):
        for name, col in zip(names, ("#1f77b4", "#2ca02c")):  # one or two fits per panel
            x, L, (c, aa, al) = fits[name]                     # points and fit
            xs = np.logspace(np.log10(x.min() / 1.5), np.log10(x.max() * 1.5), 100)  # a smooth curve
            lab = "final" if name in ("model size", "data") else "best checkpoint"
            a.plot(x, L, "o", color=col, label=f"{lab} held-out loss")
            a.plot(xs, c + aa * xs ** -al, color=col, alpha=0.7, label=f"fit ({lab}): {c:.2f} + a x^-{al:.2f}")
        a.set_xscale("log"); a.set_xlabel(xlabel + " (log scale)"); a.set_ylabel("held-out bits per character")
        a.set_title(f"Loss against {names[0]}", fontsize=10); a.legend(fontsize=7); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("scaling_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved scaling_results.png")


def draw_diagram():                                     # the experiment's grid
    fig, ax = plt.subplots(figsize=(12, 3.6))                   # canvas
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")     # coordinates in "boxes"
    ax.text(6, 3.35, "Day 95: seven runs, one budget (3,000 steps each)", ha="center", fontsize=12, fontweight="bold")

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(0.2, 2.45, "model size\n(all the data)", fontsize=9, va="center")
    for k, d in enumerate(WIDTHS):                                   # the size row
        box(3.0 + k * 2.3, 2.45, 2.0, 0.6, f"width {d}", "#DCEBF7")
    ax.text(0.2, 1.2, "data\n(width 64)", fontsize=9, va="center")
    for k, f in enumerate(FRACTIONS):                                # the data row
        box(3.0 + k * 2.3, 1.2, 2.0, 0.6, f"{'all' if f == 1 else f'1/{round(1 / f)}'} of the text", "#E5F2E1")
    ax.text(6.9, 0.35, "width 64 on all the text is one run, shared by both rows", ha="center", fontsize=8, style="italic")
    plt.savefig("scaling_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved scaling_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 96 takes a trained mini-GPT and fine-tunes it on a narrower real task: one kind of text, little data.")


def main():                                             # runs every section in order
    _, _, vocab, _, _ = d92.data()                         # the vocabulary
    demo_count(len(vocab))                                 # 1. weights
    sizes, datas = demo_runs()                             # 2. the runs
    fits = demo_fits(sizes, datas)                         # 3. power laws
    demo_overfit(datas)                                    # 4. data running out
    plot_results(sizes, datas, fits)                       # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 5. Day 96


if __name__ == "__main__":                              # run only when executed directly (and not in the worker processes)
    main()                                                 # kick off the lesson
