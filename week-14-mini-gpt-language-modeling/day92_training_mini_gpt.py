"""
Day 92: Training the Mini-GPT on Real Text -- Honest Perplexity Numbers

Day 90's tiny GPT trained for 4,000 steps at one fixed learning rate and
ended level with a counting model. Today it is trained properly, and every
number about it is reported the way it should be:

  * a learning-rate schedule: a short warm-up, then a cosine decay (new)
  * the peak learning rate chosen by a short, real sweep, not by guessing
  * a bigger model (4 blocks of 128, context 128) next to Day 90's (2 of 64)
  * held-out loss measured with full context (a sliding window), every 500 steps,
    together with the same measurement on training text, so overfitting shows
  * perplexity reported per character AND per word, because a perplexity
    means nothing without its unit

The model code is Day 90's, imported from its script (Day 90 in turn imports
Week 12's block). The data is Day 90's: Tiny Shakespeare, the first 90% to
train on, the last 10% held out.

The three training runs are independent, so they run at the same time in
separate processes (multiprocessing); timings are therefore those of a busy
machine. Every line of code below carries its own comment, continuing Day
52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import multiprocessing as mp                           # multiprocessing: independent runs in parallel processes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 90's script
import time                                            # time: perf_counter, for timings

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, os.path.join(HERE, "..", "week-13-tokenization-embeddings"))  # Day 90's folder
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing; kept quiet to be sure
    import day90_tiny_gpt as d90                        # Day 90: the GPT (forward, backward, Adam), its data and tokenizer

SMALL = dict(T=64, B=32, d=64, h=4, ff=256, layers=2)  # Day 90's model and batch
BIG = dict(T=128, B=16, d=128, h=4, ff=512, layers=4)  # today's bigger model: same tokens per batch (2,048)
STEPS, CHECK_EVERY = 5000, 500                          # training steps; measure this often
WARMUP, FLOOR = 200, 0.1                                # warm-up steps; the cosine decays to 10% of the peak
SWEEP, SWEEP_STEPS = (1e-3, 3e-3, 6e-3), 300            # peak learning rates tried for the big model, and for how long
PEAK_SMALL = 3e-3                                       # Day 90's learning rate, kept for the small model


# =============================================================================
# 1. THE SCHEDULE AND THE MEASUREMENT -- NEW
# =============================================================================
def lr_schedule(step, peak, warmup=WARMUP, total=STEPS, floor=FLOOR):  # NEW: linear warm-up, then cosine decay to floor * peak
    if step < warmup:                                     # the first steps
        return peak * (step + 1) / warmup                   # grow linearly from almost 0 to the peak
    progress = (step - warmup) / max(1, total - warmup)   # 0 at the end of the warm-up, 1 at the last step
    return peak * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * min(1.0, progress))))  # half a cosine, from peak to floor * peak


def eval_bits(P, ids, T, h, max_windows=None):          # NEW: bits per token with full context: windows every T/2, only their last T/2 positions scored
    half = T // 2                                         # stride, and the number of scored positions per window
    starts = np.arange(0, len(ids) - T - 1, half)[:max_windows]  # window starts
    nats, count = 0.0, 0                                  # summed -ln p; tokens scored
    for s in range(0, len(starts), 32):                   # 32 windows at a time
        st = starts[s:s + 32]                               # their starts
        x = np.stack([ids[i:i + T] for i in st]); y = np.stack([ids[i + 1:i + T + 1] for i in st])  # inputs, next tokens
        logp = d90.log_softmax(d90.gpt_forward(x, P, h)[0][:, half:])  # Day 90's model; the last half of each window
        nats -= np.take_along_axis(logp, y[:, half:, None], axis=-1).sum()  # -ln p of the right tokens
        count += y[:, half:].size                           # how many
    return nats / count / np.log(2)                       # bits per token (= per character here)


def data():                                             # Day 90's text, split and tokenized as characters (Day 84's tokenizer, 0 merges)
    tr_text, va_text = d90.load_text()                    # 90% / 10%
    merges, vocab, tr, va = d90.tokenize(tr_text, va_text, 0)  # characters
    return tr_text, va_text, vocab, tr, va                # texts, vocabulary, ids


# =============================================================================
# 2. ONE TRAINING RUN -- runs in its own process
# =============================================================================
def train_run(job):                                     # NEW: one configuration, trained and measured; returns what the report needs
    name, cfg, peak, schedule, steps, seed = job          # what to train
    _, _, vocab, tr, va = data()                          # each process loads the data itself
    tr_eval = tr[:len(va)]                                # a slice of TRAINING text as long as the held-out text
    with threadpool_limits(1, "blas"):                    # one BLAS thread per process (Day 64)
        rng = np.random.RandomState(seed)                   # weights and batches
        P = d90.init_gpt(rng, len(vocab), d=cfg["d"], ff=cfg["ff"], layers=cfg["layers"], t=cfg["T"])  # Day 90's model
        state, recent, busy = {"t": 0, "m": {}, "v": {}}, [], 0.0  # Adam's state; recent losses; training seconds
        measure = steps > SWEEP_STEPS                       # the short sweep runs only record their training loss
        curve = [(0, 0.0, np.nan, eval_bits(P, va, cfg["T"], cfg["h"]), eval_bits(P, tr_eval, cfg["T"], cfg["h"]))] if measure else []
        T, B = cfg["T"], cfg["B"]                           # context and batch
        for step in range(1, steps + 1):                    # training
            t0 = time.perf_counter()                          # time the step only
            lr = lr_schedule(step - 1, peak, warmup=min(WARMUP, steps // 3), total=steps) if schedule else peak  # warm-up: 200 steps, or a third of a short run
            o = rng.randint(0, len(tr) - T - 1, B)            # random windows
            x = np.stack([tr[i:i + T] for i in o]); y = np.stack([tr[i + 1:i + T + 1] for i in o])  # inputs, next tokens
            loss, G = d90.gpt_loss_grads(x, y, P, h=cfg["h"])  # Day 90's forward and backward
            d90.adam_step(P, G, state, lr); recent.append(loss)  # Day 37's Adam with Day 60's clipping (Day 90)
            busy += time.perf_counter() - t0                  # training time
            if (measure and step % CHECK_EVERY == 0) or (not measure and step % 50 == 0):  # a check
                row = (step, lr, np.mean(recent) / np.log(2))   # step, learning rate, mean training-batch bits since the last check
                if measure:                                     # full measurements for the real runs
                    row += (eval_bits(P, va, T, cfg["h"]), eval_bits(P, tr_eval, T, cfg["h"]))
                curve.append(row); recent = []
    return dict(name=name, curve=np.array(curve), P=P, sec=busy / steps, weights=sum(a.size for _, a in d90.w12.leaves(P)))


def run_parallel(jobs):                                 # train several jobs at once, one process each
    with mp.get_context("spawn").Pool(len(jobs)) as pool:  # fresh processes (the safe way on every platform)
        return pool.map(train_run, jobs)                    # results in the order of the jobs


# =============================================================================
# DEMOS
# =============================================================================
def demo_schedule():
    print("=" * 70)
    print("1. THE LEARNING-RATE SCHEDULE -- warm up, then a cosine decay")
    print("=" * 70)
    print(f"Linear warm-up over the first {WARMUP} steps, then half a cosine down to {FLOOR:.0%} of the peak at step {STEPS}. "
          "Shown for a peak of 1.0, so the numbers are fractions of the peak.")
    print("\n  step: " + ", ".join(f"{s}: {lr_schedule(s, 1.0):.3f}" for s in (0, 99, 199, 200, 1000, 2600, 4000, 4999)))
    print("Actual output from this lesson's run.")


def demo_sweep():
    print("\n" + "=" * 70)
    print("2. CHOOSING THE PEAK LEARNING RATE -- a short real sweep for the big model")
    print("=" * 70)
    print(f"The big model ({BIG['layers']} blocks, d = {BIG['d']}, context {BIG['T']}) at three peak learning rates for "
          f"{SWEEP_STEPS} steps each (100-step warm-up inside the {SWEEP_STEPS}), seed 0, all three at once. Mean training "
          "loss over each 50 steps, in bits per character.")
    res = run_parallel([(f"sweep {lr}", BIG, lr, True, SWEEP_STEPS, 0) for lr in SWEEP])  # three short runs
    print(f"\n{'step':>6}" + "".join(f"{'lr ' + str(lr):>12}" for lr in SWEEP))
    for i in range(len(res[0]["curve"])):                 # one row per 50 steps
        print(f"{int(res[0]['curve'][i, 0]):>6}" + "".join(f"{r['curve'][i, 2]:>12.3f}" for r in res))
    best = SWEEP[int(np.argmin([r["curve"][-1, 2] for r in res]))]  # the lowest final training loss
    print(f"chosen for the big model: peak learning rate {best} (lowest training loss at step {SWEEP_STEPS})")
    print("Actual output from this lesson's run.")
    return best, res                                       # the main runs use it; the plot shows the sweep


def demo_training(best):
    print("\n" + "=" * 70)
    print("3. THREE RUNS -- Day 90's model with and without the schedule, and the big model")
    print("=" * 70)
    print(f"{STEPS} steps each, seed 0, at the same time in three processes. Every {CHECK_EVERY} steps: held-out bits per "
          "character with full context (windows every T/2 characters, only their last T/2 positions scored), and the "
          "same measurement on a slice of TRAINING text of the same length.")
    jobs = [("small, constant lr", SMALL, PEAK_SMALL, False, STEPS, 0), ("small, schedule", SMALL, PEAK_SMALL, True, STEPS, 0),
            ("big, schedule", BIG, best, True, STEPS, 0)]
    t0 = time.perf_counter(); res = run_parallel(jobs)    # train all three
    print(f"(wall time for the three runs together: {(time.perf_counter() - t0) / 60:.0f} minutes)")
    for r, (name, cfg, peak, sched, _, _) in zip(res, jobs):  # one table per run
        print(f"\n{name}: {r['weights']} weights, context {cfg['T']}, peak lr {peak}, {r['sec'] * 1000:.0f} ms per step")
        print(f"{'step':>6}{'lr':>10}{'train batches':>15}{'train text':>12}{'held-out':>10}")
        for row in r["curve"]:                              # one row per check
            tb = "-" if np.isnan(row[2]) else f"{row[2]:.3f}"
            print(f"{int(row[0]):>6}{row[1]:>10.5f}{tb:>15}{row[4]:>12.3f}{row[3]:>10.3f}")
    print("Actual output from this lesson's run.")
    return res                                             # later sections and the plot use them


def demo_honest(res, tr_text):
    print("\n" + "=" * 70)
    print("4. HONEST NUMBERS -- the best checkpoint, the gap, and perplexity with its unit")
    print("=" * 70)
    words = tr_text.split()                               # whitespace-separated words of the training text
    cpw = len(tr_text) / len(words)                       # characters per word, spaces and newlines included
    print(f"Perplexity = 2^(bits per token): the number of equally likely choices the model is, on average, choosing "
          f"between. Per character it is small; per word it is large. The training text has {cpw:.2f} characters per "
          "word (spaces and newlines included), so bits per word = bits per character x that.")
    print(f"\n{'run':<20}{'final held-out':>15}{'best (step)':>15}{'train text':>12}{'gap':>7}{'ppl/char':>10}{'ppl/word':>10}")
    for r in res:                                         # each run
        c = r["curve"]; fin = c[-1]; b = int(np.argmin(c[:, 3]))  # final row; best held-out check
        print(f"{r['name']:<20}{fin[3]:>15.3f}{f'{c[b, 3]:.3f} ({int(c[b, 0])})':>15}{fin[4]:>12.3f}{fin[3] - fin[4]:>7.3f}"
              f"{2 ** fin[3]:>10.2f}{2 ** (fin[3] * cpw):>10.0f}")
    print("(gap = held-out minus training-text bits: how much worse the model does on text it never saw)")
    print("Actual output from this lesson's run.")
    return cpw                                             # the guide quotes it


def save_model(r, vocab):                               # the big model's weights, for Days 93 and 94
    flat = {k.replace(".", "__"): v.astype(np.float32) for k, v in d90.w12.leaves(r["P"])}  # dotted names -> flat keys, float32
    np.savez_compressed(os.path.join(HERE, "day92_mini_gpt.npz"), vocab=np.array(vocab), **flat, **{f"cfg__{k}": v for k, v in BIG.items()})
    print(f"\nsaved the big model to day92_mini_gpt.npz ({r['weights']} weights as float32, its vocabulary and settings)")


def plot_results(sweep, res):                           # three panels: the schedule | the sweep | held-out and training-text curves
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))          # canvas
    s = np.arange(STEPS); ax[0].plot(s, [lr_schedule(i, 1.0) for i in s], color="#1f77b4")  # the schedule, peak 1
    ax[0].set_xlabel("step"); ax[0].set_ylabel("learning rate (fraction of the peak)"); ax[0].set_title("Warm-up, then cosine decay", fontsize=10)
    ax[0].grid(alpha=0.3)
    for r, lr in zip(sweep, SWEEP):                          # the sweep
        ax[1].plot(r["curve"][:, 0], r["curve"][:, 2], marker="o", markersize=3, label=f"peak {lr}")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("training loss (bits per character)"); ax[1].set_title(f"Big model, first {SWEEP_STEPS} steps", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    colors = ["#7f7f7f", "#1f77b4", "#d62728"]               # one colour per run
    for r, col in zip(res, colors):                          # held-out (solid) and training text (dashed)
        ax[2].plot(r["curve"][:, 0], r["curve"][:, 3], marker="o", markersize=3, color=col, label=f"{r['name']}, held-out")
        ax[2].plot(r["curve"][:, 0], r["curve"][:, 4], linestyle="--", color=col, label=f"{r['name']}, training text")
    ax[2].set_ylim(1.4, 3.5); ax[2].set_xlabel("step"); ax[2].set_ylabel("bits per character (full context)")
    ax[2].set_title("Held-out against training text", fontsize=10); ax[2].legend(fontsize=7, loc="upper right"); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("training_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved training_results.png")


def draw_diagram():                                     # the evaluation protocol: sliding windows, only the second half scored
    fig, ax = plt.subplots(figsize=(13, 3.8))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.8); ax.axis("off")     # coordinates in "boxes"
    ax.text(6.5, 3.55, "Day 92: measuring held-out loss with full context", ha="center", fontsize=12, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.5, 2.55), 12.0, 0.45, boxstyle="round,pad=0.02", facecolor="#F2F2F2", edgecolor="#333333"))
    ax.text(6.5, 2.77, "held-out text, one character after another", ha="center", va="center", fontsize=9)
    for k in range(4):                                             # four windows, each half a window after the last
        x0 = 0.5 + k * 2.4; y = 1.9 - k * 0.42
        ax.add_patch(FancyBboxPatch((x0, y), 2.4, 0.3, boxstyle="round,pad=0.01", facecolor="#DCEBF7", edgecolor="#333333"))
        ax.add_patch(FancyBboxPatch((x0 + 2.4, y), 2.4, 0.3, boxstyle="round,pad=0.01", facecolor="#E5F2E1", edgecolor="#333333"))
        ax.text(x0 + 1.2, y + 0.15, "context only", ha="center", va="center", fontsize=8)
        ax.text(x0 + 3.6, y + 0.15, "scored", ha="center", va="center", fontsize=8)
    ax.text(6.5, 0.12, "every scored character has at least T/2 characters before it; each is scored exactly once",
            ha="center", fontsize=8.5, style="italic")
    plt.savefig("training_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved training_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("The big model is saved. Day 93 writes with it in four ways -- greedy, temperature, top-k and top-p "
          "sampling -- and measures what each does to the text.")


def main():                                             # runs every section in order
    tr_text, va_text, vocab, tr, va = data()               # the corpus
    demo_schedule()                                        # 1. the schedule
    best, sweep = demo_sweep()                             # 2. the peak learning rate
    res = demo_training(best)                              # 3. the three runs
    demo_honest(res, tr_text)                              # 4. honest numbers
    save_model(res[2], vocab)                              # for Days 93 and 94
    plot_results(sweep, res)                               # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 5. Day 93


if __name__ == "__main__":                              # run only when executed directly (and not in the worker processes)
    main()                                                 # kick off the lesson
