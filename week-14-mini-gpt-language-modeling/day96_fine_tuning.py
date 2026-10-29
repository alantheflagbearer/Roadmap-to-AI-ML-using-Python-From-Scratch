"""
Day 96: Fine-Tuning the Mini-GPT on a Narrower Real Task

Day 92's mini-GPT learned from 90% of Tiny Shakespeare. The last 10%, which
it never saw, ends with a play that appears nowhere in its training text:
The Tempest (Prospero, Miranda, Ariel, Caliban). The narrow task is to model
that one play, from a small amount of it:

    The Tempest, in order:  70% to train on  |  15% dev  |  15% test

Fine-tuning means starting from the trained weights and training a little
more on the new text, usually at a smaller learning rate. Today it is
compared with the alternatives, each measured on the play's test part
(held-out bits per character with full context, Day 92's measure) and, for
forgetting, on the rest of the held-out text (the end of The Taming of the
Shrew, which neither the pretraining nor the fine-tuning ever saw):

  * zero-shot: the pretrained model as it is
  * from scratch: the same model trained only on The Tempest
  * full fine-tuning (learning rate chosen on dev from three)
  * fine-tuning only the top block and the output layer
  * fine-tuning with replay: half of every batch from the original training text
  * full fine-tuning on a quarter of the play's training part

Every run keeps its best checkpoint by dev loss (checked every 25 steps).
Runs are independent, so they train at the same time in separate processes.
Day 92's code (data, schedule, measurement, Day 90's model) and Day 93's
model loader are imported from their scripts. Every line of code below
carries its own comment, continuing Day 52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import multiprocessing as mp                           # multiprocessing: independent runs in parallel processes
import os                                              # os: paths
import re                                              # re: speaker lines
import sys                                             # sys: path, to import Days 92 and 93
import time                                            # time: perf_counter, for timings

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Days 92 and 93 sit in the same folder
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing; kept quiet to be sure
    import day92_training_mini_gpt as d92               # Day 92: data, schedule, full-context measurement
    import day93_sampling_strategies as d93             # Day 93: the model loader, generation
d90 = d92.d90                                           # Day 90's model code, as Day 92 imported it

START_MARK = "Master:\nBoatswain!"                       # The Tempest's first lines, in the held-out text
SPLITS = (0.70, 0.15, 0.15)                             # train / dev / test, in the play's own order
STEPS, CHECK_EVERY, WARMUP = 400, 25, 20                # fine-tuning budget; dev checks; warm-up steps
FT_LRS = (1e-4, 3e-4, 1e-3)                             # fine-tuning peak learning rates tried on dev
SCRATCH_LR = 3e-3                                       # the from-scratch run uses Day 92's pretraining peak
TOP = ("blocks.3.", "Wout", "bout")                     # 'top only': the last block and the output layer
SMALL_FRAC = 0.25                                       # the small-data run: a quarter of the play's training part
N_PROMPTS, GEN_LEN = 20, 300                            # samples: prompts from the play's test part; characters written


# =============================================================================
# 1. THE TASK
# =============================================================================
def task_data():                                        # NEW: The Tempest split in order, the rest of the held-out text, and the training text
    tr_text, va_text = d90.load_text()                    # Day 90's split
    s = va_text.index(START_MARK)                         # where The Tempest begins
    play, rest = va_text[s:], va_text[:s]                 # the play; the end of The Taming of the Shrew
    a, b = int(SPLITS[0] * len(play)), int((SPLITS[0] + SPLITS[1]) * len(play))  # split points
    return tr_text, play[:a], play[a:b], play[b:], rest   # training text, play train / dev / test, general held-out


def ids(text, vocab):                                   # characters -> ids with the model's vocabulary
    ix = {c: i for i, c in enumerate(vocab)}              # character -> id
    return np.array([ix[c] for c in text])                # every character is in the vocabulary (checked in section 1)


# =============================================================================
# 2. FINE-TUNING -- NEW
# =============================================================================
def adam_subset(P, G, state, lr, keys, clip=1.0):       # NEW: Day 37's Adam (as in Day 90) on the named weights only; the rest stay frozen
    g = {k: v for k, v in d90.w12.leaves(G) if k in keys}  # gradients of the trainable weights
    scale = min(1.0, clip / np.sqrt(sum(np.sum(a ** 2) for a in g.values())))  # clipping over the trainable weights (Day 60)
    state["t"] += 1; t = state["t"]                       # step count
    for k, A in d90.w12.leaves(P):                        # every weight ...
        if k not in keys:                                   # ... frozen ones are skipped
            continue
        m = state["m"].setdefault(k, np.zeros_like(A)); v = state["v"].setdefault(k, np.zeros_like(A))
        gk = g[k] * scale
        m[:] = 0.9 * m + 0.1 * gk; v[:] = 0.999 * v + 0.001 * gk ** 2
        A -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def snapshot(P):                                        # a copy of every weight, for the best checkpoint
    return {k: (snapshot(v) if isinstance(v, dict) else v.copy()) for k, v in P.items()}


def fine_tune(job):                                     # NEW: one run; returns its dev curve and its best checkpoint's numbers
    name, init, which, lr, frac, replay, seed = job       # what to run
    tr_text, ft_tr, ft_dev, ft_te, rest = task_data()     # the data
    P0, vocab, cfg = d93.load_model()                     # Day 92's model (its weights, vocabulary, settings)
    T, B, h = cfg["T"], cfg["B"], cfg["h"]                # context, batch, heads
    tr, dev, te, gen = (ids(x, vocab) for x in (ft_tr, ft_dev, ft_te, rest))  # as ids
    tr = tr[:int(len(tr) * frac)]                         # the small-data run uses only the first part
    old = ids(tr_text, vocab)                             # the original training text, for replay
    with threadpool_limits(1, "blas"):                    # one BLAS thread per process
        rng = np.random.RandomState(seed)                   # batches (and weights, from scratch)
        P = P0 if init == "pretrained" else d90.init_gpt(rng, len(vocab), d=cfg["d"], ff=cfg["ff"], layers=cfg["layers"], t=T)
        keys = {k for k, _ in d90.w12.leaves(P) if which == "all" or k.startswith(TOP)}  # trainable weights
        state = {"t": 0, "m": {}, "v": {}}                  # Adam's state
        best = (d92.eval_bits(P, dev, T, h), 0, snapshot(P))  # (dev bits, step, weights): the untrained start counts too
        curve = [(0, best[0])]                              # dev bits at every check
        t0 = time.perf_counter()                            # time it
        for step in range(1, STEPS + 1):                    # fine-tuning
            o = rng.randint(0, len(tr) - T - 1, B)            # windows from the play
            x = np.stack([tr[i:i + T] for i in o]); y = np.stack([tr[i + 1:i + T + 1] for i in o])
            if replay:                                        # half of the batch from the original training text
                o2 = rng.randint(0, len(old) - T - 1, B // 2)
                x[B // 2:] = np.stack([old[i:i + T] for i in o2]); y[B // 2:] = np.stack([old[i + 1:i + T + 1] for i in o2])
            _, G = d90.gpt_loss_grads(x, y, P, h=h)           # Day 90's forward and backward
            adam_subset(P, G, state, d92.lr_schedule(step - 1, lr, warmup=WARMUP, total=STEPS), keys)  # Day 92's schedule
            if step % CHECK_EVERY == 0:                       # a dev check
                d = d92.eval_bits(P, dev, T, h); curve.append((step, d))
                if d < best[0]:                                 # a new best checkpoint
                    best = (d, step, snapshot(P))
        Pb = best[2]                                        # the best checkpoint's weights
        return dict(name=name, curve=np.array(curve), best_step=best[1], dev=best[0], test=d92.eval_bits(Pb, te, T, h),
                    general=d92.eval_bits(Pb, gen, T, h), trainable=sum(A.size for k, A in d90.w12.leaves(P) if k in keys),
                    minutes=(time.perf_counter() - t0) / 60, P=Pb if name == "full fine-tuning" else None)


def run_parallel(jobs):                                 # one process per job (Day 92's way)
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        return pool.map(fine_tune, jobs)


# =============================================================================
# DEMOS
# =============================================================================
def demo_task():
    print("=" * 70)
    print("1. THE TASK -- one play the model has never seen")
    print("=" * 70)
    tr_text, ft_tr, ft_dev, ft_te, rest = task_data()     # the data
    P, vocab, cfg = d93.load_model()                      # Day 92's model
    play = ft_tr + ft_dev + ft_te                         # the whole play
    names = lambda s: set(re.findall(r"^([A-Z][A-Z' ]+):$", s, re.M))  # speaker names in a text (capitals, then a colon, alone on a line)
    new = sorted(names(play) - names(tr_text))            # speakers of the play that never speak in the training text
    print(f"The Tempest starts {len(tr_text) + len(rest):,} characters into Tiny Shakespeare and runs {len(play):,} "
          f"characters to the end of the file. Split in order: train {len(ft_tr):,}, dev {len(ft_dev):,}, test {len(ft_te):,}. "
          f"The rest of the held-out text (the end of The Taming of the Shrew, {len(rest):,} characters) measures forgetting.")
    print(f"every character of the play is in the model's vocabulary: {set(play) <= set(vocab)}")
    print(f"speakers of the play who never speak in the training text: {len(new)} of {len(names(play))}: " + ", ".join(new[:12]))
    print("\nzero-shot, Day 92's model as it is (held-out bits per character, full context):")
    T, h = cfg["T"], cfg["h"]                             # context, heads
    with threadpool_limits(1, "blas"):                    # one BLAS thread
        z = [d92.eval_bits(P, ids(x, vocab), T, h) for x in (ft_dev, ft_te, rest)]
    print(f"  The Tempest dev {z[0]:.3f}, test {z[1]:.3f}; rest of the held-out text {z[2]:.3f}")
    print("Actual output from this lesson's run.")
    return z, new                                          # the table and the samples use them


def demo_sweep():
    print("\n" + "=" * 70)
    print("2. THE FINE-TUNING LEARNING RATE -- chosen on dev")
    print("=" * 70)
    print(f"Full fine-tuning at three peak learning rates ({STEPS} steps, {WARMUP}-step warm-up then Day 92's cosine, "
          f"batches of 16 windows of 128, seed 0), all at once. Day 92 pretrained at a peak of 0.003. Best dev bits and the "
          f"step it came at (checks every {CHECK_EVERY} steps).")
    res = run_parallel([("full fine-tuning", "pretrained", "all", lr, 1.0, False, 0) for lr in FT_LRS])
    print(f"\n{'peak lr':>8}{'best dev':>10}{'at step':>9}{'dev at the end':>16}")
    for lr, r in zip(FT_LRS, res):                         # one row per rate
        print(f"{lr:>8}{r['dev']:>10.3f}{r['best_step']:>9}{r['curve'][-1, 1]:>16.3f}")
    k = int(np.argmin([r["dev"] for r in res])); best_lr = FT_LRS[k]  # the lowest dev loss
    print(f"chosen: peak learning rate {best_lr}")
    print("Actual output from this lesson's run.")
    return best_lr, res[k]                                 # the main comparison reuses the chosen run


def demo_compare(best_lr, full, zero):
    print("\n" + "=" * 70)
    print("3. SIX WAYS TO GET A MODEL OF THE TEMPEST")
    print("=" * 70)
    print(f"Four more runs at the same time: from scratch (peak {SCRATCH_LR}), top block and output layer only, replay "
          f"(half of each batch from the original training text) and a quarter of the play's training part; the "
          f"fine-tuning runs at the chosen peak {best_lr}. Each at its best dev checkpoint. Forgetting: the rest of the "
          "held-out text, never trained on by any of them.")
    jobs = [("from scratch", "scratch", "all", SCRATCH_LR, 1.0, False, 0), ("top block + output only", "pretrained", "top", best_lr, 1.0, False, 0),
            ("replay", "pretrained", "all", best_lr, 1.0, True, 0), ("a quarter of the data", "pretrained", "all", best_lr, SMALL_FRAC, False, 0)]
    res = [full] + run_parallel(jobs)                     # the chosen sweep run is the full fine-tuning row
    print(f"\n{'model':<26}{'trained weights':>16}{'best step':>10}{'dev':>8}{'test':>8}{'rest of held-out':>18}")
    print(f"{'zero-shot (Day 92 as is)':<26}{0:>16}{'-':>10}{zero[0]:>8.3f}{zero[1]:>8.3f}{zero[2]:>18.3f}")
    for r in res:                                         # one row per run
        print(f"{r['name']:<26}{r['trainable']:>16}{r['best_step']:>10}{r['dev']:>8.3f}{r['test']:>8.3f}{r['general']:>18.3f}")
    print("Actual output from this lesson's run.")
    return res                                             # the plot, the samples and the save use them


def demo_samples(full, new):
    print("\n" + "=" * 70)
    print("4. WHAT FINE-TUNING CHANGES IN THE TEXT -- the play's own characters")
    print("=" * 70)
    tr_text, ft_tr, ft_dev, ft_te, rest = task_data()     # the data
    P0, vocab, cfg = d93.load_model()                     # Day 92's model, again
    starts = np.random.RandomState(5).randint(0, len(ft_te) - 64, N_PROMPTS)  # prompts from the play's test part
    ix = {c: i for i, c in enumerate(vocab)}
    prompts = [[ix[c] for c in ft_te[s:s + 32]] for s in starts]  # 32-character prompts
    print(f"Both models continue the same {N_PROMPTS} prompts from the play's test part by {GEN_LEN} characters (top-p 0.9, "
          "Day 93's sampler, seed 0). How many of the speaker lines they write name one of the play's own new speakers?")
    out = {}                                              # model -> texts
    with threadpool_limits(1, "blas"):                    # one BLAS thread
        for name, P in (("Day 92 (zero-shot)", P0), ("full fine-tuning", full["P"])):
            conts = d93.generate(P, cfg, prompts, GEN_LEN, np.random.RandomState(0), top_p=0.9)
            out[name] = ["".join(vocab[i] for i in c) for c in conts]
    for name, texts in out.items():                       # count speaker lines
        sp = [m for t in texts for m in re.findall(r"^([A-Z][A-Z' ]+):$", t, re.M)]
        share = np.mean([s in new for s in sp]) if sp else 0.0
        print(f"  {name:<20}: {len(sp)} speaker lines, {share:.3f} of them name a speaker new in The Tempest")
    print("\nprompt 1 (" + repr(ft_te[starts[0]:starts[0] + 32]) + "), first 160 characters:")
    for name, texts in out.items():
        print(f"\n--- {name} ---\n{texts[0][:160]}")
    print("Actual output from this lesson's run.")


def save_model(full, vocab, cfg):                       # the fine-tuned weights, in Day 92's file format, for Day 97
    flat = {k.replace(".", "__"): v.astype(np.float32) for k, v in d90.w12.leaves(full["P"])}
    np.savez_compressed(os.path.join(HERE, "day96_finetuned_gpt.npz"), vocab=np.array(vocab), **flat,
                        **{f"cfg__{k}": v for k, v in cfg.items()})
    print("\nsaved the fully fine-tuned model to day96_finetuned_gpt.npz (Day 92's format; Day 93's loader reads it)")


def plot_results(sweep_lr, res, zero):                  # two panels: dev curves | test and forgetting per model
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))           # canvas
    for r in res:                                             # dev curves
        ax[0].plot(r["curve"][:, 0], r["curve"][:, 1], marker="o", markersize=2, label=r["name"])
    ax[0].axhline(zero[0], color="#7f7f7f", linestyle=":", label="zero-shot")
    for r in res:                                             # a curve that never comes into view gets a note instead
        if r["curve"][:, 1].min() > 4.5:
            ax[0].text(200, 4.35, f"{r['name']}: above this panel throughout (best dev {r['dev']:.3f})", ha="center", fontsize=8)
    ax[0].set_xlabel("fine-tuning step"); ax[0].set_ylabel("dev bits per character (The Tempest)"); ax[0].set_ylim(1.8, 4.5)
    ax[0].set_title("Dev loss while training", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    names = ["zero-shot"] + [r["name"] for r in res]; x = np.arange(len(names))
    test = [zero[1]] + [r["test"] for r in res]; gen = [zero[2]] + [r["general"] for r in res]
    ax[1].bar(x - 0.2, test, width=0.4, color="#1f77b4", label="The Tempest, test")
    ax[1].bar(x + 0.2, gen, width=0.4, color="#ff7f0e", label="rest of held-out (forgetting)")
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax[1].set_ylim(1.5, max(test + gen) + 0.3); ax[1].set_ylabel("bits per character (best dev checkpoint)")
    ax[1].set_title("The new play, and what was forgotten", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("fine_tuning_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved fine_tuning_results.png")


def draw_diagram():                                     # pretraining, then the ways to adapt
    fig, ax = plt.subplots(figsize=(13, 4.2))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(-0.15, 4.2); ax.axis("off")  # coordinates in "boxes" (room below the lowest box)
    ax.text(6.5, 3.95, "Day 96: from a general model to one play", ha="center", fontsize=12, fontweight="bold")

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    box(1.6, 2.0, 2.6, 1.0, "Day 92's mini-GPT\n(90% of Shakespeare)", "#DCEBF7")
    box(1.6, 0.6, 2.6, 0.6, "random weights", "#F2F2F2")
    ways = [("full fine-tuning", 3.3), ("top block + output only", 2.5), ("replay: half old text", 1.7),
            ("a quarter of the data", 0.9)]
    for name, y in ways:                                             # the fine-tuning variants
        box(5.6, y, 3.0, 0.55, name, "#E5F2E1", fs=8)
        ax.annotate("", xy=(4.08, y), xytext=(2.92, 2.0), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(9.0, 2.0), xytext=(7.12, y), arrowprops=dict(arrowstyle="->", color="#333333"))
    box(5.6, 0.2, 3.0, 0.4, "from scratch", "#F2F2F2", fs=8)
    ax.annotate("", xy=(4.08, 0.2), xytext=(2.92, 0.6), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(9.0, 1.9), xytext=(7.12, 0.2), arrowprops=dict(arrowstyle="->", color="#333333"))
    box(10.9, 2.0, 3.6, 1.2, "The Tempest: 70% train,\nbest checkpoint on 15% dev,\nmeasured on 15% test", "#FAEEDA")
    plt.savefig("fine_tuning_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved fine_tuning_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 97 is Week 14's build: the week's six parts assembled into one evaluation of these models -- perplexity, "
          "and BLEU and ROUGE written from scratch -- and a test of what each metric can and cannot tell apart.")


def main():                                             # runs every section in order
    zero, new = demo_task()                                # 1. the task
    best_lr, full = demo_sweep()                           # 2. the learning rate
    res = demo_compare(best_lr, full, zero)                # 3. six ways
    demo_samples(full, new)                                # 4. samples
    _, vocab, cfg = d93.load_model()                       # the vocabulary and settings, for the file
    save_model(full, vocab, cfg)                           # for Day 97
    plot_results(best_lr, res, zero)                       # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 5. Day 97


if __name__ == "__main__":                              # run only when executed directly (and not in the worker processes)
    main()                                                 # kick off the lesson
