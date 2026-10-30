"""
Day 99: Fine-Tuning a BERT-Style Model on a Real Classification Dataset

Day 98 used a DistilBERT that someone else had fine-tuned for sentiment on all
67,349 of SST-2's training items. Today we do the fine-tuning ourselves, on
this CPU, starting from plain pretrained DistilBERT (distilbert-base-uncased:
masked-word pretraining only, no sentiment labels ever seen), with a new
two-class head on top.

The data: SST-2. Training items come from SST-2's training set (a fixed
shuffle; the first n of it). SST-2's 872 validation sentences are split once,
at random, into a dev half (436, for choosing the checkpoint) and a test half
(436, reported once per model).

  1. The setup: what is new in the model (a randomly initialised head), and
     the training loop (AdamW, linear warm-up and decay, best dev checkpoint).
  2. Five fine-tuning runs in parallel processes: 250, 1,000 (three seeds) and
     4,000 training items.
  3. The alternatives on the same test half: a frozen encoder with a logistic
     regression on its [CLS] vectors, the from-scratch bag of words (Day 98's),
     and Day 98's ready-made model.
  4. Paired comparisons: a bootstrap of the DIFFERENCE between two models on
     the same sentences, which is tighter than comparing two separate intervals.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported script quiet
import copy                                            # copy: deepcopy, for the best checkpoint
import io                                              # io: StringIO, where that output goes
import multiprocessing as mp                           # multiprocessing: independent runs in parallel processes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 98
import time                                            # time: perf_counter, for timings

import numpy as np                                     # numpy: the from-scratch parts
import torch                                           # torch: training the model
from datasets import disable_progress_bars            # datasets: its progress bars (SST-2 is loaded through Day 98)
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # the library's entry points
from transformers.utils import logging as hf_logging   # the library's logging
from huggingface_hub.utils import logging as hub_logging  # the model hub's logging
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 98 sits in the same folder
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing; kept quiet to be sure
    import day98_pretrained_transformers as d98         # Day 98: the bag-of-words model, bootstrap_ci, the ready-made model
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()  # no progress bars or notices in the output
hub_logging.set_verbosity_error(); disable_progress_bars()            # the same for the hub and datasets

BASE = "distilbert/distilbert-base-uncased"             # plain pretrained DistilBERT: no sentiment training
LR, BATCH, EPOCHS, MAX_LEN = 5e-5, 16, 2, 64            # the BERT paper's usual range is 2e-5 to 5e-5; not tuned here
WARMUP_FRAC, WEIGHT_DECAY = 0.1, 0.01                   # 10% of steps warming up, then linear decay to 0; AdamW's decay
CHECKS = 8                                              # dev checks per run
THREADS_PER_RUN = 3                                     # PyTorch threads in each parallel process
RUNS = [(250, 0), (1000, 0), (1000, 1), (1000, 2), (4000, 0)]  # (training items, seed)


# =============================================================================
# DATA
# =============================================================================
def data():                                             # NEW: training items in a fixed shuffle; validation split into dev / test halves
    Xtr, ytr = d98.sst2("train")                          # Day 98's loader
    Xva, yva = d98.sst2("validation")
    order = np.random.RandomState(0).permutation(len(Xtr))  # one fixed shuffle of the training set
    Xtr, ytr = [Xtr[i] for i in order], ytr[order]        # the first n of it are 'n training items'
    half = np.random.RandomState(1).permutation(len(Xva))  # one fixed split of validation
    dev, test = half[:len(Xva) // 2], half[len(Xva) // 2:]
    return (Xtr, ytr), ([Xva[i] for i in dev], yva[dev]), ([Xva[i] for i in test], yva[test])


def predict(tok, model, texts, bs=32):                  # predicted labels for many texts (eval mode, no gradients)
    model.eval()                                          # dropout off
    out = []                                              # one array per batch
    with torch.no_grad():                                 # inference only
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out.append(model(**enc).logits.argmax(-1).numpy())  # the larger score's class
    return np.concatenate(out)


# =============================================================================
# FINE-TUNING -- NEW
# =============================================================================
def lr_at(step, total):                                 # NEW: linear warm-up, then linear decay to 0 (the BERT schedule)
    warm = max(1, int(WARMUP_FRAC * total))               # warm-up steps
    return LR * (step + 1) / warm if step < warm else LR * max(0.0, (total - step) / (total - warm))


def fine_tune(job):                                     # NEW: one run; returns its dev curve and best checkpoint's test predictions
    n, seed = job                                         # training items, seed
    torch.set_num_threads(THREADS_PER_RUN); torch.manual_seed(seed)  # this process's threads; the head's initial weights and dropout
    (Xtr, ytr), (Xd, yd), (Xt, yt) = data()
    Xtr, ytr = Xtr[:n], ytr[:n]                           # the first n training items
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)  # pretrained body + a NEW random 2-class head
    no_decay = lambda name: name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name  # the usual exemptions from weight decay
    groups = [{"params": [p for nm, p in model.named_parameters() if not no_decay(nm)], "weight_decay": WEIGHT_DECAY},
              {"params": [p for nm, p in model.named_parameters() if no_decay(nm)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=LR)                # Adam with decoupled weight decay
    steps_per_epoch = int(np.ceil(n / BATCH)); total = EPOCHS * steps_per_epoch  # the whole budget
    check_at = set(np.linspace(0, total, CHECKS + 1).astype(int)[1:])  # evenly spaced dev checks, the last at the end
    rng = np.random.RandomState(seed)                     # batch order
    curve = [(0, (predict(tok, model, Xd) == yd).mean())]  # dev accuracy before any training (the head is random)
    best = (curve[0][1], 0, copy.deepcopy(model.state_dict()))  # (dev accuracy, step, weights)
    t0, step = time.perf_counter(), 0                     # time it; steps so far
    for epoch in range(EPOCHS):                           # passes over the n items
        order = rng.permutation(n)                          # a new order each epoch
        for i in range(0, n, BATCH):                        # mini-batches
            model.train()                                     # dropout on while training
            idx = order[i:i + BATCH]
            enc = tok([Xtr[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            loss = model(**enc, labels=torch.tensor(ytr[idx])).loss  # cross-entropy, computed by the model
            loss.backward()                                   # every gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip the global norm at 1 (Day 60)
            for g in opt.param_groups:                        # this step's learning rate
                g["lr"] = lr_at(step, total)
            opt.step(); opt.zero_grad(); step += 1           # update, clear gradients, count
            if step in check_at:                              # a dev check
                acc = (predict(tok, model, Xd) == yd).mean(); curve.append((step, acc))
                if acc > best[0]:                               # a new best checkpoint
                    best = (acc, step, copy.deepcopy(model.state_dict()))
    minutes = (time.perf_counter() - t0) / 60             # training time, checks included
    model.load_state_dict(best[2])                        # back to the best checkpoint
    pt = predict(tok, model, Xt)                          # its test predictions
    return dict(n=n, seed=seed, curve=np.array(curve), best_step=best[1], dev=best[0], test=(pt == yt).mean(),
                correct=(pt == yt).astype(float), steps=total, minutes=minutes)


def run_parallel(jobs):                                 # one process per job
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        return pool.map(fine_tune, jobs)


# =============================================================================
# FROM-SCRATCH COMPARISONS -- NEW
# =============================================================================
def paired_bootstrap(a, b, n_boot=2000, seed=0):        # NEW: 95% interval for accuracy(a) - accuracy(b) on the SAME examples
    rng = np.random.RandomState(seed)                     # fixed rng
    idx = rng.randint(0, len(a), (n_boot, len(a)))        # resample example indices, shared by both models
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)         # the difference in each resample
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


def cls_features(texts, bs=32):                         # NEW: the frozen encoder's [CLS] vector for each text
    tok = AutoTokenizer.from_pretrained(BASE); enc_model = AutoModel.from_pretrained(BASE).eval()  # the body only, no head
    out = []
    with torch.no_grad():                                 # nothing is trained here
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out.append(enc_model(**enc).last_hidden_state[:, 0].numpy())  # position 0 is [CLS]
    return np.concatenate(out)                            # (n, 768)


def logistic(F, y, Fd, yd, l2s=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0), steps=500, lr=0.01):  # NEW: logistic regression on fixed features, L2 chosen on dev
    mu, sd = F.mean(0), F.std(0) + 1e-6                   # standardise with the training features' statistics
    Z, Zd = (F - mu) / sd, (Fd - mu) / sd
    best, devs = None, {}                                 # (dev accuracy, l2, w, b); dev accuracy of every l2
    for l2 in l2s:                                        # each penalty
        w, b = np.zeros(Z.shape[1]), 0.0                    # start at zero
        m, v = np.zeros(Z.shape[1] + 1), np.zeros(Z.shape[1] + 1)  # Adam's averages
        for t in range(1, steps + 1):                       # full-batch Adam
            p = 1 / (1 + np.exp(-(Z @ w + b)))                # P(positive)
            g = np.r_[Z.T @ (p - y) / len(y) + l2 * w, np.mean(p - y)]  # gradient: weights, then bias
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
            s = lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            w -= s[:-1]; b -= s[-1]
        acc = np.mean(((Zd @ w + b) > 0) == yd); devs[l2] = acc  # dev accuracy
        if best is None or acc > best[0]:
            best = (acc, l2, w.copy(), b)
    return best, (mu, sd), devs


# =============================================================================
# DEMOS
# =============================================================================
def demo_setup():
    print("=" * 70)
    print("1. THE SETUP -- a pretrained body and a new head")
    print("=" * 70)
    (Xtr, ytr), (Xd, yd), (Xt, yt) = data()
    torch.manual_seed(0)                                  # the head's random start
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
    body = sum(p.numel() for nm, p in model.named_parameters() if nm.startswith("distilbert."))  # pretrained weights
    head = sum(p.numel() for nm, p in model.named_parameters() if not nm.startswith("distilbert."))  # new weights
    print(f"{BASE}: {body:,} pretrained weights in the body; {head:,} NEW weights in the head "
          f"({', '.join(nm for nm, _ in model.named_parameters() if not nm.startswith('distilbert.'))})")
    tok = AutoTokenizer.from_pretrained(BASE)
    acc = (predict(tok, model, Xd) == yd).mean()          # before any training
    print(f"dev accuracy with the random head, before training: {acc:.3f} (dev: {len(Xd)} sentences, {yd.mean():.3f} positive)")
    print(f"data: training items from a fixed shuffle of SST-2's {len(Xtr):,}; dev {len(Xd)} and test {len(Xt)} sentences "
          f"(SST-2 validation split in two, test {yt.mean():.3f} positive)")
    print(f"training: AdamW, peak learning rate {LR}, {int(WARMUP_FRAC * 100)}% warm-up then linear decay to 0, batches of "
          f"{BATCH}, {EPOCHS} epochs, inputs cut at {MAX_LEN} tokens, gradient norm clipped at 1, weight decay {WEIGHT_DECAY} "
          f"(not on biases or LayerNorm); {CHECKS} dev checks per run, best dev checkpoint kept")
    print("Actual output from this lesson's run.")


def demo_runs():
    print("\n" + "=" * 70)
    print("2. FIVE FINE-TUNING RUNS (in parallel processes)")
    print("=" * 70)
    t0 = time.perf_counter()                              # wall time for all five
    res = run_parallel(RUNS)
    wall = (time.perf_counter() - t0) / 60
    print(f"{len(RUNS)} runs at the same time, {THREADS_PER_RUN} threads each (wall time {wall:.0f} minutes; timings vary between runs)")
    print(f"\n{'items':>6}{'seed':>6}{'steps':>7}{'best step':>11}{'dev':>8}{'test':>8}{'test 95% interval':>21}{'minutes':>9}")
    for r in res:                                         # one row per run
        lo, hi = d98.bootstrap_ci(r["correct"])
        print(f"{r['n']:>6}{r['seed']:>6}{r['steps']:>7}{r['best_step']:>11}{r['dev']:>8.3f}{r['test']:>8.3f}"
              f"      [{lo:.3f}, {hi:.3f}]{r['minutes']:>9.1f}")
    t = [r["test"] for r in res if r["n"] == 1000]
    print(f"1,000 items, three seeds: test {', '.join(f'{x:.3f}' for x in t)}; spread {max(t) - min(t):.3f}")
    print("Actual output from this lesson's run.")
    return res                                              # later sections use them


def demo_alternatives(res):
    print("\n" + "=" * 70)
    print("3. THE ALTERNATIVES ON THE SAME TEST HALF")
    print("=" * 70)
    torch.set_num_threads(d98.THREADS)                    # back to Day 98's thread count in this process
    (Xtr, ytr), (Xd, yd), (Xt, yt) = data()
    print("Frozen DistilBERT: its [CLS] vectors as fixed features, a logistic regression (NumPy) on top, L2 chosen on dev. "
          "Bag of words: Day 98's from-scratch model. Ready-made: Day 98's model, fine-tuned by its authors on all items.")
    rows, dev_rows = {}, []                               # name -> test correct (0/1 per sentence); dev accuracy by L2
    Fd, Ft = cls_features(Xd), cls_features(Xt)           # dev and test features, once
    print(f"\n{'model':<38}{'items':>7}{'test':>8}{'95% interval':>18}")
    for n in (250, 1000, 4000):                           # the same training sizes as the runs
        F = cls_features(Xtr[:n])
        (dacc, l2, w, b), (mu, sd), devs = logistic(F, ytr[:n], Fd, yd); dev_rows.append((n, devs))
        c = ((((Ft - mu) / sd) @ w + b > 0) == yt).astype(float); rows[f"frozen + logistic, {n}"] = c
        bow = d98.BagOfWords(Xtr[:n]).fit(Xtr[:n], ytr[:n])
        cb = (bow.predict(Xt) == yt).astype(float); rows[f"bag of words, {n}"] = cb
        for name, cc, extra in ((f"frozen DistilBERT + logistic (L2 {l2:g})", c, n), ("bag of words (Day 98)", cb, n)):
            lo, hi = d98.bootstrap_ci(cc)
            print(f"{name:<38}{extra:>7,}{cc.mean():>8.3f}      [{lo:.3f}, {hi:.3f}]")
    for r in res:                                         # the fine-tuned runs, seed 0
        if r["seed"] == 0:
            rows[f"fine-tuned, {r['n']}"] = r["correct"]; lo, hi = d98.bootstrap_ci(r["correct"])
            print(f"{'fine-tuned DistilBERT (seed 0)':<38}{r['n']:>7,}{r['test']:>8.3f}      [{lo:.3f}, {hi:.3f}]")
    tok, model = d98.load()                               # Day 98's ready-made model
    c = (d98.predict_proba(tok, model, Xt).argmax(1) == yt).astype(float); rows["ready-made (Day 98)"] = c
    lo, hi = d98.bootstrap_ci(c)
    print(f"{'ready-made (Day 98)':<38}{67349:>7,}{c.mean():>8.3f}      [{lo:.3f}, {hi:.3f}]")
    print("\nthe frozen baseline's dev accuracy for each L2 penalty (the highest is chosen; the test half is never consulted):")
    for n, devs in dev_rows:
        print(f"  {n:>5,} items: " + ", ".join(f"{l2:g}: {a:.3f}" for l2, a in devs.items()))
    print("Actual output from this lesson's run.")
    return rows                                             # section 4 and the plot use them


def demo_paired(rows):
    print("\n" + "=" * 70)
    print("4. PAIRED COMPARISONS -- the difference, resampled on the same sentences")
    print("=" * 70)
    print("For each pair: mean difference in test accuracy, its 95% paired-bootstrap interval, and the share of resamples "
          "in which the first model is ahead.")
    pairs = [("fine-tuned, 1000", "frozen + logistic, 1000"), ("fine-tuned, 1000", "bag of words, 1000"),
             ("fine-tuned, 4000", "fine-tuned, 1000"), ("ready-made (Day 98)", "fine-tuned, 4000"),
             ("frozen + logistic, 4000", "bag of words, 4000")]
    out = []
    for a, b in pairs:                                    # each comparison
        m, lo, hi, share = paired_bootstrap(rows[a], rows[b]); out.append((a, b, m, lo, hi))
        print(f"  {a:<25} minus {b:<25} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  ahead in {share:.3f}")
    a, b = rows["fine-tuned, 1000"], rows["bag of words, 1000"]
    (lo_a, hi_a), (lo_b, hi_b) = d98.bootstrap_ci(a), d98.bootstrap_ci(b)  # the two separate intervals
    print(f"\nwidth of the separate 95% intervals: {hi_a - lo_a:.3f} and {hi_b - lo_b:.3f}; of the paired interval for the "
          f"difference: {out[1][4] - out[1][3]:.3f}; the two models agree on {np.mean(a == b):.3f} of test sentences")
    print("Actual output from this lesson's run.")
    return out                                              # the plot uses them


def plot_results(res, rows):                            # two panels: dev curves | test accuracy by method and size
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))           # canvas
    for r in res:                                             # dev curves
        ax[0].plot(r["curve"][:, 0], r["curve"][:, 1], marker="o", markersize=3, label=f"{r['n']} items, seed {r['seed']}")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("dev accuracy"); ax[0].set_ylim(0.4, 1.0)
    ax[0].set_title("Fine-tuning: dev accuracy while training", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    sizes = [250, 1000, 4000]
    for key, col, lab in (("fine-tuned", "#d62728", "fine-tuned (seed 0)"), ("frozen + logistic", "#2ca02c", "frozen + logistic"),
                          ("bag of words", "#ff7f0e", "bag of words")):
        acc = [rows[f"{key}, {n}"].mean() for n in sizes]
        lo = [acc[i] - d98.bootstrap_ci(rows[f"{key}, {n}"])[0] for i, n in enumerate(sizes)]
        hi = [d98.bootstrap_ci(rows[f"{key}, {n}"])[1] - acc[i] for i, n in enumerate(sizes)]
        ax[1].errorbar(sizes, acc, yerr=[lo, hi], marker="o", capsize=4, color=col, label=lab)
    ax[1].axhline(rows["ready-made (Day 98)"].mean(), color="#1f77b4", linestyle="--", label="ready-made (67,349 items)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("training items (log scale)"); ax[1].set_ylabel("test accuracy (436 sentences)")
    ax[1].set_title("Test accuracy against training items", fontsize=10); ax[1].legend(fontsize=8, loc="lower right"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("bert_finetune_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved bert_finetune_results.png")


def draw_diagram():                                     # three ways to use a pretrained encoder
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(-0.1, 4.4); ax.axis("off")  # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 99: three ways to turn a pretrained encoder into a classifier", ha="center", fontsize=12, fontweight="bold")
    box(1.6, 2.0, 2.6, 1.1, "distilbert-base-uncased\n(masked-word pretraining,\nno sentiment labels)", "#DCEBF7")
    rows = [(3.3, "fine-tune everything", "body + NEW head\nall 66.9M weights trained", "#FBE3E3"),
            (2.0, "frozen features", "[CLS] vector, body frozen\nlogistic regression (NumPy)", "#E5F2E1"),
            (0.7, "ready-made (Day 98)", "fine-tuned by its authors\non all 67,349 items", "#EEEDFE")]
    for y, name, text, col in rows:                                  # the three routes
        box(5.4, y, 2.8, 0.95, f"{name}\n{text}", col, fs=8)
        ax.annotate("", xy=(3.95, y), xytext=(2.92, 2.0), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(8.55, 2.0), xytext=(6.85, y), arrowprops=dict(arrowstyle="->", color="#333333"))
    box(10.6, 2.0, 4.0, 1.3, "SST-2: train on the first n items\nchoose the checkpoint on dev (436)\nreport on test (436)\n+ bag of words baseline", "#FAEEDA")
    plt.savefig("bert_finetune_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved bert_finetune_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 100 moves from one label per sentence to one label per word: named entity recognition with a pretrained "
          "model on a real dataset, scored by entity spans.")


def main():                                             # runs every section in order
    torch.set_num_threads(d98.THREADS)                     # the main process's threads
    demo_setup()                                           # 1. the setup
    res = demo_runs()                                      # 2. the runs
    rows = demo_alternatives(res)                          # 3. the alternatives
    demo_paired(rows)                                      # 4. paired comparisons
    plot_results(res, rows)                                # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 5. Day 100


if __name__ == "__main__":                              # run only when executed directly (and not in the worker processes)
    main()                                                 # kick off the lesson
