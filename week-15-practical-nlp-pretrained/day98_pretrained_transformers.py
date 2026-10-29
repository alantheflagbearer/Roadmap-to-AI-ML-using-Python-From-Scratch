"""
Day 98: A Pretrained-Transformers Library, and a Real Pretrained Model

Weeks 11-14 built attention, the Transformer and a mini-GPT from scratch in
NumPy. Week 15 switches to the tools practitioners use: the Hugging Face
`transformers` library (on PyTorch) and models other people pretrained on far
more text and compute than one CPU has.

Today's model is DistilBERT fine-tuned for sentiment on SST-2 (the Stanford
Sentiment Treebank, movie-review sentences labelled positive or negative):
distilbert/distilbert-base-uncased-finetuned-sst-2-english. Today's data is
SST-2's validation set, 872 real sentences.

  1. The library's three objects -- tokenizer, model, pipeline -- and a check
     that the pipeline is just tokenizer -> model -> softmax.
  2. The tokenizer: WordPiece subwords, special tokens, padding and the
     attention mask (does padding change the answer?).
  3. Accuracy on real data, with a bootstrap confidence interval, and
     calibration (are its 99% answers right 99% of the time?).
  4. What pretraining buys: a bag-of-words logistic regression written from
     scratch in NumPy, trained on SST-2's training set, on the same sentences.
  5. Where it fails: the most confident mistakes, negation, sentence length.
  6. Speed on a CPU: batch size and threads.

A caveat that matters for honesty: the model was fine-tuned on SST-2's
training set by its authors, and SST-2's validation set is the standard
development set, which they may have used to choose settings. Its score here
is an upper-end estimate, not a clean held-out test.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import re                                              # re: words, for the bag-of-words model and the negation check
import time                                            # time: perf_counter, for the speed section

import numpy as np                                     # numpy: the from-scratch parts (baseline, bootstrap, calibration)
import torch                                           # torch: the library the pretrained model runs on
from datasets import disable_progress_bars, load_dataset  # datasets: SST-2, downloaded once into the local cache
from transformers.utils import logging as hf_logging   # the library's own logging, to keep the output readable
from huggingface_hub.utils import logging as hub_logging  # the model hub's logging (its sign-in notice)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline  # the library's three entry points
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # the pretrained sentiment model
THREADS = 4                                             # PyTorch CPU threads for everything except the threads test
BATCH = 32                                              # sentences per forward pass
N_BOOT = 2000                                           # bootstrap resamples
TRAIN_SIZES = (1000, 10000, 67349)                      # bag-of-words training sizes (67,349 = all of SST-2's training set)
torch.set_num_threads(THREADS)                          # fixed, so timings are comparable within the run
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()  # the library's progress bars and notices off: the output is the lesson's
disable_progress_bars()                                 # the same for the datasets library
hub_logging.set_verbosity_error()                       # no notice about anonymous downloads


# =============================================================================
# THE MODEL AND THE DATA
# =============================================================================
def load():                                             # the tokenizer and the model, from the local cache after the first download
    tok = AutoTokenizer.from_pretrained(MODEL)            # the tokenizer saved with the model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()  # weights + classification head; eval() turns dropout off
    return tok, model                                   # both, ready to use


def sst2(split):                                        # one split of SST-2 as (sentences, labels)
    d = load_dataset("nyu-mll/glue", "sst2", split=split)  # GLUE's copy of SST-2
    return list(d["sentence"]), np.array(d["label"])      # 0 = negative, 1 = positive


def predict_proba(tok, model, texts, bs=BATCH):         # NEW: class probabilities for many texts, a batch at a time
    out = []                                              # one array per batch
    with torch.no_grad():                                 # inference only: no gradients, less memory, faster
        for i in range(0, len(texts), bs):                  # batch by batch
            enc = tok(texts[i:i + bs], padding=True, truncation=True, return_tensors="pt")  # ids, padded to the batch's longest, and the mask
            out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())  # scores -> probabilities
    return np.concatenate(out)                            # (n, 2)


# =============================================================================
# FROM-SCRATCH PARTS -- NEW
# =============================================================================
def words(s):                                           # lowercase words (letters, digits, apostrophes)
    return re.findall(r"[a-z0-9']+", s.lower())         # runs of letters, digits and apostrophes


def bootstrap_ci(correct, n_boot=N_BOOT, seed=0):       # NEW: 95% interval for an accuracy by resampling the examples
    rng = np.random.RandomState(seed)                     # fixed rng
    idx = rng.randint(0, len(correct), (n_boot, len(correct)))  # n_boot resamples, with replacement
    accs = correct[idx].mean(axis=1)                      # the accuracy of each resample
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)  # the middle 95% of the resampled accuracies


def calibration(conf, correct, n_bins=10):              # NEW: reliability bins and the expected calibration error
    edges = np.linspace(0.5, 1.0, n_bins + 1)             # two classes: confidence is always at least 0.5
    rows, ece = [], 0.0                                   # (low, high, count, mean confidence, accuracy); ECE
    for lo, hi in zip(edges[:-1], edges[1:]):             # each bin
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)  # the answers in it
        if m.sum() == 0:                                    # an empty bin adds nothing
            continue                                    # skip it
        rows.append((lo, hi, m.sum(), conf[m].mean(), correct[m].mean()))  # the bin's size, mean confidence and accuracy
        ece += m.mean() * abs(conf[m].mean() - correct[m].mean())  # gap, weighted by the bin's share of answers
    return rows, ece                                    # the table and the single summary number


class BagOfWords:                                       # NEW: logistic regression on which words occur, trained by Adam
    def __init__(self, texts, min_count=2):               # the vocabulary: words seen at least min_count times
        counts = {}                                         # word -> count
        for t in texts:                                 # every training text ...
            for w in set(words(t)):                     # ... each distinct word once
                counts[w] = counts.get(w, 0) + 1        # texts containing the word
        self.vocab = {w: i for i, w in enumerate(sorted(w for w, c in counts.items() if c >= min_count))}  # word -> column

    def encode(self, texts):                              # (word ids, sentence index) of every distinct known word
        ids, rows = [], []                              # two parallel lists
        for r, t in enumerate(texts):                   # every text, with its row number
            for w in set(words(t)):                     # ... each distinct word once
                if w in self.vocab:                     # unknown words are ignored
                    ids.append(self.vocab[w]); rows.append(r)  # one (word, sentence) pair
        return np.array(ids, dtype=np.int64), np.array(rows, dtype=np.int64)  # as arrays, for bincount

    def scores(self, ids, rows, n):                       # one score per sentence: bias + the weights of its words
        return self.b + np.bincount(rows, weights=self.w[ids], minlength=n)  # sum each sentence's word weights

    def fit(self, texts, y, steps=300, lr=0.05, l2=1e-4):  # full-batch Adam on the mean log-loss plus a small L2 penalty
        ids, rows = self.encode(texts); n = len(texts)  # the training texts as pairs; how many
        self.w, self.b = np.zeros(len(self.vocab)), 0.0     # start at zero: every sentence 50/50
        m, v = np.zeros(len(self.vocab) + 1), np.zeros(len(self.vocab) + 1)  # Adam's averages (weights, then bias)
        for t in range(1, steps + 1):                   # Adam steps (Day 37's update)
            p = 1 / (1 + np.exp(-self.scores(ids, rows, n)))  # predicted P(positive)
            g_s = (p - y) / n                                 # d(mean log-loss)/d(score)
            g = np.r_[np.bincount(ids, weights=g_s[rows], minlength=len(self.vocab)) + l2 * self.w, g_s.sum()]  # gradient: per word, then the bias
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2  # the two moving averages
            step = lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)  # bias-corrected step
            self.w -= step[:-1]; self.b -= step[-1]     # update weights and bias
        return self                                     # so fit() can be chained

    def predict(self, texts):                             # 1 = positive
        ids, rows = self.encode(texts)                  # the texts as (word, sentence) pairs
        return (self.scores(ids, rows, len(texts)) > 0).astype(int)  # score above 0 means P(positive) above 0.5


# =============================================================================
# DEMOS
# =============================================================================
def demo_library(tok, model):
    print("=" * 70)
    print("1. THE LIBRARY -- tokenizer, model, pipeline")
    print("=" * 70)
    n = sum(p.numel() for p in model.parameters())        # every weight
    print(f"model: {MODEL}")
    print(f"  {type(model).__name__}, {n:,} weights, {model.config.n_layers} layers, width {model.config.dim}, "
          f"{model.config.n_heads} heads, labels {model.config.id2label}")
    print(f"tokenizer: {type(tok).__name__}, vocabulary {tok.vocab_size:,}, longest input {tok.model_max_length} tokens")
    texts = ["a gorgeous, witty, seductive movie.", "the plot is nothing but boilerplate cliches."]  # two short reviews
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tok)  # the one-line interface
    res = pipe(texts)                                     # [{'label': ..., 'score': ...}, ...]
    enc = tok(texts, padding=True, return_tensors="pt")   # the same, by hand: ids and mask
    with torch.no_grad():
        logits = model(**enc).logits                        # (2, 2) scores
    probs = torch.softmax(logits, dim=-1).numpy()         # softmax by hand
    for t, r, lg, p in zip(texts, res, logits.numpy(), probs):
        print(f"  {t!r}\n    pipeline: {r['label']} {r['score']:.6f} | by hand: logits {np.round(lg, 3)}, softmax {p.max():.6f}")
    gap = max(abs(r["score"] - p.max()) for r, p in zip(res, probs))
    print(f"largest gap between the pipeline and tokenizer -> model -> softmax: {gap:.1e}")
    print("Actual output from this lesson's run.")


def demo_tokenizer(tok, model, X):
    print("\n" + "=" * 70)
    print("2. THE TOKENIZER -- WordPiece pieces, special tokens, padding")
    print("=" * 70)
    for s in ("The movie wasn't unwatchable, just forgettable.", "Spielberg's best since Jaws"):
        print(f"  {s!r} -> {tok.tokenize(s)}")
    enc = tok(X[:3], padding=True)                        # three real sentences, padded together
    print("\nthree validation sentences padded into one batch (ids and attention mask of the shortest):")
    k = int(np.argmin([sum(m) for m in enc["attention_mask"]]))  # the shortest
    print(f"  ids  {enc['input_ids'][k]}\n  mask {enc['attention_mask'][k]}")
    print(f"  [CLS] = {tok.cls_token_id}, [SEP] = {tok.sep_token_id}, [PAD] = {tok.pad_token_id}")
    n_words = n_split = n_unk = n_tok = 0                 # over the whole validation set
    for s in X:
        ws = s.split()                                      # SST-2 is already split on spaces
        pieces = [tok.tokenize(w) for w in ws]
        n_words += len(ws); n_split += sum(len(p) > 1 for p in pieces); n_tok += sum(len(p) for p in pieces)
        n_unk += sum(t == tok.unk_token for p in pieces for t in p)
    print(f"\nover all {len(X)} validation sentences: {n_words:,} words -> {n_tok:,} pieces ({n_tok / n_words:.3f} per word); "
          f"{n_split / n_words:.3f} of words split into more than one piece; [UNK] pieces: {n_unk}")
    alone = predict_proba(tok, model, X[:64], bs=1)       # each sentence on its own: no padding
    padded = predict_proba(tok, model, X[:64], bs=64)     # all 64 in one batch: most are padded
    print(f"64 sentences one at a time vs padded into one batch: largest probability difference {np.abs(alone - padded).max():.1e}")
    print("Actual output from this lesson's run.")


def demo_accuracy(tok, model, X, y):
    print("\n" + "=" * 70)
    print("3. ACCURACY AND CALIBRATION ON SST-2 VALIDATION")
    print("=" * 70)
    t0 = time.perf_counter()                              # time the whole set
    P = predict_proba(tok, model, X)                      # (872, 2)
    secs = time.perf_counter() - t0
    pred, conf = P.argmax(1), P.max(1)                    # label and confidence
    correct = (pred == y).astype(float)                   # 1 where right
    lo, hi = bootstrap_ci(correct)                        # 95% interval
    print(f"{len(X)} sentences ({(y == 1).sum()} positive, {(y == 0).sum()} negative) in {secs:.1f} s with {THREADS} threads")
    print(f"accuracy {correct.mean():.4f}, 95% bootstrap interval [{lo:.4f}, {hi:.4f}] ({N_BOOT} resamples)")
    cm = np.array([[np.sum((y == a) & (pred == b)) for b in (0, 1)] for a in (0, 1)])  # rows: true, columns: predicted
    print(f"confusion matrix (rows true neg/pos, columns predicted neg/pos): {cm.tolist()}")
    for c, name in ((0, "negative"), (1, "positive")):
        print(f"  {name}: precision {cm[c, c] / cm[:, c].sum():.3f}, recall {cm[c, c] / cm[c, :].sum():.3f}")
    rows, ece = calibration(conf, correct)
    print("\ncalibration: answers grouped by confidence")
    print(f"{'confidence':>14}{'answers':>9}{'mean conf':>11}{'accuracy':>10}")
    for a, b, n, mc, acc in rows:
        print(f"  {a:.2f} - {b:.2f}{n:>9}{mc:>11.3f}{acc:>10.3f}")
    print(f"expected calibration error: {ece:.4f}; answers with confidence above 0.99: {(conf > 0.99).mean():.3f} of all")
    print("Actual output from this lesson's run.")
    return P, correct, rows                                 # later sections use them


def demo_baseline(y, correct):
    print("\n" + "=" * 70)
    print("4. WHAT PRETRAINING BUYS -- a bag-of-words model from scratch")
    print("=" * 70)
    Xtr, ytr = sst2("train")                              # 67,349 labelled phrases and sentences
    Xva, _ = sst2("validation")
    print(f"Logistic regression on which words occur (a word counts if it appears in at least 2 training texts), "
          f"300 full-batch Adam steps, written in NumPy. SST-2's training set has {len(Xtr):,} items, many of them short "
          f"phrases cut from the sentences; mean length {np.mean([len(s.split()) for s in Xtr]):.1f} words, against "
          f"{np.mean([len(s.split()) for s in Xva]):.1f} in validation.")
    rng = np.random.RandomState(0)                        # the random subsets
    order = rng.permutation(len(Xtr))
    res = []                                              # (size, accuracy, interval)
    print(f"\n{'training items':>15}{'vocabulary':>12}{'accuracy':>10}{'95% interval':>20}")
    for n in TRAIN_SIZES:
        sub = order[:n]                                     # the first n of a fixed shuffle
        bow = BagOfWords([Xtr[i] for i in sub]).fit([Xtr[i] for i in sub], ytr[sub])
        c = (bow.predict(Xva) == y).astype(float)
        lo, hi = bootstrap_ci(c); res.append((n, c.mean(), lo, hi))
        print(f"{n:>15,}{len(bow.vocab):>12,}{c.mean():>10.4f}      [{lo:.4f}, {hi:.4f}]")
    lo, hi = bootstrap_ci(correct)
    print(f"{'DistilBERT':>15}{'30,522':>12}{correct.mean():>10.4f}      [{lo:.4f}, {hi:.4f}]")
    both = np.mean((c == 1) & (correct == 1)); only_d = np.mean((c == 0) & (correct == 1)); only_b = np.mean((c == 1) & (correct == 0))
    print(f"on the same sentences (bag of words on all items): both right {both:.3f}, only DistilBERT right {only_d:.3f}, "
          f"only bag of words right {only_b:.3f}")
    print("Actual output from this lesson's run.")
    return res                                              # the plot uses them


def demo_failures(X, y, P, correct):
    print("\n" + "=" * 70)
    print("5. WHERE IT FAILS")
    print("=" * 70)
    conf = P.max(1); wrong = np.where(correct == 0)[0]    # confidence; the mistakes
    print(f"{len(wrong)} mistakes; the five made with the highest confidence:")
    for i in wrong[np.argsort(-conf[wrong])][:5]:
        lab = ["negative", "positive"][y[i]]
        print(f"  true {lab}, said the other with {conf[i]:.4f}: {X[i][:110]!r}")
    neg = np.array([bool(re.search(r"\b(not|no|never|n't)\b|n't", s)) for s in X])  # sentences with a negation word
    print(f"\nwith a negation word (not, no, never, n't): {neg.sum()} sentences, accuracy {correct[neg].mean():.3f}; "
          f"without: {(~neg).sum()}, accuracy {correct[~neg].mean():.3f}")
    L = np.array([len(s.split()) for s in X])             # words per sentence
    cuts = np.percentile(L, [33.3, 66.7])                 # thirds by length
    for name, m in (("short", L <= cuts[0]), ("middle", (L > cuts[0]) & (L <= cuts[1])), ("long", L > cuts[1])):
        print(f"  {name:<6} sentences ({L[m].min()}-{L[m].max()} words, {m.sum()} of them): accuracy {correct[m].mean():.3f}")
    lo_n, hi_n = bootstrap_ci(correct[neg]); lo_o, hi_o = bootstrap_ci(correct[~neg])
    print(f"95% intervals: negation [{lo_n:.3f}, {hi_n:.3f}], no negation [{lo_o:.3f}, {hi_o:.3f}]")
    print("Actual output from this lesson's run.")


def demo_speed(tok, model, X):
    print("\n" + "=" * 70)
    print("6. SPEED ON A CPU -- batch size and threads")
    print("=" * 70)
    texts = X[:128]                                       # the same 128 sentences every time
    print("Sentences per second on the first 128 validation sentences; median of 3 timings each. Timings on this machine "
          "vary from run to run, so only the comparisons within this run mean anything.")
    res = {}                                              # (threads, batch) -> sentences per second
    for th, bs in ((THREADS, 1), (THREADS, 8), (THREADS, 32), (THREADS, 128), (1, 32)):
        torch.set_num_threads(th)                           # this setting
        ts = []
        for _ in range(3):                                  # three timings
            t0 = time.perf_counter(); predict_proba(tok, model, texts, bs=bs); ts.append(time.perf_counter() - t0)
        res[(th, bs)] = len(texts) / np.median(ts)
        print(f"  {th} thread(s), batch {bs:>3}: {res[(th, bs)]:6.1f} sentences per second")
    torch.set_num_threads(THREADS)                        # back to the default
    print("Actual output from this lesson's run.")
    return res                                              # the plot uses them


def plot_results(rows, base, correct, speed):           # three panels: calibration | accuracy vs data | speed
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))           # canvas
    mc = [r[3] for r in rows]; acc = [r[4] for r in rows]; n = [r[2] for r in rows]
    ax[0].plot([0.5, 1], [0.5, 1], ":", color="#7f7f7f", label="perfect calibration")
    ax[0].scatter(mc, acc, s=[max(15, k / 3) for k in n], color="#1f77b4", label="confidence bins (size = answers)")
    ax[0].set_xlabel("mean confidence in bin"); ax[0].set_ylabel("accuracy in bin"); ax[0].set_xlim(0.48, 1.02); ax[0].set_ylim(0.0, 1.05)
    ax[0].set_title("Calibration on SST-2 validation", fontsize=10); ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)
    sizes = [b[0] for b in base]; accs = [b[1] for b in base]
    err = [[b[1] - b[2] for b in base], [b[3] - b[1] for b in base]]
    ax[1].errorbar(sizes, accs, yerr=err, marker="o", capsize=4, color="#ff7f0e", label="bag of words (from scratch)")
    lo, hi = bootstrap_ci(correct)
    ax[1].axhspan(lo, hi, color="#1f77b4", alpha=0.25, label="DistilBERT, 95% interval")
    ax[1].axhline(correct.mean(), color="#1f77b4")
    ax[1].set_xscale("log"); ax[1].set_xlabel("bag-of-words training items (log scale)"); ax[1].set_ylabel("validation accuracy")
    ax[1].set_title("What pretraining buys", fontsize=10); ax[1].legend(fontsize=8, loc="lower right"); ax[1].grid(alpha=0.3)
    keys = list(speed); ax[2].bar(range(len(keys)), [speed[k] for k in keys], color=["#2ca02c"] * 4 + ["#9467bd"])
    ax[2].set_xticks(range(len(keys))); ax[2].set_xticklabels([f"{t} thr\nbatch {b}" for t, b in keys], fontsize=8)
    ax[2].set_ylabel("sentences per second"); ax[2].set_title("CPU speed", fontsize=10); ax[2].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("pretrained_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved pretrained_results.png")


def draw_diagram():                                     # what the pipeline does
    fig, ax = plt.subplots(figsize=(13, 3.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 3.35, "Day 98: what pipeline('sentiment-analysis') does", ha="center", fontsize=12, fontweight="bold")
    steps = [(1.2, "text\n\"a witty movie.\"", "#F2F2F2"), (3.5, "tokenizer\nWordPiece -> ids\n+ [CLS] [SEP], mask", "#FAEEDA"),
             (6.2, "DistilBERT\n6 layers, width 768\n(pretrained)", "#DCEBF7"), (8.8, "classification head\n-> 2 scores (logits)", "#E5F2E1"),
             (11.4, "softmax\n-> POSITIVE 0.9999", "#EEEDFE")]
    for i, (x, t, c) in enumerate(steps):                            # the boxes and the arrows between them
        box(x, 1.6, 2.1 if i else 1.7, 1.3, t, c)
        if i:
            ax.annotate("", xy=(x - 1.08, 1.6), xytext=(steps[i - 1][0] + (1.08 if i > 1 else 0.88), 1.6),
                        arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 0.45, "Checked in section 1: the pipeline's answer equals tokenizer -> model -> softmax done by hand.",
            ha="center", fontsize=9, style="italic")
    plt.savefig("pretrained_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved pretrained_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 99 fine-tunes a BERT-style model ourselves: plain DistilBERT (no sentiment training) on part of SST-2's "
          "training set, on this CPU, and compares it with today's ready-made model and the bag-of-words baseline.")


def main():                                             # runs every section in order
    tok, model = load()                                    # the pretrained model
    X, y = sst2("validation")                              # the 872 validation sentences
    demo_library(tok, model)                               # 1. the library
    demo_tokenizer(tok, model, X)                          # 2. the tokenizer
    P, correct, rows = demo_accuracy(tok, model, X, y)     # 3. accuracy and calibration
    base = demo_baseline(y, correct)                       # 4. the bag-of-words baseline
    demo_failures(X, y, P, correct)                        # 5. failures
    speed = demo_speed(tok, model, X)                      # 6. speed
    plot_results(rows, base, correct, speed)               # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 7. Day 99


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
