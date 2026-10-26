"""
Day 91: Causal Language Modeling -- Next-Token Prediction From Scratch

A language model gives a probability to every possible text. Written one
token at a time, the chain rule of probability makes that a sequence of
next-token predictions:

    p(x_1, ..., x_n) = p(x_1) p(x_2 | x_1) p(x_3 | x_1, x_2) ... p(x_n | x_1 .. x_(n-1))
    loss = -(1/n) sum_t log p(x_t | x_<t)          bits per token = loss / ln 2

Three ways to make the conditional p(next | before), all on Day 90's data
(Tiny Shakespeare, characters, the same 90% / 10% split):

  * counting: n-gram models, with Day 90's add-k smoothing and with
    interpolated Kneser-Ney smoothing (Chen & Goodman, 1998), from scratch
  * a fixed-window neural model (Bengio et al., 2003): embed the last k
    characters, one hidden layer, a softmax
  * Day 90's GPT, whose window is its whole context

And one check that makes the chain rule concrete: summed over every possible
string, a causal model's probabilities add up to exactly 1. Without the
causal mask they do not, so the 'probabilities' mean nothing.

What is measured, reported as it came out:
  1. The chain rule, checked: a causal and a non-causal model, all strings.
  2. Counting models: add-k against Kneser-Ney, 1 to 8 characters of context.
  3. A fixed-window neural model with 2, 4, 8 and 16 characters of context.
  4. All of them side by side with Day 90's GPT.

Day 90's model code is imported from its script (and Week 12's through it).
Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep an imported script quiet
import io                                              # io: StringIO, where that output goes
import itertools                                       # itertools: product, to list every possible string
import os                                              # os: paths
import sys                                             # sys: path, to import Day 90's script
import time                                            # time: perf_counter, for timings
from collections import Counter, defaultdict           # Counter and defaultdict: the n-gram tables

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, os.path.join(HERE, "..", "week-13-tokenization-embeddings"))  # Day 90's folder
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing; kept quiet to be sure
    import day90_tiny_gpt as d90                        # Day 90: the GPT, its data and the add-k counting model

ORDERS = (1, 2, 3, 4, 5, 6, 7, 8, 9)                   # n-gram orders: n - 1 characters of context
DISCOUNTS = (0.5, 0.75, 0.9, 0.95, 0.99)                # Kneser-Ney discounts tried on held-back training text
WINDOWS = (2, 4, 8, 16)                                 # context lengths for the neural window model
EMB, HID = 24, 256                                      # its embedding size per character, hidden units
STEPS, BATCH, LR = 4000, 512, 3e-3                      # its training: Adam steps, windows per batch, learning rate
DAY90 = dict(full_context="2.525", addk4="2.554")      # Day 90's results, copied from day90_run_output.txt


# =============================================================================
# 1. THE CHAIN RULE, CHECKED
# =============================================================================
def sequence_logprob(P, seqs, causal, h=2):             # NEW: log p of each whole sequence, as a sum of next-token log-probabilities
    BOS = P["E"].shape[0] - 1                             # the last token id marks the beginning of a sequence
    x = np.concatenate([np.full((len(seqs), 1), BOS), seqs[:, :-1]], axis=1)  # inputs: BOS, then the sequence shifted right
    n = x.shape[1]                                        # length
    if causal:                                            # Day 90's GPT: each position sees only itself and the past
        lg, _ = d90.gpt_forward(x, P, h)
    else:                                                 # the same weights with Day 78's encoder block: every position sees everything
        z = d90.embed(P["E"], x) + d90.embed(P["pos"], np.arange(n))
        for k in sorted(P["blocks"]):
            z, _ = d90.w12.encoder_block(z, P["blocks"][k], h, np.ones(x.shape, bool))  # no causal mask
        lg = d90.output_layer(z, P["Wout"], P["bout"])
    logp = d90.log_softmax(lg[..., :BOS])                 # a distribution over the real tokens only (BOS is never predicted)
    return np.take_along_axis(logp, seqs[..., None], axis=-1)[..., 0].sum(axis=1)  # sum over positions


# =============================================================================
# 2. COUNTING MODELS
# =============================================================================
class KneserNey:                                        # NEW: interpolated Kneser-Ney n-gram model over characters
    def __init__(self, text, order, D=0.75):              # count every n-gram up to the order
        self.n, self.D, self.V = order, D, sorted(set(text))  # order, discount, the characters
        self.cnt = [Counter() for _ in range(order + 1)]   # cnt[k][g]: how often the k-gram g occurs (raw counts)
        for k in range(1, order + 1):                       # every length
            self.cnt[k].update(text[i:i + k] for i in range(len(text) - k + 1))
        self.cont = [Counter() for _ in range(order + 1)]  # cont[k][g]: how many different characters precede g
        for k in range(1, order):                           # lower orders only
            for g in self.cnt[k + 1]:                         # every (k + 1)-gram seen
                self.cont[k][g[1:]] += 1                        # one more distinct left neighbour for its last k characters
        self.tab = [None] + [self._tables(k) for k in range(1, order + 1)]  # per order: context totals and distinct followers

    def _tables(self, k):                                 # for every context of length k - 1: the total count and how many characters follow it
        src = self.cnt[k] if k == self.n else self.cont[k]  # raw counts at the top order, continuation counts below (the KN idea)
        tot, kinds = defaultdict(float), defaultdict(int)   # context -> total; context -> distinct next characters
        for g, c in src.items():                            # every k-gram
            tot[g[:-1]] += c; kinds[g[:-1]] += 1
        return src, tot, kinds

    def prob(self, ctx, ch, k=None):                      # p(ch | the last k - 1 characters of ctx), interpolated down to a uniform guess
        k = self.n if k is None else k                      # start at the top order
        if k == 0:                                          # below unigrams: every character equally likely
            return 1.0 / len(self.V)
        src, tot, kinds = self.tab[k]                       # this order's tables
        h = ctx[len(ctx) - (k - 1):] if k > 1 else ""       # the context this order uses
        if tot[h] == 0:                                     # context never seen: use the lower order alone
            return self.prob(ctx, ch, k - 1)
        top = max(src[h + ch] - self.D, 0.0) / tot[h]       # discounted count
        lam = self.D * kinds[h] / tot[h]                    # the probability mass taken away, given to the lower order
        return top + lam * self.prob(ctx, ch, k - 1)

    def bits(self, text, start):                          # mean bits per character of text[start:], each with its real preceding context
        return -np.mean([np.log2(self.prob(text[max(0, i - self.n + 1):i], text[i])) for i in range(start, len(text))])


# =============================================================================
# 3. A FIXED-WINDOW NEURAL MODEL (Bengio et al., 2003)
# =============================================================================
class WindowLM:                                         # NEW: the last k characters -> embeddings side by side -> tanh layer -> softmax
    def __init__(self, V, k, seed=0):
        rng = np.random.RandomState(seed)                   # fixed rng
        self.k = k                                          # context length
        self.P = dict(E=rng.randn(V, EMB) * 0.1, W1=rng.randn(k * EMB, HID) / np.sqrt(k * EMB), b1=np.zeros(HID),
                      W2=rng.randn(HID, V) / np.sqrt(HID) * 0.1, b2=np.zeros(V))  # table, hidden layer, output layer (x 0.1, Day 81)

    def logits(self, X):                                  # X: (n, k) ids of the preceding characters
        self.X = X; self.e = self.P["E"][X].reshape(len(X), -1)  # the k embeddings side by side (Day 85's lookup)
        self.hd = np.tanh(self.e @ self.P["W1"] + self.P["b1"])  # hidden layer
        return self.hd @ self.P["W2"] + self.P["b2"]        # a score for every character

    def grads(self, X, y):                                # mean cross-entropy and its gradients
        lg = self.logits(X); n = len(y)                     # forward
        loss, dz = d90.softmax_xent(lg, y)                  # Day 86's loss (from Day 90's script)
        G = dict(W2=self.hd.T @ dz, b2=dz.sum(axis=0))      # output layer
        dpre = (dz @ self.P["W2"].T) * (1 - self.hd ** 2)   # back through tanh
        G["W1"], G["b1"] = self.e.T @ dpre, dpre.sum(axis=0)  # hidden layer
        G["E"] = d90.embed_backward((dpre @ self.P["W1"].T).reshape(n, self.k, EMB), self.P["E"], X)  # Day 85's backward lookup
        return loss, G


def contexts(ids, k, start):                            # every position from start on, with its k preceding ids
    idx = np.arange(start, len(ids))                      # target positions
    return ids[idx[:, None] - np.arange(k, 0, -1)[None, :]], ids[idx]  # (n, k) contexts, (n,) targets


def train_window(tr, va, V, k):                         # Adam on random positions; held-out bits per character at the end
    m = WindowLM(V, k); state = {"t": 0, "m": {}, "v": {}}  # model; Day 37's Adam state (Day 90's adam_step)
    rng = np.random.RandomState(1)                        # batches
    t0 = time.perf_counter()                              # time it
    for step in range(STEPS):                             # training
        i = rng.randint(k, len(tr), BATCH)                  # random target positions
        X = tr[i[:, None] - np.arange(k, 0, -1)[None, :]]   # their contexts
        _, G = m.grads(X, tr[i])                            # gradients
        d90.adam_step(m.P, G, state, LR)                    # Day 90's Adam with clipping (works on any dict of arrays)
    X, y = contexts(va, k, 16)                            # held-out: every position from 16 on, so all models score the same characters
    lp = d90.log_softmax(np.concatenate([m.logits(X[s:s + 8192]) for s in range(0, len(X), 8192)]))  # log-probabilities
    bits = -lp[np.arange(len(y)), y].mean() / np.log(2)   # bits per character
    return bits, sum(a.size for a in m.P.values()), time.perf_counter() - t0


# =============================================================================
# DEMOS
# =============================================================================
def demo_chain_rule():
    print("=" * 70)
    print("1. THE CHAIN RULE, CHECKED -- does a model define a probability for every string?")
    print("=" * 70)
    print("A tiny untrained GPT (Day 90's, 3 real tokens plus a beginning-of-sequence token, d = 8, 2 blocks). For "
          "every one of the 3^4 = 81 strings of length 4, log p(string) = sum of log p(next token | tokens before). "
          "If the model is a real probability model, the 81 probabilities add up to 1.")
    rng = np.random.RandomState(0)                        # fixed rng
    P = d90.init_gpt(rng, 4, d=8, ff=16, layers=2, t=4)   # 4 ids: tokens 0, 1, 2 and BOS = 3
    for _, A in d90.w12.leaves(P):                        # move the weights off their start so the model is not near-uniform
        A += 0.5 * rng.randn(*A.shape)
    seqs = np.array(list(itertools.product(range(3), repeat=4)))  # all 81 strings
    for causal in (True, False):                          # with and without the causal mask
        p = np.exp(sequence_logprob(P, seqs, causal))       # the probability of each string
        print(f"\n{'causal (Day 90)' if causal else 'no causal mask (Day 78 block)':<30}: sum over 81 strings = {p.sum():.6f}; "
              f"largest {p.max():.4f}, smallest {p.min():.2e}")
    print("Actual output from this lesson's run.")


def demo_counting(tr_text, va_text):
    print("\n" + "=" * 70)
    print("2. COUNTING MODELS -- add-k (Day 90) against Kneser-Ney")
    print("=" * 70)
    print("Both count what followed each context in the training text. Add-k adds k to every count. Kneser-Ney subtracts "
          "a discount D from every seen count, gives the saved probability to the next shorter context, and below the top "
          "order counts how many DIFFERENT characters came before a string instead of how often it occurred. D is chosen "
          "on the last 10% of the training text (order 5), k as on Day 90. Held-out bits per character; every model "
          "scores the same characters (from the 17th on), each with its real preceding context.")
    cut = int(0.9 * len(tr_text))                         # held-back part of the training text
    t0 = time.perf_counter()
    D = min(DISCOUNTS, key=lambda D: KneserNey(tr_text[:cut], 5, D).bits(tr_text[cut:cut + 30000], 16))  # D on held-back text
    print(f"\ndiscount chosen on held-back training text: D = {D} (tried {', '.join(map(str, DISCOUNTS))})")
    print(f"\n{'context':>9}{'add-k':>9}{'Kneser-Ney':>12}{'contexts seen':>15}")
    rows = []                                             # (context length, add-k bits, KN bits)
    for n in ORDERS:                                      # each order
        kn = KneserNey(tr_text, n, D)                       # Kneser-Ney on all the training text
        b_kn = kn.bits(va_text, 16)                         # held-out
        b_ak = d90.ngram_bits(tr_text, va_text[16 - n + 1:], n, len(kn.V))[0] if n <= 6 else np.nan  # Day 90's add-k (up to order 6; slow above)
        rows.append((n - 1, b_ak, b_kn))
        print(f"{n - 1:>9}" + (f"{b_ak:>9.3f}" if not np.isnan(b_ak) else f"{'-':>9}") + f"{b_kn:>12.3f}{len(kn.tab[n][1]):>15,}")
    print(f"({time.perf_counter() - t0:.0f} s for all the counting models)")
    print("Actual output from this lesson's run.")
    return rows                                            # the plot shows them


def demo_window(tr, va, V):
    print("\n" + "=" * 70)
    print("3. A FIXED-WINDOW NEURAL MODEL -- Bengio et al. (2003)")
    print("=" * 70)
    print(f"The last k characters, each as a {EMB}-number vector (one shared table), side by side into {HID} tanh units, "
          f"then a softmax over the 65 characters. Adam (Day 37, with Day 60's clipping), learning rate {LR}, {STEPS} steps "
          f"of {BATCH} random positions, seed 0. Held-out: the same characters as section 2.")
    print(f"\n{'k':>4}{'weights':>9}{'bits/char':>11}{'seconds':>9}")
    rows = []                                             # (k, bits)
    with threadpool_limits(1, "blas"):                    # one BLAS thread
        for k in WINDOWS:                                   # each window
            b, n, s = train_window(tr, va, V, k)
            rows.append((k, b))
            print(f"{k:>4}{n:>9}{b:>11.3f}{s:>9.0f}")
    print("Actual output from this lesson's run.")
    return rows                                            # the plot shows them


def demo_summary(count_rows, win_rows):
    print("\n" + "=" * 70)
    print("4. SIDE BY SIDE -- the best of each, and Day 90's GPT")
    print("=" * 70)
    bk = min(count_rows, key=lambda r: r[2]); bw = min(win_rows, key=lambda r: r[1])  # the best of each family
    print(f"Kneser-Ney, best ({bk[0]} characters of context): {bk[2]:.3f} bits per character")
    print(f"window model, best (k = {bw[0]}):                {bw[1]:.3f}")
    print(f"Day 90's GPT, at least 32 characters of context: {DAY90['full_context']} (Day 90's run output; scored on Day 90's windows)")
    print(f"Day 90's add-k, 4 characters of context:         {DAY90['addk4']} (Day 90's run output)")
    print("Actual output from this lesson's run.")


def plot_results(count_rows, win_rows):                 # one panel: bits per character against context length
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.6))         # canvas
    c = np.array([r[0] for r in count_rows])                 # context lengths
    ax.plot(c, [r[1] for r in count_rows], marker="o", color="#7f7f7f", label="counting, add-k (Day 90)")
    ax.plot(c, [r[2] for r in count_rows], marker="o", color="#1f77b4", label="counting, Kneser-Ney")
    ax.plot([r[0] for r in win_rows], [r[1] for r in win_rows], marker="s", color="#ff7f0e", label="window neural model")
    ax.axhline(float(DAY90["full_context"]), color="#d62728", linestyle="--", label="Day 90's GPT (32+ characters of context)")
    ax.set_xscale("symlog", linthresh=1); ax.set_xticks([0, 1, 2, 4, 8, 16]); ax.set_xticklabels(["0", "1", "2", "4", "8", "16"])
    ax.set_ylim(1.8, 5.0); ax.set_xlabel("characters of context"); ax.set_ylabel("held-out bits per character")
    ax.set_title("Next-character prediction: three ways", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("causal_lm_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved causal_lm_results.png")


def draw_diagram():                                     # the chain rule as next-token predictions
    fig, ax = plt.subplots(figsize=(12, 3.6))                   # canvas
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")     # coordinates in "boxes"
    ax.text(6, 3.35, "Day 91: p(text) is a product of next-token predictions", ha="center", fontsize=12, fontweight="bold")
    toks = ["BOS", "R", "O", "M", "E"]                             # a toy text: "ROME"
    for i, t in enumerate(toks[:4]):                               # the inputs: the text shifted right (the last character is only a target)
        ax.add_patch(FancyBboxPatch((0.8 + i * 2.2, 1.95), 1.4, 0.5, boxstyle="round,pad=0.03", facecolor="#FAEEDA", edgecolor="#333333"))
        ax.text(1.5 + i * 2.2, 2.2, t, ha="center", va="center", fontsize=10)
    for i in range(4):                                             # each prediction reads only what came before
        ax.add_patch(FancyBboxPatch((0.6 + i * 2.2, 0.75), 1.8, 0.55, boxstyle="round,pad=0.03", facecolor="#DCEBF7", edgecolor="#333333"))
        ax.text(1.5 + i * 2.2, 1.02, f"p({toks[i + 1]} | {''.join(toks[1:i + 1]) or 'BOS'})", ha="center", va="center", fontsize=8.5)
        ax.annotate("", xy=(1.5 + i * 2.2, 1.32), xytext=(1.5 + i * 2.2, 1.93), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(0.3, 2.2, "inputs", ha="right", va="center", fontsize=8.5); ax.text(0.3, 1.02, "predictions", ha="right", va="center", fontsize=8.5)
    ax.text(6, 0.25, "multiply the four = p('ROME');  the causal mask makes sure no prediction sees its own answer",
            ha="center", fontsize=8.5, style="italic")
    plt.savefig("causal_lm_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved causal_lm_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("The GPT is only one way to make the conditional. Day 92 trains it properly -- a schedule, a bigger model, "
          "full-context measurement -- to see whether it beats the best counting model.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        tr_text, va_text = d90.load_text()                   # Day 90's split
        _, vocab, tr, va = d90.tokenize(tr_text, va_text, 0)  # characters (Day 84's tokenizer, 0 merges)
        demo_chain_rule()                                    # 1. chain rule
        count_rows = demo_counting(tr_text, va_text)         # 2. counting
    win_rows = demo_window(tr, va, len(vocab))              # 3. window model
    demo_summary(count_rows, win_rows)                     # 4. side by side
    plot_results(count_rows, win_rows)                     # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 5. Day 92


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
