"""
Day 85: Word Embeddings -- One-Hot Against Dense, on a Real Vocabulary

A tokenizer (Day 84) turns text into ids. A network needs numbers it can
compute with. Two ways to turn an id into a vector:

    one-hot:  a vector of V numbers (V = vocabulary size), all 0 except a 1 at the word's id
    dense:    a row of a learned table E of V rows and d columns, d much smaller than V

They are closer than they look: a one-hot vector times a matrix IS a row of
that matrix. The real differences are size, sharing and what the vectors
can come to mean. Today measures them on Day 62's real vocabulary (2,082
entries) and its part-of-speech tagging task, with a small tagger that
reads a word and its two neighbours.

What is measured, reported as it came out:
  1. One-hot times a matrix is a lookup: equal results, very different cost.
  2. One-hot vectors know nothing: every pair of words is equally far apart.
  3. The same tagger with one-hot input, a frozen random table, and learned
     tables of 2 to 128 numbers per word, three seeds each.
  4. What the learned table knows: nearest neighbours, and whether words
     that share a part of speech end up close together.

Day 62's data code is reused verbatim. Every line of code below carries its
own comment, continuing Day 52's convention.
"""
import copy                                            # copy: deepcopy, to keep the best model during training
import os                                              # os: build the path to Week 9's dataset folder
import time                                            # time: perf_counter, for timings
from collections import Counter                        # Counter counts things (the vocabulary)

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
DATA_DIR = os.path.join(HERE, "..", "week-09-sequence-data-rnns-i", "ud_english_ewt_pos")  # Day 62's dataset, used where it is
PAD, UNK = 0, 1                                         # Day 62's reserved ids: 0 = padding, 1 = a word not in the vocabulary
SEEDS = [0, 1, 2]                                       # three seeds for every trained comparison
EPOCHS = 30                                   # training budget
LRS = (0.25, 0.5, 1.0, 2.0)                             # learning rates tried on validation, for each kind of input
DIMS = (2, 8, 32, 128)                                  # dense table widths compared


# =============================================================================
# 0. REUSED -- Day 62's data code, verbatim
# =============================================================================
def read_tagged(name):                                  # Day 62: read one "word<TAB>tag" file; a blank line ends a sentence
    sentences, current = [], []                           # all sentences so far, and the one being read
    with open(os.path.join(DATA_DIR, name + ".txt"), encoding="utf-8") as f:  # open the file
        for line in f:                                      # one token per line
            line = line.rstrip("\n")                          # drop the newline
            if not line:                                      # blank line -> the sentence is finished
                sentences.append(current); current = []         # store it, start a new one
            else:                                             # a "word<TAB>tag" line
                word, tag = line.split("\t")                    # the two columns
                current.append((word.lower(), tag))             # lowercase the word, keep the tag
    return sentences                                        # a list of sentences, each a list of (word, tag) pairs


def encode(sentences, stoi, ttoi):                      # Day 62: sentences -> padded id arrays plus a tag array
    T = max(len(s) for s in sentences)                    # the longest sentence sets the width
    ids = np.full((len(sentences), T), PAD, dtype=int)    # word ids, padding everywhere to start
    tags = np.zeros((len(sentences), T), dtype=int)       # tag ids; the value under padding is never used
    lens = np.zeros(len(sentences), dtype=int)            # the real length of each sentence
    for row, sent in enumerate(sentences):                 # one sentence at a time
        ids[row, :len(sent)] = [stoi.get(w, UNK) for w, _ in sent]  # word -> id, unknown words -> UNK
        tags[row, :len(sent)] = [ttoi[t] for _, t in sent]   # tag -> id
        lens[row] = len(sent)                                # remember where the real words stop
    mask = np.arange(T)[None, :] < lens[:, None]           # True on real words, False on padding
    return ids, tags, mask                                  # everything a model needs for these sentences


def prepare_data():                                     # Day 62: load the three files, build the vocabularies, encode
    splits = {name: read_tagged(name) for name in ["train", "val", "test"]}  # sentences of (word, tag) pairs
    counts = Counter(w for s in splits["train"] for w, _ in s)  # word counts in the TRAINING sentences
    itos = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= 2)  # word vocabulary, as on Day 61
    stoi = {w: i for i, w in enumerate(itos)}               # word -> id
    tag_names = sorted({t for s in splits["train"] for _, t in s})  # the tag set, alphabetical
    ttoi = {t: i for i, t in enumerate(tag_names)}          # tag -> id
    data = [encode(splits[name], stoi, ttoi) for name in ["train", "val", "test"]]  # (ids, tags, mask) x 3
    return itos, tag_names, data                            # what this lesson needs


# =============================================================================
# 1. THE WINDOW TAGGER -- NEW: a word and its two neighbours, as one-hot rows or dense vectors
# =============================================================================
def windows(ids, mask):                                 # NEW: every real word with its left and right neighbours (PAD at the edges)
    left = np.concatenate([np.full((len(ids), 1), PAD), ids[:, :-1]], axis=1)  # the word before each position
    right = np.concatenate([ids[:, 1:], np.full((len(ids), 1), PAD)], axis=1)  # the word after
    return np.stack([left[mask], ids[mask], right[mask]], axis=1)  # (real words, 3): previous, this, next


class WindowTagger:                                     # NEW: [x(prev); x(word); x(next)] -> tanh hidden layer -> tag
    def __init__(self, V, n_tags, kind="dense", d=32, H=64, lr=0.5, seed=0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.V, self.kind, self.lr = V, kind, lr           # vocabulary, input kind, learning rate
        width = 3 * V if kind == "onehot" else 3 * d       # the input layer's width
        if kind != "onehot":                               # dense or frozen: one table shared by the three positions
            self.E = rng.randn(V, d) * 0.1                   # Day 62's starting spread
        self.W1 = rng.randn(width, H) / np.sqrt(3 * (d if kind != "onehot" else 1)); self.b1 = np.zeros(H)  # hidden layer
        self.W2 = rng.randn(H, n_tags) / np.sqrt(H); self.b2 = np.zeros(n_tags)  # output layer

    def inputs(self, win):                                 # (n, 3) word ids -> (n, width) input rows
        if self.kind == "onehot":                          # the literal one-hot version: three V-long rows of 0s and one 1
            X = np.zeros((len(win), 3 * self.V))             # mostly zeros
            X[np.arange(len(win))[:, None], win + np.arange(3) * self.V] = 1.0  # one 1 in each of the three blocks
            return X                                         # (n, 3V)
        return self.E[win].reshape(len(win), -1)          # dense: three table rows side by side, (n, 3d)

    def forward(self, win):                                # -> tag probabilities (n, T)
        self.win, self.X = win, self.inputs(win)           # inputs
        self.Hh = np.tanh(self.X @ self.W1 + self.b1)      # hidden layer
        z = self.Hh @ self.W2 + self.b2                    # scores
        e = np.exp(z - z.max(axis=1, keepdims=True))       # softmax ...
        self.p = e / e.sum(axis=1, keepdims=True)          # ... per word
        return self.p                                      # hand back

    def step(self, win, y):                                # one SGD step on the mean cross-entropy
        p = self.forward(win)                              # forward
        n = len(y)                                         # batch size
        dz = p.copy(); dz[np.arange(n), y] -= 1; dz /= n   # softmax + cross-entropy gradient
        dW2, db2 = self.Hh.T @ dz, dz.sum(axis=0)          # output layer
        dpre = (dz @ self.W2.T) * (1 - self.Hh ** 2)       # back through tanh
        dW1, db1 = self.X.T @ dpre, dpre.sum(axis=0)       # hidden layer
        if self.kind == "dense":                           # the table is trained too
            dX = (dpre @ self.W1.T).reshape(n, 3, -1)        # gradient for each of the three vectors
            dE = np.zeros_like(self.E); np.add.at(dE, win, dX)  # Day 61: rows that appear several times add up
            self.E -= self.lr * dE                           # update the table
        for W, g in ((self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2)):  # the layers
            W -= self.lr * g                                 # plain SGD, in place

    def predict(self, win):                                # most probable tag per word, in chunks
        return np.concatenate([self.forward(win[s:s + 4096]).argmax(axis=1) for s in range(0, len(win), 4096)])


def fit(data, V, n_tags, seed, epochs=EPOCHS, **kw):    # train on the window examples; best epoch on VALIDATION
    (win_tr, y_tr), (win_va, y_va), (win_te, y_te) = data  # the three splits
    net = WindowTagger(V, n_tags, seed=seed, **kw)        # fresh model
    rng = np.random.RandomState(seed)                     # shuffles the examples each epoch
    best = (-1.0, None, -1)                               # validation accuracy, snapshot, epoch
    seconds = []                                          # training time per epoch
    for epoch in range(epochs):                           # the budget
        start = time.perf_counter()                         # time the epoch
        perm = rng.permutation(len(y_tr))                   # fresh order
        for s in range(0, len(perm), 64):                   # mini-batches of 64 words
            idx = perm[s:s + 64]                              # this batch
            net.step(win_tr[idx], y_tr[idx])                  # one SGD step
        seconds.append(time.perf_counter() - start)         # training seconds this epoch
        val = (net.predict(win_va) == y_va).mean()          # validation accuracy
        if val > best[0]:                                   # a new best
            best = (val, copy.deepcopy(net), epoch)           # snapshot
    net = best[1]                                         # keep the best
    return dict(net=net, val=best[0], epoch=best[2], test=(net.predict(win_te) == y_te).mean(),
                seconds=np.median(seconds))               # everything later sections need


RUNS = {}                                               # cache: each configuration trains once


def fit_cached(data, V, n_tags, seed, **kw):            # fit(), remembered by its settings
    key = (seed, tuple(sorted(kw.items())))               # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = fit(data, V, n_tags, seed, **kw)      # train and remember
    return RUNS[key]                                      # later requests reuse it


def window_data(data):                                  # Day 62's padded arrays -> (windows, tags) per split
    return [(windows(ids, mask), tags[mask]) for ids, tags, mask in data]


def cosine_neighbours(E, i, k=5):                       # the k rows of E with the largest cosine similarity to row i
    N = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)  # unit-length rows
    sim = N @ N[i]                                        # cosine with every row
    sim[i] = -np.inf                                      # not itself
    return np.argsort(-sim)[:k]                           # the k most similar


# =============================================================================
# DEMOS
# =============================================================================
def demo_lookup(V):
    print("=" * 70)
    print("1. A ONE-HOT ROW TIMES A MATRIX IS A LOOKUP")
    print("=" * 70)
    print(f"Day 62's vocabulary has V = {V} entries. A word's one-hot vector is V numbers, all 0 but one 1. "
          "Multiplying it by a (V x d) matrix picks out one row of that matrix: an embedding lookup.")
    rng = np.random.RandomState(0)                        # fixed rng
    E = rng.randn(V, 32)                                  # a (V x 32) table
    ids = rng.randint(0, V, size=1000)                    # 1,000 random word ids
    one_hot = np.zeros((1000, V)); one_hot[np.arange(1000), ids] = 1.0  # their one-hot rows
    print(f"\nlargest |one_hot @ E - E[ids]| over 1,000 words: {np.abs(one_hot @ E - E[ids]).max():.1e}")
    t_mm = min(timeit(lambda: one_hot @ E) for _ in range(5)); t_lk = min(timeit(lambda: E[ids]) for _ in range(5))  # best of 5
    print(f"time for the 1,000 words: matrix product {t_mm * 1000:.3f} ms, lookup {t_lk * 1000:.3f} ms "
          f"(the product is {t_mm / t_lk:.0f} times slower here; only the ratio is meaningful)")
    print(f"memory for the inputs: one-hot {one_hot.nbytes / 1e6:.1f} MB, ids {ids.nbytes / 1e3:.1f} kB")
    print("Actual output from this lesson's run.")


def timeit(fn, n=20):                                   # mean seconds per call over n calls
    start = time.perf_counter()                           # clock
    for _ in range(n):                                    # n calls
        fn()                                                # the work
    return (time.perf_counter() - start) / n              # per call


def demo_geometry_onehot(V):
    print("\n" + "=" * 70)
    print("2. ONE-HOT VECTORS KNOW NOTHING ABOUT WORDS")
    print("=" * 70)
    print("Distances and similarities between one-hot vectors, for any two different words.")
    a, b, c = np.zeros(V), np.zeros(V), np.zeros(V); a[5] = b[6] = c[700] = 1.0  # three words' one-hot vectors
    print(f"\ndistance(word 5, word 6) = {np.linalg.norm(a - b):.4f}; distance(word 5, word 700) = {np.linalg.norm(a - c):.4f} "
          f"(sqrt 2 = {np.sqrt(2):.4f}); dot products {a @ b:.0f} and {a @ c:.0f}")
    print("Every pair of different words is equally far apart and has similarity 0: 'good' is no closer to 'great' "
          "than to 'the'.")
    print("Actual output from this lesson's run.")


def demo_training(V, nt, data):
    print("\n" + "=" * 70)
    print("3. ONE-HOT, DENSE AND FROZEN INPUTS -- the same window tagger")
    print("=" * 70)
    print(f"Week 9's tagging task (Day 62's data and vocabulary), the word and its two neighbours, hidden layer 64, "
          f"plain SGD, batches of 64 words, {EPOCHS} epochs, best epoch on validation. 'one-hot': three one-hot rows "
          "into the hidden layer; 'dense': one learned table of d numbers per word, shared by the three positions; "
          "'frozen': the same table, random and never trained. Each kind of input gets its own learning rate, chosen "
          "on validation (seed 0, d = 32 for the tables).")
    grid = {}                                             # kind -> chosen learning rate
    print(f"\nvalidation accuracy, seed 0: " + "".join(f"{'lr ' + str(lr):>10}" for lr in LRS))
    for kind in ("onehot", "frozen", "dense"):            # the three kinds
        vals = [fit_cached(data, V, nt, 0, lr=lr, kind=kind, d=32)["val"] for lr in LRS]  # one run per rate
        grid[kind] = LRS[int(np.argmax(vals))]              # chosen on validation
        print(f"  {kind:<26}" + "".join(f"{v:>10.4f}" for v in vals) + f"   -> lr {grid[kind]}")
    print("Actual output from this lesson's run.")
    configs = [("one-hot", dict(kind="onehot")), ("frozen, d = 32", dict(kind="frozen", d=32))] + \
              [(f"dense, d = {d}", dict(kind="dense", d=d)) for d in DIMS]
    print(f"\nthree seeds at the chosen rates (the dense tables all use the d = 32 rate), TEST accuracy:")
    print(f"\n{'input':<16}{'weights':>9}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}{'best epoch':>12}{'s / epoch':>11}")
    table = {}                                            # label -> mean test accuracy
    for label, kw in configs:                             # each input
        runs = [fit_cached(data, V, nt, s, lr=grid[kw["kind"]], **kw) for s in SEEDS]  # three seeds
        net = runs[0]["net"]                                # one of them, to count weights
        n = sum(a.size for a in (net.W1, net.b1, net.W2, net.b2)) + (net.E.size if kw["kind"] == "dense" else 0)  # trained weights
        table[label] = np.mean([r["test"] for r in runs])   # mean test accuracy
        print(f"{label:<16}{n:>9}" + "".join(f"{r['test']:>9.4f}" for r in runs) + f"{table[label]:>9.4f}"
              f"{np.mean([r['epoch'] + 1 for r in runs]):>12.1f}{np.median([r['seconds'] for r in runs]):>11.2f}")
    print("(weights: the ones training changes; the frozen table is not counted. best epoch: mean of the seeds)")
    print("Actual output from this lesson's run.")
    return table, grid                                     # the plot and Section 4 reuse them


def demo_neighbours(itos, V, nt, data, word_tags, grid):
    print("\n" + "=" * 70)
    print("4. WHAT THE LEARNED TABLE KNOWS -- nearest neighbours, dense d = 32, seed 0")
    print("=" * 70)
    print("Cosine similarity between table rows. For a few words, the five nearest. Then, for the 500 most frequent "
          "training words: how often a word's nearest neighbour has the same most-common tag, for the learned "
          "table, for the frozen random table, and what a random pair of these words gives.")
    E = fit_cached(data, V, nt, 0, lr=grid["dense"], kind="dense", d=32)["net"].E  # the learned table
    R = fit_cached(data, V, nt, 0, lr=grid["frozen"], kind="frozen", d=32)["net"].E  # the frozen random one
    stoi = {w: i for i, w in enumerate(itos)}             # word -> id
    print()
    for w in ("the", "is", "good", "said", "you", "very", "3", "january"):  # a few words
        if w in stoi:                                       # in the vocabulary
            print(f"  {w:<8} -> " + ", ".join(itos[j] for j in cosine_neighbours(E, stoi[w])))
    top = [i for i in np.argsort(-word_tags["count"])[:500]]  # the 500 most frequent training words
    tags = word_tags["tag"]                               # each word's most common tag
    def same_tag_share(T):                                # share of the 500 whose nearest neighbour (among the 500) shares its tag
        N = T[top] / (np.linalg.norm(T[top], axis=1, keepdims=True) + 1e-12)  # unit-length rows
        sim = N @ N.T                                         # every cosine similarity
        np.fill_diagonal(sim, -np.inf)                        # a word is not its own neighbour
        return np.mean(tags[top][sim.argmax(axis=1)] == tags[top])  # same tag as the nearest one?

    rng = np.random.RandomState(1); pairs = rng.choice(top, size=(5000, 2))  # random pairs of these words
    chance = np.mean(tags[pairs[:, 0]] == tags[pairs[:, 1]])  # how often two random ones share a tag
    print(f"\nnearest neighbour has the same most-common tag (500 most frequent words): learned {same_tag_share(E):.3f}, "
          f"frozen random {same_tag_share(R):.3f}; two random words {chance:.3f}")
    print("Actual output from this lesson's run.")
    return E, top                                          # the plot reuses them


def word_tag_table(V, data):                            # each vocabulary word's training count and most common tag
    ids, tags, mask = data[0]                             # the training split
    counts = np.zeros(V, dtype=int); np.add.at(counts, ids[mask], 1)  # how often each word occurs
    by_tag = np.zeros((V, tags.max() + 1), dtype=int); np.add.at(by_tag, (ids[mask], tags[mask]), 1)  # word x tag counts
    return dict(count=counts, tag=by_tag.argmax(axis=1))  # the two tables


def plot_results(table, E, top, itos, word_tags, tag_names):  # two panels: accuracy by input | the learned table in 2-D
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))          # canvas
    labels = list(table)                                    # panel 1
    colors = ["#7f7f7f", "#c7c7c7"] + ["#1f77b4"] * len(DIMS)
    ax[0].barh(range(len(labels)), [table[l] for l in labels], color=colors)
    for i, l in enumerate(labels):                          # the value on each bar
        ax[0].text(table[l] + 0.002, i, f"{table[l]:.4f}", va="center", fontsize=8)
    ax[0].set_yticks(range(len(labels))); ax[0].set_yticklabels(labels, fontsize=8); ax[0].invert_yaxis()
    lo = min(table.values()) - 0.05                          # zoom in on the differences
    ax[0].set_xlim(lo, max(table.values()) + 0.03); ax[0].set_xlabel("test accuracy (mean of 3 seeds)")
    ax[0].set_title("The same tagger, different inputs", fontsize=10)
    X = E[top] - E[top].mean(axis=0)                        # panel 2: the 500 words, centred
    U, s, Vt = np.linalg.svd(X, full_matrices=False)        # principal directions (Day 87 does this properly)
    Y = X @ Vt[:2].T                                        # the two largest
    tg = word_tags["tag"][top]                              # their most common tags
    for t, color in zip(["NOUN", "VERB", "ADJ", "ADP", "DET", "PRON"], ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]):
        sel = tg == tag_names.index(t)                        # the words with this tag
        ax[1].scatter(Y[sel, 0], Y[sel, 1], s=10, color=color, label=t, alpha=0.7)
    ax[1].set_xlabel("first principal direction"); ax[1].set_ylabel("second principal direction")
    ax[1].set_title("Learned table, 500 most frequent words (seed 0)", fontsize=10); ax[1].legend(fontsize=7, loc="best", markerscale=1.5)
    plt.tight_layout(); plt.savefig("embeddings_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved embeddings_results.png")


def draw_diagram():                                     # one-hot times a matrix = picking a row
    fig, ax = plt.subplots(figsize=(13, 3.8))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.8); ax.axis("off")     # coordinates in "boxes"
    ax.text(6.5, 3.55, "Day 85: a one-hot vector times a table is a lookup of one row", ha="center", fontsize=12, fontweight="bold")
    for j in range(10):                                         # a one-hot vector of length 10
        ax.add_patch(FancyBboxPatch((0.6 + j * 0.42, 1.6), 0.34, 0.4, boxstyle="round,pad=0.01",
                                    facecolor="#FAEEDA" if j == 3 else "#F2F2F2", edgecolor="#333333", linewidth=0.7))
        ax.text(0.77 + j * 0.42, 1.8, "1" if j == 3 else "0", ha="center", va="center", fontsize=8)
    ax.text(2.7, 1.25, "one-hot: word 3 of V", ha="center", fontsize=8, style="italic")
    ax.text(5.15, 1.8, "x", ha="center", va="center", fontsize=14)
    for r in range(10):                                         # a 10 x 4 table
        for col in range(4):
            ax.add_patch(FancyBboxPatch((5.6 + col * 0.42, 3.05 - r * 0.27), 0.34, 0.2, boxstyle="round,pad=0.01",
                                        facecolor="#DCEBF7" if r == 3 else "#F2F2F2", edgecolor="#333333", linewidth=0.5))
    ax.text(6.4, 0.2, "table E: V rows x d columns", ha="center", fontsize=8, style="italic")
    ax.text(7.75, 1.8, "=", ha="center", va="center", fontsize=14)
    for col in range(4):                                        # the result: row 3
        ax.add_patch(FancyBboxPatch((8.2 + col * 0.42, 1.6), 0.34, 0.4, boxstyle="round,pad=0.01",
                                    facecolor="#DCEBF7", edgecolor="#333333", linewidth=0.7))
    ax.text(9.0, 1.25, "row 3 of E: the word's vector", ha="center", fontsize=8, style="italic")
    ax.text(11.6, 1.8, "in code:\nE[3]", ha="center", va="center", fontsize=10, fontweight="bold")
    plt.savefig("embeddings_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved embeddings_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Today's table learned from tags. Day 86 learns word vectors from raw text alone, by predicting a word's "
          "neighbours: Word2Vec's skip-gram, from scratch.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        itos, tag_names, data = prepare_data()               # Day 62's data and vocabulary
        V, nt = len(itos), len(tag_names)                    # sizes
        demo_lookup(V)                                       # 1. one-hot times a matrix
        demo_geometry_onehot(V)                              # 2. one-hot geometry
        wdata = window_data(data)                            # the window examples
        table, grid = demo_training(V, nt, wdata)            # 3. training
        word_tags = word_tag_table(V, data)                  # each word's count and tag
        E, top = demo_neighbours(itos, V, nt, wdata, word_tags, grid)  # 4. neighbours
        plot_results(table, E, top, itos, word_tags, tag_names)  # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 5. Day 86


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
