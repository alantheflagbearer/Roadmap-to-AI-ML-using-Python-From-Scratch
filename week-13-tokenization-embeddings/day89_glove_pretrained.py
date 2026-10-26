"""
Day 89: Pretrained Embeddings -- GloVe From Scratch, Saved, Loaded and Used

Word2Vec (Day 86) learns from one (centre, context) pair at a time. GloVe
(Pennington, Socher & Manning, 2014) first counts how often every two words
appear near each other in the whole corpus, then fits vectors to those
counts:

    X_ij = how often word j appears within 10 words of word i (nearer words count more: 1 / distance)
    loss = sum over pairs with X_ij > 0 of  f(X_ij) (w_i . w~_j + b_i + b~_j - log X_ij)^2
    f(x) = (x / 100)^0.75, at most 1   -- rare pairs count less, very common pairs do not dominate

The vectors are written to a text file in GloVe's own format (one line per
word: the word, then its numbers), loaded back, and used the way pretrained
vectors are used: as the starting table of a model trained on a different,
labelled task -- Day 85's part-of-speech tagger, on Day 62's data -- frozen,
or fine-tuned, with all the labelled sentences or only a tenth of them.

The pretraining corpus is Day 86's 2,000,000 words of Wikipedia text, which
is real but small: the GloVe vectors people download were trained on
billions of words. That changes how much they help, not how they are used.

What is measured, reported as it came out:
  1. The co-occurrence counts, and a gradient check of GloVe's loss.
  2. Training, measured after every epoch; GloVe against Day 86's Word2Vec.
  3. Saving and loading: GloVe's text format, and Day 86's word2vec format.
  4. Pretrained vectors in Day 85's tagger: random, Word2Vec or GloVe;
     frozen or fine-tuned; all or a tenth of the labelled sentences.

Day 85's tagger and Day 86's corpus and scores are imported from their own
scripts. Every line of code below carries its own comment, continuing Day
52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import copy                                            # copy: deepcopy, to keep the best tagger during training
import gzip                                            # gzip: vector files are stored compressed
import io                                              # io: StringIO, where that output goes
import os                                              # os: the folder this script lives in
import sys                                             # sys: path, to import Days 85 and 86
import time                                            # time: perf_counter, for timings

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Days 85 and 86 sit in the same folder
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing; kept quiet to be sure
    import day85_word_embeddings as d85                 # Day 85: Day 62's data, the window tagger
    import day86_word2vec_skipgram as d86               # Day 86: the corpus, vocabulary and scores

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

D = 50                                                  # numbers per word, as Day 86
WIN = 10                                                # co-occurrence window, GloVe's default
XMAX, ALPHA = 100.0, 0.75                               # the weighting function's settings (GloVe's defaults)
LR, EPOCHS, BATCH = 0.05, 25, 4096                      # AdaGrad's learning rate (GloVe's default), passes, pairs per update
GLOVE_FILE = os.path.join(HERE, "day89_glove_vectors.txt.gz")  # where today's vectors are saved
W2V_FILE = os.path.join(HERE, "day86_vectors.txt.gz")  # Day 86's saved vectors
LRS = d85.LRS                                           # Day 85's learning-rate grid, chosen on validation per setting
SEEDS = [0, 1, 2]                                       # three seeds per tagger setting
FRACTIONS = (1.0, 0.1)                                  # all the labelled training sentences, or the first tenth


# =============================================================================
# 1. CO-OCCURRENCE COUNTS AND GLOVE
# =============================================================================
def cooccurrence(ids, V, window=WIN):                   # NEW: X_ij summed over the corpus, as (i, j, X_ij) for nonzero pairs
    keys, wts = np.zeros(0, dtype=np.int64), np.zeros(0)  # pair keys i * V + j so far, and their summed weights
    for off in range(1, window + 1):                      # distance between the two words
        a, b = ids[:-off].astype(np.int64), ids[off:].astype(np.int64)  # every pair at this distance
        k = np.concatenate([keys, a * V + b, b * V + a])    # both orders (the counts are symmetric), added to the totals so far
        w = np.concatenate([wts, np.full(2 * len(a), 1.0 / off)])  # each new pair weighs 1 / distance
        keys, inv = np.unique(k, return_inverse=True)       # distinct pairs
        wts = np.bincount(inv, weights=w)                   # summed weight per distinct pair
    return keys // V, keys % V, wts                       # rows, columns, counts


def glove_loss_grads(W, Wt, b, bt, i, j, x):            # NEW: GloVe's weighted squared error and its gradients, for pairs (i, j)
    f = np.minimum(1.0, (x / XMAX) ** ALPHA)              # the weighting function
    e = (W[i] * Wt[j]).sum(axis=1) + b[i] + bt[j] - np.log(x)  # how far w_i . w~_j + b_i + b~_j is from log X_ij
    g = 2 * f * e                                         # dloss/d(inner part)
    return (f * e ** 2).sum(), g[:, None] * Wt[j], g[:, None] * W[i], g, g  # loss, dW_i, dWt_j, db_i, dbt_j


class GloVe:                                            # NEW: two tables and two biases, trained with AdaGrad
    def __init__(self, V, d=D, seed=0):                   # V words, d numbers each
        rng = np.random.RandomState(seed)                   # fixed rng
        self.W, self.Wt = (rng.rand(V, d) - 0.5) / d, (rng.rand(V, d) - 0.5) / d  # word and context vectors
        self.b, self.bt = np.zeros(V), np.zeros(V)          # biases
        self.acc = [np.ones_like(a) for a in (self.W, self.Wt, self.b, self.bt)]  # AdaGrad's sums of squared gradients (GloVe starts them at 1)

    def step(self, i, j, x, lr):                          # one AdaGrad update from a batch of pairs
        loss, *grads = glove_loss_grads(self.W, self.Wt, self.b, self.bt, i, j, x)  # gradients
        for P, A, g, r in zip((self.W, self.Wt, self.b, self.bt), self.acc, grads, (i, j, i, j)):  # each array
            np.add.at(A, r, g ** 2)                           # accumulate squared gradients per row
            np.add.at(P, r, -lr * g / np.sqrt(A[r]))          # step, smaller for rows that have had large gradients
        return loss                                         # summed over the batch

    def vectors(self):                                    # GloVe's final vectors: word plus context vector
        return self.W + self.Wt


def train_glove(rows, cols, X, itos, stoi, seed=0):     # AdaGrad over the nonzero pairs, scores after every epoch
    model = GloVe(len(itos), seed=seed); rng = np.random.RandomState(seed + 100)  # fresh model; data-order rng
    hist = []                                             # (loss per pair, seconds, category, capitals, gender) per epoch
    for ep in range(EPOCHS):                              # passes
        t0 = time.perf_counter(); perm = rng.permutation(len(X)); loss = 0.0  # time; shuffle; summed loss
        for s in range(0, len(X), BATCH):                   # batches
            k = perm[s:s + BATCH]                             # this batch's pairs
            loss += model.step(rows[k], cols[k], X[k], LR)    # update
        hist.append((loss / len(X), time.perf_counter() - t0) + scores(model.vectors(), itos, stoi))
    return model, np.array(hist)                          # trained model, trajectory


def scores(E, itos, stoi):                              # Day 86's three scores (category, capitals, gender)
    return d86.evaluate(E, itos, stoi)


# =============================================================================
# 2. VECTOR FILES
# =============================================================================
def save_vectors(path, words, M):                       # NEW: GloVe's text format: 'word x1 x2 ... xd', one line per word, no header
    with gzip.open(path, "wt", encoding="utf-8") as f:    # compressed text
        for w, v in zip(words, M):                          # one line per word
            f.write(w + " " + " ".join(f"{x:.5f}" for x in v) + "\n")


def load_vectors(path):                                 # NEW: GloVe's format, or word2vec's (whose first line is 'count dim')
    opener = gzip.open if path.endswith(".gz") else open  # compressed or not
    with opener(path, "rt", encoding="utf-8") as f:       # read text
        lines = f.read().splitlines()                       # every line
    first = lines[0].split()                              # word2vec files start with two integers
    if len(first) == 2 and all(t.isdigit() for t in first):  # a header: skip it
        lines = lines[1:]
    words = [ln.split(" ", 1)[0] for ln in lines]         # the first token of each line
    M = np.array([[float(x) for x in ln.split(" ")[1:]] for ln in lines])  # the rest
    return words, M                                       # words, (n, d) matrix


# =============================================================================
# 3. PRETRAINED VECTORS IN DAY 85'S TAGGER
# =============================================================================
def start_table(ewt_itos, words, M, seed):              # NEW: Day 85's random starting table, with pretrained rows where the word is known
    E = np.random.RandomState(seed).randn(len(ewt_itos), D) * 0.1  # Day 85's start (spread 0.1)
    if M is None:                                         # no pretrained vectors: the random table
        return E, 0
    ix = {w: i for i, w in enumerate(words)}              # pretrained word -> row
    hit = [(r, ix[w]) for r, w in enumerate(ewt_itos) if w in ix]  # tagger rows that have a pretrained vector
    rows, src = np.array([h[0] for h in hit]), np.array([h[1] for h in hit])
    scale = np.linalg.norm(E, axis=1).mean() / np.linalg.norm(M[src], axis=1).mean()  # same average row length as the random rows
    E[rows] = M[src] * scale                              # pretrained rows, rescaled; the rest stay random
    return E, len(hit)                                    # table, how many rows came from the file


def fit_with(wdata, V, nt, seed, E0, train_table, lr, epochs=d85.EPOCHS):  # Day 85's fit(), with a given starting table
    (win_tr, y_tr), (win_va, y_va), (win_te, y_te) = wdata  # the three splits
    net = d85.WindowTagger(V, nt, kind="dense" if train_table else "frozen", d=D, lr=lr, seed=seed)  # Day 85's tagger
    net.E = E0.copy()                                     # the starting table
    rng = np.random.RandomState(seed); best = (-1.0, None)  # shuffles; best validation accuracy and snapshot
    for epoch in range(epochs):                           # Day 85's budget
        perm = rng.permutation(len(y_tr))                   # fresh order
        for s in range(0, len(perm), 64):                   # mini-batches of 64 words
            net.step(win_tr[perm[s:s + 64]], y_tr[perm[s:s + 64]])  # one SGD step (the table moves only if trained)
        val = (net.predict(win_va) == y_va).mean()          # validation accuracy
        if val > best[0]:                                   # a new best
            best = (val, copy.deepcopy(net))                  # snapshot
    return best[0], (best[1].predict(win_te) == y_te).mean()  # validation and test accuracy of the best epoch


# =============================================================================
# DEMOS
# =============================================================================
def demo_counts(ids, itos):
    print("=" * 70)
    print("1. COUNTING FIRST -- the co-occurrence matrix, and GloVe's gradient")
    print("=" * 70)
    print(f"Day 86's corpus as ids (its {len(itos):,}-word vocabulary); every pair within {WIN} words, weighted "
          f"1 / distance, both orders.")
    t0 = time.perf_counter()                              # time the counting
    V = len(itos); rows, cols, X = cooccurrence(ids, V)   # the counts
    print(f"\nnonzero pairs: {len(X):,} of {V * V:,} possible ({len(X) / V ** 2:.2%}); counted in {time.perf_counter() - t0:.0f} s")
    ix = {w: i for i, w in enumerate(itos)}               # word -> id
    keyset = rows * V + cols                              # sorted pair keys, for lookups
    for a, b in [("ice", "solid"), ("steam", "solid"), ("ice", "gas"), ("steam", "gas")]:  # GloVe paper's example words
        k = ix[a] * V + ix[b]; p = np.searchsorted(keyset, k)  # where the pair would be
        print(f"  X({a}, {b}) = {X[p] if p < len(X) and keyset[p] == k else 0.0:.2f}")
    print(f"weight f(X): X = 1 -> {min(1, (1 / XMAX) ** ALPHA):.3f}, X = 10 -> {min(1, (10 / XMAX) ** ALPHA):.3f}, "
          f"X >= {XMAX:.0f} -> 1")
    rng = np.random.RandomState(0)                        # tiny gradient check
    W, Wt, b, bt = rng.randn(6, 3), rng.randn(6, 3), rng.randn(6), rng.randn(6)  # 6 words, 3 numbers
    i, j, x = np.array([0, 2, 0, 5]), np.array([1, 3, 4, 0]), np.array([3.0, 150.0, 0.5, 12.0])  # word 0 twice; one count above XMAX
    _, gW, gWt, gb, gbt = glove_loss_grads(W, Wt, b, bt, i, j, x)  # analytic, per pair
    full = [np.zeros_like(a) for a in (W, Wt, b, bt)]     # collected per row
    for F, g, r in zip(full, (gW, gWt, gb, gbt), (i, j, i, j)):
        np.add.at(F, r, g)
    worst, ok, count = 0.0, True, 0                       # largest difference; every entry passes; entries
    for A, G in zip((W, Wt, b, bt), full):                # every array
        for pos in np.ndindex(A.shape):                     # every entry
            old = A[pos]                                      # remember it
            A[pos] = old + 1e-6; up = glove_loss_grads(W, Wt, b, bt, i, j, x)[0]  # nudge up
            A[pos] = old - 1e-6; dn = glove_loss_grads(W, Wt, b, bt, i, j, x)[0]  # nudge down
            A[pos] = old                                      # restore
            num = (up - dn) / 2e-6; count += 1                # central difference
            worst = max(worst, abs(num - G[pos])); ok &= abs(num - G[pos]) <= 1e-10 + 1e-6 * abs(num)
    print(f"gradient check: {count} entries, largest |numeric - analytic| = {worst:.2e}, all pass: {ok}")
    print("Actual output from this lesson's run.")
    return rows, cols, X                                   # GloVe trains on them


def demo_training(rows, cols, X, itos, stoi):
    print("\n" + "=" * 70)
    print("2. TRAINING GLOVE -- measured after every epoch, against Day 86's Word2Vec")
    print("=" * 70)
    print(f"d = {D}, AdaGrad with learning rate {LR}, {EPOCHS} passes over the nonzero pairs in batches of {BATCH}, "
          f"seed 0; vectors = word + context vector. Day 86's three scores, the whole vocabulary searched.")
    model, h = train_glove(rows, cols, X, itos, stoi)     # train
    print(f"\n{'epoch':>6}{'loss/pair':>11}{'category':>10}{'capitals':>10}{'gender':>8}")
    for e in [e for e in (1, 2, 5, 10, 15, 20, 25) if e <= len(h)]:  # selected epochs
        r = h[e - 1]
        print(f"{e:>6}{r[0]:>11.4f}{r[2]:>10.3f}{r[3]:>10.3f}{r[4]:>8.3f}")
    print(f"({np.median(h[:, 1]):.0f} s per epoch here)")
    w_words, w_M = load_vectors(W2V_FILE)                 # Day 86's saved vectors (its seed 0)
    sw = scores(w_M, w_words, {w: i for i, w in enumerate(w_words)})  # the same scores
    print(f"Day 86's Word2Vec (its saved seed 0 vectors, 4 decimals): category {sw[0]:.3f}, capitals {sw[1]:.3f}, gender {sw[2]:.3f}")
    G = model.vectors()                                   # GloVe's vectors
    print("\nnearest neighbours, GloVe / Word2Vec:")
    for w in ("three", "january", "france", "physics", "king"):  # a few words
        g = d86.neighbours(G, w, itos, stoi, k=5)
        v = d86.neighbours(w_M, w, w_words, {x: i for i, x in enumerate(w_words)}, k=5)
        print(f"  {w:<8} {', '.join(g):<46} / {', '.join(v)}")
    print("Actual output from this lesson's run.")
    return G, h, sw                                        # saved next; plotted


def demo_files(G, itos):
    print("\n" + "=" * 70)
    print("3. SAVING AND LOADING -- the files pretrained vectors come in")
    print("=" * 70)
    save_vectors(GLOVE_FILE, itos, G)                     # today's vectors, GloVe's format
    words, M = load_vectors(GLOVE_FILE)                   # back from the file
    print(f"saved {len(words):,} words x {M.shape[1]} numbers to day89_glove_vectors.txt.gz (GloVe's format, "
          f"{os.path.getsize(GLOVE_FILE) / 1e6:.1f} MB compressed)")
    print(f"loaded back: same words {words == itos}; largest |difference| {np.abs(M - G).max():.1e} "
          f"(5 decimals kept)")
    with gzip.open(GLOVE_FILE, "rt") as f:                # the first line, as stored
        print("first line: " + f.readline()[:72] + " ...")
    ww, wm = load_vectors(W2V_FILE)                       # Day 86's file has a header line
    print(f"Day 86's file (word2vec format, with a 'count dim' header): {len(ww):,} words x {wm.shape[1]} numbers")
    print("Actual output from this lesson's run.")
    return (words, M), (ww, wm)                            # both, for the tagger


def demo_tagger(glove, w2v):
    print("\n" + "=" * 70)
    print("4. USING PRETRAINED VECTORS -- Day 85's tagger, on Day 62's data")
    print("=" * 70)
    ewt_itos, tag_names, data = d85.prepare_data()        # Day 62's data and vocabulary
    V, nt = len(ewt_itos), len(tag_names)                 # sizes
    for name, (words, M) in (("GloVe", glove), ("Word2Vec", w2v)):  # coverage of the tagger's vocabulary
        _, hit = start_table(ewt_itos, words, M, 0)
        print(f"{name}: {hit} of the tagger's {V} vocabulary entries have a pretrained vector "
              f"(text8 has no digits, punctuation or capitals)")
    print(f"Day 85's tagger with a table of {D} numbers per word: 'frozen' never changes the table, 'fine-tuned' trains "
          f"it. Learning rate chosen on validation from {LRS} for every setting (seed 0), then three seeds; "
          f"TEST accuracy. 'a tenth': the first 10% of the training sentences.")
    sources = [("random", None), ("Word2Vec", w2v), ("GloVe", glove)]  # where the starting table comes from
    results = {}                                          # (fraction, source, trained) -> mean test accuracy
    print(f"\n{'training data':<15}{'start':<10}{'table':<12}{'lr':>5}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}")
    for frac in FRACTIONS:                                # all, then a tenth
        tr = tuple(a[:int(round(len(a) * frac))] for a in data[0])  # the first sentences
        wdata = d85.window_data([tr, data[1], data[2]])     # Day 85's windows
        for src, vec in sources:                            # three starting tables
            for trained in (False, True):                     # frozen or fine-tuned
                E0 = lambda s: start_table(ewt_itos, *(vec if vec else (None, None)), s)[0]  # the start, per seed
                vals = [fit_with(wdata, V, nt, 0, E0(0), trained, lr)[0] for lr in LRS]  # validation, seed 0
                lr = LRS[int(np.argmax(vals))]                # chosen on validation
                tests = [fit_with(wdata, V, nt, s, E0(s), trained, lr)[1] for s in SEEDS]  # three seeds
                results[(frac, src, trained)] = np.mean(tests)
                print(f"{('all' if frac == 1 else 'a tenth') + f' ({len(tr[0])})':<15}{src:<10}"
                      f"{'fine-tuned' if trained else 'frozen':<12}{lr:>5}" + "".join(f"{t:>9.4f}" for t in tests)
                      + f"{np.mean(tests):>9.4f}")
    print("Actual output from this lesson's run.")
    return results                                         # the plot shows them


def plot_results(h, sw, results):                       # three panels: GloVe by epoch | tagger, all data | tagger, a tenth
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))          # canvas
    ep = np.arange(1, len(h) + 1)                           # epochs
    for j, (name, color) in enumerate([("category", "#1f77b4"), ("capitals", "#2ca02c"), ("gender", "#d62728")]):
        ax[0].plot(ep, h[:, 2 + j], marker="o", markersize=3, color=color, label=f"GloVe, {name}")
        ax[0].axhline(sw[j], color=color, linestyle="--", linewidth=1, label=f"Word2Vec (Day 86), {name}")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("score"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title("GloVe, measured after every epoch", fontsize=10); ax[0].legend(fontsize=7, loc="center right"); ax[0].grid(alpha=0.3)
    for a, frac, title in ((ax[1], 1.0, "Tagger, all labelled sentences"), (ax[2], 0.1, "Tagger, a tenth of them")):
        x = np.arange(3)                                      # three starting tables
        fr = [results[(frac, s, False)] for s in ("random", "Word2Vec", "GloVe")]
        ft = [results[(frac, s, True)] for s in ("random", "Word2Vec", "GloVe")]
        a.bar(x - 0.2, fr, width=0.4, color="#9ecae1", label="frozen table"); a.bar(x + 0.2, ft, width=0.4, color="#1f77b4", label="fine-tuned")
        for k in range(3):                                    # the numbers on the bars
            a.text(x[k] - 0.2, fr[k] + 0.005, f"{fr[k]:.3f}", ha="center", fontsize=7); a.text(x[k] + 0.2, ft[k] + 0.005, f"{ft[k]:.3f}", ha="center", fontsize=7)
        a.set_xticks(x); a.set_xticklabels(["random start", "Word2Vec", "GloVe"]); lo = min(fr + ft) - 0.05
        a.set_ylim(lo, max(fr + ft) + 0.03); a.set_ylabel("test accuracy (3 seeds)"); a.set_title(title, fontsize=10)
        a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("glove_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved glove_results.png")


def draw_diagram():                                     # pretrain, save, load, reuse
    fig, ax = plt.subplots(figsize=(13, 3.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                     # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.1))

    ax.text(6.5, 3.35, "Day 89: pretrain on raw text, save, load, reuse on a labelled task", ha="center", fontsize=12, fontweight="bold")
    steps = [("2M words of\nWikipedia text", "#FAEEDA"), ("count pairs within\n10 words: X_ij", "#DCEBF7"),
             ("fit w_i . w~_j + b_i + b~_j\nto log X_ij", "#DCEBF7"), ("save: 'word x1 ... x50'\n(GloVe text file)", "#E5F2E1"),
             ("load into Day 85's\ntagger: frozen or\nfine-tuned", "#F2F2F2")]
    for k, (text, color) in enumerate(steps):                        # five steps left to right
        box(1.3 + k * 2.6, 1.7, 2.2, 1.1, text, color, fs=8)
        if k:
            arrow(1.3 + (k - 1) * 2.6 + 1.12, 1.7, 1.3 + k * 2.6 - 1.12, 1.7)
    ax.text(6.5, 0.45, "no labels are used until the last step", ha="center", fontsize=8, style="italic")
    plt.savefig("glove_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved glove_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 90 is Week 13's build: a tiny character-level GPT-style model on Shakespeare, assembled from the week's "
          "exercise parts and Week 12's Transformer parts.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        words = d86.load_text8()                             # Day 86's corpus
        itos, stoi, freq, ids = d86.build_vocab(words)       # Day 86's vocabulary (most frequent first)
        rows, cols, X = demo_counts(ids, itos)               # 1. counts
        G, h, sw = demo_training(rows, cols, X, itos, stoi)  # 2. training
        glove, w2v = demo_files(G, itos)                     # 3. files
        results = demo_tagger(glove, w2v)                    # 4. the tagger
        plot_results(h, sw, results)                         # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 5. Day 90


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
