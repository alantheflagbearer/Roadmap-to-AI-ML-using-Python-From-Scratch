"""
Day 86: Word2Vec (Skip-Gram) From Scratch, on a Real Small Corpus

Day 85's table learned from part-of-speech tags: someone had to label every
word. Word2Vec (Mikolov et al., 2013) learns word vectors from raw text
alone, with one idea: a word is known by the company it keeps. The
skip-gram model takes each word in turn as the centre and learns to predict
the words around it:

    centre word c  ->  its vector v_c (a row of W_in)  ->  a score u_o . v_c for every context word o

Predicting the context with a full softmax over the vocabulary costs one
dot product per vocabulary word, for every pair. Negative sampling replaces
it with a cheaper question: is this (centre, context) pair real, or one of
k made-up pairs whose context word was drawn at random?

    loss = -log s(u_o . v_c) - sum over k noise words n of log s(-u_n . v_c)      (s = the sigmoid)

The corpus is real text: the first 2,000,000 words of text8 (Wikipedia,
lowercased, letters only), kept in ./corpora. Nobody labels anything.

What is measured, reported as it came out:
  1. How skip-gram turns a sentence into (centre, context) pairs.
  2. The two objectives: gradient checks, and what each costs per batch.
  3. Training, three seeds, measured after every epoch: word categories
     (do months sit next to months?) and two small analogy sets.
  4. Nearest neighbours, and a few analogies answered.
  5. What each ingredient does: no training, no subsampling, a smaller
     window, a single noise word.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import gzip                                            # gzip: the corpus is stored compressed
import os                                              # os: build the path to the corpus
import time                                            # time: perf_counter, for timings
from collections import Counter                        # Counter counts words

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
CORPUS = os.path.join(HERE, "corpora", "text8_first_2M_words.txt.gz")  # the first 2M words of text8
MIN_COUNT = 5                                           # words seen fewer times are dropped, as in word2vec
D = 50                                                  # numbers per word
WINDOW = 5                                              # context words on each side (at most; see skipgram_pairs)
NEG = 5                                                 # noise words per real pair
SUBSAMPLE = 1e-4                                        # frequent-word subsampling threshold (Mikolov et al.)
EPOCHS = 5                                              # passes over the corpus
LR0 = 0.025                                             # word2vec's starting learning rate, falling linearly to near 0
BATCH = 1024                                            # pairs per update
SEEDS = [0, 1, 2]                                       # three seeds for the main run

CATEGORIES = {                                          # hand-made word groups for the category check (all in the vocabulary)
    "country": "france germany italy spain england china japan india russia egypt greece poland sweden canada brazil "
               "mexico portugal ireland austria norway".split(),
    "number": "two three four five six seven eight nine ten twenty".split(),
    "month": "january february march april june july august september october november december".split(),
    "colour": "red blue green yellow black white brown orange purple pink".split(),
    "family": "father mother son daughter brother sister wife husband uncle aunt".split(),
    "element": "hydrogen oxygen carbon nitrogen iron gold silver copper helium sodium".split(),
    "instrument": "piano guitar violin drums flute organ accordion keyboards synthesizer".split(),
    "science": "physics chemistry biology mathematics astronomy geology economics psychology philosophy linguistics".split(),
}
CAPITALS = [("paris", "france"), ("berlin", "germany"), ("rome", "italy"), ("madrid", "spain"), ("london", "england"),
            ("moscow", "russia"), ("tokyo", "japan"), ("beijing", "china"), ("athens", "greece"), ("cairo", "egypt"),
            ("ottawa", "canada"), ("warsaw", "poland"), ("stockholm", "sweden"), ("lisbon", "portugal"),
            ("vienna", "austria"), ("dublin", "ireland"), ("oslo", "norway")]  # capital : country
GENDER = [("king", "queen"), ("man", "woman"), ("father", "mother"), ("son", "daughter"), ("brother", "sister"),
          ("husband", "wife"), ("boy", "girl"), ("uncle", "aunt"), ("prince", "princess"), ("he", "she")]  # male : female


# =============================================================================
# 1. THE CORPUS AND THE PAIRS
# =============================================================================
def load_text8(path=CORPUS):                            # NEW: the corpus as a list of words
    with gzip.open(path, "rt", encoding="ascii") as f:    # decompress while reading
        return f.read().split()                           # text8 is lowercase words separated by single spaces


def build_vocab(words, min_count=MIN_COUNT):            # NEW: frequent words, most frequent first
    counts = Counter(words)                               # how often each word occurs
    itos = [w for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if c >= min_count]  # id -> word
    stoi = {w: i for i, w in enumerate(itos)}             # word -> id
    freq = np.array([counts[w] for w in itos], dtype=float)  # each kept word's count
    ids = np.array([stoi[w] for w in words if w in stoi])  # the corpus as ids; rare words removed (word2vec does the same)
    return itos, stoi, freq, ids                          # vocabulary, lookup, counts, corpus


def subsample(ids, freq, t, rng):                       # NEW: drop frequent words at random (keep prob sqrt(t/f) + t/f)
    f = freq[ids] / len(ids)                              # each token's word frequency
    keep = np.minimum(1.0, np.sqrt(t / f) + t / f)        # word2vec's keep probability: 'the' is mostly dropped, rare words always kept
    return ids[rng.rand(len(ids)) < keep]                 # the kept tokens, in order


def skipgram_pairs(ids, window, rng):                   # NEW: every (centre, context) pair, window shrunk at random per centre
    b = rng.randint(1, window + 1, size=len(ids))         # each centre's own window, 1..window (word2vec: nearer words count more)
    C, O = [], []                                         # centres, contexts
    for off in range(1, window + 1):                      # distance between the two words
        right = b[:-off] >= off                             # centre i reads i + off if its window reaches that far
        C.append(ids[:-off][right]); O.append(ids[off:][right])  # pairs to the right
        left = b[off:] >= off                               # centre i reads i - off
        C.append(ids[off:][left]); O.append(ids[:-off][left])  # pairs to the left
    return np.concatenate(C), np.concatenate(O)           # one entry per pair


def noise_cdf(freq):                                    # NEW: the noise distribution, unigram counts to the power 0.75
    p = freq ** 0.75                                      # flattened: rare words drawn more often than their count says
    return np.cumsum(p / p.sum())                         # cumulative, for sampling with searchsorted


# =============================================================================
# 2. THE TWO OBJECTIVES
# =============================================================================
def sigmoid(z):                                         # 1 / (1 + e^-z), without overflow for large negative z
    return np.exp(-np.logaddexp(0, -z))                   # = 1 / (1 + e^-z)


def sgns_loss_grads(Win, Wout, c, tgt):                 # NEW: negative-sampling loss and gradients for a batch of pairs
    v, U = Win[c], Wout[tgt]                              # centre vectors (n, d); context + noise vectors (n, k+1, d)
    z = np.einsum("nd,nkd->nk", v, U)                     # one score per (pair, candidate)
    loss = np.logaddexp(0, -z[:, 0]).sum() + np.logaddexp(0, z[:, 1:]).sum()  # -log s(z_pos) - sum log s(-z_neg), stable
    gz = sigmoid(z); gz[:, 0] -= 1                        # dloss/dz: s(z) - 1 for the real pair, s(z) for noise
    dv = np.einsum("nk,nkd->nd", gz, U)                   # back to the centre vectors
    dU = gz[:, :, None] * v[:, None, :]                   # back to the context and noise vectors
    return loss, dv, dU                                   # loss summed over pairs, and per-pair gradients


def softmax_loss_grads(Win, Wout, c, o):                # NEW: full-softmax loss and gradients (the version negative sampling replaces)
    v = Win[c]                                            # centre vectors (n, d)
    z = v @ Wout.T                                        # a score for EVERY vocabulary word: (n, V)
    z = z - z.max(axis=1, keepdims=True)                  # stability
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))  # log-softmax
    loss = -logp[np.arange(len(c)), o].sum()             # -log p(context | centre), summed
    gz = np.exp(logp); gz[np.arange(len(c)), o] -= 1      # softmax minus one-hot
    return loss, gz @ Wout, gz.T @ v                      # loss, gradient for the centre vectors, gradient for all of Wout


class SkipGram:                                         # NEW: two tables, trained with negative sampling
    def __init__(self, V, d=D, seed=0):                   # V words, d numbers each
        rng = np.random.RandomState(seed)                   # fixed rng
        self.Win = (rng.rand(V, d) - 0.5) / d               # centre-word vectors: word2vec's small uniform start
        self.Wout = np.zeros((V, d))                        # context-word vectors: word2vec starts them at 0

    def step(self, c, tgt, lr):                           # one update from a batch of (centre, [context, noise...]) rows
        loss, dv, dU = sgns_loss_grads(self.Win, self.Wout, c, tgt)  # gradients
        np.add.at(self.Win, c, -lr * dv)                    # a word used by several pairs collects every update (Day 85)
        np.add.at(self.Wout, tgt, -lr * dU)                 # the same for context and noise words
        return loss                                         # summed over the batch


def train(words_ids, freq, V, seed, window=WINDOW, neg=NEG, t=SUBSAMPLE, epochs=EPOCHS, itos=None, stoi=None):
    rng = np.random.RandomState(seed + 100)               # the data-order rng
    model = SkipGram(V, seed=seed)                        # fresh tables
    cdf = noise_cdf(freq)                                 # the noise distribution
    total, done = epochs * len(words_ids), 0              # tokens to process, and processed so far (for the learning rate)
    hist = []                                             # (loss per pair, category score, capitals, gender) after every epoch
    for ep in range(epochs):                              # passes over the corpus
        kept = subsample(words_ids, freq, t, rng) if t > 0 else words_ids  # a fresh subsample every epoch
        C, O = skipgram_pairs(kept, window, rng)            # this epoch's pairs
        perm = rng.permutation(len(C)); C, O = C[perm], O[perm]  # shuffled
        loss = 0.0                                          # summed over the epoch
        for s in range(0, len(C), BATCH):                   # one batch at a time
            c, o = C[s:s + BATCH], O[s:s + BATCH]             # the batch
            lr = LR0 * max(1e-4, 1 - (done + s / len(C) * len(words_ids)) / total)  # linear decay, as in word2vec
            tgt = np.concatenate([o[:, None], np.searchsorted(cdf, rng.rand(len(c), neg))], axis=1)  # context, then noise
            loss += model.step(c, tgt, lr)                    # update
        done += len(words_ids)                              # an epoch's worth of tokens
        hist.append((loss / len(C) / (1 + neg),) + evaluate(model.Win, itos, stoi))  # per candidate, then the scores
    return model, np.array(hist)                          # trained model, its trajectory


RUNS = {}                                               # cache: each configuration trains once


def train_cached(*args, **kw):                          # train(), remembered by its settings
    key = (kw.get("seed"), kw.get("window", WINDOW), kw.get("neg", NEG), kw.get("t", SUBSAMPLE))  # what defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = train(*args, **kw)                      # train and remember
    return RUNS[key]                                      # later requests reuse it


# =============================================================================
# 3. MEASURING WHAT THE VECTORS KNOW
# =============================================================================
def unit(E):                                            # every row scaled to length 1, so dot products are cosines
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def category_score(E, stoi):                            # NEW: share of category words whose nearest category word is in the same group
    words = [(w, c) for c, ws in CATEGORIES.items() for w in ws]  # (word, group) for every listed word
    ii = np.array([stoi[w] for w, _ in words]); cc = np.array([c for _, c in words])  # ids, groups
    N = unit(E[ii]); S = N @ N.T; np.fill_diagonal(S, -np.inf)  # cosines between listed words; a word is not its own neighbour
    return np.mean(cc[S.argmax(axis=1)] == cc)            # same group as the nearest


def analogy_score(E, pairs, itos, stoi):                # NEW: a:b :: c:? answered by the word nearest to b - a + c (3CosAdd)
    N = unit(E); ok = n = 0                               # unit vectors; right answers; questions
    for a, b in pairs:                                    # one known pair ...
        for c, d in pairs:                                  # ... applied to every other pair
            if a == c: continue                               # not the same pair
            q = N[stoi[b]] - N[stoi[a]] + N[stoi[c]]          # e.g. france - paris + rome
            s = N @ q; s[[stoi[a], stoi[b], stoi[c]]] = -np.inf  # the question words are not allowed as answers
            ok += itos[int(s.argmax())] == d; n += 1          # is the nearest word the expected one?
    return ok / n                                         # share right


def evaluate(E, itos, stoi):                            # the three scores
    return category_score(E, stoi), analogy_score(E, CAPITALS, itos, stoi), analogy_score(E, GENDER, itos, stoi)


def neighbours(E, w, itos, stoi, k=6):                  # the k words with the largest cosine similarity to w
    N = unit(E); s = N @ N[stoi[w]]; s[stoi[w]] = -np.inf  # cosines; not the word itself
    return [itos[i] for i in np.argsort(-s)[:k]]          # most similar first


# =============================================================================
# DEMOS
# =============================================================================
def demo_pairs(words):
    print("=" * 70)
    print("1. FROM TEXT TO PAIRS -- every word in turn is the centre")
    print("=" * 70)
    print("The corpus starts like this. With a window of 2, each centre word is paired with up to 2 words on each side; "
          "the model must predict each context word from the centre word alone.")
    s = words[:8]                                         # the first eight words
    print("\ntext: " + " ".join(s))
    for i in (0, 3):                                      # two centres
        ctx = [s[j] for j in range(max(0, i - 2), min(len(s), i + 3)) if j != i]  # the window around it
        print(f"centre '{s[i]}': pairs " + ", ".join(f"({s[i]}, {w})" for w in ctx))
    print("Actual output from this lesson's run.")


def demo_objectives(V):
    print("\n" + "=" * 70)
    print("2. THE TWO OBJECTIVES -- gradient checks, and the cost of each")
    print("=" * 70)
    print("A tiny case (12 words, 4 numbers each, 3 pairs, 2 noise words each), every table entry nudged by +-1e-6, "
          "Day 62's pass rule. Then one batch of 1,024 pairs at this lesson's size, best of 5 timings.")
    rng = np.random.RandomState(0)                        # fixed rng
    Win, Wout = rng.randn(12, 4) * 0.5, rng.randn(12, 4) * 0.5  # tiny tables
    c, o = np.array([1, 5, 1]), np.array([2, 7, 3])       # centre 1 used twice, so the adding-up matters
    tgt = np.concatenate([o[:, None], rng.randint(0, 12, (3, 2))], axis=1)  # context, then two noise words
    for name, f in [("negative sampling", lambda A, B: sgns_loss_grads(A, B, c, tgt)),
                    ("full softmax", lambda A, B: softmax_loss_grads(A, B, c, o))]:
        _, dv, dU = f(Win, Wout)                            # analytic, per pair
        gin = np.zeros_like(Win); np.add.at(gin, c, dv)     # per-pair centre gradients, collected per word
        if dU.ndim == 3:                                    # negative sampling: per (pair, candidate)
            gout = np.zeros_like(Wout); np.add.at(gout, tgt, dU)
        else:                                               # full softmax: already one row per word
            gout = dU
        worst, ok, count = 0.0, True, 0                     # largest difference; every entry passes; entries
        for A, G in ((Win, gin), (Wout, gout)):             # both tables
            for pos in np.ndindex(A.shape):                   # every entry
                old = A[pos]                                    # remember it
                A[pos] = old + 1e-6; up = f(Win, Wout)[0]       # nudge up
                A[pos] = old - 1e-6; dn = f(Win, Wout)[0]       # nudge down
                A[pos] = old                                    # restore
                num = (up - dn) / 2e-6; count += 1              # central difference
                worst = max(worst, abs(num - G[pos])); ok &= abs(num - G[pos]) <= 1e-10 + 1e-6 * abs(num)
        print(f"{name:<18} gradient check: {count} entries, largest |numeric - analytic| = {worst:.2e}, all pass: {ok}")
    model = SkipGram(V); model.Wout = np.random.RandomState(1).randn(V, D) * 0.1  # this lesson's size
    c = np.random.RandomState(2).randint(0, V, BATCH); o = np.random.RandomState(3).randint(0, V, BATCH)  # a batch
    tgt = np.concatenate([o[:, None], np.random.RandomState(4).randint(0, V, (BATCH, NEG))], axis=1)

    def best(fn):                                         # fastest of 5 timings
        times = []                                          # one per repeat
        for _ in range(5):                                  # 5 repeats
            t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)  # one call
        return min(times)                                   # fastest

    t_neg = best(lambda: sgns_loss_grads(model.Win, model.Wout, c, tgt))  # negative sampling
    t_soft = best(lambda: softmax_loss_grads(model.Win, model.Wout, c, o))  # full softmax
    print(f"\none batch of {BATCH} pairs, V = {V}, d = {D}: negative sampling ({NEG} noise words) {t_neg * 1000:.1f} ms, "
          f"full softmax {t_soft * 1000:.1f} ms -- {t_soft / t_neg:.0f} times slower (only the ratio is meaningful)")
    print(f"scores computed per pair: {1 + NEG} against {V}")
    print("Actual output from this lesson's run.")
    return t_soft / t_neg                                  # the plot shows it


def demo_training(ids, freq, itos, stoi):
    print("\n" + "=" * 70)
    print("3. TRAINING -- three seeds, measured after every epoch")
    print("=" * 70)
    print(f"{len(ids)} tokens after dropping words seen fewer than {MIN_COUNT} times; vocabulary {len(itos)}; d = {D}, "
          f"window up to {WINDOW}, {NEG} noise words, subsampling {SUBSAMPLE}, {EPOCHS} epochs, batches of {BATCH}, "
          f"learning rate {LR0} falling linearly. Scores: category = share of the {sum(len(v) for v in CATEGORIES.values())} "
          f"listed words whose nearest listed word is in the same group ({len(CATEGORIES)} groups); capitals = "
          f"{len(CAPITALS) * (len(CAPITALS) - 1)} questions like paris:france :: rome:?; gender = "
          f"{len(GENDER) * (len(GENDER) - 1)} questions like king:queen :: man:?. The whole vocabulary is searched.")
    print(f"\nseed 0, after each epoch:\n{'epoch':>6}{'loss':>8}{'category':>10}{'capitals':>10}{'gender':>8}")
    t0 = time.perf_counter()                              # time the first run
    _, h = train_cached(ids, freq, len(itos), seed=0, itos=itos, stoi=stoi)  # seed 0
    secs = time.perf_counter() - t0                       # its wall time
    for e, row in enumerate(h, 1):                        # one row per epoch
        print(f"{e:>6}{row[0]:>8.3f}{row[1]:>10.3f}{row[2]:>10.3f}{row[3]:>8.3f}")
    print(f"(loss: mean per scored candidate, the real pair or a noise word; {secs / EPOCHS:.0f} s per epoch here)")
    print(f"\nfinal scores per seed:\n{'seed':>6}{'category':>10}{'capitals':>10}{'gender':>8}")
    finals = []                                           # one row per seed
    for s in SEEDS:                                       # three seeds
        _, h = train_cached(ids, freq, len(itos), seed=s, itos=itos, stoi=stoi)
        finals.append(h[-1, 1:]); print(f"{s:>6}" + "".join(f"{v:>10.3f}" if i < 2 else f"{v:>8.3f}" for i, v in enumerate(h[-1, 1:])))
    m = np.mean(finals, axis=0)                           # means
    print(f"{'mean':>6}{m[0]:>10.3f}{m[1]:>10.3f}{m[2]:>8.3f}")
    print("Actual output from this lesson's run.")


def demo_neighbours(ids, freq, itos, stoi):
    print("\n" + "=" * 70)
    print("4. WHAT THE VECTORS KNOW -- nearest neighbours and analogies, seed 0")
    print("=" * 70)
    print("Cosine similarity between centre-word vectors (W_in). Nobody told the model what any word means.")
    E = train_cached(ids, freq, len(itos), seed=0, itos=itos, stoi=stoi)[0].Win  # the trained table
    print()
    for w in ["three", "january", "france", "red", "physics", "computer", "war", "music", "king", "river"]:
        print(f"  {w:<9} -> " + ", ".join(neighbours(E, w, itos, stoi)))
    print("\nanalogies, a:b :: c:? -- the word nearest to b - a + c, question words excluded:")
    N = unit(E)                                           # unit vectors
    for a, b, c in [("paris", "france", "rome"), ("paris", "france", "tokyo"), ("man", "woman", "king"),
                    ("father", "mother", "brother"), ("france", "french", "germany"), ("big", "bigger", "small")]:
        q = N[stoi[b]] - N[stoi[a]] + N[stoi[c]]          # the analogy vector
        s = N @ q; s[[stoi[a], stoi[b], stoi[c]]] = -np.inf  # question words excluded
        top = [itos[i] for i in np.argsort(-s)[:3]]       # three best answers
        print(f"  {a}:{b} :: {c}:?  -> {', '.join(top)}")
    print("Actual output from this lesson's run.")
    return E                                               # saved for Day 87


def demo_ablations(ids, freq, itos, stoi):
    print("\n" + "=" * 70)
    print("5. WHAT EACH INGREDIENT DOES -- one change at a time, seed 0")
    print("=" * 70)
    print("Final scores after the same 5 epochs, changing one setting of section 3 at a time. 'untrained' is the "
          "starting table: what the scores give by chance.")
    rows = [("untrained (random table)", evaluate(SkipGram(len(itos), seed=0).Win, itos, stoi), None)]  # chance level
    for label, kw in [("section 3's settings", {}), ("no subsampling", {"t": 0.0}), ("window 2 instead of 5", {"window": 2}),
                      ("1 noise word instead of 5", {"neg": 1})]:
        t0 = time.perf_counter()                            # time each run
        _, h = train_cached(ids, freq, len(itos), seed=0, itos=itos, stoi=stoi, **kw)  # one change
        rows.append((label, h[-1, 1:], time.perf_counter() - t0 if kw else None))
    print(f"\n{'setting':<28}{'category':>10}{'capitals':>10}{'gender':>8}{'  s per epoch':>14}")
    for label, sc, secs in rows:                          # one row per setting
        extra = f"{secs / EPOCHS:>14.0f}" if secs is not None else ""
        print(f"{label:<28}{sc[0]:>10.3f}{sc[1]:>10.3f}{sc[2]:>8.3f}{extra}")
    print("(seconds per epoch only for runs trained in this section; the section 3 run was timed there)")
    print("Actual output from this lesson's run.")
    return rows                                            # the plot reuses them


def save_vectors(E, itos):                              # every word's vector, word2vec text format, for Days 87 and 89
    path = os.path.join(HERE, "day86_vectors.txt.gz")     # next to this script
    with gzip.open(path, "wt", encoding="ascii") as f:    # compressed text
        f.write(f"{len(itos)} {E.shape[1]}\n")              # header: words, numbers per word
        for i in range(len(itos)):                          # one line per word, most frequent first
            f.write(itos[i] + " " + " ".join(f"{x:.4f}" for x in E[i]) + "\n")  # word, then its vector
    print(f"\nsaved all {len(itos)} words' vectors to day86_vectors.txt.gz (for Days 87 and 89)")


def plot_results(ids, freq, itos, stoi, rows, ratio):   # three panels: scores by epoch | ablations | cost
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))          # canvas
    hs = [train_cached(ids, freq, len(itos), seed=s, itos=itos, stoi=stoi)[1] for s in SEEDS]  # three trajectories
    ep = np.arange(1, EPOCHS + 1)                           # epochs
    for j, (name, color) in enumerate([("category", "#1f77b4"), ("capitals", "#2ca02c"), ("gender", "#d62728")], start=1):
        for k, h in enumerate(hs):                            # each seed
            ax[0].plot(ep, h[:, j], marker="o", color=color, alpha=0.8, label=name if k == 0 else None)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("score (3 seeds)"); ax[0].set_ylim(0, 1.02); ax[0].set_xticks(ep)
    ax[0].set_title("Measured after every epoch", fontsize=10); ax[0].legend(fontsize=8, loc="center right"); ax[0].grid(alpha=0.3)
    labels = [r[0] for r in rows]; x = np.arange(len(rows))  # ablation bars
    for j, (name, color) in enumerate([("category", "#1f77b4"), ("capitals", "#2ca02c"), ("gender", "#d62728")]):
        ax[1].barh(x + (j - 1) * 0.27, [r[1][j] for r in rows], height=0.27, color=color, label=name)
    ax[1].set_yticks(x); ax[1].set_yticklabels(labels, fontsize=8); ax[1].invert_yaxis(); ax[1].set_xlim(0, 1.32)
    ax[1].set_xlabel("final score (seed 0)"); ax[1].set_title("One change at a time", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper right"); ax[1].grid(alpha=0.3, axis="x")
    ax[2].bar(["negative sampling\n(%d noise words)" % NEG, "full softmax\n(%d words)" % len(itos)], [1, ratio],
              color=["#1f77b4", "#7f7f7f"])
    ax[2].set_yscale("log"); ax[2].set_ylabel("time per batch, relative (log scale)")
    ax[2].set_title("Cost of one batch of %d pairs" % BATCH, fontsize=10); ax[2].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("word2vec_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved word2vec_results.png")


def draw_diagram(V):                                    # skip-gram with negative sampling, one pair
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                     # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.1))

    ax.text(6.5, 4.15, "Day 86: skip-gram with negative sampling, for one (centre, context) pair", ha="center",
            fontsize=12, fontweight="bold")
    box(1.3, 2.2, 2.0, 0.9, "centre word\n'france'", "#FAEEDA")
    box(4.0, 2.2, 2.2, 0.9, "v = W_in['france']\n(50 numbers)", "#DCEBF7")
    arrow(2.32, 2.2, 2.88, 2.2)
    rows = [(3.3, "real context 'spain'", "#E5F2E1", "s(u . v) -> 1"), (2.2, "noise word 'tuesday'", "#F7E1E1", "s(u . v) -> 0"),
            (1.1, "noise word 'banks'", "#F7E1E1", "s(u . v) -> 0")]
    for y, name, color, goal in rows:                                # the real pair and two of the five noise words
        box(7.2, y, 2.4, 0.7, "u = W_out[" + name.split("'")[1] + "]\n" + name, color, fs=8)
        arrow(5.12, 2.2, 5.98, y)
        box(10.2, y, 2.2, 0.7, "score u . v\n" + goal, "#F2F2F2", fs=8)
        arrow(8.42, y, 9.08, y)
    ax.text(10.2, 0.35, "five noise words per pair here (two shown), drawn from counts^0.75", ha="center", fontsize=8,
            style="italic")
    ax.text(4.0, 0.9, f"full softmax would score\nall {V:,} words instead", ha="center", fontsize=8, style="italic")
    plt.savefig("word2vec_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved word2vec_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT COMES NEXT")
    print("=" * 70)
    print("Fifty numbers per word cannot be looked at directly. Day 87 projects these vectors to two dimensions with "
          "PCA and t-SNE, and checks how much of the category structure survives the projection.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        words = load_text8()                                 # the corpus
        itos, stoi, freq, ids = build_vocab(words)           # vocabulary and ids
        demo_pairs(words)                                    # 1. pairs
        ratio = demo_objectives(len(itos))                   # 2. objectives
        demo_training(ids, freq, itos, stoi)                 # 3. training
        E = demo_neighbours(ids, freq, itos, stoi)           # 4. what it knows
        rows = demo_ablations(ids, freq, itos, stoi)         # 5. ingredients
        save_vectors(E, itos)                                # for Day 87
        plot_results(ids, freq, itos, stoi, rows, ratio)     # the figure
        draw_diagram(len(itos))                              # the diagram
        note_on_what_comes_next()                            # 6. Day 87


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
