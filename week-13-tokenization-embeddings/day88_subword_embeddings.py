"""
Day 88: Subword Embeddings and Out-of-Vocabulary Words (fastText From Scratch)

Day 86's Word2Vec has one row per word. A word it never saw -- or saw fewer
than 5 times -- gets nothing. fastText (Bojanowski et al., 2017) builds a
word's vector from its pieces: the word itself plus every character n-gram
of 3 to 6 letters, with < and > marking the word's start and end:

    where  ->  <wh, whe, her, ere, re>, <whe, wher, here, ere>, ... , and the word 'where' itself
    v(where) = the mean of the rows for all of these

The n-grams are hashed into a fixed number of rows (buckets), so any string
of letters has a vector -- including words never seen in training.

The test is a real one. Every second word of six of Day 86's word groups
(31 words: 'january', 'four', 'physics', ...) are removed from the corpus
before training, so neither model ever sees them. Afterwards fastText gives
them vectors from their n-grams; Word2Vec cannot. A spelling-only baseline
(which known word shares the most n-grams?) checks whether fastText adds
anything beyond spelling.

What is measured, reported as it came out:
  1. Character n-grams and hashing.
  2. A gradient check of the subword input.
  3. Word2Vec and fastText trained on the same corpus: known words.
  4. The 31 held-out words: group check, against spelling alone and chance.
  5. Rare real words (seen 1-4 times, dropped from both vocabularies): is the
     word's stem among its neighbours?
  6. Why: which rows dominate a word's vector.
  7. What the subwords cost.

Day 86's corpus, vocabulary, pair-making, negative-sampling and scoring code
is imported from its own script. Every line of code below carries its own
comment, continuing Day 52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep an imported script quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: the folder this script lives in
import sys                                             # sys: path, to import Day 86's script
import textwrap                                        # textwrap: wrap long bar labels
import time                                            # time: perf_counter, for timings
from collections import Counter                        # Counter counts words

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 86's script sits in the same folder
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing; kept quiet to be sure
    import day86_word2vec_skipgram as d86               # Day 86: corpus, vocabulary, pairs, negative sampling, scores

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

NMIN, NMAX = 3, 6                                       # n-gram lengths (fastText's defaults)
BUCKETS = 200_000                                       # rows shared by all n-grams (fastText uses 2,000,000)
LR_FT = 0.05                                            # fastText's default skip-gram learning rate (Word2Vec keeps Day 86's 0.025)
HELD = {g: ws[::2] for g, ws in d86.CATEGORIES.items()  # every second word of six groups, never shown in training
        if g in ("number", "month", "colour", "element", "instrument", "science")}
HELD_WORDS = {w for ws in HELD.values() for w in ws}    # as one set
SUFFIXES = ("s", "es", "ed", "ing", "ly", "er", "ers", "ism", "ist", "ists", "ian", "ians", "al", "ic")  # for the stem test


# =============================================================================
# 1. N-GRAMS AND HASHING
# =============================================================================
def char_ngrams(word, nmin=NMIN, nmax=NMAX):            # NEW: every character n-gram of '<word>', lengths nmin..nmax
    w = "<" + word + ">"                                  # start and end marks: 'her' inside a word differs from '<her>'
    return [w[i:i + n] for n in range(nmin, nmax + 1) for i in range(len(w) - n + 1)]  # all of them, in order


def fnv1a(s):                                           # NEW: the 32-bit FNV-1a hash fastText uses (Python's hash() changes per run)
    h = 2166136261                                        # the FNV offset basis
    for b in s.encode("utf-8"):                           # byte by byte
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF             # xor, multiply by the FNV prime, keep 32 bits
    return h                                              # an integer 0 .. 2^32 - 1


def ngram_rows(word, V):                                # NEW: the table rows for a word's n-grams (after the V word rows)
    return [V + fnv1a(g) % BUCKETS for g in char_ngrams(word)]  # one bucket per n-gram; different n-grams can share one


# =============================================================================
# 2. THE MODEL
# =============================================================================
class FastText:                                         # NEW: Day 86's skip-gram, with the centre vector built from subwords
    def __init__(self, itos, d=d86.D, seed=0):            # vocabulary, numbers per row
        V = len(itos)                                       # known words
        rng = np.random.RandomState(seed)                   # fixed rng
        self.Win = (rng.rand(V + BUCKETS, d) - 0.5) / d     # word rows, then n-gram rows: word2vec's small uniform start
        self.Wout = np.zeros((V, d))                        # context vectors, as Day 86
        rows = [[i] + ngram_rows(w, V) for i, w in enumerate(itos)]  # each word: its own row, then its n-grams
        self.cnt = np.array([len(r) for r in rows])         # rows per word
        self.start = np.cumsum(self.cnt) - self.cnt         # where each word's list starts in the flat array
        self.flat = np.array([x for r in rows for x in r])  # every word's rows, one after another

    def gather(self, c):                                  # the rows of words c, flattened, with weights and segment starts
        n = self.cnt[c]                                     # rows per word in the batch
        seg = np.cumsum(n) - n                              # where each word's rows start in the batch's flat list
        pos = np.arange(n.sum()) - np.repeat(seg, n) + np.repeat(self.start[c], n)  # positions in self.flat
        return self.flat[pos], np.repeat(1.0 / n, n), seg, n  # rows, each row's weight 1/n, segment starts, counts

    def centre(self, c):                                  # centre vectors for word ids c: the mean of their rows
        rows, w, seg, _ = self.gather(c)                    # which rows, with what weight
        return np.add.reduceat(self.Win[rows] * w[:, None], seg, axis=0)  # weighted sum per word

    def step(self, c, tgt, lr):                           # one update; same loss as Day 86
        rows, w, seg, n = self.gather(c)                    # the centre words' rows
        v = np.add.reduceat(self.Win[rows] * w[:, None], seg, axis=0)  # centre vectors
        U = self.Wout[tgt]                                  # context + noise vectors
        z = np.einsum("nd,nkd->nk", v, U)                   # scores
        loss = np.logaddexp(0, -z[:, 0]).sum() + np.logaddexp(0, z[:, 1:]).sum()  # Day 86's loss
        gz = d86.sigmoid(z); gz[:, 0] -= 1                  # dloss/dz
        dv = np.einsum("nk,nkd->nd", gz, U)                 # back to the centre vectors
        np.add.at(self.Win, rows, -lr * w[:, None] * np.repeat(dv, n, axis=0))  # each row gets its share of its word's gradient
        np.add.at(self.Wout, tgt, -lr * gz[:, :, None] * v[:, None, :])  # context and noise vectors
        return loss                                         # summed over the batch

    def vectors(self):                                    # every known word's vector
        return np.concatenate([self.centre(np.arange(s, min(s + 4096, len(self.cnt)))) for s in range(0, len(self.cnt), 4096)])

    def oov(self, word, V):                               # a vector for ANY word: the mean of its n-gram rows (no word row)
        return self.Win[ngram_rows(word, V)].mean(axis=0)


def train(model, ids, freq, lr0, seed, scorer):         # Day 86's train(), with the model and the scores passed in
    rng = np.random.RandomState(seed + 100)               # the data-order rng (Day 86's rule)
    cdf = d86.noise_cdf(freq)                             # Day 86's noise distribution
    total, done, hist = d86.EPOCHS * len(ids), 0, []      # tokens to process; processed; scores per epoch
    for ep in range(d86.EPOCHS):                          # Day 86's 5 epochs
        t0 = time.perf_counter()                            # time each epoch
        kept = d86.subsample(ids, freq, d86.SUBSAMPLE, rng)  # Day 86's subsampling
        C, O = d86.skipgram_pairs(kept, d86.WINDOW, rng)    # Day 86's pairs
        perm = rng.permutation(len(C)); C, O = C[perm], O[perm]  # shuffled
        for s in range(0, len(C), d86.BATCH):               # batches
            c, o = C[s:s + d86.BATCH], O[s:s + d86.BATCH]     # the batch
            lr = lr0 * max(1e-4, 1 - (done + s / len(C) * len(ids)) / total)  # linear decay
            tgt = np.concatenate([o[:, None], np.searchsorted(cdf, rng.rand(len(c), d86.NEG))], axis=1)  # context, noise
            model.step(c, tgt, lr)                            # update
        done += len(ids)                                    # an epoch of tokens
        hist.append((time.perf_counter() - t0,) + scorer(model))  # seconds, then the scores
    return np.array(hist)                                 # one row per epoch


# =============================================================================
# 3. MEASURES
# =============================================================================
def known_scores(E, itos, stoi):                        # Day 86's three scores, on the category words both models know
    keep = {g: [w for w in ws if w not in HELD_WORDS] for g, ws in d86.CATEGORIES.items()}  # held-out words left out
    words = [(w, g) for g, ws in keep.items() for w in ws]  # (word, group)
    ii = np.array([stoi[w] for w, _ in words]); gg = np.array([g for _, g in words])  # ids, groups
    N = d86.unit(E[ii]); S = N @ N.T; np.fill_diagonal(S, -np.inf)  # cosines
    return (np.mean(gg[S.argmax(axis=1)] == gg), d86.analogy_score(E, d86.CAPITALS, itos, stoi),
            d86.analogy_score(E, d86.GENDER, itos, stoi))  # category, capitals, gender


def held_check(sim_of, cand, cand_groups):              # share of held-out words whose nearest known category word is in the same group
    ok = []                                               # one True/False per held-out word
    for g, ws in HELD.items():                            # each group
        for w in ws:                                        # each held-out word
            ok.append(cand_groups[int(np.argmax(sim_of(w)))] == g)  # the most similar candidate's group
    return np.mean(ok)                                    # share right


def jaccard(a, b):                                      # shared n-grams / all n-grams of two words: spelling similarity
    A, B = set(char_ngrams(a)), set(char_ngrams(b))       # the two n-gram sets
    return len(A & B) / len(A | B)                        # 0 = nothing shared, 1 = the same word


# =============================================================================
# DEMOS
# =============================================================================
def demo_ngrams(itos):
    print("=" * 70)
    print("1. A WORD AS ITS PIECES -- character n-grams and hashing")
    print("=" * 70)
    print(f"Each word is wrapped in < and >, then cut into every piece of {NMIN} to {NMAX} characters. Each piece is "
          f"hashed (32-bit FNV-1a, as in fastText) into one of {BUCKETS:,} rows.")
    g = char_ngrams("where")                              # fastText's own example
    print(f"\n'where': {len(g)} n-grams: " + " ".join(g))
    print("the piece '<her>' (the whole word 'her') is among the n-grams of 'her': " + str("<her>" in char_ngrams("her"))
          + "; of 'where': " + str("<her>" in g) + " ('where' has 'her' but not '<her>')")
    allg = {x for w in itos for x in char_ngrams(w)}      # every distinct n-gram of the vocabulary
    used = len({fnv1a(x) % BUCKETS for x in allg})         # distinct buckets they land in
    lens = [len(char_ngrams(w)) for w in itos]            # n-grams per word
    print(f"the {len(itos)} known words have {len(allg):,} distinct n-grams; they fill {used:,} of the {BUCKETS:,} rows "
          f"({len(allg) - used:,} n-grams share a row with another); a word has {np.mean(lens):.1f} n-grams on average")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("2. THE SUBWORD INPUT -- checking its gradient")
    print("=" * 70)
    print("A tiny model (3 words, 20 buckets, 4 numbers per row); 3 pairs, 2 noise words each. The loss is Day 86's; "
          "the centre vector is the mean of the word's rows. Every input row entry nudged by +-1e-6.")
    global BUCKETS                                        # shrink the table for the check, then restore it
    saved, BUCKETS = BUCKETS, 20                          # 20 buckets: n-grams of different words collide, as they would
    m = FastText(["cat", "cats", "dog"], d=4, seed=0)     # tiny model
    m.Win = np.random.RandomState(1).randn(*m.Win.shape) * 0.5; m.Wout = np.random.RandomState(2).randn(3, 4) * 0.5
    c, tgt = np.array([0, 1, 0]), np.array([[1, 2, 0], [0, 2, 1], [2, 1, 1]])  # pairs and noise
    lossf = lambda: (lambda z: np.logaddexp(0, -z[:, 0]).sum() + np.logaddexp(0, z[:, 1:]).sum())(
        np.einsum("nd,nkd->nk", m.centre(c), m.Wout[tgt]))  # the loss as a function of the current tables
    before = m.Win.copy()                                 # analytic gradient: one step with lr 1 moves by -gradient
    m.step(c, tgt, 1.0); G = before - m.Win; m.Win = before.copy(); m.Wout = np.random.RandomState(2).randn(3, 4) * 0.5
    worst, ok, count = 0.0, True, 0                       # largest difference; every entry passes; entries
    for pos in np.ndindex(m.Win.shape):                   # every input-row entry
        old = m.Win[pos]                                    # remember it
        m.Win[pos] = old + 1e-6; up = lossf()               # nudge up
        m.Win[pos] = old - 1e-6; dn = lossf()               # nudge down
        m.Win[pos] = old                                    # restore
        num = (up - dn) / 2e-6; count += 1                  # central difference
        worst = max(worst, abs(num - G[pos])); ok &= abs(num - G[pos]) <= 1e-10 + 1e-6 * abs(num)
    BUCKETS = saved                                       # restore
    print(f"\ngradient check: {count} entries (3 word rows + 20 buckets, 4 numbers each), largest |numeric - analytic| = "
          f"{worst:.2e}, all pass: {ok}")
    print("Actual output from this lesson's run.")


def demo_training(words):
    print("\n" + "=" * 70)
    print("3. TWO MODELS, ONE CORPUS -- the known words")
    print("=" * 70)
    print(f"The {len(HELD_WORDS)} held-out words are removed from Day 86's corpus first: " +
          "; ".join(f"{g}: {' '.join(ws)}" for g, ws in HELD.items()) + ". "
          f"Then Day 86's settings for both (d = {d86.D}, window {d86.WINDOW}, {d86.NEG} noise words, {d86.EPOCHS} epochs, "
          f"seed 0), each model at its own default learning rate (Word2Vec {d86.LR0}, fastText {LR_FT}). Category: Day 86's "
          f"check on the {sum(len(v) for v in d86.CATEGORIES.values()) - len(HELD_WORDS)} category words both models know.")
    kept = [w for w in words if w not in HELD_WORDS]      # the corpus without the held-out words
    itos, stoi, freq, ids = d86.build_vocab(kept)         # Day 86's vocabulary
    assert not HELD_WORDS & set(stoi)                     # they really are unknown
    w2v = d86.SkipGram(len(itos), seed=0)                 # Day 86's model
    hw = train(w2v, ids, freq, d86.LR0, 0, lambda m: known_scores(m.Win, itos, stoi))  # Word2Vec
    ft = FastText(itos, seed=0)                           # fastText
    hf = train(ft, ids, freq, LR_FT, 0, lambda m: known_scores(m.vectors(), itos, stoi))  # fastText
    print(f"\n{len(ids)} tokens, vocabulary {len(itos)}\n{'model':<10}{'epoch':>6}{'category':>10}{'capitals':>10}{'gender':>8}{'s/epoch':>9}")
    for name, h in (("Word2Vec", hw), ("fastText", hf)):  # both trajectories
        for e, row in enumerate(h, 1):                      # one row per epoch
            print(f"{name:<10}{e:>6}{row[1]:>10.3f}{row[2]:>10.3f}{row[3]:>8.3f}{row[0]:>9.0f}")
    print("Actual output from this lesson's run.")
    return itos, stoi, freq, ft, hw, hf                   # later sections use them


def demo_held_out(itos, stoi, ft):
    print("\n" + "=" * 70)
    print("4. WORDS NEVER SEEN -- the held-out words")
    print("=" * 70)
    print("For each held-out word: which known category word is most similar? Right if it is in the same group "
          "(e.g. 'january' -> 'february'). Candidates: the category words both models know.")
    V = len(itos)                                         # known words
    cand = [(w, g) for g, ws in d86.CATEGORIES.items() for w in ws if w not in HELD_WORDS]  # known category words
    cg = np.array([g for _, g in cand]); ci = np.array([stoi[w] for w, _ in cand])  # their groups and ids
    Ef = d86.unit(ft.vectors()[ci])                       # fastText vectors of the candidates
    ft_sim = lambda w: Ef @ (ft.oov(w, V) / np.linalg.norm(ft.oov(w, V)))  # cosine, held-out word against candidates
    sp_sim = lambda w: np.array([jaccard(w, c) for c, _ in cand])  # spelling only
    chance = np.mean([np.mean(cg == g) for g, ws in HELD.items() for _ in ws])  # expected share for a random pick
    rows = [("Word2Vec", None), ("fastText, n-grams", held_check(ft_sim, cand, cg)), ("spelling only", held_check(sp_sim, cand, cg)),
            ("chance (random pick)", chance)]
    print(f"\n{'method':<22}{'right':>8}")
    for name, v in rows:                                  # one row per method
        print(f"{name:<22}" + (f"{v:>8.3f}" if v is not None else "     no vector: never saw the words"))
    print(f"per group, fastText / spelling:")
    for g, ws in HELD.items():                            # each group
        f = np.mean([cg[int(np.argmax(ft_sim(w)))] == g for w in ws]); s = np.mean([cg[int(np.argmax(sp_sim(w)))] == g for w in ws])
        print(f"  {g:<11} {len(ws)} words   {f:.2f} / {s:.2f}")
    print("\nnearest known words of the whole vocabulary (fastText):")
    Ev = d86.unit(ft.vectors())                           # every known word
    for w in ("january", "four", "physics", "green", "silver", "violin"):
        if w in HELD_WORDS:                                 # only held-out words
            q = ft.oov(w, V); s = Ev @ (q / np.linalg.norm(q))  # cosine to every known word
            print(f"  {w:<10} -> " + ", ".join(itos[i] for i in np.argsort(-s)[:6]))
    print("Actual output from this lesson's run.")
    return rows                                            # the plot shows them


def demo_rare(words, itos, stoi, ft):
    print("\n" + "=" * 70)
    print("5. RARE REAL WORDS -- seen 1-4 times, so dropped from both vocabularies")
    print("=" * 70)
    print("Rare words made of a known word plus a common ending (" + ", ".join(SUFFIXES) + "), e.g. 'pianists' = "
          "'pianist' + 's'. Is the known stem among the word's 10 most similar known words?")
    V = len(itos); cnt = Counter(words)                   # known words; counts
    rare = sorted(w for w, c in cnt.items() if 1 <= c <= 4 and w not in HELD_WORDS)  # dropped words
    tests = []                                            # (rare word, its stem)
    for w in rare:                                        # each rare word
        for suf in SUFFIXES:                                # each ending
            if w.endswith(suf) and w[:-len(suf)] in stoi and len(w) - len(suf) >= 4:  # a known stem of 4+ letters
                tests.append((w, w[:-len(suf)])); break
    rng = np.random.RandomState(0)                        # a fixed sample, to keep the spelling baseline quick
    tests = [tests[i] for i in sorted(rng.choice(len(tests), size=min(400, len(tests)), replace=False))]
    Ev = d86.unit(ft.vectors())                           # every known word, fastText
    grams = [set(char_ngrams(x)) for x in itos]           # every known word's n-grams, for the spelling baseline
    hit_ft = hit_sp = 0                                   # stems found
    for w, stem in tests:                                 # each test word
        q = ft.oov(w, V); s = Ev @ (q / np.linalg.norm(q))  # fastText similarity to every known word
        hit_ft += stoi[stem] in np.argsort(-s)[:10]         # stem in the top 10?
        A = set(char_ngrams(w)); j = np.array([len(A & B) / len(A | B) for B in grams])  # spelling similarity
        hit_sp += stoi[stem] in np.argsort(-j)[:10]         # stem in the top 10?
    print(f"\n{len(rare)} rare words; {len(tests)} sampled with a known stem")
    print(f"stem among the 10 most similar known words: fastText {hit_ft / len(tests):.3f}, spelling only {hit_sp / len(tests):.3f}")
    print("examples (fastText's 4 most similar known words):")
    for w, stem in tests[:: max(1, len(tests) // 8)][:8]:  # eight examples, spread out
        q = ft.oov(w, V); s = Ev @ (q / np.linalg.norm(q))  # similarity
        print(f"  {w:<16} -> " + ", ".join(itos[i] for i in np.argsort(-s)[:4]))
    print("Actual output from this lesson's run.")
    return hit_ft / len(tests), hit_sp / len(tests)       # the plot shows them


def demo_why(itos, ft):
    print("\n" + "=" * 70)
    print("6. WHY -- which rows dominate a word's vector")
    print("=" * 70)
    print("A word's vector is the mean of its rows. If some rows are much longer than others, they decide the "
          "direction. Row length after training, by how many known words share the row:")
    V = len(itos)                                         # known words
    use = Counter(r for i in range(V) for r in set(ft.flat[ft.start[i] + 1:ft.start[i] + ft.cnt[i]].tolist()))  # n-gram row -> known words using it (once each)
    rows = np.array(list(use)); cnt = np.array([use[r] for r in rows])  # rows and their sharing counts
    norms = np.linalg.norm(ft.Win[rows], axis=1)          # their lengths
    print(f"\n{'row':<30}{'rows':>8}{'mean length':>13}")
    bands = [("word rows", np.linalg.norm(ft.Win[:V], axis=1).mean())]  # (label, mean length), for the plot
    print(f"{'word rows (one per word)':<30}{V:>8}{bands[0][1]:>13.3f}")
    for lo, hi in [(1, 2), (2, 10), (10, 100), (100, 10 ** 9)]:  # sharing bands
        m = (cnt >= lo) & (cnt < hi)                        # rows in this band
        label = f"n-gram, used by {lo}" + (f"-{hi - 1}" if hi - lo > 1 and hi < 10 ** 9 else "+" if hi >= 10 ** 9 else "") + " words"
        print(f"{label:<30}{m.sum():>8}{norms[m].mean():>13.3f}"); bands.append((label[8:].replace(" words", ""), norms[m].mean()))
    rng = np.random.RandomState(0)                        # 300 known words
    Ev = d86.unit(ft.vectors()); sims = []                # every known word's full vector
    for i in rng.choice(V, 300, replace=False):           # each sampled word
        q = ft.oov(itos[i], V); sims.append(Ev[i] @ q / np.linalg.norm(q))  # its n-gram-only vector against its full vector
    print(f"\nknown words: cosine between the n-gram-only vector and the full vector (word row included), "
          f"mean of 300: {np.mean(sims):.4f}")
    print("Actual output from this lesson's run.")
    return bands                                           # the plot shows them


def demo_cost(itos, hw, hf):
    print("\n" + "=" * 70)
    print("7. WHAT THE SUBWORDS COST")
    print("=" * 70)
    V = len(itos)                                         # known words
    print(f"weights: Word2Vec {2 * V * d86.D:,} (two tables of {V} x {d86.D}); fastText {(2 * V + BUCKETS) * d86.D:,} "
          f"(plus {BUCKETS:,} n-gram rows)")
    print(f"time per epoch in this run: Word2Vec {np.median(hw[:, 0]):.0f} s, fastText {np.median(hf[:, 0]):.0f} s "
          f"({np.median(hf[:, 0]) / np.median(hw[:, 0]):.1f} times; median of 5 epochs)")
    print("Actual output from this lesson's run.")


def plot_results(hw, hf, rows, stems, bands):           # four panels: known words | held-out words | rare words | row lengths
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))          # canvas
    ep = np.arange(1, len(hw) + 1)                          # epochs
    for h, name, ls in ((hw, "Word2Vec", "--"), (hf, "fastText", "-")):
        for j, (score, color) in enumerate([("category", "#1f77b4"), ("capitals", "#2ca02c"), ("gender", "#d62728")], start=1):
            ax[0].plot(ep, h[:, j], linestyle=ls, marker="o", color=color, label=f"{name}, {score}")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("score"); ax[0].set_ylim(0, 1.02); ax[0].set_xticks(ep)
    ax[0].set_title("Known words, same corpus", fontsize=10); ax[0].legend(fontsize=7, ncol=2, loc="center right"); ax[0].grid(alpha=0.3)
    names = [r[0] for r in rows[1:]]; vals = [r[1] for r in rows[1:]]  # Word2Vec has no bar: no vector
    ax[1].bar([textwrap.fill(n, 12) for n in names], vals, color=["#1f77b4", "#ff7f0e", "#7f7f7f"]); ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("held-out words with a same-group nearest word"); ax[1].set_title("Words never seen (Word2Vec: no vector)", fontsize=10)
    ax[1].grid(alpha=0.3, axis="y")
    ax[2].bar(["fastText", "spelling only"], stems, color=["#1f77b4", "#ff7f0e"]); ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("stem among the 10 most similar"); ax[2].set_title("Rare real words: is the stem nearby?", fontsize=10)
    ax[2].grid(alpha=0.3, axis="y")
    ax[3].bar([textwrap.fill(b[0], 8) for b in bands], [b[1] for b in bands], color=["#7f7f7f"] + ["#1f77b4"] * (len(bands) - 1))
    ax[3].set_xlabel("n-gram rows, by how many known words use them"); ax[3].set_ylabel("mean row length after training")
    ax[3].set_title("Shared pieces grow long and dominate", fontsize=10); ax[3].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("subword_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved subword_results.png")


def draw_diagram():                                     # a word's vector from its pieces
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                     # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.1))

    ax.text(6.5, 4.15, "Day 88: a word's vector from its pieces (fastText)", ha="center", fontsize=12, fontweight="bold")
    box(1.3, 2.1, 2.0, 0.9, "'pianists'\n(seen 1-4 times)", "#FAEEDA")
    pieces = ["word row (known words only)", "<pi, pia, ian, ...", "<pia, pian, iani, ...", "... up to 6 letters: ists>"]
    for k, p in enumerate(pieces):                                   # the pieces
        y = 3.4 - k * 0.85
        box(4.6, y, 3.0, 0.6, p, "#DCEBF7" if k else "#EEEEEE", fs=8)
        arrow(2.32, 2.1, 3.08, y)
        arrow(6.12, y, 7.38, 2.1)
    box(8.6, 2.1, 2.3, 0.9, "hash each piece\ninto 200,000 rows", "#E5F2E1")
    box(11.5, 2.1, 2.4, 0.9, "v = mean of\nthe rows", "#F2F2F2")
    arrow(9.77, 2.1, 10.28, 2.1)
    ax.text(9.8, 1.15, "an unseen word has no word row:" + "\n" + "its vector is the mean of its n-gram rows", ha="center",
            fontsize=8, style="italic")
    plt.savefig("subword_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved subword_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Days 86 and 88 learned vectors from raw text. Day 89 trains GloVe from word co-occurrence counts, saves the "
          "vectors in GloVe's text format, loads them back, and uses them as pretrained vectors in Day 85's tagger.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        words = d86.load_text8()                             # Day 86's corpus
        _, stoi_all, _, _ = d86.build_vocab(words)           # the full vocabulary, for section 1
        demo_ngrams([w for w in stoi_all if w not in HELD_WORDS])  # 1. n-grams
        demo_gradient_check()                                # 2. gradient check
        itos, stoi, freq, ft, hw, hf = demo_training(words)  # 3. both models
        rows = demo_held_out(itos, stoi, ft)                 # 4. held-out words
        stems = demo_rare(words, itos, stoi, ft)             # 5. rare words
        bands = demo_why(itos, ft)                           # 6. why
        demo_cost(itos, hw, hf)                              # 7. cost
        plot_results(hw, hf, rows, stems, bands)             # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 8. Day 89


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
