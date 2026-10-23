"""
Day 84: Byte-Pair Encoding, From Scratch, on Real Text

Week 13 turns to the input side of the models built so far. Every model
since Day 61 has read WORDS, with a fixed vocabulary: Day 62's tagger kept
the 2,080 words seen at least twice in training and turned every other
word into <unk> -- 20.2% of the test words. Byte-pair encoding (BPE;
Sennrich, Haddow and Birch, 2016, after Gage, 1994) builds a vocabulary of
SUBWORDS instead:

    start: every word is a sequence of characters, plus an end-of-word mark </w>
    repeat: find the most frequent adjacent pair of symbols in the corpus,
            merge it into one new symbol everywhere, and remember the merge
    encode a new word: apply the remembered merges, in the order learned

Frequent words end up as one symbol; rare and unseen words are spelled out
from smaller pieces, so nothing is ever unknown unless it contains a
character never seen in training.

What is built and measured, reported as it came out, on Week 9's real
sentences (Universal Dependencies English Web Treebank; trained on the
2,001 training sentences, tested on the 1,647 test sentences):

  1. BPE by hand on Sennrich et al.'s small example.
  2. Training on real text: the first merges, and the time they take.
  3. Encoding is lossless: every test sentence decodes back exactly.
  4. Vocabulary size against sequence length.
  5. No more <unk>: what happens to the words Day 62 could not read,
     and the characters it still cannot.
  6. Frequent words are one piece, rare words several.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: build the path to Week 9's dataset folder
import time                                            # time: perf_counter, to time training
from collections import Counter                        # Counter counts things

import numpy as np                                     # numpy: arrays for the summaries and plots
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
DATA_DIR = os.path.join(HERE, "..", "week-09-sequence-data-rnns-i", "ud_english_ewt_pos")  # Day 62's dataset, used where it is
EOW = "</w>"                                            # the end-of-word mark: "er" inside a word and "er" at its end stay different
N_MERGES = 4000                                         # merges learned on the real text (the sweep uses the first 0..4000)
SWEEP = (0, 100, 250, 500, 1000, 2000, 4000)            # vocabulary sizes compared


# =============================================================================
# 0. REUSED -- Day 62's reader, verbatim
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


# =============================================================================
# 1. BYTE-PAIR ENCODING -- NEW
# =============================================================================
def pair_counts(vocab):                                 # NEW: how often each adjacent pair of symbols occurs, over the corpus
    counts = Counter()                                    # pair -> count
    for symbols, freq in vocab.items():                   # each distinct word, as a tuple of symbols, and how often it occurs
        for a, b in zip(symbols, symbols[1:]):              # every adjacent pair in it
            counts[a, b] += freq                              # counted once per occurrence of the word
    return counts                                         # Counter of pairs


def merge_word(symbols, pair, new):                     # NEW: replace every occurrence of pair in one word by the new symbol
    out, i = [], 0                                        # the merged word; position
    while i < len(symbols):                               # left to right
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:  # the pair starts here
            out.append(new); i += 2                           # one symbol for two
        else:
            out.append(symbols[i]); i += 1                    # unchanged
    return tuple(out)                                     # tuples can be dictionary keys


def train_bpe(word_counts, n_merges):                   # NEW: learn n_merges merges from a Counter of words
    vocab = {tuple(w) + (EOW,): c for w, c in word_counts.items()}  # every word spelled out, with its end mark
    merges = []                                           # the merges, in the order learned, with their counts
    for _ in range(n_merges):                             # one merge per round
        counts = pair_counts(vocab)                         # count every pair (the simple way: from scratch each round)
        if not counts:                                      # every word is already one symbol
            break                                             # nothing left to merge
        best = max(counts, key=lambda p: (counts[p], p))    # the most frequent pair; ties broken by the pair itself, so runs repeat
        new = best[0] + best[1]                             # the new symbol is the two joined
        vocab = {merge_word(s, best, new): c for s, c in vocab.items()}  # merge it everywhere
        merges.append((best, counts[best]))                 # remember it, and how often it occurred
    return merges                                         # the learned merges


def encode_word(word, ranks):                           # NEW: a word -> its subword symbols, applying merges in the order learned
    symbols = tuple(word) + (EOW,)                        # spelled out, with its end mark
    while len(symbols) > 1:                               # until no learned merge applies
        pairs = [(ranks.get(p, np.inf), p) for p in zip(symbols, symbols[1:])]  # each adjacent pair's rank (inf: never learned)
        rank, best = min(pairs)                             # the earliest-learned pair present
        if rank == np.inf:                                  # none of the pairs was ever learned
            break                                             # done
        symbols = merge_word(symbols, best, best[0] + best[1])  # apply that merge
    return symbols                                        # the subword symbols


def decode(symbols):                                    # NEW: subword symbols -> text (words separated by spaces)
    return "".join(symbols).replace(EOW, " ").strip()     # join everything; every end mark becomes a space


def ranks_of(merges, k):                                # NEW: the first k merges as pair -> rank (lower = learned earlier)
    return {pair: i for i, (pair, _) in enumerate(merges[:k])}  # a lookup table for encode_word


# =============================================================================
# DEMOS
# =============================================================================
def demo_by_hand():
    print("=" * 70)
    print("1. BPE BY HAND -- Sennrich et al. (2016)'s example corpus")
    print("=" * 70)
    print("Four words with their counts: low x5, lower x2, newest x6, widest x3. Every word starts as characters "
          "plus </w>. Each round merges the most frequent adjacent pair everywhere.")
    corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})  # the example
    merges = train_bpe(corpus, 10)                        # ten rounds
    vocab = {tuple(w) + (EOW,): c for w, c in corpus.items()}  # to show the words after each round
    print(f"\n{'round':>6}  {'merge':<16}{'count':>6}   words afterwards")
    for i, (pair, c) in enumerate(merges, 1):             # each round
        vocab = {merge_word(s, pair, pair[0] + pair[1]): n for s, n in vocab.items()}  # apply it, for the printout
        print(f"{i:>6}  {pair[0] + ' + ' + pair[1]:<16}{c:>6}   " + "  ".join(" ".join(s) for s in vocab))
    print("\nencoding words never seen in this corpus with the 10 merges:")
    for w in ("lowest", "newer", "wider"):                # three new words
        print(f"  {w:<8} -> {' '.join(encode_word(w, ranks_of(merges, 10)))}")
    print("Actual output from this lesson's run.")


def demo_train(train_words):
    print("\n" + "=" * 70)
    print(f"2. TRAINING ON REAL TEXT -- {N_MERGES} merges on Week 9's training sentences")
    print("=" * 70)
    counts = Counter(train_words)                         # word -> count
    chars = sorted({ch for w in counts for ch in w})      # every character in the training text
    print(f"{len(train_words)} words ({len(counts)} distinct) in the 2,001 training sentences, lowercased; "
          f"{len(chars)} different characters, so {len(chars) + 1} starting symbols with </w>.")
    start = time.perf_counter()                           # time the training
    merges = train_bpe(counts, N_MERGES)                  # learn the merges
    seconds = time.perf_counter() - start                 # how long it took
    print(f"\nthe first 15 merges (pair, and how often it occurred when merged):")
    for i in range(0, 15, 5):                             # three lines of five
        print("  " + "   ".join(f"{a}+{b} ({c})" for (a, b), c in merges[i:i + 5]))
    print(f"merges 1000, 2000, 4000: " + ", ".join(f"{merges[k - 1][0][0]}+{merges[k - 1][0][1]} ({merges[k - 1][1]})" for k in (1000, 2000, 4000)))
    print(f"{len(merges)} merges took {seconds:.1f} s here (every pair recounted every round)")
    print("Actual output from this lesson's run.")
    return merges, chars                                   # later sections use them


def demo_lossless(test_sents, merges):
    print("\n" + "=" * 70)
    print("3. ENCODING IS LOSSLESS -- every test sentence back from its subwords")
    print("=" * 70)
    print("Each of the 1,647 test sentences (lowercased words joined by spaces) is encoded with all the merges, then "
          "decoded. A tokenizer must give back exactly what it was given.")
    ranks = ranks_of(merges, len(merges))                 # all merges
    wrong, n_tok, n_words = 0, 0, 0                       # mismatches, tokens, words
    for sent in test_sents:                               # each test sentence
        words = [w for w, _ in sent]                        # its words
        symbols = [s for w in words for s in encode_word(w, ranks)]  # every word's subwords, in order
        wrong += decode(symbols) != " ".join(words)         # does it come back exactly?
        n_tok += len(symbols); n_words += len(words)        # running totals
    print(f"\nsentences that do not decode back exactly: {wrong} of {len(test_sents)}")
    print(f"tokens: {n_tok} for {n_words} words ({n_tok / n_words:.3f} per word)")
    sent = [w for w, _ in test_sents[3]]                  # one example
    print(f"\nexample: {' '.join(sent)}")
    print("encoded: " + " | ".join(" ".join(encode_word(w, ranks)) for w in sent))
    print("Actual output from this lesson's run.")


def demo_sweep(train_words, test_words, merges, chars):
    print("\n" + "=" * 70)
    print("4. VOCABULARY SIZE AGAINST SEQUENCE LENGTH")
    print("=" * 70)
    print("The first k merges, for several k. Vocabulary = the starting symbols plus k merged symbols. Measured on the "
          "test words: tokens per word, and the share of test words that are a single token. Day 62's word "
          "vocabulary for comparison: 2,082 entries, one token per word, but 20.2% of test words are <unk>.")
    print(f"\n{'merges':>7}{'vocabulary':>12}{'tokens per word':>17}{'one-token words':>17}")
    res = {}                                              # k -> (vocabulary, tokens per word, single-token share)
    for k in SWEEP:                                       # each size
        ranks = ranks_of(merges, k)                         # the first k merges
        lens = [len(encode_word(w, ranks)) for w in test_words]  # tokens per test word (with its end mark)
        res[k] = (len(chars) + 1 + k, np.mean(lens), np.mean(np.array(lens) == 1))  # the three numbers
        print(f"{k:>7}{res[k][0]:>12}{res[k][1]:>17.3f}{res[k][2]:>17.3f}")
    print("(with 0 merges every character is a token, and </w> is its own token: a 3-letter word is 4 tokens)")
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_unknown(train_words, test_words, merges, chars):
    print("\n" + "=" * 70)
    print("5. NO MORE <unk> -- the test words Day 62 could not read")
    print("=" * 70)
    print("Day 62 kept only words seen at least twice in training. The test words outside that vocabulary, and how "
          "BPE (all merges) handles them.")
    counts = Counter(train_words)                         # training word counts
    kept = {w for w, c in counts.items() if c >= 2}       # Day 62's vocabulary (without <pad> and <unk>)
    unk = [w for w in test_words if w not in kept]        # Day 62's <unk> words, every occurrence
    new_chars = sorted({ch for w in test_words for ch in w} - set(chars))  # characters never seen in training
    ranks = ranks_of(merges, len(merges))                 # all merges
    pieces = [len(encode_word(w, ranks)) for w in unk]    # how many pieces each needs
    print(f"\ntest words outside Day 62's vocabulary: {len(unk)} of {len(test_words)} ({len(unk) / len(test_words):.1%})")
    print(f"characters in the test text never seen in training: {len(new_chars)}")
    for ch in new_chars:                                  # each one, as its Unicode number (this console cannot print all of them)
        print(f"  U+{ord(ch):04X}: {len(ch.encode('utf-8'))} byte(s) in UTF-8; in {sum(ch in w for w in test_words)} test word(s)")
    print(f"test words with such a character: {sum(any(ch in new_chars for ch in w) for w in test_words)} -- BPE keeps the character "
          "as a symbol of its own, which a model trained on this vocabulary has never seen. A byte-level BPE (as in GPT-2) "
          "starts from the 256 possible byte values instead, so no input can ever fall outside it.")
    print(f"BPE pieces per Day-62-unknown word: mean {np.mean(pieces):.2f}; one piece {np.mean(np.array(pieces) == 1):.3f}")
    longer = sorted({w for w in unk if len(w) >= 7 and w.isascii()})  # longer unknown words that this console can print
    examples = longer[:: max(1, len(longer) // 8)][:8]    # 8 of them, spread through the alphabet
    print("examples:")
    for w in examples:                                    # each example
        print(f"  {w:<16} -> {' '.join(encode_word(w, ranks))}")
    print("Actual output from this lesson's run.")


def demo_frequency(train_words, test_words, merges):
    print("\n" + "=" * 70)
    print("6. FREQUENT WORDS ARE ONE PIECE, RARE WORDS SEVERAL")
    print("=" * 70)
    print("Test words grouped by how often they occurred in the training sentences; pieces per word with 1000 and "
          "with all merges.")
    counts = Counter(train_words)                         # training word counts
    bands = [("10 or more", lambda c: c >= 10), ("2-9", lambda c: 2 <= c <= 9), ("once", lambda c: c == 1),  # four frequency bands
             ("never", lambda c: c == 0)]                 # four bands
    print(f"\n{'seen in training':<18}{'test words':>11}{'pieces, 1000 merges':>21}{'pieces, ' + str(N_MERGES) + ' merges':>21}")
    res = {}                                              # band -> (pieces at 1000, at all)
    for label, test in bands:                             # each band
        ws = [w for w in test_words if test(counts[w])]     # its test words
        a = np.mean([len(encode_word(w, ranks_of(merges, 1000))) for w in ws])  # 1000 merges
        b = np.mean([len(encode_word(w, ranks_of(merges, len(merges)))) for w in ws])  # all merges
        res[label] = (a, b)                                 # for the plot
        print(f"{label:<18}{len(ws):>11}{a:>21.2f}{b:>21.2f}")
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def plot_results(sweep, merges, freq):                  # three panels: vocabulary vs length | merge counts | pieces by frequency
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))          # canvas
    ks = list(sweep)                                        # panel 1
    ax[0].plot([sweep[k][0] for k in ks], [sweep[k][1] for k in ks], marker="o", color="#1f77b4")
    for k in ks:                                            # label each point with its merge count
        ax[0].annotate(f"{k}", (sweep[k][0], sweep[k][1]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax[0].axhline(1.0, color="#7f7f7f", linestyle="--", label="a whole word (Day 62, but 20.2% <unk>)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("vocabulary size (log scale)"); ax[0].set_ylabel("tokens per test word")
    ax[0].set_title("Vocabulary against length", fontsize=10); ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(alpha=0.3)
    ax[1].plot(np.arange(1, len(merges) + 1), [c for _, c in merges], color="#1f77b4")  # panel 2: how often each merge occurred
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("merge number"); ax[1].set_ylabel("pair count when merged")
    ax[1].set_title("Each merge covers less text", fontsize=10); ax[1].grid(alpha=0.3, which="both")
    labels = list(freq); x = np.arange(len(labels))         # panel 3: pieces by frequency band
    ax[2].bar(x - 0.2, [freq[l][0] for l in labels], width=0.4, color="#9ecae1", label="1000 merges")
    ax[2].bar(x + 0.2, [freq[l][1] for l in labels], width=0.4, color="#08519c", label=f"{N_MERGES} merges")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels); ax[2].set_xlabel("times the word was seen in training")
    ax[2].set_ylabel("pieces per test word (with </w>)"); ax[2].set_title("Rare words are spelled out", fontsize=10)
    ax[2].legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig("bpe_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved bpe_results.png")


def draw_diagram():                                     # the BPE loop, and encoding a new word
    fig, ax = plt.subplots(figsize=(13, 4.0))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.0); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333"):                    # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    ax.text(6.5, 3.75, "Day 84: byte-pair encoding -- learn merges, then apply them", ha="center", fontsize=12, fontweight="bold")
    box(1.5, 2.6, 2.4, 0.7, "words as characters\nl o w e r </w>", "#F2F2F2")         # training loop
    box(4.6, 2.6, 2.4, 0.7, "count every adjacent\npair (x word count)", "#DCEBF7")
    box(7.7, 2.6, 2.4, 0.7, "merge the most\nfrequent pair everywhere", "#FAEEDA")
    box(10.8, 2.6, 2.4, 0.7, "remember it:\nmerge number k", "#E5F2E1")
    arrow(2.72, 2.6, 3.38, 2.6); arrow(5.82, 2.6, 6.48, 2.6); arrow(8.92, 2.6, 9.58, 2.6)  # the training loop, left to right
    ax.annotate("", xy=(4.6, 2.98), xytext=(10.8, 2.98), arrowprops=dict(arrowstyle="->", color="#999999", linewidth=1.0,
                                                                        connectionstyle="arc3,rad=0.12"))  # repeat
    ax.text(7.7, 3.58, "repeat k times", ha="center", fontsize=8, color="#777777")
    box(2.6, 0.9, 3.4, 0.7, "a new word: \"lowest\"\nl o w e s t </w>", "#F2F2F2")      # encoding
    box(7.0, 0.9, 3.8, 0.7, "apply the learned merges,\nearliest first, while any applies", "#FAEEDA")
    box(11.0, 0.9, 2.6, 0.7, "low est</w>", "#E5F2E1", bold=True)
    arrow(4.32, 0.9, 5.08, 0.9); arrow(8.92, 0.9, 9.68, 0.9)  # encoding a new word
    plt.savefig("bpe_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved bpe_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("A tokenizer turns text into ids. Day 85 turns ids into vectors: one-hot vectors against dense "
          "embeddings, on a real vocabulary.")


def main():                                             # runs every section in order
    demo_by_hand()                                        # 1. by hand
    train = read_tagged("train"); test = read_tagged("test")  # Week 9's sentences
    train_words = [w for s in train for w, _ in s]        # every training word, in order
    test_words = [w for s in test for w, _ in s]          # every test word
    merges, chars = demo_train(train_words)               # 2. training
    demo_lossless(test, merges)                           # 3. lossless
    sweep = demo_sweep(train_words, test_words, merges, chars)  # 4. vocabulary against length
    demo_unknown(train_words, test_words, merges, chars)  # 5. no more <unk>
    freq = demo_frequency(train_words, test_words, merges)  # 6. frequency
    plot_results(sweep, merges, freq)                     # the figure
    draw_diagram()                                        # the diagram
    note_on_what_comes_next()                             # 7. Day 85


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
