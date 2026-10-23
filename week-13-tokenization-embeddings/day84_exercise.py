"""
Day 84 Exercise -- A Byte-Pair-Encoding Tokenizer in Four Functions

The first of Week 13's parts for Day 90's tiny GPT-style model: a BPE
tokenizer on its own, small enough to read in one go.

    train(text, n_merges)   learn merges from a text
    encode(text, merges)    text -> token ids
    decode(ids, vocab)      token ids -> text

It works on the characters of the whole text (spaces included, so there is
no separate end-of-word mark), checks that decoding gives back exactly the
input, and shows the sequence getting shorter as merges are added. The
training text is the first part of Day 61's real review sentences; the
test text is the rest.
"""
import os                                              # os: build the path to Week 9's review sentences
from collections import Counter                        # Counter counts pairs

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week-09-sequence-data-rnns-i",
                    "sentiment_labelled_sentences", "yelp_labelled.txt")  # Day 61's Yelp sentences


def merge(seq, pair, new):                              # replace every occurrence of pair in a list of symbols
    out, i = [], 0                                        # result; position
    while i < len(seq):                                   # left to right
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:  # the pair starts here
            out.append(new); i += 2                           # one symbol for two
        else:
            out.append(seq[i]); i += 1                        # unchanged
    return out                                            # the merged list


def train(text, n_merges):                              # learn merges: the most frequent adjacent pair, n_merges times
    seq, merges = list(text), []                          # the text as characters; the merges learned
    for _ in range(n_merges):                             # one merge per round
        counts = Counter(zip(seq, seq[1:]))                 # every adjacent pair and its count
        if not counts:                                      # one symbol left
            break                                             # nothing to merge
        best = max(counts, key=lambda p: (counts[p], p))    # most frequent; ties broken by the pair itself
        merges.append(best)                                 # remember it
        seq = merge(seq, best, best[0] + best[1])           # apply it to the training text
    vocab = sorted(set(text)) + [a + b for a, b in merges]  # every symbol: the characters, then the merged ones
    return merges, vocab                                  # merges in order, and the vocabulary


def encode(text, merges, vocab):                        # text -> ids, applying the merges in the order learned
    seq = list(text)                                      # characters
    for pair in merges:                                   # earliest merge first
        seq = merge(seq, pair, pair[0] + pair[1])           # apply it
    index = {s: i for i, s in enumerate(vocab)}           # symbol -> id
    return [index[s] for s in seq]                        # a KeyError here means a character never seen in training


def decode(ids, vocab):                                 # ids -> text
    return "".join(vocab[i] for i in ids)                 # every symbol is its own piece of text


lines = [l.split("\t")[0].lower() for l in open(DATA, encoding="utf-8").read().splitlines() if l.strip()]  # 1,000 sentences
train_text, test_text = "\n".join(lines[:800]), "\n".join(lines[800:])  # 800 to train, 200 to test
chars = set(train_text)                                   # characters seen in training
test_text = "".join(ch for ch in test_text if ch in chars)  # drop the few unseen characters (Day 84's lesson: they cannot be encoded)
for k in (0, 50, 200, 800):                               # four vocabulary sizes
    merges, vocab = train(train_text, k)                    # learn k merges
    ids = encode(test_text, merges, vocab)                  # tokenize the test text
    assert decode(ids, vocab) == test_text                  # lossless
    print(f"{k:>4} merges: vocabulary {len(vocab):>4}, test text {len(test_text)} characters -> {len(ids)} tokens "
          f"({len(test_text) / len(ids):.2f} characters per token), decodes back exactly: True")
merges, vocab = train(train_text, 800)                    # the largest
print("first 8 merges:", " | ".join(repr(a + b) for a, b in merges[:8]))
print("'the food was amazing' ->", [vocab[i] for i in encode("the food was amazing", merges, vocab)])
