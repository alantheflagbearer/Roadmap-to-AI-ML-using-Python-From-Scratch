"""
Day 89 Exercise -- Vector Files: Save, Load, Look Up

The sixth and last of Week 13's parts for Day 90's tiny GPT-style model.
Pretrained vectors travel as text files; Day 90 saves its learned character
table the same way, loads it back, and asks which characters it learned to
treat alike.

    save_vectors(path, words, M)   one line per word: the word, then its numbers (GloVe's format)
    load_vectors(path)             reads GloVe's format, or word2vec's (a 'count dim' first line)
    most_similar(words, M, w, k)   the k words whose vectors have the largest cosine with w's

Checked: a round trip loses only the rounding to 5 decimals, both file
formats load, and a token that is itself a space or a symbol survives.
"""
import os                                              # os: a temporary file path
import tempfile                                        # tempfile: somewhere to write the test files

import numpy as np                                     # numpy: every array/matrix operation below is built on this


def save_vectors(path, words, M):                       # NEW: GloVe's text format; spaces and newlines inside a token are escaped
    with open(path, "w", encoding="utf-8") as f:          # plain text
        for w, v in zip(words, M):                          # one line per token
            tok = w.replace(" ", "<sp>").replace("\n", "<nl>")  # a character model has ' ' and newline as tokens
            f.write(tok + " " + " ".join(f"{x:.5f}" for x in v) + "\n")


def load_vectors(path):                                 # NEW: GloVe's format, or word2vec's with its 'count dim' header
    with open(path, encoding="utf-8") as f:               # plain text
        lines = f.read().splitlines()                       # every line
    first = lines[0].split()                              # a word2vec file starts with two integers
    if len(first) == 2 and all(t.isdigit() for t in first):  # header: skip it
        lines = lines[1:]
    words = [ln.split(" ", 1)[0].replace("<sp>", " ").replace("<nl>", "\n") for ln in lines]  # undo the escaping
    return words, np.array([[float(x) for x in ln.split(" ")[1:]] for ln in lines])  # tokens, (n, d) matrix


def most_similar(words, M, w, k=3):                     # NEW: the k nearest tokens to w by cosine similarity
    N = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)  # unit rows
    s = N @ N[words.index(w)]; s[words.index(w)] = -np.inf  # cosines; not w itself
    return [(words[i], round(float(s[i]), 3)) for i in np.argsort(-s)[:k]]  # most similar first


rng = np.random.RandomState(0)                          # fixed rng
words = ["a", "e", "i", "x", " ", "\n", "!"]              # a few character tokens, including a space and a newline
M = rng.randn(len(words), 4)                              # their vectors
M[1] = M[0] + 0.1 * rng.randn(4); M[2] = M[0] + 0.1 * rng.randn(4)  # make the vowels alike
path = os.path.join(tempfile.mkdtemp(), "chars.txt")     # a temporary file
save_vectors(path, words, M)                              # save
w2, M2 = load_vectors(path)                               # load
print(f"round trip: same tokens {w2 == words}; largest |difference| {np.abs(M2 - M).max():.1e} (5 decimals kept)")
print("first two lines of the file:", open(path, encoding="utf-8").read().splitlines()[:2])
with open(path, "w", encoding="utf-8") as f:              # the same vectors in word2vec's format: a header line first
    f.write(f"{len(words)} 4\n" + "".join((w.replace(' ', '<sp>').replace('\n', '<nl>')) + " " + " ".join(f"{x:.5f}" for x in v) + "\n"
                                        for w, v in zip(words, M)))
w3, M3 = load_vectors(path)                               # load again
print(f"word2vec format (with a header): same tokens {w3 == words}, same vectors {np.allclose(M3, M2)}")
print(f"most similar to 'a': {most_similar(w2, M2, 'a')}")
