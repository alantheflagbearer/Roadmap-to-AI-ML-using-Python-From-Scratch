"""
Day 88 Exercise -- A Word's Vector From Its Character N-Grams

The fifth of Week 13's parts for Day 90's tiny GPT-style model. Day 90's
model invents words as it writes; char_ngrams is how it checks whether an
invented word is built from pieces that real words use.

    char_ngrams(word)          '<word>' cut into every piece of 3 to 6 characters
    bucket(gram)               a 32-bit FNV-1a hash of the piece, modulo the number of rows
    subword_vector(word, T)    the mean of the table rows of the word's pieces
    subword_backward(word, dv) its gradient: each piece's row gets dv / (number of pieces)

Checked: a numerical gradient check (with a piece used twice in one word, so
the adding-up matters), the same vector for the same word on every run (a
fixed hash, not Python's hash()), and a vector for a word no table was ever
trained on.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this

B = 1000                                                # rows in the table (small: some pieces still share a row)


def char_ngrams(word, nmin=3, nmax=6):                  # NEW: every character n-gram of '<word>', lengths nmin..nmax
    w = "<" + word + ">"                                  # start and end marks
    return [w[i:i + n] for n in range(nmin, nmax + 1) for i in range(len(w) - n + 1)]  # all of them, in order


def bucket(gram):                                       # NEW: FNV-1a, 32 bits, modulo B: the same row on every run
    h = 2166136261                                        # the FNV offset basis
    for b in gram.encode("utf-8"):                        # byte by byte
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF             # xor, multiply by the FNV prime, keep 32 bits
    return h % B                                          # a row


def subword_vector(word, T):                            # NEW: the mean of the word's pieces' rows
    return T[[bucket(g) for g in char_ngrams(word)]].mean(axis=0)


def subword_backward(word, dv, T):                      # NEW: gradient for the table
    rows = [bucket(g) for g in char_ngrams(word)]         # the pieces' rows (a row can appear more than once)
    dT = np.zeros_like(T); np.add.at(dT, rows, dv / len(rows))  # each piece's share, added up per row
    return dT                                             # rows of other pieces stay 0


print("'banana':", " ".join(char_ngrams("banana")))     # 'ana' occurs twice
g = char_ngrams("banana"); print(f"{len(g)} pieces, {len(set(g))} different; rows {[bucket(x) for x in g[:6]]} ...")
rng = np.random.RandomState(0)                          # fixed rng
T = rng.randn(B, 4); u = rng.randn(4)                    # a table of 4 numbers per row; L = u . v
dT = subword_backward("banana", u, T)                    # analytic
worst = 0.0                                              # largest |numeric - analytic|
for pos in np.ndindex(T.shape):                          # every entry of the table
    old = T[pos]                                           # remember it
    T[pos] = old + 1e-6; up = u @ subword_vector("banana", T)  # nudge up
    T[pos] = old - 1e-6; dn = u @ subword_vector("banana", T)  # nudge down
    T[pos] = old                                           # restore
    worst = max(worst, abs((up - dn) / 2e-6 - dT[pos]))    # largest difference so far
print(f"gradient check, every entry of the table: largest |numeric - analytic| = {worst:.2e}")
print(f"the same rows on every run (fixed hash): bucket('<ba') = {bucket('<ba')}; Python's own hash() changes from run to run")
T50 = rng.randn(B, 50)                                   # a table of 50 numbers per row (4 is too few for cosines to mean much)
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)  # cosine similarity
v = subword_vector("bananas", T50)                       # a word nobody trained on
print(f"'bananas' was never seen, but shares {len(set(char_ngrams('bananas')) & set(g))} of its "
      f"{len(char_ngrams('bananas'))} pieces with 'banana'. Random 50-number table: cosine with 'banana' "
      f"{cos(v, subword_vector('banana', T50)):.3f}, with 'xylophone' {cos(v, subword_vector('xylophone', T50)):.3f} "
      f"(before any training, similarity comes only from shared pieces)")
