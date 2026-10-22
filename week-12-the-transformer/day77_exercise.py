"""
Day 77 Exercise -- Sinusoidal Positional Encoding as One Reusable Function

The first of Week 12's parts for Day 83's Transformer. It builds the
encoding table on its own:

    PE[p, 2i] = sin(p / 10000^(2i/d)),   PE[p, 2i+1] = cos(p / 10000^(2i/d))

and checks the three things the rest of the week relies on: every number
lies between -1 and 1; moving k positions along is the same rotation R_k
for every starting position (so PE[p] . PE[p + k] depends on k only); and
once the table is added to the word vectors, Day 75's attention is no
longer blind to word order.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sinusoidal_encoding(n, d):                          # NEW: (n, d) table, row p = position p
    p = np.arange(n)[:, None]                             # positions as a column
    i = np.arange(d // 2)[None, :]                        # pair index as a row
    angle = p / 10000 ** (2 * i / d)                      # one angle per (position, pair)
    PE = np.zeros((n, d))                                 # the table
    PE[:, 0::2], PE[:, 1::2] = np.sin(angle), np.cos(angle)  # sine in even columns, cosine in odd ones
    return PE                                             # hand back


def scaled_dot_product_attention(Q, K, V):              # day75_exercise.py, without the mask
    S = np.einsum("mid,mjd->mij", Q, K) / np.sqrt(Q.shape[-1])  # scaled scores
    A = np.exp(S - S.max(axis=-1, keepdims=True)); A /= A.sum(axis=-1, keepdims=True)  # softmax over the keys
    return np.einsum("mij,mjd->mid", A, V)                # weighted average of the values


d, n = 16, 50                                             # size 16, positions 0..49
PE = sinusoidal_encoding(n, d)                            # the table
print(f"range of the numbers: {PE.min():.3f} .. {PE.max():.3f}")

k = 3                                                     # a shift of 3 positions
R = np.zeros((d, d))                                      # the rotation for this shift, built from k alone
for i in range(d // 2):                                   # one 2 x 2 block per pair
    a = k / 10000 ** (2 * i / d)                            # this pair's turn over k positions
    R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]  # turns (sin, cos) on by a
print(f"largest |PE[p + 3] - PE[p] R_3| over p = 0..46: {np.abs(PE[k:] - PE[:-k] @ R).max():.2e}")
dots = (PE[:-k] * PE[k:]).sum(axis=1)                     # PE[p] . PE[p + 3] for every p
print(f"PE[p] . PE[p + 3] over every p: from {dots.min():.6f} to {dots.max():.6f}")

rng = np.random.RandomState(0)                            # fixed rng
X = rng.randn(2, 6, d)                                    # 2 sentences of 6 word vectors
Wq, Wk, Wv = (rng.randn(d, d) / np.sqrt(d) for _ in range(3))  # random projections
perm = np.array([5, 2, 0, 4, 1, 3])                       # one shuffle of the 6 words
for label, pos in [("without", 0.0), ("with", PE[:6])]:   # the two cases
    Z = scaled_dot_product_attention((X + pos) @ Wq, (X + pos) @ Wk, (X + pos) @ Wv)  # original order
    Xs = X[:, perm] + pos                                   # the words shuffled; the positions stay put
    Zs = scaled_dot_product_attention(Xs @ Wq, Xs @ Wk, Xs @ Wv)  # shuffled order
    print(f"{label:<7} the encoding: largest |output(shuffled) - shuffled(output)| = {np.abs(Zs - Z[:, perm]).max():.2e}")
