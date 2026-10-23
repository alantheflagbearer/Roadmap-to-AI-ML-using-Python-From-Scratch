"""
Day 85 Exercise -- An Embedding Layer, Forward and Backward

The second of Week 13's parts for Day 90's tiny GPT-style model: the layer
that turns token ids into vectors.

    forward:   X = E[ids]                    (a lookup: row ids[i] of the table, for every position)
    backward:  dE[t] = sum of dX over every position whose id is t      (np.add.at)

It checks the lookup against the one-hot matrix product it replaces, checks
the gradient numerically (with a token that appears twice, so the adding-up
matters), and shows that rows of tokens not in the batch get no gradient.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def embed(E, ids):                                      # NEW: (V, d) table, ids of any shape -> vectors of shape ids.shape + (d,)
    return E[ids]                                         # fancy indexing: one row per id


def embed_backward(dX, E, ids):                         # NEW: gradient for the table, from the gradient at the looked-up vectors
    dE = np.zeros_like(E)                                 # one row per token
    np.add.at(dE, ids, dX)                                # a token used at several positions collects all of their gradients
    return dE                                             # rows of unused tokens stay 0


rng = np.random.RandomState(0)                          # fixed rng
V, d = 12, 4                                              # 12 tokens, 4 numbers per token
E = rng.randn(V, d)                                       # the table
ids = np.array([[3, 7, 3, 0], [5, 5, 5, 9]])              # 2 sequences of 4 tokens: 3 twice, 5 three times
U = rng.randn(*ids.shape, d)                              # L = sum(U * X)

X = embed(E, ids)                                         # forward
one_hot = np.eye(V)[ids]                                  # the same ids as one-hot rows, (2, 4, V)
print(f"lookup against one-hot @ E: largest difference {np.abs(X - one_hot @ E).max():.1e}; shapes {X.shape} and {(one_hot @ E).shape}")

dE = embed_backward(U, E, ids)                            # analytic gradient
worst = 0.0                                               # largest |numeric - analytic|
for pos in np.ndindex(E.shape):                           # every entry of the table
    old = E[pos]                                            # remember it
    E[pos] = old + 1e-6; up = (U * embed(E, ids)).sum()    # nudge up
    E[pos] = old - 1e-6; dn = (U * embed(E, ids)).sum()    # nudge down
    E[pos] = old                                            # restore
    worst = max(worst, abs((up - dn) / 2e-6 - dE[pos]))     # largest difference so far
print(f"gradient check, every entry of the table: largest |numeric - analytic| = {worst:.2e}")
print(f"token 5 appears 3 times: its gradient equals the sum of its three positions' gradients: "
      f"{np.allclose(dE[5], U[1, 0] + U[1, 1] + U[1, 2])}")
unused = [t for t in range(V) if t not in ids]            # tokens not in the batch
print(f"tokens not in the batch {unused}: largest gradient {np.abs(dE[unused]).max():.1e}")
dE_wrong = np.zeros_like(E); dE_wrong[ids] = U            # the common mistake: plain assignment keeps only the LAST position of a repeated id
print(f"with dE[ids] = dX instead of np.add.at, token 5's gradient is off by {np.abs(dE_wrong[5] - dE[5]).max():.2f}")
