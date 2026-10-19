"""
Day 75 Exercise -- Scaled Dot-Product Attention as One Reusable Function

Days 73-74's exercises each built an attention read inside something
else. This one writes the core on its own, in the general shape the
Transformer uses: any number of queries, any number of keys, a mask, and
the 1/sqrt(d) scaling:

    A = softmax( Q K^T / sqrt(d) )   (masked keys get weight 0),   Z = A V

It is gradient-checked for Q, K and V, and the scaling is checked in
numbers: with random inputs of spread 1, the scores' spread is about
sqrt(d) unscaled and about 1 scaled. Day 76's multi-head attention runs
this function once per head.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def scaled_dot_product_attention(Q, K, V, mask=None):  # NEW: Q (m, nq, d), K (m, nk, d), V (m, nk, dv); mask (m, nq, nk)
    d = Q.shape[-1]                                       # the query and key size
    S = np.einsum("mid,mjd->mij", Q, K) / np.sqrt(d)      # scaled scores
    if mask is not None:
        S = np.where(mask, S, -np.inf)                      # masked keys get minus infinity ...
    e = np.exp(S - S.max(axis=-1, keepdims=True))         # ... and so weight exactly 0
    A = e / e.sum(axis=-1, keepdims=True)                 # weights over the keys
    return np.einsum("mij,mjd->mid", A, V), A             # outputs, and the weights (the backward pass needs them)


def scaled_dot_product_attention_backward(dZ, Q, K, V, A):  # NEW: gradients for Q, K and V
    d = Q.shape[-1]
    dA = np.einsum("mid,mjd->mij", dZ, V)                 # Z = A V
    dV = np.einsum("mij,mid->mjd", A, dZ)
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True)) / np.sqrt(d)  # back through the softmax and the scaling
    return np.einsum("mij,mjd->mid", dS, K), np.einsum("mij,mid->mjd", dS, Q), dV  # dQ, dK, dV


rng = np.random.RandomState(0)                          # fixed rng
m, nq, nk, d, dv = 2, 3, 5, 4, 2                          # 2 examples, 3 queries, 5 keys, sizes 4 and 2
Q, K, V = rng.randn(m, nq, d), rng.randn(m, nk, d), rng.randn(m, nk, dv)
mask = np.ones((m, nq, nk), bool); mask[1, :, 3:] = False  # example 2: its last two keys are padding
U = rng.randn(m, nq, dv)                                  # L = sum(U * Z)

Z, A = scaled_dot_product_attention(Q, K, V, mask)        # forward
dQ, dK, dV = scaled_dot_product_attention_backward(U, Q, K, V, A)  # analytic gradients
worst = 0.0
for X, G in [(Q, dQ), (K, dK), (V, dV)]:                  # each input
    for pos in np.ndindex(X.shape):
        old = X[pos]
        X[pos] = old + 1e-6; up = (U * scaled_dot_product_attention(Q, K, V, mask)[0]).sum()
        X[pos] = old - 1e-6; dn = (U * scaled_dot_product_attention(Q, K, V, mask)[0]).sum()
        X[pos] = old
        worst = max(worst, abs((up - dn) / 2e-6 - G[pos]))
print(f"gradient check (Q, K, V): largest |numeric - analytic| = {worst:.2e}; weight on masked keys: {A[1, :, 3:].max():.1e}")

for dd in [16, 256]:                                      # the reason for the scaling, in two numbers each
    q, k = rng.randn(4000, dd), rng.randn(4000, dd)
    s = (q * k).sum(axis=1)                               # unscaled dot products
    print(f"d = {dd:>3}: spread of q . k = {s.std():6.2f} (sqrt(d) = {np.sqrt(dd):5.2f}); after dividing by sqrt(d): {(s / np.sqrt(dd)).std():.2f}")
