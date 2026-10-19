"""
Day 71 Exercise -- One Additive Attention Read, Forward and Backward

Day 70's exercise read a memory with given scores. This one adds where the
scores come from in additive (Bahdanau) attention:

    e_i = va . tanh(q Wa + m_i Ua)        q: the query (the decoder's state), m_i: memory slot i
    w   = softmax(e)
    out = sum_i w_i m_i

It computes one read, then its gradient with respect to the query, the
memory and the three weight arrays, and checks every entry numerically.
No sequences and no training: just the attention read.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def softmax(z):                                         # weights along the last axis, summing to 1
    e = np.exp(z - z.max(axis=-1, keepdims=True))         # subtract the row max for stability, exponentiate
    return e / e.sum(axis=-1, keepdims=True)              # normalise


def additive_read(q, M, P):                             # NEW: q (m, H), M (m, n, H) -> the read (m, H) and a cache
    t = np.tanh((q @ P["Wa"])[:, None, :] + M @ P["Ua"])  # (m, n, A): query and every key added, squashed
    w = softmax(t @ P["va"])                              # (m, n): one weight per memory slot
    return np.einsum("mn,mnh->mh", w, M), (t, w)          # the weighted sum of the slots


def additive_read_backward(dout, q, M, P, cache):       # NEW: gradients of the read for q, M, Wa, Ua, va
    t, w = cache                                          # what the forward pass computed
    dM = w[:, :, None] * dout[:, None, :]                 # out = sum w_i m_i  ->  through the weighted sum
    dw = np.einsum("mh,mnh->mn", dout, M)                 # ... and through the weights
    de = w * (dw - (w * dw).sum(axis=1, keepdims=True))   # back through the softmax
    dva = np.einsum("mn,mna->a", de, t)                   # e = t . va
    da = de[:, :, None] * P["va"] * (1 - t ** 2)          # back through tanh
    dq = da.sum(axis=1) @ P["Wa"].T                       # the query was added to every slot
    dWa = q.T @ da.sum(axis=1)                            # query weights
    dUa = np.einsum("mnh,mna->ha", M, da)                 # key weights
    dM += da @ P["Ua"].T                                  # the memory also made the keys
    return dict(q=dq, M=dM, Wa=dWa, Ua=dUa, va=dva)


rng = np.random.RandomState(0)                          # fixed rng
m, n, H, A = 2, 4, 3, 5                                   # 2 queries, 4 memory slots of 3 numbers, attention size 5
q, M = rng.randn(m, H), rng.randn(m, n, H)                # the query and the memory
P = {"Wa": rng.randn(H, A), "Ua": rng.randn(H, A), "va": rng.randn(A)}  # the weights

out, cache = additive_read(q, M, P)                       # one read
print("weights per slot:", cache[1].round(3).tolist())
print("read (first query):", out[0].round(4).tolist())

U = rng.randn(m, H)                                       # a random direction: L = sum(U * read)
grads = additive_read_backward(U, q, M, P, cache)         # analytic gradients
arrays = dict(q=q, M=M, **P)                              # every input of the read
worst = 0.0                                               # largest |numeric - analytic|
for name, X in arrays.items():                            # each input
    for pos in np.ndindex(X.shape):                         # each entry
        old = X[pos]
        X[pos] = old + 1e-6; up = (U * additive_read(q, M, P)[0]).sum()
        X[pos] = old - 1e-6; dn = (U * additive_read(q, M, P)[0]).sum()
        X[pos] = old
        worst = max(worst, abs((up - dn) / 2e-6 - grads[name][pos]))
print(f"gradient check (q, M, Wa, Ua, va): largest |numeric - analytic| = {worst:.2e}")
