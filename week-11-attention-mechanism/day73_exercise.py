"""
Day 73 Exercise -- Dot-Product and General Attention Reads, Forward and Backward

Day 71's exercise built one additive attention read. This one builds the
two multiplicative scores of Luong et al. (2015) in the same shape:

    dot:      e_i = q . m_i              (no weights)
    general:  e_i = (q Wg) . m_i         (one weight matrix)
    w = softmax(e),  out = sum_i w_i m_i

and checks every gradient numerically. The dot read is the one Week 11's
remaining days (self-attention, scaling, multiple heads) are built on.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def softmax(z):                                         # weights along the last axis, summing to 1
    e = np.exp(z - z.max(axis=-1, keepdims=True))         # subtract the row max for stability, exponentiate
    return e / e.sum(axis=-1, keepdims=True)              # normalise


def dot_read(q, M, Wg=None):                            # NEW: q (m, H), M (m, n, H); Wg = None for dot, a matrix for general
    qq = q if Wg is None else q @ Wg                      # the query, projected for the general score
    w = softmax(np.einsum("mh,mnh->mn", qq, M))           # one weight per memory slot
    return np.einsum("mn,mnh->mh", w, M), (qq, w)         # the weighted sum, and a cache


def dot_read_backward(dout, q, M, cache, Wg=None):      # NEW: gradients of the read for q, M and Wg
    qq, w = cache                                         # what the forward pass computed
    dM = w[:, :, None] * dout[:, None, :]                 # through the weighted sum ...
    dw = np.einsum("mh,mnh->mn", dout, M)                 # ... and into the weights
    de = w * (dw - (w * dw).sum(axis=1, keepdims=True))   # back through the softmax
    dqq = np.einsum("mn,mnh->mh", de, M)                  # e_i = qq . m_i  ->  d qq = sum_i de_i m_i
    dM += de[:, :, None] * qq[:, None, :]                 # ... and d m_i += de_i qq
    if Wg is None:
        return dict(q=dqq, M=dM)                          # dot: the query is used directly
    return dict(q=dqq @ Wg.T, M=dM, Wg=q.T @ dqq)         # general: back through qq = q Wg


rng = np.random.RandomState(0)                          # fixed rng
m, n, H = 2, 4, 3                                         # 2 queries, 4 memory slots of 3 numbers
q, M, Wg = rng.randn(m, H), rng.randn(m, n, H), rng.randn(H, H)  # query, memory, general weights
U = rng.randn(m, H)                                       # a random direction: L = sum(U * read)

for name, W in [("dot", None), ("general", Wg)]:           # both scores
    out, cache = dot_read(q, M, W)                         # one read
    grads = dot_read_backward(U, q, M, cache, W)           # analytic gradients
    arrays = dict(q=q, M=M) if W is None else dict(q=q, M=M, Wg=Wg)  # every input of this read
    worst = 0.0                                            # largest |numeric - analytic|
    for key, X in arrays.items():                          # each input
        for pos in np.ndindex(X.shape):                      # each entry
            old = X[pos]
            X[pos] = old + 1e-6; up = (U * dot_read(q, M, W)[0]).sum()
            X[pos] = old - 1e-6; dn = (U * dot_read(q, M, W)[0]).sum()
            X[pos] = old
            worst = max(worst, abs((up - dn) / 2e-6 - grads[key][pos]))
    print(f"{name:<8} weights {cache[1][0].round(3).tolist()}  gradient check: largest |numeric - analytic| = {worst:.2e}")
print(f"general with Wg = identity equals dot: {np.allclose(dot_read(q, M, np.eye(H))[0], dot_read(q, M)[0])}")
