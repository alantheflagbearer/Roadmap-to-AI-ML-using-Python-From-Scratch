"""
Day 74 Exercise -- One Self-Attention Layer, Forward and Backward

The lesson's new piece on its own: queries, keys and values are three
projections of the SAME inputs, and every position reads a weighted
average of all the values:

    Q = X Wq,  K = X Wk,  V = X Wv,   A = softmax(Q K^T),   Z = A V

It checks every gradient numerically (with one sequence padded, so padded
positions must be ignored as keys), and checks that shuffling the input
positions shuffles the outputs in exactly the same way.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def masked_softmax(scores, mask):                      # from day70_exercise.py: padded keys get weight exactly 0
    s = np.where(mask, scores, -np.inf)                   # padded scores -> minus infinity
    e = np.exp(s - s.max(axis=-1, keepdims=True))         # stable exponentials; exp(-inf) = 0
    return e / e.sum(axis=-1, keepdims=True)              # weights over the real keys


def self_attention(X, P, mask):                         # NEW: X (m, n, D) -> Z (m, n, d), and a cache
    Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]       # three projections of the same inputs
    A = masked_softmax(np.einsum("mid,mjd->mij", Q, K), mask[:, None, :])  # A[m, i, j]: how much i reads j
    return np.einsum("mij,mjd->mid", A, V), (X, Q, K, V, A)


def self_attention_backward(dZ, P, cache):              # NEW: gradients for X, Wq, Wk, Wv
    X, Q, K, V, A = cache
    dA = np.einsum("mid,mjd->mij", dZ, V)                 # through Z = A V ...
    dV = np.einsum("mij,mid->mjd", A, dZ)                 # ... into the values
    dS = A * (dA - (A * dA).sum(axis=2, keepdims=True))   # back through each row's softmax
    dQ, dK = np.einsum("mij,mjd->mid", dS, K), np.einsum("mij,mid->mjd", dS, Q)  # S = Q K^T
    f = lambda T: T.reshape(-1, T.shape[-1])              # all positions as rows
    return dict(X=dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T,  # X made all three
                Wq=f(X).T @ f(dQ), Wk=f(X).T @ f(dK), Wv=f(X).T @ f(dV))


rng = np.random.RandomState(0)                          # fixed rng
m, n, D, d = 2, 4, 3, 2                                   # 2 sequences, 4 positions, 3 inputs, attention size 2
X = rng.randn(m, n, D)                                    # inputs
P = {k: rng.randn(D, d) for k in ["Wq", "Wk", "Wv"]}      # the three projections
mask = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], bool)       # sequence 2 has a padded last position
U = rng.randn(m, n, d) * mask[:, :, None]                 # L = sum(U * Z) over the real positions only

Z, cache = self_attention(X, P, mask)                     # forward
grads = self_attention_backward(U, P, cache)              # analytic gradients
arrays = dict(X=X, **P)                                   # every input
worst = 0.0                                               # largest |numeric - analytic|
for name, A_ in arrays.items():
    for pos in np.ndindex(A_.shape):
        old = A_[pos]
        A_[pos] = old + 1e-6; up = (U * self_attention(X, P, mask)[0]).sum()
        A_[pos] = old - 1e-6; dn = (U * self_attention(X, P, mask)[0]).sum()
        A_[pos] = old
        worst = max(worst, abs((up - dn) / 2e-6 - grads[name][pos]))
print(f"gradient check (X, Wq, Wk, Wv): largest |numeric - analytic| = {worst:.2e}")
print(f"weight on the padded key, largest over all queries: {cache[4][1, :, 3].max():.1e}")
print(f"gradient at the padded input position (its value is never read): {np.abs(grads['X'][1, 3]).max():.1e}")

perm = np.array([2, 0, 3, 1])                             # shuffle the 4 positions of sequence 1
Zp, _ = self_attention(X[:1, perm], P, mask[:1])          # shuffled input
print(f"shuffle the positions: largest |output(shuffled) - shuffled(output)| = {np.abs(Zp - Z[:1, perm]).max():.2e}")
