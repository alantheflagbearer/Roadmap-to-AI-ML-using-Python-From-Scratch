"""
Day 80 Exercise -- Causal Self-Attention and Cross-Attention

The fourth of Week 12's parts for Day 83's Transformer: the two
attentions a decoder needs, built on Day 75's function.

    causal self-attention:  queries, keys, values from the decoder; key j > query i is masked out
    cross-attention:        queries from the decoder, keys and values from the ENCODER (another length)

It checks that the causal mask leaves exactly zero weight on later
positions and that changing a later input leaves earlier outputs
unchanged, and gradient-checks cross-attention for the decoder input,
the encoder output and all four weight matrices.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def scaled_dot_product_attention(Q, K, V, mask=None):  # day75_exercise.py, verbatim: any nq queries, any nk keys
    d = Q.shape[-1]                                       # the query and key size
    S = np.einsum("mid,mjd->mij", Q, K) / np.sqrt(d)      # scaled scores, (m, nq, nk)
    if mask is not None:                                  # masked keys ...
        S = np.where(mask, S, -np.inf)                      # ... get minus infinity
    e = np.exp(S - S.max(axis=-1, keepdims=True))         # ... and so weight exactly 0
    A = e / e.sum(axis=-1, keepdims=True)                 # weights over the keys
    return np.einsum("mij,mjd->mid", A, V), A             # outputs, and the weights


def causal_mask(n):                                     # NEW: True where query i may read key j (j <= i)
    return np.tril(np.ones((n, n), dtype=bool))           # lower triangle


def cross_attention(Y, E, P, emask):                    # NEW: decoder states Y (m, ny, d) read encoder output E (m, ne, d)
    Q, K, V = Y @ P["Wq"], E @ P["Wk"], E @ P["Wv"]       # queries from the decoder; keys and values from the encoder
    Z, A = scaled_dot_product_attention(Q, K, V, emask[:, None, :])  # every decoder position reads every real encoder position
    return Z @ P["Wo"], (Y, E, Q, K, V, A, Z)             # output projection, cache


def cross_attention_backward(dO, P, cache):             # NEW: gradients for Y, E and the four weights
    Y, E, Q, K, V, A, Z = cache                           # what the forward pass computed
    f = lambda T: T.reshape(-1, T.shape[-1])              # every position as one row
    dZ = dO @ P["Wo"].T                                   # back through Wo
    dA = np.einsum("mid,mjd->mij", dZ, V); dV = np.einsum("mij,mid->mjd", A, dZ)  # Z = A V
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True)) / np.sqrt(Q.shape[-1])  # softmax and scaling
    dQ, dK = np.einsum("mij,mjd->mid", dS, K), np.einsum("mij,mid->mjd", dS, Q)  # S = Q K^T
    g = dict(Wo=f(Z).T @ f(dO), Wq=f(Y).T @ f(dQ), Wk=f(E).T @ f(dK), Wv=f(E).T @ f(dV))  # weights
    return dQ @ P["Wq"].T, dK @ P["Wk"].T + dV @ P["Wv"].T, g  # dY, dE (it made keys AND values), weights


rng = np.random.RandomState(0)                          # fixed rng
d = 4                                                     # size

X = rng.randn(2, 6, d)                                    # a decoder input: 2 sequences of 6 positions
Wq, Wk, Wv = (rng.randn(d, d) for _ in range(3))          # random projections
Z, A = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv, causal_mask(6)[None])  # causal self-attention
X2 = X.copy(); X2[:, 4] = rng.randn(2, d)                 # change position 4 only
Z2, _ = scaled_dot_product_attention(X2 @ Wq, X2 @ Wk, X2 @ Wv, causal_mask(6)[None])  # again
print(f"causal self-attention: largest weight on a later position = {A[:, ~causal_mask(6)].max():.1e}; "
      f"change position 4 -> outputs 0-3 move by {np.abs(Z2 - Z)[:, :4].max():.1e}, outputs 4-5 by {np.abs(Z2 - Z)[:, 4:].max():.2f}")

Y, E = rng.randn(2, 3, d), rng.randn(2, 5, d)            # 3 decoder positions, 5 encoder positions
emask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], bool)  # the second encoder sequence has 2 padded positions
P = {k: rng.randn(d, d) / np.sqrt(d) for k in ("Wq", "Wk", "Wv", "Wo")}  # cross-attention weights
U = rng.randn(2, 3, d)                                    # L = sum(U * output)
O, cache = cross_attention(Y, E, P, emask)                # forward
dY, dE, g = cross_attention_backward(U, P, cache)         # analytic gradients
worst = 0.0                                               # largest |numeric - analytic|
for name, T in dict(Y=Y, E=E, **P).items():               # the two inputs and the four weights
    G = {"Y": dY, "E": dE}.get(name, g.get(name))           # its analytic gradient
    for pos in np.ndindex(T.shape):                         # every entry
        old = T[pos]                                          # remember it
        T[pos] = old + 1e-6; up = (U * cross_attention(Y, E, P, emask)[0]).sum()  # nudge up
        T[pos] = old - 1e-6; dn = (U * cross_attention(Y, E, P, emask)[0]).sum()  # nudge down
        T[pos] = old                                          # restore
        worst = max(worst, abs((up - dn) / 2e-6 - G[pos]))    # largest difference so far
print(f"cross-attention: weights are {cache[5].shape[1]} decoder x {cache[5].shape[2]} encoder positions; "
      f"weight on padded encoder positions {cache[5][1][:, 3:].max():.1e}; gradient at them {np.abs(dE[1, 3:]).max():.1e}")
print(f"cross-attention gradient check (Y, E, Wq, Wk, Wv, Wo): largest |numeric - analytic| = {worst:.2e}")
