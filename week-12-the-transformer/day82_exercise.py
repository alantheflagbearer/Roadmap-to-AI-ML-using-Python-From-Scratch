"""
Day 82 Exercise -- The Whole Transformer, Backward, From the Week's Parts

The sixth and last of Week 12's parts for Day 83: the backward pass of
Day 81's exercise model, built from the earlier exercise scripts:

    Day 78  encoder_block_backward          the encoder block
    Day 79  layer_norm_backward             every add & norm
    Day 80  cross_attention_backward        the decoder's cross-attention

New today: decoder_block_backward (the three parts in reverse, with the
causal self-attention's backward pass written out) and
transformer_backward (output layer -> decoder -> encoder -> the one
embedding table, which collects from both sides). Every entry of every
weight is checked numerically, with padded source and target positions.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sinusoidal_encoding(n, d):                          # NEW: (n, d) table, row p = position p
    p = np.arange(n)[:, None]                             # positions as a column
    i = np.arange(d // 2)[None, :]                        # pair index as a row
    angle = p / 10000 ** (2 * i / d)                      # one angle per (position, pair)
    PE = np.zeros((n, d))                                 # the table
    PE[:, 0::2], PE[:, 1::2] = np.sin(angle), np.cos(angle)  # sine in even columns, cosine in odd ones
    return PE                                             # hand back


def attention(Q, K, V, mask):                           # day75_exercise.py: scaled dot-product attention with a key mask
    S = np.where(mask, np.einsum("mid,mjd->mij", Q, K) / np.sqrt(Q.shape[-1]), -np.inf)  # scaled, masked scores
    A = np.exp(S - S.max(axis=-1, keepdims=True)); A /= A.sum(axis=-1, keepdims=True)  # softmax over the keys
    return np.einsum("mij,mjd->mid", A, V), A             # outputs and weights


def split(T, h):                                        # Day 76: (m, n, d) -> (m * h, n, d / h)
    m, n, d = T.shape                                     # sizes
    return T.reshape(m, n, h, d // h).transpose(0, 2, 1, 3).reshape(m * h, n, d // h)  # heads become batch entries


def merge(T, h):                                        # Day 76: the inverse
    mh, n, dk = T.shape                                   # sizes
    return T.reshape(mh // h, h, n, dk).transpose(0, 2, 1, 3).reshape(mh // h, n, h * dk)  # heads side by side


def encoder_block(x, P, h, mask):                       # NEW: x (m, n, d) -> y (m, n, d), and a cache
    km = np.repeat(mask[:, None, None, :], h, axis=1).reshape(-1, 1, x.shape[1])  # padded keys, per head
    Q, K, V = (split(x @ P[w], h) for w in ("Wq", "Wk", "Wv"))  # Day 74's projections, cut into heads
    Zh, A = attention(Q, K, V, km)                        # every head at once
    Z = merge(Zh, h)                                      # heads side by side
    h1, n1 = layer_norm(x + Z @ P["Wo"], P["g1"], P["b1"])  # add & norm
    u = h1 @ P["W1"] + P["c1"]; r = np.maximum(u, 0)      # FFN: widen, ReLU
    y, n2 = layer_norm(h1 + r @ P["W2"] + P["c2"], P["g2"], P["b2"])  # FFN back to d, add & norm
    return y, (x, Q, K, V, A, Z, h1, n1, u, r, n2)        # output, and everything the backward pass needs


def encoder_block_backward(dy, P, h, cache):            # NEW: gradients for x and every weight
    x, Q, K, V, A, Z, h1, n1, u, r, n2 = cache            # what the forward pass computed
    f = lambda T: T.reshape(-1, T.shape[-1])              # every position as one row
    g = {}                                                # weight gradients
    ds2, g["g2"], g["b2"] = layer_norm_backward(dy, P["g2"], n2)  # second norm
    g["W2"], g["c2"] = f(r).T @ f(ds2), f(ds2).sum(axis=0)  # FFN, second layer
    du = (ds2 @ P["W2"].T) * (u > 0)                      # through W2 and the ReLU
    g["W1"], g["c1"] = f(h1).T @ f(du), f(du).sum(axis=0)  # FFN, first layer
    dh1 = ds2 + du @ P["W1"].T                            # the residual and the FFN both read h1
    ds1, g["g1"], g["b1"] = layer_norm_backward(dh1, P["g1"], n1)  # first norm
    g["Wo"] = f(Z).T @ f(ds1)                             # output projection
    dZh = split(ds1 @ P["Wo"].T, h)                       # back into the heads
    dA = np.einsum("mid,mjd->mij", dZh, V); dV = np.einsum("mij,mid->mjd", A, dZh)  # Z = A V
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True)) / np.sqrt(Q.shape[-1])  # softmax and scaling
    dQ, dK = merge(np.einsum("mij,mjd->mid", dS, K), h), merge(np.einsum("mij,mid->mjd", dS, Q), h)  # S = Q K^T
    dV = merge(dV, h)                                     # heads side by side
    g["Wq"], g["Wk"], g["Wv"] = f(x).T @ f(dQ), f(x).T @ f(dK), f(x).T @ f(dV)  # projections
    dx = ds1 + dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T  # the residual and the three projections
    return dx, g                                          # input gradient, weight gradients


def layer_norm(x, g, b, eps=1e-5):                      # NEW: x (..., d) -> same shape; statistics over the last axis
    sd = np.sqrt(x.var(axis=-1, keepdims=True) + eps)     # each position's spread
    xhat = (x - x.mean(axis=-1, keepdims=True)) / sd      # mean 0, spread 1 per position
    return xhat * g + b, (xhat, sd)                       # output and cache


def layer_norm_backward(dy, g, cache):                  # NEW: gradients for x, g and b
    xhat, sd = cache                                      # what the forward pass computed
    dh = dy * g                                           # through the scale
    dx = (dh - dh.mean(axis=-1, keepdims=True) - xhat * (dh * xhat).mean(axis=-1, keepdims=True)) / sd  # through mean and spread
    flat = lambda T: T.reshape(-1, T.shape[-1])           # every position as a row
    return dx, flat(dy * xhat).sum(axis=0), flat(dy).sum(axis=0)  # dx, dg, db


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


def decoder_block(y, enc, P, h, ymask, emask):          # NEW: masked self-attention, cross-attention, FFN; each with add & norm
    m, n, _ = y.shape                                     # batch, decoder length
    allowed = ymask[:, None, :] & causal_mask(n)[None]    # real keys that are not in the future (Day 80)
    km = np.repeat(allowed[:, None], h, axis=1).reshape(m * h, n, n)  # per head
    Q, K, V = (split(y @ P[w], h) for w in ("sq", "sk", "sv"))  # Day 78's projections, cut into heads
    Zh, A = attention(Q, K, V, km)                        # Day 78's attention with the causal mask
    Z = merge(Zh, h)                                      # heads side by side
    h1, n1 = layer_norm(y + Z @ P["so"], P["g1"], P["b1"])  # add & norm (Day 79's LayerNorm)
    ca, cc = cross_attention(h1, enc, P["cross"], emask)  # Day 80: queries from the decoder, keys and values from the encoder
    h2, n2 = layer_norm(h1 + ca, P["g2"], P["b2"])        # add & norm
    u = h2 @ P["W1"] + P["c1"]; r = np.maximum(u, 0)      # FFN: widen, ReLU (Day 78)
    out, n3 = layer_norm(h2 + r @ P["W2"] + P["c2"], P["g3"], P["b3"])  # back to d, add & norm
    return out, (y, Q, K, V, A, Z, h1, n1, cc, h2, n2, u, r, n3)  # output, and everything the backward pass needs


def transformer_forward(src, tgt_in, P, h, smask, tmask):  # NEW: the whole model -> log-probabilities (m, T, V)
    n, T = src.shape[1], tgt_in.shape[1]                  # source and target lengths
    PE = sinusoidal_encoding(max(n, T), P["E"].shape[1])  # Day 77's table
    x = P["E"][src] + PE[:n]                              # source embeddings plus positions
    enc, ec = encoder_block(x, P["enc"], h, smask)        # Day 78's encoder block
    y = P["E"][tgt_in] + PE[:T]                           # target embeddings plus positions (the same table)
    dec, dc = decoder_block(y, enc, P["dec"], h, tmask, smask)  # today's decoder block
    logits = dec @ P["Wout"] + P["bout"]                  # a score per token
    logp = logits - logits.max(axis=-1, keepdims=True)    # log-softmax ...
    logp -= np.log(np.exp(logp).sum(axis=-1, keepdims=True))  # ... in two steps
    return logp, (src, tgt_in, x, ec, enc, y, dc, dec)    # and the cache (Day 82's backward pass)


def init_params(rng, V, d, ff, out_scale=0.1):          # NEW: every weight, small output layer (Day 81's lesson)
    s = lambda a, b: rng.randn(a, b) / np.sqrt(a)         # a weight matrix
    ln = lambda: (np.ones(d), np.zeros(d))                # a LayerNorm's scale and shift
    enc = dict(Wq=s(d, d), Wk=s(d, d), Wv=s(d, d), Wo=s(d, d), W1=s(d, ff), c1=np.zeros(ff), W2=s(ff, d), c2=np.zeros(d))  # the encoder block's weights (Day 78's names)
    enc["g1"], enc["b1"] = ln(); enc["g2"], enc["b2"] = ln()  # the encoder block's two norms
    dec = dict(sq=s(d, d), sk=s(d, d), sv=s(d, d), so=s(d, d), W1=s(d, ff), c1=np.zeros(ff), W2=s(ff, d), c2=np.zeros(d),  # the decoder block's weights
               cross=dict(Wq=s(d, d), Wk=s(d, d), Wv=s(d, d), Wo=s(d, d)))  # the decoder block, cross-attention nested
    for i in "123":                                       # the decoder block's three norms
        dec["g" + i], dec["b" + i] = ln()  # scale 1, shift 0
    return dict(E=rng.randn(V, d), enc=enc, dec=dec, Wout=s(d, V) * out_scale, bout=np.zeros(V))  # embeddings, both blocks, output layer

def decoder_block_backward(dout, P, h, cache):          # NEW: gradients for y, the encoder output, and every weight
    y, Q, K, V, A, Z, h1, n1, cc, h2, n2, u, r, n3 = cache  # what decoder_block computed
    f = lambda T: T.reshape(-1, T.shape[-1])              # every position as one row
    g = {}                                                # weight gradients
    ds3, g["g3"], g["b3"] = layer_norm_backward(dout, P["g3"], n3)  # third norm (Day 79)
    g["W2"], g["c2"] = f(r).T @ f(ds3), f(ds3).sum(axis=0)  # FFN, second layer
    du = (ds3 @ P["W2"].T) * (u > 0)                      # through W2 and the ReLU
    g["W1"], g["c1"] = f(h2).T @ f(du), f(du).sum(axis=0)  # FFN, first layer
    dh2 = ds3 + du @ P["W1"].T                            # the residual and the FFN both read h2
    ds2, g["g2"], g["b2"] = layer_norm_backward(dh2, P["g2"], n2)  # second norm
    dh1_c, denc, g["cross"] = cross_attention_backward(ds2, P["cross"], cc)  # Day 80: h1 made the queries, enc the keys and values
    dh1 = ds2 + dh1_c                                     # the residual and the cross-attention both read h1
    ds1, g["g1"], g["b1"] = layer_norm_backward(dh1, P["g1"], n1)  # first norm
    g["so"] = f(Z).T @ f(ds1)                             # self-attention's output projection
    dZh = split(ds1 @ P["so"].T, h)                       # back into the heads
    dA = np.einsum("mid,mjd->mij", dZh, V); dV = np.einsum("mij,mid->mjd", A, dZh)  # Z = A V
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True)) / np.sqrt(Q.shape[-1])  # softmax and scaling
    dQ, dK = merge(np.einsum("mij,mjd->mid", dS, K), h), merge(np.einsum("mij,mid->mjd", dS, Q), h)  # S = Q K^T
    dV = merge(dV, h)                                     # heads side by side
    g["sq"], g["sk"], g["sv"] = f(y).T @ f(dQ), f(y).T @ f(dK), f(y).T @ f(dV)  # the three projections
    dy = ds1 + dQ @ P["sq"].T + dK @ P["sk"].T + dV @ P["sv"].T  # the residual and y's three roles
    return dy, denc, g                                    # input, encoder output, weights


def transformer_backward(logp, y, tmask, P, h, cache):  # NEW: every gradient of the mean cross-entropy over real target positions
    src, tgt_in, x, ec, enc, yin, dc, dec = cache         # what transformer_forward computed
    m, T, Vn = logp.shape                                 # batch, target length, tokens
    dlog = np.exp(logp); np.put_along_axis(dlog, y[..., None], np.take_along_axis(dlog, y[..., None], -1) - 1, -1)  # softmax - one-hot
    dlog *= tmask[..., None] / tmask.sum()                # real positions only, averaged
    G = dict(Wout=dec.reshape(-1, dec.shape[-1]).T @ dlog.reshape(-1, Vn), bout=dlog.reshape(-1, Vn).sum(axis=0))  # output layer
    dy, denc, G["dec"] = decoder_block_backward(dlog @ P["Wout"].T, P["dec"], h, dc)  # today's decoder backward
    dx, G["enc"] = encoder_block_backward(denc, P["enc"], h, ec)  # Day 78's encoder backward
    G["E"] = np.zeros_like(P["E"])                        # one table for both sides
    np.add.at(G["E"], tgt_in, dy); np.add.at(G["E"], src, dx)  # the two contributions add (positions get no gradient)
    return G                                              # nested like P


def loss_of(P, src, tgt_in, y, h, smask, tmask):        # mean cross-entropy over the real target positions
    logp, _ = transformer_forward(src, tgt_in, P, h, smask, tmask)  # forward
    return -(np.take_along_axis(logp, y[..., None], axis=-1)[..., 0] * tmask).sum() / tmask.sum()  # padding counts 0


def leaves(D, prefix=""):                               # every array inside nested dictionaries, with a dotted name
    for k, v in D.items():                                # each entry
        if isinstance(v, dict):                             # a nested part (enc, dec, cross)
            yield from leaves(v, prefix + k + ".")            # its arrays
        else:
            yield prefix + k, v                               # an array


rng = np.random.RandomState(0)                          # fixed rng
V, d, ff, h = 10, 8, 12, 2                                # tiny: 10 tokens, size 8, FFN 12, 2 heads
P = init_params(rng, V, d, ff)                            # Day 81 exercise's weights
for _, A in leaves(P):                                    # move every weight a little, so g = 1, b = 0 and zeros are not special
    A += 0.05 * rng.randn(*A.shape)                         # in place: the array inside P changes
src = rng.randint(1, 9, size=(3, 5)); tgt_in = np.concatenate([np.full((3, 1), 9), rng.randint(1, 9, size=(3, 4))], axis=1)  # GO + 4
y = rng.randint(0, V, size=(3, 5))                        # targets
smask = np.ones((3, 5), bool); smask[1, 3:] = False       # padded source positions
tmask = np.ones((3, 5), bool); tmask[2, 4:] = False       # a padded target position

logp, cache = transformer_forward(src, tgt_in, P, h, smask, tmask)  # forward
G = dict(leaves(transformer_backward(logp, y, tmask, P, h, cache)))  # analytic gradients, flat
worst, n = 0.0, 0                                         # largest |numeric - analytic|, entries checked
for name, A in leaves(P):                                 # every weight array
    for pos in np.ndindex(A.shape):                         # every entry
        old = A[pos]                                          # remember it
        A[pos] = old + 1e-5; up = loss_of(P, src, tgt_in, y, h, smask, tmask)  # nudge up
        A[pos] = old - 1e-5; dn = loss_of(P, src, tgt_in, y, h, smask, tmask)  # nudge down
        A[pos] = old                                          # restore
        worst = max(worst, abs((up - dn) / 2e-5 - G[name][pos])); n += 1  # largest difference so far
print(f"gradient check, whole model ({n} entries, padded source and target): largest |numeric - analytic| = {worst:.2e}")
print(f"gradient on the embedding rows of tokens never used: {np.abs(G['E'][[t for t in range(V) if t not in src and t not in tgt_in]]).max():.1e}")
