"""
Day 81 Exercise -- The Whole Transformer, Forward, From the Week's Parts

The fifth of Week 12's parts for Day 83: one encoder block and one decoder
block joined into a complete forward pass, built only from the earlier
exercise scripts:

    Day 77  sinusoidal_encoding            positions, added to the embeddings
    Day 78  attention, split, merge,        the encoder block (and the heads of the
            encoder_block                   decoder's self-attention)
    Day 79  layer_norm                      every add & norm
    Day 80  causal_mask, cross_attention    the decoder's two attentions

    source -> E + PE -> encoder_block --------------+
    GO + target so far -> E + PE -> decoder_block <-+ (cross-attention) -> scores -> log-softmax

New today: decoder_block (the three parts with add & norm) and
transformer_forward. Checks: the scores are a probability distribution,
the untrained loss is close to log 10 (with a small output layer, Day 81's
lesson), a later decoder input does not move earlier scores, and padded
source positions are invisible.
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


def layer_norm(x, g, b, eps=1e-5):                      # NEW: x (..., d) -> same shape; statistics over the last axis
    sd = np.sqrt(x.var(axis=-1, keepdims=True) + eps)     # each position's spread
    xhat = (x - x.mean(axis=-1, keepdims=True)) / sd      # mean 0, spread 1 per position
    return xhat * g + b, (xhat, sd)                       # output and cache


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


rng = np.random.RandomState(0)                          # fixed rng
V, d, ff, h = 10, 16, 32, 2                               # 10 tokens (Day 69's digits, EOS and GO), size 16, FFN 32, 2 heads
P = init_params(rng, V, d, ff)                            # untrained weights
src = rng.randint(1, 9, size=(3, 6))                      # 3 source sequences of 6 digits
tgt_in = np.concatenate([np.full((3, 1), 9), src], axis=1)  # GO + the digits (COPY, teacher forcing)
y = np.concatenate([src, np.zeros((3, 1), int)], axis=1)  # the digits + EOS
smask, tmask = np.ones((3, 6), bool), np.ones((3, 7), bool)  # no padding yet

logp, _ = transformer_forward(src, tgt_in, P, h, smask, tmask)  # forward
loss = -np.take_along_axis(logp, y[..., None], axis=-1).mean()  # mean cross-entropy
print(f"scores: {logp.shape}; untrained loss {loss:.3f} (log {V} = {np.log(V):.3f}); probabilities sum to {np.exp(logp).sum(axis=-1).min():.6f}..{np.exp(logp).sum(axis=-1).max():.6f}")

t2 = tgt_in.copy(); t2[:, 4] = (t2[:, 4] % 8) + 1          # change decoder input 4
d4 = np.abs(transformer_forward(src, t2, P, h, smask, tmask)[0] - logp)  # how every score moved
print(f"change decoder input 4: positions 0-3 move by {d4[:, :4].max():.1e}, positions 4-6 by up to {d4[:, 4:].max():.2f}")

sm = smask.copy(); sm[:, 4:] = False                      # pretend the last 2 source positions are padding
base, _ = transformer_forward(src, tgt_in, P, h, sm, tmask)  # with that padding
s2 = src.copy(); s2[:, 4:] = rng.randint(1, 9, size=(3, 2))  # new digits under the padding
print(f"change padded source positions: largest change in any score = "
      f"{np.abs(transformer_forward(s2, tgt_in, P, h, sm, tmask)[0] - base).max():.1e}")
