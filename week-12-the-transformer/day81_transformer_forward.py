"""
Day 81: The Whole Transformer -- One Forward Pass on a Toy Copy Task

Days 77-80 built the parts: positional encoding, the encoder block, layer
normalisation, the decoder block with its causal mask and cross-attention.
Today they are put together into the model of Vaswani et al. (2017):

    source digits -> embedding + positions -> N encoder blocks ----------------+
                                                                                 | keys and values
    GO + target so far -> embedding + positions -> N decoder blocks <--- cross-attention
                                                       -> linear -> softmax -> next token

and run FORWARD only, untrained, on Day 69's COPY and REVERSE digit task.
Training needs the backward pass through all of it (Day 82); the model is
trained on Day 83. Today checks that the assembled model behaves as its
parts promise before any weight is learned:

  1. One example through the whole model, with the shape at every stage.
  2. Where the weights are.
  3. The masks, end to end: source padding is invisible, later target
     tokens are invisible, real source digits reach every output.
  4. Before training: the loss against log V, for three embedding spreads.
  5. Greedy decoding one token at a time, against one teacher-forced pass.
  6. What a forward pass costs as sequences grow.

Days 75-80's code is reused verbatim. Every line of code below carries its
own comment, continuing Day 52's convention.
"""
import time                                            # time: perf_counter, to time forward passes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

EOS, GO = 0, 9                                          # Day 69: token 0 = end of sequence, 9 = "start writing"; digits are 1..8
V = 10                                                  # vocabulary: EOS, eight digits, GO
SEEDS = [0, 1, 2]                                       # three seeds where a result could depend on the draw
D, F, HEADS = 32, 64, 4                                 # model size, feed-forward size, heads (small: a 10-token task)
MAX_LEN = 128                                           # positions the encoding is built for


# =============================================================================
# 0. REUSED -- Days 75-80, verbatim
# =============================================================================
def softmax(z):                                         # Days 57-77: weights along the last axis, summing to 1
    e = np.exp(z - z.max(axis=-1, keepdims=True))         # subtract the row max for stability, exponentiate
    return e / e.sum(axis=-1, keepdims=True)              # normalise


def scaled_dot_product_attention(Q, K, V, mask=None):  # day75_exercise.py, verbatim (via Days 76-77)
    d = Q.shape[-1]                                       # the query and key size
    S = np.einsum("mid,mjd->mij", Q, K) / np.sqrt(d)      # scaled scores
    if mask is not None:                                  # Day 70's masking, if there are padded keys
        S = np.where(mask, S, -np.inf)                      # masked keys get minus infinity ...
    e = np.exp(S - S.max(axis=-1, keepdims=True))         # ... and so weight exactly 0
    A = e / e.sum(axis=-1, keepdims=True)                 # weights over the keys
    return np.einsum("mij,mjd->mid", A, V), A             # outputs, and the weights (the backward pass needs them)


def scaled_dot_product_attention_backward(dZ, Q, K, V, A):  # day75_exercise.py, verbatim (via Days 76-77)
    d = Q.shape[-1]                                       # the query and key size
    dA = np.einsum("mid,mjd->mij", dZ, V)                 # Z = A V
    dV = np.einsum("mij,mid->mjd", A, dZ)                 # ... into the values
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True)) / np.sqrt(d)  # back through the softmax and the scaling
    return np.einsum("mij,mjd->mid", dS, K), np.einsum("mij,mid->mjd", dS, Q), dV  # dQ, dK, dV


def split_heads(T, h):                                  # Day 76: (m, n, d) -> (m * h, n, d / h)
    m, n, d = T.shape                                     # batch, positions, size
    return T.reshape(m, n, h, d // h).transpose(0, 2, 1, 3).reshape(m * h, n, d // h)  # each head becomes a batch entry


def merge_heads(T, h):                                  # Day 76: (m * h, n, d / h) -> (m, n, d)
    mh, n, dk = T.shape                                   # batch times heads, positions, head size
    return T.reshape(mh // h, h, n, dk).transpose(0, 2, 1, 3).reshape(mh // h, n, h * dk)  # the heads side by side again


def sinusoidal_encoding(n, d):                          # day77_exercise.py: (n, d) table, row p = position p
    p = np.arange(n)[:, None]                             # positions as a column
    i = np.arange(d // 2)[None, :]                        # pair index as a row
    angle = p / 10000 ** (2 * i / d)                      # one angle per (position, pair)
    PE = np.zeros((n, d))                                 # the table
    PE[:, 0::2], PE[:, 1::2] = np.sin(angle), np.cos(angle)  # sine in even columns, cosine in odd ones
    return PE                                             # hand back


# =============================================================================
# 1. THE BLOCKS -- Day 78's encoder block and Day 80's decoder block, verbatim
# =============================================================================
def layer_norm(x, g, b, eps=1e-5):                      # NEW: each position's vector to mean 0, spread 1, then scale g and shift b
    mu = x.mean(axis=-1, keepdims=True)                   # the mean of each vector's d numbers
    sd = np.sqrt(x.var(axis=-1, keepdims=True) + eps)     # their spread (eps keeps it above 0)
    xhat = (x - mu) / sd                                  # normalised: mean 0, spread 1
    return xhat * g + b, (xhat, sd)                       # learned scale and shift; the cache


def layer_norm_backward(dy, g, cache):                  # NEW: gradients for x, g and b
    xhat, sd = cache                                      # what the forward pass computed
    dxhat = dy * g                                        # through the scale
    dx = (dxhat - dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)) / sd  # through mean and spread
    flat = lambda T: T.reshape(-1, T.shape[-1])           # every position as one row
    return dx, flat(dy * xhat).sum(axis=0), flat(dy).sum(axis=0)  # dx, dg, db


def ffn(x, P):                                          # NEW: the same two-layer network applied to every position on its own
    u = x @ P["W1"] + P["c1"]                             # widen to F numbers
    r = np.maximum(u, 0.0)                                # ReLU
    return r @ P["W2"] + P["c2"], (x, u, r)               # back to D numbers; the cache


def ffn_backward(dy, P, cache):                         # NEW: gradients for x, W1, c1, W2, c2
    x, u, r = cache                                       # what the forward pass computed
    flat = lambda T: T.reshape(-1, T.shape[-1])           # every position as one row
    du = (dy @ P["W2"].T) * (u > 0)                       # back through W2, then the ReLU
    g = dict(W2=flat(r).T @ flat(dy), c2=flat(dy).sum(axis=0), W1=flat(x).T @ flat(du), c1=flat(du).sum(axis=0))  # the FFN's four weight gradients
    return du @ P["W1"].T, g                              # dx and the weight gradients


class EncoderBlock:                                     # NEW: attention and FFN, each with a residual and a LayerNorm
    def __init__(self, rng, pre_ln=False, residual=True, norm=True, use_ffn=True):
        d, f, heads = D, F, HEADS                          # the sizes in force when the block is made (Section 3 shrinks them)
        self.h, self.pre, self.res, self.norm, self.use_ffn = heads, pre_ln, residual, norm, use_ffn  # settings
        s = lambda a, b: rng.randn(a, b) / np.sqrt(a)      # a weight matrix with Day 74's scale
        self.P = dict(Wq=s(d, d), Wk=s(d, d), Wv=s(d, d), Wo=s(d, d),  # Day 76's attention
                      g1=np.ones(d), b1=np.zeros(d),       # the first LayerNorm (starts as "no change")
                      W1=s(d, f), c1=np.zeros(f), W2=s(f, d), c2=np.zeros(d),  # the feed-forward network
                      g2=np.ones(d), b2=np.zeros(d))       # the second LayerNorm

    def _norm(self, x, which):                             # a LayerNorm, or nothing for the "no norm" ablation
        if not self.norm:                                    # ablation: leave the vectors as they are
            return x, None                                     # no cache needed
        return layer_norm(x, self.P["g" + which], self.P["b" + which])  # normalise

    def _norm_back(self, dy, which, cache, g):             # the backward pass of _norm
        if not self.norm:                                    # nothing was done
            return dy                                          # the gradient passes straight through
        dx, g["g" + which], g["b" + which] = layer_norm_backward(dy, self.P["g" + which], cache)  # LayerNorm's backward
        return dx                                          # gradient at its input

    def forward(self, x, pe, kmask):                       # x (m, T, d) -> (m, T, d); pe is added to queries and keys (Day 77)
        P, h, c = self.P, self.h, {}                         # parameters, heads, cache
        a_in, c["n1"] = self._norm(x, "1") if self.pre else (x, None)  # pre-LN: normalise before attention
        Xqk = a_in + pe                                      # Day 77's "qk": positions only where the layer decides where to look
        Q, K, V = split_heads(Xqk @ P["Wq"], h), split_heads(Xqk @ P["Wk"], h), split_heads(a_in @ P["Wv"], h)  # Day 76
        Zh, A = scaled_dot_product_attention(Q, K, V, kmask)  # Day 75's function, every head
        Z = merge_heads(Zh, h)                               # heads side by side
        s1 = Z @ P["Wo"] + (x if self.res else 0.0)          # the residual: add the block's input back
        h1, n1 = (s1, None) if self.pre else self._norm(s1, "1")  # post-LN: normalise after adding
        c["n1"] = c["n1"] if self.pre else n1                # the first norm's cache, whichever place it was
        c.update(a_in=a_in, Xqk=Xqk, Q=Q, K=K, V=V, A=A, Z=Z)  # attention's cache
        if not self.use_ffn:                                 # ablation: no feed-forward part
            self.c = c                                         # keep the cache
            return h1                                          # the block ends here
        f_in, c["n2"] = self._norm(h1, "2") if self.pre else (h1, None)  # pre-LN: normalise before the FFN
        fo, c["ffn"] = ffn(f_in, P)                          # the feed-forward network
        s2 = fo + (h1 if self.res else 0.0)                  # the second residual
        y, n2 = (s2, None) if self.pre else self._norm(s2, "2")  # post-LN: normalise after adding
        c["n2"] = c["n2"] if self.pre else n2                # the second norm's cache
        self.c = c                                           # keep the cache
        return y                                          # the block's output

    def backward(self, dy):                                # gradient at the output -> gradient at the input, and every weight's
        P, h, c = self.P, self.h, self.c                     # parameters, heads, cache
        g = {k: np.zeros_like(v) for k, v in P.items()}      # unused parts (an ablation) keep zero gradients
        flat = lambda T: T.reshape(-1, T.shape[-1])          # every position as one row
        if self.use_ffn:                                     # back through the feed-forward half
            ds2 = dy if self.pre else self._norm_back(dy, "2", c["n2"], g)  # post-LN: through the second norm first
            df_in, gf = ffn_backward(ds2, P, c["ffn"])         # through the FFN
            g.update(gf)                                       # its weights
            if self.pre:                                       # pre-LN: the FFN's input was normalised
                df_in = self._norm_back(df_in, "2", c["n2"], g)  # through the second norm
            dh1 = df_in + (ds2 if self.res else 0.0)           # the residual: the gradient also skips the FFN
        else:
            dh1 = dy                                           # no FFN: the gradient arrives at h1 directly
        ds1 = dh1 if self.pre else self._norm_back(dh1, "1", c["n1"], g)  # post-LN: through the first norm
        g["Wo"] = flat(c["Z"]).T @ flat(ds1)                 # output projection
        dQ, dK, dV = scaled_dot_product_attention_backward(split_heads(ds1 @ P["Wo"].T, h), c["Q"], c["K"], c["V"], c["A"])  # Day 75's backward
        dQ, dK, dV = merge_heads(dQ, h), merge_heads(dK, h), merge_heads(dV, h)  # heads side by side
        g["Wq"], g["Wk"], g["Wv"] = flat(c["Xqk"]).T @ flat(dQ), flat(c["Xqk"]).T @ flat(dK), flat(c["a_in"]).T @ flat(dV)  # projections
        da_in = dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T  # attention's input made all three
        if self.pre:                                         # pre-LN: attention's input was normalised
            da_in = self._norm_back(da_in, "1", c["n1"], g)    # through the first norm
        return da_in + (ds1 if self.res else 0.0), g      # the residual: the gradient also skips attention


def causal_mask(n):                                     # NEW: (n, n), True where query i may read key j, i.e. j <= i
    return np.tril(np.ones((n, n), dtype=bool))           # the lower triangle, diagonal included


def attend(Xq, Xk, Xv, P, prefix, h, mask):             # NEW: Day 77's attention with three separate inputs
    Q = split_heads(Xq @ P[prefix + "q"], h)              # queries: from the decoder
    K = split_heads(Xk @ P[prefix + "k"], h)              # keys: from the decoder (self) or the encoder (cross)
    V = split_heads(Xv @ P[prefix + "v"], h)              # values: likewise
    Zh, A = scaled_dot_product_attention(Q, K, V, mask)   # Day 75's function, every head
    Z = merge_heads(Zh, h)                                # heads side by side
    return Z @ P[prefix + "o"], dict(Q=Q, K=K, V=V, A=A, Z=Z, Xq=Xq, Xk=Xk, Xv=Xv)  # output projection, cache


def attend_backward(dO, P, prefix, h, c):               # NEW: gradients for the three inputs and the four weights
    flat = lambda T: T.reshape(-1, T.shape[-1])           # every position as one row
    g = {prefix + "o": flat(c["Z"]).T @ flat(dO)}         # output projection
    dQ, dK, dV = scaled_dot_product_attention_backward(split_heads(dO @ P[prefix + "o"].T, h), c["Q"], c["K"], c["V"], c["A"])  # Day 75's backward, every head
    dQ, dK, dV = merge_heads(dQ, h), merge_heads(dK, h), merge_heads(dV, h)  # heads side by side
    g[prefix + "q"], g[prefix + "k"], g[prefix + "v"] = flat(c["Xq"]).T @ flat(dQ), flat(c["Xk"]).T @ flat(dK), flat(c["Xv"]).T @ flat(dV)  # each projection's own input
    return dQ @ P[prefix + "q"].T, dK @ P[prefix + "k"].T, dV @ P[prefix + "v"].T, g  # dXq, dXk, dXv, weights


class DecoderBlock:                                     # NEW: post-LN decoder block (Vaswani et al. 2017)
    def __init__(self, rng, causal=True, cross=True):
        self.h, self.causal, self.cross = HEADS, causal, cross  # settings: the mask, and whether there is an encoder
        s = lambda a, b: rng.randn(a, b) / np.sqrt(a)      # a weight matrix with Day 74's scale
        self.P = dict(sq=s(D, D), sk=s(D, D), sv=s(D, D), so=s(D, D), g1=np.ones(D), b1=np.zeros(D))  # self-attention, norm 1
        if cross:                                          # attention to the encoder's output
            self.P.update(cq=s(D, D), ck=s(D, D), cv=s(D, D), co=s(D, D), g2=np.ones(D), b2=np.zeros(D))  # cross-attention, norm 2
        self.P.update(W1=s(D, F), c1=np.zeros(F), W2=s(F, D), c2=np.zeros(D), g3=np.ones(D), b3=np.zeros(D))  # FFN, norm 3

    def forward(self, y, pe, ymask, enc=None, emask=None):  # y (m, n, d) decoder inputs; enc (m, ne, d) encoder output
        P, h, c = self.P, self.h, {}                         # parameters, heads, cache
        m, n = ymask.shape                                   # batch, decoder length
        allowed = np.broadcast_to(ymask[:, None, :], (m, n, n))  # query i may read every real key ...
        if self.causal:                                      # ... except, with the causal mask, the ones after it
            allowed = allowed & causal_mask(n)[None]  # future keys out
        sm = np.repeat(allowed[:, None], h, axis=1).reshape(m * h, n, n)  # per head
        Xqk = y + pe                                         # Day 77's "qk": positions in queries and keys only
        a, c["self"] = attend(Xqk, Xqk, y, P, "s", h, sm)    # masked self-attention
        h1, c["n1"] = layer_norm(y + a, P["g1"], P["b1"])    # add & norm
        if self.cross:                                       # every decoder position reads the whole encoder output
            cm = np.repeat(np.broadcast_to(emask[:, None, None, :], (m, 1, n, emask.shape[1])), h, axis=1).reshape(m * h, n, -1)  # padded encoder positions out, per head
            ca, c["cross"] = attend(h1, enc, enc, P, "c", h, cm)  # queries from the decoder, keys and values from the encoder
            h2, c["n2"] = layer_norm(h1 + ca, P["g2"], P["b2"])  # add & norm
        else:
            h2 = h1                                            # decoder-only: no encoder to read
        f, c["ffn"] = ffn(h2, dict(W1=P["W1"], c1=P["c1"], W2=P["W2"], c2=P["c2"]))  # Day 78's FFN
        out, c["n3"] = layer_norm(h2 + f, P["g3"], P["b3"])  # add & norm
        self.c = c                                           # keep the cache
        return out                                        # the block's output

    def backward(self, dout):                              # gradients for y, the encoder output, and every weight
        P, h, c = self.P, self.h, self.c                     # parameters, heads, cache
        g = {}                                               # weight gradients
        ds3, g["g3"], g["b3"] = layer_norm_backward(dout, P["g3"], c["n3"])  # third norm
        dh2, gf = ffn_backward(ds3, dict(W1=P["W1"], c1=P["c1"], W2=P["W2"], c2=P["c2"]), c["ffn"])  # FFN
        g.update(gf); dh2 = dh2 + ds3                        # the residual
        denc = None                                          # no encoder: no encoder gradient
        if self.cross:                                       # back through cross-attention
            ds2, g["g2"], g["b2"] = layer_norm_backward(dh2, P["g2"], c["n2"])  # second norm
            dq, dk, dv, gc = attend_backward(ds2, P, "c", h, c["cross"])  # cross-attention
            g.update(gc); dh1 = ds2 + dq; denc = dk + dv       # the residual; the encoder made keys and values
        else:
            dh1 = dh2                                          # decoder-only
        ds1, g["g1"], g["b1"] = layer_norm_backward(dh1, P["g1"], c["n1"])  # first norm
        dq, dk, dv, gs = attend_backward(ds1, P, "s", h, c["self"])  # self-attention
        g.update(gs)                                         # its weights
        return ds1 + dq + dk + dv, denc, g                # the residual and all three of y's roles; the encoder gradient


# =============================================================================
# 2. THE TOY TASK -- Day 69's data, verbatim
# =============================================================================
def make_batch(rng, m, n, task):                        # Day 69's data, verbatim: m sequences of n digits
    src = rng.randint(1, 9, size=(m, n))                  # random digits 1..8
    tgt = src if task == "copy" else src[:, ::-1]         # COPY: the same digits; REVERSE: backwards
    dec_in = np.concatenate([np.full((m, 1), GO), tgt], axis=1)    # decoder input: GO, then the target shifted right
    dec_out = np.concatenate([tgt, np.full((m, 1), EOS)], axis=1)  # decoder output: the target, then EOS
    return np.eye(V)[src], np.eye(V)[dec_in], dec_out, src  # one-hot source, one-hot decoder input, targets, raw digits


def fixed_test(n, task, m=300):                         # Day 70's fixed test sets: the same sequences as Days 70-73 used
    return make_batch(np.random.RandomState(12345 + n), m, n, task)  # a seed per length


def as_ids(batch):                                      # NEW: the Transformer reads token ids, not one-hot rows
    Xs, Xd, Y, src = batch                                # Day 69's four arrays
    return src, Xd.argmax(axis=-1), Y                     # source ids, decoder-input ids, targets


# =============================================================================
# 3. THE WHOLE TRANSFORMER -- NEW: the week's blocks put together
# =============================================================================
def head_mask(mask, h):                                 # NEW: (m, n) real-position mask -> (m * h, 1, n) key mask per head (Day 77)
    return np.repeat(mask[:, None, None, :], h, axis=1).reshape(mask.shape[0] * h, 1, mask.shape[1])  # one copy per head, heads as batch entries


class Transformer:                                      # NEW: source -> encoder blocks; target so far -> decoder blocks reading the encoder
    def __init__(self, n_enc=1, n_dec=1, seed=0, embed_scale=1.0, out_scale=1.0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.P = dict(E=rng.randn(V, D) * embed_scale)     # ONE embedding table: source and target are the same digits
        self.enc = [EncoderBlock(rng) for _ in range(n_enc)]  # Day 78's block, post-LN
        self.dec = [DecoderBlock(rng, causal=True, cross=True) for _ in range(n_dec)]  # Day 80's block
        self.P.update(Wout=rng.randn(D, V) / np.sqrt(D) * out_scale, bout=np.zeros(V))  # a score for every token (Section 4 shrinks it)
        self.PE = sinusoidal_encoding(MAX_LEN, D)          # Day 77's table, added to the embeddings (the paper's place)

    def params(self):                                      # every trainable array, by name
        out = dict(self.P)                                   # embeddings and output layer
        for i, b in enumerate(self.enc):                     # encoder blocks: "e0.Wq", ...
            out.update({f"e{i}.{k}": v for k, v in b.P.items()})  # prefixed with the block's place
        for i, b in enumerate(self.dec):                     # decoder blocks: "d0.sq", ...
            out.update({f"d{i}.{k}": v for k, v in b.P.items()})  # prefixed with the block's place
        return out                                        # the same arrays, not copies

    def encode(self, src, smask):                          # (m, n) source ids -> (m, n, D) encoder output
        x = self.P["E"][src] + self.PE[:src.shape[1]]        # embeddings plus positions
        km = head_mask(smask, HEADS)                         # padded source positions, per head
        for b in self.enc:                                   # through the encoder stack
            x = b.forward(x, 0.0, km)                          # Day 78's block; positions were added once, at the input
        return x                                          # every source position, in context

    def decode(self, tgt_in, tmask, enc, smask):           # (m, T) target-so-far ids -> (m, T, D)
        y = self.P["E"][tgt_in] + self.PE[:tgt_in.shape[1]]  # embeddings plus positions
        for b in self.dec:                                   # through the decoder stack
            y = b.forward(y, 0.0, tmask, enc, smask)           # Day 80's block: causal self-attention, cross-attention, FFN
        return y                                          # every target position

    def forward(self, src, smask, tgt_in, tmask):          # the whole model: log-probabilities of the next token, (m * T, V)
        self.enc_out = self.encode(src, smask)               # read the source
        self.dec_out = self.decode(tgt_in, tmask, self.enc_out, smask)  # write, reading the source
        logits = self.dec_out.reshape(-1, D) @ self.P["Wout"] + self.P["bout"]  # a score per token
        logp = logits - logits.max(axis=1, keepdims=True)    # log-softmax (Day 78)
        return logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))  # ... minus log of the summed exponentials

    def loss(self, src, smask, tgt_in, tmask, y):          # mean cross-entropy over the real target positions
        logp = self.forward(src, smask, tgt_in, tmask)[np.arange(y.size), y.ravel()]  # log-probability of each true token
        return -np.sum(logp * tmask.ravel()) / tmask.sum()  # padding counts 0

    def generate(self, src, n_steps):                      # greedy decoding: each new token is fed back in
        m = len(src); smask = np.ones(src.shape, bool)       # no source padding in the toy task
        enc = self.encode(src, smask)                        # the source is read once
        tokens = np.full((m, 1), GO)                         # every output starts with GO
        steps = []                                           # each step's log-probabilities
        for _ in range(n_steps):                             # one token per step
            y = self.decode(tokens, np.ones(tokens.shape, bool), enc, smask)  # the decoder on everything written so far
            logits = y[:, -1] @ self.P["Wout"] + self.P["bout"]  # only the last position predicts the next token
            lp = logits - logits.max(axis=1, keepdims=True); lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))  # log-softmax, as in forward()
            steps.append(lp)                                   # keep it (Section 5)
            tokens = np.concatenate([tokens, lp.argmax(axis=1)[:, None]], axis=1)  # the most probable token is written
        return tokens[:, 1:], np.stack(steps, axis=1)      # the tokens (GO dropped), and (m, n_steps, V)


# =============================================================================
# DEMOS
# =============================================================================
def demo_trace():
    print("=" * 70)
    print("1. ONE EXAMPLE THROUGH THE WHOLE MODEL -- shapes at every stage")
    print("=" * 70)
    print(f"One untrained Transformer (seed 0): {D} numbers per token, {HEADS} heads, FFN {F}, one encoder block and "
          "one decoder block. One REVERSE example of 5 digits; the decoder is given GO plus the right answer, shifted "
          "(teacher forcing).")
    src, tgt_in, y = as_ids(make_batch(np.random.RandomState(1), 1, 5, "reverse"))  # one example
    net = Transformer(seed=0)                             # untrained
    smask, tmask = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
    print(f"\nsource ids {src[0].tolist()}   decoder input {tgt_in[0].tolist()}   target {y[0].tolist()}")
    x = net.P["E"][src] + net.PE[:5]                      # stage by stage, for the printout
    print(f"source embeddings + positions   {x.shape}")
    enc = net.encode(src, smask)                          # the encoder
    print(f"encoder output                  {enc.shape}")
    print(f"decoder embeddings + positions  {(net.P['E'][tgt_in] + net.PE[:6]).shape}")
    dec = net.decode(tgt_in, tmask, enc, smask)           # the decoder
    print(f"decoder output                  {dec.shape}")
    print(f"cross-attention weights, per head (decoder positions x source positions): {net.dec[0].c['cross']['A'].shape[1:]}")
    lp = net.forward(src, smask, tgt_in, tmask)           # the whole model
    print(f"scores for the next token       {lp.shape}  (6 positions x {V} tokens)")
    print(f"untrained prediction at each position: {lp.argmax(axis=1).tolist()}")
    print("Actual output from this lesson's run.")


def demo_params():
    print("\n" + "=" * 70)
    print("2. WHERE THE WEIGHTS ARE")
    print("=" * 70)
    print("Weight counts by part, for one block on each side and for two.")
    for n in (1, 2):                                      # two sizes
        net = Transformer(n_enc=n, n_dec=n)                 # fresh
        P = net.params()                                    # every array
        part = lambda pre: sum(v.size for k, v in P.items() if k.startswith(pre))  # total of the arrays whose name starts so
        emb, out = P["E"].size, P["Wout"].size + P["bout"].size  # embeddings, output layer
        enc, dec = part("e"), part("d")                     # the two stacks ("E" is upper case, so not counted here)
        print(f"\n{n} + {n} blocks: embeddings {emb}, encoder {enc}, decoder {dec}, output layer {out}; total {emb + enc + dec + out}")
        b = net.dec[0].P                                    # one decoder block, part by part
        sa = sum(b[k].size for k in ("sq", "sk", "sv", "so")); ca = sum(b[k].size for k in ("cq", "ck", "cv", "co"))  # the two attentions
        ff = sum(b[k].size for k in ("W1", "c1", "W2", "c2")); ln = sum(b[k].size for k in b if k[0] in "gb")  # FFN and the three norms
        if n == 1:                                           # the breakdown once is enough
            print(f"  one decoder block: self-attention {sa}, cross-attention {ca}, FFN {ff}, LayerNorms {ln}")
    print("Actual output from this lesson's run.")


def demo_masks():
    print("\n" + "=" * 70)
    print("3. THE MASKS, END TO END")
    print("=" * 70)
    print("Untrained model (seed 0), 4 COPY examples of 8 digits. Three changes, and the largest change in the "
          "scores they cause where nothing should (or something should) change.")
    src, tgt_in, y = as_ids(make_batch(np.random.RandomState(2), 4, 8, "copy"))  # 4 examples
    net = Transformer(seed=0)                             # untrained
    sm = np.ones(src.shape, bool); sm[:, 6:] = False       # pretend the last 2 source positions are padding
    tm = np.ones(tgt_in.shape, bool)                      # every target position real
    base = net.forward(src, sm, tgt_in, tm).reshape(4, 9, V)  # the reference scores
    s2 = src.copy(); s2[:, 6:] = np.random.RandomState(3).randint(1, 9, size=(4, 2))  # (a) new digits under the padding
    a = np.abs(net.forward(s2, sm, tgt_in, tm).reshape(4, 9, V) - base).max()  # largest change anywhere
    t2 = tgt_in.copy(); t2[:, 5] = (t2[:, 5] % 8) + 1      # (b) a different decoder input at position 5
    d = np.abs(net.forward(src, sm, t2, tm).reshape(4, 9, V) - base)  # change at every position
    s3 = src.copy(); s3[:, 2] = (s3[:, 2] % 8) + 1         # (c) a different real source digit at position 2
    c = np.abs(net.forward(s3, sm, tgt_in, tm).reshape(4, 9, V) - base)  # change at every position
    print(f"\n(a) change the PADDED source positions:          every score changes by at most {a:.2e}")
    print(f"(b) change decoder input 5: scores at positions 0-4 change by {d[:, :5].max():.2e}, at 5-8 by up to {d[:, 5:].max():.2e}")
    print(f"(c) change a REAL source digit: scores change at every decoder position, smallest change {c.max(axis=2).min():.2e}")
    print("Actual output from this lesson's run.")


def demo_initial_loss():
    print("\n" + "=" * 70)
    print("4. BEFORE ANY TRAINING -- the loss against log V")
    print("=" * 70)
    print(f"A model that knows nothing should spread its bets evenly over the {V} tokens: loss log {V} = "
          f"{np.log(V):.3f}. The untrained loss on the 300 fixed 8-digit COPY test sequences, three seeds, for three "
          "spreads of the embedding table, and then with the output layer's starting weights 10 times smaller.")
    src, tgt_in, y = as_ids(fixed_test(8, "copy"))        # fixed test sequences
    sm, tm = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
    print(f"\n{'model':<26}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'largest probability (mean)':>29}")
    res = {}                                              # label -> losses
    for label, kw in [("embedding spread 0.1", dict(embed_scale=0.1)), ("embedding spread 1", {}),  # four versions of the untrained model
                      ("embedding spread 3", dict(embed_scale=3.0)), ("output layer x 0.1", dict(out_scale=0.1))]:
        losses, top = [], []                                # per seed
        for s in SEEDS:                                     # three seeds
            net = Transformer(seed=s, **kw)                   # untrained
            losses.append(net.loss(src, sm, tgt_in, tm, y))   # mean cross-entropy
            top.append(np.exp(net.forward(src, sm, tgt_in, tm)).max(axis=1).mean())  # how confident it already is
        res[label] = losses                                 # for the plot
        print(f"{label:<26}" + "".join(f"{v:>9.3f}" for v in losses) + f"{np.mean(top):>29.3f}")
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_decoding():
    print("\n" + "=" * 70)
    print("5. GREEDY DECODING, AND WHY ONE PASS IS ENOUGH IN TRAINING")
    print("=" * 70)
    print("The untrained model writes 9 tokens for each of the 300 fixed 8-digit COPY sequences, one at a time, "
          "feeding each token back in. Then the tokens it wrote are given to the decoder all at once (GO plus "
          "the first 8), in ONE pass, as in training. The causal mask (Day 80) says the two must agree.")
    src, tgt_in, y = as_ids(fixed_test(8, "copy"))        # fixed test sequences
    net = Transformer(seed=0)                             # untrained
    tokens, steps = net.generate(src, 9)                  # one token at a time
    fed = np.concatenate([np.full((300, 1), GO), tokens[:, :8]], axis=1)  # what the decoder saw at each step
    one = net.forward(src, np.ones(src.shape, bool), fed, np.ones(fed.shape, bool)).reshape(300, 9, V)  # one pass
    print(f"\nlargest difference between the two: {np.abs(one - steps).max():.2e}; same tokens chosen: {(one.argmax(axis=2) == tokens).all()}")
    vals, counts = np.unique(tokens, return_counts=True)  # what the untrained model writes
    print(f"untrained model: exact match {np.all(tokens == y, axis=1).mean():.3f}, token accuracy {(tokens == y).mean():.3f}; "
          f"its most written token is {vals[counts.argmax()]} ({counts.max() / tokens.size:.3f} of all tokens)")
    print("Actual output from this lesson's run.")


def demo_cost():
    print("\n" + "=" * 70)
    print("6. WHAT A FORWARD PASS COSTS AS SEQUENCES GROW")
    print("=" * 70)
    print("One forward pass (teacher forcing), batch of 32, source and target of n digits; best of 5 timings of 10 "
          "passes, matrix library on one thread. Attention compares every position with every position, so part of "
          "the work grows like n squared. Only ratios within this run are meaningful.")
    net = Transformer(seed=0)                             # untrained
    t = {}                                                # n -> seconds per pass
    for n in (8, 16, 32, 64):                             # four lengths
        src, tgt_in, y = as_ids(make_batch(np.random.RandomState(4), 32, n - 1, "copy"))  # n - 1 digits, so n decoder positions
        sm, tm = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
        best = np.inf                                       # fastest repeat
        for _ in range(5):                                  # 5 repeats
            start = time.perf_counter()                       # clock
            for _ in range(10):                               # 10 passes
                net.forward(src, sm, tgt_in, tm)                # the whole model
            best = min(best, (time.perf_counter() - start) / 10)  # per pass
        t[n] = best                                         # keep it
    print(f"\n{'n':>4}{'ms per pass':>13}{'vs n = 8':>10}")
    for n in t:                                           # each length
        print(f"{n:>4}{t[n] * 1000:>13.2f}{t[n] / t[8]:>10.2f}")
    print("(doubling n would double the cost if everything grew like n, and quadruple it if everything grew like n squared)")
    print("Actual output from this lesson's run.")
    return t                                               # the plot reuses it


def plot_results(res, t):                               # three panels: initial loss | cost | untrained cross-attention
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))          # canvas
    labels = list(res)                                      # panel 1: initial loss by model
    for i, lab in enumerate(labels):                        # each model, three seeds
        ax[0].scatter([i] * 3, res[lab], color="#d62728" if "output" in lab else "#1f77b4", s=35)
    ax[0].axhline(np.log(V), color="#7f7f7f", linestyle="--", label=f"log {V} (even bets)")
    ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(["emb. 0.1", "emb. 1", "emb. 3", "output\nx 0.1"], fontsize=8)
    ax[0].set_ylabel("untrained loss (three seeds)"); ax[0].set_title("Before training", fontsize=10); ax[0].legend(fontsize=8)
    ns = list(t)                                            # panel 2: cost
    ax[1].plot(ns, [t[n] / t[8] for n in ns], marker="o", color="#1f77b4", label="measured")
    ax[1].plot(ns, [n / 8 for n in ns], linestyle=":", color="#7f7f7f", label="like n")
    ax[1].plot(ns, [(n / 8) ** 2 for n in ns], linestyle="--", color="#7f7f7f", label="like n squared")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log", base=2); ax[1].set_xlabel("sequence length n")
    ax[1].set_ylabel("forward-pass time, relative to n = 8"); ax[1].set_title("Cost of one forward pass", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left"); ax[1].grid(alpha=0.3, which="both")
    src, tgt_in, y = as_ids(fixed_test(8, "reverse"))       # panel 3: untrained cross-attention on one REVERSE example
    net = Transformer(seed=0)  # untrained
    net.forward(src[:1], np.ones((1, 8), bool), tgt_in[:1], np.ones((1, 9), bool))  # fills the cross-attention cache
    A = net.dec[0].c["cross"]["A"].reshape(1, HEADS, 9, 8).mean(axis=1)[0]  # mean over heads
    im = ax[2].imshow(A, cmap="Blues", vmin=0, vmax=max(0.3, A.max()))
    ax[2].set_xlabel("source position"); ax[2].set_ylabel("decoder position"); fig.colorbar(im, ax=ax[2], fraction=0.04)
    ax[2].set_title("Untrained cross-attention (mean of heads)", fontsize=10)
    plt.tight_layout(); plt.savefig("transformer_forward_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved transformer_forward_results.png")


def draw_diagram():                                     # the whole Transformer, with the week's parts named
    fig, ax = plt.subplots(figsize=(13, 5.0))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.0); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333"):                    # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    ax.text(6.5, 4.75, "Day 81: the whole Transformer, one forward pass", ha="center", fontsize=12, fontweight="bold")
    box(1.3, 3.6, 2.0, 0.55, "source digits", "#F2F2F2")                      # encoder side
    box(1.3, 1.4, 2.0, 0.55, "GO + target so far", "#F2F2F2")                 # decoder side
    box(3.7, 3.6, 2.0, 0.6, "embedding + PE\n(Day 77)", "#FAEEDA", fs=8)
    box(3.7, 1.4, 2.0, 0.6, "embedding + PE\n(same table)", "#FAEEDA", fs=8)
    box(6.4, 3.6, 2.3, 0.75, "N encoder blocks\n(Day 78)", "#DCEBF7")
    box(6.4, 1.4, 2.3, 0.75, "N decoder blocks\n(Day 80)", "#F4EEFC")
    box(9.2, 1.4, 1.9, 0.6, "linear + softmax", "#E5F2E1", fs=8)
    box(11.6, 1.4, 1.9, 0.6, "next-token\nprobabilities", "#F2F2F2", fs=8)
    arrow(2.32, 3.6, 2.68, 3.6); arrow(4.72, 3.6, 5.23, 3.6)                  # source path
    arrow(2.32, 1.4, 2.68, 1.4); arrow(4.72, 1.4, 5.23, 1.4)                  # target path
    arrow(6.4, 3.2, 6.4, 1.8, color="#C97A16")                                # encoder output into cross-attention
    ax.text(6.55, 2.5, "encoder output: keys and values\nof every cross-attention", ha="left", fontsize=7.5, color="#C97A16")
    arrow(7.57, 1.4, 8.23, 1.4); arrow(10.17, 1.4, 10.63, 1.4)  # decoder -> output -> probabilities
    ax.text(6.5, 4.3, "causal mask in the decoder's self-attention; padding masks for the source in both stacks",
            ha="center", fontsize=8, style="italic")
    ax.annotate("", xy=(1.3, 1.05), xytext=(11.6, 1.05), arrowprops=dict(arrowstyle="->", color="#999999", linewidth=1.0,
                                                                        connectionstyle="arc3,rad=-0.12"))  # generation feeds back, under the boxes
    ax.text(6.5, 0.12, "generating: the chosen token is appended and fed back", ha="center", fontsize=7.5, color="#777777")
    plt.savefig("transformer_forward_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved transformer_forward_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Every block's backward pass already exists (Days 78 and 80). Day 82 chains them into one backward "
          "pass for the whole model and checks every gradient numerically; Day 83 trains it.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_trace()                                         # 1. one example, stage by stage
        demo_params()                                        # 2. where the weights are
        demo_masks()                                         # 3. the masks
        res = demo_initial_loss()                            # 4. before training
        demo_decoding()                                      # 5. greedy decoding
        t = demo_cost()                                      # 6. cost
        plot_results(res, t)                                 # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 7. Day 82


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
