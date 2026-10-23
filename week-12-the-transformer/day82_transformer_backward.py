"""
Day 82: The Backward Pass Through the Whole Transformer, and a Gradient Check

Day 81 ran the assembled Transformer forward. Training needs the gradient
of the loss with respect to every one of its weights. Each block's own
backward pass already exists (Day 78's encoder block, Day 80's decoder
block); today chains them, in reverse order, from the loss back to the
embedding table:

    loss -> output layer -> decoder blocks (last first) -> target embeddings
                               |  every decoder block's cross-attention sends a gradient
                               v  to the encoder output: they ADD UP
                            encoder blocks (last first) -> source embeddings
    source and target share ONE embedding table: its gradient is the sum of both

What is built and measured, reported as it came out:

  1. The order of the backward pass, with each hand-over point's size.
  2. A gradient check of the whole model -- two blocks per side, padded
     source and target -- every entry of every parameter.
  3. Would the check catch a bug? One deliberate mistake in the chain.
  4. Where the gradient goes, part by part, before training.
  5. Does the gradient teach? Fitting one fixed batch.
  6. What the backward pass costs against the forward pass.

Days 75-81's code is reused verbatim. Every line of code below carries its
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
# 0. REUSED -- Days 75-81, verbatim (everything up to section 4 is Day 81's code)
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
# 4. THE WHOLE BACKWARD PASS -- NEW
# =============================================================================
class TrainableTransformer(Transformer):                # NEW: Day 81's model, plus the backward pass through all of it
    def forward(self, src, smask, tgt_in, tmask):          # Day 81's forward, remembering what the backward pass needs
        self.src, self.tgt_in, self.smask, self.tmask = src, tgt_in, smask, tmask  # the inputs and their masks
        self.logp = super().forward(src, smask, tgt_in, tmask)  # unchanged
        return self.logp                                  # (m * T, V)

    def gradients(self, y, broken=False):                  # every gradient of Day 81's mean cross-entropy; broken: Section 3's bug
        m, T = self.tgt_in.shape                             # batch, decoder length
        mask = self.tmask.ravel()                            # real target positions
        dlog = np.exp(self.logp); dlog[np.arange(y.size), y.ravel()] -= 1  # softmax minus one-hot ...
        dlog *= mask[:, None] / mask.sum()                   # ... on the real positions, averaged
        G = dict(Wout=self.dec_out.reshape(-1, D).T @ dlog, bout=dlog.sum(axis=0))  # the output layer
        dy = (dlog @ self.P["Wout"].T).reshape(m, T, D)      # gradient at the top of the decoder
        denc = np.zeros_like(self.enc_out)                   # gradient at the encoder output: every decoder block adds to it
        for i in reversed(range(len(self.dec))):             # back down the decoder, last block first
            dy, de, g = self.dec[i].backward(dy)               # Day 80's backward: the block's input, the encoder output, weights
            denc = de if broken else denc + de                 # the BUG keeps only the last-visited block's share
            G.update({f"d{i}.{k}": v for k, v in g.items()})   # its weights
        G["E"] = np.zeros_like(self.P["E"])                  # ONE table, reached from both sides
        np.add.at(G["E"], self.tgt_in, dy)                   # the target embeddings (the positions are fixed: no gradient)
        dx = denc                                            # the encoder's output gradient ...
        for i in reversed(range(len(self.enc))):             # ... back down the encoder
            dx, g = self.enc[i].backward(dx)                   # Day 78's backward
            G.update({f"e{i}.{k}": v for k, v in g.items()})   # its weights
        np.add.at(G["E"], self.src, dx)                      # the source embeddings: the same table
        return G                                          # dict: name -> gradient

    def train_step(self, src, smask, tgt_in, tmask, y, lr, clip=1.0):  # one SGD step with Day 60's clipping
        self.forward(src, smask, tgt_in, tmask)              # forward pass
        G, P = self.gradients(y), self.params()              # gradients, and the arrays they belong to
        norm = np.sqrt(sum(np.sum(v ** 2) for v in G.values()))  # global norm
        for k, v in G.items():                               # plain SGD
            P[k] -= lr * min(1.0, clip / norm) * v             # in place: the blocks see it
        return norm                                       # for the record


def gradient_check(net, batch, broken=False, eps=1e-5):  # NEW: every entry of every parameter, Day 62's rule, by parameter group
    src, sm, tgt_in, tm, y = batch                        # inputs, masks, targets
    net.forward(src, sm, tgt_in, tm); G = net.gradients(y, broken)  # analytic
    groups = {}                                           # group -> (largest difference, every entry passes)
    for k, A in net.params().items():                     # every parameter array
        worst, ok = 0.0, True                               # this array
        for pos in np.ndindex(A.shape):                     # every entry
            old = A[pos]                                      # remember it
            A[pos] = old + eps; up = net.loss(src, sm, tgt_in, tm, y)  # nudge up
            A[pos] = old - eps; dn = net.loss(src, sm, tgt_in, tm, y)  # nudge down
            A[pos] = old                                      # restore
            num = (up - dn) / (2 * eps)                       # central difference
            worst = max(worst, abs(num - G[k][pos])); ok &= abs(num - G[k][pos]) <= 1e-10 + 1e-6 * abs(num)  # Day 62's rule
        g = k if k in ("E", "Wout", "bout") else k.split(".")[0]  # group: E, the output layer, or a block ("e0", "d1", ...)
        w, o = groups.get(g, (0.0, True)); groups[g] = (max(w, worst), o and ok)  # combine
    return groups                                          # one line per group


def tiny_batch():                                       # the batch every gradient check uses: padded on both sides
    rng = np.random.RandomState(3)                        # fixed rng
    src = rng.randint(1, 9, size=(3, 5)); tgt_in = np.concatenate([np.full((3, 1), GO), rng.randint(1, 9, size=(3, 4))], axis=1)  # 3 sources; GO + 4 digits
    y = rng.randint(0, V, size=(3, 5))                    # targets (any tokens: the check only needs a loss)
    sm = np.ones((3, 5), bool); sm[1, 3:] = False; sm[2, 4:] = False  # padded source positions
    tm = np.ones((3, 5), bool); tm[1, 4:] = False         # one padded target position
    return src, sm, tgt_in, tm, y                          # everything gradient_check needs


# =============================================================================
# DEMOS
# =============================================================================
def demo_chain():
    print("=" * 70)
    print("1. THE ORDER OF THE BACKWARD PASS -- one untrained model, one batch")
    print("=" * 70)
    print("The gradient at each hand-over point, for 32 COPY examples of 8 digits, and its size (the square root "
          "of the sum of squares). The forward pass goes source -> encoder -> decoder -> scores; the backward pass "
          "goes the other way, and the encoder's gradient arrives only through the cross-attention.")
    src, tgt_in, y = as_ids(make_batch(np.random.RandomState(5), 32, 8, "copy"))  # one batch
    sm, tm = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
    net = TrainableTransformer(seed=0, out_scale=0.1)     # Day 81's smaller output layer
    net.forward(src, sm, tgt_in, tm); G = net.gradients(y)  # one backward pass
    size = lambda a: np.sqrt((a ** 2).sum())              # the size of a gradient
    print(f"\n{'stage':<44}{'shape':>14}{'size':>11}")
    print(f"{'output layer Wout':<44}{str(G['Wout'].shape):>14}{size(G['Wout']):>11.2e}")
    for k in ("d0.W2", "d0.cq", "d0.ck", "d0.sq", "e0.W2", "e0.Wq", "E"):  # down the model
        print(f"{k:<44}{str(G[k].shape):>14}{size(G[k]):>11.2e}")
    print("(d0 = the decoder block, e0 = the encoder block; cq, ck: cross-attention's query and key weights; sq: self-attention's)")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("2. GRADIENT CHECK -- the whole model, every entry of every parameter")
    print("=" * 70)
    print("Tiny sizes (d = 8, 2 heads, FFN 12), TWO encoder blocks and TWO decoder blocks, 3 examples with padded "
          "source and target positions; every entry nudged by +-1e-5; Day 62's pass rule |numeric - analytic| <= "
          "1e-10 + 1e-6 |numeric|. Reported by part.")
    global D, F, HEADS                                    # tiny sizes, for the checks only
    saved = (D, F, HEADS); D, F, HEADS = 8, 12, 2         # restored at the end of Section 3
    net = TrainableTransformer(n_enc=2, n_dec=2, seed=1, out_scale=0.1)  # tiny, two blocks per side
    for b in net.enc + net.dec:                           # move the norms off g = 1, b = 0 so they are tested
        for k in b.P:  # every weight of the block
            if k[0] in "gb" and k[1:].isdigit():            # g1, b1, g2, ...
                b.P[k] = b.P[k] + 0.1 * np.random.RandomState(7).randn(D)  # nudged off its starting value
    n = sum(v.size for v in net.params().values())        # how many entries are checked
    groups = gradient_check(net, tiny_batch())            # the check
    print(f"\n{n} entries checked")
    print(f"{'part':<26}{'largest |numeric - analytic|':>30}{'every entry passes':>20}")
    names = {"E": "embedding table (shared)", "Wout": "output layer W", "bout": "output layer b"}  # readable names for the printout
    for g, (w, ok) in groups.items():                     # each part
        label = names[g] if g in names else ("encoder" if g[0] == "e" else "decoder") + f" block {int(g[1:]) + 1}"  # a readable name
        print(f"{label:<26}{w:>30.2e}{str(ok):>20}")
    print("Actual output from this lesson's run.")
    return net, saved                                      # Section 3 reuses the tiny model


def demo_broken(net, saved):
    print("\n" + "=" * 70)
    print("3. WOULD THE CHECK CATCH A BUG? -- one deliberate mistake")
    print("=" * 70)
    print("Both decoder blocks read the same encoder output, so the encoder's gradient is the SUM of what each "
          "decoder block sends back. The bug keeps only the share of the last block visited (block 1) and drops "
          "block 2's. Same tiny model, same batch.")
    global D, F, HEADS                                    # still the tiny sizes
    groups = gradient_check(net, tiny_batch(), broken=True)  # the check, with the bug
    print(f"\n{'part':<26}{'largest |numeric - analytic|':>30}{'every entry passes':>20}")
    names = {"E": "embedding table (shared)", "Wout": "output layer W", "bout": "output layer b"}  # readable names for the printout
    for g, (w, ok) in groups.items():                     # each part
        label = names[g] if g in names else ("encoder" if g[0] == "e" else "decoder") + f" block {int(g[1:]) + 1}"  # a readable name
        print(f"{label:<26}{w:>30.2e}{str(ok):>20}")
    D, F, HEADS = saved                                   # back to the lesson's sizes
    print("Actual output from this lesson's run.")


def demo_gradient_sizes():
    print("\n" + "=" * 70)
    print("4. WHERE THE GRADIENT GOES -- sizes by part, untrained, three seeds")
    print("=" * 70)
    print("The size of each part's gradient (square root of the sum of squares over all its weights) for 32 COPY "
          "examples of 8 digits, with one block per side and with two.")
    src, tgt_in, y = as_ids(make_batch(np.random.RandomState(5), 32, 8, "copy"))  # one batch
    sm, tm = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
    print(f"\n{'blocks':>7}{'seed':>5}" + "".join(f"{p:>12}" for p in ("embedding", "enc 1", "enc 2", "dec 1", "dec 2", "output")))
    res = {}                                              # n -> list of per-part sizes (mean over seeds)
    for n in (1, 2):                                      # one and two blocks per side
        rows = []                                           # per seed
        for s in SEEDS:                                     # three seeds
            net = TrainableTransformer(n_enc=n, n_dec=n, seed=s, out_scale=0.1)  # untrained
            net.forward(src, sm, tgt_in, tm); G = net.gradients(y)  # one backward pass
            part = lambda pre: np.sqrt(sum(np.sum(v ** 2) for k, v in G.items() if k.startswith(pre)))  # one part's size
            row = [part("E"), part("e0."), part("e1.") if n == 2 else np.nan, part("d0."), part("d1.") if n == 2 else np.nan,  # six parts
                   np.sqrt(np.sum(G["Wout"] ** 2) + np.sum(G["bout"] ** 2))]  # six parts (nan where there is no block)
            rows.append(row)  # keep this seed's sizes
            print(f"{n:>7}{s:>5}" + "".join(f"{v:>12.2e}" if not np.isnan(v) else f"{'-':>12}" for v in row))
        res[n] = np.array(rows, dtype=float).mean(axis=0)   # mean over seeds, per part (nan stays nan where there is no block)
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_overfit():
    print("\n" + "=" * 70)
    print("5. DOES THE GRADIENT TEACH? -- fit one fixed batch")
    print("=" * 70)
    print("A gradient that passes the check should drive the loss on a FIXED batch towards 0. 32 REVERSE examples "
          "of 6 digits, one block per side, plain SGD with Day 60's clipping, 600 steps, at three learning rates. "
          "Loss and token accuracy on the same 32 examples (memorising them is the point here, not generalising).")
    src, tgt_in, y = as_ids(make_batch(np.random.RandomState(6), 32, 6, "reverse"))  # the fixed batch
    sm, tm = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
    print(f"\n{'lr':>6}" + "".join(f"{'@' + str(s):>9}" for s in (0, 50, 100, 200, 400, 600)) + f"{'token acc':>11}")
    curves = {}                                           # lr -> losses
    for lr in (0.1, 0.5, 2.0):                            # three learning rates
        net = TrainableTransformer(seed=0, out_scale=0.1)   # fresh
        losses = [net.loss(src, sm, tgt_in, tm, y)]         # step 0
        for step in range(1, 601):                          # 600 steps on the same batch
            net.train_step(src, sm, tgt_in, tm, y, lr)        # one SGD step
            losses.append(net.loss(src, sm, tgt_in, tm, y))   # loss after it
        acc = (net.forward(src, sm, tgt_in, tm).argmax(axis=1).reshape(y.shape) == y).mean()  # token accuracy
        curves[lr] = losses                                 # for the plot
        print(f"{lr:>6}" + "".join(f"{losses[s]:>9.3f}" for s in (0, 50, 100, 200, 400, 600)) + f"{acc:>11.3f}")
    print("Actual output from this lesson's run.")
    return curves                                          # the plot reuses them


def demo_cost():
    print("\n" + "=" * 70)
    print("6. WHAT THE BACKWARD PASS COSTS -- against the forward pass, this run")
    print("=" * 70)
    print("One block per side, batch of 32, 12 digits; best of 5 timings of 10 passes, matrix library on one "
          "thread. Only ratios within this run are meaningful.")
    src, tgt_in, y = as_ids(make_batch(np.random.RandomState(7), 32, 12, "copy"))  # one batch
    sm, tm = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
    net = TrainableTransformer(seed=0, out_scale=0.1)     # untrained

    def best(fn):                                         # fastest of 5 timings of 10 calls (Day 71)
        times = []                                          # one timing per repeat
        for _ in range(5):                                  # 5 repeats
            start = time.perf_counter()                       # clock
            for _ in range(10):                               # 10 calls
                fn()                                            # the work being timed
            times.append((time.perf_counter() - start) / 10)  # per call
        return min(times)                                   # fastest

    f = best(lambda: net.forward(src, sm, tgt_in, tm))    # forward only
    fb = best(lambda: (net.forward(src, sm, tgt_in, tm), net.gradients(y)))  # forward and backward
    print(f"\nforward {f * 1000:.2f} ms; forward + backward {fb * 1000:.2f} ms; the backward pass alone is "
          f"{(fb - f) / f:.2f} times the forward pass")
    print("Actual output from this lesson's run.")


def plot_results(res, curves):                          # two panels: gradient sizes by part | fitting one batch
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))          # canvas
    labels = ["embedding", "enc 1", "enc 2", "dec 1", "dec 2", "output"]  # panel 1: sizes by part
    x = np.arange(len(labels))                              # bar positions
    for j, (n, color) in enumerate([(1, "#9ecae1"), (2, "#08519c")]):
        vals = np.nan_to_num(res[n])                          # missing blocks drawn as 0 height
        ax[0].bar(x + (j - 0.5) * 0.38, vals, width=0.38, color=color, label=f"{n} block{'s' if n > 1 else ''} per side")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_yscale("log")
    ax[0].set_ylabel("gradient size (mean of 3 seeds)"); ax[0].set_title("Where the gradient goes, untrained", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")
    for lr, color in zip(curves, ["#9ecae1", "#3182bd", "#08519c"]):  # panel 2: fitting one batch
        ax[1].plot(curves[lr], color=color, label=f"lr {lr}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("SGD steps on the same 32 examples"); ax[1].set_ylabel("loss")
    ax[1].set_title("Fitting one fixed batch", fontsize=10); ax[1].legend(fontsize=8, loc="upper right"); ax[1].grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("transformer_backward_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved transformer_backward_results.png")


def draw_diagram():                                     # the backward pass, right to left
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#d62728"):                    # one (red, backward) arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.3))

    ax.text(6.5, 4.15, "Day 82: the backward pass through the whole Transformer", ha="center", fontsize=12, fontweight="bold")
    box(11.9, 1.4, 1.7, 0.6, "loss", "#F2F2F2", bold=True)
    box(9.7, 1.4, 1.9, 0.6, "output layer", "#E5F2E1")
    box(7.1, 1.4, 2.4, 0.75, "decoder blocks\n(last block first)", "#F4EEFC")
    box(4.2, 1.4, 2.2, 0.6, "target embeddings", "#FAEEDA")
    box(7.1, 3.1, 2.4, 0.75, "encoder blocks\n(last block first)", "#DCEBF7")
    box(4.2, 3.1, 2.2, 0.6, "source embeddings", "#FAEEDA")
    box(1.4, 2.25, 1.9, 0.8, "ONE table E:\nboth add up", "#FAEEDA", bold=True, fs=8)
    arrow(11.03, 1.4, 10.67, 1.4); arrow(8.73, 1.4, 8.32, 1.4); arrow(5.88, 1.4, 5.32, 1.4)  # loss -> output -> decoder -> target embeddings
    arrow(7.1, 1.8, 7.1, 2.7)                                                # decoder -> encoder output
    ax.text(7.25, 2.25, "the SUM of every decoder block's\ncross-attention gradient", ha="left", fontsize=7.5, color="#d62728")
    arrow(5.88, 3.1, 5.32, 3.1)  # encoder -> source embeddings
    arrow(3.08, 1.5, 2.37, 2.05); arrow(3.08, 3.0, 2.37, 2.45)  # both into the one table
    ax.text(6.5, 0.4, "each block's own backward pass is Day 78's or Day 80's; the positional encoding is fixed, so it gets no gradient",
            ha="center", fontsize=8, style="italic")
    plt.savefig("transformer_backward_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved transformer_backward_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 83 is the week's build: a tiny Transformer assembled from Week 12's exercise scripts, trained on "
          "COPY and REVERSE and compared with Week 11's attention LSTM.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_chain()                                         # 1. the order of the backward pass
        net, saved = demo_gradient_check()                   # 2. every gradient
        demo_broken(net, saved)                              # 3. a deliberate bug
        res = demo_gradient_sizes()                          # 4. where the gradient goes
        curves = demo_overfit()                              # 5. fitting one batch
        demo_cost()                                          # 6. cost
        plot_results(res, curves)                            # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 7. Day 83


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
