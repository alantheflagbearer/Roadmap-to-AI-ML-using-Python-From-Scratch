"""
Day 83: The Week 12 Build -- a Tiny Transformer From the Exercise Parts, Trained

Week 12's last day does two jobs at once, like Days 62, 69 and 76. The
syllabus topic is training a tiny Transformer end to end, with honest
results. And this is the week's integration day: the model is assembled
only from the short exercise scripts of Days 77-82.

The parts, and where each one goes:

  Day 77 exercise  sinusoidal_encoding                 positions, added to the embeddings
  Day 78 exercise  encoder_block (+ backward)          the encoder; its attention and heads
  Day 79 exercise  layer_norm (+ backward)             every add & norm
  Day 80 exercise  causal_mask, cross_attention        the decoder's two attentions
  Day 81 exercise  decoder_block, transformer_forward  the decoder and the whole forward pass
  Day 82 exercise  decoder_block_backward,             the whole backward pass
                   transformer_backward

The only new code is greedy decoding and the training loop. The test is
the one Weeks 10-11 used: COPY and REVERSE of 3-20 digits (Day 69's data,
Day 70's fixed test sets), trained exactly as Day 71 trained its attention
LSTM -- plain SGD, lr 0.5, clip 1.0, batches of 32, 20,000 steps, three
seeds -- so the two can be compared directly.

What is measured, reported as it came out:
  1. The build fits: a gradient check of the assembled model.
  2. COPY, three seeds, against Day 71's attention LSTM.
  3. REVERSE, three seeds.
  4. Sequences longer than any seen in training.
  5. What the cross-attention learned (Day 72's measurements).
  6. How sensitive training is to the learning rate.
  7. What a training step costs, against Day 71's model.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep an imported script quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: the path to Week 11's folder (Day 71's script)
import sys                                             # sys: path, to import Day 71's script
import time                                            # time: perf_counter, to time training steps

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
EOS, GO = 0, 9                                          # Day 69: token 0 = end of sequence, 9 = "start writing"; digits are 1..8
V = 10                                                  # vocabulary: EOS, eight digits, GO
SEEDS = [0, 1, 2]                                       # Day 71's three seeds
D, F, HEADS = 32, 64, 4                                 # model size (Day 71's hidden size), feed-forward size, heads
LR, STEPS = 0.5, 20000                                  # Day 71's learning rate and budget
CHECKS = (500, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 15000, 20000)  # when the test sets are scored
EVAL_LENGTHS = (4, 8, 12, 16, 20)                       # Day 70's test lengths
DAY71 = dict(weights=17866, n20_2500="0.000", n20_5000="0.992", tok25="0.742",  # Day 71's results, copied from day71_run_output.txt
             checks=[2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000], curve20=[0.0, 0.992, 1, 1, 1, 1, 1, 1])


# =============================================================================
# 1. THE PARTS -- Week 12's exercise scripts, verbatim
# =============================================================================
# ---- from day77_exercise.py, verbatim ----
def sinusoidal_encoding(n, d):                          # NEW: (n, d) table, row p = position p
    p = np.arange(n)[:, None]                             # positions as a column
    i = np.arange(d // 2)[None, :]                        # pair index as a row
    angle = p / 10000 ** (2 * i / d)                      # one angle per (position, pair)
    PE = np.zeros((n, d))                                 # the table
    PE[:, 0::2], PE[:, 1::2] = np.sin(angle), np.cos(angle)  # sine in even columns, cosine in odd ones
    return PE                                             # hand back


# ---- from day78_exercise.py, verbatim ----
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


# ---- from day79_exercise.py, verbatim ----
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


# ---- from day80_exercise.py, verbatim ----
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


# ---- from day81_exercise.py, verbatim ----
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


# ---- from day82_exercise.py, verbatim ----
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


# =============================================================================
# 2. THE TOY TASK -- Day 69's data and Day 70's test sets, verbatim
# =============================================================================
def make_batch(rng, m, n, task):                        # Day 69's data, verbatim: m sequences of n digits
    src = rng.randint(1, 9, size=(m, n))                  # random digits 1..8
    tgt = src if task == "copy" else src[:, ::-1]         # COPY: the same digits; REVERSE: backwards
    dec_in = np.concatenate([np.full((m, 1), GO), tgt], axis=1)    # decoder input: GO, then the target shifted right
    dec_out = np.concatenate([tgt, np.full((m, 1), EOS)], axis=1)  # decoder output: the target, then EOS
    return np.eye(V)[src], np.eye(V)[dec_in], dec_out, src  # one-hot source, one-hot decoder input, targets, raw digits


def fixed_test(n, task, m=300):                         # Day 70's fixed test sets: the same sequences Days 70-73 used
    return make_batch(np.random.RandomState(12345 + n), m, n, task)  # a seed per length


# =============================================================================
# 3. TRAINING AND DECODING -- NEW (the only code not from an exercise script)
# =============================================================================
def generate(P, src, n_steps):                          # NEW: greedy decoding with the exercise model, one token at a time
    m = len(src); smask = np.ones(src.shape, bool)        # no source padding in the toy task
    tok = np.full((m, 1), GO)                             # every output starts with GO
    for _ in range(n_steps):                              # one token per step
        logp, _ = transformer_forward(src, tok, P, HEADS, smask, np.ones(tok.shape, bool))  # Day 81's forward pass
        tok = np.concatenate([tok, logp[:, -1].argmax(axis=-1)[:, None]], axis=1)  # the most probable token is written
    return tok[:, 1:]                                     # GO dropped


def evaluate(P, n, task):                               # Day 70's measures: exact match and token accuracy, greedy decoding
    _, _, Y, src = fixed_test(n, task)                    # the fixed test sequences of length n
    pred = generate(P, src, Y.shape[1])                   # n digits and EOS
    return np.all(pred == Y, axis=1).mean(), (pred == Y).mean(), (pred == Y).mean(axis=0)  # exact, token, per position


def train(task, seed, lr=LR, steps=STEPS):              # NEW: Day 71's training loop for the exercise model
    P = init_params(np.random.RandomState(seed), V, D, F)  # Day 81 exercise's weights (output layer x 0.1)
    rng = np.random.RandomState(seed + 100)               # Day 69-73's data-stream seed rule
    curve = []                                            # exact match per length at every check
    for step in range(1, steps + 1):                      # plain SGD on fresh batches
        n = rng.randint(3, 21)                              # a random length 3..20 (Day 70)
        _, Xd, Y, src = make_batch(rng, 32, n, task)        # one batch
        tgt_in = Xd.argmax(axis=-1)                         # decoder-input ids
        sm, tm = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
        logp, cache = transformer_forward(src, tgt_in, P, HEADS, sm, tm)  # Day 81 exercise
        G = dict(leaves(transformer_backward(logp, Y, tm, P, HEADS, cache)))  # Day 82 exercise
        norm = np.sqrt(sum(np.sum(g ** 2) for g in G.values()))  # global norm (Day 60)
        for k, A in leaves(P):                              # every weight
            A -= lr * min(1.0, 1.0 / norm) * G[k]             # SGD, clipped at 1.0 as on Day 71
        if step in CHECKS:                                  # a check step
            curve.append([evaluate(P, n_, task)[0] for n_ in EVAL_LENGTHS])  # exact match at each length
    return P, np.array(curve)                             # final weights, curve (checks, lengths)


RUNS = {}                                               # cache: each configuration trains once


def train_cached(*args, **kw):                          # train(), remembered by its arguments
    key = (args, tuple(sorted(kw.items())))               # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = train(*args, **kw)                    # train and remember
    return RUNS[key]                                      # later requests reuse it


# =============================================================================
# DEMOS
# =============================================================================
def demo_parts():
    print("=" * 70)
    print("1. THE BUILD -- Week 12's six exercise parts, and a check that they fit")
    print("=" * 70)
    print("Every function above section 2 is copied verbatim from an exercise script (a banner above each group says "
          "which). Then a gradient check of the assembled model at tiny sizes (d = 8, 2 heads, FFN 12), padded source "
          "and target, every entry nudged by +-1e-5.")
    n = sum(A.size for _, A in leaves(init_params(np.random.RandomState(0), V, D, F)))  # weights at the lesson's size
    print(f"\nthe lesson's model: d = {D}, FFN {F}, {HEADS} heads, one encoder block and one decoder block: {n} weights "
          f"(Day 71's attention LSTM: {DAY71['weights']})")
    rng = np.random.RandomState(0)                        # fixed rng
    P = init_params(rng, V, 8, 12)                        # tiny
    for _, A in leaves(P):                                # move every weight a little off its starting value
        A += 0.05 * rng.randn(*A.shape)
    src = rng.randint(1, 9, size=(3, 5)); tgt_in = np.concatenate([np.full((3, 1), GO), rng.randint(1, 9, size=(3, 4))], axis=1)
    y = rng.randint(0, V, size=(3, 5)); sm = np.ones((3, 5), bool); sm[1, 3:] = False; tm = np.ones((3, 5), bool); tm[2, 4:] = False
    logp, cache = transformer_forward(src, tgt_in, P, 2, sm, tm)  # forward
    G = dict(leaves(transformer_backward(logp, y, tm, P, 2, cache)))  # analytic
    worst, ok, count = 0.0, True, 0                       # largest difference, every entry passes, entries
    for name, A in leaves(P):                             # every array
        for pos in np.ndindex(A.shape):                     # every entry
            old = A[pos]                                      # remember it
            A[pos] = old + 1e-5; up = loss_of(P, src, tgt_in, y, 2, sm, tm)  # nudge up
            A[pos] = old - 1e-5; dn = loss_of(P, src, tgt_in, y, 2, sm, tm)  # nudge down
            A[pos] = old                                      # restore
            num = (up - dn) / 2e-5; count += 1                # central difference
            worst = max(worst, abs(num - G[name][pos])); ok &= abs(num - G[name][pos]) <= 1e-10 + 1e-6 * abs(num)
    print(f"gradient check: {count} entries, largest |numeric - analytic| = {worst:.2e}, every entry passes: {ok}")
    print("Actual output from this lesson's run.")
    return n                                               # later sections quote it


def demo_training(task):
    title = "2. COPY" if task == "copy" else "3. REVERSE"
    print("\n" + "=" * 70)
    print(f"{title} -- Day 71's setup exactly, three seeds")
    print("=" * 70)
    print(f"Lengths 3-20, plain SGD lr {LR}, clip 1.0, batches of 32, {STEPS} steps, the same data stream and the same "
          "300 fixed test sequences per length as Days 70-73. Exact match (every digit and EOS right, greedy "
          "decoding), mean of three seeds.")
    curves = np.mean([train_cached(task, s)[1] for s in SEEDS], axis=0)  # mean over seeds: (checks, lengths)
    print(f"\n{'steps':>7}" + "".join(f"{'n=' + str(n):>8}" for n in EVAL_LENGTHS))
    for c, row in zip(CHECKS, curves):                    # one row per check
        print(f"{c:>7}" + "".join(f"{v:>8.3f}" for v in row))
    if task == "copy":                                    # Day 71's numbers, for comparison
        print(f"Day 71's attention LSTM, the same setup (from its run output): at 20 digits {DAY71['n20_2500']} after 2500 "
              f"steps, {DAY71['n20_5000']} after 5000, 1.000 from 7500")
    print("Actual output from this lesson's run.")
    print("\nper seed, final exact match at 20 digits, and the first check at which every length is 1.000:")
    for s in SEEDS:                                       # each seed
        c = train_cached(task, s)[1]                        # its curve
        done = [CHECKS[i] for i in range(len(CHECKS)) if np.all(c[i] == 1.0)]  # checks with every length perfect
        print(f"  seed {s}: {c[-1, -1]:.3f}   all perfect from: {str(done[0]) + ' steps' if done else 'never'}")
    print("Actual output from this lesson's run.")
    return curves                                          # the plot reuses them


def demo_longer():
    print("\n" + "=" * 70)
    print("4. LONGER THAN ANY TRAINING SEQUENCE -- 25, 30 and 40 digits")
    print("=" * 70)
    print("The trained models (3-20 digits) on 300 fixed sequences of each new length. Day 71's attention LSTM: exact "
          f"match 0.000 at 25 digits, token accuracy {DAY71['tok25']}.")
    print(f"\n{'task':<9}{'length':>7}{'exact':>8}{'token acc':>11}   token acc, positions 1-20 / 21 on")
    per = {}                                              # task -> per-position accuracy at 30, mean of seeds
    for task in ("copy", "reverse"):                      # both tasks
        for n in (20, 25, 30, 40):                          # lengths
            res = [evaluate(train_cached(task, s)[0], n, task) for s in SEEDS]  # three seeds
            ex_, tok = np.mean([r[0] for r in res]), np.mean([r[1] for r in res])  # means
            pos = np.mean([r[2] for r in res], axis=0)       # per position
            extra = f"   {pos[:20].mean():.3f} / {pos[20:n].mean():.3f}" if n > 20 else ""  # inside vs beyond the training lengths
            print(f"{task:<9}{n:>7}{ex_:>8.3f}{tok:>11.3f}{extra}")
            if n == 30:                                       # kept for the plot
                per[task] = pos
    print("Actual output from this lesson's run.")
    return per                                             # the plot reuses it


def demo_alignment():
    print("\n" + "=" * 70)
    print("5. WHAT CROSS-ATTENTION LEARNED -- Day 72's measurements, 20 digits, seed 0")
    print("=" * 70)
    print("For each output digit, the cross-attention weights over the 20 source positions (one head of size "
          f"{D}, Day 80's exercise). Target: the source position holding the digit being written (COPY: the same "
          "position; REVERSE: the mirror). Entropy: log 20 = 3.00 means spread evenly.")
    out = {}                                              # task -> (m, 20, 20) weights
    print(f"\n{'task':<9}{'argmax on target':>18}{'weight on target':>18}{'entropy':>9}")
    for task in ("copy", "reverse"):                      # both tasks
        P = train_cached(task, 0)[0]                        # seed 0
        _, Xd, Y, src = fixed_test(20, task)                # 300 test sequences
        tgt_in = Xd.argmax(axis=-1)                         # teacher forcing: the decoder sees the right previous digits
        _, cache = transformer_forward(src, tgt_in, P, HEADS, np.ones(src.shape, bool), np.ones(tgt_in.shape, bool))
        A = cache[6][8][5][:, :20, :]                       # the decoder block's cross-attention cache: weights, EOS step left out
        j = np.arange(20); target = j if task == "copy" else 19 - j  # where each output's digit sits
        entropy = -(A * np.log(A + 1e-12)).sum(axis=2)      # spread of each row
        print(f"{task:<9}{(A.argmax(axis=2) == target).mean():>18.3f}{A[:, j, target].mean():>18.3f}{entropy.mean():>9.3f}")
        out[task] = A                                       # for the plot
    print("Actual output from this lesson's run.")
    return out                                             # the plot reuses it


def demo_learning_rate():
    print("\n" + "=" * 70)
    print("6. HOW SENSITIVE IS IT TO THE LEARNING RATE? -- COPY, seed 0, 5000 steps")
    print("=" * 70)
    print("The same training at four learning rates, stopped at 5000 steps. Exact match at 12 and 20 digits.")
    print(f"\n{'lr':>6}{'n=12':>8}{'n=20':>8}")
    for lr in (0.1, 0.5, 1.0, 2.0):                        # four rates
        P, _ = train_cached("copy", 0, lr=lr, steps=5000)   # a short run
        print(f"{lr:>6}{evaluate(P, 12, 'copy')[0]:>8.3f}{evaluate(P, 20, 'copy')[0]:>8.3f}")
    print("Actual output from this lesson's run.")


def demo_cost():
    print("\n" + "=" * 70)
    print("7. WHAT A TRAINING STEP COSTS -- against Day 71's attention LSTM, measured in this run")
    print("=" * 70)
    print("A batch of 32 sequences of 20 digits, forward and backward, best of 5 timings of 20 steps, matrix "
          "library on one thread. Day 71's model is imported from its own script. Only the ratio is meaningful.")
    _, Xd, Y, src = make_batch(np.random.RandomState(5), 32, 20, "copy")  # one batch
    Xs = np.eye(V)[src]; tgt_in = Xd.argmax(axis=-1)      # one-hot for Day 71, ids for the Transformer
    sm, tm = np.ones(src.shape, bool), np.ones(tgt_in.shape, bool)  # no padding
    P = init_params(np.random.RandomState(0), V, D, F)    # the Transformer
    sys.path.insert(0, os.path.join(HERE, "..", "week-11-attention-mechanism"))  # Day 71's folder
    with contextlib.redirect_stdout(io.StringIO()):       # its import prints nothing anyway; kept quiet to be sure
        import day71_additive_attention as d71
    lstm = d71.AttnSeq2Seq(hidden=32, seed=0)             # Day 71's model, hidden 32

    def best(fn):                                         # fastest of 5 timings of 20 calls
        times = []                                          # one timing per repeat
        for _ in range(5):                                  # 5 repeats
            start = time.perf_counter()                       # clock
            for _ in range(20):                               # 20 calls
                fn()                                            # the work being timed
            times.append((time.perf_counter() - start) / 20)  # per call
        return min(times)                                   # fastest

    def step():                                           # one Transformer training step's arithmetic
        logp, c = transformer_forward(src, tgt_in, P, HEADS, sm, tm)  # forward
        transformer_backward(logp, Y, tm, P, HEADS, c)      # backward

    tt = best(step)                                       # the Transformer
    tl = best(lambda: lstm.loss_and_grads(Xs, Xd, Y))     # Day 71: forward and backward
    print(f"\nTransformer: {tt * 1000:.1f} ms per step; Day 71's attention LSTM: {tl * 1000:.1f} ms; "
          f"the LSTM takes {tl / tt:.2f} times as long")
    print("Actual output from this lesson's run.")


def plot_results(copy, rev, per, align):                # four panels: learning curves | beyond 20 | two alignments
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))          # canvas
    j20 = EVAL_LENGTHS.index(20)                            # the 20-digit column
    ax[0].plot(CHECKS, copy[:, j20], marker="o", color="#1f77b4", label="Transformer, COPY")
    ax[0].plot(CHECKS, rev[:, j20], marker="o", color="#ff7f0e", label="Transformer, REVERSE")
    ax[0].plot(DAY71["checks"], DAY71["curve20"], marker="s", linestyle="--", color="#7f7f7f", label="Day 71 LSTM, COPY")
    ax[0].set_xscale("log"); ax[0].set_xlabel("training steps"); ax[0].set_ylabel("exact match, 20 digits (3 seeds)")
    ax[0].set_ylim(-0.02, 1.02); ax[0].set_title("Learning speed", fontsize=10); ax[0].legend(fontsize=7, loc="lower right"); ax[0].grid(alpha=0.3)
    for task, color in [("copy", "#1f77b4"), ("reverse", "#ff7f0e")]:  # panel 2: per position at 30 digits
        ax[1].plot(np.arange(1, 32), per[task], marker="o", markersize=3, color=color, label=task)
    ax[1].axvline(20.5, color="#333333", linestyle=":"); ax[1].text(21, 0.55, "beyond the\ntraining lengths", fontsize=8)
    ax[1].set_xlabel("output position (30 digits + EOS)"); ax[1].set_ylabel("token accuracy (3 seeds)"); ax[1].set_ylim(-0.02, 1.05)
    ax[1].set_title("Longer than any training sequence", fontsize=10); ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.3)
    for a, task in [(ax[2], "copy"), (ax[3], "reverse")]:  # panels 3-4: mean cross-attention, 20 digits
        im = a.imshow(align[task].mean(axis=0), cmap="Blues", vmin=0, vmax=1)
        a.set_xlabel("source position"); a.set_ylabel("output digit"); a.set_title(f"Cross-attention, {task.upper()} (seed 0)", fontsize=10)
    fig.colorbar(im, ax=ax[3], fraction=0.04)
    plt.tight_layout(); plt.savefig("tiny_transformer_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved tiny_transformer_results.png")


def draw_diagram():                                     # the build: six exercise parts into one model
    fig, ax = plt.subplots(figsize=(13, 4.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8, bold=False):            # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333"):                    # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.1))

    ax.text(6.5, 4.35, "Day 83: Week 12's build -- a tiny Transformer from the six exercise parts", ha="center",
            fontsize=12, fontweight="bold")
    parts = [("Day 77", "sinusoidal_encoding"), ("Day 78", "encoder_block (+ backward)"), ("Day 79", "layer_norm (+ backward)"),
             ("Day 80", "causal_mask, cross_attention"), ("Day 81", "decoder_block, transformer_forward"),
             ("Day 82", "decoder_block_backward,\ntransformer_backward")]
    for i, (day, name) in enumerate(parts):                                  # the six parts, left column
        y = 3.6 - i * 0.62
        box(2.2, y, 3.6, 0.5, f"{day}: {name}", "#FAEEDA", fs=7.5)
        arrow(4.02, y, 5.18, 2.05)                                           # each into the model
    box(6.6, 2.05, 2.7, 1.3, "one encoder block\n+ one decoder block\nd = 32, 4 heads, FFN 64", "#DCEBF7", fs=8.5)
    box(9.6, 2.05, 2.0, 0.9, "train: SGD lr 0.5,\nclip 1.0 (Day 71)", "#E5F2E1")
    box(11.9, 2.05, 1.8, 0.9, "COPY and\nREVERSE,\n3-20 digits", "#F2F2F2")
    arrow(7.97, 2.05, 8.58, 2.05); arrow(10.62, 2.05, 10.98, 2.05)
    ax.text(9.6, 0.4, "the only new code: greedy decoding and the training loop", ha="center", fontsize=8, style="italic")
    plt.savefig("tiny_transformer_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved tiny_transformer_diagram.png")


def note_on_the_week():
    print("\n" + "=" * 70)
    print("8. THE WEEK, AND WHAT COMES NEXT")
    print("=" * 70)
    print("Week 12 built the Transformer part by part: positions (Day 77), the encoder block (78), its "
          "normalisation (79), the decoder block and its mask (80), the whole forward pass (81) and the whole "
          "backward pass (82); today the parts trained together. Week 13 turns to the input side: how text becomes "
          "tokens (Day 84, byte-pair encoding) and how tokens become vectors.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_parts()                                         # 1. the build, checked
        copy = demo_training("copy")                         # 2. COPY
        rev = demo_training("reverse")                       # 3. REVERSE
        per = demo_longer()                                  # 4. longer than training
        align = demo_alignment()                             # 5. what cross-attention learned
        demo_learning_rate()                                 # 6. learning-rate sensitivity
        demo_cost()                                          # 7. cost
        plot_results(copy, rev, per, align)                  # the figure
        draw_diagram()                                       # the diagram
        note_on_the_week()                                   # 8. Day 84


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
