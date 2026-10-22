"""
Day 80: The Transformer Decoder Block, and Masked Self-Attention

The encoder (Day 78) reads a whole input at once. The decoder WRITES an
output one token at a time, so at position i it may only use positions
0..i -- the future has not been written yet. A causal MASK enforces
this inside self-attention:

    scores[i, j] = -infinity  for every j > i     ->  weight exactly 0

A decoder block has three parts, each with a residual and a LayerNorm
(post-LN, as in Day 78):

    h1  = LayerNorm( y  + MaskedSelfAttention(y) )
    h2  = LayerNorm( h1 + CrossAttention(queries h1, keys and values from the ENCODER) )
    out = LayerNorm( h2 + FFN(h2) )

What is built and measured, reported as it came out:

  1. The causal mask, by hand.
  2. The future cannot leak in: change a later word and the earlier
     outputs do not move.
  3. A gradient check of the full decoder block (with cross-attention,
     including the gradient for the encoder's output) and of a next-word
     model with and without the mask.
  4. Why the mask: a next-word model on Week 9's real sentences, trained
     with and without it, scored the way it was trained and the way it
     would be used (each word from the words before it only).
  5. With the mask, one pass over a sentence equals one pass per word.
  6. Where the unmasked model looks.

Days 62-78's code is reused verbatim. Every line of code below carries
its own comment, continuing Day 52's convention.
"""
import copy                                            # copy: deepcopy, to keep a snapshot of the best weights during training
import os                                              # os: build the path to Week 9's dataset folder
from collections import Counter                        # Counter counts things (the vocabulary)

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
DATA_DIR = os.path.join(HERE, "..", "week-09-sequence-data-rnns-i", "ud_english_ewt_pos")  # Day 62's dataset, used where it is
PAD, UNK = 0, 1                                         # reserved word ids: 0 = padding, 1 = a word not in the vocabulary
SEEDS = [0, 1, 2]                                       # three seeds for every trained comparison
EPOCHS = 25                                             # Day 62's training budget
LR = 2.0                                                # Day 78's learning rate, chosen there on validation
LM_EPOCHS = 20                                          # next-word training budget (each epoch also scores validation word by word)
D, F, HEADS = 32, 128, 4                                # model size, feed-forward size (4 x D, as in the paper), heads
PE_SCALE = 3.0                                          # Day 77's choice on validation: the encoding x 3, in queries and keys only
MAX_LEN = 128                                           # positions the encoding is built for


# =============================================================================
# 0. REUSED -- Days 62, 75, 76 and 77, verbatim
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


def read_tagged(name):                                  # Day 62: read one "word<TAB>tag" file; a blank line ends a sentence
    sentences, current = [], []                           # all sentences so far, and the one being read
    with open(os.path.join(DATA_DIR, name + ".txt"), encoding="utf-8") as f:  # open the file
        for line in f:                                      # one token per line
            line = line.rstrip("\n")                          # drop the newline
            if not line:                                      # blank line -> the sentence is finished
                sentences.append(current); current = []         # store it, start a new one
            else:                                             # a "word<TAB>tag" line
                word, tag = line.split("\t")                    # the two columns
                current.append((word.lower(), tag))             # lowercase the word, keep the tag
    return sentences                                        # a list of sentences, each a list of (word, tag) pairs


def encode(sentences, stoi, ttoi):                      # Day 62: sentences -> padded id arrays plus a tag array
    T = max(len(s) for s in sentences)                    # the longest sentence sets the width
    ids = np.full((len(sentences), T), PAD, dtype=int)    # word ids, padding everywhere to start
    tags = np.zeros((len(sentences), T), dtype=int)       # tag ids; the value under padding is never used
    lens = np.zeros(len(sentences), dtype=int)            # the real length of each sentence
    for row, sent in enumerate(sentences):                 # one sentence at a time
        ids[row, :len(sent)] = [stoi.get(w, UNK) for w, _ in sent]  # word -> id, unknown words -> UNK
        tags[row, :len(sent)] = [ttoi[t] for _, t in sent]   # tag -> id
        lens[row] = len(sent)                                # remember where the real words stop
    mask = np.arange(T)[None, :] < lens[:, None]           # True on real words, False on padding
    return ids, tags, mask                                  # everything a model needs for these sentences


def prepare_data():                                     # Day 62: load the three files, build the vocabularies, encode
    splits = {name: read_tagged(name) for name in ["train", "val", "test"]}  # sentences of (word, tag) pairs
    counts = Counter(w for s in splits["train"] for w, _ in s)  # word counts in the TRAINING sentences
    itos = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= 2)  # word vocabulary, as on Day 61
    stoi = {w: i for i, w in enumerate(itos)}               # word -> id
    tag_names = sorted({t for s in splits["train"] for _, t in s})  # the tag set, alphabetical
    ttoi = {t: i for i, t in enumerate(tag_names)}          # tag -> id
    data = [encode(splits[name], stoi, ttoi) for name in ["train", "val", "test"]]  # (ids, tags, mask) x 3
    return itos, tag_names, data                            # what this lesson needs


def token_accuracy(pred, tags, mask):                   # Day 62: fraction of REAL words tagged correctly
    return (pred == tags)[mask].mean()                    # padding positions are left out by the mask


# =============================================================================
# 1. THE NEW PARTS -- layer normalisation and the feed-forward network
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


# =============================================================================
# 2. THE DECODER BLOCK -- NEW: causal self-attention, cross-attention, FFN
# =============================================================================
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
# 3. A NEXT-WORD MODEL -- NEW: decoder-only, on Week 9's sentences
# =============================================================================
def lm_data(itos, data):                                # NEW: [BOS] + words as input, words + [EOS] as target
    BOS, EOS = len(itos), len(itos) + 1                   # two new token ids after the vocabulary
    out = []                                              # one (inputs, targets, mask) per split
    for ids, _, mask in data:                             # Day 62's three splits; the tags are not used
        n = mask.sum(axis=1)                                # real length of each sentence
        x = np.concatenate([np.full((len(ids), 1), BOS), ids], axis=1)  # BOS in front
        y = np.concatenate([ids, np.zeros((len(ids), 1), dtype=int)], axis=1)  # one extra column for EOS
        y[np.arange(len(ids)), n] = EOS                     # EOS right after the last word
        m = np.arange(x.shape[1])[None, :] <= n[:, None]   # n words + EOS are predicted
        x[~m] = PAD                                         # padding after BOS + n words
        out.append((x, y, m))                               # this split
    return out, len(itos) + 2                             # the splits, and the new vocabulary size


class NextWord:                                         # NEW: words -> one decoder block (no encoder) -> the next word
    def __init__(self, V, causal=True, lr=1.0, clip=1.0, seed=0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.V, self.lr, self.clip = V, lr, clip           # vocabulary, training settings
        self.E = rng.randn(V, D) * 0.1                     # embeddings, as on Days 62-79
        self.block = DecoderBlock(rng, causal=causal, cross=False)  # the decoder block
        self.Wout = rng.randn(D, V) / np.sqrt(D); self.bout = np.zeros(V)  # a score for every word
        self.PE = sinusoidal_encoding(MAX_LEN, D) * PE_SCALE  # Day 77's encoding

    def params(self):                                      # every trainable array, by name
        return dict(E=self.E, Wout=self.Wout, bout=self.bout, **{"b." + k: v for k, v in self.block.P.items()})  # the block's weights prefixed with 'b.'

    def forward(self, x, mask, last_only=False):           # (m, n) -> log-probabilities of the next word at every position
        m, n = x.shape                                       # batch, length
        self.x = x                                           # for the backward pass
        self.y = self.block.forward(self.E[x], self.PE[:n][None], mask)  # one block
        out = self.y[:, -1] if last_only else self.y.reshape(m * n, -1)  # last_only: the last position's scores only (Section 4)
        logits = out @ self.Wout + self.bout                 # a score per word
        self.logp = logits - logits.max(axis=1, keepdims=True)  # log-softmax (Day 78)
        self.logp -= np.log(np.exp(self.logp).sum(axis=1, keepdims=True))  # ... minus log of the sum of exponentials
        return self.logp                                  # (m * n, V)

    def loss(self, x, y, mask):                            # mean cross-entropy over the predicted positions
        logp = self.forward(x, mask)[np.arange(y.size), y.ravel()]  # log-probability of each true next word
        return -np.sum(logp * mask.ravel()) / mask.sum()   # padding counts 0

    def gradients(self, y, mask):                          # every gradient of that loss
        m, n = self.x.shape                                  # batch, length
        p = np.exp(self.logp); p[np.arange(y.size), y.ravel()] -= 1  # softmax + cross-entropy ...
        dlog = p * mask.ravel()[:, None] / mask.sum()        # ... on the predicted positions only
        G = dict(Wout=self.y.reshape(m * n, -1).T @ dlog, bout=dlog.sum(axis=0))  # output layer
        dx, _, g = self.block.backward((dlog @ self.Wout.T).reshape(m, n, -1))  # the block
        G.update({"b." + k: v for k, v in g.items()})        # its weights
        G["E"] = np.zeros_like(self.E); np.add.at(G["E"], self.x, dx)  # Day 61
        return G                                          # dict: name -> gradient

    def train_step(self, x, y, mask):                      # one SGD step with Day 60's clipping
        self.forward(x, mask)                                # forward pass
        G, P = self.gradients(y, mask), self.params()        # gradients, and the arrays they belong to
        norm = np.sqrt(sum(np.sum(v ** 2) for v in G.values()))  # global norm
        for k, v in G.items():                               # plain SGD
            P[k] -= self.lr * min(1.0, self.clip / norm) * v   # in place


def evaluate(net, split, honest):                       # accuracy and cross-entropy of the next word
    x, y, mask = split                                    # one split
    right, nll, count = 0, 0.0, 0                         # running totals
    if not honest:                                        # one pass over whole sentences: every position sees the full input
        for s in range(0, len(x), 32):                      # batches of 32 sentences (the scores are V numbers per word)
            xb, yb, mb = x[s:s + 32], y[s:s + 32], mask[s:s + 32]  # this batch
            T = mb.sum(axis=1).max(); xb, yb, mb = xb[:, :T], yb[:, :T], mb[:, :T]  # pad to its longest
            lp = net.forward(xb, xb != PAD).reshape(len(xb), T, -1)  # log-probabilities
            pick = np.take_along_axis(lp, yb[..., None], axis=-1)[..., 0]  # log-probability of each true next word
            right += ((lp.argmax(axis=-1) == yb) & mb).sum(); nll -= pick[mb].sum(); count += mb.sum()  # running totals over the real positions
        return right / count, nll / count                 # accuracy, mean cross-entropy
    lens = mask.sum(axis=1)                               # positions to predict per sentence
    for L in range(1, lens.max() + 1):                    # predict position L-1 from the first L inputs ONLY
        rows = np.where(lens >= L)[0]                       # sentences long enough
        for s in range(0, len(rows), 256):                  # batches of 256
            r = rows[s:s + 256]                               # these sentences
            xb = x[r, :L]                                     # inputs 0..L-1: nothing after them is given
            lp = net.forward(xb, xb != PAD, last_only=True)  # the prediction at the last given position
            right += (lp.argmax(axis=-1) == y[r, L - 1]).sum(); nll -= lp[np.arange(len(r)), y[r, L - 1]].sum(); count += len(r)  # running totals
    return right / count, nll / count                     # accuracy, mean cross-entropy


def fit_lm(splits, V, seed, causal, epochs=LM_EPOCHS, batch_size=32, lr=LR):  # one training run, best epoch on validation (honest)
    (x_tr, y_tr, m_tr), val, test = splits                # the three splits
    net = NextWord(V, causal=causal, lr=lr, seed=seed)      # fresh model
    rng = np.random.RandomState(seed)                       # shuffles the sentences each epoch
    lens = m_tr.sum(axis=1)                                 # predicted positions per sentence
    best = (np.inf, None, -1)                               # the best validation cross-entropy
    for epoch in range(epochs):                           # full budget
        perm = rng.permutation(len(x_tr))                      # fresh order
        for start in range(0, len(perm), batch_size):          # mini-batches
            idx = perm[start:start + batch_size]                 # this batch
            T = lens[idx].max()                                  # pad to its longest
            net.train_step(x_tr[idx, :T], y_tr[idx, :T], m_tr[idx, :T])  # one step
        ce = evaluate(net, val, honest=True)[1]               # validation cross-entropy, predicting from the past only
        if ce < best[0]:                                      # a new best
            best = (ce, copy.deepcopy(net.params()), epoch)     # snapshot
    for k, v in best[1].items():                            # restore it in place
        net.params()[k][...] = v                              # the block sees it too
    return dict(net=net, val_ce=best[0], epoch=best[2])     # model and its record


LM_RUNS = {}                                            # cache: each configuration trains once


def fit_lm_cached(splits, V, seed, causal):             # fit_lm(), remembered
    if (seed, causal) not in LM_RUNS:                     # first request: train it
        LM_RUNS[(seed, causal)] = fit_lm(splits, V, seed, causal)  # train and remember
    return LM_RUNS[(seed, causal)]                        # later requests reuse it


# =============================================================================
# DEMOS
# =============================================================================
def demo_mask():
    print("=" * 70)
    print("1. THE CAUSAL MASK -- each position may read itself and earlier positions only")
    print("=" * 70)
    print("Five positions, random scores (seed 0), one head. Masked keys get minus infinity, so weight 0; "
          "every row still sums to 1.")
    rng = np.random.RandomState(0)                        # fixed rng
    Q, K, Vv = rng.randn(1, 5, 4), rng.randn(1, 5, 4), rng.randn(1, 5, 4)  # one sequence
    _, A = scaled_dot_product_attention(Q, K, Vv, causal_mask(5)[None])  # Day 75's function with the mask
    print("\nmask (1 = may read):")
    for row in causal_mask(5).astype(int):                # the mask itself
        print("  " + " ".join(str(v) for v in row))
    print("weights:")
    for row in A[0]:                                      # the weights
        print("  " + " ".join(f"{v:.3f}" for v in row) + f"   sum {row.sum():.3f}")
    print(f"largest weight on a future position: {A[0][~causal_mask(5)].max():.1e}")
    print("Actual output from this lesson's run.")


def demo_no_future(V):
    print("\n" + "=" * 70)
    print("2. THE FUTURE CANNOT LEAK IN -- change a later word, watch the earlier outputs")
    print("=" * 70)
    print("One untrained decoder block (seed 0) on 4 random 12-word sequences. Word 8 of each is replaced "
          "by another word; the largest change in the outputs at positions 0-7 (which come before it) and "
          "at positions 8-11 is reported, with and without the causal mask.")
    rng = np.random.RandomState(5)                        # fixed rng
    x = rng.randint(2, V, size=(4, 12)); x2 = x.copy(); x2[:, 8] = rng.randint(2, V, size=4)  # the change
    mask = np.ones((4, 12), bool)                         # no padding
    print(f"\n{'model':<16}{'positions 0-7':>16}{'positions 8-11':>16}")
    for causal in (True, False):                          # with and without the mask
        net = NextWord(V, causal=causal, seed=0)            # untrained
        a = net.forward(x, mask).reshape(4, 12, -1); b = net.forward(x2, mask).reshape(4, 12, -1)  # before and after
        print(f"{'causal mask' if causal else 'no mask':<16}{np.abs(a - b)[:, :8].max():>16.2e}{np.abs(a - b)[:, 8:].max():>16.2e}")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("3. GRADIENT CHECK -- the decoder block with cross-attention, and the next-word model")
    print("=" * 70)
    print("Tiny sizes (d = 8, 2 heads, FFN 12). The decoder block: 2 sequences of 4 decoder positions reading "
          "6 encoder positions (one sequence padded on both sides); L = sum(U * output), and the gradient "
          "for the ENCODER output is checked too. The next-word model: 3 padded sentences, both masks. "
          "Every entry nudged by +-1e-5; Day 62's pass rule.")
    global D, F, HEADS, PE_SCALE                          # tiny sizes for this check only
    saved = (D, F, HEADS, PE_SCALE)                       # restored afterwards
    D, F, HEADS, PE_SCALE = 8, 12, 2, 1.0                 # tiny
    rng = np.random.RandomState(3)                        # fixed rng
    blk = DecoderBlock(rng, causal=True, cross=True)      # the full decoder block
    for k in blk.P:                                       # move the norms' g and b off 1 and 0 so they are tested
        if k[0] in "gb" and k[1:].isdigit():  # a norm's scale or shift (g1, b1, ...)
            blk.P[k] = blk.P[k] + 0.1 * rng.randn(D)  # nudged off its starting value
    y = rng.randn(2, 4, D); enc = rng.randn(2, 6, D)      # decoder inputs, encoder output
    ym = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], bool); em = np.array([[1] * 6, [1, 1, 1, 1, 0, 0]], bool)  # padding
    pe = sinusoidal_encoding(4, D)[None]                  # positions
    U = rng.randn(2, 4, D) * ym[:, :, None]               # L = sum(U * out) over real positions
    L = lambda: (U * blk.forward(y, pe, ym, enc, em)).sum()  # the loss
    blk.forward(y, pe, ym, enc, em); dy, denc, g = blk.backward(U)  # analytic gradients
    worst, ok = 0.0, True                                 # largest difference; every entry passes?
    for name, A, G in [("y", y, dy), ("enc", enc, denc)] + [(k, blk.P[k], g[k]) for k in blk.P]:  # inputs and weights
        for pos in np.ndindex(A.shape):                     # every entry
            old = A[pos]                                      # remember it
            A[pos] = old + 1e-5; up = L(); A[pos] = old - 1e-5; dn = L(); A[pos] = old  # nudge both ways, restore
            num = (up - dn) / 2e-5                            # central difference
            worst = max(worst, abs(num - G[pos])); ok &= abs(num - G[pos]) <= 1e-10 + 1e-6 * abs(num)  # Day 62's rule
    print(f"decoder block (self + cross + FFN): largest |numeric - analytic| = {worst:.2e}; every entry passes: {ok}")
    x = np.array([[10, 2, 3, 4, 5], [10, 6, 2, 0, 0], [10, 3, 3, 7, 0]]); yy = rng.randint(2, 12, size=x.shape)  # tiny sentences
    m = x != PAD                                          # predicted positions
    for causal in (True, False):                          # both models
        net = NextWord(12, causal=causal, seed=1)           # tiny
        net.forward(x, m); G = net.gradients(yy, m)         # analytic
        worst, ok = 0.0, True                               # reset
        for k, A in net.params().items():                   # every parameter
            for pos in np.ndindex(A.shape):                   # every entry
                old = A[pos]  # remember it
                A[pos] = old + 1e-5; up = net.loss(x, yy, m); A[pos] = old - 1e-5; dn = net.loss(x, yy, m); A[pos] = old  # nudge both ways, restore
                num = (up - dn) / 2e-5                          # central difference
                worst = max(worst, abs(num - G[k][pos])); ok &= abs(num - G[k][pos]) <= 1e-10 + 1e-6 * abs(num)  # Day 62's rule
        print(f"next-word model, {'causal mask' if causal else 'no mask':<12}: largest |numeric - analytic| = {worst:.2e}; every entry passes: {ok}")
    D, F, HEADS, PE_SCALE = saved                         # back to the lesson's sizes
    print("Actual output from this lesson's run.")


def demo_baselines(splits, V, itos):
    print("\n" + "=" * 70)
    print("4a. WHAT IS A GOOD SCORE? -- two counting baselines")
    print("=" * 70)
    print("Predict the next word with no model at all: always the most common training word, or the word "
          "that most often followed the previous word in training (a bigram table).")
    (x_tr, y_tr, m_tr), _, (x_te, y_te, m_te) = splits    # training and test
    counts = np.bincount(y_tr[m_tr], minlength=V)         # how often each word is a target
    uni = (y_te[m_te] == counts.argmax()).mean()          # always the most common
    pair = np.zeros((V, V), dtype=np.int32)               # previous word -> next word counts
    np.add.at(pair, (x_tr[m_tr], y_tr[m_tr]), 1)          # every (input, target) pair of training
    best_next = pair.argmax(axis=1)                       # the most common follower of each word
    best_next[pair.sum(axis=1) == 0] = counts.argmax()    # never seen: fall back to the most common word
    bi = (best_next[x_te[m_te]] == y_te[m_te]).mean()     # accuracy on the test set
    names = itos + ["<bos>", "<eos>"]                      # every token's name
    print(f"\nmost common target in training: '{names[counts.argmax()]}' (words outside the vocabulary); "
          f"share of test targets that are '<unk>': {(y_te[m_te] == UNK).mean():.4f}")
    print(f"always predicting it: {uni:.4f}      bigram table: {bi:.4f}")
    print("Actual output from this lesson's run.")
    return uni, bi                                         # the plot reuses them


def demo_leak(splits, V):
    print("\n" + "=" * 70)
    print("4. WHY THE MASK -- a next-word model trained with and without it")
    print("=" * 70)
    print(f"Week 9's sentences as a language-modelling task: input [BOS] w1 .. wn, target w1 .. wn [EOS]. One "
          f"decoder block, no encoder, size {D}, {HEADS} heads, Day 77's encoding; SGD lr {LR}, clip 1.0, batches of "
          f"32, {LM_EPOCHS} epochs, best epoch by validation cross-entropy measured from the past only. Two test "
          "scores: 'whole sentence' gives the model the full input in one pass, as in training; 'past only' "
          "predicts each word from the words before it, as when generating text.")
    print(f"\n{'model':<14}{'seed':>5}{'whole sentence: acc':>21}{'CE':>7}{'past only: acc':>16}{'CE':>7}{'best epoch':>12}")
    res = {}                                               # (causal) -> list of (whole acc, past acc)
    for causal in (True, False):                          # with and without the mask
        for s in SEEDS:                                     # three seeds
            r = fit_lm_cached(splits, V, s, causal)           # trained model
            wa, wc = evaluate(r["net"], splits[2], honest=False)  # whole sentence
            pa, pc = evaluate(r["net"], splits[2], honest=True)  # past only
            res.setdefault(causal, []).append((wa, pa, wc, pc))  # for the plot
            print(f"{'causal mask' if causal else 'no mask':<14}{s:>5}{wa:>21.4f}{wc:>7.3f}{pa:>16.4f}{pc:>7.3f}{r['epoch'] + 1:>12}")
    print("(CE: mean cross-entropy in nats per word; lower is better)")
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_same_answer(splits, V):
    print("\n" + "=" * 70)
    print("5. WITH THE MASK, ONE PASS = ONE PASS PER WORD")
    print("=" * 70)
    print("For the trained causal model (seed 0): the log-probabilities from one pass over the whole "
          "sentence, against those from feeding each prefix separately, on 200 test sentences.")
    x, y, m = splits[2]                                   # test split
    net = fit_lm_cached(splits, V, 0, True)["net"]        # trained causal model
    xb, mb = x[:200], m[:200]; T = mb.sum(axis=1).max(); xb, mb = xb[:, :T], mb[:, :T]  # 200 sentences
    full = np.concatenate([net.forward(xb[s:s + 20], xb[s:s + 20] != PAD).reshape(-1, T, V) for s in range(0, 200, 20)])  # one pass
    worst = 0.0                                           # largest difference
    for L in range(1, T + 1):                             # every prefix length
        rows = mb[:, L - 1]                                 # sentences that predict at position L-1
        if rows.any():                                      # some do
            part = net.forward(xb[rows, :L], xb[rows, :L] != PAD, last_only=True)  # prefix only
            worst = max(worst, np.abs(part - full[rows, L - 1]).max())  # compare
    print(f"\nlargest difference over every predicted position: {worst:.2e}")
    print("Actual output from this lesson's run.")


def demo_what_it_reads(splits, V):
    print("\n" + "=" * 70)
    print("6. WHERE THE UNMASKED MODEL LOOKS -- seed 0, 300 test sentences")
    print("=" * 70)
    print("The mean attention weight from each position to the position one ahead (whose input IS the "
          "word to predict), to itself, and to the position one behind, averaged over the heads.")
    x, y, m = splits[2]                                   # test split
    xb = x[:300]; T = (xb != PAD).sum(axis=1).max(); xb = xb[:, :T]  # 300 sentences
    real = xb != PAD                                      # real input positions
    print(f"\n{'model':<14}{'one ahead':>11}{'itself':>9}{'one behind':>12}")
    for causal in (True, False):                          # both models
        net = fit_lm_cached(splits, V, 0, causal)["net"]    # seed 0
        net.forward(xb, real, last_only=True)               # fills the attention cache (scores only for the last position)
        A = net.block.c["self"]["A"].reshape(300, HEADS, T, T).mean(axis=1)  # mean over heads
        i = np.arange(T - 1)                                # positions with a next position
        ok = real[:, i] & real[:, i + 1]                    # both real
        ahead = A[:, i, i + 1][ok].mean(); same = A[:, i, i][ok].mean()  # one ahead, itself
        j = np.arange(1, T); okb = real[:, j] & real[:, j - 1]  # positions with a previous position
        behind = A[:, j, j - 1][okb].mean()                 # one behind
        print(f"{'causal mask' if causal else 'no mask':<14}{ahead:>11.3f}{same:>9.3f}{behind:>12.3f}")
    print("Actual output from this lesson's run.")


def plot_results(res, base):                            # two panels: the two scores per model | the counting baselines
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))          # canvas
    labels = ["causal mask", "no mask"]                     # the two models
    for i, causal in enumerate((True, False)):              # panel 1: whole sentence vs past only, three seeds
        w = [r[0] for r in res[causal]]; p = [r[1] for r in res[causal]]  # the two scores
        ax[0].scatter([i - 0.15] * 3, w, color="#9ecae1", s=35, label="whole sentence (as in training)" if i == 0 else None)
        ax[0].scatter([i + 0.15] * 3, p, color="#08519c", s=35, label="past only (as when generating)" if i == 0 else None)
    ax[0].axhline(base[1], color="#d62728", linestyle="--", label="bigram table")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(labels); ax[0].set_ylim(0, 1.02); ax[0].set_ylabel("next-word accuracy (test)")
    ax[0].set_title("Trained with and without the causal mask", fontsize=10); ax[0].legend(fontsize=8, loc="upper center")
    for i, causal in enumerate((True, False)):              # panel 2: cross-entropy, past only
        ax[1].scatter([i] * 3, [r[3] for r in res[causal]], color="#08519c", s=35)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(labels); ax[1].set_ylabel("cross-entropy, past only (nats per word)")
    ax[1].set_title("Predicting from the past only (lower is better)", fontsize=10); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("causal_mask_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved causal_mask_results.png")


def draw_diagram():                                     # the decoder block, and the mask
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333"):                    # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    ax.text(6.5, 4.15, "Day 80: one Transformer decoder block (post-LN)", ha="center", fontsize=12, fontweight="bold")
    y = 2.3                                                                  # main line
    box(0.7, y, 0.9, 0.6, "y", "#F2F2F2", bold=True)                         # decoder input
    box(2.5, y, 2.0, 0.8, "masked self-\nattention", "#F4EEFC")             # causal self-attention
    box(4.35, y, 1.2, 0.6, "add & norm", "#DCEBF7", fs=8)
    box(6.2, y, 2.0, 0.8, "cross-attention\nQ: decoder, K, V: encoder", "#FDE7D3", fs=8)
    box(8.05, y, 1.2, 0.6, "add & norm", "#DCEBF7", fs=8)
    box(9.7, y, 1.5, 0.8, "FFN (Day 78)", "#E5F2E1")
    box(11.3, y, 1.2, 0.6, "add & norm", "#DCEBF7", fs=8)
    box(12.5, y, 0.7, 0.6, "out", "#F2F2F2", bold=True)
    arrow(1.17, y, 1.48, y); arrow(3.52, y, 3.73, y); arrow(4.97, y, 5.18, y); arrow(7.22, y, 7.43, y)  # the main line, left half
    arrow(8.67, y, 8.93, y); arrow(10.47, y, 10.68, y); arrow(11.92, y, 12.13, y)  # the main line, right half
    box(6.2, 3.55, 2.2, 0.5, "encoder output", "#F2F2F2")                    # the encoder
    arrow(6.2, 3.28, 6.2, 2.72)                                              # encoder -> cross-attention
    for i in range(4):                                                       # a small picture of the causal mask
        for j in range(4):  # four columns
            ax.add_patch(FancyBboxPatch((1.7 + j * 0.42, 0.25 + (3 - i) * 0.3), 0.36, 0.25, boxstyle="round,pad=0.01",
                                        facecolor="#7A4EC9" if j <= i else "#EEEEEE", edgecolor="#333333", linewidth=0.5))
    ax.text(3.6, 0.85, "causal mask: row i may read\ncolumns 0..i (purple)", ha="left", fontsize=7.5, style="italic")
    ax.text(9.0, 0.85, "decoder-only models (Section 4) leave out\nthe cross-attention and its add & norm", ha="center", fontsize=7.5, style="italic")
    plt.savefig("decoder_block_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved decoder_block_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 81 joins the two halves: encoder blocks (Day 78) read the input, decoder blocks (today) write "
          "the output while reading the encoder through cross-attention -- a full Transformer forward pass on a "
          "toy copy task.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_mask()                                          # 1. the causal mask
        itos, _, data = prepare_data()                       # Day 62's sentences
        splits, V = lm_data(itos, data)                      # as a next-word task
        demo_no_future(V)                                    # 2. no leak from the future
        demo_gradient_check()                                # 3. gradients
        base = demo_baselines(splits, V, itos)               # 4a. counting baselines
        res = demo_leak(splits, V)                           # 4. with and without the mask
        demo_same_answer(splits, V)                          # 5. one pass = one pass per word
        demo_what_it_reads(splits, V)                        # 6. where the unmasked model looks
        plot_results(res, base)                              # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 7. Day 81


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
