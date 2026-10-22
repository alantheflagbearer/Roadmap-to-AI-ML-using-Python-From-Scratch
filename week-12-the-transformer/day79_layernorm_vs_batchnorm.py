"""
Day 79: LayerNorm vs BatchNorm -- a Real Comparison, and Why Transformers Use LayerNorm

Day 78's block normalises with LAYER normalisation: each position's
vector of d numbers is shifted to mean 0 and spread 1 on its own. Day 54
used BATCH normalisation in a CNN: each FEATURE is normalised over all
the examples in the batch, and running averages of its mean and variance
are kept for test time.

    LayerNorm:  mean and variance over the d features of ONE position
    BatchNorm:  mean and variance of ONE feature over every position of the batch

Today puts both inside Day 78's encoder block, on Week 9's real tagging
data, where sentences have different lengths and are padded.

What is measured, reported as it came out:

  1. The two norms by hand, on the same numbers.
  2. Does one sentence's output depend on the OTHER sentences in its batch?
  3. Does padding change the output?
  4. A gradient check of BatchNorm inside the block.
  5. Training with batches of 32, three seeds: LayerNorm, BatchNorm with
     padding left out of its statistics, and BatchNorm with it counted.
  6. Batches of 4.
  7. BatchNorm at test time: running averages or batch statistics?
  8. The cost of each.

Days 63-78's code is reused verbatim; the one change is that Day 78's
tagger takes its block class as a setting. Every line of code below
carries its own comment, continuing Day 52's convention.
"""
import copy                                            # copy: deepcopy, to keep a snapshot of the best weights during training
import time                                            # time: perf_counter, to time training steps
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
# 2. THE ENCODER BLOCK -- NEW
# =============================================================================
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


class EncoderTagger:                                    # NEW: words -> N encoder blocks -> a tag for every word
    def __init__(self, vocab_size, n_tags, n_blocks=1, pre_ln=False, pe_every_block=True, lr=1.0, clip=1.0, seed=0,
                 block_cls=EncoderBlock, **block_kw):  # CHANGED (Day 79): the block class is a setting
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.n_tags, self.pre, self.pe_every, self.lr, self.clip = n_tags, pre_ln, pe_every_block, lr, clip  # settings
        self.P = dict(E=rng.randn(vocab_size, D) * 0.1)    # embedding table, as on Days 62-77
        self.blocks = [block_cls(rng, pre_ln=pre_ln, **block_kw) for _ in range(n_blocks)]  # the stack
        self.P.update(Wout=rng.randn(D, n_tags) / np.sqrt(D), bout=np.zeros(n_tags))  # a tag for every position
        if pre_ln:                                           # pre-LN ends with one more LayerNorm
            self.P.update(gf=np.ones(D), bf=np.zeros(D))       # starts as "no change"
        self.PE = sinusoidal_encoding(MAX_LEN, D) * PE_SCALE  # Day 77's encoding and strength

    def params(self):                                      # every trainable array, by name (the blocks' prefixed by their index)
        out = dict(self.P)                                   # the tagger's own
        for i, b in enumerate(self.blocks):                  # each block
            out.update({f"{i}.{k}": v for k, v in b.P.items()})  # e.g. "0.Wq"
        return out                                        # the same arrays, not copies

    def forward(self, ids):                                # ids (m, T) -> log-probabilities of the tags, one row per position
        m, T = ids.shape                                     # batch, padded length
        self.ids = ids                                       # kept for the backward pass
        mask = ids != PAD                                    # real words
        kmask = np.repeat(mask[:, None, None, :], HEADS, axis=1).reshape(m * HEADS, 1, T)  # padded keys, per head
        x = self.P["E"][ids]                                 # word vectors
        for i, b in enumerate(self.blocks):                  # through the stack
            pe = self.PE[:T][None] if (i == 0 or self.pe_every) else 0.0  # positions in every block, or only the first
            x = b.forward(x, pe, kmask)                        # one block
        if self.pre:                                         # pre-LN's final norm
            x, self.nf = layer_norm(x, self.P["gf"], self.P["bf"])  # normalise the last block's output
        self.y = x                                           # the stack's output
        logits = x.reshape(m * T, -1) @ self.P["Wout"] + self.P["bout"]  # a score per tag
        self.logp = logits - logits.max(axis=1, keepdims=True)  # NEW: log-softmax, computed without taking log of a probability
        self.logp -= np.log(np.exp(self.logp).sum(axis=1, keepdims=True))  # ... so the loss needs no "+ 1e-9" (Section 3)
        return self.logp                                  # hand back

    def loss(self, ids, tags, mask):                       # Day 62's mean cross-entropy over the real words
        logp = self.forward(ids)[np.arange(tags.size), tags.ravel()]  # log-probability of each word's true tag
        return -np.sum(logp * mask.ravel()) / mask.sum()   # padding counts 0

    def gradients(self, tags, mask):                       # every gradient of that loss
        m, T = self.ids.shape                                # batch, padded length
        dlog = (np.exp(self.logp) - np.eye(self.n_tags)[tags.ravel()]) * mask.ravel()[:, None] / mask.sum()  # softmax + CE
        G = dict(Wout=self.y.reshape(m * T, -1).T @ dlog, bout=dlog.sum(axis=0))  # output layer
        dx = (dlog @ self.P["Wout"].T).reshape(m, T, -1)     # gradient at the stack's output
        if self.pre:                                         # back through the final norm
            dx, G["gf"], G["bf"] = layer_norm_backward(dx, self.P["gf"], self.nf)  # LayerNorm's backward
        for i in reversed(range(len(self.blocks))):          # back through the stack, last block first
            dx, g = self.blocks[i].backward(dx)                # one block
            G.update({f"{i}.{k}": v for k, v in g.items()})    # its gradients, prefixed
        G["E"] = np.zeros_like(self.P["E"]); np.add.at(G["E"], self.ids, dx)  # Day 61
        return G                                          # dict: name -> gradient

    def train_step(self, ids, tags, mask):                 # one SGD step with Day 60's clipping
        self.forward(ids)                                    # forward pass
        G, P = self.gradients(tags, mask), self.params()     # gradients, and the arrays they belong to
        norm = np.sqrt(sum(np.sum(v ** 2) for v in G.values()))  # global norm
        for k, v in G.items():                               # plain SGD
            P[k] -= self.lr * min(1.0, self.clip / norm) * v   # in place, shrunk only if too long

    def predict(self, ids):                                # predicted tag for every position, in batches of 256 sentences (Day 77)
        out = np.zeros(ids.shape, dtype=int)                 # predictions, padding left at 0
        for s in range(0, len(ids), 256):                    # batches keep the attention arrays small
            b = ids[s:s + 256]                                 # these sentences
            T = (b != PAD).sum(axis=1).max()                   # pad only to the longest one
            out[s:s + 256, :T] = self.forward(b[:, :T]).argmax(axis=1).reshape(len(b), T)  # most probable tag
        return out                                        # hand back

def fit(data, vocab_size, n_tags, seed, epochs=EPOCHS, batch_size=32, **kw):  # Day 78's fit(), with NormTagger and the batch size as a setting
    (ids_tr, tags_tr, mask_tr), (ids_va, tags_va, mask_va), (ids_te, tags_te, mask_te) = data  # the three splits
    net = NormTagger(vocab_size, n_tags, seed=seed, **kw)  # fresh network
    rng = np.random.RandomState(seed)                       # shuffles the sentences each epoch (Day 62)
    lens = mask_tr.sum(axis=1)                              # real length of every training sentence
    best_val, best, best_epoch = -1.0, None, -1             # the best validation snapshot
    for epoch in range(epochs):                           # full training budget
        perm = rng.permutation(len(ids_tr))                    # fresh sentence order
        for start in range(0, len(perm), batch_size):          # mini-batches
            idx = perm[start:start + batch_size]                 # this batch's sentences
            T = lens[idx].max()                                  # pad only to the longest sentence in this batch
            net.train_step(ids_tr[idx, :T], tags_tr[idx, :T], mask_tr[idx, :T])  # one SGD step
        val = token_accuracy(net.predict(ids_va), tags_va, mask_va)  # validation accuracy now (running statistics for BN)
        if val > best_val:                                     # a new best on VALIDATION
            best_val, best, best_epoch = val, net.snapshot(), epoch  # weights AND running statistics
    net.restore(best)                                       # keep the best snapshot
    acc = lambda split: token_accuracy(net.predict(split[0]), split[1], split[2])  # accuracy on one split
    return dict(net=net, val=best_val, epoch=best_epoch, test=acc(data[2]), train=acc(data[0]))  # everything later sections need


RUNS = {}                                               # cache: each configuration trains once


def fit_cached(data, vocab_size, n_tags, seed, **kw):   # fit(), remembered by its settings (Day 67)
    key = (seed, tuple(sorted(kw.items())))               # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = fit(data, vocab_size, n_tags, seed, **kw)  # train and remember
    return RUNS[key]                                      # later requests reuse it


# =============================================================================
# 1. BATCH NORMALISATION -- NEW, and the block with the norm as a setting
# =============================================================================
def batch_norm(x, g, b, mask, eps=1e-5):                # NEW: each FEATURE to mean 0, spread 1 over the batch's real positions
    real = x[mask]                                        # (N, d): every real position of every sentence in the batch
    mu, var = real.mean(axis=0), real.var(axis=0)         # one mean and one variance per feature
    sd = np.sqrt(var + eps)                               # spread (eps keeps it above 0)
    xhat = (x - mu) / sd                                  # every position, padded ones too, uses the same statistics
    return xhat * g + b, (xhat, sd, mask), mu, var      # output, cache, and the statistics (for the running averages)


def batch_norm_backward(dy, g, cache):                  # NEW: gradients for x, g and b
    xhat, sd, mask = cache                                # what the forward pass computed; mask = positions that shaped the statistics
    d, N = xhat.shape[-1], mask.sum()                     # features, real positions
    dxhat = dy * g                                        # through the scale
    mean_d = dxhat.reshape(-1, d).sum(axis=0) / N         # the statistics were means over N real positions ...
    mean_dx = (dxhat * xhat).reshape(-1, d).sum(axis=0) / N  # ... and every position used them
    dx = (dxhat - mask[..., None] * (mean_d + xhat * mean_dx)) / sd  # padded positions get only the direct path
    return dx, (dy * xhat).reshape(-1, d).sum(axis=0), dy.reshape(-1, d).sum(axis=0)  # dx, dg, db


class NormBlock(EncoderBlock):                          # NEW: Day 78's block with the norm as a setting
    def __init__(self, rng, kind="layer", momentum=0.9, **kw):
        super().__init__(rng, **kw)                          # Day 78's weights, the same random draws
        self.kind, self.momentum = kind, momentum            # "layer", "batch" (padding left out) or "batch_all" (padding counted)
        self.training, self.update_running = True, True      # train mode: batch statistics; eval mode: running averages
        self.running = {w: (np.zeros(D), np.ones(D)) for w in ("1", "2")}  # running mean and variance per norm
        self.pos_mask = None                                 # real positions of the current batch, set by the tagger

    def _norm(self, x, which):                             # LayerNorm (Day 78) or BatchNorm
        if self.kind == "layer":                             # Day 78's path
            return super()._norm(x, which)                     # unchanged
        g, b = self.P["g" + which], self.P["b" + which]      # learned scale and shift
        mask = self.pos_mask if self.kind == "batch" else np.ones(x.shape[:2], bool)  # which positions count
        if not self.training:                                # eval mode: the running averages, no batch statistics
            mu, var = self.running[which]                      # learned during training
            return (x - mu) / np.sqrt(var + 1e-5) * g + b, None  # no backward pass in eval mode
        y, cache, mu, var = batch_norm(x, g, b, mask)        # train mode: this batch's statistics
        if self.update_running:                              # keep running averages for eval mode
            rm, rv = self.running[which]                       # the old averages
            self.running[which] = (self.momentum * rm + (1 - self.momentum) * mu, self.momentum * rv + (1 - self.momentum) * var)  # move 10% toward this batch
        return y, cache                                    # the mask travels inside the cache

    def _norm_back(self, dy, which, cache, g):             # the backward pass of _norm
        if self.kind == "layer":                             # Day 78's path
            return super()._norm_back(dy, which, cache, g)     # unchanged
        dx, g["g" + which], g["b" + which] = batch_norm_backward(dy, self.P["g" + which], cache)  # BatchNorm's backward
        return dx                                          # gradient at the norm's input


class NormTagger(EncoderTagger):                        # NEW: Day 78's tagger built from NormBlocks
    def __init__(self, vocab_size, n_tags, **kw):
        super().__init__(vocab_size, n_tags, block_cls=NormBlock, **kw)  # Day 78's tagger, same weights

    def set_mode(self, training, update_running=True):     # train or eval mode, for every block
        for b in self.blocks:                                # each block
            b.training, b.update_running = training, update_running  # BatchNorm reads these

    def forward(self, ids):                                # tell every block where the real words are, then Day 78's forward
        for b in self.blocks:                                # each block
            b.pos_mask = ids != PAD                            # real positions
        return super().forward(ids)                       # unchanged

    def predict(self, ids, batch_stats=False):             # eval mode by default: BatchNorm uses its running averages
        self.set_mode(training=batch_stats, update_running=False)  # batch_stats=True: each test batch's own statistics
        out = super().predict(ids)                           # Day 78's batched prediction
        self.set_mode(training=True)                         # back to train mode
        return out                                        # hand back

    def snapshot(self):                                    # weights and running averages
        return copy.deepcopy(self.params()), [copy.deepcopy(b.running) for b in self.blocks]  # independent copies

    def restore(self, snap):                               # put a snapshot back
        P, running = snap                                    # its two parts
        for k, v in P.items():                               # weights, in place
            self.params()[k][...] = v                          # the blocks see it too
        for b, r in zip(self.blocks, running):               # running averages
            b.running = r                                      # replace


# =============================================================================
# DEMOS
# =============================================================================
def demo_by_hand():
    print("=" * 70)
    print("1. TWO WAYS TO NORMALISE THE SAME NUMBERS")
    print("=" * 70)
    print("A batch of 2 sentences x 3 positions x 4 features. LayerNorm makes every POSITION's 4 numbers "
          "mean 0 and spread 1; BatchNorm makes every FEATURE's 6 numbers (all positions of the batch) "
          "mean 0 and spread 1. g = 1, b = 0.")
    x = np.arange(24, dtype=float).reshape(2, 3, 4) ** 1.5 / 10  # numbers that differ along both axes
    ln, _ = layer_norm(x, np.ones(4), np.zeros(4))        # Day 78's LayerNorm
    bn, _, _, _ = batch_norm(x, np.ones(4), np.zeros(4), np.ones((2, 3), bool))  # BatchNorm, every position real
    print("\nLayerNorm: mean and spread of each position's 4 numbers")
    print("  means  " + " ".join(f"{v:7.3f}" for v in ln.mean(axis=2).ravel()) + "   spreads " + " ".join(f"{v:.3f}" for v in ln.std(axis=2).ravel()))
    print("BatchNorm: mean and spread of each feature's 6 numbers")
    print("  means  " + " ".join(f"{v:7.3f}" for v in bn.reshape(-1, 4).mean(axis=0)) + "   spreads " + " ".join(f"{v:.3f}" for v in bn.reshape(-1, 4).std(axis=0)))
    print("BatchNorm's positions are NOT normalised one by one: position means " + " ".join(f"{v:.3f}" for v in bn.mean(axis=2).ravel()))
    print("Actual output from this lesson's run.")


def demo_batch_dependence(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("2. DOES A SENTENCE'S OUTPUT DEPEND ON THE OTHER SENTENCES IN ITS BATCH?")
    print("=" * 70)
    print("One untrained block (seed 0, the same weights for every norm). The first test sentence is put in "
          "a batch with 7 other test sentences, then with 7 different ones. Reported: the largest change in "
          "its tag log-probabilities.")
    ids = data[2][0]                                      # test word ids
    rng = np.random.RandomState(4)                        # fixed choice of companions
    others = rng.permutation(np.arange(1, len(ids)))[:14]  # 14 other sentences
    print(f"\n{'norm':<28}{'train mode':>14}{'eval mode':>14}")
    for kind, label in [("layer", "LayerNorm"), ("batch", "BatchNorm (padding left out)"), ("batch_all", "BatchNorm (padding counted)")]:
        net = NormTagger(len(itos), len(tag_names), kind=kind, seed=0)  # untrained
        row = []                                             # train mode, eval mode
        for training in (True, False):                       # both modes
            net.set_mode(training, update_running=False)       # do not touch the running averages
            outs = []                                          # the first sentence's output in each batch
            for comp in (others[:7], others[7:]):              # two different sets of companions
                b = ids[np.concatenate([[0], comp])]             # sentence 0 first
                T = (b != PAD).sum(axis=1).max(); n0 = (b[0] != PAD).sum()  # padded width; sentence 0's length
                outs.append(net.forward(b[:, :T]).reshape(len(b), T, -1)[0, :n0])  # its real words only
            row.append(np.abs(outs[0] - outs[1]).max())        # how much the companions changed it
        print(f"{label:<28}{row[0]:>14.2e}{row[1]:>14.2e}")
    print("Actual output from this lesson's run.")


def demo_padding(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("3. DOES PADDING CHANGE THE OUTPUT?")
    print("=" * 70)
    print("The same 8 test sentences, padded to their own longest (as in training) and then with 20 more "
          "padding positions. Train mode, one untrained block. Reported: the largest change in any real "
          "word's tag log-probabilities.")
    ids = data[2][0][:8]                                  # 8 test sentences
    T = (ids != PAD).sum(axis=1).max()                    # their longest
    wide = np.zeros((8, T + 20), dtype=int); wide[:, :T] = ids[:, :T]  # 20 extra padding positions
    real = (ids[:, :T] != PAD).ravel()                    # real words, in the narrow layout
    print(f"\n{'norm':<28}{'largest change':>16}")
    for kind, label in [("layer", "LayerNorm"), ("batch", "BatchNorm (padding left out)"), ("batch_all", "BatchNorm (padding counted)")]:
        net = NormTagger(len(itos), len(tag_names), kind=kind, seed=0)  # untrained
        net.set_mode(True, update_running=False)             # train mode, running averages untouched
        a = net.forward(ids[:, :T])                          # narrow
        b = net.forward(wide).reshape(8, T + 20, -1)[:, :T].reshape(8 * T, -1)  # wide, cut back to the narrow layout
        print(f"{label:<28}{np.abs(a - b)[real].max():>16.2e}")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("4. GRADIENT CHECK -- BatchNorm inside the block, train mode")
    print("=" * 70)
    print("Day 78's tiny tagger (size 8, 2 heads, FFN 12, positions x 1), three padded sentences; every "
          "entry nudged by +-1e-5 (running averages not updated during the check); Day 62's pass rule.")
    global D, F, HEADS, PE_SCALE                          # the tiny sizes, for this check only
    saved = (D, F, HEADS, PE_SCALE)                       # restored afterwards
    D, F, HEADS, PE_SCALE = 8, 12, 2, 1.0                 # tiny
    rng = np.random.RandomState(3)                        # fixed rng
    ids = np.array([[2, 3, 4, 5, 6], [7, 2, 3, 0, 0], [4, 4, 5, 6, 0]])  # three sentences, padded
    tags = rng.randint(0, 3, size=ids.shape); mask = ids != PAD  # random tags, the real words
    for kind, kw in [("layer", {}), ("batch", {}), ("batch_all", {}), ("batch", dict(n_blocks=2, pre_ln=True))]:
        net = NormTagger(8, 3, kind=kind, seed=1, **kw)     # tiny tagger
        net.set_mode(True, update_running=False)            # batch statistics, averages untouched
        net.forward(ids); G = net.gradients(tags, mask)     # analytic gradients
        worst, ok = 0.0, True                               # largest difference; every entry passes?
        for k, A in net.params().items():                   # every parameter
            for pos in np.ndindex(A.shape):                   # every entry
                old = A[pos]                                    # remember it
                A[pos] = old + 1e-5; up = net.loss(ids, tags, mask)  # nudge up
                A[pos] = old - 1e-5; dn = net.loss(ids, tags, mask)  # nudge down
                A[pos] = old                                    # restore
                num = (up - dn) / 2e-5                          # central difference
                worst = max(worst, abs(num - G[k][pos]))        # largest difference so far
                ok &= abs(num - G[k][pos]) <= 1e-10 + 1e-6 * abs(num)  # Day 62's rule
        label = kind + (", 2 pre-LN blocks" if kw else "")    # row label
        print(f"{label:<26}: largest |numeric - analytic| = {worst:.2e}; every entry passes: {ok}")
    D, F, HEADS, PE_SCALE = saved                         # back to the lesson's sizes
    print("Actual output from this lesson's run.")


def demo_training(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("5. TRAINING -- one block, batches of 32, three seeds")
    print("=" * 70)
    print(f"Day 78's tagger and settings (lr {LR}, Day 78's choice on validation; {EPOCHS} epochs, best epoch "
          "on validation). BatchNorm keeps running averages (momentum 0.9) and uses them at test time. "
          "Test accuracy; training accuracy at the best epoch.")
    V, nt = len(itos), len(tag_names)                     # sizes
    print(f"\n{'norm':<30}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}{'train':>9}")
    table = {}                                             # label -> mean test accuracy
    for kind, label in [("layer", "LayerNorm"), ("batch", "BatchNorm (padding left out)"), ("batch_all", "BatchNorm (padding counted)")]:
        runs = [fit_cached(data, V, nt, s, kind=kind, lr=LR) for s in SEEDS]  # three seeds
        table[label] = [r["test"] for r in runs]             # per seed
        print(f"{label:<30}" + "".join(f"{r['test']:>9.4f}" for r in runs) + f"{np.mean(table[label]):>9.4f}{np.mean([r['train'] for r in runs]):>9.4f}")
    print("Actual output from this lesson's run.")
    return table                                           # the plot reuses it


def demo_small_batch(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("6. SMALL BATCHES -- 4 sentences per batch, seed 0")
    print("=" * 70)
    print("The same tagger with batches of 4: BatchNorm's statistics now come from about 50 words. Same "
          f"number of epochs (so 8 times as many steps). First at the same learning rate ({LR}); then at "
          f"{LR / 8}, eight times smaller, so that an epoch moves the weights about as far as with batches of 32.")
    V, nt = len(itos), len(tag_names)                     # sizes
    print(f"\n{'norm':<30}{'batch 32, lr ' + str(LR):>18}{'batch 4, lr ' + str(LR):>17}{'batch 4, lr ' + str(LR / 8):>18}")
    small = {}                                             # label -> (batch 32, batch 4, batch 4 smaller lr)
    for kind, label in [("layer", "LayerNorm"), ("batch", "BatchNorm (padding left out)")]:
        a = fit_cached(data, V, nt, 0, kind=kind, lr=LR)["test"]  # Section 5's seed 0
        b = fit_cached(data, V, nt, 0, kind=kind, lr=LR, batch_size=4)["test"]  # batches of 4, same rate
        c = fit_cached(data, V, nt, 0, kind=kind, lr=LR / 8, batch_size=4)["test"]  # batches of 4, rate / 8
        small[label] = (a, b, c)                             # for the plot
        print(f"{label:<30}{a:>18.4f}{b:>17.4f}{c:>18.4f}")
    print("Actual output from this lesson's run.")
    return small                                           # the plot reuses it


def demo_train_vs_eval(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("7. TRAINING STATISTICS OR RUNNING AVERAGES? -- the trained BatchNorm tagger, seed 0")
    print("=" * 70)
    print("The same trained weights, three ways of normalising at test time: the running averages "
          "(what Section 5 reports), or each test batch's own statistics with batches of 256 or of 4 "
          "sentences. LayerNorm has no such choice.")
    V, nt = len(itos), len(tag_names)                     # sizes
    ids, tags, mask = data[2]                             # the test split
    net = fit_cached(data, V, nt, 0, kind="batch", lr=LR)["net"]  # Section 5's seed 0
    run = token_accuracy(net.predict(ids), tags, mask)    # running averages
    b256 = token_accuracy(net.predict(ids, batch_stats=True), tags, mask)  # predict() batches 256 sentences at a time
    pred4 = np.zeros(ids.shape, dtype=int)                # batches of 4
    net.set_mode(True, update_running=False)              # batch statistics
    for s in range(0, len(ids), 4):                       # 4 sentences at a time
        b = ids[s:s + 4]; T = (b != PAD).sum(axis=1).max()  # pad to the longest of the 4
        pred4[s:s + 4, :T] = net.forward(b[:, :T]).argmax(axis=1).reshape(len(b), T)  # most probable tag
    net.set_mode(True)                                    # back to normal
    b4 = token_accuracy(pred4, tags, mask)                # accuracy
    print(f"\nrunning averages: {run:.4f}   test batches of 256: {b256:.4f}   test batches of 4: {b4:.4f}")
    print("Actual output from this lesson's run.")
    return run, b256, b4                                   # the plot reuses them


def demo_cost(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("8. WHAT EACH NORM COSTS -- one training step, measured in this run")
    print("=" * 70)
    print("One block, a batch of 32 training sentences, forward and backward; best of 5 timings of 10 steps. "
          "Only the ratio within this run is meaningful.")
    ids, tags, mask = data[0]                             # training split
    b, t, mk = ids[:32], tags[:32], mask[:32]; T = mk.sum(axis=1).max()  # one batch
    times = {}                                             # kind -> seconds per step
    for kind in ("layer", "batch"):                        # the two norms
        net = NormTagger(len(itos), len(tag_names), kind=kind, seed=0)  # fresh
        best = np.inf                                        # fastest repeat
        for _ in range(5):                                   # 5 repeats
            start = time.perf_counter()                        # clock
            for _ in range(10):                                # 10 steps
                net.forward(b[:, :T]); net.gradients(t[:, :T], mk[:, :T])  # forward and backward
            best = min(best, (time.perf_counter() - start) / 10)  # per step
        times[kind] = best                                   # keep it
    print(f"\nLayerNorm {times['layer'] * 1000:.1f} ms, BatchNorm {times['batch'] * 1000:.1f} ms per step; ratio {times['batch'] / times['layer']:.2f}")
    print("Actual output from this lesson's run.")


def plot_results(table, small, tve):                    # three panels: batch 32 | batch 4 | test-time statistics
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))          # canvas
    labels = list(table)                                    # panel 1: three norms, three seeds
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]              # one colour per norm
    for i, k in enumerate(labels):                          # each norm
        ax[0].scatter([i] * 3, table[k], color=colors[i], s=30)  # the seeds
        ax[0].bar(i, np.mean(table[k]), color=colors[i], alpha=0.35)  # their mean
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["LayerNorm", "BN, padding\nleft out", "BN, padding\ncounted"], fontsize=8)
    lo = min(min(v) for v in table.values()) - 0.02         # zoom in on the differences
    ax[0].set_ylim(lo, max(max(v) for v in table.values()) + 0.01); ax[0].set_ylabel("test accuracy")
    ax[0].set_title("Batches of 32 (three seeds, bar = mean)", fontsize=10)
    names = list(small)                                     # panel 2: batch 32 vs 4
    x = np.arange(len(names))                               # bar positions
    for j, (lab, col) in enumerate([(f"batch 32, lr {LR}", "#c6dbef"), (f"batch 4, lr {LR}", "#6baed6"), (f"batch 4, lr {LR / 8}", "#08519c")]):
        ax[1].bar(x + (j - 1) * 0.27, [small[k][j] for k in names], width=0.27, color=col, label=lab)  # one bar per setting
    ax[1].set_xticks(x); ax[1].set_xticklabels(["LayerNorm", "BatchNorm"], fontsize=9)
    vals = [v for k in names for v in small[k]]             # every bar
    ax[1].set_ylim(min(vals) - 0.03, max(vals) + 0.035); ax[1].set_ylabel("test accuracy (seed 0)")
    ax[1].set_title("Smaller batches", fontsize=10); ax[1].legend(fontsize=8, loc="upper center", ncol=3)
    ax[2].bar(range(3), tve, color=["#2ca02c", "#98df8a", "#d62728"])  # panel 3: test-time statistics
    for i, v in enumerate(tve):                             # the value on each bar
        ax[2].text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["running\naverages", "test batches\nof 256", "test batches\nof 4"], fontsize=8)
    ax[2].set_ylim(min(tve) - 0.05, max(tve) + 0.03); ax[2].set_ylabel("test accuracy (BatchNorm, seed 0)")
    ax[2].set_title("BatchNorm at test time", fontsize=10)
    plt.tight_layout(); plt.savefig("layernorm_vs_batchnorm.png", dpi=110); plt.close()  # save and free
    print("\nsaved layernorm_vs_batchnorm.png")


def draw_diagram():                                     # which numbers each norm averages over
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))             # two panels
    for a, title, axis in [(ax[0], "LayerNorm: one position's features", "row"), (ax[1], "BatchNorm: one feature over the batch", "col")]:
        a.set_xlim(0, 8.5); a.set_ylim(0, 6.6); a.axis("off")   # coordinates in "cells"
        a.text(4.25, 6.25, title, ha="center", fontsize=11, fontweight="bold")
        for s in range(2):                                      # two sentences
            for p in range(3):                                    # three positions each
                y = 5.0 - (s * 3 + p) * 0.8                         # one row per position
                pad = s == 1 and p == 2                             # the last position of sentence 2 is padding
                for f in range(6):                                  # six features
                    hit = (axis == "row" and s == 0 and p == 1) or (axis == "col" and f == 2 and not pad)  # the averaged cells
                    color = "#F5B971" if hit else ("#EEEEEE" if pad else "#DCEBF7")
                    a.add_patch(FancyBboxPatch((1.6 + f * 1.0, y - 0.32), 0.85, 0.64, boxstyle="round,pad=0.02",
                                               facecolor=color, edgecolor="#333333", linewidth=0.8))
                a.text(1.4, y, f"sentence {s + 1}, word {p + 1}" + (" (pad)" if pad else ""), ha="right", va="center", fontsize=7.5)
        a.text(4.6, 0.25, "orange = the numbers averaged together; grey = padding", ha="center", fontsize=8, style="italic")
    plt.savefig("norms_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved norms_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("9. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 80 builds the Transformer's DECODER block: self-attention that may only look at earlier "
          "positions (a causal mask), attention to the encoder's output, and the same add & norm.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_by_hand()                                       # 1. the two norms by hand
        itos, tag_names, data = prepare_data()               # Day 62's data
        demo_batch_dependence(itos, tag_names, data)         # 2. the batch matters to BatchNorm
        demo_padding(itos, tag_names, data)                  # 3. so does padding
        demo_gradient_check()                                # 4. gradients
        table = demo_training(itos, tag_names, data)         # 5. training
        small = demo_small_batch(itos, tag_names, data)      # 6. small batches
        tve = demo_train_vs_eval(itos, tag_names, data)      # 7. test-time statistics
        demo_cost(itos, tag_names, data)                     # 8. cost
        plot_results(table, small, tve)                      # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 9. Day 80


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
