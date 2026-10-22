"""
Day 78: The Transformer Encoder Block, From Scratch

Day 77's tagger was one attention layer and a small classifier. The
Transformer wraps attention in a BLOCK that can be stacked:

    h = LayerNorm( x + MultiHeadAttention(x) )        "add & norm"
    y = LayerNorm( h + FFN(h) )                        FFN(h) = ReLU(h W1 + c1) W2 + c2, at every position

That is the original ("post-LN") order. Most later models move the norm
in front of each part ("pre-LN"):

    h = x + MultiHeadAttention( LayerNorm(x) )
    y = h + FFN( LayerNorm(h) )                        and one LayerNorm after the last block

Three new parts: the RESIDUAL connection (x + ...), LAYER NORMALISATION
(each position's vector rescaled to mean 0 and spread 1, then a learned
scale and shift) and the position-wise FEED-FORWARD network.

What is built and measured, reported as it came out:

  1. Layer normalisation by hand.
  2. The feed-forward network works on every position on its own.
  3. A gradient check of post-LN and pre-LN stacks, and of a block with
     each part removed.
  4. Why the residual: the gradient that reaches the input through 1 to
     8 untrained blocks, with and without it.
  5. One block on Week 9's real tagging task, three seeds.
  6. Each part removed in turn.
  7. Deeper: 2 and 4 blocks, post-LN and pre-LN, and where the
     positional encoding goes.

Every line of code below carries its own comment, continuing Day 52's
convention.
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
LRS = (0.5, 1.0, 2.0)                                   # learning rates tried on validation for one block
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
    def __init__(self, vocab_size, n_tags, n_blocks=1, pre_ln=False, pe_every_block=True, lr=1.0, clip=1.0, seed=0, **block_kw):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.n_tags, self.pre, self.pe_every, self.lr, self.clip = n_tags, pre_ln, pe_every_block, lr, clip  # settings
        self.P = dict(E=rng.randn(vocab_size, D) * 0.1)    # embedding table, as on Days 62-77
        self.blocks = [EncoderBlock(rng, pre_ln=pre_ln, **block_kw) for _ in range(n_blocks)]  # the stack
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


def fit(data, vocab_size, n_tags, seed, epochs=EPOCHS, batch_size=32, **kw):  # Day 77's fit(): best epoch on VALIDATION
    (ids_tr, tags_tr, mask_tr), (ids_va, tags_va, mask_va), (ids_te, tags_te, mask_te) = data  # the three splits
    net = EncoderTagger(vocab_size, n_tags, seed=seed, **kw)  # fresh network
    rng = np.random.RandomState(seed)                       # shuffles the sentences each epoch (Day 62)
    lens = mask_tr.sum(axis=1)                              # real length of every training sentence
    best_val, best_P, best_epoch = -1.0, None, -1           # the best validation snapshot
    for epoch in range(epochs):                           # full training budget
        perm = rng.permutation(len(ids_tr))                    # fresh sentence order
        for start in range(0, len(perm), batch_size):          # mini-batches
            idx = perm[start:start + batch_size]                 # this batch's sentences
            T = lens[idx].max()                                  # pad only to the longest sentence in this batch
            net.train_step(ids_tr[idx, :T], tags_tr[idx, :T], mask_tr[idx, :T])  # one SGD step
        val = token_accuracy(net.predict(ids_va), tags_va, mask_va)  # validation accuracy now
        if val > best_val:                                     # a new best on VALIDATION
            best_val, best_P, best_epoch = val, copy.deepcopy(net.params()), epoch  # snapshot every weight
    for k, v in best_P.items():                             # restore the best snapshot ...
        net.params()[k][...] = v                              # ... in place, so the blocks see it too
    acc = lambda split: token_accuracy(net.predict(split[0]), split[1], split[2])  # accuracy on one split
    return dict(net=net, val=best_val, epoch=best_epoch, test=acc(data[2]), train=acc(data[0]))  # everything later sections need


RUNS = {}                                               # cache: each configuration trains once


def fit_cached(data, vocab_size, n_tags, seed, **kw):   # fit(), remembered by its settings (Day 67)
    key = (seed, tuple(sorted(kw.items())))               # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = fit(data, vocab_size, n_tags, seed, **kw)  # train and remember
    return RUNS[key]                                      # later requests reuse it


# =============================================================================
# DEMOS
# =============================================================================
def demo_layer_norm():
    print("=" * 70)
    print("1. LAYER NORMALISATION BY HAND -- each position's vector on its own")
    print("=" * 70)
    print("Two positions with 4 numbers each, g = 1 and b = 0 (where training starts). Each row is "
          "shifted to mean 0 and divided by its spread; the other row plays no part.")
    x = np.array([[1.0, 2.0, 3.0, 6.0], [10.0, 10.0, 10.0, 14.0]])  # two positions
    y, _ = layer_norm(x, np.ones(4), np.zeros(4))         # normalise
    mu, sd = x.mean(axis=1), x.std(axis=1)                # by hand: mean and spread of each row
    print(f"\n{'row':>4}{'input':>26}{'mean':>7}{'spread':>8}{'output':>34}")
    for r in range(2):                                    # each row
        print(f"{r:>4}{str(x[r].tolist()):>26}{mu[r]:>7.2f}{sd[r]:>8.4f}   " + " ".join(f"{v:7.4f}" for v in y[r]))
    by_hand = (x - mu[:, None]) / np.sqrt(sd[:, None] ** 2 + 1e-5)  # the formula, written out
    print(f"largest difference from the formula: {np.abs(y - by_hand).max():.2e}; output means "
          f"{y.mean(axis=1).round(6).tolist()}, spreads {y.std(axis=1).round(4).tolist()}")
    print("Actual output from this lesson's run.")


def demo_ffn():
    print("\n" + "=" * 70)
    print("2. THE FEED-FORWARD NETWORK WORKS ON EVERY POSITION ON ITS OWN")
    print("=" * 70)
    print("Random input (3 sentences, 6 positions, size 32) and random weights. Shuffle the positions: "
          "if the FFN mixes nothing between positions, its outputs shuffle with them exactly. And one "
          "position fed alone gives the same output as inside the sentence.")
    rng = np.random.RandomState(0)                        # fixed rng
    x = rng.randn(3, 6, D)                                # inputs
    P = dict(W1=rng.randn(D, F) / np.sqrt(D), c1=np.zeros(F), W2=rng.randn(F, D) / np.sqrt(F), c2=np.zeros(D))  # random FFN
    perm = rng.permutation(6)                             # one shuffle
    out, _ = ffn(x, P)                                    # all positions
    out_s, _ = ffn(x[:, perm], P)                         # shuffled
    alone, _ = ffn(x[:, 2:3], P)                          # position 2 on its own
    print(f"\nshuffled: largest |output(shuffled) - shuffled(output)| = {np.abs(out_s - out[:, perm]).max():.2e}")
    print(f"position 2 alone vs inside the sentence: largest difference = {np.abs(alone - out[:, 2:3]).max():.2e}")
    print(f"weights: W1 {D} x {F}, W2 {F} x {D} -- the same {D * F * 2 + F + D} numbers at every position")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("3. GRADIENT CHECK -- post-LN and pre-LN stacks, and each part removed")
    print("=" * 70)
    print("A tiny tagger (8 words, 3 tags, size 8, 2 heads, FFN size 12, positions x 1), three padded "
          "sentences; every entry of every parameter nudged by +-1e-5; Day 62's pass rule "
          "|numeric - analytic| <= 1e-10 + 1e-6 |numeric|. The loss is a log-softmax: no '+ 1e-9'.")
    global D, F, HEADS, PE_SCALE                          # the tiny sizes, for this check only
    saved = (D, F, HEADS, PE_SCALE)                       # restored afterwards
    D, F, HEADS, PE_SCALE = 8, 12, 2, 1.0                 # tiny
    rng = np.random.RandomState(3)                        # fixed rng
    ids = np.array([[2, 3, 4, 5, 6], [7, 2, 3, 0, 0], [4, 4, 5, 6, 0]])  # three sentences, padded
    tags = rng.randint(0, 3, size=ids.shape); mask = ids != PAD  # random tags, the real words
    cases = [("2 blocks, post-LN", dict(n_blocks=2)), ("2 blocks, pre-LN", dict(n_blocks=2, pre_ln=True)),
             ("1 block, no residual", dict(residual=False)), ("1 block, no norm", dict(norm=False)),
             ("1 block, no FFN", dict(use_ffn=False))]
    for label, kw in cases:                               # each model
        net = EncoderTagger(8, 3, seed=1, **kw)             # tiny tagger
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
        print(f"{label:<22}: largest |numeric - analytic| = {worst:.2e}; every entry passes: {ok}")
    D, F, HEADS, PE_SCALE = saved                         # back to the lesson's sizes
    print("Actual output from this lesson's run.")


def demo_residual():
    print("\n" + "=" * 70)
    print("4. WHAT KEEPS THE GRADIENT ALIVE -- residuals and LayerNorm, untrained blocks")
    print("=" * 70)
    print("A stack of 1 to 8 untrained blocks (seed 0) on 32 real training sentences; a random "
          "gradient of size 1 per real word arrives at the top. Reported: the mean size of the gradient "
          "that reaches the word vectors at the bottom. Five versions of the block: post-LN and pre-LN with "
          "residuals, post-LN without them, and the block with no LayerNorm, with and without residuals.")
    itos, _, data = prepare_data()                        # Day 62's data
    ids, _, mask = data[0]                                # training split
    b = ids[:32]; T = mask[:32].sum(axis=1).max(); b = b[:, :T]  # 32 sentences
    real = b != PAD                                       # real words
    rng = np.random.RandomState(1)                        # the same top gradient for every model
    top = rng.randn(len(b), T, D) * real[:, :, None]      # random on real words, zero on padding
    top /= np.sqrt((top ** 2).sum(axis=-1, keepdims=True) + 1e-12)  # size exactly 1 per real word
    kinds = [("post-LN, residual", dict()), ("post-LN, none", dict(residual=False)), ("pre-LN, residual", dict(pre_ln=True)),
             ("no LN, residual", dict(norm=False)), ("no LN, none", dict(norm=False, residual=False))]
    print(f"\n{'blocks':>7}" + "".join(f"{k:>19}" for k, _ in kinds))
    res = {k: [] for k, _ in kinds}                       # label -> sizes by depth
    for n in (1, 2, 4, 8):                                 # depths
        for label, kw in kinds:                              # each version
            net = EncoderTagger(len(itos), 17, n_blocks=n, seed=0, **kw)  # untrained
            net.P["E"] = np.random.RandomState(2).randn(*net.P["E"].shape) * 0.1  # the same word vectors for every model
            x = net.P["E"][b]                                  # word vectors
            kmask = np.repeat(real[:, None, None, :], HEADS, axis=1).reshape(len(b) * HEADS, 1, T)  # padded keys
            for blk in net.blocks:                             # forward through the stack
                x = blk.forward(x, net.PE[:T][None], kmask)      # one block
            dx = top                                           # the gradient arriving at the top (pre-LN's final norm left out)
            for blk in reversed(net.blocks):                   # back down
                dx, _ = blk.backward(dx)                         # one block
            res[label].append(np.sqrt((dx ** 2).sum(axis=-1))[real].mean())  # mean size at the bottom
        print(f"{n:>7}" + "".join(f"{res[k][-1]:>19.3e}" for k, _ in kinds))
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_one_block(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("5. ONE BLOCK ON REAL TAGGING -- learning rate on validation, then three seeds")
    print("=" * 70)
    print(f"Day 62's data. Embedding {D}, {HEADS} heads, FFN {F}, Day 77's encoding (x {PE_SCALE}, queries and "
          f"keys only), post-LN; plain SGD, clip 1.0, batches of 32, {EPOCHS} epochs, best epoch on "
          "validation. The tagger is the block plus one output layer: the FFN takes the place of Day 77's hidden layer.")
    V, nt = len(itos), len(tag_names)                     # sizes
    vals = [fit_cached(data, V, nt, 0, lr=lr)["val"] for lr in LRS]  # validation accuracy per learning rate
    lr = LRS[int(np.argmax(vals))]                        # chosen on validation
    print("\nvalidation accuracy, seed 0: " + "  ".join(f"lr {a}: {v:.4f}" for a, v in zip(LRS, vals)) + f"  -> lr {lr}")
    runs = [fit_cached(data, V, nt, s, lr=lr) for s in SEEDS]  # three seeds
    print(f"test accuracy: " + "  ".join(f"{r['test']:.4f}" for r in runs) + f"   mean {np.mean([r['test'] for r in runs]):.4f}"
          f"   (training accuracy {np.mean([r['train'] for r in runs]):.4f})")
    print("Day 77's single attention layer + hidden layer (same encoding): 0.8487")
    print("Actual output from this lesson's run.")
    return lr                                              # later sections use it


def demo_ablations(itos, tag_names, data, lr):
    print("\n" + "=" * 70)
    print("6. EACH PART REMOVED -- one block, three seeds")
    print("=" * 70)
    print("The same tagger with one part taken out at a time. Test accuracy (best epoch on validation), "
          "and training accuracy at that epoch.")
    V, nt = len(itos), len(tag_names)                     # sizes
    rows = [("full block", {}), ("no feed-forward", dict(use_ffn=False)), ("no LayerNorm", dict(norm=False)),
            ("no residual", dict(residual=False))]
    print(f"\n{'model':<18}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}{'train':>9}")
    table = {}                                             # label -> mean test accuracy
    for label, kw in rows:                                 # each model
        runs = [fit_cached(data, V, nt, s, lr=lr, **kw) for s in SEEDS]  # three seeds
        table[label] = np.mean([r["test"] for r in runs])    # mean test accuracy
        print(f"{label:<18}" + "".join(f"{r['test']:>9.4f}" for r in runs) + f"{table[label]:>9.4f}{np.mean([r['train'] for r in runs]):>9.4f}")
    print("Actual output from this lesson's run.")
    return table                                           # the plot reuses it


def demo_depth(itos, tag_names, data, lr):
    print("\n" + "=" * 70)
    print("7. DEEPER -- 2 and 4 blocks, post-LN and pre-LN, seed 0")
    print("=" * 70)
    print("The same settings and learning rate. Training accuracy is at the best validation epoch, to "
          "tell a model that cannot fit (low training accuracy) from one that fits but does not "
          "generalise. 'positions' says whether Day 77's encoding is added in every block or only the first.")
    V, nt = len(itos), len(tag_names)                     # sizes
    rows = [("1 block", dict()), ("2 blocks, post-LN", dict(n_blocks=2)), ("2 blocks, pre-LN", dict(n_blocks=2, pre_ln=True)),
            ("4 blocks, post-LN", dict(n_blocks=4)), ("4 blocks, pre-LN", dict(n_blocks=4, pre_ln=True)),
            ("2 blocks, post-LN, first", dict(n_blocks=2, pe_every_block=False)),
            ("4 blocks, post-LN, first", dict(n_blocks=4, pe_every_block=False))]
    print(f"\n{'model':<26}{'positions':>11}{'val':>9}{'test':>9}{'train':>9}{'best epoch':>12}")
    out = {}                                               # label -> (test, train)
    for label, kw in rows:                                 # each model
        r = fit_cached(data, V, nt, 0, lr=lr, **kw)          # seed 0
        out[label] = (r["test"], r["train"])                 # for the plot
        where = "first" if not kw.get("pe_every_block", True) else "every"  # where the encoding went
        print(f"{label.replace(', first', ''):<26}{where:>11}{r['val']:>9.4f}{r['test']:>9.4f}{r['train']:>9.4f}{r['epoch'] + 1:>12}")
    print("Actual output from this lesson's run.")
    return out                                             # the plot reuses it


def plot_results(res, table, depth):                    # three panels: gradient by depth | ablations | depth: train vs test
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))          # canvas
    ns = [1, 2, 4, 8]                                       # depths of Section 4
    styles = {"post-LN, residual": ("#1f77b4", "-"), "post-LN, none": ("#1f77b4", "--"), "pre-LN, residual": ("#d62728", "-"),
              "no LN, residual": ("#2ca02c", "-"), "no LN, none": ("#2ca02c", "--")}  # colour = norm, dashed = no residual
    for label, (color, ls) in styles.items():               # panel 1: gradient size by depth
        ax[0].plot(ns, res[label], marker="o", color=color, linestyle=ls, label=label)
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xlabel("untrained blocks")
    ax[0].set_ylabel("gradient size at the word vectors"); ax[0].set_title("Gradient reaching the bottom", fontsize=10)
    ax[0].legend(fontsize=7, loc="lower left"); ax[0].grid(alpha=0.3, which="both")
    labels = list(table)                                    # panel 2: ablations
    ax[1].bar(range(len(labels)), [table[k] for k in labels], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    for i, k in enumerate(labels):                          # the value on each bar
        ax[1].text(i, table[k] + 0.003, f"{table[k]:.4f}", ha="center", fontsize=8)
    lo = min(table.values()) - 0.03                         # zoom in on the differences
    ax[1].set_ylim(lo, max(table.values()) + 0.02); ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("test accuracy (mean of 3 seeds)"); ax[1].set_title("One block, each part removed", fontsize=10)
    names = list(depth)                                     # panel 3: depth, train vs test
    x = np.arange(len(names))                               # bar positions
    ax[2].bar(x - 0.2, [depth[k][1] for k in names], width=0.4, color="#9ecae1", label="training")
    ax[2].bar(x + 0.2, [depth[k][0] for k in names], width=0.4, color="#08519c", label="test")
    ax[2].set_xticks(x); ax[2].set_xticklabels([k.replace("blocks", "bl.").replace("block", "bl.") for k in names], fontsize=6.5, rotation=25, ha="right")
    ax[2].set_ylim(0.75, 1.0); ax[2].set_ylabel("accuracy (seed 0)"); ax[2].set_title("Deeper stacks: fit vs generalise", fontsize=10)
    ax[2].legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig("encoder_block_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved encoder_block_results.png")


def draw_diagram():                                     # the post-LN encoder block, and where pre-LN moves the norms
    fig, ax = plt.subplots(figsize=(13, 4.2))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333", ls="-"):            # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2, linestyle=ls))

    ax.text(6.5, 3.95, "Day 78: one Transformer encoder block (post-LN, as in the paper)", ha="center", fontsize=12, fontweight="bold")
    y = 2.0                                                                   # the main line
    box(0.7, y, 0.9, 0.6, "x", "#F2F2F2", bold=True)                          # input
    box(2.5, y, 1.9, 0.75, "multi-head\nattention (Day 76)", "#F4EEFC")      # attention
    box(4.4, y, 0.7, 0.6, "+", "#FAEEDA", fs=11, bold=True)                   # residual add
    box(5.8, y, 1.3, 0.6, "LayerNorm", "#DCEBF7")                             # norm
    box(7.7, y, 1.7, 0.75, "FFN\nReLU(h W1) W2", "#E5F2E1")                   # feed-forward
    box(9.5, y, 0.7, 0.6, "+", "#FAEEDA", fs=11, bold=True)                   # residual add
    box(10.9, y, 1.3, 0.6, "LayerNorm", "#DCEBF7")                            # norm
    box(12.4, y, 0.8, 0.6, "y", "#F2F2F2", bold=True)                         # output
    arrow(1.17, y, 1.53, y); arrow(3.47, y, 4.03, y); arrow(4.77, y, 5.13, y); arrow(6.47, y, 6.83, y)
    arrow(8.57, y, 9.13, y); arrow(9.87, y, 10.23, y); arrow(11.57, y, 11.98, y)
    ax.annotate("", xy=(4.4, 2.32), xytext=(0.7, 2.32), arrowprops=dict(arrowstyle="->", color="#C97A16", linewidth=1.3,
                                                                       connectionstyle="arc3,rad=-0.35"))  # residual 1
    ax.annotate("", xy=(9.5, 2.32), xytext=(5.8, 2.32), arrowprops=dict(arrowstyle="->", color="#C97A16", linewidth=1.3,
                                                                       connectionstyle="arc3,rad=-0.35"))  # residual 2
    ax.text(2.5, 3.35, "residual: x skips attention", ha="center", fontsize=8, color="#C97A16")
    ax.text(7.7, 3.35, "residual: h skips the FFN", ha="center", fontsize=8, color="#C97A16")
    ax.text(2.5, 1.2, "Day 77's encoding added to\nqueries and keys", ha="center", fontsize=7.5, style="italic")
    ax.text(7.7, 1.2, "the same weights at every\nposition, size 32 -> 128 -> 32", ha="center", fontsize=7.5, style="italic")
    ax.text(6.5, 0.4, "pre-LN: the two LayerNorms move in front of attention and the FFN, and one more LayerNorm follows the last block",
            ha="center", fontsize=8.5, style="italic")
    plt.savefig("encoder_block_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved encoder_block_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Today's block normalises each position's vector on its own: LayerNorm. CNNs (Day 54) used "
          "BatchNorm, which normalises each feature across the batch. Day 79 puts the two side by side "
          "inside this block, on padded sentences of different lengths.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_layer_norm()                                    # 1. LayerNorm by hand
        demo_ffn()                                           # 2. the FFN is position-wise
        demo_gradient_check()                                # 3. gradients
        res = demo_residual()                                # 4. why the residual
        itos, tag_names, data = prepare_data()               # Day 62's data
        lr = demo_one_block(itos, tag_names, data)           # 5. one block
        table = demo_ablations(itos, tag_names, data, lr)    # 6. each part removed
        depth = demo_depth(itos, tag_names, data, lr)        # 7. deeper
        plot_results(res, table, depth)                      # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 8. Day 79


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
