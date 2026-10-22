"""
Day 77: Positional Encoding, From Scratch -- Giving Attention a Sense of Order

Day 74 proved it on real sentences: self-attention cannot tell one order
of the words from another. Shuffle a sentence and every word's output
just moves with it. Vaswani et al. (2017) fix this by adding a vector
that depends only on the POSITION to every word vector:

    PE[p, 2i]   = sin(p / 10000^(2i/d))
    PE[p, 2i+1] = cos(p / 10000^(2i/d))          p = position, i = 0 .. d/2 - 1

Each pair of numbers (2i, 2i+1) is a point on a circle that turns at its
own speed as p grows: fast for the first pairs, very slow for the last.

What is built and measured, reported as it came out:

  1. The encoding itself, and how fast each pair turns.
  2. Its key property: moving k positions along is the SAME rotation for
     every starting position, so PE[p] . PE[q] depends only on p - q.
  3. Order becomes visible: Day 74's shuffle test, without and with it.
  4. A toy task that needs position (write the previous token), with no
     encoding, the sinusoidal one and a learned table, and then on
     sequences longer than any seen in training.
  5. Week 9's real tagging task with Day 76's multi-head layer: where to
     add the encoding and how strongly, chosen on validation.
  6. What the trained heads look at, by offset.
  7. Day 62's shuffle test on the real test set.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import copy                                            # copy: deepcopy, to keep a snapshot of the best weights during training
import os                                              # os: build the path to Week 9's dataset folder
import time                                            # time: perf_counter, to time training runs
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
LR = 2.0                                                # Day 74's learning rate, chosen there on validation
HEADS = 4                                               # Day 76's layer with 4 heads of size 8
SCALES = (0.3, 1.0, 3.0, 10.0)                          # strengths of the encoding tried on validation
TOY_V = 10                                              # toy task: token 0 means "nothing before", tokens 1..9 are real
TOY_LEN = (8, 16)                                       # toy training sequences have 8 to 16 tokens
TOY_STEPS = 3000                                        # toy training steps
MAX_LEN = 128                                           # positions the encodings are built for (the longest test sentence has 81 words)


# =============================================================================
# 0. REUSED -- Day 76's multi-head attention, and Day 62's data, verbatim
# =============================================================================
def softmax(z):                                         # Days 57-76: weights along the last axis, summing to 1
    e = np.exp(z - z.max(axis=-1, keepdims=True))         # subtract the row max for stability, exponentiate
    return e / e.sum(axis=-1, keepdims=True)              # normalise


def scaled_dot_product_attention(Q, K, V, mask=None):  # day75_exercise.py, verbatim (via Day 76)
    d = Q.shape[-1]                                       # the query and key size
    S = np.einsum("mid,mjd->mij", Q, K) / np.sqrt(d)      # scaled scores
    if mask is not None:                                  # Day 70's masking, if there are padded keys
        S = np.where(mask, S, -np.inf)                      # masked keys get minus infinity ...
    e = np.exp(S - S.max(axis=-1, keepdims=True))         # ... and so weight exactly 0
    A = e / e.sum(axis=-1, keepdims=True)                 # weights over the keys
    return np.einsum("mij,mjd->mid", A, V), A             # outputs, and the weights (the backward pass needs them)


def scaled_dot_product_attention_backward(dZ, Q, K, V, A):  # day75_exercise.py, verbatim (via Day 76)
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


def multi_head_attention(X, P, h, mask=None):           # Day 76, verbatim: X (m, n, d) -> (m, n, d), and a cache
    Q, K, V = (split_heads(X @ P[w], h) for w in ("Wq", "Wk", "Wv"))  # Day 74's three projections, cut into h heads
    Zh, A = scaled_dot_product_attention(Q, K, V, mask)   # Day 75's function, every head at once
    Z = merge_heads(Zh, h)                                # the heads' outputs side by side
    return Z @ P["Wo"], dict(X=X, Q=Q, K=K, V=V, A=A, Z=Z)  # the output projection mixes the heads


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
# 1. THE SINUSOIDAL ENCODING -- NEW
# =============================================================================
def sinusoidal_encoding(n, d):                          # NEW: (n, d) -- row p is position p's vector
    p = np.arange(n)[:, None]                             # positions 0 .. n-1, as a column
    i = np.arange(d // 2)[None, :]                        # pair index 0 .. d/2 - 1, as a row
    angle = p / 10000 ** (2 * i / d)                      # pair i turns by 1 / 10000^(2i/d) radians per position
    PE = np.zeros((n, d))                                 # one row per position
    PE[:, 0::2] = np.sin(angle)                           # even columns: sine
    PE[:, 1::2] = np.cos(angle)                           # odd columns: cosine of the same angle
    return PE                                             # every number between -1 and 1


def shift_rotation(k, d):                               # NEW: the (d, d) matrix R_k with PE[p + k] = PE[p] R_k for EVERY p
    R = np.zeros((d, d))                                  # block diagonal: one 2 x 2 rotation per pair
    for i in range(d // 2):                               # each pair
        a = k / 10000 ** (2 * i / d)                        # how far this pair turns in k positions
        R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]  # (sin, cos) of angle + a
    return R                                              # depends on k, never on p


# =============================================================================
# 2. THE TOY TASK -- NEW: write the token one position back
# =============================================================================
def toy_batch(rng, m, n):                               # NEW: m sequences of n tokens; the answer at p is the token at p - 1
    tok = rng.randint(1, TOY_V, size=(m, n))              # tokens 1..9
    y = np.concatenate([np.zeros((m, 1), int), tok[:, :-1]], axis=1)  # position 0 has nothing before it: answer 0
    return tok, y                                         # inputs and answers


class ToyNet:                                           # NEW: tokens (+ positions) -> one attention head -> a token per position
    def __init__(self, d, pe, seed=0, lr=0.1, clip=1.0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.pe, self.lr, self.clip = pe, lr, clip         # "none", "sin" or "learned"; training settings
        self.P = dict(E=rng.randn(TOY_V, d),               # token embeddings, spread 1
                      Wq=rng.randn(d, d) / np.sqrt(d), Wk=rng.randn(d, d) / np.sqrt(d),  # query and key projections
                      Wv=rng.randn(d, d) / np.sqrt(d),     # value projection
                      Wout=rng.randn(d, TOY_V) / np.sqrt(d), bout=np.zeros(TOY_V))  # which token?
        if pe == "learned":                                # a learned table: one free vector per position
            self.P["PE"] = rng.randn(MAX_LEN, d)             # spread 1, like the sinusoidal numbers
        self.SIN = sinusoidal_encoding(MAX_LEN, d)         # the fixed table

    def forward(self, tok):                                # (m, n) tokens -> (m, n, 10) probabilities
        P = self.P                                           # parameters
        n = tok.shape[1]                                     # length
        X = P["E"][tok]                                      # token vectors
        if self.pe == "sin":                                 # the fixed encoding
            X = X + self.SIN[:n]                               # added to every token vector
        if self.pe == "learned":                             # the learned table
            X = X + P["PE"][:n]                                # rows 0..n-1
        Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]      # one head (Day 75's function below)
        Z, A = scaled_dot_product_attention(Q, K, V)         # every position reads every position
        p = softmax(Z @ P["Wout"] + P["bout"])               # a token distribution per position
        self.cache = dict(tok=tok, X=X, Q=Q, K=K, V=V, A=A, Z=Z, p=p)  # for the backward pass
        return p                                          # hand back

    def loss(self, tok, y):                                # mean cross-entropy over every position
        p = self.forward(tok)                                # forward pass
        return -np.log(np.take_along_axis(p, y[..., None], axis=-1) + 1e-12).mean()  # probability of each right answer

    def gradients(self, tok, y):                           # every gradient of that loss
        P = self.P                                           # parameters
        p = self.forward(tok); C = self.cache                # forward pass
        m, n = tok.shape                                     # batch, length
        dl = p.copy()                                        # softmax + cross-entropy gradient ...
        dl[np.arange(m)[:, None], np.arange(n)[None, :], y] -= 1; dl /= m * n  # ... averaged over every position
        flat = lambda T: T.reshape(-1, T.shape[-1])          # (m, n, d) -> (m * n, d)
        g = dict(Wout=flat(C["Z"]).T @ flat(dl), bout=flat(dl).sum(axis=0))  # output layer
        dQ, dK, dV = scaled_dot_product_attention_backward(dl @ P["Wout"].T, C["Q"], C["K"], C["V"], C["A"])  # Day 75's backward
        g.update(Wq=flat(C["X"]).T @ flat(dQ), Wk=flat(C["X"]).T @ flat(dK), Wv=flat(C["X"]).T @ flat(dV))  # projections
        dX = dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T  # X made all three
        g["E"] = np.zeros_like(P["E"]); np.add.at(g["E"], tok, dX)  # token embeddings
        if self.pe == "learned":                             # the learned table's rows 0..n-1
            g["PE"] = np.zeros_like(P["PE"]); g["PE"][:n] = dX.sum(axis=0)  # summed over the batch
        return g                                          # dict: parameter name -> gradient

    def train_step(self, tok, y):                          # one SGD step with Day 60's clipping
        g = self.gradients(tok, y)                           # forward and backward
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))  # global norm
        for k in g:                                          # plain SGD
            self.P[k] -= self.lr * min(1.0, self.clip / norm) * g[k]  # shrink only if too long


def toy_accuracy(net, n):                               # share of positions 1..n-1 answered right, on 500 fixed sequences
    tok, y = toy_batch(np.random.RandomState(999 + n), 500, n)  # fixed test sequences of length n
    right = net.forward(tok).argmax(axis=-1) == y         # right or wrong, per position
    return right[:, 1:].mean(), right.mean(axis=0)        # overall (position 0 left out: it is trivial), and per position


def toy_train(pe, seed, d=32):                          # one toy run
    net = ToyNet(d, pe, seed=seed)                          # fresh model
    rng = np.random.RandomState(seed + 100)                 # the training data stream
    for step in range(TOY_STEPS):                         # plain SGD on fresh batches
        tok, y = toy_batch(rng, 32, rng.randint(TOY_LEN[0], TOY_LEN[1] + 1))  # a random length 8..16
        net.train_step(tok, y)                                # one update
    return net                                            # the trained model


# =============================================================================
# 3. THE TAGGER -- NEW: Day 76's layer, with the encoding added where chosen
# =============================================================================
def positional_attention(Xqk, Xv, P, h, mask):          # NEW: queries and keys from Xqk, values from Xv (Day 76 when they are equal)
    Q, K = split_heads(Xqk @ P["Wq"], h), split_heads(Xqk @ P["Wk"], h)  # where to look: from Xqk
    V = split_heads(Xv @ P["Wv"], h)                      # what to read: from Xv
    Zh, A = scaled_dot_product_attention(Q, K, V, mask)   # Day 75's function, every head at once
    Z = merge_heads(Zh, h)                                # heads side by side
    return Z @ P["Wo"], dict(Q=Q, K=K, V=V, A=A, Z=Z)     # output projection, and the cache


class PositionalTagger:                                 # NEW: words (+ positions) -> multi-head self-attention -> a tag per word
    def __init__(self, vocab_size, n_tags, place="qk", scale=1.0, D=32, H=32, heads=HEADS, lr=LR, clip=1.0, seed=0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.n_tags, self.place, self.h, self.lr, self.clip = n_tags, place, heads, lr, clip  # settings
        self.P = dict(E=rng.randn(vocab_size, D) * 0.1,   # embedding table, as on Days 62-74
                      Wq=rng.randn(D, D) / np.sqrt(D), Wk=rng.randn(D, D) / np.sqrt(D),  # query and key projections
                      Wv=rng.randn(D, D) / np.sqrt(D), Wo=rng.randn(D, D) / np.sqrt(D),  # value and output projections
                      W1=rng.randn(2 * D, H) / np.sqrt(2 * D), b1=np.zeros(H),  # hidden layer on [word ; attention output] (Day 74)
                      Why=rng.randn(H, n_tags) / np.sqrt(H), by=np.zeros(n_tags))  # output layer
        self.PE = sinusoidal_encoding(MAX_LEN, D) * scale  # the encoding, times its strength

    def forward(self, ids):                                # ids (m, T) -> tag probabilities, one row per position
        P = self.P                                           # parameters
        m, T = ids.shape                                     # batch, padded length
        self.ids, mask = ids, ids != PAD                     # only padding has id 0
        X = P["E"][ids]                                      # word vectors
        pos = self.PE[:T][None] if self.place != "none" else 0.0  # the encoding, if used
        self.Xqk = X + pos                                   # queries and keys always see the positions (if any)
        self.Xv = X + pos if self.place == "all" else X      # values and the word's own vector see them only for "all"
        kmask = np.repeat(mask[:, None, None, :], self.h, axis=1).reshape(m * self.h, 1, T)  # padded keys, per head
        self.O, self.c = positional_attention(self.Xqk, self.Xv, P, self.h, kmask)  # context from the whole sentence
        self.XZ = np.concatenate([self.Xv, self.O], axis=2).reshape(m * T, -1)  # each word's own vector and its context
        self.Hh = np.tanh(self.XZ @ P["W1"] + P["b1"])       # hidden layer
        self.probs = softmax(self.Hh @ P["Why"] + P["by"])   # a tag distribution for every position
        return self.probs                                 # hand back

    def loss(self, ids, tags, mask):                       # Day 62's mean cross-entropy over the real words
        p = self.forward(ids)[np.arange(tags.size), tags.ravel()]  # probability of each word's true tag
        return -np.sum(np.log(p + 1e-9) * mask.ravel()) / mask.sum()  # padding counts 0

    def gradients(self, tags, mask):                       # gradients of Day 62's masked cross-entropy
        P, c = self.P, self.c                                # parameters, attention cache
        m, T = self.ids.shape                                # batch, padded length
        D = P["E"].shape[1]                                  # embedding size
        onehot = np.eye(self.n_tags)[tags.ravel()]           # true tags, one-hot
        dlog = (self.probs - onehot) * mask.ravel()[:, None] / mask.sum()  # zero on padding (Day 62)
        g = dict(Why=self.Hh.T @ dlog, by=dlog.sum(axis=0))  # output layer
        dpre = (dlog @ P["Why"].T) * (1 - self.Hh ** 2)      # back through tanh
        g["W1"], g["b1"] = self.XZ.T @ dpre, dpre.sum(axis=0)  # hidden layer
        dXZ = (dpre @ P["W1"].T).reshape(m, T, -1)           # gradient for [word ; context]
        flat = lambda A: A.reshape(-1, A.shape[-1])          # (m, T, d) -> (m * T, d)
        dO = dXZ[:, :, D:]                                   # the attention output's part
        g["Wo"] = flat(c["Z"]).T @ flat(dO)                  # output projection
        dQ, dK, dV = scaled_dot_product_attention_backward(split_heads(dO @ P["Wo"].T, self.h), c["Q"], c["K"], c["V"], c["A"])  # Day 75's backward, every head
        dQ, dK, dV = merge_heads(dQ, self.h), merge_heads(dK, self.h), merge_heads(dV, self.h)  # heads side by side
        g["Wq"], g["Wk"] = flat(self.Xqk).T @ flat(dQ), flat(self.Xqk).T @ flat(dK)  # queries and keys came from Xqk
        g["Wv"] = flat(self.Xv).T @ flat(dV)                 # values came from Xv
        dX = dXZ[:, :, :D] + dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T  # every path back to the word vectors
        g["E"] = np.zeros_like(P["E"]); np.add.at(g["E"], self.ids, dX)  # Day 61 (the encoding itself is fixed)
        return g                                          # dict: parameter name -> gradient

    def train_step(self, ids, tags, mask):                 # one SGD step with Day 60's clipping
        self.forward(ids)                                    # forward pass
        g = self.gradients(tags, mask)                       # backward pass
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))  # global norm
        for k in g:                                          # plain SGD
            self.P[k] -= self.lr * min(1.0, self.clip / norm) * g[k]  # shrink only if too long

    def predict(self, ids):                                # predicted tag for every position, in batches of 256 sentences
        out = np.zeros(ids.shape, dtype=int)                 # predictions, padding left at 0
        for s in range(0, len(ids), 256):                    # batches keep the attention arrays small
            b = ids[s:s + 256]                                 # these sentences
            T = (b != PAD).sum(axis=1).max()                   # pad only to the longest one
            out[s:s + 256, :T] = self.forward(b[:, :T]).argmax(axis=1).reshape(len(b), T)  # most probable tag
        return out                                        # hand back


def fit(data, vocab_size, n_tags, seed, epochs=EPOCHS, batch_size=32, **kw):  # Day 67's fit(): best epoch on VALIDATION
    (ids_tr, tags_tr, mask_tr), (ids_va, tags_va, mask_va), (ids_te, tags_te, mask_te) = data  # the three splits
    net = PositionalTagger(vocab_size, n_tags, seed=seed, **kw)  # fresh network
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
            best_val, best_P, best_epoch = val, copy.deepcopy(net.P), epoch  # snapshot the weights
    net.P = best_P                                          # keep the best snapshot
    return net, best_val, best_epoch, token_accuracy(net.predict(ids_te), tags_te, mask_te)  # model, val, epoch, test


RUNS = {}                                               # cache: each configuration trains once


def fit_cached(data, vocab_size, n_tags, seed, **kw):   # fit(), remembered by its settings (Day 67)
    key = (seed, tuple(sorted(kw.items())))               # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = fit(data, vocab_size, n_tags, seed, **kw)  # train and remember
    return RUNS[key]                                      # later requests reuse it


def offset_weights(net, ids, offsets=(-3, -2, -1, 0, 1, 2, 3)):  # mean weight each head puts k words away, over real words
    b = ids[:500]; T = (b != PAD).sum(axis=1).max(); b = b[:, :T]  # 500 test sentences
    net.forward(b)                                        # fills the attention cache
    A = net.c["A"].reshape(len(b), net.h, T, T)           # (m, heads, query word, key word)
    real = b != PAD                                       # real words
    out = np.zeros((net.h, len(offsets)))                 # head x offset
    for j, k in enumerate(offsets):                       # each offset
        i = np.arange(T); i = i[(i + k >= 0) & (i + k < T)]  # query positions whose word k away exists in the padded array
        ok = real[:, i] & real[:, i + k]                     # both words real
        out[:, j] = [A[:, hh, i, i + k][ok].mean() for hh in range(net.h)]  # mean weight, per head
    return out                                            # (heads, offsets)


# =============================================================================
# DEMOS
# =============================================================================
def demo_encoding():
    print("=" * 70)
    print("1. THE ENCODING -- a vector per position, d = 32")
    print("=" * 70)
    print("The first six numbers of the encoding at positions 0-4 (pairs: (sin, cos) of one angle), "
          "and how many positions each pair takes to turn once around its circle (2 pi 10000^(2i/d)).")
    PE = sinusoidal_encoding(MAX_LEN, 32)                 # the table
    print(f"\n{'position':>9}" + "".join(f"{'col ' + str(c):>9}" for c in range(6)))
    for p in range(5):                                    # five positions
        print(f"{p:>9}" + "".join(f"{v:>9.4f}" for v in PE[p, :6]))
    print("Actual output from this lesson's run.")
    print("\npositions per full turn, pair by pair (pair 0 is columns 0-1):")
    periods = 2 * np.pi * 10000 ** (2 * np.arange(16) / 32)  # one period per pair
    print("  " + "  ".join(f"{i}: {t:.1f}" for i, t in enumerate(periods[:6])) + "  ...  15: " + f"{periods[15]:.0f}")
    print("Actual output from this lesson's run.")
    return PE                                              # the plot reuses it


def demo_rotation():
    print("\n" + "=" * 70)
    print("2. MOVING ALONG IS A ROTATION -- the same one for every starting position")
    print("=" * 70)
    print("For each shift k, one matrix R_k (a 2 x 2 rotation per pair, built from k alone) is applied "
          "to the encoding at EVERY position p = 0..99. If PE[p + k] = PE[p] R_k exactly, then "
          "PE[p] . PE[p + k] cannot depend on p -- only on k.")
    d = 32                                                # size
    PE = sinusoidal_encoding(MAX_LEN, d)                  # the table
    print(f"\n{'k':>4}{'largest |PE[p+k] - PE[p] R_k|':>32}{'PE[p] . PE[p+k]: smallest':>28}{'largest':>10}")
    for k in (1, 2, 5, 20):                               # four shifts
        R = shift_rotation(k, d)                            # built from k alone
        err = np.abs(PE[k:100 + k] - PE[:100] @ R).max()    # every p at once
        dots = (PE[:100] * PE[k:100 + k]).sum(axis=1)       # the dot product at every p
        print(f"{k:>4}{err:>32.2e}{dots.min():>28.4f}{dots.max():>10.4f}")
    print("Actual output from this lesson's run.")
    print("\nPE[0] . PE[k] for k = 0..8 (k = 0 gives d / 2 = 16; nearby positions score highest, but not in a smooth line):")
    print("  " + "  ".join(f"{(PE[0] * PE[k]).sum():.2f}" for k in range(9)))
    print("Actual output from this lesson's run.")


def demo_order_visible():
    print("\n" + "=" * 70)
    print("3. ORDER BECOMES VISIBLE -- Day 74's shuffle test, without and with the encoding")
    print("=" * 70)
    print("Random word vectors (4 sentences, 7 words, size 8), Day 76's layer with 2 heads. Shuffle "
          "the words; compare each word's output after the shuffle with its output before.")
    rng = np.random.RandomState(0)                        # fixed rng
    X = rng.randn(4, 7, 8)                                # word vectors
    P = {w: rng.randn(8, 8) for w in ("Wq", "Wk", "Wv", "Wo")}  # random projections
    perm = rng.permutation(7)                             # one shuffle
    PE = sinusoidal_encoding(7, 8)                        # positions 0..6
    print(f"\nshuffle {perm.tolist()}")
    for label, pos in [("no encoding", 0.0), ("with the encoding", PE)]:  # the two cases
        out, _ = multi_head_attention(X + pos, P, 2)        # original order
        out_s, _ = multi_head_attention(X[:, perm] + pos, P, 2)  # shuffled words, positions stay where they are
        print(f"{label:<19}: largest |output(shuffled) - shuffled(output)| = {np.abs(out_s - out[:, perm]).max():.2e}")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("4a. GRADIENT CHECK -- the toy network and the tagger, every encoding")
    print("=" * 70)
    print("Tiny sizes (d = 4 for the toy, 8 with 2 heads for the tagger; 3 sequences, the tagger's padded "
          "to different lengths); every entry of every parameter nudged by +-1e-5; Day 62's pass rule "
          "|numeric - analytic| <= 1e-10 + 1e-6 |numeric|.")
    rng = np.random.RandomState(3)                        # fixed rng
    tok, y = toy_batch(rng, 3, 5)                         # a toy batch
    ids = np.array([[2, 3, 4, 5, 6], [7, 2, 3, 0, 0], [4, 4, 5, 6, 0]])  # three sentences, padded
    tags = rng.randint(0, 3, size=ids.shape); mask = ids != PAD  # random tags, the real words
    cases = [(f"toy, {pe}", ToyNet(4, pe, seed=1), lambda n: n.loss(tok, y), lambda n: n.gradients(tok, y))
             for pe in ("none", "sin", "learned")]        # the three toy encodings
    for place in ("none", "all", "qk"):                   # the three tagger placements
        net = PositionalTagger(8, 3, place=place, scale=1.0, D=8, H=5, heads=2, seed=1)  # a tiny tagger
        cases.append((f"tagger, {place}", net, lambda n: n.loss(ids, tags, mask),
                      lambda n: (n.forward(ids), n.gradients(tags, mask))[1]))
    for label, net, loss_fn, grad_fn in cases:            # each model
        g = grad_fn(net)                                    # analytic gradients
        worst, ok = 0.0, True                               # largest difference; every entry passes?
        for k, A in net.P.items():                          # every parameter
            for pos in np.ndindex(A.shape):                   # every entry
                old = A[pos]                                    # remember it
                A[pos] = old + 1e-5; up = loss_fn(net)          # nudge up
                A[pos] = old - 1e-5; dn = loss_fn(net)          # nudge down
                A[pos] = old                                    # restore
                num = (up - dn) / 2e-5                          # central difference
                worst = max(worst, abs(num - g[k][pos]))        # largest difference so far
                ok &= abs(num - g[k][pos]) <= 1e-10 + 1e-6 * abs(num)  # Day 62's rule
        print(f"{label:<16}: largest |numeric - analytic| = {worst:.2e}; every entry passes: {ok}")
    print("Actual output from this lesson's run.")


def demo_toy():
    print("\n" + "=" * 70)
    print("4. A TASK THAT NEEDS POSITION -- write the token one position back")
    print("=" * 70)
    print(f"Sequences of {TOY_LEN[0]}-{TOY_LEN[1]} random tokens (1-9); the answer at every position is the "
          "token just before it. One attention head, size 32, plain SGD lr 0.1, clip 1.0, batches of 32, "
          f"{TOY_STEPS} steps, three seeds. Guessing: 1/9 = 0.111. Test: 500 fixed sequences of each "
          "length; 24, 32 and 48 are longer than any training sequence.")
    print(f"\n{'encoding':<11}" + "".join(f"{'n=' + str(n):>8}" for n in (16, 24, 32, 48)) + "   n=16 per seed")
    toy = {}                                               # encoding -> per-position accuracy at n = 32 (seed mean)
    for pe in ("none", "sin", "learned"):                  # the three encodings
        nets = [toy_train(pe, s) for s in SEEDS]             # three seeds
        accs = np.array([[toy_accuracy(net, n)[0] for n in (16, 24, 32, 48)] for net in nets])  # (seeds, lengths)
        toy[pe] = np.mean([toy_accuracy(net, 32)[1] for net in nets], axis=0)  # per position at n = 32
        print(f"{pe:<11}" + "".join(f"{v:>8.3f}" for v in accs.mean(axis=0)) + "   " + " ".join(f"{v:.3f}" for v in accs[:, 0]))
    print("Actual output from this lesson's run.")
    print("\nat n = 32, by position (mean of seeds): positions 1-15 were seen in training, 16-31 never")
    for pe in ("sin", "learned"):                          # the two encodings
        print(f"  {pe:<8} positions 1-15: {toy[pe][1:16].mean():.3f}   positions 16-31: {toy[pe][16:].mean():.3f}")
    print("Actual output from this lesson's run.")
    return toy                                             # the plot reuses it


def demo_tagger(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("5. REAL TAGGING -- where to add the encoding, and how strongly, chosen on validation")
    print("=" * 70)
    print(f"Day 62's data and Day 74's tagger, with Day 76's layer ({HEADS} heads of 8, size 32), lr {LR}, "
          f"{EPOCHS} epochs, best epoch on validation. 'all': the encoding is added to the word vectors "
          "(queries, keys, values and the word's own vector, as in the Transformer paper); 'qk': added "
          "only where the layer decides WHERE to look (queries and keys). Seed 0, validation accuracy:")
    V, nt = len(itos), len(tag_names)                     # vocabulary and tag-set sizes
    print(f"\n{'place':<7}" + "".join(f"{'x ' + str(s):>9}" for s in SCALES))
    best = {}                                              # place -> scale with the best validation accuracy
    for place in ("all", "qk"):                            # the two placements
        vals = [fit_cached(data, V, nt, 0, place=place, scale=s)[1] for s in SCALES]  # validation accuracy per strength
        best[place] = SCALES[int(np.argmax(vals))]           # chosen on validation
        print(f"{place:<7}" + "".join(f"{v:>9.4f}" for v in vals))
    print(f"no encoding: {fit_cached(data, V, nt, 0, place='none')[1]:.4f}")
    print(f"chosen on validation: all x {best['all']}, qk x {best['qk']}")
    print("Actual output from this lesson's run.")
    print("\nthree seeds, TEST accuracy (each run's best epoch chosen on validation):")
    print(f"\n{'model':<26}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}")
    configs = [("no encoding", dict(place="none")), (f"all, x {best['all']}", dict(place="all", scale=best["all"])),
               (f"qk, x {best['qk']}", dict(place="qk", scale=best["qk"]))]
    table = {}                                             # label -> test accuracies
    for label, kw in configs:                              # each model
        table[label] = [fit_cached(data, V, nt, s, **kw)[3] for s in SEEDS]  # test accuracy per seed
        print(f"{label:<26}" + "".join(f"{v:>9.4f}" for v in table[label]) + f"{np.mean(table[label]):>9.4f}")
    print("for comparison: Day 74's single-head tagger 0.8070, Day 62's RNN 0.8327, Day 68's LSTM 0.8377, "
          "bidirectional LSTM 0.8577")
    print("Actual output from this lesson's run.")
    return best, configs, table                            # later sections reuse them


def demo_offsets(itos, tag_names, data, configs):
    print("\n" + "=" * 70)
    print("6. WHAT THE HEADS LOOK AT -- mean weight on the word k positions away, seed 0")
    print("=" * 70)
    print("Over 500 test sentences, real words only. With 10 words a uniform head gives about 0.1 "
          "everywhere; a head that tracks position puts a large weight on one offset.")
    V, nt = len(itos), len(tag_names)                     # sizes
    ids_te = data[2][0]                                   # test word ids
    offs = (-3, -2, -1, 0, 1, 2, 3)                       # offsets shown
    res = {}                                              # label -> (heads, offsets)
    for label, kw in configs:                             # the three models
        res[label] = offset_weights(fit_cached(data, V, nt, 0, **kw)[0], ids_te, offs)  # head x offset
        print(f"\n{label}:   " + "".join(f"{'k=' + str(k):>8}" for k in offs))
        for hh in range(HEADS):                              # each head
            print(f"  head {hh + 1}        " + "".join(f"{v:>8.3f}" for v in res[label][hh]))
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_shuffle(itos, tag_names, data, configs):
    print("\n" + "=" * 70)
    print("7. DAY 62'S SHUFFLE TEST -- the real test sentences with their words in random order")
    print("=" * 70)
    print("Each word keeps its own true tag. Day 74's tagger did not change a single prediction; "
          "Day 62's RNN dropped from 0.8327 to 0.7487. Seed 0 of each model.")
    V, nt = len(itos), len(tag_names)                     # sizes
    ids, tags, mask = data[2]                             # the test split
    rng = np.random.RandomState(7)                        # fixed shuffles
    ids_s, tags_s = ids.copy(), tags.copy()               # shuffled copies
    for r in range(len(ids)):                             # each sentence
        n = mask[r].sum(); p = rng.permutation(n)           # its real words, shuffled
        ids_s[r, :n], tags_s[r, :n] = ids[r, :n][p], tags[r, :n][p]  # words move with their tags
    print(f"\n{'model':<26}{'normal order':>14}{'shuffled':>10}")
    for label, kw in configs:                             # the three models
        net = fit_cached(data, V, nt, 0, **kw)[0]           # seed 0
        print(f"{label:<26}{token_accuracy(net.predict(ids), tags, mask):>14.4f}{token_accuracy(net.predict(ids_s), tags_s, mask):>10.4f}")
    print("Actual output from this lesson's run.")


def plot_results(PE, toy, offsets, configs):            # three panels: the encoding | toy by position | heads by offset
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))          # canvas
    im = ax[0].imshow(PE[:64].T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)  # rows: columns of the encoding
    ax[0].set_xlabel("position"); ax[0].set_ylabel("column of the encoding (d = 32)")
    ax[0].set_title("The sinusoidal encoding", fontsize=10); fig.colorbar(im, ax=ax[0], fraction=0.04)
    for pe, color in [("sin", "#1f77b4"), ("learned", "#ff7f0e"), ("none", "#7f7f7f")]:  # panel 2: per position at n = 32
        ax[1].plot(np.arange(1, 32), toy[pe][1:], marker="o", markersize=3, color=color, label=pe)
    ax[1].axvline(15.5, color="#333333", linestyle=":"); ax[1].text(16, 0.6, "never seen\nin training", fontsize=8)
    ax[1].set_xlabel("position (32-token sequences)"); ax[1].set_ylabel("accuracy (mean of 3 seeds)"); ax[1].set_ylim(-0.02, 1.05)
    ax[1].set_title("Previous-token task, longer than training", fontsize=10); ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.3)
    offs = np.arange(-3, 4)                                 # panel 3: the qk model's heads by offset
    label = configs[2][0]                                   # the qk model
    for hh, color in zip(range(HEADS), ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]):
        ax[2].plot(offs, offsets[label][hh], marker="o", color=color, label=f"head {hh + 1}")
    ax[2].plot(offs, offsets[configs[1][0]].mean(axis=0), color="#7f7f7f", linestyle="--", label=f"{configs[1][0]}, mean of heads")
    ax[2].set_xlabel("offset k (word k positions away)"); ax[2].set_ylabel("mean attention weight")
    ax[2].set_title(f"Tagger heads ({label}), seed 0", fontsize=10); ax[2].legend(fontsize=7, loc="upper left"); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("positional_encoding_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved positional_encoding_results.png")


def draw_diagram():                                     # where the encoding enters
    fig, ax = plt.subplots(figsize=(13, 4.0))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.0); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333"):                    # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    ax.text(6.5, 3.75, "Day 77: adding position to self-attention", ha="center", fontsize=12, fontweight="bold")
    box(1.1, 2.6, 1.7, 0.6, "word vector E[w]", "#F2F2F2")                  # the word
    box(1.1, 1.2, 1.7, 0.6, "PE[p]  (sin, cos)", "#FAEEDA", bold=True)      # the position
    box(3.6, 2.6, 1.3, 0.6, "+", "#FAEEDA", fs=11, bold=True)               # the sum
    arrow(1.97, 2.6, 2.93, 2.6); arrow(1.97, 1.35, 2.98, 2.42)              # word and position into the sum
    box(6.0, 2.95, 2.0, 0.55, "queries, keys", "#DCEBF7")                   # where to look
    box(6.0, 1.7, 2.0, 0.55, "values", "#E5F2E1")                           # what to read
    arrow(4.27, 2.7, 4.98, 2.95)                                            # sum -> queries and keys
    arrow(4.27, 2.5, 4.98, 1.8, color="#999999")                            # sum -> values, only for "all"
    arrow(1.97, 2.45, 4.98, 1.62)                                           # word alone -> values, for "qk"
    ax.text(3.7, 1.15, "qk: values from the word alone\nall: values from word + position", ha="center", fontsize=7.5, style="italic")
    box(9.0, 2.3, 2.4, 0.8, "Day 76's multi-head\nattention", "#F4EEFC")    # the layer
    arrow(7.02, 2.95, 7.78, 2.45); arrow(7.02, 1.7, 7.78, 2.15)             # into the layer
    box(11.7, 2.3, 1.6, 0.7, "a tag per word", "#F2F2F2")                   # the output
    arrow(10.22, 2.3, 10.88, 2.3)                                           # layer -> tagger
    ax.text(6.5, 0.35, "PE[p + k] = PE[p] R_k: the same rotation for every p, so a fixed pair of query and key weights can pick 'k words away'",
            ha="center", fontsize=8.5, style="italic")
    plt.savefig("positional_encoding_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved positional_encoding_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Today's tagger is one attention layer and a small classifier. Day 78 builds the Transformer's "
          "encoder block around the attention: a residual connection, layer normalisation and a "
          "position-wise feed-forward network, and stacks it.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        PE = demo_encoding()                                 # 1. the encoding
        demo_rotation()                                      # 2. shifts are rotations
        demo_order_visible()                                 # 3. order is visible
        demo_gradient_check()                                # 4a. gradients
        toy = demo_toy()                                     # 4. the toy task
        itos, tag_names, data = prepare_data()               # Day 62's data
        best, configs, table = demo_tagger(itos, tag_names, data)  # 5. real tagging
        offsets = demo_offsets(itos, tag_names, data, configs)  # 6. what the heads look at
        demo_shuffle(itos, tag_names, data, configs)         # 7. the shuffle test
        plot_results(PE, toy, offsets, configs)              # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 8. Day 78


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
