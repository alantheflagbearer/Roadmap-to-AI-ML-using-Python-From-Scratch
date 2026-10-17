"""
Day 67: RNN vs LSTM vs GRU -- a Real Comparison on Week 9's Tagging Task

Days 63-66 built two gated cells and showed, on Day 59's sign task, that
they carry information much further than the plain RNN. That task was
built to need long memory. Today asks the practical question: on a REAL
task, does the cell matter?

The task is Day 62's: part-of-speech tagging of 4,078 real sentences
from the Universal Dependencies English Web Treebank. The model is Day
62's tagger with ONE thing changed -- the recurrent cell -- so the three
cells are compared like for like:

    words -> embedding -> [ RNN | LSTM | GRU ] -> a tag for every word

What is measured, reported as it came out:

  0. The three cells inside one tagger, and their weight counts.
  1. A gradient check of all three, through the whole tagger.
  2. Fair settings: each cell gets its own learning rate (and gate bias),
     chosen on VALIDATION accuracy only.
  3. The comparison: three seeds per cell, test accuracy at the best
     validation epoch, against Day 62's baselines.
  4. Where the cells differ: unambiguous, ambiguous and unknown words,
     and position in the sentence.
  5. How far back each tagger actually looks: change one earlier word and
     see whether the prediction k words later changes.
  6. A fairer size match: an RNN with as many recurrent weights as the LSTM.
  7. What each costs in training time.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import copy                                            # copy: deepcopy, to keep a snapshot of the best network during training
import os                                              # os: build the path to Week 9's dataset folder
import time                                            # time: perf_counter, to time every training epoch
from collections import Counter, defaultdict           # Counter counts things; defaultdict creates a missing entry on first use

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
CELLS = ["rnn", "lstm", "gru"]                          # the three recurrent cells being compared
NAMES = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU"}    # display names
COLORS = {"rnn": "#7f7f7f", "lstm": "#1f77b4", "gru": "#2ca02c"}  # one colour per cell, used in every plot
SEEDS = [0, 1, 2]                                       # every trained comparison uses these three seeds (Day 62's)
EPOCHS = 25                                             # Day 62's training budget

# =============================================================================
# 0. REUSED -- verbatim from Days 57-66
# =============================================================================
def sigmoid(z):                                         # squashes any number into the range 0..1 -- Day 63
    return 1.0 / (1.0 + np.exp(-z))                       # large negative -> 0, zero -> 0.5, large positive -> 1


def softmax(z):                                         # logits -> probabilities, one row per example -- Days 57-62
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract the row max for numerical stability
    exp_z = np.exp(z_shifted)                             # exponentiate
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row to sum to 1


def one_hot(y, n_classes):                              # integer labels -> one-hot rows -- Day 56
    out = np.zeros((y.shape[0], n_classes))               # start all-zero
    out[np.arange(y.shape[0]), y] = 1                     # set the true class's column to 1
    return out                                            # the one-hot matrix


def clip_by_global_norm(grads, max_norm):               # Day 60's clipper, verbatim (Day 65's form)
    norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))  # global norm over every parameter's gradient
    if max_norm is None or norm <= max_norm:               # no threshold, or already short enough
        return grads, norm, 1.0                              # untouched
    scale = max_norm / norm                                 # the factor that makes the norm exactly max_norm
    return {k: g * scale for k, g in grads.items()}, norm, scale  # same factor everywhere -> direction preserved


# --- the RNN: Day 62's forward and backward, verbatim ---------------------------
def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):        # ONE timestep -- Day 57
    return np.tanh(x_t @ Wxh + h_prev @ Whh + bh)         # this step's input AND the previous hidden state, squashed with tanh


def rnn_forward(X, h0, Wxh, Whh, bh):                   # every hidden state of a batch of sequences -- Day 62
    m, T, _ = X.shape                                     # batch size, sequence length, embedding size
    H = np.zeros((m, T, Whh.shape[0]))                    # will hold every timestep's hidden state
    h = h0                                                 # start from the given initial hidden state
    for t in range(T):                                     # one word at a time
        h = rnn_step_forward(X[:, t, :], h, Wxh, Whh, bh)    # the SAME weights at every t
        H[:, t, :] = h                                       # store this timestep's hidden state
    return H                                               # every hidden state


def rnn_backward(dH, X, H, h0, Wxh, Whh):               # BPTT with a gradient arriving at EVERY step -- Day 62
    m, T, _ = X.shape                                     # batch size, sequence length, embedding size
    dWxh = np.zeros_like(Wxh)                             # accumulators -- start at zero,
    dWhh = np.zeros_like(Whh)                             # get one contribution PER timestep
    dbh = np.zeros(Whh.shape[0])
    dX = np.zeros_like(X)                                 # the gradient for every word vector
    dh = np.zeros_like(h0)                                # nothing special arrives at the end
    for t in reversed(range(T)):                           # walk BACKWARD through time
        dh = dh + dH[:, t, :]                                # this step's own output adds its gradient
        h_prev = H[:, t - 1, :] if t > 0 else h0              # the state that fed timestep t
        da = dh * (1.0 - H[:, t, :] ** 2)                     # undo tanh using the cached output
        dWxh += X[:, t, :].T @ da                             # ACCUMULATE -- the same Wxh was used at every t
        dWhh += h_prev.T @ da                                 # ACCUMULATE
        dbh += da.sum(axis=0)                                 # ACCUMULATE
        dX[:, t, :] = da @ Wxh.T                              # gradient for this position's word vector
        dh = da @ Whh.T                                       # hand gradient back to h_{t-1}
    return dWxh, dWhh, dbh, dX                              # summed over every timestep


# --- the LSTM: Days 63-65, verbatim -----------------------------------------------
def lstm_step_forward(x_t, h_prev, c_prev, W, b):       # one step of an LSTM cell -- Day 63
    H = h_prev.shape[1]                                   # hidden size
    a = np.concatenate([x_t, h_prev], axis=1) @ W + b     # input and old hidden state side by side, times ONE weight matrix
    f = sigmoid(a[:, 0 * H:1 * H])                        # forget gate: how much of the old cell state to KEEP (0..1)
    i = sigmoid(a[:, 1 * H:2 * H])                        # input gate: how much of the candidate to WRITE (0..1)
    g = np.tanh(a[:, 2 * H:3 * H])                        # candidate: the new value that could be written (-1..1)
    o = sigmoid(a[:, 3 * H:4 * H])                        # output gate: how much of the cell state to SHOW (0..1)
    c = f * c_prev + i * g                                # new cell state: keep part of the old, add part of the new
    h = o * np.tanh(c)                                    # new hidden state: a gated, squashed view of the cell state
    return h, c, (f, i, g, o)                             # both states, and the gate values


def init_lstm(input_dim, hidden_dim, seed, forget_bias=0.0, weight_scale=1.0):  # random LSTM weights -- Day 63
    rng = np.random.RandomState(seed)                     # private rng -> reproducible
    W = rng.randn(input_dim + hidden_dim, 4 * hidden_dim) * weight_scale / np.sqrt(input_dim + hidden_dim)  # one block per gate
    b = np.zeros(4 * hidden_dim)                          # biases, in the same f, i, g, o order
    b[:hidden_dim] = forget_bias                          # the forget gate's bias
    return W, b                                           # W: (input + hidden, 4 * hidden);  b: (4 * hidden,)


def lstm_forward(X, h0, c0, W, b, mask=None):           # the forward pass with its cache -- Day 64
    m, T, D = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    if mask is None:                                      # no padding: every position is real
        mask = np.ones((m, T))
    Hs, Cs = np.zeros((m, T, H)), np.zeros((m, T, H))     # every hidden state and every cell state
    F, I, G, O = (np.zeros((m, T, H)) for _ in range(4))  # every gate value, kept for the backward pass
    h, c = h0, c0                                         # the two states start where the caller says
    for t in range(T):                                    # one step at a time
        h_new, c_new, (f, i, g, o) = lstm_step_forward(X[:, t, :], h, c, W, b)  # Day 63's step, unchanged
        m_t = mask[:, t:t + 1]                              # 1 = real input at this step, 0 = padding
        h = m_t * h_new + (1.0 - m_t) * h                   # padding: keep the old hidden state
        c = m_t * c_new + (1.0 - m_t) * c                   # padding: keep the old cell state too
        Hs[:, t], Cs[:, t] = h, c                           # store the states
        F[:, t], I[:, t], G[:, t], O[:, t] = f, i, g, o     # store the gates
    cache = dict(X=X, h0=h0, c0=c0, W=W, b=b, mask=mask, H=Hs, C=Cs, F=F, I=I, G=G, O=O)  # everything backward will need
    return Hs, Cs, cache                                  # all hidden states, all cell states, and the cache


def lstm_backward(dH, dh_last, dc_last, cache):         # walk the cache backwards -- Day 65
    X, h0, c0, W, mask = cache["X"], cache["h0"], cache["c0"], cache["W"], cache["mask"]  # what the forward pass saw
    Hs, Cs, F, I, G, O = (cache[k] for k in ["H", "C", "F", "I", "G", "O"])  # what it computed
    m, T, D = X.shape                                     # batch size, sequence length, input size
    dW, db, dX = np.zeros_like(W), np.zeros(W.shape[1]), np.zeros_like(X)  # accumulators
    dh, dc = dh_last.copy(), dc_last.copy()               # gradients arriving at the final h and the final c
    for t in reversed(range(T)):                          # backward through time
        dh = dh + dH[:, t]                                  # plus whatever the loss sends straight to h_t (many-to-many)
        m_t = mask[:, t:t + 1]                              # 1 = this step was real, 0 = padding
        h_prev = Hs[:, t - 1] if t > 0 else h0              # the states that fed step t
        c_prev = Cs[:, t - 1] if t > 0 else c0
        f, i, g, o = F[:, t], I[:, t], G[:, t], O[:, t]     # this step's gates
        tanh_c = np.tanh(Cs[:, t])                          # recomputed rather than cached (Day 64)
        dh_real, dc_real = dh * m_t, dc * m_t               # only a real step passes gradient into its own computation
        do = dh_real * tanh_c                               # h = o * tanh(c)  ->  dL/do
        dc_new = dc_real + dh_real * o * (1.0 - tanh_c ** 2)  # ... and dL/dc: from the next step AND through tanh(c)
        da = np.concatenate([dc_new * c_prev * f * (1.0 - f),  # forget gate, then undo its sigmoid
                             dc_new * g * i * (1.0 - i),       # input gate, then undo its sigmoid
                             dc_new * i * (1.0 - g ** 2),      # candidate, then undo its tanh
                             do * o * (1.0 - o)], axis=1)      # output gate, then undo its sigmoid
        z = np.concatenate([X[:, t], h_prev], axis=1)       # what was multiplied by W at this step
        dW += z.T @ da                                      # ACCUMULATE: the same W at every step
        db += da.sum(axis=0)                                # ACCUMULATE
        dz = da @ W.T                                       # back through the matrix product
        dX[:, t] = dz[:, :D]                                # the part belonging to the input
        dh = dz[:, D:] + dh * (1.0 - m_t)                   # the part belonging to h_prev; padding passes dh straight through
        dc = dc_new * f + dc * (1.0 - m_t)                  # the direct route: times the forget gate
    return dW, db, dX, dh, dc                             # dh and dc are now the gradients at h0 and c0


# --- the GRU: Day 66, verbatim ----------------------------------------------------
def gru_step_forward(x_t, h_prev, W, b):                # one step of a GRU cell -- Day 66
    D, H = x_t.shape[1], h_prev.shape[1]                  # input size, hidden size
    zr = np.concatenate([x_t, h_prev], axis=1) @ W[:, :2 * H] + b[:2 * H]  # raw scores of the two gates, in one product
    z = sigmoid(zr[:, :H])                                # update gate: how much of the old state to KEEP (0..1)
    r = sigmoid(zr[:, H:])                                # reset gate: how much of the old state the candidate may SEE (0..1)
    n = np.tanh(x_t @ W[:D, 2 * H:] + (r * h_prev) @ W[D:, 2 * H:] + b[2 * H:])  # candidate, from x and the RESET old state
    h = z * h_prev + (1.0 - z) * n                        # keep part of the old state, fill the rest with the candidate
    return h, (z, r, n)                                   # the one new state, and the gates


def init_gru(input_dim, hidden_dim, seed, keep_bias=0.0):  # random GRU weights, columns in z, r, n order -- Day 66
    rng = np.random.RandomState(seed)                     # private rng -> reproducible
    W = rng.randn(input_dim + hidden_dim, 3 * hidden_dim) / np.sqrt(input_dim + hidden_dim)  # one block per gate
    b = np.zeros(3 * hidden_dim)                          # biases
    b[:hidden_dim] = keep_bias                            # the update gate's bias
    return W, b                                           # W: (input + hidden, 3 * hidden);  b: (3 * hidden,)


def gru_forward(X, h0, W, b, mask=None):                # the GRU loop, with padding and a cache -- Day 66
    m, T, D = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    if mask is None:                                      # no padding: every position is real
        mask = np.ones((m, T))
    Hs = np.zeros((m, T, H))                              # every hidden state
    Z, R, N = (np.zeros((m, T, H)) for _ in range(3))     # every gate value, for the backward pass
    h = h0                                                # starting state
    for t in range(T):                                    # one step at a time
        h_new, (z, r, n) = gru_step_forward(X[:, t], h, W, b)  # one GRU step
        m_t = mask[:, t:t + 1]                              # 1 = real step, 0 = padding
        h = m_t * h_new + (1.0 - m_t) * h                   # padding carries the state forward
        Hs[:, t] = h                                        # store it
        Z[:, t], R[:, t], N[:, t] = z, r, n                 # store the gates
    cache = dict(X=X, h0=h0, W=W, b=b, mask=mask, H=Hs, Z=Z, R=R, N=N)  # everything backward needs
    return Hs, cache                                      # all hidden states, and the cache


def gru_backward(dH, dh_last, cache):                   # walk the cache backwards -- Day 66
    X, h0, W, mask = cache["X"], cache["h0"], cache["W"], cache["mask"]  # what the forward pass saw
    Hs, Z, R, N = cache["H"], cache["Z"], cache["R"], cache["N"]  # what it computed
    m, T, D = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    dW, db, dX = np.zeros_like(W), np.zeros(W.shape[1]), np.zeros_like(X)  # accumulators
    dh = dh_last.copy()                                   # gradient arriving at the final hidden state
    for t in reversed(range(T)):                          # backward through time
        dh = dh + dH[:, t]                                  # plus whatever the loss sends straight to h_t
        m_t = mask[:, t:t + 1]                              # 1 = real step, 0 = padding
        h_prev = Hs[:, t - 1] if t > 0 else h0              # the state that fed step t
        z, r, n, x = Z[:, t], R[:, t], N[:, t], X[:, t]     # this step's gates and input
        d = dh * m_t                                        # only a real step passes gradient into its own computation
        dz = d * (h_prev - n)                               # h = z*h_prev + (1-z)*n  ->  dL/dz
        dn = d * (1.0 - z)                                  # ... dL/dn
        dh_prev = d * z + dh * (1.0 - m_t)                  # the direct route: times the update gate
        da_n = dn * (1.0 - n ** 2)                          # undo the candidate's tanh
        rh = r * h_prev                                     # what the candidate's hidden weights multiplied
        dW[:D, 2 * H:] += x.T @ da_n                        # candidate, input weights
        dW[D:, 2 * H:] += rh.T @ da_n                       # candidate, hidden weights
        db[2 * H:] += da_n.sum(axis=0)                      # candidate bias
        d_rh = da_n @ W[D:, 2 * H:].T                       # gradient for (r * h_prev)
        dr = d_rh * h_prev                                  # ... split into the reset gate
        dh_prev += d_rh * r                                 # ... and the old state, through the reset gate
        da_zr = np.concatenate([dz * z * (1.0 - z), dr * r * (1.0 - r)], axis=1)  # undo the two gates' sigmoids
        zin = np.concatenate([x, h_prev], axis=1)           # what the gates' weights multiplied
        dW[:, :2 * H] += zin.T @ da_zr                      # gate weights
        db[:2 * H] += da_zr.sum(axis=0)                     # gate biases
        dzin = da_zr @ W[:, :2 * H].T                       # back through the gates' product
        dX[:, t] = dzin[:, :D] + da_n @ W[:D, 2 * H:].T     # the input reached both the gates and the candidate
        dh = dh_prev + dzin[:, D:]                          # every route back to h_prev, added up
    return dW, db, dX, dh                                 # after the loop, dh is the gradient at h0


# --- the data: Day 62, verbatim ---------------------------------------------------
def read_tagged(name):                                  # read one "word<TAB>tag" file; a blank line ends a sentence
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


def encode(sentences, stoi, ttoi):                      # sentences -> padded id arrays plus a tag array -- Day 62
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


def prepare_data():                                     # load the three files, build the vocabularies, encode -- Day 62
    splits = {name: read_tagged(name) for name in ["train", "val", "test"]}  # sentences of (word, tag) pairs
    counts = Counter(w for s in splits["train"] for w, _ in s)  # word counts in the TRAINING sentences
    itos = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= 2)  # word vocabulary, as on Day 61
    stoi = {w: i for i, w in enumerate(itos)}               # word -> id
    tag_names = sorted({t for s in splits["train"] for _, t in s})  # the tag set, alphabetical
    ttoi = {t: i for i, t in enumerate(tag_names)}          # tag -> id
    data = [encode(splits[name], stoi, ttoi) for name in ["train", "val", "test"]]  # (ids, tags, mask) x 3
    return splits, counts, itos, stoi, tag_names, ttoi, data  # everything later sections need


def token_accuracy(pred, tags, mask):                   # fraction of REAL words tagged correctly -- Day 62
    return (pred == tags)[mask].mean()                    # padding positions are left out by the mask


# =============================================================================
# 1. ONE TAGGER, THREE CELLS -- NEW today
# =============================================================================
class Tagger:                                           # NEW: Day 62's RNNTagger with the recurrent cell as a setting
    def __init__(self, vocab_size, n_tags, cell="rnn", embed_dim=32, hidden_dim=32, lr=1.0, clip=1.0, seed=0, gate_bias=1.0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.cell, self.n_tags, self.hidden_dim = cell, n_tags, hidden_dim  # which cell, and the sizes needed later
        self.lr, self.clip = lr, clip                       # learning rate and clip threshold
        self.E = rng.randn(vocab_size, embed_dim) * 0.1      # embedding table (Day 61), drawn first: the same for every cell
        if cell == "rnn":                                    # the plain RNN, initialised EXACTLY as Day 62's tagger
            self.Wxh = rng.randn(embed_dim, hidden_dim) * np.sqrt(1.0 / embed_dim)  # input-to-hidden weights
            self.Whh = rng.randn(hidden_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)  # recurrent weights
            self.bh = np.zeros(hidden_dim)                     # hidden bias
            self.cell_params = ["Wxh", "Whh", "bh"]            # the cell's own parameters, by name
        elif cell == "lstm":                                 # Day 63's initialisation, forget bias = gate_bias
            self.W, self.b = init_lstm(embed_dim, hidden_dim, seed, forget_bias=gate_bias)
            self.cell_params = ["W", "b"]
        else:                                                # Day 66's initialisation, keep bias = gate_bias
            self.W, self.b = init_gru(embed_dim, hidden_dim, seed, keep_bias=gate_bias)
            self.cell_params = ["W", "b"]
        self.Why = rng.randn(hidden_dim, n_tags) * np.sqrt(1.0 / hidden_dim)  # output weights: hidden -> tags
        self.by = np.zeros(n_tags)                           # output bias
        self.params = ["E"] + self.cell_params + ["Why", "by"]  # every trainable parameter, by name

    def n_weights(self):                                   # how many numbers this network learns
        return {name: getattr(self, name).size for name in self.params}  # per parameter

    def run_cell(self, X):                                 # NEW: the only place the three cells differ going forward
        h0 = np.zeros((X.shape[0], self.hidden_dim))         # every sentence starts from a zero state
        if self.cell == "rnn":
            return rnn_forward(X, h0, self.Wxh, self.Whh, self.bh), h0  # Day 62
        if self.cell == "lstm":
            Hs, _, self.cache = lstm_forward(X, h0, h0, self.W, self.b)  # Day 64 (c0 is also zero)
            return Hs, h0
        Hs, self.cache = gru_forward(X, h0, self.W, self.b)  # Day 66
        return Hs, h0

    def forward(self, ids):                                # ids: (m, T) word ids -> probabilities, one row per word
        m, T = ids.shape                                     # batch size, padded length
        self.ids = ids                                       # cache for backward
        self.X = self.E[ids]                                 # embedding lookup, (m, T, D)
        self.H, self.h0 = self.run_cell(self.X)              # every hidden state, (m, T, H)
        self.H2 = self.H.reshape(m * T, self.hidden_dim)     # every word of every sentence as its own row
        self.probs = softmax(self.H2 @ self.Why + self.by)   # a tag distribution for EVERY word, (m*T, n_tags)
        return self.probs                                    # hand back

    def loss(self, tags, mask):                            # mean cross-entropy over the REAL words only -- Day 62
        p_true = self.probs[np.arange(tags.size), tags.ravel()]  # probability given to each word's true tag
        return -np.sum(np.log(p_true + 1e-9) * mask.ravel()) / mask.sum()  # padding counts 0; divide by the real word count

    def gradients(self, tags, mask):                       # gradients of that loss for every parameter; no update
        m, T = self.ids.shape                                # batch size, padded length
        dlogits = (self.probs - one_hot(tags.ravel(), self.n_tags)) * mask.ravel()[:, None] / mask.sum()  # zero on padding
        grads = dict(Why=self.H2.T @ dlogits, by=dlogits.sum(axis=0))  # output-layer gradients
        dH = (dlogits @ self.Why.T).reshape(m, T, self.hidden_dim)  # a gradient for EVERY timestep's hidden state
        zeros = np.zeros((m, self.hidden_dim))               # nothing extra arrives after the last step
        if self.cell == "rnn":                               # NEW: dispatch to the matching backward pass
            grads["Wxh"], grads["Whh"], grads["bh"], dX = rnn_backward(dH, self.X, self.H, self.h0, self.Wxh, self.Whh)
        elif self.cell == "lstm":
            grads["W"], grads["b"], dX, _, _ = lstm_backward(dH, zeros, zeros, self.cache)
        else:
            grads["W"], grads["b"], dX, _ = gru_backward(dH, zeros, self.cache)
        grads["E"] = np.zeros_like(self.E)                   # one gradient row per vocabulary word
        np.add.at(grads["E"], self.ids, dX)                  # add each position's gradient into its word's row (Day 61)
        return grads                                          # dict: parameter name -> gradient

    def train_step(self, ids, tags, mask):                 # one SGD step with Day 60's clipping; returns the loss
        self.forward(ids)                                    # forward pass
        loss = self.loss(tags, mask)                         # batch loss, before the update
        grads, _, _ = clip_by_global_norm(self.gradients(tags, mask), self.clip)  # gradients, clipped
        for name, g in grads.items():                        # plain SGD
            getattr(self, name)[...] -= self.lr * g            # in place
        return loss                                          # for the record

    def predict(self, ids):                                # predicted tag id for every position, (m, T)
        pred = self.forward(ids).argmax(axis=1).reshape(ids.shape)  # most probable tag per row, back in sentence shape
        self.drop_caches()                                   # a prediction needs no backward pass: free the memory
        return pred                                          # hand back

    def drop_caches(self):                                 # forget everything a forward pass stored for the backward pass
        for name in ["ids", "X", "H", "h0", "H2", "probs", "cache"]:  # the big per-batch arrays
            self.__dict__.pop(name, None)                      # remove it if present

    def snapshot(self):                                    # a copy of the network that holds only its parameters and settings
        self.drop_caches()                                   # nothing per-batch is worth copying
        return copy.deepcopy(self)                           # an independent copy


def fit(data, vocab_size, n_tags, seed, epochs=EPOCHS, batch_size=32, **net_kwargs):  # Day 62's fit(), plus a timer
    (ids_tr, tags_tr, mask_tr), (ids_va, tags_va, mask_va), (ids_te, tags_te, mask_te) = data  # the three splits
    net = Tagger(vocab_size, n_tags, seed=seed, **net_kwargs)  # fresh network
    rng = np.random.RandomState(seed)                       # shuffles the order of sentences each epoch
    lens_tr = mask_tr.sum(axis=1)                           # real length of every training sentence
    hist = dict(loss=[], train=[], val=[], test=[], seconds=[])  # one entry per epoch
    best_val, best_net, best_epoch = -1.0, None, -1          # the snapshot with the best validation accuracy
    for epoch in range(epochs):                              # full training budget
        start_time = time.perf_counter()                       # NEW: time the training part of the epoch
        perm = rng.permutation(len(ids_tr))                    # fresh sentence order
        losses = []                                            # this epoch's per-step losses
        for start in range(0, len(perm), batch_size):          # mini-batches
            idx = perm[start:start + batch_size]                 # this batch's sentences
            T_batch = lens_tr[idx].max()                         # pad only to the longest sentence IN THIS BATCH
            losses.append(net.train_step(ids_tr[idx, :T_batch], tags_tr[idx, :T_batch], mask_tr[idx, :T_batch]))  # one step
        hist["seconds"].append(time.perf_counter() - start_time)  # NEW: seconds spent training this epoch
        hist["loss"].append(np.mean(losses))                   # mean training loss
        hist["train"].append(token_accuracy(net.predict(ids_tr), tags_tr, mask_tr))  # training accuracy NOW
        hist["val"].append(token_accuracy(net.predict(ids_va), tags_va, mask_va))    # validation accuracy NOW
        hist["test"].append(token_accuracy(net.predict(ids_te), tags_te, mask_te))   # test accuracy NOW (never used to choose)
        if hist["val"][-1] > best_val:                           # a new best on VALIDATION
            best_val, best_net, best_epoch = hist["val"][-1], net.snapshot(), epoch  # snapshot it (parameters only)
    return best_net, best_epoch, {k: np.array(v) for k, v in hist.items()}  # best snapshot, when, and the full record


RUNS = {}                                               # cache: identical runs requested by two sections train only once


def fit_cached(data, vocab_size, n_tags, seed, **net_kwargs):  # fit(), remembered by its settings
    key = (seed, tuple(sorted(net_kwargs.items())))          # everything that defines the run
    if key not in RUNS:                                      # first request
        RUNS[key] = fit(data, vocab_size, n_tags, seed, **net_kwargs)  # train it
    return RUNS[key]                                         # later requests reuse the result


# =============================================================================
# DEMOS
# =============================================================================
def demo_cells(itos, tag_names, data):
    print("=" * 70)
    print("0. ONE TAGGER, THREE CELLS -- what is shared and what is swapped")
    print("=" * 70)
    print("Day 62's tagger: words -> embedding (32) -> recurrent cell (hidden 32) -> a tag for "
          "every word. Only the cell changes. The embedding table is drawn first from the "
          "same seed, so all three start from identical word vectors; the RNN is initialised "
          "exactly as on Day 62.")
    V, K = len(itos), len(tag_names)                           # vocabulary size, number of tags
    print(f"\nvocabulary {V} words, {K} tags; training sentences {len(data[0][0])}")
    print(f"\n{'cell':<6}{'embedding':>11}{'cell weights':>14}{'output layer':>14}{'total':>9}{'cell share':>12}")
    for cell in CELLS:                                         # one row per cell
        n = Tagger(V, K, cell=cell).n_weights()                  # weight count of each parameter
        cell_w = sum(n[p] for p in n if p not in ("E", "Why", "by"))  # the recurrent cell's own weights
        out_w = n["Why"] + n["by"]                               # the output layer's
        total = sum(n.values())                                  # everything
        print(f"{NAMES[cell]:<6}{n['E']:>11}{cell_w:>14}{out_w:>14}{total:>9}{cell_w / total:>12.1%}")
    print("Actual output from this lesson's run.")


def numeric_gradient(net, name, ids, tags, mask, eps=1e-5):  # central-difference gradient of the masked loss -- Day 62
    P = getattr(net, name)                                  # the parameter array itself, not a copy
    numeric = np.zeros_like(P)                              # same shape as the parameter
    for pos in np.ndindex(P.shape):                         # every single entry
        old = P[pos]                                          # remember it
        P[pos] = old + eps; net.forward(ids); loss_up = net.loss(tags, mask)    # loss with this entry nudged up
        P[pos] = old - eps; net.forward(ids); loss_down = net.loss(tags, mask)  # loss with this entry nudged down
        P[pos] = old                                          # restore
        numeric[pos] = (loss_up - loss_down) / (2 * eps)      # the measured slope
    return numeric                                          # one slope per entry


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("1. GRADIENT CHECK -- all three cells, through the whole tagger")
    print("=" * 70)
    print("Day 62's check: a tiny tagger (12 words, 3 tags, embedding 4, hidden 5), four "
          "sentences of lengths 5, 2, 4, 1 with one word repeated, every entry of every "
          "parameter nudged by +-1e-5. Pass rule: |numeric - analytic| <= 1e-10 + 1e-6 * "
          "|numeric| for every entry.")
    rng = np.random.RandomState(3)                             # private rng for the check (Day 62's)
    lens = np.array([5, 2, 4, 1])                              # four different lengths
    mask = np.arange(5)[None, :] < lens[:, None]               # real-word mask
    ids = rng.randint(2, 12, size=(4, 5))                      # random real word ids
    ids[0, 3] = ids[0, 1]                                      # the SAME word twice in one sentence
    ids[~mask] = PAD                                           # pad past each sentence's length
    tags = rng.randint(0, 3, size=(4, 5))                      # random tags
    print(f"\n{'cell':<6}{'parameter':>10}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>7}")
    verdict = {}                                               # cell -> did every entry pass?
    for cell in CELLS:                                         # each cell in turn
        net = Tagger(12, 3, cell=cell, embed_dim=4, hidden_dim=5, clip=None, seed=1)  # tiny network
        net.forward(ids)                                         # forward pass
        analytic = net.gradients(tags, mask)                     # what the backward pass computed
        ok_all = True                                            # every parameter passes?
        for name in net.params:                                  # every parameter of this cell's tagger
            numeric = numeric_gradient(net, name, ids, tags, mask)  # measured slopes
            diff = np.abs(numeric - analytic[name])                # absolute disagreement
            rel = diff / (np.abs(numeric) + np.abs(analytic[name]) + 1e-12)  # relative disagreement
            ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(numeric)))  # the pass rule
            ok_all = ok_all and ok                                 # track the verdict
            print(f"{NAMES[cell]:<6}{name:>10}{numeric.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>7}")
        verdict[cell] = ok_all and bool(np.all(analytic["E"][PAD] == 0))  # plus: padding's embedding row gets no gradient
    print("every entry passes, and <pad>'s embedding gradient is exactly zero: "
          + ", ".join(f"{NAMES[c]} {verdict[c]}" for c in CELLS))
    print("Actual output from this lesson's run.")


def demo_settings(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("2. FAIR SETTINGS -- each cell gets its own learning rate, chosen on validation")
    print("=" * 70)
    print("One learning rate for all three would favour whichever cell it happens to suit. "
          "Each cell is trained at four learning rates (and the gated cells at gate bias 0 "
          "and 1), seed 0, 25 epochs, Day 62's batch size 32 and clip 1.0. The setting with "
          "the best VALIDATION accuracy is kept; test accuracy is not looked at here.")
    V, K = len(itos), len(tag_names)                           # sizes
    lrs = [0.5, 1.0, 2.0, 4.0]                                 # four learning rates, each double the last
    grid = {"rnn": [(lr, 0.0) for lr in lrs],                  # the RNN has no gate bias
            "lstm": [(lr, gb) for gb in [0.0, 1.0] for lr in lrs],  # forget bias 0 or 1
            "gru": [(lr, gb) for gb in [0.0, 1.0] for lr in lrs]}   # keep bias 0 or 1
    print(f"\n{'cell':<6}{'lr':>5}{'gate bias':>11}{'best val':>10}{'at epoch':>10}{'val at end':>12}{'train at end':>14}")
    chosen = {}                                                # cell -> (lr, gate bias) with the best validation accuracy
    for cell in CELLS:                                         # each cell
        best = (-1.0, None)                                      # best validation accuracy and its setting
        for lr, gb in grid[cell]:                                # every setting in its grid
            _, be, h = fit_cached(data, V, K, seed=0, cell=cell, lr=lr, gate_bias=gb)  # train (seed 0)
            print(f"{NAMES[cell]:<6}{lr:>5}{gb if cell != 'rnn' else '-':>11}{h['val'][be]:>10.4f}{be:>10}"
                  f"{h['val'][-1]:>12.4f}{h['train'][-1]:>14.4f}")
            if h["val"][be] > best[0]:                             # a better validation accuracy
                best = (h["val"][be], (lr, gb))                      # remember the setting
        chosen[cell] = best[1]                                   # keep the winner
    print("\nchosen on validation: " + ", ".join(
        f"{NAMES[c]} lr {chosen[c][0]}" + (f" gate bias {chosen[c][1]}" if c != "rnn" else "") for c in CELLS))
    print("Actual output from this lesson's run.")
    return chosen                                              # later sections train at these settings


def demo_compare(itos, tag_names, data, chosen):
    print("\n" + "=" * 70)
    print("3. THE COMPARISON -- three seeds per cell, at its chosen setting")
    print("=" * 70)
    print("Test accuracy at each run's best validation epoch (Day 62's rule). The no-memory "
          "baseline is Day 62's most-frequent-tag: every word gets the tag it had most often "
          "in training.")
    V, K = len(itos), len(tag_names)                           # sizes
    (ids_tr, tags_tr, mask_tr), _, (ids_te, tags_te, mask_te) = data  # training and test splits
    seen = defaultdict(Counter)                                # word id -> how often each tag was seen with it (Day 62)
    for w, t in zip(ids_tr[mask_tr], tags_tr[mask_tr]):        # every real training word
        seen[w][t] += 1                                          # count the pair
    mft = np.full(V, seen[UNK].most_common(1)[0][0])           # default: the tag most common among <unk> training words
    for w, c in seen.items():                                  # every word seen in training
        mft[w] = c.most_common(1)[0][0]                          # its single most frequent tag
    mft_acc = token_accuracy(mft[ids_te], tags_te, mask_te)    # the baseline on test
    print(f"\n{'cell':<6}{'seed':>5}{'best epoch':>12}{'val there':>11}{'test there':>12}{'train at end':>14}")
    tests = {}                                                 # cell -> test accuracy per seed
    for cell in CELLS:                                         # each cell
        lr, gb = chosen[cell]                                    # its chosen setting
        tests[cell] = []                                         # test accuracy per seed
        for seed in SEEDS:                                       # three seeds
            _, be, h = fit_cached(data, V, K, seed=seed, cell=cell, lr=lr, gate_bias=gb)  # train (or reuse seed 0)
            tests[cell].append(h["test"][be])                      # the honest number
            print(f"{NAMES[cell]:<6}{seed:>5}{be:>12}{h['val'][be]:>11.4f}{h['test'][be]:>12.4f}{h['train'][-1]:>14.4f}")
    print(f"\n{'cell':<6}{'mean test':>11}{'min':>8}{'max':>8}{'vs most-frequent-tag':>22}")
    for cell in CELLS:                                         # summary per cell
        t = np.array(tests[cell])                                # three seeds
        print(f"{NAMES[cell]:<6}{t.mean():>11.4f}{t.min():>8.4f}{t.max():>8.4f}{t.mean() - mft_acc:>+22.4f}")
    n = int(mask_te.sum())                                     # test words
    print(f"most-frequent-tag baseline on test: {mft_acc:.4f}; test words {n}, one standard error "
          f"about {np.sqrt(0.84 * 0.16 / n):.4f}")
    print("Actual output from this lesson's run.")
    _, _, h_rnn = fit_cached(data, V, K, seed=0, cell="rnn", lr=1.0, gate_bias=0.0)  # the RNN at Day 62's own setting
    print(f"\nconsistency check: the RNN at Day 62's setting (lr 1.0, seed 0) reaches test "
          f"{h_rnn['test'][int(np.argmax(h_rnn['val']))]:.4f} at its best validation epoch (Day 62 reported 0.8338)")
    print("Actual output from this lesson's run.")
    seen_ambiguous = np.array([len(seen[w]) > 1 for w in range(V)])  # words that had MORE than one tag in training
    seen_ambiguous[UNK] = False                                # <unk> is its own group
    return tests, mft, mft_acc, seen_ambiguous                 # later sections reuse these


def demo_where(itos, tag_names, data, chosen, mft, ambiguous):
    print("\n" + "=" * 70)
    print("4. WHERE THE CELLS DIFFER -- word groups and position in the sentence")
    print("=" * 70)
    print("Day 62's three word groups (UNAMBIGUOUS: known word with one tag in training; "
          "AMBIGUOUS: known word with several; UNKNOWN: <unk>), and the word's position in "
          "its sentence. Accuracy is the mean of the three seeds.")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    groups = {"all words": mask_te,                             # every real test word
              "unambiguous": mask_te & (ids_te != UNK) & ~ambiguous[ids_te],  # known, one tag in training
              "ambiguous": mask_te & ambiguous[ids_te],            # known, several tags in training
              "unknown": mask_te & (ids_te == UNK)}                # not in the vocabulary
    pos = np.tile(np.arange(ids_te.shape[1]), (len(ids_te), 1))  # each word's position in its sentence
    spans = [(0, 1), (1, 3), (3, 10), (10, 20), (20, 100)]     # position ranges: first word, 2nd-3rd, 4th-10th, ...
    span_names = ["1st word", "2nd-3rd", "4th-10th", "11th-20th", "21st on"]  # their labels
    for (a, b), name in zip(spans, span_names):                # add each range as a group
        groups[name] = mask_te & (pos >= a) & (pos < b)
    preds = {cell: [fit_cached(data, V, K, seed=s, cell=cell, lr=chosen[cell][0], gate_bias=chosen[cell][1])[0].predict(ids_te)
                    for s in SEEDS] for cell in CELLS}         # test predictions: 3 seeds per cell
    preds["most-freq"] = [mft[ids_te]]                         # the no-memory baseline
    word_groups = ["all words", "unambiguous", "ambiguous", "unknown"]  # the first table
    table = {}                                                 # model -> {group: accuracy}
    for title, cols in [("by word group", word_groups), ("by position in the sentence", span_names)]:
        print(f"\n{title}:")
        print(f"{'':<11}" + "".join(f"{g:>12}" for g in cols))
        print(f"{'share':<11}" + "".join(f"{groups[g].sum() / mask_te.sum():>12.1%}" for g in cols))
        for model in ["most-freq"] + CELLS:                      # one row per model
            accs = {g: np.mean([(p == tags_te)[groups[g]].mean() for p in preds[model]]) for g in cols}  # mean over seeds
            table.setdefault(model, {}).update(accs)               # keep for the plot
            label = NAMES.get(model, model)                        # display name
            print(f"{label:<11}" + "".join(f"{accs[g]:>12.4f}" for g in cols))
    print("Actual output from this lesson's run.")
    print(f"\nhow often two cells (seed 0 each) give a word DIFFERENT tags, and who is right then:")
    print(f"{'pair':<12}{'disagree':>10}{'first right':>13}{'second right':>14}{'neither':>9}")
    for a, b in [("rnn", "lstm"), ("rnn", "gru"), ("lstm", "gru")]:  # every pair of cells
        pa, pb = preds[a][0], preds[b][0]                          # seed-0 predictions
        dis = (pa != pb) & mask_te                                 # real words where they disagree
        print(f"{NAMES[a] + '-' + NAMES[b]:<12}{dis.sum() / mask_te.sum():>10.2%}"
              f"{((pa == tags_te) & dis).sum() / dis.sum():>13.1%}{((pb == tags_te) & dis).sum() / dis.sum():>14.1%}"
              f"{((pa != tags_te) & (pb != tags_te) & dis).sum() / dis.sum():>9.1%}")
    print("Actual output from this lesson's run.")
    return table, groups                                       # the plot reuses the table


def probs_at(net, ids, t_idx, chunk=1000):              # tag distribution at position t_idx[j] of sentence j, a chunk at a time
    out = []                                              # one block of rows per chunk
    for s in range(0, len(ids), chunk):                   # chunks keep the forward pass's memory small
        p = net.forward(ids[s:s + chunk]).reshape(len(ids[s:s + chunk]), ids.shape[1], -1)  # (chunk, T, K)
        out.append(p[np.arange(len(p)), t_idx[s:s + chunk]])  # keep only each sentence's target position
    net.drop_caches()                                     # free the forward pass's arrays
    return np.concatenate(out)                            # (n, K)


def demo_how_far(itos, tag_names, data, chosen):
    print("\n" + "=" * 70)
    print("5. HOW FAR BACK DOES EACH TAGGER LOOK? -- change one earlier word")
    print("=" * 70)
    print("For every test sentence, a target word t is picked at random; the word k positions "
          "EARLIER is replaced by a random vocabulary word, and the tagger's prediction at t is "
          "compared before and after. Two numbers per distance k: how often the predicted tag "
          "changes, and the mean total-variation distance between the two probability "
          "distributions (0 = identical, 1 = no overlap). Words after t cannot matter: the "
          "taggers read left to right. 5 random draws per sentence, three seeds per cell.")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    lens = mask_te.sum(axis=1)                                 # real length of every test sentence
    distances = [1, 2, 3, 4, 6, 8, 12]                         # how far back the change is made
    results = {cell: {"flip": [], "tv": []} for cell in CELLS}  # cell -> lists over distances
    counts = []                                                # how many (sentence, draw) pairs each distance used
    for k in distances:                                        # each distance
        rng = np.random.RandomState(100 + k)                     # the SAME random changes for every cell
        rows = np.where(lens > k)[0]                             # sentences long enough to have a word k before another
        draws = []                                               # (row, target, replacement word) triples
        for _ in range(5):                                       # 5 draws per sentence
            for r in rows:                                         # each long-enough sentence
                t = rng.randint(k, lens[r])                          # a target position with a word k before it
                draws.append((r, t, rng.randint(2, V)))              # a random real vocabulary word (not pad, not unk)
        draws = np.array(draws)                                  # (n, 3)
        counts.append(len(draws))                                # remember the sample size
        r_idx, t_idx, new_w = draws[:, 0], draws[:, 1], draws[:, 2]  # unpack
        changed = ids_te[r_idx].copy()                           # one modified copy of the sentence per draw
        changed[np.arange(len(draws)), t_idx - k] = new_w        # replace the word k before the target
        same = changed[np.arange(len(draws)), t_idx - k] == ids_te[r_idx, t_idx - k]  # the random word happened to be the same
        for cell in CELLS:                                       # each cell
            flips, tvs = [], []                                    # per seed
            for s in SEEDS:                                        # three seeds
                net = fit_cached(data, V, K, seed=s, cell=cell, lr=chosen[cell][0], gate_bias=chosen[cell][1])[0]
                p_orig = probs_at(net, ids_te[r_idx], t_idx)         # the target's tag distribution before the change
                p_new = probs_at(net, changed, t_idx)                # ... and after
                keep = ~same                                         # count only draws where the word really changed
                flips.append((p_orig.argmax(1) != p_new.argmax(1))[keep].mean())  # share of predictions that changed
                tvs.append(0.5 * np.abs(p_orig - p_new).sum(axis=1)[keep].mean())  # mean total-variation distance
            results[cell]["flip"].append(np.mean(flips))            # mean over seeds
            results[cell]["tv"].append(np.mean(tvs))
    print(f"\n{'k':>4}{'draws':>8}" + "".join(f"{NAMES[c] + ' changed':>14}" for c in CELLS)
          + "".join(f"{NAMES[c] + ' TV':>10}" for c in CELLS))
    for j, k in enumerate(distances):                          # one row per distance
        print(f"{k:>4}{counts[j]:>8}" + "".join(f"{results[c]['flip'][j]:>14.2%}" for c in CELLS)
              + "".join(f"{results[c]['tv'][j]:>10.4f}" for c in CELLS))
    print("Actual output from this lesson's run.")
    return distances, results                                  # the plot reuses these


def demo_size_match(itos, tag_names, data, chosen, tests):
    print("\n" + "=" * 70)
    print("6. A FAIRER SIZE MATCH -- an RNN with as many cell weights as the LSTM")
    print("=" * 70)
    print("At hidden 32 the LSTM's cell has four times the RNN's weights. An RNN with hidden 76 "
          "has about as many cell weights as the LSTM at 32 (and the GRU at 38 likewise). Same "
          "learning rate as each cell's chosen one, three seeds.")
    V, K = len(itos), len(tag_names)                           # sizes
    print(f"\n{'model':<16}{'cell weights':>14}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}")
    for cell, hidden in [("rnn", 32), ("rnn", 76), ("gru", 38), ("lstm", 32)]:  # three sizes of cell weight count
        lr, gb = chosen[cell]                                    # that cell's chosen setting
        accs = []                                                # test accuracy per seed
        for s in SEEDS:                                          # three seeds
            if hidden == 32:                                       # the runs from Section 3
                _, be, h = fit_cached(data, V, K, seed=s, cell=cell, lr=lr, gate_bias=gb)
            else:                                                  # the size-matched runs
                _, be, h = fit_cached(data, V, K, seed=s, cell=cell, lr=lr, gate_bias=gb, hidden_dim=hidden)
            accs.append(h["test"][be])                             # test accuracy at the best validation epoch
        n = Tagger(V, K, cell=cell, hidden_dim=hidden).n_weights()  # weight counts
        cell_w = sum(n[p] for p in n if p not in ("E", "Why", "by"))  # the cell's own
        print(f"{NAMES[cell] + ' hidden ' + str(hidden):<16}{cell_w:>14}" + "".join(f"{a:>9.4f}" for a in accs)
              + f"{np.mean(accs):>9.4f}")
    print("Actual output from this lesson's run.")


def demo_cost(itos, tag_names, data, chosen):
    print("\n" + "=" * 70)
    print("7. WHAT EACH CELL COSTS -- seconds per training epoch")
    print("=" * 70)
    print("Measured inside every training run above (training steps only, not evaluation), "
          "matrix library on one thread. Median over all 25 epochs of the three seeds at the "
          "chosen settings. Timings vary by machine.")
    V, K = len(itos), len(tag_names)                           # sizes
    med = {}                                                   # cell -> median seconds per epoch
    for cell in CELLS:                                         # each cell
        secs = np.concatenate([fit_cached(data, V, K, seed=s, cell=cell, lr=chosen[cell][0], gate_bias=chosen[cell][1])[2]["seconds"]
                               for s in SEEDS])                  # every epoch of the three seeds
        med[cell] = np.median(secs)                              # the typical epoch
    print(f"\n{'cell':<6}{'seconds / epoch':>17}{'vs RNN':>9}")
    for cell in CELLS:                                         # one row per cell
        print(f"{NAMES[cell]:<6}{med[cell]:>17.2f}{med[cell] / med['rnn']:>9.2f}")
    print("Actual output from this lesson's run.")


def plot_results(itos, tag_names, data, chosen, table, distances, results):  # two figures from the measurements above
    V, K = len(itos), len(tag_names)                           # sizes
    fig, ax = plt.subplots(figsize=(9, 4.6))                    # validation curves, all seeds
    for cell in CELLS:                                         # each cell
        for s in SEEDS:                                          # each seed, one line
            h = fit_cached(data, V, K, seed=s, cell=cell, lr=chosen[cell][0], gate_bias=chosen[cell][1])[2]
            ax.plot(h["val"], color=COLORS[cell], alpha=0.85, linewidth=1.4,
                    label=f"{NAMES[cell]} (lr {chosen[cell][0]})" if s == 0 else None)  # label the first seed only
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (real words)"); ax.set_ylim(0.6, 0.86)
    ax.set_title("Day 62's tagger with three different cells: validation accuracy, three seeds each", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("cells_validation_curves.png", dpi=110); plt.close()  # save and free
    print("\nsaved cells_validation_curves.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))       # where they differ | how far they look
    groups = ["all words", "unambiguous", "ambiguous", "unknown"]  # panel 1 clusters
    x = np.arange(len(groups))                                  # one cluster per group
    models = ["most-freq"] + CELLS                              # four bars per cluster
    for j, model in enumerate(models):                          # each model
        a1.bar(x + (j - 1.5) * 0.2, [table[model][g] for g in groups], width=0.2,
               color=COLORS.get(model, "#d9d9d9"), edgecolor="#555555", linewidth=0.5,
               label=NAMES.get(model, "most-frequent-tag"))
    a1.set_xticks(x); a1.set_xticklabels(groups); a1.set_ylim(0, 1.15); a1.set_ylabel("test accuracy (mean of 3 seeds)")
    a1.set_title("Where the cells differ", fontsize=10); a1.legend(fontsize=8, loc="upper right", ncol=4)
    for cell in CELLS:                                         # panel 2: one line per cell
        a2.plot(distances, results[cell]["tv"], color=COLORS[cell], marker="o", markersize=4, label=NAMES[cell])
    a2.set_yscale("log"); a2.set_xlabel("distance k: the changed word is k positions before the target")
    a2.set_ylabel("mean change in the target's tag distribution (TV)")
    a2.set_title("How far back each tagger looks", fontsize=10); a2.legend(fontsize=8); a2.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("cells_where_and_how_far.png", dpi=110); plt.close()  # save and free
    print("saved cells_where_and_how_far.png")


def draw_diagram():                                     # one tagger, a slot for the cell, and the three cells that fit it
    fig, ax = plt.subplots(figsize=(13, 5.4))                   # wide canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")      # plain coordinate grid, no axes

    def box(x, y, w, h, title, sub, color, lw=1.2):              # a rounded box with a bold title and a smaller second line
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#333333", linewidth=lw))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1, color="#333333"):                  # one arrow from (x0, y0) to (x1, y1)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5))

    ax.text(6.5, 5.1, "Day 67: one tagger, the recurrent cell swapped, everything else identical", ha="center",
            fontsize=12, fontweight="bold")
    chain = [(0.3, "words -> ids", "Day 62's vocabulary", "#E6F1F8"), (2.9, "embedding", "32 numbers per word", "#E6F1F8"),
             (5.5, "recurrent cell", "RNN | LSTM | GRU", "#FAEEDA"), (8.1, "tag scores", "softmax at every word", "#E6F1F8"),
             (10.7, "loss on real words", "Day 62's mask", "#E6F1F8")]
    for x, title, sub, color in chain:                           # the tagger, left to right
        box(x, 3.4, 2.1, 1.0, title, sub, color, lw=2.2 if title == "recurrent cell" else 1.2)
    for x, _, _, _ in chain[:-1]:                                # arrows between neighbours
        arrow(x + 2.14, 3.9, x + 2.46, 3.9)
    cells = [(0.6, "RNN (Days 57-62)", "h = tanh([x, h] W + b)", "1 block of weights", "#F2F2F2"),
             (4.75, "LSTM (Days 63-65)", "c = f*c + i*g;  h = o*tanh(c)", "4 blocks, 2 states", "#E3EEF8"),
             (8.9, "GRU (Day 66)", "h = z*h + (1 - z)*n", "3 blocks, 1 state", "#E5F2E1")]
    for x, title, eq, sub, color in cells:                       # the three cells that fit the slot
        ax.add_patch(FancyBboxPatch((x, 0.5), 3.5, 1.5, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x + 1.75, 1.67, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + 1.75, 1.22, eq, ha="center", va="center", fontsize=8.5, family="monospace")
        ax.text(x + 1.75, 0.8, sub, ha="center", va="center", fontsize=8)
        arrow(x + 1.75, 2.06, 6.55, 3.34, "#B07A1A")             # each one plugs into the cell slot
    ax.text(6.5, 0.15, "same data, same embedding start, same batches, same clipping -- only the cell and its learning rate differ",
            ha="center", fontsize=8.5, style="italic")
    plt.savefig("three_cells_tagger_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved three_cells_tagger_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Every tagger today reads left to right, so the tag of a word is decided before the "
          "words after it are seen. Day 62 measured what that costs: a right-to-left tagger is "
          "right on many words the left-to-right one gets wrong. Day 68 builds the two standard "
          "extensions: stacking cells in layers, and a bidirectional tagger that reads both ways "
          "at once.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread: same results, much faster on this machine (Day 64)
        splits, counts, itos, stoi, tag_names, ttoi, data = prepare_data()  # load and encode once (Day 62)
        demo_cells(itos, tag_names, data)                    # 0. one tagger, three cells
        demo_gradient_check()                                # 1. all three are correct
        chosen = demo_settings(itos, tag_names, data)        # 2. each cell's own learning rate, on validation
        tests, mft, mft_acc, ambiguous = demo_compare(itos, tag_names, data, chosen)  # 3. the comparison
        table, groups = demo_where(itos, tag_names, data, chosen, mft, ambiguous)  # 4. where they differ
        distances, results = demo_how_far(itos, tag_names, data, chosen)  # 5. how far back they look
        demo_size_match(itos, tag_names, data, chosen, tests)  # 6. equal cell weights
        demo_cost(itos, tag_names, data, chosen)             # 7. time per epoch
        plot_results(itos, tag_names, data, chosen, table, distances, results)  # the two figures
        draw_diagram()                                       # the diagram used in the guides
        note_on_what_comes_next()                            # 8. Day 68


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
