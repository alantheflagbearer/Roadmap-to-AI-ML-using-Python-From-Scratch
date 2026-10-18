"""
Day 68: Stacked and Bidirectional RNNs, From Scratch

Day 67 put three recurrent cells into Day 62's tagger and found that the
cell barely matters on this task: every tagger reads left to right, so
when it tags a word it has not yet seen the words that follow. Day 62
had already measured what that leaves out: a right-to-left copy of the
tagger is right on many words the left-to-right one gets wrong.

Today's two standard extensions, built on Day 67's LSTM tagger:

  BIDIRECTIONAL  a second LSTM reads each sentence right to left; at every
                 word the two states are joined, so the tag sees the whole
                 sentence.
  STACKED        a second layer of LSTMs reads the first layer's states
                 instead of word vectors.

What is built and measured, reported as it came out:

  0. Four taggers: 1 or 2 layers, one direction or two; their sizes.
  1. Checks: reversing padded sentences; a gradient check through two
     layers and two directions; a word's tag now depends on later words.
  2. All four trained, three seeds each, plus a size control.
  3. Where reading both ways helps: word groups, and the tags that gain.
  4. Which words each tagger looks at: Day 67's probe, now also AFTER the
     target word.
  5. What each costs in training time.

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
SEEDS = [0, 1, 2]                                       # every trained comparison uses these three seeds (Day 62's)
EPOCHS = 25                                             # Day 62's training budget
LR, FORGET_BIAS = 4.0, 1.0                                # Day 67's validation choice for the LSTM tagger

# =============================================================================
# 0. REUSED -- verbatim from Days 56-67
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


def reverse_padded(A, lens):                            # NEW: write every sequence backwards; padding stays where it is
    T = A.shape[1]                                        # padded length
    t = np.arange(T)[None, :]                             # position index, (1, T)
    idx = np.where(t < lens[:, None], lens[:, None] - 1 - t, t)  # real positions flipped, padding positions unchanged
    if A.ndim == 3:                                       # (m, T, D): the same flip for every feature
        idx = idx[:, :, None]
    return np.take_along_axis(A, idx, axis=1)             # gather; doing it twice gives back the original


# =============================================================================
# 1. A TAGGER WITH LAYERS AND TWO DIRECTIONS -- NEW today
# =============================================================================
class DeepTagger:                                       # NEW: Day 67's LSTM tagger, with stacked layers and an optional backward reader
    def __init__(self, vocab_size, n_tags, layers=1, bidirectional=False, embed_dim=32, hidden_dim=32,
                 lr=LR, clip=1.0, seed=0, forget_bias=FORGET_BIAS):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.layers, self.bi, self.n_tags, self.hidden_dim = layers, bidirectional, n_tags, hidden_dim  # the shape of the network
        self.lr, self.clip = lr, clip                       # learning rate and clip threshold
        self.E = rng.randn(vocab_size, embed_dim) * 0.1      # embedding table, drawn first (as on Day 67)
        self.directions = ["f", "b"] if bidirectional else ["f"]  # f = left to right, b = right to left
        width = hidden_dim * len(self.directions)            # what one layer hands to the next: both directions side by side
        self.params = ["E"]                                  # every trainable parameter, by name
        for l in range(layers):                              # each layer
            in_dim = embed_dim if l == 0 else width            # layer 0 reads word vectors, later layers read the layer below
            for j, d in enumerate(self.directions):            # each direction has its OWN LSTM
                W, b = init_lstm(in_dim, hidden_dim, seed + 1000 * (2 * l + j), forget_bias)  # layer 0 forward = Day 67's seed
                setattr(self, f"W{l}{d}", W); setattr(self, f"b{l}{d}", b)  # stored as W0f, b0f, W0b, ...
                self.params += [f"W{l}{d}", f"b{l}{d}"]
        self.Why = rng.randn(width, n_tags) * np.sqrt(1.0 / width)  # output weights: top layer -> tags
        self.by = np.zeros(n_tags)                           # output bias
        self.params += ["Why", "by"]

    def n_weights(self):                                   # how many numbers this network learns
        return {name: getattr(self, name).size for name in self.params}  # per parameter

    def forward(self, ids):                                # ids: (m, T) -> probabilities, one row per word
        m, T = ids.shape                                     # batch size, padded length
        self.ids, self.lens = ids, (ids != PAD).sum(axis=1)  # real length of every sentence (only padding has id 0)
        h0 = np.zeros((m, self.hidden_dim))                  # every LSTM starts from zero states
        inp = self.E[ids]                                    # layer 0's input: word vectors, (m, T, D)
        self.caches = []                                     # one entry per layer: the caches of its LSTMs
        for l in range(self.layers):                         # bottom layer first
            Hf, _, cf = lstm_forward(inp, h0, h0, getattr(self, f"W{l}f"), getattr(self, f"b{l}f"))  # left to right (Day 64)
            caches = {"f": cf}                                 # keep for backward
            if self.bi:                                        # NEW: the same input, read right to left by a second LSTM
                Xr = reverse_padded(inp, self.lens)              # every sentence backwards
                Hr, _, cb = lstm_forward(Xr, h0, h0, getattr(self, f"W{l}b"), getattr(self, f"b{l}b"))
                caches["b"] = cb                                 # keep for backward
                Hf = np.concatenate([Hf, reverse_padded(Hr, self.lens)], axis=2)  # flip its states back; join at each word
            self.caches.append(caches)                         # this layer's caches
            inp = Hf                                           # NEW (stacking): this layer's states are the next layer's input
        self.top = inp.reshape(m * T, -1)                    # the top layer's state at every word, one row each
        self.probs = softmax(self.top @ self.Why + self.by)  # a tag distribution for every word
        return self.probs                                    # hand back

    def loss(self, tags, mask):                            # mean cross-entropy over the REAL words only -- Day 62
        p_true = self.probs[np.arange(tags.size), tags.ravel()]  # probability given to each word's true tag
        return -np.sum(np.log(p_true + 1e-9) * mask.ravel()) / mask.sum()  # padding counts 0

    def gradients(self, tags, mask):                       # gradients of that loss for every parameter
        m, T = self.ids.shape                                # batch size, padded length
        H = self.hidden_dim                                  # hidden size of one direction
        dlogits = (self.probs - one_hot(tags.ravel(), self.n_tags)) * mask.ravel()[:, None] / mask.sum()  # zero on padding
        grads = dict(Why=self.top.T @ dlogits, by=dlogits.sum(axis=0))  # output layer
        d_out = (dlogits @ self.Why.T).reshape(m, T, -1)     # gradient at the top layer's states
        zeros = np.zeros((m, H))                             # nothing extra arrives after the last step
        for l in reversed(range(self.layers)):               # top layer first
            caches = self.caches[l]                            # this layer's LSTM caches
            grads[f"W{l}f"], grads[f"b{l}f"], d_in, _, _ = lstm_backward(d_out[:, :, :H], zeros, zeros, caches["f"])  # Day 65
            if self.bi:                                        # NEW: the right-to-left half of the gradient
                dHr = reverse_padded(d_out[:, :, H:], self.lens)  # into the reversed LSTM's own step order
                grads[f"W{l}b"], grads[f"b{l}b"], dXr, _, _ = lstm_backward(dHr, zeros, zeros, caches["b"])
                d_in = d_in + reverse_padded(dXr, self.lens)     # the input fed BOTH directions: add both gradients
            d_out = d_in                                       # NEW (stacking): the gradient for the layer below
        grads["E"] = np.zeros_like(self.E)                   # one gradient row per vocabulary word
        np.add.at(grads["E"], self.ids, d_out)               # add each position's gradient into its word's row (Day 61)
        return grads                                          # dict: parameter name -> gradient

    def train_step(self, ids, tags, mask):                 # one SGD step with Day 60's clipping -- Day 67
        self.forward(ids)                                    # forward pass
        loss = self.loss(tags, mask)                         # batch loss, before the update
        grads, _, _ = clip_by_global_norm(self.gradients(tags, mask), self.clip)  # gradients, clipped
        for name, g in grads.items():                        # plain SGD
            getattr(self, name)[...] -= self.lr * g            # in place
        return loss                                          # for the record

    def drop_caches(self):                                 # forget everything a forward pass stored -- Day 67
        for name in ["ids", "lens", "caches", "top", "probs"]:  # the big per-batch arrays
            self.__dict__.pop(name, None)                      # remove it if present

    def predict(self, ids):                                # predicted tag id for every position, (m, T) -- Day 67
        pred = self.forward(ids).argmax(axis=1).reshape(ids.shape)  # most probable tag per row
        self.drop_caches()                                   # free the memory
        return pred                                          # hand back

    def snapshot(self):                                    # a copy holding only the parameters and settings -- Day 67
        self.drop_caches()                                   # nothing per-batch is worth copying
        return copy.deepcopy(self)                           # an independent copy


def fit(data, vocab_size, n_tags, seed, epochs=EPOCHS, batch_size=32, **net_kwargs):  # Day 67's fit(), with DeepTagger
    (ids_tr, tags_tr, mask_tr), (ids_va, tags_va, mask_va), (ids_te, tags_te, mask_te) = data  # the three splits
    net = DeepTagger(vocab_size, n_tags, seed=seed, **net_kwargs)  # fresh network
    rng = np.random.RandomState(seed)                       # shuffles the order of sentences each epoch
    lens_tr = mask_tr.sum(axis=1)                           # real length of every training sentence
    hist = dict(loss=[], train=[], val=[], test=[], seconds=[])  # one entry per epoch
    best_val, best_net, best_epoch = -1.0, None, -1          # the snapshot with the best validation accuracy
    for epoch in range(epochs):                              # full training budget
        start_time = time.perf_counter()                       # time the training part of the epoch
        perm = rng.permutation(len(ids_tr))                    # fresh sentence order
        losses = []                                            # this epoch's per-step losses
        for start in range(0, len(perm), batch_size):          # mini-batches
            idx = perm[start:start + batch_size]                 # this batch's sentences
            T_batch = lens_tr[idx].max()                         # pad only to the longest sentence IN THIS BATCH
            losses.append(net.train_step(ids_tr[idx, :T_batch], tags_tr[idx, :T_batch], mask_tr[idx, :T_batch]))  # one step
        hist["seconds"].append(time.perf_counter() - start_time)  # seconds spent training this epoch
        hist["loss"].append(np.mean(losses))                   # mean training loss
        hist["train"].append(token_accuracy(net.predict(ids_tr), tags_tr, mask_tr))  # training accuracy NOW
        hist["val"].append(token_accuracy(net.predict(ids_va), tags_va, mask_va))    # validation accuracy NOW
        hist["test"].append(token_accuracy(net.predict(ids_te), tags_te, mask_te))   # test accuracy NOW (never used to choose)
        if hist["val"][-1] > best_val:                           # a new best on VALIDATION
            best_val, best_net, best_epoch = hist["val"][-1], net.snapshot(), epoch  # snapshot it
    return best_net, best_epoch, {k: np.array(v) for k, v in hist.items()}  # best snapshot, when, and the full record


RUNS = {}                                               # cache: identical runs requested by two sections train only once


def fit_cached(data, vocab_size, n_tags, seed, **net_kwargs):  # fit(), remembered by its settings -- Day 67
    key = (seed, tuple(sorted(net_kwargs.items())))          # everything that defines the run
    if key not in RUNS:                                      # first request
        RUNS[key] = fit(data, vocab_size, n_tags, seed, **net_kwargs)  # train it
    return RUNS[key]                                         # later requests reuse the result


CONFIGS = [("1 layer, left to right", dict(layers=1, bidirectional=False)),   # = Day 67's LSTM tagger
           ("2 layers, left to right", dict(layers=2, bidirectional=False)),  # stacking only
           ("1 layer, bidirectional", dict(layers=1, bidirectional=True)),    # two directions only
           ("2 layers, bidirectional", dict(layers=2, bidirectional=True))]   # both


# =============================================================================
# DEMOS
# =============================================================================
def demo_shapes(itos, tag_names):
    print("=" * 70)
    print("0. FOUR TAGGERS -- one layer or two, one direction or two")
    print("=" * 70)
    print(f"Day 67's LSTM tagger (embedding 32, hidden 32 per direction, lr {LR}, forget bias "
          f"{FORGET_BIAS}: Day 67's validation choice for the LSTM). A bidirectional layer runs a "
          "second LSTM over each sentence backwards and joins the two states at every word, so "
          "it hands 64 numbers per word upward. A second layer reads the first layer's states "
          "instead of word vectors.")
    V, K = len(itos), len(tag_names)                           # vocabulary size, number of tags
    print(f"\n{'tagger':<26}{'LSTMs':>6}{'LSTM weights':>14}{'output layer':>14}{'total':>9}")
    for name, cfg in CONFIGS:                                  # each configuration
        net = DeepTagger(V, K, **cfg)                            # build it (untrained)
        n = net.n_weights()                                      # weight counts
        lstm_w = sum(v for k, v in n.items() if k[1:2].isdigit())  # every LSTM's weights and biases (named W0f, b0f, ...)
        print(f"{name:<26}{len(net.directions) * net.layers:>6}{lstm_w:>14}{n['Why'] + n['by']:>14}{sum(n.values()):>9}")
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


def demo_checks():
    print("\n" + "=" * 70)
    print("1. CHECKS -- reversing with padding, and gradients through two layers and two directions")
    print("=" * 70)
    print("(a) reverse_padded must flip only the real words of each sentence and leave padding in "
          "place; done twice it must give back the original. (b) Day 62's gradient check (12 words, "
          "3 tags, embedding 4, hidden 5, four sentences of lengths 5, 2, 4, 1) on the 2-layer "
          "bidirectional tagger, every entry of all 11 parameter arrays. Pass rule: |numeric - "
          "analytic| <= 1e-10 + 1e-6 * |numeric|.")
    ids = np.array([[5, 6, 7, 0, 0], [8, 9, 0, 0, 0]])         # two padded sentences
    lens = (ids != PAD).sum(axis=1)                            # their real lengths
    print(f"\n(a) sentences {ids.tolist()} -> reversed {reverse_padded(ids, lens).tolist()}; "
          f"reversed twice equals the original: {np.array_equal(reverse_padded(reverse_padded(ids, lens), lens), ids)}")
    rng = np.random.RandomState(3)                             # private rng for the check (Day 62's)
    lens = np.array([5, 2, 4, 1])                              # four different lengths
    mask = np.arange(5)[None, :] < lens[:, None]               # real-word mask
    ids = rng.randint(2, 12, size=(4, 5))                      # random real word ids
    ids[0, 3] = ids[0, 1]                                      # the SAME word twice in one sentence
    ids[~mask] = PAD                                           # pad past each sentence's length
    tags = rng.randint(0, 3, size=(4, 5))                      # random tags
    net = DeepTagger(12, 3, layers=2, bidirectional=True, embed_dim=4, hidden_dim=5, clip=None, seed=1, forget_bias=0.5)
    net.forward(ids)                                           # forward pass
    analytic = net.gradients(tags, mask)                       # what the backward pass computed
    print(f"\n(b) {'parameter':>10}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>7}")
    ok_all = True                                              # every parameter passes?
    for name in net.params:                                    # all 11 arrays
        numeric = numeric_gradient(net, name, ids, tags, mask)   # measured slopes
        diff = np.abs(numeric - analytic[name])                  # absolute disagreement
        rel = diff / (np.abs(numeric) + np.abs(analytic[name]) + 1e-12)  # relative disagreement
        ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(numeric)))  # the pass rule
        ok_all = ok_all and ok                                   # track the verdict
        print(f"    {name:>10}{numeric.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>7}")
    print(f"every entry passes: {ok_all};  <pad>'s embedding gradient exactly zero: {bool(np.all(analytic['E'][PAD] == 0))}")
    print("Actual output from this lesson's run.")
    changed = ids.copy(); changed[0, 4] = 3                    # change the LAST word of sentence 0 (a real word there)
    p0 = net.forward(ids).reshape(4, 5, 3)[0, 0].copy()        # the FIRST word's tag distribution before
    p1 = net.forward(changed).reshape(4, 5, 3)[0, 0]           # ... and after the last word changed
    one = DeepTagger(12, 3, layers=2, bidirectional=False, embed_dim=4, hidden_dim=5, seed=1, forget_bias=0.5)  # one direction
    q0 = one.forward(ids).reshape(4, 5, 3)[0, 0].copy()        # the same test, left-to-right tagger
    q1 = one.forward(changed).reshape(4, 5, 3)[0, 0]
    print(f"\nchanging the LAST word moves the FIRST word's tag probabilities by: bidirectional "
          f"{np.abs(p1 - p0).max():.2e}, left to right {np.abs(q1 - q0).max():.2e}")
    print("Actual output from this lesson's run.")


def mft_baseline(itos, data):                           # Day 62's most-frequent-tag baseline and its word groups
    V = len(itos)                                             # vocabulary size
    (ids_tr, tags_tr, mask_tr), _, (ids_te, tags_te, mask_te) = data  # training and test splits
    seen = defaultdict(Counter)                                # word id -> how often each tag was seen with it
    for w, t in zip(ids_tr[mask_tr], tags_tr[mask_tr]):        # every real training word
        seen[w][t] += 1                                          # count the pair
    mft = np.full(V, seen[UNK].most_common(1)[0][0])           # default: the tag most common among <unk> training words
    for w, c in seen.items():                                  # every word seen in training
        mft[w] = c.most_common(1)[0][0]                          # its single most frequent tag
    ambiguous = np.array([len(seen[w]) > 1 for w in range(V)])  # words with MORE than one tag in training
    ambiguous[UNK] = False                                     # <unk> is its own group
    groups = {"all words": mask_te,                             # Day 62's three word groups
              "unambiguous": mask_te & (ids_te != UNK) & ~ambiguous[ids_te],
              "ambiguous": mask_te & ambiguous[ids_te],
              "unknown": mask_te & (ids_te == UNK)}
    return mft, groups                                         # baseline table and groups


def demo_train(itos, tag_names, data, mft, groups):
    print("\n" + "=" * 70)
    print("2. TRAINING THE FOUR TAGGERS -- three seeds each")
    print("=" * 70)
    print(f"Day 67's setup unchanged: 25 epochs, batch 32, clip 1.0, lr {LR}, forget bias "
          f"{FORGET_BIAS}; test accuracy at each run's best validation epoch. Plus a size control: "
          "a 1-layer left-to-right tagger with hidden 50, which has about as many LSTM weights as "
          "the 1-layer bidirectional one.")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    print(f"\n{'tagger':<30}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}{'best epochs':>14}{'train at end':>14}")
    results = {}                                               # name -> (test accuracies, predictions)
    rows = CONFIGS[:1] + [("1 layer, left to right, h 50", dict(layers=1, bidirectional=False, hidden_dim=50))] + CONFIGS[1:]
    for name, cfg in rows:                                     # every configuration, plus the size control
        accs, preds, epochs, trains = [], [], [], []             # per seed
        for s in SEEDS:                                          # three seeds
            net, be, h = fit_cached(data, V, K, seed=s, **cfg)     # train
            accs.append(h["test"][be]); epochs.append(be); trains.append(h["train"][-1])  # the honest numbers
            preds.append(net.predict(ids_te))                      # test predictions, for Section 3
        results[name] = (accs, preds)                            # keep
        print(f"{name:<30}" + "".join(f"{a:>9.4f}" for a in accs) + f"{np.mean(accs):>9.4f}"
              f"{'/'.join(str(e) for e in epochs):>14}{np.mean(trains):>14.4f}")
    print(f"most-frequent-tag baseline on test: {token_accuracy(mft[ids_te], tags_te, mask_te):.4f}; "
          f"one standard error about {np.sqrt(0.85 * 0.15 / mask_te.sum()):.4f}")
    print("Actual output from this lesson's run.")
    return results                                             # later sections reuse the predictions


def demo_lr_check(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("2b. IS DAY 67'S LEARNING RATE STILL RIGHT FOR THE BIGGER TAGGERS?")
    print("=" * 70)
    print(f"lr {LR} was chosen on Day 67 for a 1-layer left-to-right tagger. Each bigger tagger is "
          f"also trained at half that, seed 0, and the two are compared on VALIDATION accuracy only.")
    V, K = len(itos), len(tag_names)                           # sizes
    print(f"\n{'tagger':<26}{'val at lr ' + str(LR):>15}{'val at lr ' + str(LR / 2):>15}")
    for name, cfg in CONFIGS[1:]:                              # the three bigger taggers
        _, be, h = fit_cached(data, V, K, seed=0, **cfg)         # Day 67's learning rate (trained in Section 2)
        _, be2, h2 = fit_cached(data, V, K, seed=0, lr=LR / 2, **cfg)  # half of it
        print(f"{name:<26}{h['val'][be]:>15.4f}{h2['val'][be2]:>15.4f}")
    print("Actual output from this lesson's run.")


def demo_where(itos, tag_names, data, mft, groups, results):
    print("\n" + "=" * 70)
    print("3. WHERE READING BOTH WAYS HELPS -- word groups, and the tags that gain most")
    print("=" * 70)
    print("Day 62's word groups, mean of three seeds. Then the tags whose accuracy changes most "
          "from the 1-layer left-to-right tagger to the 1-layer bidirectional one (seed 0 each).")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    cols = ["all words", "unambiguous", "ambiguous", "unknown"]  # the groups
    print(f"\n{'':<26}" + "".join(f"{g:>13}" for g in cols))
    print(f"{'share of test words':<26}" + "".join(f"{groups[g].sum() / mask_te.sum():>13.1%}" for g in cols))
    table = {"most-frequent-tag": {g: (mft[ids_te] == tags_te)[groups[g]].mean() for g in cols}}  # baseline row
    print(f"{'most-frequent-tag':<26}" + "".join(f"{table['most-frequent-tag'][g]:>13.4f}" for g in cols))
    for name, _ in CONFIGS:                                    # the four taggers
        table[name] = {g: np.mean([(p == tags_te)[groups[g]].mean() for p in results[name][1]]) for g in cols}
        print(f"{name:<26}" + "".join(f"{table[name][g]:>13.4f}" for g in cols))
    print("Actual output from this lesson's run.")
    p_uni, p_bi = results["1 layer, left to right"][1][0], results["1 layer, bidirectional"][1][0]  # seed-0 predictions
    print(f"\n{'true tag':>9}{'test words':>12}{'left to right':>15}{'bidirectional':>15}{'change':>9}")
    stats = []                                                 # (change, tag, count, acc uni, acc bi)
    for k in range(K):                                         # every tag
        sel = mask_te & (tags_te == k)                           # test words with this true tag
        a_u, a_b = (p_uni == tags_te)[sel].mean(), (p_bi == tags_te)[sel].mean()  # accuracy of each tagger on them
        stats.append((a_b - a_u, k, int(sel.sum()), a_u, a_b))   # keep
    for change, k, n, a_u, a_b in sorted(stats, reverse=True)[:6]:  # the six biggest gains
        print(f"{tag_names[k]:>9}{n:>12}{a_u:>15.4f}{a_b:>15.4f}{change:>+9.4f}")
    print("Actual output from this lesson's run.")
    return table                                               # the plot reuses it


def probs_at(net, ids, t_idx, chunk=1000):              # tag distribution at position t_idx[j] of sentence j -- Day 67
    out = []                                              # one block of rows per chunk
    for s in range(0, len(ids), chunk):                   # chunks keep the forward pass's memory small
        p = net.forward(ids[s:s + chunk]).reshape(len(ids[s:s + chunk]), ids.shape[1], -1)  # (chunk, T, K)
        out.append(p[np.arange(len(p)), t_idx[s:s + chunk]])  # keep only each sentence's target position
    net.drop_caches()                                     # free the forward pass's arrays
    return np.concatenate(out)                            # (n, K)


def demo_how_far(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("4. WHICH WORDS DOES EACH TAGGER LOOK AT? -- change one word before OR after the target")
    print("=" * 70)
    print("Day 67's probe, now in both directions: the word k positions before (k < 0) or after "
          "(k > 0) a random target word is replaced by a random vocabulary word, and the mean "
          "total-variation change of the target's tag distribution is measured. 5 draws per "
          "sentence, three seeds, the 1-layer taggers and the 2-layer bidirectional one.")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    lens = mask_te.sum(axis=1)                                 # real length of every test sentence
    offsets = [-8, -4, -2, -1, 1, 2, 4, 8]                     # where the changed word is, relative to the target
    models = ["1 layer, left to right", "1 layer, bidirectional", "2 layers, bidirectional"]  # three taggers
    cfg = dict(CONFIGS)                                        # name -> settings
    tv = {name: [] for name in models}                         # name -> mean TV change per offset
    for k in offsets:                                          # each offset
        rng = np.random.RandomState(200 + k + 50)                # the SAME random changes for every tagger
        rows = np.where(lens > abs(k))[0]                        # sentences long enough
        draws = []                                               # (row, target, replacement word)
        for _ in range(5):                                       # 5 draws per sentence
            for r in rows:
                t = rng.randint(max(0, -k), lens[r] - max(0, k))   # a target with a real word at t + k
                draws.append((r, t, rng.randint(2, V)))            # a random real vocabulary word
        draws = np.array(draws)                                  # (n, 3)
        r_idx, t_idx, new_w = draws[:, 0], draws[:, 1], draws[:, 2]  # unpack
        changed = ids_te[r_idx].copy()                           # one modified copy per draw
        changed[np.arange(len(draws)), t_idx + k] = new_w        # replace the word at offset k
        keep = changed[np.arange(len(draws)), t_idx + k] != ids_te[r_idx, t_idx + k]  # the word really changed
        for name in models:                                      # each tagger
            vals = []                                              # per seed
            for s in SEEDS:
                net = fit_cached(data, V, K, seed=s, **cfg[name])[0]  # trained in Section 2
                p0, p1 = probs_at(net, ids_te[r_idx], t_idx), probs_at(net, changed, t_idx)  # before and after
                vals.append(0.5 * np.abs(p0 - p1).sum(axis=1)[keep].mean())  # mean total-variation distance
            tv[name].append(np.mean(vals))                         # mean over seeds
    print(f"\n{'offset k':>9}" + "".join(f"{m:>26}" for m in models))
    for j, k in enumerate(offsets):                            # one row per offset
        print(f"{k:>+9}" + "".join(f"{tv[m][j]:>26.4f}" for m in models))
    print("Actual output from this lesson's run.")
    return offsets, tv                                         # the plot reuses these


def demo_cost(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("5. WHAT EACH COSTS -- seconds per training epoch")
    print("=" * 70)
    print("Median over every epoch of the three seeds; training steps only; matrix library on one "
          "thread. Timings vary by machine.")
    V, K = len(itos), len(tag_names)                           # sizes
    base = None                                                # the 1-layer left-to-right time
    print(f"\n{'tagger':<26}{'seconds / epoch':>17}{'vs 1 layer, left to right':>27}")
    for name, cfg in CONFIGS:                                  # each tagger
        secs = np.median(np.concatenate([fit_cached(data, V, K, seed=s, **cfg)[2]["seconds"] for s in SEEDS]))  # typical epoch
        base = secs if base is None else base                    # the first row is the reference
        print(f"{name:<26}{secs:>17.2f}{secs / base:>27.2f}")
    print("Actual output from this lesson's run.")


def plot_results(itos, tag_names, data, table, offsets, tv):  # one figure: word groups | which words matter
    V, K = len(itos), len(tag_names)                           # sizes
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))       # two panels
    cols = ["all words", "unambiguous", "ambiguous", "unknown"]  # panel 1 clusters
    models = ["most-frequent-tag"] + [n for n, _ in CONFIGS]    # five bars per cluster
    colors = ["#d9d9d9", "#9ecae1", "#3182bd", "#fdae6b", "#e6550d"]  # baseline grey, blues = left to right, oranges = bidirectional
    x = np.arange(len(cols))                                    # cluster positions
    for j, (name, color) in enumerate(zip(models, colors)):     # each model
        a1.bar(x + (j - 2) * 0.16, [table[name][g] for g in cols], width=0.16, color=color,
               edgecolor="#555555", linewidth=0.5, label=name)
    a1.set_xticks(x); a1.set_xticklabels(cols); a1.set_ylim(0, 1.25); a1.set_ylabel("test accuracy (mean of 3 seeds)")
    a1.set_title("Where reading both ways helps", fontsize=10); a1.legend(fontsize=7.5, loc="upper right", ncol=2)
    for name, color in zip(["1 layer, left to right", "1 layer, bidirectional", "2 layers, bidirectional"],
                           ["#3182bd", "#fdae6b", "#e6550d"]):  # panel 2: one line per tagger
        ks = [(k, v) for k, v in zip(offsets, tv[name]) if v > 0]  # a log scale cannot show 0: plot only the non-zero points
        a2.plot([k for k, _ in ks], [v for _, v in ks], color=color, marker="o", markersize=4, label=name)
    a2.text(4.5, 0.03, "left to right:\nexactly 0 for every\nword AFTER the target", ha="center", fontsize=8, color="#3182bd")
    a2.axvline(0, color="#999999", linewidth=1); a2.set_yscale("log")
    a2.set_xlabel("offset k of the changed word (negative = before the target, positive = after)")
    a2.set_ylabel("mean change in the target's tag distribution (TV)")
    a2.set_title("Which words each tagger looks at", fontsize=10); a2.legend(fontsize=8, loc="lower center")
    a2.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("deep_bidirectional_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved deep_bidirectional_results.png")


def draw_diagram():                                     # a 2-layer bidirectional tagger over a 5-word sentence
    fig, ax = plt.subplots(figsize=(13, 7.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.4); ax.axis("off")      # plain coordinate grid, no axes

    def box(x, y, w, h, text, color, fs=8.5):                    # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, color="#333333"):                  # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    ax.text(6.5, 7.15, "Day 68: two layers, each reading the sentence in both directions", ha="center",
            fontsize=12, fontweight="bold")
    words = ["the", "old", "man", "the", "boats"]               # "man" is a VERB here: only the words after it say so
    xs = [2.2, 4.4, 6.6, 8.8, 11.0]                             # one column per word
    rows = {"word": 0.5, "f1": 1.45, "b1": 2.25, "j1": 3.1, "f2": 3.95, "b2": 4.75, "j2": 5.6, "tag": 6.45}  # heights
    for name, label in [("word", "word vectors"), ("f1", "layer 1"), ("j1", "joined"), ("f2", "layer 2"),
                        ("j2", "joined"), ("tag", "tag per word")]:  # row labels on the left
        ax.text(0.1, rows[name] + (0.4 if name in ("f1", "f2") else 0), label, fontsize=8.5,
                fontweight="bold" if name in ("f1", "f2") else "normal", color="#333333" if name in ("f1", "f2") else "#666666")
    for j, (x, w) in enumerate(zip(xs, words)):                  # one column per word
        box(x, rows["word"], 1.3, 0.42, w, "#F2F2F2", fs=9)      # the word
        for layer, below in [(1, "word"), (2, "j1")]:            # two layers
            yf, yb, yj = rows[f"f{layer}"], rows[f"b{layer}"], rows[f"j{layer}"]  # this layer's rows
            box(x - 0.5, yf, 0.85, 0.42, "fwd", "#DCEBF7")       # left-to-right LSTM step
            box(x + 0.5, yb, 0.85, 0.42, "bwd", "#FDE7D3")       # right-to-left LSTM step
            box(x, yj, 1.5, 0.42, "[ fwd ; bwd ]", "#F4EEFC", fs=8)  # the two states joined at this word
            arrow(x - 0.25, rows[below] + 0.22, x - 0.5, yf - 0.22)  # input -> forward step
            arrow(x + 0.25, rows[below] + 0.22, x + 0.5, yb - 0.22)  # input -> backward step
            arrow(x - 0.5, yf + 0.22, x - 0.3, yj - 0.22, "#555555")  # forward state -> join
            arrow(x + 0.5, yb + 0.22, x + 0.3, yj - 0.22, "#555555")  # backward state -> join
            if j < len(xs) - 1:
                arrow(x - 0.06, yf, xs[j + 1] - 0.94, yf, "#1F6F99")  # forward state passes right
                arrow(xs[j + 1] + 0.06, yb, x + 0.94, yb, "#C0611B")  # backward state passes left
        box(x, rows["tag"], 1.3, 0.42, "tag", "#E5F2E1", fs=9)   # the output
        arrow(x, rows["j2"] + 0.22, x, rows["tag"] - 0.22)       # top joined state -> tag
    ax.text(6.5, 0.02, "each tag sees the whole sentence: the forward LSTMs bring the words before it, the backward LSTMs "
            "the words after it", ha="center", fontsize=8.5, style="italic")
    plt.savefig("deep_bidirectional_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved deep_bidirectional_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT COMES NEXT")
    print("=" * 70)
    print("A bidirectional layer can only be used when the whole input is available before any "
          "output is needed -- true for tagging a finished sentence, not for writing a sentence "
          "word by word. Day 69, the week's build, uses both kinds: an ENCODER that reads a whole "
          "input sequence (and may read it both ways), and a DECODER that writes an output "
          "sequence one token at a time, left to right.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread: same results, much faster on this machine (Day 64)
        splits, counts, itos, stoi, tag_names, ttoi, data = prepare_data()  # load and encode once (Day 62)
        demo_shapes(itos, tag_names)                         # 0. four taggers
        demo_checks()                                        # 1. reversing, gradients, dependence on later words
        mft, groups = mft_baseline(itos, data)               # Day 62's baseline and word groups
        results = demo_train(itos, tag_names, data, mft, groups)  # 2. train all four (+ size control)
        demo_lr_check(itos, tag_names, data)                 # 2b. is Day 67's learning rate still right?
        table = demo_where(itos, tag_names, data, mft, groups, results)  # 3. where reading both ways helps
        offsets, tv = demo_how_far(itos, tag_names, data)    # 4. which words each tagger looks at
        demo_cost(itos, tag_names, data)                     # 5. time per epoch
        plot_results(itos, tag_names, data, table, offsets, tv)  # the figure
        draw_diagram()                                       # the diagram used in the guides
        note_on_what_comes_next()                            # 6. Day 69


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
