"""
Day 74: Self-Attention, From Scratch -- Attention Within One Sequence

Days 71-73 let a decoder attend to an ENCODER's states. Self-attention
lets every position of ONE sequence attend to every position of the same
sequence, itself included:

    Q = X Wq,  K = X Wk,  V = X Wv          three projections of the same inputs
    A = softmax(Q K^T)   row by row         how much position i reads position j
    Z = A V                                 each position's output: a weighted average

Today the scores are plain dot products (Day 73); Day 75 adds the scaling.

What is built and measured, reported as it came out:

  1. One self-attention layer by hand.
  2. Self-attention ignores order: shuffle the positions and the outputs
     shuffle with them, exactly.
  3. A gradient check of a self-attention tagger, with padding masked.
  4. That tagger on Week 9's real part-of-speech data (Day 62's), against
     the same tagger with the attention removed.
  5. The shuffle test from Day 62.
  6. What each word attends to.
  7. Where the context helps.
  8. What it costs.

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
SEEDS = [0, 1, 2]                                       # Day 62's three seeds
EPOCHS = 25                                             # Day 62's training budget
LRS = [0.5, 1.0, 2.0, 4.0]                              # learning rates tried on validation (Day 67's grid)

# =============================================================================
# 0. REUSED -- verbatim from Days 56-67
# =============================================================================
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


def masked_softmax(scores, mask):                      # day70_exercise.py, verbatim: softmax over the real slots only
    s = np.where(mask, scores, -np.inf)                   # a padded slot's score becomes minus infinity
    s = s - s.max(axis=-1, keepdims=True)                 # subtract the largest real score, for stability
    e = np.exp(s)                                         # exp(-inf) = 0, so padding contributes nothing
    return e / e.sum(axis=-1, keepdims=True)              # weights over the real slots, summing to 1


# =============================================================================
# 1. SELF-ATTENTION -- NEW
# =============================================================================
def self_attention(X, Wq, Wk, Wv, mask):                # NEW: every position attends to every real position of the SAME sequence
    Q, K, Vv = X @ Wq, X @ Wk, X @ Wv                     # queries, keys and values: three projections of the same inputs
    S = np.einsum("mid,mjd->mij", Q, K)                   # scores: S[m, i, j] = q_i . k_j (unscaled; Day 75 scales them)
    A = masked_softmax(S, mask[:, None, :])               # weights over the KEYS j; padded keys get weight 0
    Z = np.einsum("mij,mjd->mid", A, Vv)                  # each position's output: a weighted average of all the values
    return Z, dict(X=X, Q=Q, K=K, V=Vv, A=A)              # outputs, and what the backward pass needs


def self_attention_backward(dZ, cache, Wq, Wk, Wv):     # NEW: gradients for X, Wq, Wk, Wv
    X, Q, K, Vv, A = (cache[k] for k in ["X", "Q", "K", "V", "A"])  # what the forward pass computed
    dA = np.einsum("mid,mjd->mij", dZ, Vv)                # Z = A V  ->  dA = dZ V^T
    dV = np.einsum("mij,mid->mjd", A, dZ)                 # ...  and dV = A^T dZ
    dS = A * (dA - (A * dA).sum(axis=2, keepdims=True))   # back through each row's softmax (zero where A is 0)
    dQ = np.einsum("mij,mjd->mid", dS, K)                 # S = Q K^T  ->  dQ = dS K
    dK = np.einsum("mij,mid->mjd", dS, Q)                 # ...  and dK = dS^T Q
    dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T                # X made the queries, the keys AND the values
    flat = lambda T: T.reshape(-1, T.shape[-1])           # (m, n, d) -> (m*n, d), to sum over every position
    return dX, flat(X).T @ flat(dQ), flat(X).T @ flat(dK), flat(X).T @ flat(dV)  # dX, dWq, dWk, dWv


class SelfAttnTagger:                                   # NEW: words -> embedding -> self-attention -> a tag for every word
    def __init__(self, vocab_size, n_tags, embed_dim=32, attn_dim=32, hidden_dim=32, lr=1.0, clip=1.0, seed=0, attend=True):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        D, d, H = embed_dim, attn_dim, hidden_dim          # sizes
        self.n_tags, self.lr, self.clip, self.attend = n_tags, lr, clip, attend  # settings; attend=False is the no-context control
        self.E = rng.randn(vocab_size, D) * 0.1            # embedding table, drawn first (as on Days 62-68)
        self.Wq = rng.randn(D, d) / np.sqrt(D)             # query projection
        self.Wk = rng.randn(D, d) / np.sqrt(D)             # key projection
        self.Wv = rng.randn(D, d) / np.sqrt(D)             # value projection
        self.W1 = rng.randn(D + d, H) / np.sqrt(D + d)     # hidden layer on [word vector ; attention output]
        self.b1 = np.zeros(H)                             # hidden bias
        self.Why = rng.randn(H, n_tags) / np.sqrt(H)       # output layer
        self.by = np.zeros(n_tags)                        # output bias
        self.params = ["E", "Wq", "Wk", "Wv", "W1", "b1", "Why", "by"]  # every trainable parameter, by name

    def forward(self, ids):                                # ids (m, T) -> tag probabilities, one row per position
        m, T = ids.shape                                     # batch size, padded length
        self.ids, self.mask = ids, ids != PAD                # only padding has id 0 (Day 68)
        self.X = self.E[ids]                                 # word vectors, (m, T, D)
        if self.attend:                                   # with attention ...
            self.Z, self.cache = self_attention(self.X, self.Wq, self.Wk, self.Wv, self.mask)  # context from the whole sentence
        else:
            self.Z = np.zeros((m, T, self.Wv.shape[1]))        # control: no context at all
        self.XZ = np.concatenate([self.X, self.Z], axis=2).reshape(m * T, -1)  # each word's own vector and its context
        self.Hh = np.tanh(self.XZ @ self.W1 + self.b1)       # hidden layer
        self.probs = softmax(self.Hh @ self.Why + self.by)   # a tag distribution for every position
        return self.probs                                 # hand back

    def loss(self, tags, mask):                            # mean cross-entropy over the real words -- Day 62
        p_true = self.probs[np.arange(tags.size), tags.ravel()]  # probability of each word's true tag
        return -np.sum(np.log(p_true + 1e-9) * mask.ravel()) / mask.sum()  # padding counts 0

    def gradients(self, tags, mask):                       # gradients of that loss for every parameter
        m, T = self.ids.shape                                # batch size, padded length
        D = self.E.shape[1]                                  # embedding size
        dlog = (self.probs - one_hot(tags.ravel(), self.n_tags)) * mask.ravel()[:, None] / mask.sum()  # zero on padding
        g = dict(Why=self.Hh.T @ dlog, by=dlog.sum(axis=0))  # output layer
        dpre = (dlog @ self.Why.T) * (1 - self.Hh ** 2)      # back through tanh
        g["W1"], g["b1"] = self.XZ.T @ dpre, dpre.sum(axis=0)  # hidden layer
        dXZ = (dpre @ self.W1.T).reshape(m, T, -1)           # gradient for [word vector ; context]
        dX = dXZ[:, :, :D].copy()                            # the word vector's own part
        if self.attend:                                      # NEW: back through self-attention
            dXa, g["Wq"], g["Wk"], g["Wv"] = self_attention_backward(dXZ[:, :, D:], self.cache, self.Wq, self.Wk, self.Wv)
            dX += dXa                                          # the word vectors also made the queries, keys and values
        else:
            g["Wq"], g["Wk"], g["Wv"] = (np.zeros_like(self.Wq), np.zeros_like(self.Wk), np.zeros_like(self.Wv))
        g["E"] = np.zeros_like(self.E)                       # one gradient row per vocabulary word
        np.add.at(g["E"], self.ids, dX)                      # Day 61
        return g                                          # dict: parameter name -> gradient

    def train_step(self, ids, tags, mask):                 # one SGD step with Day 60's clipping -- Day 67
        self.forward(ids)                                    # forward pass
        loss = self.loss(tags, mask)                         # batch loss
        grads, _, _ = clip_by_global_norm(self.gradients(tags, mask), self.clip)  # gradients, clipped
        for name, gr in grads.items():                       # plain SGD
            getattr(self, name)[...] -= self.lr * gr           # in place
        return loss                                       # for the record

    def drop_caches(self):                                 # free the per-batch arrays (Day 67)
        for name in ["ids", "mask", "X", "Z", "cache", "XZ", "Hh", "probs"]:  # the big per-batch arrays
            self.__dict__.pop(name, None)                 # remove it if present

    def predict(self, ids):                                # predicted tag for every position (Day 67)
        pred = self.forward(ids).argmax(axis=1).reshape(ids.shape)  # most probable tag, back in sentence shape
        self.drop_caches()                                # a prediction needs no backward pass
        return pred                                       # hand back

    def snapshot(self):                                    # a copy holding only the parameters (Day 67)
        self.drop_caches()                                # a prediction needs no backward pass
        return copy.deepcopy(self)                        # an independent copy


def fit(data, vocab_size, n_tags, seed, epochs=EPOCHS, batch_size=32, **net_kwargs):  # Day 67's fit(), with this tagger
    (ids_tr, tags_tr, mask_tr), (ids_va, tags_va, mask_va), (ids_te, tags_te, mask_te) = data  # the three splits
    net = SelfAttnTagger(vocab_size, n_tags, seed=seed, **net_kwargs)  # fresh network
    rng = np.random.RandomState(seed)                       # shuffles the sentences each epoch (Day 62)
    lens_tr = mask_tr.sum(axis=1)                           # real length of every training sentence
    hist = dict(train=[], val=[], test=[], seconds=[])      # one entry per epoch
    best_val, best_net, best_epoch = -1.0, None, -1          # the best validation snapshot
    for epoch in range(epochs):                           # full training budget
        start_time = time.perf_counter()                       # time the training part of the epoch
        perm = rng.permutation(len(ids_tr))                    # fresh sentence order
        for start in range(0, len(perm), batch_size):          # mini-batches
            idx = perm[start:start + batch_size]          # this batch's sentences
            T_batch = lens_tr[idx].max()                         # pad only to the longest sentence in this batch
            net.train_step(ids_tr[idx, :T_batch], tags_tr[idx, :T_batch], mask_tr[idx, :T_batch])  # one SGD step
        hist["seconds"].append(time.perf_counter() - start_time)  # seconds spent training this epoch
        hist["train"].append(token_accuracy(net.predict(ids_tr), tags_tr, mask_tr))  # training accuracy now
        hist["val"].append(token_accuracy(net.predict(ids_va), tags_va, mask_va))  # validation accuracy now
        hist["test"].append(token_accuracy(net.predict(ids_te), tags_te, mask_te))  # recorded, never used to choose
        if hist["val"][-1] > best_val:                           # a new best on VALIDATION
            best_val, best_net, best_epoch = hist["val"][-1], net.snapshot(), epoch  # snapshot it
    return best_net, best_epoch, {k: np.array(v) for k, v in hist.items()}  # best snapshot, when, and the record


RUNS = {}                                               # cache: each configuration trains once


def fit_cached(data, vocab_size, n_tags, seed, **net_kwargs):  # fit(), remembered by its settings -- Day 67
    key = (seed, tuple(sorted(net_kwargs.items())))       # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = fit(data, vocab_size, n_tags, seed, **net_kwargs)  # train and remember
    return RUNS[key]                                      # later requests reuse it


# =============================================================================
# DEMOS
# =============================================================================
def demo_by_hand():
    print("=" * 70)
    print("1. ONE SELF-ATTENTION LAYER -- by hand, then checked")
    print("=" * 70)
    print("Three positions with 2-number inputs x = (1, 0), (0, 1), (1, 1); Wq = Wk = identity, Wv = "
          "[[1, 0], [0, 2]]. Each position's query is compared with every key (its own included), the "
          "scores become weights, and the output is the weighted average of the values.")
    X = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])    # one sequence of three positions
    Wq = Wk = np.eye(2); Wv = np.array([[1.0, 0.0], [0.0, 2.0]])  # hand-chosen projections
    Z, cache = self_attention(X, Wq, Wk, Wv, np.ones((1, 3), bool))  # the function
    S = X[0] @ X[0].T                                       # scores by hand: x_i . x_j (Wq = Wk = I)
    A = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)    # softmax of each row, by hand
    print(f"\nscores by hand (row i = position i's query against every key):\n{S}")
    print(f"weights by hand:\n{A.round(4)}")
    print(f"output by hand A (X Wv):\n{(A @ (X[0] @ Wv)).round(4)}")
    print(f"largest difference from self_attention: {np.abs(Z[0] - A @ (X[0] @ Wv)).max():.2e}")
    print("Actual output from this lesson's run.")


def demo_order_blind():
    print("\n" + "=" * 70)
    print("2. SELF-ATTENTION IGNORES ORDER -- shuffle the positions, the outputs shuffle with them")
    print("=" * 70)
    print("Random inputs (4 sequences, 7 positions, 5 numbers) and random projections. The positions of "
          "each sequence are shuffled; if every output simply moves with its input, self-attention "
          "cannot tell one order from another.")
    rng = np.random.RandomState(7)                          # fixed rng
    X = rng.randn(4, 7, 5)                                  # inputs
    Wq, Wk, Wv = (rng.randn(5, 4) for _ in range(3))        # projections
    mask = np.ones((4, 7), bool)                            # no padding
    Z, _ = self_attention(X, Wq, Wk, Wv, mask)              # original order
    perm = rng.permutation(7)                               # a shuffle of the 7 positions
    Zp, _ = self_attention(X[:, perm], Wq, Wk, Wv, mask)    # shuffled inputs
    print(f"\nshuffle {perm.tolist()}: largest |output(shuffled) - shuffled(output)| = {np.abs(Zp - Z[:, perm]).max():.2e}")
    print(f"for comparison, outputs themselves are of size {np.abs(Z).mean():.3f} on average")
    print("Actual output from this lesson's run.")


def numeric_gradient(net, name, ids, tags, mask, eps=1e-5):  # Day 62's central differences, verbatim
    P = getattr(net, name)                                  # the parameter array itself
    numeric = np.zeros_like(P)                              # same shape
    for pos in np.ndindex(P.shape):                         # every entry
        old = P[pos]                                          # remember it
        P[pos] = old + eps; net.forward(ids); loss_up = net.loss(tags, mask)    # nudge up
        P[pos] = old - eps; net.forward(ids); loss_down = net.loss(tags, mask)  # nudge down
        P[pos] = old                                          # restore
        numeric[pos] = (loss_up - loss_down) / (2 * eps)      # the measured slope
    return numeric                                        # one slope per entry


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("3. GRADIENT CHECK -- the self-attention tagger, with padding")
    print("=" * 70)
    print("Day 62's check: 12 words, 3 tags, embedding 4, attention size 3, hidden 5, four sentences of "
          "lengths 5, 2, 4, 1 (so padded keys must be masked out), one word repeated. Pass rule: "
          "|numeric - analytic| <= 1e-10 + 1e-6 * |numeric| for every entry.")
    rng = np.random.RandomState(3)                             # Day 62's rng and data
    lens = np.array([5, 2, 4, 1])                         # four different lengths
    mask = np.arange(5)[None, :] < lens[:, None]          # real-word mask
    ids = rng.randint(2, 12, size=(4, 5))                 # random real word ids
    ids[0, 3] = ids[0, 1]                                 # the SAME word twice in one sentence
    ids[~mask] = PAD                                      # pad past each sentence's length
    tags = rng.randint(0, 3, size=(4, 5))                 # random tags
    net = SelfAttnTagger(12, 3, embed_dim=4, attn_dim=3, hidden_dim=5, clip=None, seed=1)  # tiny tagger
    net.forward(ids)                                           # forward pass
    analytic = net.gradients(tags, mask)                       # backward pass
    print(f"\n{'parameter':>10}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>7}")
    ok_all = True                                         # every parameter passes?
    for name in net.params:                                    # every parameter
        numeric = numeric_gradient(net, name, ids, tags, mask)  # measured slopes
        diff = np.abs(numeric - analytic[name])           # absolute disagreement
        rel = diff / (np.abs(numeric) + np.abs(analytic[name]) + 1e-12)  # relative disagreement
        ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(numeric)))  # Day 62's pass rule
        ok_all = ok_all and ok                            # track the verdict
        print(f"{name:>10}{numeric.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>7}")
    net.forward(ids)                                           # weights on the padded keys must be exactly 0
    pad_w = net.cache["A"][np.broadcast_to(~mask[:, None, :], net.cache["A"].shape)]  # every weight placed on a padded key
    print(f"every entry passes: {ok_all};  <pad>'s embedding gradient exactly zero: {bool(np.all(analytic['E'][PAD] == 0))};  "
          f"largest weight on a padded key: {pad_w.max():.1e}")
    print("Actual output from this lesson's run.")


def demo_settings(itos, tag_names, data):
    print("\n" + "=" * 70)
    print("4. TRAINING ON WEEK 9'S TAGGING TASK -- learning rate on validation, then three seeds")
    print("=" * 70)
    print("Day 67's setup: Day 62's data, embedding 32, attention size 32, hidden 32, batch 32, clip 1.0, "
          "25 epochs; the learning rate is chosen on VALIDATION (seed 0). Then three seeds at that "
          "rate, with and without the attention output (the control sees each word on its own).")
    V, K = len(itos), len(tag_names)                           # sizes
    print(f"\n{'lr':>5}{'best val':>10}{'at epoch':>10}{'train at end':>14}")
    best = (-1.0, None)                                        # best validation accuracy and its learning rate
    for lr in LRS:                                        # each learning rate
        _, be, h = fit_cached(data, V, K, 0, lr=lr)            # seed 0
        print(f"{lr:>5}{h['val'][be]:>10.4f}{be:>10}{h['train'][-1]:>14.4f}")
        if h["val"][be] > best[0]:                        # a better validation accuracy
            best = (h["val"][be], lr)                     # remember the rate
    lr = best[1]                                               # the chosen learning rate
    print(f"chosen on validation: lr {lr}")
    print(f"\n{'model':<26}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}")
    res = {}                                                   # model -> test accuracies
    for label, kw in [("self-attention", dict(lr=lr)), ("no attention (control)", dict(lr=lr, attend=False))]:
        accs = []                                         # test accuracy per seed
        for s in SEEDS:                                   # three seeds
            _, be, h = fit_cached(data, V, K, s, **kw)    # train (or reuse)
            accs.append(h["test"][be])                           # test at the best validation epoch
        res[label] = accs                                 # keep
        print(f"{label:<26}" + "".join(f"{a:>9.4f}" for a in accs) + f"{np.mean(accs):>9.4f}")
    print("Actual output from this lesson's run.")
    return lr, res                                        # the chosen rate and the results


def demo_shuffle(itos, tag_names, data, lr):
    print("\n" + "=" * 70)
    print("5. THE SHUFFLE TEST -- the same test sentences with their words in random order")
    print("=" * 70)
    print("Day 62's test: the words of every test sentence shuffled (each keeps its own true tag). "
          "Day 62's left-to-right RNN dropped from 0.8327 to 0.7487. Seed 0 of each model here.")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    rng = np.random.RandomState(0)                             # Day 62's shuffle rng
    order = np.tile(np.arange(ids_te.shape[1]), (len(ids_te), 1))  # identity order in every row
    for row, n in enumerate(mask_te.sum(axis=1)):              # shuffle only the real positions
        order[row, :n] = rng.permutation(n)               # shuffle only the real positions
    ids_sh, tags_sh = np.take_along_axis(ids_te, order, axis=1), np.take_along_axis(tags_te, order, axis=1)  # each word keeps its own tag
    print(f"\n{'model':<26}{'normal order':>14}{'shuffled':>10}{'predictions changed':>21}")
    for label, kw in [("self-attention", dict(lr=lr)), ("no attention (control)", dict(lr=lr, attend=False))]:
        net = fit_cached(data, V, K, 0, **kw)[0]                 # seed 0
        p, p_sh = net.predict(ids_te), net.predict(ids_sh)       # both orders
        back = np.empty_like(p_sh)                               # shuffled predictions, put back in the original order
        np.put_along_axis(back, order, p_sh, axis=1)      # undo the shuffle
        changed = (back != p)[mask_te].mean()                    # share of words whose predicted tag changed
        print(f"{label:<26}{token_accuracy(p, tags_te, mask_te):>14.4f}{token_accuracy(p_sh, tags_sh, mask_te):>10.4f}{changed:>21.4f}")
    print("Actual output from this lesson's run.")


def demo_what_it_attends(itos, tag_names, data, lr):
    print("\n" + "=" * 70)
    print("6. WHAT EACH WORD ATTENDS TO -- itself, its neighbours, which words")
    print("=" * 70)
    print("The trained self-attention tagger (seed 0) on all test sentences: the mean weight a word puts "
          "on itself, on words at each distance, and the words that receive the most weight on average "
          "(over every occurrence, among words seen at least 50 times in the test set).")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data               # the test split
    net = fit_cached(data, V, K, 0, lr=lr)[0]                  # seed 0
    net.forward(ids_te); A = net.cache["A"]                    # (m, T, T) weights
    n_real = mask_te.sum(axis=1)                               # real lengths
    i, j = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[2]), indexing="ij")  # query and key positions
    dist = np.abs(i - j)                                       # how far apart
    valid = mask_te[:, :, None] & mask_te[:, None, :]          # real query AND real key
    uniform = (1.0 / n_real)[:, None, None] * np.ones_like(A)  # what an even spread would give
    print(f"\n{'distance':>9}{'mean weight':>13}{'if spread evenly':>18}")
    for d in [0, 1, 2, 3, 5, 10]:                         # a few distances
        sel = valid & (dist == d)                         # real pairs at this distance
        print(f"{d:>9}{A[sel].mean():>13.4f}{uniform[sel].mean():>18.4f}")
    received = (A * mask_te[:, :, None]).sum(axis=1)          # weight each key receives, summed over the real queries (even share = 1)
    total, count = np.zeros(V), np.zeros(V)                    # per word id: summed share, occurrences
    np.add.at(total, ids_te[mask_te], received[mask_te]); np.add.at(count, ids_te[mask_te], 1)  # sum per word id
    common = np.where(count >= 50)[0]                          # words with enough occurrences
    share = total[common] / count[common]                      # mean share received, relative to an even spread
    print(f"\nmost attended words (weight received from the whole sentence; an even spread gives 1.00):")
    for k in np.argsort(-share)[:8]:                           # the eight most attended
        print(f"  {itos[common[k]]!r:<10}{share[k]:>8.2f}  ({int(count[common[k]])} occurrences)")
    print(f"median over these {len(common)} common words: {np.median(share):.2f}")
    net.drop_caches()                                     # free the memory
    print("Actual output from this lesson's run.")


def demo_groups(itos, tag_names, data, lr):
    print("\n" + "=" * 70)
    print("7. WHERE THE CONTEXT HELPS -- Day 62's word groups")
    print("=" * 70)
    print("Unambiguous, ambiguous and unknown test words (Day 62's definitions), mean of three seeds.")
    V, K = len(itos), len(tag_names)                      # sizes
    (ids_tr, tags_tr, mask_tr), _, (ids_te, tags_te, mask_te) = data  # training and test splits
    seen = defaultdict(Counter)                                # word id -> tag counts in training (Day 62)
    for w, t in zip(ids_tr[mask_tr], tags_tr[mask_tr]):   # every real training word
        seen[w][t] += 1                                   # count the pair
    ambiguous = np.array([len(seen[w]) > 1 for w in range(V)]); ambiguous[UNK] = False  # words with several tags in training
    groups = {"all words": mask_te, "unambiguous": mask_te & (ids_te != UNK) & ~ambiguous[ids_te],
              "ambiguous": mask_te & ambiguous[ids_te], "unknown": mask_te & (ids_te == UNK)}
    print(f"\n{'model':<26}" + "".join(f"{g:>13}" for g in groups))
    table = {}                                            # model -> accuracy per group
    for label, kw in [("self-attention", dict(lr=lr)), ("no attention (control)", dict(lr=lr, attend=False))]:
        preds = [fit_cached(data, V, K, s, **kw)[0].predict(ids_te) for s in SEEDS]  # test predictions, three seeds
        table[label] = [np.mean([(p == tags_te)[groups[g]].mean() for p in preds]) for g in groups]  # mean over seeds, per group
        print(f"{label:<26}" + "".join(f"{a:>13.4f}" for a in table[label]))
    print("Actual output from this lesson's run.")
    return list(groups), table                            # the plot reuses them


def demo_cost(itos, tag_names, data, lr):
    print("\n" + "=" * 70)
    print("8. WHAT IT COSTS -- seconds per training epoch, in this run")
    print("=" * 70)
    print("Median over every epoch of the three seeds; training steps only; matrix library on one thread. "
          "Only the ratio within this run is meaningful.")
    V, K = len(itos), len(tag_names)                      # sizes
    med = {}                                              # model -> median seconds per epoch
    for label, kw in [("self-attention", dict(lr=lr)), ("no attention (control)", dict(lr=lr, attend=False))]:
        med[label] = np.median(np.concatenate([fit_cached(data, V, K, s, **kw)[2]["seconds"] for s in SEEDS]))
        print(f"{label:<26}{med[label]:>8.2f} s per epoch")
    print(f"ratio {med['self-attention'] / med['no attention (control)']:.2f}")
    print("Actual output from this lesson's run.")


def plot_results(itos, tag_names, data, lr, groups, table):  # attention weights of one sentence | accuracy by word group
    V, K = len(itos), len(tag_names)                      # sizes
    splits, _, _, stoi, _, _, _ = prepare_data()              # the raw test sentences, for the words
    _, _, (ids_te, tags_te, mask_te) = data               # the test split
    lens = mask_te.sum(axis=1)                            # real length of every test sentence
    plain = [r for r in np.where(lens == 10)[0]                # 10-word test sentences made of ordinary words only
             if all(w.isalpha() for w, _ in splits["test"][r])]
    row = int(plain[0])                                        # the first of them
    net = fit_cached(data, V, K, 0, lr=lr)[0]             # the seed-0 tagger
    net.forward(ids_te[row:row + 1, :10]); A = net.cache["A"][0]; net.drop_caches()  # its weights on that sentence
    words = [w for w, _ in splits["test"][row]]           # the words, for the axis labels
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
    a1.imshow(A, cmap="Blues", vmin=0, vmax=A.max())
    a1.set_xticks(range(10)); a1.set_xticklabels(words, rotation=60, fontsize=8)
    a1.set_yticks(range(10)); a1.set_yticklabels(words, fontsize=8)
    a1.set_xlabel("key (attended word)"); a1.set_ylabel("query (word being tagged)")
    a1.set_title("Self-attention weights, one test sentence (seed 0)", fontsize=10)
    x = np.arange(len(groups))                            # one cluster per word group
    for k, (label, color) in enumerate([("no attention (control)", "#bbbbbb"), ("self-attention", "#1f77b4")]):
        a2.bar(x + (k - 0.5) * 0.36, table[label], width=0.36, color=color, label=label)
    a2.set_xticks(x); a2.set_xticklabels(groups); a2.set_ylim(0, 1.15); a2.set_ylabel("test accuracy (mean of 3 seeds)")
    a2.set_title("Where context from self-attention helps", fontsize=10); a2.legend(fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout(); plt.savefig("self_attention_tagger.png", dpi=110); plt.close()
    print("\nsaved self_attention_tagger.png")


def draw_diagram():                                     # one self-attention layer over four positions
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333", lw=1.1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=lw))

    ax.text(6.0, 5.15, "Day 74: self-attention -- every position reads every position of the same sequence", ha="center",
            fontsize=12, fontweight="bold")
    xs = [1.5, 4.5, 7.5, 10.5]                                  # four positions
    for k, x in enumerate(xs):
        box(x, 0.55, 1.4, 0.45, f"x{k + 1}", "#F2F2F2", fs=9)    # the input
        for dx, name, color in [(-0.75, "q", "#FDE7D3"), (0.0, "k", "#DCEBF7"), (0.75, "v", "#E5F2E1")]:
            box(x + dx, 1.55, 0.6, 0.42, f"{name}{k + 1}", color, fs=8.5)  # its query, key, value
            arrow(x, 0.79, x + dx, 1.33)
        box(x, 4.25, 1.6, 0.5, f"z{k + 1}", "#FAEEDA", fs=9, bold=True)  # its output
    q3x = xs[2] - 0.75                                          # position 3's query, drawn reading every key
    box(6.0, 2.95, 6.2, 0.5, "z3 = sum_j a3j vj,   a3j = softmax_j(q3 . kj)", "#F4EEFC", fs=8.5)  # position 3's read
    for x in xs:
        arrow(x, 1.77, 6.0 + (x - 6.0) * 0.45, 2.69, "#1F6F99", 1.0)  # every key -> the scores
        arrow(x + 0.75, 1.77, 6.0 + (x + 0.75 - 6.0) * 0.45, 2.69, "#2F7D32", 1.0)  # every value -> the weighted sum
    arrow(q3x, 1.77, 6.25, 2.69, "#C0611B", 1.4)                # position 3's query -> its scores
    arrow(6.0, 3.21, xs[2], 3.99, "#C97A16", 1.4)               # the read -> z3
    ax.text(0.2, 3.6, "blue: keys   green: values   orange: position 3's query", fontsize=8, color="#555555")
    ax.text(6.0, 0.05, "drawn for position 3; every position does the same at once. Nothing here depends on the ORDER of the x's.",
            ha="center", fontsize=8.5, style="italic")
    plt.savefig("self_attention_diagram.png", dpi=120, bbox_inches="tight"); plt.close()
    print("saved self_attention_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("9. WHAT COMES NEXT")
    print("=" * 70)
    print("Today's scores were plain dot products q . k. Day 75 divides them by the square root of the "
          "query size -- scaled dot-product attention -- and measures what that changes numerically.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        splits, counts, itos, stoi, tag_names, ttoi, data = prepare_data()  # Day 62's data
        demo_by_hand()                                       # 1. one layer by hand
        demo_order_blind()                                   # 2. order does not matter
        demo_gradient_check()                                # 3. gradients, with padding
        lr, res = demo_settings(itos, tag_names, data)       # 4. train on real data
        demo_shuffle(itos, tag_names, data, lr)              # 5. the shuffle test
        demo_what_it_attends(itos, tag_names, data, lr)      # 6. what it attends to
        groups, table = demo_groups(itos, tag_names, data, lr)  # 7. word groups
        demo_cost(itos, tag_names, data, lr)                 # 8. cost
        plot_results(itos, tag_names, data, lr, groups, table)  # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 9. Day 75


if __name__ == "__main__":
    main()
