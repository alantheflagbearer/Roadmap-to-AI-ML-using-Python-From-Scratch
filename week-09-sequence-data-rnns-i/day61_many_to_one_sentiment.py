"""
Day 61: A Many-to-One RNN Classifier on Real Short-Text Sentiment

Days 56-60 trained on synthetic sequences of numbers. Today the RNN reads
real sentences: 3,000 one-sentence reviews from Amazon, IMDb and Yelp
(Kotzias et al., KDD 2015 -- UCI "Sentiment Labelled Sentences"), each
labelled positive (1) or negative (0). "Many-to-one" means MANY inputs
(one word per timestep) and ONE output (a single label per sentence).

Three new pieces are needed before Day 58's RNN can read text:

  1. A text pipeline: sentence -> tokens -> integer ids, padded to a
     common length so sentences can be stacked into one array.
  2. An embedding layer: a lookup table with one learned vector per word.
     Forward is a row lookup; backward ADDS the gradient into the rows
     that were looked up (a word used twice gets two contributions).
  3. Masking: a padded position must not change the hidden state, so the
     state after the padding equals the state after the last REAL word.

Each is checked: a numeric gradient check through embedding + masked RNN
for both readouts, and a batch-invariance check (a sentence gives the
same prediction alone and inside a padded batch -- only with masking).

Then the honest part. An order-blind bag-of-words baseline is trained on
the same split, and the RNN is tested for whether it uses word order at
all -- the one thing it has that the baseline does not:

  - which state to read: the LAST hidden state vs. the MEAN of all of them
  - shuffle the words of every test sentence: does accuracy change?
  - train on shuffled sentences: does the RNN get worse?
  - hand-written negation probes ("good" vs. "not good")

The results are reported as measured, including where the RNN loses.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import copy                                            # copy: deepcopy, to keep a snapshot of the best network during training
import os                                              # os: build the path to the dataset folder next to this script
import re                                              # re: the regular expression that splits a sentence into tokens
from collections import Counter                        # Counter: counts how often each word appears

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_labelled_sentences")  # dataset folder beside this file
PAD, UNK = 0, 1                                         # reserved ids: 0 = padding, 1 = a word not in the vocabulary
MAX_LEN = 40                                            # sentences longer than this many tokens are cut off

# =============================================================================
# 0. SHARED HELPERS -- reused from Days 57-60
# =============================================================================
def softmax(z):                                         # logits -> probabilities, one row per example
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract the row max for numerical stability
    exp_z = np.exp(z_shifted)                             # exponentiate
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row to sum to 1


def one_hot(y, n_classes):                              # integer labels -> one-hot rows
    out = np.zeros((y.shape[0], n_classes))               # all-zero matrix
    out[np.arange(y.shape[0]), y] = 1                     # one 1 per row at the true class
    return out                                            # the one-hot matrix


def cross_entropy_loss(probs, y_onehot):                # mean cross-entropy over the batch
    return -np.sum(y_onehot * np.log(probs + 1e-9)) / probs.shape[0]  # -log(prob of true class), averaged


def clip_by_global_norm(grads, max_norm):               # Day 60's clipper, verbatim: cap the gradient's length, keep its direction
    norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))  # global norm over every parameter's gradient
    if max_norm is None or norm <= max_norm:               # no threshold, or already short enough
        return grads, norm, 1.0                              # untouched
    scale = max_norm / norm                                 # the factor that makes the norm exactly max_norm
    return {k: g * scale for k, g in grads.items()}, norm, scale  # same factor everywhere -> direction preserved


# =============================================================================
# 1. THE TEXT PIPELINE -- NEW today: sentence -> tokens -> integer ids
# =============================================================================
def load_sentences():                                   # read the three files into parallel lists
    texts, labels, sources = [], [], []                   # the sentence, its 0/1 label, which website it came from
    for name in ["amazon_cells", "imdb", "yelp"]:          # the three sources, 1,000 sentences each
        with open(os.path.join(DATA_DIR, name + "_labelled.txt"), encoding="utf-8") as f:  # one "sentence<TAB>label" per line
            for line in f:                                  # every line of the file
                text, label = line.rstrip("\n").rsplit("\t", 1)  # split on the LAST tab: sentence | label
                texts.append(text); labels.append(int(label)); sources.append(name.split("_")[0])  # keep all three
    return texts, np.array(labels), np.array(sources)       # labels and sources as arrays for easy indexing


def tokenize(sentence):                                 # sentence -> list of lowercase tokens
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[!?]", sentence.lower())  # words, contractions like don't, and ! or ? as tokens


def build_vocab(token_lists, min_count=2):              # choose which words get their own id
    counts = Counter(w for toks in token_lists for w in toks)  # how many times each word appears in the TRAINING sentences
    itos = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= min_count)  # id -> word; rare words are left out
    return itos, {w: i for i, w in enumerate(itos)}        # and the reverse map, word -> id


def encode(token_lists, stoi, max_len=MAX_LEN):          # token lists -> one padded integer array + the true lengths
    ids = np.full((len(token_lists), max_len), PAD, dtype=int)  # start with every position = padding
    lens = np.zeros(len(token_lists), dtype=int)           # how many real tokens each sentence has
    for row, toks in enumerate(token_lists):               # one sentence at a time
        seq = [stoi.get(w, UNK) for w in toks][:max_len] or [UNK]  # word -> id (unknown -> UNK), cut to max_len; never empty
        ids[row, :len(seq)] = seq                            # real tokens go at the START; padding stays at the end
        lens[row] = len(seq)                                 # remember where the real tokens stop
    return ids, lens                                        # (n, max_len) ids and (n,) lengths


def shuffle_within(ids, lens, seed):                     # shuffle the words INSIDE each sentence; padding stays put
    rng = np.random.RandomState(seed)                      # private rng
    out = ids.copy()                                        # never modify the caller's array
    for row in range(len(lens)):                            # one sentence at a time
        out[row, :lens[row]] = rng.permutation(ids[row, :lens[row]])  # same words, random order
    return out                                              # same shape, same lengths


# =============================================================================
# 2. EMBEDDING + MASKED RNN -- NEW today
# =============================================================================
def embedding_forward(E, ids):                           # NEW today: word ids -> word vectors
    return E[ids]                                          # row lookup: (m, T) ids -> (m, T, D) vectors


def embedding_backward(dX, ids, vocab_size):             # NEW today: gradient for the lookup table
    dE = np.zeros((vocab_size, dX.shape[-1]))              # one gradient row per word in the vocabulary
    np.add.at(dE, ids, dX)                                 # ADD each position's gradient into its word's row; repeats accumulate
    return dE                                              # rows of words not in this batch stay exactly zero


def rnn_forward_masked(X, mask, h0, Wxh, Whh, bh):        # Day 57's recurrence + NEW: padded steps leave the state alone
    m, T, _ = X.shape                                      # batch size, sequence length, embedding size
    H = np.zeros((m, T, Whh.shape[0]))                     # every hidden state
    h = h0                                                  # initial state
    for t in range(T):                                      # one word at a time
        h_new = np.tanh(X[:, t, :] @ Wxh + h @ Whh + bh)      # the usual recurrence
        m_t = mask[:, t:t + 1]                                # 1 where this position is a real word, 0 where it is padding
        h = m_t * h_new + (1.0 - m_t) * h                     # real word: take the new state; padding: carry the old one
        H[:, t, :] = h                                        # store it
    return H                                                # H[:, -1] is the state after each sentence's last REAL word


def rnn_backward_masked(dH_direct, dh_last, X, mask, H, h0, Wxh, Whh):  # Day 58's BPTT through the masked recurrence
    m, T, _ = X.shape                                      # batch size, sequence length, embedding size
    dWxh = np.zeros_like(Wxh); dWhh = np.zeros_like(Whh); dbh = np.zeros(Whh.shape[0])  # accumulators over timesteps
    dX = np.zeros_like(X)                                   # gradient for every input vector, for the embedding table
    dh = dh_last                                            # gradient arriving at the final state
    for t in reversed(range(T)):                            # backward through time
        dh = dh + dH_direct[:, t, :]                          # plus any gradient the readout sends straight to h_t (mean readout)
        m_t = mask[:, t:t + 1]                                # the same mask the forward pass used
        h_prev = H[:, t - 1, :] if t > 0 else h0              # the state that fed step t
        da = dh * m_t * (1.0 - H[:, t, :] ** 2)               # undo tanh -- only for real words; padding contributes nothing
        dWxh += X[:, t, :].T @ da / m                         # ACCUMULATE
        dWhh += h_prev.T @ da / m                             # ACCUMULATE
        dbh += da.sum(axis=0) / m                             # ACCUMULATE
        dX[:, t, :] = da @ Wxh.T / m                          # gradient for this position's word vector
        dh = da @ Whh.T + dh * (1.0 - m_t)                    # real word: back through Whh; padding: pass straight through
    return dWxh, dWhh, dbh, dX                               # everything summed over timesteps


class SentimentRNN:                                     # embedding -> masked RNN -> ONE label per sentence (many-to-one)
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=16, lr=0.1, clip=1.0, readout="last", seed=0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.vocab_size, self.hidden_dim = vocab_size, hidden_dim  # sizes needed later
        self.lr, self.clip, self.readout = lr, clip, readout  # learning rate, clip threshold (or None), "last" or "mean"
        self.E = rng.randn(vocab_size, embed_dim) * 0.1      # the embedding table: one small random vector per word
        self.Wxh = rng.randn(embed_dim, hidden_dim) * np.sqrt(1.0 / embed_dim)  # input-to-hidden weights
        self.Whh = rng.randn(hidden_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)  # recurrent weights
        self.bh = np.zeros(hidden_dim)                       # hidden bias
        self.Why = rng.randn(hidden_dim, 2) * np.sqrt(1.0 / hidden_dim)  # output weights: hidden -> 2 classes
        self.by = np.zeros(2)                                # output bias

    def forward(self, ids, lens, use_mask=True):           # ids: (m, T) word ids; lens: (m,) real lengths
        m, T = ids.shape                                     # batch size, padded length
        self.ids, self.lens = ids, lens                      # cache for backward
        self.mask = (np.arange(T)[None, :] < lens[:, None]).astype(float) if use_mask else np.ones((m, T))  # 1 = real word
        self.h0 = np.zeros((m, self.hidden_dim))             # initial state
        self.X = embedding_forward(self.E, ids)              # (m, T, D) word vectors
        self.H = rnn_forward_masked(self.X, self.mask, self.h0, self.Wxh, self.Whh, self.bh)  # every hidden state
        if self.readout == "last":                           # many-to-one, option A
            self.feat = self.H[:, -1, :]                       # the state after the last real word
        else:                                                # many-to-one, option B
            self.feat = (self.H * self.mask[:, :, None]).sum(axis=1) / lens[:, None]  # the average state over real words
        self.probs = softmax(self.feat @ self.Why + self.by)  # ONE prediction per sentence
        return self.probs                                     # hand back

    def gradients(self, y):                                # gradients of the mean loss w.r.t. all six parameters; no update
        m, T = self.ids.shape                                # batch size, padded length
        dlogits = self.probs - one_hot(y, 2)                 # softmax + cross-entropy gradient
        grads = dict(Why=self.feat.T @ dlogits / m, by=dlogits.sum(axis=0) / m)  # output-layer gradients
        dfeat = dlogits @ self.Why.T                         # gradient arriving at the readout feature
        if self.readout == "last":                           # only the final state was read
            dH_direct, dh_last = np.zeros_like(self.H), dfeat  # so all of it enters at the end
        else:                                                # every real state was read, with weight 1/length
            dH_direct = dfeat[:, None, :] * self.mask[:, :, None] / self.lens[:, None, None]  # each gets a direct share
            dh_last = np.zeros_like(dfeat)                     # nothing extra at the end
        grads["Wxh"], grads["Whh"], grads["bh"], dX = rnn_backward_masked(
            dH_direct, dh_last, self.X, self.mask, self.H, self.h0, self.Wxh, self.Whh)  # BPTT
        grads["E"] = embedding_backward(dX, self.ids, self.vocab_size)  # route input gradients into the lookup table
        return grads                                          # dict: parameter name -> gradient

    def train_step(self, ids, lens, y):                    # one SGD step; returns (loss, was this step clipped?)
        loss = cross_entropy_loss(self.forward(ids, lens), one_hot(y, 2))  # forward pass and batch loss
        grads, _, scale = clip_by_global_norm(self.gradients(y), self.clip)  # Day 60's clipping
        for name, g in grads.items():                        # plain SGD
            getattr(self, name)[...] -= self.lr * g            # in place
        return loss, scale < 1.0                              # loss before the update, and whether clipping fired

    def predict(self, ids, lens, use_mask=True):           # predicted class per sentence
        return self.forward(ids, lens, use_mask).argmax(axis=1)  # the more probable class

    def accuracy(self, ids, lens, y, use_mask=True):       # classification accuracy
        return (self.predict(ids, lens, use_mask) == y).mean()  # fraction correct


def fit(data, vocab_size, seed, order="original", epochs=40, batch_size=32, **net_kwargs):  # train one network, tracking every epoch
    (ids_tr, len_tr, y_tr), (ids_va, len_va, y_va), (ids_te, len_te, y_te) = data  # the three splits
    net = SentimentRNN(vocab_size, seed=seed, **net_kwargs)  # fresh network
    rng = np.random.RandomState(seed)                       # shuffles the ORDER OF SENTENCES each epoch
    hist = dict(loss=[], train=[], val=[], test=[], clipped=[])  # one entry per epoch
    best_val, best_net, best_epoch = -1.0, None, -1          # the snapshot with the best validation accuracy
    for epoch in range(epochs):                              # full training budget
        if order == "original":                                # sentences as written
            src = ids_tr
        elif order == "shuffled once":                         # each sentence's words scrambled ONE fixed way, reused every epoch
            src = shuffle_within(ids_tr, len_tr, seed=1000)
        else:                                                  # "reshuffled each epoch": a new scramble every epoch
            src = shuffle_within(ids_tr, len_tr, seed=1000 + epoch)
        perm = rng.permutation(len(y_tr))                      # fresh sentence order
        losses, clips = [], []                                 # this epoch's per-step records
        for start in range(0, len(perm), batch_size):          # mini-batches
            idx = perm[start:start + batch_size]                 # this batch's sentences
            T_batch = len_tr[idx].max()                          # pad only to the longest sentence IN THIS BATCH
            loss, was_clipped = net.train_step(src[idx][:, :T_batch], len_tr[idx], y_tr[idx])  # one SGD step
            losses.append(loss); clips.append(was_clipped)       # record it
        hist["loss"].append(np.mean(losses)); hist["clipped"].append(np.mean(clips))  # epoch averages
        hist["train"].append(net.accuracy(ids_tr, len_tr, y_tr))  # accuracy on the training sentences, original order
        hist["val"].append(net.accuracy(ids_va, len_va, y_va))    # validation accuracy NOW
        hist["test"].append(net.accuracy(ids_te, len_te, y_te))   # test accuracy NOW (recorded, never used to choose)
        if hist["val"][-1] > best_val:                           # a new best on VALIDATION
            best_val, best_net, best_epoch = hist["val"][-1], copy.deepcopy(net), epoch  # snapshot it
    return best_net, best_epoch, {k: np.array(v) for k, v in hist.items()}  # best snapshot, when, and the full record


RUNS = {}                                               # cache: identical runs requested by two sections train only once


def fit_cached(data, vocab_size, seed, order="original", **net_kwargs):  # fit(), remembered by its settings
    key = (seed, order, tuple(sorted(net_kwargs.items())))   # everything that defines the run
    if key not in RUNS:                                      # first request
        RUNS[key] = fit(data, vocab_size, seed, order, **net_kwargs)  # train it
    return RUNS[key]                                         # later requests reuse the result


# =============================================================================
# 3. THE ORDER-BLIND BASELINE -- bag-of-words logistic regression
# =============================================================================
def bag_of_words(ids, vocab_size):                       # (n, T) ids -> (n, V) matrix: 1 if the word appears, else 0
    X = np.zeros((ids.shape[0], vocab_size))               # one column per vocabulary word
    rows = np.repeat(np.arange(ids.shape[0]), ids.shape[1])  # the row index of every (sentence, position) pair
    X[rows, ids.ravel()] = 1.0                              # mark each word that occurs; ORDER IS DISCARDED here
    X[:, PAD] = 0.0                                         # padding is not a word
    return X                                                # presence/absence features


def train_bow(X_tr, y_tr, X_va, y_va, lr=2.0, epochs=600, l2=1e-4):  # softmax regression by full-batch gradient descent
    W = np.zeros((X_tr.shape[1], 2)); b = np.zeros(2)       # one weight per (word, class), plus a bias
    best = (-1.0, None, None, -1)                           # best validation accuracy, its W, its b, its epoch
    for epoch in range(epochs):                             # full budget
        dlogits = (softmax(X_tr @ W + b) - one_hot(y_tr, 2)) / len(y_tr)  # softmax + cross-entropy gradient
        W -= lr * (X_tr.T @ dlogits + l2 * W)                 # gradient step with a little L2 regularization
        b -= lr * dlogits.sum(axis=0)                         # bias step
        val = ((X_va @ W + b).argmax(axis=1) == y_va).mean()  # validation accuracy NOW
        if val > best[0]:                                     # a new best on validation
            best = (val, W.copy(), b.copy(), epoch)             # snapshot it
    return best                                              # (val accuracy, W, b, epoch)


# =============================================================================
# DEMOS
# =============================================================================
def prepare_data():                                     # load, split, build the vocabulary, encode -- used by every section
    texts, y, sources = load_sentences()                    # 3,000 sentences
    tokens = [tokenize(t) for t in texts]                   # each as a list of tokens
    perm = np.random.RandomState(0).permutation(len(texts))  # one fixed random split
    split = dict(train=perm[:2000], val=perm[2000:2400], test=perm[2400:])  # 2,000 / 400 / 600
    itos, stoi = build_vocab([tokens[i] for i in split["train"]])  # vocabulary from TRAINING sentences only
    data = [encode([tokens[i] for i in split[s]], stoi) + (y[split[s]],) for s in ["train", "val", "test"]]  # (ids, lens, y) x 3
    return texts, y, sources, tokens, split, itos, stoi, data  # everything later sections need


def demo_text_pipeline(texts, y, sources, tokens, split, itos, stoi, data):
    print("=" * 70)
    print("1. TEXT -> NUMBERS -- tokenize, build a vocabulary, encode, pad")
    print("=" * 70)
    print("A network multiplies numbers; it cannot multiply words. Each sentence is split "
          "into tokens, each token is replaced by an integer id from a vocabulary built on "
          "the training sentences only, and every sentence is padded with id 0 to a common "
          "length so a batch fits in one array.")
    lengths = np.array([len(t) for t in tokens])             # tokens per sentence, before truncation
    print(f"\nsentences: {len(texts)}  (positive {int(y.sum())}, negative {int((1 - y).sum())})  "
          f"sources: " + ", ".join(f"{s} {int((sources == s).sum())}" for s in ["amazon", "imdb", "yelp"]))
    print(f"split: train {len(split['train'])}, validation {len(split['val'])}, test {len(split['test'])}  "
          f"(positive fraction {y[split['train']].mean():.3f} / {y[split['val']].mean():.3f} / {y[split['test']].mean():.3f})")
    print(f"tokens per sentence: min {lengths.min()}, median {int(np.median(lengths))}, "
          f"90th percentile {int(np.percentile(lengths, 90))}, max {lengths.max()}; "
          f"{int((lengths > MAX_LEN).sum())} sentences longer than {MAX_LEN} are truncated; "
          f"{int((lengths == 0).sum())} have no word tokens and become a single <unk>")
    counts = Counter(w for i in split["train"] for w in tokens[i])  # training word counts
    print(f"distinct words in training sentences: {len(counts)}; kept (seen at least twice): "
          f"{len(itos) - 2}; vocabulary size with <pad> and <unk>: {len(itos)}")
    ids_te, len_te, _ = data[2]                               # the encoded test split
    real = ids_te[ids_te != PAD]                              # every real test token
    print(f"test tokens that are <unk>: {(real == UNK).mean():.1%}")
    i = int(split["train"][4])                                # one training sentence, as a worked example
    ids_one, len_one = encode([tokens[i]], stoi)              # encode it on its own
    print(f"\nworked example (training sentence, label {y[i]}):")
    print(f"  text   : {texts[i]}")
    print(f"  tokens : {tokens[i]}")
    print(f"  ids    : {ids_one[0, :len_one[0]].tolist()}  + {MAX_LEN - len_one[0]} padding zeros")
    print(f"  decoded: {[itos[j] for j in ids_one[0, :len_one[0]]]}")
    print("Actual output from this lesson's run.")


def demo_checks():
    print("\n" + "=" * 70)
    print("2. CHECKING THE NEW PIECES -- gradient check, and batch invariance")
    print("=" * 70)
    print("(a) Numeric gradient check of all six parameters, on a tiny network and a batch of "
          "four sentences with DIFFERENT lengths and a repeated word, for both readouts. "
          "(b) Batch invariance: a 3-word sentence must get the same prediction alone as it "
          "does inside a batch padded to 40 positions.")
    rng = np.random.RandomState(3)                             # private rng for the check
    lens = np.array([5, 2, 4, 1])                              # four different lengths
    ids = rng.randint(2, 12, size=(4, 5))                      # random word ids from a 12-word vocabulary
    ids[0, 3] = ids[0, 1]                                      # the SAME word twice in one sentence
    ids[np.arange(5)[None, :] >= lens[:, None]] = PAD          # pad past each sentence's length
    y = np.array([0, 1, 1, 0])                                 # labels
    print(f"\n{'readout':>8}" + "".join(f"{p:>11}" for p in ["E", "Wxh", "Whh", "bh", "Why", "by"]) + f"{'max':>11}")
    for readout in ["last", "mean"]:                           # both many-to-one readouts
        net = SentimentRNN(12, embed_dim=4, hidden_dim=5, readout=readout, clip=None, seed=1)  # tiny network
        net.forward(ids, lens)                                   # forward pass
        analytic = net.gradients(y)                              # the gradients backward computed
        errs = []                                                # one max relative error per parameter
        for name in ["E", "Wxh", "Whh", "bh", "Why", "by"]:        # every parameter
            P = getattr(net, name)                                 # the parameter array itself
            numeric = np.zeros_like(P)                             # numeric gradient, same shape
            for pos in np.ndindex(P.shape):                        # every single entry
                old = P[pos]                                         # remember it
                P[pos] = old + 1e-5; lp = cross_entropy_loss(net.forward(ids, lens), one_hot(y, 2))  # loss nudged up
                P[pos] = old - 1e-5; lm = cross_entropy_loss(net.forward(ids, lens), one_hot(y, 2))  # loss nudged down
                P[pos] = old                                         # restore
                numeric[pos] = (lp - lm) / 2e-5                      # central difference
            errs.append(np.max(np.abs(numeric - analytic[name]) / (np.abs(numeric) + np.abs(analytic[name]) + 1e-12)))
        print(f"{readout:>8}" + "".join(f"{e:>11.2e}" for e in errs) + f"{max(errs):>11.2e}")
        if readout == "last":                                    # two structural facts, shown once
            unused = [w for w in range(12) if w not in ids]        # words that never appear in the batch
            pad_ok = np.all(analytic["E"][PAD] == 0)               # padding must receive no gradient
            unused_ok = np.all(analytic["E"][unused] == 0)         # nor may words that were never looked up
    print(f"gradient row of <pad> is exactly zero: {pad_ok};  rows of the {len(unused)} unused words are exactly zero: {unused_ok}")
    print("Actual output from this lesson's run.")

    net = SentimentRNN(50, seed=0)                               # an untrained network, default sizes
    sent = np.array([[7, 21, 33]])                               # one 3-word sentence
    alone = net.forward(sent, np.array([3]))[0]                  # prediction for it alone: no padding at all
    batch = np.full((2, MAX_LEN), PAD); batch[0, :3] = sent      # the same sentence padded to 40 ...
    batch[1] = rng.randint(2, 50, size=MAX_LEN)                  # ... next to a full-length 40-word sentence
    batch_lens = np.array([3, MAX_LEN])                          # true lengths
    masked = net.forward(batch, batch_lens)[0]                   # with masking
    unmasked = net.forward(batch, batch_lens, use_mask=False)[0]  # without: the RNN keeps running over 37 padding steps
    print(f"\nP(positive) for the 3-word sentence:")
    print(f"  alone, no padding            : {alone[1]:.10f}")
    print(f"  in the batch, WITH mask      : {masked[1]:.10f}   |difference| = {abs(masked[1] - alone[1]):.2e}")
    print(f"  in the batch, WITHOUT mask   : {unmasked[1]:.10f}   |difference| = {abs(unmasked[1] - alone[1]):.2e}")
    print("Actual output from this lesson's run.")


def demo_baseline(itos, data):
    print("\n" + "=" * 70)
    print("3. THE ORDER-BLIND BASELINE -- bag-of-words logistic regression")
    print("=" * 70)
    print("Each sentence becomes a vector of 0s and 1s: which vocabulary words appear in it. "
          "Word ORDER is thrown away. A softmax regression on those vectors is the bar the "
          "RNN has to clear: anything the RNN adds must come from order.")
    (ids_tr, _, y_tr), (ids_va, _, y_va), (ids_te, _, y_te) = data  # the three splits
    V = len(itos)                                                # vocabulary size
    X_tr, X_va, X_te = [bag_of_words(ids, V) for ids in (ids_tr, ids_va, ids_te)]  # order-free features
    val, W, b, epoch = train_bow(X_tr, y_tr, X_va, y_va)         # train; keep the best-validation weights
    test = ((X_te @ W + b).argmax(axis=1) == y_te).mean()         # test accuracy at that epoch
    print(f"\nbest validation accuracy {val:.3f} at epoch {epoch}; test accuracy there {test:.3f}  "
          f"(chance = 0.500; 600 test sentences, so one standard error is roughly 0.02)")
    pull = W[:, 1] - W[:, 0]                                      # how strongly each word pulls toward "positive"
    order = np.argsort(pull)                                      # most negative first
    print(f"most negative words: {', '.join(itos[i] for i in order[:10])}")
    print(f"most positive words: {', '.join(itos[i] for i in order[::-1][:10])}")
    print("Actual output from this lesson's run.")
    return dict(W=W, b=b, test=test, val=val, X_te=X_te)           # later sections compare against this


def demo_train_rnn(itos, data, bow):
    print("\n" + "=" * 70)
    print("4. TRAINING THE MANY-TO-ONE RNN -- last hidden state, tracked every epoch")
    print("=" * 70)
    print("Embedding 16, hidden 16, learning rate 0.1, clip threshold 1.0, batch 32, 40 "
          "epochs, seed 0. The network reads a sentence word by word and predicts from the "
          "hidden state after the last real word. Validation accuracy picks the epoch; test "
          "accuracy is recorded alongside but never used to choose.")
    (ids_tr, len_tr, y_tr), _, (ids_te, len_te, y_te) = data    # splits needed below
    net, best_epoch, h = fit_cached(data, len(itos), seed=0, lr=0.1, readout="last")  # the run
    print(f"\n{'epoch':>6}{'train loss':>12}{'train acc':>11}{'val acc':>9}{'test acc':>10}{'steps clipped':>15}")
    for e in range(0, 40, 3):                                     # every third epoch; the last one printed is epoch 39
        print(f"{e:>6}{h['loss'][e]:>12.4f}{h['train'][e]:>11.3f}{h['val'][e]:>9.3f}{h['test'][e]:>10.3f}{h['clipped'][e]:>15.1%}")
    print("Actual output from this lesson's run.")
    print(f"\nbest validation accuracy {h['val'][best_epoch]:.3f} at epoch {best_epoch}: "
          f"train {h['train'][best_epoch]:.3f}, test {h['test'][best_epoch]:.3f}")
    print(f"at the end (epoch 39): train {h['train'][-1]:.3f}, validation {h['val'][-1]:.3f}, test {h['test'][-1]:.3f}")
    print(f"bag-of-words baseline on the same split: validation {bow['val']:.3f}, test {bow['test']:.3f}")
    unmasked = net.accuracy(ids_te, len_te, y_te, use_mask=False)  # same trained network, padding NOT masked
    print(f"same best-epoch network, but reading the state after the padding instead (no mask): "
          f"test {unmasked:.3f}")
    print("Actual output from this lesson's run.")

    fig, ax = plt.subplots(figsize=(9, 4.6))                      # one panel
    ax.plot(h["train"], color="#1f77b4", label="RNN, training sentences")  # training accuracy
    ax.plot(h["val"], color="#ff7f0e", label="RNN, validation sentences")  # validation accuracy
    ax.plot(h["test"], color="#2ca02c", alpha=0.7, label="RNN, test sentences")  # test accuracy
    ax.axhline(bow["test"], color="black", linestyle="--", linewidth=1, label=f"bag-of-words baseline, test ({bow['test']:.3f})")
    ax.axvline(best_epoch, color="#ff7f0e", alpha=0.3, linewidth=5, label=f"best validation epoch ({best_epoch})")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_ylim(0.45, 1.02); ax.legend(fontsize=8, loc="center right")
    ax.set_title("Many-to-one RNN on real sentences: training accuracy climbs to ~1.0,\nvalidation and test stall -- it memorizes")
    plt.tight_layout(); plt.savefig("sentiment_training_curve.png", dpi=110); plt.close()  # save and free
    print("\nsaved sentiment_training_curve.png")


SEEDS = [0, 1, 2, 3, 4]                                 # every comparison below is averaged over these five seeds


def demo_readout(itos, data, bow):
    print("\n" + "=" * 70)
    print("5. WHICH STATE TO READ? -- last hidden state vs. the mean of all of them")
    print("=" * 70)
    print("Two ways to turn many hidden states into one prediction. LAST: read only the final "
          "state, so everything about the sentence must survive to the end. MEAN: average the "
          "states after every real word, so each word has a direct path to the output. Five "
          "seeds each, at two learning rates; test accuracy is taken at each run's best "
          "validation epoch. One row repeats LAST without Day 60's clipping.")
    _, _, (ids_te, len_te, y_te) = data                          # the test split
    rows = {}                                                    # (readout, lr, clip) -> per-seed arrays
    print(f"\n{'readout':>8}{'lr':>6}{'clip':>6}{'val mean':>10}{'test mean':>11}{'test std':>10}{'test min':>10}{'test max':>10}{'final train':>13}")
    for readout, lr, clip in [("last", 0.1, 1.0), ("last", 0.3, 1.0), ("mean", 0.1, 1.0), ("mean", 0.3, 1.0), ("last", 0.1, None)]:
        val, test, train, nets = [], [], [], []                    # per-seed results
        for seed in SEEDS:                                          # five seeds
            net, be, h = fit_cached(data, len(itos), seed=seed, lr=lr, readout=readout, clip=clip)  # train (or reuse)
            val.append(h["val"][be]); test.append(h["test"][be]); train.append(h["train"][-1]); nets.append(net)
        rows[(readout, lr, clip)] = dict(val=np.array(val), test=np.array(test), nets=nets)  # keep for below
        print(f"{readout:>8}{lr:>6}{str(clip):>6}{np.mean(val):>10.3f}{np.mean(test):>11.3f}{np.std(test):>10.3f}"
              f"{np.min(test):>10.3f}{np.max(test):>10.3f}{np.mean(train):>13.3f}")
    print(f"{'bag-of-words baseline':>20}{bow['val']:>10.3f}{bow['test']:>11.3f}")
    print("Actual output from this lesson's run.")

    best = {r: max([k for k in rows if k[0] == r and k[2] == 1.0], key=lambda k: rows[k]["val"].mean()) for r in ["last", "mean"]}
    print(f"\nlearning rate chosen on validation: last -> {best['last'][1]}, mean -> {best['mean'][1]}")
    buckets = [(1, 6), (7, 12), (13, 20), (21, MAX_LEN)]           # sentence-length ranges, in tokens
    bow_ok = (bow["X_te"] @ bow["W"] + bow["b"]).argmax(axis=1) == y_te  # baseline right/wrong per test sentence
    print(f"\ntest accuracy by sentence length (mean over the five seeds):")
    print(f"{'length':>10}{'sentences':>11}{'last':>8}{'mean':>8}{'bag-of-words':>14}")
    by_len = {}                                                  # for the plot
    for lo, hi in buckets:                                        # one length range at a time
        sel = (len_te >= lo) & (len_te <= hi)                       # test sentences in this range
        acc = {r: np.mean([(n.predict(ids_te, len_te) == y_te)[sel].mean() for n in rows[best[r]]["nets"]]) for r in ["last", "mean"]}
        by_len[(lo, hi)] = (acc["last"], acc["mean"], bow_ok[sel].mean())  # keep
        print(f"{f'{lo}-{hi}':>10}{int(sel.sum()):>11}{acc['last']:>8.3f}{acc['mean']:>8.3f}{bow_ok[sel].mean():>14.3f}")
    print("Actual output from this lesson's run.")
    return rows, best, by_len                                     # section 6 and the plot reuse these


def demo_word_order(itos, stoi, data, bow, rows, best, by_len):
    print("\n" + "=" * 70)
    print("6. DOES THE RNN USE WORD ORDER? -- the one thing it has that the baseline does not")
    print("=" * 70)
    print("Three tests on the LAST-state RNN. (a) Shuffle the words inside every TEST sentence: "
          "how many predictions flip, and does accuracy change? (b) TRAIN on scrambled "
          "sentences -- scrambled once and reused, or re-scrambled every epoch -- and test on "
          "normal sentences. If real word order carried usable signal, (b) should hurt. "
          "(c) Hand-written negation probes.")
    _, _, (ids_te, len_te, y_te) = data                          # the test split
    lr = best["last"][1]                                         # the validation-chosen learning rate for LAST
    print(f"\n{'trained on':>24}{'val':>8}{'test':>8}{'test std':>10}{'test, words shuffled':>22}{'predictions flipped':>21}")
    order_res = {}                                               # training mode -> mean test accuracy, for the plot
    for order in ["original", "shuffled once", "reshuffled each epoch"]:  # the three training conditions
        val, test, shuf, flip = [], [], [], []                     # per-seed results
        for seed in SEEDS:                                          # five seeds
            net, be, h = fit_cached(data, len(itos), seed=seed, order=order, lr=lr, readout="last", clip=1.0)  # train (or reuse)
            p0 = net.predict(ids_te, len_te)                          # predictions on the normal test sentences
            ps = [net.predict(shuffle_within(ids_te, len_te, s), len_te) for s in range(5)]  # on five different scrambles
            val.append(h["val"][be]); test.append((p0 == y_te).mean())  # validation and test accuracy, normal word order
            shuf.append(np.mean([(p == y_te).mean() for p in ps]))    # accuracy with the words scrambled
            flip.append(np.mean([(p != p0).mean() for p in ps]))      # fraction of predictions that changed
        order_res[order] = (np.mean(test), np.std(test))           # keep for the plot
        print(f"{order:>24}{np.mean(val):>8.3f}{np.mean(test):>8.3f}{np.std(test):>10.3f}{np.mean(shuf):>22.3f}"
              f"{np.mean(flip):>21.1%}")
    print(f"{'bag-of-words baseline':>24}{bow['val']:>8.3f}{bow['test']:>8.3f}{'':>10}{bow['test']:>22.3f}{0:>21.1%}")
    print("Actual output from this lesson's run.")

    probes = [("good", "not good"), ("bad", "not bad"), ("i love this phone", "i do not love this phone"),
              ("the food was great", "the food was not great"), ("would recommend", "would not recommend"),
              ("this movie is terrible", "this movie is not terrible")]  # written for this lesson, NOT from the dataset
    net = fit_cached(data, len(itos), seed=0, lr=lr, readout="last", clip=1.0)[0]  # the seed-0 network from Section 4/5

    def p_pos_rnn(s):                                            # the RNN's P(positive) for one sentence
        ids, lens = encode([tokenize(s)], stoi)                    # encode it like any other sentence
        return net.forward(ids, lens)[0, 1]

    def p_pos_bow(s):                                            # the baseline's P(positive) for one sentence
        ids, _ = encode([tokenize(s)], stoi)                       # same encoding
        return softmax(bag_of_words(ids, len(itos)) @ bow["W"] + bow["b"])[0, 1]

    print(f"\nnegation probes (hand-written for this lesson, not from the dataset) -- P(positive):")
    print(f"{'sentence':<30}{'RNN':>7}{'BoW':>7}   {'negated sentence':<30}{'RNN':>7}{'BoW':>7}")
    for plain, negated in probes:                                 # one pair per row
        print(f"{plain:<30}{p_pos_rnn(plain):>7.3f}{p_pos_bow(plain):>7.3f}   {negated:<30}{p_pos_rnn(negated):>7.3f}{p_pos_bow(negated):>7.3f}")
    print("Actual output from this lesson's run.")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)  # readout | length | word order
    labels = ["last\n(read final state)", "mean\n(average all states)", "bag-of-words\n(no order)"]  # panel 1 bars
    vals = [rows[best["last"]]["test"].mean(), rows[best["mean"]]["test"].mean(), bow["test"]]  # their test accuracies
    errs = [rows[best["last"]]["test"].std(), rows[best["mean"]]["test"].std(), 0]  # seed-to-seed spread
    a1.bar(range(3), vals, yerr=errs, color=["#1f77b4", "#9467bd", "#7f7f7f"], capsize=4)  # bars with error bars
    a1.set_xticks(range(3)); a1.set_xticklabels(labels, fontsize=8); a1.set_title("Which state to read", fontsize=10)
    a1.set_ylabel("test accuracy")
    x = np.arange(len(by_len))                                    # panel 2: accuracy by sentence length
    for j, (name, color) in enumerate([("last", "#1f77b4"), ("mean", "#9467bd"), ("bag-of-words", "#7f7f7f")]):
        a2.bar(x + (j - 1) * 0.27, [v[j] for v in by_len.values()], width=0.27, color=color, label=name)
    a2.set_xticks(x); a2.set_xticklabels([f"{lo}-{hi} words" for lo, hi in by_len], fontsize=8)
    a2.set_title("Longer sentences are harder for all three", fontsize=10); a2.legend(fontsize=7, loc="lower left")
    names = list(order_res)                                       # panel 3: training on scrambled sentences
    a3.bar(range(3), [order_res[n][0] for n in names], yerr=[order_res[n][1] for n in names],
           color=["#1f77b4", "#ff7f0e", "#2ca02c"], capsize=4)
    a3.set_xticks(range(3)); a3.set_xticklabels(["original\nword order", "scrambled once", "re-scrambled\nevery epoch"], fontsize=8)
    a3.set_title("LAST-state RNN: scrambling the training words does not hurt", fontsize=10)
    for ax in (a1, a2, a3):                                       # shared styling
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1); ax.set_ylim(0.45, 0.9)
    plt.tight_layout(); plt.savefig("sentiment_readout_and_order.png", dpi=110); plt.close()  # save and free
    print("\nsaved sentiment_readout_and_order.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Every piece needed to run an RNN on real text now exists and is checked: tokens, "
          "ids, padding, an embedding table, masking, a many-to-one readout, clipping. What "
          "2,000 sentences and a vanilla RNN did NOT deliver is a use for word order -- the "
          "numbers above are the evidence. Day 62 changes the output side: many-to-many, one "
          "label per word instead of one per sentence, where position is the whole task. "
          "Week 10 changes the cell: the LSTM (Day 63).")


def main():                                             # runs every section in order
    texts, y, sources, tokens, split, itos, stoi, data = prepare_data()  # load and encode once
    demo_text_pipeline(texts, y, sources, tokens, split, itos, stoi, data)  # 1. text -> numbers
    demo_checks()                                          # 2. gradient check + batch invariance
    bow = demo_baseline(itos, data)                        # 3. the order-blind baseline
    demo_train_rnn(itos, data, bow)                        # 4. train the RNN, tracked
    rows, best, by_len = demo_readout(itos, data, bow)     # 5. last vs. mean readout
    demo_word_order(itos, stoi, data, bow, rows, best, by_len)  # 6. does it use word order?
    note_on_what_comes_next()                              # 7. bridge to Day 62 and Week 10


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
