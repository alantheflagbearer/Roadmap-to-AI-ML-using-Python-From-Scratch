"""
Day 62: The Week 9 Build -- a Many-to-Many RNN Tagger From Six Exercise Parts

Week 9's last day does two jobs at once. The syllabus topic is the
MANY-TO-MANY RNN: one output for EVERY input, instead of Day 61's single
output per sentence. And this is the week's integration day: the six
short exercise scripts from Days 56-61 are put together into one final
trained model. So the final model IS the many-to-many network.

The task is part-of-speech tagging: label every word of a sentence as
NOUN, VERB, ADJ, ... The data is real -- 4,078 hand-tagged sentences from
the Universal Dependencies English Web Treebank (Silveira et al., 2014).

The six parts, and where each one goes:

  Day 56 exercise  softmax, one_hot, PlainFCNet -> the no-memory baseline
  Day 57 exercise  rnn_step_forward, rnn_forward -> the forward pass
  Day 58 exercise  rnn_backward (BPTT)           -> the backward pass
  Day 59 exercise  grad_norm_by_timestep         -> a diagnostic probe
  Day 60 exercise  clip_by_global_norm           -> safe update steps
  Day 61 exercise  word ids, padding, embedding  -> text in, vectors out

Only ONE idea is new today: an output, a loss and a gradient at EVERY
timestep. In the backward pass that is a single added line.

What the lesson measures, reported as it came out:
  - a numeric gradient check through the whole assembled model
  - two no-memory baselines: "most frequent tag for this word", and Day
    56's dense network looking at one word at a time
  - the tagger, trained with accuracy tracked every epoch, three seeds
  - WHERE memory helps: unambiguous, ambiguous and unknown words; the same
    tagger with its memory switched off; words shuffled at test time
  - the Day 59 probe and Day 60's clipping at work inside it
  - a limit: this RNN reads left to right, so it never sees what follows;
    a right-to-left copy of the same tagger shows what that leaves out

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import copy                                            # copy: deepcopy, to keep a snapshot of the best network during training
import os                                              # os: build the path to the dataset folder next to this script
from collections import Counter, defaultdict           # Counter counts things; defaultdict creates a missing entry on first use

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the parts diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ud_english_ewt_pos")  # dataset folder beside this file
PAD, UNK = 0, 1                                         # reserved word ids: 0 = padding, 1 = a word not in the vocabulary

# =============================================================================
# PART 1 -- from day56_exercise.py, verbatim: helpers and the no-memory network
# =============================================================================
def relu(z):                                            # ReLU activation
    return np.maximum(0, z)                              # elementwise max(0, z)


def drelu(z):                                           # derivative of ReLU
    return (z > 0).astype(z.dtype)                       # 1 where z>0, else 0


def softmax(z):                                         # turns logits into a probability distribution per row
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract each row's max for numerical stability
    exp_z = np.exp(z_shifted)                             # exponentiate
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row to sum to 1


def one_hot(y, n_classes):                              # integer labels -> one-hot rows
    out = np.zeros((y.shape[0], n_classes))               # start all-zero
    out[np.arange(y.shape[0]), y] = 1                     # set the true class's column to 1
    return out                                            # the one-hot matrix


class PlainFCNet:                                        # Day 56's dense network, unchanged: it has NO memory of earlier inputs
    def __init__(self, T, hidden=16, n_classes=2, lr=0.2, seed=42):  # constructor
        rng = np.random.RandomState(seed)                   # this network's own RNG
        self.W1 = rng.randn(T, hidden) * np.sqrt(2.0 / T)     # FC1 weights, He-scaled
        self.b1 = np.zeros(hidden)                            # FC1 biases
        self.W2 = rng.randn(hidden, n_classes) * np.sqrt(2.0 / hidden)  # FC2 weights
        self.b2 = np.zeros(n_classes)                         # FC2 biases
        self.lr = lr                                          # learning rate

    def forward(self, X):                                  # forward pass
        self.X = X                                            # cache input
        self.Z1 = X @ self.W1 + self.b1                       # FC1 pre-activation
        self.A1 = relu(self.Z1)                               # FC1 activation
        self.Z2 = self.A1 @ self.W2 + self.b2                 # FC2 pre-activation (logits)
        self.probs = softmax(self.Z2)                         # class probabilities
        return self.probs                                     # hand back

    def backward(self, y_onehot):                          # backward pass + SGD update
        m = y_onehot.shape[0]                                 # batch size
        dZ2 = self.probs - y_onehot                           # combined softmax+CE gradient
        dW2 = self.A1.T @ dZ2 / m                              # FC2 weight grad
        db2 = dZ2.sum(axis=0) / m                              # FC2 bias grad
        dA1 = dZ2 @ self.W2.T                                 # grad into FC1 activation
        dZ1 = dA1 * drelu(self.Z1)                             # FC1 pre-activation grad
        dW1 = self.X.T @ dZ1 / m                               # FC1 weight grad
        db1 = dZ1.sum(axis=0) / m                              # FC1 bias grad
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1      # SGD update FC1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2      # SGD update FC2


# =============================================================================
# PART 2 -- from day57_exercise.py: the RNN cell and the forward pass
# =============================================================================
def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):        # ONE timestep -- verbatim from day57_exercise.py
    return np.tanh(x_t @ Wxh + h_prev @ Whh + bh)         # combine this step's input AND the previous hidden state, squash with tanh


def rnn_forward(X, h0, Wxh, Whh, bh):                   # runs the cell once per timestep, collecting every hidden state
    m, T, _ = X.shape                                     # CHANGED: X is now (batch, time, embedding) -- a VECTOR per word, not one number
    H = np.zeros((m, T, Whh.shape[0]))                    # will hold every timestep's hidden state
    h = h0                                                 # start from the given initial hidden state
    for t in range(T):                                     # walk through the sentence one word at a time
        h = rnn_step_forward(X[:, t, :], h, Wxh, Whh, bh)    # CHANGED: X[:, t, :] instead of X[:, t:t+1]; the SAME weights at every t
        H[:, t, :] = h                                       # store this timestep's hidden state
    return H                                               # every hidden state -- today ALL of them are used, not just the last


# =============================================================================
# PART 3 -- from day58_exercise.py: backpropagation through time
# =============================================================================
def rnn_backward(dH, X, H, h0, Wxh, Whh):               # BPTT; dH[:, t, :] is the gradient each timestep's OWN output sends to h_t
    m, T, _ = X.shape                                     # batch size, sequence length, embedding size
    dWxh = np.zeros_like(Wxh)                             # accumulators -- start at zero,
    dWhh = np.zeros_like(Whh)                             # get one contribution PER timestep
    dbh = np.zeros(Whh.shape[0])
    dX = np.zeros_like(X)                                 # Day 61's addition: the gradient for every word vector
    dh = np.zeros_like(h0)                                # CHANGED: nothing special arrives at the end any more
    for t in reversed(range(T)):                           # walk BACKWARD through time
        dh = dh + dH[:, t, :]                                # NEW TODAY: this step's own output adds its gradient -- the many-to-many line
        h_prev = H[:, t - 1, :] if t > 0 else h0              # the state that fed timestep t
        da = dh * (1.0 - H[:, t, :] ** 2)                     # undo tanh using the cached output
        dWxh += X[:, t, :].T @ da                             # ACCUMULATE -- the same Wxh was used at every t
        dWhh += h_prev.T @ da                                 # ACCUMULATE
        dbh += da.sum(axis=0)                                 # ACCUMULATE
        dX[:, t, :] = da @ Wxh.T                              # gradient for this position's word vector
        dh = da @ Whh.T                                       # hand gradient back to h_{t-1}: the "through time" step
    return dWxh, dWhh, dbh, dX                              # summed over every timestep


# =============================================================================
# PART 4 -- from day59_exercise.py: how much gradient reaches each timestep
# =============================================================================
def grad_norm_by_timestep(dH, H, Whh):                  # Day 59's probe, with the same one-line change as rnn_backward
    T = H.shape[1]                                        # sequence length
    norms = np.zeros(T)                                   # one gradient norm per timestep
    dh = np.zeros_like(H[:, 0, :])                        # CHANGED: start from zero; gradient joins step by step
    for t in reversed(range(T)):                           # walk backward through time
        dh = dh + dH[:, t, :]                                # this step's own gradient (all zero except the last step = Day 59 exactly)
        norms[t] = np.linalg.norm(dh)                        # RECORD the total gradient at h_t
        dh = (dh * (1.0 - H[:, t, :] ** 2)) @ Whh.T          # undo tanh, then hand back to h_{t-1}
    return norms                                           # norms[0] = what reaches the very first word


# =============================================================================
# PART 5 -- from day60_exercise.py, verbatim: gradient clipping
# =============================================================================
def clip_by_global_norm(grads, max_norm):               # scale ALL gradients by one factor if their combined length is too big
    norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))  # global norm: every gradient treated as one long vector
    if norm <= max_norm:                                   # already short enough
        return grads, norm                                   # untouched -- an exact no-op
    scale = max_norm / norm                                 # the factor that makes the norm exactly max_norm
    return {k: g * scale for k, g in grads.items()}, norm   # SAME factor for every entry -> direction preserved


# =============================================================================
# PART 6 -- from day61_exercise.py: words -> ids -> padded arrays -> vectors
# =============================================================================
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


def encode(sentences, stoi, ttoi):                      # sentences -> padded id arrays, like Day 61, plus a tag array
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


# =============================================================================
# THE ASSEMBLED MODEL -- parts 1-6 wired together, plus the one new idea
# =============================================================================
class RNNTagger:                                        # words -> embedding -> RNN -> one tag prediction PER WORD
    def __init__(self, vocab_size, n_tags, embed_dim=32, hidden_dim=32, lr=1.0, clip=1.0, seed=0, memory=True):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.n_tags, self.hidden_dim = n_tags, hidden_dim    # sizes needed later
        self.lr, self.clip, self.memory = lr, clip, memory   # learning rate, clip threshold (or None), is the recurrent link on?
        self.E = rng.randn(vocab_size, embed_dim) * 0.1      # embedding table (Day 61): one vector per word
        self.Wxh = rng.randn(embed_dim, hidden_dim) * np.sqrt(1.0 / embed_dim)  # input-to-hidden weights (Day 57)
        self.Whh = rng.randn(hidden_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim) * float(memory)  # recurrent weights; all zero if memory is off
        self.bh = np.zeros(hidden_dim)                       # hidden bias
        self.Why = rng.randn(hidden_dim, n_tags) * np.sqrt(1.0 / hidden_dim)  # output weights: hidden -> tags
        self.by = np.zeros(n_tags)                           # output bias

    def forward(self, ids):                                # ids: (m, T) word ids -> probabilities, one ROW per word
        m, T = ids.shape                                     # batch size, padded length
        self.ids = ids                                       # cache for backward
        self.h0 = np.zeros((m, self.hidden_dim))             # initial state
        self.X = self.E[ids]                                 # Day 61: embedding lookup, (m, T, D)
        self.H = rnn_forward(self.X, self.h0, self.Wxh, self.Whh, self.bh)  # Day 57: every hidden state, (m, T, H)
        self.H2 = self.H.reshape(m * T, self.hidden_dim)     # NEW: treat every word of every sentence as its own row
        self.probs = softmax(self.H2 @ self.Why + self.by)   # NEW: a tag distribution for EVERY word, (m*T, n_tags)
        return self.probs                                     # hand back

    def loss(self, tags, mask):                            # mean cross-entropy over the REAL words only
        p_true = self.probs[np.arange(tags.size), tags.ravel()]  # probability given to each word's true tag
        return -np.sum(np.log(p_true + 1e-9) * mask.ravel()) / mask.sum()  # padding is multiplied by 0; divide by the real word count

    def gradients(self, tags, mask):                       # gradients of that loss w.r.t. all six parameters; no update
        m, T = self.ids.shape                                # batch size, padded length
        dlogits = (self.probs - one_hot(tags.ravel(), self.n_tags)) * mask.ravel()[:, None] / mask.sum()  # zero on padding
        grads = dict(Why=self.H2.T @ dlogits, by=dlogits.sum(axis=0))  # output-layer gradients
        dH = (dlogits @ self.Why.T).reshape(m, T, self.hidden_dim)  # NEW: a gradient for EVERY timestep's hidden state
        grads["Wxh"], grads["Whh"], grads["bh"], dX = rnn_backward(dH, self.X, self.H, self.h0, self.Wxh, self.Whh)  # Day 58
        grads["E"] = np.zeros_like(self.E)                   # Day 61: one gradient row per vocabulary word
        np.add.at(grads["E"], self.ids, dX)                  # Day 61: add each position's gradient into its word's row
        if not self.memory:                                  # memory switched off
            grads["Whh"] = np.zeros_like(self.Whh)             # keep the recurrent weights at exactly zero
        return grads                                          # dict: parameter name -> gradient

    def train_step(self, ids, tags, mask):                 # one SGD step; returns (loss, gradient norm, was it clipped?)
        self.forward(ids)                                    # forward pass
        loss = self.loss(tags, mask)                         # batch loss, before the update
        grads = self.gradients(tags, mask)                   # all gradients
        if self.clip is None:                                # clipping switched off
            norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))  # still measure the norm, for the record
        else:                                                # clipping on
            grads, norm = clip_by_global_norm(grads, self.clip)  # Day 60
        for name, g in grads.items():                        # plain SGD
            getattr(self, name)[...] -= self.lr * g            # in place
        return loss, norm, self.clip is not None and norm > self.clip  # the third value: did clipping change this step?

    def predict(self, ids):                                # predicted tag id for every position, (m, T)
        return self.forward(ids).argmax(axis=1).reshape(ids.shape)  # most probable tag per row, back in sentence shape


def token_accuracy(pred, tags, mask):                   # fraction of REAL words tagged correctly
    return (pred == tags)[mask].mean()                    # padding positions are left out by the mask


def fit(data, vocab_size, n_tags, seed, epochs=25, batch_size=32, **net_kwargs):  # train one tagger, tracking every epoch
    (ids_tr, tags_tr, mask_tr), (ids_va, tags_va, mask_va), (ids_te, tags_te, mask_te) = data  # the three splits
    net = RNNTagger(vocab_size, n_tags, seed=seed, **net_kwargs)  # fresh network
    rng = np.random.RandomState(seed)                       # shuffles the order of sentences each epoch
    lens_tr = mask_tr.sum(axis=1)                           # real length of every training sentence
    hist = dict(loss=[], train=[], val=[], test=[], clipped=[], max_norm=[])  # one entry per epoch
    best_val, best_net, best_epoch = -1.0, None, -1          # the snapshot with the best validation accuracy
    for epoch in range(epochs):                              # full training budget
        perm = rng.permutation(len(ids_tr))                    # fresh sentence order
        losses, norms, clips = [], [], []                      # this epoch's per-step records
        for start in range(0, len(perm), batch_size):          # mini-batches
            idx = perm[start:start + batch_size]                 # this batch's sentences
            T_batch = lens_tr[idx].max()                         # pad only to the longest sentence IN THIS BATCH
            loss, norm, clipped = net.train_step(ids_tr[idx, :T_batch], tags_tr[idx, :T_batch], mask_tr[idx, :T_batch])  # one step
            losses.append(loss); norms.append(norm); clips.append(clipped)  # record it
        hist["loss"].append(np.mean(losses)); hist["clipped"].append(np.mean(clips)); hist["max_norm"].append(np.max(norms))
        hist["train"].append(token_accuracy(net.predict(ids_tr), tags_tr, mask_tr))  # training accuracy NOW
        hist["val"].append(token_accuracy(net.predict(ids_va), tags_va, mask_va))    # validation accuracy NOW
        hist["test"].append(token_accuracy(net.predict(ids_te), tags_te, mask_te))   # test accuracy NOW (recorded, never used to choose)
        if hist["val"][-1] > best_val:                           # a new best on VALIDATION
            best_val, best_net, best_epoch = hist["val"][-1], copy.deepcopy(net), epoch  # snapshot it
    return best_net, best_epoch, {k: np.array(v) for k, v in hist.items()}  # best snapshot, when, and the full record


RUNS = {}                                               # cache: identical runs requested by two sections train only once


def fit_cached(data, vocab_size, n_tags, seed, name="left-to-right", **net_kwargs):  # fit(), remembered by its settings
    key = (name, seed, tuple(sorted(net_kwargs.items())))    # everything that defines the run (name tells datasets apart)
    if key not in RUNS:                                      # first request
        RUNS[key] = fit(data, vocab_size, n_tags, seed, **net_kwargs)  # train it
    return RUNS[key]                                         # later requests reuse the result


SEEDS = [0, 1, 2]                                       # every trained comparison below uses these three seeds


# =============================================================================
# DEMOS
# =============================================================================
def prepare_data():                                     # load the three files, build the vocabularies, encode
    splits = {name: read_tagged(name) for name in ["train", "val", "test"]}  # sentences of (word, tag) pairs
    counts = Counter(w for s in splits["train"] for w, _ in s)  # word counts in the TRAINING sentences
    itos = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= 2)  # word vocabulary, as on Day 61
    stoi = {w: i for i, w in enumerate(itos)}               # word -> id
    tag_names = sorted({t for s in splits["train"] for _, t in s})  # the tag set, alphabetical
    ttoi = {t: i for i, t in enumerate(tag_names)}          # tag -> id
    data = [encode(splits[name], stoi, ttoi) for name in ["train", "val", "test"]]  # (ids, tags, mask) x 3
    return splits, counts, itos, stoi, tag_names, ttoi, data  # everything later sections need


def demo_parts():
    print("=" * 70)
    print("0. THE SIX PARTS -- what each exercise script contributes today")
    print("=" * 70)
    print("Code is copied from each dayNN_exercise.py. Lines that had to change are marked "
          "CHANGED in this script; lines that exist only because of today's topic are marked "
          "NEW.")
    print(f"\n{'from':<18}{'what is reused':<42}{'changed today?'}")
    for row in [("Day 56 exercise", "softmax, one_hot, PlainFCNet", "no -- verbatim"),
                ("Day 57 exercise", "rnn_step_forward, rnn_forward", "2 lines (a vector per word)"),
                ("Day 58 exercise", "rnn_backward (BPTT)", "1 new line (gradient at EVERY step)"),
                ("Day 59 exercise", "grad_norm_by_timestep", "the same 1 new line"),
                ("Day 60 exercise", "clip_by_global_norm", "no -- verbatim"),
                ("Day 61 exercise", "ids + padding, E[ids], np.add.at, mask", "mask applied to the LOSS")]:
        print(f"{row[0]:<18}{row[1]:<42}{row[2]}")


def demo_data(splits, counts, itos, tag_names, data):
    print("\n" + "=" * 70)
    print("1. THE TASK AND THE DATA -- one tag for every word")
    print("=" * 70)
    print("Part-of-speech tagging: label each word with its grammatical role. Source: Universal "
          "Dependencies English Web Treebank (blogs, emails, reviews, newsgroups, answers). "
          "Train = the treebank's dev file, validation and test = its test file, split at a "
          "document boundary, so no document is shared between train and test.")
    print(f"\n{'split':<12}{'sentences':>10}{'words':>9}{'longest':>9}")
    for name, (ids, _, mask) in zip(["train", "validation", "test"], data):
        print(f"{name:<12}{len(ids):>10}{int(mask.sum()):>9}{ids.shape[1]:>9}")
    ids_te, _, mask_te = data[2]                               # the encoded test split
    print(f"\ndistinct words in training sentences: {len(counts)}; kept (seen at least twice): "
          f"{len(itos) - 2}; vocabulary size with <pad> and <unk>: {len(itos)}")
    print(f"test words that are <unk>: {(ids_te[mask_te] == UNK).mean():.1%}")
    tag_counts = Counter(t for s in splits["train"] for _, t in s)  # how common each tag is in training
    n_words = sum(tag_counts.values())                          # total training words
    print(f"\n{len(tag_names)} tags, by share of training words:")
    line = ""                                                   # build rows of six tags each
    for i, (tag, c) in enumerate(tag_counts.most_common()):       # most common first
        line += f"{tag:>6} {c / n_words:>5.1%}  "                   # tag and its share
        if i % 6 == 5 or i == len(tag_counts) - 1:                  # row full, or last tag
            print(" " + line.rstrip()); line = ""                     # print the row, start the next
    sent = splits["train"][5]                                   # one training sentence, as a worked example
    print(f"\nworked example (training sentence):")
    print("  words: " + " ".join(f"{w:<9}" for w, _ in sent[:9]))
    print("  tags : " + " ".join(f"{t:<9}" for _, t in sent[:9]))
    print("Actual output from this lesson's run.")


def numeric_gradient(net, name, ids, tags, mask, eps):    # central-difference gradient of the masked loss for ONE parameter
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
    print("2. THE ASSEMBLED MODEL -- gradient check through all six parts at once")
    print("=" * 70)
    print("A tiny tagger (12 words, 3 tags, embedding 4, hidden 5) on four sentences of lengths "
          "5, 2, 4, 1, with one word repeated. Every entry of every parameter is nudged by "
          "+-1e-5 and the change in the masked loss is compared with the gradient the "
          "assembled backward pass computed. Pass rule: |numeric - analytic| <= 1e-10 + "
          "1e-6 * |numeric| for every entry.")
    rng = np.random.RandomState(3)                             # private rng for the check
    lens = np.array([5, 2, 4, 1])                              # four different lengths
    mask = np.arange(5)[None, :] < lens[:, None]               # real-word mask
    ids = rng.randint(2, 12, size=(4, 5))                      # random real word ids
    ids[0, 3] = ids[0, 1]                                      # the SAME word twice in one sentence
    ids[~mask] = PAD                                           # pad past each sentence's length
    tags = rng.randint(0, 3, size=(4, 5))                      # random tags
    net = RNNTagger(12, 3, embed_dim=4, hidden_dim=5, clip=None, seed=1)  # tiny network
    net.forward(ids)                                           # forward pass
    analytic = net.gradients(tags, mask)                       # the gradients backward computed
    print(f"\n{'parameter':>10}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'all entries pass':>18}")
    worst = (0.0, None, None)                                  # largest relative error seen: (value, parameter, position)
    all_pass = True                                            # does every parameter pass?
    for name in ["E", "Wxh", "Whh", "bh", "Why", "by"]:        # every parameter
        numeric = numeric_gradient(net, name, ids, tags, mask, 1e-5)  # measured slopes
        diff = np.abs(numeric - analytic[name])                  # absolute disagreement, entry by entry
        rel = diff / (np.abs(numeric) + np.abs(analytic[name]) + 1e-12)  # relative disagreement
        ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(numeric)))  # the pass rule
        all_pass = all_pass and ok                               # track the overall verdict
        if rel.max() > worst[0]:                                 # a new worst relative error
            worst = (rel.max(), name, np.unravel_index(rel.argmax(), rel.shape))  # remember where
        print(f"{name:>10}{numeric.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>18}")
    print(f"every parameter passes: {all_pass};  gradient row of <pad> exactly zero: {np.all(analytic['E'][PAD] == 0)}")
    print("Actual output from this lesson's run.")

    _, name, pos = worst                                       # the single entry with the largest relative error
    print(f"\nthe largest relative error is at {name}{tuple(int(p) for p in pos)}, where the gradient itself is tiny: "
          f"{analytic[name][pos]:.3e}. The same entry at three nudge sizes:")
    print(f"{'nudge':>8}{'numeric':>16}{'analytic':>16}{'|difference|':>15}")
    for eps in [1e-4, 1e-5, 1e-6]:                              # a real bug gives the same error at every nudge size
        numeric = numeric_gradient(net, name, ids, tags, mask, eps)[pos]  # re-measure with this nudge
        print(f"{eps:>8.0e}{numeric:>16.6e}{analytic[name][pos]:>16.6e}{abs(numeric - analytic[name][pos]):>15.2e}")
    print("Actual output from this lesson's run.")


def demo_baselines(splits, itos, stoi, tag_names, ttoi, data):
    print("\n" + "=" * 70)
    print("3. TWO BASELINES WITH NO MEMORY -- each word tagged on its own")
    print("=" * 70)
    print("(a) Most-frequent-tag: give every word the tag it had most often in training; "
          "unknown words get the most common tag of rare training words. No learning. "
          "(b) Day 56's PlainFCNet, fed ONE word at a time as a one-hot vector. Both see only "
          "the current word, so the same word always gets the same tag.")
    (ids_tr, tags_tr, mask_tr), (ids_va, tags_va, mask_va), (ids_te, tags_te, mask_te) = data  # the three splits
    V, K = len(itos), len(tag_names)                           # vocabulary size, number of tags
    seen = defaultdict(Counter)                                # word id -> how often each tag was seen with it
    for w, t in zip(ids_tr[mask_tr], tags_tr[mask_tr]):        # every real training word
        seen[w][t] += 1                                          # count the pair
    mft = np.full(V, seen[UNK].most_common(1)[0][0])           # default: the tag most common among <unk> training words
    for w, c in seen.items():                                  # every word seen in training
        mft[w] = c.most_common(1)[0][0]                          # its single most frequent tag
    print(f"\n(a) most-frequent-tag: unknown words are tagged {tag_names[mft[UNK]]}; "
          f"validation {token_accuracy(mft[ids_va], tags_va, mask_va):.4f}, "
          f"test {token_accuracy(mft[ids_te], tags_te, mask_te):.4f}")

    def one_hot_words(word_ids):                               # (n,) word ids -> (n, V) one-hot rows, using Day 56's one_hot
        return one_hot(word_ids, V)

    def fc_predict(net, ids):                                  # tag every position with the dense net, a chunk at a time
        flat = ids.ravel()                                       # every position as its own example
        out = [net.forward(one_hot_words(flat[i:i + 2000])).argmax(axis=1) for i in range(0, len(flat), 2000)]  # chunks keep memory small
        return np.concatenate(out).reshape(ids.shape)            # back to sentence shape

    fc = PlainFCNet(V, hidden=32, n_classes=K, lr=1.0, seed=7)  # Day 56's class, unchanged; its "sequence" is one one-hot word
    words, labels = ids_tr[mask_tr], tags_tr[mask_tr]           # every real training word and its tag, order ignored
    rng = np.random.RandomState(7)                              # shuffling rng
    best = (-1.0, None, -1)                                     # best validation accuracy, its network, its epoch
    print(f"\n(b) PlainFCNet on one word at a time ({V} inputs -> 32 -> {K}):")
    print(f"{'epoch':>6}{'train acc':>11}{'val acc':>9}{'test acc':>10}")
    for epoch in range(20):                                     # 20 passes over the training words
        perm = rng.permutation(len(words))                        # shuffle the words
        for start in range(0, len(perm), 128):                    # mini-batches of 128 words
            idx = perm[start:start + 128]                           # this batch
            fc.forward(one_hot_words(words[idx]))                   # forward
            fc.backward(one_hot(labels[idx], K))                    # backward + update
        accs = [token_accuracy(fc_predict(fc, i), t, m) for i, t, m in data]  # train, validation, test accuracy now
        if accs[1] > best[0]:                                     # a new best on validation
            best = (accs[1], copy.deepcopy(fc), epoch)              # snapshot it
        if epoch % 3 == 0 or epoch == 19:                         # print every third epoch and the last
            print(f"{epoch:>6}{accs[0]:>11.4f}{accs[1]:>9.4f}{accs[2]:>10.4f}")
    fc_pred = fc_predict(best[1], ids_te)                       # test predictions at the best validation epoch
    print(f"best validation {best[0]:.4f} at epoch {best[2]}; test there "
          f"{token_accuracy(fc_pred, tags_te, mask_te):.4f}")
    ambiguous = np.array([len(seen[w]) > 1 for w in range(V)])  # words that had MORE than one tag in training
    ambiguous[UNK] = False                                      # <unk> is its own group
    fc_unk = tag_names[best[1].forward(one_hot_words(np.array([UNK]))).argmax()]  # the one tag PlainFCNet gives every unknown word
    print(f"tag given to EVERY unknown word: most-frequent-tag -> {tag_names[mft[UNK]]}, PlainFCNet -> {fc_unk}")
    print("Actual output from this lesson's run.")
    return dict(mft=mft, mft_pred=mft[ids_te], fc_pred=fc_pred, ambiguous=ambiguous)  # later sections compare against these


def demo_train(itos, tag_names, data, base):
    print("\n" + "=" * 70)
    print("4. TRAINING THE TAGGER -- tracked every epoch, three seeds")
    print("=" * 70)
    print("Embedding 32, hidden 32, learning rate 1.0, clip threshold 1.0, batch 32 sentences, "
          "25 epochs. The loss is the mean over real words. Validation accuracy picks the "
          "epoch; test accuracy is recorded alongside but never used to choose.")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    net, best_epoch, h = fit_cached(data, V, K, seed=0, lr=1.0, clip=1.0)  # the seed-0 run, shown in detail
    print(f"\nseed 0:")
    print(f"{'epoch':>6}{'train loss':>12}{'train acc':>11}{'val acc':>9}{'test acc':>10}{'steps clipped':>15}{'largest grad':>14}")
    for e in list(range(0, 25, 3)):                              # every third epoch (the last printed is epoch 24)
        print(f"{e:>6}{h['loss'][e]:>12.4f}{h['train'][e]:>11.4f}{h['val'][e]:>9.4f}{h['test'][e]:>10.4f}"
              f"{h['clipped'][e]:>15.1%}{h['max_norm'][e]:>14.3f}")
    print("Actual output from this lesson's run.")
    print(f"\n{'seed':>5}{'best epoch':>12}{'val there':>11}{'test there':>12}{'train at end':>14}{'test at end':>13}")
    tests = []                                                 # test accuracy at each seed's best validation epoch
    for seed in SEEDS:                                         # three seeds
        _, be, hs = fit_cached(data, V, K, seed=seed, lr=1.0, clip=1.0)  # train (or reuse)
        tests.append(hs["test"][be])                             # keep the honest number
        print(f"{seed:>5}{be:>12}{hs['val'][be]:>11.4f}{hs['test'][be]:>12.4f}{hs['train'][-1]:>14.4f}{hs['test'][-1]:>13.4f}")
    mft_acc = token_accuracy(base["mft_pred"], tags_te, mask_te)  # baseline (a) on test
    fc_acc = token_accuracy(base["fc_pred"], tags_te, mask_te)    # baseline (b) on test
    print(f"\nRNN tagger test accuracy, mean of {len(SEEDS)} seeds: {np.mean(tests):.4f}  "
          f"(min {np.min(tests):.4f}, max {np.max(tests):.4f})")
    print(f"no-memory baselines on the same test words: most-frequent-tag {mft_acc:.4f}, PlainFCNet {fc_acc:.4f}")
    print(f"test set: {int(mask_te.sum())} words, so one standard error is about "
          f"{np.sqrt(0.82 * 0.18 / mask_te.sum()):.3f}")
    print("Actual output from this lesson's run.")

    fig, ax = plt.subplots(figsize=(9, 4.6))                    # one panel
    ax.plot(h["train"], color="#1f77b4", label="RNN tagger, training words")  # training accuracy
    ax.plot(h["val"], color="#ff7f0e", label="RNN tagger, validation words")  # validation accuracy
    ax.plot(h["test"], color="#2ca02c", alpha=0.7, label="RNN tagger, test words")  # test accuracy
    ax.axhline(mft_acc, color="black", linestyle="--", linewidth=1, label=f"most-frequent-tag baseline, test ({mft_acc:.3f})")
    ax.axvline(best_epoch, color="#ff7f0e", alpha=0.3, linewidth=5, label=f"best validation epoch ({best_epoch})")
    ax.set_xlabel("epoch"); ax.set_ylabel("word-level accuracy"); ax.set_ylim(0.55, 1.0); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Many-to-many RNN tagger (seed 0): one prediction per word, tracked every epoch")
    plt.tight_layout(); plt.savefig("tagger_training_curve.png", dpi=110); plt.close()  # save and free
    print("\nsaved tagger_training_curve.png")


def demo_where_memory_helps(splits, itos, stoi, tag_names, data, base):
    print("\n" + "=" * 70)
    print("5. WHERE DOES MEMORY HELP? -- the RNN against models that see one word at a time")
    print("=" * 70)
    print("Test words are split into three groups: UNAMBIGUOUS (known word that had one tag in "
          "training), AMBIGUOUS (known word that had several), UNKNOWN (<unk>). A no-memory "
          "model must give every <unk> the same tag. Two controls: the SAME tagger trained "
          "with its recurrent weights held at zero (memory off), and the trained tagger fed "
          "test sentences with their words shuffled.")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    groups = {"all words": mask_te,                             # every real test word
              "unambiguous": mask_te & (ids_te != UNK) & ~base["ambiguous"][ids_te],  # known, one tag in training
              "ambiguous": mask_te & base["ambiguous"][ids_te],    # known, several tags in training
              "unknown": mask_te & (ids_te == UNK)}                # not in the vocabulary
    preds = {"most-frequent-tag": [base["mft_pred"]], "PlainFCNet (Day 56)": [base["fc_pred"]],
             "RNN, memory off": [], "RNN tagger": []}           # model -> list of test predictions (one per seed)
    shuffled_acc, flips = [], []                                # for the shuffle test
    rng = np.random.RandomState(0)                              # one fixed shuffle of every test sentence
    order = np.tile(np.arange(ids_te.shape[1]), (len(ids_te), 1))  # start from the identity order in every row
    for row, n in enumerate(mask_te.sum(axis=1)):               # one sentence at a time
        order[row, :n] = rng.permutation(n)                       # shuffle only the real positions
    ids_sh = np.take_along_axis(ids_te, order, axis=1)          # the words, shuffled
    tags_sh = np.take_along_axis(tags_te, order, axis=1)        # each word keeps its own true tag
    for seed in SEEDS:                                          # three seeds
        net = fit_cached(data, V, K, seed=seed, lr=1.0, clip=1.0)[0]  # the tagger from Section 4
        off = fit_cached(data, V, K, seed=seed, lr=1.0, clip=1.0, memory=False)[0]  # identical, memory off
        preds["RNN tagger"].append(net.predict(ids_te)); preds["RNN, memory off"].append(off.predict(ids_te))  # test predictions
        shuffled_acc.append(token_accuracy(net.predict(ids_sh), tags_sh, mask_te))  # accuracy with the words shuffled
    print(f"\n{'':<22}" + "".join(f"{g:>14}" for g in groups))
    print(f"{'share of test words':<22}" + "".join(f"{groups[g].sum() / mask_te.sum():>14.1%}" for g in groups))
    table = {}                                                  # model -> accuracies per group, for the plot
    for name, plist in preds.items():                           # one model per row
        table[name] = [np.mean([(p == tags_te)[groups[g]].mean() for p in plist]) for g in groups]  # mean over seeds
        print(f"{name:<22}" + "".join(f"{a:>14.4f}" for a in table[name]))
    off0 = preds["RNN, memory off"][0]                          # seed 0 of the memory-off tagger
    off_unk = Counter(off0[groups["unknown"]]).most_common(1)[0]  # the tag it gives unknown words, and how often
    print(f"tag the memory-off tagger (seed 0) gives unknown words: {tag_names[off_unk[0]]} "
          f"({off_unk[1] / groups['unknown'].sum():.0%} of them)")
    print("Actual output from this lesson's run.")

    prev = np.full_like(ids_te, -1)                             # the id of the word BEFORE each position; -1 = no word before
    prev[:, 1:] = ids_te[:, :-1]                                # shift the ids one step to the right
    p_rnn0 = preds["RNN tagger"][0]                             # seed-0 tagger predictions, to show which tag it picks
    print(f"\nUNKNOWN test words, grouped by the word just before them (accuracy = mean of {len(SEEDS)} seeds):")
    print(f"{'word before':<21}{'count':>6}{'most common true tag':>22}{'RNN mostly says':>17}{'RNN acc':>9}{'most-freq acc':>15}")
    for label, before in [("(start of sentence)", [-1]), ("the / a / an", ["the", "a", "an"]), ("to", ["to"]),
                          ("i / you / we / they", ["i", "you", "we", "they"]), ("of / in / for / on", ["of", "in", "for", "on"]),
                          (",", [","]), ("another unknown word", [UNK])]:
        before_ids = [stoi[b] if isinstance(b, str) else b for b in before]  # words -> ids; -1 and UNK are already ids
        sel = groups["unknown"] & np.isin(prev, before_ids)       # unknown test words that follow one of these
        true_tag, true_n = Counter(tags_te[sel]).most_common(1)[0]  # what they most often really are
        rnn_tag, rnn_n = Counter(p_rnn0[sel]).most_common(1)[0]     # what the seed-0 tagger most often calls them
        rnn_acc = np.mean([(p == tags_te)[sel].mean() for p in preds["RNN tagger"]])  # tagger accuracy on them
        print(f"{label:<21}{int(sel.sum()):>6}{f'{tag_names[true_tag]} ({true_n / sel.sum():.0%})':>22}"
              f"{f'{tag_names[rnn_tag]} ({rnn_n / sel.sum():.0%})':>17}{rnn_acc:>9.3f}{(base['mft_pred'] == tags_te)[sel].mean():>15.3f}")
    print("Actual output from this lesson's run.")
    rnn_all = table["RNN tagger"][0]                            # the tagger's overall test accuracy (mean of seeds)
    print(f"\nshuffle test -- RNN tagger on normal test sentences {rnn_all:.4f}, on the same sentences "
          f"with words shuffled {np.mean(shuffled_acc):.4f}  (most-frequent-tag is {table['most-frequent-tag'][0]:.4f} either way)")
    print("Actual output from this lesson's run.")

    net = fit_cached(data, V, K, seed=0, lr=1.0, clip=1.0)[0]   # seed-0 tagger, for examples
    p_rnn, p_mft = net.predict(ids_te), base["mft_pred"]        # both models' test predictions
    gain = ((p_rnn == tags_te) & (p_mft != tags_te) & mask_te).sum(axis=1)  # words per sentence the RNN fixes
    loss_ = ((p_rnn != tags_te) & (p_mft == tags_te) & mask_te).sum(axis=1)  # words per sentence the RNN breaks
    lens = mask_te.sum(axis=1)                                  # sentence lengths
    short = np.where((lens >= 5) & (lens <= 9))[0]              # short sentences fit on one line
    pick_good = short[np.argmax(gain[short] - loss_[short])]    # the short sentence where the RNN helps most
    pick_bad = short[np.argmax(loss_[short] - gain[short])]     # the short sentence where the RNN hurts most
    for title, row in [("a short test sentence where memory helps most", pick_good),
                       ("a short test sentence where the RNN does worse", pick_bad)]:
        n = lens[row]                                             # its length
        words = [w for w, _ in splits["test"][row]]               # the original words
        print(f"\n{title}:")
        print("  word      " + " ".join(f"{w[:8]:<8}" for w in words))
        print("  in vocab? " + " ".join(f"{'yes' if ids_te[row, j] != UNK else 'NO':<8}" for j in range(n)))
        print("  true tag  " + " ".join(f"{tag_names[tags_te[row, j]]:<8}" for j in range(n)))
        print("  most-freq " + " ".join(f"{tag_names[p_mft[row, j]]:<8}" for j in range(n)))
        print("  RNN       " + " ".join(f"{tag_names[p_rnn[row, j]]:<8}" for j in range(n)))
    print("Actual output from this lesson's run.")
    return table, groups                                        # the plot in Section 6 reuses the table


def demo_parts_at_work(itos, tag_names, data, table, groups):
    print("\n" + "=" * 70)
    print("6. TWO PARTS DOING THEIR JOB -- Day 59's probe and Day 60's clipping")
    print("=" * 70)
    print("(a) Day 59's probe on the trained tagger: how much gradient reaches each word's hidden "
          "state, for today's loss (every word has its own) and for a loss on the LAST word "
          "only (Day 61's shape). All 20-word test sentences, same network. (b) Day 60's "
          "clipping: the tagger trained with and without it, at this lesson's hidden size (32) "
          "and at a larger one (64).")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    net = fit_cached(data, V, K, seed=0, lr=1.0, clip=1.0)[0]   # the trained seed-0 tagger
    rows = np.where(mask_te.sum(axis=1) == 20)[0]              # test sentences with exactly 20 words: no padding needed
    ids20, tags20 = ids_te[rows, :20], tags_te[rows, :20]       # those sentences
    net.forward(ids20)                                          # forward pass, fills net.H and net.probs
    dlogits = (net.probs - one_hot(tags20.ravel(), K)) / tags20.size  # every word's output gradient (no padding here)
    dH_every = (dlogits @ net.Why.T).reshape(len(rows), 20, -1)  # many-to-many: a gradient at every step
    dH_last = np.zeros_like(dH_every); dH_last[:, -1, :] = dH_every[:, -1, :]  # many-to-one shape: only the last step
    n_every = grad_norm_by_timestep(dH_every, net.H, net.Whh)   # Day 59's probe, today's loss
    n_last = grad_norm_by_timestep(dH_last, net.H, net.Whh)     # Day 59's probe, last-word-only loss
    direct = np.array([np.linalg.norm(dH_every[:, t, :]) for t in range(20)])  # what each step's OWN output contributes
    print(f"\n(a) {len(rows)} test sentences of exactly 20 words; gradient norm at each word's hidden state:")
    print(f"{'word position':>14}{'loss on every word':>20}{'own output only':>17}{'loss on last word only':>24}")
    for t in [0, 4, 9, 14, 17, 18, 19]:                          # a few positions from first to last
        print(f"{t:>14}{n_every[t]:>20.2e}{direct[t]:>17.2e}{n_last[t]:>24.2e}")
    print(f"ratio first word / last word:  every-word loss {n_every[0] / n_every[-1]:.2f},  "
          f"last-word-only loss {n_last[0] / n_last[-1]:.2e}")
    print("Actual output from this lesson's run.")

    print(f"\n(b) test accuracy at the best validation epoch, three seeds each:")
    print(f"{'hidden':>7}{'lr':>5}{'clip':>6}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}{'steps clipped':>15}{'largest grad':>14}")
    settings = [(32, 1.0), (32, 2.0), (64, 1.5), (64, 2.0)]      # (hidden size, learning rate); the first is this lesson's own setting
    clip_res = {}                                               # (hidden, lr, clip) -> per-seed test accuracies, for the plot
    for hidden, lr in settings:                                 # four settings
        for clip in [1.0, None]:                                   # with and without Day 60's clipping
            runs = [fit_cached(data, V, K, seed=s, lr=lr, clip=clip, hidden_dim=hidden) for s in SEEDS]  # three runs
            accs = [h["test"][be] for _, be, h in runs]              # test accuracy at each run's best validation epoch
            clip_res[(hidden, lr, clip)] = accs                      # keep
            print(f"{hidden:>7}{lr:>5}{str(clip):>6}" + "".join(f"{a:>9.4f}" for a in accs) + f"{np.mean(accs):>9.4f}"
                  f"{np.mean([h['clipped'].mean() for _, _, h in runs]):>15.2%}{np.max([h['max_norm'].max() for _, _, h in runs]):>14.2f}")
    print("Actual output from this lesson's run.")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.5, 4.4))  # memory | gradient probe | clipping
    names = ["most-frequent-tag", "RNN, memory off", "RNN tagger"]  # panel 1: three models
    x = np.arange(len(groups))                                  # one cluster per word group
    for j, (name, color) in enumerate(zip(names, ["#7f7f7f", "#ff7f0e", "#1f77b4"])):
        a1.bar(x + (j - 1) * 0.27, table[name], width=0.27, color=color, label=name)  # accuracy per group
    a1.set_xticks(x); a1.set_xticklabels(list(groups), fontsize=8); a1.set_ylim(0, 1.3); a1.set_ylabel("test accuracy")
    a1.set_title("Where memory helps", fontsize=10); a1.legend(fontsize=7, loc="upper right")
    a2.semilogy(n_every, color="#1f77b4", marker="o", markersize=3, label="loss on every word (today)")  # panel 2
    a2.semilogy(n_last, color="#d62728", marker="o", markersize=3, label="loss on the last word only")
    a2.set_xlabel("word position in a 20-word sentence"); a2.set_ylabel("gradient norm at that word's hidden state")
    a2.set_title("Day 59's probe on the trained tagger", fontsize=10); a2.legend(fontsize=7, loc="lower right")
    x = np.arange(len(settings))                                # panel 3: clipping on/off for each setting
    for j, (clip, color, label) in enumerate([(1.0, "#2ca02c", "clipped at 1.0"), (None, "#d62728", "no clipping")]):
        a3.bar(x + (j - 0.5) * 0.36, [np.mean(clip_res[(hd, lr, clip)]) for hd, lr in settings], width=0.36, color=color, alpha=0.55, label=label)
        for i, (hd, lr) in enumerate(settings):                     # every individual seed as a dot
            a3.scatter([x[i] + (j - 0.5) * 0.36] * 3, clip_res[(hd, lr, clip)], color="black", s=12, zorder=3)
    a3.set_xticks(x); a3.set_xticklabels([f"hidden {hd}\nlr {lr}" for hd, lr in settings], fontsize=8)
    a3.set_ylim(0, 1.2); a3.set_ylabel("test accuracy")
    a3.set_title("Day 60's clipping (bars = mean, dots = seeds)", fontsize=10); a3.legend(fontsize=7, loc="upper left", ncol=2)
    plt.tight_layout(); plt.savefig("tagger_memory_and_parts.png", dpi=110); plt.close()  # save and free
    print("\nsaved tagger_memory_and_parts.png")


def reverse_sentences(ids, tags, mask):                 # every sentence written backwards; padding stays at the end
    ids_r, tags_r = ids.copy(), tags.copy()                # never modify the caller's arrays
    for row, n in enumerate(mask.sum(axis=1)):              # one sentence at a time
        ids_r[row, :n] = ids[row, :n][::-1]                   # the real words, last to first
        tags_r[row, :n] = tags[row, :n][::-1]                 # each tag travels with its word
    return ids_r, tags_r, mask                              # same shapes, same mask


def demo_limits(itos, tag_names, data, base, groups):
    print("\n" + "=" * 70)
    print("7. A LIMIT -- this RNN reads left to right only")
    print("=" * 70)
    print("The hidden state at word t has seen words 0..t and nothing after it. To measure what "
          "that leaves out, the SAME tagger is trained on every sentence written backwards -- "
          "a right-to-left reader, which sees only what FOLLOWS each word -- and its "
          "predictions are flipped back and compared, word by word, with the left-to-right "
          "tagger's. Three seeds each.")
    V, K = len(itos), len(tag_names)                           # sizes
    _, _, (ids_te, tags_te, mask_te) = data                    # the test split
    data_rev = [reverse_sentences(*split) for split in data]   # train, validation and test, all reversed
    n_real = mask_te.sum(axis=1)                               # real length of every test sentence
    fwd, bwd = [], []                                          # per-seed test predictions of both readers
    print(f"\n{'seed':>5}{'left-to-right':>15}{'right-to-left':>15}{'both right':>12}{'at least one right':>20}{'they disagree':>15}")
    for seed in SEEDS:                                         # three seeds
        p_f = fit_cached(data, V, K, seed=seed, lr=1.0, clip=1.0)[0].predict(ids_te)  # the tagger from Section 4
        p_rev = fit_cached(data_rev, V, K, seed=seed, name="right-to-left", lr=1.0, clip=1.0)[0].predict(data_rev[2][0])  # reversed reader
        p_b = p_rev.copy()                                       # its predictions, to be flipped back into normal word order
        for row, n in enumerate(n_real):                         # one sentence at a time
            p_b[row, :n] = p_rev[row, :n][::-1]                    # undo the reversal
        ok_f, ok_b = (p_f == tags_te)[mask_te], (p_b == tags_te)[mask_te]  # right/wrong per real test word
        fwd.append(p_f); bwd.append(p_b)                         # keep both
        print(f"{seed:>5}{ok_f.mean():>15.4f}{ok_b.mean():>15.4f}{(ok_f & ok_b).mean():>12.4f}{(ok_f | ok_b).mean():>20.4f}"
              f"{(p_f != p_b)[mask_te].mean():>15.4f}")
    print("Actual output from this lesson's run.")
    print(f"\n{'word group':<14}{'left-to-right':>15}{'right-to-left':>15}")
    for g in ["unambiguous", "ambiguous", "unknown"]:           # the three word groups from Section 5
        print(f"{g:<14}{np.mean([(p == tags_te)[groups[g]].mean() for p in fwd]):>15.4f}"
              f"{np.mean([(p == tags_te)[groups[g]].mean() for p in bwd]):>15.4f}")
    print("Actual output from this lesson's run.")

    print(f"\nthe tags the left-to-right tagger (seed 0) gets wrong most often:")
    print(f"{'true tag':>9}{'test words':>12}{'accuracy':>10}{'most common wrong answer':>27}")
    p0 = fwd[0]                                                 # seed 0's predictions
    stats = []                                                  # (accuracy, tag id, count) for every tag
    for k in range(K):                                          # every tag
        sel = mask_te & (tags_te == k)                            # test words whose true tag is k
        stats.append(((p0 == tags_te)[sel].mean(), k, int(sel.sum())))  # accuracy on them
    for acc, k, n in sorted(stats)[:6]:                         # the six tags with the lowest accuracy
        wrong = Counter(p0[mask_te & (tags_te == k) & (p0 != k)])  # what was predicted instead
        w_tag, w_n = wrong.most_common(1)[0]                      # the most common mistake
        print(f"{tag_names[k]:>9}{n:>12}{acc:>10.4f}{f'{tag_names[w_tag]} ({w_n / n:.0%} of them)':>27}")
    print("Actual output from this lesson's run.")


def draw_parts_diagram():                               # a box-and-arrow picture of how the six parts fit together
    fig, ax = plt.subplots(figsize=(13, 5.6))                   # wide canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")      # plain coordinate grid, no axes

    def box(x, y, w, h, title, sub, color):                      # one rounded box with a bold title and a smaller second line
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1, color="#333333"):                  # one arrow from (x0, y0) to (x1, y1)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5))

    ax.text(6.5, 5.3, "Week 9 build: six exercise parts -> one many-to-many tagger", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.2, 4.55, "FORWARD (left to right)", fontsize=9, color="#1F6F99", fontweight="bold")
    forward = [(0.2, "words -> ids", "Day 61: vocab, padding", "#E6F1F8"), (2.8, "embedding E[ids]", "Day 61: one vector per word", "#E6F1F8"),
               (5.4, "RNN cell, every word", "Day 57: h_t = tanh(...)", "#EAF3DE"), (8.0, "tag scores per word", "NEW: softmax at EVERY step", "#FAEEDA"),
               (10.6, "loss on real words", "Day 61 mask, on the loss", "#E6F1F8")]
    for x, title, sub, color in forward:                         # the five forward boxes
        box(x, 3.35, 2.2, 1.0, title, sub, color)
    for x, _, _, _ in forward[:-1]:                              # arrows between neighbours
        arrow(x + 2.24, 3.85, x + 2.56, 3.85)
    ax.text(10.4, 2.75, "BACKWARD (right to left)", fontsize=9, color="#B03A3A", fontweight="bold", ha="right")
    backward = [(10.6, "gradient per word", "NEW: one dH[:, t] per step", "#FAEEDA"), (8.0, "BPTT", "Day 58: + dH[:, t] each step", "#EAF3DE"),
                (5.4, "clip, then update", "Day 60: cap the step length", "#F4EEFC"), (2.8, "embedding gradient", "Day 61: np.add.at", "#E6F1F8")]
    for x, title, sub, color in backward:                        # the four backward boxes
        box(x, 1.55, 2.2, 1.0, title, sub, color)
    for x, _, _, _ in backward[:-1]:                             # arrows between neighbours, pointing left
        arrow(x - 0.04, 2.05, x - 0.36, 2.05, "#B03A3A")
    arrow(11.7, 3.31, 11.7, 2.63, "#B03A3A")                     # loss -> gradient per word
    box(0.2, 1.55, 2.2, 1.0, "Day 59 probe", "gradient norm per step", "#F2F2F2")  # a diagnostic, beside the chain
    box(0.2, 0.15, 4.8, 0.95, "baseline: Day 56 PlainFCNet", "one word at a time, no memory -- the bar to beat", "#F2F2F2")
    box(5.4, 0.15, 7.4, 0.95, "what is NEW today", "an output, a loss and a gradient at every timestep (1 line in BPTT)", "#FAEEDA")
    plt.savefig("week9_parts_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("\nsaved week9_parts_diagram.png")


def note_on_the_week():
    print("\n" + "=" * 70)
    print("8. THE WEEK IN ONE PARAGRAPH")
    print("=" * 70)
    print("Day 56 showed why sequences need something new; Day 57 built the recurrent cell; "
          "Day 58 trained it with BPTT; Day 59 measured how gradient fades or explodes over "
          "long distances; Day 60 capped the step so explosions cannot wreck training; Day 61 "
          "put real text in. Today those six parts became one model that tags every word of "
          "a sentence. What still limits it is what Day 59 measured: a plain RNN cannot carry "
          "information far. Week 10 replaces the cell with one built for that: the LSTM "
          "(Day 63).")


def main():                                             # runs every section in order
    splits, counts, itos, stoi, tag_names, ttoi, data = prepare_data()  # load and encode once
    demo_parts()                                           # 0. the six parts
    demo_data(splits, counts, itos, tag_names, data)       # 1. the task and the data
    demo_gradient_check()                                  # 2. the assembled model is correct
    base = demo_baselines(splits, itos, stoi, tag_names, ttoi, data)  # 3. two no-memory baselines
    demo_train(itos, tag_names, data, base)                # 4. train the tagger, tracked
    table, groups = demo_where_memory_helps(splits, itos, stoi, tag_names, data, base)  # 5. where memory helps
    demo_parts_at_work(itos, tag_names, data, table, groups)  # 6. the probe and the clipper at work
    demo_limits(itos, tag_names, data, base, groups)       # 7. a limit: left to right only
    draw_parts_diagram()                                   # the parts diagram used in the guides
    note_on_the_week()                                     # 8. the week in one paragraph


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
