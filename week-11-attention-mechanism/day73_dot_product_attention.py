"""
Day 73: Dot-Product (Luong) Attention, and a Real Comparison With Additive

Day 71 scored each encoder state h against the decoder's state s with an
ADDITIVE score: project both, add, squash, score. Luong, Pham and Manning
(2015) proposed MULTIPLICATIVE scores:

    dot:      e_i = s . h_i              no weights at all
    general:  e_i = (s Wg) . h_i         one weight matrix
    additive: e_i = va . tanh(s Wa + h_i Ua + ba)     (Day 71)

Everything else is kept exactly as on Day 71 -- the same encoder, decoder,
data stream, seeds and test sequences -- so the three scores are compared
like for like. (Luong et al. also changed other parts of the model, such
as which decoder state forms the query; those changes are NOT made here,
so that only the score differs.)

What is measured, reported as it came out:

  1. The three scores on one hand-worked step.
  2. A gradient check of the dot and general scores through the model.
  3. COPY of 3-20 digits with each score, three seeds, tracked in training.
  4. REVERSE, and a smaller state.
  5. Where each score looks (Day 72's measurements).
  6. How large the raw scores are, before and after training.
  7. What each score costs per training step.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import time                                            # time: perf_counter, to time one training step in Section 7

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

EOS, GO = 0, 9                                          # token 0 = end of sequence, token 9 = "start writing"; digits are 1..8 (Day 69)
V = 10                                                  # vocabulary: EOS, eight digits, GO (Day 69)
SEEDS = [0, 1, 2]                                       # Day 71's three seeds
EVAL_LENGTHS = (4, 8, 12, 16, 20)                       # Day 70's test lengths
STEPS = 20000                                           # Day 71's training budget
EVERY = 2500                                            # test accuracy every EVERY steps (Day 71)
SCORES = ("additive", "dot", "general")                 # the three score functions compared
COLORS = ("#7f7f7f", "#d62728", "#1f77b4")              # one colour per score, in every plot

# =============================================================================
# REUSED -- verbatim from Day 71 (the LSTM, the soft read, the step backward and additive attention)
# =============================================================================
def sigmoid(z):                                         # squashes any number into the range 0..1
    return 1.0 / (1.0 + np.exp(-z))                       # large negative -> 0, zero -> 0.5, large positive -> 1


def lstm_step_forward(x_t, h_prev, c_prev, W, b):       # ONE timestep of an LSTM cell
    H = h_prev.shape[1]                                   # hidden size
    a = np.concatenate([x_t, h_prev], axis=1) @ W + b     # input and old hidden state side by side, times one weight matrix
    f = sigmoid(a[:, 0 * H:1 * H])                        # forget gate: how much of the old cell state to keep
    i = sigmoid(a[:, 1 * H:2 * H])                        # input gate: how much of the candidate to write
    g = np.tanh(a[:, 2 * H:3 * H])                        # candidate: the new value that could be written
    o = sigmoid(a[:, 3 * H:4 * H])                        # output gate: how much of the cell state to show
    c = f * c_prev + i * g                                # new cell state
    h = o * np.tanh(c)                                    # new hidden state
    return h, c, (f, i, g, o)                             # both states, and the gates


def lstm_forward(X, h0, c0, W, b):                      # the loop from day64_exercise.py, also keeping the gates (day65)
    m, T, _ = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    Hs, Cs = np.zeros((m, T, H)), np.zeros((m, T, H))     # every hidden state and every cell state
    gates = []                                            # (f, i, g, o) at every step, for the backward pass
    h, c = h0, c0                                         # start from the given states -- today: the ENCODER'S final states
    for t in range(T):                                    # one step at a time
        h, c, fiog = lstm_step_forward(X[:, t, :], h, c, W, b)  # same weights every step
        Hs[:, t, :], Cs[:, t, :] = h, c                     # store both states
        gates.append(fiog)                                  # and the gates
    return Hs, Cs, gates                                  # everything backward needs


def lstm_backward(dHs, dh_last, dc_last, X, h0, c0, W, Hs, Cs, gates):  # Day 69: see the three marked lines
    m, T, D = X.shape                                     # batch size, sequence length, input size
    dW, db = np.zeros_like(W), np.zeros(W.shape[1])       # accumulators, one contribution per step
    dh, dc = dh_last.copy(), dc_last.copy()               # Day 69: a gradient may arrive at the last c too (not just h)
    for t in reversed(range(T)):                          # walk BACKWARD through time
        dh = dh + dHs[:, t, :]                              # day67's change: this step's own output adds its gradient
        f, i, g, o = gates[t]                               # this step's gates
        h_prev = Hs[:, t - 1, :] if t > 0 else h0           # the states that fed step t
        c_prev = Cs[:, t - 1, :] if t > 0 else c0
        tanh_c = np.tanh(Cs[:, t, :])                       # tanh of this step's cell state
        do = dh * tanh_c                                    # through h = o * tanh(c), to the output gate
        dc = dc + dh * o * (1 - tanh_c ** 2)                # ... and to the cell state
        da = np.concatenate([dc * c_prev * f * (1 - f),     # forget gate, then its sigmoid
                             dc * g * i * (1 - i),          # input gate, then its sigmoid
                             dc * i * (1 - g ** 2),         # candidate, then its tanh
                             do * o * (1 - o)], axis=1)     # output gate, then its sigmoid
        z = np.concatenate([X[:, t, :], h_prev], axis=1)    # what W multiplied at this step
        dW += z.T @ da                                      # ACCUMULATE -- the same W at every step
        db += da.sum(axis=0)                                # ACCUMULATE
        dh = (da @ W.T)[:, D:]                              # back to h_{t-1}
        dc = dc * f                                         # back to c_{t-1}: times the forget gate
    return dW, db, dh, dc                                 # Day 69: also the gradients at h0 and c0 -- the encoder needs them


def softmax(z):                                         # logits -> probabilities, one row per example
    e = np.exp(z - z.max(axis=1, keepdims=True))          # subtract the row max for stability, exponentiate
    return e / e.sum(axis=1, keepdims=True)               # rows sum to 1


def masked_softmax(scores, mask):                      # softmax over the real slots only; padded slots get weight 0
    s = np.where(mask, scores, -np.inf)                   # a padded slot's score becomes minus infinity
    s = s - s.max(axis=-1, keepdims=True)                 # subtract the largest real score, for stability
    e = np.exp(s)                                         # exp(-inf) = 0, so padding contributes nothing
    return e / e.sum(axis=-1, keepdims=True)              # weights over the real slots, summing to 1


def soft_read(memory, scores, mask):                   # read a weighted average of the memory slots
    w = masked_softmax(scores, mask)                      # (m, n) weights
    return np.einsum("mn,mnd->md", w, memory), w          # (m, d) read vector, and the weights used


def lstm_step_backward(dh, dc, x, h_prev, c_prev, c, gates, W):  # Day 71: gradients of one LSTM step
    f, i, g, o = gates                                    # this step's gates
    D = x.shape[1]                                        # input size
    tanh_c = np.tanh(c)                                   # tanh of this step's cell state
    do = dh * tanh_c                                      # through h = o * tanh(c), to the output gate
    dc = dc + dh * o * (1 - tanh_c ** 2)                  # ... and to the cell state
    da = np.concatenate([dc * c_prev * f * (1 - f),       # forget gate, then its sigmoid
                         dc * g * i * (1 - i),            # input gate, then its sigmoid
                         dc * i * (1 - g ** 2),           # candidate, then its tanh
                         do * o * (1 - o)], axis=1)       # output gate, then its sigmoid
    z = np.concatenate([x, h_prev], axis=1)               # what W multiplied at this step
    dz = da @ W.T                                         # back through the matrix product
    return dz[:, :D], dz[:, D:], dc * f, z.T @ da, da.sum(axis=0)  # dx, dh_prev, dc_prev, dW, db


def additive_scores(s_prev, keys, Wa, va):             # Day 71: e_i = va . tanh(s_prev Wa + keys_i), one score per encoder step
    t = np.tanh((s_prev @ Wa)[:, None, :] + keys)         # (m, n, A): the query added to every key, squashed
    return t @ va, t                                      # (m, n) scores, and t for the backward pass


def additive_scores_backward(de, t, s_prev, Wa, va):   # Day 71: gradients of the scores w.r.t. s_prev, the keys, Wa and va
    dva = np.einsum("mn,mna->a", de, t)                   # e = t . va
    da = de[:, :, None] * va * (1 - t ** 2)               # back through tanh: (m, n, A)
    dq = da.sum(axis=1)                                   # the query was added to every key: sum over steps
    return dq @ Wa.T, da, s_prev.T @ dq, dva              # d s_prev, d keys, dWa, dva


class AttnSeq2Seq:                                      # Day 71: Day 69's encoder-decoder, with the decoder reading every encoder state
    def __init__(self, hidden=32, attn=32, seed=0, forget_bias=1.0, lr=0.5, clip=1.0, init_from_encoder=True):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        H, A = hidden, attn                                # sizes
        self.H, self.lr, self.clip, self.init_from_encoder = H, lr, clip, init_from_encoder  # settings
        self.P = {"We": rng.randn(V + H, 4 * H) / np.sqrt(V + H), "be": np.zeros(4 * H),  # encoder LSTM (Day 69)
                  "Wd": rng.randn(V + 2 * H, 4 * H) / np.sqrt(V + 2 * H), "bd": np.zeros(4 * H),  # decoder LSTM: input [y ; context]
                  "Wa": rng.randn(H, A) / np.sqrt(H), "Ua": rng.randn(H, A) / np.sqrt(H), "ba": np.zeros(A),  # attention
                  "va": rng.randn(A) / np.sqrt(A),
                  "Why": rng.randn(2 * H, V) / np.sqrt(2 * H), "by": np.zeros(V)}  # output layer: from [s ; context]
        self.P["be"][:H] = forget_bias; self.P["bd"][:H] = forget_bias  # forget biases

    def encode(self, Xs):                                  # the encoder: every state is kept, not just the last
        m = Xs.shape[0]                                      # batch size
        z = np.zeros((m, self.H))                            # zero starting states
        Hs, Cs, gates = lstm_forward(Xs, z, z, self.P["We"], self.P["be"])  # Day 64/65's loop
        keys = Hs @ self.P["Ua"] + self.P["ba"]              # each encoder state's key, computed once per sequence
        s, c = (Hs[:, -1], Cs[:, -1]) if self.init_from_encoder else (z, z)  # the decoder's start (Day 69) or zeros
        return Hs, Cs, gates, keys, s, c                     # memory, caches and the decoder's start

    def step(self, y_onehot, s, c, Hs, keys):              # ONE decoder step with attention
        e, t = additive_scores(s, keys, self.P["Wa"], self.P["va"])  # how well each encoder state matches the decoder's state
        ctx, alpha = soft_read(Hs, e, np.ones(e.shape, bool))  # day70 exercise: weights and the context (no padding today)
        x = np.concatenate([y_onehot, ctx], axis=1)          # the decoder reads the previous token AND the context
        s_new, c_new, gates = lstm_step_forward(x, s, c, self.P["Wd"], self.P["bd"])  # Day 63's step
        out = np.concatenate([s_new, ctx], axis=1)           # the output layer sees the new state and the context
        return s_new, c_new, dict(s_prev=s, c_prev=c, t=t, alpha=alpha, ctx=ctx, x=x, s=s_new, c=c_new, gates=gates, out=out)  # new states, and the step's cache

    def loss_and_grads(self, Xs, Xd, Y):                   # forward and backward through encoder, attention and decoder
        P, H = self.P, self.H                                # parameters, hidden size
        m, T = Y.shape                                       # batch, output steps
        Hs, Cs, gates, keys, s, c = self.encode(Xs)          # encoder
        s0, c0 = s, c                                        # the decoder's starting states
        steps, loss = [], 0.0                                # every decoder step's cache; the loss
        g = {k: np.zeros_like(v) for k, v in P.items()}      # gradients
        douts = []                                           # gradient at each step's [s ; context]
        for j in range(T):                                   # teacher forcing: the true previous token as input
            s, c, st = self.step(Xd[:, j], s, c, Hs, keys)     # one step
            p = softmax(st["out"] @ P["Why"] + P["by"])        # this step's prediction
            loss -= np.log(p[np.arange(m), Y[:, j]]).sum() / (m * T)  # mean cross-entropy over batch and steps
            dl = p.copy(); dl[np.arange(m), Y[:, j]] -= 1; dl /= m * T  # softmax + cross-entropy gradient
            g["Why"] += st["out"].T @ dl; g["by"] += dl.sum(axis=0)  # output layer
            douts.append(dl @ P["Why"].T); steps.append(st)    # keep for the backward loop
        dHs = np.zeros_like(Hs); dkeys = np.zeros_like(keys)  # gradients for the memory and its keys
        ds, dc = np.zeros((m, H)), np.zeros((m, H))          # nothing arrives after the last step
        for j in reversed(range(T)):                         # backward through the decoder
            st = steps[j]                                      # this step's cache
            ds = ds + douts[j][:, :H]                          # the output layer read s ...
            dctx = douts[j][:, H:].copy()                      # ... and the context
            dx, ds_prev, dc_prev, dW, db = lstm_step_backward(ds, dc, st["x"], st["s_prev"], st["c_prev"], st["c"], st["gates"], P["Wd"])  # back through the decoder step
            g["Wd"] += dW; g["bd"] += db                       # decoder LSTM
            dctx += dx[:, V:]                                  # the decoder's input also contained the context
            alpha = st["alpha"]                                # this step's attention weights
            dHs += alpha[:, :, None] * dctx[:, None, :]        # context = sum_i alpha_i h_i  ->  d h_i += alpha_i d context
            dalpha = np.einsum("mh,mnh->mn", dctx, Hs)         # d alpha_i = d context . h_i
            de = alpha * (dalpha - (alpha * dalpha).sum(axis=1, keepdims=True))  # back through the softmax
            dsp, dk, dWa, dva = additive_scores_backward(de, st["t"], st["s_prev"], P["Wa"], P["va"])  # back through the scores
            g["Wa"] += dWa; g["va"] += dva; dkeys += dk        # attention weights; keys collect over steps
            ds, dc = ds_prev + dsp, dc_prev                    # the previous state fed the LSTM AND the scores
        g["Ua"] = np.einsum("mnh,mna->ha", Hs, dkeys)        # keys = Hs Ua + ba
        g["ba"] = dkeys.sum(axis=(0, 1))                  # the key bias was added at every step of every sequence
        dHs += dkeys @ P["Ua"].T                             # the encoder states also made the keys
        z = np.zeros((m, H))                                 # the encoder started from zeros
        dh_last, dc_last = (ds, dc) if self.init_from_encoder else (z, z)  # the decoder's start came from the encoder's end
        g["We"], g["be"], _, _ = lstm_backward(dHs, dh_last, dc_last, Xs, z, z, P["We"], Hs, Cs, gates)  # Day 69's backward
        return loss, g                                       # loss and every gradient

    def train_step(self, Xs, Xd, Y):                       # one SGD step with Day 60's clipping (Day 69's rule)
        loss, g = self.loss_and_grads(Xs, Xd, Y)          # forward and backward
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))  # global gradient norm
        scale = min(1.0, self.clip / norm)                # shrink only if too long
        for k in self.P:                                  # plain SGD on every parameter
            self.P[k] -= self.lr * scale * g[k]           # in place
        return loss                                       # for the record

    def generate(self, Xs, n_steps, return_weights=False):  # greedy decoding (Day 69), keeping the attention weights
        Hs, _, _, keys, s, c = self.encode(Xs)               # encoder
        token = np.full(len(Xs), GO)                         # start with GO
        out, weights = [], []                                # tokens written; weights used
        for _ in range(n_steps):                             # one token per step
            s, c, st = self.step(np.eye(V)[token], s, c, Hs, keys)  # attend, then step
            token = (st["out"] @ self.P["Why"] + self.P["by"]).argmax(axis=1)  # most probable token
            out.append(token); weights.append(st["alpha"])     # keep both
        pred = np.stack(out, axis=1)                         # (m, n_steps)
        return (pred, np.stack(weights, axis=1)) if return_weights else pred  # weights: (m, n_steps, n)


def make_batch(rng, m, n, task):                        # Day 69's data, verbatim: m sequences of n digits
    src = rng.randint(1, 9, size=(m, n))                  # random digits 1..8
    tgt = src if task == "copy" else src[:, ::-1]         # COPY: the same digits; REVERSE: backwards
    dec_in = np.concatenate([np.full((m, 1), GO), tgt], axis=1)    # decoder input: GO, then the target shifted right
    dec_out = np.concatenate([tgt, np.full((m, 1), EOS)], axis=1)  # decoder output: the target, then EOS
    return np.eye(V)[src], np.eye(V)[dec_in], dec_out, src  # one-hot source, one-hot decoder input, targets, raw digits


def fixed_test(n, task, m=300):                         # Day 70's fixed test sets: the same sequences as Day 70 used
    return make_batch(np.random.RandomState(12345 + n), m, n, task)  # a seed per length: Day 70's exact test sequences


def exact_and_token(net, batch):                        # Day 70: greedy decoding; exact match and token accuracy
    Xs, _, Y, _ = batch                                   # source and targets
    pred = net.generate(Xs, Y.shape[1])                   # digits + EOS, greedy
    return np.all(pred == Y, axis=1).mean(), (pred == Y).mean()  # whole sequence right; share of tokens right


# =============================================================================
# 1. TWO MULTIPLICATIVE SCORES -- NEW (Luong, Pham and Manning, 2015)
# =============================================================================
def dot_scores(s_prev, Hs):                             # NEW: e_i = s . h_i -- no weights at all
    return np.einsum("mh,mnh->mn", s_prev, Hs)            # one dot product per encoder step


def dot_scores_backward(de, s_prev, Hs):                # NEW: gradients of the dot scores w.r.t. the query and the memory
    ds = np.einsum("mn,mnh->mh", de, Hs)                  # e_i = s . h_i  ->  d s = sum_i de_i h_i
    dHs = de[:, :, None] * s_prev[:, None, :]             # ... and d h_i = de_i s
    return ds, dHs                                        # gradients for the query and the memory


def general_scores(s_prev, Hs, Wg):                     # NEW: e_i = (s Wg) . h_i -- one weight matrix between them
    q = s_prev @ Wg                                       # the query, projected once per step
    return np.einsum("mh,mnh->mn", q, Hs), q              # scores, and q for the backward pass


def general_scores_backward(de, s_prev, Hs, q, Wg):     # NEW: gradients w.r.t. the query, the memory and Wg
    dq = np.einsum("mn,mnh->mh", de, Hs)                  # d q = sum_i de_i h_i
    dHs = de[:, :, None] * q[:, None, :]                  # d h_i = de_i q
    return dq @ Wg.T, dHs, s_prev.T @ dq                  # d s, d Hs, d Wg


class ScoredAttnSeq2Seq(AttnSeq2Seq):                    # NEW: Day 71's model with the score function as a setting
    def __init__(self, score="additive", hidden=32, attn=32, seed=0, forget_bias=1.0, lr=0.5, clip=1.0):
        self.score = score                                 # "additive" (Day 71), "dot" or "general"
        if score == "additive":                            # exactly Day 71's model, same random draws
            super().__init__(hidden=hidden, attn=attn, seed=seed, forget_bias=forget_bias, lr=lr, clip=clip)
            return                                        # nothing else to set up: Day 71's weights are complete
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        H = hidden                                         # hidden size
        self.H, self.lr, self.clip, self.init_from_encoder = H, lr, clip, True  # settings (decoder starts from the encoder)
        self.P = {"We": rng.randn(V + H, 4 * H) / np.sqrt(V + H), "be": np.zeros(4 * H),  # encoder LSTM: the same draws as Day 71
                  "Wd": rng.randn(V + 2 * H, 4 * H) / np.sqrt(V + 2 * H), "bd": np.zeros(4 * H)}  # decoder LSTM: the same draws
        if score == "general":                             # the one weight matrix of the general score
            self.P["Wg"] = rng.randn(H, H) / np.sqrt(H)   # the general score's weight matrix
        self.P.update(Why=rng.randn(2 * H, V) / np.sqrt(2 * H), by=np.zeros(V))  # output layer
        self.P["be"][:H] = forget_bias; self.P["bd"][:H] = forget_bias  # forget biases

    def encode(self, Xs):                                  # Day 71's encoder; the multiplicative scores use the states as keys
        if self.score == "additive":                      # Day 71's path
            return super().encode(Xs)                        # Day 71, unchanged
        m = Xs.shape[0]                                      # batch size
        z = np.zeros((m, self.H))                            # zero starting states
        Hs, Cs, gates = lstm_forward(Xs, z, z, self.P["We"], self.P["be"])  # Day 64/65's loop
        return Hs, Cs, gates, Hs, Hs[:, -1], Cs[:, -1]       # the keys ARE the encoder states

    def scores(self, s, Hs, keys):                         # NEW: the only thing that differs between the three models
        if self.score == "additive":                      # Day 71's path
            return additive_scores(s, keys, self.P["Wa"], self.P["va"])  # Day 71: e and the tanh values
        if self.score == "dot":                           # the dot score
            return dot_scores(s, Hs), None                   # nothing to cache
        return general_scores(s, Hs, self.P["Wg"])           # e and the projected query

    def step(self, y_onehot, s, c, Hs, keys):              # Day 71's decoder step, with the score as a setting
        e, cache = self.scores(s, Hs, keys)                  # one score per encoder state
        ctx, alpha = soft_read(Hs, e, np.ones(e.shape, bool))  # day70 exercise: weights and the context
        x = np.concatenate([y_onehot, ctx], axis=1)          # the decoder reads the previous token AND the context
        s_new, c_new, gates = lstm_step_forward(x, s, c, self.P["Wd"], self.P["bd"])  # Day 63's step
        out = np.concatenate([s_new, ctx], axis=1)           # the output layer sees the new state and the context
        return s_new, c_new, dict(s_prev=s, c_prev=c, t=cache, alpha=alpha, ctx=ctx, x=x, s=s_new, c=c_new, gates=gates, out=out)

    def loss_and_grads(self, Xs, Xd, Y):                   # Day 71's loss_and_grads; only the score's backward pass differs
        if self.score == "additive":                      # Day 71's path
            return super().loss_and_grads(Xs, Xd, Y)         # Day 71, unchanged
        P, H = self.P, self.H                                # parameters, hidden size
        m, T = Y.shape                                       # batch, output steps
        Hs, Cs, gates, keys, s, c = self.encode(Xs)          # encoder
        steps, loss = [], 0.0                                # every decoder step's cache; the loss
        g = {k: np.zeros_like(v) for k, v in P.items()}      # gradients
        douts = []                                           # gradient at each step's [s ; context]
        for j in range(T):                                   # teacher forcing
            s, c, st = self.step(Xd[:, j], s, c, Hs, keys)     # one step
            p = softmax(st["out"] @ P["Why"] + P["by"])        # this step's prediction
            loss -= np.log(p[np.arange(m), Y[:, j]]).sum() / (m * T)  # mean cross-entropy
            dl = p.copy(); dl[np.arange(m), Y[:, j]] -= 1; dl /= m * T  # softmax + cross-entropy gradient
            g["Why"] += st["out"].T @ dl; g["by"] += dl.sum(axis=0)  # output layer
            douts.append(dl @ P["Why"].T); steps.append(st)    # keep for the backward loop
        dHs = np.zeros_like(Hs)                              # gradient for the memory
        ds, dc = np.zeros((m, H)), np.zeros((m, H))          # nothing arrives after the last step
        for j in reversed(range(T)):                         # backward through the decoder (Day 71)
            st = steps[j]                                 # this step's cache
            ds = ds + douts[j][:, :H]                          # the output layer read s ...
            dctx = douts[j][:, H:].copy()                      # ... and the context
            dx, ds_prev, dc_prev, dW, db = lstm_step_backward(ds, dc, st["x"], st["s_prev"], st["c_prev"], st["c"], st["gates"], P["Wd"])  # back through the decoder step
            g["Wd"] += dW; g["bd"] += db                       # decoder LSTM
            dctx += dx[:, V:]                                  # the decoder's input also contained the context
            alpha = st["alpha"]                                # this step's weights
            dHs += alpha[:, :, None] * dctx[:, None, :]        # through the weighted sum (Day 71)
            dalpha = np.einsum("mh,mnh->mn", dctx, Hs)         # d alpha_i = d context . h_i
            de = alpha * (dalpha - (alpha * dalpha).sum(axis=1, keepdims=True))  # back through the softmax
            if self.score == "dot":                            # NEW: the dot score's backward pass
                dsp, dH_score = dot_scores_backward(de, st["s_prev"], Hs)  # gradients for the query and the memory
            else:                                              # NEW: the general score's backward pass
                dsp, dH_score, dWg = general_scores_backward(de, st["s_prev"], Hs, st["t"], P["Wg"])  # ... and for Wg
                g["Wg"] += dWg                            # accumulate over the steps
            dHs += dH_score                                    # the memory also made the scores
            ds, dc = ds_prev + dsp, dc_prev                    # the previous state fed the LSTM AND the scores
        z = np.zeros((m, H))                                 # the encoder started from zeros
        g["We"], g["be"], _, _ = lstm_backward(dHs, ds, dc, Xs, z, z, P["We"], Hs, Cs, gates)  # Day 69's backward
        return loss, g                                    # the loss and every gradient


def train(score, task, hidden, seed, steps=STEPS, lo=3, hi=20, every=EVERY):  # Day 71's training loop, with the score as a setting
    net = ScoredAttnSeq2Seq(score, hidden=hidden, seed=seed)  # fresh model
    rng = np.random.RandomState(seed + 100)                 # Day 69-71's data-stream seed rule: the same batches for every score
    tests = {n: fixed_test(n, task) for n in EVAL_LENGTHS}  # Day 70's fixed test sets
    curve = []                                              # exact match per length, every `every` steps
    for step in range(1, steps + 1):                        # plain SGD on fresh batches
        n = rng.randint(lo, hi + 1)                           # a random length 3..20
        Xs, Xd, Y, _ = make_batch(rng, 32, n, task)           # one batch
        net.train_step(Xs, Xd, Y)                             # one update (Day 71's rule)
        if step % every == 0:                                 # progress check
            curve.append([exact_and_token(net, tests[n])[0] for n in EVAL_LENGTHS])  # exact match at each length
    return net, np.array(curve)                             # final model and its curve


RUNS = {}                                               # cache: each configuration trains once


def train_cached(*args):                                # train(), remembered by its arguments
    if args not in RUNS:                                  # first request: train it
        RUNS[args] = train(*args)                           # train and remember
    return RUNS[args]                                     # later requests reuse it


def alignment(net, task, n):                            # Day 72's measurements: where the decoder looks
    Xs, _, Y, _ = fixed_test(n, task)                     # fixed test sequences
    pred, W = net.generate(Xs, n + 1, return_weights=True)  # weights: (m, n + 1, n)
    W = W[:, :n]                                          # leave out the EOS step
    j = np.arange(n)                                      # output steps
    target = j if task == "copy" else n - 1 - j           # where each output's digit sits
    entropy = -(W * np.log(W + 1e-12)).sum(axis=2)        # spread of each row
    return (W.argmax(axis=2) == target).mean(), W[:, j, target].mean(), entropy.mean()  # argmax on target, weight on target, entropy


# =============================================================================
# DEMOS
# =============================================================================
def demo_one_step():
    print("=" * 70)
    print("1. THREE SCORES FOR THE SAME STEP -- by hand, then checked")
    print("=" * 70)
    print("Day 71's example: three encoder states h = (1, 0), (0, 1), (1, 1) and the decoder state "
          "s = (1, -1). Additive uses Day 71's hand-chosen weights; general uses Wg = identity, "
          "which makes it the dot score; then Wg = [[2, 0], [0, 0]], which ignores the second number.")
    Hs = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # three encoder states, one sequence
    s = np.array([[1.0, -1.0]])                             # the decoder's previous state
    keys = Hs @ np.array([[0.5, 0.0], [0.0, 0.5]])          # Day 71's keys (Ua = 0.5 I, ba = 0)
    e_add, _ = additive_scores(s, keys, np.eye(2), np.array([1.0, -1.0]))  # Day 71's additive score
    e_dot = dot_scores(s, Hs)                               # s . h_i
    e_gen_i, _ = general_scores(s, Hs, np.eye(2))           # (s I) . h_i
    e_gen_2, _ = general_scores(s, Hs, np.array([[2.0, 0.0], [0.0, 0.0]]))  # (s Wg) . h_i
    print(f"\n{'encoder state':<16}{'dot by hand':>13}{'dot':>8}{'general, I':>12}{'general, Wg':>13}{'additive':>10}")
    for i in range(3):                                      # each encoder state
        hand = s[0] @ Hs[0, i]                                # s . h_i by hand
        print(f"{str(Hs[0, i].tolist()):<16}{hand:>13.4f}{e_dot[0, i]:>8.4f}{e_gen_i[0, i]:>12.4f}{e_gen_2[0, i]:>13.4f}{e_add[0, i]:>10.4f}")
    print(f"\n{'weights':<16}" + "".join(f"{name:>22}" for name in ["dot", "general, Wg", "additive"]))
    w = [softmax(e)[0] for e in (e_dot, e_gen_2, e_add)]    # the softmax of each score vector
    for i in range(3):                                      # one row per encoder state
        print(f"{str(Hs[0, i].tolist()):<16}" + "".join(f"{wk[i]:>22.4f}" for wk in w))
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("2. GRADIENT CHECK -- the dot and general scores, through the whole model")
    print("=" * 70)
    print("Day 71's check (hidden 4, 3 REVERSE sequences of 5 digits, every entry nudged by +-1e-5, "
          "Day 62's pass rule). The additive model is Day 71's own code, checked there.")
    Xs, Xd, Y, _ = make_batch(np.random.RandomState(4), 3, 5, "reverse")  # Day 71's tiny batch
    print(f"\n{'score':<9}{'parameter':>10}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>7}")
    verdict = {}                                           # score -> did every entry pass?
    for score in ["dot", "general"]:                      # the two new scores
        net = ScoredAttnSeq2Seq(score, hidden=4, seed=1, forget_bias=0.5)  # tiny model
        _, g = net.loss_and_grads(Xs, Xd, Y)                 # analytic gradients
        ok_all = True                                        # every parameter passes?
        for name, P in net.P.items():                        # every parameter
            num = np.zeros_like(P)                             # numeric gradient
            for pos in np.ndindex(P.shape):                    # every entry
                old = P[pos]                                     # remember it
                P[pos] = old + 1e-5; up = net.loss_and_grads(Xs, Xd, Y)[0]    # nudge up
                P[pos] = old - 1e-5; down = net.loss_and_grads(Xs, Xd, Y)[0]  # nudge down
                P[pos] = old                                     # restore
                num[pos] = (up - down) / 2e-5                    # measured slope
            diff = np.abs(num - g[name])                       # absolute disagreement
            rel = diff / (np.abs(num) + np.abs(g[name]) + 1e-12)  # relative disagreement
            ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(num)))  # Day 62's pass rule
            ok_all = ok_all and ok                             # track the verdict
            print(f"{score:<9}{name:>10}{P.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>7}")
        verdict[score] = ok_all                              # keep
    print(f"every entry passes: dot {verdict['dot']}, general {verdict['general']}")
    print("Actual output from this lesson's run.")


def demo_training():
    print("\n" + "=" * 70)
    print("3. THE COMPARISON -- COPY, 3-20 digits, three scores, three seeds")
    print("=" * 70)
    print(f"Day 71's Section 3 setup exactly (hidden 32, {STEPS} steps, plain SGD lr 0.5, clip 1.0, the "
          "same data stream and test sequences); only the score differs. Exact match every "
          f"{EVERY} steps, mean of three seeds, at 12 and 20 digits.")
    curves = {sc: np.mean([train_cached(sc, "copy", 32, s)[1] for s in SEEDS], axis=0) for sc in SCORES}  # mean curves
    j12, j20 = EVAL_LENGTHS.index(12), EVAL_LENGTHS.index(20)  # columns for 12 and 20 digits
    print(f"\n{'steps':>7}" + "".join(f"{sc + ' n=12':>15}{sc + ' n=20':>15}" for sc in SCORES))
    for i in range(len(curves[SCORES[0]])):                # one row per checkpoint
        print(f"{(i + 1) * EVERY:>7}" + "".join(f"{curves[sc][i, j12]:>15.3f}{curves[sc][i, j20]:>15.3f}" for sc in SCORES))
    print("Actual output from this lesson's run.")
    print(f"\nfinal exact match at 20 digits per seed, and the first checkpoint where every seed is at 1.000 on all lengths:")
    for sc in SCORES:                                      # each score
        runs = [train_cached(sc, "copy", 32, s)[1] for s in SEEDS]  # per-seed curves
        done = [i for i in range(len(runs[0])) if all(np.all(r[i] == 1.0) for r in runs)]  # checkpoints where all are perfect
        first = f"{(done[0] + 1) * EVERY} steps" if done else "not within the budget"  # the earliest such checkpoint, if any
        print(f"  {sc:<9}" + "  ".join(f"seed {s} {r[-1][j20]:.3f}" for s, r in zip(SEEDS, runs)) + f"   all perfect from: {first}")
    print("Actual output from this lesson's run.")
    return curves                                          # the plot reuses them


def demo_variants():
    print("\n" + "=" * 70)
    print("4. REVERSE, AND A SMALLER STATE -- final exact match, mean of three seeds")
    print("=" * 70)
    print("The same three scores on REVERSE at hidden 32, and on COPY at hidden 16 (the dot score's "
          "query and keys then have 16 numbers).")
    print(f"\n{'model':<26}" + "".join(f"{'n=' + str(n):>8}" for n in EVAL_LENGTHS) + f"{'weights':>9}")
    for task, hidden in [("reverse", 32), ("copy", 16)]:   # two variants
        for sc in SCORES:                                    # each score
            final = np.mean([train_cached(sc, task, hidden, s)[1][-1] for s in SEEDS], axis=0)  # final exact match
            n_w = sum(v.size for v in ScoredAttnSeq2Seq(sc, hidden=hidden).P.values())  # weight count
            print(f"{task + ', h ' + str(hidden) + ', ' + sc:<26}" + "".join(f"{e:>8.3f}" for e in final) + f"{n_w:>9}")
    print("Actual output from this lesson's run.")


def demo_alignment():
    print("\n" + "=" * 70)
    print("5. WHERE EACH SCORE LOOKS -- Day 72's measurements, COPY, 20 digits")
    print("=" * 70)
    print("The trained COPY models of Section 3: the share of output digits whose largest weight is on "
          "the target, the mean weight on the target, and the row entropy (log 20 = 3.00). Mean of "
          "three seeds.")
    print(f"\n{'score':<10}{'argmax on target':>18}{'weight on target':>18}{'entropy':>9}")
    res = {}                                               # score -> the three numbers
    for sc in SCORES:                                     # each score
        res[sc] = np.mean([alignment(train_cached(sc, "copy", 32, s)[0], "copy", 20) for s in SEEDS], axis=0)  # Day 72's three numbers, mean of seeds
        print(f"{sc:<10}{res[sc][0]:>18.3f}{res[sc][1]:>18.3f}{res[sc][2]:>9.3f}")
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_score_scale():
    print("\n" + "=" * 70)
    print("6. HOW LARGE ARE THE SCORES? -- before and after training")
    print("=" * 70)
    print("For the first decoder step of the 300 fixed 20-digit COPY test sequences: the spread "
          "(standard deviation over the 20 encoder positions) of the raw scores, and the largest "
          "attention weight. Untrained (seed 0) and trained (Section 3, seed 0). A large spread makes "
          "the softmax nearly one-hot: Day 75 is about this.")
    Xs, _, _, _ = fixed_test(20, "copy")                   # 20-digit test inputs
    print(f"\n{'score':<10}{'spread, untrained':>19}{'max weight, untrained':>23}{'spread, trained':>17}{'max weight, trained':>21}")
    for sc in SCORES:                                     # each score
        row = []                                             # four numbers
        for net in [ScoredAttnSeq2Seq(sc, hidden=32, seed=0), train_cached(sc, "copy", 32, 0)[0]]:  # untrained, trained
            Hs, _, _, keys, s, _ = net.encode(Xs)              # the decoder's first query is the encoder's final state
            e, _ = net.scores(s, Hs, keys)                     # first-step scores
            row += [e.std(axis=1).mean(), softmax(e).max(axis=1).mean()]  # spread and largest weight, mean over sequences
        print(f"{sc:<10}{row[0]:>19.3f}{row[1]:>23.3f}{row[2]:>17.3f}{row[3]:>21.3f}")
    print("Actual output from this lesson's run.")


def demo_cost():
    print("\n" + "=" * 70)
    print("7. WHAT EACH SCORE COSTS -- one training step, measured in this run")
    print("=" * 70)
    print("Hidden 32, a batch of 32 sequences of 20 digits, forward and backward; best of 5 timings of "
          "50 steps each, matrix library on one thread. Only ratios within this run are meaningful.")
    Xs, Xd, Y, _ = make_batch(np.random.RandomState(5), 32, 20, "copy")  # one fixed batch

    def best(fn):                                         # fastest of 5 timings of 50 calls (Day 71)
        times = []                                          # one timing per repeat
        for _ in range(5):                                  # 5 repeats
            start = time.perf_counter()                       # clock
            for _ in range(50):                               # 50 calls
                fn()                                            # the work being timed
            times.append(time.perf_counter() - start)         # elapsed
        return min(times) / 50                              # per call

    t = {sc: best(lambda net=ScoredAttnSeq2Seq(sc, hidden=32): net.loss_and_grads(Xs, Xd, Y)) for sc in SCORES}  # each score
    print(f"\n{'score':<10}{'ms per step':>13}{'vs additive':>13}{'score weights':>15}")
    for sc in SCORES:                                     # each score
        n_score = sum(ScoredAttnSeq2Seq(sc, hidden=32).P[k].size for k in ["Wa", "Ua", "ba", "va", "Wg"]
                      if k in ScoredAttnSeq2Seq(sc, hidden=32).P)  # the score's own weights
        print(f"{sc:<10}{t[sc] * 1000:>13.1f}{t[sc] / t['additive']:>13.2f}{n_score:>15}")
    print("Actual output from this lesson's run.")


def plot_results(curves, align):                        # two panels: learning speed | attention sharpness
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))    # canvas
    steps = np.arange(1, len(curves[SCORES[0]]) + 1) * EVERY  # checkpoint steps
    for sc, color in zip(SCORES, COLORS):                   # panel 1: exact match at 20 digits
        a1.plot(steps, curves[sc][:, EVAL_LENGTHS.index(20)], marker="o", markersize=4, color=color, label=sc)
    a1.set_xlabel("training steps"); a1.set_ylabel("exact match, 20 digits (mean of 3 seeds)"); a1.set_ylim(-0.02, 1.02)
    a1.set_title("How fast each score learns COPY", fontsize=10); a1.legend(fontsize=8, loc="lower right"); a1.grid(alpha=0.3)
    x = np.arange(len(SCORES))                              # panel 2: entropy and weight on target
    a2.bar(x - 0.2, [align[sc][1] for sc in SCORES], width=0.4, color=COLORS, alpha=0.9, label="weight on target")
    a2.bar(x + 0.2, [align[sc][2] / np.log(20) for sc in SCORES], width=0.4, color=COLORS, alpha=0.4, hatch="//",
           label="entropy / log 20")
    a2.set_xticks(x); a2.set_xticklabels(SCORES); a2.set_ylim(0, 1.15)
    a2.set_title("How sharply each looks (COPY, 20 digits)", fontsize=10); a2.legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig("score_functions_compared.png", dpi=110); plt.close()  # save and free
    print("\nsaved score_functions_compared.png")


def draw_diagram():                                     # the three score functions, side by side
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")      # plain coordinate grid

    def box(x, y, w, h, text, color, fs=8.5, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333"):                    # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    ax.text(6.5, 4.15, "Day 73: three ways to score a decoder state s against an encoder state h", ha="center",
            fontsize=12, fontweight="bold")
    for x, title, steps_, formula, wts in [
            (2.2, "additive (Day 71)", ["s Wa  +  h Ua + ba", "tanh", "va . (...)"], "e = va . tanh(s Wa + h Ua + ba)", "Wa, Ua, ba, va"),
            (6.5, "dot (Luong)", ["s . h"], "e = s . h", "no weights"),
            (10.8, "general (Luong)", ["q = s Wg", "q . h"], "e = (s Wg) . h", "Wg")]:
        box(x, 3.55, 3.6, 0.45, title, "#F4EEFC", fs=9.5, bold=True)   # the score's name
        ys = np.linspace(2.8, 1.4, len(steps_)) if len(steps_) > 1 else [2.1]  # one box per operation
        prev = 3.32                                                      # bottom of the name box
        for y, op in zip(ys, steps_):                     # one box per operation
            box(x, y, 2.6, 0.42, op, "#FAEEDA")                          # one operation
            arrow(x, prev, x, y + 0.23)                                  # from the previous box
            prev = y - 0.21                               # the bottom of this box, for the next arrow
        box(x, 0.75, 3.6, 0.42, formula, "#E6F1F8", fs=8.5)              # the formula
        arrow(x, prev, x, 0.98)
        ax.text(x, 0.28, "weights: " + wts, ha="center", fontsize=8.5, style="italic")
    plt.savefig("score_functions_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved score_functions_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("So far the decoder attends to the ENCODER. Day 74 lets every position of one sequence attend "
          "to every other position of the same sequence: self-attention, the building block of the "
          "Transformer.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_one_step()                                      # 1. three scores by hand
        demo_gradient_check()                                # 2. gradients
        curves = demo_training()                             # 3. the comparison
        demo_variants()                                      # 4. reverse and a smaller state
        align = demo_alignment()                             # 5. where each score looks
        demo_score_scale()                                   # 6. how large the scores are
        demo_cost()                                          # 7. cost
        plot_results(curves, align)                          # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 8. Day 74


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
