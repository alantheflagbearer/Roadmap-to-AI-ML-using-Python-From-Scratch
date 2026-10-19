"""
Day 71: Additive (Bahdanau) Attention, From Scratch

Day 70 measured the bottleneck: when the decoder may only look at the
encoder's final state, copying 20 digits fails, and it fails because of
how MUCH must fit in that state, not how FAR back the digits are.

Attention (Bahdanau, Cho and Bengio, 2015) keeps every encoder state and
lets the decoder look back at all of them at every output step:

    e_i     = va . tanh(s_prev Wa + h_i Ua + ba)     a score for encoder state i
    alpha   = softmax(e)                             weights over the input steps
    context = sum_i alpha_i h_i                      Day 70 exercise's soft read
    s, c    = LSTM([y_prev ; context], s_prev, c_prev)
    output  = softmax([s ; context] Why + by)

The score is ADDITIVE: the decoder's state and each encoder state are
projected, added, squashed and scored by one vector.

What is built and measured, reported as it came out:

  1. One attention step by hand, checked against the functions.
  2. A gradient check through attention, decoder and encoder.
  3. Day 70's hardest test (COPY, 3-20 digits) with attention.
  4. A smaller state, REVERSE, and a decoder that starts from zeros.
  5. Sequences longer than any seen in training.
  6. What attention costs per training step.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import time                                            # time: perf_counter, to time one training step in Section 6

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

EOS, GO = 0, 9                                          # token 0 = end of sequence, token 9 = "start writing"; digits are 1..8 (Day 69)
V = 10                                                  # vocabulary: EOS, eight digits, GO (Day 69)
SEEDS = [0, 1, 2]                                       # three seeds for every trained comparison
EVAL_LENGTHS = (4, 8, 12, 16, 20)                       # Day 70's test lengths
STEPS = 20000                                           # training steps per run (half of Day 70's main budget)
EVERY = 2500                                            # test accuracy is measured every EVERY steps

# =============================================================================
# REUSED -- verbatim from Day 69 (its LSTM, output layer and model, for the comparison in Section 6)
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


def outputs_loss(Hs, Y, Why, by):                       # day67's per-step output layer, as its own function
    m, T = Y.shape                                        # batch, steps
    loss, dHs = 0.0, np.zeros_like(Hs)                    # total loss; gradient at every step's hidden state
    dWhy, dby = np.zeros_like(Why), np.zeros_like(by)     # output-layer gradients
    for t in range(T):                                    # every step has its own output
        p = softmax(Hs[:, t] @ Why + by)                    # this step's prediction
        loss -= np.log(p[np.arange(m), Y[:, t]]).sum() / (m * T)  # cross-entropy, averaged over batch and steps
        dlog = p.copy(); dlog[np.arange(m), Y[:, t]] -= 1; dlog /= m * T  # softmax + cross-entropy gradient
        dWhy += Hs[:, t].T @ dlog; dby += dlog.sum(axis=0)  # accumulate the output layer's gradients
        dHs[:, t] = dlog @ Why.T                            # what this step's output sends back to h_t
    return loss, dHs, dWhy, dby                           # everything the backward pass needs


class Seq2Seq:                                          # encoder LSTM -> final (h, c) -> decoder LSTM -> a token per step
    def __init__(self, hidden=32, bidirectional=False, seed=0, forget_bias=1.0, lr=0.5, clip=1.0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.H, self.bi, self.lr, self.clip = hidden, bidirectional, lr, clip  # sizes and training settings
        He = hidden // 2 if bidirectional else hidden      # each encoder direction gets half when there are two
        self.P = {"Wef": rng.randn(V + He, 4 * He) / np.sqrt(V + He), "bef": np.zeros(4 * He)}  # encoder, left to right
        if bidirectional:                                  # day68: a second encoder LSTM, reading right to left
            self.P.update(Web=rng.randn(V + He, 4 * He) / np.sqrt(V + He), beb=np.zeros(4 * He))
        self.P.update(Wd=rng.randn(V + hidden, 4 * hidden) / np.sqrt(V + hidden), bd=np.zeros(4 * hidden),  # decoder
                      Why=rng.randn(hidden, V) / np.sqrt(hidden), by=np.zeros(V))  # decoder's output layer (day67)
        for name, size in [("bef", He), ("beb", He), ("bd", hidden)]:  # forget biases
            if name in self.P:
                self.P[name][:size] = forget_bias

    def encode(self, Xs):                                  # read the source; return the final (h, c) and the caches
        m = Xs.shape[0]                                      # batch size
        He = self.P["Wef"].shape[1] // 4                     # hidden size of one encoder direction
        z = np.zeros((m, He))                                # the encoder starts from zero states
        Hf, Cf, gf = lstm_forward(Xs, z, z, self.P["Wef"], self.P["bef"])  # part 2: left to right
        caches = [("f", Xs, Hf, Cf, gf)]                     # for the backward pass
        h, c = Hf[:, -1], Cf[:, -1]                          # the final states: everything the decoder will get
        if self.bi:                                          # day68: the same source read backwards
            Xr = Xs[:, ::-1]                                   # reversed (every sequence in a batch has the same length)
            Hr, Cr, gr = lstm_forward(Xr, z, z, self.P["Web"], self.P["beb"])
            caches.append(("b", Xr, Hr, Cr, gr))
            h = np.concatenate([h, Hr[:, -1]], axis=1)          # join the two final hidden states ...
            c = np.concatenate([c, Cr[:, -1]], axis=1)          # ... and the two final cell states
        return h, c, caches                                  # the "thought vector" (h, c), and the caches

    def loss_and_grads(self, Xs, Xd, Y):                   # one forward and backward pass through encoder AND decoder
        h0, c0, caches = self.encode(Xs)                     # the encoder's final states
        Hd, Cd, gd = lstm_forward(Xd, h0, c0, self.P["Wd"], self.P["bd"])  # the decoder starts from them, not from zeros
        loss, dHs, dWhy, dby = outputs_loss(Hd, Y, self.P["Why"], self.P["by"])  # part 4: a loss at every output step
        g = {"Why": dWhy, "by": dby}                         # gradients so far
        z = np.zeros_like(h0)                                # nothing arrives after the decoder's last step
        g["Wd"], g["bd"], dh0, dc0 = lstm_backward(dHs, z, z, Xd, h0, c0, self.P["Wd"], Hd, Cd, gd)  # part 3 through the decoder
        He = self.P["Wef"].shape[1] // 4                     # hidden size of one encoder direction
        for j, (d, X, Hs, Cs, gates) in enumerate(caches):   # the decoder's initial-state gradient flows into the encoder
            part = slice(j * He, (j + 1) * He)                 # this direction's share of h0 and c0
            ze = np.zeros((X.shape[0], He))                    # the encoder started from zeros
            g["We" + d], g["be" + d], _, _ = lstm_backward(np.zeros_like(Hs), dh0[:, part], dc0[:, part],  # only the LAST state
                                                        X, ze, ze, self.P["We" + d], Hs, Cs, gates)       # received gradient
        return loss, g                                       # loss, and a gradient for every parameter

    def train_step(self, Xs, Xd, Y):                       # one SGD step with day60's clipping
        loss, g = self.loss_and_grads(Xs, Xd, Y)             # forward and backward
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))  # global norm
        scale = min(1.0, self.clip / norm)                   # shrink only if too long
        for k in self.P:                                     # plain SGD
            self.P[k] -= self.lr * scale * g[k]
        return loss                                          # for the record

    def generate(self, Xs, n_steps, zero_state=False):     # greedy decoding -- the decoder feeds itself its own output
        h, c, _ = self.encode(Xs)                            # the encoder's final states
        if zero_state:                                       # control: throw the encoder's information away
            h, c = np.zeros_like(h), np.zeros_like(c)
        token = np.full(len(Xs), GO)                         # every sequence starts with GO
        out = []                                             # the generated tokens
        for _ in range(n_steps):                             # one token per step
            h, c, _ = lstm_step_forward(np.eye(V)[token], h, c, self.P["Wd"], self.P["bd"])  # part 1: one decoder step
            token = (h @ self.P["Why"] + self.P["by"]).argmax(axis=1)  # the most probable next token ...
            out.append(token)                                  # ... is kept, and becomes the next input
        return np.stack(out, axis=1)                         # (m, n_steps)

    def teacher_forced(self, Xs, Xd):                      # predictions when every decoder input is the TRUE previous token
        h0, c0, _ = self.encode(Xs)                          # the encoder's final states
        Hd, _, _ = lstm_forward(Xd, h0, c0, self.P["Wd"], self.P["bd"])  # the training-time forward pass
        return (Hd @ self.P["Why"] + self.P["by"]).argmax(axis=2)  # most probable token at every step


# =============================================================================
# REUSED -- from day70_exercise.py: reading from a memory with soft weights
# =============================================================================
def masked_softmax(scores, mask):                      # softmax over the real slots only; padded slots get weight 0
    s = np.where(mask, scores, -np.inf)                   # a padded slot's score becomes minus infinity
    s = s - s.max(axis=-1, keepdims=True)                 # subtract the largest real score, for stability
    e = np.exp(s)                                         # exp(-inf) = 0, so padding contributes nothing
    return e / e.sum(axis=-1, keepdims=True)              # weights over the real slots, summing to 1


def soft_read(memory, scores, mask):                   # read a weighted average of the memory slots
    w = masked_softmax(scores, mask)                      # (m, n) weights
    return np.einsum("mn,mnd->md", w, memory), w          # (m, d) read vector, and the weights used


# =============================================================================
# 1. ONE LSTM STEP, BACKWARDS -- NEW (the body of Day 65's loop, for a single step)
# =============================================================================
def lstm_step_backward(dh, dc, x, h_prev, c_prev, c, gates, W):  # NEW: gradients of one LSTM step
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


# =============================================================================
# 2. ADDITIVE (BAHDANAU) ATTENTION -- NEW
# =============================================================================
def additive_scores(s_prev, keys, Wa, va):             # NEW: e_i = va . tanh(s_prev Wa + keys_i), one score per encoder step
    t = np.tanh((s_prev @ Wa)[:, None, :] + keys)         # (m, n, A): the query added to every key, squashed
    return t @ va, t                                      # (m, n) scores, and t for the backward pass


def additive_scores_backward(de, t, s_prev, Wa, va):   # NEW: gradients of the scores w.r.t. s_prev, the keys, Wa and va
    dva = np.einsum("mn,mna->a", de, t)                   # e = t . va
    da = de[:, :, None] * va * (1 - t ** 2)               # back through tanh: (m, n, A)
    dq = da.sum(axis=1)                                   # the query was added to every key: sum over steps
    return dq @ Wa.T, da, s_prev.T @ dq, dva              # d s_prev, d keys, dWa, dva


class AttnSeq2Seq:                                      # NEW: Day 69's encoder-decoder, with the decoder reading every encoder state
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


def train(task, hidden, seed, steps=STEPS, lo=3, hi=20, every=EVERY, **kw):  # Day 70's training loop, for the attention model
    net = AttnSeq2Seq(hidden=hidden, seed=seed, **kw)       # fresh model
    rng = np.random.RandomState(seed + 100)                 # Day 69/70's data-stream seed rule
    tests = {n: fixed_test(n, task) for n in EVAL_LENGTHS}  # fixed test sets
    curve = []                                              # exact match per length, every `every` steps
    for step in range(1, steps + 1):                        # plain SGD on fresh batches
        n = rng.randint(lo, hi + 1)                           # a random length
        Xs, Xd, Y, _ = make_batch(rng, 32, n, task)           # one batch
        net.train_step(Xs, Xd, Y)                             # one update
        if step % every == 0:                             # progress check
            curve.append([exact_and_token(net, tests[n])[0] for n in EVAL_LENGTHS])  # exact match at each length
    return net, np.array(curve)                           # final model and its curve


RUNS = {}                                               # cache: each configuration trains once


def train_cached(*args, **kwargs):                      # train(), remembered by its arguments
    key = (args, tuple(sorted(kwargs.items())))           # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = train(*args, **kwargs)                # train and remember
    return RUNS[key]                                      # later requests reuse the result


# =============================================================================
# DEMOS
# =============================================================================
def demo_one_step():
    print("=" * 70)
    print("1. ONE ATTENTION STEP -- by hand, then checked")
    print("=" * 70)
    print("Three encoder states of 2 numbers, a decoder state of 2 numbers, attention size 2, "
          "hand-chosen weights. Scores, weights and the context are computed with plain "
          "arithmetic and compared with additive_scores and soft_read.")
    Hs = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # three encoder states, one sequence
    s = np.array([[1.0, -1.0]])                             # the decoder's previous state
    Wa = np.array([[1.0, 0.0], [0.0, 1.0]]); Ua = np.array([[0.5, 0.0], [0.0, 0.5]])  # query and key weights
    ba = np.zeros(2); va = np.array([1.0, -1.0])            # key bias and the scoring vector
    keys = Hs @ Ua + ba                                     # keys
    e, _ = additive_scores(s, keys, Wa, va)                 # the function
    ctx, alpha = soft_read(Hs, e, np.ones(e.shape, bool))   # day70's read
    print(f"\n{'encoder state':<16}{'key':<14}{'tanh(query + key)':<22}{'score by hand':>15}{'function':>10}{'weight':>9}")
    hand = []                                               # scores computed by hand
    for i in range(3):                                    # each encoder state
        q = s[0] @ Wa                                         # the query: (1, -1)
        tt = np.tanh(q + keys[0, i])                          # tanh(query + key)
        hand.append(tt @ va)                                  # va . tanh(...)
        print(f"{str(Hs[0, i].tolist()):<16}{str(keys[0, i].tolist()):<14}{str(tt.round(4).tolist()):<22}"
              f"{hand[-1]:>15.4f}{e[0, i]:>10.4f}{alpha[0, i]:>9.4f}")
    w_hand = np.exp(hand) / np.exp(hand).sum()              # softmax by hand
    print(f"weights by hand {w_hand.round(4).tolist()}; context = sum of weight x state = {ctx[0].round(4).tolist()} "
          f"(by hand {(w_hand[:, None] * Hs[0]).sum(axis=0).round(4).tolist()})")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("2. GRADIENT CHECK -- through attention, decoder and encoder")
    print("=" * 70)
    print("A tiny model (hidden 4, attention size 3), 3 REVERSE sequences of 5 digits, every entry of "
          "every parameter nudged by +-1e-5; Day 62's pass rule. Twice: with the decoder starting "
          "from the encoder's final state (Day 69), and starting from zeros.")
    Xs, Xd, Y, _ = make_batch(np.random.RandomState(4), 3, 5, "reverse")  # tiny batch (Day 69's)
    print(f"\n{'decoder start':<15}{'parameter':>10}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>7}")
    verdict = {}                                          # decoder start -> did every entry pass?
    for init in [True, False]:                             # both starts
        net = AttnSeq2Seq(hidden=4, attn=3, seed=1, forget_bias=0.5, init_from_encoder=init)  # tiny model
        _, g = net.loss_and_grads(Xs, Xd, Y)                 # analytic gradients
        ok_all = True                                     # every parameter passes?
        for name, P in net.P.items():                        # every parameter
            num = np.zeros_like(P)                        # numeric gradient
            for pos in np.ndindex(P.shape):               # every entry
                old = P[pos]                              # remember it
                P[pos] = old + 1e-5; up = net.loss_and_grads(Xs, Xd, Y)[0]  # nudge up
                P[pos] = old - 1e-5; down = net.loss_and_grads(Xs, Xd, Y)[0]  # nudge down
                P[pos] = old                              # restore
                num[pos] = (up - down) / 2e-5             # measured slope
            diff = np.abs(num - g[name])                  # absolute disagreement
            rel = diff / (np.abs(num) + np.abs(g[name]) + 1e-12)  # relative disagreement
            ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(num)))  # Day 62's pass rule
            ok_all = ok_all and ok                        # track the verdict
            print(f"{'encoder end' if init else 'zeros':<15}{name:>10}{P.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>7}")
        verdict[init] = ok_all                            # keep
    print(f"every entry passes: decoder starts from the encoder {verdict[True]}, from zeros {verdict[False]}")
    print("Actual output from this lesson's run.")


def demo_training():
    print("\n" + "=" * 70)
    print("3. DAY 70'S HARDEST TEST, WITH ATTENTION -- COPY, 3-20 digits")
    print("=" * 70)
    print(f"Exactly Day 70's Section 2 setup (lengths 3-20, plain SGD lr 0.5, clip 1.0, batches of 32, "
          f"seeds 0, 1, 2, the same 300 fixed test sequences per length), hidden 32, attention size 32, "
          f"{STEPS} steps; exact match every {EVERY} steps.")
    print(f"\n{'steps':>7}" + "".join(f"{'n=' + str(n):>8}" for n in EVAL_LENGTHS) + "   (mean of 3 seeds)")
    curves = np.mean([train_cached("copy", 32, s)[1] for s in SEEDS], axis=0)  # mean curve
    for i in range(len(curves)):                          # one row per checkpoint
        print(f"{(i + 1) * EVERY:>7}" + "".join(f"{e:>8.3f}" for e in curves[i]))
    print("Actual output from this lesson's run.")
    print(f"\nfinal exact match per seed:")
    for s in SEEDS:                                       # each seed
        print(f"  seed {s}: " + "  ".join(f"n={n} {e:.3f}" for n, e in zip(EVAL_LENGTHS, train_cached('copy', 32, s)[1][-1])))
    print("Actual output from this lesson's run.")
    return curves                                         # the plot reuses it


def demo_variants():
    print("\n" + "=" * 70)
    print("4. SMALLER STATE, REVERSE, AND NO HAND-OVER -- final exact match, mean of 3 seeds")
    print("=" * 70)
    print("(a) hidden 16 (a quarter of Day 70's largest state); (b) REVERSE; (c) the decoder starts "
          "from zeros instead of the encoder's final state, so attention is its ONLY view of the "
          "input. Everything else as Section 3.")
    rows = [("copy, hidden 32 (Section 3)", ("copy", 32), {}), ("copy, hidden 16", ("copy", 16), {}),
            ("reverse, hidden 32", ("reverse", 32), {}), ("copy, hidden 32, start from zeros", ("copy", 32), dict(init_from_encoder=False))]
    print(f"\n{'model':<36}" + "".join(f"{'n=' + str(n):>8}" for n in EVAL_LENGTHS) + f"{'weights':>9}")
    for label, (task, hidden), kw in rows:                # each variant
        final = np.mean([train_cached(task, hidden, s, **kw)[1][-1] for s in SEEDS], axis=0)  # final exact match, mean of seeds
        n_w = sum(v.size for v in AttnSeq2Seq(hidden=hidden, **kw).P.values())  # its weight count
        print(f"{label:<36}" + "".join(f"{e:>8.3f}" for e in final) + f"{n_w:>9}")
    print("Actual output from this lesson's run.")


def demo_longer():
    print("\n" + "=" * 70)
    print("5. LONGER THAN ANY TRAINING SEQUENCE -- 25, 30 and 40 digits")
    print("=" * 70)
    print("The Section 3 models (trained on 3-20 digits) on 300 fixed sequences of each new length.")
    print(f"\n{'length':>8}{'exact match':>13}{'token acc':>11}{'per seed (exact)':>26}")
    for n in [20, 25, 30, 40]:                            # the last training length, then three longer ones
        scores = [exact_and_token(train_cached("copy", 32, s)[0], fixed_test(n, "copy")) for s in SEEDS]  # each seed's model
        e, t = np.mean(scores, axis=0)                    # mean over seeds
        print(f"{n:>8}{e:>13.3f}{t:>11.3f}{'  '.join(f'{x:.3f}' for x, _ in scores):>26}")
    print("Actual output from this lesson's run.")


def demo_cost():
    print("\n" + "=" * 70)
    print("6. WHAT ATTENTION COSTS -- one training step, measured in this run")
    print("=" * 70)
    print("Hidden 32, a batch of 32 sequences of 20 digits; best of 5 timings of 50 steps each, matrix "
          "library on one thread. Only the ratio is meaningful: this machine's speed varies between "
          "runs (Day 69).")
    Xs, Xd, Y, _ = make_batch(np.random.RandomState(5), 32, 20, "copy")  # one fixed batch
    plain, attn = Seq2Seq(hidden=32, seed=0), AttnSeq2Seq(hidden=32, seed=0)  # Day 69's model and today's

    def best(fn):                                         # fastest of 5 timings of 50 calls
        times = []                                        # one timing per repeat
        for _ in range(5):                                # 5 repeats
            start = time.perf_counter()                   # clock
            for _ in range(50):                           # 50 calls
                fn()                                      # the work being timed
            times.append(time.perf_counter() - start)     # elapsed
        return min(times) / 50                            # the least disturbed repeat, per call

    t_plain = best(lambda: plain.loss_and_grads(Xs, Xd, Y))  # Day 69: forward + backward
    t_attn = best(lambda: attn.loss_and_grads(Xs, Xd, Y))    # today
    n_plain, n_attn = sum(v.size for v in plain.P.values()), sum(v.size for v in attn.P.values())  # weight counts
    print(f"\nDay 69 encoder-decoder: {t_plain * 1000:.1f} ms per step, {n_plain} weights")
    print(f"with additive attention: {t_attn * 1000:.1f} ms per step, {n_attn} weights; time ratio {t_attn / t_plain:.2f}")
    print("Actual output from this lesson's run.")


def plot_results(curves):                               # learning curves with attention, per length
    fig, ax = plt.subplots(figsize=(9, 4.4))
    steps = np.arange(1, len(curves) + 1) * EVERY
    for j, (n, color) in enumerate(zip(EVAL_LENGTHS, ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"])):
        ax.plot(steps, curves[:, j], marker="o", markersize=3, color=color, label=f"n={n}")
    ax.set_xlabel("training steps"); ax.set_ylabel("exact match (test, mean of 3 seeds)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("COPY with additive attention, hidden 32: every length is learned", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("attention_training_curves.png", dpi=110); plt.close()
    print("\nsaved attention_training_curves.png")


def draw_diagram():                                     # one decoder step with additive attention
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333", lw=1.2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=lw))

    ax.text(6.5, 5.35, "Day 71: one decoder step with additive (Bahdanau) attention", ha="center", fontsize=12, fontweight="bold")
    xs = [0.8, 1.9, 3.0, 4.1]                                   # encoder states h_1..h_4
    for j, x in enumerate(xs):
        box(x, 0.7, 0.85, 0.5, f"h{j + 1}", "#DCEBF7")             # encoder state
        box(x, 2.0, 0.95, 0.5, f"e{j + 1}", "#F4EEFC", fs=8)       # its score
        arrow(x, 0.97, x, 1.73)                                    # state -> score
        arrow(x, 2.27, 5.25, 3.25 - 0.06 * j, "#7A4EC9", 1.0)      # score -> softmax
    ax.text(2.45, 0.08, "every encoder state is kept (the memory)", ha="center", fontsize=8.5, style="italic")
    ax.text(2.45, 4.3, "score: e_i = va . tanh(s Wa + h_i Ua + ba)", ha="center", fontsize=8.5, family="monospace")
    box(6.2, 3.2, 1.8, 0.55, "softmax -> weights", "#F4EEFC", fs=8.5)  # scores into weights
    box(8.6, 3.2, 2.3, 0.55, "context = sum a_i h_i", "#FAEEDA", fs=8.5, bold=True)  # the weighted read
    arrow(7.13, 3.2, 7.42, 3.2, "#C97A16")                         # weights -> context
    box(11.1, 3.2, 1.5, 0.6, "decoder LSTM", "#E5F2E1", fs=8.5)    # the decoder step
    arrow(9.78, 3.2, 10.32, 3.2, "#C97A16")                        # context -> decoder
    box(11.1, 1.6, 1.8, 0.5, "previous token y", "#F2F2F2", fs=8.5)  # the decoder's other input
    arrow(11.1, 1.86, 11.1, 2.88)                                  # previous token -> decoder
    box(8.6, 1.6, 1.9, 0.5, "previous state s", "#E5F2E1", fs=8.5)  # the query
    arrow(7.62, 1.6, 4.65, 1.95, "#2F7D32", 1.0)                   # s feeds every score (drawn to the score row)
    arrow(9.58, 1.6, 10.4, 2.92, "#2F7D32", 1.0)                   # s -> decoder LSTM
    box(11.1, 4.6, 2.8, 0.5, "next token from [s ; context]", "#FDE7D3", fs=8)  # the output
    arrow(11.1, 3.52, 11.1, 4.33)                                  # decoder -> output
    arrow(8.6, 3.49, 9.68, 4.42, "#C97A16", 1.0)                   # context also feeds the output
    plt.savefig("additive_attention_diagram.png", dpi=120, bbox_inches="tight"); plt.close()
    print("saved additive_attention_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Every decoder step produced a set of attention weights: where in the input it looked. Day 72 "
          "draws them as alignment heatmaps and checks whether the model looks where a person would.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_one_step()                                      # 1. one attention step by hand
        demo_gradient_check()                                # 2. gradients
        curves = demo_training()                             # 3. Day 70's hardest test
        demo_variants()                                      # 4. smaller state, reverse, no hand-over
        demo_longer()                                        # 5. longer than training
        demo_cost()                                          # 6. cost
        plot_results(curves)                                 # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 7. Day 72


if __name__ == "__main__":
    main()
