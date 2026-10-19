"""
Day 72: Looking Inside Attention -- Alignment Heatmaps on the Toy Task

Day 71's attention model copies and reverses 20-digit sequences that the
plain encoder-decoder could not. At every output step it produced a set
of weights over the input positions: where it looked. Today reads those
weights.

What is measured, reported as it came out:

  1. One 12-digit COPY and REVERSE alignment, printed as numbers.
  2. Alignment over the whole test set: how often the largest weight sits
     on the input digit being written, and where the rest of the weight goes.
  3. Why the leftover weight lands on the NEXT encoder state: a probe of
     which encoder states still hold a given digit.
  4. When the alignment appears during training, next to accuracy.
  5. Where attention breaks on sequences longer than any seen in training.

The models are Day 71's, retrained here with the same seeds and the same
data stream, so they are the same models (Section 1 checks this against
Day 71's numbers in the guide).

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import copy                                            # copy: deepcopy, to keep snapshots of the weights during training

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

EOS, GO = 0, 9                                          # token 0 = end of sequence, token 9 = "start writing"; digits are 1..8 (Day 69)
V = 10                                                  # vocabulary: EOS, eight digits, GO (Day 69)
SEEDS = [0, 1, 2]                                       # the three seeds of Day 71
STEPS = 20000                                           # Day 71's training budget
SNAPSHOTS = {250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000}  # training steps at which the weights are copied

# =============================================================================
# REUSED -- verbatim from Day 71 (its LSTM, attention model and data)
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
# NEW TODAY -- reading the attention weights
# =============================================================================
def train_with_snapshots(task, hidden, seed, steps=STEPS, snap_at=SNAPSHOTS):  # NEW: Day 71's loop, keeping copies of the model
    net = AttnSeq2Seq(hidden=hidden, seed=seed)             # fresh model (Day 71)
    rng = np.random.RandomState(seed + 100)                 # Day 69-71's data-stream seed rule: the same batches as Day 71
    snaps = {0: copy.deepcopy(net.P)}                       # the weights before any training
    for step in range(1, steps + 1):                        # plain SGD on fresh batches
        n = rng.randint(3, 21)                                # a random length 3..20
        Xs, Xd, Y, _ = make_batch(rng, 32, n, task)           # one batch
        net.train_step(Xs, Xd, Y)                             # one update
        if step in snap_at:                                   # a snapshot step
            snaps[step] = copy.deepcopy(net.P)                  # keep a copy of every weight
    return net, snaps                                       # the final model and its snapshots


MODELS = {}                                             # cache: (task, seed) -> (model, snapshots)


def model(task, seed):                                  # the trained model for a task and seed, trained once
    if (task, seed) not in MODELS:                        # first request
        MODELS[(task, seed)] = train_with_snapshots(task, 32, seed)  # train it
    return MODELS[(task, seed)]                           # later requests reuse it


def with_weights(P, hidden=32):                          # a model object carrying a given set of weights
    net = AttnSeq2Seq(hidden=hidden)                       # the right shapes
    net.P = copy.deepcopy(P)                               # the snapshot's values
    return net


def target_position(task, n):                           # NEW: for output step j, the input position holding the digit to write
    j = np.arange(n)                                      # output steps 0..n-1 (the EOS step is left out)
    return j if task == "copy" else n - 1 - j             # COPY: the same position; REVERSE: mirrored


def alignment_stats(net, task, n, m=300):                # NEW: where the decoder looks, compared with where the digit is
    Xs, _, Y, src = fixed_test(n, task, m)                # Day 70's fixed test sequences
    pred, W = net.generate(Xs, n + 1, return_weights=True)  # W: (m, n + 1, n) -- one row of weights per output step
    W = W[:, :n]                                          # leave out the EOS step
    target = target_position(task, n)                     # where each output's digit sits in the input
    argmax = W.argmax(axis=2)                             # where each output step looked most
    offset = argmax - target[None, :]                     # 0 = exactly on the digit; +1 = one step later; ...
    on_target = W[:, np.arange(n), target]                # the weight placed on the right position
    entropy = -(W * np.log(W + 1e-12)).sum(axis=2)        # how spread out each row is (0 = all on one position)
    correct = pred[:, :n] == Y[:, :n]                     # was the digit written correctly?
    return dict(offset=offset, on_target=on_target, entropy=entropy, correct=correct, W=W, src=src, pred=pred)


def fit_probe(F, y, iters=400, lr=0.5):                 # Day 70's linear probe, verbatim: a softmax classifier by gradient descent
    mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-8         # standardise each feature
    Z = (F - mu) / sd                                     # standardised features
    W = np.zeros((F.shape[1], V)); b = np.zeros(V)        # weights and biases, start at zero
    for _ in range(iters):                                # full-batch gradient descent
        p = softmax(Z @ W + b)                              # predicted digit probabilities
        p[np.arange(len(y)), y] -= 1                        # softmax + cross-entropy gradient
        W -= lr * Z.T @ p / len(y); b -= lr * p.mean(axis=0)  # update
    return lambda G: ((G - mu) / sd @ W + b).argmax(axis=1)  # a function that predicts digits from new features


# =============================================================================
# DEMOS
# =============================================================================
def demo_one_example():
    print("=" * 70)
    print("1. ONE ALIGNMENT, AS NUMBERS -- a 12-digit COPY and REVERSE")
    print("=" * 70)
    print(f"Day 71's model (hidden 32, attention 32, COPY or REVERSE, seed 0, {STEPS} steps, retrained "
          "here with the same seeds, so the same weights). Each row is one output step; each column "
          "an input position; the numbers are the attention weights (x 100, rounded). The first test "
          "sequence of length 12.")
    for task in ["copy", "reverse"]:                       # both tasks
        st = alignment_stats(model(task, 0)[0], task, 12)    # weights on the fixed 12-digit sequences
        W, src, pred = st["W"][0], st["src"][0], st["pred"][0]  # the first sequence
        print(f"\n{task.upper()}: input {src.tolist()} -> written {pred[:12].tolist()}")
        print("   in:" + "".join(f"{d:>4}" for d in src))  # the input digits as column headers
        for j in range(12):                                  # one output step per row
            print(f"out {pred[j]:>1}" + "".join(f"{int(round(100 * w)):>4}" if w >= 0.005 else "   ." for w in W[j]))
    print("Actual output from this lesson's run.")


def demo_statistics():
    print("\n" + "=" * 70)
    print("2. ALIGNMENT OVER THE WHOLE TEST SET -- where the decoder looks")
    print("=" * 70)
    print("For every output digit: the input position holding it (the TARGET), where the attention "
          "weight is largest, and how the weight is spread. 300 fixed test sequences per length, "
          "three seeds. 'next' is the encoder state one step AFTER the target (it was computed after "
          "reading the digit); 'previous' is the one before. Entropy is in nats; 0 = all weight on one "
          f"position, log(n) = spread evenly (log 20 = {np.log(20):.2f}).")
    print(f"\n{'task':<9}{'n':>4}{'argmax on target':>18}{'weight: target':>16}{'next':>8}{'previous':>10}{'entropy':>9}{'exact':>8}")
    for task in ["copy", "reverse"]:                       # both tasks
        for n in [8, 12, 20]:                                # three lengths
            rows = []                                          # one row of numbers per seed
            for s in SEEDS:
                st = alignment_stats(model(task, s)[0], task, n)  # weights on the test sequences
                tgt = target_position(task, n)                   # the target per output step
                W, j = st["W"], np.arange(n)                     # weights; output steps
                nxt, prv = tgt + 1, tgt - 1                      # input positions after and before the target
                ok_n, ok_p = nxt < n, prv >= 0                   # those that exist
                rows.append([(st["offset"] == 0).mean(), st["on_target"].mean(),
                             W[:, j[ok_n], nxt[ok_n]].mean(), W[:, j[ok_p], prv[ok_p]].mean(),
                             st["entropy"].mean(), np.all(st["correct"], axis=1).mean()])  # six numbers per seed
            r = np.mean(rows, axis=0)                          # mean over seeds
            print(f"{task:<9}{n:>4}{r[0]:>18.3f}{r[1]:>16.3f}{r[2]:>8.3f}{r[3]:>10.3f}{r[4]:>9.3f}{r[5]:>8.3f}")
    print("Actual output from this lesson's run.")


def demo_why_next():
    print("\n" + "=" * 70)
    print("3. WHY THE NEXT STATE? -- which encoder states still hold a digit")
    print("=" * 70)
    print("A linear probe (Day 70's) reads the digit at input position p from ONE encoder state h_q, "
          "for q = p - 1 (before the digit was read), p, p + 1, p + 2 and p + 4. Day 71's COPY encoder "
          "(seed 0), 20-digit inputs, p = 8; 3,000 training and 1,000 test sequences.")
    net = model("copy", 0)[0]                              # the trained COPY model
    rng = np.random.RandomState(41)                        # probe data
    Xtr, _, _, src_tr = make_batch(rng, 3000, 20, "copy")  # probe training inputs
    Xte, _, _, src_te = make_batch(rng, 1000, 20, "copy")  # probe test inputs
    Htr, Hte = net.encode(Xtr)[0], net.encode(Xte)[0]      # every encoder state, (m, 20, 32)
    p = 7                                                  # input position 8 (index 7)
    print(f"\n{'probe reads x_8 from':<22}" + "".join(f"{'h_' + str(q + 1):>8}" for q in [p - 1, p, p + 1, p + 2, p + 4]))
    acc = [(fit_probe(Htr[:, q], src_tr[:, p])(Hte[:, q]) == src_te[:, p]).mean() for q in [p - 1, p, p + 1, p + 2, p + 4]]
    print(f"{'accuracy':<22}" + "".join(f"{a:>8.3f}" for a in acc))
    print(f"guessing: {1 / 8:.3f}")
    print("Actual output from this lesson's run.")


def demo_during_training():
    print("\n" + "=" * 70)
    print("4. WHEN DOES THE ALIGNMENT APPEAR? -- snapshots during training")
    print("=" * 70)
    print("The COPY and REVERSE models' saved weights at 0 to 20,000 steps: the share of output digits "
          "whose largest attention weight is on the target, and exact match, at n = 12. Mean of three "
          "seeds.")
    print(f"\n{'steps':>7}{'copy: argmax on target':>24}{'exact':>8}{'reverse: argmax on target':>27}{'exact':>8}")
    curve = {t: [] for t in ["copy", "reverse"]}           # task -> rows of (on target, exact)
    for step in sorted(SNAPSHOTS | {0}):                   # every snapshot, starting untrained
        vals = []                                            # four numbers for this step
        for task in ["copy", "reverse"]:
            per = []                                           # per seed
            for s in SEEDS:
                st = alignment_stats(with_weights(model(task, s)[1][step]), task, 12)  # the snapshot's weights
                per.append([(st["offset"] == 0).mean(), np.all(st["correct"], axis=1).mean()])
            curve[task].append(np.mean(per, axis=0)); vals += list(curve[task][-1])  # mean over seeds
        print(f"{step:>7}{vals[0]:>24.3f}{vals[1]:>8.3f}{vals[2]:>27.3f}{vals[3]:>8.3f}")
    print("Actual output from this lesson's run.")
    return {t: np.array(v) for t, v in curve.items()}      # the plot reuses it


def demo_beyond():
    print("\n" + "=" * 70)
    print("5. WHERE IT BREAKS -- attention on sequences longer than any in training")
    print("=" * 70)
    print("Day 71 found exact match falls on 30- and 40-digit COPY inputs. Here, for the COPY models "
          "(three seeds): by output position, the share of digits written correctly and the share "
          "whose largest weight is on the target; then, across all output digits, the weight on the "
          "target for digits written right and for digits written wrong.")
    for n in [30, 40]:                                     # two lengths beyond training
        sts = [alignment_stats(model("copy", s)[0], "copy", n) for s in SEEDS]  # three seeds
        ok = np.mean([st["correct"].mean(axis=0) for st in sts], axis=0)  # accuracy by output position
        on = np.mean([(st["offset"] == 0).mean(axis=0) for st in sts], axis=0)  # argmax-on-target by position
        cols = list(range(0, n, 5)) + [n - 1]                # every 5th position, and the last
        print(f"\nn = {n}")
        print(f"{'output position':<22}" + "".join(f"{c + 1:>6}" for c in cols))
        print(f"{'digit right':<22}" + "".join(f"{ok[c]:>6.2f}" for c in cols))
        print(f"{'argmax on target':<22}" + "".join(f"{on[c]:>6.2f}" for c in cols))
        right = np.concatenate([st["on_target"][st["correct"]] for st in sts])  # weight on target, digits written right
        wrong = np.concatenate([st["on_target"][~st["correct"]] for st in sts])  # ... digits written wrong
        print(f"weight on target: digits right {right.mean():.3f} ({len(right)}), digits wrong "
              f"{wrong.mean() if len(wrong) else float('nan'):.3f} ({len(wrong)})")
    print("Actual output from this lesson's run.")


def plot_heatmaps(curve):                               # heatmaps for COPY, REVERSE and a 40-digit COPY; alignment over training
    fig = plt.figure(figsize=(16, 4.6))                     # canvas
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.5, 1.3])  # four panels
    for k, (task, n, title) in enumerate([("copy", 12, "COPY, 12 digits"), ("reverse", 12, "REVERSE, 12 digits"),
                                          ("copy", 40, "COPY, 40 digits (never trained)")]):  # three heatmaps
        ax = fig.add_subplot(gs[0, k])                       # one panel
        st = alignment_stats(model(task, 0)[0], task, n)     # seed 0's weights
        ax.imshow(st["W"][0], cmap="Blues", vmin=0, vmax=1, aspect="auto")  # the first test sequence
        ax.set_xlabel("input position"); ax.set_ylabel("output step"); ax.set_title(title, fontsize=10)
        if n == 12:
            ax.set_xticks(range(n)); ax.set_xticklabels(st["src"][0], fontsize=7)  # input digits along the bottom
            ax.set_yticks(range(n)); ax.set_yticklabels(st["pred"][0][:n], fontsize=7)  # written digits on the side
    ax = fig.add_subplot(gs[0, 3])                           # alignment over training
    steps = sorted(SNAPSHOTS | {0})                          # snapshot steps
    for task, color in [("copy", "#1f77b4"), ("reverse", "#d62728")]:
        ax.plot(steps, curve[task][:, 0], color=color, marker="o", markersize=3, label=f"{task}: argmax on target")
        ax.plot(steps, curve[task][:, 1], color=color, marker="s", markersize=3, linestyle="--", label=f"{task}: exact match")
    ax.set_xscale("symlog", linthresh=500); ax.set_xlabel("training steps (log scale after 500)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Alignment appears with accuracy (n = 12)", fontsize=10); ax.legend(fontsize=7, loc="center right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("attention_heatmaps.png", dpi=110); plt.close()  # save and free
    print("\nsaved attention_heatmaps.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT COMES NEXT")
    print("=" * 70)
    print("Today's score was additive: the decoder's state and each encoder state are projected, added, "
          "squashed and scored. Day 73 builds the simpler dot-product (Luong) score and compares the two.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_one_example()                                   # 1. one alignment as numbers
        demo_statistics()                                    # 2. alignment over the test set
        demo_why_next()                                      # 3. why the next state
        curve = demo_during_training()                       # 4. during training
        demo_beyond()                                        # 5. beyond the training lengths
        plot_heatmaps(curve)                                 # the figure
        note_on_what_comes_next()                            # 6. Day 73


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
