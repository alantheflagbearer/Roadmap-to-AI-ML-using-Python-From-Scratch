"""
Day 69: The Week 10 Build -- a Sequence-to-Sequence Encoder-Decoder From
the Exercise Parts

Week 10's last day does two jobs at once, like Day 62 did for Week 9. The
syllabus topic is SEQUENCE-TO-SEQUENCE: one LSTM (the ENCODER) reads an
input sequence and hands its final state to a second LSTM (the DECODER),
which writes an output sequence one token at a time. And this is the
week's integration day: the model is assembled from the short exercise
scripts of Days 63-68.

The task is a toy one, on purpose: sequences of 3 to 8 digits (1-8), and
the output is either the same sequence (COPY) or the sequence backwards
(REVERSE). Every output must pass through the encoder's final state --
two vectors of 32 numbers -- so the task tests exactly what seq2seq
claims: that a sequence can be squeezed into a state and written out
again.

The parts, and where each one goes:

  Day 63 exercise  lstm_step_forward               -> one step of both LSTMs; the decoder's step at generation time
  Day 64 exercise  lstm_forward (the loop)          -> the encoder's and the decoder's forward pass
  Day 65 exercise  lstm_backward                    -> backpropagation through both
  Day 66 exercise  gru_step_forward                 -> not used: Days 66-67 gave no reason to prefer the GRU here
  Day 67 exercise  an output and a loss at EVERY step -> the decoder
  Day 68 exercise  a bidirectional layer            -> an optional encoder that reads both ways

Only ONE idea is new today: the decoder STARTS from the encoder's final
(h, c) instead of zeros, so the decoder's gradient at its initial state
flows back into the encoder. In code: lstm_backward now also returns the
gradients at h0 and c0, and the encoder's backward pass starts from them.

What the lesson measures, reported as it came out:
  - a gradient check through the whole encoder-decoder, both encoders
  - training on COPY and REVERSE, three seeds each, accuracy tracked
  - two controls: the encoder's state removed, and teacher forcing vs the
    decoder feeding itself
  - where the errors are: which output positions fail, for each task
  - the bidirectional encoder from Day 68's exercise on the same tasks

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import time                                            # time: perf_counter, to time each training run

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

EOS, GO = 0, 9                                          # token 0 = end of sequence, token 9 = "start writing"; digits are 1..8
V = 10                                                  # vocabulary: EOS, eight digits, GO
LENGTHS = range(3, 9)                                   # training and test sequences have 3 to 8 digits
SEEDS = [0, 1, 2]                                       # three seeds for every trained comparison
STEPS = 20000                                           # training steps per run
EVERY = 1000                                            # test accuracy is measured every EVERY steps

# =============================================================================
# PART 1 -- from day63_exercise.py, verbatim: one LSTM step
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


# =============================================================================
# PART 2 -- from day64/65_exercise.py, verbatim: the loop over time
# =============================================================================
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


# =============================================================================
# PART 3 -- from day65_exercise.py (+ day67's change): backpropagation
# =============================================================================
def lstm_backward(dHs, dh_last, dc_last, X, h0, c0, W, Hs, Cs, gates):  # CHANGED: see the three marked lines
    m, T, D = X.shape                                     # batch size, sequence length, input size
    dW, db = np.zeros_like(W), np.zeros(W.shape[1])       # accumulators, one contribution per step
    dh, dc = dh_last.copy(), dc_last.copy()               # CHANGED: a gradient may arrive at the last c too (not just h)
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
    return dW, db, dh, dc                                 # CHANGED: also the gradients at h0 and c0 -- the encoder needs them


# =============================================================================
# PART 4 -- from day67_exercise.py: a softmax and a loss at every step
# =============================================================================
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


# =============================================================================
# PART 5 -- from day68_exercise.py: reading a sequence in both directions
# =============================================================================
# (day68's bidirectional layer joined the two directions' states at EVERY step;
#  an encoder needs only the FINAL state of each direction, so only that is kept)


# =============================================================================
# THE ASSEMBLED MODEL -- encoder + decoder, plus the one new idea
# =============================================================================
class Seq2Seq:                                          # NEW: encoder LSTM -> final (h, c) -> decoder LSTM -> a token per step
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

    def encode(self, Xs):                                  # NEW: read the source; return the final (h, c) and the caches
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
        Hd, Cd, gd = lstm_forward(Xd, h0, c0, self.P["Wd"], self.P["bd"])  # NEW: the decoder starts from them, not from zeros
        loss, dHs, dWhy, dby = outputs_loss(Hd, Y, self.P["Why"], self.P["by"])  # part 4: a loss at every output step
        g = {"Why": dWhy, "by": dby}                         # gradients so far
        z = np.zeros_like(h0)                                # nothing arrives after the decoder's last step
        g["Wd"], g["bd"], dh0, dc0 = lstm_backward(dHs, z, z, Xd, h0, c0, self.P["Wd"], Hd, Cd, gd)  # part 3 through the decoder
        He = self.P["Wef"].shape[1] // 4                     # hidden size of one encoder direction
        for j, (d, X, Hs, Cs, gates) in enumerate(caches):   # NEW: the decoder's initial-state gradient flows into the encoder
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

    def generate(self, Xs, n_steps, zero_state=False):     # NEW: greedy decoding -- the decoder feeds itself its own output
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


def make_batch(rng, m, n, task):                        # m sequences of n digits, and what the decoder sees and must write
    src = rng.randint(1, 9, size=(m, n))                  # random digits 1..8
    tgt = src if task == "copy" else src[:, ::-1]         # COPY: the same digits; REVERSE: backwards
    dec_in = np.concatenate([np.full((m, 1), GO), tgt], axis=1)    # decoder input: GO, then the target shifted right
    dec_out = np.concatenate([tgt, np.full((m, 1), EOS)], axis=1)  # decoder output: the target, then EOS
    return np.eye(V)[src], np.eye(V)[dec_in], dec_out, src  # one-hot source, one-hot decoder input, targets, raw digits


def test_sets(task, n_per_length=300):                  # fixed test sequences: 300 of every length 3..8
    rng = np.random.RandomState(12345)                    # the same test sets for every model of this task
    return {n: make_batch(rng, n_per_length, n, task) for n in LENGTHS}  # length -> batch


def evaluate(net, tests, zero_state=False):             # exact-match and token accuracy, greedy decoding
    exact, tokens = [], []                                # one value per length
    for n, (Xs, _, Y, _) in tests.items():                # each length
        pred = net.generate(Xs, n + 1, zero_state)           # n digits + EOS
        exact.append(np.all(pred == Y, axis=1).mean())       # whole sequence right, including EOS
        tokens.append((pred == Y).mean())                    # share of output tokens right
    return np.mean(exact), np.mean(tokens), exact         # averages over lengths, and exact-match per length


def train(task, bidirectional, seed, steps=STEPS, every=EVERY):  # one training run, accuracy tracked
    net = Seq2Seq(bidirectional=bidirectional, seed=seed)  # fresh model
    rng = np.random.RandomState(seed + 100)                # the training data stream
    tests = test_sets(task)                                # fixed test sets
    curve, losses = [], []                                 # exact-match every `every` steps; recent losses
    start = time.perf_counter()                            # time the run
    for step in range(1, steps + 1):                       # plain SGD on fresh batches
        n = rng.randint(3, 9)                                # a random length 3..8 for this batch
        Xs, Xd, Y, _ = make_batch(rng, 32, n, task)          # 32 sequences of that length
        losses.append(net.train_step(Xs, Xd, Y))             # one step
        if step % every == 0:                                # progress check
            curve.append(evaluate(net, tests)[0])              # exact-match accuracy on the fixed test sets
    return net, np.array(curve), np.mean(losses[-every:]), time.perf_counter() - start  # model, curve, final loss, seconds


RUNS = {}                                               # cache: each (task, encoder, seed) trains once


def train_cached(task, bidirectional, seed):            # train(), remembered
    key = (task, bidirectional, seed)                     # what defines the run
    if key not in RUNS:                                   # first request
        RUNS[key] = train(task, bidirectional, seed)        # train it
    return RUNS[key]                                      # later requests reuse it


# =============================================================================
# DEMOS
# =============================================================================
def demo_parts():
    print("=" * 70)
    print("0. THE PARTS -- what each exercise script contributes today")
    print("=" * 70)
    print("Code is copied from each dayNN_exercise.py. Lines that had to change are marked "
          "CHANGED; lines that exist only because of today's topic are marked NEW.")
    print(f"\n{'from':<18}{'what is reused':<38}{'changed today?'}")
    for row in [("Day 63 exercise", "sigmoid, lstm_step_forward", "no -- verbatim"),
                ("Day 64 exercise", "lstm_forward (the loop)", "no -- verbatim (day65's form)"),
                ("Day 65 exercise", "lstm_backward", "3 lines: gradient in at c, out at h0, c0"),
                ("Day 66 exercise", "gru_step_forward", "not used"),
                ("Day 67 exercise", "a softmax and loss at every step", "moved into its own function"),
                ("Day 68 exercise", "a second LSTM reading backwards", "final states only")]:
        print(f"{row[0]:<18}{row[1]:<38}{row[2]}")
    Xs, Xd, Y, src = make_batch(np.random.RandomState(0), 2, 5, "reverse")  # a worked example
    print(f"\nworked example (REVERSE, 5 digits; EOS = {EOS}, GO = {GO}):")
    for k in range(2):                                     # two sequences
        print(f"  source {src[k].tolist()}  decoder reads {Xd[k].argmax(1).tolist()}  decoder must write {Y[k].tolist()}")
    for bi in [False, True]:                               # weight counts of both models
        net = Seq2Seq(bidirectional=bi)                      # untrained
        print(f"{'bidirectional' if bi else 'left-to-right'} encoder: " +
              ", ".join(f"{k} {v.shape}" for k, v in net.P.items()) + f"; total {sum(v.size for v in net.P.values())}")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("1. GRADIENT CHECK -- through the decoder AND back into the encoder")
    print("=" * 70)
    print("A tiny model (hidden 4; the bidirectional encoder 2 + 2), 3 sequences of 5 digits, "
          "REVERSE. Every entry of every parameter is nudged by +-1e-5. The encoder's weights "
          "receive gradient ONLY through the decoder's initial state, so if the new line were "
          "wrong, the encoder rows below would fail. Pass rule: |numeric - analytic| <= 1e-10 + "
          "1e-6 * |numeric|.")
    Xs, Xd, Y, _ = make_batch(np.random.RandomState(4), 3, 5, "reverse")  # tiny batch
    print(f"\n{'encoder':<15}{'parameter':>10}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>7}")
    verdict = {}                                           # encoder -> did every entry pass?
    for bi in [False, True]:                               # both encoders
        net = Seq2Seq(hidden=4, bidirectional=bi, seed=1, forget_bias=0.5)  # tiny model
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
            ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(num)))  # the pass rule
            ok_all = ok_all and ok                             # track the verdict
            print(f"{'bidirectional' if bi else 'left-to-right':<15}{name:>10}{P.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>7}")
        verdict[bi] = ok_all                                 # keep
    print(f"every entry passes: left-to-right encoder {verdict[False]}, bidirectional encoder {verdict[True]}")
    print("Actual output from this lesson's run.")


def demo_training():
    print("\n" + "=" * 70)
    print("2. TRAINING -- COPY and REVERSE, three seeds each")
    print("=" * 70)
    print(f"Hidden 32, forget bias 1, plain SGD lr 0.5 with clipping at 1.0, batches of 32 sequences "
          f"of one random length 3..8, {STEPS} steps. Test: 300 fixed sequences of every "
          "length 3..8, decoded greedily (the decoder feeds itself its own previous output). EXACT "
          "MATCH = every digit and the final EOS right.")
    print(f"\n{'task':<9}{'encoder':<15}{'seed':>5}{'final loss':>12}{'exact match':>13}{'token acc':>11}{'seconds':>9}")
    summary = {}                                           # (task, bi) -> exact-match per seed
    for task in ["copy", "reverse"]:                       # both tasks
        tests = test_sets(task)                              # its fixed test sets
        for bi in [False]:                                   # the plain encoder here; Section 5 adds the bidirectional one
            summary[(task, bi)] = []                           # per seed
            for s in SEEDS:                                    # three seeds
                net, curve, loss, secs = train_cached(task, bi, s)  # train (or reuse)
                ex, tok, _ = evaluate(net, tests)                # final test accuracy
                summary[(task, bi)].append(ex)                   # keep
                print(f"{task:<9}{'left-to-right':<15}{s:>5}{loss:>12.4f}{ex:>13.4f}{tok:>11.4f}{secs:>9.1f}")
    print("Actual output from this lesson's run.")
    print(f"\nexact match by length (mean of {len(SEEDS)} seeds):")
    print(f"{'task':<9}" + "".join(f"{'n=' + str(n):>9}" for n in LENGTHS))
    for task in ["copy", "reverse"]:                       # both tasks
        tests = test_sets(task)                              # fixed test sets
        per = np.mean([evaluate(train_cached(task, False, s)[0], tests)[2] for s in SEEDS], axis=0)  # per length
        print(f"{task:<9}" + "".join(f"{e:>9.3f}" for e in per))
    print("Actual output from this lesson's run.")
    return summary                                         # later sections reuse it


def demo_controls():
    print("\n" + "=" * 70)
    print("3. TWO CONTROLS -- remove the encoder's state; feed the decoder the truth")
    print("=" * 70)
    print("(a) The trained models with the encoder's final (h, c) replaced by zeros at test time: "
          "the decoder alone, knowing nothing about the source. Guessing gives 1/8 per digit. "
          "(b) TEACHER FORCING (training-time decoding: every decoder input is the TRUE previous "
          "digit) against greedy decoding (each input is the model's own previous output, as in "
          "real use). Token accuracy, mean of three seeds.")
    print(f"\n{'task':<9}{'normal':>10}{'encoder state zeroed':>22}{'teacher-forced':>16}{'greedy':>9}")
    for task in ["copy", "reverse"]:                       # both tasks
        tests = test_sets(task)                              # fixed test sets
        normal, zeroed, tf, greedy = [], [], [], []          # per seed
        for s in SEEDS:                                      # three seeds
            net = train_cached(task, False, s)[0]              # trained in Section 2
            normal.append(evaluate(net, tests)[1])             # token accuracy, normal
            zeroed.append(evaluate(net, tests, zero_state=True)[1])  # token accuracy, encoder state removed
            tf.append(np.mean([(net.teacher_forced(Xs, Xd) == Y).mean() for Xs, Xd, Y, _ in tests.values()]))  # teacher forcing
            greedy.append(normal[-1])                          # greedy is the normal evaluation
        print(f"{task:<9}{np.mean(normal):>10.4f}{np.mean(zeroed):>22.4f}{np.mean(tf):>16.4f}{np.mean(greedy):>9.4f}")
    print("Actual output from this lesson's run.")


def demo_positions():
    print("\n" + "=" * 70)
    print("4. WHERE THE ERRORS ARE -- accuracy at each output position, 8-digit sequences")
    print("=" * 70)
    print("Position 1 is the first digit the decoder writes. For COPY that is the FIRST digit the "
          "encoder read, 8 steps before its final state; for REVERSE it is the LAST digit it read. "
          "Greedy decoding, mean of three seeds, the 300 test sequences of length 8.")
    acc = {}                                               # task -> accuracy per output position
    for task in ["copy", "reverse"]:                       # both tasks
        Xs, _, Y, src = test_sets(task)[8]                   # the 8-digit test sequences
        acc[task] = np.mean([(train_cached(task, False, s)[0].generate(Xs, 9)[:, :8] == Y[:, :8]).mean(axis=0)
                             for s in SEEDS], axis=0)       # per output position, mean over seeds
    print(f"\n{'output position':<22}" + "".join(f"{p:>7}" for p in range(1, 9)))
    print(f"{'COPY: read at step':<22}" + "".join(f"{p:>7}" for p in range(1, 9)))
    print(f"{'  accuracy':<22}" + "".join(f"{a:>7.3f}" for a in acc["copy"]))
    print(f"{'REVERSE: read at step':<22}" + "".join(f"{p:>7}" for p in range(8, 0, -1)))
    print(f"{'  accuracy':<22}" + "".join(f"{a:>7.3f}" for a in acc["reverse"]))
    print("Actual output from this lesson's run.")
    return acc                                             # the plot reuses it


def demo_bidirectional(summary):
    print("\n" + "=" * 70)
    print("5. DAY 68'S BIDIRECTIONAL ENCODER -- same tasks, same decoder size")
    print("=" * 70)
    print("Two encoder LSTMs of 16 units (one reads left to right, one right to left); their final "
          "states are joined into the decoder's 32. Everything else as in Section 2. The seed columns are the FINAL "
          "model; 'best checkpoint' is the best exact match at any of the 20 checkpoints (every 1,000 steps), mean "
          "of the three seeds -- it shows whether a low final number is a late dip or a run that never learned.")
    print(f"\n{'task':<9}{'encoder':<15}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'mean':>9}{'best checkpoint':>17}{'seconds':>9}")
    for task in ["copy", "reverse"]:                       # both tasks
        tests = test_sets(task)                              # fixed test sets
        for bi in [False, True]:                             # both encoders
            runs = [train_cached(task, bi, s) for s in SEEDS]  # trained (or reused)
            ex = [evaluate(r[0], tests)[0] for r in runs]      # exact match per seed
            peak = np.mean([r[1].max() for r in runs])          # best exact match at any checkpoint, mean over seeds
            print(f"{task:<9}{'bidirectional' if bi else 'left-to-right':<15}" + "".join(f"{e:>9.4f}" for e in ex)
                  + f"{np.mean(ex):>9.4f}{peak:>17.4f}{np.mean([r[3] for r in runs]):>9.1f}")
    print("Actual output from this lesson's run.")


def plot_results(acc):                                  # two panels: learning curves | accuracy by output position
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))   # canvas
    styles = {("copy", False): ("#1f77b4", "-"), ("reverse", False): ("#d62728", "-"),
              ("copy", True): ("#1f77b4", "--"), ("reverse", True): ("#d62728", "--")}  # colour = task, dashes = bidirectional
    for (task, bi), (color, ls) in styles.items():         # each task and encoder
        for s in SEEDS:                                      # every seed, one line
            curve = train_cached(task, bi, s)[1]               # exact match every EVERY steps
            a1.plot(np.arange(1, len(curve) + 1) * EVERY, curve, color=color, linestyle=ls, alpha=0.8, linewidth=1.3,
                    label=f"{task}, {'bidirectional' if bi else 'left-to-right'} encoder" if s == 0 else None)
    a1.set_xlabel("training steps"); a1.set_ylabel("exact match, test (lengths 3-8)"); a1.set_ylim(-0.02, 1.02)
    a1.set_title("Learning to copy and to reverse, three seeds each", fontsize=10); a1.legend(fontsize=8, loc="lower right")
    a1.grid(alpha=0.3)
    pos = np.arange(1, 9)                                   # output positions
    a2.bar(pos - 0.2, acc["copy"], width=0.4, color="#1f77b4", label="copy")
    a2.bar(pos + 0.2, acc["reverse"], width=0.4, color="#d62728", label="reverse")
    a2.set_xlabel("output position (8-digit sequences)"); a2.set_ylabel("token accuracy (axis starts at 0.94)"); a2.set_ylim(0.94, 1.008)
    a2.set_title("Where the errors are (left-to-right encoder)", fontsize=10); a2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(); plt.savefig("seq2seq_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved seq2seq_results.png")


def draw_diagram():                                     # the encoder-decoder on a REVERSE example, with the parts named
    fig, ax = plt.subplots(figsize=(13, 5.8))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.8); ax.axis("off")      # plain coordinate grid, no axes

    def box(x, y, w, h, text, color, fs=9, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333", lw=1.3):          # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=lw))

    ax.text(6.5, 5.55, "Day 69: sequence to sequence -- the encoder's final (h, c) is all the decoder gets",
            ha="center", fontsize=12, fontweight="bold")
    enc_x = [0.9, 2.1, 3.3]                                     # encoder steps
    for j, (x, d) in enumerate(zip(enc_x, ["3", "1", "7"])):    # source digits
        box(x, 2.6, 0.9, 0.6, "LSTM", "#DCEBF7")                  # one encoder step (parts 1-2)
        box(x, 1.2, 0.6, 0.45, d, "#F2F2F2")                      # its input digit
        arrow(x, 1.44, x, 2.28)                                   # digit -> step
        if j < 2:
            arrow(x + 0.47, 2.6, enc_x[j + 1] - 0.47, 2.6, "#1F6F99")  # state passes right
    ax.text(2.1, 3.25, "ENCODER", ha="center", fontsize=9.5, fontweight="bold", color="#1F6F99")
    box(5.0, 2.6, 1.5, 0.75, "final (h, c)\n2 x 32 numbers", "#FAEEDA", fs=8.5, bold=True)  # the bottleneck
    arrow(3.77, 2.6, 4.23, 2.6, "#C97A16", 1.8)                  # encoder -> state
    dec_x = [7.0, 8.4, 9.8, 11.2]                               # decoder steps
    ins, outs = ["GO", "7", "1", "3"], ["7", "1", "3", "EOS"]   # REVERSE of 3 1 7
    for j, x in enumerate(dec_x):                               # decoder steps (part 1 at generation, parts 2-4 in training)
        box(x, 2.6, 0.9, 0.6, "LSTM", "#E5F2E1")
        box(x, 1.2, 0.7, 0.45, ins[j], "#F2F2F2", fs=8.5)          # what the decoder reads
        box(x, 4.15, 0.7, 0.45, outs[j], "#FDE7D3", fs=8.5)        # what it must write
        arrow(x, 1.44, x, 2.28); arrow(x, 2.92, x, 3.9)            # input -> step -> softmax output (part 4)
        if j < 3:
            arrow(x + 0.47, 2.6, dec_x[j + 1] - 0.47, 2.6, "#2F7D32")  # state passes right
            arrow(x + 0.38, 4.0, dec_x[j + 1] - 0.15, 1.45, "#999999", 1.0)  # greedy: each output becomes the next input
    arrow(5.77, 2.6, 6.53, 2.6, "#C97A16", 1.8)                  # state -> decoder: NEW today
    ax.text(9.1, 3.25, "DECODER", ha="center", fontsize=9.5, fontweight="bold", color="#2F7D32")
    ax.text(6.3, 1.9, "NEW: decoder\nstarts here", ha="center", fontsize=8, color="#C97A16")
    ax.text(6.5, 0.35, "parts: LSTM step (Day 63)  |  loop (Day 64)  |  backward (Day 65)  |  output at every step (Day 67)  |  "
            "optional backward-reading encoder (Day 68)", ha="center", fontsize=8.5, style="italic")
    plt.savefig("seq2seq_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved seq2seq_diagram.png")


def note_on_the_week():
    print("\n" + "=" * 70)
    print("6. THE WEEK IN ONE PARAGRAPH")
    print("=" * 70)
    print("Day 63 built the LSTM cell and its cell-state route; Day 64 looped it over time; Day 65 "
          "backpropagated through it and found the forget bias decides whether it learns; Day 66 "
          "built the GRU and found where its tied gate costs it; Day 67 put all three cells on a "
          "real task; Day 68 stacked layers and read in both directions. Today two LSTMs became "
          "an encoder and a decoder, joined only by one final state. Day 70 asks what that single "
          "state costs as sequences get longer.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread: same results, faster on this machine (Day 64)
        demo_parts()                                         # 0. the parts
        demo_gradient_check()                                # 1. the assembled model is correct
        summary = demo_training()                            # 2. train on copy and reverse
        demo_controls()                                      # 3. two controls
        acc = demo_positions()                               # 4. where the errors are
        demo_bidirectional(summary)                          # 5. Day 68's encoder
        plot_results(acc)                                    # the figure
        draw_diagram()                                       # the diagram used in the guides
        note_on_the_week()                                   # 6. the week in one paragraph


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
