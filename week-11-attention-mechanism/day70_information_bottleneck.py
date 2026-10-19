"""
Day 70: The Information Bottleneck -- Where One Fixed-Size State Stops Being Enough

Day 69's encoder-decoder copied and reversed sequences of 3 to 8 digits
almost perfectly. Everything the decoder knew about the input passed
through one place: the encoder's final (h, c), a state whose size does
not change with the length of the input. That is the INFORMATION
BOTTLENECK of the basic sequence-to-sequence model.

What is measured, reported as it came out:

  1. Day 69's model on longer sequences than it was trained on.
  2. A fair test: trained on every length it is tested on (3-20 digits),
     at three state sizes, with accuracy tracked during training.
  3. Too far, or too much? Every input has 20 digits; the model must write
     back only the first k. The distance is fixed; the amount to hold is
     not.
  4. Where the errors are in a 20-digit copy.
  5. What the final state holds: a linear probe for every input position.
  6. A follow-up: three times the training budget.
  7. Reversing the target (Sutskever et al.'s trick) at longer lengths.

No timings are reported today: the experiments run for long enough that
this machine's run-to-run speed changes (Day 69) would make them
meaningless.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
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
EVAL_LENGTHS = (4, 8, 12, 16, 20)                       # test lengths
HIDDENS = (16, 32, 64)                                  # three state sizes
KS = (4, 8, 12, 16, 20)                                 # Section 3: how many of the 20 input digits to write back
STEPS_MAIN = 40000                                      # training steps for Sections 2, 4, 7
STEPS_K = 30000                                         # training steps for Section 3
STEPS_LONG = 120000                                     # Section 6: three times the main budget
EVERY = 5000                                            # test accuracy is measured every EVERY steps

# =============================================================================
# REUSED -- verbatim from Day 69 (its LSTM, output layer, model and data)
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
            self.P[k] -= self.lr * scale * g[k]           # in place
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


def make_batch(rng, m, n, task):                        # m sequences of n digits, and what the decoder sees and must write
    src = rng.randint(1, 9, size=(m, n))                  # random digits 1..8
    tgt = src if task == "copy" else src[:, ::-1]         # COPY: the same digits; REVERSE: backwards
    dec_in = np.concatenate([np.full((m, 1), GO), tgt], axis=1)    # decoder input: GO, then the target shifted right
    dec_out = np.concatenate([tgt, np.full((m, 1), EOS)], axis=1)  # decoder output: the target, then EOS
    return np.eye(V)[src], np.eye(V)[dec_in], dec_out, src  # one-hot source, one-hot decoder input, targets, raw digits


def make_batch_first(rng, m, n, k):                     # NEW: read n digits, write back only the FIRST k of them
    src = rng.randint(1, 9, size=(m, n))                  # random digits 1..8 (the same draw as make_batch)
    tgt = src[:, :k]                                      # the first k digits
    dec_in = np.concatenate([np.full((m, 1), GO), tgt], axis=1)    # GO, then the target shifted right
    dec_out = np.concatenate([tgt, np.full((m, 1), EOS)], axis=1)  # the target, then EOS
    return np.eye(V)[src], np.eye(V)[dec_in], dec_out, src  # same four arrays as make_batch


def fixed_test(n, task, k=None, m=300):                 # NEW: 300 fixed test sequences of n digits (the same for every model)
    rng = np.random.RandomState(12345 + n)                # a seed per length, independent of training
    return make_batch(rng, m, n, task) if k is None else make_batch_first(rng, m, n, k)  # all digits, or only the first k


def exact_and_token(net, batch):                        # NEW: greedy decoding; exact match and token accuracy
    Xs, _, Y, _ = batch                                   # source and targets
    pred = net.generate(Xs, Y.shape[1])                   # as many tokens as the target has (digits + EOS)
    return np.all(pred == Y, axis=1).mean(), (pred == Y).mean()  # whole sequence right; share of tokens right


def train(task, hidden, seed, steps, lo, hi, k=None, every=EVERY, eval_lengths=EVAL_LENGTHS):  # NEW: Day 69's loop, more options
    net = Seq2Seq(hidden=hidden, seed=seed)                 # Day 69's model, unchanged
    rng = np.random.RandomState(seed + 100)                 # the training data stream (Day 69's seed rule)
    tests = {n: fixed_test(n, task, k) for n in eval_lengths}  # fixed test sets
    curve = []                                              # exact match per test length, every `every` steps
    for step in range(1, steps + 1):                        # plain SGD on fresh batches
        n = rng.randint(lo, hi + 1)                           # a random input length for this batch
        if k is None:                                     # COPY or REVERSE all n digits ...
            Xs, Xd, Y, _ = make_batch(rng, 32, n, task)         # copy or reverse all n digits
        else:
            Xs, Xd, Y, _ = make_batch_first(rng, 32, n, min(k, n))  # write back the first k
        net.train_step(Xs, Xd, Y)                             # Day 69's update, with clipping
        if step % every == 0:                                 # progress check
            curve.append([exact_and_token(net, tests[n])[0] for n in eval_lengths])  # exact match at each length
    return net, np.array(curve)                             # final model, (checkpoints, lengths)


RUNS = {}                                               # cache: each configuration trains once


def train_cached(*args, **kwargs):                      # train(), remembered by its arguments
    key = (args, tuple(sorted(kwargs.items())))           # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = train(*args, **kwargs)                # train and remember
    return RUNS[key]                                      # later requests reuse the result


# =============================================================================
# DEMOS
# =============================================================================
def demo_outside_range():
    print("=" * 70)
    print("1. DAY 69'S MODEL ON LONGER SEQUENCES -- outside its training range")
    print("=" * 70)
    print("Day 69's COPY model exactly as trained there (hidden 32, lengths 3-8, 20,000 steps, "
          "seed 0), tested on 300 fixed sequences of each length 4 to 16.")
    net, _ = train_cached("copy", 32, 0, 20000, 3, 8, eval_lengths=(8,))  # Day 69's training run, repeated
    print(f"\n{'length':>7}" + "".join(f"{n:>7}" for n in range(4, 17, 2)))
    rows = [exact_and_token(net, fixed_test(n, "copy")) for n in range(4, 17, 2)]  # each length
    print(f"{'exact':>7}" + "".join(f"{e:>7.3f}" for e, _ in rows))
    print(f"{'token':>7}" + "".join(f"{t:>7.3f}" for _, t in rows))
    print("Actual output from this lesson's run.")
    print("\nThis mixes two problems: the model never saw these lengths, AND it must hold more "
          "digits in one state. Section 2 trains on every length it is tested on.")


def demo_fair_test():
    print("\n" + "=" * 70)
    print("2. TRAINED ON 3-20 DIGITS -- exact match by length, three state sizes")
    print("=" * 70)
    print(f"COPY, lengths 3-20 in training, {STEPS_MAIN} steps, hidden 16, 32 and 64 (the decoder "
          "gets 2 x hidden numbers: h and c). Test: 300 fixed sequences per length; every length "
          "tested was also trained on. Exact match is measured every 5,000 steps.")
    final = {}                                             # hidden -> exact match per length (mean of seeds)
    print(f"\n{'hidden':>7}{'state size':>12}" + "".join(f"{'n=' + str(n):>8}" for n in EVAL_LENGTHS) + f"{'seeds':>8}")
    for hidden in HIDDENS:                                 # three state sizes
        curves = [train_cached("copy", hidden, s, STEPS_MAIN, 3, 20)[1] for s in SEEDS]  # (checkpoints, lengths) per seed
        final[hidden] = np.mean([c[-1] for c in curves], axis=0)  # final exact match, mean over seeds
        print(f"{hidden:>7}{2 * hidden:>12}" + "".join(f"{e:>8.3f}" for e in final[hidden]) + f"{len(SEEDS):>8}")
    print("Actual output from this lesson's run.")
    print(f"\nis training finished? exact match at n=12 and n=16 over the last checkpoints (mean of seeds):")
    print(f"{'hidden':>7}{'n':>4}" + "".join(f"{'@' + str(c // 1000) + 'k':>8}" for c in range(STEPS_MAIN // 2, STEPS_MAIN + 1, EVERY)))
    for hidden in HIDDENS:                                 # each size
        curves = np.mean([train_cached("copy", hidden, s, STEPS_MAIN, 3, 20)[1] for s in SEEDS], axis=0)  # mean curve
        for j, n in enumerate(EVAL_LENGTHS):              # each test length
            if n in (12, 16):                                # two lengths where it matters
                idx = range(STEPS_MAIN // 2 // EVERY - 1, STEPS_MAIN // EVERY)  # checkpoints from half-way to the end
                print(f"{hidden:>7}{n:>4}" + "".join(f"{curves[i, j]:>8.3f}" for i in idx))
    print("Actual output from this lesson's run.")
    return final                                           # the plot reuses it


def demo_amount_vs_distance():
    print("\n" + "=" * 70)
    print("3. TOO FAR, OR TOO MUCH? -- read 20 digits, write back only the first k")
    print("=" * 70)
    print("Every input has 20 digits. The target is the FIRST k of them (k = 4, 8, 12, 16, 20; "
          "k = 20 is full COPY). For every k the first digit to write was read 20 steps before "
          "the encoder's final state, so the DISTANCE is the same; only the AMOUNT to hold changes. "
          f"Hidden 32, {STEPS_K} steps, inputs always 20 digits, three seeds.")
    res = {}                                               # k -> (exact, token) mean of seeds
    print(f"\n{'k':>4}{'exact match':>13}{'token acc':>11}{'per seed (exact)':>30}")
    for k in KS:                                           # amounts to hold
        runs = [train_cached("first", 32, s, STEPS_K, 20, 20, k=k, eval_lengths=(20,)) for s in SEEDS]  # trained models
        scores = [exact_and_token(net, fixed_test(20, "first", k)) for net, _ in runs]  # final test
        res[k] = np.mean(scores, axis=0)                     # mean over seeds
        print(f"{k:>4}{res[k][0]:>13.3f}{res[k][1]:>11.3f}{'  '.join(f'{e:.3f}' for e, _ in scores):>30}")
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_where_errors():
    print("\n" + "=" * 70)
    print("4. WHERE THE ERRORS ARE -- token accuracy at each output position, 20-digit COPY")
    print("=" * 70)
    print("The hidden-32 models from Section 2 on the 300 fixed 20-digit sequences, greedy decoding, "
          "mean of three seeds. Position 1 is the first digit written (the first digit read).")
    Xs, _, Y, _ = fixed_test(20, "copy")                   # 20-digit test sequences
    acc = np.mean([(train_cached("copy", 32, s, STEPS_MAIN, 3, 20)[0].generate(Xs, 21) == Y).mean(axis=0)  # per output position, each seed
                   for s in SEEDS], axis=0)              # per position, mean over seeds
    print(f"\n{'position':>9}" + "".join(f"{p:>6}" for p in range(1, 21, 2)))
    print(f"{'accuracy':>9}" + "".join(f"{acc[p - 1]:>6.3f}" for p in range(1, 21, 2)))
    print(f"guessing a digit: {1 / 8:.3f}")
    print("Actual output from this lesson's run.")
    return acc                                             # the plot reuses it


def probe_features(net, Xs):                            # the encoder's final (h, c), joined: everything the decoder gets
    h, c, _ = net.encode(Xs)                              # Day 69's encode
    return np.concatenate([h, c], axis=1)                 # (m, 2 x hidden)


def fit_probe(F, y, iters=400, lr=0.5):                 # NEW: a linear softmax classifier, plain gradient descent
    mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-8         # standardise each feature
    Z = (F - mu) / sd                                     # standardised features
    W = np.zeros((F.shape[1], V)); b = np.zeros(V)        # weights and biases, start at zero
    for _ in range(iters):                                # full-batch gradient descent
        p = softmax(Z @ W + b)                              # predicted digit probabilities
        p[np.arange(len(y)), y] -= 1                        # softmax + cross-entropy gradient
        W -= lr * Z.T @ p / len(y); b -= lr * p.mean(axis=0)  # update
    return lambda G: ((G - mu) / sd @ W + b).argmax(axis=1)  # a function that predicts digits from new features


def demo_probe():
    print("\n" + "=" * 70)
    print("5. WHAT THE FINAL STATE HOLDS -- a linear probe for every input position")
    print("=" * 70)
    print("For 20-digit inputs, a separate linear classifier is trained to read the digit at input "
          "position p (1..20) from the encoder's final (h, c) alone: 3,000 training sequences, 1,000 "
          "test sequences. Models: the hidden-32 COPY model (Section 2, seed 0), the same "
          "architecture untrained, and the k = 4 model from Section 3 (seed 0).")
    rng = np.random.RandomState(31)                        # probe data
    Xtr, _, _, src_tr = make_batch(rng, 3000, 20, "copy")  # probe training inputs
    Xte, _, _, src_te = make_batch(rng, 1000, 20, "copy")  # probe test inputs
    models = {"trained on COPY": train_cached("copy", 32, 0, STEPS_MAIN, 3, 20)[0],  # three encoders to probe
              "untrained": Seq2Seq(hidden=32, seed=0),
              "trained, first 4 only": train_cached("first", 32, 0, STEPS_K, 20, 20, k=4, eval_lengths=(20,))[0]}
    res = {}                                               # model -> probe accuracy per position
    for name, net in models.items():                       # each encoder
        Ftr, Fte = probe_features(net, Xtr), probe_features(net, Xte)  # final states
        res[name] = [(fit_probe(Ftr, src_tr[:, p])(Fte) == src_te[:, p]).mean() for p in range(20)]  # one probe per position
    print(f"\n{'input position':<24}" + "".join(f"{p:>6}" for p in range(1, 21, 2)))
    for name, acc in res.items():                         # one row per encoder
        print(f"{name:<24}" + "".join(f"{acc[p - 1]:>6.3f}" for p in range(1, 21, 2)))
    print(f"guessing: {1 / 8:.3f}")
    print("Actual output from this lesson's run.")
    return res                                             # the plot reuses it


def demo_more_training():
    print("\n" + "=" * 70)
    print("6. FOLLOW-UP: IS IT JUST TRAINING TIME?")
    print("=" * 70)
    print(f"Hidden 32 COPY on lengths 3-20, seed 0, trained {STEPS_LONG} steps ({STEPS_LONG // STEPS_MAIN} "
          "times Section 2's budget). Exact match every 5,000 steps at three lengths.")
    _, curve = train_cached("copy", 32, 0, STEPS_LONG, 3, 20)  # the long run
    cols = [EVAL_LENGTHS.index(n) for n in (8, 12, 16)]     # three lengths
    print(f"\n{'steps':>8}{'n=8':>8}{'n=12':>8}{'n=16':>8}")
    for i in range(3, len(curve), 4):                      # every 20,000 steps
        print(f"{(i + 1) * EVERY:>8}" + "".join(f"{curve[i, c]:>8.3f}" for c in cols))
    print("Actual output from this lesson's run.")
    return curve                                           # the plot reuses it


def demo_reverse():
    print("\n" + "=" * 70)
    print("7. DOES REVERSING HELP? -- Sutskever et al.'s trick at longer lengths")
    print("=" * 70)
    print(f"REVERSE instead of COPY, everything else as Section 2 at hidden 32 ({STEPS_MAIN} steps, "
          "three seeds). Reversing puts the first digits to write next to the encoder's end, but "
          "the LAST digits to write are then the ones read longest ago.")
    copy = np.mean([train_cached("copy", 32, s, STEPS_MAIN, 3, 20)[1][-1] for s in SEEDS], axis=0)  # copy, final
    rev = np.mean([train_cached("reverse", 32, s, STEPS_MAIN, 3, 20)[1][-1] for s in SEEDS], axis=0)  # reverse, final
    print(f"\n{'task':>8}" + "".join(f"{'n=' + str(n):>8}" for n in EVAL_LENGTHS))
    print(f"{'copy':>8}" + "".join(f"{e:>8.3f}" for e in copy))
    print(f"{'reverse':>8}" + "".join(f"{e:>8.3f}" for e in rev))
    print("Actual output from this lesson's run.")


def plot_results(final, amount, probe, long_curve):     # four panels from the measurements above
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))         # canvas (the last legend sits outside its panel)
    for hidden, color in zip(HIDDENS, ["#9ecae1", "#3182bd", "#08519c"]):  # panel 1: exact match by length
        ax[0].plot(EVAL_LENGTHS, final[hidden], marker="o", color=color, label=f"hidden {hidden}")
    ax[0].set_xlabel("sequence length (digits)"); ax[0].set_ylabel("exact match (test)"); ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_title("COPY: longer sequences fail", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(KS, [amount[k][0] for k in KS], marker="o", color="#d62728", label="exact match")  # panel 2: amount
    ax[1].plot(KS, [amount[k][1] for k in KS], marker="s", color="#ff9896", label="token accuracy")
    ax[1].set_xlabel("digits to write back (input always 20)"); ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("Same distance, more to hold", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    for (name, acc), color in zip(probe.items(), ["#3182bd", "#999999", "#2ca02c"]):  # panel 3: probe
        ax[2].plot(range(1, 21), acc, marker=".", color=color, label=name)
    ax[2].axhline(1 / 8, color="#555555", linestyle=":", linewidth=1, label="guessing")
    ax[2].set_xlabel("input position (20-digit inputs)"); ax[2].set_ylabel("probe accuracy"); ax[2].set_ylim(0, 1.02)
    ax[2].set_title("What the final state holds", fontsize=10); ax[2].legend(fontsize=7.5); ax[2].grid(alpha=0.3)
    steps = np.arange(1, len(long_curve) + 1) * EVERY        # panel 4: the long run
    for n, color in zip((8, 12, 16, 20), ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]):
        ax[3].plot(steps, long_curve[:, EVAL_LENGTHS.index(n)], color=color, label=f"n={n}")
    ax[3].set_xlabel("training steps"); ax[3].set_ylabel("exact match"); ax[3].set_ylim(-0.02, 1.02)
    ax[3].set_title("Hidden 32, trained longer (seed 0)", fontsize=10); ax[3].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax[3].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("bottleneck_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved bottleneck_results.png")


def draw_diagram():                                     # the bottleneck: n inputs -> one fixed-size state -> n outputs
    fig, ax = plt.subplots(figsize=(13, 4.8))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis("off")      # plain coordinate grid, no axes

    def box(x, y, w, h, text, color, fs=9, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333", lw=1.3):          # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=lw))

    ax.text(6.5, 4.5, "The bottleneck: every input digit must pass through one state of fixed size", ha="center",
            fontsize=12, fontweight="bold")
    xs = np.linspace(0.6, 4.4, 8)                               # encoder steps
    for j, x in enumerate(xs):
        box(x, 2.3, 0.42, 0.5, "", "#DCEBF7")                    # one encoder step
        box(x, 1.1, 0.42, 0.38, str((j * 5) % 8 + 1), "#F2F2F2", fs=8)  # its digit
        arrow(x, 1.3, x, 2.04)
        if j < len(xs) - 1:
            arrow(x + 0.22, 2.3, xs[j + 1] - 0.22, 2.3, "#1F6F99", 1.0)
    ax.text(2.5, 3.0, "ENCODER reads n digits", ha="center", fontsize=9, color="#1F6F99", fontweight="bold")
    ax.text(2.5, 0.55, "... n can be 4 or 20: the state below does not grow", ha="center", fontsize=8.5, style="italic")
    box(6.5, 2.3, 1.7, 1.0, "final (h, c)\nfixed size:\n2 x hidden", "#FAEEDA", fs=9, bold=True)  # the bottleneck
    arrow(4.65, 2.3, 5.62, 2.3, "#C97A16", 2.2); arrow(7.38, 2.3, 8.35, 2.3, "#C97A16", 2.2)
    xd = np.linspace(8.6, 12.4, 8)                              # decoder steps
    for j, x in enumerate(xd):
        box(x, 2.3, 0.42, 0.5, "", "#E5F2E1")                    # one decoder step
        box(x, 3.5, 0.42, 0.38, str((j * 5) % 8 + 1), "#FDE7D3", fs=8)  # what it writes
        arrow(x, 2.56, x, 3.3)
        if j < len(xd) - 1:
            arrow(x + 0.22, 2.3, xd[j + 1] - 0.22, 2.3, "#2F7D32", 1.0)
    ax.text(10.5, 1.55, "DECODER writes n digits", ha="center", fontsize=9, color="#2F7D32", fontweight="bold")
    ax.text(10.5, 0.95, "from that state alone (Day 69)", ha="center", fontsize=8.5, style="italic")
    plt.savefig("bottleneck_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved bottleneck_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("The decoder fails because it may only look at one summary of the input. Day 71 lets it look "
          "back at EVERY encoder state, choosing at each output step which ones to read: additive "
          "(Bahdanau) attention.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread: same results, faster on this machine (Day 64)
        demo_outside_range()                                 # 1. Day 69's model, longer inputs
        final = demo_fair_test()                             # 2. trained on every length tested
        amount = demo_amount_vs_distance()                   # 3. distance fixed, amount varied
        demo_where_errors()                                  # 4. where the errors are
        probe = demo_probe()                                 # 5. what the final state holds
        long_curve = demo_more_training()                    # 6. more training
        demo_reverse()                                       # 7. reversing
        plot_results(final, amount, probe, long_curve)       # the figure
        draw_diagram()                                       # the diagram used in the guides
        note_on_what_comes_next()                            # 8. Day 71


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
