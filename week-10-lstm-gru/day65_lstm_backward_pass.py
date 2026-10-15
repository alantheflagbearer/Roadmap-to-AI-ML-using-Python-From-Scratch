"""
Day 65: The LSTM Backward Pass From Scratch, a Gradient Check, and a First Training Run

Day 64 ran the LSTM forward over whole sequences and cached every gate.
Today the cache is walked backwards. Two gradients flow back in time --
one for the hidden state h, one for the cell state c -- and at every step
the c-gradient is multiplied by that step's forget gate: the "direct
route" of Day 63, now as a gradient.

What is built and measured:

  1. lstm_backward, gradient-checked on its own (a loss that uses every
     hidden state, the last hidden state and the last cell state), with
     padding, for every weight, every input and both starting states.
  2. The same check through the classifier that is actually trained.
  3. The direct route inside the backward pass: with the gates cut off
     from h, the gradient reaching c_0 is EXACTLY the gradient at c_T times
     the product of forget gates.
  4. How much gradient reaches the first input at the start of training,
     by forget bias, on Day 59's sign task.
  5. The first LSTM training runs: Day 59's sign task (the label is the
     sign of the FIRST input) at 20, 40 and 100 steps, with Day 60's
     training setup (lr 0.05, clipping at 1.0, 150 epochs) plus a larger
     learning rate, three seeds each, test accuracy tracked during
     training. The vanilla RNN reached 0.955 at 20 steps and 0.525 at 40
     with this setup on Day 60.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the backward-pass diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

# =============================================================================
# 0. REUSED -- verbatim from Days 57-64
# =============================================================================
def sigmoid(z):                                         # squashes any number into the range 0..1 -- Day 63
    return 1.0 / (1.0 + np.exp(-z))                       # large negative -> 0, zero -> 0.5, large positive -> 1


def softmax(z):                                         # logits -> probabilities, one row per example -- Days 57-62
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract the row max for numerical stability
    exp_z = np.exp(z_shifted)                             # exponentiate
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row to sum to 1


def clip_by_global_norm(grads, max_norm):               # Day 60's clipper, verbatim
    norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))  # global norm over every parameter's gradient
    if max_norm is None or norm <= max_norm:               # no threshold, or already short enough
        return grads, norm, 1.0                              # untouched
    scale = max_norm / norm                                 # the factor that makes the norm exactly max_norm
    return {k: g * scale for k, g in grads.items()}, norm, scale  # same factor everywhere -> direction preserved


def make_sign_task(n, T, seed):                          # Day 59's task, verbatim: is the FIRST input positive?
    rng = np.random.RandomState(seed)                     # private RNG
    X = rng.randn(n, T)                                    # every input N(0, 1)
    return X, (X[:, 0] > 0).astype(int)                    # label depends only on x_0


def lstm_step_forward(x_t, h_prev, c_prev, W, b):       # one step of an LSTM cell -- verbatim from Day 63
    H = h_prev.shape[1]                                   # hidden size
    a = np.concatenate([x_t, h_prev], axis=1) @ W + b     # input and old hidden state side by side, times ONE weight matrix
    f = sigmoid(a[:, 0 * H:1 * H])                        # forget gate: how much of the old cell state to KEEP (0..1)
    i = sigmoid(a[:, 1 * H:2 * H])                        # input gate: how much of the candidate to WRITE (0..1)
    g = np.tanh(a[:, 2 * H:3 * H])                        # candidate: the new value that could be written (-1..1)
    o = sigmoid(a[:, 3 * H:4 * H])                        # output gate: how much of the cell state to SHOW (0..1)
    c = f * c_prev + i * g                                # new cell state: keep part of the old, add part of the new
    h = o * np.tanh(c)                                    # new hidden state: a gated, squashed view of the cell state
    return h, c, (f, i, g, o)                             # both states, and the gate values for inspection


def init_lstm(input_dim, hidden_dim, seed, forget_bias=0.0, weight_scale=1.0):  # random LSTM weights -- verbatim from Day 63
    rng = np.random.RandomState(seed)                     # private rng -> reproducible
    W = rng.randn(input_dim + hidden_dim, 4 * hidden_dim) * weight_scale / np.sqrt(input_dim + hidden_dim)  # one block per gate
    b = np.zeros(4 * hidden_dim)                          # biases, in the same f, i, g, o order
    b[:hidden_dim] = forget_bias                          # the forget gate's bias
    return W, b                                           # W: (input + hidden, 4 * hidden);  b: (4 * hidden,)


def lstm_forward(X, h0, c0, W, b, mask=None):           # the forward pass with its cache -- verbatim from Day 64
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
        c = m_t * c_new + (1.0 - m_t) * c                   # padding: keep the old cell state too -- BOTH states are carried
        Hs[:, t], Cs[:, t] = h, c                           # store the states
        F[:, t], I[:, t], G[:, t], O[:, t] = f, i, g, o     # store the gates
    cache = dict(X=X, h0=h0, c0=c0, W=W, b=b, mask=mask, H=Hs, C=Cs, F=F, I=I, G=G, O=O)  # everything backward will need
    return Hs, Cs, cache                                  # all hidden states, all cell states, and the cache


# =============================================================================
# 1. THE BACKWARD PASS -- NEW today
# =============================================================================
def lstm_backward(dH, dh_last, dc_last, cache):         # NEW: walk the cache backwards; returns gradients for W, b, X, h0, c0
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
        da = np.concatenate([dc_new * c_prev * f * (1.0 - f),  # forget gate: c = f*c_prev + ...  then undo its sigmoid
                             dc_new * g * i * (1.0 - i),       # input gate: ... + i*g            then undo its sigmoid
                             dc_new * i * (1.0 - g ** 2),      # candidate:  ... + i*g            then undo its tanh
                             do * o * (1.0 - o)], axis=1)      # output gate                     then undo its sigmoid
        z = np.concatenate([X[:, t], h_prev], axis=1)       # what was multiplied by W at this step
        dW += z.T @ da                                      # ACCUMULATE: the same W at every step
        db += da.sum(axis=0)                                # ACCUMULATE
        dz = da @ W.T                                       # back through the matrix product
        dX[:, t] = dz[:, :D]                                # the part belonging to the input
        dh = dz[:, D:] + dh * (1.0 - m_t)                   # the part belonging to h_prev; padding passes dh straight through
        dc = dc_new * f + dc * (1.0 - m_t)                  # THE DIRECT ROUTE: times the forget gate; padding passes dc through
    return dW, db, dX, dh, dc                             # dh and dc are now the gradients at h0 and c0


class LSTMClassifier:                                   # NEW: an LSTM read at its last state, one label per sequence (Day 61's shape)
    def __init__(self, input_dim, hidden_dim, n_classes, seed=7, forget_bias=1.0, lr=0.05, clip=1.0):
        self.W, self.b = init_lstm(input_dim, hidden_dim, seed, forget_bias)  # LSTM weights
        rng = np.random.RandomState(seed + 1)              # a separate rng for the output layer
        self.Why = rng.randn(hidden_dim, n_classes) / np.sqrt(hidden_dim)  # output weights
        self.by = np.zeros(n_classes)                      # output bias
        self.H, self.lr, self.clip = hidden_dim, lr, clip  # sizes and training settings

    def forward(self, X, mask=None):                       # X: (m, T, D) -> class probabilities (m, n_classes)
        m = X.shape[0]                                       # batch size
        zeros = np.zeros((m, self.H))                        # both states start at zero
        self.Hs, self.Cs, self.cache = lstm_forward(X, zeros, zeros, self.W, self.b, mask)  # Day 64
        self.h_last = self.Hs[:, -1]                         # the state after the last real step (padding carries it)
        self.probs = softmax(self.h_last @ self.Why + self.by)  # one prediction per sequence
        return self.probs                                    # hand back

    def loss(self, y):                                     # mean cross-entropy
        return -np.mean(np.log(self.probs[np.arange(len(y)), y] + 1e-12))  # -log(prob of the true class), averaged

    def gradients(self, y):                                # gradients of the mean loss for W, b, Why, by (and dX for checking)
        m = len(y)                                           # batch size
        dlogits = self.probs.copy()                          # softmax + cross-entropy gradient ...
        dlogits[np.arange(m), y] -= 1.0                      # ... is probs minus one-hot
        dlogits /= m                                         # averaged over the batch
        grads = dict(Why=self.h_last.T @ dlogits, by=dlogits.sum(axis=0))  # output layer
        dh_last = dlogits @ self.Why.T                       # gradient arriving at the last hidden state
        grads["W"], grads["b"], self.dX, _, _ = lstm_backward(np.zeros_like(self.Hs), dh_last, np.zeros_like(dh_last), self.cache)
        return grads                                         # dict: parameter name -> gradient

    def train_step(self, X, y):                            # one SGD step with Day 60's clipping
        self.forward(X)                                      # forward pass
        grads, norm, _ = clip_by_global_norm(self.gradients(y), self.clip)  # gradients, clipped
        for name, g in grads.items():                        # plain SGD
            getattr(self, name)[...] -= self.lr * g            # in place

    def accuracy(self, X, y):                              # classification accuracy
        return (self.forward(X).argmax(axis=1) == y).mean()  # fraction correct


# =============================================================================
# DEMOS
# =============================================================================
def numeric_grad(f, P, eps=1e-5):                       # central-difference gradient of the scalar function f() w.r.t. array P
    numeric = np.zeros_like(P)                            # one slope per entry
    for pos in np.ndindex(P.shape):                       # every entry
        old = P[pos]                                        # remember it
        P[pos] = old + eps; up = f()                        # nudge up, measure
        P[pos] = old - eps; down = f()                      # nudge down, measure
        P[pos] = old                                        # restore
        numeric[pos] = (up - down) / (2 * eps)              # the slope
    return numeric                                        # same shape as P


def report(name, numeric, analytic):                    # one row of a gradient-check table
    diff = np.abs(numeric - analytic)                     # absolute disagreement, entry by entry
    rel = diff / (np.abs(numeric) + np.abs(analytic) + 1e-12)  # relative disagreement
    ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(numeric)))  # Day 62's pass rule
    print(f"{name:>7}{numeric.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>8}")
    return ok                                             # True if every entry passes


def demo_gradient_checks():
    print("=" * 70)
    print("1. THE BACKWARD PASS ON ITS OWN -- every weight, every input, both starting states")
    print("=" * 70)
    print("4 sequences of lengths 5, 2, 4, 1 (padded to 5), 2 inputs, 3 hidden units, random "
          "weights and biases, random non-zero h0 and c0. The loss uses all three places "
          "gradient can enter: L = sum(R1 * every h_t) + sum(R2 * last h) + sum(R3 * last c), "
          "with fixed random R1, R2, R3. Pass rule (Day 62): |numeric - analytic| <= 1e-10 + "
          "1e-6 * |numeric| for every entry.")
    rng = np.random.RandomState(3)                             # private rng
    m, T, D, H = 4, 5, 2, 3                                    # small sizes
    lens = np.array([5, 2, 4, 1])                              # four different lengths
    mask = (np.arange(T)[None, :] < lens[:, None]).astype(float)  # 1 = real step
    X = rng.randn(m, T, D)                                     # inputs
    h0, c0 = rng.randn(m, H) * 0.5, rng.randn(m, H) * 0.5      # non-zero starting states
    W, b = init_lstm(D, H, seed=1, forget_bias=1.0)            # weights
    b += rng.randn(4 * H) * 0.3                                # random biases too
    R1, R2, R3 = rng.randn(m, T, H), rng.randn(m, H), rng.randn(m, H)  # fixed random weightings of the outputs

    def loss():                                                # the scalar being differentiated
        Hs, Cs, _ = lstm_forward(X, h0, c0, W, b, mask)          # forward pass
        return np.sum(R1 * Hs * mask[:, :, None]) + np.sum(R2 * Hs[:, -1]) + np.sum(R3 * Cs[:, -1])

    _, _, cache = lstm_forward(X, h0, c0, W, b, mask)          # forward once for the analytic gradients
    dW, db, dX, dh0, dc0 = lstm_backward(R1 * mask[:, :, None], R2, R3, cache)  # the gradients under test
    print(f"\n{'array':>7}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>8}")
    ok = all([report(name, numeric_grad(loss, P), A) for name, P, A in
              [("W", W, dW), ("b", b, db), ("X", X, dX), ("h0", h0, dh0), ("c0", c0, dc0)]])
    print(f"every entry passes: {ok};  gradient at padded inputs exactly zero: {np.all(dX[mask == 0] == 0)}")
    print("Actual output from this lesson's run.")

    print("\n" + "=" * 70)
    print("2. THE SAME CHECK THROUGH THE CLASSIFIER THAT IS TRAINED")
    print("=" * 70)
    print("Same sequences and lengths; labels 0, 1, 1, 0. LSTMClassifier (3 hidden units, "
          "forget bias 1, extra random biases) reads the last real hidden state; the loss is "
          "its mean cross-entropy, exactly as in training.")
    net = LSTMClassifier(D, H, 2, seed=1, forget_bias=1.0, clip=None)  # tiny classifier
    net.b += rng.randn(4 * H) * 0.3                            # random biases
    y = np.array([0, 1, 1, 0])                                 # labels
    f = lambda: (net.forward(X, mask), net.loss(y))[1]         # forward, then return only the loss
    net.forward(X, mask); grads = net.gradients(y)             # analytic gradients
    print(f"\n{'array':>7}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>8}")
    ok = all([report(name, numeric_grad(f, getattr(net, name)), grads[name]) for name in ["W", "b", "Why", "by"]])
    print(f"every entry passes: {ok}")
    print("Actual output from this lesson's run.")


def demo_direct_route():
    print("\n" + "=" * 70)
    print("3. THE DIRECT ROUTE, AS A GRADIENT")
    print("=" * 70)
    print("Day 63 measured the cell-state line going forward. Going backward, every step "
          "multiplies dL/dc by that step's forget gate. With the gates cut off from h (the "
          "rows of W that multiply h set to zero) that line is the ONLY route from c_T back to "
          "c_0, so dL/dc_0 must equal dL/dc_T times the product of the forget gates, exactly. "
          "16 units, 200 sequences of 30 N(0,1) inputs, forget bias 2.")
    m, T, H = 200, 30, 16                                      # sizes
    X = np.random.RandomState(0).randn(m, T, 1)                # inputs
    W, b = init_lstm(1, H, seed=3, forget_bias=2.0)            # weights
    zeros = np.zeros((m, H))                                   # starting states
    dc_T = np.random.RandomState(1).randn(m, H)                # some gradient arriving at the last cell state
    for label, cut in [("gates cut off from h", True), ("normal LSTM", False)]:  # with and without the indirect routes
        Wc = W.copy()                                            # never change the original weights
        if cut:                                                  # remove every connection from h
            Wc[1:, :] = 0.0
        _, _, cache = lstm_forward(X, zeros, zeros, Wc, b)       # forward
        _, _, _, _, dc0 = lstm_backward(np.zeros((m, T, H)), np.zeros((m, H)), dc_T, cache)  # gradient only enters at c_T
        direct = dc_T * np.prod(cache["F"], axis=1)              # the direct route alone: product of forget gates over time
        print(f"\n{label}: ||dL/dc_0|| = {np.linalg.norm(dc0):.6e},  direct route alone = {np.linalg.norm(direct):.6e},  "
              f"max |difference| = {np.max(np.abs(dc0 - direct)):.2e}")
    print("Actual output from this lesson's run.")


def demo_gradient_reach():
    print("\n" + "=" * 70)
    print("4. HOW MUCH GRADIENT REACHES THE FIRST INPUT -- before any training")
    print("=" * 70)
    print("Day 59's sign task: the label is the sign of x_0, so x_0 is the only input that "
          "matters. For an untrained classifier (16 units, seed 7) on the 400 test "
          "sequences: ||dL/dx_0|| divided by ||dL/dx_(T-1)||, the gradient reaching the "
          "first input relative to the last. Now measured with the backward pass, so there "
          "is no lower limit on what can be measured.")
    print(f"\n{'T':>5}" + "".join(f"{'forget bias ' + str(fb):>17}" for fb in [0.0, 1.0, 2.0]))
    table = {}                                                 # (T, forget bias) -> ratio, for the plot
    for T in [20, 40, 100]:                                    # three lengths
        X, y = make_sign_task(400, T, seed=999)                 # Day 59's exact test sequences
        row = []                                                 # one ratio per forget bias
        for fb in [0.0, 1.0, 2.0]:                               # three forget biases
            net = LSTMClassifier(1, 16, 2, seed=7, forget_bias=fb)  # untrained
            net.forward(X[:, :, None]); net.gradients(y)           # forward and backward; net.dX holds dL/dX
            norms = np.linalg.norm(net.dX[:, :, 0], axis=0)        # gradient size at every input position
            table[(T, fb)] = norms[0] / norms[-1]                  # first relative to last
            row.append(table[(T, fb)])
        print(f"{T:>5}" + "".join(f"{r:>17.3e}" for r in row))
    print("Actual output from this lesson's run.")
    return table                                               # for the summary plot


def demo_training():
    print("\n" + "=" * 70)
    print("5. THE FIRST LSTM TRAINING RUNS -- Day 59's sign task at 20, 40 and 100 steps")
    print("=" * 70)
    print("16 units, 600 training and 400 test sequences (Day 59's seeds), batch 30, 150 epochs, "
          "clipping at 1.0 (Day 60). Learning rate 0.05 (Day 60's) and 0.2; forget bias 0, 1 "
          "and 2; seeds 7, 8, 9 (initial weights and batch order). Test accuracy every 10 "
          "epochs. Reference, same data and setup, lr 0.05, clipped at 1.0 (Day 60): vanilla "
          "RNN 0.955 final at 20 steps, 0.525 at 40.")
    results = {}                                               # (T, fb, lr) -> list of (peak, final, curve) per seed
    print(f"\n{'T':>4}{'f-bias':>8}{'lr':>6}   {'peak / final test accuracy, seeds 7, 8, 9':<44}{'final >= 0.9':>13}")
    for T in [20, 40, 100]:                                    # three lengths
        X_tr, y_tr = make_sign_task(600, T, seed=42)             # Day 59's training data
        X_te, y_te = make_sign_task(400, T, seed=999)            # Day 59's test data
        X_tr, X_te = X_tr[:, :, None], X_te[:, :, None]          # one input per step: (n, T) -> (n, T, 1)
        for fb in [0.0, 1.0, 2.0]:                               # three forget biases
            for lr in [0.05, 0.2]:                                 # two learning rates
                runs = []                                            # one entry per seed
                for seed in [7, 8, 9]:                               # three seeds
                    net = LSTMClassifier(1, 16, 2, seed=seed, forget_bias=fb, lr=lr, clip=1.0)  # fresh network
                    rng = np.random.RandomState(seed)                  # batch order
                    curve = []                                         # test accuracy during training
                    for epoch in range(150):                           # 150 epochs, as on Day 60
                        order = rng.permutation(600)                     # fresh shuffle
                        for start in range(0, 600, 30):                  # batches of 30
                            idx = order[start:start + 30]                  # this batch
                            net.train_step(X_tr[idx], y_tr[idx])           # one step
                        if epoch % 10 == 0 or epoch == 149:              # every 10 epochs, and the last
                            curve.append(net.accuracy(X_te, y_te))         # test accuracy NOW
                    runs.append((max(curve), curve[-1], curve))          # peak, final, whole curve
                results[(T, fb, lr)] = runs                              # keep
                cells = "  ".join(f"{p:.3f}/{f:.3f}" for p, f, _ in runs)  # peak/final per seed
                print(f"{T:>4}{fb:>8}{lr:>6}   {cells:<44}{sum(f >= 0.9 for _, f, _ in runs):>9} of 3")
    print("Actual output from this lesson's run.")
    return results                                             # for the plot


def plot_results(reach, results):                       # two panels: gradient reach at init, and final accuracy
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))       # side by side
    colors = {0.0: "#fdae6b", 1.0: "#e6550d", 2.0: "#a63603"}   # one colour per forget bias (Day 63's palette)
    for fb in [0.0, 1.0, 2.0]:                                   # left: gradient reaching x_0 relative to x_(T-1)
        a1.semilogy([20, 40, 100], [reach[(T, fb)] for T in [20, 40, 100]], "o-", color=colors[fb], label=f"forget bias {fb}")
    a1.axhline(1.0, color="black", linewidth=0.8)
    a1.set_xlabel("sequence length T"); a1.set_ylabel("||dL/dx_0|| / ||dL/dx_(T-1)||")
    a1.set_title("Gradient reaching the first input, before training", fontsize=10); a1.legend(fontsize=8)
    x = 0                                                         # right: final test accuracy of every run
    ticks, labels = [], []                                        # tick positions and labels
    for T in [20, 40, 100]:                                       # one group per length
        for lr in [0.05, 0.2]:                                      # one sub-group per learning rate
            for j, fb in enumerate([0.0, 1.0, 2.0]):                  # one column of dots per forget bias
                finals = [f for _, f, _ in results[(T, fb, lr)]]        # three seeds
                a2.scatter([x + j * 0.25] * 3, finals, color=colors[fb], s=22, zorder=3, label=f"forget bias {fb}" if (T, lr) == (20, 0.05) else None)
            ticks.append(x + 0.25); labels.append(f"T={T}\nlr={lr}")
            x += 1.2                                                  # gap before the next sub-group
    a2.axhline(0.5, color="gray", linestyle="--", linewidth=1)     # chance
    a2.set_xticks(ticks); a2.set_xticklabels(labels, fontsize=8); a2.set_ylim(0.4, 1.02)
    a2.set_ylabel("final test accuracy"); a2.set_title("Final test accuracy, every run (3 seeds each)", fontsize=10)
    a2.legend(fontsize=8, loc="center left")
    plt.tight_layout(); plt.savefig("lstm_training_by_forget_bias.png", dpi=110); plt.close()  # save and free
    print("\nsaved lstm_training_by_forget_bias.png")


def draw_backward_diagram():                            # one step of the backward pass: where dh and dc go
    fig, ax = plt.subplots(figsize=(13, 5.4))                   # wide canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")      # plain coordinate grid

    def box(x, y, w, h, title, sub, color):                      # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x, y + 0.17, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8.3)

    def arrow(x0, y0, x1, y1, color="#333333", width=1.5):       # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=width))

    ax.text(6.5, 5.1, "One step of the backward pass (step t), gradients flowing right to left", ha="center", fontsize=12, fontweight="bold")
    ax.text(12.3, 4.15, "dc from\nstep t+1", ha="center", va="center", fontsize=9.5, color="#a63603", fontweight="bold")
    ax.text(12.3, 2.0, "dh from step t+1\n+ dH[t] from the loss", ha="center", va="center", fontsize=9, color="#08519c", fontweight="bold")
    box(8.9, 2.0, 2.6, 1.0, "through h = o * tanh(c)", "gives do, and adds to dc", "#E6F1F8")
    box(5.6, 4.15, 2.4, 1.0, "dc at step t", "dc_next + (part from h)", "#FDECEC")
    box(5.6, 2.0, 2.4, 1.0, "gate gradients", "df, di, dg, do -> da", "#F4EEFC")
    box(2.6, 2.0, 2.4, 1.0, "back through W", "dW += z^T da;  dz = da W^T", "#F2F2F2")
    arrow(11.4, 4.15, 6.85, 4.15, "#a63603", 2.4)                  # dc from the future into dc at step t
    arrow(11.25, 2.0, 10.25, 2.0, "#08519c")                       # dh into the tanh box
    arrow(8.9, 2.55, 6.4, 3.6, "#a63603")                          # tanh box adds to dc
    arrow(7.6, 2.0, 6.85, 2.0, "#08519c")                          # do into the gate box
    arrow(5.6, 3.6, 5.6, 2.55, "#a63603")                          # dc feeds df, di, dg
    arrow(4.35, 2.0, 3.85, 2.0)                                    # da into W
    arrow(4.35, 4.15, 0.9, 4.15, "#a63603", 2.4)                   # dc * f to step t-1
    ax.text(2.6, 4.38, "dc_prev = dc * f   (the direct route)", ha="center", fontsize=9, color="#a63603")
    ax.text(0.55, 4.15, "to\nt-1", ha="center", va="center", fontsize=9, color="#a63603", fontweight="bold")
    arrow(1.35, 2.0, 0.9, 2.0, "#08519c")                          # dh_prev to step t-1
    ax.text(0.55, 2.0, "to\nt-1", ha="center", va="center", fontsize=9, color="#08519c", fontweight="bold")
    ax.text(1.15, 1.35, "dh_prev = dz[:, D:]", ha="center", fontsize=8.5, color="#08519c")
    arrow(2.6, 1.45, 2.6, 0.75)                                    # dX out
    ax.text(2.6, 0.5, "dX[t] = dz[:, :D]", ha="center", fontsize=8.5)
    plt.savefig("lstm_backward_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("\nsaved lstm_backward_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT COMES NEXT")
    print("=" * 70)
    print("The LSTM now has a checked backward pass and has been trained. Day 66 builds the GRU, "
          "a simpler gated cell with one state instead of two, and Day 67 compares a trained "
          "RNN, LSTM and GRU side by side on Week 9's data.")


def main():                                             # runs every section in order
    with threadpool_limits(limits=1, user_api="blas"):     # one thread for NumPy's matrix library: faster here, same results
        demo_gradient_checks()                               # 1-2. gradient checks
        demo_direct_route()                                  # 3. the direct route as a gradient
        reach = demo_gradient_reach()                        # 4. gradient reaching x_0 before training
        results = demo_training()                            # 5. training on the sign task
        plot_results(reach, results)                         # the summary plot
        draw_backward_diagram()                              # the diagram used in the guides
        note_on_what_comes_next()                            # 6. bridge to Days 66-67


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
