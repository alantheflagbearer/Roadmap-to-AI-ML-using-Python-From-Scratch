"""
Day 66: The GRU -- a Simpler Gated Cell, From Scratch

Days 63-65 built the LSTM: two states, three gates and a candidate, and
a training result that hinged on one setting, the forget gate's bias.
The GRU (Cho et al., 2014) keeps the idea and drops parts of the
machinery: ONE state, two gates, one candidate.

    z = sigmoid([x, h_prev] Wz + bz)          update gate: how much of the old state to KEEP
    r = sigmoid([x, h_prev] Wr + br)          reset gate: how much of the old state the candidate may SEE
    n = tanh(x Wxn + (r * h_prev) Whn + bn)   candidate
    h = z * h_prev + (1 - z) * n

The keep-path is h = z * h_prev + ...: the update gate plays the forget
gate's role, on the hidden state itself.

What is built and measured, in the same order as Days 63-65:

  1. gru_step_forward, traced by hand and checked against the formulas
     written out separately; its weight count against the LSTM's.
  2. gru_forward (with padding and a cache) and gru_backward, gradient-
     checked on their own and through the classifier that is trained.
  3. The direct route: with the gates cut off from h, the gradient at h_0
     is exactly the gradient at h_T times the product of the update gates.
  4. How much gradient reaches the first input before training, by the
     update gate's bias.
  5. Training on Day 59's sign task with exactly Day 65's grid (lengths
     20/40/100, keep-bias 0/1/2, lr 0.05/0.2, seeds 7/8/9), so the table
     sits next to Day 65's LSTM table.
     A follow-up beyond the grid separates "slower" from "unable", and
     finds where the GRU's design costs it: its one update gate both keeps
     the old state and admits the new one, so a high keep-bias also blocks
     writing.
  6. What one step costs, against the LSTM.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import time                                            # time: perf_counter, to time one step of each cell

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch, Circle   # rounded boxes and circles, for the cell diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

# =============================================================================
# 0. REUSED -- verbatim from Days 57-65
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


def lstm_step_forward(x_t, h_prev, c_prev, W, b):       # one LSTM step -- verbatim from Day 63, used only to time it in Section 6
    H = h_prev.shape[1]                                   # hidden size
    a = np.concatenate([x_t, h_prev], axis=1) @ W + b     # input and old hidden state side by side, times ONE weight matrix
    f = sigmoid(a[:, 0 * H:1 * H])                        # forget gate
    i = sigmoid(a[:, 1 * H:2 * H])                        # input gate
    g = np.tanh(a[:, 2 * H:3 * H])                        # candidate
    o = sigmoid(a[:, 3 * H:4 * H])                        # output gate
    c = f * c_prev + i * g                                # new cell state
    h = o * np.tanh(c)                                    # new hidden state
    return h, c, (f, i, g, o)                             # both states, and the gates


# =============================================================================
# 1. THE GRU CELL -- NEW today
# =============================================================================
def gru_step_forward(x_t, h_prev, W, b):                # NEW: one step of a GRU cell
    D, H = x_t.shape[1], h_prev.shape[1]                  # input size, hidden size
    zr = np.concatenate([x_t, h_prev], axis=1) @ W[:, :2 * H] + b[:2 * H]  # raw scores of the two gates, in one product
    z = sigmoid(zr[:, :H])                                # update gate: how much of the old state to KEEP (0..1)
    r = sigmoid(zr[:, H:])                                # reset gate: how much of the old state the candidate may SEE (0..1)
    n = np.tanh(x_t @ W[:D, 2 * H:] + (r * h_prev) @ W[D:, 2 * H:] + b[2 * H:])  # candidate, from x and the RESET old state
    h = z * h_prev + (1.0 - z) * n                        # keep part of the old state, fill the rest with the candidate
    return h, (z, r, n)                                   # the one new state, and the gates for inspection


def init_gru(input_dim, hidden_dim, seed, keep_bias=0.0):  # NEW: random GRU weights, columns in z, r, n order
    rng = np.random.RandomState(seed)                     # private rng -> reproducible
    W = rng.randn(input_dim + hidden_dim, 3 * hidden_dim) / np.sqrt(input_dim + hidden_dim)  # one block per gate
    b = np.zeros(3 * hidden_dim)                          # biases
    b[:hidden_dim] = keep_bias                            # the update gate's bias: today's version of the forget bias
    return W, b                                           # W: (input + hidden, 3 * hidden);  b: (3 * hidden,)


# =============================================================================
# 2. FORWARD AND BACKWARD OVER WHOLE SEQUENCES -- NEW today
# =============================================================================
def gru_forward(X, h0, W, b, mask=None):                # NEW: the GRU loop, with padding and a cache (Day 64's pattern)
    m, T, D = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    if mask is None:                                      # no padding: every position is real
        mask = np.ones((m, T))
    Hs = np.zeros((m, T, H))                              # every hidden state
    Z, R, N = (np.zeros((m, T, H)) for _ in range(3))     # every gate value, for the backward pass
    h = h0                                                # starting state
    for t in range(T):                                    # one step at a time
        h_new, (z, r, n) = gru_step_forward(X[:, t], h, W, b)  # one GRU step
        m_t = mask[:, t:t + 1]                              # 1 = real step, 0 = padding
        h = m_t * h_new + (1.0 - m_t) * h                   # padding carries the state forward (one state, so one line)
        Hs[:, t] = h                                        # store it
        Z[:, t], R[:, t], N[:, t] = z, r, n                 # store the gates
    cache = dict(X=X, h0=h0, W=W, b=b, mask=mask, H=Hs, Z=Z, R=R, N=N)  # everything backward needs
    return Hs, cache                                      # all hidden states, and the cache


def gru_backward(dH, dh_last, cache):                   # NEW: walk the cache backwards; returns gradients for W, b, X, h0
    X, h0, W, mask = cache["X"], cache["h0"], cache["W"], cache["mask"]  # what the forward pass saw
    Hs, Z, R, N = cache["H"], cache["Z"], cache["R"], cache["N"]  # what it computed
    m, T, D = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    dW, db, dX = np.zeros_like(W), np.zeros(W.shape[1]), np.zeros_like(X)  # accumulators
    dh = dh_last.copy()                                   # gradient arriving at the final hidden state
    for t in reversed(range(T)):                          # backward through time
        dh = dh + dH[:, t]                                  # plus whatever the loss sends straight to h_t
        m_t = mask[:, t:t + 1]                              # 1 = real step, 0 = padding
        h_prev = Hs[:, t - 1] if t > 0 else h0              # the state that fed step t
        z, r, n, x = Z[:, t], R[:, t], N[:, t], X[:, t]     # this step's gates and input
        d = dh * m_t                                        # only a real step passes gradient into its own computation
        dz = d * (h_prev - n)                               # h = z*h_prev + (1-z)*n  ->  dL/dz
        dn = d * (1.0 - z)                                  # ... dL/dn
        dh_prev = d * z + dh * (1.0 - m_t)                  # THE DIRECT ROUTE: times the update gate; padding passes dh through
        da_n = dn * (1.0 - n ** 2)                          # undo the candidate's tanh
        rh = r * h_prev                                     # what the candidate's hidden weights multiplied
        dW[:D, 2 * H:] += x.T @ da_n                        # candidate, input weights
        dW[D:, 2 * H:] += rh.T @ da_n                       # candidate, hidden weights
        db[2 * H:] += da_n.sum(axis=0)                      # candidate bias
        d_rh = da_n @ W[D:, 2 * H:].T                       # gradient for (r * h_prev)
        dr = d_rh * h_prev                                  # ... split into the reset gate
        dh_prev += d_rh * r                                 # ... and the old state, through the reset gate
        da_zr = np.concatenate([dz * z * (1.0 - z), dr * r * (1.0 - r)], axis=1)  # undo the two gates' sigmoids
        zin = np.concatenate([x, h_prev], axis=1)           # what the gates' weights multiplied
        dW[:, :2 * H] += zin.T @ da_zr                      # gate weights
        db[:2 * H] += da_zr.sum(axis=0)                     # gate biases
        dzin = da_zr @ W[:, :2 * H].T                       # back through the gates' product
        dX[:, t] = dzin[:, :D] + da_n @ W[:D, 2 * H:].T     # the input reached both the gates and the candidate
        dh = dh_prev + dzin[:, D:]                          # every route back to h_prev, added up
    return dW, db, dX, dh                                 # after the loop, dh is the gradient at h0


class GRUClassifier:                                    # NEW: a GRU read at its last state -- Day 65's LSTMClassifier with the cell swapped
    def __init__(self, input_dim, hidden_dim, n_classes, seed=7, keep_bias=1.0, lr=0.05, clip=1.0):
        self.W, self.b = init_gru(input_dim, hidden_dim, seed, keep_bias)  # GRU weights
        rng = np.random.RandomState(seed + 1)              # a separate rng for the output layer (as on Day 65)
        self.Why = rng.randn(hidden_dim, n_classes) / np.sqrt(hidden_dim)  # output weights
        self.by = np.zeros(n_classes)                      # output bias
        self.H, self.lr, self.clip = hidden_dim, lr, clip  # sizes and training settings

    def forward(self, X, mask=None):                       # X: (m, T, D) -> class probabilities
        self.Hs, self.cache = gru_forward(X, np.zeros((X.shape[0], self.H)), self.W, self.b, mask)  # the GRU loop
        self.h_last = self.Hs[:, -1]                         # the state after the last real step
        self.probs = softmax(self.h_last @ self.Why + self.by)  # one prediction per sequence
        return self.probs                                    # hand back

    def loss(self, y):                                     # mean cross-entropy
        return -np.mean(np.log(self.probs[np.arange(len(y)), y] + 1e-12))

    def gradients(self, y):                                # gradients of the mean loss for W, b, Why, by (and dX)
        m = len(y)                                           # batch size
        dlogits = self.probs.copy()                          # probabilities ...
        dlogits[np.arange(m), y] -= 1.0                      # ... minus one-hot
        dlogits /= m                                         # averaged over the batch
        grads = dict(Why=self.h_last.T @ dlogits, by=dlogits.sum(axis=0))  # output layer
        grads["W"], grads["b"], self.dX, _ = gru_backward(np.zeros_like(self.Hs), dlogits @ self.Why.T, self.cache)
        return grads                                         # dict: parameter name -> gradient

    def train_step(self, X, y):                            # one SGD step with Day 60's clipping
        self.forward(X)                                      # forward pass
        grads, _, _ = clip_by_global_norm(self.gradients(y), self.clip)  # gradients, clipped
        for name, g in grads.items():                        # plain SGD
            getattr(self, name)[...] -= self.lr * g            # in place

    def accuracy(self, X, y):                              # classification accuracy
        return (self.forward(X).argmax(axis=1) == y).mean()  # fraction correct


# =============================================================================
# DEMOS
# =============================================================================
def numeric_grad(f, P, eps=1e-5):                       # central-difference gradient of f() w.r.t. array P -- Day 65
    numeric = np.zeros_like(P)                            # one slope per entry
    for pos in np.ndindex(P.shape):                       # every entry
        old = P[pos]                                        # remember it
        P[pos] = old + eps; up = f()                        # nudge up, measure
        P[pos] = old - eps; down = f()                      # nudge down, measure
        P[pos] = old                                        # restore
        numeric[pos] = (up - down) / (2 * eps)              # the slope
    return numeric                                        # same shape as P


def report(name, numeric, analytic):                    # one row of a gradient-check table -- Day 65
    diff = np.abs(numeric - analytic)                     # absolute disagreement
    rel = diff / (np.abs(numeric) + np.abs(analytic) + 1e-12)  # relative disagreement
    ok = bool(np.all(diff <= 1e-10 + 1e-6 * np.abs(numeric)))  # Day 62's pass rule
    print(f"{name:>7}{numeric.size:>9}{rel.max():>20.2e}{diff.max():>18.2e}{str(ok):>8}")
    return ok                                             # True if every entry passes


def demo_one_step():
    print("=" * 70)
    print("1. ONE STEP OF A GRU CELL -- traced by hand, then checked")
    print("=" * 70)
    print("One input, one hidden unit, hand-chosen weights. Each line is computed with plain "
          "arithmetic and compared with gru_step_forward. Then a random case is checked "
          "against the formulas written out gate by gate.")
    x, h_prev = 1.0, 0.5                                       # this step's input and the old hidden state
    W = np.array([[0.5, -1.0, 2.0],                            # weights from x to z, r, n
                  [1.0, 0.4, -0.6]])                           # weights from h (for z and r) and from r*h (for n)
    b = np.array([0.1, 0.0, -0.3])                             # one bias per gate
    z = sigmoid(x * W[0, 0] + h_prev * W[1, 0] + b[0])         # update gate
    r = sigmoid(x * W[0, 1] + h_prev * W[1, 1] + b[1])         # reset gate
    n = np.tanh(x * W[0, 2] + (r * h_prev) * W[1, 2] + b[2])   # candidate: note r * h_prev
    h = z * h_prev + (1 - z) * n                               # new state
    h_fn, (z_fn, r_fn, n_fn) = gru_step_forward(np.array([[x]]), np.array([[h_prev]]), W, b)
    print(f"\ninputs: x = {x}, old hidden state h = {h_prev}")
    print(f"\n{'quantity':<26}{'by hand':>11}{'function':>11}")
    for name, hand, fn in [("update gate z (sigmoid)", z, z_fn), ("reset gate r (sigmoid)", r, r_fn),
                           ("candidate n (tanh)", n, n_fn), ("new hidden state h", h, h_fn)]:
        print(f"{name:<26}{hand:>11.4f}{fn.item():>11.4f}")
    print(f"  new h = z * old h + (1 - z) * n = {z:.4f} * {h_prev} + {1 - z:.4f} * ({n:.4f}) = {h:.4f}")
    print("Actual output from this lesson's run.")

    rng = np.random.RandomState(0)                             # a bigger, random case
    D, H, m = 3, 4, 5                                          # 3 inputs, 4 hidden units, 5 examples
    W, b = init_gru(D, H, seed=1)                              # random weights
    b = rng.randn(3 * H) * 0.5                                 # random biases
    x_t, h0 = rng.randn(m, D), rng.randn(m, H)                 # random input and state
    h_fn, _ = gru_step_forward(x_t, h0, W, b)                  # the function under test
    Wx, Wh = W[:D], W[D:]                                      # input rows and hidden rows
    z = sigmoid(x_t @ Wx[:, :H] + h0 @ Wh[:, :H] + b[:H])      # the textbook formulas, one gate at a time
    r = sigmoid(x_t @ Wx[:, H:2 * H] + h0 @ Wh[:, H:2 * H] + b[H:2 * H])
    n = np.tanh(x_t @ Wx[:, 2 * H:] + (r * h0) @ Wh[:, 2 * H:] + b[2 * H:])
    print(f"\nrandom check ({m} examples, {D} inputs, {H} hidden units): max difference from the gate-by-gate "
          f"formulas {np.max(np.abs(h_fn - (z * h0 + (1 - z) * n))):.2e}")
    print(f"\n{'input':>6}{'hidden':>8}{'RNN weights':>13}{'GRU weights':>13}{'LSTM weights':>14}")
    for D, H in [(1, 16), (32, 32), (64, 128)]:                 # Days 59, 62 and a larger size
        print(f"{D:>6}{H:>8}{(D + H + 1) * H:>13}{3 * (D + H + 1) * H:>13}{4 * (D + H + 1) * H:>14}")
    print("Actual output from this lesson's run.")


def demo_gradient_checks():
    print("\n" + "=" * 70)
    print("2. FORWARD AND BACKWARD -- gradient checks, on their own and through the classifier")
    print("=" * 70)
    print("(a) 4 sequences of lengths 5, 2, 4, 1 (padded to 5), 2 inputs, 3 hidden units, random "
          "weights and biases, random non-zero h0. Loss L = sum(R1 * every h_t) + sum(R2 * last "
          "h). (b) GRUClassifier through its mean cross-entropy. Day 62's pass rule.")
    rng = np.random.RandomState(3)                             # private rng
    m, T, D, H = 4, 5, 2, 3                                    # small sizes
    lens = np.array([5, 2, 4, 1])                              # four different lengths
    mask = (np.arange(T)[None, :] < lens[:, None]).astype(float)  # 1 = real step
    X = rng.randn(m, T, D)                                     # inputs
    h0 = rng.randn(m, H) * 0.5                                 # non-zero starting state
    W, b = init_gru(D, H, seed=1, keep_bias=1.0)               # weights
    b += rng.randn(3 * H) * 0.3                                # random biases too
    R1, R2 = rng.randn(m, T, H), rng.randn(m, H)               # fixed random weightings of the outputs

    def loss():                                                # the scalar being differentiated
        Hs, _ = gru_forward(X, h0, W, b, mask)                   # forward pass
        return np.sum(R1 * Hs * mask[:, :, None]) + np.sum(R2 * Hs[:, -1])

    _, cache = gru_forward(X, h0, W, b, mask)                  # forward once for the analytic gradients
    dW, db, dX, dh0 = gru_backward(R1 * mask[:, :, None], R2, cache)  # the gradients under test
    print(f"\n(a) gru_backward on its own:")
    print(f"{'array':>7}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>8}")
    ok = all([report(name, numeric_grad(loss, P), A) for name, P, A in
              [("W", W, dW), ("b", b, db), ("X", X, dX), ("h0", h0, dh0)]])
    print(f"every entry passes: {ok};  gradient at padded inputs exactly zero: {np.all(dX[mask == 0] == 0)}")

    net = GRUClassifier(D, H, 2, seed=1, keep_bias=1.0, clip=None)  # tiny classifier
    net.b += rng.randn(3 * H) * 0.3                            # random biases
    y = np.array([0, 1, 1, 0])                                 # labels
    f = lambda: (net.forward(X, mask), net.loss(y))[1]         # forward, then return only the loss
    net.forward(X, mask); grads = net.gradients(y)             # analytic gradients
    print(f"\n(b) through GRUClassifier:")
    print(f"{'array':>7}{'entries':>9}{'max relative error':>20}{'max |difference|':>18}{'pass':>8}")
    ok = all([report(name, numeric_grad(f, getattr(net, name)), grads[name]) for name in ["W", "b", "Why", "by"]])
    print(f"every entry passes: {ok}")
    print("Actual output from this lesson's run.")


def demo_direct_route():
    print("\n" + "=" * 70)
    print("3. THE DIRECT ROUTE -- the update gate plays the forget gate's part")
    print("=" * 70)
    print("Going back one step, h = z * h_prev + ... sends dh * z straight to h_prev. With the "
          "gates and the candidate cut off from h (W's hidden rows set to zero), that is the only "
          "route from h_T back to h_0, so dL/dh_0 must be dL/dh_T times the product of the update "
          "gates. 16 units, 200 sequences of 30 N(0,1) inputs, keep-bias 2.")
    m, T, H = 200, 30, 16                                      # sizes
    X = np.random.RandomState(0).randn(m, T, 1)                # inputs
    W, b = init_gru(1, H, seed=3, keep_bias=2.0)               # weights
    dh_T = np.random.RandomState(1).randn(m, H)                # some gradient arriving at the last state
    for label, cut in [("gates cut off from h", True), ("normal GRU", False)]:
        Wc = W.copy()                                            # never change the original weights
        if cut:                                                  # remove every connection from h
            Wc[1:, :] = 0.0
        _, cache = gru_forward(X, np.zeros((m, H)), Wc, b)       # forward
        dh0 = gru_backward(np.zeros((m, T, H)), dh_T, cache)[3]  # gradient only enters at h_T
        direct = dh_T * np.prod(cache["Z"], axis=1)              # the direct route alone
        print(f"\n{label}: ||dL/dh_0|| = {np.linalg.norm(dh0):.6e},  direct route alone = {np.linalg.norm(direct):.6e},  "
              f"max |difference| = {np.max(np.abs(dh0 - direct)):.2e}")
    print("Actual output from this lesson's run.")


def demo_gradient_reach():
    print("\n" + "=" * 70)
    print("4. HOW MUCH GRADIENT REACHES THE FIRST INPUT -- before any training")
    print("=" * 70)
    print("Day 65's measurement for the GRU: untrained GRUClassifier (16 units, seed 7), Day 59's "
          "400 test sequences, ||dL/dx_0|| / ||dL/dx_(T-1)||, by the update gate's bias.")
    print(f"\n{'T':>5}" + "".join(f"{'keep bias ' + str(kb):>16}" for kb in [0.0, 1.0, 2.0]))
    table = {}                                                 # (T, keep bias) -> ratio
    for T in [20, 40, 100]:                                    # three lengths
        X, y = make_sign_task(400, T, seed=999)                 # Day 59's test sequences
        row = []                                                 # one ratio per bias
        for kb in [0.0, 1.0, 2.0]:                               # three keep-biases
            net = GRUClassifier(1, 16, 2, seed=7, keep_bias=kb)    # untrained
            net.forward(X[:, :, None]); net.gradients(y)           # forward and backward
            norms = np.linalg.norm(net.dX[:, :, 0], axis=0)        # gradient size at every input position
            table[(T, kb)] = norms[0] / norms[-1]                  # first relative to last
            row.append(table[(T, kb)])
        print(f"{T:>5}" + "".join(f"{v:>16.3e}" for v in row))
    print("Actual output from this lesson's run.")
    return table                                               # for the plot


def train_runs(T, kb, lr, seeds=(7, 8, 9), epochs=150):  # train one setting for several seeds; returns (peak, final) per seed
    X_tr, y_tr = make_sign_task(600, T, seed=42)           # Day 59's training data
    X_te, y_te = make_sign_task(400, T, seed=999)          # Day 59's test data
    X_tr, X_te = X_tr[:, :, None], X_te[:, :, None]        # one input per step
    runs = []                                              # one entry per seed
    for seed in seeds:                                     # each seed
        net = GRUClassifier(1, 16, 2, seed=seed, keep_bias=kb, lr=lr, clip=1.0)  # fresh network
        rng = np.random.RandomState(seed)                    # batch order
        curve = []                                           # test accuracy during training
        for epoch in range(epochs):                          # the training budget
            order = rng.permutation(600)                       # fresh shuffle
            for start in range(0, 600, 30):                    # batches of 30
                idx = order[start:start + 30]                    # this batch
                net.train_step(X_tr[idx], y_tr[idx])             # one step
            if epoch % 10 == 0 or epoch == epochs - 1:         # every 10 epochs, and the last
                curve.append(net.accuracy(X_te, y_te))           # test accuracy NOW
        runs.append((max(curve), curve[-1]))                   # peak and final
    return runs                                            # [(peak, final), ...]


def demo_training():
    print("\n" + "=" * 70)
    print("5. TRAINING ON DAY 59'S SIGN TASK -- exactly Day 65's grid, with the GRU")
    print("=" * 70)
    print("16 units, Day 59's 600/400 sequences, batch 30, 150 epochs, clipping at 1.0, lr 0.05 "
          "and 0.2, keep-bias 0, 1, 2, seeds 7, 8, 9. Test accuracy every 10 epochs. 'Learned' = "
          "final test accuracy >= 0.9. Day 65's LSTM, same grid, learned 0 of 18 runs at bias 0, "
          "10 of 18 at bias 1, 16 of 18 at bias 2.")
    results = {}                                               # (T, keep bias, lr) -> list of (peak, final) per seed
    print(f"\n{'T':>4}{'k-bias':>8}{'lr':>6}   {'peak / final test accuracy, seeds 7, 8, 9':<44}{'final >= 0.9':>13}")
    for T in [20, 40, 100]:                                    # three lengths
        for kb in [0.0, 1.0, 2.0]:                               # three keep-biases
            for lr in [0.05, 0.2]:                                 # two learning rates
                runs = train_runs(T, kb, lr)                         # three seeds
                results[(T, kb, lr)] = runs                          # keep
                cells = "  ".join(f"{p:.3f}/{f:.3f}" for p, f in runs)  # peak/final per seed
                print(f"{T:>4}{kb:>8}{lr:>6}   {cells:<44}{sum(f >= 0.9 for _, f in runs):>9} of 3")
    learned = {kb: sum(f >= 0.9 for (T, k, lr), runs in results.items() if k == kb for _, f in runs) for kb in [0.0, 1.0, 2.0]}
    print(f"\nruns that learned, out of 18 per keep-bias: " + ", ".join(f"bias {kb}: {n}" for kb, n in learned.items()))
    print("Actual output from this lesson's run.")
    return results                                             # for the plot


def demo_follow_up():
    print("\n" + "=" * 70)
    print("5b. FOLLOW-UP -- slower, or unable?")
    print("=" * 70)
    print("Where the GRU fell short of Day 65's LSTM: 40 steps at Day 60's learning rate 0.05, and "
          "every run at 100 steps. Settings beyond the grid, one change at a time, seeds 7, 8, 9: "
          "40 steps at lr 0.05 with three times as many epochs; 40 and 100 steps with keep-bias 3 "
          "and 4 at lr 0.2. First: at the first step, how much of the candidate the cell can write, "
          "(1 - z), compared with the LSTM, whose input gate is separate from its forget gate.")
    X, _ = make_sign_task(400, 40, seed=999)                    # the test sequences
    print(f"\n{'keep bias':>10}{'GRU keeps z':>13}{'GRU writes 1 - z':>18}{'LSTM keeps f':>14}{'LSTM writes i':>15}")
    for kb in [2.0, 3.0, 4.0]:                                  # three keep-biases
        W, b = init_gru(1, 16, seed=7, keep_bias=kb)              # the seed-7 GRU at this bias
        _, (z, _, _) = gru_step_forward(X[:, :1], np.zeros((400, 16)), W, b)  # gates at the first step
        print(f"{kb:>10}{z.mean():>13.3f}{1 - z.mean():>18.3f}{sigmoid(kb):>14.3f}{sigmoid(0.0):>15.3f}")
    print("(the LSTM columns are its gates at a raw score of the bias alone: f = sigmoid(bias), i = sigmoid(0))")
    print(f"\n{'setting':<44}{'peak / final, seeds 7, 8, 9':<40}{'final >= 0.9':>13}")
    rows = {}                                                  # label -> runs
    for label, T, kb, lr, epochs in [("T=40, keep-bias 2, lr 0.05, 450 epochs", 40, 2.0, 0.05, 450),
                                     ("T=40, keep-bias 3, lr 0.2", 40, 3.0, 0.2, 150),
                                     ("T=40, keep-bias 4, lr 0.2", 40, 4.0, 0.2, 150),
                                     ("T=100, keep-bias 3, lr 0.2", 100, 3.0, 0.2, 150),
                                     ("T=100, keep-bias 4, lr 0.2", 100, 4.0, 0.2, 150)]:
        runs = train_runs(T, kb, lr, epochs=epochs)               # three seeds
        rows[label] = runs                                        # keep
        cells = "  ".join(f"{p:.3f}/{f:.3f}" for p, f in runs)    # peak/final per seed
        print(f"{label:<44}{cells:<40}{sum(f >= 0.9 for _, f in runs):>9} of 3")
    print("Actual output from this lesson's run.")


def demo_cost():
    print("\n" + "=" * 70)
    print("6. WHAT ONE STEP COSTS -- GRU against LSTM")
    print("=" * 70)
    print("20,000 single steps, input 32, hidden 32, batch 32, matrix library on one thread, best "
          "of 5 timings. Timings vary by machine.")
    rng = np.random.RandomState(0)                             # random data
    x, h, c = rng.randn(32, 32), rng.randn(32, 32), rng.randn(32, 32)  # one batch
    Wg, bg = init_gru(32, 32, seed=0)                          # GRU weights
    Wl, bl = rng.randn(64, 128) / 8.0, np.zeros(128)           # LSTM weights of the same sizes

    def best(fn):                                              # fastest of 5 timings of 20,000 calls
        times = []                                               # one entry per repeat
        for _ in range(5):                                       # 5 repeats
            start = time.perf_counter()                            # start the clock
            for _ in range(20000):                                 # 20,000 steps
                fn()
            times.append(time.perf_counter() - start)              # seconds
        return min(times)                                        # least disturbed run

    t_gru = best(lambda: gru_step_forward(x, h, Wg, bg))       # GRU step
    t_lstm = best(lambda: lstm_step_forward(x, h, c, Wl, bl))  # LSTM step
    print(f"\nGRU {t_gru:.3f} s, LSTM {t_lstm:.3f} s for 20,000 steps; GRU / LSTM time ratio {t_gru / t_lstm:.2f}")
    print("Actual output from this lesson's run.")


def plot_results(reach, results):                       # two panels, in Day 65's layout
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))       # side by side
    colors = {0.0: "#9ecae1", 1.0: "#4292c6", 2.0: "#08519c"}   # blues for the GRU (the LSTM's were oranges)
    for kb in [0.0, 1.0, 2.0]:                                   # left: gradient reach before training
        a1.semilogy([20, 40, 100], [reach[(T, kb)] for T in [20, 40, 100]], "o-", color=colors[kb], label=f"keep bias {kb}")
    a1.axhline(1.0, color="black", linewidth=0.8)
    a1.set_xlabel("sequence length T"); a1.set_ylabel("||dL/dx_0|| / ||dL/dx_(T-1)||")
    a1.set_title("GRU: gradient reaching the first input, before training", fontsize=10); a1.legend(fontsize=8)
    x = 0                                                         # right: final accuracy of every run
    ticks, labels = [], []                                        # tick positions and labels
    for T in [20, 40, 100]:                                       # one group per length
        for lr in [0.05, 0.2]:                                      # one sub-group per learning rate
            for j, kb in enumerate([0.0, 1.0, 2.0]):                  # one column per keep-bias
                finals = [f for _, f in results[(T, kb, lr)]]           # three seeds
                a2.scatter([x + j * 0.25] * 3, finals, color=colors[kb], s=22, zorder=3, label=f"keep bias {kb}" if (T, lr) == (20, 0.05) else None)
            ticks.append(x + 0.25); labels.append(f"T={T}\nlr={lr}")
            x += 1.2                                                  # gap before the next sub-group
    a2.axhline(0.5, color="gray", linestyle="--", linewidth=1)     # chance
    a2.set_xticks(ticks); a2.set_xticklabels(labels, fontsize=8); a2.set_ylim(0.4, 1.02)
    a2.set_ylabel("final test accuracy"); a2.set_title("GRU: final test accuracy, every run (3 seeds each)", fontsize=10)
    a2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # outside the axes, so no dot is hidden
    plt.tight_layout(); plt.savefig("gru_training_by_keep_bias.png", dpi=110); plt.close()  # save and free
    print("\nsaved gru_training_by_keep_bias.png")


def draw_cell_diagram():                                # one step of a GRU cell, in Day 63's style
    fig, ax = plt.subplots(figsize=(13, 6.2))                   # wide canvas
    ax.set_xlim(0, 13.4); ax.set_ylim(0, 6.2); ax.axis("off")    # plain coordinate grid

    def box(x, y, w, h, title, sub, color):                      # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x, y + (0.17 if sub else 0.0), title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8.5)

    def op(x, y, symbol):                                        # a small circle with an operation symbol
        ax.add_patch(Circle((x, y), 0.27, facecolor="white", edgecolor="#333333", linewidth=1.4))
        ax.text(x, y, symbol, ha="center", va="center", fontsize=12, fontweight="bold")

    def arrow(x0, y0, x1, y1, color="#333333", width=1.5):       # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=width))

    ax.text(6.7, 5.9, "One step of a GRU cell: one state, two gates, one candidate", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.75, 4.9, "h(t-1)\nold state", ha="center", va="center", fontsize=9.5, color="#08519c", fontweight="bold")
    ax.text(12.65, 4.9, "h(t)\nnew state", ha="center", va="center", fontsize=9.5, color="#08519c", fontweight="bold")
    arrow(1.45, 4.9, 3.33, 4.9, "#08519c", 2.6); op(3.6, 4.9, "x")    # the old state times z ...
    arrow(3.87, 4.9, 10.13, 4.9, "#08519c", 2.6); op(10.4, 4.9, "+")  # ... plus (1 - z) * n ...
    arrow(10.67, 4.9, 11.95, 4.9, "#08519c", 2.6)                     # ... is the new state
    ax.text(6.9, 5.12, "the keep path: z * h(t-1), no weight matrix, no squashing", ha="center", fontsize=8, color="#08519c")
    box(3.6, 2.6, 2.1, 1.0, "update gate z", "sigmoid: keep 0 to 1", "#FDECEC")
    box(6.3, 2.2, 2.0, 1.0, "reset gate r", "sigmoid: 0 to 1", "#EAF3DE")
    box(9.0, 2.2, 2.0, 1.0, "candidate n", "tanh: -1 to 1", "#E6F1F8")
    arrow(3.6, 3.14, 3.6, 4.62)                                       # z up to the keep-path multiply
    ax.plot([4.4, 4.9, 10.0], [3.14, 4.25, 4.25], color="#B03A3A", linewidth=1.5)  # 1 - z, routed above the boxes
    arrow(10.0, 4.25, 10.22, 3.66, "#B03A3A")                         # ... into (1 - z) * n
    ax.text(6.1, 4.37, "1 - z", ha="center", fontsize=8.5, color="#B03A3A")
    arrow(7.6, 4.88, 7.6, 3.68, "#08519c")                            # h(t-1) branches down ...
    op(7.6, 3.4, "x")                                                 # ... into r * h(t-1)
    ax.text(7.22, 3.88, "r * h(t-1)", ha="right", fontsize=8)
    arrow(6.6, 2.74, 7.38, 3.22)                                      # r into r * h(t-1)
    arrow(7.82, 3.22, 8.55, 2.74)                                     # r * h(t-1) into the candidate
    op(10.4, 3.4, "x")                                                # (1 - z) * n
    ax.text(10.75, 3.4, "(1 - z) * n", ha="left", va="center", fontsize=8)
    arrow(9.6, 2.74, 10.2, 3.18)                                      # n into (1 - z) * n
    arrow(10.4, 3.68, 10.4, 4.62)                                     # ... and up into the add
    box(5.8, 0.75, 7.6, 0.8, "x(t) and h(t-1), side by side", "feed both gates; x(t) also feeds the candidate", "#F2F2F2")
    arrow(3.6, 1.19, 3.6, 2.06)                                       # shared input up to the update gate
    arrow(6.3, 1.19, 6.3, 1.66)                                       # ... to the reset gate
    arrow(9.0, 1.19, 9.0, 1.66)                                       # x(t) into the candidate
    plt.savefig("gru_cell_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("\nsaved gru_cell_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Three recurrent cells now exist with checked gradients: the vanilla RNN, the LSTM and "
          "the GRU. Day 67 compares them trained on real data, Week 9's part-of-speech tagging, "
          "instead of a task built to need long memory.")


def main():                                             # runs every section in order
    with threadpool_limits(limits=1, user_api="blas"):     # one thread for NumPy's matrix library: faster here, same results
        demo_one_step()                                      # 1. the cell, by hand and checked
        demo_gradient_checks()                               # 2. forward and backward, checked
        demo_direct_route()                                  # 3. the direct route through z
        reach = demo_gradient_reach()                        # 4. gradient reaching x_0 before training
        results = demo_training()                            # 5. Day 65's training grid, GRU
        demo_follow_up()                                     # 5b. 40 steps: too slow, or unable?
        demo_cost()                                          # 6. time per step
        plot_results(reach, results)                         # the summary plot
        draw_cell_diagram()                                  # the diagram used in the guides
        note_on_what_comes_next()                            # 7. bridge to Day 67


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
