"""
Day 65 Exercise -- The LSTM Backward Pass, Isolated

Day 64's exercise looped the LSTM step over time. This one adds the
matching backward pass and checks it against numeric gradients -- the
LSTM version of Day 58's exercise. As there, the "loss" is just the sum
of the final hidden state, the simplest scalar to differentiate, so the
only thing being tested is backpropagation through the LSTM.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sigmoid(z):                                         # squashes any number into the range 0..1 -- from day63_exercise.py
    return 1.0 / (1.0 + np.exp(-z))


def lstm_step_forward(x_t, h_prev, c_prev, W, b):       # ONE timestep -- unchanged from day63_exercise.py
    H = h_prev.shape[1]                                   # hidden size
    a = np.concatenate([x_t, h_prev], axis=1) @ W + b     # all four gates' raw scores in one product
    f = sigmoid(a[:, 0 * H:1 * H])                        # forget gate
    i = sigmoid(a[:, 1 * H:2 * H])                        # input gate
    g = np.tanh(a[:, 2 * H:3 * H])                        # candidate
    o = sigmoid(a[:, 3 * H:4 * H])                        # output gate
    c = f * c_prev + i * g                                # new cell state
    h = o * np.tanh(c)                                    # new hidden state
    return h, c, (f, i, g, o)                             # both states, and the gates


def lstm_forward(X, h0, c0, W, b):                      # the loop from day64_exercise.py, now also keeping the gates
    m, T, _ = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    Hs, Cs = np.zeros((m, T, H)), np.zeros((m, T, H))     # every hidden state and every cell state
    gates = []                                            # (f, i, g, o) at every step, for the backward pass
    h, c = h0, c0                                         # start from the given states
    for t in range(T):                                    # one step at a time
        h, c, fiog = lstm_step_forward(X[:, t, :], h, c, W, b)  # same weights every step
        Hs[:, t, :], Cs[:, t, :] = h, c                     # store both states
        gates.append(fiog)                                  # and the gates
    return Hs, Cs, gates                                  # everything backward needs


def lstm_backward(dh_last, X, h0, c0, W, Hs, Cs, gates):  # NEW: backpropagation through the LSTM
    m, T, D = X.shape                                     # batch size, sequence length, input size
    dW, db = np.zeros_like(W), np.zeros(W.shape[1])       # accumulators, one contribution per step
    dh, dc = dh_last, np.zeros_like(c0)                   # gradient arrives at the last h; none at the last c
    for t in reversed(range(T)):                          # walk BACKWARD through time
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
    return dW, db                                         # summed over every step


rng = np.random.RandomState(0)                          # fixed rng
D, H, T = 2, 3, 4                                         # 2 inputs, 3 hidden units, 4 steps
W = rng.randn(D + H, 4 * H) * 0.5                         # weights
b = rng.randn(4 * H) * 0.1                                # biases
X = rng.randn(2, T, D)                                    # a batch of 2 sequences
h0, c0 = np.zeros((2, H)), np.zeros((2, H))               # start from zero


def loss_fn():                                          # the simplest scalar: sum of the final hidden state
    return lstm_forward(X, h0, c0, W, b)[0][:, -1, :].sum()


Hs, Cs, gates = lstm_forward(X, h0, c0, W, b)            # forward pass
dW, db = lstm_backward(np.ones((2, H)), X, h0, c0, W, Hs, Cs, gates)  # d(sum of h_last)/d h_last = all ones

worst = 0.0                                               # worst relative error seen
for name, param, grad in [("W", W, dW), ("b", b, db)]:
    for idx in np.ndindex(param.shape):                     # check EVERY entry
        orig = param[idx]
        param[idx] = orig + 1e-5; lp = loss_fn()             # nudge up
        param[idx] = orig - 1e-5; lm = loss_fn()             # nudge down
        param[idx] = orig                                    # restore
        numeric = (lp - lm) / 2e-5                           # central difference
        worst = max(worst, abs(numeric - grad[idx]) / max(abs(numeric) + abs(grad[idx]), 1e-8))
    print(f"{name}: all {param.size} entries checked")

print(f"\nmax relative error: {worst:.2e} ({'PASS' if worst < 1e-6 else 'FAIL'})")
