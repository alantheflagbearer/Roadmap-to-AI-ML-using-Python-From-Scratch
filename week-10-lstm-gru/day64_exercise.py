"""
Day 64 Exercise -- The LSTM Forward Pass, Isolated

Day 63's exercise built one LSTM step. This one puts that step inside a
loop over time, carrying BOTH states (h and c) from step to step, and
checks the loop against three steps written out by hand -- exactly what
Day 57's exercise did for the RNN cell. No masking, no timing, no
training: just the loop.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sigmoid(z):                                         # squashes any number into the range 0..1 -- from day63_exercise.py
    return 1.0 / (1.0 + np.exp(-z))


def lstm_step_forward(x_t, h_prev, c_prev, W, b):       # ONE timestep of an LSTM cell -- unchanged from day63_exercise.py
    H = h_prev.shape[1]                                   # hidden size
    a = np.concatenate([x_t, h_prev], axis=1) @ W + b     # all four gates' raw scores in one product
    f = sigmoid(a[:, 0 * H:1 * H])                        # forget gate
    i = sigmoid(a[:, 1 * H:2 * H])                        # input gate
    g = np.tanh(a[:, 2 * H:3 * H])                        # candidate
    o = sigmoid(a[:, 3 * H:4 * H])                        # output gate
    c = f * c_prev + i * g                                # new cell state
    h = o * np.tanh(c)                                    # new hidden state
    return h, c, (f, i, g, o)                             # both states, and the gates


def lstm_forward(X, h0, c0, W, b):                      # NEW: run the step once per timestep
    m, T, _ = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    Hs, Cs = np.zeros((m, T, H)), np.zeros((m, T, H))     # every hidden state and every cell state
    h, c = h0, c0                                         # start from the given states
    for t in range(T):                                    # one step at a time
        h, c, _ = lstm_step_forward(X[:, t, :], h, c, W, b)  # the SAME weights at every step; h AND c carried forward
        Hs[:, t, :], Cs[:, t, :] = h, c                     # store both
    return Hs, Cs                                         # all hidden states and all cell states


rng = np.random.RandomState(0)                          # fixed rng for reproducible numbers
D, H = 2, 3                                               # 2 inputs, 3 hidden units
W = rng.randn(D + H, 4 * H) * 0.5                         # one weight matrix for all four gates
b = rng.randn(4 * H) * 0.1                                # one bias per gate per unit
X = rng.randn(1, 3, D)                                    # one sequence of 3 steps
h0, c0 = np.zeros((1, H)), np.zeros((1, H))               # start from zero

Hs, Cs = lstm_forward(X, h0, c0, W, b)                   # the loop being checked

h1, c1, _ = lstm_step_forward(X[:, 0, :], h0, c0, W, b)   # step 0 by hand
h2, c2, _ = lstm_step_forward(X[:, 1, :], h1, c1, W, b)   # step 1, fed step 0's states
h3, c3, _ = lstm_step_forward(X[:, 2, :], h2, c2, W, b)   # step 2, fed step 1's states

print("hidden states from the loop (one row per step):")
print(Hs[0].round(4))
print("cell states from the loop:")
print(Cs[0].round(4))
max_diff = max(np.max(np.abs(Hs[:, 0] - h1)), np.max(np.abs(Hs[:, 1] - h2)), np.max(np.abs(Hs[:, 2] - h3)),
               np.max(np.abs(Cs[:, 0] - c1)), np.max(np.abs(Cs[:, 1] - c2)), np.max(np.abs(Cs[:, 2] - c3)))
print(f"\nmax difference from the hand-unrolled steps: {max_diff:.2e} ({'match' if max_diff < 1e-12 else 'MISMATCH'})")
