"""
Day 63 Exercise -- The LSTM Cell, Isolated

The main lesson's one new function on its own: a single step of an LSTM
cell. It is run on one small example and checked against the same step
written out gate by gate. No sequences, no training -- Day 64's exercise
will loop this function over time, exactly as Day 57's looped the RNN
cell.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


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
    return h, c, (f, i, g, o)                             # both states, and the gates for inspection


rng = np.random.RandomState(0)                          # fixed rng for reproducible numbers
D, H = 3, 2                                               # 3 inputs, 2 hidden units
W = rng.randn(D + H, 4 * H) * 0.5                         # one weight matrix: (inputs + hidden) rows, four gate blocks of H columns
b = rng.randn(4 * H) * 0.1                                # one bias per gate per unit
x_t = rng.randn(1, D)                                     # one example's input for this step
h_prev = rng.randn(1, H) * 0.5                            # the old hidden state
c_prev = rng.randn(1, H) * 0.5                            # the old cell state

h, c, (f, i, g, o) = lstm_step_forward(x_t, h_prev, c_prev, W, b)  # the function being checked

Wx, Wh = W[:D], W[D:]                                     # the rows that multiply x, and the rows that multiply h
f_hand = sigmoid(x_t @ Wx[:, 0:2] + h_prev @ Wh[:, 0:2] + b[0:2])  # forget gate, written out on its own
i_hand = sigmoid(x_t @ Wx[:, 2:4] + h_prev @ Wh[:, 2:4] + b[2:4])  # input gate
g_hand = np.tanh(x_t @ Wx[:, 4:6] + h_prev @ Wh[:, 4:6] + b[4:6])  # candidate
o_hand = sigmoid(x_t @ Wx[:, 6:8] + h_prev @ Wh[:, 6:8] + b[6:8])  # output gate
c_hand = f_hand * c_prev + i_hand * g_hand                # new cell state by hand
h_hand = o_hand * np.tanh(c_hand)                         # new hidden state by hand

print("forget gate f :", f.round(4))                     # every gate is between 0 and 1 (g between -1 and 1)
print("input gate  i :", i.round(4))
print("candidate   g :", g.round(4))
print("output gate o :", o.round(4))
print("old cell state:", c_prev.round(4))
print("new cell state:", c.round(4), "  = f * old + i * g")
print("new hidden    :", h.round(4), "  = o * tanh(new cell state)")
max_diff = max(np.max(np.abs(c - c_hand)), np.max(np.abs(h - h_hand)))  # largest disagreement with the written-out version
print(f"\nmax difference from the gate-by-gate version: {max_diff:.2e} ({'match' if max_diff < 1e-12 else 'MISMATCH'})")
print(f"weights in this cell: {W.size + b.size}  (a vanilla RNN of the same size has {D * H + H * H + H})")
