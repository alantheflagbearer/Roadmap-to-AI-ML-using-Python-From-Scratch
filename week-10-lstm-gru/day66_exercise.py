"""
Day 66 Exercise -- The GRU Cell, Isolated

The main lesson's new cell on its own: one GRU step, run on one small
example and checked against the same step written out gate by gate,
like Day 63's exercise for the LSTM. No sequences, no training.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sigmoid(z):                                         # squashes any number into the range 0..1
    return 1.0 / (1.0 + np.exp(-z))


def gru_step_forward(x_t, h_prev, W, b):                # ONE timestep of a GRU cell; W's columns are [z | r | n]
    D, H = x_t.shape[1], h_prev.shape[1]                  # input size, hidden size
    zr = np.concatenate([x_t, h_prev], axis=1) @ W[:, :2 * H] + b[:2 * H]  # both gates' raw scores in one product
    z = sigmoid(zr[:, :H])                                # update gate: share of the old state to keep
    r = sigmoid(zr[:, H:])                                # reset gate: share of the old state the candidate may see
    n = np.tanh(x_t @ W[:D, 2 * H:] + (r * h_prev) @ W[D:, 2 * H:] + b[2 * H:])  # candidate
    h = z * h_prev + (1.0 - z) * n                        # keep part, write the rest
    return h, (z, r, n)                                   # the new state, and the gates


rng = np.random.RandomState(0)                          # fixed rng
D, H = 3, 2                                               # 3 inputs, 2 hidden units
W = rng.randn(D + H, 3 * H) * 0.5                         # one weight matrix: three blocks of H columns
b = rng.randn(3 * H) * 0.1                                # one bias per gate per unit
x_t = rng.randn(1, D)                                     # one example's input
h_prev = rng.randn(1, H) * 0.5                            # the old state

h, (z, r, n) = gru_step_forward(x_t, h_prev, W, b)        # the function being checked

Wx, Wh = W[:D], W[D:]                                     # rows for x, rows for h
z_hand = sigmoid(x_t @ Wx[:, 0:2] + h_prev @ Wh[:, 0:2] + b[0:2])  # update gate, written out
r_hand = sigmoid(x_t @ Wx[:, 2:4] + h_prev @ Wh[:, 2:4] + b[2:4])  # reset gate
n_hand = np.tanh(x_t @ Wx[:, 4:6] + (r_hand * h_prev) @ Wh[:, 4:6] + b[4:6])  # candidate
h_hand = z_hand * h_prev + (1 - z_hand) * n_hand          # new state by hand

print("update gate z:", z.round(4), "  (share of the old state kept)")
print("reset gate r :", r.round(4))
print("candidate n  :", n.round(4))
print("old state    :", h_prev.round(4))
print("new state    :", h.round(4), "  = z * old + (1 - z) * n")
print(f"\nmax difference from the gate-by-gate version: {np.max(np.abs(h - h_hand)):.2e} "
      f"({'match' if np.max(np.abs(h - h_hand)) < 1e-12 else 'MISMATCH'})")
print(f"weights: GRU {W.size + b.size}, LSTM of the same size {4 * (D + H + 1) * H}, RNN {(D + H + 1) * H}")
