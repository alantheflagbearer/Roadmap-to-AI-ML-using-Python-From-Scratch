"""
Day 63: The LSTM Cell -- Gates Explained, Starting From Day 59's Problem

Week 9 ended on a limit. Day 59 measured it: in a vanilla RNN, whatever
is in the hidden state gets multiplied, at EVERY step, by the recurrent
weights and by tanh's slope. Over 50 steps that product was anywhere
from 1e-22 to 50,000 depending on one number (the spectral radius of
Whh). Yesterday's probe showed it again on a trained network.

The LSTM (Hochreiter & Schmidhuber, 1997) changes the cell itself. It
keeps a second state, the CELL STATE c, that is updated by a rule with no
weight matrix and no squashing in its path:

    c_t = f_t * c_{t-1} + i_t * g_t        (* = entry by entry)
    h_t = o_t * tanh(c_t)

f, i and o are GATES: numbers between 0 and 1, computed from the current
input and the previous hidden state, that decide how much to keep (f),
how much to write (i) and how much to show (o). g is the candidate value
to write.

Today is about ONE cell and ONE step. The forward pass over whole
sequences is Day 64 and training is Day 65. What is built and measured:

  1. lstm_step_forward, traced by hand on one small example and checked
     against a second, spelled-out implementation.
  2. What each gate does, by setting the gates directly.
  3. Store / hold / read: a cell with hand-set WEIGHTS stores one number,
     ignores up to 999 distracting inputs, and gives it back -- and how
     sharp the gates must be for that to work.
  4. Day 59's question, asked of the LSTM: nudge the memory at step 0 and
     measure how much of the nudge is left T steps later. The answer
     depends almost entirely on the forget gate.
  5. What the cell costs: four times the weights, and measured time.

Nothing is trained today, and the lesson says so where it matters.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import time                                            # time: perf_counter, to measure how long a step takes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch, Circle   # rounded boxes and circles, for the cell diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

# =============================================================================
# 0. HELPERS -- two reused from Week 9, one new
# =============================================================================
def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):        # ONE timestep of the vanilla RNN -- verbatim from Day 57
    return np.tanh(x_t @ Wxh + h_prev @ Whh + bh)         # combine this step's input and the previous hidden state, squash with tanh


def spectral_radius(W):                                 # the largest |eigenvalue| of a square matrix -- verbatim from Day 59
    return np.max(np.abs(np.linalg.eigvals(W)))           # governs how repeated multiplication by W grows or shrinks vectors


def sigmoid(z):                                         # NEW in this week: squashes any number into the range 0..1
    return 1.0 / (1.0 + np.exp(-z))                       # large negative -> 0, zero -> 0.5, large positive -> 1


# =============================================================================
# 1. THE LSTM CELL -- one timestep
# =============================================================================
def lstm_step_forward(x_t, h_prev, c_prev, W, b):       # NEW today: one step of an LSTM cell
    H = h_prev.shape[1]                                   # hidden size
    a = np.concatenate([x_t, h_prev], axis=1) @ W + b     # input and old hidden state side by side, times ONE weight matrix
    f = sigmoid(a[:, 0 * H:1 * H])                        # forget gate: how much of the old cell state to KEEP (0..1)
    i = sigmoid(a[:, 1 * H:2 * H])                        # input gate: how much of the candidate to WRITE (0..1)
    g = np.tanh(a[:, 2 * H:3 * H])                        # candidate: the new value that could be written (-1..1)
    o = sigmoid(a[:, 3 * H:4 * H])                        # output gate: how much of the cell state to SHOW (0..1)
    c = f * c_prev + i * g                                # new cell state: keep part of the old, add part of the new
    h = o * np.tanh(c)                                    # new hidden state: a gated, squashed view of the cell state
    return h, c, (f, i, g, o)                             # both states, and the gate values for inspection


def init_lstm(input_dim, hidden_dim, seed, forget_bias=0.0, weight_scale=1.0):  # random LSTM weights
    rng = np.random.RandomState(seed)                     # private rng -> reproducible
    W = rng.randn(input_dim + hidden_dim, 4 * hidden_dim) * weight_scale / np.sqrt(input_dim + hidden_dim)  # one block per gate
    b = np.zeros(4 * hidden_dim)                          # biases, in the same f, i, g, o order
    b[:hidden_dim] = forget_bias                          # the forget gate's bias: the one setting Section 4 turns
    return W, b                                           # W: (input + hidden, 4 * hidden);  b: (4 * hidden,)


def cell_update(c_prev, f, i, g, o):                    # the two update lines on their own, with the gates GIVEN
    c = f * c_prev + i * g                                # same line as in lstm_step_forward
    return c, o * np.tanh(c)                              # new cell state, new hidden state


# =============================================================================
# DEMOS
# =============================================================================
def demo_one_step():
    print("=" * 70)
    print("1. ONE STEP OF AN LSTM CELL -- traced by hand, then checked")
    print("=" * 70)
    print("A cell with one input and one hidden unit, so every quantity is a single number. "
          "The weights are chosen by hand, not learned. Each line below is computed with "
          "plain arithmetic and compared with lstm_step_forward.")
    x, h_prev, c_prev = 1.0, 0.5, -0.8                        # this step's input, the old hidden state, the old cell state
    W = np.array([[0.5, -1.0, 2.0, 1.5],                       # weights from the input x to the f, i, g, o gates
                  [1.0, 0.4, -0.6, 0.2]])                      # weights from the old hidden state h to the f, i, g, o gates
    b = np.array([0.1, 0.0, 0.0, -0.3])                        # one bias per gate
    a_f = x * W[0, 0] + h_prev * W[1, 0] + b[0]                # forget gate's raw score
    a_i = x * W[0, 1] + h_prev * W[1, 1] + b[1]                # input gate's raw score
    a_g = x * W[0, 2] + h_prev * W[1, 2] + b[2]                # candidate's raw score
    a_o = x * W[0, 3] + h_prev * W[1, 3] + b[3]                # output gate's raw score
    f, i, g, o = sigmoid(a_f), sigmoid(a_i), np.tanh(a_g), sigmoid(a_o)  # squash each one
    c = f * c_prev + i * g                                     # new cell state
    h = o * np.tanh(c)                                         # new hidden state
    h_fn, c_fn, (f_fn, i_fn, g_fn, o_fn) = lstm_step_forward(np.array([[x]]), np.array([[h_prev]]), np.array([[c_prev]]), W, b)
    print(f"\ninputs: x = {x}, old hidden state h = {h_prev}, old cell state c = {c_prev}")
    print(f"\n{'quantity':<26}{'raw score':>11}{'by hand':>11}{'function':>11}")
    for name, raw, hand, fn in [("forget gate f (sigmoid)", a_f, f, f_fn), ("input gate i (sigmoid)", a_i, i, i_fn),
                                ("candidate g (tanh)", a_g, g, g_fn), ("output gate o (sigmoid)", a_o, o, o_fn)]:
        print(f"{name:<26}{raw:>11.4f}{hand:>11.4f}{fn.item():>11.4f}")
    print(f"{'new cell state c':<26}{'':>11}{c:>11.4f}{c_fn.item():>11.4f}")
    print(f"{'new hidden state h':<26}{'':>11}{h:>11.4f}{h_fn.item():>11.4f}")
    print(f"  new c = f * old c + i * g = {f:.4f} * ({c_prev}) + {i:.4f} * {g:.4f} = {c:.4f}")
    print(f"  new h = o * tanh(new c)   = {o:.4f} * ({np.tanh(c):.4f}) = {h:.4f}")
    print("Actual output from this lesson's run.")

    rng = np.random.RandomState(0)                             # a bigger, random case
    D, H, m = 3, 4, 5                                          # 3 inputs, 4 hidden units, a batch of 5
    W, b = init_lstm(D, H, seed=1)                             # random weights
    b = rng.randn(4 * H) * 0.5                                 # random biases too, so nothing is hidden by zeros
    x_t, h0, c0 = rng.randn(m, D), rng.randn(m, H), rng.randn(m, H)  # random input and states
    h_fn, c_fn, _ = lstm_step_forward(x_t, h0, c0, W, b)       # the function under test
    Wx, Wh = W[:D], W[D:]                                      # the input rows and the hidden rows of W
    gate = lambda k: x_t @ Wx[:, k * H:(k + 1) * H] + h0 @ Wh[:, k * H:(k + 1) * H] + b[k * H:(k + 1) * H]  # gate k's raw scores
    c_ref = sigmoid(gate(0)) * c0 + sigmoid(gate(1)) * np.tanh(gate(2))  # the textbook formulas, one gate at a time
    h_ref = sigmoid(gate(3)) * np.tanh(c_ref)
    print(f"\nrandom check ({m} examples, {D} inputs, {H} hidden units): four separate gate formulas vs. the function -- "
          f"max difference in c {np.max(np.abs(c_fn - c_ref)):.2e}, in h {np.max(np.abs(h_fn - h_ref)):.2e}")
    print("Actual output from this lesson's run.")


def demo_gates():
    print("\n" + "=" * 70)
    print("2. WHAT EACH GATE DOES -- the gates set directly, no weights involved")
    print("=" * 70)
    print("Old cell state c = 0.80 and candidate g = -0.50 in every row. Only the three gates "
          "change. c_new = f * 0.80 + i * (-0.50);  h_new = o * tanh(c_new).")
    c_prev, g = 0.80, -0.50                                    # the memory, and the value offered for writing
    print(f"\n{'what the gates say':<34}{'f':>5}{'i':>5}{'o':>5}{'c_new':>9}{'h_new':>9}")
    for name, f, i, o in [("keep the memory, show it", 1, 0, 1), ("keep the memory, hide it", 1, 0, 0),
                          ("erase the memory", 0, 0, 1), ("replace it with the candidate", 0, 1, 1),
                          ("add the candidate to it", 1, 1, 1), ("half of each", 0.5, 0.5, 1)]:
        c, h = cell_update(c_prev, f, i, g, o)                   # the two update lines
        print(f"{name:<34}{f:>5}{i:>5}{o:>5}{c:>9.3f}{h:>9.3f}")
    print("Actual output from this lesson's run.")


def handset_memory_cell(k):                             # weights for a 1-unit cell that stores, holds and reads; inputs = [value, write, read]
    W = np.zeros((4, 4))                                  # rows: value, write flag, read flag, old h;  columns: f, i, g, o
    b = np.zeros(4)                                       # one bias per gate
    W[1, 0], b[0] = -k, k / 2                             # forget gate: score +k/2 normally (keep), -k/2 when write = 1 (clear)
    W[1, 1], b[1] = k, -k / 2                             # input gate: score -k/2 normally (shut), +k/2 when write = 1 (open)
    W[0, 2] = 1.0                                         # candidate = tanh(value)
    W[2, 3], b[3] = k, -k / 2                             # output gate: shut unless read = 1
    return W, b                                           # k is the "sharpness": bigger k pushes the gates closer to exactly 0 and 1


def demo_store_hold_read():
    print("\n" + "=" * 70)
    print("3. STORE, HOLD, READ -- a memory cell with hand-set weights")
    print("=" * 70)
    print("One LSTM unit, three inputs per step: a value, a WRITE flag and a READ flag. Step 0 "
          "writes a value v. Every later step carries a random distracting value (std 0.5) "
          "with write = 0. The last step sets read = 1. The weights are set by hand with a "
          "sharpness k: a gate's raw score is +k/2 when it should be open and -k/2 when shut. "
          "1,000 different stored values per row.")
    n = 1000                                                   # number of independent trials
    rng = np.random.RandomState(0)                             # one rng for the stored values and the distractors
    v = rng.uniform(-1, 1, n)                                  # the values to store
    noise = rng.randn(n, 999) * 0.5                            # distracting values for up to 999 later steps
    stored = np.tanh(v)                                        # what the cell state should hold: g = tanh(v), written with i ~ 1
    shown = int(np.argmax(stored > 0.6))                       # the first trial storing a clearly visible value, for the plot
    traces = {}                                                # k -> that trial's cell state at every step
    print(f"\n{'k':>4}{'open gate':>13}{'shut gate':>12}{'steps held':>12}{'mean |c - stored|':>19}{'correlation':>13}"
          f"{'max |h| before read':>21}")
    for k in [10, 20, 40]:                                      # three sharpness settings
        W, b = handset_memory_cell(k)                            # the hand-set weights
        for T in [10, 100, 1000]:                                # total sequence lengths
            X = np.zeros((n, T, 3))                                # (trial, step, [value, write, read])
            X[:, 0, 0], X[:, 0, 1] = v, 1.0                        # step 0: the value, with the write flag up
            X[:, 1:, 0] = noise[:, :T - 1]                         # later steps: distractors, write flag down
            X[:, -1, 2] = 1.0                                      # last step: read flag up
            h, c = np.zeros((n, 1)), np.zeros((n, 1))              # start empty
            hidden_peak, trace = 0.0, []                           # largest |h| before the read step; the shown trial's cell state
            for t in range(T):                                     # run the cell
                h, c, _ = lstm_step_forward(X[:, t, :], h, c, W, b)  # one step
                trace.append(c[shown, 0])                            # record the shown trial
                if t < T - 1:                                        # every step except the read step
                    hidden_peak = max(hidden_peak, np.abs(h).max())    # how much leaks out while the output gate is shut
            if T == 1000:                                          # keep the longest trace for the plot
                traces[k] = np.array(trace)
            print(f"{k:>4}{sigmoid(k / 2):>13.9f}{sigmoid(-k / 2):>12.2e}{T - 1:>12}{np.mean(np.abs(c[:, 0] - stored)):>19.2e}"
                  f"{np.corrcoef(c[:, 0], stored)[0, 1]:>13.6f}{hidden_peak:>21.2e}")
    print("Actual output from this lesson's run.")

    print(f"\nprediction for the leak: a forget gate of value f keeps f^T of the memory after T steps.")
    print(f"{'k':>4}{'f':>13}{'f^9':>10}{'f^99':>10}{'f^999':>10}")
    for k in [10, 20, 40]:                                      # the same three settings
        f = sigmoid(k / 2)                                        # the open forget gate's value
        print(f"{k:>4}{f:>13.9f}{f ** 9:>10.4f}{f ** 99:>10.4f}{f ** 999:>10.4f}")
    print("Actual output from this lesson's run.")

    print(f"\nfor contrast, the simplest hand-set RNN memory: one unit, h = tanh(h + value), recurrent weight 1. "
          f"It has no gates, so every value is added in.")
    print(f"{'steps held':>11}{'distractors: corr':>20}{'no distractors: corr':>23}{'no distractors: 0.9 becomes':>30}")
    rnn_trace = {}                                             # label -> the shown trial's hidden state at every step
    one = np.ones((1, 1))                                      # the 1x1 weight "matrix" holding the single weight 1.0
    for T in [10, 100, 1000]:                                   # the same three lengths
        h_noisy = rnn_step_forward(v[:, None], np.zeros((n, 1)), one, one, 0.0)  # step 0: store v
        h_quiet = h_noisy.copy()                                  # a second copy that will see no distractors
        h_demo = np.tanh(np.array([[0.9]]))                       # one trial storing exactly 0.9
        tr_noisy, tr_quiet = [h_noisy[shown, 0]], [h_quiet[shown, 0]]  # the shown trial's traces
        for t in range(T - 1):                                    # the hold steps
            h_noisy = rnn_step_forward(noise[:, t:t + 1], h_noisy, one, one, 0.0)  # distractor added in
            h_quiet = rnn_step_forward(np.zeros((n, 1)), h_quiet, one, one, 0.0)   # input exactly zero
            h_demo = np.tanh(h_demo)                                # the single 0.9 trial, zero input
            tr_noisy.append(h_noisy[shown, 0]); tr_quiet.append(h_quiet[shown, 0])  # record the shown trial
        if T == 1000:                                             # keep the longest traces for the plot
            rnn_trace = {"RNN, distractors": np.array(tr_noisy), "RNN, no distractors": np.array(tr_quiet)}
        print(f"{T - 1:>11}{np.corrcoef(h_noisy[:, 0], stored)[0, 1]:>20.4f}{np.corrcoef(h_quiet[:, 0], stored)[0, 1]:>23.4f}"
              f"{h_demo.item():>30.4f}")
    print("Actual output from this lesson's run.")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))                  # one panel: one trial's memory over 1,000 steps
    steps = np.arange(1, 1001)                                  # step numbers, starting at 1 so a log axis works
    for k, color in [(10, "#d62728"), (20, "#ff7f0e"), (40, "#2ca02c")]:  # the three LSTM settings
        ax.plot(steps, traces[k], color=color, label=f"LSTM cell state, sharpness k = {k}")
    ax.plot(steps, rnn_trace["RNN, no distractors"], color="#1f77b4", linestyle="--", label="one-unit RNN, no distractors")
    ax.plot(steps, rnn_trace["RNN, distractors"], color="#7f7f7f", linewidth=0.5, alpha=0.55, label="one-unit RNN, with distractors")
    ax.axhline(stored[shown], color="black", linestyle=":", linewidth=1, label=f"the value stored at step 1 ({stored[shown]:.3f})")
    ax.set_xscale("log"); ax.set_xlabel("step (log scale)"); ax.set_ylabel("memory in one trial")  # log x: 1 to 1,000
    ax.set_title("Holding one number for 1,000 steps: how sharp must the gates be?", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")                     # legend
    plt.tight_layout(); plt.savefig("lstm_store_hold_read.png", dpi=110); plt.close()  # save and free
    print("\nsaved lstm_store_hold_read.png")


def rnn_final_state(h0, X, Wxh, Whh):                   # run Day 57's RNN (no bias) over X starting from h0; return the last hidden state
    h = h0                                                # starting state
    for t in range(X.shape[1]):                           # one step at a time
        h = rnn_step_forward(X[:, t:t + 1], h, Wxh, Whh, 0.0)  # the vanilla recurrence
    return h                                              # h after the last step


def lstm_final_cell(c0, X, W, b):                       # run the LSTM cell over X starting from cell state c0 (hidden state starts at 0)
    h, c = np.zeros_like(c0), c0                          # starting states
    kept = np.ones_like(c0)                               # running product of the forget gates, unit by unit
    for t in range(X.shape[1]):                           # one step at a time
        h, c, (f, _, _, _) = lstm_step_forward(X[:, t:t + 1], h, c, W, b)  # one LSTM step
        kept = kept * f                                     # how much of c0 the DIRECT path still carries
    return c, kept                                        # final cell state, and the product of every forget gate


def nudge_sensitivity(run, state0, direction, eps=1e-4):  # how big is the change at the end, per unit of nudge at the start?
    change = run(state0 + eps * direction) - run(state0 - eps * direction)  # final state nudged up, minus nudged down
    return np.linalg.norm(change) / (2 * eps)             # size of that change per unit of nudge (a central difference)


def fmt(value):                                         # print a sensitivity, or say it is below what the nudge method can measure
    return f"{value:.2e}" if value >= 1e-9 else "<1e-09"


def demo_memory_survival():
    print("\n" + "=" * 70)
    print("4. DAY 59'S QUESTION, ASKED OF THE LSTM -- how much of a nudge at step 0 survives T steps?")
    print("=" * 70)
    print("16 units, 200 sequences of N(0,1) inputs -- Day 59's setup. The state at step 0 is "
          "nudged by a tiny amount in a random direction, and the size of the change at step "
          "T is divided by the size of the nudge. 1.0 means fully kept; below 1 fading; above "
          "1 growing. RNN: the hidden state, at five spectral radii of Whh (Day 59's sweep, "
          "same weights). LSTM: the cell state, at four forget-gate biases, all other weights "
          "random. Nothing is trained.")
    m, H = 200, 16                                             # batch size, hidden size
    direction = np.random.RandomState(1).randn(m, H)           # the direction of the nudge
    direction /= np.linalg.norm(direction)                     # scaled to length 1 over the whole batch
    zero = np.zeros((m, H))                                    # every state starts at zero
    lengths = [5, 10, 20, 50, 100]                             # how many steps the nudge has to survive
    rhos, biases = [0.5, 1.0, 1.5, 2.0, 3.0], [0.0, 1.0, 2.0, 4.0]  # the RNN sweep and the LSTM sweep

    W, b = init_lstm(1, H, seed=3, forget_bias=2.0)            # first, check the measuring tool on a case with a known answer
    W[1:, :] = 0.0                                             # cut every connection from h: the gates now depend on the input only
    X = np.random.RandomState(0).randn(m, 20)                  # 20 steps of input
    kept = lstm_final_cell(zero, X, W, b)[1]                   # product of the forget gates
    measured = nudge_sensitivity(lambda c0: lstm_final_cell(c0, X, W, b)[0], zero, direction)  # the nudge measurement
    exact = np.linalg.norm(kept * direction)                   # with no path through h, c_T = (product of f) * c_0 + something fixed
    print(f"\ncheck of the measuring tool (gates cut off from h, so the exact answer is known): "
          f"measured {measured:.10f}, exact {exact:.10f}, difference {abs(measured - exact):.1e}")
    print("Actual output from this lesson's run.")

    rnn_curves = {rho: [] for rho in rhos}                     # rho -> sensitivity at every length, for the plot
    lstm_curves = {fb: [] for fb in biases}                    # forget bias -> sensitivity at every length
    table = {T: {} for T in lengths}                           # T -> {label: value}, for printing
    for T in range(1, 101):                                    # every length from 1 to 100 (the plot uses all of them)
        X = np.random.RandomState(0).randn(m, T)                 # Day 59's inputs: N(0,1) at every step
        for rho in rhos:                                         # the RNN sweep
            rng = np.random.RandomState(3)                         # Day 59's seed-3 network ...
            Wxh = rng.randn(1, H) * np.sqrt(2.0 / 1)               # ... same input weights
            Whh = rng.randn(H, H) * np.sqrt(2.0 / H)               # ... same recurrent weights
            Whh *= rho / spectral_radius(Whh)                      # rescaled to the wanted spectral radius, as on Day 59
            s = nudge_sensitivity(lambda h0: rnn_final_state(h0, X, Wxh, Whh), zero, direction)  # nudge h_0, measure h_T
            rnn_curves[rho].append(s)                              # keep for the plot
            if T in table: table[T][("rnn", rho)] = s              # keep for the table
        for fb in biases:                                        # the LSTM sweep
            W, b = init_lstm(1, H, seed=3, forget_bias=fb)         # identical random weights; only the forget bias differs
            s = nudge_sensitivity(lambda c0: lstm_final_cell(c0, X, W, b)[0], zero, direction)  # nudge c_0, measure c_T
            c_T, kept = lstm_final_cell(zero, X, W, b)             # the forget gates along the way
            lstm_curves[fb].append(s)                              # keep for the plot
            if T in table:                                         # keep for the tables
                table[T][("lstm", fb)] = s
                table[T][("direct", fb)] = np.linalg.norm(kept * direction)  # the direct cell-state path alone
                table[T][("gate", fb)] = kept.mean() ** (1.0 / T)            # a typical single forget-gate value

    print(f"\nvanilla RNN -- nudge the hidden state at step 0, measure the hidden state at step T:")
    print(f"{'T':>5}" + "".join(f"{'rho=' + str(r):>12}" for r in rhos))
    for T in lengths:
        print(f"{T:>5}" + "".join(f"{fmt(table[T][('rnn', r)]):>12}" for r in rhos))
    print("Actual output from this lesson's run.")
    print(f"\nLSTM -- nudge the cell state at step 0, measure the cell state at step T:")
    print(f"{'T':>5}" + "".join(f"{'f-bias=' + str(fb):>14}" for fb in biases))
    for T in lengths:
        print(f"{T:>5}" + "".join(f"{fmt(table[T][('lstm', fb)]):>14}" for fb in biases))
    print("Actual output from this lesson's run.")
    print(f"\nLSTM, the direct path only: the product of the forget gates (no nudge needed; this part is exact):")
    print(f"{'T':>5}" + "".join(f"{'f-bias=' + str(fb):>14}" for fb in biases))
    for T in lengths:
        print(f"{T:>5}" + "".join(f"{table[T][('direct', fb)]:>14.2e}" for fb in biases))
    print(f"{'f':>5}" + "".join(f"{sigmoid(fb):>14.3f}" for fb in biases) + "   <- sigmoid(f-bias)")
    print("Actual output from this lesson's run.")

    print(f"\ncan the LSTM blow up? forget bias 4, T = 50, every weight multiplied by a factor:")
    print(f"{'weights x':>10}{'nudge sensitivity':>19}{'direct path':>13}")
    X = np.random.RandomState(0).randn(m, 50)                  # 50 steps of input
    for scale in [1.0, 3.0, 10.0]:                              # normal weights, then 3x and 10x larger
        W, b = init_lstm(1, H, seed=3, forget_bias=4.0, weight_scale=scale)  # same random draw, scaled
        s = nudge_sensitivity(lambda c0: lstm_final_cell(c0, X, W, b)[0], zero, direction)  # total sensitivity
        kept = lstm_final_cell(zero, X, W, b)[1]                 # the direct path
        print(f"{scale:>10}{s:>19.2e}{np.linalg.norm(kept * direction):>13.2e}")
    print("Actual output from this lesson's run.")

    fig, ax = plt.subplots(figsize=(9.5, 5.6))                  # one panel: survival against distance
    T_axis = np.arange(1, 101)                                  # 1..100 steps
    for rho, color in zip(rhos, ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]):  # RNN: blues, dashed
        ax.semilogy(T_axis, np.clip(rnn_curves[rho], 1e-9, None), color=color, linestyle="--", label=f"RNN, spectral radius {rho}")
    for fb, color in zip(biases, ["#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]):  # LSTM: oranges, solid
        ax.semilogy(T_axis, np.clip(lstm_curves[fb], 1e-9, None), color=color, linewidth=2, label=f"LSTM, forget bias {fb}")
    ax.axhline(1.0, color="black", linewidth=0.8)               # 1.0 = the nudge is exactly preserved
    ax.set_ylim(1e-9, 1e6); ax.set_xlabel("T: number of steps since the nudge")  # the measurable range
    ax.set_ylabel("size of change at step T / size of nudge")
    ax.set_title("How much of a nudge at step 0 is left T steps later (untrained networks)", fontsize=11)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))  # legend below the plot
    plt.tight_layout(); plt.savefig("lstm_memory_survival.png", dpi=110); plt.close()  # save and free
    print("\nsaved lstm_memory_survival.png")


def demo_cost():
    print("\n" + "=" * 70)
    print("5. WHAT THE CELL COSTS -- weights and time")
    print("=" * 70)
    print("Same input size and hidden size for both cells. Weights are counted from the arrays; "
          "time is measured for 20,000 single steps on a batch of 32.")
    print(f"\n{'input':>6}{'hidden':>8}{'RNN weights':>13}{'LSTM weights':>14}{'ratio':>7}")
    for D, H in [(1, 16), (32, 32), (64, 128)]:                 # Day 59's size, Day 62's tagger size, a larger one
        W, b = init_lstm(D, H, seed=0)                            # LSTM arrays
        n_rnn = D * H + H * H + H                                 # Wxh + Whh + bh
        print(f"{D:>6}{H:>8}{n_rnn:>13}{W.size + b.size:>14}{(W.size + b.size) / n_rnn:>7.1f}")
    D, H, m = 32, 32, 32                                        # the timing case
    rng = np.random.RandomState(0)                              # random inputs and weights
    x, h, c = rng.randn(m, D), rng.randn(m, H), rng.randn(m, H)  # one batch
    W, b = init_lstm(D, H, seed=0)                              # LSTM weights
    Wxh, Whh, bh = rng.randn(D, H), rng.randn(H, H), np.zeros(H)  # RNN weights
    start = time.perf_counter()                                 # start the clock
    for _ in range(20000):                                      # 20,000 LSTM steps
        lstm_step_forward(x, h, c, W, b)
    t_lstm = time.perf_counter() - start                        # seconds for the LSTM
    start = time.perf_counter()                                 # restart the clock
    for _ in range(20000):                                      # 20,000 RNN steps
        rnn_step_forward(x, h, Wxh, Whh, bh)
    t_rnn = time.perf_counter() - start                         # seconds for the RNN
    print(f"\ntime for 20,000 steps (input 32, hidden 32, batch 32): RNN {t_rnn:.2f} s, LSTM {t_lstm:.2f} s, "
          f"ratio {t_lstm / t_rnn:.1f}  (timings vary from run to run and machine to machine)")
    print("Actual output from this lesson's run.")


def draw_cell_diagram():                                # a box-and-arrow picture of one LSTM step
    fig, ax = plt.subplots(figsize=(13, 6.6))                   # wide canvas
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 6.6); ax.axis("off")    # plain coordinate grid, no axes

    def box(x, y, w, h, title, sub, color):                      # a rounded box centred on (x, y) with one or two lines of text
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x, y + (0.17 if sub else 0.0), title, ha="center", va="center", fontsize=9.5, fontweight="bold")  # title, centred if alone
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8.5)

    def op(x, y, symbol):                                        # a small circle holding an operation symbol
        ax.add_patch(Circle((x, y), 0.27, facecolor="white", edgecolor="#333333", linewidth=1.4))
        ax.text(x, y, symbol, ha="center", va="center", fontsize=13, fontweight="bold")

    def arrow(x0, y0, x1, y1, color="#333333", width=1.5):       # one arrow from (x0, y0) to (x1, y1)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=width))

    ax.text(6.5, 6.25, "One step of an LSTM cell", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.75, 5.3, "c(t-1)\nold cell state", ha="center", va="center", fontsize=9.5, color="#a63603", fontweight="bold")
    ax.text(12.2, 5.3, "c(t)\nnew cell state", ha="center", va="center", fontsize=9.5, color="#a63603", fontweight="bold")
    arrow(1.45, 5.3, 2.7, 5.3, "#a63603", 2.6); op(3.0, 5.3, "x")  # the cell-state line: old memory times the forget gate
    arrow(3.3, 5.3, 6.0, 5.3, "#a63603", 2.6); op(6.3, 5.3, "+")   # ... plus what is written in
    arrow(6.6, 5.3, 11.45, 5.3, "#a63603", 2.6)                    # ... is the new memory
    ax.text(4.65, 5.52, "keep part of the old memory", ha="center", fontsize=8, color="#a63603")
    ax.text(8.6, 5.52, "the cell-state line: no weight matrix, no squashing", ha="center", fontsize=8, color="#a63603")
    box(3.0, 2.9, 1.9, 1.0, "forget gate f", "sigmoid: 0 to 1", "#FDECEC")   # the four gate boxes
    box(5.2, 2.9, 1.9, 1.0, "input gate i", "sigmoid: 0 to 1", "#EAF3DE")
    box(7.4, 2.9, 1.9, 1.0, "candidate g", "tanh: -1 to 1", "#E6F1F8")
    box(9.6, 2.9, 1.9, 1.0, "output gate o", "sigmoid: 0 to 1", "#F4EEFC")
    arrow(3.0, 3.44, 3.0, 5.0)                                     # f goes up to the first multiply
    op(6.3, 4.2, "x")                                              # i times g
    arrow(5.2, 3.44, 6.08, 4.02); arrow(7.4, 3.44, 6.52, 4.02)     # i and g into that multiply
    arrow(6.3, 4.48, 6.3, 5.0)                                     # ... and up into the add
    ax.text(6.75, 4.6, "write part of\nthe candidate", ha="left", va="center", fontsize=8)
    box(11.3, 4.2, 1.0, 0.6, "tanh", "", "#F2F2F2")                # squash the new cell state
    arrow(11.3, 5.3, 11.3, 4.54)                                   # from the cell-state line down into tanh
    op(11.3, 2.9, "x")                                             # ... times the output gate
    arrow(11.3, 3.86, 11.3, 3.2); arrow(10.6, 2.9, 11.0, 2.9)      # tanh and o into the last multiply
    arrow(11.58, 2.9, 12.0, 2.9)                                   # ... is the new hidden state
    ax.text(12.75, 2.9, "h(t)\nnew hidden\nstate", ha="center", va="center", fontsize=9.5, color="#08519c", fontweight="bold")
    box(6.3, 0.95, 8.5, 0.8, "x(t) and h(t-1), side by side", "this step's input and the old hidden state feed all four boxes", "#F2F2F2")
    for x in [3.0, 5.2, 7.4, 9.6]:                                 # one arrow from the shared input up to each box
        arrow(x, 1.39, x, 2.36)
    plt.savefig("lstm_cell_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("\nsaved lstm_cell_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT TODAY DID NOT SHOW")
    print("=" * 70)
    print("Every gate today was set by hand or left random. Section 3 shows a cell CAN hold a "
          "number for 1,000 steps; Section 4 shows that with the forget gate near 1 a nudge "
          "survives where the RNN's fades or explodes. Neither shows that training FINDS such "
          "gate values. That needs the forward pass over whole sequences (Day 64), the "
          "backward pass with a gradient check (Day 65), and a real comparison of trained "
          "RNN, LSTM and GRU (Day 67).")


def main():                                             # runs every section in order
    demo_one_step()                                        # 1. the cell, by hand and checked
    demo_gates()                                           # 2. what each gate does
    demo_store_hold_read()                                 # 3. a hand-set memory cell
    demo_memory_survival()                                 # 4. Day 59's question, for the LSTM
    demo_cost()                                            # 5. weights and time
    draw_cell_diagram()                                    # the cell diagram used in the guides
    note_on_what_comes_next()                              # 6. what today did not show


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
