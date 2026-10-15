"""
Day 64: The LSTM Forward Pass From Scratch

Day 63 built ONE step of an LSTM cell. Today that step goes inside a loop
over time, exactly as Day 57 put the RNN cell inside a loop. The loop
carries TWO states from step to step (the hidden state h and the cell
state c) and stores everything the backward pass (Day 65) will need.

What is built and measured:

  1. lstm_forward: the loop, with its cache. Checked against the same
     steps called one by one by hand, and run at lengths 5, 50 and 500.
  2. Padding and masking with two states: on a padded step BOTH h and c
     are carried forward unchanged (Day 61's rule, applied twice). Checked:
     a short sequence gives the same final states alone and padded.
  3. A faster forward pass: the input part of every gate is computed for
     all timesteps in one matrix product before the loop. Same numbers.
     The first timing showed an 8x speed-up; most of it turned out to be
     this machine's multi-threaded matrix library being slow on mid-sized
     products, not the trick. Both timings are shown.
  4. What the forward pass produces over 200 steps, by forget bias: with
     the forget gate near 1 the cell state keeps adding up, and tanh(c)
     saturates. Day 63 showed what a high forget bias buys; this shows
     part of what it costs.
  5. What the cache costs in memory, compared with an RNN's.

Nothing is trained today: training needs the backward pass (Day 65).

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import time                                            # time: perf_counter, to time the two forward passes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_info, threadpool_limits  # read and limit how many CPU threads NumPy's matrix library uses
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the unrolled-network diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

# =============================================================================
# 0. FROM DAY 63, VERBATIM -- the gate function, one LSTM step, and the initializer
# =============================================================================
def sigmoid(z):                                         # squashes any number into the range 0..1
    return 1.0 / (1.0 + np.exp(-z))                       # large negative -> 0, zero -> 0.5, large positive -> 1


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


# =============================================================================
# 1. THE FORWARD PASS -- NEW today
# =============================================================================
def lstm_forward(X, h0, c0, W, b, mask=None):           # NEW: run the cell over a whole batch of sequences
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


def lstm_forward_fast(X, h0, c0, W, b):                 # NEW: same result, with the input part done before the loop
    m, T, D = X.shape                                     # batch size, sequence length, input size
    H = h0.shape[1]                                       # hidden size
    Wx, Wh = W[:D], W[D:]                                 # the rows that multiply x, and the rows that multiply h
    A = (X.reshape(m * T, D) @ Wx + b).reshape(m, T, 4 * H)  # the input part of every gate at every step, in ONE product
    A = A.transpose(1, 0, 2).copy()                       # time first, so A[t] is one contiguous block of memory
    Hs, Cs = np.zeros((m, T, H)), np.zeros((m, T, H))     # every hidden and cell state
    h, c = h0, c0                                         # starting states
    for t in range(T):                                    # the loop is still needed: h_t depends on h_{t-1}
        a = A[t] + h @ Wh                                   # only the recurrent part is left to compute here
        f, i = sigmoid(a[:, :H]), sigmoid(a[:, H:2 * H])     # forget and input gates
        g, o = np.tanh(a[:, 2 * H:3 * H]), sigmoid(a[:, 3 * H:])  # candidate and output gate
        c = f * c + i * g                                   # cell state
        h = o * np.tanh(c)                                  # hidden state
        Hs[:, t], Cs[:, t] = h, c                           # store
    return Hs, Cs                                         # same shapes as lstm_forward


def rnn_forward(X, h0, Wxh, Whh, bh):                   # Day 57's RNN forward pass, with a vector input per step (as on Day 62)
    m, T, _ = X.shape                                     # batch size, sequence length, input size
    Hs = np.zeros((m, T, Whh.shape[0]))                   # every hidden state
    h = h0                                                # starting state
    for t in range(T):                                    # one step at a time
        h = np.tanh(X[:, t, :] @ Wxh + h @ Whh + bh)        # the vanilla recurrence
        Hs[:, t] = h                                        # store it
    return Hs                                             # the only thing an RNN's backward pass needs besides X


# =============================================================================
# DEMOS
# =============================================================================
def demo_forward_checks():
    print("=" * 70)
    print("1. THE FORWARD PASS -- checked against the same steps done by hand")
    print("=" * 70)
    print("A batch of 4 sequences, 3 steps each, 2 inputs, 5 hidden units, random weights. "
          "lstm_forward runs the loop; the reference calls Day 63's lstm_step_forward three "
          "times by hand, passing h and c along manually.")
    rng = np.random.RandomState(0)                             # private rng
    m, T, D, H = 4, 3, 2, 5                                    # small sizes, so every number can be inspected
    X = rng.randn(m, T, D)                                     # the inputs
    h0, c0 = rng.randn(m, H) * 0.5, rng.randn(m, H) * 0.5      # non-zero starting states, so they are really used
    W, b = init_lstm(D, H, seed=1)                             # random weights
    Hs, Cs, cache = lstm_forward(X, h0, c0, W, b)              # the loop under test
    h1, c1, _ = lstm_step_forward(X[:, 0], h0, c0, W, b)       # step 0 by hand
    h2, c2, _ = lstm_step_forward(X[:, 1], h1, c1, W, b)       # step 1, fed step 0's states
    h3, c3, _ = lstm_step_forward(X[:, 2], h2, c2, W, b)       # step 2, fed step 1's states
    print(f"\n{'step':>5}{'max |h loop - h by hand|':>27}{'max |c loop - c by hand|':>27}")
    for t, (h_ref, c_ref) in enumerate([(h1, c1), (h2, c2), (h3, c3)]):  # compare every step
        print(f"{t:>5}{np.max(np.abs(Hs[:, t] - h_ref)):>27.2e}{np.max(np.abs(Cs[:, t] - c_ref)):>27.2e}")
    print("Actual output from this lesson's run.")

    print(f"\nwhat the cache holds for the backward pass (m={m}, T={T}, D={D}, H={H}):")
    for name in ["X", "h0", "c0", "W", "b", "mask", "H", "C", "F", "I", "G", "O"]:  # every entry
        print(f"  {name:<5} shape {str(cache[name].shape):<12}")
    print("Actual output from this lesson's run.")

    W, b = init_lstm(3, 8, seed=2)                             # a second network: 3 inputs, 8 hidden units
    print(f"\nany length works -- the same weights ({W.size + b.size} numbers) on three different lengths:")
    for T in [5, 50, 500]:                                     # three lengths
        X = rng.randn(2, T, 3)                                   # 2 sequences of length T
        Hs, Cs, _ = lstm_forward(X, np.zeros((2, 8)), np.zeros((2, 8)), W, b)  # same weights every time
        print(f"  T={T:<4} hidden states {str(Hs.shape):<13} cell states {str(Cs.shape):<13} "
              f"largest |h| {np.abs(Hs).max():.3f}  largest |c| {np.abs(Cs).max():.3f}")
    print("Actual output from this lesson's run.")


def demo_masking():
    print("\n" + "=" * 70)
    print("2. PADDING WITH TWO STATES -- both h and c must be carried through padding")
    print("=" * 70)
    print("A 3-step sequence is run alone, then padded to 8 steps inside a batch next to a "
          "full 8-step sequence. With the mask, padded steps carry h AND c forward unchanged. "
          "Two wrong versions are tried as well: no mask at all, and a mask on h only.")
    rng = np.random.RandomState(3)                             # private rng
    D, H = 2, 6                                                # 2 inputs, 6 hidden units
    W, b = init_lstm(D, H, seed=4)                             # random weights
    short = rng.randn(1, 3, D)                                 # the 3-step sequence
    zeros = lambda m: np.zeros((m, H))                         # starting states of the right shape
    h_alone, c_alone, _ = lstm_forward(short, zeros(1), zeros(1), W, b)  # reference: no padding at all
    batch = np.zeros((2, 8, D)); batch[0, :3] = short[0]       # the short sequence, then 5 padding steps of zeros
    batch[1] = rng.randn(8, D)                                 # a full-length neighbour
    mask = np.array([[1.0] * 3 + [0.0] * 5, [1.0] * 8])        # 1 = real step, 0 = padding
    Hm, Cm, _ = lstm_forward(batch, zeros(2), zeros(2), W, b, mask=mask)  # with the mask
    Hn, Cn, _ = lstm_forward(batch, zeros(2), zeros(2), W, b)  # no mask: padding treated as real zeros
    h, c = zeros(2), zeros(2)                                  # third version, written out: carry h only
    for t in range(8):                                         # one step at a time
        h_new, c, _ = lstm_step_forward(batch[:, t], h, c, W, b)  # c is always replaced -- the mistake
        h = mask[:, t:t + 1] * h_new + (1 - mask[:, t:t + 1]) * h  # h is carried correctly
    print(f"\n{'version':<26}{'max |h_final - alone|':>23}{'max |c_final - alone|':>23}")
    for name, h_last, c_last in [("mask on h and c", Hm[0, -1], Cm[0, -1]), ("no mask", Hn[0, -1], Cn[0, -1]),
                                 ("mask on h only", h[0], c[0])]:
        print(f"{name:<26}{np.max(np.abs(h_last - h_alone[0, -1])):>23.2e}{np.max(np.abs(c_last - c_alone[0, -1])):>23.2e}")
    print("Actual output from this lesson's run.")


def best_time(fn, *args, repeats=7):                    # time a call several times and keep the fastest
    times = []                                            # one entry per repeat
    for _ in range(repeats):                              # repeat to reduce noise from the rest of the machine
        start = time.perf_counter()                         # start the clock
        fn(*args)                                           # the call being timed
        times.append(time.perf_counter() - start)           # seconds it took
    return min(times)                                     # the fastest run is the least disturbed one


def demo_fast_forward():
    print("\n" + "=" * 70)
    print("3. A FASTER FORWARD PASS? -- the input part of every gate, computed before the loop")
    print("=" * 70)
    print("x_t @ Wx does not depend on the previous step, so it can be done for all T steps at "
          "once in one big matrix product; only h @ Wh has to stay inside the loop. Checked for "
          "identical results, then timed: best of 7 runs, several layer sizes.")
    rng = np.random.RandomState(5)                             # private rng
    X = rng.randn(16, 40, 10)                                  # a check case: 16 sequences, 40 steps, 10 inputs
    W, b = init_lstm(10, 12, seed=6)                           # 12 hidden units
    h0, c0 = np.zeros((16, 12)), np.zeros((16, 12))            # zero starting states
    Hs, Cs, _ = lstm_forward(X, h0, c0, W, b)                  # the loop version
    Hf, Cf = lstm_forward_fast(X, h0, c0, W, b)                # the precomputed version
    print(f"\nsame result? max difference in h {np.max(np.abs(Hs - Hf)):.1e}, in c {np.max(np.abs(Cs - Cf)):.1e}")
    print("Actual output from this lesson's run.")
    blas = [x for x in threadpool_info() if x["user_api"] == "blas"]  # NumPy's matrix library and its thread setting
    print(f"\nmatrix library: {blas[0]['internal_api']}, using {blas[0]['num_threads']} threads by default")
    print(f"\n{'':>21}{'---- default threads ----':>27}{'------ one thread ------':>27}")
    print(f"{'D':>4}{'H':>5}{'batch':>6}{'steps':>6}{'loop':>9}{'precomp':>9}{'speed-up':>9}"
          f"{'loop':>9}{'precomp':>9}{'speed-up':>9}")
    rows = []                                                  # (label, speed-up with default threads, speed-up with one thread)
    for D, H, m, T in [(1, 16, 200, 200), (32, 32, 32, 100), (64, 64, 32, 100), (128, 128, 32, 100), (300, 128, 64, 50)]:
        X = rng.randn(m, T, D)                                   # random inputs of this size
        W, b = init_lstm(D, H, seed=0)                           # random weights of this size
        h0, c0 = np.zeros((m, H)), np.zeros((m, H))              # zero starting states
        t_loop = best_time(lstm_forward, X, h0, c0, W, b)        # the straightforward version, default threads
        t_fast = best_time(lstm_forward_fast, X, h0, c0, W, b)   # the precomputed version, default threads
        with threadpool_limits(limits=1, user_api="blas"):       # the same two, with the matrix library on one thread
            t_loop1 = best_time(lstm_forward, X, h0, c0, W, b)
            t_fast1 = best_time(lstm_forward_fast, X, h0, c0, W, b)
        rows.append((f"D={D}, H={H}", t_loop / t_fast, t_loop1 / t_fast1))  # keep for the summary
        print(f"{D:>4}{H:>5}{m:>6}{T:>6}{t_loop:>9.4f}{t_fast:>9.4f}{t_loop / t_fast:>9.2f}"
              f"{t_loop1:>9.4f}{t_fast1:>9.4f}{t_loop1 / t_fast1:>9.2f}")
    print("Times in seconds, best of 7 runs. Actual output from this lesson's run; timings vary by machine.")

    x_h = rng.randn(32, 256); W_big = rng.randn(256, 512)      # the loop version's per-step product at D = H = 128
    h_only = rng.randn(32, 128); W_h = rng.randn(128, 512)     # the precomputed version's per-step product
    reps = 300                                                 # average over many calls
    timings = {}                                               # (which product, thread setting) -> microseconds per call
    for label, limit in [("default threads", None), ("one thread", 1)]:  # both thread settings
        with threadpool_limits(limits=limit, user_api="blas"):   # None leaves the default alone
            for name, A, B in [("[x, h] @ W  (32x256 @ 256x512)", x_h, W_big), ("h @ Wh      (32x128 @ 128x512)", h_only, W_h)]:
                start = time.perf_counter()                        # start the clock
                for _ in range(reps):                              # the same product many times
                    A @ B
                timings[(name, label)] = (time.perf_counter() - start) / reps * 1e6  # microseconds per call
    print(f"\none matrix product per step, at D = H = 128, microseconds per call:")
    print(f"{'product':<34}{'default threads':>17}{'one thread':>12}")
    for name in ["[x, h] @ W  (32x256 @ 256x512)", "h @ Wh      (32x128 @ 128x512)"]:
        print(f"{name:<34}{timings[(name, 'default threads')]:>17.1f}{timings[(name, 'one thread')]:>12.1f}")
    print("Actual output from this lesson's run.")
    return rows                                                # the speed-ups, for the summary plot


def demo_long_sequences():
    print("\n" + "=" * 70)
    print("4. WHAT THE FORWARD PASS PRODUCES OVER 200 STEPS -- by forget bias")
    print("=" * 70)
    print("16 units, 200 sequences of 200 N(0,1) inputs (Day 59's kind of input), random "
          "weights (seed 3, as on Day 63). Only the forget bias changes. Measured at four "
          "points in time: the typical and largest size of the cell state, and the fraction "
          "of units whose tanh(c) is saturated (|tanh(c)| > 0.99).")
    m, T, H = 200, 200, 16                                     # batch, length, hidden size
    X = np.random.RandomState(0).randn(m, T, 1)                # the inputs, shared by every setting
    curves = {}                                                # forget bias -> (mean |c| over time, saturated fraction over time)
    print(f"\n{'f-bias':>7}{'step':>6}{'mean |c|':>10}{'max |c|':>10}{'saturated':>11}{'mean |h|':>10}")
    for fb in [0.0, 1.0, 2.0, 4.0]:                             # the four settings from Day 63
        W, b = init_lstm(1, H, seed=3, forget_bias=fb)          # identical weights except the forget bias
        Hs, Cs, _ = lstm_forward(X, np.zeros((m, H)), np.zeros((m, H)), W, b)  # the forward pass
        saturated = np.abs(np.tanh(Cs)) > 0.99                   # True where tanh(c) is flat
        curves[fb] = (np.abs(Cs).mean(axis=(0, 2)), saturated.mean(axis=(0, 2)))  # averaged over batch and units, per step
        for t in [0, 9, 49, 199]:                                # steps 1, 10, 50, 200
            print(f"{fb:>7}{t + 1:>6}{np.abs(Cs[:, t]).mean():>10.3f}{np.abs(Cs[:, t]).max():>10.2f}"
                  f"{saturated[:, t].mean():>11.1%}{np.abs(Hs[:, t]).mean():>10.3f}")
    print("Actual output from this lesson's run.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))       # left: size of c; right: saturation
    steps = np.arange(1, T + 1)                                 # step numbers
    for fb, color in zip([0.0, 1.0, 2.0, 4.0], ["#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]):  # same colours as Day 63
        a1.plot(steps, curves[fb][0], color=color, linewidth=2, label=f"forget bias {fb}")
        a2.plot(steps, curves[fb][1], color=color, linewidth=2, label=f"forget bias {fb}")
    a1.set_xlabel("step"); a1.set_ylabel("mean |cell state|"); a1.set_title("The cell state adds up when f is near 1", fontsize=10)
    a2.set_xlabel("step"); a2.set_ylabel("fraction with |tanh(c)| > 0.99"); a2.set_ylim(-0.02, 1)
    a2.set_title("...and tanh(c) saturates", fontsize=10)
    for ax in (a1, a2):                                          # shared styling
        ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig("lstm_cell_state_growth.png", dpi=110); plt.close()  # save and free
    print("\nsaved lstm_cell_state_growth.png")


def demo_memory():
    print("\n" + "=" * 70)
    print("5. WHAT THE CACHE COSTS -- memory kept for the backward pass, LSTM vs. RNN")
    print("=" * 70)
    print("Batch 32, 100 steps, 32 inputs, 128 hidden units, float64. Each array's real size "
          "is read with .nbytes.")
    m, T, D, H = 32, 100, 32, 128                              # a mid-sized case
    rng = np.random.RandomState(7)                             # private rng
    X = rng.randn(m, T, D)                                     # inputs
    W, b = init_lstm(D, H, seed=0)                             # LSTM weights
    _, _, cache = lstm_forward(X, np.zeros((m, H)), np.zeros((m, H)), W, b)  # LSTM forward, with its cache
    Hr = rnn_forward(X, np.zeros((m, H)), rng.randn(D, H), rng.randn(H, H) * 0.05, np.zeros(H))  # RNN forward
    lstm_parts = {k: cache[k].nbytes for k in ["X", "H", "C", "F", "I", "G", "O"]}  # the per-step arrays
    rnn_parts = {"X": X.nbytes, "H": Hr.nbytes}                # an RNN's backward needs only the inputs and the hidden states
    print(f"\n{'array':<7}{'LSTM (MB)':>11}{'RNN (MB)':>10}")
    for k in ["X", "H", "C", "F", "I", "G", "O"]:               # one row per array
        print(f"{k:<7}{lstm_parts[k] / 1e6:>11.2f}{(rnn_parts[k] / 1e6 if k in rnn_parts else 0):>10.2f}")
    total_l, total_r = sum(lstm_parts.values()), sum(rnn_parts.values())  # totals
    print(f"{'total':<7}{total_l / 1e6:>11.2f}{total_r / 1e6:>10.2f}    ratio {total_l / total_r:.1f}")
    print(f"per example per step: LSTM {(D + 6 * H)} numbers (D + 6H), RNN {D + H} numbers (D + H)")
    print("Actual output from this lesson's run.")


def draw_unrolled_diagram():                            # three LSTM steps side by side, with both states flowing right
    fig, ax = plt.subplots(figsize=(13, 5.2))                   # wide canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")      # plain coordinate grid

    def arrow(x0, y0, x1, y1, color="#333333", width=1.5):       # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=width))

    ax.text(6.5, 4.95, "The forward pass: one cell, the same weights, used at every step", ha="center", fontsize=12, fontweight="bold")
    xs = [2.6, 6.5, 10.4]                                        # centres of the three cells
    for k, x in enumerate(xs):                                   # one cell per step
        ax.add_patch(FancyBboxPatch((x - 1.2, 1.7), 2.4, 1.9, boxstyle="round,pad=0.05", facecolor="#F2F2F2", edgecolor="#333333", linewidth=1.3))
        ax.text(x, 2.95, f"LSTM step t={k}", ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(x, 2.45, "lstm_step_forward\n(Day 63)", ha="center", va="center", fontsize=8.5)
        ax.text(x, 0.55, f"x[t={k}]", ha="center", va="center", fontsize=10, color="#08519c", fontweight="bold")
        arrow(x, 0.8, x, 1.62, "#08519c")                         # input into the cell
        arrow(x, 3.68, x, 4.3, "#08519c")                         # hidden state out, stored in H
        ax.text(x, 4.5, f"H[:, {k}], C[:, {k}] stored", ha="center", va="center", fontsize=8.5)
    ax.text(0.45, 3.3, "c0", ha="center", va="center", fontsize=10, color="#a63603", fontweight="bold")  # starting states
    ax.text(0.45, 2.05, "h0", ha="center", va="center", fontsize=10, color="#08519c", fontweight="bold")
    arrow(0.7, 3.3, 1.35, 3.3, "#a63603", 2.4); arrow(0.7, 2.05, 1.35, 2.05, "#08519c")
    for x0, x1 in zip(xs[:-1], xs[1:]):                         # states handed from each step to the next
        arrow(x0 + 1.25, 3.3, x1 - 1.25, 3.3, "#a63603", 2.4)
        arrow(x0 + 1.25, 2.05, x1 - 1.25, 2.05, "#08519c")
        ax.text((x0 + x1) / 2, 3.47, "c", ha="center", fontsize=9, color="#a63603")
        ax.text((x0 + x1) / 2, 2.2, "h", ha="center", fontsize=9, color="#08519c")
    arrow(11.65, 3.3, 12.3, 3.3, "#a63603", 2.4); arrow(11.65, 2.05, 12.3, 2.05, "#08519c")  # and on to step 3 ...
    ax.text(12.6, 2.7, "...", ha="center", va="center", fontsize=14)
    plt.savefig("lstm_unrolled_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("\nsaved lstm_unrolled_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT COMES NEXT")
    print("=" * 70)
    print("The forward pass is done and its cache holds every gate value at every step. Day "
          "65 walks backward through that cache -- two gradients flowing back in time, one "
          "for h and one for c -- and checks the result numerically. After that the LSTM can "
          "finally be trained.")


def main():                                             # runs every section in order
    demo_forward_checks()                                  # 1. the loop, checked by hand
    demo_masking()                                         # 2. padding with two states
    demo_fast_forward()                                    # 3. precomputing the input part
    demo_long_sequences()                                  # 4. 200 steps, by forget bias
    demo_memory()                                          # 5. the cache's memory cost
    draw_unrolled_diagram()                                # the diagram used in the guides
    note_on_what_comes_next()                              # 6. bridge to Day 65


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
