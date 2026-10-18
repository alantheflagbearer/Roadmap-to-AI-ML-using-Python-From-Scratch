"""
Day 68 Exercise -- A Bidirectional LSTM Layer, Isolated

Day 67's exercise gave the LSTM an output at every step. Its state at
step t has seen only steps 0..t. A bidirectional layer runs a SECOND LSTM
over the sequence backwards and joins the two states at every step, so
step t sees the whole sequence. This exercise isolates exactly that,
gradient-checks it, and trains it next to the one-directional version
on a task only the bidirectional layer can solve: at each step, output
the NEXT token.

Day 69's encoder can use this layer.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sigmoid(z):                                         # squashes any number into the range 0..1 -- from day63_exercise.py
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):                                         # logits -> probabilities, one row per example
    e = np.exp(z - z.max(axis=1, keepdims=True))          # subtract the row max for stability, exponentiate
    return e / e.sum(axis=1, keepdims=True)               # rows sum to 1


def lstm_forward(X, W, b):                              # verbatim from day67_exercise.py
    m, T, D = X.shape                                     # batch, steps, input size
    H = W.shape[1] // 4                                   # hidden size
    h, c = np.zeros((m, H)), np.zeros((m, H))             # both states start at zero
    Hs, Cs, gates = [], [], []                            # every state and every step's gates
    for t in range(T):                                    # one step at a time
        a = np.concatenate([X[:, t], h], axis=1) @ W + b    # all four gates' raw scores in one product
        f, i, o = sigmoid(a[:, :H]), sigmoid(a[:, H:2 * H]), sigmoid(a[:, 3 * H:])  # forget, input, output gates
        g = np.tanh(a[:, 2 * H:3 * H])                      # candidate
        c = f * c + i * g                                   # cell state
        h = o * np.tanh(c)                                  # hidden state
        Hs.append(h); Cs.append(c); gates.append((f, i, g, o))  # keep everything for backward
    return Hs, Cs, gates                                  # lists with one entry per step


def lstm_backward(dHs, X, W, Hs, Cs, gates):            # verbatim from day67_exercise.py: a gradient at every step
    m, T, D = X.shape                                     # batch, steps, input size
    H = W.shape[1] // 4                                   # hidden size
    dW, db = np.zeros_like(W), np.zeros(W.shape[1])       # accumulators
    dh, dc = np.zeros((m, H)), np.zeros((m, H))           # nothing arrives after the last step
    for t in reversed(range(T)):                          # backward through time
        dh = dh + dHs[t]                                    # this step's own output adds its gradient
        f, i, g, o = gates[t]                               # this step's gates
        h_prev = Hs[t - 1] if t > 0 else np.zeros((m, H))   # the states that fed step t
        c_prev = Cs[t - 1] if t > 0 else np.zeros((m, H))
        tc = np.tanh(Cs[t])                                 # tanh of this step's cell state
        dc = dc + dh * o * (1 - tc ** 2)                    # gradient at c: from the next step and through h = o*tanh(c)
        da = np.concatenate([dc * c_prev * f * (1 - f), dc * g * i * (1 - i),
                             dc * i * (1 - g ** 2), dh * tc * o * (1 - o)], axis=1)  # through each gate's squashing
        z = np.concatenate([X[:, t], h_prev], axis=1)       # what W multiplied at this step
        dW += z.T @ da; db += da.sum(axis=0)                # accumulate: the same W at every step
        dh = (da @ W.T)[:, D:]                              # gradient handed back to h_prev
        dc = dc * f                                         # gradient handed back to c_prev: the direct route
    return dW, db                                         # gradients for the LSTM's weights


def loss_and_grads(X, Y, P, bidirectional):             # NEW: the LSTM reading forwards, and (optionally) a second one backwards
    m, T = Y.shape                                        # batch, steps
    Hf, Cf, gf = lstm_forward(X, P["Wf"], P["bf"])        # forward LSTM: its state at t has seen steps 0..t
    if bidirectional:                                     # NEW: a second LSTM over the REVERSED sequence
        Hr, Cr, gr = lstm_forward(X[:, ::-1], P["Wb"], P["bb"])  # its step s reads original step T-1-s
        Hs = [np.concatenate([Hf[t], Hr[T - 1 - t]], axis=1) for t in range(T)]  # join both states at each ORIGINAL step t
    else:
        Hs = Hf                                             # one direction only
    loss, dHs = 0.0, []                                   # total loss, and the gradient at each step's joined state
    g = {k: np.zeros_like(v) for k, v in P.items()}       # gradients, start at zero (unused ones stay zero)
    for t in range(T):                                    # an output at every step (Day 67's exercise)
        p = softmax(Hs[t] @ P["Why"] + P["by"])             # this step's prediction
        loss -= np.log(p[np.arange(m), Y[:, t]]).sum() / (m * T)  # cross-entropy, averaged over batch and steps
        dlog = p.copy(); dlog[np.arange(m), Y[:, t]] -= 1; dlog /= m * T  # softmax + cross-entropy gradient
        g["Why"] += Hs[t].T @ dlog; g["by"] += dlog.sum(axis=0)  # output layer
        dHs.append(dlog @ P["Why"].T)                       # gradient at this step's joined state
    H = P["Wf"].shape[1] // 4                             # hidden size of each direction
    g["Wf"], g["bf"] = lstm_backward([d[:, :H] for d in dHs], X, P["Wf"], Hf, Cf, gf)  # forward half of each gradient
    if bidirectional:                                     # NEW: the backward half goes to the reversed LSTM ...
        dHr = [dHs[T - 1 - s][:, H:] for s in range(T)]     # ... in ITS step order (reversed)
        g["Wb"], g["bb"] = lstm_backward(dHr, X[:, ::-1], P["Wb"], Hr, Cr, gr)
    return loss, g, Hs                                    # loss, gradients, and the joined states


def make_batch(rng, m, T, V):                           # the task: at each step, output the NEXT token
    tokens = rng.randint(1, V, size=(m, T))               # random tokens 1..V-1 (0 means "no next token")
    Y = np.zeros((m, T), dtype=int)                       # the last step's target is 0 ...
    Y[:, :-1] = tokens[:, 1:]                             # ... every other step's is the token after it
    return np.eye(V)[tokens], Y                           # one-hot inputs, integer targets


def new_params(rng, V, H, bidirectional):               # random weights; forget bias 1 in each LSTM
    P = {"Wf": rng.randn(V + H, 4 * H) / np.sqrt(V + H), "bf": np.zeros(4 * H)}  # forward LSTM
    if bidirectional:
        P.update(Wb=rng.randn(V + H, 4 * H) / np.sqrt(V + H), bb=np.zeros(4 * H))  # backward LSTM
    out = 2 * H if bidirectional else H                   # the joined state is twice as wide
    P.update(Why=rng.randn(out, V) / np.sqrt(out), by=np.zeros(V))  # output layer
    P["bf"][:H] = 1.0                                     # forget biases
    if bidirectional: P["bb"][:H] = 1.0
    return P


V, H = 5, 16                                              # 5 symbols (0 = no next token), 16 hidden units per direction
P = new_params(np.random.RandomState(0), V, H, True)      # a bidirectional layer, for the gradient check
X, Y = make_batch(np.random.RandomState(1), 3, 6, V)      # a tiny batch
_, g, _ = loss_and_grads(X, Y, P, True)                   # analytic gradients
worst = 0.0                                               # largest |numeric - analytic| over every entry
for name in P:                                            # every parameter, both directions
    for pos in np.ndindex(P[name].shape):                   # every entry
        old = P[name][pos]                                    # remember it
        P[name][pos] = old + 1e-5; up = loss_and_grads(X, Y, P, True)[0]    # nudge up
        P[name][pos] = old - 1e-5; down = loss_and_grads(X, Y, P, True)[0]  # nudge down
        P[name][pos] = old                                    # restore
        worst = max(worst, abs((up - down) / 2e-5 - g[name][pos]))  # compare
print(f"gradient check, every entry of both LSTMs and the output layer: largest |numeric - analytic| = {worst:.2e}")

for bidirectional in [False, True]:                       # the same training for both layers
    rng = np.random.RandomState(0)                          # same data stream for both
    P = new_params(np.random.RandomState(2), V, H, bidirectional)  # fresh weights
    for step in range(1, 1501):                             # plain SGD on fresh batches
        X, Y = make_batch(rng, 32, 12, V)                     # 32 sequences of 12 tokens
        loss, g, _ = loss_and_grads(X, Y, P, bidirectional)   # forward and backward
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))  # Day 60's global norm
        for name in P:                                        # update, clipped at norm 1.0
            P[name] -= 0.5 * g[name] * min(1.0, 1.0 / norm)
    X, Y = make_batch(np.random.RandomState(99), 500, 12, V)  # 500 fresh test sequences
    _, _, Hs = loss_and_grads(X, Y, P, bidirectional)         # forward pass
    pred = np.stack([(h @ P["Why"] + P["by"]).argmax(axis=1) for h in Hs], axis=1)  # a prediction per step
    print(f"{'bidirectional' if bidirectional else 'forward only ':<14} weights {sum(v.size for v in P.values()):>5}"
          f"  final training loss {loss:.4f}  test accuracy, steps 1-11: {(pred[:, :-1] == Y[:, :-1]).mean():.3f}"
          f"  (guessing among 4 tokens: 0.250)")
