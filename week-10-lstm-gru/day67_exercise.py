"""
Day 67 Exercise -- An LSTM With an Output at Every Step

Day 65's exercise backpropagated through an LSTM whose loss looked only
at the final state. Today's main lesson used an output at EVERY step (a
tag per word). This exercise isolates exactly that: a softmax layer on
every hidden state, a loss summed over all steps, and the backward pass
that receives a gradient at every step. It is gradient-checked, then
trained on a tiny task that needs memory: at each step, output the token
seen TWO steps earlier.

Day 69's sequence-to-sequence decoder is this network.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sigmoid(z):                                         # squashes any number into the range 0..1 -- from day63_exercise.py
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):                                         # logits -> probabilities, one row per example
    e = np.exp(z - z.max(axis=1, keepdims=True))          # subtract the row max for stability, exponentiate
    return e / e.sum(axis=1, keepdims=True)               # rows sum to 1


def lstm_forward(X, W, b):                              # Day 65 exercise's loop, starting from zero states
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


def lstm_backward(dHs, X, W, Hs, Cs, gates):            # NEW: a gradient arrives at EVERY step's hidden state
    m, T, D = X.shape                                     # batch, steps, input size
    H = W.shape[1] // 4                                   # hidden size
    dW, db = np.zeros_like(W), np.zeros(W.shape[1])       # accumulators
    dh, dc = np.zeros((m, H)), np.zeros((m, H))           # nothing arrives after the last step
    for t in reversed(range(T)):                          # backward through time
        dh = dh + dHs[t]                                    # NEW: this step's own output adds its gradient
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


def loss_and_grads(X, Y, P):                            # NEW: softmax at every step, mean loss over every step
    Hs, Cs, gates = lstm_forward(X, P["W"], P["b"])       # forward through the LSTM
    m, T = Y.shape                                        # batch, steps
    loss, dHs = 0.0, []                                   # total loss, and the gradient at each step's h
    g = {"Why": np.zeros_like(P["Why"]), "by": np.zeros_like(P["by"])}  # output-layer gradients
    for t in range(T):                                    # every step has its own output
        p = softmax(Hs[t] @ P["Why"] + P["by"])             # this step's prediction
        loss -= np.log(p[np.arange(m), Y[:, t]]).sum() / (m * T)  # cross-entropy, averaged over batch AND steps
        dlog = p.copy(); dlog[np.arange(m), Y[:, t]] -= 1; dlog /= m * T  # softmax + cross-entropy gradient
        g["Why"] += Hs[t].T @ dlog; g["by"] += dlog.sum(axis=0)  # accumulate the output layer's gradients
        dHs.append(dlog @ P["Why"].T)                       # what this step's output sends back to h_t
    g["W"], g["b"] = lstm_backward(dHs, X, P["W"], Hs, Cs, gates)  # backward through time
    return loss, g, Hs                                    # loss, all gradients, and the hidden states


def make_batch(rng, m, T, V):                           # the task: output the token seen two steps earlier
    tokens = rng.randint(1, V, size=(m, T))               # random tokens 1..V-1 (0 means "nothing yet")
    Y = np.zeros((m, T), dtype=int)                       # targets: 0 for the first two steps ...
    Y[:, 2:] = tokens[:, :-2]                             # ... then the token from two steps back
    return np.eye(V)[tokens], Y                           # one-hot inputs, integer targets


V, H = 5, 16                                              # 5 symbols (0 = nothing yet), 16 hidden units
rng = np.random.RandomState(0)                            # fixed rng
P = {"W": rng.randn(V + H, 4 * H) / np.sqrt(V + H), "b": np.zeros(4 * H),  # LSTM weights
     "Why": rng.randn(H, V) / np.sqrt(H), "by": np.zeros(V)}  # output layer
P["b"][:H] = 1.0                                          # forget bias 1, the usual start

X, Y = make_batch(rng, 3, 6, V)                           # a tiny batch for the gradient check
_, g, _ = loss_and_grads(X, Y, P)                         # analytic gradients
worst = 0.0                                               # largest |numeric - analytic| over every entry
for name in P:                                            # every parameter
    for pos in np.ndindex(P[name].shape):                   # every entry
        old = P[name][pos]                                    # remember it
        P[name][pos] = old + 1e-5; up = loss_and_grads(X, Y, P)[0]    # nudge up
        P[name][pos] = old - 1e-5; down = loss_and_grads(X, Y, P)[0]  # nudge down
        P[name][pos] = old                                    # restore
        worst = max(worst, abs((up - down) / 2e-5 - g[name][pos]))  # compare
print(f"gradient check, every entry of W, b, Why, by: largest |numeric - analytic| = {worst:.2e}")

def test_accuracy():                                    # accuracy on the same 500 test sequences every time
    X, Y = make_batch(np.random.RandomState(99), 500, 12, V)  # fixed test sequences
    _, _, Hs = loss_and_grads(X, Y, P)                    # forward pass on them
    pred = np.stack([(h @ P["Why"] + P["by"]).argmax(axis=1) for h in Hs], axis=1)  # a prediction per step
    return (pred == Y).mean(), (pred[:, 2:] == Y[:, 2:]).mean()  # every step, and steps 3-12 (where memory is needed)


acc_all, acc_mem = test_accuracy()                        # before any training
print(f"before training: test accuracy at every step {acc_all:.3f}, steps 3-12 only {acc_mem:.3f} "
      f"(guessing among 4 tokens: 0.250)")
for step in range(1, 1501):                               # plain SGD on fresh batches
    X, Y = make_batch(rng, 32, 12, V)                       # 32 sequences of 12 tokens
    loss, g, Hs = loss_and_grads(X, Y, P)                   # forward and backward
    norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))  # Day 60's global norm
    for name in P:                                          # update, clipped at norm 1.0
        P[name] -= 0.5 * g[name] * min(1.0, 1.0 / norm)
    if step % 300 == 0:                                     # progress report
        acc_all, acc_mem = test_accuracy()                    # after this many updates
        print(f"step {step:>5}  training loss {loss:.4f}  test accuracy at every step {acc_all:.3f}"
              f"  (steps 3-12 only: {acc_mem:.3f})")
print("An untrained LSTM already beats guessing on steps 3-12: its state mixes the last few tokens, "
      "and the random output layer happens to line up with some of them.")
