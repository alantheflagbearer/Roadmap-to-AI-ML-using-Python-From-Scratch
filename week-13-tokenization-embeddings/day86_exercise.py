"""
Day 86 Exercise -- The Output Layer: a Full Softmax Over the Vocabulary

The third of Week 13's parts for Day 90's tiny GPT-style model. Negative
sampling is a shortcut for learning word vectors; a language model needs
the real thing: a probability for EVERY token in the vocabulary, at every
position. That is the layer skip-gram's full-softmax version uses:

    logits = h W + b                  (one score per vocabulary token)
    loss   = mean of -log softmax(logits)[y]
    dlogits = (softmax(logits) - one_hot(y)) / (number of positions)

It works on any leading shape: (positions, d) for skip-gram, (batch, time,
d) for a sequence model. Checked three ways: equal scores give log V, a
numerical gradient check of every weight and input, and a tiny next-word
model (a full-softmax skip-gram whose window is one word to the right) that
learns which words follow which in a 12-word text.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def output_layer(h, W, b):                              # NEW: (..., d) -> (..., V) scores
    return h @ W + b                                      # one score per vocabulary token


def softmax_xent(logits, y):                            # NEW: mean cross-entropy, and its gradient for the logits
    z = logits - logits.max(axis=-1, keepdims=True)       # subtract the largest score: same softmax, no overflow
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))  # log-softmax
    flat, yf = logp.reshape(-1, logp.shape[-1]), y.reshape(-1)  # every position as one row
    loss = -flat[np.arange(len(yf)), yf].mean()           # mean of -log p(correct token)
    d = np.exp(flat); d[np.arange(len(yf)), yf] -= 1      # softmax minus one-hot
    return loss, (d / len(yf)).reshape(logits.shape)      # loss, dloss/dlogits


def output_layer_backward(dlogits, h, W):               # NEW: gradients for h, W and b
    f = lambda T: T.reshape(-1, T.shape[-1])              # every position as one row
    return dlogits @ W.T, f(h).T @ f(dlogits), f(dlogits).sum(axis=0)  # dh, dW, db


rng = np.random.RandomState(0)                          # fixed rng
V, d = 7, 4                                               # 7 tokens, 4 numbers in
h = rng.randn(2, 3, d); W = rng.randn(d, V) * 0.5; b = rng.randn(V) * 0.1  # a (batch 2, time 3) input and the layer
y = rng.randint(0, V, size=(2, 3))                        # the right token at each position

print(f"equal scores for all {V} tokens: loss {softmax_xent(np.zeros((5, V)), np.zeros(5, int))[0]:.4f}, log V = {np.log(V):.4f}")

loss_of = lambda: softmax_xent(output_layer(h, W, b), y)[0]  # the loss as a function of the current arrays
loss, dl = softmax_xent(output_layer(h, W, b), y)          # forward
dh, dW, db = output_layer_backward(dl, h, W)               # analytic gradients
worst, ok, count = 0.0, True, 0                            # largest difference; every entry passes; entries
for A, G in ((h, dh), (W, dW), (b, db)):                   # input, weights, bias
    for pos in np.ndindex(A.shape):                          # every entry
        old = A[pos]                                           # remember it
        A[pos] = old + 1e-6; up = loss_of()                    # nudge up
        A[pos] = old - 1e-6; dn = loss_of()                    # nudge down
        A[pos] = old                                           # restore
        num = (up - dn) / 2e-6; count += 1                     # central difference
        worst = max(worst, abs(num - G[pos])); ok &= abs(num - G[pos]) <= 1e-10 + 1e-6 * abs(num)
print(f"gradient check: {count} entries (h, W, b), largest |numeric - analytic| = {worst:.2e}, all pass: {ok}")

text = "the cat sat on the mat the dog sat on the log".split()  # a 12-word text
vocab = sorted(set(text)); ix = {w: i for i, w in enumerate(vocab)}  # 7 words
c = np.array([ix[w] for w in text[:-1]]); o = np.array([ix[w] for w in text[1:]])  # (word, next word) pairs
E = rng.randn(len(vocab), 8) * 0.1; W2 = np.zeros((8, len(vocab))); b2 = np.zeros(len(vocab))  # a table and the output layer
for step in range(301):                                    # plain gradient descent on the 11 pairs
    loss, dl = softmax_xent(output_layer(E[c], W2, b2), o)   # forward
    dh, dW2, db2 = output_layer_backward(dl, E[c], W2)       # backward
    dE = np.zeros_like(E); np.add.at(dE, c, dh)              # Day 85's embedding backward
    E -= 1.0 * dE; W2 -= 1.0 * dW2; b2 -= 1.0 * db2          # update
    if step in (0, 100, 300):                                # a few checkpoints
        print(f"toy next-word model, step {step:>3}: loss {loss:.3f}")
p = np.exp(output_layer(E[ix["sat"]], W2, b2)); p /= p.sum()  # what follows 'sat'?
print(f"after 'sat': p(on) = {p[ix['on']]:.3f}; after 'the': " +
      ", ".join(f"p({w}) = {q:.2f}" for w, q in zip(vocab, (lambda e: e / e.sum())(np.exp(output_layer(E[ix['the']], W2, b2)))) if q > 0.05))
