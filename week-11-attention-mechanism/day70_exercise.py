"""
Day 70 Exercise -- Reading From a Memory With Soft Weights

Today's lesson showed the bottleneck: the decoder sees ONE summary of the
input. The way out (Day 71) is to keep EVERY encoder state as a memory and
let the decoder read a weighted average of them. This exercise isolates
that read on its own, with no training:

  scores (one per memory slot) -> masked softmax -> weights -> weighted sum

It checks three things: a peaked score vector reads back one slot almost
exactly, equal scores read back the plain average, and padded slots get
weight exactly 0. Then it checks the gradient of the read numerically.
Week 11's integration day (76) is built from this and the next exercises.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def masked_softmax(scores, mask):                      # softmax over the real slots only; padded slots get weight 0
    s = np.where(mask, scores, -np.inf)                   # a padded slot's score becomes minus infinity
    s = s - s.max(axis=-1, keepdims=True)                 # subtract the largest real score, for stability
    e = np.exp(s)                                         # exp(-inf) = 0, so padding contributes nothing
    return e / e.sum(axis=-1, keepdims=True)              # weights over the real slots, summing to 1


def soft_read(memory, scores, mask):                   # NEW: read a weighted average of the memory slots
    w = masked_softmax(scores, mask)                      # (m, n) weights
    return np.einsum("mn,mnd->md", w, memory), w          # (m, d) read vector, and the weights used


rng = np.random.RandomState(0)                          # fixed rng
m, n, d = 2, 5, 3                                         # 2 sequences, 5 memory slots each, 3 numbers per slot
memory = rng.randn(m, n, d)                               # e.g. one encoder state per input step
mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], bool)  # the second sequence has only 3 real steps

peaked = np.full((m, n), 0.0); peaked[:, 1] = 20.0       # a score vector that strongly prefers slot 2
read, w = soft_read(memory, peaked, mask)
print("peaked scores -> weights:", w.round(4)[0], "; read minus slot 2:", f"{np.abs(read - memory[:, 1]).max():.2e}")

read, w = soft_read(memory, np.zeros((m, n)), mask)      # equal scores
plain_avg = memory[1, :3].mean(axis=0)                    # the average of the 3 REAL slots of sequence 2
print("equal scores, sequence 2 weights:", w[1].round(4), "; read minus average of its 3 real slots:",
      f"{np.abs(read[1] - plain_avg).max():.2e}")

scores = rng.randn(m, n)                                  # random scores
U = rng.randn(m, d)                                       # a random direction: L = sum(U * read)
read, w = soft_read(memory, scores, mask)
dread = U                                                 # dL/dread
dmemory = w[:, :, None] * dread[:, None, :]               # NEW: read = sum_i w_i memory_i  ->  dL/dmemory_i = w_i dread
dw = np.einsum("md,mnd->mn", dread, memory)               # dL/dw_i = dread . memory_i
dscores = w * (dw - (w * dw).sum(axis=1, keepdims=True))  # back through the softmax (zero where w = 0)

worst = 0.0                                               # largest |numeric - analytic|
for P, G in [(memory, dmemory), (scores, dscores)]:       # both inputs
    for pos in np.ndindex(P.shape):
        old = P[pos]
        P[pos] = old + 1e-6; up = (U * soft_read(memory, scores, mask)[0]).sum()
        P[pos] = old - 1e-6; dn = (U * soft_read(memory, scores, mask)[0]).sum()
        P[pos] = old
        worst = max(worst, abs((up - dn) / 2e-6 - G[pos]))
print(f"gradient check (memory and scores): largest |numeric - analytic| = {worst:.2e}; "
      f"gradient at padded slots exactly 0: {bool(np.all(dmemory[~mask] == 0) and np.all(dscores[~mask] == 0))}")
