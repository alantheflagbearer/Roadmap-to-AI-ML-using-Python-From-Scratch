"""
Day 92 Exercise -- A Learning-Rate Schedule and an Honest Evaluator

The second of Week 14's parts for Day 97's build.

    lr_schedule(step, peak, warmup, total, floor)   linear warm-up, then half a cosine down to floor * peak
    eval_bits(logp_fn, ids, T)                       held-out bits per token with full context: windows every
                                                     T/2 tokens, only their last T/2 positions scored

eval_bits takes the model as a function (ids of shape (m, T) -> log-probabilities
(m, T, V)), so the same evaluator scores any model. Checked: the schedule
starts small, peaks at the end of the warm-up and ends at the floor; every
scored token is scored exactly once; a uniform model scores log2(V); and a
model that knows the next token exactly scores 0.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def lr_schedule(step, peak, warmup, total, floor=0.1):  # NEW: linear warm-up, then cosine decay to floor * peak
    if step < warmup:                                     # the first steps
        return peak * (step + 1) / warmup                   # grow linearly
    progress = (step - warmup) / max(1, total - warmup)   # 0 after the warm-up, 1 at the end
    return peak * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * min(1.0, progress))))  # half a cosine


def eval_bits(logp_fn, ids, T):                         # NEW: bits per token, every scored token with at least T/2 tokens of context
    half = T // 2                                         # stride and scored positions per window
    starts = np.arange(0, len(ids) - T - 1, half)         # window starts
    nats, scored = 0.0, []                                # summed -ln p; which positions were scored
    for s in range(0, len(starts), 32):                   # 32 windows at a time
        st = starts[s:s + 32]
        x = np.stack([ids[i:i + T] for i in st]); y = np.stack([ids[i + 1:i + T + 1] for i in st])  # inputs, next tokens
        lp = logp_fn(x)[:, half:]                           # the model's log-probabilities, last half only
        nats -= np.take_along_axis(lp, y[:, half:, None], axis=-1).sum()
        scored += [i + 1 + j for i in st for j in range(half, T)]  # the text positions these targets sit at
    return nats / len(scored) / np.log(2), np.array(scored)  # bits per token, and the scored positions


steps = [0, 99, 199, 200, 2000, 4999]                     # a few steps of a 5,000-step run with a 200-step warm-up
print("schedule, peak 1:", ", ".join(f"{s}: {lr_schedule(s, 1.0, 200, 5000):.3f}" for s in steps))
lrs = np.array([lr_schedule(s, 1.0, 200, 5000) for s in range(5000)])
print(f"largest value {lrs.max():.3f} at step {lrs.argmax()}; never rises after the warm-up: {np.all(np.diff(lrs[200:]) <= 1e-12)}")

rng = np.random.RandomState(0)                            # fixed rng
V, T = 10, 16                                             # 10 tokens, context 16
ids = rng.randint(0, V, 1000)                             # a random 'text'
uniform = lambda x: np.full(x.shape + (V,), -np.log(V))   # a model that knows nothing
bits, pos = eval_bits(uniform, ids, T)
print(f"uniform model: {bits:.4f} bits per token (log2 {V} = {np.log2(V):.4f}); {len(pos)} positions scored, "
      f"each once: {len(set(pos)) == len(pos)}, all after the first {T // 2}: {pos.min() > T // 2}")
count_up = np.arange(1000) % V                            # a 'text' where the next token is always the current one plus 1 (mod V)
def knows(x):                                             # a model that knows that rule: probability 1 on (x + 1) mod V
    lp = np.full(x.shape + (V,), -1e9)                      # almost no probability anywhere ...
    np.put_along_axis(lp, ((x + 1) % V)[..., None], 0.0, axis=-1)  # ... except log 1 = 0 on the right token
    return lp
print(f"a model that always knows the next token: {eval_bits(knows, count_up, T)[0]:.4f} bits per token")
