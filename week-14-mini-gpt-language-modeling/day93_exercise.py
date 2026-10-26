"""
Day 93 Exercise -- One Function for Every Sampling Strategy

The third of Week 14's parts for Day 97's build.

    sample_next(logits, rng, temperature, top_k, top_p)
        temperature 0   greedy: the most probable token
        temperature t   softmax(logits / t)
        top_k k         keep the k most probable tokens
        top_p p         keep the smallest set whose probability reaches p
        then renormalise and draw (inverse CDF), one token per row

Checked against exact distributions with many draws, and at the two limits:
a very low temperature behaves like greedy, a very high one like a uniform
draw.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sample_next(logits, rng, temperature=1.0, top_k=None, top_p=None):  # NEW: one token per row of logits (m, V)
    if temperature == 0:                                  # greedy
        return logits.argmax(axis=-1)
    z = logits / temperature                              # sharpen or flatten
    p = np.exp(z - z.max(axis=-1, keepdims=True)); p /= p.sum(axis=-1, keepdims=True)  # softmax
    if top_k is not None:                                 # the k most probable tokens
        p = np.where(p >= np.sort(p, axis=-1)[:, -top_k][:, None], p, 0.0)
    if top_p is not None:                                 # the smallest set reaching top_p
        order = np.argsort(-p, axis=-1); sp = np.take_along_axis(p, order, axis=-1)  # sorted, most probable first
        keep = np.zeros_like(p, dtype=bool); np.put_along_axis(keep, order, np.cumsum(sp, axis=-1) - sp < top_p, axis=-1)
        p = np.where(keep, p, 0.0)
    p /= p.sum(axis=-1, keepdims=True)                    # renormalise
    return np.minimum((np.cumsum(p, axis=-1) < rng.rand(len(p), 1)).sum(axis=-1), p.shape[-1] - 1)  # inverse-CDF draw


rng = np.random.RandomState(0)                          # fixed rng
p = np.array([0.4, 0.3, 0.2, 0.1]); lg = np.log(p)[None].repeat(100000, axis=0)  # four tokens, 100,000 rows
freq = lambda **kw: np.bincount(sample_next(lg, rng, **kw), minlength=4) / len(lg)  # observed frequencies
print("temperature 1  :", np.round(freq(), 3), "expected", p)
print("top-k 2        :", np.round(freq(top_k=2), 3), "expected", np.round(np.r_[p[:2], 0, 0] / 0.7, 3))
print("top-p 0.65     :", np.round(freq(top_p=0.65), 3), "expected", np.round(np.r_[p[:2], 0, 0] / 0.7, 3), "(0.4 < 0.65, 0.4 + 0.3 >= 0.65: two tokens)")
print("temperature 0.05:", np.round(freq(temperature=0.05), 3), "-> almost always the top token, like greedy", sample_next(lg[:5], rng, temperature=0))
print("temperature 50 :", np.round(freq(temperature=50), 3), "-> close to uniform (0.25 each)")
