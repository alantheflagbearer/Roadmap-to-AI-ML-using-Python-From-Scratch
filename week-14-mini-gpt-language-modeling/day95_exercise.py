"""
Day 95 Exercise -- Counting Weights and Fitting a Power Law

The fifth of Week 14's parts for Day 97's build.

    count_weights(V, d, layers, t, ff)   weights of Day 90's GPT by formula
    fit_power_law(N, L)                  L = c + a N^-alpha: try floors c on a grid, fit a line in log-log space

Checked: the formula against a model built the slow way (summing array sizes
of the same shapes), and the fit on made-up losses with a known c, a and
alpha, without and with noise -- to see how well four points pin alpha down.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def count_weights(V, d, layers, t, ff=None):            # NEW: Day 90's GPT, by formula
    ff = ff or 4 * d                                      # feed-forward width
    block = 4 * d * d + 2 * d * ff + ff + d + 4 * d       # Wq Wk Wv Wo; W1 W2; c1 c2; two LayerNorms
    return V * d + t * d + layers * block + d * V + V     # token and position tables, blocks, output layer and bias


def fit_power_law(N, L):                                # NEW: L = c + a N^-alpha
    N, L = np.asarray(N, float), np.asarray(L, float)
    best = None                                           # (squared error, c, a, alpha)
    for c in np.linspace(0, L.min() * 0.999, 400):        # the floor must be below every loss
        slope, icpt = np.polyfit(np.log(N), np.log(L - c), 1)  # log(L - c) = log a - alpha log N
        err = np.sum((c + np.exp(icpt) * N ** slope - L) ** 2)
        if best is None or err < best[0]:
            best = (err, c, np.exp(icpt), -slope)
    return best[1:]


shapes = lambda V, d, L, t: [(V, d), (t, d)] + L * [(d, d)] * 4 + L * [(d, 4 * d), (4 * d,), (4 * d, d), (d,)] + L * [(d,)] * 4 + [(d, V), (V,)]
for d in (16, 64):                                        # two widths
    slow = sum(int(np.prod(s)) for s in shapes(65, d, 2, 64))  # every array of Day 90's model, counted one by one
    print(f"width {d}: formula {count_weights(65, d, 2, 64)}, counted array by array {slow}")
N = np.array([1e4, 3e4, 1e5, 4e5])                        # four model sizes
L = 1.5 + 30 * N ** -0.35                                 # losses from a known law: c = 1.5, a = 30, alpha = 0.35
c, a, al = fit_power_law(N, L)
print(f"exact losses: fitted c = {c:.3f}, a = {a:.1f}, alpha = {al:.3f}  (true 1.5, 30, 0.35)")
rng = np.random.RandomState(0)                            # noise like a single training run's
for trial in range(3):                                    # three noisy versions
    Ln = L + 0.01 * rng.randn(4)                            # +-0.01 bits
    c, a, al = fit_power_law(N, Ln)
    print(f"with noise of 0.01 bits, try {trial + 1}: fitted c = {c:.3f}, alpha = {al:.3f}")
