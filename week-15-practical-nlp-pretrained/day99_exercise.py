"""
Day 99 Exercise -- Bootstrap Intervals, Separate and Paired

The second of Week 15's parts for Day 104's build, which compares several
question-answering setups on the same questions.

    bootstrap_ci(values, n_boot, seed)        95% interval for a mean (an accuracy, an F1), by resampling examples
    paired_bootstrap(a, b, n_boot, seed)      95% interval for mean(a) - mean(b) on the SAME examples, and how often a is ahead

Checked on made-up results with a known answer: how often a 95% interval
covers the true accuracy, and why the paired interval for a difference is
narrower than the two separate intervals suggest when two systems agree on
most examples.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def bootstrap_ci(values, n_boot=2000, seed=0):          # NEW: middle 95% of the mean over resamples of the examples
    rng = np.random.RandomState(seed)                     # fixed rng
    idx = rng.randint(0, len(values), (n_boot, len(values)))  # each row: one resample, with replacement
    means = np.asarray(values, float)[idx].mean(axis=1)   # the mean of each resample
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired_bootstrap(a, b, n_boot=2000, seed=0):        # NEW: the difference, resampling examples for both systems together
    a, b = np.asarray(a, float), np.asarray(b, float)     # per-example scores of the two systems
    rng = np.random.RandomState(seed)                     # fixed rng
    idx = rng.randint(0, len(a), (n_boot, len(a)))        # the SAME resampled examples for both
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)         # difference per resample
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


rng = np.random.RandomState(0)                          # fixed rng for the made-up results
true_acc, n = 0.85, 436                                 # a system that is right 85% of the time, a test set of 436
covered = 0                                             # how many intervals contain the true accuracy
for trial in range(200):                                # 200 imaginary test sets
    c = (rng.rand(n) < true_acc).astype(float)            # right/wrong on each example
    lo, hi = bootstrap_ci(c, n_boot=1000, seed=trial)
    covered += lo <= true_acc <= hi
print(f"200 test sets of {n} from a system with true accuracy {true_acc}: the 95% interval covered it {covered} times ({covered / 200:.3f})")

base = (rng.rand(n) < 0.85).astype(float)               # system B's right/wrong
a = base.copy()                                         # system A: the same, except ...
flip = rng.choice(np.where(base == 0)[0], 13, replace=False)  # ... it fixes 13 of B's mistakes
a[flip] = 1.0
print(f"\nA = B plus 13 fixed mistakes: accuracy A {a.mean():.3f}, B {base.mean():.3f}, they agree on {np.mean(a == base):.3f} of examples")
la, ha = bootstrap_ci(a); lb, hb = bootstrap_ci(base)
print(f"separate intervals: A [{la:.3f}, {ha:.3f}], B [{lb:.3f}, {hb:.3f}] -> they overlap: {la <= hb}")
m, lo, hi, share = paired_bootstrap(a, base)
print(f"paired interval for A - B: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], A ahead in {share:.3f} of resamples -> excludes 0: {lo > 0}")
b2 = np.roll(a, 7)                                      # A's results shifted by 7 examples: the same accuracy, errors on unrelated examples
m, lo, hi, share = paired_bootstrap(a, b2)
print(f"A against a system with the same accuracy but unrelated errors: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> excludes 0: {lo > 0 or hi < 0}")
