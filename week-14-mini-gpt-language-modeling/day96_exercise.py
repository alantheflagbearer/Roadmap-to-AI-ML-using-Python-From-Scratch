"""
Day 96 Exercise -- Model Files and Fine-Tuning Helpers

The sixth and last of Week 14's parts for Day 97's build, which loads two
models (Day 92's and Day 96's) and compares them.

    save_params(path, P)        nested dictionaries of arrays -> one .npz file ('blocks.0.W' stored as 'blocks__0__W')
    load_params(path)           and back
    adam_subset(P, G, state, lr, keys)   Adam on the named weights only: everything else stays frozen
    snapshot(P)                 a deep copy, for keeping the best checkpoint

Checked: a file round trip is exact; frozen weights do not move by even one
bit while the trainable ones do; a snapshot does not change when the model
keeps training.
"""
import os                                             # os: a temporary file path
import tempfile                                       # tempfile: somewhere to write the test file

import numpy as np                                    # numpy: every array/matrix operation below is built on this


def leaves(D, prefix=""):                               # every array inside nested dictionaries, with a dotted name (Day 82's exercise)
    for k, v in D.items():
        if isinstance(v, dict):
            yield from leaves(v, prefix + k + ".")
        else:
            yield prefix + k, v


def save_params(path, P):                               # NEW: one compressed file, one named array per weight
    np.savez_compressed(path, **{k.replace(".", "__"): v for k, v in leaves(P)})


def load_params(path):                                  # NEW: the file back into nested dictionaries
    P, f = {}, np.load(path)                              # result; the file
    for k in f.files:                                     # every stored array
        parts = k.split("__"); node = P                     # 'blocks__0__W' -> ['blocks', '0', 'W']
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = f[k]
    return P


def adam_subset(P, G, state, lr, keys):                 # NEW: Adam (Day 37's four lines) on the named weights only
    g = dict(leaves(G)); state["t"] += 1; t = state["t"]  # gradients by name; step count
    for k, A in leaves(P):                                # every weight ...
        if k not in keys:                                   # ... frozen ones are skipped
            continue
        m = state["m"].setdefault(k, np.zeros_like(A)); v = state["v"].setdefault(k, np.zeros_like(A))
        m[:] = 0.9 * m + 0.1 * g[k]; v[:] = 0.999 * v + 0.001 * g[k] ** 2
        A -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def snapshot(P):                                        # NEW: a deep copy of nested dictionaries of arrays
    return {k: (snapshot(v) if isinstance(v, dict) else v.copy()) for k, v in P.items()}


rng = np.random.RandomState(0)                          # fixed rng
P = {"blocks": {"0": {"W": rng.randn(4, 4)}, "1": {"W": rng.randn(4, 4)}}, "Wout": rng.randn(4, 3)}  # a tiny 'model'
path = os.path.join(tempfile.mkdtemp(), "model.npz")    # a temporary file
save_params(path, P); Q = load_params(path)             # round trip
print(f"round trip: same names {sorted(k for k, _ in leaves(Q)) == sorted(k for k, _ in leaves(P))}, "
      f"largest |difference| {max(np.abs(Q_ - P_).max() for (_, Q_), (_, P_) in zip(sorted(leaves(Q)), sorted(leaves(P)))):.1e}")
before = snapshot(P)                                    # the weights before training (a deep copy)
raw = P["Wout"].copy()                                  # an independent copy of one array, to check the snapshot against
shallow = dict(P)                                       # a shallow copy: a new dictionary holding the SAME arrays
keys = {"blocks.1.W", "Wout"}                           # 'top only': the last block and the output layer
state = {"t": 0, "m": {}, "v": {}}                      # Adam's state
for _ in range(5):                                      # five steps with random gradients
    G = {"blocks": {"0": {"W": rng.randn(4, 4)}, "1": {"W": rng.randn(4, 4)}}, "Wout": rng.randn(4, 3)}
    adam_subset(P, G, state, 0.1, keys)
for k, A in leaves(P):                                  # what moved
    B = dict(leaves(before))[k]
    print(f"{k:<12} trainable: {k in keys!s:<6} unchanged bit for bit: {np.array_equal(A, B)}")
print(f"after training, the snapshot still equals the old values: {np.array_equal(before['Wout'], raw)}; "
      f"the shallow copy does not (it shares the arrays): {np.array_equal(shallow['Wout'], raw)}")
