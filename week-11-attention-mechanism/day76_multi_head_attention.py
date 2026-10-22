"""
Day 76: The Week 11 Build -- Multi-Head Attention From the Exercise Parts

Week 11's last day does two jobs at once, like Days 62 and 69 did. The
syllabus topic is MULTI-HEAD ATTENTION: instead of one attention read of
size d, run h smaller reads of size d/h side by side, each with its own
slice of the query, key and value projections, then join them:

    Q = X Wq,  K = X Wk,  V = X Wv           (as on Day 74)
    split Q, K, V into h heads of size d/h
    head_i = softmax( Q_i K_i^T / sqrt(d/h) ) V_i    (Day 75's function, once per head)
    output = concat(head_1, ..., head_h) Wo

And this is the week's integration day: the model is assembled from the
short exercise scripts of Days 70-75.

The parts, and where each one goes:

  Day 70 exercise  masked softmax, weighted read     -> inside Day 75's function (its mask and its read)
  Day 71 exercise  additive read                     -> not used: the Transformer scores with dot products
  Day 72 exercise  alignment_metrics                 -> what each trained head looks at (Section 6)
  Day 73 exercise  dot and general reads             -> the dot score, with learned projections in front
  Day 74 exercise  Q, K, V from the SAME sequence     -> the projections of the multi-head layer
  Day 75 exercise  scaled_dot_product_attention      -> run once per head, verbatim

The test is a task built so that ONE position must fetch TWO things at
once: two-key associative recall. Each example shows 16 (key, value)
pairs and asks for two keys; the model must answer both values. With one
head the query position has one softmax: it can look at both pairs, but
then it reads their AVERAGE. With two heads it has two softmaxes.

Every model below has the same size d = 64, so the same number of
weights whatever h is: multi-head attention divides the size, it does not
add to it.

What the lesson measures, reported as it came out:
  1. The parts fit: a loop of Day 75's function over the heads gives the
     same answer as one call on all heads; the weight count does not depend on h.
  2. A gradient check through the whole model, one and two heads.
  3. Two-key recall with 1, 2, 4 and 8 heads, three seeds.
  4. The control: ordinary one-key recall, where one head is enough.
  5. Why one head fails: a bigger head, four times the training, and
     attention FORCED onto the two right pairs.
  6. What each trained head looks at (Day 72's measurements).
  7. The cost of more heads.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import time                                            # time: perf_counter, to time training steps

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

N_KEYS = 32                                             # Day 75's recall task: 32 possible keys and 32 possible values
N_PAIRS = 16                                            # each example shows 16 (key, value) pairs
D = 64                                                  # model size: every model has d = 64, split into h heads
HEADS = (1, 2, 4, 8)                                    # head counts compared (head sizes 64, 32, 16, 8)
SEEDS = [0, 1, 2]                                       # three seeds for every trained comparison
STEPS = 2000                                            # Day 75's training budget
CHECKS = (100, 250, 500, 1000, 2000)                    # Day 75's checkpoints
COLORS = {1: "#d62728", 2: "#1f77b4", 4: "#2ca02c", 8: "#9467bd"}  # one colour per head count, in every plot


# =============================================================================
# 0. THE PARTS -- verbatim from the exercise scripts of Days 72 and 75
# =============================================================================
def scaled_dot_product_attention(Q, K, V, mask=None):  # day75_exercise.py, verbatim: Q (m, nq, d), K (m, nk, d), V (m, nk, dv)
    d = Q.shape[-1]                                       # the query and key size
    S = np.einsum("mid,mjd->mij", Q, K) / np.sqrt(d)      # scaled scores
    if mask is not None:                                  # Day 70's masking, if there are padded keys
        S = np.where(mask, S, -np.inf)                      # masked keys get minus infinity ...
    e = np.exp(S - S.max(axis=-1, keepdims=True))         # ... and so weight exactly 0
    A = e / e.sum(axis=-1, keepdims=True)                 # weights over the keys
    return np.einsum("mij,mjd->mid", A, V), A             # outputs, and the weights (the backward pass needs them)


def scaled_dot_product_attention_backward(dZ, Q, K, V, A):  # day75_exercise.py, verbatim: gradients for Q, K and V
    d = Q.shape[-1]                                       # the query and key size
    dA = np.einsum("mid,mjd->mij", dZ, V)                 # Z = A V
    dV = np.einsum("mij,mid->mjd", A, dZ)                 # ... into the values
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True)) / np.sqrt(d)  # back through the softmax and the scaling
    return np.einsum("mij,mjd->mid", dS, K), np.einsum("mij,mid->mjd", dS, Q), dV  # dQ, dK, dV


def alignment_metrics(W, target):                       # day72_exercise.py, verbatim: W (rows, n) of weights; target[j] = right input for row j
    j = np.arange(len(target))                            # one index per row
    entropy = -(W * np.log(W + 1e-12)).sum(axis=1)        # one number per row
    return (W.argmax(axis=1) == target).mean(), W[j, target].mean(), entropy.mean()  # argmax on target, weight on target, entropy


def softmax(z):                                         # Days 57-75: weights along the last axis, summing to 1
    e = np.exp(z - z.max(axis=-1, keepdims=True))         # subtract the row max for stability, exponentiate
    return e / e.sum(axis=-1, keepdims=True)              # normalise


# =============================================================================
# 1. MULTI-HEAD ATTENTION -- NEW: split, Day 75's function per head, join
# =============================================================================
def split_heads(T, h):                                  # NEW: (m, n, d) -> (m * h, n, d / h): each head becomes its own batch entry
    m, n, d = T.shape                                     # batch, positions, size
    return T.reshape(m, n, h, d // h).transpose(0, 2, 1, 3).reshape(m * h, n, d // h)  # head i = columns i*d/h .. (i+1)*d/h - 1


def merge_heads(T, h):                                  # NEW: (m * h, n, d / h) -> (m, n, d): the heads side by side again
    mh, n, dk = T.shape                                   # batch times heads, positions, head size
    return T.reshape(mh // h, h, n, dk).transpose(0, 2, 1, 3).reshape(mh // h, n, h * dk)  # the exact inverse of split_heads


def multi_head_attention(X, P, h, mask=None):           # NEW: X (m, n, d) -> (m, n, d), and a cache for the backward pass
    Q, K, V = (split_heads(X @ P[w], h) for w in ("Wq", "Wk", "Wv"))  # Day 74's three projections, cut into h heads
    Zh, A = scaled_dot_product_attention(Q, K, V, mask)   # Day 75's function, every head at once (heads are batch entries)
    Z = merge_heads(Zh, h)                                # the heads' outputs side by side
    return Z @ P["Wo"], dict(X=X, Q=Q, K=K, V=V, A=A, Z=Z)  # the output projection mixes the heads


def multi_head_attention_backward(dO, P, h, c, scores=True):  # NEW: gradients for X, Wq, Wk, Wv, Wo
    flat = lambda T: T.reshape(-1, T.shape[-1])           # (m, n, d) -> (m * n, d)
    g = dict(Wo=flat(c["Z"]).T @ flat(dO))                # output = Z Wo
    dZh = split_heads(dO @ P["Wo"].T, h)                  # back to the heads
    dQ, dK, dV = scaled_dot_product_attention_backward(dZh, c["Q"], c["K"], c["V"], c["A"])  # Day 75's backward, every head
    if not scores:                                        # Section 5's forced attention: the weights did not come from the scores
        dQ, dK = np.zeros_like(dQ), np.zeros_like(dK)       # so no gradient reaches the queries or keys
    dQ, dK, dV = merge_heads(dQ, h), merge_heads(dK, h), merge_heads(dV, h)  # the heads side by side again
    X = c["X"]                                            # the layer's input
    g.update(Wq=flat(X).T @ flat(dQ), Wk=flat(X).T @ flat(dK), Wv=flat(X).T @ flat(dV))  # the three projections
    dX = dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T  # X made all three (Day 74)
    return dX, g                                          # input gradient and weight gradients


# =============================================================================
# 2. THE TWO-KEY RECALL TASK -- NEW (Day 75's task, asked about two keys)
# =============================================================================
def make_batch(rng, m, two=True, n_keys=N_KEYS, n_pairs=N_PAIRS):  # NEW: 16 pairs and two asked keys per example
    keys = np.array([rng.permutation(n_keys)[:n_pairs] for _ in range(m)])  # 16 different keys per example (Day 75)
    vals = rng.randint(0, n_keys, size=(m, n_pairs))      # a random value for each key (Day 75)
    pa = rng.randint(0, n_pairs, size=m)                  # the first pair asked about
    pb = (pa + rng.randint(1, n_pairs, size=m)) % n_pairs  # the second: always a different pair
    if not two:                                           # Section 4's control: ask about the same key twice
        pb = pa                                             # one pair to fetch, not two
    r = np.arange(m)                                      # example index
    return keys, vals, keys[r, pa], keys[r, pb], vals[r, pa], vals[r, pb], pa, pb  # pairs, asked keys, answers, positions


class TwoKeyRecall:                                     # NEW: pair tokens + one query token -> multi-head self-attention -> two answers
    def __init__(self, d, heads, seed=0, lr=0.1, clip=1.0, n_keys=N_KEYS):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.d, self.h, self.lr, self.clip = d, heads, lr, clip  # size, heads, training settings
        self.P = dict(Ek=rng.randn(n_keys, d),             # key embeddings, spread 1 (Day 75)
                      Ev=rng.randn(n_keys, d),             # value embeddings
                      Ea=rng.randn(n_keys, d),             # the FIRST asked key, as the query token sees it
                      Eb=rng.randn(n_keys, d),             # the SECOND asked key: a separate table, so the two are told apart
                      Wq=rng.randn(d, d) / np.sqrt(d),     # query projection (all heads together)
                      Wk=rng.randn(d, d) / np.sqrt(d),     # key projection
                      Wv=rng.randn(d, d) / np.sqrt(d),     # value projection
                      Wo=rng.randn(d, d) / np.sqrt(d),     # output projection: mixes the heads
                      Ua=rng.randn(d, n_keys) / np.sqrt(d), ba=np.zeros(n_keys),  # first answer: which value?
                      Ub=rng.randn(d, n_keys) / np.sqrt(d), bb=np.zeros(n_keys))  # second answer

    def forward(self, keys, vals, qa, qb, force=None):     # 17 tokens through one multi-head layer; force = (pa, pb) for Section 5
        P = self.P                                           # parameters
        pairs = P["Ek"][keys] + P["Ev"][vals]                # each pair as one token (Day 75), (m, 16, d)
        query = P["Ea"][qa] + P["Eb"][qb]                    # the query token, (m, d)
        X = np.concatenate([pairs, query[:, None, :]], axis=1)  # 17 tokens; the query is the last one
        out, c = multi_head_attention(X, P, self.h)          # every token attends to every token
        if force is not None:                                # Section 5: overwrite the query token's weights
            c = self.forced_read(c, force)                     # the right pairs, by hand
            out = c["Z"] @ P["Wo"]                             # the output from the forced read
        o = out[:, -1]                                       # only the query token's output is used
        pa, pb = softmax(o @ P["Ua"] + P["ba"]), softmax(o @ P["Ub"] + P["bb"])  # the two answers
        self.cache = dict(c=c, keys=keys, vals=vals, qa=qa, qb=qb, o=o, pa=pa, pb=pb, forced=force is not None)
        return pa, pb                                     # probabilities of the 32 values, twice

    def forced_read(self, c, force):                      # Section 5: the query token's weights set by hand
        A = c["A"].copy()                                    # (m * h, 17, 17) weights, heads as batch entries
        m = len(force[0]); r = np.arange(m)                  # examples
        A = A.reshape(m, self.h, A.shape[1], A.shape[2])     # (m, h, 17, 17)
        A[:, :, -1, :] = 0.0                                 # clear the query row of every head
        if self.h == 1:                                      # one head: half on each right pair
            A[r, 0, -1, force[0]] += 0.5; A[r, 0, -1, force[1]] += 0.5  # (+= so the two halves add)
        else:                                                # two or more heads: head 1 on pair a, every other head on pair b
            A[r, 0, -1, force[0]] = 1.0                        # head 1: all its weight on pair a
            A[r[:, None], np.arange(1, self.h)[None, :], -1, force[1][:, None]] = 1.0  # every other head: all on pair b
        A = A.reshape(m * self.h, A.shape[2], A.shape[3])    # back to heads as batch entries
        Zh = np.einsum("mij,mjd->mid", A, c["V"])            # read the values with the forced weights
        return dict(c, A=A, Z=merge_heads(Zh, self.h))     # the cache, with the forced weights and read

    def loss_and_grads(self, keys, vals, qa, qb, ya, yb, force=None):  # mean cross-entropy of both answers, and every gradient
        P = self.P                                           # parameters
        pa, pb = self.forward(keys, vals, qa, qb, force)     # forward pass
        C = self.cache                                       # what it computed
        m = len(ya); r = np.arange(m)                        # batch size, example index
        loss = -(np.log(pa[r, ya] + 1e-12) + np.log(pb[r, yb] + 1e-12)).mean() / 2  # mean of the two cross-entropies
        da = pa.copy(); da[r, ya] -= 1; da /= 2 * m          # softmax + cross-entropy gradient, first answer
        db = pb.copy(); db[r, yb] -= 1; db /= 2 * m          # ... and second answer
        g = dict(Ua=C["o"].T @ da, ba=da.sum(axis=0), Ub=C["o"].T @ db, bb=db.sum(axis=0))  # the two output layers
        dout = np.zeros(C["c"]["X"].shape)                   # gradient at every token's output ...
        dout[:, -1] = da @ P["Ua"].T + db @ P["Ub"].T        # ... only the query token's is non-zero
        dX, ga = multi_head_attention_backward(dout, P, self.h, C["c"], scores=not C["forced"])  # back through the layer
        g.update(ga)                                         # Wq, Wk, Wv, Wo
        for k in ("Ek", "Ev", "Ea", "Eb"):                   # embedding tables
            g[k] = np.zeros_like(P[k])                         # start at zero
        np.add.at(g["Ek"], C["keys"], dX[:, :-1]); np.add.at(g["Ev"], C["vals"], dX[:, :-1])  # pair tokens
        np.add.at(g["Ea"], C["qa"], dX[:, -1]); np.add.at(g["Eb"], C["qb"], dX[:, -1])  # the query token
        return loss, g                                    # the loss and every gradient

    def train_step(self, keys, vals, qa, qb, ya, yb, force=None):  # one SGD step with Day 60's clipping (Day 75)
        loss, g = self.loss_and_grads(keys, vals, qa, qb, ya, yb, force)  # forward and backward
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))  # global norm
        for k in self.P:                                     # plain SGD
            self.P[k] -= self.lr * min(1.0, self.clip / norm) * g[k]  # shrink only if too long
        return loss                                       # for the record


TESTS = {two: make_batch(np.random.RandomState(999), 2000, two) for two in (True, False)}  # 2,000 fixed test examples per task


def evaluate(net, two=True, force=False):               # share right: first answer, second answer, both
    keys, vals, qa, qb, ya, yb, pa_, pb_ = TESTS[two]     # the fixed test examples
    pa, pb = net.forward(keys, vals, qa, qb, (pa_, pb_) if force else None)  # predictions
    a, b = pa.argmax(axis=1) == ya, pb.argmax(axis=1) == yb  # right or wrong, per answer
    return np.array([a.mean(), b.mean(), (a & b).mean()])  # first, second, both


def train(heads, two, seed, d=D, steps=STEPS, force=False):  # one training run; "both right" at every check
    net = TwoKeyRecall(d, heads, seed=seed)                 # fresh model
    rng = np.random.RandomState(seed + 100)                 # the training data stream (Day 75's seed rule)
    curve = []                                              # test accuracy at each check
    checks = [c * steps // STEPS for c in CHECKS]           # the same checkpoints, stretched for a longer run
    for step in range(1, steps + 1):                      # plain SGD on fresh batches
        keys, vals, qa, qb, ya, yb, pa, pb = make_batch(rng, 64, two)  # 64 fresh examples
        net.train_step(keys, vals, qa, qb, ya, yb, (pa, pb) if force else None)  # one update
        if step in checks:                                # a check step
            curve.append(evaluate(net, two, force))       # first, second, both
    return net, np.array(curve)                           # model and its curve (checks, 3)


RUNS = {}                                               # cache: each configuration trains once


def train_cached(*args, **kwargs):                      # train(), remembered by its arguments (Day 75)
    key = (args, tuple(sorted(kwargs.items())))           # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = train(*args, **kwargs)                # train and remember
    return RUNS[key]                                      # later requests reuse it


# =============================================================================
# DEMOS
# =============================================================================
def demo_parts():
    print("=" * 70)
    print("1. THE PARTS FIT -- one call on all heads, or Day 75's function head by head")
    print("=" * 70)
    print("The heads are folded into the batch axis, so Day 75's function runs all of them in one call. "
          "Check it against the plain way: cut the projected Q, K and V into column slices and call "
          "Day 75's function on each slice separately, then put the outputs side by side.")
    rng = np.random.RandomState(1)                         # fixed rng
    X = rng.randn(3, 5, 8)                                 # 3 sequences, 5 positions, size 8
    P = {w: rng.randn(8, 8) for w in ("Wq", "Wk", "Wv", "Wo")}  # random projections
    print(f"\n{'heads':>6}{'head size':>11}{'largest |one call - head by head|':>36}")
    for h in (1, 2, 4):                                    # three head counts
        out, _ = multi_head_attention(X, P, h)               # one call
        dk = 8 // h                                          # head size
        Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]      # the projections, uncut
        parts = [scaled_dot_product_attention(Q[..., i * dk:(i + 1) * dk], K[..., i * dk:(i + 1) * dk],
                                              V[..., i * dk:(i + 1) * dk])[0] for i in range(h)]  # Day 75's function per head
        loop = np.concatenate(parts, axis=-1) @ P["Wo"]      # side by side, then the output projection
        print(f"{h:>6}{dk:>11}{np.abs(out - loop).max():>36.2e}")
    print("Actual output from this lesson's run.")
    print(f"\nweights of the recall model (d = {D}) for each head count:")
    for h in HEADS:                                        # every head count used below
        n = sum(v.size for v in TwoKeyRecall(D, h).P.values())  # total weights
        print(f"  {h} head{'s' if h > 1 else ' '} of size {D // h:>2}: {n} weights")
    print("Actual output from this lesson's run.")


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("2. GRADIENT CHECK -- the whole recall model, one and two heads")
    print("=" * 70)
    print("Size 4, 6 possible keys, 4 pairs per example, 3 examples; every entry of every parameter "
          "nudged by +-1e-5; Day 62's pass rule |numeric - analytic| <= 1e-10 + 1e-6 |numeric|. The "
          "last row has the attention FORCED (Section 5), so the query and key weights get no gradient.")
    batch = make_batch(np.random.RandomState(3), 3, n_keys=6, n_pairs=4)  # a tiny batch
    for heads, force in [(1, False), (2, False), (2, True)]:  # three checks
        net = TwoKeyRecall(4, heads, seed=1, n_keys=6)       # tiny model
        f = batch[6:] if force else None                     # forced positions, if any
        _, g = net.loss_and_grads(*batch[:6], force=f)       # analytic gradients
        worst, ok = 0.0, True                                # largest difference; every entry passes?
        for k, A in net.P.items():                           # every parameter
            for pos in np.ndindex(A.shape):                    # every entry
                old = A[pos]                                     # remember it
                A[pos] = old + 1e-5; up = net.loss_and_grads(*batch[:6], force=f)[0]  # nudge up
                A[pos] = old - 1e-5; dn = net.loss_and_grads(*batch[:6], force=f)[0]  # nudge down
                A[pos] = old                                     # restore
                num = (up - dn) / 2e-5                           # central difference
                worst = max(worst, abs(num - g[k][pos]))         # largest difference so far
                ok &= abs(num - g[k][pos]) <= 1e-10 + 1e-6 * abs(num)  # Day 62's rule
        label = f"{heads} head{'s' if heads > 1 else ''}" + (", forced" if force else "")  # row label
        print(f"{label:<17}: largest |numeric - analytic| = {worst:.2e}; every entry passes: {ok}")
    print("Actual output from this lesson's run.")


def demo_training():
    print("\n" + "=" * 70)
    print("3. TWO-KEY RECALL -- 1, 2, 4 and 8 heads, the same d = 64, three seeds")
    print("=" * 70)
    print("Each example: 16 (key, value) pairs, two asked keys (always different pairs); the model "
          "must give both values. Guessing gets one right 1/32 = 0.031 of the time. Plain SGD lr 0.1, "
          "clip 1.0, batches of 64, 2000 steps (Day 75's settings); 2,000 fixed test examples.")
    print(f"\nshare with BOTH answers right, mean of three seeds:")
    print(f"{'heads':>6}{'size':>6}" + "".join(f"{'@' + str(c):>8}" for c in CHECKS) + f"{'first':>8}{'second':>8}")
    curves = {}                                            # head count -> mean curve (checks, 3)
    for h in HEADS:                                        # every head count
        curves[h] = np.mean([train_cached(h, True, s)[1] for s in SEEDS], axis=0)  # mean of the seeds
        print(f"{h:>6}{D // h:>6}" + "".join(f"{v:>8.3f}" for v in curves[h][:, 2]) + f"{curves[h][-1, 0]:>8.3f}{curves[h][-1, 1]:>8.3f}")
    print("('first' and 'second': each answer on its own, at the end)")
    print("Actual output from this lesson's run.")
    print("\nper seed, both right at the end:")
    for h in HEADS:                                        # every head count
        print(f"  {h} head{'s' if h > 1 else ' '}: " + "  ".join(f"{train_cached(h, True, s)[1][-1, 2]:.3f}" for s in SEEDS))
    print("Actual output from this lesson's run.")
    return curves                                          # the plot reuses them


def demo_control():
    print("\n" + "=" * 70)
    print("4. THE CONTROL -- one-key recall, where one head should be enough")
    print("=" * 70)
    print("The same models and settings, but both asked keys are the same key: one pair to fetch. "
          "Share right (both answers are then the same value), mean of three seeds.")
    print(f"\n{'heads':>6}{'size':>6}" + "".join(f"{'@' + str(c):>8}" for c in CHECKS))
    curves = {}                                            # head count -> mean curve
    for h in HEADS:                                        # every head count
        curves[h] = np.mean([train_cached(h, False, s)[1] for s in SEEDS], axis=0)  # mean of the seeds
        print(f"{h:>6}{D // h:>6}" + "".join(f"{v:>8.3f}" for v in curves[h][:, 2]))
    print("Actual output from this lesson's run.")
    return curves                                          # the plot reuses them


def demo_why_one_fails():
    print("\n" + "=" * 70)
    print("5. WHY ONE HEAD FAILS -- size, time, or the average?")
    print("=" * 70)
    print("Three follow-ups on two-key recall, three seeds each: (a) one head twice as large (d = 128, "
          "more weights than any model above); (b) one head trained four times as long (8,000 steps); "
          "(c) attention FORCED onto the two right pairs: one head puts half its weight on each, two "
          "heads put all of one head on each. The queries and keys then get no gradient; everything "
          "else is trained. The question is whether the model can answer from what it reads.")
    rows = [("one head, d = 128", dict(heads=1, d=128)), ("one head, 8000 steps", dict(heads=1, steps=8000)),
            ("one head, forced 1/2 + 1/2", dict(heads=1, force=True)), ("two heads, forced", dict(heads=2, force=True))]
    print(f"\n{'model':<28}{'first':>8}{'second':>8}{'both':>8}   both, per seed")
    res = {}                                               # label -> mean final (first, second, both)
    for label, kw in rows:                                 # each follow-up
        h = kw.pop("heads")                                  # head count
        finals = [train_cached(h, True, s, **kw)[1][-1] for s in SEEDS]  # final accuracies, per seed
        res[label] = np.mean(finals, axis=0)                 # mean of the seeds
        print(f"{label:<28}" + "".join(f"{v:>8.3f}" for v in res[label]) + "   " + " ".join(f"{f[2]:.3f}" for f in finals))
    print("Actual output from this lesson's run.")
    return res                                             # the guide quotes these


def demo_heads():
    print("\n" + "=" * 70)
    print("6. WHAT EACH HEAD LOOKS AT -- Day 72's measurements, query token, seed 0")
    print("=" * 70)
    print("For the trained two-key models of Section 3 (seed 0): each head's weights from the query "
          "token over the 17 tokens, on the 2,000 test examples, measured against pair a (the first "
          "asked key) and pair b (the second). Weight on target: the mean weight on that pair.")
    keys, vals, qa, qb, ya, yb, pa, pb = TESTS[True]      # the fixed test examples
    print(f"\n{'model':<10}{'head':>5}{'argmax on a':>13}{'weight on a':>13}{'argmax on b':>13}{'weight on b':>13}{'entropy':>9}")
    heads_w = {}                                           # head count -> weights on a and b per head
    for h in (1, 2, 4):                                    # three trained models
        net = train_cached(h, True, 0)[0]                    # seed 0
        net.forward(keys, vals, qa, qb)                      # fills the cache
        A = net.cache["c"]["A"].reshape(len(qa), h, N_PAIRS + 1, N_PAIRS + 1)[:, :, -1, :]  # (m, h, 17): the query row of every head
        heads_w[h] = []                                      # per head: (weight on a, weight on b)
        for i in range(h):                                   # each head
            ma, mb = alignment_metrics(A[:, i], pa), alignment_metrics(A[:, i], pb)  # Day 72's three numbers, both targets
            heads_w[h].append((ma[1], mb[1]))                  # the two weights, for the plot
            print(f"{str(h) + ' head' + ('s' if h > 1 else ''):<10}{i + 1:>5}{ma[0]:>13.3f}{ma[1]:>13.3f}{mb[0]:>13.3f}{mb[1]:>13.3f}{ma[2]:>9.3f}")
    print("(entropy over 17 tokens: log 17 = 2.83 means spread evenly)")
    print("Actual output from this lesson's run.")
    return heads_w                                         # the plot reuses it


def demo_cost():
    print("\n" + "=" * 70)
    print("7. WHAT MORE HEADS COST -- one training step, measured in this run")
    print("=" * 70)
    print("d = 64, a batch of 64 examples, forward and backward; best of 5 timings of 20 steps each, "
          "matrix library on one thread. The weight count is the same for every head count (Section 1); "
          "only ratios within this run are meaningful.")
    batch = make_batch(np.random.RandomState(5), 64)       # one fixed batch
    t = {}                                                 # head count -> seconds per step
    for h in HEADS:                                        # every head count
        net = TwoKeyRecall(D, h)                             # fresh model
        times = []                                           # one timing per repeat
        for _ in range(5):                                   # 5 repeats
            start = time.perf_counter()                        # clock
            for _ in range(20):                                # 20 steps
                net.loss_and_grads(*batch[:6])                   # forward and backward
            times.append((time.perf_counter() - start) / 20)   # per step
        t[h] = min(times)                                    # the fastest repeat
    print(f"\n{'heads':>6}{'ms per step':>13}{'vs one head':>13}")
    for h in HEADS:                                        # every head count
        print(f"{h:>6}{t[h] * 1000:>13.2f}{t[h] / t[1]:>13.2f}")
    print("Actual output from this lesson's run.")


def plot_results(two, one, heads_w):                    # three panels: two-key | one-key | what the heads look at
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))          # canvas
    for h in HEADS:                                         # panels 1-2: learning curves
        ax[0].plot(CHECKS, two[h][:, 2], marker="o", color=COLORS[h], label=f"{h} head{'s' if h > 1 else ''} of {D // h}")
        ax[1].plot(CHECKS, one[h][:, 2], marker="o", color=COLORS[h], label=f"{h} head{'s' if h > 1 else ''} of {D // h}")
    for a, title in [(ax[0], "Two-key recall: both answers right"), (ax[1], "Control: one-key recall")]:
        a.set_xscale("log"); a.set_xlabel("training steps"); a.set_ylabel("test accuracy (mean of 3 seeds)")
        a.set_ylim(-0.02, 1.02); a.set_title(title, fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8, loc="center right")
    labels, wa, wb = [], [], []                             # panel 3: weight on pair a and pair b, per head
    for h in (1, 2, 4):                                     # three trained models
        for i, (x, y) in enumerate(heads_w[h]):               # each head
            labels.append(f"{h}h:{i + 1}"); wa.append(x); wb.append(y)  # one bar pair per head
    x = np.arange(len(labels))                              # bar positions
    ax[2].bar(x - 0.2, wa, width=0.4, color="#1f77b4", label="weight on pair a")
    ax[2].bar(x + 0.2, wb, width=0.4, color="#ff7f0e", label="weight on pair b")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=8); ax[2].set_ylim(0, 1.15)
    ax[2].set_xlabel("model : head"); ax[2].set_ylabel("mean weight from the query token")
    ax[2].set_title("What each head looks at (seed 0)", fontsize=10); ax[2].legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig("multi_head_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved multi_head_results.png")


def draw_diagram():                                     # multi-head attention, with the week's parts named
    fig, ax = plt.subplots(figsize=(13, 4.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5, bold=False):          # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333"):                    # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    ax.text(6.5, 4.35, "Day 76: multi-head attention, built from Week 11's exercise parts", ha="center",
            fontsize=12, fontweight="bold")
    box(0.8, 2.2, 1.0, 0.6, "X", "#F2F2F2", bold=True)                           # the input tokens
    box(2.7, 2.2, 2.0, 1.2, "Q = X Wq\nK = X Wk\nV = X Wv", "#DCEBF7")            # Day 74's projections
    ax.text(2.7, 1.35, "Day 74 exercise", ha="center", fontsize=7.5, style="italic")
    arrow(1.32, 2.2, 1.68, 2.2)                                                    # X -> projections
    box(4.9, 2.2, 1.5, 0.8, "split into\nh heads", "#FAEEDA")                     # NEW
    arrow(3.72, 2.2, 4.13, 2.2)                                                    # projections -> split
    ys = [3.4, 2.6, 1.8, 1.0]                                                      # four heads drawn
    for i, y in enumerate(ys):                                                     # each head
        box(7.4, y, 3.0, 0.55, f"head {i + 1}: softmax(Q K^T / sqrt(d/h)) V", "#F4EEFC", fs=7.5)
        arrow(5.67, 2.2, 5.86, y)                                                    # split -> head
        arrow(8.94, y, 9.36, 2.2)                                                    # head -> concat
    ax.text(7.4, 0.45, "Day 75 exercise, once per head (Days 70, 73 inside)", ha="center", fontsize=7.5, style="italic")
    box(10.0, 2.2, 1.2, 0.8, "concat", "#FAEEDA")                                  # NEW
    box(11.85, 2.2, 1.5, 0.8, "output\n= concat Wo", "#E5F2E1")                    # NEW: mixes the heads
    arrow(10.62, 2.2, 11.08, 2.2)                                                  # concat -> Wo
    ax.text(11.85, 1.35, "same weights for any h", ha="center", fontsize=7.5, style="italic")
    plt.savefig("multi_head_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved multi_head_diagram.png")


def note_on_the_week():
    print("\n" + "=" * 70)
    print("8. THE WEEK, AND WHAT COMES NEXT")
    print("=" * 70)
    print("Week 11 went from a decoder that sees one summary of its input (Day 70) to attention over "
          "every encoder state (Days 71-73), attention within one sequence (Day 74), its scaling "
          "(Day 75) and several attentions side by side (today). Day 74 showed what is still missing: "
          "self-attention cannot tell one order of the words from another. Week 12 builds the "
          "Transformer, starting on Day 77 with positional encoding.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_parts()                                         # 1. the parts fit
        demo_gradient_check()                                # 2. gradients
        two = demo_training()                                # 3. two-key recall
        one = demo_control()                                 # 4. one-key recall
        demo_why_one_fails()                                 # 5. size, time, or the average
        heads_w = demo_heads()                               # 6. what each head looks at
        demo_cost()                                          # 7. cost
        plot_results(two, one, heads_w)                      # the figure
        draw_diagram()                                       # the diagram
        note_on_the_week()                                   # 8. Day 77


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
