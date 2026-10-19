"""
Day 75: Scaled Dot-Product Attention -- What the 1/sqrt(d) Actually Does

Days 73-74 scored with plain dot products, q . k. Vaswani et al. (2017)
divide them by sqrt(d), the length of the query and key vectors:

    A = softmax( Q K^T / sqrt(d) ),   Z = A V

Their reason: if the numbers in q and k have spread 1, q . k is a sum of
d such products and has spread sqrt(d). Large scores push the softmax
towards one-hot, where its gradient is nearly zero. Today measures each
step of that argument, then trains a model that needs attention to see
whether it matters in practice.

What is measured, reported as it came out:

  1. The spread of q . k as d grows, with and without the scaling.
  2. What that spread does to the softmax: largest weight and entropy.
  3. What it does to the gradient flowing back through the softmax.
  4. Training: an associative-recall task that can only be solved by
     attending to the right position, at four sizes, scaled and unscaled.
  5. Why sqrt(d)? The same model with no scaling, 1/sqrt(d) and 1/d.
  6. Scaling or initialisation? Unscaled scores with smaller initial
     query and key weights start exactly where the scaled model starts.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import time                                            # time: perf_counter, to time each training run

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

N_KEYS = 32                                             # the recall task: 32 possible keys and 32 possible values
N_PAIRS = 16                                            # each example shows 16 (key, value) pairs
SEEDS = [0, 1, 2]                                       # three seeds for every trained comparison
STEPS = 2000                                            # training steps per run
CHECKS = (100, 250, 500, 1000, 2000)                    # steps at which test accuracy is measured
DIMS = (16, 64, 128)                                    # attention sizes trained with three seeds


# =============================================================================
# 0. REUSED -- softmax, Days 57-74
# =============================================================================
def softmax(z):                                         # weights along the last axis, summing to 1
    e = np.exp(z - z.max(axis=-1, keepdims=True))         # subtract the row max for stability, exponentiate
    return e / e.sum(axis=-1, keepdims=True)              # normalise


def softmax_backward(dA, A):                            # Day 70 exercise: gradient of the scores from the gradient of the weights
    return A * (dA - (A * dA).sum(axis=-1, keepdims=True))  # zero wherever a weight is exactly 0 or 1


# =============================================================================
# 1. THE ASSOCIATIVE-RECALL TASK -- NEW
# =============================================================================
def make_batch(rng, m):                                 # NEW: 16 (key, value) pairs and one query key per example
    keys = np.array([rng.permutation(N_KEYS)[:N_PAIRS] for _ in range(m)])  # 16 different keys per example
    vals = rng.randint(0, N_KEYS, size=(m, N_PAIRS))      # a random value for each key
    pick = rng.randint(0, N_PAIRS, size=m)                # which pair is asked about
    return keys, vals, keys[np.arange(m), pick], vals[np.arange(m), pick], pick  # pairs, query, answer, its position


class Recall:                                           # NEW: one scaled dot-product attention read from the query to the pairs
    def __init__(self, d, scale="sqrt", seed=0, lr=0.1, clip=1.0, qk_init=1.0):
        rng = np.random.RandomState(seed)                  # private rng -> reproducible init
        self.d, self.lr, self.clip = d, lr, clip           # size and training settings
        self.c = {"none": 1.0, "sqrt": 1 / np.sqrt(d), "d": 1 / d}[scale]  # the factor that multiplies q . k
        self.P = dict(Ek=rng.randn(N_KEYS, d),             # key embeddings: numbers of spread 1, as after a Transformer's normalisation
                      Ev=rng.randn(N_KEYS, d),             # value embeddings
                      Eq=rng.randn(d),                     # marks the query position
                      Wq=rng.randn(d, d) / np.sqrt(d) * qk_init,  # query projection; qk_init < 1 shrinks it (Section 6)
                      Wk=rng.randn(d, d) / np.sqrt(d) * qk_init,  # key projection
                      Wv=rng.randn(d, d) / np.sqrt(d),     # value projection
                      Wo=rng.randn(d, N_KEYS) / np.sqrt(d), bo=np.zeros(N_KEYS))  # output layer: which value?

    def forward(self, keys, vals, qkey):                   # attention from the query to the 16 pairs, then a guess
        P = self.P                                           # parameters
        X = P["Ek"][keys] + P["Ev"][vals]                    # each pair as one vector, (m, 16, d)
        xq = P["Ek"][qkey] + P["Eq"]                         # the query vector, (m, d)
        q, K, Vv = xq @ P["Wq"], X @ P["Wk"], X @ P["Wv"]    # query, keys and values
        S = self.c * np.einsum("md,mnd->mn", q, K)           # scaled scores, one per pair
        A = softmax(S)                                       # attention weights
        ctx = np.einsum("mn,mnd->md", A, Vv)                 # what the query reads
        p = softmax(ctx @ P["Wo"] + P["bo"])                 # probabilities of the 32 values
        self.cache = dict(keys=keys, vals=vals, qkey=qkey, X=X, xq=xq, q=q, K=K, V=Vv, S=S, A=A, ctx=ctx, p=p)
        return p                                          # hand back

    def loss_and_grads(self, keys, vals, qkey, y):         # mean cross-entropy and every gradient
        P = self.P                                           # parameters
        p = self.forward(keys, vals, qkey)                   # forward pass
        C = self.cache                                       # what it computed
        m = len(y)                                           # batch size
        loss = -np.log(p[np.arange(m), y] + 1e-12).mean()    # mean cross-entropy
        dl = p.copy(); dl[np.arange(m), y] -= 1; dl /= m     # softmax + cross-entropy gradient
        g = dict(Wo=C["ctx"].T @ dl, bo=dl.sum(axis=0))      # output layer
        dctx = dl @ P["Wo"].T                                # gradient at the read
        dA = np.einsum("md,mnd->mn", dctx, C["V"])           # ctx = sum_n A_n v_n  ->  dA
        dV = C["A"][:, :, None] * dctx[:, None, :]           # ... and dV
        dS = softmax_backward(dA, C["A"]) * self.c           # back through the softmax AND the scaling
        dq = np.einsum("mn,mnd->md", dS, C["K"])             # S = c q . k  ->  dq
        dK = dS[:, :, None] * C["q"][:, None, :]             # ... and dK
        flat = lambda T: T.reshape(-1, T.shape[-1])          # (m, n, d) -> (m*n, d)
        g["Wq"] = C["xq"].T @ dq                             # query projection
        g["Wk"] = flat(C["X"]).T @ flat(dK)                  # key projection
        g["Wv"] = flat(C["X"]).T @ flat(dV)                  # value projection
        dX = dK @ P["Wk"].T + dV @ P["Wv"].T                 # the pair vectors made keys and values
        dxq = dq @ P["Wq"].T                                 # the query vector made the query
        g["Ek"], g["Ev"] = np.zeros_like(P["Ek"]), np.zeros_like(P["Ev"])  # embedding gradients
        np.add.at(g["Ek"], C["keys"], dX); np.add.at(g["Ev"], C["vals"], dX)  # each pair's key and value embeddings
        np.add.at(g["Ek"], C["qkey"], dxq); g["Eq"] = dxq.sum(axis=0)  # the query's key embedding and the query marker
        return loss, g                                    # the loss and every gradient

    def train_step(self, keys, vals, qkey, y):             # one SGD step with Day 60's clipping
        loss, g = self.loss_and_grads(keys, vals, qkey, y)   # forward and backward
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))  # global norm
        for k in self.P:                                     # plain SGD
            self.P[k] -= self.lr * min(1.0, self.clip / norm) * g[k]  # shrink only if too long
        return loss                                       # for the record


TEST = make_batch(np.random.RandomState(999), 2000)     # 2,000 fixed test examples, the same for every model


def accuracy(net):                                      # share of test examples answered correctly
    keys, vals, qkey, y, _ = TEST                         # the fixed test examples
    return (net.forward(keys, vals, qkey).argmax(axis=1) == y).mean()  # share answered correctly


def train(d, scale, seed, qk_init=1.0, steps=STEPS):    # one training run; accuracy at every check
    net = Recall(d, scale, seed=seed, qk_init=qk_init)      # fresh model
    rng = np.random.RandomState(seed + 100)                 # the training data stream
    curve = []                                              # test accuracy at each check
    start = time.perf_counter()                             # time the run
    for step in range(1, steps + 1):                      # plain SGD on fresh batches
        keys, vals, qkey, y, _ = make_batch(rng, 64)          # 64 fresh examples
        net.train_step(keys, vals, qkey, y)                   # one update
        if step in CHECKS:                                # a check step
            curve.append(accuracy(net))                   # test accuracy now
    return net, np.array(curve), time.perf_counter() - start  # model, curve, seconds


RUNS = {}                                               # cache: each configuration trains once


def train_cached(*args, **kwargs):                      # train(), remembered by its arguments
    key = (args, tuple(sorted(kwargs.items())))           # everything that defines the run
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = train(*args, **kwargs)                # train and remember
    return RUNS[key]                                      # later requests reuse it


# =============================================================================
# DEMOS
# =============================================================================
def demo_spread():
    print("=" * 70)
    print("1. HOW LARGE IS q . k? -- 10,000 random pairs of vectors at each size")
    print("=" * 70)
    print("Every number in q and k drawn with mean 0 and spread (standard deviation) 1, as a "
          "Transformer's normalisation layers aim for.")
    rng = np.random.RandomState(1)                         # fixed rng
    print(f"\n{'d':>6}{'spread of q . k':>17}{'sqrt(d)':>9}{'spread of q . k / sqrt(d)':>27}")
    for d in [4, 16, 64, 256, 1024]:                      # growing sizes
        q, k = rng.randn(10000, d), rng.randn(10000, d)      # 10,000 random pairs
        s = (q * k).sum(axis=1)                              # their dot products
        print(f"{d:>6}{s.std():>17.3f}{np.sqrt(d):>9.3f}{(s / np.sqrt(d)).std():>27.3f}")
    print("Actual output from this lesson's run.")


def demo_softmax():
    print("\n" + "=" * 70)
    print("2. WHAT LARGE SCORES DO TO THE SOFTMAX -- 16 keys per query")
    print("=" * 70)
    print("For 5,000 random queries, each against 16 random keys (numbers of spread 1): the largest "
          f"weight and the entropy of the weights (log 16 = {np.log(16):.3f} means spread evenly, 0 means one-hot).")
    rng = np.random.RandomState(2)                        # fixed rng
    print(f"\n{'d':>6}{'unscaled: max weight':>22}{'entropy':>9}{'scaled: max weight':>21}{'entropy':>9}")
    rows = {}                                              # d -> four numbers, for the plot
    for d in [4, 16, 64, 256, 1024]:                      # growing sizes
        q, K = rng.randn(5000, 1, d), rng.randn(5000, 16, d)  # queries and keys
        S = (q * K).sum(axis=2)                              # unscaled scores, (5000, 16)
        r = []                                            # the numbers for this size
        for A in [softmax(S), softmax(S / np.sqrt(d))]:      # unscaled, scaled
            r += [A.max(axis=1).mean(), (-(A * np.log(A + 1e-30)).sum(axis=1)).mean()]  # largest weight and entropy, mean over queries
        rows[d] = r                                       # keep, for the plot
        print(f"{d:>6}{r[0]:>22.3f}{r[1]:>9.3f}{r[2]:>21.3f}{r[3]:>9.3f}")
    print("Actual output from this lesson's run.")
    return rows                                           # the plot reuses them


def demo_gradient():
    print("\n" + "=" * 70)
    print("3. WHAT IT DOES TO THE GRADIENT -- back through the softmax")
    print("=" * 70)
    print("Same queries and keys as Section 2, and a random gradient arriving at the weights (spread 1). "
          "The table gives the size of the gradient that reaches the scores, and the share of queries "
          "where it is below 1e-6 (the softmax is effectively frozen). For the scaled version the "
          "gradient at the UNSCALED scores is reported, so both columns measure the same thing: what "
          "reaches q and k.")
    rng = np.random.RandomState(2)                         # the same rng and draws as Section 2
    print(f"\n{'d':>6}{'unscaled: gradient size':>25}{'frozen':>9}{'scaled: gradient size':>23}{'frozen':>9}")
    rows = {}                                             # d -> numbers, for the plot
    for d in [4, 16, 64, 256, 1024]:                      # growing sizes
        q, K = rng.randn(5000, 1, d), rng.randn(5000, 16, d)  # 5,000 queries, 16 keys each
        S = (q * K).sum(axis=2)                           # unscaled scores
        dA = np.random.RandomState(3).randn(5000, 16)        # the same upstream gradient for both
        r = []                                            # the numbers for this size
        for c in [1.0, 1 / np.sqrt(d)]:                      # unscaled, scaled
            dS = softmax_backward(dA, softmax(c * S)) * c      # gradient at the raw scores q . k
            size = np.linalg.norm(dS, axis=1)                  # one size per query
            r += [np.median(size), (size < 1e-6).mean()]  # median size, share frozen
        rows[d] = r                                       # keep, for the plot
        print(f"{d:>6}{r[0]:>25.2e}{r[1]:>9.3f}{r[2]:>23.2e}{r[3]:>9.3f}")
    print("(median over the 5,000 queries)")
    print("Actual output from this lesson's run.")
    return rows                                           # the plot reuses them


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("4a. GRADIENT CHECK -- the recall model, scaled and unscaled")
    print("=" * 70)
    print("Size 4, 3 examples, every entry of every parameter nudged by +-1e-5; Day 62's pass rule.")
    keys, vals, qkey, y, _ = make_batch(np.random.RandomState(1), 3)  # tiny batch
    for scale in ["none", "sqrt"]:                        # unscaled and scaled
        net = Recall(4, scale, seed=2)                       # tiny model
        _, g = net.loss_and_grads(keys, vals, qkey, y)       # analytic gradients
        worst, ok_all = 0.0, True                         # largest difference; every entry passes?
        for name, P in net.P.items():                        # every parameter
            for pos in np.ndindex(P.shape):                    # every entry
                old = P[pos]                              # remember it
                P[pos] = old + 1e-5; up = net.loss_and_grads(keys, vals, qkey, y)[0]  # nudge up
                P[pos] = old - 1e-5; down = net.loss_and_grads(keys, vals, qkey, y)[0]  # nudge down
                P[pos] = old                              # restore
                num = (up - down) / 2e-5                  # measured slope
                diff = abs(num - g[name][pos])            # disagreement
                worst, ok_all = max(worst, diff), ok_all and diff <= 1e-10 + 1e-6 * abs(num)  # Day 62's pass rule
        print(f"scale {scale:<5}: largest |numeric - analytic| = {worst:.2e}; every entry passes: {ok_all}")
    print("Actual output from this lesson's run.")


def demo_training():
    print("\n" + "=" * 70)
    print("4. TRAINING -- associative recall, scaled and unscaled, three seeds")
    print("=" * 70)
    print(f"Each example: {N_PAIRS} (key, value) pairs with different keys out of {N_KEYS}, and a query key; "
          f"the answer is the value paired with it (guessing: 1/{N_KEYS} = {1 / N_KEYS:.3f}). The only way to "
          "answer is to attend to the right pair. Embeddings have spread 1; plain SGD lr 0.1, clip 1.0, "
          f"batches of 64, {STEPS} steps; test accuracy on 2,000 fixed examples, mean of three seeds.")
    print(f"\n{'d':>5}{'scaling':>9}{'attention entropy at start':>28}" + "".join(f"{'@' + str(c):>8}" for c in CHECKS))
    curves = {}                                            # (d, scale) -> mean curve
    for d in DIMS:                                        # each attention size
        for scale in ["none", "sqrt"]:                    # unscaled and scaled
            runs = [train_cached(d, scale, s) for s in SEEDS]  # three seeds
            curves[(d, scale)] = np.mean([r[1] for r in runs], axis=0)  # mean curve
            net0 = Recall(d, scale, seed=0)                  # the untrained seed-0 model
            net0.forward(*TEST[:3]); A = net0.cache["A"]     # its attention on the test set
            ent = (-(A * np.log(A + 1e-30)).sum(axis=1)).mean()  # entropy at the start (log 16 = 2.77)
            print(f"{d:>5}{scale:>9}{ent:>28.3f}" + "".join(f"{a:>8.3f}" for a in curves[(d, scale)]))
    print("Actual output from this lesson's run.")
    print(f"\nper seed, accuracy at 250 steps:")
    for d in DIMS:                                        # each attention size
        print(f"  d = {d:<4}" + "   ".join(f"{scale}: " + " ".join(f"{train_cached(d, scale, s)[1][1]:.3f}" for s in SEEDS)
                                          for scale in ["none", "sqrt"]))
    print("Actual output from this lesson's run.")
    return curves                                         # the plot reuses them


def demo_which_factor():
    print("\n" + "=" * 70)
    print("5. WHY sqrt(d)? -- no scaling, 1/sqrt(d) and 1/d, at d = 64")
    print("=" * 70)
    print("The same model and data; only the factor in front of q . k changes. Mean of three seeds.")
    print(f"\n{'factor':>12}{'entropy at start':>18}" + "".join(f"{'@' + str(c):>8}" for c in CHECKS))
    for scale, label in [("none", "1"), ("sqrt", "1/sqrt(64)"), ("d", "1/64")]:  # three factors
        curve = np.mean([train_cached(64, scale, s)[1] for s in SEEDS], axis=0)  # mean curve over seeds
        net0 = Recall(64, scale, seed=0); net0.forward(*TEST[:3]); A = net0.cache["A"]  # the untrained model's attention
        ent = (-(A * np.log(A + 1e-30)).sum(axis=1)).mean()  # its entropy at the start
        print(f"{label:>12}{ent:>18.3f}" + "".join(f"{a:>8.3f}" for a in curve))
    print("Actual output from this lesson's run.")


def demo_init_instead():
    print("\n" + "=" * 70)
    print("6. SCALING, OR JUST A SMALLER START? -- d = 64")
    print("=" * 70)
    print("Unscaled scores, but the query and key weights start 64^(1/4) = 2.83 times smaller. Then "
          "q . k at the start is exactly the scaled model's q . k / 8 -- the two begin identically. "
          "Do they train identically? Mean of three seeds.")
    shrink = 64 ** -0.25                                   # (1/2.83)^2 = 1/8 = 1/sqrt(64)
    a, b = Recall(64, "sqrt", seed=0), Recall(64, "none", seed=0, qk_init=shrink)  # the two models
    pa, pb = a.forward(*TEST[:3]), b.forward(*TEST[:3])   # their first predictions
    print(f"\nlargest difference between the two models' starting attention weights: "
          f"{np.abs(a.cache['A'] - b.cache['A']).max():.2e}")
    print(f"\n{'model':<34}" + "".join(f"{'@' + str(c):>8}" for c in CHECKS))
    for label, args, kw in [("scaled (1/sqrt(d))", (64, "sqrt"), {}), ("unscaled, smaller Wq and Wk", (64, "none"), dict(qk_init=shrink)),
                            ("unscaled (Section 4)", (64, "none"), {})]:
        curve = np.mean([train_cached(*args, s, **kw)[1] for s in SEEDS], axis=0)  # mean curve over seeds
        print(f"{label:<34}" + "".join(f"{x:>8.3f}" for x in curve))
    print("Actual output from this lesson's run.")


def demo_larger():
    print("\n" + "=" * 70)
    print("7. ONE LARGER SIZE -- d = 256, seed 0 only")
    print("=" * 70)
    print("The same comparison at d = 256, for one seed (a run at this size takes several minutes here).")
    print(f"\n{'scaling':>9}" + "".join(f"{'@' + str(c):>8}" for c in CHECKS))
    for scale in ["none", "sqrt"]:                        # unscaled and scaled
        print(f"{scale:>9}" + "".join(f"{a:>8.3f}" for a in train_cached(256, scale, 0)[1]))
    print("Actual output from this lesson's run.")


def plot_results(soft, grad, curves):                   # three panels: softmax, gradient, training
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))          # canvas
    ds = sorted(soft)                                       # the sizes of Sections 2-3
    ax[0].plot(ds, [soft[d][1] for d in ds], marker="o", color="#d62728", label="unscaled")
    ax[0].plot(ds, [soft[d][3] for d in ds], marker="o", color="#1f77b4", label="scaled by 1/sqrt(d)")
    ax[0].axhline(np.log(16), color="#999999", linestyle=":", label="spread evenly (log 16)")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("d"); ax[0].set_ylabel("entropy of the weights (16 keys)")
    ax[0].set_title("Large scores make the softmax one-hot", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(ds, [grad[d][0] for d in ds], marker="o", color="#d62728", label="unscaled")
    ax[1].plot(ds, [grad[d][2] for d in ds], marker="o", color="#1f77b4", label="scaled by 1/sqrt(d)")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xlabel("d")
    ax[1].set_ylabel("gradient reaching q . k (median)"); ax[1].set_title("... and the gradient vanishes", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    for d, color in zip(DIMS, ["#9ecae1", "#3182bd", "#08519c"]):  # panel 3: training curves
        ax[2].plot(CHECKS, curves[(d, "sqrt")], marker="o", color=color, label=f"d={d}, scaled")
        ax[2].plot(CHECKS, curves[(d, "none")], marker="s", linestyle="--", color=color, label=f"d={d}, unscaled")
    ax[2].set_xscale("log"); ax[2].set_xlabel("training steps"); ax[2].set_ylabel("test accuracy (mean of 3 seeds)")
    ax[2].set_ylim(-0.02, 1.02); ax[2].set_title("Associative recall", fontsize=10); ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(); plt.savefig("scaling_results.png", dpi=110); plt.close()
    print("\nsaved scaling_results.png")


def draw_diagram():                                     # scaled dot-product attention as a pipeline
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#333333"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, linewidth=1.3))

    ax.text(6.5, 3.35, "Day 75: scaled dot-product attention", ha="center", fontsize=12, fontweight="bold")
    box(0.9, 2.3, 1.2, 0.5, "Q", "#FDE7D3", bold=True); box(0.9, 1.3, 1.2, 0.5, "K", "#DCEBF7", bold=True)
    box(3.0, 1.8, 1.7, 0.6, "Q K^T", "#F2F2F2"); arrow(1.52, 2.25, 2.13, 1.9); arrow(1.52, 1.35, 2.13, 1.7)
    box(5.3, 1.8, 1.9, 0.6, "divide by sqrt(d)", "#FAEEDA", bold=True); arrow(3.87, 1.8, 4.33, 1.8)
    box(7.6, 1.8, 1.6, 0.6, "softmax", "#F4EEFC"); arrow(6.27, 1.8, 6.78, 1.8)
    box(9.9, 2.85, 1.2, 0.45, "V", "#E5F2E1", bold=True)   # the values
    box(9.9, 1.8, 1.4, 0.6, "A V", "#F2F2F2"); arrow(8.42, 1.8, 9.18, 1.8); arrow(9.9, 2.6, 9.9, 2.13)  # weights and values meet
    box(11.9, 1.8, 1.3, 0.6, "Z", "#FAEEDA", bold=True); arrow(10.62, 1.8, 11.23, 1.8)
    ax.text(5.3, 0.85, "keeps the scores' spread near 1 whatever d is,", ha="center", fontsize=8.5, style="italic")
    ax.text(5.3, 0.5, "so the softmax is not one-hot at the start", ha="center", fontsize=8.5, style="italic")
    plt.savefig("scaled_attention_diagram.png", dpi=120, bbox_inches="tight"); plt.close()
    print("saved scaled_attention_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 76, the week's build, runs several scaled dot-product attentions side by side on smaller "
          "projections and joins their outputs: multi-head attention, assembled from this week's exercise "
          "scripts.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_spread()                                        # 1. the spread of q . k
        soft = demo_softmax()                                # 2. what it does to the softmax
        grad = demo_gradient()                               # 3. what it does to the gradient
        demo_gradient_check()                                # 4a. the model is correct
        curves = demo_training()                             # 4. training
        demo_which_factor()                                  # 5. why sqrt(d)
        demo_init_instead()                                  # 6. scaling or initialisation
        demo_larger()                                        # 7. d = 256
        plot_results(soft, grad, curves)                     # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 8. Day 76


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
