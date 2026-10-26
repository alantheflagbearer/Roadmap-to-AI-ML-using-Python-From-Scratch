"""
Day 87: Embedding Visualization -- PCA and t-SNE From Scratch, and What
Survives the Projection

Day 86's word vectors have 50 numbers each; a picture has two. Two ways to
squeeze them, both written from scratch here:

    PCA:    the two directions along which the vectors spread the most (a rotation, then keep two axes)
    t-SNE:  place points in 2-D so that each point's NEAREST neighbours stay near (van der Maaten & Hinton, 2008)

A picture of word vectors is easy to over-read, so every plot here comes with
a number: how many words keep a same-group word as their nearest neighbour,
and how many of each word's 10 nearest neighbours in 50-D are still among its
10 nearest in 2-D. Two controls test the pictures themselves: t-SNE on the
untrained (random) table, and t-SNE on made-up clusters whose true sizes and
distances are known.

What is measured, reported as it came out:
  1. PCA from scratch: agreement with NumPy's SVD, and how much of the spread
     two components keep.
  2. t-SNE from scratch: a gradient check of its loss.
  3. 500 words in 2-D, PCA against t-SNE (five seeds): the category check
     and neighbourhood preservation.
  4. Controls: the random table; perplexity; clusters of different sizes.

Day 86's vectors are read from day86_vectors.txt.gz, written by Day 86's
script. Every line of code below carries its own comment, continuing Day
52's convention.
"""
import gzip                                            # gzip: Day 86's vectors are stored compressed
import os                                              # os: build the path to the vectors

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
VECTORS = os.path.join(HERE, "day86_vectors.txt.gz")    # Day 86's whole vocabulary, most frequent first, 50 numbers each
N_OTHER = 410                                           # frequent non-category words added, for 500 points in all
PERPLEXITY = 30                                         # t-SNE's effective number of neighbours (the usual default)
TSNE_SEEDS = [0, 1, 2, 3, 4]                            # five t-SNE runs from different random starts
K = 10                                                  # neighbourhood size for the preservation measure

CATEGORIES = {                                          # Day 86's word groups, verbatim
    "country": "france germany italy spain england china japan india russia egypt greece poland sweden canada brazil "
               "mexico portugal ireland austria norway".split(),
    "number": "two three four five six seven eight nine ten twenty".split(),
    "month": "january february march april june july august september october november december".split(),
    "colour": "red blue green yellow black white brown orange purple pink".split(),
    "family": "father mother son daughter brother sister wife husband uncle aunt".split(),
    "element": "hydrogen oxygen carbon nitrogen iron gold silver copper helium sodium".split(),
    "instrument": "piano guitar violin drums flute organ accordion keyboards synthesizer".split(),
    "science": "physics chemistry biology mathematics astronomy geology economics psychology philosophy linguistics".split(),
}


# =============================================================================
# 1. PCA
# =============================================================================
def load_vectors(path=VECTORS):                         # NEW: word2vec text format -> (words, matrix)
    with gzip.open(path, "rt", encoding="ascii") as f:    # decompress while reading
        n, d = map(int, f.readline().split())              # header: words, numbers per word
        rows = [line.split() for line in f]                # one word per line
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows])  # words, (n, d) matrix


def pca(X, k):                                          # NEW: the k directions of largest spread, and the data in them
    Xc = X - X.mean(axis=0)                               # centre every column
    C = Xc.T @ Xc / (len(X) - 1)                          # covariance matrix (d, d)
    vals, vecs = np.linalg.eigh(C)                        # eigen-decomposition of a symmetric matrix, smallest first
    order = np.argsort(vals)[::-1]                        # largest spread first
    vals, vecs = vals[order], vecs[:, order]              # reordered
    return Xc @ vecs[:, :k], vecs[:, :k], vals / vals.sum()  # projections, directions, share of spread per direction


# =============================================================================
# 2. t-SNE
# =============================================================================
def sq_dists(X):                                        # all pairwise squared distances
    s = (X ** 2).sum(axis=1)                              # squared lengths
    return np.maximum(s[:, None] + s[None, :] - 2 * X @ X.T, 0)  # |a|^2 + |b|^2 - 2 a.b, never below 0


def affinities(X, perplexity):                          # NEW: t-SNE's P: Gaussian neighbour probabilities, symmetrised
    D = sq_dists(X); n = len(X)                           # distances; points
    P = np.zeros((n, n)); target = np.log(perplexity)    # conditional probabilities; the entropy each row must have
    for i in range(n):                                    # one point at a time
        lo, hi, beta = 0.0, np.inf, 1.0                     # bisection on beta = 1 / (2 sigma^2)
        d = np.delete(D[i], i)                              # distances to the others
        for _ in range(60):                                 # bisection steps
            p = np.exp(-(d - d.min()) * beta); p /= p.sum()   # Gaussian weights around point i
            H = -(p * np.log(p + 1e-12)).sum()                # their entropy
            if abs(H - target) < 1e-6: break                  # perplexity reached
            if H > target: lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2  # too spread: narrower
            else: hi, beta = beta, (beta + lo) / 2            # too narrow: wider
        P[i, np.arange(n) != i] = p                         # row i
    return (P + P.T) / (2 * n)                            # symmetric, sums to 1


def kl_and_grad(P, Y):                                  # NEW: t-SNE's loss KL(P || Q) and its gradient for the 2-D points
    num = 1 / (1 + sq_dists(Y)); np.fill_diagonal(num, 0)  # Student-t similarities in 2-D
    Q = np.maximum(num / num.sum(), 1e-12)                # their probabilities
    kl = np.sum(P * np.log(np.maximum(P, 1e-12) / Q))     # how badly Q matches P
    W = (P - Q) * num                                     # pull (P > Q) or push (P < Q), weighted
    grad = 4 * (W.sum(axis=1)[:, None] * Y - W @ Y)       # = 4 sum_j (p_ij - q_ij) num_ij (y_i - y_j)
    return kl, grad                                       # loss and gradient


def tsne(X, perplexity=PERPLEXITY, seed=0, iters=1000, lr=100.0, exag=12.0, exag_iters=250):  # NEW: t-SNE, exact
    P = affinities(X, perplexity)                         # neighbour probabilities in the original space
    rng = np.random.RandomState(seed)                     # the random start
    Y = rng.randn(len(X), 2) * 1e-4                       # tiny random 2-D positions
    vel, gains = np.zeros_like(Y), np.ones_like(Y)        # momentum, per-coordinate step sizes (van der Maaten's code)
    for it in range(iters):                               # gradient descent
        _, g = kl_and_grad(P * (exag if it < exag_iters else 1.0), Y)  # early exaggeration: clusters form first
        gains = np.where(np.sign(g) != np.sign(vel), gains + 0.2, gains * 0.8).clip(0.01)  # grow steps that keep direction
        mom = 0.5 if it < 250 else 0.8                      # more momentum later
        vel = mom * vel - lr * gains * g                    # momentum step
        Y = Y + vel; Y -= Y.mean(axis=0)                    # move, keep centred
    return Y, kl_and_grad(P, Y)[0]                        # positions, final loss


# =============================================================================
# 3. MEASURES
# =============================================================================
def group_share(Y, groups):                             # share of category points whose nearest category point is in the same group
    D = sq_dists(Y); np.fill_diagonal(D, np.inf)          # distances; not itself
    return np.mean(groups[D.argmin(axis=1)] == groups)    # same group as the nearest


def preserved(X, Y, k=K):                               # share of each point's k nearest neighbours in X still among its k nearest in Y
    def knn(Z):                                           # k nearest, by Euclidean distance
        D = sq_dists(Z); np.fill_diagonal(D, np.inf)        # not itself
        return np.argsort(D, axis=1)[:, :k]                 # indices
    a, b = knn(X), knn(Y)                                 # neighbours before and after
    return np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))])  # overlap, averaged


def unit(E):                                            # every row scaled to length 1 (Day 86 compared directions, by cosine)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


# =============================================================================
# DEMOS
# =============================================================================
def demo_pca(words, E):
    print("=" * 70)
    print("1. PCA FROM SCRATCH -- the directions of largest spread")
    print("=" * 70)
    print("Centre the vectors, take the covariance matrix's eigenvectors, largest eigenvalue first. Checked against "
          "NumPy's SVD of the centred data: the same directions (up to sign) and the same projections.")
    Z, comps, share = pca(E, 50)                          # every direction
    U, S, Vt = np.linalg.svd(E - E.mean(axis=0), full_matrices=False)  # NumPy's answer
    print(f"\n{len(words)} words x {E.shape[1]} numbers (Day 86's table, rows scaled to length 1)")
    print(f"largest |difference| from SVD: directions {np.abs(np.abs(comps) - np.abs(Vt.T)).max():.1e}, "
          f"projections {np.abs(np.abs(Z) - np.abs(U * S)).max():.1e}")
    print(f"share of the spread kept: 2 directions {share[:2].sum():.3f}, 10 directions {share[:10].sum():.3f}, "
          f"25 directions {share[:25].sum():.3f}")
    print(f"the largest direction holds {share[0]:.3f}; a direction of 50 equally spread ones would hold {1 / 50:.3f}")
    print("Actual output from this lesson's run.")
    return share                                           # the plot shows it


def demo_gradient_check():
    print("\n" + "=" * 70)
    print("2. t-SNE FROM SCRATCH -- checking its gradient")
    print("=" * 70)
    print("t-SNE moves 2-D points downhill on KL(P || Q). A tiny case: 8 random points in 5-D, perplexity 3, random "
          "2-D positions; every coordinate nudged by +-1e-6.")
    rng = np.random.RandomState(0)                        # fixed rng
    X, Y = rng.randn(8, 5), rng.randn(8, 2)               # tiny data and positions
    P = affinities(X, 3)                                  # neighbour probabilities
    _, g = kl_and_grad(P, Y)                              # analytic
    worst, ok = 0.0, True                                 # largest difference; every entry passes
    for pos in np.ndindex(Y.shape):                       # every coordinate
        old = Y[pos]                                        # remember it
        Y[pos] = old + 1e-6; up = kl_and_grad(P, Y)[0]      # nudge up
        Y[pos] = old - 1e-6; dn = kl_and_grad(P, Y)[0]      # nudge down
        Y[pos] = old                                        # restore
        num = (up - dn) / 2e-6                              # central difference
        worst = max(worst, abs(num - g[pos])); ok &= abs(num - g[pos]) <= 1e-10 + 1e-6 * abs(num)
    print(f"\ngradient check: 16 coordinates, largest |numeric - analytic| = {worst:.2e}, all pass: {ok}")
    print(f"P sums to {P.sum():.6f}; symmetric: {np.allclose(P, P.T)}")
    print("Actual output from this lesson's run.")


def pick_points(words, E):                              # the 90 category words plus the 410 most frequent other words
    ix = {w: i for i, w in enumerate(words)}              # word -> row
    cat = [(w, g) for g, ws in CATEGORIES.items() for w in ws]  # (word, group)
    others = [i for i, w in enumerate(words) if w not in dict(cat)][:N_OTHER]  # frequent, not in a group
    idx = np.array([ix[w] for w, _ in cat] + others)      # rows: category words first
    return idx, np.array([g for _, g in cat]), [w for w, _ in cat]  # rows, the category words' groups, their words


def demo_projections(words, E, idx, groups):
    print("\n" + "=" * 70)
    print("3. 500 WORDS IN TWO DIMENSIONS -- PCA against t-SNE")
    print("=" * 70)
    print(f"The {len(groups)} category words of Day 86 and the {N_OTHER} most frequent other words. Group check: share of "
          f"category words whose nearest category word (in that space) is in the same group. Preserved: share of each "
          f"word's {K} nearest neighbours (of the 500) in 50-D that are still among its {K} nearest in 2-D. t-SNE: "
          f"perplexity {PERPLEXITY}, 1000 steps, five random starts.")
    X = E[idx]; nc = len(groups)                          # the 500 points; how many are category words
    Zp = pca(X, 2)[0]                                     # PCA to 2-D
    rows = [("50-D (no projection)", group_share(X[:nc], groups), 1.0),
            ("PCA, 2-D", group_share(Zp[:nc], groups), preserved(X, Zp))]
    ts = []                                               # the five t-SNE runs
    for s in TSNE_SEEDS:                                  # five random starts
        Y, kl = tsne(X, seed=s)                             # t-SNE
        ts.append((Y, kl)); rows.append((f"t-SNE, 2-D, seed {s}", group_share(Y[:nc], groups), preserved(X, Y)))
    print(f"\n{'space':<24}{'group check':>12}{'preserved':>11}")
    for name, gs, pr in rows:                             # one row per space
        print(f"{name:<24}{gs:>12.3f}{pr:>11.3f}")
    print("t-SNE's final KL per seed: " + ", ".join(f"{kl:.3f}" for _, kl in ts))
    print("Actual output from this lesson's run.")
    return Zp, ts, rows                                    # the plot reuses them


def demo_controls(E, idx, groups):  # E: the whole trained table
    print("\n" + "=" * 70)
    print("4. CONTROLS -- what the pictures can and cannot be trusted with")
    print("=" * 70)
    print("(a) The same 500 words with Day 86's untrained table (seed 0's random start, rows scaled to length 1).")
    rng = np.random.RandomState(0)                        # Day 86's SkipGram start: (rand - 0.5) / d, seed 0
    R = unit((rng.rand(len(E), 50) - 0.5) / 50)[idx]      # Day 86's starting table (same vocabulary size, same seed), the same rows
    Yr, _ = tsne(R, seed=0)                               # t-SNE of noise
    print(f"random table: group check 50-D {group_share(R[:len(groups)], groups):.3f}, t-SNE 2-D "
          f"{group_share(Yr[:len(groups)], groups):.3f}; preserved {preserved(R, Yr):.3f}")
    print("\n(b) Perplexity, on the trained table (seed 0):")
    X = E[idx]                                            # the trained 500
    for perp in (5, 30, 100):                             # three neighbourhood sizes
        Y, _ = tsne(X, perplexity=perp, seed=0)             # t-SNE
        print(f"  perplexity {perp:>3}: group check {group_share(Y[:len(groups)], groups):.3f}, preserved {preserved(X, Y):.3f}")
    print("\n(c) Made-up clusters in 50-D whose true sizes and distances are known: A and B, 100 points each, A's spread "
          "10 times B's; C, 100 points, 4 times as far from A as B is.")
    rng = np.random.RandomState(1)                        # fixed rng
    centres = np.zeros((3, 50)); centres[1, 0] = 20; centres[2, 0] = -80  # B at distance 20 from A, C at 80
    spread = [1.0, 0.1, 0.1]                              # A wide, B and C tight (A's points are about 10 apart, well short of 20)
    S = np.concatenate([centres[j] + spread[j] * rng.randn(100, 50) for j in range(3)])  # 300 points
    lab = np.repeat([0, 1, 2], 100)                       # which cluster
    out = {}                                              # name -> (spread A / spread B, dist(A, C) / dist(A, B))
    for name, Z in [("50-D (true)", S), ("PCA, 2-D", pca(S, 2)[0]), ("t-SNE, 2-D", tsne(S, seed=0)[0])]:
        c = np.array([Z[lab == j].mean(axis=0) for j in range(3)])  # cluster centres
        r = [np.sqrt(((Z[lab == j] - c[j]) ** 2).sum(axis=1)).mean() for j in range(3)]  # mean distance to own centre
        out[name] = (r[0] / r[1], np.linalg.norm(c[0] - c[2]) / np.linalg.norm(c[0] - c[1]))
        print(f"  {name:<12} size A / size B = {out[name][0]:6.2f}   distance A-C / distance A-B = {out[name][1]:6.2f}")
    print("Actual output from this lesson's run.")
    return Yr                                              # the plot shows it


def plot_results(share, Zp, ts, Yr, groups, cwords):    # four panels: spread per direction | PCA | t-SNE | t-SNE of noise
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))          # canvas
    ax[0].bar(np.arange(1, 21), share[:20], color="#1f77b4"); ax[0].axhline(1 / 50, color="#d62728", linestyle="--", label="equal spread (1/50)")
    ax[0].set_xlabel("PCA direction"); ax[0].set_ylabel("share of the spread"); ax[0].set_title("PCA: spread per direction", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
    colors = dict(zip(CATEGORIES, plt.cm.tab10.colors))   # one colour per group
    nc = len(groups)                                      # category points come first
    for a, Y, title in [(ax[1], Zp, "PCA, 2-D"), (ax[2], ts[0][0], "t-SNE, 2-D (seed 0)"), (ax[3], Yr, "t-SNE of the untrained table")]:
        a.scatter(Y[nc:, 0], Y[nc:, 1], s=6, color="#cccccc", label="other frequent words")  # the 410 others, grey
        for g in CATEGORIES:                                # each group in its colour
            m = groups == g
            a.scatter(Y[:nc][m, 0], Y[:nc][m, 1], s=22, color=colors[g], label=g)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
    for i, w in enumerate(cwords):                        # label a few words on the t-SNE panel
        if w in ("france", "three", "january", "red", "father", "gold", "piano", "physics"):
            ax[2].annotate(w, ts[0][0][i], fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[3].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout(); plt.savefig("visualization_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved visualization_results.png")


def draw_diagram():                                     # PCA and t-SNE side by side
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                     # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.1))

    ax.text(6.5, 4.15, "Day 87: two ways from 50 numbers per word to a picture", ha="center", fontsize=12, fontweight="bold")
    box(1.4, 2.1, 2.2, 1.0, "500 word vectors\n50 numbers each", "#FAEEDA")
    box(4.6, 3.1, 2.6, 0.9, "PCA: covariance ->\neigenvectors", "#DCEBF7")
    box(8.2, 3.1, 2.8, 0.9, "keep the 2 directions\nof largest spread", "#DCEBF7")
    box(11.4, 3.1, 2.6, 0.9, "a rotation: distances\nshrink, never invented", "#F2F2F2", fs=8)
    box(4.6, 1.1, 2.6, 0.9, "t-SNE: P = Gaussian\nneighbour probabilities", "#E5F2E1")
    box(8.2, 1.1, 2.8, 0.9, "move 2-D points until\nStudent-t Q matches P", "#E5F2E1")
    box(11.4, 1.1, 2.6, 0.9, "keeps near neighbours;\nsizes, gaps not kept", "#F2F2F2", fs=8)
    arrow(2.52, 2.35, 3.28, 3.0); arrow(2.52, 1.85, 3.28, 1.2)
    arrow(5.92, 3.1, 6.78, 3.1); arrow(9.62, 3.1, 10.08, 3.1)
    arrow(5.92, 1.1, 6.78, 1.1); arrow(9.62, 1.1, 10.08, 1.1)
    plt.savefig("visualization_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved visualization_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 86 dropped every word seen fewer than 5 times, and has no vector for a word it never saw. Day 88 builds "
          "word vectors from character n-grams (fastText), so that a new word gets a vector from its pieces.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        words, E = load_vectors()                            # Day 86's table
        E = unit(E)                                          # Day 86 compared directions (cosine); lengths are left out
        share = demo_pca(words, E)                           # 1. PCA
        demo_gradient_check()                                # 2. t-SNE's gradient
        idx, groups, cwords = pick_points(words, E)          # the 500 words
        Zp, ts, rows = demo_projections(words, E, idx, groups)  # 3. PCA against t-SNE
        Yr = demo_controls(E, idx, groups)                   # 4. controls
        plot_results(share, Zp, ts, Yr, groups, cwords)      # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 5. Day 88


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
