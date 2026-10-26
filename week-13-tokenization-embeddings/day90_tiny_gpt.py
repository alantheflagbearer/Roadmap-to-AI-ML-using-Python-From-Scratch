"""
Day 90: The Week 13 Build -- a Tiny Character-Level GPT-Style Model on Shakespeare

Week 13's last day does two jobs, like Days 62, 69, 76 and 83. The syllabus
topic is a tiny character-level GPT-style model, combining Weeks 11-13. And
this is the week's integration day: the model is assembled from Week 13's
exercise scripts and Week 12's Transformer parts.

A GPT-style model is a stack of Transformer blocks that may only look
backwards: position t reads positions 0..t and predicts the token at t + 1.
It is trained on nothing but raw text.

The parts, and where each one goes:

  Day 84 exercise  merge, train, encode, decode       characters (0 merges) or BPE tokens -> ids, and back
  Day 85 exercise  embed, embed_backward              the token table AND the position table
  Day 86 exercise  output_layer, softmax_xent, ...    the next-token scores and the loss
  Day 87 exercise  pca                                 a picture of the learned character table
  Day 88 exercise  char_ngrams                         are the model's invented words made of real pieces?
  Day 89 exercise  save_vectors, load_vectors, ...     the character table to a file, and its neighbours
  Week 12 (Day 83's build, imported)                   attention and heads (Days 75-76), layer_norm (79),
                                                       causal_mask (80), encoder_block_backward (78)
  Day 37                                               Adam's update rule

The one new piece of model code: a GPT block is Day 78's encoder block with
Day 80's causal mask in place of the padding mask -- one line differs -- and
its backward pass is Day 78's encoder_block_backward unchanged, because the
mask only decides which attention weights are 0.

The text is Tiny Shakespeare (1,115,394 characters, in ./corpora): the first
90% to train on, the last 10% to measure on.

What is measured, reported as it came out:
  1. The build fits: a gradient check of the whole model, and the causal mask.
  2. Tokens: characters, and BPE pieces from Day 84's tokenizer.
  3. What to beat: n-gram counting models (1 to 5 characters of context).
  4. Training, measured every 500 steps, two seeds: bits per character,
     and how much context it really uses.
  5. Characters against BPE tokens, at the same model size and budget.
  6. What it writes: samples, and its invented words.
  7. What its character table learned.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep an imported script quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths to the corpus and to Week 12's folder
import re                                              # re: splitting text into words
import sys                                             # sys: path, to import Day 83's build
import tempfile                                        # tempfile: where the character table's file goes during the check
import time                                            # time: perf_counter, for timings
from collections import Counter                        # Counter counts pairs (Day 84) and n-grams

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, os.path.join(HERE, "..", "week-12-the-transformer"))  # Week 12's folder
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing; kept quiet to be sure
    import day83_tiny_transformer as w12                # Week 12's build: attention, heads, layer_norm, causal_mask, block backward

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

CORPUS = os.path.join(HERE, "corpora", "tinyshakespeare.txt")  # Tiny Shakespeare
T, B = 64, 32                                           # context length (tokens), sequences per batch
D, HEADS, F, LAYERS = 64, 4, 256, 2                     # model width, heads, feed-forward width, blocks
LR, STEPS = 3e-3, 4000                                  # Adam's learning rate, training steps
CHECK_EVERY = 500                                       # measure on held-out text this often
SEEDS = [0, 1]                                          # two seeds for the character model
MERGES = 200                                            # BPE merges for section 5


# =============================================================================
# 1. THE PARTS -- Week 13's exercise scripts, verbatim
# =============================================================================
# ---- from day84_exercise.py, verbatim ----
def merge(seq, pair, new):                              # replace every occurrence of pair in a list of symbols
    out, i = [], 0                                        # result; position
    while i < len(seq):                                   # left to right
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:  # the pair starts here
            out.append(new); i += 2                           # one symbol for two
        else:
            out.append(seq[i]); i += 1                        # unchanged
    return out                                            # the merged list


def train(text, n_merges):                              # learn merges: the most frequent adjacent pair, n_merges times
    seq, merges = list(text), []                          # the text as characters; the merges learned
    for _ in range(n_merges):                             # one merge per round
        counts = Counter(zip(seq, seq[1:]))                 # every adjacent pair and its count
        if not counts:                                      # one symbol left
            break                                             # nothing to merge
        best = max(counts, key=lambda p: (counts[p], p))    # most frequent; ties broken by the pair itself
        merges.append(best)                                 # remember it
        seq = merge(seq, best, best[0] + best[1])           # apply it to the training text
    vocab = sorted(set(text)) + [a + b for a, b in merges]  # every symbol: the characters, then the merged ones
    return merges, vocab                                  # merges in order, and the vocabulary


def encode(text, merges, vocab):                        # text -> ids, applying the merges in the order learned
    seq = list(text)                                      # characters
    for pair in merges:                                   # earliest merge first
        seq = merge(seq, pair, pair[0] + pair[1])           # apply it
    index = {s: i for i, s in enumerate(vocab)}           # symbol -> id
    return [index[s] for s in seq]                        # a KeyError here means a character never seen in training


def decode(ids, vocab):                                 # ids -> text
    return "".join(vocab[i] for i in ids)                 # every symbol is its own piece of text


# ---- from day85_exercise.py, verbatim ----
def embed(E, ids):                                      # NEW: (V, d) table, ids of any shape -> vectors of shape ids.shape + (d,)
    return E[ids]                                         # fancy indexing: one row per id


def embed_backward(dX, E, ids):                         # NEW: gradient for the table, from the gradient at the looked-up vectors
    dE = np.zeros_like(E)                                 # one row per token
    np.add.at(dE, ids, dX)                                # a token used at several positions collects all of their gradients
    return dE                                             # rows of unused tokens stay 0


# ---- from day86_exercise.py, verbatim ----
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


# ---- from day87_exercise.py, verbatim ----
def pca(X, k):                                          # NEW: the k directions of largest spread, and the data in them
    Xc = X - X.mean(axis=0)                               # centre every column
    C = Xc.T @ Xc / (len(X) - 1)                          # covariance matrix (d, d)
    vals, vecs = np.linalg.eigh(C)                        # eigen-decomposition of a symmetric matrix, smallest first
    order = np.argsort(vals)[::-1]                        # largest spread first
    vals, vecs = vals[order], vecs[:, order]              # reordered
    return Xc @ vecs[:, :k], vecs[:, :k], vals / vals.sum()  # projections, directions, share of spread per direction


# ---- from day88_exercise.py, verbatim ----
def char_ngrams(word, nmin=3, nmax=6):                  # NEW: every character n-gram of '<word>', lengths nmin..nmax
    w = "<" + word + ">"                                  # start and end marks
    return [w[i:i + n] for n in range(nmin, nmax + 1) for i in range(len(w) - n + 1)]  # all of them, in order


# ---- from day89_exercise.py, verbatim ----
def save_vectors(path, words, M):                       # NEW: GloVe's text format; spaces and newlines inside a token are escaped
    with open(path, "w", encoding="utf-8") as f:          # plain text
        for w, v in zip(words, M):                          # one line per token
            tok = w.replace(" ", "<sp>").replace("\n", "<nl>")  # a character model has ' ' and newline as tokens
            f.write(tok + " " + " ".join(f"{x:.5f}" for x in v) + "\n")


def load_vectors(path):                                 # NEW: GloVe's format, or word2vec's with its 'count dim' header
    with open(path, encoding="utf-8") as f:               # plain text
        lines = f.read().splitlines()                       # every line
    first = lines[0].split()                              # a word2vec file starts with two integers
    if len(first) == 2 and all(t.isdigit() for t in first):  # header: skip it
        lines = lines[1:]
    words = [ln.split(" ", 1)[0].replace("<sp>", " ").replace("<nl>", "\n") for ln in lines]  # undo the escaping
    return words, np.array([[float(x) for x in ln.split(" ")[1:]] for ln in lines])  # tokens, (n, d) matrix


def most_similar(words, M, w, k=3):                     # NEW: the k nearest tokens to w by cosine similarity
    N = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)  # unit rows
    s = N @ N[words.index(w)]; s[words.index(w)] = -np.inf  # cosines; not w itself
    return [(words[i], round(float(s[i]), 3)) for i in np.argsort(-s)[:k]]  # most similar first


# =============================================================================
# 2. THE MODEL -- NEW: Day 78's block with Day 80's causal mask, stacked
# =============================================================================
def gpt_block(x, P, h):                                 # NEW: Day 78's encoder_block with the causal mask -- the only line that differs
    m, n, _ = x.shape                                     # batch, length
    km = np.broadcast_to(w12.causal_mask(n), (m * h, n, n))  # Day 80: query i may read keys 0..i, for every sequence and head
    Q, K, V = (w12.split(x @ P[w], h) for w in ("Wq", "Wk", "Wv"))  # Day 74's projections, cut into heads (Day 76)
    Zh, A = w12.attention(Q, K, V, km)                    # Day 75's attention, every head at once
    Z = w12.merge(Zh, h)                                  # heads side by side
    h1, n1 = w12.layer_norm(x + Z @ P["Wo"], P["g1"], P["b1"])  # add & norm (Day 79)
    u = h1 @ P["W1"] + P["c1"]; r = np.maximum(u, 0)      # FFN: widen, ReLU
    y, n2 = w12.layer_norm(h1 + r @ P["W2"] + P["c2"], P["g2"], P["b2"])  # FFN back to d, add & norm
    return y, (x, Q, K, V, A, Z, h1, n1, u, r, n2)        # Day 78's cache layout, so Day 78's backward pass applies


def init_gpt(rng, V, d=D, ff=F, layers=LAYERS, t=T):    # NEW: Day 81 exercise's scales; a learned position table (as in GPT)
    s = lambda a, b: rng.randn(a, b) / np.sqrt(a)         # a weight matrix
    blk = lambda: dict(Wq=s(d, d), Wk=s(d, d), Wv=s(d, d), Wo=s(d, d), W1=s(d, ff), c1=np.zeros(ff), W2=s(ff, d),
                       c2=np.zeros(d), g1=np.ones(d), b1=np.zeros(d), g2=np.ones(d), b2=np.zeros(d))  # Day 78's names
    return dict(E=rng.randn(V, d) * 0.1, pos=rng.randn(t, d) * 0.1, blocks={str(k): blk() for k in range(layers)},
                Wout=s(d, V) * 0.1, bout=np.zeros(V))     # token table, position table, blocks, output layer (x 0.1, Day 81)


def gpt_forward(ids, P, h=HEADS):                       # NEW: ids (m, n) -> scores (m, n, V) for the NEXT token at every position
    n = ids.shape[1]                                      # length
    x = embed(P["E"], ids) + embed(P["pos"], np.arange(n))  # Day 85's lookup, twice: the token and its position
    caches = []                                           # one per block
    for k in sorted(P["blocks"]):                         # bottom to top
        x, c = gpt_block(x, P["blocks"][k], h); caches.append(c)
    return output_layer(x, P["Wout"], P["bout"]), (ids, x, caches)  # Day 86's output layer, and the cache


def gpt_loss_grads(ids, y, P, h=HEADS):                 # NEW: mean cross-entropy and every gradient
    logits, (ids, x, caches) = gpt_forward(ids, P, h)     # forward
    loss, dl = softmax_xent(logits, y)                    # Day 86's loss
    dx, dW, db = output_layer_backward(dl, x, P["Wout"])  # Day 86's backward pass
    G = dict(Wout=dW, bout=db, blocks={})                 # gradients, in P's shape
    for k, c in reversed(list(zip(sorted(P["blocks"]), caches))):  # top to bottom
        dx, G["blocks"][k] = w12.encoder_block_backward(dx, P["blocks"][k], h, c)  # Day 78's backward pass, unchanged
    G["E"] = embed_backward(dx, P["E"], ids)              # Day 85: a token used many times collects every gradient
    G["pos"] = embed_backward(dx, P["pos"], np.broadcast_to(np.arange(ids.shape[1]), ids.shape))  # the same for positions
    return loss, G                                        # loss and gradients


def adam_step(P, G, state, lr, clip=1.0):               # Day 37's step_adam (same four lines), for nested dictionaries, after Day 60's clipping
    g = dict(w12.leaves(G))                               # gradients by dotted name
    scale = min(1.0, clip / np.sqrt(sum(np.sum(a ** 2) for a in g.values())))  # global-norm clipping at 1.0
    state["t"] += 1; t = state["t"]                       # step count, for the bias correction
    for k, A in w12.leaves(P):                            # every weight
        m = state["m"].setdefault(k, np.zeros_like(A)); v = state["v"].setdefault(k, np.zeros_like(A))  # moment estimates
        gk = g[k] * scale                                   # the clipped gradient
        m[:] = 0.9 * m + 0.1 * gk                           # Day 37: m = beta1 m + (1 - beta1) g
        v[:] = 0.999 * v + 0.001 * gk ** 2                  # Day 37: v = beta2 v + (1 - beta2) g^2
        A -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)  # Day 37: bias-corrected step


# =============================================================================
# 3. DATA, BASELINES, TRAINING AND SAMPLING -- NEW
# =============================================================================
def load_text():                                        # Tiny Shakespeare, split 90% / 10%
    text = open(CORPUS, encoding="utf-8").read()          # the whole file
    cut = int(0.9 * len(text))                            # where the held-out part starts
    return text[:cut], text[cut:]                         # training text, held-out text


def tokenize(tr_text, va_text, n_merges):               # Day 84's tokenizer: characters (0 merges) or BPE
    merges, vocab = train(tr_text, n_merges)              # Day 84's train()
    return merges, vocab, np.array(encode(tr_text, merges, vocab)), np.array(encode(va_text, merges, vocab))  # ids


def windows(ids):                                       # every non-overlapping window of T inputs and their next tokens
    n = (len(ids) - 1) // T                               # how many fit
    return ids[:n * T].reshape(n, T), ids[1:n * T + 1].reshape(n, T)  # inputs, targets


def bits_per_char(P, vx, vy, vocab):                    # held-out loss in bits per CHARACTER (comparable across tokenizers)
    nats = 0.0                                            # summed -log p over every predicted token
    for s in range(0, len(vx), 64):                       # 64 windows at a time
        lg, _ = gpt_forward(vx[s:s + 64], P)                # scores
        nats += softmax_xent(lg, vy[s:s + 64])[0] * vy[s:s + 64].size  # mean times count = sum
    chars = sum(len(vocab[i]) for i in vy.ravel())        # characters those tokens spell out
    return nats / chars / np.log(2)                       # nats -> bits, per character


def log_softmax(lg):                                    # log-probabilities over the last axis
    z = lg - lg.max(axis=-1, keepdims=True)               # stability
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))  # log p


def position_bits(P, vx, vy):                           # NEW: held-out bits per token at each position of the window
    tot = np.zeros(vx.shape[1])                           # summed -log2 p, per position
    for s in range(0, len(vx), 64):                       # 64 windows at a time
        logp = log_softmax(gpt_forward(vx[s:s + 64], P)[0])  # log-probabilities
        tot -= np.take_along_axis(logp, vy[s:s + 64][..., None], axis=-1)[..., 0].sum(axis=0) / np.log(2)  # the right tokens', per position
    return tot / len(vx)                                  # mean over windows: the prediction at position p sees p + 1 tokens


def context_bits(P, ids, vocab):                        # NEW: bits per character when every scored token has at least T/2 tokens of context
    h = T // 2                                            # windows start every h tokens; only their last h positions are scored
    starts = np.arange(0, len(ids) - T - 1, h)            # window starts
    nats, chars = 0.0, 0                                  # summed -ln p; characters scored
    for s in range(0, len(starts), 64):                   # 64 windows at a time
        st = starts[s:s + 64]                               # their starts
        x = np.stack([ids[i:i + T] for i in st]); y = np.stack([ids[i + 1:i + T + 1] for i in st])  # inputs, next tokens
        logp = log_softmax(gpt_forward(x, P)[0][:, h:])     # the last h positions only
        nats -= np.take_along_axis(logp, y[:, h:, None], axis=-1).sum()  # -ln p of the right tokens
        chars += sum(len(vocab[i]) for i in y[:, h:].ravel())  # characters they spell
    return nats / chars / np.log(2)                       # bits per character


def train_gpt(tr, vx, vy, vocab, seed, steps=STEPS):    # NEW: Adam on random windows; held-out bits per character every CHECK_EVERY steps
    rng = np.random.RandomState(seed)                     # weights and batches
    P = init_gpt(rng, len(vocab)); state = {"t": 0, "m": {}, "v": {}}  # fresh model and optimiser
    cpt = np.array([len(v) for v in vocab])[tr].mean()    # characters per training token (1 for characters)
    curve, recent, busy = [(0, np.nan, bits_per_char(P, vx, vy, vocab))], [], 0.0  # untrained point; recent losses; training seconds
    for step in range(1, steps + 1):                      # training steps
        t0 = time.perf_counter()                            # time the step only (the held-out checks are not counted)
        o = rng.randint(0, len(tr) - T - 1, B)              # B random starting points
        x = np.stack([tr[i:i + T] for i in o]); y = np.stack([tr[i + 1:i + T + 1] for i in o])  # inputs, next tokens
        loss, G = gpt_loss_grads(x, y, P)                   # forward and backward
        adam_step(P, G, state, LR); recent.append(loss)     # update; remember the loss
        busy += time.perf_counter() - t0                    # training time so far
        if step % CHECK_EVERY == 0:                         # a check
            curve.append((step, np.mean(recent) / np.log(2) / cpt, bits_per_char(P, vx, vy, vocab))); recent = []  # train, held-out
    return P, np.array(curve), busy / steps               # weights, (step, train bits/char, held-out bits/char), training seconds per step


RUNS = {}                                               # cache: each configuration trains once


def train_cached(key, *args, **kw):                     # train_gpt(), remembered by a name
    if key not in RUNS:                                   # first request: train it
        RUNS[key] = train_gpt(*args, **kw)                  # train and remember
    return RUNS[key]                                      # later requests reuse it


def ngram_bits(tr_text, va_text, order, vocab_size):    # NEW: a counting model with (order - 1) characters of context, add-k smoothing
    def counts(text):                                     # how often each context, and each (context, next character), occurs
        c1, c2 = Counter(), Counter()                       # context counts; (context, character) counts
        for i in range(order - 1, len(text)):               # every position with a full context
            ctx = text[i - order + 1:i]; c1[ctx] += 1; c2[ctx, text[i]] += 1
        return c1, c2

    def bits(c1, c2, text, k):                            # mean bits per character of text under add-k smoothing
        tot = 0.0                                           # summed bits
        for i in range(order - 1, len(text)):               # every predicted character
            ctx = text[i - order + 1:i]
            tot -= np.log2((c2[ctx, text[i]] + k) / (c1[ctx] + k * vocab_size))
        return tot / (len(text) - order + 1)

    cut = int(0.9 * len(tr_text))                         # k is chosen on the last 10% of the TRAINING text
    c1, c2 = counts(tr_text[:cut])                        # counts without it
    k = min((0.01, 0.1, 1.0), key=lambda k: bits(c1, c2, tr_text[cut:], k))  # the best smoothing there
    c1, c2 = counts(tr_text)                              # counts from all the training text
    return bits(c1, c2, va_text, k), k, len(c1)           # held-out bits per character, the k used, distinct contexts


def sample(P, vocab, n, rng, start):                    # NEW: write n tokens, each drawn from the model's distribution (temperature 1)
    ids = list(start)                                     # the prompt
    for _ in range(n):                                    # one token at a time
        lg, _ = gpt_forward(np.array([ids[-T:]]), P)        # the last T tokens are all the model can see
        p = np.exp(lg[0, -1] - lg[0, -1].max()); p /= p.sum()  # softmax of the last position's scores
        ids.append(int(rng.choice(len(vocab), p=p)))        # draw the next token
    return decode(ids[len(start):], vocab)                # the new text (Day 84's decode)


# =============================================================================
# DEMOS
# =============================================================================
def demo_parts():
    print("=" * 70)
    print("1. THE BUILD -- the parts together, checked")
    print("=" * 70)
    print("Every function in section 1 is copied verbatim from a Week 13 exercise script (a banner names each one); "
          "Week 12's parts are imported from Day 83's build. A gradient check of the whole model at tiny sizes "
          "(10 tokens, length 6, d = 8, 2 heads, FFN 12, 2 blocks), every entry nudged by +-1e-5, Day 62's pass rule.")
    rng = np.random.RandomState(0)                        # fixed rng
    P = init_gpt(rng, 10, d=8, ff=12, layers=2, t=6)      # tiny model
    for _, A in w12.leaves(P):                            # move every weight a little off its start
        A += 0.05 * rng.randn(*A.shape)
    x, y = rng.randint(0, 10, (3, 6)), rng.randint(0, 10, (3, 6))  # 3 sequences, next tokens
    _, G = gpt_loss_grads(x, y, P, h=2); G = dict(w12.leaves(G))  # analytic
    worst, ok, count = 0.0, True, 0                       # largest difference; every entry passes; entries
    for name, A in w12.leaves(P):                         # every array
        for pos in np.ndindex(A.shape):                     # every entry
            old = A[pos]                                      # remember it
            A[pos] = old + 1e-5; up = gpt_loss_grads(x, y, P, h=2)[0]  # nudge up
            A[pos] = old - 1e-5; dn = gpt_loss_grads(x, y, P, h=2)[0]  # nudge down
            A[pos] = old                                      # restore
            num = (up - dn) / 2e-5; count += 1                # central difference
            worst = max(worst, abs(num - G[name][pos])); ok &= abs(num - G[name][pos]) <= 1e-10 + 1e-6 * abs(num)
    print(f"\ngradient check: {count} entries, largest |numeric - analytic| = {worst:.2e}, every entry passes: {ok}")
    lg, _ = gpt_forward(x, P, h=2); x2 = x.copy(); x2[:, 4] = (x2[:, 4] + 1) % 10  # change the token at position 4
    lg2, _ = gpt_forward(x2, P, h=2)                      # again
    print(f"changing the token at position 4: largest change in the scores at positions 0-3 {np.abs(lg2[:, :4] - lg[:, :4]).max():.1e}, "
          f"at positions 4-5 {np.abs(lg2[:, 4:] - lg[:, 4:]).max():.1e} (the past cannot see the future)")
    n = sum(A.size for _, A in w12.leaves(init_gpt(np.random.RandomState(0), 65)))  # the lesson's size, 65 characters
    print(f"the lesson's model: {LAYERS} blocks, d = {D}, {HEADS} heads, FFN {F}, context {T}: {n} weights for 65 characters")
    print("Actual output from this lesson's run.")


def demo_tokens(tr_text, va_text):
    print("\n" + "=" * 70)
    print("2. TOKENS -- characters, or BPE pieces (Day 84's tokenizer)")
    print("=" * 70)
    print(f"{len(tr_text):,} training characters, {len(va_text):,} held-out characters. With 0 merges Day 84's tokenizer "
          f"gives characters; with {MERGES} merges, BPE pieces (spaces and newlines are ordinary characters, so pieces "
          "can cross word boundaries).")
    out = {}                                              # name -> (merges, vocab, train ids, held-out ids)
    for name, k in (("characters", 0), ("BPE", MERGES)):  # two tokenizations
        t0 = time.perf_counter(); out[name] = tokenize(tr_text, va_text, k)  # Day 84's train and encode
        merges, vocab, tr, va = out[name]
        print(f"\n{name}: vocabulary {len(vocab)}, held-out text {len(va_text):,} characters -> {len(va):,} tokens "
              f"({len(va_text) / len(va):.2f} characters per token); decodes back exactly: {decode(list(va), vocab) == va_text}; "
              f"{time.perf_counter() - t0:.0f} s")
    print("characters: " + repr("".join(out["characters"][1]))[:90])
    print("first BPE merges: " + " | ".join(repr(a + b) for a, b in out["BPE"][0][:10]))
    print("Actual output from this lesson's run.")
    return out                                             # later sections train on them


def demo_baselines(tr_text, va_text, V):
    print("\n" + "=" * 70)
    print("3. WHAT TO BEAT -- counting models")
    print("=" * 70)
    print("An n-gram model predicts the next character from the previous n - 1 by counting how often each followed "
          "in the training text; add-k smoothing (k chosen on the last 10% of the training text) keeps unseen "
          "combinations possible. Held-out bits per character: log2(65) = 6.02 is a uniform guess.")
    print(f"\n{'context':<20}{'bits/char':>10}{'k':>7}{'contexts seen':>15}")
    out = {}                                              # order -> bits
    for order in (1, 2, 3, 4, 5):                         # context of 0..4 characters
        b, k, nc = ngram_bits(tr_text, va_text, order, V)   # count and score
        out[order] = b
        print(f"{str(order - 1) + ' characters':<20}{b:>10.3f}{k:>7}{nc:>15,}")
    print("Actual output from this lesson's run.")
    return out                                             # the plot shows them


def demo_training(tok):
    print("\n" + "=" * 70)
    print("4. TRAINING -- characters, two seeds, measured every 500 steps")
    print("=" * 70)
    merges, vocab, tr, va = tok["characters"]             # character ids
    vx, vy = windows(va)                                  # every held-out window
    print(f"Adam (Day 37) with learning rate {LR}, clipping at 1.0 (Day 60), batches of {B} random windows of {T} "
          f"characters, {STEPS} steps. Held-out: all {len(vx)} non-overlapping windows of the last 10% (each window "
          f"starts with no context). Bits per character, train = mean over the last {CHECK_EVERY} steps' batches.")
    curves = []                                           # one per seed
    for s in SEEDS:                                       # two seeds
        P, c, sec = train_cached(("char", s), tr, vx, vy, vocab, s)
        curves.append(c)
    print(f"\n{'step':>6}" + "".join(f"{'train s' + str(s):>10}{'held-out s' + str(s):>13}" for s in SEEDS))
    for i in range(len(curves[0])):                       # one row per check
        print(f"{int(curves[0][i, 0]):>6}" + "".join(f"{c[i, 1]:>10.3f}{c[i, 2]:>13.3f}" if i else f"{'-':>10}{c[i, 2]:>13.3f}" for c in curves))
    print(f"({train_cached(('char', 0))[2] * 1000:.0f} ms per training step here, held-out checks not counted)")
    print("Actual output from this lesson's run.")
    pos = position_bits(train_cached(("char", 0))[0], vx, vy)  # seed 0, by position in the window
    print("\nThe held-out windows above start with no context. Seed 0's held-out bits per character by position in the "
          "window (the prediction at position p is made from p + 1 characters):")
    print("  " + ", ".join(f"position {i}: {pos[i]:.2f}" for i in (0, 1, 2, 4, 8, 16, 32, 63)))
    print("every character scored with at least 32 characters of context: " +
          ", ".join(f"seed {s} {context_bits(train_cached(('char', s))[0], va, vocab):.3f}" for s in SEEDS))
    print("Actual output from this lesson's run.")
    return curves, pos                                     # the plot shows them


def demo_bpe(tok):
    print("\n" + "=" * 70)
    print("5. CHARACTERS AGAINST BPE TOKENS -- same model, same steps, seed 0")
    print("=" * 70)
    merges, vocab, tr, va = tok["BPE"]                    # BPE ids
    vx, vy = windows(va)                                  # held-out windows of T BPE tokens
    P, c, sec = train_cached(("bpe", 0), tr, vx, vy, vocab, 0)  # train
    cc = train_cached(("char", 0))[1]                     # the character model's curve
    n = sum(A.size for _, A in w12.leaves(P))             # weights
    print(f"The same model with a {len(vocab)}-token vocabulary ({n} weights: the token table and output layer grow). "
          f"A window of {T} BPE tokens spans about {T * np.array([len(v) for v in vocab])[va].mean():.0f} characters, "
          f"against {T} for the character model. Both measured in bits per CHARACTER on the same held-out text.")
    print(f"\n{'step':>6}{'characters':>12}{'BPE':>8}")
    for i in range(len(c)):                               # one row per check
        print(f"{int(c[i, 0]):>6}{cc[i, 2]:>12.3f}{c[i, 2]:>8.3f}")
    print(f"with at least 32 tokens of context for every scored token: characters {context_bits(train_cached(('char', 0))[0], tok['characters'][3], tok['characters'][1]):.3f}, "
          f"BPE {context_bits(P, va, vocab):.3f} bits per character")
    print(f"ms per step: characters {train_cached(('char', 0))[2] * 1000:.0f}, BPE {sec * 1000:.0f}")
    print("Actual output from this lesson's run.")
    return c                                               # the plot shows it


def demo_samples(tok, tr_text, va_text):
    print("\n" + "=" * 70)
    print("6. WHAT IT WRITES -- samples, and the words it makes up (character model, seed 0)")
    print("=" * 70)
    merges, vocab, tr, va = tok["characters"]             # characters
    P = train_cached(("char", 0))[0]                      # the trained model
    start = encode("\n", merges, vocab)                   # start after a newline
    print("Drawn at temperature 1 (plain sampling, every character from the model's distribution):\n")
    print(sample(P, vocab, 300, np.random.RandomState(0), start))
    long = sample(P, vocab, 5000, np.random.RandomState(1), start)  # a longer sample for counting
    known = set(re.findall(r"[a-z]+", tr_text.lower()))  # every word in the training text
    gen = re.findall(r"[a-z]+", long.lower())             # the sample's words
    new = sorted({w for w in gen if w not in known and len(w) >= 4})  # invented words of 4+ letters
    grams = {g for w in known for g in char_ngrams(w)}    # every piece (Day 88) of every training word
    piece_share = lambda ws: np.mean([np.mean([g in grams for g in char_ngrams(w)]) for w in ws])  # share of a word's pieces seen in training
    letters = np.array(list("".join(re.findall(r"[a-z]", tr_text.lower()))))  # training letters, for random strings
    rng = np.random.RandomState(2)                        # random strings with the same lengths and letter frequencies
    rand = ["".join(rng.choice(letters, len(w))) for w in new]
    real_new = sorted({w for w in re.findall(r"[a-z]+", va_text.lower()) if w not in known and len(w) >= 4})  # real words unseen in training
    print(f"\nin a 5,000-character sample: {len(gen)} words, {np.mean([w in known for w in gen]):.3f} of them real "
          f"training words; {len(new)} different invented words of 4+ letters, e.g. " + ", ".join(new[:8]))
    print(f"share of each word's character pieces (3-6, Day 88) that occur in training words: invented {piece_share(new):.3f}, "
          f"real held-out words never seen in training {piece_share(real_new):.3f} ({len(real_new)} words), "
          f"random letters {piece_share(rand):.3f}")
    print("Actual output from this lesson's run.")
    return piece_share(new), piece_share(real_new), piece_share(rand)  # the plot shows them


def demo_table(tok):
    print("\n" + "=" * 70)
    print("7. WHAT THE CHARACTER TABLE LEARNED -- seed 0")
    print("=" * 70)
    merges, vocab, tr, va = tok["characters"]             # characters
    E = train_cached(("char", 0))[0]["E"]                 # the learned token table (65 x 64)
    path = os.path.join(HERE, "day90_char_vectors.txt")   # Day 89's format
    save_vectors(path, vocab, E); w2, E2 = load_vectors(path)  # save and load back (Day 89)
    print(f"saved to day90_char_vectors.txt and loaded back: same tokens {w2 == vocab}, largest |difference| {np.abs(E2 - E).max():.1e}")
    for ch in ("a", "e", "T", "?", " ", "\n"):            # a few characters' neighbours (Day 89's most_similar)
        print(f"  {repr(ch):<6} -> " + ", ".join(f"{repr(w)} {s:.2f}" for w, s in most_similar(w2, E2, ch, k=4)))
    low = [c for c in "abcdefghijklmnopqrstuvwxyz" if c.upper() in vocab]  # letters with both cases
    case = np.mean([c.upper() in [w for w, _ in most_similar(vocab, E, c, k=5)] for c in low])  # upper case among the 5 nearest?
    letters = list("abcdefghijklmnopqrstuvwxyz")          # the lowercase letters
    sub = E[[vocab.index(c) for c in letters]]            # their rows of the table
    nearest = lambda c: most_similar(letters, sub, c, k=1)[0][0]  # the most similar other lowercase letter (Day 89)
    near_v = np.mean([nearest(v) in "aeiou" for v in "aeiou"])  # is a vowel's nearest letter a vowel?
    print(f"a lowercase letter's capital is among its 5 nearest characters: {case:.3f} of {len(low)} letters")
    print(f"a vowel's nearest other lowercase letter is a vowel: {near_v:.3f} of 5 (by chance about 4/25 = 0.16)")
    Z, _, share = pca(E, 2)                               # Day 87's PCA
    print(f"PCA (Day 87): two directions keep {share[:2].sum():.3f} of the table's spread")
    print("Actual output from this lesson's run.")
    return Z, vocab                                        # the plot shows them


def demo_cost(tok):
    print("\n" + "=" * 70)
    print("8. WHAT IT COST")
    print("=" * 70)
    for name, key in (("characters, seed 0", ("char", 0)), ("characters, seed 1", ("char", 1)), ("BPE, seed 0", ("bpe", 0))):
        P, c, sec = train_cached(key)                       # a finished run
        print(f"{name:<20} {sum(A.size for _, A in w12.leaves(P)):>7} weights, {sec * 1000:.0f} ms per step, "
              f"{sec * STEPS / 60:.1f} minutes for {STEPS} steps")
    print("(one CPU thread for the matrix library; only ratios within this run are meaningful)")
    print("Actual output from this lesson's run.")


def plot_results(curves, pos, cb, base, pieces, Z, vocab):  # four panels: learning | context | invented words | the character table
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))          # canvas
    for s, c in zip(SEEDS, curves):                         # the character model, two seeds
        ax[0].plot(c[:, 0], c[:, 2], marker="o", markersize=3, color="#1f77b4", alpha=0.6 + 0.4 * (s == 0), label=f"GPT, characters, seed {s}")
    ax[0].plot(cb[:, 0], cb[:, 2], marker="s", markersize=3, color="#ff7f0e", label=f"GPT, BPE ({MERGES} merges), seed 0")
    for order, ls in ((2, ":"), (3, "--"), (5, "-.")):    # three counting models
        ax[0].axhline(base[order], color="#7f7f7f", linestyle=ls, linewidth=1, label=f"{order - 1}-character context, counted")
    ax[0].set_ylim(1.5, 5.0); ax[0].set_xlabel("training step"); ax[0].set_ylabel("held-out bits per character")
    ax[0].set_title("Learning, measured every 500 steps", fontsize=10); ax[0].legend(fontsize=7, loc="upper right"); ax[0].grid(alpha=0.3)
    ax[1].plot(np.arange(T), pos, marker="o", markersize=2, color="#1f77b4", label="GPT, characters, seed 0")  # loss by context length
    for order, ls in ((2, ":"), (3, "--"), (5, "-.")):    # the counting models, for comparison
        ax[1].axhline(base[order], color="#7f7f7f", linestyle=ls, linewidth=1, label=f"{order - 1}-character context, counted")
    ax[1].set_ylim(1.5, 5.0); ax[1].set_xlabel("position in the window (the prediction sees position + 1 characters)"); ax[1].set_ylabel("held-out bits per character")
    ax[1].set_title("Does it use more than 4 characters of context?", fontsize=10); ax[1].legend(fontsize=7, loc="upper right"); ax[1].grid(alpha=0.3)
    ax[2].bar(["invented by\nthe model", "real words unseen\nin training", "random\nletters"], pieces, color=["#1f77b4", "#2ca02c", "#7f7f7f"])
    ax[2].set_ylim(0, 1); ax[2].set_ylabel("share of a word's pieces (3-6 chars)\nseen in training words")
    ax[2].set_title("Are its made-up words made of real pieces?", fontsize=10); ax[2].grid(alpha=0.3, axis="y")
    for i, ch in enumerate(vocab):                         # the character table in 2-D (Day 87's PCA)
        color = "#1f77b4" if ch.islower() else "#d62728" if ch.isupper() else "#2ca02c"
        ax[3].text(Z[i, 0], Z[i, 1], {" ": "sp", "\n": "nl"}.get(ch, ch), color=color, fontsize=9, ha="center", va="center")
    ax[3].set_xlim(Z[:, 0].min() - 0.3, Z[:, 0].max() + 0.3); ax[3].set_ylim(Z[:, 1].min() - 0.3, Z[:, 1].max() + 0.3)
    ax[3].set_title("Character table, PCA (blue lower, red upper, green other)", fontsize=10); ax[3].set_xticks([]); ax[3].set_yticks([])
    plt.tight_layout(); plt.savefig("tiny_gpt_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved tiny_gpt_results.png")


def draw_diagram():                                     # the build: the parts into one model
    fig, ax = plt.subplots(figsize=(13, 5.0))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.0); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8):                       # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                     # one arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.1))

    ax.text(6.5, 4.75, "Day 90: Week 13's build -- a tiny character-level GPT from the week's parts", ha="center",
            fontsize=12, fontweight="bold")
    flow = [(4.0, "text -> ids\n(Day 84: train, encode)", "#FAEEDA"), (3.15, "token + position vectors\n(Day 85: embed, twice)", "#FAEEDA"),
            (2.3, f"{LAYERS} GPT blocks: Day 78's block\nwith Day 80's causal mask", "#DCEBF7"),
            (1.45, "next-token scores, loss\n(Day 86: output_layer, softmax_xent)", "#FAEEDA"),
            (0.6, "Adam (Day 37), clipping (Day 60)", "#E5F2E1")]
    for y, text, color in flow:                                      # the model, bottom to top in the data's order
        box(3.4, y, 4.2, 0.65, text, color)
    for (y0, _, _), (y1, _, _) in zip(flow[:-1], flow[1:]):          # arrows down the column
        arrow(3.4, y0 - 0.33, 3.4, y1 + 0.33)
    box(9.6, 3.4, 4.6, 0.75, "Week 12 (imported from Day 83): attention and heads,\nlayer_norm, causal_mask, encoder_block_backward", "#DCEBF7")
    arrow(7.28, 3.2, 5.52, 2.45)
    box(9.6, 1.9, 4.6, 0.95, "afterwards: the character table in 2-D (Day 87: pca),\nto a file and its neighbours (Day 89),\n"
        "its invented words' pieces (Day 88: char_ngrams)", "#F2F2F2")
    arrow(5.52, 1.45, 7.28, 1.75)
    plt.savefig("tiny_gpt_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved tiny_gpt_diagram.png")


def note_on_the_week():
    print("\n" + "=" * 70)
    print("9. THE WEEK, AND WHAT COMES NEXT")
    print("=" * 70)
    print("Week 13 turned text into numbers: tokens (Day 84), vectors (85), vectors learned from raw text (86, 88, 89) "
          "and looked at (87); today the parts trained a language model. Week 14 studies that model properly: "
          "next-token prediction (Day 91), honest perplexity numbers (92) and sampling strategies (93).")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        tr_text, va_text = load_text()                       # the corpus
        demo_parts()                                         # 1. the build, checked
        tok = demo_tokens(tr_text, va_text)                  # 2. tokens
        base = demo_baselines(tr_text, va_text, len(tok["characters"][1]))  # 3. counting models
        curves, pos = demo_training(tok)                     # 4. training
        cb = demo_bpe(tok)                                   # 5. characters against BPE
        pieces = demo_samples(tok, tr_text, va_text)         # 6. samples
        Z, vocab = demo_table(tok)                           # 7. the character table
        demo_cost(tok)                                       # 8. cost
        plot_results(curves, pos, cb, base, pieces, Z, vocab)  # the figure
        draw_diagram()                                       # the diagram
        note_on_the_week()                                   # 9. Week 14


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
