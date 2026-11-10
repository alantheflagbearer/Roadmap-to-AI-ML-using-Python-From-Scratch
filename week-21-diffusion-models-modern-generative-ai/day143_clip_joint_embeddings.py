"""
Day 143: Text-to-Image Models Conceptually -- CLIP-Style Joint Embeddings, a Real Demo

Every generator so far chose its own digit (or took a digit label, Day 137).
Text-to-image models take a sentence instead. Their first piece is usually a
pair of encoders trained so that an image and a caption describing it land
close together in one shared space: CLIP (Radford et al., 2021), trained on
400 million image-caption pairs from the web.

Today's version is built from scratch on MNIST. There are no human captions, so
each digit's caption is written by program from its label and two things
measured from its pixels: how thick its strokes are and which way it leans.
  * an image encoder (Day 126's dense network) and a bag-of-words text encoder
  * CLIP's symmetric contrastive loss, with its gradient written out by hand
  * zero-shot classification: naming digits with no classifier trained
  * retrieval by description, including a combination never seen in training

  1. By hand: the contrastive loss on three pairs, and its gradient.
  2. Captions from measurements.
  3. Training the two encoders.
  4. Zero-shot classification, and what the prompt changes.
  5. Finding digits by description.
  6. A combination never seen in training.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier parts
import time                                            # time: how long training takes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)

NUM = "zero one two three four five six seven eight nine".split()                      # the digit words
VOCAB = NUM + "a the digit handwritten written thin thick leaning left right upright and".split()  # every word the captions use
IDX = {w: i for i, w in enumerate(VOCAB)}               # word -> its column in a bag of words
ACTS = ["relu", "linear"]                               # both encoders: one hidden ReLU layer, a linear output
TAU = 0.1                                               # the temperature: similarities are divided by it before the softmax (the only value tried)
EPOCHS = 10                                             # passes over the 60,000 training digits
HOLD = (7, 1)                                           # the held-out combination: a seven is never called "thick" in section 6's training


# =============================================================================
# THE PARTS -- NEW
# =============================================================================
def measure(X, y, n_train=60000):                       # NEW: thickness and lean of every digit, each cut into thirds within its own digit
    ink = X.mean(1)                                                                   # stroke thickness, roughly: the average pixel
    R = np.repeat(np.arange(28.0), 28)[None]; Cc = np.tile(np.arange(28.0), 28)[None]  # each pixel's row and column
    m = X.sum(1, keepdims=True) + 1e-9; mr = (X * R).sum(1, keepdims=True) / m; mc = (X * Cc).sum(1, keepdims=True) / m  # the ink's centre
    slant = -(X * (R - mr) * (Cc - mc)).sum(1) / (X * (R - mr) ** 2).sum(1)           # how far the ink moves right per row up: positive leans right
    thick = np.zeros(len(X), int); lean = np.zeros(len(X), int)
    for d in range(10):                                                               # thirds within each digit, from the training digits only
        tr = y[:n_train] == d; k = y == d
        a1, a2 = np.percentile(ink[:n_train][tr], [100 / 3, 200 / 3]); b1, b2 = np.percentile(slant[:n_train][tr], [100 / 3, 200 / 3])
        thick[k] = np.where(ink[k] < a1, -1, np.where(ink[k] > a2, 1, 0))             # -1 thin, 0 neither, 1 thick
        lean[k] = np.where(slant[k] < b1, -1, np.where(slant[k] > b2, 1, 0))          # -1 leans left, 0 upright, 1 leans right
    return ink, slant, thick, lean


def caption(d, th, ln, rng, hold=None):                  # NEW: a random caption for a digit d with thickness th and lean ln
    w = [NUM[d]]
    if th != 0 and rng.rand() < 0.6 and (d, th) != hold:                               # mention the thickness 60% of the time (never for the held-out pair)
        w.append("thick" if th > 0 else "thin")
    if ln != 0 and rng.rand() < 0.6:                                                  # mention a clear lean 60% of the time
        w += ["leaning", "right" if ln > 0 else "left"]
    elif ln == 0 and rng.rand() < 0.3:                                                # and an upright digit 30% of the time
        w.append("upright")
    return w + [["a"], ["the", "digit"], ["handwritten"], ["written"], []][rng.randint(5)]  # plus filler words; order never matters (bag of words)


def bow(captions):                                      # NEW: each caption as word counts over VOCAB, divided by its length; unknown words are dropped
    B = np.zeros((len(captions), len(VOCAB)), np.float32)
    for r, ws in enumerate(captions):
        for w in ws:
            if w in IDX:
                B[r, IDX[w]] += 1
        B[r] /= max(B[r].sum(), 1)                                                    # an average of word vectors, not a sum
    return B


def clip_loss(Zi, Zt, tau=TAU):                         # NEW: CLIP's symmetric contrastive loss, and its gradient for both encoders
    ni = np.linalg.norm(Zi, axis=1, keepdims=True) + 1e-12; nt = np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-12
    U, V = Zi / ni, Zt / nt; S = U @ V.T / tau; n = len(S); I = np.arange(n)            # unit vectors; S[i, j]: image i against caption j
    P1 = np.exp(S - S.max(1, keepdims=True)); P1 /= P1.sum(1, keepdims=True)           # each image's softmax over the batch's captions
    P2 = np.exp(S - S.max(0, keepdims=True)); P2 /= P2.sum(0, keepdims=True)           # each caption's softmax over the batch's images
    loss = float(-(np.log(P1[I, I] + 1e-30).mean() + np.log(P2[I, I] + 1e-30).mean()) / 2)  # the matching pair should win both ways
    G1 = P1.copy(); G1[I, I] -= 1; G2 = P2.copy(); G2[I, I] -= 1; dS = (G1 + G2) / (2 * n)  # softmax gradients: probability minus target
    dU = dS @ V / tau; dV = dS.T @ U / tau                                            # back through S = U V^T / tau
    return loss, (dU - U * (U * dU).sum(1, keepdims=True)) / ni, (dV - V * (V * dV).sum(1, keepdims=True)) / nt  # through the normalisation (Day 131)


def train_clip(X, y, thick, lean, epochs=EPOCHS, hold=None, seed=0, batch=256, lr=1e-3, every=None):  # NEW: both encoders, trained together
    img = d126.init_net([784, 256, 64], seed); txt = d126.init_net([len(VOCAB), 256, 64], seed + 1)  # 64-number shared space
    oi, ot = d126.Adam(img, lr), d126.Adam(txt, lr); rng = np.random.RandomState(seed); log = []
    for ep in range(1, epochs + 1):
        o = rng.permutation(len(X)); L = []
        for s in range(0, len(X) - batch + 1, batch):                                 # full batches: every batch has 255 wrong captions per image
            b = o[s:s + batch]; caps = [caption(y[i], thick[i], lean[i], rng, hold) for i in b]  # a fresh random caption each time
            Ai = d126.forward(img, X[b], ACTS); At = d126.forward(txt, bow(caps), ACTS)
            loss, gi, gt = clip_loss(Ai[-1].astype(np.float64), At[-1].astype(np.float64))
            oi.step(img, d126.backward(img, Ai, ACTS, gi.astype(np.float32))[0])       # Day 126's backward, once per encoder
            ot.step(txt, d126.backward(txt, At, ACTS, gt.astype(np.float32))[0]); L.append(loss)
        log.append((ep, float(np.mean(L)), every(img, txt) if every else None))      # the epoch's average loss, and an optional check
    return img, txt, log


def embed_images(img, X):                               # NEW: images -> unit vectors in the shared space
    Z = d126.forward(img, X, ACTS)[-1].astype(np.float64); return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def embed_texts(txt, captions):                         # NEW: captions (lists of words) -> unit vectors in the shared space
    Z = d126.forward(txt, bow(captions), ACTS)[-1].astype(np.float64); return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def zero_shot(img, txt, X, y, templates=("the digit {}",)):  # NEW: name each image by its most similar prompt, averaged over templates
    T = sum(embed_texts(txt, [t.format(w).split() for w in NUM]) for t in templates)    # one text vector per digit (summed over templates)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return float(((embed_images(img, X) @ T.T).argmax(1) == y).mean())


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- the contrastive loss on three image-caption pairs")
    print("=" * 70)
    Zi = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]); Zt = np.array([[0.8, 0.6], [0.0, 1.0], [1.0, 0.0]])  # already unit length
    loss, gi, gt = clip_loss(Zi, Zt, 1.0); S = Zi @ Zt.T
    print("three images and three captions, unit vectors in 2-D; pair i is image i with caption i; temperature 1")
    print("similarities (rows images, columns captions):"); print("\n".join("   " + " ".join(f"{v:5.2f}" for v in row) for row in S))
    p1 = np.exp(S[0, 0]) / np.exp(S[0]).sum(); p2 = np.exp(S[0, 0]) / np.exp(S[:, 0]).sum()
    print(f"image 0 picks caption 0 with probability e^0.8 / (e^0.8 + e^0 + e^1) = {p1:.4f}; caption 0 picks image 0 with {p2:.4f}")
    by_hand = -(sum(np.log(np.exp(S[i, i]) / np.exp(S[i]).sum()) for i in range(3)) + sum(np.log(np.exp(S[i, i]) / np.exp(S[:, i]).sum()) for i in range(3))) / 6
    print(f"loss: code {loss:.6f}, by hand {by_hand:.6f} (the average of the six -log probabilities)")
    rng = np.random.RandomState(0); A = rng.randn(6, 4); B = rng.randn(6, 4); _, gA, gB = clip_loss(A, B); worst = 0.0
    for M, G in ((A, gA), (B, gB)):
        for idx in np.ndindex(M.shape):                                                # every entry, nudged both ways
            old = M[idx]; M[idx] = old + 1e-6; up = clip_loss(A, B)[0]; M[idx] = old - 1e-6; dn = clip_loss(A, B)[0]; M[idx] = old
            worst = max(worst, abs((up - dn) / 2e-6 - G[idx]))
    print(f"gradient against finite differences, all 48 entries of two random 6 x 4 batches, temperature {TAU}: largest gap {worst:.2e}")
    print("Actual output from this lesson's run.")


def demo_captions(X, y, ink, slant, thick, lean):
    print("\n" + "=" * 70)
    print("2. CAPTIONS FROM MEASUREMENTS -- no human wrote any of them")
    print("=" * 70)
    print("thickness: the average pixel; lean: how far the ink's column moves per row (positive: leans right); each cut into")
    print("thirds within its own digit, from the 60,000 training digits, so 'thick' means thick for that digit")
    print(f"{'digit':>6} {'ink: thin below':>16} {'thick above':>12} {'lean: left below':>17} {'right above':>12}")
    for d in range(10):
        k = y[:60000] == d; a = np.percentile(ink[:60000][k], [100 / 3, 200 / 3]); b = np.percentile(slant[:60000][k], [100 / 3, 200 / 3])
        print(f"{NUM[d]:>6} {a[0]:16.4f} {a[1]:12.4f} {b[0]:17.4f} {b[1]:12.4f}")
    rng = np.random.RandomState(3)
    print(f"vocabulary: {len(VOCAB)} words; six random captions for training digit 0 ({NUM[y[0]]}, thickness {thick[0]}, lean {lean[0]}):")
    print("   " + " | ".join(" ".join(caption(y[0], thick[0], lean[0], rng)) for _ in range(6)))
    print(f"share of training digits that are thick, by the measure: {np.mean(thick[:60000] == 1):.3f}; leaning right {np.mean(lean[:60000] == 1):.3f}")
    print("Actual output from this lesson's run.")


def demo_train(X, y, thick, lean, te):
    print("\n" + "=" * 70)
    print("3. TRAINING THE TWO ENCODERS")
    print("=" * 70)
    print(f"image encoder 784 -> 256 -> 64; text encoder {len(VOCAB)} (bag of words) -> 256 -> 64; both Day 126's networks and Adam (lr 0.001);")
    print(f"batches of 256 digits with a fresh random caption each; temperature {TAU}; {EPOCHS} epochs over the 60,000 training digits;")
    print("zero-shot accuracy after each epoch: 2,000 test digits named by the prompt 'the digit <word>'; a loss of log 256 = "
          f"{np.log(256):.3f} would mean no better than chance")
    t0 = time.perf_counter()
    img, txt, log = train_clip(X[:60000], y[:60000], thick[:60000], lean[:60000], every=lambda i, t: zero_shot(i, t, X[te], y[te]))
    print(f"{'epoch':>6} {'loss':>8} {'zero-shot accuracy':>19}")
    for ep, loss, acc in log:
        print(f"{ep:6d} {loss:8.4f} {acc:19.4f}")
    print(f"({time.perf_counter() - t0:.0f} s, including the checks)")
    Ei = embed_images(img, X[te]); caps = [caption(y[i], thick[i], lean[i], np.random.RandomState(i)) for i in te]; Et = embed_texts(txt, caps)
    S = Ei @ Et.T; r = (S.argmax(1) == np.arange(len(te))).mean(); same = (y[te][S.argmax(1)] == y[te]).mean()
    print(f"image -> caption among the same 2,000 test digits' captions: the exact caption ranked first {r:.4f} (chance 0.0005); "
          f"a caption of the right digit first {same:.4f}")
    print("Actual output from this lesson's run.")
    return img, txt, log


def demo_zero_shot(img, txt, X, y, te):
    print("\n" + "=" * 70)
    print("4. ZERO-SHOT CLASSIFICATION -- and what the prompt changes")
    print("=" * 70)
    print("2,000 test digits; each prompt is filled with the ten digit words and each image takes the most similar one.")
    print("For comparison, Day 138's trained classifier scored 0.9829 on the full test set.")
    print(f"{'prompt':>36} {'accuracy':>9}   words the encoder does not know")
    for t in ("{}", "a {}", "the digit {}", "handwritten {}", "a photo of the number {}", "a thick {}", "a thin {} leaning left",
              "{} upright and thick", "not a {}"):
        unk = [w for w in t.replace("{}", "").split() if w not in IDX]
        print(f"{repr(t):>36} {zero_shot(img, txt, X[te], y[te], (t,)):9.4f}   {' '.join(unk) if unk else '-'}")
    ens = ("{}", "a {}", "the digit {}", "handwritten {}", "written {}")
    print(f"{'ensemble of 5 plain prompts':>36} {zero_shot(img, txt, X[te], y[te], ens):9.4f}")
    a = embed_texts(txt, ["a thin seven leaning left".split()]); b = embed_texts(txt, ["left seven thin leaning a".split()])
    c = embed_texts(txt, ["a seven".split()]); d = embed_texts(txt, ["a seven".split() + ["purple", "elephant"]])
    print(f"word order: 'a thin seven leaning left' against 'left seven thin leaning a': cosine {float((a @ b.T)[0, 0]):.6f}")
    print(f"unknown words: 'a seven' against 'a seven purple elephant': cosine {float((c @ d.T)[0, 0]):.6f}")
    print("Actual output from this lesson's run.")


def retrieve(img, txt, X, words, pool, k=50):            # the k pool images most similar to a caption
    return pool[np.argsort(-(embed_images(img, X[pool]) @ embed_texts(txt, [words.split()])[0]))[:k]]


def demo_retrieval(img, txt, X, y, ink, slant, thick, lean):
    print("\n" + "=" * 70)
    print("5. FINDING DIGITS BY DESCRIPTION -- text -> image retrieval over the 10,000 test digits")
    print("=" * 70)
    print("top 50 for each query; 'right digit': share of them with that label; then the share whose measured attribute matches")
    print("the query (each attribute covers about a third of each digit, so 0.33 is what chance gives)")
    print(f"{'query':>26} {'right digit':>12} {'attribute matches':>18} {'mean ink':>9} {'mean lean':>10}")
    pool = np.arange(60000, 70000); rows = {}
    for q in ("three", "a thick three", "a thin three", "four leaning right", "four leaning left", "a thick one", "a thin eight upright"):
        top = retrieve(img, txt, X, q, pool); ws = q.split(); d = next(NUM.index(w) for w in ws if w in NUM)  # whole words: "upright" is not "right"
        att = thick if ("thick" in ws or "thin" in ws) else lean; want = 1 if ("thick" in ws or "right" in ws) else (-1 if ("thin" in ws or "left" in ws) else 0)
        match = "-" if q == "three" else f"{np.mean(att[top] == want):.2f}"
        rows[q] = top; print(f"{q:>26} {np.mean(y[top] == d):12.2f} {match:>18} {ink[top].mean():9.4f} {slant[top].mean():10.4f}")
    k3 = pool[y[pool] == 3]
    print(f"all test threes: mean ink {ink[k3].mean():.4f}, mean lean {slant[k3].mean():.4f}")
    print("Actual output from this lesson's run.")
    return rows


def demo_held_out(X, y, ink, thick, lean, te, img, txt):
    print("\n" + "=" * 70)
    print("6. A COMBINATION NEVER SEEN IN TRAINING -- 'thick' about a seven")
    print("=" * 70)
    print("a second pair of encoders, trained exactly as in section 3 except that a thick seven's caption never says 'thick'")
    print("(it still says 'seven', and other digits are still called thick). Section 3's encoders, which saw the combination,")
    sev = np.arange(60000, 70000)[y[60000:] == 7]
    print(f"are the control. Queries rank the {len(sev):,} test sevens; thick share 0.33 is chance")
    i2, t2, _ = train_clip(X[:60000], y[:60000], thick[:60000], lean[:60000], hold=HOLD)
    n_hold = sum(1 for i in range(60000) if y[i] == 7 and thick[i] == 1)
    print(f"(training sevens measured thick, never called thick: {n_hold}; zero-shot accuracy of the held-out encoders "
          f"{zero_shot(i2, t2, X[te], y[te]):.4f})")
    print(f"{'encoders':>14} {'query':>16} {'thick share, top 50':>20} {'mean ink, top 50':>17}")
    for name, a, b in (("saw it", img, txt), ("held out", i2, t2)):
        for q in ("a thick seven", "a thin seven", "a thick three"):
            pool = sev if "seven" in q else np.arange(60000, 70000)[y[60000:] == 3]; top = retrieve(a, b, X, q, pool)
            print(f"{name:>14} {q:>16} {np.mean(thick[top] == (1 if 'thick' in q else -1)):20.2f} {ink[top].mean():17.4f}")
    print(f"all test sevens: mean ink {ink[sev].mean():.4f}")
    print("Actual output from this lesson's run.")
    return i2, t2


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(img, txt, X, y, log, rows, out):        # six panels
    fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0]); ep = [r[0] for r in log]
    ax.plot(ep, [r[1] for r in log], marker="o", label="contrastive loss"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax2 = ax.twinx(); ax2.plot(ep, [r[2] for r in log], marker="s", color="#C44E52", label="zero-shot accuracy"); ax2.set_ylabel("zero-shot accuracy")
    ax.set_title("training the two encoders", fontsize=10); ax.legend(loc="center right", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    ax = fig.add_subplot(gs[0, 1]); pool = np.arange(60000, 62000); E = embed_images(img, X[pool])  # the shared space, flattened by PCA
    T = embed_texts(txt, [[w] for w in NUM]); mu = E.mean(0); _, _, Vt = np.linalg.svd(E - mu, full_matrices=False)
    P = (E - mu) @ Vt[:2].T; Q = (T - mu) @ Vt[:2].T
    ax.scatter(P[:, 0], P[:, 1], c=y[pool], cmap="tab10", s=4, alpha=0.5)
    for d in range(10):
        ax.scatter(*Q[d], marker="*", s=180, color=plt.cm.tab10(d), edgecolor="black"); ax.annotate(NUM[d], Q[d], fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("the shared space (2 of 64 directions): images (dots),\nthe ten digit words (stars)", fontsize=10)

    def strip(ax, I, title):                                                          # 8 images in one row
        ax.imshow(np.hstack([X[i].reshape(28, 28) for i in I[:8]]), cmap="gray_r", vmin=0, vmax=1); ax.set_title(title, fontsize=10); ax.axis("off")
    sub = gs[0, 2:4].subgridspec(4, 1); sub2 = gs[1, 0:2].subgridspec(4, 1)
    for j, q in enumerate(("a thick three", "a thin three", "four leaning right", "four leaning left")):
        strip(fig.add_subplot(sub[j]), rows[q], f"top 8 for '{q}'")
    for j, (q, I) in enumerate(rows["held"].items()):
        strip(fig.add_subplot(sub2[j]), I, q)
    ax = fig.add_subplot(gs[1, 2:4]); sev = np.arange(60000, 70000)[y[60000:] == 7]
    ax.hist(X[sev].mean(1), bins=40, color="lightgray", label="all test sevens")
    for q, c in (("held out: a thick seven", "#C44E52"), ("held out: a thin seven", "#4C72B0")):
        ax.hist(X[rows["held"][q]].mean(1), bins=15, alpha=0.7, color=c, label=f"top 50, {q}")
    ax.set_xlabel("ink (average pixel)"); ax.set_ylabel("sevens"); ax.set_title("a combination never seen in training", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the two encoders and the loss
    fig, ax = plt.subplots(figsize=(13, 4.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.3, 3.1, 2.0, 0.8, "a digit image\n784 pixels", "#E2F0D9"); box(4.2, 3.1, 2.4, 0.8, "image encoder\n784 -> 256 -> 64", "#DDEBF7")
    box(1.3, 1.1, 2.0, 0.8, "'a thick seven\nleaning right'", "#FFF2CC"); box(4.2, 1.1, 2.4, 0.8, "bag of words (22)\n-> 256 -> 64", "#FCE4D6")
    box(7.3, 3.1, 1.6, 0.8, "unit vector u", "#EEEEEE"); box(7.3, 1.1, 1.6, 0.8, "unit vector v", "#EEEEEE")
    box(10.6, 2.1, 3.6, 1.4, "S = U V^T / tau  (256 x 256 per batch)\nloss: each image picks its caption,\nand each caption its image\n(softmax both ways)", "#F4CCCC")
    arrow(2.3, 3.1, 3.0, 3.1); arrow(5.4, 3.1, 6.5, 3.1); arrow(2.3, 1.1, 3.0, 1.1); arrow(5.4, 1.1, 6.5, 1.1)
    arrow(8.1, 3.1, 8.8, 2.5); arrow(8.1, 1.1, 8.8, 1.7)
    ax.set_title("Day 143: a CLIP-style pair of encoders, trained so matching images and captions land close together", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "clip_joint_embeddings_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("The text encoder now turns a caption into 64 numbers that line up with images. Day 144 feeds those numbers into")
    print("Day 141's diffusion model, so a prompt chooses what it draws, and experiments with the prompts themselves.")


def main():                                             # runs every section in order
    X, y = d126m.mnist()                                                              # Day 126's 70,000 digits, pixels in [0, 1]
    ink, slant, thick, lean = measure(X, y)                                           # the measured attributes
    te = 60000 + np.random.RandomState(0).permutation(10000)[:2000]                    # 2,000 test digits, in random order (the stored order is not)
    demo_hand()                                                                       # 1. the loss by hand
    demo_captions(X, y, ink, slant, thick, lean)                                      # 2. captions
    img, txt, log = demo_train(X, y, thick, lean, te)                                 # 3. training
    demo_zero_shot(img, txt, X, y, te)                                                # 4. zero-shot
    rows = demo_retrieval(img, txt, X, y, ink, slant, thick, lean)                    # 5. retrieval
    i2, t2 = demo_held_out(X, y, ink, thick, lean, te, img, txt)                      # 6. the held-out combination
    sev = np.arange(60000, 70000)[y[60000:] == 7]
    rows["held"] = {f"held out: {q}": retrieve(i2, t2, X, q, sev) for q in ("a thick seven", "a thin seven")}
    rows["held"].update({f"saw it: {q}": retrieve(img, txt, X, q, sev) for q in ("a thick seven", "a thin seven")})
    plot_results(img, txt, X, y, log, rows, os.path.join(HERE, "clip_joint_embeddings_results.png"))
    draw_diagram()                                                                    # the two encoders
    print("\nsaved clip_joint_embeddings_results.png\nsaved clip_joint_embeddings_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
