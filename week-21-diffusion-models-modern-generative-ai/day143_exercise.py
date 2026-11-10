"""
Day 143 Exercise -- A Shared Space for Digits and Words

The fourth of Week 21's parts for Day 146's capstone.

    measure(X, y)                    each digit's thickness and lean, cut into thirds within its own digit
    caption(d, th, ln, rng, hold)    a random short caption for one digit ("seven thick leaning left a")
    bow(captions)                    captions as averaged word counts over the 22-word vocabulary
    clip_loss(Zi, Zt, tau)           CLIP's symmetric contrastive loss, with its gradient for both encoders
    train_clip(X, y, thick, lean)    an image encoder and a text encoder trained together
    embed_images, embed_texts        either kind of input -> a unit vector in the shared 64-number space
    zero_shot(img, txt, X, y)        name each image by the most similar prompt

Built on Day 126's network code and Adam, imported unchanged.
Checked by hand: the loss's gradient against finite differences, the loss of
random and perfectly matched embeddings, one caption's bag of words, the lean of
a made-up slanted stroke, and one epoch of training on 10,000 digits.
"""
import os                                              # os: the paths to earlier weeks' folders
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # as in the lesson
import sys                                             # sys: path, to import them
import numpy as np                                    # numpy: every array/matrix operation below is built on this
HERE = os.path.dirname(os.path.abspath(__file__))       # this week's folder
for w in ("week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
import contextlib                                      # contextlib: redirect_stdout, to keep their checks quiet
import io                                              # io: StringIO, where that output goes
with contextlib.redirect_stdout(io.StringIO()):
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
NUM = "zero one two three four five six seven eight nine".split()                      # as in the lesson
VOCAB = NUM + "a the digit handwritten written thin thick leaning left right upright and".split()
IDX = {w: i for i, w in enumerate(VOCAB)}
ACTS = ["relu", "linear"]
TAU = 0.1
EPOCHS = 10


def measure(X, y, n_train=60000):                       # NEW: Day 143's version, unchanged
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


def caption(d, th, ln, rng, hold=None):                  # NEW: Day 143's version, unchanged
    w = [NUM[d]]
    if th != 0 and rng.rand() < 0.6 and (d, th) != hold:                               # mention the thickness 60% of the time (never for the held-out pair)
        w.append("thick" if th > 0 else "thin")
    if ln != 0 and rng.rand() < 0.6:                                                  # mention a clear lean 60% of the time
        w += ["leaning", "right" if ln > 0 else "left"]
    elif ln == 0 and rng.rand() < 0.3:                                                # and an upright digit 30% of the time
        w.append("upright")
    return w + [["a"], ["the", "digit"], ["handwritten"], ["written"], []][rng.randint(5)]  # plus filler words; order never matters (bag of words)


def bow(captions):                                      # NEW: Day 143's version, unchanged
    B = np.zeros((len(captions), len(VOCAB)), np.float32)
    for r, ws in enumerate(captions):
        for w in ws:
            if w in IDX:
                B[r, IDX[w]] += 1
        B[r] /= max(B[r].sum(), 1)                                                    # an average of word vectors, not a sum
    return B


def clip_loss(Zi, Zt, tau=TAU):                         # NEW: Day 143's version, unchanged
    ni = np.linalg.norm(Zi, axis=1, keepdims=True) + 1e-12; nt = np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-12
    U, V = Zi / ni, Zt / nt; S = U @ V.T / tau; n = len(S); I = np.arange(n)            # unit vectors; S[i, j]: image i against caption j
    P1 = np.exp(S - S.max(1, keepdims=True)); P1 /= P1.sum(1, keepdims=True)           # each image's softmax over the batch's captions
    P2 = np.exp(S - S.max(0, keepdims=True)); P2 /= P2.sum(0, keepdims=True)           # each caption's softmax over the batch's images
    loss = float(-(np.log(P1[I, I] + 1e-30).mean() + np.log(P2[I, I] + 1e-30).mean()) / 2)  # the matching pair should win both ways
    G1 = P1.copy(); G1[I, I] -= 1; G2 = P2.copy(); G2[I, I] -= 1; dS = (G1 + G2) / (2 * n)  # softmax gradients: probability minus target
    dU = dS @ V / tau; dV = dS.T @ U / tau                                            # back through S = U V^T / tau
    return loss, (dU - U * (U * dU).sum(1, keepdims=True)) / ni, (dV - V * (V * dV).sum(1, keepdims=True)) / nt  # through the normalisation (Day 131)


def train_clip(X, y, thick, lean, epochs=EPOCHS, hold=None, seed=0, batch=256, lr=1e-3, every=None):  # NEW: Day 143's version, unchanged
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


def embed_images(img, X):                               # NEW: Day 143's version, unchanged
    Z = d126.forward(img, X, ACTS)[-1].astype(np.float64); return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def embed_texts(txt, captions):                         # NEW: Day 143's version, unchanged
    Z = d126.forward(txt, bow(captions), ACTS)[-1].astype(np.float64); return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def zero_shot(img, txt, X, y, templates=("the digit {}",)):  # NEW: Day 143's version, unchanged
    T = sum(embed_texts(txt, [t.format(w).split() for w in NUM]) for t in templates)    # one text vector per digit (summed over templates)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return float(((embed_images(img, X) @ T.T).argmax(1) == y).mean())


rng = np.random.RandomState(0); A = rng.randn(6, 4); B = rng.randn(6, 4); _, gA, gB = clip_loss(A, B, 0.5)
num = []
for M, G in ((A, gA), (B, gB)):
    Mp = M.copy(); Mp[2, 1] += 1e-6; Mm = M.copy(); Mm[2, 1] -= 1e-6
    fp = clip_loss(Mp, B, 0.5)[0] if M is A else clip_loss(A, Mp, 0.5)[0]; fm = clip_loss(Mm, B, 0.5)[0] if M is A else clip_loss(A, Mm, 0.5)[0]
    num.append((G[2, 1], (fp - fm) / 2e-6))
print(f"gradient check, image side: {num[0][0]:.8f} against {num[0][1]:.8f}; text side: {num[1][0]:.8f} against {num[1][1]:.8f}")
E = np.eye(8)                                                                           # 8 pairs, each embedding exactly its partner's
print(f"8 perfectly matched pairs, tau 0.1: loss {clip_loss(E, E, 0.1)[0]:.6f}; all embeddings equal: loss {clip_loss(np.ones((8, 3)), np.ones((8, 3)), 0.1)[0]:.4f} (log 8 = {np.log(8):.4f})")
b = bow([["seven", "thick", "seven", "zebra"]])[0]
print(f"bag of words of 'seven thick seven zebra': seven {b[IDX['seven']]:.3f}, thick {b[IDX['thick']]:.3f}, total {b.sum():.3f} ('zebra' is not in the vocabulary)")
Xs = np.zeros((2, 784))
for r in range(28):
    Xs[0, r * 28 + 14 + (13 - r) // 3] = 1.0                                             # a stroke moving right as it goes up
    Xs[1, r * 28 + 14 - (13 - r) // 3] = 1.0                                             # a stroke moving left as it goes up
_, sl, _, _ = measure(np.vstack([Xs] * 10), np.repeat(np.arange(10), 2), n_train=20)          # both strokes, labelled as each digit in turn
print(f"slant of a stroke leaning right: {sl[0]:.3f}; leaning left: {sl[1]:.3f} (about 1/3 of a column per row, either way)")
X, y = d126m.mnist(); ink, slant, thick, lean = measure(X, y)
img, txt, log = train_clip(X[:10000], y[:10000], thick[:10000], lean[:10000], epochs=1)
te = 60000 + np.random.RandomState(0).permutation(10000)[:2000]
print(f"one epoch on 10,000 digits: loss {log[0][1]:.4f} (log 256 = {np.log(256):.4f} for guessing); zero-shot accuracy on 2,000 test digits {zero_shot(img, txt, X[te], y[te]):.3f}")
