"""
Day 144: Prompt Engineering for Generative Models -- Real Experimentation

Day 143 trained a text encoder that turns a short caption into 64 numbers.
Today those numbers steer Day 141's diffusion model: the network gets the
caption's embedding as an extra input, so a prompt chooses what it draws. The
standard way to make it listen is classifier-free guidance (Ho and Salimans,
2022): train one network both with and without the caption, and at sampling
time push its guess away from the no-caption guess, towards the caption.

Everything is written from scratch in NumPy, on Day 126's networks and Adam,
Day 138's scorer, Day 140's schedule, Day 141's model, Day 142's averaged
weights and DDIM, and Day 143's encoders, all imported unchanged:
  * a diffusion model that takes a caption embedding, trained with the
    caption dropped 15% of the time
  * classifier-free guidance, and negative prompts
  * prompts varied one change at a time, with the drawings measured

  1. By hand: guidance in the x_0 guess is guidance in the noise guess.
  2. Training the text-conditioned model.
  3. The guidance scale: following the prompt against quality and variety.
  4. Paraphrases and words the model does not know.
  5. Attributes, and a combination never seen in training.
  6. Negative prompts.
  7. Contradictory prompts.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier parts
import time                                            # time: how long training and sampling take

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts (Days 140-143)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day139_gan_build_and_forward_diffusion as d139  # Day 139: train_classifier, Scorer (Day 138's)
    import day140_exercise as d140                      # Day 140: ABAR, time_embed
    import day141_exercise as d141                      # Day 141: to_pixels
    import day142_exercise as d142                      # Day 142: ema_update, ddim_sample, grey_share
    import day143_exercise as d143                      # Day 143: measure, caption, train_clip, embed_texts, NUM

ABAR = d140.ABAR                                        # Day 139's linear schedule: alpha-bar after each of the 1,000 steps
ACTS = ["relu", "relu", "linear"]                       # two hidden ReLU layers and a linear output, as on Day 141
SIZES = [784 + 32 + 64, 512, 512, 784]                  # x_t, the time embedding and the caption embedding in; a digit out
STEPS = 20000                                           # training steps (batches of 128)
DROP = 0.15                                             # the share of training captions replaced by 'no caption' (all zeros)
HOLD = (7, 1)                                           # Day 143's held-out pair: a seven is never called thick, in either model's training
SAMPLE_STEPS = 100                                      # DDIM steps with eta 0 for every sample here (Day 142: FID 45.8 against 38.1 for 1,000)
W_DEFAULT = 2.0                                         # the guidance scale for sections 4-7, chosen from section 3's sweep
N = 200                                                 # samples per prompt


# =============================================================================
# THE PARTS -- NEW
# =============================================================================
def guess_c(net, xt, t, c):                             # NEW: the x_0 guess, given x_t, the step t and a caption embedding c
    A = d126.forward(net, np.concatenate([xt, d140.time_embed(t), c], 1).astype(np.float32), ACTS)  # 784 + 32 + 64 inputs per row
    return A, A[-1].astype(np.float64)                                                # every layer's output (for backward), and the guess


def train_cond(X2, y, thick, lean, txt, steps=STEPS, drop=DROP, hold=HOLD, seed=0, batch=128, lr=1e-3, every=2000):  # NEW
    net = d126.init_net(SIZES, seed); net[-1] = (net[-1][0] * 0, net[-1][1])          # Day 126's network, output layer started at zero (Day 141)
    opt = d126.Adam(net, lr); ema = [(W.copy(), b.copy()) for W, b in net]            # Adam, and Day 142's averaged copy
    rng = np.random.RandomState(seed); log = []; run = 0.0
    for s in range(1, steps + 1):
        i = rng.randint(0, len(X2), batch); x0 = X2[i].astype(np.float64); t = rng.randint(1, len(ABAR), batch)  # random digits, a random t each
        e = rng.standard_normal(x0.shape); xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * e  # Day 139's shortcut
        c = d143.embed_texts(txt, [d143.caption(y[k], thick[k], lean[k], rng, hold) for k in i])  # a fresh random caption for each, embedded
        c[rng.rand(batch) < drop] = 0.0                                               # some captions dropped: the same network learns 'no caption'
        A, x0h = guess_c(net, xt, t, c); miss = x0h - x0                              # Day 141's x_0 loss
        grads, _ = d126.backward(net, A, ACTS, (miss * 2 / miss.size).astype(np.float32))
        opt.step(net, grads); d142.ema_update(ema, net); run += (miss ** 2).mean()     # one Adam step, then the averaged copy follows
        if s % every == 0:
            log.append((s, run / every)); run = 0.0                                   # the average loss over the last `every` steps
    return net, ema, log


def cfg_sample(net, cpos, n, w, cneg=None, steps=SAMPLE_STEPS, seed=1):  # NEW: classifier-free guidance inside Day 142's DDIM (eta 0)
    cp = np.repeat(cpos[None], n, 0)                                                  # the prompt's embedding, one row per sample
    cn = np.zeros_like(cp) if cneg is None else np.repeat(cneg[None], n, 0)           # 'no caption', or a negative prompt in its place

    def x0_fn(x, t):                                                                  # the guided x_0 guess at step t
        tt = np.full(len(x), t); a = guess_c(net, x, tt, cp)[1]                        # with the prompt
        if w == 1.0 and cneg is None:
            return a                                                                  # w = 1 is plain conditioning: one network call is enough
        b = guess_c(net, x, tt, cn)[1]                                                # without it (or with the negative prompt)
        return b + w * (a - b)                                                        # w = 0: ignore the prompt; w > 1: push past it
    return d141.to_pixels(d142.ddim_sample(x0_fn, n, np.random.RandomState(seed), steps, 0.0))


def ink_lean(F):                                        # NEW: Day 143's two measurements (same formulas), for any batch of images
    R = np.repeat(np.arange(28.0), 28)[None]; Cc = np.tile(np.arange(28.0), 28)[None]  # each pixel's row and column
    m = F.sum(1, keepdims=True) + 1e-9; mr = (F * R).sum(1, keepdims=True) / m; mc = (F * Cc).sum(1, keepdims=True) / m  # the ink's centre
    return F.mean(1), -(F * (R - mr) * (Cc - mc)).sum(1) / (F * (R - mr) ** 2).sum(1)  # thickness (average pixel) and lean


def said(sc, F):                                        # NEW: the digit Day 138's classifier sees in each image
    return np.concatenate([sc.head.forward(sc.feats(F[i:i + 1000])).argmax(1) for i in range(0, len(F), 1000)])


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- guidance in the x_0 guess is guidance in the noise guess")
    print("=" * 70)
    rng = np.random.RandomState(0); t = 400; xt = rng.standard_normal((5, 784))       # any x_t, any step
    a, b = np.tanh(rng.standard_normal((5, 784))), np.tanh(rng.standard_normal((5, 784)))  # two made-up x_0 guesses: with and without the prompt
    eps = lambda x0: (xt - np.sqrt(ABAR[t]) * x0) / np.sqrt(1 - ABAR[t])               # the noise each x_0 guess implies (Day 140)
    print("eps(x_0) = (x_t - sqrt(abar_t) x_0) / sqrt(1 - abar_t) is a straight-line function of x_0, so mixing guesses commutes with it:")
    for w in (0.0, 1.0, 3.0):
        g = b + w * (a - b)                                                           # guidance applied to the x_0 guesses
        print(f"  w = {w}: largest difference between eps(guided x_0) and guided eps: {np.abs(eps(g) - (eps(b) + w * (eps(a) - eps(b)))).max():.2e}")
    print("w = 0 returns the no-caption guess exactly, w = 1 the captioned guess exactly; a negative prompt simply takes the no-caption guess's place")
    print("Actual output from this lesson's run.")


def demo_train(X, y, X2, thick, lean):
    print("\n" + "=" * 70)
    print("2. TRAINING THE TEXT-CONDITIONED MODEL")
    print("=" * 70)
    t0 = time.perf_counter(); img, txt, _ = d143.train_clip(X[:60000], y[:60000], thick[:60000], lean[:60000], hold=HOLD)  # Day 143's encoders
    print(f"Day 143's encoders, trained with thick sevens never called thick: {time.perf_counter() - t0:.0f} s")
    print(f"diffusion model: Day 141's network with 64 more inputs ({SIZES[0]} -> 512 -> 512 -> 784), output read as x_0, output layer started at zero;")
    print(f"each training digit gets a fresh random caption (Day 143's, the same held-out pair), embedded by the text encoder, which is frozen;")
    print(f"{DROP:.0%} of captions replaced by zeros ('no caption'); Adam lr 0.001, batches of 128, {STEPS:,} steps, averaged copy decay 0.999")
    t0 = time.perf_counter(); net, ema, log = train_cond(X2[:60000], y[:60000], thick[:60000], lean[:60000], txt); took = time.perf_counter() - t0
    print(f"{'step':>8} {'loss':>8}")
    for s, l in log:
        print(f"{s:8,d} {l:8.4f}")
    print(f"({took:.0f} s, including the text encoder embedding every caption)")
    print("Actual output from this lesson's run.")
    return txt, ema, log


def demo_scale(sc, txt, ema, X):
    print("\n" + "=" * 70)
    print("3. THE GUIDANCE SCALE -- following the prompt against quality and variety")
    print("=" * 70)
    print(f"prompts 'zero' ... 'nine', {N} samples each (2,000 in all), DDIM {SAMPLE_STEPS} steps; 'right digit': the share Day 138's classifier")
    print("names as the prompted digit (0.10 is chance); FID, precision and recall against real training digits; ink: the average pixel")
    print(f"{'w':>5} {'right digit':>12} {'IS':>6} {'FID':>8} {'precision':>10} {'recall':>7} {'grey share':>11} {'ink':>6} {'time':>6}")
    yy = np.repeat(np.arange(10), N); out = []; keep = {}
    for w in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0):
        t0 = time.perf_counter(); F = np.vstack([cfg_sample(ema, d143.embed_texts(txt, [[word]])[0], N, w, seed=1 + k) for k, word in enumerate(d143.NUM)])
        took = time.perf_counter() - t0; i, fid, p, r, _ = sc(F); acc = float(np.mean(said(sc, F) == yy))
        out.append((w, acc, fid, r)); keep[w] = F[::N // 4][:40]                      # 4 samples of each digit, for the figure
        print(f"{w:5.1f} {acc:12.3f} {i:6.2f} {fid:8.2f} {p:10.3f} {r:7.3f} {d142.grey_share(F):11.4f} {F.mean():6.3f} {took:5.0f}s")
    T = X[np.random.RandomState(0).permutation(10000)[:2000] + 60000]                  # 2,000 real test digits in random order, for comparison
    print(f"{'real':>5} {'':>12} {'':>6} {'':>8} {'':>10} {'':>7} {d142.grey_share(T):11.4f} {T.mean():6.3f}   (2,000 real test digits)")
    print("Actual output from this lesson's run.")
    return out, keep


def run_prompt(sc, txt, ema, prompt, w=W_DEFAULT, neg=None, want=None):  # one prompt, N samples, measured
    F = cfg_sample(ema, d143.embed_texts(txt, [prompt.split()])[0], N, w, None if neg is None else d143.embed_texts(txt, [neg.split()])[0])
    lab = said(sc, F); ink, lean = ink_lean(F); cnt = np.bincount(lab, minlength=10); top = np.argsort(-cnt)[:2]
    right = f"{np.mean(lab == want):.3f}" if want is not None else "-"
    print(f"{repr(prompt):>30} {repr(neg) if neg else '-':>16} {right:>7} {ink.mean():7.4f} {lean.mean():7.4f} "
          f"{d142.grey_share(F):7.4f}   {d143.NUM[top[0]]} {cnt[top[0]]}, {d143.NUM[top[1]]} {cnt[top[1]]}")
    return F


def header():
    print(f"{N} samples per prompt from the same starting noise (seed 1), guidance w = {W_DEFAULT}; 'right': the share named as the intended digit;")
    print("ink and lean: Day 143's measurements, averaged; classified as: the two digits Day 138's classifier names most often")
    print(f"{'prompt':>30} {'negative prompt':>16} {'right':>7} {'ink':>7} {'lean':>7} {'grey':>7}   classified as")


def demo_paraphrase(sc, txt, ema, X, y):
    print("\n" + "=" * 70)
    print("4. PARAPHRASES, AND WORDS THE MODEL DOES NOT KNOW")
    print("=" * 70)
    header(); keep = {}
    for p in ("seven", "a seven", "the digit seven", "handwritten seven", "seven written a", "a photo of the number seven", "seven purple elephant", "7", "sept"):
        keep[p] = run_prompt(sc, txt, ema, p, want=7)[:8]
    e7, e0 = d143.embed_texts(txt, [["7"], []])                                        # '7' and an empty caption
    print(f"'7' against an empty caption: cosine {float(e7 @ e0):.6f}; an empty caption against 'no caption' (zeros, used in training): "
          f"{'different' if np.abs(e0).max() > 0 else 'the same'} (largest entry {np.abs(e0).max():.3f})")
    k = y[:60000] == 7; ink, lean = ink_lean(X[:60000][k])
    print(f"real training sevens: ink {ink.mean():.4f}, lean {lean.mean():.4f}")
    print("Actual output from this lesson's run.")
    return keep


def demo_attributes(sc, txt, ema, X, y):
    print("\n" + "=" * 70)
    print("5. ATTRIBUTES, AND A COMBINATION NEVER SEEN IN TRAINING")
    print("=" * 70)
    header(); keep = {}
    for p, d in (("three", 3), ("a thick three", 3), ("a thin three", 3), ("four", 4), ("four leaning right", 4), ("four leaning left", 4),
                 ("four upright", 4), ("seven", 7), ("a thick seven", 7), ("a thin seven", 7)):
        keep[p] = run_prompt(sc, txt, ema, p, want=d)[:8]
    for d in (3, 4, 7):
        ink, lean = ink_lean(X[:60000][y[:60000] == d])
        print(f"real training {d143.NUM[d]}s: ink {ink.mean():.4f}, lean {lean.mean():.4f}")
    print("'a thick seven': neither model's training captions ever put 'thick' with 'seven'")
    print("Actual output from this lesson's run.")
    return keep


def demo_negative(sc, txt, ema):
    print("\n" + "=" * 70)
    print("6. NEGATIVE PROMPTS -- guide away from a second prompt instead of from 'no caption'")
    print("=" * 70)
    header(); keep = {}
    for p, n, d in (("three", None, 3), ("three", "thick", 3), ("three", "thin", 3), ("four", "leaning right", 4), ("four", "leaning left", 4),
                    ("three", "eight", 3), ("one", "seven", 1)):
        keep[(p, n)] = run_prompt(sc, txt, ema, p, neg=n, want=d)[:8]
    print("Actual output from this lesson's run.")
    return keep


def demo_contradiction(sc, txt, ema):
    print("\n" + "=" * 70)
    print("7. CONTRADICTORY PROMPTS")
    print("=" * 70)
    header(); keep = {}
    for p in ("three seven", "three eight", "one zero", "thick thin seven", "leaning left right four"):
        keep[p] = run_prompt(sc, txt, ema, p)[:8]
    print("Actual output from this lesson's run.")
    return keep


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(scale, keep_w, keep, log, out):        # the sweep, and samples for each kind of prompt
    fig = plt.figure(figsize=(17, 15)); gs = fig.add_gridspec(5, 4, height_ratios=[1.6, 1, 1, 1, 1])  # the sweep, then 16 prompt panels

    def grid(ax, F, r, c, title):                                                     # r x c images tiled into one picture
        ax.imshow(np.vstack([np.hstack([F[i * c + j].reshape(28, 28) for j in range(c)]) for i in range(r)]), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[0, 0]); ax.plot([s for s, _ in log], [l for _, l in log], marker="o")
    ax.set_xlabel("training step"); ax.set_xticks([5000, 10000, 15000, 20000]); ax.set_ylabel("loss (x_0 squared error)"); ax.set_title("training", fontsize=10)
    ax = fig.add_subplot(gs[0, 1]); w = [r[0] for r in scale]
    ax.plot(w, [r[1] for r in scale], marker="o", label="right digit"); ax.plot(w, [r[3] for r in scale], marker="s", label="recall")
    ax.set_xlabel("guidance scale w"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc="center right"); ax.set_title("following the prompt, and variety", fontsize=10)
    ax2 = ax.twinx(); ax2.plot(w, [r[2] for r in scale], marker="^", color="#C44E52"); ax2.set_ylabel("FID (red triangles)", color="#C44E52")
    for j, ww in enumerate((0.0, 2.0)):
        grid(fig.add_subplot(gs[0, 2 + j]), keep_w[ww], 4, 10, f"'zero' ... 'nine', four of each in order, w = {ww:.0f}")
    panels = [("seven", "'seven'"), ("a photo of the number seven", "'a photo of the number seven'"), ("7", "'7' (unknown word)"),
              ("seven written a", "'seven written a'"),
              ("a thick three", "'a thick three'"), ("a thin three", "'a thin three'"), ("four leaning right", "'four leaning right'"),
              ("four leaning left", "'four leaning left'"),
              ("a thick seven", "'a thick seven' (never in training)"), (("three", "thick"), "'three', negative 'thick'"),
              (("three", "thin"), "'three', negative 'thin'"), (("four", "leaning right"), "'four', negative 'leaning right'"),
              ("three seven", "'three seven'"), ("one zero", "'one zero'"), ("thick thin seven", "'thick thin seven'"), ("a thin seven", "'a thin seven'")]
    for k, (key, title) in enumerate(panels):
        grid(fig.add_subplot(gs[1 + k // 4, k % 4]), keep[key], 2, 4, title)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # one network, called twice per step
    fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.2, 3.4, 1.9, 0.8, "prompt\n'a thick three'", "#FFF2CC"); box(3.6, 3.4, 2.0, 0.8, "Day 143's text\nencoder (frozen)", "#FCE4D6")
    box(1.2, 2.3, 1.9, 0.8, "x_t and step t", "#E2EFDA"); box(3.6, 1.2, 2.0, 0.9, "no caption\n(zeros), or a\nnegative prompt", "#EEEEEE", fs=8)
    box(6.5, 3.4, 2.1, 0.8, "network: guess a\n(with the prompt)", "#DDEBF7"); box(6.5, 1.2, 2.1, 0.8, "network: guess b\n(without it)", "#DDEBF7")
    box(9.6, 2.3, 2.4, 0.95, "guided x_0 guess\nb + w (a - b)", "#F4CCCC"); box(12.0, 2.3, 1.5, 0.8, "DDIM step\n(Day 142)", "#E2F0D9")
    arrow(2.15, 3.4, 2.6, 3.4); arrow(4.6, 3.4, 5.45, 3.4); arrow(4.6, 1.2, 5.45, 1.2); arrow(2.15, 2.4, 5.45, 3.05); arrow(2.15, 2.2, 5.45, 1.55)
    arrow(7.55, 3.3, 8.4, 2.55); arrow(7.55, 1.3, 8.4, 2.05); arrow(10.8, 2.3, 11.25, 2.3)
    ax.text(6.5, 0.3, f"training: the same network sees real captions and, {DROP:.0%} of the time, zeros, so it can make both guesses", ha="center", fontsize=8.5)
    ax.set_title("Day 144: classifier-free guidance -- the same network asked twice, and the answers pushed apart", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "prompt_engineering_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Guidance made the model follow its prompts, and also changed how thick and how varied its digits are.")
    print("Day 145 looks at skew like that deliberately (bias), at telling generated images from real ones (deepfakes), and at")
    print("copying (memorisation), with real case studies.")


def main():                                             # runs every section in order
    X, y = d126m.mnist(); X2 = (X * 2 - 1).astype(np.float32)                         # Day 126's 70,000 digits; [-1, 1] for the diffusion model
    _, _, thick, lean = d143.measure(X, y)                                            # Day 143's measured attributes, for the captions
    demo_hand()                                                                       # 1. guidance by hand
    t0 = time.perf_counter(); body, head = d139.train_classifier(X[:60000], y[:60000])  # Day 138's feature network, for every score below
    sc = d139.Scorer(body, head, X[:10000], X[10000:30000]); print(f"\n(feature network for the scores: {time.perf_counter() - t0:.0f} s)")
    txt, ema, log = demo_train(X, y, X2, thick, lean)                                 # 2. training
    scale, keep_w = demo_scale(sc, txt, ema, X)                                       # 3. the guidance scale
    keep = demo_paraphrase(sc, txt, ema, X, y)                                        # 4. paraphrases
    keep.update(demo_attributes(sc, txt, ema, X, y))                                  # 5. attributes
    keep.update(demo_negative(sc, txt, ema))                                          # 6. negative prompts
    keep.update(demo_contradiction(sc, txt, ema))                                     # 7. contradictions
    plot_results(scale, keep_w, keep, log, os.path.join(HERE, "prompt_engineering_results.png"))
    draw_diagram()                                                                    # classifier-free guidance
    print("\nsaved prompt_engineering_results.png\nsaved prompt_engineering_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
