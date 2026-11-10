"""
Day 146: The Week 21 Build -- a Text-to-Digit Generator, End to End

The week's six exercise parts become one small text-to-image system: type a
prompt, get handwritten digits. Day 143's text encoder turns the prompt into
64 numbers, Day 144's text-conditioned diffusion model (Day 141's network,
Day 140's schedule) draws from it with classifier-free guidance, Day 142's
DDIM decides how many steps that takes, and Day 145's measurements check
what comes out.

Earlier build days (125, 132, 139) pasted their week's parts into one file.
This week's parts cannot be pasted: two of their constants clash, and one of
the clashes would break Day 143's encoders without a single error. So the
parts are imported, and section 1 measures what pasting would have done.

  1. The parts: imported, not pasted -- what pasting would break.
  2. The model, trained end to end, and checked against Days 143 and 144.
  3. The front end: prompts as the encoder can read them, and refusals.
  4. Speed: how many sampling steps a request needs.
  5. A model card: what this generator does to each digit.

Data: MNIST (cached by Day 126). Every line written for today carries its
own comment, continuing Day 52's convention.
"""
import ast                                             # ast: reading the exercise files' constants
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported parts quiet
import io                                              # io: StringIO, where that output goes
import re                                              # re: splitting prompts into words, and reading earlier days' outputs
import sys                                             # sys: path, to import earlier parts
import time                                            # time: how long each step takes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts (Days 140-145)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day139_gan_build_and_forward_diffusion as d139  # Day 139: train_classifier, Scorer (Day 138's)
    import day141_exercise as d141                      # Day 141: nearest_train
    import day142_exercise as d142                      # Day 142: grey_share
    import day143_exercise as d143                      # Day 143: measure, train_clip, bow, embed_texts, zero_shot, NUM, IDX
    import day144_exercise as d144                      # Day 144: train_cond, cfg_sample, ink_lean, said (Day 145's checks are rebuilt in section 5)

HOLD = (7, 1)                                           # Day 144's held-out pair: a seven is never called thick
N = 200                                                 # samples per prompt
W_GUIDE = 2.0                                           # Day 144's best guidance scale
STEP_LIST = (10, 25, 50, 100)                           # DDIM steps tried in section 4
NEAR = 2.0                                              # Day 145's 'near copy': this close (pixel distance) to a training digit


# =============================================================================
# THE PARTS -- NEW
# =============================================================================
def clashes(days=range(140, 146)):                      # NEW: constants given different values in more than one exercise file
    seen = {}
    for d in days:
        src = open(os.path.join(HERE, f"day{d}_exercise.py"), encoding="utf-8").read(); tree = ast.parse(src)
        last = max(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef))  # the parts end here; the checks below them are not parts
        for n in tree.body:                                                           # top-level constants: NAME = value, in capitals
            if isinstance(n, ast.Assign) and n.lineno < last and isinstance(n.targets[0], ast.Name) and n.targets[0].id.isupper():
                seen.setdefault(n.targets[0].id, []).append((d, ast.get_source_segment(src, n.value)))
    return {k: v for k, v in seen.items() if len({s for _, s in v}) > 1}             # only names whose values differ


def zero_shot_acts(img, txt, X, y, acts):               # NEW: Day 143's zero_shot, with the activation list passed in
    T = d126.forward(txt, d143.bow([f"the digit {w}".split() for w in d143.NUM]), acts)[-1].astype(np.float64)  # one text vector per digit
    Z = d126.forward(img, X, acts)[-1].astype(np.float64)                             # one image vector per digit image
    zero = float(np.mean(np.vstack([T, Z]) == 0))                                     # the share of embedding entries that are exactly 0
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12; Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12  # unit length
    return float(((Z @ T.T).argmax(1) == y).mean()), zero                             # accuracy, and the share of zeros


def normalise(prompt):                                  # NEW: a prompt as the encoder's words: lower case, numerals as words
    words = re.findall(r"[a-z0-9]+", prompt.lower())                                  # letters and digits only: 'Seven!' -> ['seven']
    words = [d143.NUM[int(w)] if len(w) == 1 and w.isdigit() else w for w in words]   # '7' -> 'seven'
    return [w for w in words if w in d143.IDX], [w for w in words if w not in d143.IDX]  # words the encoder knows, and the rest


def generate(txt, ema, prompt, n=N, w=W_GUIDE, negative=None, steps=100, seed=1):  # NEW: the front door: prompt in, digits (or a refusal) out
    known, unknown = normalise(prompt)
    digits = sorted({k for k in known if k in d143.NUM}, key=d143.NUM.index)          # the digit words in the prompt, once each
    if len(digits) != 1:                                                              # none, or two different digits: Day 144 drew eights or picked one
        return None, "refused: " + ("no digit word it knows" if not digits else "two digits (" + " and ".join(digits) + ")")
    c = d143.embed_texts(txt, [known])[0]                                             # the prompt's embedding
    cn = None if negative is None else d143.embed_texts(txt, [normalise(negative)[0]])[0]  # and the negative prompt's, if any
    note = "drawn" + (f"; ignored {', '.join(repr(u) for u in unknown)}" if unknown else "")  # say what was dropped
    return d144.cfg_sample(ema, c, n, w, cn, steps, seed), note


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_parts(img, txt, X, y, te):
    print("=" * 70)
    print("1. THE PARTS -- imported, not pasted: what pasting would break")
    print("=" * 70)
    print("constants given different values in this week's exercise files (in a single file, only the last would survive):")
    for k, v in clashes().items():
        print(f"  {k}: " + "; ".join(f"Day {d}: {s}" for d, s in v))
    right = d143.zero_shot(img, txt, X[te], y[te])                                    # Day 143's own function, its own ACTS
    acc, zero = zero_shot_acts(img, txt, X[te], y[te], d144.ACTS)                     # the same encoders run with Day 144's ACTS
    _, zero0 = zero_shot_acts(img, txt, X[te], y[te], d143.ACTS)                      # and with Day 143's, for the share of zeros
    print(f"Day 143's encoders (2 layers) run with Day 143's ACTS {d143.ACTS}: zero-shot accuracy {right:.4f}, "
          f"embedding entries exactly 0: {zero0:.3f}")
    print(f"the same encoders run with Day 144's ACTS {d144.ACTS}, as a pasted file would: zero-shot accuracy {acc:.4f}, "
          f"entries exactly 0: {zero:.3f}")
    print("no error either way: Day 126's forward pairs layers with activations using zip, which stops at the shorter list,")
    print("so the encoder's last layer quietly becomes a ReLU")
    print("Actual output from this lesson's run.")


def demo_model(sc, img, txt, X, y, X2, thick, lean, te):
    print("\n" + "=" * 70)
    print("2. THE MODEL -- trained end to end, and checked against Days 143 and 144")
    print("=" * 70)
    z = d143.zero_shot(img, txt, X[te], y[te])
    old = float(re.search(r"zero-shot accuracy of the held-out encoders (\d\.\d+)", open(os.path.join(HERE, "day143_run_output.txt"), encoding="utf-8").read()).group(1))
    print(f"text encoder (Day 143's train_clip, the held-out pair): zero-shot accuracy {z:.4f}; Day 143's run printed {old:.4f}: "
          f"{'the same' if f'{z:.4f}' == f'{old:.4f}' else 'DIFFERENT'}")
    t0 = time.perf_counter(); net, ema, log = d144.train_cond(X2[:60000], y[:60000], thick[:60000], lean[:60000], txt)  # Day 144's model
    print(f"text-conditioned diffusion model (Day 144's train_cond, {log[-1][0]:,} steps): loss {log[0][1]:.4f} -> {log[-1][1]:.4f}, "
          f"{time.perf_counter() - t0:.0f} s")
    yy = np.repeat(np.arange(10), N)                                                  # the digit each sample was asked to be
    t0 = time.perf_counter(); F = np.vstack([d144.cfg_sample(ema, d143.embed_texts(txt, [[w]])[0], N, W_GUIDE, seed=1 + k) for k, w in enumerate(d143.NUM)])
    took = time.perf_counter() - t0; i, fid, p, r, _ = sc(F); acc = float(np.mean(d144.said(sc, F) == yy))
    row = f"{W_GUIDE:5.1f} {acc:12.3f} {i:6.2f} {fid:8.2f} {p:10.3f} {r:7.3f} {d142.grey_share(F):11.4f} {F.mean():6.3f}"  # Day 144's row format
    old = next(ln for ln in open(os.path.join(HERE, "day144_run_output.txt"), encoding="utf-8").read().split("\n") if ln.startswith("  2.0 "))
    print(f"prompts 'zero' ... 'nine', {N} samples each, w = 2, DDIM 100 steps, the same starting noise as Day 144:")
    print(f"{'':>9}{'w':>5} {'right digit':>12} {'IS':>6} {'FID':>8} {'precision':>10} {'recall':>7} {'grey share':>11} {'ink':>6}")
    print(f"{'today':>9}{row}   ({took:.0f} s)")
    print(f"{'Day 144':>9}{old[:len(row)]}")
    print(f"every printed digit the same as Day 144's run: {row == old[:len(row)]}")
    print("Actual output from this lesson's run.")
    return net, ema, log, F, took


def demo_front(sc, txt, ema):
    print("\n" + "=" * 70)
    print("3. THE FRONT END -- prompts as the encoder can read them, and refusals")
    print("=" * 70)
    print(f"{N} samples per prompt, w = 2, the same starting noise (seed 1); 'raw': the prompt split on spaces and embedded as typed,")
    print("as on Day 144; 'front end': normalise, then refuse unless exactly one digit word is left; 'right': named as the intended digit")
    print(f"{'prompt':>24} {'raw: right':>11} {'raw: named as':>22}   {'front end':<40} {'right':>6}")
    tests = (("seven", 7), ("7", 7), ("Seven!", 7), ("SEVEN", 7), ("a thick 3", 3), ("4 leaning right", 4), ("draw me a 9 please", 9),
             ("sept", 7), ("a picture of a cat", None), ("three seven", None))
    keep = {}
    for prompt, d in tests:
        F = d144.cfg_sample(ema, d143.embed_texts(txt, [prompt.split()])[0], N, W_GUIDE); lab = d144.said(sc, F)  # raw, as Day 144 did
        cnt = np.bincount(lab, minlength=10); top = np.argsort(-cnt)[:2]                  # the two digits it was most often named as
        G, note = generate(txt, ema, prompt)                                          # through the front end
        rr = "-" if d is None else f"{np.mean(lab == d):.3f}"                         # no single intended digit: no 'right'
        fr = "-" if G is None or d is None else f"{np.mean(d144.said(sc, G) == d):.3f}"
        print(f"{prompt!r:>24} {rr:>11} {d143.NUM[top[0]] + ' ' + str(cnt[top[0]]) + ', ' + d143.NUM[top[1]] + ' ' + str(cnt[top[1]]):>22}   {note:<40} {fr:>6}")
        keep[prompt] = (F[:6], None if G is None else G[:6], note)
    print("Actual output from this lesson's run.")
    return keep


def demo_steps(sc, txt, ema, F100, t100):
    print("\n" + "=" * 70)
    print("4. SPEED -- how many sampling steps a request needs")
    print("=" * 70)
    print(f"prompts 'zero' ... 'nine', {N} samples each, w = 2 (two network calls per step), DDIM with no fresh noise, the same starting noise;")
    print("time for all 2,000 samples on this machine (timings vary from run to run here; compare within this table)")
    print(f"{'steps':>6} {'right digit':>12} {'FID':>8} {'recall':>7} {'grey share':>11} {'time':>7} {'ms per digit':>13}")
    yy = np.repeat(np.arange(10), N); rows = []; keep = {}
    for s in STEP_LIST:
        if s == 100:
            F, took = F100, t100                                                      # section 2's samples: the same settings
        else:
            t0 = time.perf_counter(); F = np.vstack([d144.cfg_sample(ema, d143.embed_texts(txt, [[w]])[0], N, W_GUIDE, steps=s, seed=1 + k) for k, w in enumerate(d143.NUM)])
            took = time.perf_counter() - t0
        _, fid, _, r, _ = sc(F); acc = float(np.mean(d144.said(sc, F) == yy)); rows.append((s, acc, fid, took)); keep[s] = F[::N // 4][:40]
        print(f"{s:6d} {acc:12.3f} {fid:8.2f} {r:7.3f} {d142.grey_share(F):11.4f} {took:6.1f}s {1000 * took / len(F):13.2f}")
    print("Actual output from this lesson's run.")
    return rows, keep


def demo_card(sc, F, X, y, ink, slant, te):
    print("\n" + "=" * 70)
    print("5. A MODEL CARD -- what this generator does to each digit")
    print("=" * 70)
    print(f"section 2's samples ({N} per prompt, w = 2); 'thick', 'thin', 'lean R' and 'lean L' use Day 143's cut points for that digit,")
    print("so real training digits score a third on each; ink: the average pixel; nearest: median pixel distance to the nearest")
    print(f"training digit, for the samples and for real test digits of the same kind; near copies: closer than {NEAR}")
    print("named: the share Day 138's classifier names as the prompted digit; lean R / lean L: leaning right / left")
    print(f"{'prompt':>7} {'named':>6} {'ink':>6} {'real':>6} {'thick':>6} {'thin':>6} {'lean R':>6} {'lean L':>6} {'nearest':>8} {'test':>6} {'near copies':>12}")
    ik, sl = d144.ink_lean(F); lab = d144.said(sc, F); dF, _ = d141.nearest_train(F, X[:60000])  # Day 144's measurements, Day 141's distance
    dT, _ = d141.nearest_train(X[te], X[:60000]); card = []                           # the same distance for 2,000 real test digits
    for d in range(10):
        k = slice(d * N, (d + 1) * N); tr = y[:60000] == d                            # this prompt's samples; this digit's training images
        a1, a2 = np.percentile(ink[:60000][tr], [100 / 3, 200 / 3]); b1, b2 = np.percentile(slant[:60000][tr], [100 / 3, 200 / 3])  # Day 143's cut points
        c = (np.mean(lab[k] == d), ik[k].mean(), ink[:60000][tr].mean(), np.mean(ik[k] > a2), np.mean(ik[k] < a1), np.mean(sl[k] > b2),
             np.mean(sl[k] < b1), np.median(dF[k]), np.median(dT[y[te] == d]), np.mean(dF[k] < NEAR)); card.append(c)
        print(f"{d143.NUM[d]:>7} {c[0]:6.3f} {c[1]:6.3f} {c[2]:6.3f} {c[3]:6.3f} {c[4]:6.3f} {c[5]:6.3f} {c[6]:6.3f} {c[7]:8.2f} {c[8]:6.2f} {c[9]:12.3f}")
    c = np.array(card); print(f"{'all':>7} {c[:, 0].mean():6.3f} {c[:, 1].mean():6.3f} {c[:, 2].mean():6.3f} {c[:, 3].mean():6.3f} {c[:, 4].mean():6.3f} "
                              f"{c[:, 5].mean():6.3f} {c[:, 6].mean():6.3f} {np.median(dF):8.2f} {np.median(dT):6.2f} {np.mean(dF < NEAR):12.3f}")
    print(f"closest sample to any training digit: {dF.min():.2f}; real test digits closer than {NEAR}: {np.mean(dT < NEAR):.3f}")
    print("Actual output from this lesson's run.")
    return c


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(log, rows, keep_s, keep_f, card, out):  # training, speed, the front end, the model card
    fig = plt.figure(figsize=(17, 12)); gs = fig.add_gridspec(4, 4, height_ratios=[1, 0.6, 0.6, 1])

    def grid(ax, F, r, c, title):                                                     # r x c images tiled into one picture
        ax.imshow(np.vstack([np.hstack([F[i * c + j].reshape(28, 28) for j in range(c)]) for i in range(r)]), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[0, 0]); ax.plot([s for s, _ in log], [l for _, l in log], marker="o")  # training loss
    ax.set_xlabel("training step"); ax.set_ylabel("loss (x_0 squared error)"); ax.set_title("training", fontsize=10)
    ax.set_xticks([5000, 10000, 15000, 20000])
    ax = fig.add_subplot(gs[0, 1]); s = [x[0] for x in rows]                          # speed against quality
    ax.plot(s, [x[2] for x in rows], marker="o", color="#C44E52"); ax.set_xscale("log"); ax.set_xticks(s); ax.set_xticklabels(s)
    ax.set_xlabel("DDIM steps"); ax.set_ylabel("FID (lower is better)", color="#C44E52")
    ax2 = ax.twinx(); ax2.plot(s, [x[3] for x in rows], marker="s", color="#4C72B0"); ax2.set_ylabel("seconds for 2,000 digits", color="#4C72B0")
    ax.set_title("speed against quality, w = 2", fontsize=10)
    grid(fig.add_subplot(gs[0, 2]), keep_s[STEP_LIST[0]], 4, 10, f"'zero' ... 'nine', {STEP_LIST[0]} steps")
    grid(fig.add_subplot(gs[0, 3]), keep_s[100], 4, 10, "'zero' ... 'nine', 100 steps")
    for j, p in enumerate(("7", "Seven!", "a thick 3", "draw me a 9 please", "sept", "a picture of a cat", "three seven", "4 leaning right")):
        ax = fig.add_subplot(gs[1 + j // 4, j % 4])                                   # one panel per prompt, two rows of four
        raw, fe, note = keep_f[p]
        if fe is None:                                                                # refused: only the raw row exists
            grid(ax, raw, 1, 6, f"{p!r}: raw (above);\nfront end {note}")
        else:
            grid(ax, np.vstack([raw, fe]), 2, 6, f"{p!r}: raw (top), front end (bottom)")
    ax = fig.add_subplot(gs[3, 0:2]); x = np.arange(10)                               # the model card: ink
    ax.bar(x - 0.2, card[:, 1], 0.4, label="samples, w = 2", color="#C44E52"); ax.bar(x + 0.2, card[:, 2], 0.4, label="real training digits", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(d143.NUM); ax.set_ylabel("ink (average pixel)"); ax.legend(fontsize=8)
    ax.set_title("model card: ink per prompt", fontsize=10)
    ax = fig.add_subplot(gs[3, 2:4])                                                  # the model card: attributes
    for i, (col, lab, c) in enumerate(((3, "thick", "#C44E52"), (4, "thin", "#DD8452"), (5, "leaning right", "#4C72B0"), (6, "leaning left", "#55A868"))):
        ax.bar(x + (i - 1.5) * 0.2, card[:, col], 0.2, label=lab, color=c)
    ax.axhline(1 / 3, color="gray", ls="--", label="real digits (a third each)"); ax.set_xticks(x); ax.set_xticklabels(d143.NUM)
    ax.set_ylabel("share of samples"); ax.legend(fontsize=8, ncol=5, loc="upper center"); ax.set_ylim(0, 1.15)
    ax.set_title("model card: thickness and lean per prompt", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # the week's parts as one system
    fig, ax = plt.subplots(figsize=(15, 5.2)); ax.set_xlim(0, 15); ax.set_ylim(0.4, 4.8); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.1, 3.2, 1.7, 0.9, "prompt\n'Seven!'", "#FFF2CC")
    box(3.4, 3.2, 2.1, 1.2, "front end (today)\nlower case, numerals\nas words; one digit\nword, or refuse", "#FCE4D6")
    box(3.4, 1.0, 2.1, 0.8, "refused, with\nthe reason", "#F4CCCC")
    box(6.0, 3.2, 2.0, 1.0, "text encoder\n(Day 143)\n64 numbers", "#DDEBF7")
    box(8.9, 3.2, 2.6, 1.2, "guided DDIM\nb + w (a - b), w = 2\n(Days 140-142, 144)\n10 to 100 steps", "#DDEBF7")
    box(8.9, 1.0, 2.6, 0.8, "noise, 784 numbers\n(seed)", "#EEEEEE")
    box(11.6, 3.2, 1.6, 0.9, "digits", "#E2F0D9")
    box(13.7, 3.2, 2.0, 1.4, "checks (Day 145)\nnamed digit, ink,\nlean, nearest\ntraining digit", "#E2F0D9")
    arrow(1.95, 3.2, 2.35, 3.2); arrow(3.4, 2.6, 3.4, 1.4); arrow(4.45, 3.2, 5.0, 3.2); arrow(7.0, 3.2, 7.6, 3.2)
    arrow(8.9, 1.4, 8.9, 2.6); arrow(10.2, 3.2, 10.8, 3.2); arrow(12.4, 3.2, 12.7, 3.2)
    ax.text(7.5, 4.4, "trained once: the encoder on captioned digits (Day 143), then the diffusion model on digits and their embedded "
            "captions, 15% dropped (Day 144)", ha="center", fontsize=8.5, style="italic")
    ax.set_title("Day 146: the week's parts as one text-to-digit generator", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "text_to_digit_capstone_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Week 22 turns to time series: Day 147 decomposes a real series into trend, seasonality and what is left over.")


def main():                                             # runs every section in order
    X, y = d126m.mnist(); X2 = (X * 2 - 1).astype(np.float32)                         # Day 126's 70,000 digits; [-1, 1] for the diffusion model
    ink, slant, thick, lean = d143.measure(X, y)                                      # Day 143's measured attributes, for captions and the card
    te = 60000 + np.random.RandomState(0).permutation(10000)[:2000]                    # 2,000 test digits in random order (Day 143's)
    t0 = time.perf_counter(); body, head = d139.train_classifier(X[:60000], y[:60000])  # Day 138's feature network, for naming and scoring
    sc = d139.Scorer(body, head, X[:10000], X[10000:30000])
    t1 = time.perf_counter(); img, txt, _ = d143.train_clip(X[:60000], y[:60000], thick[:60000], lean[:60000], hold=HOLD)  # Day 143's encoders, as on Day 144
    print(f"(feature network for the scores: {t1 - t0:.0f} s; text and image encoders: {time.perf_counter() - t1:.0f} s)\n")
    demo_parts(img, txt, X, y, te)                                                    # 1. the parts
    net, ema, log, F, took = demo_model(sc, img, txt, X, y, X2, thick, lean, te)      # 2. the model, checked
    keep_f = demo_front(sc, txt, ema)                                                 # 3. the front end
    rows, keep_s = demo_steps(sc, txt, ema, F, took)                                  # 4. speed
    card = demo_card(sc, F, X, y, ink, slant, te)                                     # 5. the model card
    plot_results(log, rows, keep_s, keep_f, card, os.path.join(HERE, "text_to_digit_capstone_results.png"))
    draw_diagram()                                                                    # the system
    print("\nsaved text_to_digit_capstone_results.png\nsaved text_to_digit_capstone_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
