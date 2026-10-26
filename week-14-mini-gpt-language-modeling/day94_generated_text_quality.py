"""
Day 94: Real Generated Text From the Mini-GPT -- an Honest Quality Assessment

Reading a few good samples is how generated text usually gets judged, and it
is the easiest way to fool yourself. Today 120 passages from Day 92's
mini-GPT are judged by measurements, against two references written from the
same 120 held-out prompts:

  * the real continuation (what Shakespeare wrote next)
  * Day 91's Kneser-Ney counting model, sampling the same way

Four questions:

  1. Can a simple classifier tell it from real text? A logistic regression on
     character 1-3-gram counts, trained from scratch, five-fold cross-validated.
     50% accuracy would mean it cannot tell; 100% that it always can.
  2. How much is copied? The share of every k-character piece of the passage
     that occurs word for word in the training text -- also measured on the
     real held-out text, which shares common phrases with training by chance.
  3. Does it keep the play's shape? Speaker lines ('NAME:'), line lengths.
  4. Words: real words, repetition (distinct word pairs), and how odd Day 91's
     Kneser-Ney judge finds the text.

The model is Day 92's, loaded from day92_mini_gpt.npz. Generation is Day 93's
code (top-p 0.9 and temperature 1.0), imported from its script. Every line
of code below carries its own comment, continuing Day 52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: words and speaker lines
import sys                                             # sys: path, to import Days 90, 91 and 93
import zlib                                            # zlib: crc32, a fixed hash for the classifier's features

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Days 91 and 93 sit in the same folder
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing; kept quiet to be sure
    import day93_sampling_strategies as d93             # Day 93: the sampler, generation, the model loader, text_stats
d90, d91 = d93.d90, d93.d91                              # Days 90 and 91, as Day 93 imported them

N, PROMPT_LEN, GEN_LEN = 120, 32, 300                   # passages; prompt length; characters generated after each
KS = (8, 12, 16, 24, 32)                                # piece lengths for the copying measure
BUCKETS, FOLDS = 4096, 5                                # classifier: hashed features, cross-validation folds
L2, CLF_STEPS, CLF_LR = 1e-3, 500, 0.5                  # classifier: regularisation, gradient steps, step size
SPEAKER = re.compile(r"^[A-Z][A-Z' ]+:$")               # a speaker line: capitals, then a colon, alone on the line


# =============================================================================
# 1. A REFERENCE GENERATOR: KNESER-NEY SAMPLING
# =============================================================================
def kn_generate(kn, vocab, prompt_text, n, rng, top_p=None):  # NEW: write n characters with Day 91's Kneser-Ney model
    text = prompt_text                                    # the text so far
    for _ in range(n):                                    # one character at a time
        ctx = text[-(kn.n - 1):]                            # the context the model uses
        p = np.array([kn.prob(ctx, c) for c in vocab])      # its distribution over the next character
        nxt = d93.sample_next(np.log(p)[None], rng, top_p=top_p)[0]  # Day 93's sampler
        text += vocab[nxt]
    return text[len(prompt_text):]                        # the new characters only


# =============================================================================
# 2. THE MEASUREMENTS -- NEW
# =============================================================================
def features(texts):                                    # hashed counts of every 1-, 2- and 3-character piece, each row scaled to length 1
    X = np.zeros((len(texts), BUCKETS))                   # one row per text
    for r, t in enumerate(texts):                         # each text
        for n in (1, 2, 3):                                 # piece lengths
            for i in range(len(t) - n + 1):
                X[r, zlib.crc32(t[i:i + n].encode()) % BUCKETS] += 1  # a fixed hash, the same on every run
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def logistic_cv(Xa, Xb, seed=0):                        # NEW: five-fold cross-validated accuracy of a logistic regression telling a from b
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]  # data and labels
    folds = np.random.RandomState(seed).permutation(len(y)) % FOLDS  # a fold for every example
    acc = []                                              # one accuracy per fold
    for f in range(FOLDS):                                # each fold held out in turn
        tr, te = folds != f, folds == f                     # train and test rows
        w, b = np.zeros(X.shape[1]), 0.0                    # weights and bias
        for _ in range(CLF_STEPS):                          # full-batch gradient descent on the mean log-loss
            p = 1 / (1 + np.exp(-(X[tr] @ w + b)))            # predicted probability of 'b'
            g = p - y[tr]                                     # the log-loss gradient for the scores
            w -= CLF_LR * (X[tr].T @ g / tr.sum() + L2 * w); b -= CLF_LR * g.mean()
        acc.append(np.mean(((X[te] @ w + b) > 0) == y[te]))  # accuracy on the held-out fold
    return np.mean(acc), np.std(acc)


def copy_share(texts, train_text, k):                   # NEW: share of every k-character piece of the texts that occurs in the training text
    seen = {train_text[i:i + k] for i in range(len(train_text) - k + 1)}  # every k-piece of the training text
    pieces = [t[i:i + k] for t in texts for i in range(len(t) - k + 1)]  # every k-piece of the texts
    return np.mean([p in seen for p in pieces])


def shape_stats(texts):                                 # speaker lines and line lengths
    lines = [ln for t in texts for ln in t.split("\n")[1:-1] if ln.strip()]  # complete, non-empty lines
    return np.mean([bool(SPEAKER.match(ln)) for ln in lines]), np.mean([len(ln) for ln in lines])


# =============================================================================
# DEMOS
# =============================================================================
def make_passages(P, cfg, vocab, tr_text, va_text):
    print("=" * 70)
    print("1. FOUR KINDS OF PASSAGE, FROM THE SAME 120 PROMPTS")
    print("=" * 70)
    ix = {c: i for i, c in enumerate(vocab)}              # character -> id
    starts = np.random.RandomState(11).randint(0, len(va_text) - PROMPT_LEN - GEN_LEN, N)  # places in the held-out text
    prompts_txt = [va_text[s:s + PROMPT_LEN] for s in starts]  # the prompts
    prompts = [[ix[c] for c in p] for p in prompts_txt]   # as ids
    src = {"real": [va_text[s + PROMPT_LEN:s + PROMPT_LEN + GEN_LEN] for s in starts]}  # what really followed
    for name, how in (("GPT, temperature 1.0", dict(temperature=1.0)), ("GPT, top-p 0.9", dict(top_p=0.9))):
        conts = d93.generate(P, cfg, prompts, GEN_LEN, np.random.RandomState(0), **how)  # Day 93's generation
        src[name] = ["".join(vocab[i] for i in c) for c in conts]
    kn = d91.KneserNey(tr_text, d93.JUDGE_ORDER, 0.9)    # Day 91's best counting model
    rng = np.random.RandomState(0)                        # its sampling rng
    src["Kneser-Ney, top-p 0.9"] = [kn_generate(kn, vocab, p, GEN_LEN, rng, top_p=0.9) for p in prompts_txt]
    print(f"{N} prompts of {PROMPT_LEN} characters from random places in the held-out text; {GEN_LEN} characters after "
          "each: the real continuation, Day 92's GPT at temperature 1.0 and at top-p 0.9 (Day 93's sampler, seed 0), and "
          f"Day 91's Kneser-Ney model ({d93.JUDGE_ORDER - 1} characters of context) at top-p 0.9.")
    print("\nthe first 200 characters after prompt 1 (" + repr(prompts_txt[0]) + "):")
    for name, texts in src.items():                       # one example from each
        print(f"\n--- {name} ---\n{texts[0][:200]}")
    print("Actual output from this lesson's run.")
    return src                                             # every later section uses them


def demo_classifier(src):
    print("\n" + "=" * 70)
    print("2. CAN A SIMPLE CLASSIFIER TELL IT FROM REAL SHAKESPEARE?")
    print("=" * 70)
    print(f"Each passage becomes {BUCKETS} hashed counts of its 1-, 2- and 3-character pieces. A logistic regression "
          f"(L2 {L2}, {CLF_STEPS} gradient steps) learns real against generated; accuracy on held-out folds, "
          f"{FOLDS}-fold cross-validation. 0.5 = cannot tell. The real passages are split in two as a control.")
    Xr = features(src["real"]); out = {}                  # the real passages' features
    print(f"\n{'real against':<26}{'accuracy':>10}{'spread over folds':>19}")
    half = len(Xr) // 2                                   # the control: real against real
    m, s = logistic_cv(Xr[:half], Xr[half:]); out["real (other half)"] = m
    print(f"{'real (other half)':<26}{m:>10.3f}{s:>19.3f}")
    for name in [k for k in src if k != "real"]:          # each generator
        m, s = logistic_cv(Xr, features(src[name])); out[name] = m
        print(f"{name:<26}{m:>10.3f}{s:>19.3f}")
    print("Actual output from this lesson's run.")
    return out                                             # the plot shows them


def demo_copying(src, tr_text):
    print("\n" + "=" * 70)
    print("3. HOW MUCH IS COPIED? -- pieces that occur word for word in the training text")
    print("=" * 70)
    print("For each piece length k: the share of all k-character pieces of the passages found somewhere in the "
          "training text. Real held-out text is the reference: it shares common phrases with training by chance.")
    print(f"\n{'source':<26}" + "".join(f"{'k=' + str(k):>9}" for k in KS))
    out = {}                                              # source -> shares
    for name, texts in src.items():                       # each source
        out[name] = [copy_share(texts, tr_text, k) for k in KS]
        print(f"{name:<26}" + "".join(f"{v:>9.3f}" for v in out[name]))
    print("Actual output from this lesson's run.")
    return out                                             # the plot shows them


def demo_shape_words(src, tr_text):
    print("\n" + "=" * 70)
    print("4. THE PLAY'S SHAPE, AND THE WORDS")
    print("=" * 70)
    known = set(re.findall(r"[a-z']+", tr_text.lower()))  # every training word
    judge = d91.KneserNey(tr_text, d93.JUDGE_ORDER, 0.9)  # Day 93's judge
    print("Speaker lines: complete lines that are a name in capitals followed by a colon. Distinct pairs, real words and "
          "judge bits are Day 93's measures.")
    print(f"\n{'source':<26}{'speaker lines':>14}{'line length':>13}{'distinct pairs':>16}{'real words':>12}{'judge bits':>12}")
    for name, texts in src.items():                       # each source
        sp, ll = shape_stats(texts); d, r, j = d93.text_stats(texts, known, judge)
        print(f"{name:<26}{sp:>14.3f}{ll:>13.1f}{d:>16.3f}{r:>12.3f}{j:>12.3f}")
    print("Actual output from this lesson's run.")


def plot_results(clf, cp):                              # two panels: classifier accuracy | copying by piece length
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))           # canvas
    names = list(clf); x = np.arange(len(names))              # sources
    ax[0].bar(x, [clf[n] for n in names], color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"][:len(names)])
    ax[0].axhline(0.5, color="#d62728", linestyle="--", label="0.5: cannot tell")
    ax[0].set_xticks(x); ax[0].set_xticklabels([n.replace(", ", ",\n") for n in names], fontsize=8); ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("cross-validated accuracy"); ax[0].set_title("Can a classifier tell it from real text?", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3, axis="y")
    for name, col in zip(cp, ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]):  # copying curves
        ax[1].plot(KS, cp[name], marker="o", color=col, label=name)
    ax[1].set_xlabel("piece length k (characters)"); ax[1].set_ylabel("share of pieces found in the training text")
    ax[1].set_title("How much is copied word for word?", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("quality_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved quality_results.png")


def draw_diagram():                                     # the assessment: four sources, four tests
    fig, ax = plt.subplots(figsize=(12, 3.8))                   # canvas
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.8); ax.axis("off")     # coordinates in "boxes"
    ax.text(6, 3.55, "Day 94: judging generated text by measurement", ha="center", fontsize=12, fontweight="bold")

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    box(1.4, 1.75, 2.2, 1.0, "120 held-out\nprompts", "#FAEEDA")
    for k, (name, color) in enumerate([("real continuation", "#F2F2F2"), ("GPT, temperature 1.0", "#DCEBF7"),
                                       ("GPT, top-p 0.9", "#DCEBF7"), ("Kneser-Ney, top-p 0.9", "#E5F2E1")]):
        y = 3.0 - k * 0.8
        box(4.8, y, 2.8, 0.55, name, color, fs=8)
        ax.annotate("", xy=(3.38, y), xytext=(2.52, 1.75), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(8.0, 1.75), xytext=(6.22, y), arrowprops=dict(arrowstyle="->", color="#333333"))
    box(9.9, 1.75, 3.6, 1.6, "1. classifier: real or not?\n2. copied from training?\n3. the play's shape\n4. words and the judge",
        "#F2F2F2", fs=8.5)
    plt.savefig("quality_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved quality_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 95 asks how the held-out loss itself moves with model size and with the amount of training text: "
          "scaling laws, at the smallest possible scale.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        P, vocab, cfg = d93.load_model()                     # Day 92's model
        tr_text, va_text = d90.load_text()                   # Day 90's split
        src = make_passages(P, cfg, vocab, tr_text, va_text)  # 1. the passages
        clf = demo_classifier(src)                           # 2. the classifier
        cp = demo_copying(src, tr_text)                      # 3. copying
        demo_shape_words(src, tr_text)                       # 4. shape and words
        plot_results(clf, cp)                                # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 5. Day 95


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
