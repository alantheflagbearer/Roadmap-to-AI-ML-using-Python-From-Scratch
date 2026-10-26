"""
Day 93: Sampling Strategies From Scratch -- Greedy, Temperature, Top-k, Top-p

A trained language model gives a probability for every next token. Writing
text means choosing one, again and again. Four ways to choose, all from
scratch:

    greedy        always the most probable token
    temperature   divide the scores by t before the softmax: t < 1 sharpens, t > 1 flattens
    top-k         keep only the k most probable tokens, renormalise, draw
    top-p         keep the smallest set whose probabilities add up to p (nucleus sampling,
                  Holtzman et al., 2020), renormalise, draw

The model is Day 92's big mini-GPT (4 blocks, d = 128, context 128), loaded
from day92_mini_gpt.npz. Each strategy continues the same 20 prompts taken
from the held-out text, and is measured, not just read:

  * repetition: the share of word pairs in the text that are distinct
  * real words: the share of words that occur in the training text
  * how likely the model itself finds the text (low bits = safe, bland text)
  * how Shakespeare-like an outside judge finds it: Day 91's Kneser-Ney model,
    trained on the training text, scores the generated text

What is measured, reported as it came out:
  1. The samplers draw from exactly the distributions they claim (toy check).
  2. Seven strategies on 20 prompts, measured.
  3. One prompt, every strategy, so the numbers can be read as text.

Days 90 and 91 are imported from their scripts. Every line of code below
carries its own comment, continuing Day 52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: splitting text into words
import sys                                             # sys: path, to import Days 90 and 91

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 91 sits in the same folder
sys.path.insert(0, os.path.join(HERE, "..", "week-13-tokenization-embeddings"))  # Day 90's folder
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing; kept quiet to be sure
    import day90_tiny_gpt as d90                        # Day 90: the GPT's forward pass, the data
    import day91_causal_language_modeling as d91        # Day 91: the Kneser-Ney model, as the outside judge

MODEL = os.path.join(HERE, "day92_mini_gpt.npz")        # Day 92's saved big model
N_PROMPTS, PROMPT_LEN, GEN_LEN = 20, 32, 400            # prompts from the held-out text; their length; characters written after each
STRATEGIES = [("greedy", dict(temperature=0)), ("temperature 0.5", dict(temperature=0.5)),
              ("temperature 0.8", dict(temperature=0.8)), ("temperature 1.0", dict(temperature=1.0)),
              ("temperature 1.2", dict(temperature=1.2)), ("top-k 5", dict(top_k=5)), ("top-p 0.9", dict(top_p=0.9))]
JUDGE_ORDER = 7                                         # Kneser-Ney with 6 characters of context: Day 91's best


# =============================================================================
# 1. THE SAMPLER -- NEW
# =============================================================================
def sample_next(logits, rng, temperature=1.0, top_k=None, top_p=None):  # NEW: one token per row of logits (m, V)
    if temperature == 0:                                  # greedy
        return logits.argmax(axis=-1)                       # the most probable token, every time
    z = logits / temperature                              # temperature: sharpen (< 1) or flatten (> 1)
    p = np.exp(z - z.max(axis=-1, keepdims=True)); p /= p.sum(axis=-1, keepdims=True)  # softmax
    if top_k is not None:                                 # keep only the k most probable tokens
        kth = np.sort(p, axis=-1)[:, -top_k][:, None]       # the k-th largest probability, per row
        p = np.where(p >= kth, p, 0.0)                      # everything smaller set to 0 (ties at the k-th are kept)
    if top_p is not None:                                 # keep the smallest set whose mass reaches top_p
        order = np.argsort(-p, axis=-1)                     # tokens from most to least probable
        sp = np.take_along_axis(p, order, axis=-1)          # their probabilities, sorted
        keep_sorted = np.cumsum(sp, axis=-1) - sp < top_p   # a token stays if the mass BEFORE it is still below top_p
        keep = np.zeros_like(p, dtype=bool); np.put_along_axis(keep, order, keep_sorted, axis=-1)  # back to token order
        p = np.where(keep, p, 0.0)                          # the rest set to 0
    p /= p.sum(axis=-1, keepdims=True)                    # renormalise what is left
    u = rng.rand(len(p), 1)                               # one uniform number per row
    return np.minimum((np.cumsum(p, axis=-1) < u).sum(axis=-1), p.shape[-1] - 1)  # inverse-CDF draw (clipped against rounding at the top)


# =============================================================================
# 2. THE MODEL AND GENERATION
# =============================================================================
def load_model(path=MODEL):                             # Day 92's file -> the nested weights, the vocabulary, the settings
    f = np.load(path)                                     # the saved arrays
    P, cfg = {}, {}                                       # weights; settings
    for k in f.files:                                     # every saved array
        if k == "vocab":                                    # the vocabulary
            continue
        if k.startswith("cfg__"):                           # a setting
            cfg[k[5:]] = int(f[k]); continue
        parts = k.split("__"); node = P                     # 'blocks__0__Wq' -> P['blocks']['0']['Wq']
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = f[k].astype(np.float64)           # back to float64 for the arithmetic
    return P, [str(v) for v in f["vocab"]], cfg


def generate(P, cfg, prompts, n, rng, **how):           # NEW: continue every prompt n tokens at once (one row per prompt)
    ids = [list(p) for p in prompts]                      # the texts so far
    for _ in range(n):                                    # one token per step, for all prompts together
        x = np.array([s[-cfg["T"]:] for s in ids])          # the last T tokens of each (all the same length here)
        lg = d90.gpt_forward(x, P, cfg["h"])[0][:, -1]      # Day 90's forward pass: scores for the next token
        for s, t in zip(ids, sample_next(lg, rng, **how)):  # draw one per prompt
            s.append(int(t))
    return [s[len(p):] for s, p in zip(ids, prompts)]     # the new tokens only


def model_bits(P, cfg, prompts, conts):                 # how surprising the model itself finds its continuations (bits per character)
    full = np.array([list(p) + list(c) for p, c in zip(prompts, conts)])  # prompt + continuation
    tot, cnt = 0.0, 0                                     # summed -log2 p; characters
    for s in range(PROMPT_LEN, full.shape[1]):            # every continuation position
        x = full[:, max(0, s - cfg["T"]):s]                 # its context (at most T)
        lp = d90.log_softmax(d90.gpt_forward(x, P, cfg["h"])[0][:, -1])  # log-probabilities of the next token
        tot -= lp[np.arange(len(full)), full[:, s]].sum() / np.log(2); cnt += len(full)
    return tot / cnt                                      # bits per character


def text_stats(texts, known, judge):                    # distinct word pairs, real words, outside judge's bits per character
    words = [re.findall(r"[a-z']+", t.lower()) for t in texts]  # the words of each text
    pairs = [tuple(w[i:i + 2]) for w in words for i in range(len(w) - 1)]  # word pairs, within texts
    flat = [x for w in words for x in w]                  # all words
    jb = np.mean([judge.bits(t, JUDGE_ORDER) for t in texts if len(t) > JUDGE_ORDER])  # Kneser-Ney's bits per character
    return len(set(pairs)) / max(1, len(pairs)), np.mean([x in known for x in flat]), jb


# =============================================================================
# DEMOS
# =============================================================================
def demo_toy():
    print("=" * 70)
    print("1. DO THE SAMPLERS DRAW WHAT THEY CLAIM? -- a toy distribution, 200,000 draws each")
    print("=" * 70)
    p = np.array([0.5, 0.2, 0.15, 0.1, 0.05]); lg = np.log(p)[None].repeat(200000, axis=0)  # five tokens
    expect = {"temperature 1.0": p, "temperature 0.5": p ** 2 / (p ** 2).sum(),  # t = 0.5: p^(1/t), renormalised
              "top-k 2": np.r_[p[:2], 0, 0, 0] / p[:2].sum(), "top-p 0.8": np.r_[p[:3], 0, 0] / p[:3].sum()}  # top-p 0.8: 0.5 + 0.2 < 0.8, so three tokens
    how = {"temperature 1.0": dict(), "temperature 0.5": dict(temperature=0.5), "top-k 2": dict(top_k=2), "top-p 0.8": dict(top_p=0.8)}
    print(f"probabilities: {p.tolist()}\n\n{'sampler':<17}{'expected':<42}{'drawn':<42}{'largest gap':>11}")
    rng = np.random.RandomState(0)                        # fixed rng
    for name in expect:                                   # each sampler
        got = np.bincount(sample_next(lg, rng, **how[name]), minlength=5) / 200000  # observed frequencies
        print(f"{name:<17}{str(np.round(expect[name], 3).tolist()):<42}{str(np.round(got, 3).tolist()):<42}{np.abs(got - expect[name]).max():>11.4f}")
    print(f"greedy: {sample_next(lg[:3], rng, temperature=0).tolist()} (always token 0)")
    print("(with 200,000 draws, random gaps of a few thousandths are expected)")
    print("Actual output from this lesson's run.")


def demo_strategies(P, cfg, vocab, tr_text, va_text):
    print("\n" + "=" * 70)
    print("2. SEVEN STRATEGIES, 20 PROMPTS -- measured")
    print("=" * 70)
    ix = {c: i for i, c in enumerate(vocab)}              # character -> id
    rng = np.random.RandomState(7)                        # where the prompts start
    starts = rng.randint(0, len(va_text) - PROMPT_LEN - GEN_LEN, N_PROMPTS)  # random places in the held-out text
    prompts = [[ix[c] for c in va_text[s:s + PROMPT_LEN]] for s in starts]  # 32-character prompts
    known = set(re.findall(r"[a-z']+", tr_text.lower()))  # every training word
    judge = d91.KneserNey(tr_text, JUDGE_ORDER, 0.9)      # Day 91's best counting model, Day 91's discount
    real = [va_text[s + PROMPT_LEN:s + PROMPT_LEN + GEN_LEN] for s in starts]  # what really followed each prompt
    print(f"Day 92's model ({sum(a.size for _, a in d90.w12.leaves(P))} weights) continues {N_PROMPTS} prompts of "
          f"{PROMPT_LEN} held-out characters by {GEN_LEN} characters with each strategy (seed 0 for every strategy). "
          f"Distinct pairs: distinct word pairs / all word pairs. Real words: share in the training text. Model bits: "
          f"how surprising the model finds its own text. Judge bits: Day 91's Kneser-Ney ({JUDGE_ORDER - 1} characters of "
          "context) on the same text. The first row is the real held-out continuation.")
    d, r, j = text_stats(real, known, judge)              # the real text, for reference
    print(f"\n{'strategy':<17}{'distinct pairs':>15}{'real words':>12}{'model bits':>12}{'judge bits':>12}")
    print(f"{'(real text)':<17}{d:>15.3f}{r:>12.3f}{model_bits(P, cfg, prompts, [[ix[c] for c in t] for t in real]):>12.3f}{j:>12.3f}")
    out = {"(real text)": real}                           # strategy -> its texts
    rows = []                                             # for the plot
    for name, how in STRATEGIES:                          # each strategy
        conts = generate(P, cfg, prompts, GEN_LEN, np.random.RandomState(0), **how)  # write
        texts = ["".join(vocab[i] for i in c) for c in conts]  # as text
        d, r, j = text_stats(texts, known, judge); mb = model_bits(P, cfg, prompts, conts)
        rows.append((name, d, r, mb, j)); out[name] = texts
        print(f"{name:<17}{d:>15.3f}{r:>12.3f}{mb:>12.3f}{j:>12.3f}")
    print("Actual output from this lesson's run.")
    return rows, out, prompts                              # the plot and section 3 use them


def demo_example(out, prompts, vocab):
    print("\n" + "=" * 70)
    print("3. ONE PROMPT, EVERY STRATEGY -- the first 160 characters of prompt 1's continuations")
    print("=" * 70)
    print("prompt: " + repr("".join(vocab[i] for i in prompts[0])))
    for name, texts in out.items():                       # the real text first, then each strategy
        print(f"\n--- {name} ---\n{texts[0][:160]}")
    print("Actual output from this lesson's run.")


def plot_results(rows):                                 # two panels: repetition and realness | the two kinds of bits
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))           # canvas
    names = [r[0] for r in rows]; x = np.arange(len(rows))    # strategies
    ax[0].bar(x - 0.2, [r[1] for r in rows], width=0.4, color="#1f77b4", label="distinct word pairs")
    ax[0].bar(x + 0.2, [r[2] for r in rows], width=0.4, color="#2ca02c", label="real words")
    ax[0].set_xticks(x); ax[0].set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8); ax[0].set_ylim(0, 1.25)
    ax[0].set_title("Repetition and real words", fontsize=10); ax[0].legend(fontsize=8, loc="upper center", ncol=2); ax[0].grid(alpha=0.3, axis="y")
    ax[1].plot(x, [r[3] for r in rows], marker="o", color="#d62728", label="model's own bits (how safe)")
    ax[1].plot(x, [r[4] for r in rows], marker="s", color="#7f7f7f", label="Kneser-Ney judge's bits (how odd)")
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax[1].set_ylabel("bits per character"); ax[1].set_title("Safe for the model, odd for the judge", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("sampling_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved sampling_results.png")


def draw_diagram():                                     # one distribution, four ways to cut it
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))           # canvas
    p = np.array([0.42, 0.22, 0.14, 0.09, 0.06, 0.04, 0.03])  # an example next-token distribution (illustration only)
    t05 = p ** 2 / (p ** 2).sum()                             # temperature 0.5
    k3 = np.r_[p[:3], np.zeros(4)] / p[:3].sum()              # top-k 3
    keep = np.cumsum(p) - p < 0.8; tp = np.where(keep, p, 0) / p[keep].sum()  # top-p 0.8
    g = np.r_[1, np.zeros(6)]                                 # greedy
    for a, q, title in zip(ax, (g, t05, k3, tp), ("greedy", "temperature 0.5", "top-k 3", "top-p 0.8")):
        a.bar(np.arange(7), p, color="#dddddd", label="model")  # the original
        a.bar(np.arange(7), q, color="#1f77b4", alpha=0.8, label="sampled from")  # after the cut
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_ylim(0, 1.05)
    ax[0].legend(fontsize=7)
    fig.suptitle("Day 93: four ways to choose the next token from the same distribution", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig("sampling_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved sampling_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("4. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 94 takes a lot more generated text and asks harder questions: can a simple classifier tell it from real "
          "Shakespeare, and how much of it is copied from the training text?")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        demo_toy()                                           # 1. the samplers
        P, vocab, cfg = load_model()                         # Day 92's model
        tr_text, va_text = d90.load_text()                   # Day 90's split
        rows, out, prompts = demo_strategies(P, cfg, vocab, tr_text, va_text)  # 2. strategies
        demo_example(out, prompts, vocab)                    # 3. one prompt
        plot_results(rows)                                   # the figure
        draw_diagram()                                       # the diagram
        note_on_what_comes_next()                            # 4. Day 94


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
