"""
Day 97: The Week 14 Build -- Language-Model Evaluation From the Week's Parts:
Perplexity, BLEU and ROUGE

Week 14's last day does two jobs, like Days 83 and 90. The syllabus topic is
language-model evaluation metrics written from scratch: perplexity, BLEU
and ROUGE. And this is the week's integration day: the evaluation is
assembled from the six exercise scripts of Days 91-96.

The parts, and where each one goes:

  Day 91 exercise  KneserNey                          the counting baseline
  Day 92 exercise  lr_schedule, eval_bits             perplexity with full context; a schedule for section 6
  Day 93 exercise  sample_next                        writing continuations
  Day 94 exercise  copy_share, distinct_pairs, ...    the text measures
  Day 95 exercise  count_weights, fit_power_law       model sizes; loss against context length
  Day 96 exercise  load_params, adam_subset, ...      the two saved models; fine-tuning in section 6
  Day 90 (imported, as on Days 92-96)                 the GPT's forward and backward pass

The new code is BLEU (Papineni et al., 2002) and ROUGE-1, ROUGE-2 and
ROUGE-L (Lin, 2004), and the experiments. The task is Day 96's: The Tempest,
a play the pretrained model never saw, scored on its test part.

What is measured, reported as it came out:
  1. The build: the parts fit; the two models load; BLEU and ROUGE on cases
     with known answers.
  2. Perplexity: four models of The Tempest's test text.
  3. BLEU and ROUGE: each model continues 24 prompts; the real continuation is
     the reference -- with controls (the reference itself, an unrelated real
     passage, the prompt repeated).
  4. Day 94's measures on the same continuations.
  5. How the loss falls with context: a power law fitted with Day 95's part.
  6. Metrics as a training signal: perplexity and BLEU during a short fine-tune.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: words (and the speaker pattern of Day 94's part)
import sys                                             # sys: path, to import Days 90 and 96
from collections import Counter, defaultdict           # Counter, defaultdict: n-gram tables (Day 91's part) and BLEU/ROUGE counts

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from threadpoolctl import threadpool_limits            # limit NumPy's matrix library to one thread (faster here; see Day 64)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 96 sits in the same folder
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing; kept quiet to be sure
    import day96_fine_tuning as d96                     # Day 96: The Tempest's split (and, through it, Day 90's model code)
d90 = d96.d90                                           # Day 90's GPT, as Days 92-96 imported it

MODELS = {"Day 92 (general)": "day92_mini_gpt.npz", "Day 96 (fine-tuned)": "day96_finetuned_gpt.npz"}  # the two saved models
N_PROMPTS, PROMPT_LEN, CONT_LEN = 24, 64, 150           # continuations: 24 non-overlapping (prompt, continuation) windows fill the test part
SEEDS = [0, 1, 2]                                       # three sampling seeds for the BLEU and ROUGE table
KN_ORDER, KN_D = 7, 0.9                                 # Day 91's best Kneser-Ney: 6 characters of context, discount 0.9
FT_STEPS, FT_CHECK, FT_LR = 200, 50, 3e-4               # section 6: a short top-only fine-tune of Day 92's model, checked every 50 steps
SPEAKER = re.compile(r"^[A-Z][A-Z' ]+:$")               # from day94_exercise.py, verbatim: capitals, then a colon, alone on the line


# =============================================================================
# 1. THE PARTS -- Week 14's exercise scripts, verbatim
# =============================================================================
# ---- from day91_exercise.py, verbatim ----
class KneserNey:                                        # NEW: interpolated Kneser-Ney n-gram model over characters
    def __init__(self, text, order, D=0.75):              # count every n-gram up to the order
        self.n, self.D, self.V = order, D, sorted(set(text))  # order, discount, the characters
        self.cnt = [Counter() for _ in range(order + 1)]   # cnt[k][g]: how often the k-gram g occurs
        for k in range(1, order + 1):                       # every length
            self.cnt[k].update(text[i:i + k] for i in range(len(text) - k + 1))
        self.cont = [Counter() for _ in range(order + 1)]  # cont[k][g]: how many different characters precede g
        for k in range(1, order):                           # lower orders only
            for g in self.cnt[k + 1]:                         # every (k + 1)-gram seen
                self.cont[k][g[1:]] += 1                        # one more distinct left neighbour
        self.tab = [None] + [self._tables(k) for k in range(1, order + 1)]  # per order: counts, context totals, distinct followers

    def _tables(self, k):                                 # context totals and distinct next characters, for order k
        src = self.cnt[k] if k == self.n else self.cont[k]  # raw counts at the top, continuation counts below
        tot, kinds = defaultdict(float), defaultdict(int)   # context -> total; context -> distinct followers
        for g, c in src.items():                            # every k-gram
            tot[g[:-1]] += c; kinds[g[:-1]] += 1
        return src, tot, kinds

    def prob(self, ctx, ch, k=None):                      # p(ch | the last k - 1 characters of ctx)
        k = self.n if k is None else k                      # start at the top order
        if k == 0:                                          # below unigrams: uniform
            return 1.0 / len(self.V)
        src, tot, kinds = self.tab[k]                       # this order's tables
        h = ctx[len(ctx) - (k - 1):] if k > 1 else ""       # the context this order uses
        if tot[h] == 0:                                     # never seen: the lower order alone
            return self.prob(ctx, ch, k - 1)
        return max(src[h + ch] - self.D, 0.0) / tot[h] + self.D * kinds[h] / tot[h] * self.prob(ctx, ch, k - 1)

    def bits(self, text, start):                          # mean bits per character of text[start:]
        return -np.mean([np.log2(self.prob(text[max(0, i - self.n + 1):i], text[i])) for i in range(start, len(text))])


# ---- from day92_exercise.py, verbatim ----
def lr_schedule(step, peak, warmup, total, floor=0.1):  # NEW: linear warm-up, then cosine decay to floor * peak
    if step < warmup:                                     # the first steps
        return peak * (step + 1) / warmup                   # grow linearly
    progress = (step - warmup) / max(1, total - warmup)   # 0 after the warm-up, 1 at the end
    return peak * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * min(1.0, progress))))  # half a cosine


def eval_bits(logp_fn, ids, T):                         # NEW: bits per token, every scored token with at least T/2 tokens of context
    half = T // 2                                         # stride and scored positions per window
    starts = np.arange(0, len(ids) - T - 1, half)         # window starts
    nats, scored = 0.0, []                                # summed -ln p; which positions were scored
    for s in range(0, len(starts), 32):                   # 32 windows at a time
        st = starts[s:s + 32]
        x = np.stack([ids[i:i + T] for i in st]); y = np.stack([ids[i + 1:i + T + 1] for i in st])  # inputs, next tokens
        lp = logp_fn(x)[:, half:]                           # the model's log-probabilities, last half only
        nats -= np.take_along_axis(lp, y[:, half:, None], axis=-1).sum()
        scored += [i + 1 + j for i in st for j in range(half, T)]  # the text positions these targets sit at
    return nats / len(scored) / np.log(2), np.array(scored)  # bits per token, and the scored positions


# ---- from day93_exercise.py, verbatim ----
def sample_next(logits, rng, temperature=1.0, top_k=None, top_p=None):  # NEW: one token per row of logits (m, V)
    if temperature == 0:                                  # greedy
        return logits.argmax(axis=-1)
    z = logits / temperature                              # sharpen or flatten
    p = np.exp(z - z.max(axis=-1, keepdims=True)); p /= p.sum(axis=-1, keepdims=True)  # softmax
    if top_k is not None:                                 # the k most probable tokens
        p = np.where(p >= np.sort(p, axis=-1)[:, -top_k][:, None], p, 0.0)
    if top_p is not None:                                 # the smallest set reaching top_p
        order = np.argsort(-p, axis=-1); sp = np.take_along_axis(p, order, axis=-1)  # sorted, most probable first
        keep = np.zeros_like(p, dtype=bool); np.put_along_axis(keep, order, np.cumsum(sp, axis=-1) - sp < top_p, axis=-1)
        p = np.where(keep, p, 0.0)
    p /= p.sum(axis=-1, keepdims=True)                    # renormalise
    return np.minimum((np.cumsum(p, axis=-1) < rng.rand(len(p), 1)).sum(axis=-1), p.shape[-1] - 1)  # inverse-CDF draw


# ---- from day94_exercise.py, verbatim ----
def copy_share(texts, train_text, k):                   # NEW: copied pieces of length k
    seen = {train_text[i:i + k] for i in range(len(train_text) - k + 1)}  # every k-piece of the training text
    pieces = [t[i:i + k] for t in texts for i in range(len(t) - k + 1)]  # every k-piece of the texts
    return np.mean([p in seen for p in pieces])


def distinct_pairs(texts):                              # NEW: repetition of word pairs
    words = [re.findall(r"[a-z']+", t.lower()) for t in texts]
    pairs = [tuple(w[i:i + 2]) for w in words for i in range(len(w) - 1)]
    return len(set(pairs)) / max(1, len(pairs))


def real_word_rate(texts, known):                       # NEW: words that exist
    words = [x for t in texts for x in re.findall(r"[a-z']+", t.lower())]
    return np.mean([x in known for x in words])


def speaker_share(texts):                               # NEW: the play's layout
    lines = [ln for t in texts for ln in t.split("\n")[1:-1] if ln.strip()]  # complete, non-empty lines
    return np.mean([bool(SPEAKER.match(ln)) for ln in lines])


# ---- from day95_exercise.py, verbatim ----
def count_weights(V, d, layers, t, ff=None):            # NEW: Day 90's GPT, by formula
    ff = ff or 4 * d                                      # feed-forward width
    block = 4 * d * d + 2 * d * ff + ff + d + 4 * d       # Wq Wk Wv Wo; W1 W2; c1 c2; two LayerNorms
    return V * d + t * d + layers * block + d * V + V     # token and position tables, blocks, output layer and bias


def fit_power_law(N, L):                                # NEW: L = c + a N^-alpha
    N, L = np.asarray(N, float), np.asarray(L, float)
    best = None                                           # (squared error, c, a, alpha)
    for c in np.linspace(0, L.min() * 0.999, 400):        # the floor must be below every loss
        slope, icpt = np.polyfit(np.log(N), np.log(L - c), 1)  # log(L - c) = log a - alpha log N
        err = np.sum((c + np.exp(icpt) * N ** slope - L) ** 2)
        if best is None or err < best[0]:
            best = (err, c, np.exp(icpt), -slope)
    return best[1:]


# ---- from day96_exercise.py, verbatim ----
def leaves(D, prefix=""):                               # every array inside nested dictionaries, with a dotted name (Day 82's exercise)
    for k, v in D.items():
        if isinstance(v, dict):
            yield from leaves(v, prefix + k + ".")
        else:
            yield prefix + k, v


def save_params(path, P):                               # NEW: one compressed file, one named array per weight
    np.savez_compressed(path, **{k.replace(".", "__"): v for k, v in leaves(P)})


def load_params(path):                                  # NEW: the file back into nested dictionaries
    P, f = {}, np.load(path)                              # result; the file
    for k in f.files:                                     # every stored array
        parts = k.split("__"); node = P                     # 'blocks__0__W' -> ['blocks', '0', 'W']
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = f[k]
    return P


def adam_subset(P, G, state, lr, keys):                 # NEW: Adam (Day 37's four lines) on the named weights only
    g = dict(leaves(G)); state["t"] += 1; t = state["t"]  # gradients by name; step count
    for k, A in leaves(P):                                # every weight ...
        if k not in keys:                                   # ... frozen ones are skipped
            continue
        m = state["m"].setdefault(k, np.zeros_like(A)); v = state["v"].setdefault(k, np.zeros_like(A))
        m[:] = 0.9 * m + 0.1 * g[k]; v[:] = 0.999 * v + 0.001 * g[k] ** 2
        A -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def snapshot(P):                                        # NEW: a deep copy of nested dictionaries of arrays
    return {k: (snapshot(v) if isinstance(v, dict) else v.copy()) for k, v in P.items()}


# =============================================================================
# 2. BLEU AND ROUGE -- NEW
# =============================================================================
def words(text):                                        # NEW: lowercase words (letters and apostrophes)
    return re.findall(r"[a-z']+", text.lower())


def ngrams(ws, n):                                      # NEW: the n-word sequences of a word list, counted
    return Counter(tuple(ws[i:i + n]) for i in range(len(ws) - n + 1))


def bleu(hyps, refs, max_n=4):                          # NEW: corpus BLEU (Papineni et al., 2002): clipped n-gram precisions, geometric mean, brevity penalty
    match, total = np.zeros(max_n), np.zeros(max_n)       # clipped matches and hypothesis n-grams, per n
    hyp_len = ref_len = 0                                 # total lengths, for the brevity penalty
    for h, r in zip(hyps, refs):                          # each (hypothesis, reference) pair
        hw, rw = words(h), words(r); hyp_len += len(hw); ref_len += len(rw)
        for n in range(1, max_n + 1):                       # 1- to 4-grams
            hc, rc = ngrams(hw, n), ngrams(rw, n)
            match[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())  # a hypothesis n-gram counts at most as often as in the reference
            total[n - 1] += max(0, len(hw) - n + 1)
    if match[0] == 0:                                     # not one shared word: BLEU is 0
        return 0.0, match / np.maximum(total, 1)
    prec = np.r_[match[0] / total[0], (match[1:] + 1) / (total[1:] + 1)]  # BLEU+1 (Lin & Och, 2004): add one to the 2- to 4-gram counts, so one empty order does not zero the score
    bp = 1.0 if hyp_len > ref_len else np.exp(1 - ref_len / max(hyp_len, 1))  # brevity penalty: short hypotheses are penalised
    return bp * np.exp(np.mean(np.log(prec))), match / np.maximum(total, 1)  # BLEU, and the raw precisions per n


def rouge_n(hyp, ref, n):                               # NEW: ROUGE-N (Lin, 2004): n-gram overlap; F1 of recall and precision
    hc, rc = ngrams(words(hyp), n), ngrams(words(ref), n)  # n-gram counts
    overlap = sum(min(c, rc[g]) for g, c in hc.items())   # shared n-grams, clipped
    rec = overlap / max(1, sum(rc.values())); prec = overlap / max(1, sum(hc.values()))
    return 0.0 if overlap == 0 else 2 * rec * prec / (rec + prec)


def rouge_l(hyp, ref):                                  # NEW: ROUGE-L: the longest common subsequence of words; F1
    a, b = words(hyp), words(ref)                         # word lists
    L = np.zeros((len(a) + 1, len(b) + 1), dtype=int)     # dynamic-programming table of LCS lengths
    for i in range(1, len(a) + 1):                        # every prefix of the hypothesis ...
        for j in range(1, len(b) + 1):                      # ... against every prefix of the reference
            L[i, j] = L[i - 1, j - 1] + 1 if a[i - 1] == b[j - 1] else max(L[i - 1, j], L[i, j - 1])
    lcs = L[-1, -1]                                       # its length
    return 0.0 if lcs == 0 else 2 * (lcs / len(b)) * (lcs / len(a)) / ((lcs / len(b)) + (lcs / len(a)))


def rouge(hyps, refs):                                  # mean ROUGE-1, -2 and -L F1 over the pairs
    return tuple(np.mean([f(h, r) for h, r in zip(hyps, refs)]) for f in (lambda h, r: rouge_n(h, r, 1),
                                                                          lambda h, r: rouge_n(h, r, 2), rouge_l))


# =============================================================================
# 3. MODELS AND CONTINUATIONS -- NEW
# =============================================================================
def load_model(name):                                   # one saved model through Day 96's part: weights, vocabulary, settings
    P = load_params(os.path.join(HERE, MODELS[name]))     # day96_exercise.py's load_params
    vocab = [str(v) for v in P.pop("vocab")]              # the vocabulary was stored beside the weights
    cfg = {k: int(v) for k, v in P.pop("cfg").items()}    # so were the settings ('cfg__T' -> P['cfg']['T'])
    return to64(P), vocab, cfg                            # weights back to float64


def to64(P):                                            # float32 storage back to float64 arithmetic, through the nesting
    return {k: (to64(v) if isinstance(v, dict) else v.astype(np.float64)) for k, v in P.items()}


def gpt_logp(P, cfg):                                   # a model as a function: ids (m, T) -> log-probabilities (m, T, V) (what Day 92's eval_bits takes)
    return lambda x: d90.log_softmax(d90.gpt_forward(x, P, cfg["h"])[0])


def gpt_continue(P, cfg, prompts, n, rng, **how):       # continue every prompt n tokens, all at once, with Day 93's sampler
    seqs = [list(p) for p in prompts]                     # the texts so far
    for _ in range(n):                                    # one token per step
        lg = d90.gpt_forward(np.array([s[-cfg["T"]:] for s in seqs]), P, cfg["h"])[0][:, -1]  # next-token scores
        for s, t in zip(seqs, sample_next(lg, rng, **how)):  # day93_exercise.py's sample_next
            s.append(int(t))
    return [s[len(p):] for s, p in zip(seqs, prompts)]    # new tokens only


def kn_continue(kn, vocab, prompt, n, rng, **how):      # continue one prompt with a Kneser-Ney model and the same sampler
    text = prompt                                         # the text so far
    for _ in range(n):                                    # one character at a time
        p = np.array([kn.prob(text[-(kn.n - 1):], c) for c in vocab])  # day91_exercise.py's KneserNey
        text += vocab[sample_next(np.log(p)[None], rng, **how)[0]]
    return text[len(prompt):]


def position_bits(logp_fn, ids_, T):                    # NEW: held-out bits per token at each position of non-overlapping windows
    n = (len(ids_) - 1) // T                              # windows that fit
    x, y = ids_[:n * T].reshape(n, T), ids_[1:n * T + 1].reshape(n, T)  # inputs, next tokens
    lp = np.concatenate([logp_fn(x[s:s + 32]) for s in range(0, n, 32)])  # log-probabilities
    return -np.take_along_axis(lp, y[..., None], axis=-1)[..., 0].mean(axis=0) / np.log(2)  # mean over windows, per position


# =============================================================================
# DEMOS
# =============================================================================
def demo_build(task):
    print("=" * 70)
    print("1. THE BUILD -- the parts, the two models, and BLEU/ROUGE on known cases")
    print("=" * 70)
    print("Every function in section 1 is copied verbatim from a Week 14 exercise script (a banner names each one); "
          "Day 90's GPT is imported through Day 96's script.")
    models = {}                                           # name -> (weights, vocabulary, settings)
    for name in MODELS:                                   # both saved models
        P, vocab, cfg = load_model(name); models[name] = (P, vocab, cfg)
        n = sum(A.size for _, A in leaves(P))               # weights counted from the file
        print(f"{name}: {n} weights loaded from {MODELS[name]}; Day 95's formula gives "
              f"{count_weights(len(vocab), cfg['d'], cfg['layers'], cfg['T'], cfg['ff'])}")
    a, b = models["Day 92 (general)"][0], dict(leaves(models["Day 96 (fine-tuned)"][0]))
    moved = {k: np.abs(A - b[k]).max() for k, A in leaves(a)}  # how far fine-tuning moved each array
    print(f"largest change made by fine-tuning: {max(moved.values()):.3f} (in {max(moved, key=moved.get)}); arrays unchanged: "
          f"{sum(v == 0 for v in moved.values())} of {len(moved)}")
    s = "the king is come to the island"                   # known cases for the metrics
    print("\nBLEU and ROUGE on known cases (hypothesis | reference):")
    for h, r in ((s, s), ("the king is come", s), ("island the to come is king the", s), ("my mistress waits below", s)):
        b4 = bleu([h], [r])[0]; r1, r2, rl = rouge([h], [r])
        print(f"  {h!r:<36} BLEU {b4:.3f}  ROUGE-1 {r1:.3f}  ROUGE-2 {r2:.3f}  ROUGE-L {rl:.3f}")
    print("Actual output from this lesson's run.")
    return models                                          # later sections use them


def demo_perplexity(models, task):
    print("\n" + "=" * 70)
    print("2. PERPLEXITY -- four models of The Tempest's test text")
    print("=" * 70)
    tr_text, ft_tr, ft_dev, ft_te, rest = task           # Day 96's split
    print(f"Bits per character on the play's test part ({len(ft_te):,} characters), and perplexity per character "
          f"(2^bits) and per word (2^(bits x characters per word)). The GPTs: Day 92's full-context measure (day92_exercise.py's "
          f"eval_bits). Kneser-Ney ({KN_ORDER - 1} characters of context): every character from the {KN_ORDER}th, with its "
          "real preceding context.")
    cpw = len(ft_te) / len(ft_te.split())                 # characters per word in the test part
    rows, kns = [], []                                    # (name, bits); the two counting models
    for name, text in (("Kneser-Ney, training text", tr_text), ("Kneser-Ney, + the play's train part", tr_text + ft_tr)):
        kn = KneserNey(text, KN_ORDER, KN_D); kns.append(kn); rows.append((name, kn.bits(ft_te, KN_ORDER - 1)))
    for name, (P, vocab, cfg) in models.items():          # the two GPTs
        ix = {c: i for i, c in enumerate(vocab)}
        rows.append((name, eval_bits(gpt_logp(P, cfg), np.array([ix[c] for c in ft_te]), cfg["T"])[0]))
    print(f"\n{'model':<36}{'bits/char':>10}{'ppl/char':>10}{'ppl/word':>10}")
    for name, b in rows:                                  # one row per model
        print(f"{name:<36}{b:>10.3f}{2 ** b:>10.2f}{2 ** (b * cpw):>10.0f}")
    print(f"({cpw:.2f} characters per word in the test part; the two kinds of score are on slightly different positions)")
    print("Actual output from this lesson's run.")
    return rows, kns[1]                                    # the plot shows the rows; section 3 uses the second counting model


def demo_bleu(models, task, kn):
    print("\n" + "=" * 70)
    print("3. BLEU AND ROUGE -- continuations against what Shakespeare wrote")
    print("=" * 70)
    tr_text, ft_tr, ft_dev, ft_te, rest = task
    starts = np.arange(N_PROMPTS) * (PROMPT_LEN + CONT_LEN)  # evenly spaced, so no two windows share any text
    prompts = [ft_te[s:s + PROMPT_LEN] for s in starts]   # 64-character prompts
    refs = [ft_te[s + PROMPT_LEN:s + PROMPT_LEN + CONT_LEN] for s in starts]  # what really followed: the references
    other = refs[1:] + refs[:1]                           # a control: the real continuation of a DIFFERENT prompt (never overlapping)
    print(f"{N_PROMPTS} non-overlapping prompts of {PROMPT_LEN} characters, evenly spaced through the play's test part; {CONT_LEN} characters written after each "
          f"(top-p 0.9, day93_exercise.py's sampler). Reference: the real next {CONT_LEN} characters. BLEU: corpus BLEU-4, word "
          f"level, BLEU+1 smoothing (one added to the 2- to 4-gram counts). ROUGE: mean F1. {len(SEEDS)} sampling seeds; mean "
          "(spread). Controls: the reference itself; a real continuation of another prompt (real Shakespeare, wrong place); "
          "the prompt itself, repeated to the same length.")
    vocab = models["Day 92 (general)"][1]; ix = {c: i for i, c in enumerate(vocab)}
    systems = {}                                          # name -> list (one per seed) of hypothesis lists
    for name, (P, _, cfg) in models.items():              # the two GPTs
        systems[name] = []
        for s in SEEDS:                                     # three seeds
            conts = gpt_continue(P, cfg, [[ix[c] for c in p] for p in prompts], CONT_LEN, np.random.RandomState(s), top_p=0.9)
            systems[name].append(["".join(vocab[i] for i in c) for c in conts])
    systems["Kneser-Ney (+ play)"] = [[kn_continue(kn, vocab, p, CONT_LEN, np.random.RandomState(s), top_p=0.9) for p in prompts] for s in SEEDS]
    controls = {"(reference itself)": refs, "(real text, wrong place)": other, "(prompt repeated)": [(p * 3)[:CONT_LEN] for p in prompts]}
    print(f"\n{'system':<26}{'BLEU-4':>14}{'ROUGE-1':>14}{'ROUGE-2':>14}{'ROUGE-L':>14}")
    out = {}                                              # name -> (BLEU, R1, R2, RL) means
    for name, hyps in controls.items():                   # controls: one value each
        sc = (bleu(hyps, refs)[0],) + rouge(hyps, refs); out[name] = sc
        print(f"{name:<26}" + "".join(f"{v:>14.3f}" for v in sc))
    for name, runs in systems.items():                    # systems: mean and spread over seeds
        sc = np.array([(bleu(h, refs)[0],) + rouge(h, refs) for h in runs]); out[name] = sc.mean(axis=0)
        print(f"{name:<26}" + "".join(f"{m:>8.3f} ({s:.3f})" for m, s in zip(sc.mean(axis=0), sc.std(axis=0))))
    print("Actual output from this lesson's run.")
    return out, systems, refs                              # later sections use them


def demo_measures(systems, refs, task):
    print("\n" + "=" * 70)
    print("4. DAY 94'S MEASURES ON THE SAME CONTINUATIONS (seed 0)")
    print("=" * 70)
    seen = task[0] + task[1]                              # everything either model trained on: the training text and the play's train part
    known = set(re.findall(r"[a-z']+", seen.lower()))    # its words
    print("Copied pieces are looked up in, and real words checked against, the training text plus the play's train part "
          "(everything either model was trained on).")
    print(f"\n{'text':<26}{'copied (k=16)':>14}{'distinct pairs':>16}{'real words':>12}{'speaker lines':>15}")
    for name, texts in [("(real continuations)", refs)] + [(k, v[0]) for k, v in systems.items()]:
        print(f"{name:<26}{copy_share(texts, seen, 16):>14.3f}{distinct_pairs(texts):>16.3f}"
              f"{real_word_rate(texts, known):>12.3f}{speaker_share(texts):>15.3f}")
    print("Actual output from this lesson's run.")


def demo_context(models, task):
    print("\n" + "=" * 70)
    print("5. HOW THE LOSS FALLS WITH CONTEXT -- a power law, fitted with Day 95's part")
    print("=" * 70)
    rest = task[4]                                        # the general held-out text: 15 times the play's test part, so far less noise
    P, vocab, cfg = models["Day 92 (general)"]            # the general model, on general text
    ix = {c: i for i, c in enumerate(vocab)}
    pos = position_bits(gpt_logp(P, cfg), np.array([ix[c] for c in rest]), cfg["T"])  # bits at each position
    n = np.arange(1, len(pos) + 1)                        # the prediction at position p (from 0) sees p + 1 characters
    print(f"Day 92's model on the rest of the held-out text ({len(rest):,} characters; {(len(rest) - 1) // cfg['T']} "
          f"non-overlapping windows of {cfg['T']}): bits per character by how many characters the prediction sees, and "
          "L(n) = c + a n^-alpha fitted from n = 1, 4 and 16 on.")
    print("\n  " + ", ".join(f"{k}: {pos[k - 1]:.2f}" for k in (1, 2, 4, 8, 16, 32, 64, 128)))
    fits = {}                                             # first n -> (c, a, alpha)
    for lo in (1, 4, 16):                                 # three starting points
        c, a, al = fit_power_law(n[lo - 1:], pos[lo - 1:]); fits[lo] = (c, a, al)  # day95_exercise.py's fit
        gap = np.abs(c + a * n[lo - 1:] ** -al - pos[lo - 1:]).max()  # worst point
        print(f"  fit from n = {lo:>2}: c = {c:.3f}, a = {a:.3f}, alpha = {al:.3f}; largest gap {gap:.3f} bits "
              f"(c searched on [0, {0.999 * pos[lo - 1:].min():.3f}])")
    print(f"  mean bits at n = 1-8: {pos[:8].mean():.3f}; n = 9-16: {pos[8:16].mean():.3f}; n = 17-64: {pos[16:64].mean():.3f}; "
          f"n = 65-128: {pos[64:].mean():.3f}")
    print("Actual output from this lesson's run.")
    return n, pos, fits                                    # the plot shows them


def demo_training_signal(models, task):
    print("\n" + "=" * 70)
    print("6. METRICS AS A TRAINING SIGNAL -- perplexity and BLEU during a short fine-tune")
    print("=" * 70)
    tr_text, ft_tr, ft_dev, ft_te, rest = task
    P, vocab, cfg = load_model("Day 92 (general)")        # a fresh copy of the general model
    ix = {c: i for i, c in enumerate(vocab)}; tr, dev = (np.array([ix[c] for c in t]) for t in (ft_tr, ft_dev))
    keys = {k for k, _ in leaves(P) if k.startswith(("blocks.3.", "Wout", "bout"))}  # top block and output layer (Day 96's 'top only')
    starts = np.random.RandomState(4).randint(0, len(ft_dev) - PROMPT_LEN - CONT_LEN, 20)  # 20 dev prompts
    prompts = [[ix[c] for c in ft_dev[s:s + PROMPT_LEN]] for s in starts]
    refs = [ft_dev[s + PROMPT_LEN:s + PROMPT_LEN + CONT_LEN] for s in starts]
    def check():                                          # dev bits, and dev BLEU with a fixed sampling seed
        hyps = ["".join(vocab[i] for i in c) for c in gpt_continue(P, cfg, prompts, CONT_LEN, np.random.RandomState(0), top_p=0.9)]
        return eval_bits(gpt_logp(P, cfg), dev, cfg["T"])[0], bleu(hyps, refs)[0]
    print(f"Day 92's model, top block and output layer only (day96_exercise.py's adam_subset), {FT_STEPS} steps at peak "
          f"{FT_LR} with day92_exercise.py's schedule; every {FT_CHECK} steps, dev bits per character and BLEU-4 of 20 dev "
          "continuations (one fixed sampling seed). The best checkpoint by dev bits is kept with snapshot().")
    rng = np.random.RandomState(0); state = {"t": 0, "m": {}, "v": {}}
    T, B = cfg["T"], cfg["B"]
    rows = [(0,) + check()]; best = (rows[0][1], 0, snapshot(P))  # step 0
    for step in range(1, FT_STEPS + 1):                   # fine-tuning
        o = rng.randint(0, len(tr) - T - 1, B)
        x = np.stack([tr[i:i + T] for i in o]); y = np.stack([tr[i + 1:i + T + 1] for i in o])
        _, G = d90.gpt_loss_grads(x, y, P, h=cfg["h"])      # Day 90's backward pass
        adam_subset(P, G, state, lr_schedule(step - 1, FT_LR, 10, FT_STEPS), keys)  # day96_exercise.py, day92_exercise.py
        if step % FT_CHECK == 0:
            rows.append((step,) + check())
            if rows[-1][1] < best[0]:
                best = (rows[-1][1], step, snapshot(P))
    print(f"\n{'step':>6}{'dev bits':>10}{'dev BLEU-4':>12}")
    for r in rows:
        print(f"{r[0]:>6}{r[1]:>10.3f}{r[2]:>12.4f}")
    print(f"best checkpoint by dev bits: step {best[1]}")
    print("Actual output from this lesson's run.")
    return np.array(rows)                                  # the plot shows them


def plot_results(ppl, bl, ctx, sig):                    # four panels: perplexity | BLEU/ROUGE with controls | loss vs context | training signal
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))           # canvas
    ax[0].barh([r[0] for r in ppl], [r[1] for r in ppl], color=["#7f7f7f", "#bbbbbb", "#1f77b4", "#d62728"])
    ax[0].invert_yaxis(); ax[0].set_xlabel("bits per character, The Tempest test"); ax[0].set_xlim(1.5, max(r[1] for r in ppl) + 0.2)
    ax[0].set_title("Perplexity separates the models", fontsize=10); ax[0].grid(alpha=0.3, axis="x")
    names = [k for k in bl if k != "(reference itself)"]; x = np.arange(len(names))
    for j, (lab, col) in enumerate([("BLEU-4", "#1f77b4"), ("ROUGE-1", "#2ca02c"), ("ROUGE-L", "#ff7f0e")]):
        ax[1].bar(x + (j - 1) * 0.27, [bl[n][[0, 1, 3][j]] for n in names], width=0.27, color=col, label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(" ", "\n", 2) for n in names], fontsize=7)
    ax[1].set_title("BLEU and ROUGE (the reference itself scores 1)", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    n, pos, fits = ctx
    ax[2].plot(n, pos, ".", color="#1f77b4", label="Day 92 model, general held-out text")
    for lo, col in ((1, "#d62728"), (16, "#2ca02c")):          # the fit from the start, and from n = 16
        c, a, al = fits[lo]; ax[2].plot(n[lo - 1:], c + a * n[lo - 1:] ** -al, color=col, label=f"fit from n = {lo}: alpha {al:.2f}")
    ax[2].set_xscale("log"); ax[2].set_xlabel("characters seen (log scale)"); ax[2].set_ylabel("bits per character")
    ax[2].set_title("Loss against context", fontsize=10); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    a2 = ax[3].twinx()                                        # two scales: bits and BLEU
    ax[3].plot(sig[:, 0], sig[:, 1], marker="o", color="#1f77b4", label="dev bits per character")
    a2.plot(sig[:, 0], sig[:, 2], marker="s", color="#ff7f0e", label="dev BLEU-4")
    ax[3].set_xlabel("fine-tuning step"); ax[3].set_ylabel("dev bits per character", color="#1f77b4"); a2.set_ylabel("dev BLEU-4", color="#ff7f0e")
    ax[3].set_title("During fine-tuning", fontsize=10); ax[3].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("evaluation_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved evaluation_results.png")


def draw_diagram():                                     # the build: the six parts into one evaluation
    fig, ax = plt.subplots(figsize=(13, 4.8))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8):                       # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.55, "Day 97: Week 14's build -- one evaluation from the week's six parts", ha="center", fontsize=12, fontweight="bold")
    parts = [("Day 91", "KneserNey: the counting baseline"), ("Day 92", "lr_schedule, eval_bits: perplexity"),
             ("Day 93", "sample_next: continuations"), ("Day 94", "copy_share, distinct_pairs, ..."),
             ("Day 95", "count_weights, fit_power_law"), ("Day 96", "load_params, adam_subset, snapshot")]
    for i, (day, name) in enumerate(parts):                          # the parts, left column
        y = 3.8 - i * 0.62
        box(2.6, y, 4.4, 0.5, f"{day}: {name}", "#FAEEDA", fs=7.5)
        ax.annotate("", xy=(5.6, 2.25), xytext=(4.82, y), arrowprops=dict(arrowstyle="->", color="#333333"))
    box(7.0, 2.25, 2.6, 1.2, "NEW: BLEU-4,\nROUGE-1, -2, -L\n(from scratch)", "#DCEBF7", fs=8.5)
    box(10.7, 2.25, 3.6, 2.0, "The evaluation:\nperplexity of 4 models,\nBLEU/ROUGE with controls,\ntext measures (The Tempest),\nloss vs context (general text),\nmetrics during fine-tuning",
        "#E5F2E1", fs=8)
    ax.annotate("", xy=(8.88, 2.25), xytext=(8.32, 2.25), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("evaluation_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved evaluation_diagram.png")


def note_on_the_week():
    print("\n" + "=" * 70)
    print("7. THE WEEK, AND WHAT COMES NEXT")
    print("=" * 70)
    print("Week 14 studied one language model end to end: the chain rule and honest baselines (Day 91), training and "
          "perplexity (92), sampling (93), judging generated text (94), scaling (95), fine-tuning (96) and, today, the "
          "metrics. Week 15 turns to practical NLP with pretrained models.")


def main():                                             # runs every section in order
    with threadpool_limits(1, "blas"):                     # one BLAS thread (Day 64)
        task = d96.task_data()                               # Day 96's split of The Tempest
        models = demo_build(task)                            # 1. the build
        ppl, kn = demo_perplexity(models, task)              # 2. perplexity
        bl, systems, refs = demo_bleu(models, task, kn)          # 3. BLEU and ROUGE
        demo_measures(systems, refs, task)                   # 4. Day 94's measures
        ctx = demo_context(models, task)                     # 5. loss against context
        sig = demo_training_signal(models, task)             # 6. metrics during fine-tuning
        plot_results(ppl, bl, ctx, sig)                      # the figure
        draw_diagram()                                       # the diagram
        note_on_the_week()                                   # 7. Week 15


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
