"""
Day 101: Text Summarization -- Extractive vs Abstractive, on Real News

Two ways to shorten an article:

  * EXTRACTIVE: pick the most important sentences and copy them out unchanged.
  * ABSTRACTIVE: write a new, shorter text, possibly with words and sentences
    that are not in the article at all.

The data is the CNN/DailyMail test set (news articles, each with 3-4
'highlights' written by the journalists, used as the reference summary):
100 articles drawn at random. The systems:

  lead-3          the first three sentences (a famously strong baseline for news)
  TextRank        from scratch: sentences as TF-IDF vectors, a similarity graph,
                  PageRank by power iteration, the top three in article order
  T5-small        google-t5/t5-small, a pretrained text-to-text Transformer
                  (60 million weights), given 'summarize: ' + the article
  oracle          the best three sentences chosen by looking at the reference
                  (an upper bound for extractive methods, not a real system)

Scored with Day 97's ROUGE-1, ROUGE-2 and ROUGE-L (imported unchanged), with
Day 99's bootstrap for intervals, plus checks ROUGE cannot do: how much of each
summary is new wording, and whether its numbers and words appear in the article.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: sentences, words, numbers, names
import sys                                             # sys: path, to import Days 97 and 99
import time                                            # time: perf_counter, for timings

import numpy as np                                     # numpy: the from-scratch parts
import torch                                           # torch: T5 runs on it
from datasets import disable_progress_bars, load_dataset  # datasets: CNN/DailyMail from the local cache
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # the library's entry points
from transformers.utils import logging as hf_logging   # the library's logging
from huggingface_hub.utils import logging as hub_logging  # the model hub's logging
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 99's exercise sits here
sys.path.insert(0, os.path.join(HERE, "..", "week-14-mini-gpt-language-modeling"))  # Day 97's ROUGE
with contextlib.redirect_stdout(io.StringIO()):         # the imports print nothing
    import day97_lm_evaluation as d97                   # Day 97: rouge_n, rouge_l, words
    import day99_exercise as d99                        # Day 99: bootstrap_ci, paired_bootstrap
if hasattr(sys.stdout, "reconfigure"):                  # (not when another script imports this one quietly)
    sys.stdout.reconfigure(encoding="utf-8")            # the articles' curly quotes and accents print as UTF-8 on Windows too
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()  # no progress bars or notices in the output
hub_logging.set_verbosity_error(); disable_progress_bars()            # the same for the hub and datasets
torch.set_num_threads(4)                                # CPU threads for T5

MODEL = "google-t5/t5-small"                            # the abstractive summariser
N_ARTICLES, K = 100, 3                                  # articles; sentences per extractive summary
SYSTEMS = ["lead-3", "TextRank", "T5-small", "oracle"]  # in the order they are reported


# =============================================================================
# SENTENCES AND WORDS -- NEW
# =============================================================================
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "gen", "gov", "sen", "rep", "lt", "col", "sgt", "capt", "u.s",
          "u.k", "no", "vs", "inc", "co", "corp", "ltd", "jan", "feb", "mar", "apr", "aug", "sept", "oct", "nov", "dec"}  # not sentence ends


def split_sentences(text):                              # NEW: sentence boundaries by rule (an exercise part, Day 104 reuses it)
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'‘“(]?[A-Z0-9])", text.strip())  # a stop, space, then a capital or digit
    out = []                                              # sentences, with false splits joined back
    for p in parts:
        prev = out[-1].split()[-1].lower().rstrip(".") if out else ""  # the word before the full stop
        if out and (prev in ABBREV or re.fullmatch(r"[a-z]", prev)):  # 'Mr.' or an initial 'J.' is not a sentence end
            out[-1] += " " + p
        else:
            out.append(p)
    return [s.strip() for s in out if len(s.split()) >= 3]  # drop fragments of one or two words


# =============================================================================
# THE SYSTEMS -- NEW (except T5)
# =============================================================================
def lead(sents, k=K):                                   # the first k sentences
    return " ".join(sents[:k])


def tfidf(sents):                                       # NEW: one TF-IDF vector per sentence, rows of unit length
    toks = [d97.words(s) for s in sents]                  # Day 97's lowercase words
    vocab = {w: i for i, w in enumerate(sorted({w for t in toks for w in t}))}  # every word in the article
    tf = np.zeros((len(sents), len(vocab)))               # term counts
    for r, t in enumerate(toks):
        for w in t:
            tf[r, vocab[w]] += 1
    idf = np.log(len(sents) / (1 + (tf > 0).sum(0))) + 1  # rarer words across sentences weigh more
    v = tf * idf
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def textrank(sents, k=K, d=0.85, iters=100):            # NEW: PageRank on the sentence-similarity graph
    V = tfidf(sents); S = V @ V.T                         # cosine similarity of every pair (rows have unit length)
    np.fill_diagonal(S, 0)                                # no self-links
    W = S / (S.sum(1, keepdims=True) + 1e-12)             # each sentence shares its vote among the others
    r = np.ones(len(sents)) / len(sents)                  # start uniform
    for _ in range(iters):                                # power iteration
        r = (1 - d) / len(sents) + d * W.T @ r
    top = sorted(np.argsort(-r)[:k])                      # the k highest scores, back in article order
    return " ".join(sents[i] for i in top), r


def oracle(sents, ref, k=K):                            # NEW: greedily add the sentence that most raises ROUGE-1 + ROUGE-2 against the reference
    chosen, best = [], 0.0
    for _ in range(k):
        gains = [(d97.rouge_n(" ".join(sents[j] for j in sorted(chosen + [i])), ref, 1) +
                  d97.rouge_n(" ".join(sents[j] for j in sorted(chosen + [i])), ref, 2), i) for i in range(len(sents)) if i not in chosen]
        if not gains or max(gains)[0] <= best:             # stop when nothing helps
            break
        best, i = max(gains); chosen.append(i)
    return " ".join(sents[j] for j in sorted(chosen))


def t5_summaries(articles):                             # the abstractive system, with the settings saved in T5's own config
    tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    cfg = model.config.task_specific_params["summarization"]  # prefix, beams, length limits chosen by T5's authors
    out, t0 = [], time.perf_counter()
    with torch.no_grad():                                 # inference only
        for a in articles:                                  # one article at a time
            enc = tok(cfg["prefix"] + a, truncation=True, max_length=512, return_tensors="pt")  # T5 reads at most 512 tokens
            ids = model.generate(**enc, num_beams=cfg["num_beams"], max_length=cfg["max_length"], min_length=cfg["min_length"],
                                 length_penalty=cfg["length_penalty"], no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
                                 early_stopping=cfg["early_stopping"])
            out.append(tok.decode(ids[0], skip_special_tokens=True))
    return out, cfg, (time.perf_counter() - t0) / len(articles), model


# =============================================================================
# CHECKS ROUGE CANNOT DO -- NEW
# =============================================================================
def novel_trigrams(summary, article):                   # share of the summary's word 3-grams that are NOT in the article
    s, a = d97.words(summary), set(d97.ngrams(d97.words(article), 3))
    tri = [tuple(s[i:i + 3]) for i in range(len(s) - 2)]
    return np.mean([t not in a for t in tri]) if tri else 0.0


def unsupported(summary, article):                      # numbers, and words of 4+ letters, in the summary with no match in the article
    nums = set(re.findall(r"\d[\d,.]*\d|\d", summary)) - set(re.findall(r"\d[\d,.]*\d|\d", article))
    stems = {w[:5] for w in re.findall(r"[a-z]+", article.lower())}  # the article's words, cut to 5 letters ('suffer' covers 'suffered')
    new = {w for w in re.findall(r"[a-z]+", summary.lower()) if len(w) >= 4 and w[:5] not in stems}  # lowercase: T5-small often lowercases words
    return nums | new


# =============================================================================
# DEMOS
# =============================================================================
def load_articles():                                    # 100 random test articles and their highlights
    d = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    idx = np.random.RandomState(0).choice(len(d), N_ARTICLES, replace=False)  # a fixed random sample
    return [d[int(i)]["article"] for i in idx], [d[int(i)]["highlights"].replace(" .\n", ". ").replace("\n", " ") for i in idx]


def demo_data(arts, refs):
    print("=" * 70)
    print("1. THE DATA -- 100 CNN/DailyMail test articles")
    print("=" * 70)
    S = [split_sentences(a) for a in arts]                # every article as sentences
    aw, rw = [len(a.split()) for a in arts], [len(r.split()) for r in refs]
    print(f"articles: {np.mean(aw):.0f} words on average (10th-90th percentile {np.percentile(aw, 10):.0f}-{np.percentile(aw, 90):.0f}), "
          f"{np.mean([len(s) for s in S]):.1f} sentences; reference highlights: {np.mean(rw):.0f} words")
    print("\nfirst article's opening sentences (rule-based splitting):")
    for s in S[0][:3]:
        print(f"  | {s[:120]}")
    print(f"its reference: {refs[0][:240]}")
    print("Actual output from this lesson's run.")
    return S                                                # later sections use them


def demo_systems(arts, refs, S):
    print("\n" + "=" * 70)
    print("2. FOUR SUMMARIES OF EACH ARTICLE")
    print("=" * 70)
    t5, cfg, secs, model = t5_summaries(arts)             # the slow one first
    print(f"T5-small: {sum(p.numel() for p in model.parameters()):,} weights; settings from its config: {cfg}")
    print(f"  {secs:.1f} s per article on this CPU (4 threads); input cut at 512 tokens")
    out = {"lead-3": [lead(s) for s in S], "TextRank": [textrank(s)[0] for s in S], "T5-small": t5,
           "oracle": [oracle(s, r) for s, r in zip(S, refs)]}
    print("\nthe first article, each system:")
    for name in SYSTEMS:
        print(f"  {name:<9}| {out[name][0][:230]}")
    print("Actual output from this lesson's run.")
    return out                                              # later sections use them


def demo_rouge(refs, out):
    print("\n" + "=" * 70)
    print("3. ROUGE AGAINST THE JOURNALISTS' HIGHLIGHTS")
    print("=" * 70)
    print("Mean F1 over 100 articles (Day 97's functions; words are lowercase letters only, so numbers are ignored), with "
          "Day 99's 95% bootstrap interval for ROUGE-2, and the mean length.")
    scores = {}                                           # system -> (n articles, 3) array of R1, R2, RL
    print(f"\n{'system':<10}{'ROUGE-1':>9}{'ROUGE-2':>9}{'ROUGE-L':>9}{'ROUGE-2 95% interval':>24}{'words':>7}")
    for name in SYSTEMS:
        sc = np.array([(d97.rouge_n(h, r, 1), d97.rouge_n(h, r, 2), d97.rouge_l(h, r)) for h, r in zip(out[name], refs)])
        scores[name] = sc; lo, hi = d99.bootstrap_ci(sc[:, 1])
        print(f"{name:<10}{sc[:, 0].mean():>9.3f}{sc[:, 1].mean():>9.3f}{sc[:, 2].mean():>9.3f}      [{lo:.3f}, {hi:.3f}]"
              f"{np.mean([len(h.split()) for h in out[name]]):>7.0f}")
    print("\npaired differences in ROUGE-2 (Day 99's paired bootstrap): mean, 95% interval, share of resamples ahead")
    for a, b in (("lead-3", "TextRank"), ("lead-3", "T5-small"), ("T5-small", "TextRank"), ("oracle", "lead-3")):
        m, lo, hi, share = d99.paired_bootstrap(scores[a][:, 1], scores[b][:, 1])
        print(f"  {a:<9} minus {b:<9} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  ahead in {share:.3f}")
    print("Actual output from this lesson's run.")
    return scores                                           # the plot uses them


def demo_faithfulness(arts, out):
    print("\n" + "=" * 70)
    print("4. WHAT ROUGE CANNOT SEE -- new wording and unsupported details")
    print("=" * 70)
    print("Novel 3-grams: share of the summary's word triples that never occur in the article (0 = pure copying). "
          "Unsupported: numbers, and words of four or more letters, in the summary whose first five letters match no word "
          "of the article (lowercase, letters only, since T5-small often lowercases words; 'suffered' counts as supported by 'suffer').")
    print(f"\n{'system':<10}{'novel 3-grams':>15}{'summaries with an unsupported detail':>40}")
    for name in ("lead-3", "TextRank", "T5-small"):
        nov = np.mean([novel_trigrams(h, a) for h, a in zip(out[name], arts)])
        uns = [unsupported(h, a) for h, a in zip(out[name], arts)]
        print(f"{name:<10}{nov:>15.3f}{sum(bool(u) for u in uns):>40}")
    found = [(i, u) for i, u in enumerate(unsupported(h, a) for h, a in zip(out["T5-small"], arts)) if u]
    print(f"\nT5-small's unsupported details, all {len(found)} cases: " + "; ".join(f"article {i}: {sorted(u)}" for i, u in found))
    print("\neach case: the T5 sentence holding the first flagged item, and the article's opening")
    for i, u in found:                                    # the evidence, so each case can be judged by reading
        item = sorted(u)[0]
        sent = next((x for x in re.split(r"(?<=[.!?])\s+", out["T5-small"][i]) if item in x.lower()), out["T5-small"][i])
        print(f"  article {i:>2} [{item}] T5: {sent[:150]}")
        print(f"             article: {arts[i][:150].replace(chr(10), ' ')}")
    print("Actual output from this lesson's run.")


def plot_results(scores, out, refs):                    # two panels: ROUGE by system | ROUGE-2 per article, lead-3 vs T5
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))           # canvas
    x = np.arange(len(SYSTEMS))
    for j, (lab, col) in enumerate((("ROUGE-1", "#2ca02c"), ("ROUGE-2", "#1f77b4"), ("ROUGE-L", "#ff7f0e"))):
        ax[0].bar(x + (j - 1) * 0.27, [scores[s][:, j].mean() for s in SYSTEMS], width=0.27, color=col, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(SYSTEMS); ax[0].set_ylabel("mean F1 (100 articles)")
    ax[0].set_title("ROUGE against the reference highlights", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
    ax[1].scatter(scores["lead-3"][:, 1], scores["T5-small"][:, 1], s=14, color="#9467bd")
    ax[1].plot([0, 1], [0, 1], ":", color="#7f7f7f"); lim = max(scores["lead-3"][:, 1].max(), scores["T5-small"][:, 1].max()) + 0.05
    ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim); ax[1].set_xlabel("lead-3 ROUGE-2"); ax[1].set_ylabel("T5-small ROUGE-2")
    ax[1].set_title("Per article: lead-3 against T5-small", fontsize=10); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("summarization_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved summarization_results.png")


def draw_diagram():                                     # extractive vs abstractive
    fig, ax = plt.subplots(figsize=(13, 4.0))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.0); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 3.75, "Day 101: extractive vs abstractive summarization", ha="center", fontsize=12, fontweight="bold")
    box(1.5, 1.8, 2.4, 1.2, "news article\n(~700 words)\nsplit into sentences", "#F2F2F2")
    rows = [(3.0, "lead-3: first 3 sentences", "#FAEEDA"), (2.2, "TextRank: TF-IDF graph + PageRank", "#FAEEDA"),
            (1.4, "T5-small: writes new text", "#DCEBF7"), (0.6, "oracle: best 3 (peeks at reference)", "#EEEDFE")]
    for y, t, c in rows:                                             # the four systems
        box(5.6, y, 3.6, 0.6, t, c, fs=8)
        ax.annotate("", xy=(3.78, y), xytext=(2.72, 1.8), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(8.9, 1.8), xytext=(7.42, y), arrowprops=dict(arrowstyle="->", color="#333333"))
    box(10.9, 1.8, 3.8, 1.5, "score each summary:\nROUGE-1/2/L vs highlights (Day 97)\nnovel 3-grams\nunsupported numbers and words", "#E5F2E1")
    plt.savefig("summarization_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved summarization_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 102 is question answering: a pretrained model that points at the answer inside a paragraph, measured on "
          "real SQuAD questions with the official exact-match and F1, written from scratch.")


def main():                                             # runs every section in order
    arts, refs = load_articles()                           # the data
    S = demo_data(arts, refs)                              # 1. the data
    out = demo_systems(arts, refs, S)                      # 2. the summaries
    scores = demo_rouge(refs, out)                         # 3. ROUGE
    demo_faithfulness(arts, out)                           # 4. faithfulness
    plot_results(scores, out, refs)                        # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 5. Day 102


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
