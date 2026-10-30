"""
Day 103: Semantic Search with Embeddings -- Real Vector-Similarity Search

Day 102 handed the model the right paragraph. In practice something has to
FIND it among thousands. Two families of search:

  * keyword (sparse): score paragraphs by the question's words they contain,
    weighted by rarity -- TF-IDF, and BM25 (Robertson et al., the standard).
  * semantic (dense): turn every text into one vector with a pretrained
    sentence-embedding model, and return the paragraphs whose vectors point
    the same way as the question's (cosine similarity).

The collection: every distinct paragraph of SQuAD v1.1's validation set (48
Wikipedia articles). The queries: Day 102's 1,000 questions. A search is right
if it returns the paragraph the question was written about.

  1. Embeddings: sentence-transformers/all-MiniLM-L6-v2 through the plain
     transformers library, with mean pooling written by hand.
  2. Four searches, from scratch except the encoder: TF-IDF, BM25, dense, and
     a hybrid (reciprocal rank fusion of BM25 and dense).
  3. Recall@1/5/10 and MRR, with Day 99's bootstrap and paired comparisons.
  4. When dense search helps: questions that share few words with their
     paragraph -- and a caveat about how SQuAD's questions were written.
  5. Speed and memory.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: words
import sys                                             # sys: path, to import Days 99 and 102
import time                                            # time: perf_counter, for timings
from collections import Counter                        # Counter: term counts

import numpy as np                                     # numpy: the from-scratch parts
import torch                                           # torch: the encoder runs on it
from datasets import disable_progress_bars             # datasets: its progress bars (SQuAD is loaded through Day 102)
from transformers import AutoModel, AutoTokenizer      # the library's entry points
from transformers.utils import logging as hf_logging   # the library's logging
from huggingface_hub.utils import logging as hub_logging  # the model hub's logging
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Days 99 and 102 sit here
with contextlib.redirect_stdout(io.StringIO()):         # the imports print nothing
    import day99_exercise as d99                        # Day 99: bootstrap_ci, paired_bootstrap
    import day102_question_answering as d102            # Day 102: the same 1,000 questions
if hasattr(sys.stdout, "reconfigure"):                  # (not when another script imports this one quietly)
    sys.stdout.reconfigure(encoding="utf-8")            # the articles' curly quotes and accents print as UTF-8 on Windows too
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()  # no progress bars or notices in the output
hub_logging.set_verbosity_error(); disable_progress_bars()            # the same for the hub and datasets
torch.set_num_threads(4)                                # CPU threads

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"      # a small sentence-embedding model (6 layers, 384 dimensions)
KS = (1, 5, 10)                                         # recall@k
STOP = {"the", "a", "an", "of", "in", "to", "is", "was", "and", "for", "on", "by", "with", "as", "at", "that", "from", "are",
        "were", "it", "its", "be", "this", "or", "which", "what", "who", "when", "where", "how", "why", "did", "does", "do"}


def words(text):                                        # lowercase word tokens, stop words removed
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOP]


# =============================================================================
# DENSE: EMBEDDINGS -- NEW (the pooling and the search; the encoder is pretrained)
# =============================================================================
def mean_pool(hidden, mask):                            # NEW: average the token vectors, ignoring padding
    m = mask.unsqueeze(-1).float()                        # (batch, tokens, 1): 1 for real tokens
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)  # sum of real token vectors / how many


def embed(tok, enc, texts, bs=32, max_len=256):         # NEW: unit-length sentence vectors, (n, 384)
    out = []
    with torch.no_grad():                                 # inference only
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            v = mean_pool(enc(**b).last_hidden_state, b["attention_mask"])  # one vector per text
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())  # length 1, so dot product = cosine
    return np.concatenate(out)


def ranks_of(scores, gold):                             # NEW: the rank (1 = best) of each query's gold paragraph
    g = scores[np.arange(len(gold)), gold]                # the gold paragraph's score
    return 1 + (scores > g[:, None]).sum(1)               # how many paragraphs beat it, plus one


# =============================================================================
# SPARSE: TF-IDF AND BM25 -- NEW
# =============================================================================
class BM25:                                             # NEW: Okapi BM25 over a fixed collection
    def __init__(self, docs, k1=1.5, b=0.75):              # the usual defaults
        self.tf = [Counter(words(d)) for d in docs]         # term counts per paragraph
        self.len = np.array([sum(t.values()) for t in self.tf])  # paragraph lengths (in kept words)
        self.avg, self.k1, self.b, self.N = self.len.mean(), k1, b, len(docs)
        df = Counter(w for t in self.tf for w in t)         # how many paragraphs contain each word
        self.idf = {w: np.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}  # rare words count more

    def scores(self, query):                              # one score per paragraph
        s = np.zeros(self.N)
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)  # long paragraphs need more matches
        for w in set(words(query)):                         # each distinct query word
            if w in self.idf:
                f = np.array([t.get(w, 0) for t in self.tf])
                s += self.idf[w] * f * (self.k1 + 1) / (f + norm)  # saturating: the 5th occurrence adds less than the 1st
        return s


def tfidf_matrix(docs, queries):                        # NEW: unit-length TF-IDF vectors for paragraphs and questions
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in words(d)}))}
    def vec(texts):
        M = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            for w, c in Counter(words(t)).items():
                if w in vocab:
                    M[r, vocab[w]] = c
        return M
    D, Q = vec(docs), vec(queries)
    idf = np.log(len(docs) / (1 + (D > 0).sum(0))) + 1    # from the paragraphs only
    D, Q = D * idf, Q * idf
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12), Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)


def rrf(*score_mats, k=60):                             # NEW: reciprocal rank fusion -- sum of 1 / (k + rank) over systems
    total = 0.0
    for S in score_mats:
        r = np.argsort(np.argsort(-S, axis=1), axis=1) + 1  # rank of every paragraph for every query
        total = total + 1.0 / (k + r)
    return total


# =============================================================================
# DEMOS
# =============================================================================
def collection():                                       # every distinct validation paragraph, and each question's gold index
    from datasets import load_dataset                     # imported here: only this function needs it
    d = load_dataset("rajpurkar/squad", split="validation")
    docs = list(dict.fromkeys(d["context"]))              # distinct paragraphs, in order of first appearance
    titles = {c: t for c, t in zip(d["context"], d["title"])}
    Q, _ = d102.load_questions()                          # Day 102's 1,000 questions
    index = {c: i for i, c in enumerate(docs)}
    return docs, [titles[c] for c in docs], [q["question"] for q in Q], np.array([index[q["context"]] for q in Q])


def demo_embeddings(tok, enc):
    print("=" * 70)
    print("1. EMBEDDINGS -- one vector per text, by mean pooling")
    print("=" * 70)
    n = sum(p.numel() for p in enc.parameters())
    print(f"{ENCODER}: {n:,} weights, {enc.config.num_hidden_layers} layers, vectors of {enc.config.hidden_size}")
    texts = ["How many people live in Paris?", "What is the population of the French capital?",
             "Paris is home to about two million residents.", "The cat sat on the mat."]
    V = embed(tok, enc, texts)
    print("cosine similarities (rows and columns in this order):")
    for t, row in zip(texts, V @ V.T):
        print(f"  {t[:46]:<48}" + " ".join(f"{x:6.3f}" for x in row))
    b = tok(texts[:1], return_tensors="pt"); h = enc(**b).last_hidden_state.detach()
    print(f"mean pooling by hand vs the [CLS] vector, cosine with each other: "
          f"{torch.nn.functional.cosine_similarity(mean_pool(h, b['attention_mask']), h[:, 0]).item():.3f} (they are different summaries)")
    print("Actual output from this lesson's run.")


def demo_search(tok, enc, docs, titles, queries, gold):
    print("\n" + "=" * 70)
    print("2. FOUR SEARCHES OVER EVERY SQUAD VALIDATION PARAGRAPH")
    print("=" * 70)
    print(f"collection: {len(docs):,} paragraphs from {len(set(titles))} articles, {np.mean([len(d.split()) for d in docs]):.0f} words "
          f"on average; queries: {len(queries):,} questions; the right answer is the paragraph each was written about")
    t0 = time.perf_counter(); D = embed(tok, enc, docs); t_docs = time.perf_counter() - t0
    t0 = time.perf_counter(); Qv = embed(tok, enc, queries); t_q = time.perf_counter() - t0
    long_docs = sum(len(tok(d)["input_ids"]) > 256 for d in docs)
    print(f"embedding: paragraphs {t_docs:.0f} s, questions {t_q:.1f} s (4 threads); paragraphs longer than the encoder's "
          f"256-token limit, so cut: {long_docs}")
    bm = BM25(docs); t0 = time.perf_counter(); S_bm = np.array([bm.scores(q) for q in queries]); t_bm = time.perf_counter() - t0
    Dt, Qt = tfidf_matrix(docs, queries)
    S = {"TF-IDF": Qt @ Dt.T, "BM25": S_bm, "dense (MiniLM)": Qv @ D.T}
    S["hybrid (BM25 + dense, RRF)"] = rrf(S["BM25"], S["dense (MiniLM)"])
    R = {k: ranks_of(v, gold) for k, v in S.items()}      # the gold paragraph's rank for every query, per system
    print(f"BM25 scoring of all questions: {t_bm:.1f} s (pure Python loops)")
    print("Actual output from this lesson's run.")
    return S, R, D                                         # later sections use them


def demo_metrics(R, S, gold, titles):
    print("\n" + "=" * 70)
    print("3. RECALL@K AND MRR")
    print("=" * 70)
    print("recall@k: share of questions whose paragraph is in the top k. MRR: mean of 1 / rank. Intervals: Day 99's bootstrap.")
    print(f"\n{'system':<28}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + f"{'MRR':>8}{'R@1 95% interval':>21}{'same article @1':>17}")
    tarr = np.array(titles)
    for name, r in R.items():
        hit1 = (r == 1).astype(float); lo, hi = d99.bootstrap_ci(hit1)
        top = S[name].argmax(1)                             # the system's first choice
        same = np.mean(tarr[top] == tarr[gold])             # first choice from the right article (even if the wrong paragraph)
        print(f"{name:<28}" + "".join(f"{np.mean(r <= k):>8.3f}" for k in KS) + f"{np.mean(1 / r):>8.3f}      [{lo:.3f}, {hi:.3f}]{same:>17.3f}")
    print("\npaired differences in recall@1 (Day 99): mean, 95% interval, share of resamples ahead")
    for a, b in (("dense (MiniLM)", "BM25"), ("hybrid (BM25 + dense, RRF)", "BM25"), ("hybrid (BM25 + dense, RRF)", "dense (MiniLM)"),
                 ("BM25", "TF-IDF")):
        m, lo, hi, share = d99.paired_bootstrap((R[a] == 1).astype(float), (R[b] == 1).astype(float))
        print(f"  {a:<27} minus {b:<16} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] ahead in {share:.3f}")
    print("Actual output from this lesson's run.")


def demo_overlap(R, queries, docs, gold):
    print("\n" + "=" * 70)
    print("4. WHEN DOES DENSE SEARCH HELP? -- by word overlap with the paragraph")
    print("=" * 70)
    ov = np.array([len(set(words(q)) & set(words(docs[g]))) / max(1, len(set(words(q)))) for q, g in zip(queries, gold)])  # share of the question's words in its paragraph
    cuts = np.percentile(ov, [33.3, 66.7])
    print(f"share of a question's (non-stop) words that appear in its paragraph: mean {ov.mean():.3f}; "
          f"{np.mean(ov == 1):.3f} of questions have every word there")
    print(f"\n{'overlap third':<26}{'questions':>10}" + "".join(f"{n[:14]:>16}" for n in R))
    for name, m in (("low", ov <= cuts[0]), ("middle", (ov > cuts[0]) & (ov <= cuts[1])), ("high", ov > cuts[1])):
        rng_ = f"{name} ({ov[m].min():.2f}-{ov[m].max():.2f})"
        print(f"{rng_:<26}{m.sum():>10}" + "".join(f"{np.mean(r[m] == 1):>16.3f}" for r in R.values()))
    print("(recall@1 in each third)")
    print("Actual output from this lesson's run.")
    return ov, cuts                                         # the plot uses them


def plot_results(R, ov, cuts):                          # two panels: recall@k by system | recall@1 by overlap third
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))           # canvas
    ks = np.arange(1, 21)
    for name, r in R.items():
        ax[0].plot(ks, [np.mean(r <= k) for k in ks], marker="o", markersize=3, label=name)
    ax[0].set_xlabel("k"); ax[0].set_ylabel("recall@k"); ax[0].set_xticks([1, 5, 10, 15, 20])
    ax[0].set_title("Recall@k over the SQuAD validation paragraphs", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    groups = [ov <= cuts[0], (ov > cuts[0]) & (ov <= cuts[1]), ov > cuts[1]]
    x = np.arange(3); w = 0.8 / len(R)
    for j, (name, r) in enumerate(R.items()):
        ax[1].bar(x + (j - (len(R) - 1) / 2) * w, [np.mean(r[m] == 1) for m in groups], width=w, label=name)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["low overlap", "middle", "high overlap"]); ax[1].set_ylabel("recall@1"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Recall@1 by question-paragraph word overlap", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("search_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved search_results.png")


def draw_diagram():                                     # sparse and dense search side by side
    fig, ax = plt.subplots(figsize=(13, 4.2))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 3.95, "Day 103: keyword search and semantic search", ha="center", fontsize=12, fontweight="bold")
    box(1.4, 2.0, 2.2, 1.0, "question", "#F2F2F2")
    box(4.6, 3.0, 3.2, 1.0, "sparse: words x rarity\nTF-IDF cosine, BM25", "#FAEEDA")
    box(4.6, 1.0, 3.2, 1.0, "dense: MiniLM -> mean pool\n-> unit vector (384)", "#DCEBF7")
    box(8.2, 3.0, 2.8, 1.0, "score every paragraph\n(its word counts)", "#FAEEDA")
    box(8.2, 1.0, 2.8, 1.0, "cosine with every\nparagraph vector", "#DCEBF7")
    box(11.5, 2.0, 2.6, 1.3, "rank; fuse by RRF\nrecall@k, MRR\n(gold = its paragraph)", "#E5F2E1")
    for y in (3.0, 1.0):                                             # arrows along both routes
        ax.annotate("", xy=(2.98, y), xytext=(2.52, 2.0), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(6.78, y), xytext=(6.22, y), arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.annotate("", xy=(10.18, 2.0), xytext=(9.62, y), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("search_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved search_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("5. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 104 is Week 15's build: search (today) + read (Day 102) + generate (a pretrained T5) assembled from the "
          "week's exercise parts into a small retrieval-augmented question-answering system, measured end to end.")


def main():                                             # runs every section in order
    tok = AutoTokenizer.from_pretrained(ENCODER); enc = AutoModel.from_pretrained(ENCODER).eval()  # the encoder
    docs, titles, queries, gold = collection()            # the collection and the queries
    demo_embeddings(tok, enc)                              # 1. embeddings
    S, R, D = demo_search(tok, enc, docs, titles, queries, gold)  # 2. the searches
    demo_metrics(R, S, gold, titles)                       # 3. metrics
    ov, cuts = demo_overlap(R, queries, docs, gold)        # 4. overlap
    plot_results(R, ov, cuts)                              # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 5. Day 104


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
