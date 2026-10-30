"""
Day 104: The Week 15 Build -- A Small Retrieval-Augmented Question-Answering
System From the Week's Parts

Week 15's last day does two jobs, like Days 83, 90 and 97. The syllabus topic
is a tiny retrieval-augmented generation (RAG) pipeline on real documents:
search a collection for passages relevant to a question, then have a
pretrained model answer FROM those passages. And it is the week's integration
day: the pipeline is assembled from the six exercise scripts of Days 98-103.

The parts, and where each one goes:

  Day 98 exercise   batched, run_batched, softmax              running the encoder and the readers a batch at a time
  Day 99 exercise   bootstrap_ci, paired_bootstrap             every comparison's uncertainty
  Day 100 exercise  spans, prf                                 answer types (from a NER model's tags); retrieval as sets
  Day 101 exercise  split_sentences, chunk_words               cutting paragraphs into passages
  Day 102 exercise  normalize, exact_match, f1                 scoring every answer
  Day 103 exercise  words, mean_pool, cosine_top_k, BM25       the two retrievers

The documents: every paragraph of SQuAD v1.1's validation set (48 Wikipedia
articles), cut into overlapping passages. The questions: 500 of Day 102's
1,000. Three pretrained models, all used before this week:
  sentence-transformers/all-MiniLM-L6-v2            the dense retriever (Day 103)
  distilbert/distilbert-base-cased-distilled-squad  an extractive reader (Day 102)
  google/flan-t5-small                              a generative reader: an instruction-tuned T5 (new today)

What is measured, reported as it came out:
  1. The build: the parts fit; the passages; the index.
  2. Retrieval: does a top-k passage contain the answer? BM25, dense, hybrid.
  3. End to end: closed-book (no passages), RAG with each reader, and the
     oracle (the right paragraph handed over) as an upper bound.
  4. Where RAG fails: retrieval missed it, or the reader did.
  5. How many passages: k = 1, 3, 5 for the generative reader.
  6. Which answers are hardest: by entity type, from a NER model (Day 100).

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: words (the parts) and question types
import string                                          # string: punctuation (Day 102's normalize)
import sys                                             # sys: path, to import Day 102
import time                                            # time: perf_counter, for timings
from collections import Counter                        # Counter: term counts (BM25, F1)

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import torch                                           # torch: the pretrained models run on it
from datasets import disable_progress_bars, load_dataset  # datasets: SQuAD from the local cache
from transformers import (AutoModel, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,  # the library's entry points
                          AutoModelForTokenClassification, AutoTokenizer)
from transformers.utils import logging as hf_logging   # the library's logging
from huggingface_hub.utils import logging as hub_logging  # the model hub's logging
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 102 sits here
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day102_question_answering as d102            # Day 102: the 1,000 questions and its reader's span search
if hasattr(sys.stdout, "reconfigure"):                  # (not when another script imports this one quietly)
    sys.stdout.reconfigure(encoding="utf-8")            # Wikipedia's accents print as UTF-8 on Windows too
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()  # no progress bars or notices in the output
hub_logging.set_verbosity_error(); disable_progress_bars()            # the same for the hub and datasets
torch.set_num_threads(4)                                # CPU threads

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"      # the dense retriever (Day 103)
READER = "distilbert/distilbert-base-cased-distilled-squad"  # the extractive reader (Day 102)
GENERATOR = "google/flan-t5-small"                      # the generative reader
NER = "dslim/distilbert-NER"                            # answer types (Day 100)
N_Q = 500                                               # questions (the first 500 of Day 102's 1,000)
CHUNK, OVERLAP = 100, 25                                # passage length and overlap, in words
K = 3                                                   # passages handed to a reader
PROMPT = "Answer the question using the passages.\n\n{ctx}\n\nQuestion: {q}\nAnswer:"  # the generator's instruction
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "gen", "gov", "sen", "rep", "lt", "col", "sgt", "capt", "u.s",
          "u.k", "no", "vs", "inc", "co", "corp", "ltd", "jan", "feb", "mar", "apr", "aug", "sept", "oct", "nov", "dec"}  # from day101_exercise.py, verbatim
STOP = {"the", "a", "an", "of", "in", "to", "is", "was", "and", "for", "on", "by", "with", "as", "at", "that", "from", "are",
        "were", "it", "its", "be", "this", "or", "which", "what", "who", "when", "where", "how", "why", "did", "does", "do"}  # from day103_exercise.py, verbatim


# =============================================================================
# 1. THE PARTS -- Week 15's exercise scripts, verbatim
# =============================================================================
# ---- from day98_exercise.py, verbatim ----
def batched(items, size):                               # NEW: consecutive chunks of at most `size` items
    for i in range(0, len(items), size):                  # chunk starts
        yield items[i:i + size]                             # one chunk


def run_batched(fn, items, size):                       # NEW: fn on every chunk, results stacked in order
    return np.concatenate([np.asarray(fn(chunk)) for chunk in batched(items, size)], axis=0)


def softmax(z):                                         # NEW: probabilities over the last axis
    z = z - z.max(axis=-1, keepdims=True)                 # subtracting the max changes nothing mathematically, prevents overflow
    e = np.exp(z)                                         # now every exponent is at most 1
    return e / e.sum(axis=-1, keepdims=True)


# ---- from day99_exercise.py, verbatim ----
def bootstrap_ci(values, n_boot=2000, seed=0):          # NEW: middle 95% of the mean over resamples of the examples
    rng = np.random.RandomState(seed)                     # fixed rng
    idx = rng.randint(0, len(values), (n_boot, len(values)))  # each row: one resample, with replacement
    means = np.asarray(values, float)[idx].mean(axis=1)   # the mean of each resample
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired_bootstrap(a, b, n_boot=2000, seed=0):        # NEW: the difference, resampling examples for both systems together
    a, b = np.asarray(a, float), np.asarray(b, float)     # per-example scores of the two systems
    rng = np.random.RandomState(seed)                     # fixed rng
    idx = rng.randint(0, len(a), (n_boot, len(a)))        # the SAME resampled examples for both
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)         # difference per resample
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


# ---- from day100_exercise.py, verbatim ----
def spans(tags):                                        # NEW: BIO tags -> set of (start, end, type)
    out, start, typ = set(), None, None                   # finished spans; the open span, if any
    for i, t in enumerate(tags + ["O"]):                  # an extra O closes a span that runs to the end
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):  # anything that ends the open span
            if start is not None:
                out.add((start, i, typ))
            start, typ = (i, t[2:]) if t != "O" else (None, None)  # B-X (or a stray I-X) opens a new span
    return out


def prf(gold, pred):                                    # NEW: precision, recall, F1 from lists of span sets
    tp = sum(len(g & p) for g, p in zip(gold, pred))      # exact matches
    n_pred, n_gold = sum(map(len, pred)), sum(map(len, gold))
    p, r = tp / max(1, n_pred), tp / max(1, n_gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---- from day101_exercise.py, verbatim ----
def split_sentences(text):                              # NEW: sentence boundaries by rule (Day 101's, unchanged)
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'‘“(]?[A-Z0-9])", text.strip())  # a stop, space, then a capital or digit
    out = []                                              # sentences, with false splits joined back
    for p in parts:
        prev = out[-1].split()[-1].lower().rstrip(".") if out else ""  # the word before the full stop
        if out and (prev in ABBREV or re.fullmatch(r"[a-z]", prev)):  # 'Mr.' or an initial 'J.' is not a sentence end
            out[-1] += " " + p
        else:
            out.append(p)
    return [s.strip() for s in out if len(s.split()) >= 3]  # drop fragments of one or two words


def chunk_words(text, size, overlap):                   # NEW: overlapping windows of words
    w = text.split()                                      # the words
    step = size - overlap                                 # how far each window moves
    starts = list(range(0, max(1, len(w) - overlap), step))  # a window starts every `step` words ...
    return [" ".join(w[s:s + size]) for s in starts]      # ... and the last one may be shorter


# ---- from day102_exercise.py, verbatim ----
def normalize(s):                                       # NEW: lowercase, drop punctuation, drop a/an/the, single spaces
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, golds):                           # NEW: any gold answer, after normalising both
    return float(any(normalize(pred) == normalize(g) for g in golds))


def f1(pred, golds):                                    # NEW: the best word-overlap F1 over the gold answers
    def one(p, g):
        p, g = normalize(p).split(), normalize(g).split()
        common = sum((Counter(p) & Counter(g)).values())    # shared words, with multiplicity
        if common == 0:
            return 0.0
        prec, rec = common / len(p), common / len(g)
        return 2 * prec * rec / (prec + rec)
    return max(one(pred, g) for g in golds)


# ---- from day103_exercise.py, verbatim ----
def words(text):                                        # lowercase word tokens, stop words removed
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOP]


def mean_pool(hidden, mask):                            # NEW: average the token vectors, ignoring padding
    m = mask.unsqueeze(-1).float()                        # (batch, tokens, 1)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def cosine_top_k(Q, D, k):                              # NEW: indices and scores of the k best rows of D for each row of Q
    S = Q @ D.T                                           # cosine, since rows have unit length
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]       # the k best, in no particular order (faster than a full sort)
    order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)  # now sort just those k
    idx = np.take_along_axis(idx, order, 1)
    return idx, np.take_along_axis(S, idx, 1)


class BM25:                                             # NEW: Okapi BM25 (Day 103's, unchanged)
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tf = [Counter(words(d)) for d in docs]         # term counts per document
        self.len = np.array([sum(t.values()) for t in self.tf])  # document lengths
        self.avg, self.k1, self.b, self.N = self.len.mean(), k1, b, len(docs)
        df = Counter(w for t in self.tf for w in t)         # documents containing each word
        self.idf = {w: np.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, query):                              # one score per document
        s = np.zeros(self.N)
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for w in set(words(query)):
            if w in self.idf:
                f = np.array([t.get(w, 0) for t in self.tf])
                s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return s


# =============================================================================
# 2. THE PIPELINE -- NEW
# =============================================================================
def load():                                             # the documents and the questions
    d = load_dataset("rajpurkar/squad", split="validation")
    docs = list(dict.fromkeys(d["context"]))              # every distinct paragraph
    Q, _ = d102.load_questions()                          # Day 102's 1,000 random questions ...
    return docs, Q[:N_Q]                                  # ... the first 500 of them


def word_passages(docs):                                # overlapping word windows (day101_exercise.py's chunk_words), with their paragraph
    out = [(p, i) for i, d in enumerate(docs) for p in chunk_words(d, CHUNK, OVERLAP)]
    return [p for p, _ in out], np.array([i for _, i in out])


def sentence_passages(docs):                            # whole sentences packed up to CHUNK words (day101_exercise.py's split_sentences)
    out = []
    for i, d in enumerate(docs):
        cur = []                                            # sentences in the passage being built
        for s in split_sentences(d) or [d]:
            if cur and len(" ".join(cur + [s]).split()) > CHUNK:  # the next sentence would overflow: close this passage
                out.append((" ".join(cur), i)); cur = []
            cur.append(s)
        if cur:
            out.append((" ".join(cur), i))
    return [p for p, _ in out], np.array([i for _, i in out])


def encode(tok, enc, texts):                            # unit vectors, a batch at a time (day98_exercise.py's run_batched, day103_exercise.py's mean_pool)
    def one(chunk):
        b = tok(list(chunk), padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            v = mean_pool(enc(**b).last_hidden_state, b["attention_mask"])
        return torch.nn.functional.normalize(v, dim=-1).numpy()
    return run_batched(one, texts, 64)


def rrf(*score_mats, k=60):                             # reciprocal rank fusion (Day 103's, rewritten here as new code)
    return sum(1.0 / (k + np.argsort(np.argsort(-S, axis=1), axis=1) + 1) for S in score_mats)


def contains(passage, golds):                           # does the passage contain a gold answer? (day102_exercise.py's normalize)
    p = normalize(passage)
    return any(normalize(g) in p for g in golds)


def read_extractive(tok, model, q, passages):           # Day 102's span search in each passage; the best overall, and softmax over passages
    found = [d102.answer(tok, model, q, p) for p in passages]  # (text, score) per passage
    probs = softmax(np.array([s for _, s in found]))      # day98_exercise.py's softmax: how sure it is which passage holds the answer
    k = int(np.argmax(probs))
    return found[k][0], float(probs[k])


def fit_prompt(gtok, q, passages, limit=512):           # NEW: cut the PASSAGES (never the question) so the prompt fits T5's input
    rest = len(gtok(PROMPT.format(ctx="", q=q))["input_ids"])  # tokens used by the instruction and the question
    ids = gtok("\n\n".join(passages), add_special_tokens=False)["input_ids"]  # the passages' tokens
    cut = len(ids) > limit - rest                         # do they have to be shortened?
    ctx = gtok.decode(ids[:limit - rest]) if cut else "\n\n".join(passages)  # keep the first tokens that fit
    return PROMPT.format(ctx=ctx, q=q), cut


def generate(gtok, gen, prompts):                       # flan-t5's answers, a batch at a time (day98_exercise.py's run_batched)
    def one(chunk):
        b = gtok(list(chunk), padding=True, truncation=True, max_length=512, return_tensors="pt")  # T5 reads at most 512 tokens
        with torch.no_grad():
            ids = gen.generate(**b, max_new_tokens=24, num_beams=1)  # greedy, short answers
        return np.array(gtok.batch_decode(ids, skip_special_tokens=True), dtype=object)
    return list(run_batched(one, prompts, 16))


def answer_type(ntok, nmodel, answers):                 # entity type of each gold answer, from Day 100's NER model and spans()
    out = []
    for a in answers:
        w = a.split()
        with torch.no_grad():
            pred = nmodel(**ntok(w, is_split_into_words=True, return_tensors="pt")).logits.argmax(-1)[0].numpy()
        ids = ntok(w, is_split_into_words=True).word_ids()
        tags, seen = [], set()                              # one tag per word, from its first piece (Day 100's rule)
        for pos, wi in enumerate(ids):
            if wi is not None and wi not in seen:
                tags.append(nmodel.config.id2label[int(pred[pos])]); seen.add(wi)
        sp = spans(tags)                                    # day100_exercise.py's spans
        out.append(max(sorted(sp), key=lambda s: s[1] - s[0])[2] if sp else "not an entity")  # the longest entity (the earliest, if tied)
    return out


def scores(preds, Q):                                   # EM and F1 per question (day102_exercise.py)
    golds = [q["answers"]["text"] for q in Q]
    return (np.array([exact_match(p, g) for p, g in zip(preds, golds)]), np.array([f1(p, g) for p, g in zip(preds, golds)]))


# =============================================================================
# DEMOS
# =============================================================================
def demo_build(docs, Q):
    print("=" * 70)
    print("1. THE BUILD -- the parts, the passages, the index")
    print("=" * 70)
    print("Every function in section 1 is copied verbatim from a Week 15 exercise script (a banner names each one); Day 102's "
          "span search and question sample are imported.")
    P, pid = word_passages(docs); SP, spid = sentence_passages(docs)
    print(f"documents: {len(docs):,} SQuAD validation paragraphs ({np.mean([len(d.split()) for d in docs]):.0f} words on average, "
          f"{np.mean([len(split_sentences(d)) for d in docs]):.1f} sentences)")
    print(f"word passages: {len(P):,} windows of {CHUNK} words overlapping by {OVERLAP} (chunk_words); "
          f"sentence passages: {len(SP):,}, whole sentences up to {CHUNK} words (split_sentences)")
    print(f"questions: {len(Q)}; a gold answer appears in at least one word passage for {np.mean([any(contains(p, q['answers']['text']) for p in [P[i] for i in np.where(pid == docs.index(q['context']))[0]]) for q in Q]):.3f} of them "
          "(an answer can be cut by a window edge, but the overlap usually saves it)")
    print("Actual output from this lesson's run.")
    return (P, pid), (SP, spid)                              # later sections use them


def demo_retrieval(docs, Q, word, sent, tok, enc):
    print("\n" + "=" * 70)
    print("2. RETRIEVAL -- is the answer in the top k passages?")
    print("=" * 70)
    qs = [q["question"] for q in Q]; golds = [q["answers"]["text"] for q in Q]
    res, top = {}, {}                                      # (passage type, method) -> recall at 1, 3, 5, 10; and the top-10 indices
    t0 = time.perf_counter(); Qv = encode(tok, enc, qs)
    for pname, (P, _) in (("word", word), ("sentence", sent)):
        Pv = encode(tok, enc, P)                            # dense index
        bm = BM25(P)                                        # sparse index (day103_exercise.py)
        S_b = np.array([bm.scores(q) for q in qs])          # BM25 score of every passage for every question
        idx_d, _ = cosine_top_k(Qv, Pv, 10)                 # dense: day103_exercise.py's cosine_top_k
        for mname, idx in (("BM25", np.argsort(-S_b, axis=1)[:, :10]), ("dense", idx_d),
                           ("hybrid", np.argsort(-rrf(S_b, Qv @ Pv.T), axis=1)[:, :10])):  # the ten best passages per question
            hit = np.array([[contains(P[j], g) for j in row] for row, g in zip(idx, golds)])  # does passage j contain an answer?
            res[(pname, mname)] = [hit[:, :k].any(1).mean() for k in (1, 3, 5, 10)]; top[(pname, mname)] = idx
    secs = time.perf_counter() - t0
    print(f"answer-containing passage within the top k (indexing and searching both passage types: {secs:.0f} s)")
    print(f"\n{'passages':<10}{'retriever':<9}" + "".join(f"{'k=' + str(k):>8}" for k in (1, 3, 5, 10)))
    for (pname, mname), r in res.items():
        print(f"{pname:<10}{mname:<9}" + "".join(f"{x:>8.3f}" for x in r))
    P, pid = word
    gold_sets = [{(j, 0, "P") for j in range(len(P)) if contains(P[j], g)} for g in golds]  # every passage that holds an answer
    pred_sets = [{(int(j), 0, "P") for j in row[:K]} for row in top[("word", "hybrid")]]
    p, r, f = prf(gold_sets, pred_sets)                   # day100_exercise.py's prf, on passage sets
    print(f"\nword passages, hybrid, top {K} as a set (day100_exercise.py's prf): precision {p:.3f}, recall {r:.3f} "
          f"(of {sum(map(len, gold_sets)):,} answer-holding passages; several can hold one answer)")
    print("Actual output from this lesson's run.")
    return res, top                                          # later sections use them


def demo_end_to_end(Q, word, top, docs):
    print("\n" + "=" * 70)
    print("3. END TO END -- closed book, RAG, oracle")
    print("=" * 70)
    P, _ = word
    rtok, reader = AutoTokenizer.from_pretrained(READER), AutoModelForQuestionAnswering.from_pretrained(READER).eval()
    gtok, gen = AutoTokenizer.from_pretrained(GENERATOR), AutoModelForSeq2SeqLM.from_pretrained(GENERATOR).eval()
    print(f"{GENERATOR}: {sum(p.numel() for p in gen.parameters()):,} weights; prompt: {PROMPT!r}")
    ctx = [[P[j] for j in row[:K]] for row in top[("word", "hybrid")]]  # each question's top-K word passages, hybrid retriever
    jobs = {"closed book (flan-t5)": lambda: generate(gtok, gen, [f"Question: {q['question']}\nAnswer:" for q in Q]),
            "RAG, flan-t5 reader": lambda: generate(gtok, gen, [fit_prompt(gtok, q["question"], c)[0] for q, c in zip(Q, ctx)]),
            "RAG, extractive reader": lambda: [read_extractive(rtok, reader, q["question"], c) for q, c in zip(Q, ctx)],
            "oracle paragraph, flan-t5": lambda: generate(gtok, gen, [fit_prompt(gtok, q["question"], [q["context"]])[0] for q in Q]),
            "oracle paragraph, extractive": lambda: [d102.answer(rtok, reader, q["question"], q["context"])[0] for q in Q]}
    runs, t = {}, {}                                       # system -> answers; system -> seconds
    for name, job in jobs.items():                         # each system over all questions
        t0 = time.perf_counter(); runs[name] = job(); t[name] = time.perf_counter() - t0
    ext = runs["RAG, extractive reader"]                   # (answer, passage probability) pairs ...
    runs["RAG, extractive reader"] = [a for a, _ in ext]   # ... keep the answers for scoring
    print(f"\n{'system':<31}{'exact match':>12}{'F1':>8}{'F1 95% interval':>20}{'s per question':>16}")
    S = {}
    for name, preds in runs.items():
        em, f = scores(preds, Q); S[name] = (em, f); lo, hi = bootstrap_ci(f)
        print(f"{name:<31}{em.mean():>12.3f}{f.mean():>8.3f}      [{lo:.3f}, {hi:.3f}]{t[name] / len(Q):>16.2f}")
    print("\npaired differences in F1 (day99_exercise.py): mean, 95% interval, share of resamples ahead")
    for a, b in (("RAG, flan-t5 reader", "closed book (flan-t5)"), ("RAG, extractive reader", "RAG, flan-t5 reader"),
                 ("oracle paragraph, flan-t5", "RAG, flan-t5 reader"), ("oracle paragraph, extractive", "RAG, extractive reader")):
        m, lo, hi, share = paired_bootstrap(S[a][1], S[b][1])
        print(f"  {a:<29} minus {b:<22} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] ahead in {share:.3f}")
    print("\nthree questions, every system:")
    for i in (0, 1, 2):                                   # the first three questions
        print(f"  Q: {Q[i]['question']}   gold: {Q[i]['answers']['text'][0]!r}")
        for n in runs:
            print(f"     {n:<29} {runs[n][i][:60]!r}")
    print("Actual output from this lesson's run.")
    return runs, S, ext, (rtok, reader, gtok, gen)          # later sections use them


def demo_failures(Q, word, top, S, ext):
    print("\n" + "=" * 70)
    print("4. WHERE RAG FAILS -- retrieval or reading?")
    print("=" * 70)
    P, _ = word
    golds = [q["answers"]["text"] for q in Q]
    found = np.array([any(contains(P[j], g) for j in row[:K]) for row, g in zip(top[("word", "hybrid")], golds)])  # answer in the passages?
    print(f"the top {K} passages hold a gold answer for {found.mean():.3f} of questions")
    for name in ("RAG, flan-t5 reader", "RAG, extractive reader"):
        f = S[name][1]; wrong = f == 0                      # answers with no shared word at all
        print(f"  {name:<23}: F1 = 0 on {wrong.sum()} questions -> passages lacked the answer {np.sum(wrong & ~found)}, "
              f"reader missed it although present {np.sum(wrong & found)}; F1 when the answer was retrieved {f[found].mean():.3f}, "
              f"when it was not {f[~found].mean():.3f}")
    conf = np.array([c for _, c in ext]); em = S["RAG, extractive reader"][0]
    print(f"  extractive reader's passage probability (softmax over {K} passages, so never below 1/{K}):")
    for lo_, hi_ in ((1 / K, 0.7), (0.7, 0.95), (0.95, 1.01)):  # three confidence bands
        m = (conf >= lo_) & (conf < hi_)                    # the questions in this band
        if m.any():                                         # an empty band has no exact match to report
            print(f"    [{lo_:.2f}, {min(hi_, 1.0):.2f}): {m.sum()} questions, exact match {em[m].mean():.3f}")
    print("Actual output from this lesson's run.")
    return found                                             # the plot uses it


def demo_k(Q, word, top, models):
    print("\n" + "=" * 70)
    print("5. HOW MANY PASSAGES? -- k = 1, 3, 5, flan-t5 reader")
    print("=" * 70)
    P, _ = word; _, _, gtok, gen = models
    rows = []
    for k in (1, 3, 5):
        fitted = [fit_prompt(gtok, q["question"], [P[j] for j in row[:k]]) for q, row in zip(Q, top[("word", "hybrid")])]
        cut = np.mean([c for _, c in fitted])               # share of prompts whose passages had to be shortened
        em, f = scores(generate(gtok, gen, [p for p, _ in fitted]), Q); lo, hi = bootstrap_ci(f); rows.append((k, em.mean(), f.mean(), lo, hi))
        print(f"  k = {k}: exact match {em.mean():.3f}, F1 {f.mean():.3f} [{lo:.3f}, {hi:.3f}]; passages shortened to fit 512 tokens: {cut:.3f}")
    print("Actual output from this lesson's run.")
    return rows                                              # the plot uses them


def demo_types(Q, S):
    print("\n" + "=" * 70)
    print("6. WHICH ANSWERS ARE HARDEST? -- by entity type (Day 100's NER model)")
    print("=" * 70)
    ntok, nmodel = AutoTokenizer.from_pretrained(NER), AutoModelForTokenClassification.from_pretrained(NER).eval()
    T = np.array(answer_type(ntok, nmodel, [q["answers"]["text"][0] for q in Q]))  # the first gold answer's type
    print(f"{'answer type':<15}{'questions':>10}" + "".join(f"{n.replace('oracle paragraph', 'oracle').replace(' reader', '').replace(' (flan-t5)', ''):>20}" for n in S))
    for typ in sorted(set(T), key=lambda x: -np.sum(T == x)):
        m = T == typ
        print(f"{typ:<15}{m.sum():>10}" + "".join(f"{S[n][1][m].mean():>20.3f}" for n in S))
    print("(mean F1 per type; numbers, dates and phrases are 'not an entity')")
    print("Actual output from this lesson's run.")


def plot_results(res, S, krows):                        # three panels: retrieval | end to end | k
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))           # canvas
    ks = (1, 3, 5, 10)
    for (pname, mname), r in res.items():
        ax[0].plot(ks, r, marker="o", linestyle="-" if pname == "word" else "--", label=f"{mname}, {pname} passages")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("answer in the top k"); ax[0].set_xticks(ks); ax[0].set_ylim(0.4, 1.0)
    ax[0].set_title("Retrieval", fontsize=10); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    names = list(S); x = np.arange(len(names))
    ax[1].bar(x - 0.2, [S[n][0].mean() for n in names], width=0.4, color="#ff7f0e", label="exact match")
    ax[1].bar(x + 0.2, [S[n][1].mean() for n in names], width=0.4, color="#1f77b4", label="F1")
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(", ", ",\n").replace(" (", "\n(") for n in names], fontsize=7)
    ax[1].set_ylim(0, 1); ax[1].set_title("End to end (500 SQuAD questions)", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    kk = [r[0] for r in krows]; f = [r[2] for r in krows]
    ax[2].errorbar(kk, f, yerr=[[r[2] - r[3] for r in krows], [r[4] - r[2] for r in krows]], marker="o", capsize=4, color="#2ca02c")
    ax[2].set_xlabel("passages given to flan-t5 (k)"); ax[2].set_ylabel("F1"); ax[2].set_xticks(kk)
    ax[2].set_title("How many passages?", fontsize=10); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("rag_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved rag_results.png")


def draw_diagram():                                     # the parts and the pipeline
    fig, ax = plt.subplots(figsize=(13, 5.0))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.0); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8):                       # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.75, "Day 104: Week 15's build -- retrieval-augmented question answering", ha="center", fontsize=12, fontweight="bold")
    flow = [(1.2, "question"), (3.3, "retrieve top 3\nBM25 + dense\n(RRF)"), (5.6, "passages\n(100-word windows\nof Wikipedia)"),
            (7.9, "read: flan-t5\nor extractive\nDistilBERT"), (10.0, "answer"), (12.0, "EM, F1\n+ bootstrap")]
    for i, (x, t) in enumerate(flow):                                # the pipeline, top row
        box(x, 3.4, 1.8 if i not in (0, 4) else 1.3, 1.1, t, "#DCEBF7" if i in (1, 3) else "#E5F2E1")
        if i:
            px = flow[i - 1][0]; pw = 1.8 if i - 1 not in (0, 4) else 1.3; w = 1.8 if i not in (0, 4) else 1.3
            ax.annotate("", xy=(x - w / 2 - 0.04, 3.4), xytext=(px + pw / 2 + 0.04, 3.4), arrowprops=dict(arrowstyle="->", color="#333333"))
    parts = [(1.2, "Day 98\nbatched,\nsoftmax"), (3.3, "Day 103\nBM25, mean_pool,\ncosine_top_k"), (5.6, "Day 101\nchunk_words,\nsplit_sentences"),
             (7.9, "Day 100\nspans, prf\n(answer types)"), (10.0, "Day 102\nexact_match,\nf1"), (12.0, "Day 99\nbootstrap_ci,\npaired")]
    for x, t in parts:                                               # the exercise parts, bottom row
        box(x, 1.2, 1.8, 1.2, t, "#FAEEDA", fs=7.5)
        ax.annotate("", xy=(x, 2.83), xytext=(x, 1.82), arrowprops=dict(arrowstyle="->", color="#999999"))
    plt.savefig("rag_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved rag_diagram.png")


def note_on_the_week():
    print("\n" + "=" * 70)
    print("7. THE WEEK, AND WHAT COMES NEXT")
    print("=" * 70)
    print("Week 15 used pretrained models through a library: classification (98), fine-tuning (99), NER (100), "
          "summarization (101), question answering (102), search (103) and, today, all of it as one system. Week 16 "
          "returns to classical machine learning: decision trees, written from scratch.")


def main():                                             # runs every section in order
    docs, Q = load()                                       # the data
    word, sent = demo_build(docs, Q)                       # 1. the build
    tok, enc = AutoTokenizer.from_pretrained(ENCODER), AutoModel.from_pretrained(ENCODER).eval()  # the dense retriever
    res, top = demo_retrieval(docs, Q, word, sent, tok, enc)  # 2. retrieval
    runs, S, ext, models = demo_end_to_end(Q, word, top, docs)  # 3. end to end
    demo_failures(Q, word, top, S, ext)                    # 4. failures
    krows = demo_k(Q, word, top, models)                   # 5. k
    demo_types(Q, S)                                       # 6. answer types
    plot_results(res, S, krows)                            # the figure
    draw_diagram()                                         # the diagram
    note_on_the_week()                                     # 7. Week 16


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
