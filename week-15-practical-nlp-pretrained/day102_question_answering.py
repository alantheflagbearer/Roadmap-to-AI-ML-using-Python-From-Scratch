"""
Day 102: Question Answering with a Pretrained Transformer on a Real Dataset

Extractive question answering: given a question and a paragraph that contains
the answer, point at the answer -- a span of the paragraph's own text.

  paragraph: "... The Normans were descended from Norse raiders ... from Denmark,
              Iceland and Norway ..."
  question:  "From which countries did the Norse originate?"
  answer:    "Denmark, Iceland and Norway"

The data is SQuAD v1.1 (Rajpurkar et al., 2016: crowd workers wrote questions
about Wikipedia paragraphs): 1,000 questions drawn at random from its 10,570
validation questions, each with up to several acceptable answers. The model is
distilbert/distilbert-base-cased-distilled-squad, fine-tuned by its authors on
SQuAD's training set (whose paragraphs come from different articles).

  1. How the model answers: two scores per token (start, end); the best valid
     span, searched from scratch over overlapping windows of long paragraphs,
     and mapped back to characters. (transformers 5 removed its QA pipeline.)
  2. SQuAD's official metrics, from scratch: exact match and word-overlap F1
     after normalising (lowercase, no punctuation, no a/an/the).
  3. A from-scratch baseline: the paragraph sentence that shares the most
     words with the question.
  4. Where it fails: by question type.
  5. The wrong paragraph: the same questions with an unrelated paragraph. The
     model always answers something; can its confidence tell? (This matters
     for Day 104, where a retriever may fetch the wrong paragraph.)

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported scripts quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: normalising answers, question words
import string                                          # string: punctuation, for normalising answers
import sys                                             # sys: path, to import Days 99 and 101
import time                                            # time: perf_counter, for timings
from collections import Counter                        # Counter: shared words, for F1

import numpy as np                                     # numpy: the from-scratch parts
import torch                                           # torch: the model runs on it
from datasets import disable_progress_bars, load_dataset  # datasets: SQuAD from the local cache
from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # the library's entry points
from transformers.utils import logging as hf_logging   # the library's logging
from huggingface_hub.utils import logging as hub_logging  # the model hub's logging
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Days 99 and 101 sit here
with contextlib.redirect_stdout(io.StringIO()):         # the imports print nothing
    import day99_exercise as d99                        # Day 99: bootstrap_ci, paired_bootstrap
    import day101_summarization as d101                 # Day 101: split_sentences
if hasattr(sys.stdout, "reconfigure"):                  # (not when another script imports this one quietly)
    sys.stdout.reconfigure(encoding="utf-8")            # the articles' curly quotes and accents print as UTF-8 on Windows too
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()  # no progress bars or notices in the output
hub_logging.set_verbosity_error(); disable_progress_bars()            # the same for the hub and datasets
torch.set_num_threads(4)                                # CPU threads

MODEL = "distilbert/distilbert-base-cased-distilled-squad"  # DistilBERT fine-tuned on SQuAD's training set
N_Q = 1000                                              # questions sampled from the validation set
MAX_LEN, STRIDE, MAX_ANSWER = 384, 128, 30              # window length, overlap between windows, longest answer in tokens


# =============================================================================
# ANSWERING -- NEW
# =============================================================================
def answer(tok, model, question, context):              # NEW: the best span over every window, as (text, score)
    enc = tok(question, context, truncation="only_second", max_length=MAX_LEN, stride=STRIDE,  # long paragraphs -> overlapping windows
              return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping").numpy(); enc.pop("overflow_to_sample_mapping")  # character spans of every token
    with torch.no_grad():                                 # inference only
        out = model(**enc)
    best = (-np.inf, "")                                  # (score, text)
    for w in range(len(offsets)):                         # each window
        seq = enc.sequence_ids(w)                           # 0 = question token, 1 = paragraph token, None = special
        ctx = np.array([s == 1 for s in seq])               # only paragraph tokens can be an answer
        st, en = out.start_logits[w].numpy(), out.end_logits[w].numpy()
        S = st[:, None] + en[None, :]                       # score of every (start, end) pair
        ok = np.triu(np.ones_like(S, dtype=bool)) & ~np.triu(np.ones_like(S, dtype=bool), MAX_ANSWER)  # start <= end < start + 30
        ok &= ctx[:, None] & ctx[None, :]                   # both ends inside the paragraph
        S = np.where(ok, S, -np.inf)
        i, j = np.unravel_index(np.argmax(S), S.shape)      # the best pair in this window
        if S[i, j] > best[0]:
            best = (S[i, j], context[offsets[w][i][0]:offsets[w][j][1]])  # back to the paragraph's characters
    return best[1], float(best[0])


# =============================================================================
# SQUAD'S METRICS -- NEW (Rajpurkar et al.'s official definitions)
# =============================================================================
def normalize(s):                                       # NEW: lowercase, drop punctuation, drop a/an/the, single spaces
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, golds):                           # NEW: 1 if the normalised prediction equals ANY normalised gold answer
    return float(any(normalize(pred) == normalize(g) for g in golds))


def f1(pred, golds):                                    # NEW: best word-overlap F1 against any gold answer
    def one(p, g):
        p, g = normalize(p).split(), normalize(g).split()
        common = sum((Counter(p) & Counter(g)).values())    # shared words, counted with multiplicity
        if common == 0:
            return 0.0
        prec, rec = common / len(p), common / len(g)
        return 2 * prec * rec / (prec + rec)
    return max(one(pred, g) for g in golds)


def overlap_sentence(question, context):                # NEW: the baseline -- the sentence sharing the most (non-stop) words with the question
    stop = {"the", "a", "an", "of", "in", "to", "is", "was", "what", "which", "who", "when", "where", "how", "why", "did", "does",
            "do", "and", "for", "on", "by", "with", "as", "at", "that", "from", "are", "were", "it", "its"}
    q = {w for w in re.findall(r"\w+", question.lower()) if w not in stop}
    sents = d101.split_sentences(context) or [context]   # Day 101's sentence splitter
    return max(sents, key=lambda s: len(q & set(re.findall(r"\w+", s.lower()))))


# =============================================================================
# DEMOS
# =============================================================================
def load_questions():                                   # 1,000 random validation questions
    d = load_dataset("rajpurkar/squad", split="validation")
    idx = np.random.RandomState(0).choice(len(d), N_Q, replace=False)
    return [d[int(i)] for i in idx], len(d)


def demo_how(tok, model, Q):
    print("=" * 70)
    print("1. HOW THE MODEL ANSWERS -- start and end scores, the best span")
    print("=" * 70)
    ex = Q[0]                                             # one real example
    enc = tok(ex["question"], ex["context"], truncation="only_second", max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    st, en = out.start_logits[0].numpy(), out.end_logits[0].numpy()
    print(f"question: {ex['question']}")
    print(f"paragraph ({len(ex['context'].split())} words, {len(toks)} tokens with the question): {ex['context'][:160]}...")
    print("highest start scores: " + ", ".join(f"{toks[i]!r} {st[i]:.2f}" for i in np.argsort(-st)[:3]))
    print("highest end scores:   " + ", ".join(f"{toks[i]!r} {en[i]:.2f}" for i in np.argsort(-en)[:3]))
    text, score = answer(tok, model, ex["question"], ex["context"])
    print(f"best valid span: {text!r} (start + end score {score:.2f}); gold answers: {ex['answers']['text']}")
    import transformers                                   # only for its version number
    print(f"(the library's question-answering pipeline was removed in transformers 5; installed: {transformers.__version__}; "
          "so the span search is checked against the naive alternative instead)")
    bad = 0                                               # naive spans that are not a valid answer
    for q in Q[:200]:                                     # the first 200 questions
        e = tok(q["question"], q["context"], truncation="only_second", max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            o = model(**e)
        i, j = int(o.start_logits[0].argmax()), int(o.end_logits[0].argmax())  # best start and best end, chosen separately
        inside = [s == 1 for s in e.sequence_ids(0)]      # paragraph tokens
        bad += not (i <= j < i + MAX_ANSWER and inside[i] and inside[j])  # end before start, too long, or outside the paragraph
    print(f"naive search (best start and best end chosen separately) gives an invalid span for {bad} of the first 200 questions; "
          "the constrained search never does")
    lens = [len(tok(q["question"], q["context"])["input_ids"]) for q in Q]
    print(f"question + paragraph longer than {MAX_LEN} tokens (needs more than one window): {sum(L > MAX_LEN for L in lens)} of {len(Q)}")
    print("Actual output from this lesson's run.")


def demo_scores(tok, model, Q, n_total):
    print("\n" + "=" * 70)
    print("2. SQUAD'S METRICS -- exact match and F1")
    print("=" * 70)
    t0 = time.perf_counter()
    preds = [answer(tok, model, q["question"], q["context"]) for q in Q]  # (text, score) for every question
    secs = time.perf_counter() - t0
    golds = [q["answers"]["text"] for q in Q]
    em = np.array([exact_match(p, g) for (p, _), g in zip(preds, golds)])
    f = np.array([f1(p, g) for (p, _), g in zip(preds, golds)])
    print(f"{len(Q):,} questions drawn at random from {n_total:,} ({secs / len(Q) * 1000:.0f} ms per question, 4 threads); "
          f"{np.mean([len(g) for g in golds]):.2f} gold answers per question")
    for name, v in (("exact match", em), ("F1", f)):
        lo, hi = d99.bootstrap_ci(v)
        print(f"  {name:<12} {v.mean():.4f}   95% interval [{lo:.4f}, {hi:.4f}]")
    print("\nexamples where F1 gives partial credit (exact match 0, F1 above 0):")
    shown = 0
    for (p, _), g, e, ff in zip(preds, golds, em, f):
        if e == 0 and ff > 0 and shown < 4:
            print(f"  predicted {p!r:<38} gold {g[0]!r:<34} F1 {ff:.3f}"); shown += 1
    print("Actual output from this lesson's run.")
    return preds, em, f                                     # later sections use them


def demo_baseline(Q, em, f):
    print("\n" + "=" * 70)
    print("3. A FROM-SCRATCH BASELINE -- the best-overlap sentence")
    print("=" * 70)
    sent = [overlap_sentence(q["question"], q["context"]) for q in Q]  # one sentence per question
    golds = [q["answers"]["text"] for q in Q]
    contains = np.array([any(normalize(g) in normalize(s) for g in gs) for s, gs in zip(sent, golds)])
    fb = np.array([f1(s, g) for s, g in zip(sent, golds)]); eb = np.array([exact_match(s, g) for s, g in zip(sent, golds)])
    print(f"the chosen sentence contains a gold answer: {contains.mean():.4f}")
    print(f"answering with the whole sentence: exact match {eb.mean():.4f}, F1 {fb.mean():.4f} "
          f"(mean sentence {np.mean([len(s.split()) for s in sent]):.1f} words against {np.mean([len(g[0].split()) for g in golds]):.1f} for a gold answer)")
    m, lo, hi, _ = d99.paired_bootstrap(f, fb)
    print(f"model minus baseline, F1: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (Day 99's paired bootstrap)")
    print("Actual output from this lesson's run.")
    return fb                                               # the plot uses it


def demo_types(Q, em, f):
    print("\n" + "=" * 70)
    print("4. WHERE IT FAILS -- by question type")
    print("=" * 70)
    kind = lambda q: next((w for w in ("how many", "how much", "what", "which", "who", "when", "where", "why", "how")
                           if re.search(rf"\b{w}\b", q.lower())), "other")
    K = np.array([kind(q["question"]) for q in Q])
    rows = []
    print(f"{'type':<10}{'questions':>10}{'exact match':>13}{'F1':>8}")
    for k in sorted(set(K), key=lambda k: -np.sum(K == k)):
        m = K == k; rows.append((k, m.sum(), em[m].mean(), f[m].mean()))
        print(f"{k:<10}{m.sum():>10}{em[m].mean():>13.3f}{f[m].mean():>8.3f}")
    print("Actual output from this lesson's run.")
    return rows                                             # the plot uses them


def demo_wrong_paragraph(tok, model, Q, preds, em):
    print("\n" + "=" * 70)
    print("5. THE WRONG PARAGRAPH -- does the score know?")
    print("=" * 70)
    print("The first 300 questions again, each paired with the paragraph of a question from a DIFFERENT article. SQuAD v1.1 "
          "models were never trained to say 'no answer here', so they always point at some span.")
    titles = [q["title"] for q in Q]
    wrong = []                                            # (text, score) with an unrelated paragraph
    for i, q in enumerate(Q[:300]):
        j = next(k for k in range(i + 1, i + len(Q)) if titles[k % len(Q)] != titles[i])  # the next question from another article
        wrong.append(answer(tok, model, q["question"], Q[j % len(Q)]["context"]))
    right_s = np.array([s for _, s in preds[:300]]); wrong_s = np.array([s for _, s in wrong])
    print(f"\nscore (start + end) with the right paragraph: median {np.median(right_s):.2f}; with a wrong one: median {np.median(wrong_s):.2f}")
    print("sample of answers from the wrong paragraph: " + "; ".join(f"{q['question'][:40]!r} -> {t[:30]!r}" for q, (t, _) in list(zip(Q, wrong))[:3]))
    pos = np.r_[right_s, wrong_s]; lab = np.r_[np.ones(300), np.zeros(300)]  # can a threshold separate them?
    order = np.argsort(-pos); tp = np.cumsum(lab[order]); fp = np.cumsum(1 - lab[order])
    auc = np.trapezoid(np.r_[0, tp / 300], np.r_[0, fp / 300])  # area under the ROC curve, from scratch
    print(f"area under the ROC curve, score as a 'right paragraph' detector: {auc:.3f} (0.5 = useless, 1 = perfect)")
    for th in (2.0, 5.0, 8.0):
        print(f"  refuse below {th}: keeps {np.mean(right_s >= th):.3f} of right-paragraph answers "
              f"(their exact match {em[:300][right_s >= th].mean():.3f}), lets through {np.mean(wrong_s >= th):.3f} of wrong-paragraph ones")
    print("Actual output from this lesson's run.")
    return right_s, wrong_s                                  # the plot uses them


def plot_results(f, fb, rows, right_s, wrong_s):         # three panels: model vs baseline | F1 by type | score histograms
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))           # canvas
    ax[0].bar(["DistilBERT-SQuAD", "overlap sentence"], [f.mean(), fb.mean()], color=["#1f77b4", "#ff7f0e"])
    ax[0].set_ylabel("mean F1 (1,000 questions)"); ax[0].set_ylim(0, 1); ax[0].set_title("Model against the baseline", fontsize=10); ax[0].grid(alpha=0.3, axis="y")
    ks = [r[0] for r in rows]; ax[1].barh(ks, [r[3] for r in rows], color="#2ca02c"); ax[1].invert_yaxis(); ax[1].set_xlim(0, 1.12)
    for i, r in enumerate(rows):
        ax[1].text(r[3] + 0.01, i, f"n={r[1]}", va="center", fontsize=7)
    ax[1].set_xlabel("F1"); ax[1].set_title("F1 by question type", fontsize=10); ax[1].grid(alpha=0.3, axis="x")
    bins = np.linspace(min(right_s.min(), wrong_s.min()), max(right_s.max(), wrong_s.max()), 30)
    ax[2].hist(right_s, bins=bins, alpha=0.6, color="#1f77b4", label="right paragraph")
    ax[2].hist(wrong_s, bins=bins, alpha=0.6, color="#d62728", label="wrong paragraph")
    ax[2].set_xlabel("answer score (start + end logits)"); ax[2].set_ylabel("questions")
    ax[2].set_title("Can the score tell the paragraph is wrong?", fontsize=10); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("qa_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved qa_results.png")


def draw_diagram():                                     # question + paragraph -> start/end scores -> span
    fig, ax = plt.subplots(figsize=(13, 3.8))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.8); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 3.55, "Day 102: extractive question answering", ha="center", fontsize=12, fontweight="bold")
    steps = [(1.3, "[CLS] question [SEP]\nparagraph [SEP]\n(windows of 384,\noverlap 128)", "#F2F2F2"),
             (3.9, "DistilBERT-SQuAD\ntwo scores per token:\nstart, end", "#DCEBF7"),
             (6.5, "best pair (i, j):\ni <= j < i + 30,\nboth in the paragraph", "#FAEEDA"),
             (9.1, "offsets -> characters\n-> answer text", "#E5F2E1"),
             (11.7, "exact match, F1\n(SQuAD's normalisation)", "#EEEDFE")]
    for i, (x, t, c) in enumerate(steps):                            # the boxes and the arrows between them
        box(x, 1.7, 2.2, 1.6, t, c)
        if i:
            ax.annotate("", xy=(x - 1.13, 1.7), xytext=(steps[i - 1][0] + 1.13, 1.7), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("qa_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved qa_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT COMES NEXT")
    print("=" * 70)
    print("Today the right paragraph was handed to the model. Day 103 finds it: semantic search with sentence embeddings "
          "over every SQuAD paragraph, against BM25 keyword search written from scratch.")


def main():                                             # runs every section in order
    tok = AutoTokenizer.from_pretrained(MODEL)             # the model's (cased) tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL).eval()  # start and end scores per token
    Q, n_total = load_questions()                          # the questions
    demo_how(tok, model, Q)                                # 1. how it answers
    preds, em, f = demo_scores(tok, model, Q, n_total)     # 2. metrics
    fb = demo_baseline(Q, em, f)                           # 3. the baseline
    rows = demo_types(Q, em, f)                            # 4. by type
    right_s, wrong_s = demo_wrong_paragraph(tok, model, Q, preds, em)  # 5. the wrong paragraph
    plot_results(f, fb, rows, right_s, wrong_s)            # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 6. Day 103


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
