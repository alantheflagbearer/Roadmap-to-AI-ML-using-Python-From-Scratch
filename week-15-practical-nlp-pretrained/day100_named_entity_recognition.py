"""
Day 100: Named Entity Recognition with a Pretrained Model on a Real Dataset

Day 100 of 365. Days 98 and 99 gave one label to a whole sentence. Named
entity recognition (NER) gives one label to every word: is it part of a
person's name, an organisation, a location, another kind of name (MISC), or
none of these?

  EU   rejects  German  call  to  boycott  British  lamb  .
  B-ORG  O      B-MISC   O    O     O      B-MISC    O    O

The data is CoNLL-2003 (Reuters news, 1996), the standard English NER
benchmark: 14,041 training sentences and a 3,453-sentence test set. The model
is dslim/distilbert-NER, a DistilBERT fine-tuned by its author on CoNLL-2003's
training set, so the test set here is the genuine held-out test.

  1. The task and the data: BIO tags, entity spans, how many of each type.
  2. Words and word pieces: one tag per word from a model that sees pieces.
  3. Scoring from scratch: tags -> spans -> entity-level precision, recall
     and F1 (a span counts only if its words AND its type are exactly right),
     next to the misleading per-word accuracy.
  4. A from-scratch baseline: a gazetteer (a list of every entity seen in
     training), matched longest-first.
  5. Where the model fails: entities seen in training against new ones, and
     the kinds of mistake (wrong type, wrong boundary, missed, invented).
  6. A check against the library's own pipeline.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
from collections import Counter                        # Counter: counts of entity types and mistake kinds

import numpy as np                                     # numpy: the from-scratch parts
import torch                                           # torch: the model runs on it
from datasets import disable_progress_bars, load_dataset  # datasets: CoNLL-2003 from the local cache
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline  # the library's entry points
from transformers.utils import logging as hf_logging   # the library's logging
from huggingface_hub.utils import logging as hub_logging  # the model hub's logging
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

MODEL = "dslim/distilbert-NER"                          # DistilBERT fine-tuned on CoNLL-2003's training set
TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]  # CoNLL's tag ids (the model uses the same)
TYPES = ["PER", "ORG", "LOC", "MISC"]                   # the four entity types
torch.set_num_threads(4)                                # CPU threads for inference
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()  # no progress bars or notices in the output
hub_logging.set_verbosity_error(); disable_progress_bars()            # the same for the hub and datasets


# =============================================================================
# DATA AND MODEL
# =============================================================================
def conll(split):                                       # one split: lists of words and lists of tag strings
    d = load_dataset("eriktks/conll2003", revision="refs/convert/parquet", split=split)  # the hub's parquet copy
    return list(d["tokens"]), [[TAGS[t] for t in tags] for tags in d["ner_tags"]]


def predict_tags(tok, model, sents, bs=32):             # NEW: one tag per WORD, from the tag of each word's first piece
    out = []                                              # one list of tags per sentence
    with torch.no_grad():                                 # inference only
        for i in range(0, len(sents), bs):                  # batch by batch
            batch = sents[i:i + bs]
            enc = tok(batch, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")  # words -> pieces
            pred = model(**enc).logits.argmax(-1).numpy()   # a tag id for every piece
            for b, words in enumerate(batch):               # back from pieces to words
                tags, seen = [], set()                        # this sentence's tags; words already tagged
                for pos, w in enumerate(enc.word_ids(b)):     # which word each piece came from (None for [CLS] etc.)
                    if w is not None and w not in seen:          # the FIRST piece of each word decides
                        tags.append(TAGS[pred[b, pos]]); seen.add(w)
                tags += ["O"] * (len(words) - len(tags))       # words cut off by truncation (none here) count as O
                out.append(tags)
    return out


# =============================================================================
# SCORING -- NEW
# =============================================================================
def spans(tags):                                        # NEW: BIO tags -> set of (start, end, type); end is exclusive
    out, start, typ = set(), None, None                   # finished spans; the open span, if any
    for i, t in enumerate(tags + ["O"]):                  # an extra O closes a span that runs to the end
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):  # anything that ends the open span
            if start is not None:
                out.add((start, i, typ))
            start, typ = (i, t[2:]) if t != "O" else (None, None)  # B-X (or a stray I-X) opens a new span
    return out


def prf(gold, pred):                                    # NEW: entity-level precision, recall, F1 from lists of span sets
    tp = sum(len(g & p) for g, p in zip(gold, pred))      # exact matches: same words AND same type
    n_pred, n_gold = sum(map(len, pred)), sum(map(len, gold))
    p, r = tp / max(1, n_pred), tp / max(1, n_gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, n_pred, n_gold


def by_type(gold, pred, typ):                           # the same, for one entity type
    keep = lambda S: [{s for s in x if s[2] == typ} for x in S]
    return prf(keep(gold), keep(pred))


# =============================================================================
# A FROM-SCRATCH BASELINE -- NEW
# =============================================================================
class Gazetteer:                                        # NEW: every entity string seen in training, with its most common type
    def __init__(self, sents, tags):
        votes = {}                                          # word tuple -> Counter of types
        for words, t in zip(sents, tags):                   # every training sentence
            for s, e, typ in sorted(spans(t)):                # every gold entity in it, left to right (a set has no fixed order)
                votes.setdefault(tuple(words[s:e]), Counter())[typ] += 1
        self.names = {k: c.most_common(1)[0][0] for k, c in votes.items()}  # the majority type
        self.longest = max(map(len, self.names))           # longest entry, in words

    def tag(self, words):                                 # longest match first, left to right
        tags, i = ["O"] * len(words), 0
        while i < len(words):
            for L in range(min(self.longest, len(words) - i), 0, -1):  # try the longest window first
                typ = self.names.get(tuple(words[i:i + L]))
                if typ:                                         # a known entity: tag it and jump past it
                    tags[i:i + L] = [f"B-{typ}"] + [f"I-{typ}"] * (L - 1); i += L; break
            else:                                             # no entry starts here
                i += 1
        return tags


# =============================================================================
# DEMOS
# =============================================================================
def demo_data(tr, te):
    print("=" * 70)
    print("1. THE TASK AND THE DATA -- CoNLL-2003")
    print("=" * 70)
    (Xtr, Ttr), (Xte, Tte) = tr, te
    print("a training sentence, word by word:")
    print("  " + "  ".join(f"{w}/{t}" for w, t in zip(Xtr[0], Ttr[0])))
    for name, X, T in (("train", Xtr, Ttr), ("test", Xte, Tte)):
        c = Counter(s[2] for t in T for s in spans(t))       # entities by type
        n_words = sum(map(len, X))
        print(f"{name}: {len(X):,} sentences, {n_words:,} words, {sum(c.values()):,} entities "
              f"({', '.join(f'{k} {c[k]:,}' for k in TYPES)}); words tagged O: {np.mean([x == 'O' for t in T for x in t]):.3f}")
    print("Actual output from this lesson's run.")


def demo_pieces(tok, model):
    print("\n" + "=" * 70)
    print("2. WORDS AND WORD PIECES")
    print("=" * 70)
    words = ["Schwarzenegger", "visited", "Ouagadougou", "with", "Lufthansa", "."]  # rare names split into many pieces
    enc = tok(words, is_split_into_words=True)
    print(f"words: {words}")
    print(f"pieces: {tok.convert_ids_to_tokens(enc['input_ids'])}")
    print(f"word of each piece: {enc.word_ids()}")
    with torch.no_grad():
        pred = model(**tok(words, is_split_into_words=True, return_tensors="pt")).logits.argmax(-1)[0].numpy()
    print("tag of each piece: " + " ".join(f"{p}:{TAGS[k]}" for p, k in zip(tok.convert_ids_to_tokens(enc["input_ids"]), pred)))
    print(f"one tag per word (first piece): {predict_tags(tok, model, [words])[0]}")
    print("Actual output from this lesson's run.")


def demo_scores(tok, model, te):
    print("\n" + "=" * 70)
    print("3. SCORING -- entity spans, not words")
    print("=" * 70)
    Xte, Tte = te
    P = predict_tags(tok, model, Xte)                     # the model's tags on all 3,453 test sentences
    gold, pred = [spans(t) for t in Tte], [spans(t) for t in P]
    word_acc = np.mean([a == b for t, p in zip(Tte, P) for a, b in zip(t, p)])  # per-word accuracy
    all_o = np.mean([a == "O" for t in Tte for a in t])   # what tagging everything O would score
    p, r, f, tp, n_pred, n_gold = prf(gold, pred)
    print(f"per-word tag accuracy {word_acc:.4f} (tagging every word O would already score {all_o:.4f})")
    print(f"entity level: {n_gold:,} gold entities, {n_pred:,} predicted, {tp:,} exactly right -> precision {p:.4f}, "
          f"recall {r:.4f}, F1 {f:.4f}")
    print(f"\n{'type':<6}{'gold':>7}{'precision':>11}{'recall':>9}{'F1':>8}")
    for typ in TYPES:
        p, r, f, tp, n_pred, n_gold = by_type(gold, pred, typ)
        print(f"{typ:<6}{n_gold:>7,}{p:>11.4f}{r:>9.4f}{f:>8.4f}")
    print("Actual output from this lesson's run.")
    return P, gold, pred                                   # later sections use them


def demo_gazetteer(tr, te, gold, pred):
    print("\n" + "=" * 70)
    print("4. A FROM-SCRATCH BASELINE -- the gazetteer")
    print("=" * 70)
    (Xtr, Ttr), (Xte, _) = tr, te                         # training sentences and tags; test sentences
    gz = Gazetteer(Xtr, Ttr)                              # every training entity
    G = [spans(gz.tag(w)) for w in Xte]                   # its spans on the test set
    print(f"{len(gz.names):,} distinct entity strings in training (longest {gz.longest} words), each with its most common type")
    print(f"\n{'system':<26}{'precision':>11}{'recall':>9}{'F1':>8}")
    for name, S in (("gazetteer (from scratch)", G), ("DistilBERT-NER", pred)):
        p, r, f, *_ = prf(gold, S)
        print(f"{name:<26}{p:>11.4f}{r:>9.4f}{f:>8.4f}")
    print("Actual output from this lesson's run.")
    return gz, G                                            # section 5 uses them


def demo_failures(te, gold, pred, gz, G):
    print("\n" + "=" * 70)
    print("5. WHERE THE MODEL FAILS")
    print("=" * 70)
    Xte, _ = te
    seen = lambda w, s: tuple(w[s[0]:s[1]]) in gz.names   # was this exact string an entity in training?
    for name, S in (("DistilBERT-NER", pred), ("gazetteer", G)):
        rec = {}                                            # seen / new -> (found, total)
        for w, g, p in zip(Xte, gold, S):
            for s in g:
                k = "seen in training" if seen(w, s) else "new"
                f, n = rec.get(k, (0, 0)); rec[k] = (f + (s in p), n + 1)
        print(f"{name:<15} recall on gold entities " + "; ".join(f"{k}: {f / n:.4f} of {n:,}" for k, (f, n) in sorted(rec.items())))
    kinds = Counter()                                     # what went wrong, for the model
    for g, p in zip(gold, pred):
        for s in p - g:                                     # predicted spans that are not exactly right
            same = [x for x in g if (x[0], x[1]) == (s[0], s[1])]  # same words, different type?
            over = [x for x in g if x[0] < s[1] and s[0] < x[1]]  # overlapping a gold span at all?
            kinds["right words, wrong type" if same else "wrong boundary" if over else "invented (no gold overlap)"] += 1
        for s in g:                                         # gold spans with no overlapping prediction at all
            if not any(x[0] < s[1] and s[0] < x[1] for x in p):
                kinds["missed entirely"] += 1
    print("\nthe model's mistakes by kind: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    conf = Counter((x[2], s[2]) for g, p in zip(gold, pred) for s in p for x in g if (x[0], x[1]) == (s[0], s[1]) and x[2] != s[2])
    print("wrong-type confusions (gold -> predicted), most common: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in conf.most_common(5)))
    print("Actual output from this lesson's run.")
    return rec, kinds                                       # the plot uses them


def demo_pipeline(tok, model):
    print("\n" + "=" * 70)
    print("6. A CHECK AGAINST THE LIBRARY'S PIPELINE")
    print("=" * 70)
    ner = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="first")  # the library's own grouping
    text = "Angela Merkel met Emmanuel Macron in Paris on Tuesday to discuss the European Union budget."
    words = text.replace(".", " .").split()               # the same text as words
    ours = sorted(spans(predict_tags(tok, model, [words])[0]))
    print(f"text: {text}")
    print("pipeline: " + "; ".join(f"{e['word']} ({e['entity_group']}, {e['score']:.3f})" for e in ner(text)))
    print("ours:     " + "; ".join(f"{' '.join(words[s:e])} ({t})" for s, e, t in ours))
    print("Actual output from this lesson's run.")


def plot_results(gold, pred, G, kinds):                 # two panels: F1 by type, both systems | mistakes by kind
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))           # canvas
    x = np.arange(len(TYPES) + 1)
    for j, (name, S, col) in enumerate((("gazetteer (from scratch)", G, "#ff7f0e"), ("DistilBERT-NER", pred, "#1f77b4"))):
        f1 = [by_type(gold, S, t)[2] for t in TYPES] + [prf(gold, S)[2]]
        ax[0].bar(x + (j - 0.5) * 0.38, f1, width=0.38, color=col, label=name)
    ax[0].set_xticks(x); ax[0].set_xticklabels(TYPES + ["all"]); ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("entity-level F1 (test)")
    ax[0].set_title("Entity F1 on CoNLL-2003 test, by type", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
    k = [a for a, _ in kinds.most_common()]
    ax[1].barh(k, [kinds[a] for a in k], color="#9467bd"); ax[1].invert_yaxis()
    ax[1].set_xlabel("count on the test set"); ax[1].set_title("DistilBERT-NER's mistakes by kind", fontsize=10); ax[1].grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig("ner_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved ner_results.png")


def draw_diagram():                                     # words -> pieces -> tags -> spans -> score
    fig, ax = plt.subplots(figsize=(13, 3.8))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.8); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 3.55, "Day 100: named entity recognition, from words to an entity-level score", ha="center", fontsize=12, fontweight="bold")
    steps = [(1.2, "words\nEU rejects German\ncall ...", "#F2F2F2"), (3.6, "word pieces\n(cased WordPiece)\nword_ids() remembers\nthe word of each", "#FAEEDA"),
             (6.1, "DistilBERT-NER\na tag for every piece\n(9 tags: O, B-/I- x 4)", "#DCEBF7"),
             (8.6, "one tag per word\n(first piece)\n-> BIO -> spans\n(start, end, type)", "#E5F2E1"),
             (11.4, "score spans exactly:\nprecision, recall, F1\n(vs a gazetteer\nbuilt from training)", "#EEEDFE")]
    for i, (x, t, c) in enumerate(steps):                            # the boxes and the arrows between them
        box(x, 1.7, 2.2 if i else 1.8, 1.6, t, c)
        if i:
            ax.annotate("", xy=(x - (1.13 if i < 4 else 1.13), 1.7), xytext=(steps[i - 1][0] + (1.13 if i > 1 else 0.93), 1.7),
                        arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("ner_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved ner_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 101 is summarization: picking the most important sentences (extractive, from scratch) against writing a "
          "new summary (abstractive, a pretrained T5), scored with Day 97's ROUGE on real news articles.")


def main():                                             # runs every section in order
    tr, te = conll("train"), conll("test")                 # the data
    tok = AutoTokenizer.from_pretrained(MODEL)             # the model's (cased) tokenizer
    model = AutoModelForTokenClassification.from_pretrained(MODEL).eval()  # one tag per piece
    demo_data(tr, te)                                      # 1. the data
    demo_pieces(tok, model)                                # 2. pieces
    P, gold, pred = demo_scores(tok, model, te)            # 3. scores
    gz, G = demo_gazetteer(tr, te, gold, pred)             # 4. the baseline
    rec, kinds = demo_failures(te, gold, pred, gz, G)      # 5. failures
    demo_pipeline(tok, model)                              # 6. the pipeline check
    plot_results(gold, pred, G, kinds)                     # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 7. Day 101


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
