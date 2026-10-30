"""
Day 102 Exercise -- SQuAD's Answer Metrics

The fifth of Week 15's parts for Day 104's build, which scores every answer
of its question-answering system this way.

    normalize(s)              lowercase, remove punctuation, remove a/an/the, collapse spaces
    exact_match(pred, golds)  1 if the normalised prediction equals any normalised gold answer
    f1(pred, golds)           best word-overlap F1 against any gold answer (words counted with multiplicity)

These are the definitions of Rajpurkar et al.'s official SQuAD evaluation
script, rewritten; checked on cases whose scores can be worked out by hand.
"""
import re                                             # re: removing articles
import string                                         # string: punctuation
from collections import Counter                       # Counter: shared words


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


cases = [("The Denver Broncos", ["Denver Broncos"], "articles and case are ignored"),
         ("Denver Broncos.", ["Denver Broncos"], "punctuation is ignored"),
         ("Broncos", ["Denver Broncos"], "one of two words: precision 1, recall 0.5"),
         ("the Denver Broncos football team", ["Denver Broncos"], "two of four words: precision 0.5, recall 1"),
         ("Carolina Panthers", ["Denver Broncos", "Broncos"], "no shared word"),
         ("Broncos", ["Denver Broncos", "Broncos"], "matches the second gold answer exactly"),
         ("1,000", ["1000"], "the comma is punctuation: 1,000 becomes 1000")]
print(f"{'prediction':<36}{'gold answers':<30}{'EM':>4}{'F1':>7}   why")
for p, g, why in cases:
    print(f"{p!r:<36}{str(g):<30}{exact_match(p, g):>4.0f}{f1(p, g):>7.3f}   {why}")
