"""
Day 100 Exercise -- BIO Tags to Spans, and Span-Level Precision, Recall, F1

The third of Week 15's parts for Day 104's build, which scores extracted
answers as spans of words.

    spans(tags)          BIO tags -> set of (start, end, type), end exclusive
    prf(gold, pred)      entity-level precision, recall, F1 over many sentences: a span counts only if it matches exactly

Checked on tag sequences with known answers, including the awkward cases:
an I- tag with no B- before it, two entities of one type side by side, and a
type change in the middle of a run of I- tags.
"""


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


cases = [                                               # (description, tags, expected spans)
    ("one entity", ["B-PER", "I-PER", "O"], {(0, 2, "PER")}),
    ("runs to the end", ["O", "B-LOC", "I-LOC"], {(1, 3, "LOC")}),
    ("two of one type side by side", ["B-PER", "B-PER", "I-PER"], {(0, 1, "PER"), (1, 3, "PER")}),
    ("stray I- with no B-", ["O", "I-ORG", "I-ORG", "O"], {(1, 3, "ORG")}),
    ("type change inside I- run", ["B-ORG", "I-ORG", "I-LOC"], {(0, 2, "ORG"), (2, 3, "LOC")}),
    ("nothing", ["O", "O"], set()),
]
for name, tags, want in cases:
    got = spans(tags)
    print(f"{name:<30} {tags} -> {sorted(got)}  correct: {got == want}")

gold = [{(0, 2, "PER"), (5, 6, "LOC")}, {(1, 3, "ORG")}]  # two sentences' gold spans
pred = [{(0, 2, "PER"), (5, 6, "ORG")}, {(1, 2, "ORG"), (4, 5, "MISC")}]  # right; wrong type; wrong boundary; invented
p, r, f = prf(gold, pred)
print(f"\n3 gold spans, 4 predicted, 1 exactly right: precision {p:.3f} (1/4), recall {r:.3f} (1/3), F1 {f:.3f}")
print("a wrong type or a boundary off by one word earns nothing, exactly as in the CoNLL evaluation")
