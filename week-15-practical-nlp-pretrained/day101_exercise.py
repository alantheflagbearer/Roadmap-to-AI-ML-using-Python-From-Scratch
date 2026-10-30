"""
Day 101 Exercise -- Sentences and Chunks

The fourth of Week 15's parts for Day 104's build, which searches over pieces
of documents rather than whole documents.

    split_sentences(text)                 sentence boundaries by rule: a stop, a space, a capital -- except after 'Mr.', 'U.S.', an initial
    chunk_words(text, size, overlap)      overlapping windows of `size` words, each starting `size - overlap` words after the last

Checked on sentences with known boundaries (abbreviations, initials, quotes)
and on chunks: every word is covered, neighbouring chunks share exactly
`overlap` words, and nothing is lost at the end.
"""
import re                                             # re: sentence boundaries

ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "gen", "gov", "sen", "rep", "lt", "col", "sgt", "capt", "u.s",
          "u.k", "no", "vs", "inc", "co", "corp", "ltd", "jan", "feb", "mar", "apr", "aug", "sept", "oct", "nov", "dec"}  # not sentence ends


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


text = ('Mr. Smith arrived in Washington on Monday. He met Dr. J. R. Doe at 5 p.m. "It went well," he said. '
        'U.S. officials declined to comment. Talks resume in Oct. 2025, a spokesman said.')
for s in split_sentences(text):                         # five sentences expected
    print(f"  | {s}")
print(f"{len(split_sentences(text))} sentences (expected 5)")

doc = " ".join(f"w{i}" for i in range(23))              # 23 numbered words
for size, overlap in ((10, 3), (8, 0), (30, 5)):
    chunks = chunk_words(doc, size, overlap)
    covered = sorted({x for c in chunks for x in c.split()}, key=lambda x: int(x[1:]))
    shared = [len(set(a.split()) & set(b.split())) for a, b in zip(chunks, chunks[1:])]
    print(f"size {size}, overlap {overlap}: {len(chunks)} chunks of {[len(c.split()) for c in chunks]} words; "
          f"all 23 words covered: {covered == doc.split()}; words shared by neighbours: {shared}")
