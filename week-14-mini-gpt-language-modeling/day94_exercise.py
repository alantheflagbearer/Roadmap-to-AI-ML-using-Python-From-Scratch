"""
Day 94 Exercise -- Measures of Generated Text

The fourth of Week 14's parts for Day 97's build: the numbers that stand in
for 'reading a few samples'.

    copy_share(texts, train_text, k)   share of every k-character piece of the texts found in the training text
    distinct_pairs(texts)              distinct word pairs / all word pairs (1 = no repeated pair)
    real_word_rate(texts, known)       share of words that are real (in a known vocabulary)
    speaker_share(texts)               share of complete lines that are 'NAME:' speaker lines

Checked on texts whose answers are known in advance.
"""
import re                                             # re: words and speaker lines

import numpy as np                                    # numpy: every array/matrix operation below is built on this

SPEAKER = re.compile(r"^[A-Z][A-Z' ]+:$")               # capitals, then a colon, alone on the line


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


train = "ROMEO:\nBut soft, what light through yonder window breaks?\nJULIET:\nAy me.\n"  # a tiny 'training text'
known = set(re.findall(r"[a-z']+", train.lower()))     # its words
print(f"a piece of the training text itself, k = 10: copy share {copy_share(['what light through yonder'], train, 10):.3f} (all copied)")
print(f"new text, k = 10: copy share {copy_share(['a quiet garden at noon'], train, 10):.3f} (none copied)")
print(f"'the cat the cat the cat': distinct pairs {distinct_pairs(['the cat the cat the cat']):.3f} (2 different pairs of 5)")
print(f"'but soft blorp light': real words {real_word_rate(['but soft blorp light'], known):.3f} (3 of 4)")
print(f"speaker lines in the training text: {speaker_share([train]):.3f} ('JULIET:' is 1 of the 3 complete lines; the first line, "
      "'ROMEO:', could be cut off in a passage, so first lines are never counted)")
