"""
Day 114 Exercise -- Naive Bayes for Text

The third of Week 17's parts for Day 118's build. Two classes, three kinds.

    words(text)           Day 98's tokeniser: lower-case runs of letters, digits and apostrophes
    tokens(text, ngrams)  words, plus adjacent pairs when ngrams = 2
    NaiveBayes(kind, alpha, ngrams).fit(texts, y)   kind: "multinomial", "binary" or "bernoulli"
    .log_odds(texts), .proba(texts)

Checked by hand on four tiny reviews, and on the edge cases: unseen words change
nothing, a repeated word counts twice only for the multinomial kind, and the
Bernoulli kind is moved by words that are absent.
"""
import re                                              # re: the tokeniser

import numpy as np                                     # numpy: every array/matrix operation below is built on this


def words(s):                                           # Day 98's tokeniser, unchanged
    return re.findall(r"[a-z0-9']+", s.lower())


def tokens(text, ngrams=1):                             # NEW: words, plus adjacent pairs when ngrams = 2
    w = words(text)
    return w + [a + " " + b for a, b in zip(w, w[1:])] if ngrams == 2 else w


class NaiveBayes:                                       # NEW: Day 114's three kinds, unchanged
    def __init__(self, kind="multinomial", alpha=1.0, ngrams=1):
        self.kind, self.alpha, self.ngrams = kind, alpha, ngrams

    def encode(self, texts, grow=False):
        ids, rows = [], []
        for r, t in enumerate(texts):
            toks = tokens(t, self.ngrams)
            for w in (toks if self.kind == "multinomial" else set(toks)):
                if grow and w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
                if w in self.vocab:
                    ids.append(self.vocab[w]); rows.append(r)
        return np.array(ids, dtype=np.int64), np.array(rows, dtype=np.int64)

    def fit(self, texts, y):
        self.vocab = {}; ids, rows = self.encode(texts, grow=True); V = len(self.vocab); a = self.alpha
        self.prior = np.log(np.bincount(y, minlength=2) / len(y))
        N = np.stack([np.bincount(ids[y[rows] == c], minlength=V) for c in (0, 1)]).astype(float)
        if self.kind == "bernoulli":
            n = np.bincount(y, minlength=2)[:, None]
            p = (N + a) / (n + 2 * a)
            self.lp, self.lq = np.log(p), np.log(1 - p)
        else:
            self.lp = np.log((N + a) / (N.sum(1, keepdims=True) + a * V))
        return self

    def log_odds(self, texts):
        ids, rows = self.encode(texts); n = len(texts)
        d = self.lp[1] - self.lp[0] if self.kind != "bernoulli" else (self.lp[1] - self.lq[1]) - (self.lp[0] - self.lq[0])
        s = self.prior[1] - self.prior[0] + np.bincount(rows, weights=d[ids], minlength=n)
        if self.kind == "bernoulli":
            s += self.lq[1].sum() - self.lq[0].sum()
        return s

    def proba(self, texts):
        return 1 / (1 + np.exp(-self.log_odds(texts)))


docs = ["good movie", "great fun", "bad movie", "boring bad"]; y = np.array([1, 1, 0, 0])
nb = NaiveBayes().fit(docs, y)
print(f"'good fun movie': by hand 0.008 / (0.008 + 0.002) = 0.8; multinomial gives {nb.proba(['good fun movie'])[0]:.4f}")
print(f"with an unseen word, 'good fun movie zebra': {nb.proba(['good fun movie zebra'])[0]:.4f}")
print(f"'good good movie': multinomial log-odds {nb.log_odds(['good good movie'])[0]:.4f} (good counted twice); "
      f"binary {NaiveBayes('binary').fit(docs, y).log_odds(['good good movie'])[0]:.4f} (once)")
b = NaiveBayes("bernoulli").fit(docs, y)
print(f"Bernoulli, empty text (only absent words speak): log-odds {b.log_odds([''])[0]:.4f}; 'good': {b.log_odds(['good'])[0]:.4f}")
print(f"bigrams of 'not bad at all': {tokens('not bad at all', 2)}")
