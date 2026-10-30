"""
Day 103 Exercise -- Mean Pooling, Cosine Top-k, and BM25

The sixth and last of Week 15's parts for Day 104's build: its retriever.

    mean_pool(hidden, mask)     average a text's token vectors, ignoring padding (PyTorch)
    cosine_top_k(Q, D, k)       for unit-length rows: the k most similar rows of D for every row of Q, best first
    BM25(docs).scores(query)    Okapi BM25 score of every document for one query (k1 = 1.5, b = 0.75)

Checked: padding does not change a mean-pooled vector; top-k agrees with a
brute-force sort; BM25 rewards a rare word over a common one, saturates with
repeated words, and prefers the shorter of two otherwise equal documents.
"""
import re                                             # re: words
from collections import Counter                       # Counter: term counts

import numpy as np                                    # numpy: every array/matrix operation below is built on this
import torch                                          # torch: mean pooling works on the encoder's tensors

STOP = {"the", "a", "an", "of", "in", "to", "is", "was", "and", "for", "on", "by", "with", "as", "at", "that", "from", "are",
        "were", "it", "its", "be", "this", "or", "which", "what", "who", "when", "where", "how", "why", "did", "does", "do"}


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


torch.manual_seed(0)                                    # fixed torch rng
h = torch.randn(1, 5, 4); mask = torch.ones(1, 5)      # one text of 5 tokens, vectors of 4
hp = torch.cat([h, torch.randn(1, 3, 4)], 1); mp = torch.cat([mask, torch.zeros(1, 3)], 1)  # the same text with 3 padding tokens
print(f"mean pooling with and without 3 padding tokens: largest difference {(mean_pool(h, mask) - mean_pool(hp, mp)).abs().max().item():.1e}")

rng = np.random.RandomState(0)                          # fixed rng
D = rng.randn(500, 16); D /= np.linalg.norm(D, axis=1, keepdims=True)  # 500 unit vectors
Q = rng.randn(20, 16); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
idx, sc = cosine_top_k(Q, D, 5)
brute = np.argsort(-(Q @ D.T), axis=1)[:, :5]           # a full sort, for comparison
print(f"top-5 for 20 queries over 500 vectors equals a full sort: {np.array_equal(idx, brute)}; scores descending: {bool(np.all(np.diff(sc, axis=1) <= 0))}")

docs = ["apple banana cherry", "apple apple apple banana", "banana durian", "banana fig grape kiwi lemon mango",
        "durian elderberry"]                            # 'banana' is common, 'durian' rare
bm = BM25(docs)
print(f"idf: banana (in 4 of 5 documents) {bm.idf['banana']:.3f}, durian (in 2 of 5) {bm.idf['durian']:.3f}")
s = bm.scores("apple"); print(f"'apple' once vs three times: {s[0]:.3f} vs {s[1]:.3f} (three times is not three times the score)")
s = bm.scores("banana"); print(f"'banana' in a 2-word vs a 6-word document: {s[2]:.3f} vs {s[3]:.3f} (the shorter one wins)")
print(f"best document for 'durian banana': {docs[int(np.argmax(bm.scores('durian banana')))]!r}")
