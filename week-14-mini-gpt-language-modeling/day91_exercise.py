"""
Day 91 Exercise -- A Kneser-Ney Character Model in One Class

The first of Week 14's parts for Day 97's build, where it is the counting
baseline every neural model is compared with.

    KneserNey(text, order, D)   count every n-gram up to the order
    .prob(context, ch)          p(ch | context): discounted counts, the saved mass passed to the
                                next shorter context, continuation counts below the top order
    .bits(text, start)          mean bits per character, each with its real preceding context

Checked: for any context the probabilities of all characters add up to 1
(seen and unseen contexts alike), a character never seen after a context
still gets some probability, and on a tiny real text a longer context helps.
"""
from collections import Counter, defaultdict          # Counter and defaultdict: the n-gram tables

import numpy as np                                    # numpy: every array/matrix operation below is built on this


class KneserNey:                                        # NEW: interpolated Kneser-Ney n-gram model over characters
    def __init__(self, text, order, D=0.75):              # count every n-gram up to the order
        self.n, self.D, self.V = order, D, sorted(set(text))  # order, discount, the characters
        self.cnt = [Counter() for _ in range(order + 1)]   # cnt[k][g]: how often the k-gram g occurs
        for k in range(1, order + 1):                       # every length
            self.cnt[k].update(text[i:i + k] for i in range(len(text) - k + 1))
        self.cont = [Counter() for _ in range(order + 1)]  # cont[k][g]: how many different characters precede g
        for k in range(1, order):                           # lower orders only
            for g in self.cnt[k + 1]:                         # every (k + 1)-gram seen
                self.cont[k][g[1:]] += 1                        # one more distinct left neighbour
        self.tab = [None] + [self._tables(k) for k in range(1, order + 1)]  # per order: counts, context totals, distinct followers

    def _tables(self, k):                                 # context totals and distinct next characters, for order k
        src = self.cnt[k] if k == self.n else self.cont[k]  # raw counts at the top, continuation counts below
        tot, kinds = defaultdict(float), defaultdict(int)   # context -> total; context -> distinct followers
        for g, c in src.items():                            # every k-gram
            tot[g[:-1]] += c; kinds[g[:-1]] += 1
        return src, tot, kinds

    def prob(self, ctx, ch, k=None):                      # p(ch | the last k - 1 characters of ctx)
        k = self.n if k is None else k                      # start at the top order
        if k == 0:                                          # below unigrams: uniform
            return 1.0 / len(self.V)
        src, tot, kinds = self.tab[k]                       # this order's tables
        h = ctx[len(ctx) - (k - 1):] if k > 1 else ""       # the context this order uses
        if tot[h] == 0:                                     # never seen: the lower order alone
            return self.prob(ctx, ch, k - 1)
        return max(src[h + ch] - self.D, 0.0) / tot[h] + self.D * kinds[h] / tot[h] * self.prob(ctx, ch, k - 1)

    def bits(self, text, start):                          # mean bits per character of text[start:]
        return -np.mean([np.log2(self.prob(text[max(0, i - self.n + 1):i], text[i])) for i in range(start, len(text))])


text = ("to be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles ")  # a real (short) text
kn = KneserNey(text, 4)                                   # order 4: three characters of context
for ctx in ("to ", "the", "xyz", ""):                     # seen, seen, never seen, empty contexts
    total = sum(kn.prob(ctx, ch) for ch in kn.V)            # over every character
    print(f"context {ctx!r:<7}: probabilities over all {len(kn.V)} characters sum to {total:.12f}")
after = sorted({text[i + 3] for i in range(len(text) - 3) if text[i:i + 3] == "to "})  # what really followed 'to '
print(f"after 'to ' the text has only {after}; p('b') = {kn.prob('to ', 'b'):.3f}, "
      f"p('q') (never seen there) = {kn.prob('to ', 'q'):.4f}")
for n in (1, 2, 4):                                       # longer context, on the text itself
    print(f"order {n} ({n - 1} characters of context): {KneserNey(text, n).bits(text, 3):.3f} bits per character on its own training text")
