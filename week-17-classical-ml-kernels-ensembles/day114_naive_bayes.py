"""
Day 114: Naive Bayes From Scratch -- Real Text Classification

Naive Bayes classifies a text by asking, for each class, how likely its words
would be if they were drawn independently from that class's word frequencies.
The independence assumption is plainly false (words come in phrases), yet the
model is fast, needs no training loop, and is famously hard to beat when there
is little data. Today tests all three claims on real movie reviews.

Everything is written from scratch in NumPy:
  * multinomial Naive Bayes (word counts), its binarised version (each word once
    per text), and Bernoulli Naive Bayes (which words are present AND absent)
  * Laplace (add-alpha) smoothing, chosen by 5-fold cross-validation
  * unigrams, and unigrams plus bigrams

The data: SST-2 (Stanford Sentiment Treebank, as in Week 15), from the local
cache: 67,349 training phrases and sentences, 872 validation sentences. The
logistic-regression baseline, the tokeniser and the calibration measure are
Day 98's, imported unchanged.

  1. By hand: four tiny reviews.
  2. Correctness: against scikit-learn's MultinomialNB.
  3. The three kinds, against logistic regression and Day 98's DistilBERT.
  4. Learning curves: who wins with little data?
  5. Smoothing.
  6. The independence assumption: confidence, calibration, and repeated words.
  7. What it learned, and negation.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imports quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: reading Day 98's result from its run output
import sys                                             # sys: path, to import Day 98

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.feature_extraction.text import CountVectorizer  # used ONLY for the outside check
from sklearn.naive_bayes import MultinomialNB          # used ONLY for the outside check
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
W15 = os.path.join(HERE, "..", "week-15-practical-nlp-pretrained")  # Week 15
sys.path.insert(0, W15)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):  # its imports print nothing
    import day98_pretrained_transformers as d98         # Day 98: sst2, words, calibration, BagOfWords, bootstrap_ci
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")            # real text is printed: keep Windows from using cp1252
SIZES = [100, 300, 1000, 3000, 10000, 30000, 67349]     # section 4: training items
SEEDS = 5                                               # section 4: random subsets per size
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]               # section 5: smoothing
NEGATION = {"not", "no", "never", "nor", "nothing"}     # words that negate, plus any word ending in n't


# =============================================================================
# NAIVE BAYES -- NEW
# =============================================================================
def tokens(text, ngrams=1):                             # NEW: Day 98's words, plus adjacent pairs when ngrams = 2
    w = d98.words(text)
    return w + [a + " " + b for a, b in zip(w, w[1:])] if ngrams == 2 else w


class NaiveBayes:                                       # NEW: three kinds of Naive Bayes, for two classes
    def __init__(self, kind="multinomial", alpha=1.0, ngrams=1):
        self.kind, self.alpha, self.ngrams = kind, alpha, ngrams  # "multinomial", "binary" or "bernoulli"

    def encode(self, texts, grow=False):                  # (word ids, text index) pairs; repeats kept for multinomial
        ids, rows = [], []
        for r, t in enumerate(texts):
            toks = tokens(t, self.ngrams)
            for w in (toks if self.kind == "multinomial" else set(toks)):  # binary and Bernoulli count a word once
                if grow and w not in self.vocab:
                    self.vocab[w] = len(self.vocab)               # a new word gets the next column
                if w in self.vocab:                               # unknown words are ignored at prediction time
                    ids.append(self.vocab[w]); rows.append(r)
        return np.array(ids, dtype=np.int64), np.array(rows, dtype=np.int64)

    def fit(self, texts, y):
        self.vocab = {}; ids, rows = self.encode(texts, grow=True); V = len(self.vocab); a = self.alpha
        self.prior = np.log(np.bincount(y, minlength=2) / len(y))  # log P(class)
        N = np.stack([np.bincount(ids[y[rows] == c], minlength=V) for c in (0, 1)]).astype(float)  # word counts per class
        if self.kind == "bernoulli":
            n = np.bincount(y, minlength=2)[:, None]          # texts per class
            p = (N + a) / (n + 2 * a)                           # P(word present | class)
            self.lp, self.lq = np.log(p), np.log(1 - p)       # present, absent
        else:
            self.lp = np.log((N + a) / (N.sum(1, keepdims=True) + a * V))  # P(word | class), each row summing to 1
        return self

    def log_odds(self, texts):                            # NEW: log P(positive | text) - log P(negative | text)
        ids, rows = self.encode(texts); n = len(texts)
        d = self.lp[1] - self.lp[0] if self.kind != "bernoulli" else (self.lp[1] - self.lq[1]) - (self.lp[0] - self.lq[0])
        s = self.prior[1] - self.prior[0] + np.bincount(rows, weights=d[ids], minlength=n)  # each word adds its log ratio
        if self.kind == "bernoulli":
            s += self.lq[1].sum() - self.lq[0].sum()          # every absent word counts too
        return s

    def proba(self, texts):                               # P(positive)
        return 1 / (1 + np.exp(-self.log_odds(texts)))


def lr_proba(bow, texts):                               # Day 98's logistic regression, as probabilities
    ids, rows = bow.encode(texts)
    return 1 / (1 + np.exp(-bow.scores(ids, rows, len(texts))))


def accuracy(p, y):                                     # share right at a 0.5 threshold
    return float(np.mean((p > 0.5).astype(int) == y))


def negated(text):                                      # does a sentence contain a negation?
    return any(w in NEGATION or w.endswith("n't") for w in d98.words(text))


def day98_distilbert():                                 # DistilBERT's validation accuracy, read from Day 98's own output
    s = open(os.path.join(W15, "day98_run_output.txt"), encoding="utf-8").read()
    return float(re.search(r"DistilBERT\s+30,522\s+([0-9.]+)", s).group(1))


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- four tiny reviews")
    print("=" * 70)
    docs = ["good movie", "great fun", "bad movie", "boring bad"]; y = np.array([1, 1, 0, 0])
    nb = NaiveBayes(alpha=1.0).fit(docs, y)
    print("positive: 'good movie', 'great fun'; negative: 'bad movie', 'boring bad'; 6 distinct words, 4 word slots per class, alpha 1")
    print("for 'good fun movie': positive (1+1)/(4+6) x (1+1)/(4+6) x (1+1)/(4+6) = 0.008; negative (0+1)/10 x (0+1)/10 x (1+1)/10 = 0.002")
    print(f"so P(positive) = 0.008 / (0.008 + 0.002) = 0.8; the class gives {nb.proba(['good fun movie'])[0]:.4f}")
    print(f"an unseen word changes nothing: 'good fun movie zebra' -> {nb.proba(['good fun movie zebra'])[0]:.4f}")
    print("Actual output from this lesson's run.")


def demo_check(Xtr, ytr, Xva):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- against scikit-learn's MultinomialNB, same words")
    print("=" * 70)
    nb = NaiveBayes(alpha=1.0).fit(Xtr, ytr)
    cv = CountVectorizer(tokenizer=d98.words, lowercase=False, token_pattern=None).fit(Xtr)  # the same tokeniser
    sk = MultinomialNB(alpha=1.0).fit(cv.transform(Xtr), ytr)
    lp = sk.predict_log_proba(cv.transform(Xva)); theirs = lp[:, 1] - lp[:, 0]
    ours = nb.log_odds(Xva)
    print(f"vocabulary: {len(nb.vocab):,} words here, {len(cv.vocabulary_):,} in scikit-learn")
    print(f"log-odds on the 872 validation sentences: largest gap {np.abs(ours - theirs).max():.1e}; the same predictions on {np.mean((ours > 0) == (theirs > 0)):.4f}")
    print("Actual output from this lesson's run.")


def demo_kinds(Xtr, ytr, Xva, yva):
    print("\n" + "=" * 70)
    print("3. THE THREE KINDS, AGAINST LOGISTIC REGRESSION AND DISTILBERT -- all 67,349 training items")
    print("=" * 70)
    rows = {}
    for name, m in (("multinomial NB", NaiveBayes("multinomial")), ("binary NB", NaiveBayes("binary")), ("Bernoulli NB", NaiveBayes("bernoulli")),
                    ("binary NB, + bigrams", NaiveBayes("binary", ngrams=2))):
        rows[name] = m.fit(Xtr, ytr).proba(Xva)
    bow = d98.BagOfWords(Xtr).fit(Xtr, ytr); rows["logistic regression (Day 98)"] = lr_proba(bow, Xva)
    print(f"{'model':<32}{'accuracy':>10}{'95% interval':>20}")
    for name, p in rows.items():
        c = ((p > 0.5).astype(int) == yva).astype(float); lo, hi = d98.bootstrap_ci(c)
        print(f"{name:<32}{c.mean():>10.4f}      [{lo:.4f}, {hi:.4f}]")
    print(f"{'DistilBERT (Day 98, its output)':<32}{day98_distilbert():>10.4f}")
    a, b = rows["binary NB"], rows["logistic regression (Day 98)"]
    print(f"binary NB and logistic regression give the same answer on {np.mean((a > 0.5) == (b > 0.5)):.4f} of sentences")
    print("Actual output from this lesson's run.")
    return rows                                              # later sections use them


def demo_curve(Xtr, ytr, Xva, yva):
    print("\n" + "=" * 70)
    print(f"4. LEARNING CURVES -- who wins with little data? ({SEEDS} random subsets per size)")
    print("=" * 70)
    print(f"{'training items':>15}{'binary NB':>12}{'logistic':>11}{'NB - LR':>10}{'NB ahead in':>13}")
    out = []
    for n in SIZES:
        a, b = [], []
        for s in range(SEEDS if n < len(ytr) else 1):     # all items is one subset only
            sub = np.random.RandomState(s).permutation(len(ytr))[:n]; T = [Xtr[i] for i in sub]
            a.append(accuracy(NaiveBayes("binary").fit(T, ytr[sub]).proba(Xva), yva))
            b.append(accuracy(lr_proba(d98.BagOfWords(T).fit(T, ytr[sub]), Xva), yva))
        a, b = np.array(a), np.array(b); out.append((n, a.mean(), b.mean()))
        print(f"{n:>15,}{a.mean():>12.4f}{b.mean():>11.4f}{(a - b).mean():>+10.4f}{int((a > b).sum()):>9} of {len(a)}")
    print("logistic regression is Day 98's recipe, unchanged (300 Adam steps, words in at least 2 training texts); with few items its settings were not re-tuned")
    print("Actual output from this lesson's run.")
    return out                                               # the plot uses it


def demo_alpha(Xtr, ytr, Xva, yva):
    print("\n" + "=" * 70)
    print("5. SMOOTHING -- alpha by 5-fold cross-validation on the training items")
    print("=" * 70)
    folds = np.array_split(np.random.RandomState(0).permutation(len(ytr)), 5)
    print(f"{'alpha':>8}{'CV accuracy':>13}{'validation':>12}")
    cv = []
    for a in ALPHAS:
        acc = []
        for i in range(5):
            va = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
            acc.append(accuracy(NaiveBayes("binary", a).fit([Xtr[k] for k in tr], ytr[tr]).proba([Xtr[k] for k in va]), ytr[va]))
        cv.append(np.mean(acc))
        print(f"{a:>8}{cv[-1]:>13.4f}{accuracy(NaiveBayes('binary', a).fit(Xtr, ytr).proba(Xva), yva):>12.4f}")
    print(f"chosen by CV: alpha {ALPHAS[int(np.argmax(cv))]} (the validation column is printed for comparison, not used to choose)")
    print("Actual output from this lesson's run.")


def demo_independence(rows, Xva, yva):
    print("\n" + "=" * 70)
    print("6. THE INDEPENDENCE ASSUMPTION -- confidence, calibration, repeated words")
    print("=" * 70)
    nb, lr = rows["multinomial NB"], rows["logistic regression (Day 98)"]
    for name, p in (("multinomial NB", nb), ("binary NB", rows["binary NB"]), ("logistic regression", lr)):
        conf = np.maximum(p, 1 - p); c = ((p > 0.5).astype(int) == yva).astype(float); _, e = d98.calibration(conf, c)
        print(f"{name:<22}: mean confidence {conf.mean():.4f}, accuracy {c.mean():.4f}, answers at least 99% sure {np.mean(conf >= 0.99):.4f}, ECE {e:.4f}")
    double = [t + " " + t for t in Xva]                   # every sentence written twice: no new information
    m = NaiveBayes("multinomial").fit(*train_data()); lo1, lo2 = m.log_odds(Xva), m.log_odds(double)
    b = NaiveBayes("binary").fit(*train_data()); lb1, lb2 = b.log_odds(Xva), b.log_odds(double)
    print(f"each validation sentence written twice: multinomial log-odds change by a factor of {np.median((lo2 - m.prior[1] + m.prior[0]) / (lo1 - m.prior[1] + m.prior[0])):.4f} (median, excluding the prior); "
          f"binary NB by {np.median((lb2 - b.prior[1] + b.prior[0]) / (lb1 - b.prior[1] + b.prior[0])):.4f}")
    p2 = 1 / (1 + np.exp(-lo2)); conf2 = np.maximum(p2, 1 - p2); c2 = ((p2 > 0.5).astype(int) == yva).astype(float)
    print(f"multinomial NB on the doubled sentences: accuracy {c2.mean():.4f}, mean confidence {conf2.mean():.4f}, ECE {d98.calibration(conf2, c2)[1]:.4f}")
    print("Actual output from this lesson's run.")


def demo_learned(Xtr, ytr, Xva, yva, rows):
    print("\n" + "=" * 70)
    print("7. WHAT IT LEARNED, AND NEGATION")
    print("=" * 70)
    nb = NaiveBayes("binary").fit(Xtr, ytr); inv = {i: w for w, i in nb.vocab.items()}
    ids, _ = nb.encode(Xtr); df = np.bincount(ids, minlength=len(nb.vocab))  # texts containing each word
    r = nb.lp[1] - nb.lp[0]; ok = df >= 100                # log ratio, for words in at least 100 texts
    top = np.argsort(np.where(ok, r, -np.inf))[::-1][:10]; bot = np.argsort(np.where(ok, r, np.inf))[:10]
    print("most positive (words in at least 100 texts): " + ", ".join(f"{inv[i]} {r[i]:+.2f}" for i in top))
    print("most negative: " + ", ".join(f"{inv[i]} {r[i]:+.2f}" for i in bot))
    for w in ("good", "not", "bad", "funny"):
        print(f"  '{w}': log ratio {r[nb.vocab[w]]:+.3f}")
    neg = np.array([negated(t) for t in Xva])
    print(f"\nvalidation sentences with a negation: {int(neg.sum())}; without: {int((~neg).sum())}")
    for name in ("binary NB", "binary NB, + bigrams", "logistic regression (Day 98)"):
        p = rows[name]; c = (p > 0.5).astype(int) == yva
        print(f"  {name:<30}: with negation {c[neg].mean():.4f}, without {c[~neg].mean():.4f}")
    bi = NaiveBayes("binary", ngrams=2).fit(Xtr, ytr); inv2 = {i: w for w, i in bi.vocab.items()}
    for w in ("not bad", "not good", "not funny"):
        if w in bi.vocab:
            print(f"  bigram '{w}': log ratio {bi.lp[1][bi.vocab[w]] - bi.lp[0][bi.vocab[w]]:+.3f}")
    print("Actual output from this lesson's run.")


_TRAIN = {}


def train_data():                                       # SST-2's training items, loaded once
    if not _TRAIN:
        X, y = d98.sst2("train"); _TRAIN["d"] = (X, y)
    return _TRAIN["d"]


def plot_results(curve, rows, yva):                     # two panels: learning curves | confidence
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))           # canvas
    n = [r[0] for r in curve]
    ax[0].plot(n, [r[1] for r in curve], marker="o", label="binary Naive Bayes"); ax[0].plot(n, [r[2] for r in curve], marker="s", label="logistic regression (Day 98)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("training items (log scale)"); ax[0].set_ylabel("validation accuracy"); ax[0].grid(alpha=0.3)
    ax[0].set_title("SST-2: Naive Bayes ahead with little data, logistic regression with all of it", fontsize=10); ax[0].legend(fontsize=8)
    bins = np.linspace(0.5, 1, 11)
    for name in ("multinomial NB", "logistic regression (Day 98)"):
        p = rows[name]; conf = np.maximum(p, 1 - p); c = (p > 0.5).astype(int) == yva; xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf >= lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean()); ys.append(c[m].mean())
        ax[1].plot(xs, ys, marker="o", label=name)
    ax[1].plot([0.5, 1], [0.5, 1], "k--", linewidth=0.8, label="perfect calibration"); ax[1].set_xlabel("mean confidence in a bin"); ax[1].set_ylabel("accuracy in that bin")
    ax[1].set_title("Calibration (bins with at least 5 sentences)", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("naive_bayes_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved naive_bayes_results.png")


def draw_diagram():                                     # counting, then adding log ratios
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 114: Naive Bayes, training is counting", ha="center", fontsize=12, fontweight="bold")
    box(1.7, 2.3, 2.8, 1.8, "training texts\nwith labels", "#FAEEDA")
    box(5.0, 2.3, 3.0, 1.8, "count each word\nper class, add alpha\n(no training loop)", "#DCEBF7")
    box(8.4, 2.3, 3.0, 1.8, "a new text: add up\nlog P(w | pos) - log P(w | neg)\nfor its words, plus the prior", "#E5F2E1")
    box(11.6, 2.3, 2.4, 1.8, "positive if\nthe total > 0", "#EEEDFE")
    for x0, x1 in ((3.14, 3.46), (6.54, 6.86), (9.94, 10.36)):
        ax.annotate("", xy=(x1, 2.3), xytext=(x0, 2.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(8.4, 0.9, "the 'naive' part: words are treated as independent, so evidence simply adds up", ha="center", fontsize=8.5, style="italic")
    plt.savefig("naive_bayes_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved naive_bayes_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 115 is k-nearest neighbours from scratch, with a real demonstration of the curse of dimensionality.")


def main():                                             # runs every section in order
    Xtr, ytr = train_data(); Xva, yva = d98.sst2("validation")  # SST-2 from the local cache
    demo_hand()                                            # 1. by hand
    demo_check(Xtr, ytr, Xva)                              # 2. correctness
    rows = demo_kinds(Xtr, ytr, Xva, yva)                  # 3. three kinds
    curve = demo_curve(Xtr, ytr, Xva, yva)                 # 4. learning curves
    demo_alpha(Xtr, ytr, Xva, yva)                         # 5. smoothing
    demo_independence(rows, Xva, yva)                      # 6. independence
    demo_learned(Xtr, ytr, Xva, yva, rows)                 # 7. what it learned
    plot_results(curve, rows, yva)                         # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 115


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
