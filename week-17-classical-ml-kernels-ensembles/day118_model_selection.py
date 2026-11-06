"""
Day 118: The Week 17 Build -- Model Selection and Cross-Validation Strategies

Every result this week came from cross-validation (CV): split the training rows
into folds, train on some, score on the rest. But how the folds are cut
matters, and a CV score used to CHOOSE a model is no longer a fair estimate of
that model. Today tests both, with the week's six exercise parts copied in
verbatim (one banner per file) and assembled into models.

  1. Checks: the parts are verbatim.
  2. k-fold against stratified folds, on rare classes.
  3. How good is a CV estimate? 2, 5 and 10 folds, and leave-one-out.
  4. Choosing hurts: the optimism of the best CV score, and nested CV.
  5. Time: shuffled folds against forward-chaining, on real CO2 and sunspot records.
  6. Choosing a model for the future: what each CV strategy picks.

Data: digits and breast cancer (bundled with scikit-learn), and two real time
series bundled with statsmodels: weekly Mauna Loa CO2 (1958-2001) and yearly
sunspot numbers (1700-2008).

Every line written for today carries its own comment, continuing Day 52's
convention; the copied parts keep their own.
"""
import ast                                             # ast: reading function sources out of the exercise files
import contextlib                                      # contextlib: redirect_stdout, to keep imports quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import re                                              # re: Day 114's tokeniser (in the parts)
import sys                                             # sys: path, to import Day 99's bootstrap

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import statsmodels.api as sm                           # statsmodels: ONLY for its bundled CO2 and sunspot records
from sklearn.datasets import load_breast_cancer, load_digits  # bundled datasets
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, os.path.join(HERE, "..", "week-15-practical-nlp-pretrained"))  # Day 99's exercise
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day99_exercise as d99                        # Day 99: bootstrap_ci
EPS = 1e-6                                              # Day 116's clipping constant, which its part needs
REPEATS = 50                                            # section 2: reshuffles of the folds
DRAWS = 40                                              # section 3: training sets drawn
SPLITS = 10                                             # section 4: outer splits
LAGS = 8                                                # section 5: past values used to predict the next one
KS = [1, 3, 5, 10, 20, 40]                              # section 6: kNN neighbours on the time series


SKIP = {(115, 'sqdist'), (117, 'sqdist')}                 # parts left out as duplicates
HELPERS = {'spy', 'useful', 'useless'}          # hand-check helpers, not parts


# =============================================================================
# from day112_exercise.py, verbatim: smo, bias, decision
# =============================================================================
def smo(K, y, C, tol=1e-3, max_iter=200000):            # NEW: Day 112's solver, unchanged
    n = len(y); a = np.zeros(n); f = np.zeros(n)
    for it in range(max_iter):
        G = y * f - 1
        up = ((a < C) & (y > 0)) | ((a > 0) & (y < 0))
        low = ((a < C) & (y < 0)) | ((a > 0) & (y > 0))
        v = -y * G
        i = np.where(up)[0][np.argmax(v[up])]; j = np.where(low)[0][np.argmin(v[low])]
        if v[i] - v[j] < tol:
            break
        eta = max(K[i, i] + K[j, j] - 2 * K[i, j], 1e-12)
        if y[i] != y[j]:
            L, H = max(0.0, a[j] - a[i]), min(C, C + a[j] - a[i])
        else:
            L, H = max(0.0, a[i] + a[j] - C), min(C, a[i] + a[j])
        Ei, Ej = f[i] - y[i], f[j] - y[j]
        aj = np.clip(a[j] + y[j] * (Ei - Ej) / eta, L, H)
        ai = a[i] + y[i] * y[j] * (a[j] - aj)
        ai, aj = [0.0 if v < 1e-12 * C else C if v > C * (1 - 1e-12) else v for v in (ai, aj)]  # snap rounding errors onto the bounds
        f += (ai - a[i]) * y[i] * K[:, i] + (aj - a[j]) * y[j] * K[:, j]
        a[i], a[j] = ai, aj
    return a, it + 1


def bias(a, y, K, C):                                   # NEW: b from the KKT conditions
    f = K @ (a * y); e = 1e-8 * max(C, 1)
    free = (a > e) & (a < C - e)
    if free.any():
        return float(np.mean(y[free] - f[free]))
    lo = (a <= e) & (y > 0) | (a >= C - e) & (y < 0)
    return float((np.max((y - f)[lo]) + np.min((y - f)[~lo])) / 2)


def decision(a, y, b, K_new):                           # NEW: sum_i a_i y_i K(x_i, x) + b, for each new point
    return K_new.T @ (a * y) + b


# =============================================================================
# from day113_exercise.py, verbatim: linear, poly, sqdist, rbf, phi_poly2, rff
# =============================================================================
def linear(A, B):                                       # NEW: plain dot products
    return A @ B.T


def poly(A, B, degree=2, c=1.0):                        # NEW: (x.z + c)^degree
    return (A @ B.T + c) ** degree


def sqdist(A, B):                                       # all squared distances at once
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def rbf(A, B, gamma):                                   # NEW: exp(-gamma |x - z|^2)
    return np.exp(-gamma * np.maximum(sqdist(A, B), 0))


def phi_poly2(X):                                       # NEW: the features behind (x.z + 1)^2
    n, p = X.shape; cols = [np.ones(n)] + [np.sqrt(2) * X[:, i] for i in range(p)]
    cols += [X[:, i] ** 2 for i in range(p)]
    cols += [np.sqrt(2) * X[:, i] * X[:, j] for i in range(p) for j in range(i + 1, p)]
    return np.column_stack(cols)


def rff(X, D, gamma, seed):                             # NEW: random Fourier features for the RBF kernel
    rng = np.random.RandomState(seed)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(X.shape[1], D))
    c = rng.uniform(0, 2 * np.pi, D)
    return lambda Z: np.sqrt(2.0 / D) * np.cos(Z @ W + c)


# =============================================================================
# from day114_exercise.py, verbatim: words, tokens, NaiveBayes
# =============================================================================
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


# =============================================================================
# from day115_exercise.py, verbatim: knn_predict, pca_fit
# =============================================================================
def knn_predict(Xtr, ytr, X, k, chunk=500):             # NEW: Day 115's vote, unchanged
    n_classes = int(ytr.max()) + 1; out = []
    for s in range(0, len(X), chunk):
        D = sqdist(X[s:s + chunk], Xtr)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k] if k < len(ytr) else np.tile(np.arange(len(ytr)), (len(D), 1))
        votes = np.zeros((len(D), n_classes))
        for j in range(k):
            votes[np.arange(len(D)), ytr[idx[:, j]]] += 1
        d = np.take_along_axis(D, idx, 1)
        tie = (votes == votes.max(1, keepdims=True)).sum(1) > 1
        for r in np.where(tie)[0]:
            best = np.where(votes[r] == votes[r].max())[0]
            order = idx[r][np.argsort(d[r])]
            out_r = next(ytr[i] for i in order if ytr[i] in best); votes[r, out_r] += 0.5
        out.append(np.argmax(votes, 1))
    return np.concatenate(out)


def pca_fit(X, k):                                      # NEW: the k directions of largest variance, by SVD
    mu = X.mean(0); _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k], (s ** 2) / (s ** 2).sum()


# =============================================================================
# from day116_exercise.py, verbatim: softmax, fit_softmax, features, oof, stack_fit, stack_proba
# =============================================================================
def softmax(Z):                                         # NEW: rows of scores to rows of probabilities
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def fit_softmax(F, y, k, lam=1e-3, steps=500, lr=0.5):  # NEW: Day 116's softmax regression, unchanged
    W = np.zeros((F.shape[1], k)); b = np.zeros(k); Y = np.eye(k)[y]
    m = [np.zeros_like(W), np.zeros_like(b)]; v = [np.zeros_like(W), np.zeros_like(b)]
    for t in range(1, steps + 1):
        G = (softmax(F @ W + b) - Y) / len(y)
        for i, (p, g) in enumerate(((W, F.T @ G + lam * W), (b, G.sum(0)))):
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def features(P, k):                                     # NEW: log-odds or log-probabilities
    P = np.clip(P, EPS, 1 - EPS)
    return np.log(P[:, 1] / P[:, 0])[:, None] if k == 2 else np.log(P)


def oof(fit, X, y, k, seed, folds=5):                   # NEW: out-of-fold predictions
    parts = np.array_split(np.random.RandomState(seed).permutation(len(y)), folds); P = np.zeros((len(y), k))
    for i in range(folds):
        va = parts[i]; tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        P[va] = fit(X[tr], y[tr], k, seed)(X[va])
    return P


def stack_fit(fits, X, y, k, seed):                     # NEW: base models on all rows, the meta-model on their out-of-fold predictions
    base = [f(X, y, k, seed) for f in fits]
    W, b = fit_softmax(np.hstack([features(oof(f, X, y, k, seed), k) for f in fits]), y, k)
    return base, W, b, k


def stack_proba(model, X):                              # NEW: the stacked prediction
    base, W, b, k = model
    return softmax(np.hstack([features(p(X), k) for p in base]) @ W + b)


# =============================================================================
# from day117_exercise.py, verbatim: roc_auc, avg_precision, scores, smote
# =============================================================================
def roc_auc(y, p):                                      # NEW: Day 117's rank form, unchanged
    r = np.argsort(np.argsort(p, kind="stable"), kind="stable").astype(float)
    for v in np.unique(p):
        m = p == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def avg_precision(y, p):                                # NEW: precision at each rare row, averaged
    o = np.argsort(-p, kind="stable"); hits = y[o] == 1
    prec = np.cumsum(hits) / np.arange(1, len(y) + 1)
    return float(prec[hits].mean())


def scores(y, p, threshold=0.5):                        # NEW: every metric at once
    pred = (p >= threshold).astype(int)
    tp, fp = int(np.sum((pred == 1) & (y == 1))), int(np.sum((pred == 1) & (y == 0)))
    fn, tn = int(np.sum((pred == 0) & (y == 1))), int(np.sum((pred == 0) & (y == 0)))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-12),
            "balanced acc": (rec + tn / max(tn + fp, 1)) / 2, "ROC AUC": roc_auc(y, p), "avg precision": avg_precision(y, p)}


def smote(X, k, n_new, rng):                            # NEW: Day 117's SMOTE, unchanged
    D = sqdist(X, X); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :k]
    i = rng.randint(0, len(X), n_new); j = nn[i, rng.randint(0, min(k, len(X) - 1), n_new)]
    return X[i] + rng.uniform(size=(n_new, 1)) * (X[j] - X[i])


# =============================================================================
# FOLDS -- NEW
# =============================================================================
def kfold(n, k, seed):                                  # NEW: shuffled folds, class labels ignored
    return np.array_split(np.random.RandomState(seed).permutation(n), k)


def stratified(y, k, seed):                             # NEW: each class dealt round the folds, so every fold has its share
    rng = np.random.RandomState(seed); folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, j in enumerate(idx):
            folds[i % k].append(j)                          # deal the class's rows like cards
    return [np.array(sorted(f)) for f in folds]


def forward(n, k, min_train):                           # NEW: forward-chaining: train on the past, test on the next block
    edges = np.linspace(min_train, n, k + 1).astype(int)
    return [(np.arange(edges[i]), np.arange(edges[i], edges[i + 1])) for i in range(k)]


def standardise(A, B):                                  # Day 112's scaling, unchanged
    mu, sd = A.mean(0), A.std(0) + 1e-12
    return (A - mu) / sd, (B - mu) / sd


def split(y, frac, seed):                               # Day 105's stratified split, unchanged
    rng = np.random.RandomState(seed); tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); k = int(round(len(idx) * frac)); te += list(idx[:k]); tr += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


# =============================================================================
# MODELS FROM THE PARTS -- NEW
# =============================================================================
def svm_rbf(X, y, gamma, C):                            # Days 112-113: an RBF SVM; returns its decision function
    A, mu, sd = (X - X.mean(0)) / (X.std(0) + 1e-12), X.mean(0), X.std(0) + 1e-12; t = np.where(y == 1, 1.0, -1.0)
    K = rbf(A, A, gamma); a, _ = smo(K, t, C); b = bias(a, t, K, C)
    return lambda Z: decision(a, t, b, rbf(A, (Z - mu) / sd, gamma))


def knn(X, y, k):                                       # Day 115: kNN on standardised features; returns its predictions
    mu, sd = X.mean(0), X.std(0) + 1e-12
    return lambda Z: knn_predict((X - mu) / sd, y, (Z - mu) / sd, k)


def candidates():                                       # NEW: the models the selection chooses between
    c = {f"SVM gamma {g} C {C}": (lambda X, y, g=g, C=C: (lambda f: lambda Z: (f(Z) > 0).astype(int))(svm_rbf(X, y, g, C)))
         for g in (0.003, 0.03, 0.3) for C in (0.1, 1, 10)}
    c.update({f"kNN k {k}": (lambda X, y, k=k: knn(X, y, k)) for k in (1, 5, 15, 35)})
    return c


def cv_score(fit, X, y, folds):                         # accuracy averaged over folds
    out = []
    for i in range(len(folds)):
        va = folds[i]; tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        out.append(np.mean(fit(X[tr], y[tr])(X[va]) == y[va]))
    return float(np.mean(out))


def lagged(series, lags):                               # NEW: rows of the last `lags` values, and the next value as the target
    X = np.stack([series[i:len(series) - lags + i] for i in range(lags)], 1)
    return X, series[lags:]


def knn_reg(X, y, k):                                   # Day 115's distances, averaging the neighbours' targets: kNN regression
    return lambda Z: y[np.argpartition(sqdist(Z, X), k - 1, axis=1)[:, :k]].mean(1)


def linear_reg(X, y):                                   # least squares on the lags, with an intercept
    A = np.column_stack([np.ones(len(X)), X]); w = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ w


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def series():                                           # the two real records
    co2 = sm.datasets.co2.load_pandas().data["co2"].interpolate().to_numpy()  # 59 missing weeks filled by straight lines
    sun = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy()
    return {"CO2 (weekly, ppm)": co2, "Sunspots (yearly)": sun}


# =============================================================================
# DEMOS
# =============================================================================
def demo_checks():
    print("=" * 70)
    print("1. CHECKS -- the parts are verbatim")
    print("=" * 70)
    me = open(os.path.abspath(__file__), encoding="utf-8").read(); found, total, dup = 0, 0, []
    for d in range(112, 118):
        src = open(os.path.join(HERE, f"day{d}_exercise.py"), encoding="utf-8").read(); lines = src.split("\n")
        for node in ast.parse(src).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in HELPERS:
                text = "\n".join(lines[node.lineno - 1:node.end_lineno])
                if (d, node.name) in SKIP:
                    dup.append(f"{node.name} (day{d})"); continue
                total += 1; found += text in me
    print(f"functions and classes from day112_exercise.py to day117_exercise.py found here, character for character: {found} of {total}")
    print(f"left out as duplicates of Day 113's identical sqdist: {', '.join(dup)}; the hand-check helpers of Day 116 were not copied")
    print("Actual output from this lesson's run.")


def demo_strata(sets):
    print("\n" + "=" * 70)
    print(f"2. k-FOLD AGAINST STRATIFIED FOLDS, ON RARE CLASSES -- 5 folds, {REPEATS} reshuffles")
    print("=" * 70)
    print("Day 117's metrics on kNN (k = 5) predictions; the spread is how much the CV estimate changes when only the shuffle changes")
    out = {}
    for name, (X, y) in sets.items():
        R = {"k-fold": [], "stratified": []}; zero = {"k-fold": 0, "stratified": 0}; counts = {"k-fold": [], "stratified": []}
        for s in range(REPEATS):
            for kind in R:
                folds = kfold(len(y), 5, s) if kind == "k-fold" else stratified(y, 5, s)
                f1 = []
                for i in range(5):
                    va = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i]); counts[kind].append(int(y[va].sum()))
                    if y[va].sum() == 0:
                        zero[kind] += 1                          # a fold with no rare rows cannot measure recall at all
                        continue
                    f1.append(scores(y[va], knn(X[tr], y[tr], 5)(X[va]).astype(float))["F1"])
                R[kind].append(np.mean(f1))
        out[name] = R
        print(f"\n{name}: {len(y)} rows, {int(y.sum())} rare ({y.mean():.3f})")
        for kind in R:
            c = np.array(counts[kind])
            print(f"  {kind:<11}: rare rows per fold {c.min()} to {c.max()}; folds with none {zero[kind]} of {5 * REPEATS}; CV F1 {np.mean(R[kind]):.4f}, spread across reshuffles (std) {np.std(R[kind]):.4f}")
    print("Actual output from this lesson's run.")
    return out


def demo_estimate():
    print("\n" + "=" * 70)
    print(f"3. HOW GOOD IS A CV ESTIMATE? digits, kNN (k = 3), {DRAWS} training sets of 150 images")
    print("=" * 70)
    d = load_digits(); X, y = d.data, d.target
    print("each training set's CV estimate is compared with the model's accuracy on all the other 1,647 images: the truth, as near as it can be measured")
    print(f"{'folds':>14}{'mean estimate':>15}{'mean truth':>12}{'bias':>9}{'error std':>11}")
    rows = []
    for k in (2, 5, 10, 150):
        est, tru = [], []
        for s in range(DRAWS):
            rng = np.random.RandomState(s); idx = rng.permutation(len(y)); a, rest = idx[:150], idx[150:]
            est.append(cv_score(lambda A, b: (lambda Z: knn_predict(A, b, Z, 3)), X[a], y[a], kfold(150, k, s)))
            tru.append(np.mean(knn_predict(X[a], y[a], X[rest], 3) == y[rest]))
        e = np.array(est) - np.array(tru); rows.append((k, e))
        lab = "leave-one-out" if k == 150 else f"{k}-fold"
        print(f"{lab:>14}{np.mean(est):>15.4f}{np.mean(tru):>12.4f}{e.mean():>+9.4f}{e.std():>11.4f}")
    print("fewer folds train on fewer rows, so they underestimate; every estimate still misses a given training set's true accuracy by a few points")
    print("Actual output from this lesson's run.")
    return rows                                              # the plot uses them


def demo_nested(X, y):
    print("\n" + "=" * 70)
    print(f"4. THE COST OF CHOOSING -- the best CV score, nested CV and the test ({SPLITS} splits, breast cancer)")
    print("=" * 70)
    C = candidates(); print(f"{len(C)} candidates: 9 RBF SVMs (gamma x C) and 4 kNNs; 5-fold stratified CV on the training part chooses one")
    rows = []
    for s in range(SPLITS):
        tr, te = split(y, 0.3, s); Xt, yt = X[tr], y[tr]
        cv = {n: cv_score(f, Xt, yt, stratified(yt, 5, s)) for n, f in C.items()}; best = max(cv, key=cv.get)
        test = float(np.mean(C[best](Xt, yt)(X[te]) == y[te]))
        outer = stratified(yt, 5, 100 + s); nested = []
        for i in range(5):                                  # nested: the whole choice is repeated inside each outer fold
            va = outer[i]; ft = np.concatenate([outer[j] for j in range(5) if j != i])
            inner = {n: cv_score(f, Xt[ft], yt[ft], stratified(yt[ft], 5, s)) for n, f in C.items()}
            nested.append(np.mean(C[max(inner, key=inner.get)](Xt[ft], yt[ft])(Xt[va]) == yt[va]))
        rows.append((cv[best], float(np.mean(nested)), test, best))
    r = np.array([x[:3] for x in rows])
    print(f"best CV score (what is often reported):  {r[:, 0].mean():.4f}")
    print(f"nested CV estimate:                       {r[:, 1].mean():.4f}")
    print(f"the chosen model's test accuracy:         {r[:, 2].mean():.4f}")
    for i, n in ((0, "best CV"), (1, "nested CV")):
        d = r[:, i] - r[:, 2]; lo, hi = d99.bootstrap_ci(d)
        print(f"  {n:<10} minus test: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"models chosen: {sorted({x[3]: [z[3] for z in rows].count(x[3]) for x in rows}.items())}")
    print("Actual output from this lesson's run.")
    return r


def demo_time():
    print("\n" + "=" * 70)
    print("5. TIME -- shuffled folds against forward-chaining, on real records")
    print("=" * 70)
    print(f"predict the next value from the last {LAGS}; the first 80% of each series is for choosing and estimating, the last 20% is the real future")
    print("forward-chaining: 5 blocks, each predicted by a model trained only on everything before it")
    out = {}
    for name, s in series().items():
        X, y = lagged(s, LAGS); cut = int(0.8 * len(y)); Xa, ya, Xf, yf = X[:cut], y[:cut], X[cut:], y[cut:]
        print(f"\n{name}: {len(s)} values; past {cut} rows, future {len(y) - cut} rows; RMSE in the series' own units")
        print(f"  {'model':<18}{'shuffled 5-fold':>17}{'forward-chaining':>18}{'the real future':>17}")
        res = {}
        for mname, make in (("kNN, k = 5", lambda A, b: knn_reg(A, b, 5)), ("linear on lags", linear_reg)):
            sh = []
            for f in kfold(len(ya), 5, 0):
                tr = np.setdiff1d(np.arange(len(ya)), f); sh.append(rmse(make(Xa[tr], ya[tr])(Xa[f]), ya[f]))
            fw = [rmse(make(Xa[tr], ya[tr])(Xa[te]), ya[te]) for tr, te in forward(len(ya), 5, len(ya) // 3)]
            fut = rmse(make(Xa, ya)(Xf), yf); res[mname] = (np.mean(sh), np.mean(fw), fut)
            print(f"  {mname:<18}{np.mean(sh):>17.3f}{np.mean(fw):>18.3f}{fut:>17.3f}")
        out[name] = (s, cut + LAGS, res)
    print("Actual output from this lesson's run.")
    return out


def demo_choose_time():
    print("\n" + "=" * 70)
    print("6. CHOOSING A MODEL FOR THE FUTURE -- kNN's k, chosen by each strategy")
    print("=" * 70)
    for name, s in series().items():
        X, y = lagged(s, LAGS); cut = int(0.8 * len(y)); Xa, ya, Xf, yf = X[:cut], y[:cut], X[cut:], y[cut:]
        cands = {f"kNN k {k}": (lambda A, b, k=k: knn_reg(A, b, k)) for k in KS}  # (linear regression wins everywhere, so it is left out here)
        sh, fw, fut = {}, {}, {}
        for n, make in cands.items():
            sh[n] = np.mean([rmse(make(Xa[np.setdiff1d(np.arange(len(ya)), f)], ya[np.setdiff1d(np.arange(len(ya)), f)])(Xa[f]), ya[f]) for f in kfold(len(ya), 5, 0)])
            fw[n] = np.mean([rmse(make(Xa[tr], ya[tr])(Xa[te]), ya[te]) for tr, te in forward(len(ya), 5, len(ya) // 3)])
            fut[n] = rmse(make(Xa, ya)(Xf), yf)
        a, b = min(sh, key=sh.get), min(fw, key=fw.get)
        print(f"{name}:")
        print(f"  {'':<10}" + "".join(f"{n:>11}" for n in cands))
        for lab, v in (("shuffled", sh), ("forward", fw), ("future", fut)):
            print(f"  {lab:<10}" + "".join(f"{v[n]:>11.3f}" for n in cands))
        print(f"  shuffled CV chooses {a} (future RMSE {fut[a]:.3f}); forward-chaining chooses {b} ({fut[b]:.3f}); best in hindsight {min(fut, key=fut.get)} ({min(fut.values()):.3f})")
    print("Actual output from this lesson's run.")


def plot_results(strata, rows, nested, ts):             # four panels
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.2))             # canvas
    a = ax[0, 0]; names = list(strata); x = np.arange(len(names))
    for i, kind in enumerate(("k-fold", "stratified")):
        a.bar(x + (i - 0.5) * 0.36, [np.std(strata[n][kind]) for n in names], 0.36, label=kind)
    a.set_xticks(x); a.set_xticklabels(names, fontsize=8); a.set_ylabel("std of the CV F1 across reshuffles"); a.grid(alpha=0.3, axis="y")
    a.set_title("Rare classes: how much the CV estimate wobbles", fontsize=10); a.legend(fontsize=8)
    a = ax[0, 1]; a.boxplot([e for _, e in rows], tick_labels=["2-fold", "5-fold", "10-fold", "leave-one-out"]); a.axhline(0, color="k", linewidth=0.8)
    a.set_ylabel("CV estimate minus true accuracy"); a.set_title("Digits: the error of a CV estimate", fontsize=10); a.grid(alpha=0.3, axis="y")
    a = ax[1, 0]; a.bar(range(3), nested.mean(0), color=["#d62728", "#2ca02c", "#1f77b4"]); a.set_xticks(range(3))
    a.set_xticklabels(["best CV score", "nested CV", "test accuracy"]); a.set_ylim(nested.min() - 0.02, 1.0); a.set_ylabel("accuracy")
    for i, v in enumerate(nested.mean(0)):
        a.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    a.set_title("Breast cancer: the cost of choosing", fontsize=10); a.grid(alpha=0.3, axis="y")
    a = ax[1, 1]; s, start, res = ts["CO2 (weekly, ppm)"]; X, y = lagged(s, LAGS); cut = start - LAGS
    a.plot(np.arange(len(s)), s, color="#999999", linewidth=0.8, label="CO2, the real record")
    a.plot(np.arange(start, len(s)), knn_reg(X[:cut], y[:cut], 5)(X[cut:]), color="#d62728", linewidth=1, label="kNN's forecast of the future part")
    a.axvline(start, color="k", linestyle="--", linewidth=0.8); a.set_xlabel("week since March 1958"); a.set_ylabel("ppm")
    a.set_title("CO2: kNN cannot go above anything it has seen", fontsize=10); a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("model_selection_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved model_selection_results.png")


def draw_diagram():                                     # the three ways to cut folds
    fig, ax = plt.subplots(figsize=(13, 4.6))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")     # coordinates in "boxes"
    ax.text(6.5, 4.35, "Day 118: three ways to cut folds (blue = train, orange = score)", ha="center", fontsize=12, fontweight="bold")
    rng = np.random.RandomState(3); order = rng.permutation(20)
    for row, (title, y0) in enumerate((("k-fold: shuffled, classes ignored", 3.3), ("stratified: each class shared out", 2.0), ("forward-chaining: train on the past only", 0.7))):
        ax.text(0.1, y0 + 0.62, title, fontsize=9, fontweight="bold")
        for f in range(3):                                   # three of the five rounds shown
            for i in range(20):
                if row < 2:
                    c = "#ff7f0e" if (order[i] % 5 == f if row == 0 else i % 5 == f) else "#1f77b4"
                else:
                    c = "#1f77b4" if i < 8 + 4 * f else ("#ff7f0e" if i < 12 + 4 * f else "#eeeeee")
                ax.add_patch(FancyBboxPatch((0.2 + f * 4.3 + i * 0.19, y0), 0.15, 0.4, boxstyle="square,pad=0", facecolor=c, edgecolor="none"))
    plt.savefig("model_selection_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved model_selection_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 119 begins Week 18, unsupervised learning: k-means from scratch, with a real elbow-method demonstration.")


def main():                                             # runs every section in order
    dg, bc = load_digits(), load_breast_cancer()
    m = np.where(bc.target == 0)[0]; keep = np.r_[np.where(bc.target == 1)[0], np.random.RandomState(0).choice(m, 19, replace=False)]  # Day 117's 5% malignant
    rare = {"digits: 9 against the rest": (dg.data, (dg.target == 9).astype(int)), "breast cancer, 5% malignant": (bc.data[keep], (bc.target[keep] == 0).astype(int))}
    demo_checks()                                          # 1. checks
    strata = demo_strata(rare)                             # 2. stratification
    rows = demo_estimate()                                 # 3. how good is CV
    nested = demo_nested(bc.data, bc.target)               # 4. nested CV
    ts = demo_time()                                       # 5. time
    demo_choose_time()                                     # 6. choosing for the future
    plot_results(strata, rows, nested, ts)                 # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 7. Day 119


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
