"""
Day 117: Imbalanced Data -- SMOTE Against Class Weighting

When one class is rare, a model can be 95% accurate by never predicting it.
The standard remedies change the training data (oversample the rare class, or
invent new rare examples with SMOTE) or change the loss (weight the rare class
more). Today compares them honestly, and asks what they really change.

Everything is written from scratch in NumPy:
  * metrics: precision, recall, F1, balanced accuracy, ROC AUC and average precision
  * SMOTE: new rare-class points on the line between a rare point and a rare neighbour
  * random over- and undersampling, class-weighted logistic regression, a
    class-balanced random forest, and moving the decision threshold

The data, all real rows, with the imbalance stated each time:
  * digits: one digit (a 9) against the other nine, 10% (as the data comes)
  * breast cancer: malignant tumours subsampled to 5% of the training rows
  * Diabetes: the top 10% of disease-progression scores against the rest

  1. The accuracy paradox.
  2. The metrics, checked against scikit-learn.
  3. SMOTE by hand.
  4. Six remedies, two models, three problems.
  5. Is it just the threshold?
  6. What rebalancing does to the probabilities.
  7. How rare is too rare?

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import earlier lessons

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits  # bundled datasets
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # used ONLY as an outside check
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "week-16-classical-ml-trees"))  # Weeks 16 and 17
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing
    import day116_stacking_blending as d116             # Day 116: fit_softmax, softmax; Days 107, 112, 115 through it
d107, d112, d115 = d116.d107, d116.d112, d116.d115       # the forest, the splits and scaling, the distances
standardise, stratified_split, d99 = d112.standardise, d112.stratified_split, d112.d99  # unchanged helpers
SPLITS = 20                                             # repeated splits
RATES = [0.02, 0.05, 0.1, 0.2, 0.37]                    # section 7: share of malignant tumours in training (0.37 is the data's own)
METHODS = ["none", "class weights", "random oversampling", "SMOTE", "random undersampling", "threshold moved"]  # the remedies


# =============================================================================
# METRICS -- NEW
# =============================================================================
def confusion(y, pred):                                 # true positives, false positives, false negatives, true negatives
    return int(np.sum((pred == 1) & (y == 1))), int(np.sum((pred == 1) & (y == 0))), int(np.sum((pred == 0) & (y == 1))), int(np.sum((pred == 0) & (y == 0)))


def scores(y, p, threshold=0.5):                        # NEW: every metric at once, from labels and P(rare class)
    tp, fp, fn, tn = confusion(y, (p >= threshold).astype(int))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-12),
            "balanced acc": (rec + tn / max(tn + fp, 1)) / 2, "ROC AUC": roc_auc(y, p), "avg precision": avg_precision(y, p)}


def roc_auc(y, p):                                      # NEW: the chance a random rare row outscores a random common one (ties count half)
    r = np.argsort(np.argsort(p, kind="stable"), kind="stable").astype(float)  # ranks 0..n-1
    for v in np.unique(p):                                # tied scores share their average rank
        m = p == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def avg_precision(y, p):                                # NEW: precision averaged over the rare rows, ranked by score
    o = np.argsort(-p, kind="stable"); hits = y[o] == 1
    prec = np.cumsum(hits) / np.arange(1, len(y) + 1)    # precision after each row
    return float(prec[hits].mean())                       # (ties between scores ignored; Section 2 checks how much that matters)


# =============================================================================
# REMEDIES -- NEW
# =============================================================================
def smote(X, k, n_new, rng):                            # NEW: n_new points, each between a rare point and one of its k nearest rare neighbours
    D = d115.sqdist(X, X); np.fill_diagonal(D, np.inf)    # distances among the rare points; never your own neighbour
    nn = np.argsort(D, 1)[:, :k]                          # each point's k nearest
    i = rng.randint(0, len(X), n_new); j = nn[i, rng.randint(0, min(k, len(X) - 1), n_new)]  # a point and one of its neighbours
    return X[i] + rng.uniform(size=(n_new, 1)) * (X[j] - X[i])  # somewhere on the segment between them


def resample(X, y, method, rng):                        # NEW: the training rows each data-level remedy produces
    rare, common = np.where(y == 1)[0], np.where(y == 0)[0]; need = len(common) - len(rare)
    if method == "random oversampling":
        extra = rng.choice(rare, need); return np.vstack([X, X[extra]]), np.r_[y, np.ones(need, int)]
    if method == "SMOTE":
        return np.vstack([X, smote(X[rare], 5, need, rng)]), np.r_[y, np.ones(need, int)]
    if method == "random undersampling":
        keep = np.r_[rare, rng.choice(common, len(rare), replace=False)]; return X[keep], y[keep]
    return X, y


def weighted_logistic(X, y, w, lam=1e-3, steps=500, lr=0.5):  # NEW: Day 116's softmax regression with a weight per row
    mu, sd = X.mean(0), X.std(0) + 1e-12; A = (X - mu) / sd; W = np.zeros((A.shape[1], 2)); b = np.zeros(2); Y = np.eye(2)[y]
    m = [np.zeros_like(W), np.zeros_like(b)]; v = [np.zeros_like(W), np.zeros_like(b)]
    for t in range(1, steps + 1):
        G = w[:, None] * (d116.softmax(A @ W + b) - Y) / w.sum()  # each row's gradient, scaled by its weight
        for i, (p, g) in enumerate(((W, A.T @ G + lam * W), (b, G.sum(0)))):
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return lambda Z: d116.softmax(((Z - mu) / sd) @ W + b)[:, 1]


def balanced_forest(X, y, seed, trees=60):              # NEW: each tree's bootstrap draws equally from both classes
    rng = np.random.RandomState(seed); rare, common = np.where(y == 1)[0], np.where(y == 0)[0]; m = max(1, int(np.sqrt(X.shape[1]))); fs = []
    for _ in range(trees):
        idx = np.r_[rng.choice(rare, len(rare)), rng.choice(common, len(rare))]  # a balanced bootstrap
        fs.append(d107.grow_random(X[idx], y[idx], 2, m, rng))
    return lambda Z: d107.forest_proba(fs, Z, 2)[:, 1]


def fit(model, method, X, y, seed):                     # NEW: a remedy and a model, as P(rare class) for new rows
    rng = np.random.RandomState(seed)
    if method == "threshold moved":                       # train on the data as it is; the threshold is chosen later
        method = "none"
    if method == "class weights":
        if model == "logistic":
            w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))  # each class weighs half in total
            return weighted_logistic(X, y, w)
        return balanced_forest(X, y, seed)
    A, b = resample(X, y, method, rng)
    if model == "logistic":
        return weighted_logistic(A, b, np.ones(len(b)))
    f, _ = d107.fit_forest(A, b, 2, 60, max(1, int(np.sqrt(X.shape[1]))), seed)
    return lambda Z: d107.forest_proba(f, Z, 2)[:, 1]


def best_threshold(y, p):                               # NEW: the threshold with the best F1 on validation rows
    cand = np.unique(p); f = [scores(y, p, t)["F1"] for t in cand]
    return float(cand[int(np.argmax(f))])


def problems():                                         # NEW: three rare-class problems from real rows; each returns a training/test split
    dg, bc, db = load_digits(), load_breast_cancer(), load_diabetes(scaled=False)
    def digits(s):
        y = (dg.target == 9).astype(int); tr, te = stratified_split(y, 0.3, s); return dg.data[tr], y[tr], dg.data[te], y[te]
    def cancer(s, rate=0.05):                           # malignant = 1; training malignant tumours subsampled to the given share
        y = (bc.target == 0).astype(int); tr, te = stratified_split(y, 0.3, s); rng = np.random.RandomState(500 + s)
        r, c = tr[y[tr] == 1], tr[y[tr] == 0]; keep_r = rng.choice(r, max(2, int(round(rate / (1 - rate) * len(c)))), replace=False) if rate < 0.37 else r
        tr = np.r_[c, keep_r]
        te_r, te_c = te[y[te] == 1], te[y[te] == 0]; keep_t = rng.choice(te_r, max(2, int(round(rate / (1 - rate) * len(te_c)))), replace=False) if rate < 0.37 else te_r
        te = np.r_[te_c, keep_t]                          # the test part gets the same share, so it looks like the world the model is for
        return bc.data[tr], y[tr], bc.data[te], y[te]
    def diabetes(s):
        y = (db.target >= np.quantile(db.target, 0.9)).astype(int); tr, te = stratified_split(y, 0.3, s); return db.data[tr], y[tr], db.data[te], y[te]
    return {"Digits: 9 against the rest": digits, "Breast cancer, 5% malignant": cancer, "Diabetes: top 10%": diabetes}


# =============================================================================
# DEMOS
# =============================================================================
def demo_paradox(P):
    print("=" * 70)
    print("1. THE ACCURACY PARADOX")
    print("=" * 70)
    print(f"{'problem':<30}{'rare share':>11}{'always common: acc':>20}{'recall':>8}{'logistic, as is: acc':>22}{'recall':>8}")
    for name, make in P.items():
        X, y, Xt, yt = make(0); p = fit("logistic", "none", X, y, 0)(Xt); s = scores(yt, p)
        print(f"{name:<30}{y.mean():>11.4f}{np.mean(yt == 0):>20.4f}{0.0:>8.4f}{s['accuracy']:>22.4f}{s['recall']:>8.4f}")
    print("a model that never predicts the rare class is highly 'accurate' and useless: recall, the share of rare rows found, is the number to watch")
    print("Actual output from this lesson's run.")


def demo_metrics(P):
    print("\n" + "=" * 70)
    print("2. THE METRICS, CHECKED AGAINST SCIKIT-LEARN")
    print("=" * 70)
    X, y, Xt, yt = P["Digits: 9 against the rest"](0); p = fit("logistic", "none", X, y, 0)(Xt); s = scores(yt, p)
    print(f"digits, one split: ROC AUC {s['ROC AUC']:.6f} against {roc_auc_score(yt, p):.6f}; average precision {s['avg precision']:.6f} against {average_precision_score(yt, p):.6f}; "
          f"F1 {s['F1']:.6f} against {f1_score(yt, (p >= 0.5).astype(int)):.6f}")
    q = np.round(p, 1)                                    # heavy ties, to test the tie handling
    print(f"with scores rounded to one decimal (many ties): ROC AUC {roc_auc(yt, q):.6f} against {roc_auc_score(yt, q):.6f}; average precision {avg_precision(yt, q):.6f} "
          f"against {average_precision_score(yt, q):.6f} (this version ignores ties)")
    print("Actual output from this lesson's run.")


def demo_smote():
    print("\n" + "=" * 70)
    print("3. SMOTE BY HAND")
    print("=" * 70)
    R = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]); rng = np.random.RandomState(0)
    N = smote(R, 1, 6, rng)                               # k = 1: each point's single nearest rare neighbour
    print("rare points (0,0), (1,0), (0,1), (5,5); k = 1, so each new point lies between a rare point and its nearest rare neighbour")
    for q in N:
        print(f"  new point ({q[0]:.3f}, {q[1]:.3f})")
    D = d115.sqdist(R, R); np.fill_diagonal(D, np.inf); nn = D.argmin(1)
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]       # the 2-D cross product: 0 when two vectors point the same way
    on = [min(abs(cross(R[nn[i]] - R[i], q - R[i])) for i in range(4)) < 1e-9 for q in N]  # on some point-neighbour line?
    print(f"every new point lies on such a segment: {all(on)}; (5,5)'s nearest rare neighbour is far away, so its new points cross empty space")
    print("Actual output from this lesson's run.")


def demo_compare(P):
    print("\n" + "=" * 70)
    print(f"4. SIX REMEDIES, TWO MODELS, THREE PROBLEMS -- {SPLITS} splits")
    print("=" * 70)
    print("the threshold is 0.5 except for 'threshold moved', which keeps the data as it is and picks the threshold with the best F1 on 25% of the training rows")
    print("forest 'class weights' is a balanced random forest: every tree's bootstrap draws equally from both classes")
    res = {}
    for name, make in P.items():
        for model in ("logistic", "forest"):
            R = {m: [] for m in METHODS}
            for s in range(SPLITS):
                X, y, Xt, yt = make(s)
                for m in METHODS:
                    if m == "threshold moved":                  # choose the threshold on held-out training rows, then refit on all
                        o = np.random.RandomState(s).permutation(len(y)); v = len(y) // 4
                        t = best_threshold(y[o[:v]], fit(model, m, X[o[v:]], y[o[v:]], s)(X[o[:v]])) if y[o[:v]].sum() else 0.5
                        R[m].append(scores(yt, fit(model, m, X, y, s)(Xt), t))
                    else:
                        R[m].append(scores(yt, fit(model, m, X, y, s)(Xt)))
            res[(name, model)] = {m: {k: float(np.mean([r[k] for r in v])) for k in v[0]} for m, v in R.items()}
            print(f"\n{name}, {model}:")
            print(f"  {'remedy':<22}" + "".join(f"{k:>14}" for k in ("precision", "recall", "F1", "balanced acc", "ROC AUC", "avg precision")))
            for m, r in res[(name, model)].items():
                print(f"  {m:<22}" + "".join(f"{r[k]:>14.4f}" for k in ("precision", "recall", "F1", "balanced acc", "ROC AUC", "avg precision")))
    print("Actual output from this lesson's run.")
    return res                                               # the plot uses it


def demo_threshold(res):
    print("\n" + "=" * 70)
    print("5. IS IT JUST THE THRESHOLD?")
    print("=" * 70)
    print("ROC AUC and average precision do not depend on the threshold: they measure how well rare rows are RANKED above common ones")
    print(f"{'problem, model':<40}{'AUC range across remedies':>27}{'F1 range':>12}")
    for (name, model), r in res.items():
        a = [r[m]["ROC AUC"] for m in METHODS]; f = [r[m]["F1"] for m in METHODS]
        print(f"{name + ', ' + model:<40}{min(a):>13.4f} to {max(a):.4f}{min(f):>7.4f} to {max(f):.4f}")
    print("Actual output from this lesson's run.")


def demo_probabilities(P):
    print("\n" + "=" * 70)
    print("6. WHAT REBALANCING DOES TO THE PROBABILITIES -- logistic regression")
    print("=" * 70)
    print(f"mean predicted P(rare) on test rows, against the true rare share; {SPLITS} splits")
    print(f"{'problem':<30}{'true share':>11}" + "".join(f"{m:>22}" for m in ("none", "class weights", "SMOTE")) + f"{'weights / true':>16}")
    for name, make in P.items():
        v = {m: [] for m in ("none", "class weights", "SMOTE")}; t = []
        for s in range(SPLITS):
            X, y, Xt, yt = make(s); t.append(yt.mean())
            for m in v:
                v[m].append(fit("logistic", m, X, y, s)(Xt).mean())
        print(f"{name:<30}{np.mean(t):>11.4f}" + "".join(f"{np.mean(v[m]):>22.4f}" for m in v) + f"{np.mean(v['class weights']) / np.mean(t):>16.2f}")
    print("rebalancing inflates the predicted rare share most where the classes overlap (Diabetes) and hardly at all where they separate cleanly (breast cancer)")
    print("Actual output from this lesson's run.")


def demo_rarity(P):
    print("\n" + "=" * 70)
    print("7. HOW RARE IS TOO RARE? breast cancer, logistic regression, the malignant share varied")
    print("=" * 70)
    make = P["Breast cancer, 5% malignant"]; print(f"{'malignant share':>16}{'rare training rows':>20}" + "".join(f"{m:>17}" for m in ("none: recall", "weights: recall", "SMOTE: recall", "none: AUC")))
    rows = []
    for rate in RATES:
        rec = {m: [] for m in ("none", "class weights", "SMOTE")}; auc, n1 = [], []
        for s in range(SPLITS):
            X, y, Xt, yt = make(s, rate); n1.append(y.sum())               # the same split, a different malignant share
            for m in rec:
                p = fit("logistic", m, X, y, s)(Xt); r = scores(yt, p); rec[m].append(r["recall"])
                if m == "none":
                    auc.append(r["ROC AUC"])
        rows.append((rate, {m: np.mean(v) for m, v in rec.items()}, np.mean(auc)))
        print(f"{rate:>16.2f}{np.mean(n1):>20.1f}" + "".join(f"{np.mean(rec[m]):>17.4f}" for m in rec) + f"{np.mean(auc):>17.4f}")
    print("Actual output from this lesson's run.")
    return rows                                              # the plot uses them


def plot_results(res, rows):                            # two panels: recall and F1 by remedy | rarity
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))             # canvas
    a = ax[0]; keys = [k for k in res if k[1] == "logistic"]; x = np.arange(len(METHODS)); w = 0.27
    for i, k in enumerate(keys):
        a.bar(x + (i - 1) * w, [res[k][m]["F1"] for m in METHODS], w, label=k[0])
    a.set_xticks(x); a.set_xticklabels([m.replace(" ", "\n", 1) for m in METHODS], fontsize=8); a.set_ylabel("F1 (logistic regression)"); a.set_ylim(0, 1)
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3); a.grid(alpha=0.3, axis="y"); a.set_title("F1 by remedy", fontsize=10)
    a = ax[1]; r = [x[0] for x in rows]
    for m, mk in (("none", "o"), ("class weights", "s"), ("SMOTE", "^")):
        a.plot(r, [x[1][m] for x in rows], marker=mk, label=f"recall, {m}")
    a.plot(r, [x[2] for x in rows], marker="d", linestyle="--", color="k", label="ROC AUC, no remedy")
    a.set_xscale("log"); a.set_xticks(RATES); a.set_xticklabels([f"{r:.2f}" for r in RATES]); a.minorticks_off()  # label the shares tried
    a.set_xlabel("malignant share in training (log scale)"); a.set_ylabel("score"); a.set_ylim(0, 1.02)
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.3); a.set_title("Breast cancer: how rare is too rare?", fontsize=10)
    plt.tight_layout(); plt.savefig("imbalance_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved imbalance_results.png")


def draw_diagram():                                     # where each remedy acts
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 117: three places to fix imbalance", ha="center", fontsize=12, fontweight="bold")
    box(2.0, 2.2, 3.2, 1.8, "the DATA:\noversample, SMOTE,\nundersample", "#FAEEDA")
    box(6.5, 2.2, 3.2, 1.8, "the LOSS:\nweight rare rows more\n(class weights)", "#DCEBF7")
    box(11.0, 2.2, 3.2, 1.8, "the DECISION:\nmove the threshold\non the model's scores", "#E5F2E1")
    ax.annotate("", xy=(4.86, 2.2), xytext=(3.64, 2.2), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(9.36, 2.2), xytext=(8.14, 2.2), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(4.25, 2.45, "train", ha="center", fontsize=8); ax.text(8.75, 2.45, "score", ha="center", fontsize=8)
    ax.text(6.5, 0.6, "measure with recall, precision, F1 and AUC: accuracy rewards ignoring the rare class", ha="center", fontsize=8.5, style="italic")
    plt.savefig("imbalance_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved imbalance_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 118 is the week's build: model selection and cross-validation strategies (k-fold, stratified, and time-series splits), using the week's parts.")


def main():                                             # runs every section in order
    P = problems()                                         # three rare-class problems
    demo_paradox(P)                                        # 1. the paradox
    demo_metrics(P)                                        # 2. metrics
    demo_smote()                                           # 3. SMOTE by hand
    res = demo_compare(P)                                  # 4. six remedies
    demo_threshold(res)                                    # 5. the threshold
    demo_probabilities(P)                                  # 6. probabilities
    rows = demo_rarity(P)                                  # 7. how rare
    plot_results(res, rows)                                # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 118


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
