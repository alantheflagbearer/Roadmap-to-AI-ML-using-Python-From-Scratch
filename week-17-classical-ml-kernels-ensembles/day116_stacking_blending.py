"""
Day 116: Stacking and Blending -- a Real Meta-Model Demonstration

Different models make different mistakes. Stacking trains one more model, the
meta-model, to combine their predictions: it learns which model to trust, and
how much. The catch is subtle: the meta-model must learn from predictions the
base models made on rows they had not seen, or it learns to trust whichever
model memorised its training data best.

Everything is written from scratch in NumPy, with the base models imported
unchanged from earlier lessons:
  * logistic regression (softmax regression, new today, also used as the meta-model)
  * k-nearest neighbours (Day 115), an RBF-kernel SVM (Days 112-113),
    a random forest (Day 107) and gradient boosting (Day 109)
  * out-of-fold predictions, stacking, blending and simple averaging

  1. By hand: two models, one meta-model.
  2. The leakage trap: a meta-model trained on in-sample predictions.
  3. Is it better? Stacking, blending and averaging against the best single model.
  4. What the meta-model learned, and how different the models' mistakes are.
  5. Diversity: stacking similar models against different ones.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import earlier lessons
import time                                            # time: what stacking costs

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_breast_cancer, load_digits  # bundled datasets
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "week-16-classical-ml-trees"))  # Weeks 16 and 17
with contextlib.redirect_stdout(io.StringIO()):         # their imports print nothing
    import day113_kernel_trick as d113                  # Day 113: rbf, Day 112's smo and bias through it
    import day115_k_nearest_neighbors as d115           # Day 115: sqdist
    import day109_gradient_boosting as d109             # Day 109: fit_boost, predict_reg, sigmoid; Day 107's forest through it
d112, d107 = d113.d112, d109.d107                        # Days 112 and 107
standardise, stratified_split, d99 = d112.standardise, d112.stratified_split, d112.d99  # unchanged helpers
FOLDS = 5                                               # folds for out-of-fold predictions
SPLITS = {"Breast cancer": 20, "Titanic": 20, "Digits": 5}  # repeated splits per dataset
EPS = 1e-6                                              # probabilities are clipped to [EPS, 1 - EPS] before taking logs


# =============================================================================
# SOFTMAX REGRESSION -- NEW (base model and meta-model)
# =============================================================================
def softmax(Z):                                         # rows of scores to rows of probabilities
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def fit_softmax(F, y, k, lam=1e-3, steps=500, lr=0.5):  # NEW: multinomial logistic regression, full-batch Adam, L2 penalty
    W = np.zeros((F.shape[1], k)); b = np.zeros(k); Y = np.eye(k)[y]  # weights, biases, one-hot labels
    m = [np.zeros_like(W), np.zeros_like(b)]; v = [np.zeros_like(W), np.zeros_like(b)]  # Adam's averages
    for t in range(1, steps + 1):
        G = (softmax(F @ W + b) - Y) / len(y)              # gradient of the mean cross-entropy at the scores
        for i, (p, g) in enumerate(((W, F.T @ G + lam * W), (b, G.sum(0)))):  # weights (penalised), then biases
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)  # Day 37's Adam step
    return W, b


# =============================================================================
# BASE MODELS, EACH AS fit(X, y, k, seed) -> proba(X) -- NEW wrappers around earlier lessons
# =============================================================================
def m_logistic(X, y, k, seed):                          # softmax regression on standardised features
    mu, sd = X.mean(0), X.std(0) + 1e-12; W, b = fit_softmax((X - mu) / sd, y, k)
    return lambda Z: softmax(((Z - mu) / sd) @ W + b)


def m_knn(X, y, k, seed, nn=15):                        # Day 115's distances; the vote shares as probabilities
    mu, sd = X.mean(0), X.std(0) + 1e-12; A = (X - mu) / sd
    def proba(Z):
        D = d115.sqdist((Z - mu) / sd, A); idx = np.argpartition(D, nn - 1, axis=1)[:, :nn]  # the nn nearest
        return np.stack([np.mean(y[idx] == c, 1) for c in range(k)], 1)  # share of neighbours in each class
    return proba


def m_svm(X, y, k, seed, C=1.0):                        # Days 112-113: an RBF SVM per class; softmax of its scores
    mu, sd = X.mean(0), X.std(0) + 1e-12; A = (X - mu) / sd; g = 1.0 / A.shape[1]  # gamma = 1 / features
    K = d113.rbf(A, A, g); cls = [1] if k == 2 else range(k)  # one SVM for two classes, one per class otherwise
    fits = [d113.fit_kernel_svm(K, np.where(y == c, 1.0, -1.0), C)[:2] for c in cls]
    def proba(Z):
        Kz = d113.rbf(A, (Z - mu) / sd, g)
        S = np.stack([Kz.T @ (a * np.where(y == c, 1.0, -1.0)) + bb for c, (a, bb) in zip(cls, fits)], 1)  # decision values
        return softmax(np.hstack([-S / 2, S / 2])) if k == 2 else softmax(S)  # scores squashed into shares: a rough choice
    return proba


def m_forest(X, y, k, seed, trees=60):                  # Day 107's forest
    f, _ = d107.fit_forest(X, y, k, trees, max(1, int(np.sqrt(X.shape[1]))), seed)
    return lambda Z: d107.forest_proba(f, Z, k)


def m_boost(X, y, k, seed):                             # Day 109's boosting, 150 rounds (two classes only)
    m, _ = d109.fit_boost(X, y, "logistic", 150, 0.1, 3)
    def proba(Z):
        F = np.full(len(Z), m["base"])
        for t in m["trees"]:
            F = F + m["lr"] * d109.predict_reg(t, Z)
        p = d109.sigmoid(F); return np.stack([1 - p, p], 1)
    return proba


BASE = {"logistic": m_logistic, "kNN": m_knn, "RBF SVM": m_svm, "forest": m_forest, "boosting": m_boost}  # name -> fit function


# =============================================================================
# STACKING AND BLENDING -- NEW
# =============================================================================
def features(P, k):                                     # NEW: meta-features: log-odds (two classes) or log-probabilities
    P = np.clip(P, EPS, 1 - EPS)
    return np.log(P[:, 1] / P[:, 0])[:, None] if k == 2 else np.log(P)


def oof(fit, X, y, k, seed):                            # NEW: out-of-fold predictions: each row predicted by a model that never saw it
    folds = np.array_split(np.random.RandomState(seed).permutation(len(y)), FOLDS); P = np.zeros((len(y), k))
    for i in range(FOLDS):
        va = folds[i]; tr = np.concatenate([folds[j] for j in range(FOLDS) if j != i])
        P[va] = fit(X[tr], y[tr], k, seed)(X[va])
    return P


def stack(names, X, y, Xte, k, seed, leak=False):       # NEW: base models on all rows; meta-model on their out-of-fold (or in-sample) predictions
    fits = {n: BASE[n](X, y, k, seed) for n in names}    # the base models used at test time
    tr = [features(fits[n](X) if leak else oof(BASE[n], X, y, k, seed), k) for n in names]  # what the meta-model learns from
    te = [features(fits[n](Xte), k) for n in names]
    W, b = fit_softmax(np.hstack(tr), y, k)
    return softmax(np.hstack(te) @ W + b), W, {n: fits[n](Xte) for n in names}


def blend(names, X, y, Xte, k, seed, frac=0.3):         # NEW: base models on 70%, the meta-model on their predictions for the other 30%
    rng = np.random.RandomState(seed); o = rng.permutation(len(y)); h = int(len(y) * frac); ho, fit_rows = o[:h], o[h:]
    fits = {n: BASE[n](X[fit_rows], y[fit_rows], k, seed) for n in names}
    W, b = fit_softmax(np.hstack([features(fits[n](X[ho]), k) for n in names]), y[ho], k)
    return softmax(np.hstack([features(fits[n](Xte), k) for n in names]) @ W + b)


def acc(P, y):                                          # accuracy of the most probable class
    return float(np.mean(np.argmax(P, 1) == y))


def datasets():                                         # three problems
    bc, dg = load_breast_cancer(), load_digits()
    with contextlib.redirect_stdout(io.StringIO()):
        import day106_pruning as d106
    Xt, yt = d106.load_titanic()
    return {"Breast cancer": (bc.data, bc.target), "Titanic": (Xt, yt), "Digits": (dg.data / 16.0, dg.target)}


def names_for(k):                                       # Day 109's boosting handles two classes only
    return [n for n in BASE if k == 2 or n != "boosting"]


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- two models, one meta-model")
    print("=" * 70)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])               # eight rows
    a = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.4, 0.1, 0.6])  # model A: P(class 1) for each row
    b = np.array([0.6, 0.6, 0.9, 0.4, 0.4, 0.4, 0.4, 0.3])  # model B
    P = lambda p: np.stack([1 - p, p], 1)
    print(f"model A right on {np.mean((a > 0.5) == y):.3f} of rows, model B on {np.mean((b > 0.5) == y):.3f}, their average on {np.mean(((a + b) / 2 > 0.5) == y):.3f}")
    F = np.hstack([features(P(a), 2), features(P(b), 2)]); W, c = fit_softmax(F, y, 2, lam=0.01, steps=2000)
    w = W[:, 1] - W[:, 0]                                 # the weight each model's log-odds gets, for class 1 against class 0
    print(f"the meta-model weighs A's log-odds by {w[0]:.3f} and B's by {w[1]:.3f}; it gets {acc(softmax(F @ W + c), y):.3f} of these rows right")
    wa, wb = np.where((a > 0.5) != y)[0] + 1, np.where((b > 0.5) != y)[0] + 1  # rows each model gets wrong, counted from 1
    print(f"A is wrong on rows {wa.tolist()}, B on rows {wb.tolist()}: never the same row, so a mix can fix both (on eight rows, the meta-model is scored on the rows it fitted)")
    print("Actual output from this lesson's run.")


def demo_leak(sets):
    print("\n" + "=" * 70)
    print("2. THE LEAKAGE TRAP -- a meta-model trained on in-sample predictions")
    print("=" * 70)
    print("10 splits per dataset; the same base models, the meta-model trained on two kinds of predictions")
    names = names_for(2)
    for name in ("Breast cancer", "Titanic"):
        X, y = sets[name]; R = {"out-of-fold": [], "in-sample": []}; Ws = {"out-of-fold": [], "in-sample": []}; ins = {n: [] for n in names}
        for s in range(10):
            tr, te = stratified_split(y, 0.3, s)
            for mode in R:
                P, W, base = stack(names, X[tr], y[tr], X[te], 2, s, leak=(mode == "in-sample"))
                R[mode].append(acc(P, y[te])); Ws[mode].append(W[:, 1] - W[:, 0])
            for n in names:
                ins[n].append(acc(BASE[n](X[tr], y[tr], 2, s)(X[tr]), y[tr]))  # each model scored on its own training rows
        print(f"\n{name}: each base model's accuracy on its OWN training rows: " + ", ".join(f"{n} {np.mean(v):.4f}" for n, v in ins.items()))
        for mode in R:
            w = np.mean(Ws[mode], 0)
            print(f"  {mode:<12}: test accuracy {np.mean(R[mode]):.4f}; meta-weights " + ", ".join(f"{n} {x:+.2f}" for n, x in zip(names, w)))
        d = np.array(R["out-of-fold"]) - np.array(R["in-sample"])
        print(f"  out-of-fold minus in-sample: {d.mean():+.4f}; better on {int((d > 0).sum())}, worse on {int((d < 0).sum())} of 10")
    print("Actual output from this lesson's run.")


def demo_compare(sets):
    print("\n" + "=" * 70)
    print("3. IS IT BETTER? stacking, blending and averaging against the best single model")
    print("=" * 70)
    print(f"stacking: meta-model on {FOLDS}-fold out-of-fold predictions; blending: base models on 70% of the training rows, meta-model on the other 30%")
    print("'chosen by CV' is the single base model with the best out-of-fold accuracy: what you would pick without stacking")
    res, cost = {}, {}
    for name, (X, y) in sets.items():
        k = int(y.max()) + 1; names = names_for(k); rows = []; t_stack = []
        for s in range(SPLITS[name]):
            tr, te = stratified_split(y, 0.3, s)
            t0 = time.perf_counter(); P, _, base = stack(names, X[tr], y[tr], X[te], k, s); t_stack.append(time.perf_counter() - t0)
            singles = {n: acc(base[n], y[te]) for n in names}
            cv = {n: acc(oof(BASE[n], X[tr], y[tr], k, s), y[tr]) for n in names}  # out-of-fold accuracy, training rows only
            avg = acc(np.mean([base[n] for n in names], 0), y[te])
            rows.append([singles[n] for n in names] + [singles[max(cv, key=cv.get)], avg, acc(blend(names, X[tr], y[tr], X[te], k, s), y[te]), acc(P, y[te])])
        r = np.array(rows); res[name] = (names, r); cost[name] = float(np.median(t_stack))
        print(f"\n{name}: {len(y)} rows, {k} classes, {SPLITS[name]} splits")
        for i, n in enumerate(names + ["single model chosen by CV", "simple average", "blending", "stacking"]):
            print(f"  {n:<26}{r[:, i].mean():>8.4f}  (std {r[:, i].std():.4f})")
        for j, n in ((-1, "stacking"), (-3, "average")):
            d = r[:, j] - r[:, len(names)]; lo, hi = d99.bootstrap_ci(d)
            print(f"  {n} minus the CV choice: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; better on {int((d > 0).sum())}, worse on {int((d < 0).sum())} of {SPLITS[name]}")
        print(f"  stacking took {cost[name]:.1f} s per split (median): every base model is trained {FOLDS + 1} times")
    print("the splits share rows, so these intervals are too narrow, a rough guide only")
    print("Actual output from this lesson's run.")
    return res                                               # the plot uses it


def demo_learned(sets):
    print("\n" + "=" * 70)
    print("4. WHAT THE META-MODEL LEARNED, AND HOW DIFFERENT THE MISTAKES ARE")
    print("=" * 70)
    X, y = sets["Titanic"]; names = names_for(2); Ws, E = [], []
    for s in range(SPLITS["Titanic"]):
        tr, te = stratified_split(y, 0.3, s); P, W, base = stack(names, X[tr], y[tr], X[te], 2, s)
        Ws.append(W[:, 1] - W[:, 0]); E.append(np.stack([np.argmax(base[n], 1) != y[te] for n in names]))
    w = np.mean(Ws, 0); E = np.hstack(E).astype(float)  # model, test row (all splits together)
    print("Titanic, 20 splits: mean meta-weight on each model's log-odds: " + ", ".join(f"{n} {x:+.3f}" for n, x in zip(names, w)))
    C = np.corrcoef(E)
    print(f"\ncorrelation between models' mistakes (1 = the same rows wrong):\n  {'':<10}" + "".join(f"{n:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:<10}" + "".join(f"{C[i, j]:>10.3f}" for j in range(len(names))))
    allwrong = np.mean(E.min(0) == 1); onewrong = np.mean((E.sum(0) > 0) & (E.min(0) == 0))
    print(f"test rows every model gets wrong: {allwrong:.4f}; rows at least one gets right and at least one wrong: {onewrong:.4f}")
    print("no combination of these models can fix the first kind; the second kind is all stacking has to work with")
    print("Actual output from this lesson's run.")
    return names, C                                          # the plot uses them


def demo_diversity(sets):
    print("\n" + "=" * 70)
    print("5. DIVERSITY -- stacking similar models against different ones")
    print("=" * 70)
    X, y = sets["Breast cancer"]; saved = dict(BASE); rows = []
    BASE.update({"forest A": lambda X, y, k, s: m_forest(X, y, k, s + 100), "forest B": lambda X, y, k, s: m_forest(X, y, k, s + 200)})  # forests that differ only in seed
    for s in range(10):
        tr, te = stratified_split(y, 0.3, s); row = []
        for group in (["forest", "forest A", "forest B"], ["forest", "logistic", "kNN"]):
            P, _, base = stack(group, X[tr], y[tr], X[te], 2, s)
            row += [acc(P, y[te]), max(acc(base[n], y[te]) for n in group)]
        rows.append(row)
    BASE.clear(); BASE.update(saved)                      # back to the five base models
    r = np.array(rows)
    print("breast cancer, 10 splits")
    print(f"  three forests (different seeds): stack {r[:, 0].mean():.4f}, best member on each split {r[:, 1].mean():.4f}")
    print(f"  forest, logistic, kNN:           stack {r[:, 2].mean():.4f}, best member on each split {r[:, 3].mean():.4f}")
    print("('best member on each split' is chosen by looking at the test rows, so it flatters the members)")
    print("Actual output from this lesson's run.")


def plot_results(res, names, C):                        # two panels: results | mistake correlations
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))             # canvas
    a = ax[0]; methods = ["single model chosen by CV", "simple average", "blending", "stacking"]; x = np.arange(len(res)); w = 0.2
    for i, m in enumerate(methods):
        v = [r[:, len(n) + i].mean() for n, r in res.values()]
        a.bar(x + (i - 1.5) * w, v, w, label=m)
    a.set_xticks(x); a.set_xticklabels(list(res)); a.set_ylim(0.75, 1.0); a.set_ylabel("mean test accuracy"); a.grid(alpha=0.3, axis="y")
    a.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08)); a.set_title("Combining models against choosing one", fontsize=10)
    a = ax[1]; im = a.imshow(C, cmap="viridis", vmin=0, vmax=1); plt.colorbar(im, ax=a, fraction=0.046)
    a.set_xticks(range(len(names))); a.set_xticklabels(names, fontsize=8, rotation=30); a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if C[i, j] < 0.6 else "k")
    a.set_title("Titanic: correlation between models' mistakes", fontsize=10)
    plt.tight_layout(); plt.savefig("stacking_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved stacking_results.png")


def draw_diagram():                                     # out-of-fold predictions feed the meta-model
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 116: stacking", ha="center", fontsize=12, fontweight="bold")
    box(1.5, 2.2, 2.4, 1.6, "training rows,\nsplit into\n5 folds", "#FAEEDA")
    for i, (n, y) in enumerate((("logistic", 3.4), ("kNN", 2.75), ("RBF SVM", 2.1), ("forest", 1.45), ("boosting", 0.8))):
        box(4.9, y, 2.6, 0.5, f"{n}: predict each fold\nfrom the other four", "#DCEBF7", fs=7.5)
        ax.annotate("", xy=(3.56, y), xytext=(2.74, 2.2), arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))
        ax.annotate("", xy=(7.6, 2.2), xytext=(6.24, y), arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))
    box(8.9, 2.2, 2.4, 1.6, "out-of-fold\npredictions:\none column\nper model", "#E5F2E1")
    box(11.7, 2.2, 2.2, 1.6, "meta-model\n(logistic regression)\nlearns how much\nto trust each", "#EEEDFE")
    ax.annotate("", xy=(10.56, 2.2), xytext=(10.14, 2.2), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("stacking_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved stacking_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 117 is imbalanced data: when one class is rare, accuracy misleads, and SMOTE and class weights are compared on a real rare-class problem.")


def main():                                             # runs every section in order
    sets = datasets()                                      # three problems
    demo_hand()                                            # 1. by hand
    demo_leak(sets)                                        # 2. the leakage trap
    res = demo_compare(sets)                               # 3. is it better
    names, C = demo_learned(sets)                          # 4. what it learned
    demo_diversity(sets)                                   # 5. diversity
    plot_results(res, names, C)                            # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 6. Day 117


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
