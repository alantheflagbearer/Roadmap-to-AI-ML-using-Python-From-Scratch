"""
Day 109: Gradient Boosting From Scratch -- Real Sequential Error Correction

Day 107 grew trees independently and averaged them. Boosting grows them one
after another: each small tree is trained on what the model so far still gets
wrong, and only a fraction of its answer is added.

Everything is written from scratch in NumPy:
  * a regression tree, with a split search that scores every threshold at once
  * boosting for squared error: each tree fits the residuals y - F
  * boosting for log loss: each tree fits y - p, and its leaves take a Newton step
  * shrinkage (the learning rate) and early stopping

The data: the Diabetes set bundled with scikit-learn (442 patients, 10 features,
a disease-progression score), breast cancer and Titanic.

  1. By hand: three rounds on six numbers.
  2. Correctness checks, including scikit-learn with the same settings.
  3. Sequential error correction on real data.
  4. Learning rate against rounds.
  5. Tree depth: what stumps can and cannot do.
  6. Bias and variance: boosting against Day 107's forest.
  7. Wrong labels: when correcting errors goes wrong.
  8. Is it better? Against trees and forests, and against plain linear regression.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 107

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_diabetes             # a bundled dataset
from sklearn.ensemble import GradientBoostingRegressor  # used ONLY as an outside check
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 107 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day107_random_forests as d107                # Day 107: fit_forest, forest_proba, decomposition, data
d106, d105 = d107.d106, d107.d105                       # Days 106 and 105
stratified_split = d105.stratified_split                # unchanged
LR = 0.1                                                # the usual learning rate in this lesson
DEPTH = 3                                               # the usual tree depth
ROUNDS = [0, 1, 2, 5, 10, 25, 50, 100, 200, 500]        # section 3: rounds shown
RATES = [1.0, 0.3, 0.1, 0.03, 0.01]                     # section 4: learning rates
MAX_ROUNDS = 1500                                       # section 4: the most rounds tried
DEPTHS = [1, 2, 3, 4, 6, 8]                             # section 5: tree depths
SHOW = [1, 5, 25, 100, 400]                             # section 6: rounds shown
SPLITS = 10                                             # repeated splits, unless a section says otherwise


# =============================================================================
# THE REGRESSION TREE -- NEW
# =============================================================================
def reg_split(X, r, min_leaf=1):                        # NEW: (feature, threshold, gain) that most reduces squared error
    n, best = len(r), None                                # rows; best so far
    total = r.sum(); base = total ** 2 / n                # the score before splitting
    for f in range(X.shape[1]):                           # every feature
        o = np.argsort(X[:, f], kind="stable"); xs, rs = X[o, f], r[o]  # sorted values and their targets
        sl = np.cumsum(rs)[:-1]; nl = np.arange(1, n)     # sum and size left of each gap
        ok = (xs[1:] != xs[:-1]) & (nl >= min_leaf) & (n - nl >= min_leaf)  # only between different values
        if not ok.any():
            continue
        gain = np.where(ok, sl ** 2 / nl + (total - sl) ** 2 / (n - nl) - base, -np.inf)  # squared error removed
        i = int(np.argmax(gain))
        if best is None or gain[i] > best[2] + 1e-12:     # strictly better (ties keep the earlier feature)
            best = (f, (xs[i] + xs[i + 1]) / 2, float(gain[i]))  # threshold halfway between neighbours
    return best if best and best[2] > 1e-12 else None


def grow_reg(X, r, idx, max_depth, min_leaf=1, depth=0):  # NEW: a tree on the rows idx, remembering each leaf's rows
    node = {"idx": idx}                                   # a leaf gets its value later, from the loss
    if depth >= max_depth or len(idx) < 2:
        return node
    s = reg_split(X[idx], r[idx], min_leaf)
    if s is None:                                         # nothing helps
        return node
    m = X[idx, s[0]] <= s[1]                              # who answers yes
    node.update(feature=s[0], threshold=s[1], left=grow_reg(X, r, idx[m], max_depth, min_leaf, depth + 1),
                right=grow_reg(X, r, idx[~m], max_depth, min_leaf, depth + 1))
    return node


def set_leaves(node, value):                            # NEW: give every leaf value(its rows)
    if "feature" not in node:
        node["value"] = value(node["idx"]); return
    set_leaves(node["left"], value); set_leaves(node["right"], value)


def predict_reg(node, X):                               # NEW: follow many rows down at once
    out = np.zeros(len(X)); stack = [(node, np.arange(len(X)))]  # nodes to visit, with the rows that reached them
    while stack:
        n, idx = stack.pop()
        if "feature" not in n:                              # a leaf
            out[idx] = n["value"]
            continue
        m = X[idx, n["feature"]] <= n["threshold"]
        stack.append((n["left"], idx[m])); stack.append((n["right"], idx[~m]))
    return out


# =============================================================================
# BOOSTING -- NEW
# =============================================================================
def sigmoid(F):                                         # log-odds to probability
    return 1 / (1 + np.exp(-F))


def rmse(F, y):                                         # root mean squared error
    return float(np.sqrt(np.mean((y - F) ** 2)))


def logloss(F, y):                                      # average log loss, from log-odds (stable form)
    return float(np.mean(np.logaddexp(0, -F * (2 * y - 1))))


def fit_boost(X, y, loss, rounds, lr, max_depth, evals=(), min_leaf=1):  # NEW: returns the model and the stages on evals
    y = y.astype(float); idx = np.arange(len(y))          # targets, and all the rows
    base = y.mean() if loss == "squared" else np.log(y.mean() / (1 - y.mean()))  # the best constant
    F = np.full(len(y), base); trees = []                 # the prediction so far
    E = [np.zeros((rounds + 1, len(v))) + base for v in evals]  # the prediction on each eval set, per round
    for m in range(rounds):
        if loss == "squared":
            r = y - F                                         # the residuals are the negative gradient
            leaf = lambda i: r[i].mean()                      # the best step for squared error
        else:
            p = sigmoid(F); r = y - p; h = p * (1 - p)        # gradient and curvature of the log loss
            leaf = lambda i: r[i].sum() / max(h[i].sum(), 1e-12)  # one Newton step per leaf
        t = grow_reg(X, r, idx, max_depth, min_leaf); set_leaves(t, leaf); trees.append(t)  # a tree on what is left
        F = F + lr * predict_reg(t, X)                      # add only a fraction of its answer
        for e, v in zip(E, evals):
            e[m + 1] = e[m] + lr * predict_reg(t, v)
    return {"base": base, "lr": lr, "trees": trees, "train": F}, E


def split_rows(n, frac, seed):                          # a random split for regression
    o = np.random.RandomState(seed).permutation(n); k = int(round(n * frac))
    return o[k:], o[:k]


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- three rounds on six numbers")
    print("=" * 70)
    X = np.arange(1.0, 7.0)[:, None]; y = np.array([1.0, 1.0, 2.0, 2.0, 6.0, 8.0])  # one feature, six rows
    F = np.full(6, y.mean())
    print(f"y = {y.tolist()}; start from the mean, {y.mean():.3f}; learning rate 0.5, stumps (depth 1)")
    for m in range(1, 4):
        r = y - F; t = grow_reg(X, r, np.arange(6), 1); set_leaves(t, lambda i: r[i].mean())
        print(f"round {m}: residuals {r.round(3).tolist()}; split x <= {t['threshold']}; leaves {t['left']['value']:+.3f} / {t['right']['value']:+.3f}")
        F = F + 0.5 * predict_reg(t, X)
        print(f"         prediction {F.round(3).tolist()}; squared error {np.sum((y - F) ** 2):.3f}")
    print("Actual output from this lesson's run.")


def demo_check(Xc, yc, Xd, yd):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS CHECKS")
    print("=" * 70)
    rng = np.random.RandomState(0); agree = 0              # fast split against brute force
    for trial in range(30):
        X = rng.randint(0, 6, (25, 3)).astype(float); r = rng.normal(size=25); s = reg_split(X, r); best = 0.0
        for f in range(3):
            for t in np.unique(X[:, f])[:-1]:
                m = X[:, f] <= t
                best = max(best, np.sum((r - r.mean()) ** 2) - np.sum((r[m] - r[m].mean()) ** 2) - np.sum((r[~m] - r[~m].mean()) ** 2))
        agree += s is not None and abs(s[2] - best) < 1e-9
    print(f"30 random problems: the fast split search finds the brute-force reduction in squared error in {agree} of 30")
    F = rng.normal(size=8); y = rng.randint(0, 2, 8).astype(float); eps = 1e-6
    num = np.array([(logloss(F + eps * e, y) - logloss(F - eps * e, y)) / (2 * eps) * 8 for e in np.eye(8)])  # numerical gradient
    print(f"log loss: numerical gradient against -(y - p): largest gap {np.abs(num + (y - sigmoid(F))).max():.1e}")
    tr, te = stratified_split(yc, 0.3, 0); G = np.zeros(len(tr)); yy = yc[tr].astype(float)  # one real leaf
    p = sigmoid(G); leaf = (yy - p).sum() / (p * (1 - p)).sum()  # the Newton step at F = 0, on all the rows
    grid = np.linspace(-3, 3, 60001); lo = grid[np.argmin([np.sum(np.logaddexp(0, -(G + g) * (2 * yy - 1))) for g in grid[::10]]) * 10]
    print(f"a real leaf: one Newton step {leaf:.4f}; the exact best step by grid search {lo:.4f} (Newton is an approximation of it)")
    m, _ = fit_boost(Xc[tr], yc[tr], "logistic", 200, LR, DEPTH, evals=(Xc[tr],))
    L = np.array([logloss(f, yc[tr].astype(float)) for f in _[0]])
    print(f"training log loss fell in {int((np.diff(L) < 0).sum())} of 200 rounds, from {L[0]:.4f} to {L[-1]:.4f}")
    gap_rmse, gap_max = [], []                             # against scikit-learn, same settings
    for s in range(5):
        a, b = split_rows(len(yd), 0.3, s)
        _, E = fit_boost(Xd[a], yd[a], "squared", 100, LR, DEPTH, evals=(Xd[b],))
        sk = GradientBoostingRegressor(n_estimators=100, learning_rate=LR, max_depth=DEPTH, random_state=s).fit(Xd[a], yd[a]).predict(Xd[b])
        gap_rmse.append(rmse(E[0][-1], yd[b]) - rmse(sk, yd[b])); gap_max.append(np.abs(E[0][-1] - sk).max())
    print(f"diabetes, 5 splits, 100 rounds, same settings as scikit-learn: test RMSE gap {np.mean(gap_rmse):+.4f} on average; largest prediction gap {np.max(gap_max):.4f} (target std {yd.std():.1f}; tied splits may differ)")
    print("Actual output from this lesson's run.")


def demo_correct(X, y):
    print("\n" + "=" * 70)
    print("3. SEQUENTIAL ERROR CORRECTION -- diabetes")
    print("=" * 70)
    tr, te = split_rows(len(y), 0.3, 0); rounds = ROUNDS[-1]
    m, E = fit_boost(X[tr], y[tr], "squared", rounds, LR, DEPTH, evals=(X[tr], X[te]))
    print(f"{len(tr)} training, {len(te)} test patients; depth {DEPTH}; learning rate {LR}; the mean alone gives test RMSE {rmse(np.full(len(te), y[tr].mean()), y[te]):.2f}")
    print(f"'corr' is the correlation between that round's tree and the residuals it was trained on")
    print(f"\n{'round':>7}{'train RMSE':>12}{'test RMSE':>11}{'corr':>7}{'worst residual':>16}")
    for r in ROUNDS:
        F = E[0][r]; res = y[tr] - F
        c = np.corrcoef((E[0][r] - E[0][r - 1]), y[tr] - E[0][r - 1])[0, 1] if r else float("nan")
        print(f"{r:>7}{rmse(F, y[tr]):>12.2f}{rmse(E[1][r], y[te]):>11.2f}{c:>7.3f}{np.abs(res).max():>16.1f}")
    T = [rmse(f, y[te]) for f in E[1]]; b = int(np.argmin(T))
    print(f"lowest test RMSE {T[b]:.2f} at round {b} (read off the test set, so for this table only); at round {rounds}: {T[-1]:.2f}, while training reaches {rmse(E[0][-1], y[tr]):.2f}")
    print("Actual output from this lesson's run.")
    return [rmse(f, y[tr]) for f in E[0]], T                 # the plot uses them


def stop_round(E_val, y_val, score):                    # NEW: the round with the best validation score
    return int(np.argmin([score(f, y_val) for f in E_val]))


def demo_rates(X, y):
    print("\n" + "=" * 70)
    print("4. LEARNING RATE AGAINST ROUNDS -- diabetes")
    print("=" * 70)
    print(f"{SPLITS} splits: 60% training, 20% validation (chooses the round), 20% test (used once); depth {DEPTH}; at most {MAX_ROUNDS} rounds")
    print(f"\n{'rate':>7}{'mean rounds':>13}{'at the limit':>14}{'test RMSE':>11}{'std':>8}")
    out = []
    for lr in RATES:
        rounds, scores, edge = [], [], 0
        for s in range(SPLITS):
            a, te = split_rows(len(y), 0.2, s); tr, va = a[len(a) // 4:], a[:len(a) // 4]
            _, E = fit_boost(X[tr], y[tr], "squared", MAX_ROUNDS, lr, DEPTH, evals=(X[va], X[te]))
            b = stop_round(E[0], y[va], rmse); rounds.append(b); scores.append(rmse(E[1][b], y[te])); edge += b == MAX_ROUNDS
        out.append((lr, np.mean(rounds), np.mean(scores)))
        print(f"{lr:>7}{np.mean(rounds):>13.1f}{edge:>14}{np.mean(scores):>11.2f}{np.std(scores):>8.2f}")
    print("Actual output from this lesson's run.")
    return out                                               # the plot uses them


def xor_data(n, seed):                                  # Day 106's synthetic exclusive-or, for a known interaction
    return d106.xor_data(n, seed)


def demo_depth(X, y):
    print("\n" + "=" * 70)
    print("5. TREE DEPTH -- what stumps can and cannot do")
    print("=" * 70)
    print(f"{SPLITS} splits; learning rate {LR}; at most 500 rounds; the round chosen on validation rows")
    print(f"\n{'depth':>6}{'diabetes RMSE':>15}{'rounds':>8}{'xor accuracy':>14}{'rounds':>8}")
    for d in DEPTHS:
        a_s, a_r, b_s, b_r = [], [], [], []
        for s in range(SPLITS):
            a, te = split_rows(len(y), 0.2, s); tr, va = a[len(a) // 4:], a[:len(a) // 4]
            _, E = fit_boost(X[tr], y[tr], "squared", 500, LR, d, evals=(X[va], X[te]))
            r = stop_round(E[0], y[va], rmse); a_r.append(r); a_s.append(rmse(E[1][r], y[te]))
            Xa, ya = xor_data(400, s); Xv, yv = xor_data(400, 500 + s); Xb, yb = xor_data(2000, 1000 + s)  # synthetic, on purpose
            _, E = fit_boost(Xa, ya, "logistic", 500, LR, d, evals=(Xv, Xb))
            r = stop_round(E[0], yv.astype(float), logloss); b_r.append(r); b_s.append(np.mean((E[1][r] > 0) == yb))
        print(f"{d:>6}{np.mean(a_s):>15.2f}{np.mean(a_r):>8.1f}{np.mean(b_s):>14.4f}{np.mean(b_r):>8.1f}")
    print("the exclusive-or is synthetic on purpose; two perfect questions would reach about 0.95")
    print("Actual output from this lesson's run.")


def demo_bias(X, y, tr, te):
    print("\n" + "=" * 70)
    print("6. BIAS AND VARIANCE -- boosting against Day 107's forest")
    print("=" * 70)
    runs, sub = d107.RUNS, d107.SUB
    print(f"Titanic: Day 107's {runs} training sets of {sub} passengers and the same {len(te)} test passengers; learning rate {LR}, depth {DEPTH}")
    P = np.zeros((runs, max(SHOW) + 1, len(te))); Q = np.zeros((runs, len(te)))
    for r in range(runs):
        s = np.random.RandomState(r).choice(tr, sub, replace=False)  # Day 107's training set r
        _, E = fit_boost(X[s], y[s], "logistic", max(SHOW), LR, DEPTH, evals=(X[te],)); P[r] = sigmoid(E[0])
        Q[r] = d107.forest_proba(d107.fit_forest(X[s], y[s], 2, d107.BAG, 2, r)[0], X[te], 2)[:, 1]  # Day 107's forest
    print(f"\n{'model':<24}{'variance':>10}{'bias+noise':>12}{'Brier':>8}{'accuracy':>10}")
    v, e = d107.decomposition(Q, y[te])
    print(f"{'forest (64 trees)':<24}{v:>10.4f}{e:>12.4f}{v + e:>8.4f}{np.mean((Q > 0.5) == y[te]):>10.4f}")
    rows = []
    for m in SHOW:
        v, e = d107.decomposition(P[:, m], y[te]); rows.append((m, v, e))
        name = f"boosting, {m} round" + ("s" if m > 1 else "")
        print(f"{name:<24}{v:>10.4f}{e:>12.4f}{v + e:>8.4f}{np.mean((P[:, m] > 0.5) == y[te]):>10.4f}")
    print("Actual output from this lesson's run.")


def demo_noise(X, y):
    print("\n" + "=" * 70)
    print("7. WRONG LABELS -- when correcting errors goes wrong")
    print("=" * 70)
    print(f"breast cancer, 20% of training labels flipped, test labels correct, {SPLITS} splits; boosting at learning rate {LR}, depth {DEPTH}")
    print("'pull' is the mean size of the residual |y - p|: how hard the next tree is pushed by those rows")
    keep = [1, 10, 50, 100, 200, 400]; A = np.zeros((SPLITS, len(keep))); C = A.copy(); W = A.copy(); stop, frs, frr = [], [], []
    for s in range(SPLITS):
        tr, te = stratified_split(y, 0.3, s); rng = np.random.RandomState(100 + s)
        yn = y.copy(); flip = rng.choice(tr, int(round(0.2 * len(tr))), replace=False); yn[flip] = 1 - yn[flip]  # Day 106's noise
        wrong = np.isin(tr, flip)                           # which training rows were flipped
        _, E = fit_boost(X[tr], yn[tr], "logistic", max(keep), LR, DEPTH, evals=(X[tr], X[te]))
        for i, k in enumerate(keep):
            r = np.abs(yn[tr] - sigmoid(E[0][k])); A[s, i] = np.mean((E[1][k] > 0) == y[te]); C[s, i] = r[~wrong].mean(); W[s, i] = r[wrong].mean()
        v = int(len(tr) * 0.25); o = np.random.RandomState(s).permutation(tr); va, fit = o[:v], o[v:]  # noisy validation rows too
        _, Es = fit_boost(X[fit], yn[fit], "logistic", max(keep), LR, DEPTH, evals=(X[va], X[te]))
        b = stop_round(Es[0], yn[va].astype(float), logloss); stop.append(np.mean((Es[1][b] > 0) == y[te])); frr.append(b)
        frs.append(d107.accuracy(d107.forest_proba(d107.fit_forest(X[tr], yn[tr], 2, d107.TREES, 5, s)[0], X[te], 2), y[te]))
    print(f"\n{'round':>7}{'test acc':>10}{'pull: correct rows':>20}{'pull: flipped rows':>20}")
    for i, k in enumerate(keep):
        print(f"{k:>7}{A[:, i].mean():>10.4f}{C[:, i].mean():>20.4f}{W[:, i].mean():>20.4f}")
    print(f"early stopping on a noisy validation part: test {np.mean(stop):.4f}, after {np.mean(frr):.1f} rounds; Day 107's forest on the same noisy labels: {np.mean(frs):.4f}")
    print("Actual output from this lesson's run.")
    return keep, A.mean(0), C.mean(0), W.mean(0)             # the plot uses them


def demo_compare(sets, Xd, yd):
    print("\n" + "=" * 70)
    print("8. IS IT BETTER? against trees, forests and a straight line")
    print("=" * 70)
    print(f"20 splits; boosting uses learning rate {LR}, depth {DEPTH}, at most 500 rounds, stopped on 25% of the training part; forest 100 trees")
    for name in ("Titanic", "Breast cancer"):
        X, y = sets[name]; rows = []
        for s in range(20):
            tr, te = stratified_split(y, 0.3, s)
            al, ts, b, cv, se = d106.cv_alpha(X[tr], y[tr], seed=s)
            prune = np.mean(d105.predict(d106.at_alpha(al, ts, b[int(np.argmax(cv))]), X[te]) == y[te])
            frs = d107.accuracy(d107.forest_proba(d107.fit_forest(X[tr], y[tr], 2, 100, d107.sqrt_features(X.shape[1]), s)[0], X[te], 2), y[te])
            o = np.random.RandomState(s).permutation(tr); v = len(o) // 4; va, fit = o[:v], o[v:]
            _, E = fit_boost(X[fit], y[fit], "logistic", 500, LR, DEPTH, evals=(X[va], X[te]))
            r = stop_round(E[0], y[va].astype(float), logloss); rows.append((prune, frs, np.mean((E[1][r] > 0) == y[te])))
        r = np.array(rows); d = r[:, 2] - r[:, 1]; lo, hi = d105.d99.bootstrap_ci(d)
        print(f"\n{name}: pruned tree {r[:, 0].mean():.4f}, forest {r[:, 1].mean():.4f}, boosting {r[:, 2].mean():.4f}")
        print(f"  boosting minus forest {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; better on {int((d > 0).sum())}, worse on {int((d < 0).sum())} of 20")
    rows = []
    for s in range(20):
        a, te = split_rows(len(yd), 0.2, s); tr, va = a[len(a) // 4:], a[:len(a) // 4]
        _, E = fit_boost(Xd[tr], yd[tr], "squared", 500, LR, DEPTH, evals=(Xd[va], Xd[te]))
        r = stop_round(E[0], yd[va], rmse); A = np.column_stack([np.ones(len(a)), Xd[a]])  # linear regression uses all of a
        w = np.linalg.lstsq(A, yd[a], rcond=None)[0]
        rows.append((rmse(np.full(len(te), yd[a].mean()), yd[te]), rmse(E[1][r], yd[te]), rmse(np.column_stack([np.ones(len(te)), Xd[te]]) @ w, yd[te])))
    r = np.array(rows); d = r[:, 1] - r[:, 2]; lo, hi = d105.d99.bootstrap_ci(d)
    print(f"\nDiabetes (RMSE, lower is better): the mean {r[:, 0].mean():.2f}, boosting {r[:, 1].mean():.2f}, linear regression {r[:, 2].mean():.2f}")
    print(f"  boosting minus linear regression {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]; boosting better on {int((d < 0).sum())} of 20")
    print("the splits share rows, so these intervals are too narrow, a rough guide only")
    print("Actual output from this lesson's run.")


def plot_results(tr_c, te_c, rates, noise):             # three panels: correction | rates | wrong labels
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))           # canvas
    ax[0].plot(tr_c, label="training"); ax[0].plot(te_c, label="test"); ax[0].set_xlabel("round"); ax[0].set_ylabel("RMSE")
    ax[0].set_title("Diabetes: every round lowers training error, not test error", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    lab = [str(r[0]) for r in rates]; x = np.arange(len(rates))
    ax[1].bar(x, [r[2] for r in rates], color="#1f77b4"); ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_xlabel("learning rate")
    ax[1].set_ylabel("test RMSE"); ax[1].set_ylim(min(r[2] for r in rates) - 3, max(r[2] for r in rates) + 3); ax[1].grid(alpha=0.3, axis="y")
    for i, r in enumerate(rates):
        ax[1].text(i, r[2] + 0.2, f"rounds: {r[1]:.0f}", ha="center", fontsize=8)
    ax[1].set_title("Lower rates need many more rounds", fontsize=10)
    k, a, c, w = noise
    ax[2].plot(k, c, marker="o", label="pull of correct rows"); ax[2].plot(k, w, marker="o", label="pull of flipped rows"); ax[2].set_xscale("log")
    ax[2].set_xlabel("round (log scale)"); ax[2].set_ylabel("mean |y - p| on training rows"); ax[2].grid(alpha=0.3)
    b = ax[2].twinx(); b.plot(k, a, marker="s", linestyle="--", color="#2ca02c", label="test accuracy"); b.set_ylabel("test accuracy")
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center")
    ax[2].set_title("Breast cancer with 20% flipped labels", fontsize=10)
    plt.tight_layout(); plt.savefig("boosting_results.png", dpi=110); plt.close()  # save and free
    print("\nsaved boosting_results.png")


def draw_diagram():                                     # the loop
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 109: gradient boosting", ha="center", fontsize=12, fontweight="bold")
    box(1.5, 2.4, 2.4, 1.5, "start from the\nbest constant\nF = mean of y", "#FAEEDA")
    box(4.9, 2.4, 2.6, 1.5, "what is still wrong?\nr = y - F\n(the negative gradient)", "#DCEBF7")
    box(8.3, 2.4, 2.6, 1.5, "fit a small tree\nto r, not to y", "#E5F2E1")
    box(11.5, 2.4, 2.4, 1.5, "F = F + rate x tree\n(a small step)", "#EEEDFE")
    ax.annotate("", xy=(3.54, 2.4), xytext=(2.74, 2.4), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(6.94, 2.4), xytext=(6.24, 2.4), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(10.24, 2.4), xytext=(9.64, 2.4), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(4.9, 1.61), xytext=(11.5, 1.61), arrowprops=dict(arrowstyle="->", color="#333333", connectionstyle="bar,fraction=-0.09"))
    ax.text(8.2, 0.75, "repeat; stop when a validation set stops improving", ha="center", fontsize=8.5)
    plt.savefig("boosting_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved boosting_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("9. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 110 is a production boosting library, XGBoost: the same idea with the engineering around it, compared honestly with this from-scratch version.")


def main():                                             # runs every section in order
    sets = d107.data(); Xc, yc = sets["Breast cancer"]; Xt, yt = sets["Titanic"]  # Day 107's datasets
    db = load_diabetes(scaled=False); Xd, yd = db.data, db.target  # the raw diabetes measurements
    tr, te = stratified_split(yt, 0.3, 0)                  # Day 105's Titanic split
    demo_hand()                                            # 1. by hand
    demo_check(Xc, yc, Xd, yd)                             # 2. correctness
    tc, ec = demo_correct(Xd, yd)                          # 3. sequential correction
    rates = demo_rates(Xd, yd)                             # 4. learning rate
    demo_depth(Xd, yd)                                     # 5. depth
    demo_bias(Xt, yt, tr, te)                              # 6. bias and variance
    noise = demo_noise(Xc, yc)                             # 7. wrong labels
    demo_compare(sets, Xd, yd)                             # 8. the comparison
    plot_results(tc, ec, rates, noise)                     # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 9. Day 110


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
