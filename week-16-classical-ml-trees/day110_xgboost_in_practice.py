"""
Day 110: A Production Boosting Library -- XGBoost Against the From-Scratch Version

Day 109 wrote gradient boosting in NumPy. Today uses XGBoost, the library people
actually deploy, and asks what it really adds. The first question is whether it
is the same algorithm at all, so its split rule is written from scratch too.

  * Newton boosting: the split gain and leaf value that XGBoost uses, from scratch
  * the library's speed, missing-value handling, regularisation and early stopping
  * SHAP values: an explanation of each single prediction
  * saving a model and loading it back

Data: Diabetes, breast cancer, Titanic and digits (all as in Days 105 to 109),
plus one synthetic table for timing only.

  1. Is it the same algorithm? From scratch against XGBoost, prediction by prediction.
  2. Speed: rows against seconds.
  3. Missing values: learned directions against filling in the median.
  4. Tuning: how much do the settings matter?
  5. The comparison: forests, Day 109, XGBoost and scikit-learn's boosting.
  6. Explaining one prediction: SHAP against Day 108's importance.
  7. Saving and loading.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lessons quiet
import io                                              # io: StringIO, where that output goes
import json                                            # json: reading XGBoost's trees back as text
import os                                              # os: paths
import sys                                             # sys: path, to import Day 109
import tempfile                                        # tempfile: a throwaway folder for the saved model
import time                                            # time: wall-clock timing

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import pandas as pd                                    # pandas: Titanic with its missing ages left in
import xgboost as xgb                                  # the library under test
from sklearn.datasets import load_diabetes, make_classification  # a bundled dataset; a synthetic table for timing only
from sklearn.ensemble import HistGradientBoostingClassifier  # scikit-learn's fast boosting, for comparison
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 109 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # the imports print nothing
    import day108_feature_importance as d108            # Day 108: titanic_extended, impurity and permutation importance
    import day109_gradient_boosting as d109             # Day 109: fit_boost, predict_reg, set_leaves, rmse, logloss
d107, d106, d105 = d109.d107, d109.d106, d109.d105      # Days 107, 106 and 105
stratified_split = d105.stratified_split                # unchanged
ROWS = [1000, 3000, 10000, 30000, 100000]               # section 2: table sizes
SCRATCH_MAX = 30000                                     # section 2: the most rows given to the from-scratch version
RATES = [0.0, 0.1, 0.3, 0.5]                            # section 3: shares of values removed
CONFIGS = 20                                            # section 4: random settings tried per split
TUNE_SPLITS = 10                                        # section 4: splits
SPLITS = {"Titanic": 20, "Breast cancer": 20, "Digits": 6}  # section 5: splits per dataset
BASE = dict(eta=0.1, max_depth=3, seed=0, nthread=1)    # the settings of Day 109, on one thread unless a section says otherwise


# =============================================================================
# NEWTON BOOSTING -- NEW: XGBoost's own rule, from scratch
# =============================================================================
def newton_split(X, g, h, lam):                         # NEW: the split with the largest Newton gain
    n, best = len(g), None
    G, H = g.sum(), h.sum(); parent = G ** 2 / (H + lam)  # the score of no split
    for f in range(X.shape[1]):                           # every feature
        o = np.argsort(X[:, f], kind="stable"); xs = X[o, f]  # sorted values
        GL = np.cumsum(g[o])[:-1]; HL = np.cumsum(h[o])[:-1]  # gradient and curvature left of each gap
        ok = xs[1:] != xs[:-1]                              # only between different values
        if not ok.any():
            continue
        gain = np.where(ok, 0.5 * (GL ** 2 / (HL + lam) + (G - GL) ** 2 / (H - HL + lam) - parent), -np.inf)
        i = int(np.argmax(gain))
        if best is None or gain[i] > best[2] + 1e-12:     # strictly better
            best = (f, (xs[i] + xs[i + 1]) / 2, float(gain[i]))
    return best if best and best[2] > 1e-12 else None


def grow_newton(X, g, h, idx, max_depth, lam, depth=0):  # NEW: a tree on the rows idx, leaf = -G / (H + lambda)
    node = {"value": float(-g[idx].sum() / (h[idx].sum() + lam))}  # every node knows its leaf value
    if depth >= max_depth or len(idx) < 2:
        return node
    s = newton_split(X[idx], g[idx], h[idx], lam)
    if s is None:
        return node
    m = X[idx, s[0]] <= s[1]
    node.update(feature=s[0], threshold=s[1], gain=s[2], left=grow_newton(X, g, h, idx[m], max_depth, lam, depth + 1),
                right=grow_newton(X, g, h, idx[~m], max_depth, lam, depth + 1))
    return node


def predict_lt(node, X):                                # NEW: XGBoost's convention: 32-bit numbers, and a row ON the threshold goes right
    out = np.zeros(len(X)); stack = [(node, np.arange(len(X)))]
    Z = X.astype(np.float32)                              # XGBoost stores features in 32 bits
    while stack:
        n, idx = stack.pop()
        if "feature" not in n:
            out[idx] = n["value"]
            continue
        m = Z[idx, n["feature"]] < np.float32(n["threshold"])  # strictly less, where Day 109 used <=
        stack.append((n["left"], idx[m])); stack.append((n["right"], idx[~m]))
    return out


def fit_newton(X, y, loss, rounds, lr, max_depth, lam, Xe, base):  # NEW: Newton boosting; (prediction on X, on Xe, trees)
    y = y.astype(float); idx = np.arange(len(y)); F = np.full(len(y), base); Fe = np.full(len(Xe), base); trees = []
    for _ in range(rounds):
        if loss == "squared":
            g, h = F - y, np.ones(len(y))                     # gradient and curvature of (F - y)^2 / 2
        else:
            p = d109.sigmoid(F); g, h = p - y, p * (1 - p)    # of the log loss
        t = grow_newton(X, g, h, idx, max_depth, lam)
        F = F + lr * predict_lt(t, X); Fe = Fe + lr * predict_lt(t, Xe); trees.append(t)  # the leaf values are already set
    return F, Fe, trees


def same_tree(ours, theirs):                            # NEW: does a from-scratch tree have XGBoost's questions?
    if "feature" not in ours:                             # a leaf must meet a leaf
        return "leaf" in theirs
    if "split" not in theirs or theirs["split"] != f"f{ours['feature']}":  # the same feature
        return False
    if abs(ours["threshold"] - theirs["split_condition"]) > 1e-4 * (1 + abs(ours["threshold"])):  # the same threshold
        return False
    kids = {c["nodeid"]: c for c in theirs["children"]}   # XGBoost's children, by id
    return same_tree(ours["left"], kids[theirs["yes"]]) and same_tree(ours["right"], kids[theirs["no"]])


def first_gap(ours, theirs):                            # NEW: the two gains at the first question where the trees differ
    if "feature" not in ours or "split" not in theirs:    # one side stopped: nothing to compare
        return None
    if theirs["split"] != f"f{ours['feature']}" or abs(ours["threshold"] - theirs["split_condition"]) > 1e-4 * (1 + abs(ours["threshold"])):
        return 2 * ours["gain"], theirs["gain"]             # XGBoost reports the gain without the factor of 1/2
    kids = {c["nodeid"]: c for c in theirs["children"]}
    return first_gap(ours["left"], kids[theirs["yes"]]) or first_gap(ours["right"], kids[theirs["no"]])


def fit_xgb(X, y, params, rounds, Xv=None, yv=None, stop=None):  # the library, with optional early stopping
    d = xgb.DMatrix(X, label=y)                           # its own data format
    ev = [(xgb.DMatrix(Xv, label=yv), "val")] if Xv is not None else []  # the validation part
    return xgb.train(params, d, rounds, evals=ev, early_stopping_rounds=stop, verbose_eval=False)


def margin(model, X):                                   # the raw prediction: log-odds for log loss
    it = getattr(model, "best_iteration", None)           # stop at the best round, when early stopping was used
    return model.predict(xgb.DMatrix(X), output_margin=True, iteration_range=(0, it + 1) if it is not None else (0, 0))


# =============================================================================
# DEMOS
# =============================================================================
def demo_same(Xd, yd, Xc, yc):
    print("=" * 70)
    print("1. IS IT THE SAME ALGORITHM? prediction by prediction")
    print("=" * 70)
    print(f"xgboost {xgb.__version__}; tree_method exact; no minimum child weight; 100 rounds, rate 0.1, depth 3, the same starting value; 5 splits")
    print("our predictions use XGBoost's convention: 32-bit values, and a row exactly on a threshold goes right")
    ties = []                                              # the two gains wherever the trees differ
    print(f"\n{'problem':<28}{'lambda':>7}{'same trees':>12}{'train: mean gap':>17}{'test: mean gap':>16}{'largest':>10}{'pred std':>10}")
    for name, X, y, loss, obj in (("Diabetes (squared error)", Xd, yd, "squared", "reg:squarederror"),
                                  ("Breast cancer (log loss)", Xc, yc, "logistic", "binary:logistic")):
        for lam in (0.0, 1.0, 10.0):
            gaps, top, sd, tg, same = [], [], [], [], 0
            for s in range(5):
                if loss == "squared":
                    tr, te = d109.split_rows(len(y), 0.3, s); base = float(y[tr].mean()); bs = base  # the same start
                else:
                    tr, te = stratified_split(y, 0.3, s); bs = float(y[tr].mean()); base = float(np.log(bs / (1 - bs)))
                F, ours, trees = fit_newton(X[tr], y[tr], loss, 100, 0.1, 3, lam, X[te], base)
                m = fit_xgb(X[tr], y[tr], dict(BASE, objective=obj, tree_method="exact", min_child_weight=0, base_score=bs, gamma=0, alpha=0, reg_lambda=lam), 100)
                theirs = margin(m, X[te]); gaps.append(np.abs(ours - theirs).mean()); top.append(np.abs(ours - theirs).max()); sd.append(theirs.std())
                tg.append(np.abs(F - margin(m, X[tr])).mean())
                dump = [json.loads(b) for b in m.get_dump(dump_format="json", with_stats=True)]  # XGBoost's trees as dictionaries
                same += sum(same_tree(a, b) for a, b in zip(trees, dump))
                gaps_here = [g for g in (first_gap(a, b) for a, b in zip(trees, dump)) if g]  # later trees inherit any difference
                ties += gaps_here[:1]                               # so only the first tree that differs in each run is a fair test
            print(f"{name:<28}{lam:>7.1f}{same:>8} of 500{np.mean(tg):>13.6f}{np.mean(gaps):>16.6f}{np.max(top):>10.4f}{np.mean(sd):>10.3f}")
    rel = [abs(a - b) / max(abs(a), abs(b), 1e-12) for a, b in ties]
    print(f"\nthe first tree to differ, in the {len(ties)} of 30 runs where any did: at the first differing question the two gains differ by {np.median(rel):.1e} (median), {np.max(rel):.1e} (largest), relative")
    rows = []                                              # Day 109's split rule against the Newton rule, on log loss
    for s in range(5):
        tr, te = stratified_split(yc, 0.3, s); bs = float(yc[tr].mean()); base = float(np.log(bs / (1 - bs)))
        _, E = d109.fit_boost(Xc[tr], yc[tr], "logistic", 100, 0.1, 3, evals=(Xc[te],))
        rows.append(np.abs(E[0][-1] - fit_newton(Xc[tr], yc[tr], "logistic", 100, 0.1, 3, 0.0, Xc[te], base)[1]).mean())
    print(f"Day 109's rule on log loss chooses splits on the gradient alone; its mean gap to the Newton rule is {np.mean(rows):.4f}")
    print("Actual output from this lesson's run.")


def timed(fn):                                          # seconds taken by one call
    t = time.perf_counter(); fn(); return time.perf_counter() - t


def demo_speed():
    print("\n" + "=" * 70)
    print("2. SPEED -- rows against seconds")
    print("=" * 70)
    print("synthetic table for timing only: 20 features, 100 rounds, rate 0.1, depth 3; median of 5 fits (1 fit when one takes over 20 s); timed within this one run")
    print(f"\n{'rows':>8}{'from scratch':>14}{'xgb exact':>11}{'xgb hist':>10}{'xgb hist, all threads':>23}{'sklearn hist':>14}")
    out = []
    for n in ROWS:
        X, y = make_classification(n, 20, n_informative=10, random_state=0)
        p = dict(BASE, objective="binary:logistic")
        jobs = {"exact": lambda: fit_xgb(X, y, dict(p, tree_method="exact"), 100),
                "hist": lambda: fit_xgb(X, y, dict(p, tree_method="hist"), 100),
                "all": lambda: fit_xgb(X, y, dict(p, tree_method="hist", nthread=os.cpu_count()), 100),
                "sk": lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, early_stopping=False).fit(X, y)}
        if n <= SCRATCH_MAX:
            jobs["scratch"] = lambda: d109.fit_boost(X, y, "logistic", 100, 0.1, 3)
        res = {}
        for k, f in jobs.items():
            first = timed(f)
            res[k] = first if first > 20 else float(np.median([first] + [timed(f) for _ in range(4)]))
        out.append((n, res))
        sc = f"{res['scratch']:.2f}" if "scratch" in res else "not run"
        print(f"{n:>8}{sc:>14}{res['exact']:>11.2f}{res['hist']:>10.2f}{res['all']:>23.2f}{res['sk']:>14.2f}")
    print(f"{os.cpu_count()} logical processors; scikit-learn's threads were left at their default, so that column uses them all")
    print("Actual output from this lesson's run.")
    return out                                               # the plot uses it


def demo_missing(Xc, yc):
    print("\n" + "=" * 70)
    print("3. MISSING VALUES -- learned directions against filling in the median")
    print("=" * 70)
    d = pd.read_csv(d105.TITANIC); d["is_female"] = (d["Sex"] == "female").astype(float)  # Age keeps its gaps here
    X = d[d106.FEATS].to_numpy(float); y = d["Survived"].to_numpy(); age = d106.FEATS.index("Age")
    p = dict(BASE, objective="binary:logistic", tree_method="exact"); nat, med = [], []
    for s in range(20):
        tr, te = stratified_split(y, 0.3, s); Z = X.copy(); Z[np.isnan(Z[:, age]), age] = np.nanmedian(X[tr, age])  # training median only
        nat.append(np.mean((margin(fit_xgb(X[tr], y[tr], p, 200), X[te]) > 0) == y[te]))
        med.append(np.mean((margin(fit_xgb(Z[tr], y[tr], p, 200), Z[te]) > 0) == y[te]))
    print(f"Titanic, {int(np.isnan(X[:, age]).sum())} of {len(y)} ages really missing, 20 splits, 200 rounds:")
    dd = np.array(nat) - np.array(med); lo, hi = d105.d99.bootstrap_ci(dd)
    print(f"  native handling {np.mean(nat):.4f}, median fill {np.mean(med):.4f}, gap {dd.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\nbreast cancer, values removed from every feature, 20 splits, 200 rounds:")
    print("  'at random': any value may go. 'large': only the largest in each feature go, so the gap itself is information.")
    print("  'random fill' puts a random observed value of that feature in each gap, so a filled value does not stand out")
    print(f"\n{'removed':>9}{'at random: native':>19}{'median':>9}{'large: native':>15}{'median':>9}{'random fill':>13}{'native - median':>17}{'+- SE':>8}")
    out = []
    for r in RATES:
        res = {"rand": ([], []), "big": ([], [])}; rnd = []
        for s in range(20):
            tr, te = stratified_split(yc, 0.3, s); rng = np.random.RandomState(s)
            for kind in res:
                Z = Xc.astype(float).copy()
                if r > 0:
                    mask = rng.uniform(size=Z.shape) < r if kind == "rand" else Z > np.quantile(Z, 1 - r, axis=0)  # which values go
                    Z[mask] = np.nan
                F = Z.copy(); m = np.nanmedian(Z[tr], axis=0); m = np.where(np.isnan(m), 0.0, m)  # training medians
                pos = np.where(np.isnan(F)); F[pos] = m[pos[1]]
                res[kind][0].append(np.mean((margin(fit_xgb(Z[tr], yc[tr], p, 200), Z[te]) > 0) == yc[te]))
                res[kind][1].append(np.mean((margin(fit_xgb(F[tr], yc[tr], p, 200), F[te]) > 0) == yc[te]))
                if kind == "big":                               # a fill that leaves no recognisable value behind
                    G = Z.copy()
                    for j in range(G.shape[1]):
                        seen = Z[tr, j][~np.isnan(Z[tr, j])]; gaps = np.isnan(G[:, j])  # observed training values; this column's gaps
                        G[gaps, j] = rng.choice(seen, gaps.sum()) if len(seen) else 0.0
                    rnd.append(np.mean((margin(fit_xgb(G[tr], yc[tr], p, 200), G[te]) > 0) == yc[te]))
        v = [np.mean(res["rand"][0]), np.mean(res["rand"][1]), np.mean(res["big"][0]), np.mean(res["big"][1]), np.mean(rnd)]; out.append((r, *v))
        d = np.array(res["big"][0]) - np.array(res["big"][1])
        print(f"{r:>9.0%}{v[0]:>19.4f}{v[1]:>9.4f}{v[2]:>15.4f}{v[3]:>9.4f}{v[4]:>13.4f}{d.mean():>+17.4f}{d.std() / np.sqrt(len(d)):>8.4f}")
    print("Actual output from this lesson's run.")
    return out                                               # the plot uses it


def sample_config(rng):                                 # one random setting
    pick = lambda v: v[rng.randint(len(v))]               # one value from a list
    return dict(eta=pick([0.02, 0.05, 0.1, 0.3]), max_depth=pick([2, 3, 4, 6, 8]), reg_lambda=pick([0.1, 1, 10]), gamma=pick([0, 0.1, 1]),
                subsample=pick([0.6, 0.8, 1.0]), colsample_bytree=pick([0.5, 0.8, 1.0]), min_child_weight=pick([0, 1, 5]))


def demo_tuning(sets):
    print("\n" + "=" * 70)
    print("4. TUNING -- how much do the settings matter?")
    print("=" * 70)
    print(f"{CONFIGS} random settings per split, each scored by 3-fold log loss on the training part with early stopping (at most 1,000 rounds, patience 30)")
    print(f"{TUNE_SPLITS} splits; the test part is used once")
    for name in ("Titanic", "Breast cancer"):
        X, y = sets[name]; r = []; rng = np.random.RandomState(0); spread = []
        for s in range(TUNE_SPLITS):
            tr, te = stratified_split(y, 0.3, s); folds = np.array_split(np.random.RandomState(s).permutation(tr), 3); scores = []
            cfgs = [sample_config(rng) for _ in range(CONFIGS)]
            for c in cfgs:
                p = dict(BASE, objective="binary:logistic", eval_metric="logloss", tree_method="hist", **c); ls, it = [], []
                for i in range(3):
                    va = folds[i]; fit = np.concatenate([folds[j] for j in range(3) if j != i])
                    m = fit_xgb(X[fit], y[fit], p, 1000, X[va], y[va], 30); ls.append(m.best_score); it.append(m.best_iteration)
                scores.append((np.mean(ls), int(np.mean(it)) + 1))
            b = int(np.argmin([v[0] for v in scores])); spread.append(np.ptp([v[0] for v in scores]))
            best = fit_xgb(X[tr], y[tr], dict(BASE, objective="binary:logistic", tree_method="hist", **cfgs[b]), scores[b][1])  # refit on all of it
            dflt = fit_xgb(X[tr], y[tr], dict(objective="binary:logistic", tree_method="hist", seed=0, nthread=1), 100)  # the library's defaults
            o = np.random.RandomState(s).permutation(tr); v = len(o) // 4
            es = fit_xgb(X[o[v:]], y[o[v:]], dict(BASE, objective="binary:logistic", eval_metric="logloss", tree_method="hist"), 500, X[o[:v]], y[o[:v]], 30)  # Day 109's settings
            r.append([np.mean((margin(m, X[te]) > 0) == y[te]) for m in (dflt, es, best)])
        r = np.array(r); d = r[:, 2] - r[:, 1]; lo, hi = d105.d99.bootstrap_ci(d)
        print(f"\n{name}: defaults (rate 0.3, depth 6, 100 rounds) {r[:, 0].mean():.4f}; rate 0.1, depth 3, early stopping {r[:, 1].mean():.4f}; tuned {r[:, 2].mean():.4f}")
        print(f"  tuned minus simple {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; better on {int((d > 0).sum())}, worse on {int((d < 0).sum())} of {TUNE_SPLITS}")
        print(f"  gap between the best and worst of the {CONFIGS} settings in CV log loss: {np.mean(spread):.4f} on average")
    print("Actual output from this lesson's run.")


def demo_compare(sets):
    print("\n" + "=" * 70)
    print("5. THE COMPARISON -- forests, Day 109, XGBoost and scikit-learn's boosting")
    print("=" * 70)
    print("forest: 100 trees (Day 107). Boosting: rate 0.1, depth 3, at most 500 rounds, stopped on 25% of the training part. Library defaults: as shipped")
    res = {}
    for name, (X, y) in sets.items():
        k = int(y.max()) + 1; rows = []
        for s in range(SPLITS[name]):
            tr, te = stratified_split(y, 0.3, s); o = np.random.RandomState(s).permutation(tr); v = len(o) // 4; va, fit = o[:v], o[v:]
            obj = dict(objective="binary:logistic", eval_metric="logloss") if k == 2 else dict(objective="multi:softprob", num_class=k, eval_metric="mlogloss")
            row = [d107.accuracy(d107.forest_proba(d107.fit_forest(X[tr], y[tr], k, 100, d107.sqrt_features(X.shape[1]), s)[0], X[te], k), y[te])]
            if k == 2:                                          # Day 109's version is binary only
                _, E = d109.fit_boost(X[fit], y[fit], "logistic", 500, 0.1, 3, evals=(X[va], X[te]))
                r = d109.stop_round(E[0], y[va].astype(float), d109.logloss); row.append(np.mean((E[1][r] > 0) == y[te]))
            else:
                row.append(np.nan)
            m = fit_xgb(X[fit], y[fit], dict(BASE, tree_method="hist", **obj), 500, X[va], y[va], 30)
            f = margin(m, X[te]); row.append(np.mean((f > 0) == y[te]) if k == 2 else np.mean(np.argmax(f, 1) == y[te]))
            m = fit_xgb(X[tr], y[tr], dict(obj, tree_method="hist", seed=0, nthread=1), 100)
            f = margin(m, X[te]); row.append(np.mean((f > 0) == y[te]) if k == 2 else np.mean(np.argmax(f, 1) == y[te]))
            row.append(HistGradientBoostingClassifier(random_state=s, early_stopping=False).fit(X[tr], y[tr]).score(X[te], y[te]))
            rows.append(row)
        r = np.array(rows, dtype=float); res[name] = r
        print(f"\n{name}: {len(y)} rows, {SPLITS[name]} splits")
        labels = ("forest, from scratch", "boosting, from scratch", "XGBoost, simple", "XGBoost, defaults", "scikit-learn, defaults")
        for i, n in enumerate(labels):
            print(f"  {n:<26}" + ("     n/a (binary only)" if np.isnan(r[:, i]).all() else f"{r[:, i].mean():.4f}  (std {r[:, i].std():.4f})"))
        d = r[:, 2] - r[:, 0]; lo, hi = d105.d99.bootstrap_ci(d)
        print(f"  XGBoost simple minus forest {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; better on {int((d > 0).sum())}, worse on {int((d < 0).sum())} of {SPLITS[name]}")
    print("the splits share rows, so these intervals are too narrow, a rough guide only")
    print("Actual output from this lesson's run.")
    return res                                               # the plot uses it


def demo_shap():
    print("\n" + "=" * 70)
    print("6. EXPLAINING ONE PREDICTION -- SHAP against Day 108's importance")
    print("=" * 70)
    X, y, names = d108.titanic_extended(); tr, te = stratified_split(y, 0.3, 0)
    m = fit_xgb(X[tr], y[tr], dict(BASE, objective="binary:logistic", tree_method="exact"), 150)
    C = m.predict(xgb.DMatrix(X[te]), pred_contribs=True)  # one value per feature per passenger, plus the base
    gap = np.abs(C.sum(1) - margin(m, X[te])).max()
    print(f"Titanic with Day 108's PassengerId and coin columns, 150 rounds, test accuracy {np.mean((margin(m, X[te]) > 0) == y[te]):.4f}")
    print(f"SHAP values plus the base add up to each prediction: largest gap {gap:.1e} log-odds")
    print(f"\n{'feature':<13}{'mean |SHAP|':>13}")
    a = np.abs(C[:, :-1]).mean(0)
    for j in np.argsort(-a):
        print(f"{names[j]:<13}{a[j]:>13.4f}")
    i = int(np.argmax(d109.sigmoid(margin(m, X[te]))))     # the most confident survivor
    print(f"\none passenger: the highest predicted survival in the test set, {d109.sigmoid(margin(m, X[te[i:i + 1]]))[0]:.3f}, base log-odds {C[i, -1]:+.3f}")
    print("  " + "; ".join(f"{names[j]} = {X[te[i], j]:.4g}: {C[i, j]:+.3f}" for j in np.argsort(-np.abs(C[i, :-1]))[:5]))
    j = np.argmin(d109.sigmoid(margin(m, X[te])))          # the most confident death
    print(f"and the lowest, {d109.sigmoid(margin(m, X[te[j:j + 1]]))[0]:.3f}:")
    print("  " + "; ".join(f"{names[k]} = {X[te[j], k]:.4g}: {C[j, k]:+.3f}" for k in np.argsort(-np.abs(C[j, :-1]))[:5]))
    print("Actual output from this lesson's run.")
    return names, a                                          # the plot uses them


def demo_save(Xc, yc):
    print("\n" + "=" * 70)
    print("7. SAVING AND LOADING")
    print("=" * 70)
    tr, te = stratified_split(yc, 0.3, 0)
    m = fit_xgb(Xc[tr], yc[tr], dict(BASE, objective="binary:logistic", tree_method="hist"), 200)
    with tempfile.TemporaryDirectory() as d:              # removed afterwards: nothing stays on disk
        p = os.path.join(d, "model.json"); m.save_model(p); kb = os.path.getsize(p) / 1024  # a portable text format
        n = xgb.Booster(); n.load_model(p)
        gap = np.abs(margin(n, Xc[te]) - margin(m, Xc[te])).max()
    print(f"a 200-round model saved as JSON is {kb:.0f} KB; loaded back, the largest prediction gap is {gap:.1e}")
    Z = xgb.DMatrix(Xc[te]); cached = np.median([timed(lambda: m.predict(Z)) for _ in range(20)])  # the same object every time
    t = np.median([timed(lambda: m.predict(xgb.DMatrix(Xc[te]))) for _ in range(20)])  # a new object every time
    print(f"predicting {len(te)} rows from a new data object takes {t * 1000:.2f} ms on one thread, about {t / len(te) * 1e6:.1f} microseconds per row")
    print(f"asking again with the same data object takes {cached * 1000:.2f} ms: the library keeps the last answer, so timing that measures a lookup, not a prediction")
    print("Actual output from this lesson's run.")


def plot_results(speed, missing, res, names, shap):     # four panels
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.2))             # canvas
    a = ax[0, 0]
    for k, lab, m in (("scratch", "from scratch (NumPy)", "o"), ("exact", "XGBoost exact", "s"), ("hist", "XGBoost hist", "^"), ("all", "XGBoost hist, all threads", "v"), ("sk", "scikit-learn hist", "D")):
        pts = [(n, r[k]) for n, r in speed if k in r]
        a.plot([p[0] for p in pts], [p[1] for p in pts], marker=m, label=lab)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("rows (synthetic, 20 features)"); a.set_ylabel("seconds for 100 rounds")
    a.set_title("Speed", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[0, 1]; q = [100 * r[0] for r in missing]
    a.plot(q, [r[1] for r in missing], marker="o", label="at random: native"); a.plot(q, [r[2] for r in missing], marker="o", linestyle="--", label="at random: median")
    a.plot(q, [r[3] for r in missing], marker="s", label="large: native"); a.plot(q, [r[4] for r in missing], marker="s", linestyle="--", label="large: median")
    a.plot(q, [r[5] for r in missing], marker="^", linestyle=":", label="large: random fill")
    a.set_xlabel("values removed (%)"); a.set_ylabel("test accuracy"); a.set_title("Breast cancer: missing values", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1, 0]; labels = ["forest", "boosting,\nfrom scratch", "XGBoost,\nsimple", "XGBoost,\ndefaults", "scikit-learn,\ndefaults"]; w = 0.26; x = np.arange(5)
    for i, (d, r) in enumerate(res.items()):
        v = [0.0 if np.isnan(r[:, k]).all() else r[:, k].mean() for k in x]; a.bar(x + (i - 1) * w, v, w, label=d)  # digits has no Day 109 bar
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=8); a.set_ylim(0.75, 1.03); a.set_ylabel("mean test accuracy"); a.grid(alpha=0.3, axis="y")
    a.legend(fontsize=8, ncol=3, loc="upper center"); a.set_title("Five models on three datasets", fontsize=10)
    a = ax[1, 1]; o = np.argsort(-shap); y = np.arange(len(o)); a.barh(y, shap[o], color="#9467bd"); a.set_yticks(y); a.set_yticklabels([names[j] for j in o], fontsize=8)
    a.invert_yaxis(); a.set_xlabel("mean |SHAP| (log-odds)"); a.set_title("Titanic: SHAP, with two useless columns", fontsize=10); a.grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig("xgboost_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved xgboost_results.png")


def draw_diagram():                                     # the same loop, and what is built around it
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 110: the same loop, with the engineering around it", ha="center", fontsize=12, fontweight="bold")
    box(6.5, 2.2, 3.8, 1.5, "Newton boosting, from scratch:\ngain = G^2 / (H + lambda) terms\nleaf = -G / (H + lambda)", "#DCEBF7")
    box(1.9, 3.2, 3.2, 0.95, "speed: histograms,\nmany threads", "#FAEEDA")
    box(1.9, 1.2, 3.2, 0.95, "missing values:\na learned direction", "#FAEEDA")
    box(11.1, 3.2, 3.2, 0.95, "regularisation: lambda, gamma,\nrow and column sampling", "#E5F2E1")
    box(11.1, 1.2, 3.2, 0.95, "early stopping, SHAP,\nsaving and loading", "#E5F2E1")
    ax.annotate("", xy=(4.56, 2.5), xytext=(3.54, 3.1), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(4.56, 1.9), xytext=(3.54, 1.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(8.44, 2.5), xytext=(9.46, 3.1), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(8.44, 1.9), xytext=(9.46, 1.3), arrowprops=dict(arrowstyle="->", color="#333333"))
    plt.savefig("xgboost_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved xgboost_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 111 is the week's build: trees against neural networks, head to head on tabular data, assembled from the exercise parts of Days 105 to 110.")


def main():                                             # runs every section in order
    sets = d107.data(); Xc, yc = sets["Breast cancer"]     # Day 107's datasets
    db = load_diabetes(scaled=False); Xd, yd = db.data, db.target  # Day 109's diabetes
    demo_same(Xd, yd, Xc, yc)                              # 1. the same algorithm?
    speed = demo_speed()                                   # 2. speed
    missing = demo_missing(Xc, yc)                         # 3. missing values
    demo_tuning(sets)                                      # 4. tuning
    res = demo_compare(sets)                               # 5. the comparison
    names, shap = demo_shap()                              # 6. SHAP
    demo_save(Xc, yc)                                      # 7. saving
    plot_results(speed, missing, res, names, shap)         # the figure
    draw_diagram()                                         # the diagram
    note_on_what_comes_next()                              # 8. Day 111


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
