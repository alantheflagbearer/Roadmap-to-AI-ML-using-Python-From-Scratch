"""
Day 117 Exercise -- Metrics for Rare Classes, and SMOTE

The sixth and last of Week 17's parts for Day 118's build.

    roc_auc(y, p)         the chance a random rare row outscores a random common one
    avg_precision(y, p)   precision averaged over the rare rows, ranked by score
    scores(y, p, threshold)   precision, recall, F1, balanced accuracy, and both of the above
    smote(X, k, n_new, rng)   new rare points between a rare point and one of its k nearest rare neighbours

Checked by hand on six rows: a perfect ranking scores 1, a reversed one 0, and
ties count half; and every SMOTE point lies between two rare points.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def sqdist(A, B):                                       # every squared distance (Day 113's)
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


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


y = np.array([1, 1, 0, 0, 0, 0])
print(f"perfect ranking: AUC {roc_auc(y, np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.0])):.3f}; reversed: {roc_auc(y, np.array([0.0, 0.1, 0.3, 0.2, 0.8, 0.9])):.3f}; all tied: {roc_auc(y, np.full(6, 0.5)):.3f}")
p = np.array([0.9, 0.4, 0.6, 0.2, 0.1, 0.0])           # rare rows ranked 1st and 3rd
print(f"rare rows ranked 1st and 3rd: AUC by hand 7 of 8 pairs = 0.875 -> {roc_auc(y, p):.3f}; average precision by hand (1/1 + 2/3) / 2 = 0.833 -> {avg_precision(y, p):.3f}")
s = scores(y, p)
print(f"at threshold 0.5: precision {s['precision']:.3f} (1 of 2 flagged), recall {s['recall']:.3f} (1 of 2 found), F1 {s['F1']:.3f}, balanced accuracy {s['balanced acc']:.3f}")
R = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]); N = smote(R, 2, 5, np.random.RandomState(0))
inside = all(q.min() >= -1e-12 and q.sum() <= 2 + 1e-12 for q in N)  # every point inside the triangle the three rare points span
print(f"SMOTE, 5 new points from a triangle of rare points: all on its edges or inside: {inside}; first: ({N[0, 0]:.3f}, {N[0, 1]:.3f})")
