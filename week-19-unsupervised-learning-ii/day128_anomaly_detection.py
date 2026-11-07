"""
Day 128: Anomaly Detection With Autoencoders -- Real Reconstruction-Error Thresholds

Train an autoencoder only on NORMAL examples, and it learns to rebuild them
well. Shown something unlike anything it was trained on, it rebuilds it badly.
The reconstruction error becomes an anomaly score, and a threshold on that score
turns it into an alarm. The method needs no examples of anomalies to train;
only to judge.

Everything is written from scratch in NumPy, on earlier parts:
  * Day 126's network and training; Day 117's ROC AUC and average precision
  * anomaly scores: autoencoder reconstruction error, PCA reconstruction error
    (Day 123), and the distance to the nearest normal training example
  * thresholds set from normal data alone, at a chosen false-alarm rate

The data, all real: MNIST with one digit held out as the 'anomaly', and the
Wisconsin breast-cancer measurements with malignant tumours as anomalies and
the model trained on benign ones only.

  1. By hand: a score, a threshold, and what it costs.
  2. One digit held out: the autoencoder against PCA and nearest neighbours.
  3. Every digit held out in turn: some anomalies are easy to rebuild.
  4. Thresholds from normal data: the false alarms promised and delivered.
  5. Contaminated training data: when anomalies leak into 'normal'.
  6. Tumours: a small autoencoder on 30 measurements.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier parts
import time                                            # time: how long training takes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.datasets import load_breast_cancer        # bundled dataset
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
for w in ("week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # Weeks 17 and 18's exercise parts
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: train_ae, reconstruct
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day123_exercise as d123                      # Day 123: pca_fit, transform, reconstruct
    import day119_exercise as d119                      # Day 119: sqdist
    import day117_exercise as d117                      # Day 117: roc_auc, avg_precision
ACTS = ["relu", "linear", "relu", "linear"]             # Day 126's network shape
SIZES = [784, 256, 32, 256, 784]


# =============================================================================
# ANOMALY SCORES AND THRESHOLDS -- NEW
# =============================================================================
def ae_score(net, X, acts, kind="bce"):                 # NEW: reconstruction error per example: the anomaly score
    return ((d126.reconstruct(net, X, acts, kind) - X) ** 2).sum(1)


def pca_score(X, pca, k):                               # NEW: the same score for PCA: what k directions of normal data cannot rebuild
    mu, V = pca
    return ((d123.reconstruct(d123.transform(X, mu, V, k), mu, V, k) - X) ** 2).sum(1)


def nn_score(X, ref, chunk=2000):                       # NEW: distance to the nearest normal training example
    return np.concatenate([np.sqrt(np.maximum(d119.sqdist(X[s:s + chunk], ref), 0).min(1)) for s in range(0, len(X), chunk)])


def threshold(normal_scores, false_alarm):              # NEW: the score that only `false_alarm` of normal examples exceed
    return float(np.quantile(normal_scores, 1 - false_alarm))


def alarm_stats(scores, y, t):                          # NEW: what a threshold does: y = 1 for anomalies
    flag = scores > t; tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    return {"recall": tp / max((y == 1).sum(), 1), "false alarms": fp / max((y == 0).sum(), 1), "precision": tp / max(flag.sum(), 1)}


def train(Xn, epochs=20, seed=0):                       # Day 126's training on normal images only
    return d126.train_ae(Xn, SIZES, ACTS, "bce", epochs=epochs, seed=seed)[0]


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- a score, a threshold, and what it costs")
    print("=" * 70)
    s = np.array([1.0, 2.0, 2.5, 3.0, 4.0, 9.0, 3.5, 8.0]); y = np.array([0, 0, 0, 0, 0, 1, 0, 1])  # six normal, two anomalies
    t = threshold(s[y == 0], 0.2); st = alarm_stats(s, y, t)
    print(f"scores {s.tolist()}, anomalies at 9.0 and 8.0; threshold for a 20% false-alarm rate on the six normal scores: {t:.2f}")
    print(f"flagged: {s[s > t].tolist()}; recall {st['recall']:.3f}, false alarms {st['false alarms']:.3f}, precision {st['precision']:.3f}")
    print(f"ROC AUC {d117.roc_auc(y, s):.4f}: by hand, both anomalies score above all six normal examples, so 12 of 12 pairs are ordered right")
    print("Actual output from this lesson's run.")


def demo_one(X, y, held=9):
    print("\n" + "=" * 70)
    print(f"2. ONE DIGIT HELD OUT -- train on 20,000 images of 0-8; the test set's {held}s are the anomalies")
    print("=" * 70)
    tr = np.where(y[:60000] != held)[0][:20000]; te = np.arange(60000, 70000); a = (y[te] == held).astype(int)
    t0 = time.perf_counter(); net = train(X[tr]); t1 = time.perf_counter() - t0
    mu, vals, V = d123.pca_fit(X[tr].astype(np.float64)); out = {}
    scores = {"autoencoder error": ae_score(net, X[te], ACTS), "PCA error, 32 components": pca_score(X[te].astype(np.float64), (mu, V), 32),
              "nearest normal image": nn_score(X[te].astype(np.float64), X[tr[:10000]].astype(np.float64))}
    print(f"anomalies: {a.sum()} of {len(a)} test images ({a.mean():.3f}); the autoencoder trained in {t1:.0f} s")
    print(f"  {'score':<28}{'ROC AUC':>9}{'average precision':>19}")
    for name, s in scores.items():
        out[name] = s; print(f"  {name:<28}{d117.roc_auc(a, s):>9.4f}{d117.avg_precision(a, s):>19.4f}")
    print("Actual output from this lesson's run.")
    return net, scores, a, X[tr], (mu, V)


def demo_every(X, y):
    print("\n" + "=" * 70)
    print("3. EVERY DIGIT HELD OUT IN TURN -- 10,000 normal training images and 10 epochs each, to keep the run short")
    print("=" * 70)
    te = np.arange(60000, 70000); rows = []
    print(f"  {'held out':>9}{'AE AUC':>9}{'PCA AUC':>9}{'mean error, normal':>20}{'mean error, held out':>22}")
    for d in range(10):
        tr = np.where(y[:60000] != d)[0][:10000]; a = (y[te] == d).astype(int)
        net = train(X[tr], epochs=10); s = ae_score(net, X[te], ACTS)
        mu, vals, V = d123.pca_fit(X[tr].astype(np.float64)); p = pca_score(X[te].astype(np.float64), (mu, V), 32)
        rows.append((d, d117.roc_auc(a, s), d117.roc_auc(a, p)))
        print(f"  {d:>9}{rows[-1][1]:>9.4f}{rows[-1][2]:>9.4f}{s[a == 0].mean():>20.2f}{s[a == 1].mean():>22.2f}")
    print("an AUC below 0.5 means the held-out digit is rebuilt BETTER than the normal ones on average")
    print("Actual output from this lesson's run.")
    return rows


def demo_thresholds(net, Xn, X, y, held=9):
    print("\n" + "=" * 70)
    print("4. THRESHOLDS FROM NORMAL DATA -- set on 5,000 normal images the network never trained on; judged on the test set")
    print("=" * 70)
    val = np.where(y[:60000] != held)[0][20000:25000]   # normal images after the 20,000 used for training
    sv = ae_score(net, X[val], ACTS); te = np.arange(60000, 70000); a = (y[te] == held).astype(int); s = ae_score(net, X[te], ACTS)
    print(f"  {'false alarms aimed at':>22}{'threshold':>11}{'false alarms got':>18}{'recall':>9}{'precision':>11}")
    for fa in (0.01, 0.05, 0.10, 0.20):
        t = threshold(sv, fa); st = alarm_stats(s, a, t)
        print(f"  {fa:>22.2f}{t:>11.2f}{st['false alarms']:>18.4f}{st['recall']:>9.4f}{st['precision']:>11.4f}")
    ttr = threshold(ae_score(net, Xn[:5000], ACTS), 0.05); st = alarm_stats(s, a, ttr)
    print(f"  the same 5% target set on the network's own training images instead: threshold {ttr:.2f}, false alarms got {st['false alarms']:.4f}")
    print("Actual output from this lesson's run.")


def demo_contaminated(X, y, held=9):
    print("\n" + "=" * 70)
    print("5. CONTAMINATED TRAINING DATA -- the 'normal' training set secretly contains some 9s")
    print("=" * 70)
    te = np.arange(60000, 70000); a = (y[te] == held).astype(int); normal = np.where(y[:60000] != held)[0]; nines = np.where(y[:60000] == held)[0]
    rows = []
    for share in (0.0, 0.01, 0.05, 0.10):
        k = int(round(10000 * share)); tr = np.r_[normal[:10000 - k], nines[:k]]
        net = train(X[tr], epochs=10); rows.append((share, d117.roc_auc(a, ae_score(net, X[te], ACTS))))
        print(f"  {share:>5.0%} of the 10,000 training images are 9s ({k:>4}): ROC AUC {rows[-1][1]:.4f}")
    print("Actual output from this lesson's run.")
    return rows


def demo_tumours():
    print("\n" + "=" * 70)
    print("6. TUMOURS -- trained on benign tumours only; malignant ones are the anomalies (30 measurements, standardised)")
    print("=" * 70)
    d = load_breast_cancer(); X, mal = d.data, (d.target == 0).astype(int)  # in this dataset target 0 means malignant
    rng = np.random.RandomState(0); ben = rng.permutation(np.where(mal == 0)[0]); tr = ben[:200]; te = np.r_[ben[200:], np.where(mal == 1)[0]]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12; Z = ((X - mu) / sd).astype(np.float32)  # scaled with benign training rows only
    print(f"training: {len(tr)} benign; test: {(mal[te] == 0).sum()} benign and {(mal[te] == 1).sum()} malignant")
    aucs = []
    for seed in range(5):                                  # five starting seeds: a small network on little data varies
        net = d126.train_ae(Z[tr], [30, 16, 4, 16, 30], ACTS, "mse", epochs=300, batch=32, lr=1e-3, seed=seed)[0]
        aucs.append(d117.roc_auc(mal[te], ae_score(net, Z[te], ACTS, "mse")))
    m, vals, V = d123.pca_fit(Z[tr].astype(np.float64)); p = pca_score(Z[te].astype(np.float64), (m, V), 4)
    nn = nn_score(Z[te].astype(np.float64), Z[tr].astype(np.float64))
    print(f"  autoencoder 30-16-4-16-30, 5 seeds: ROC AUC {np.mean(aucs):.4f} (from {min(aucs):.4f} to {max(aucs):.4f})")
    print(f"  PCA error, 4 components:         ROC AUC {d117.roc_auc(mal[te], p):.4f}")
    print(f"  nearest benign training tumour:  ROC AUC {d117.roc_auc(mal[te], nn):.4f}")
    print("no malignant tumour was used to train any of the three; labels are used only to judge")
    print("Actual output from this lesson's run.")
    return aucs


def plot_results(scores, a, rows, crow, Xte, net):      # four panels
    fig = plt.figure(figsize=(16, 9.6)); G = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(G[0, 0]); s = scores["autoencoder error"]; bins = np.linspace(0, np.quantile(s, 0.995), 60)
    ax.hist(s[a == 0], bins, alpha=0.6, density=True, label="normal (0-8)"); ax.hist(s[a == 1], bins, alpha=0.6, density=True, label="held out (9)")
    ax.set_xlabel("reconstruction error"); ax.set_title("Scores: normal against held-out digits", fontsize=10); ax.legend(fontsize=8)
    ax = fig.add_subplot(G[0, 1])
    for name, sc in scores.items():
        o = np.argsort(-sc); tpr = np.cumsum(a[o]) / a.sum(); fpr = np.cumsum(1 - a[o]) / (1 - a).sum(); ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8); ax.set_xlabel("false-alarm rate"); ax.set_ylabel("recall"); ax.set_title("ROC curves, 9 held out", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = fig.add_subplot(G[0, 2]); r = np.array(rows); w = 0.38
    ax.bar(r[:, 0] - w / 2, r[:, 1], w, label="autoencoder"); ax.bar(r[:, 0] + w / 2, r[:, 2], w, label="PCA, 32")
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.8); ax.set_xticks(range(10)); ax.set_xlabel("digit held out as the anomaly")
    ax.set_ylabel("ROC AUC"); ax.set_title("Every digit held out in turn", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    sub = G[1, :2].subgridspec(2, 8); o = np.argsort(-s)[:8]; R = d126.reconstruct(net, Xte[o], ACTS)  # the 8 highest-scoring test images
    for c in range(8):
        for rr, imgs in enumerate((Xte[o], R)):
            b = fig.add_subplot(sub[rr, c]); b.imshow(imgs[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1); b.set_xticks([]); b.set_yticks([])
            if c == 0:
                b.set_ylabel(("most anomalous", "rebuilt")[rr], fontsize=8)
    ax = fig.add_subplot(G[1, 2]); c = np.array(crow); ax.plot(c[:, 0] * 100, c[:, 1], marker="o")
    ax.set_xlabel("share of 9s hidden in the training set (%)"); ax.set_ylabel("ROC AUC"); ax.set_title("Contaminated training data", fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("anomaly_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved anomaly_results.png")


def draw_diagram():                                     # train on normal, score everything
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 128: anomaly detection by reconstruction", ha="center", fontsize=12, fontweight="bold")
    box(1.6, 2.4, 2.4, 1.6, "train the autoencoder\non NORMAL data only", "#E5F2E1")
    box(4.7, 2.4, 2.4, 1.6, "score = how badly\nit rebuilds an example", "#DCEBF7")
    box(7.8, 2.4, 2.4, 1.6, "threshold from held-out\nnormal scores:\na chosen false-alarm rate", "#FAEEDA")
    box(11.0, 2.4, 2.6, 1.6, "score above it: alarm", "#F7DCDC")
    for x0, x1 in ((2.84, 3.46), (5.94, 6.56), (9.04, 9.66)):
        ax.annotate("", xy=(x1, 2.4), xytext=(x0, 2.4), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 0.7, "it works when anomalies are harder to rebuild than normal data; nothing guarantees that", ha="center", fontsize=8.5)
    plt.savefig("anomaly_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved anomaly_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("7. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 129 is the variational autoencoder: codes that are random draws, and the reparameterization trick that lets gradients through them.")


def main():                                             # runs every section in order
    X, y = d126m.mnist()
    demo_hand()                                            # 1. by hand
    net, scores, a, Xn, pca = demo_one(X, y)               # 2. one digit held out
    rows = demo_every(X, y)                                # 3. every digit
    demo_thresholds(net, Xn, X, y)                         # 4. thresholds
    crow = demo_contaminated(X, y)                         # 5. contamination
    demo_tumours()                                         # 6. tumours
    plot_results(scores, a, rows, crow, X[60000:], net)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
