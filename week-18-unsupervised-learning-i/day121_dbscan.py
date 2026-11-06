"""
Day 121: DBSCAN From Scratch -- Density-Based Clustering on Non-Convex Data

k-means (Day 119) draws straight boundaries between round groups, and needs k.
DBSCAN needs neither. A point with at least min_pts neighbours within a radius
eps is a CORE point; clusters grow outward from core points through their
neighbourhoods, and points in sparse regions are left as NOISE. Clusters can
take any shape, and their number is found, not chosen.

Everything is written from scratch in NumPy:
  * neighbourhoods within eps, computed in chunks
  * DBSCAN itself: core points, expansion through neighbourhoods, border points, noise
  * the k-distance curve, the usual guide for choosing eps

The data: shapes made on purpose (so the true groups are known exactly), real
California census districts (their latitude and longitude, to find where people
cluster), and handwritten digits.

  1. By hand: eleven points.
  2. Correctness: against scikit-learn's DBSCAN.
  3. Shapes k-means cannot represent.
  4. Choosing eps: the k-distance curve, and how sensitive the result is.
  5. Real data: where Californians live.
  6. One eps for every density: DBSCAN's weakness.
  7. High dimensions: digits.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported lesson quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: paths
import sys                                             # sys: path, to import Day 119

import numpy as np                                     # numpy: every array/matrix operation below is built on this
from sklearn.cluster import DBSCAN                     # used ONLY as an outside check
from sklearn.datasets import fetch_california_housing, load_digits  # real data (California is downloaded once and cached)
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # Day 119 sits next to this file
with contextlib.redirect_stdout(io.StringIO()):         # its import prints nothing
    import day119_k_means as d119                       # Day 119: sqdist, kmeans, ari
sqdist, ari = d119.sqdist, d119.ari                     # unchanged
NOISE = -1                                              # the label for noise points, as in scikit-learn
CITIES = {"Los Angeles": (34.05, -118.24), "San Francisco": (37.77, -122.42), "Oakland": (37.80, -122.27),  # city centres (lat, lon),
          "San Jose": (37.34, -121.89), "San Diego": (32.72, -117.16), "Sacramento": (38.58, -121.49),       # only to name clusters
          "Fresno": (36.74, -119.79)}


# =============================================================================
# DBSCAN -- NEW
# =============================================================================
def neighbours(X, eps, chunk=1000):                     # NEW: for every point, the indices within eps (itself included)
    out = []
    for s in range(0, len(X), chunk):                     # a chunk of rows at a time keeps the distance table small
        D = sqdist(X[s:s + chunk], X)
        out += [np.where(row <= eps * eps)[0] for row in D]
    return out


def dbscan(X, eps, min_pts):                            # NEW: labels 0, 1, ... for clusters, NOISE for noise
    N = neighbours(X, eps); core = np.array([len(nb) >= min_pts for nb in N])  # a core point has min_pts neighbours, itself included
    lab = np.full(len(X), NOISE); c = 0
    for i in range(len(X)):
        if lab[i] != NOISE or not core[i]:
            continue                                        # already in a cluster, or cannot start one
        lab[i] = c; stack = [i]                             # a new cluster grows from this core point
        while stack:
            j = stack.pop()
            if core[j]:                                     # only core points pass the cluster on
                for v in N[j]:
                    if lab[v] == NOISE:
                        lab[v] = c; stack.append(v)         # border points join, but do not spread further
        c += 1
    return lab, core


def k_distance(X, k, chunk=1000):                       # NEW: each point's distance to its k-th nearest point (itself counted first)
    out = []
    for s in range(0, len(X), chunk):
        D = np.maximum(sqdist(X[s:s + chunk], X), 0)
        out.append(np.sqrt(np.partition(D, k - 1, axis=1)[:, k - 1]))  # the k-th smallest, without sorting the whole row
    return np.concatenate(out)


# =============================================================================
# DATA
# =============================================================================
def shapes():                                           # Day 119's two shapes, generated with the same code and seed
    rng = np.random.RandomState(0); t = rng.uniform(0, np.pi, 300)
    moons = np.r_[np.c_[np.cos(t[:150]), np.sin(t[:150])], np.c_[1 - np.cos(t[150:]), 0.5 - np.sin(t[150:])]] + rng.normal(0, 0.08, (300, 2))
    ym = np.r_[np.zeros(150, int), np.ones(150, int)]
    big = rng.normal(0, 2.0, (450, 2)); small = rng.normal([5, 0], 0.4, (50, 2)); Xs = np.r_[big, small]; ys = np.r_[np.zeros(450, int), np.ones(50, int)]
    rings_t = rng.uniform(0, 2 * np.pi, 400); r = np.r_[np.full(200, 1.0), np.full(200, 3.0)] + rng.normal(0, 0.1, 400)  # NEW: two rings, one inside the other
    rings = np.c_[r * np.cos(rings_t), r * np.sin(rings_t)]; yr = np.r_[np.zeros(200, int), np.ones(200, int)]
    return {"two moons": (moons, ym), "rings": (rings, yr), "wide + tight": (Xs, ys)}


def california():                                       # every district's position, in km from the state's mean position
    d = fetch_california_housing(); lat, lon = d.data[:, 6], d.data[:, 7]
    km = np.c_[(lon - lon.mean()) * 111.32 * np.cos(np.radians(lat.mean())), (lat - lat.mean()) * 110.57]  # degrees to km (flat-map approximation)
    return km, lat, lon, d.target                         # the target: median house value, in $100,000s


def cities_in(lab, c, lat, lon):                        # the listed cities whose nearest district falls in cluster c
    out = []
    for name, (a, b) in CITIES.items():
        i = int(np.argmin(np.hypot((lat - a) * 110.57, (lon - b) * 111.32 * np.cos(np.radians(a)))))  # the city's nearest district
        out += [name] if lab[i] == c else []
    return ", ".join(out) if out else "(none listed)"


def summary(lab):                                       # number of clusters, noise share, sizes largest first
    s = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.array([], int)
    return len(s), float(np.mean(lab == NOISE)), sorted(s.tolist(), reverse=True)


def knee_eps(X, k):                                     # NEW: eps at the knee of the sorted k-distance curve (Day 119's elbow rule)
    kd = np.sort(k_distance(X, k))[::-1]                  # largest first: a few far points, then a long flat tail
    i = d119.elbow(list(range(len(kd))), kd)              # the position furthest below the chord from first to last
    return float(kd[i]), kd


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- eleven points on a line, eps = 0.6, min_pts = 3")
    print("=" * 70)
    x = np.array([0, 0.5, 1, 1.5, 5, 5.4, 5.8, 9, 12, 12.3, 20.0]); X = x[:, None]
    lab, core = dbscan(X, 0.6, 3); N = neighbours(X, 0.6)
    for i, v in enumerate(x):
        kind = "core" if core[i] else ("border" if lab[i] != NOISE else "noise")
        print(f"  point {v:>5}: {len(N[i])} within 0.6 (itself included) -> {kind:<6} label {lab[i]}")
    print("by hand: 0.5, 1 and 5.4 have 3 points within 0.6, so they are core; 0, 1.5 and 5.8 are within 0.6 of a core point, so they")
    print("join its cluster as border points; 9, 12, 12.3 and 20 are noise (12 and 12.3 are close, but two points are fewer than min_pts)")
    print("Actual output from this lesson's run.")


def demo_check(sets, cal):
    print("\n" + "=" * 70)
    print("2. CORRECTNESS -- against scikit-learn's DBSCAN")
    print("=" * 70)
    for name, X, eps, mp in (("two moons", sets["two moons"][0], 0.2, 5), ("rings", sets["rings"][0], 0.8, 5), ("California districts", cal, 5.0, 50)):
        lab, core = dbscan(X, eps, mp); sk = DBSCAN(eps=eps, min_samples=mp).fit(X)
        sk_core = np.zeros(len(X), bool); sk_core[sk.core_sample_indices_] = True
        print(f"  {name:<21} (eps {eps}, min_pts {mp}): core points identical {bool(np.array_equal(core, sk_core))}; "
              f"labels identical {bool(np.array_equal(lab, sk.labels_))}; clusters {lab.max() + 1}")
    print("Actual output from this lesson's run.")


def demo_shapes(sets):
    print("\n" + "=" * 70)
    print("3. SHAPES k-MEANS CANNOT REPRESENT -- DBSCAN against Day 119's k-means (k = 2)")
    print("=" * 70)
    out = {}
    for name, eps in (("two moons", 0.2), ("rings", 0.8)):
        X, y = sets[name]; lab, _ = dbscan(X, eps, 5); km = d119.kmeans(X, 2, 0)[0]; out[name] = lab
        k, nz, s = summary(lab)
        print(f"  {name:<10}: DBSCAN (eps {eps}, min_pts 5) ARI {ari(lab, y):.4f}, {k} clusters, noise {nz:.3f}; k-means ARI {ari(km, y):.4f}")
    print("DBSCAN follows the shape through chains of dense neighbourhoods; k-means draws one straight boundary")
    print("Actual output from this lesson's run.")
    return out


def demo_eps(sets):
    print("\n" + "=" * 70)
    print("4. CHOOSING eps -- the k-distance curve, and how sensitive DBSCAN is to it (two moons, min_pts = 5)")
    print("=" * 70)
    X, y = sets["two moons"]; e, kd = knee_eps(X, 5)
    print(f"each point's distance to its 5th nearest point (itself first), sorted; the knee of that curve is at eps = {e:.3f}")
    print(f"  {'eps':>6}{'clusters':>10}{'noise':>8}{'ARI':>9}")
    for eps in (0.1, 0.15, 0.2, 0.3, 0.5, round(e, 3)):
        lab, _ = dbscan(X, eps, 5); k, nz, _ = summary(lab)
        print(f"  {eps:>6}{k:>10}{nz:>8.3f}{ari(lab, y):>9.4f}")
    print("too small: the moons break into fragments and noise; too large: they merge into one")
    print("Actual output from this lesson's run.")
    return kd, e


def demo_california(cal, lat, lon, value):
    print("\n" + "=" * 70)
    print("5. REAL DATA -- where Californians live: 20,640 census districts by position (eps = 5 km, min_pts = 50)")
    print("=" * 70)
    lab, core = dbscan(cal, 5.0, 50); k, nz, s = summary(lab)
    print(f"{k} clusters; {nz:.3f} of districts are noise (rural, or between towns); the knee rule would suggest eps = {knee_eps(cal, 50)[0]:.2f} km")
    print(f"  {'size':>6}  {'centre (lat, lon)':<20}{'median house value':>19}  listed cities inside")
    for c in np.argsort(-np.bincount(lab[lab >= 0]))[:7]:
        la, lo = lat[lab == c].mean(), lon[lab == c].mean()
        print(f"  {int((lab == c).sum()):>6}  ({la:.2f}, {lo:.2f})    {np.median(value[lab == c]) * 1e5:>19,.0f}  {cities_in(lab, c, lat, lon)}")
    print(f"  noise districts: median house value {np.median(value[lab == NOISE]) * 1e5:,.0f}; districts in any cluster: {np.median(value[lab != NOISE]) * 1e5:,.0f}")
    print("nobody told DBSCAN where the cities are: the clusters are where districts are densely packed")
    print("Actual output from this lesson's run.")
    return lab


def demo_density(sets, cal, lat, lon):
    print("\n" + "=" * 70)
    print("6. ONE eps FOR EVERY DENSITY -- DBSCAN's weakness")
    print("=" * 70)
    print("California, min_pts = 50: the same data at different eps")
    print(f"  {'eps (km)':>9}{'clusters':>10}{'noise':>8}{'largest':>9}  listed cities in the largest")
    scan = []
    for eps in (2.0, 3.0, 5.0, 8.0, 12.0):
        lab, _ = dbscan(cal, eps, 50); k, nz, s = summary(lab); c = int(np.argmax(np.bincount(lab[lab >= 0])))
        scan.append((eps, k, nz))
        print(f"  {eps:>9}{k:>10}{nz:>8.3f}{s[0]:>9}  {cities_in(lab, c, lat, lon)}")
    X, y = sets["wide + tight"]; best = (-1, None)
    for eps in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2):
        for mp in (5, 10, 20):
            lab, _ = dbscan(X, eps, mp); a = ari(lab, y)
            if a > best[0]:
                best = (a, (eps, mp, summary(lab)))
    a, (eps, mp, (k, nz, s)) = best
    print(f"wide + tight (450 spread-out points beside 50 tight ones): the best of 21 settings is eps {eps}, min_pts {mp}: ARI {a:.4f}, "
          f"{k} clusters, noise {nz:.3f}")
    print("an eps small enough to separate the tight group leaves most of the wide one as noise; one large enough for the wide one swallows the tight one")
    print("Actual output from this lesson's run.")
    return scan


def demo_digits():
    print("\n" + "=" * 70)
    print("7. HIGH DIMENSIONS -- digits (64 pixels), min_pts = 5")
    print("=" * 70)
    d = load_digits(); X, y = d.data, d.target; e, _ = knee_eps(X, 5)
    print(f"the knee rule suggests eps = {e:.2f}; Day 119's k-means scored ARI 0.6525 on the same images")
    print(f"  {'eps':>5}{'clusters':>10}{'noise':>8}{'ARI, all':>10}{'ARI, clustered only':>21}")
    rows = []
    for eps in (16.0, 18.0, 20.0, 22.0, 24.0, round(e, 2)):
        lab, _ = dbscan(X, eps, 5); k, nz, _ = summary(lab); m = lab != NOISE
        rows.append((eps, nz, ari(lab, y), ari(lab[m], y[m]) if m.sum() > 1 else 0.0))
        print(f"  {eps:>5}{k:>10}{nz:>8.3f}{rows[-1][2]:>10.4f}{rows[-1][3]:>21.4f}")
    print("('ARI, all' counts noise as one more group; 'clustered only' judges just the images DBSCAN placed)")
    print("Actual output from this lesson's run.")
    return rows


def plot_results(sets, shp, kd, e, cal, cal_lab, scan, rows):  # six panels
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.6))             # canvas
    for a, name in zip(ax[0, :2], ("two moons", "rings")):
        X = sets[name][0]; lab = shp[name]
        a.scatter(X[:, 0], X[:, 1], c=np.where(lab == NOISE, -1, lab), cmap="coolwarm", s=8); a.set_aspect("equal")
        a.set_title(f"{name}: DBSCAN's clusters", fontsize=10); a.set_xticks([]); a.set_yticks([])
    a = ax[0, 2]; a.plot(kd, color="#1f77b4"); a.axhline(e, color="r", linestyle="--", linewidth=0.8, label=f"knee: eps = {e:.3f}")
    a.set_xlabel("points, sorted by distance"); a.set_ylabel("distance to 5th nearest point"); a.legend(fontsize=8); a.grid(alpha=0.3)
    a.set_title("Two moons: the k-distance curve", fontsize=10)
    a = ax[1, 0]; m = cal_lab == NOISE
    a.scatter(cal[m, 0], cal[m, 1], s=1, color="#cccccc", label="noise")
    a.scatter(cal[~m, 0], cal[~m, 1], s=1, c=cal_lab[~m] % 10, cmap="tab10"); a.set_aspect("equal")
    a.set_title("California districts: DBSCAN, eps = 5 km, min_pts = 50", fontsize=10); a.set_xlabel("km east"); a.set_ylabel("km north")
    a.legend(fontsize=8, markerscale=6)
    a = ax[1, 1]; sc = np.array(scan); a.plot(sc[:, 0], sc[:, 1], marker="o", color="#1f77b4", label="clusters")
    a.set_xlabel("eps (km)"); a.set_ylabel("clusters"); a.grid(alpha=0.3); b = a.twinx()
    b.plot(sc[:, 0], sc[:, 2], marker="s", color="#d62728", label="noise share"); b.set_ylabel("noise share")
    a.set_title("California: one eps for every density", fontsize=10)
    a.legend(loc="upper center", fontsize=8); b.legend(loc="upper right", fontsize=8)
    a = ax[1, 2]; r = np.array(sorted(rows))
    a.plot(r[:, 0], r[:, 1], marker="o", label="noise share"); a.plot(r[:, 0], r[:, 2], marker="s", label="ARI, all images")
    a.plot(r[:, 0], r[:, 3], marker="^", label="ARI, clustered images only"); a.axhline(0.6525, color="k", linestyle=":", linewidth=0.8, label="Day 119's k-means")
    a.set_xlabel("eps"); a.set_title("Digits: purity against coverage", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("dbscan_results.png", dpi=100); plt.close()  # save and free
    print("\nsaved dbscan_results.png")


def draw_diagram():                                     # core, border and noise
    fig, ax = plt.subplots(figsize=(13, 4.4))                   # canvas
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")     # coordinates in "boxes"

    def box(x, y, w, h, text, color, fs=8.5):                     # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", facecolor=color,
                                    edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    ax.text(6.5, 4.15, "Day 121: DBSCAN", ha="center", fontsize=12, fontweight="bold")
    box(1.8, 2.5, 2.8, 1.6, "count each point's\nneighbours within eps", "#FAEEDA")
    box(5.3, 2.5, 3.0, 1.6, "CORE: at least\nmin_pts neighbours", "#DCEBF7")
    box(8.8, 2.5, 3.0, 1.6, "grow a cluster through\ncore points'\nneighbourhoods", "#E5F2E1")
    box(11.8, 2.5, 2.0, 1.6, "BORDER joins;\nthe rest is\nNOISE", "#EEEDFE")
    for x0, x1 in ((3.24, 3.76), (6.84, 7.26), (10.34, 10.76)):
        ax.annotate("", xy=(x1, 2.5), xytext=(x0, 2.5), arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.text(6.5, 0.9, "no k to choose, any shape allowed; one eps and one min_pts must suit every cluster", ha="center", fontsize=8.5)
    plt.savefig("dbscan_diagram.png", dpi=120, bbox_inches="tight"); plt.close()  # save and free
    print("saved dbscan_diagram.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("8. WHAT COMES NEXT")
    print("=" * 70)
    print("Day 122 is Gaussian mixture models and the EM algorithm from scratch: soft clusters, each an ellipse with its own shape and size.")


def main():                                             # runs every section in order
    sets = shapes(); cal, lat, lon, value = california()
    demo_hand()                                            # 1. by hand
    demo_check(sets, cal)                                  # 2. against scikit-learn
    shp = demo_shapes(sets)                                # 3. shapes
    kd, e = demo_eps(sets)                                 # 4. choosing eps
    cal_lab = demo_california(cal, lat, lon, value)        # 5. California
    scan = demo_density(sets, cal, lat, lon)               # 6. one eps for every density
    rows = demo_digits()                                   # 7. digits
    plot_results(sets, shp, kd, e, cal, cal_lab, scan, rows)
    draw_diagram()
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
