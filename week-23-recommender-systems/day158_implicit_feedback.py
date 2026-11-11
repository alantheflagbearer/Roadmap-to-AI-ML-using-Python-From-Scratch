"""
Day 158: Implicit Feedback and Cold Start -- Learning from What People Watch

Most real systems never see star ratings. They see that someone watched,
clicked, bought or listened, and nothing when someone ignored something. This
is implicit feedback: there are no negative examples, only a missing cell that
may mean "did not like" or "never saw". Today MovieLens 100K's ratings become
watched / not watched, and the week's question changes from "how many stars?"
to "what will this user watch next?". Built from scratch in NumPy on Day 154's
split, importing the earlier days' parts:

  1. From stars to "watched": the data, the measure, and popularity.
  2. Implicit ALS (Hu, Koren and Volinsky, 2008): every empty cell is a weak 0.
  3. Stars as confidence, and what the implicit model does for liked films.
  4. The other cold start: brand-new users, with 0 to 20 known films, and
     their age and gender when nothing else is known.

Data: F. Maxwell Harper and Joseph A. Konstan (2015), The MovieLens Datasets:
History and Context, ACM TiiS 5(4). Kept outside the repository.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported parts quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: how long each step takes

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # so the earlier days' parts can be imported from here
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day154_exercise as d154                      # Day 154: data, split, ratings table, biases
    import day155_exercise as d155                      # Day 155: low_rank and ALS for explicit ratings

LU, LI = 5, 2                                           # Day 154's bias penalties
KS = (10, 20, 50, 100)                                  # factor counts tried for implicit ALS (100 added: 50 won the first run, at the grid's edge)
LAMS = (1, 3, 10, 30, 100, 300)                         # penalties tried
ALPHAS = (1, 3, 10, 30)                                 # confidence weights tried
SVD_KS = (5, 10, 20, 50, 100)                           # ranks tried for PureSVD
REVEAL = (0, 1, 3, 5, 10, 20)                           # films known about a brand-new user


# =============================================================================
# THE PARTS (copied unchanged into day158_exercise.py)
# =============================================================================
def load_users(folder=d154.DATA):                       # NEW: age, gender (1 = female) and occupation of every user, from u.user
    rows = [x.split("|") for x in open(os.path.join(folder, "ml-100k", "u.user"), encoding="latin-1").read().strip().split("\n")]  # one list of fields per line of u.user
    occ = sorted({x[3] for x in rows})                                                # the occupations, in alphabetical order
    return (np.array([int(x[1]) for x in rows]), np.array([x[2] == "F" for x in rows]).astype(int),  # ages and genders
            np.array([occ.index(x[3]) for x in rows]))                                # occupation numbers


def implicit_als(C, k, lam, sweeps=10, seed=0, track=None):  # NEW: Hu-Koren-Volinsky ALS; C = confidence where watched, 0 elsewhere
    nu, ni = C.shape; rng = np.random.RandomState(seed)                               # seeded, so a run repeats
    X, Y = 0.01 * rng.randn(nu, k), 0.01 * rng.randn(ni, k)                           # user and film factors
    hist = []                                                                         # one record per sweep, if asked for
    for _ in range(sweeps):                                                           # each sweep: users, then films
        for A, B, Cm in ((X, Y, C), (Y, X, C.T)):                                     # solve the rows of A with B fixed
            G = B.T @ B + lam * np.eye(k)                                             # every empty cell's weight-1 term, shared by all rows
            for a in range(A.shape[0]):                                               # one row (user or film) at a time
                j = np.flatnonzero(Cm[a]); c = Cm[a, j]                               # where this row has interactions, and their confidence
                Bj = B[j]                                                             # the other side's factors there
                A[a] = np.linalg.solve(G + (Bj * (c - 1)[:, None]).T @ Bj, Bj.T @ c)  # weighted ridge: extra weight c - 1 where watched
        hist.append(track(X, Y) if track else ())                                     # measures after this sweep
    return X, Y, hist                                                                 # the factors, and the records


def implicit_objective(X, Y, C, lam):                   # NEW: sum of confidence x (watched - x . y)^2 over EVERY cell + penalty
    P = C > 0                                                                         # 1 where watched, 0 elsewhere
    W = np.where(P, C, 1.0)                                                           # confidence: c where watched, 1 elsewhere
    return (W * (P - X @ Y.T) ** 2).sum() + lam * ((X ** 2).sum() + (Y ** 2).sum())   # weighted squared error + penalty


def fold_in(Y, items, conf, lam):                       # NEW: factors for a new user from the films they watched, Y held fixed
    k = Y.shape[1]; Bj = Y[items]                                                     # the watched films' factors
    G = Y.T @ Y + lam * np.eye(k)                                                     # the same shared term as in implicit_als
    return np.linalg.solve(G + (Bj * (conf - 1)[:, None]).T @ Bj, Bj.T @ conf)        # one user step: no retraining


def cooccurrence(B):                                    # NEW: cosine between films' watcher sets (who watched both)
    n = np.sqrt(B.sum(0))                                                             # each film's length: sqrt(number of watchers)
    S = (B.T @ B) / np.maximum(np.outer(n, n), 1e-12)                                 # shared watchers / sqrt(n_a n_b)
    np.fill_diagonal(S, 0.0)                                                          # a film is not its own neighbour
    return S                                                                          # films x films


def topk_metrics(S, seen, held, k=10):                  # NEW: precision@k and recall@k against held-out watched films
    S = np.where(seen, -np.inf, S); who = np.flatnonzero(held.sum(1) > 0)             # already-watched films are never shown
    top = np.argsort(-S, axis=1, kind="stable")[:, :k]                                # each user's k best-scored films
    h = np.take_along_axis(held, top, 1)[who].sum(1)                                  # hits per user
    return h.mean() / k, (h / held[who].sum(1)).mean(), top[who]                      # precision, recall, the lists


# =============================================================================
# HELPERS FOR THE EXPERIMENTS
# =============================================================================
def setup():                                            # Day 154's data and split, plus the users' details
    u, i, r, t = d154.load_movielens(); titles, genres, gnames = d154.load_items()    # the ratings and the films
    nu, ni = u.max() + 1, i.max() + 1                                                 # users and films, counted from 0
    tr, te = d154.split_ratings(len(r), 0.2, seed=0)                                  # Day 154's 80/20 split
    fi, va = d154.split_ratings(len(tr), 0.1, seed=1); fi, va = tr[fi], tr[va]        # and its validation part
    return u, i, r, titles, nu, ni, tr, te, fi, va, load_users()                     # everything the experiments need


def mask(u, i, idx, nu, ni):                            # watched / not watched, from a set of rating positions
    M = np.zeros((nu, ni), bool); M[u[idx], i[idx]] = True                            # True where the user rated the film
    return M                                                                          # users x films


def conf_binary(M, alpha):                              # confidence 1 + alpha where watched, 0 (meaning weight 1, value 0) elsewhere
    return np.where(M, 1.0 + alpha, 0.0)                                              # the same for every watched film


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_data(u, i, r, nu, ni, tr, te, fi, va):                                        # section 1
    print("\n" + "=" * 70)
    print("1. FROM STARS TO 'WATCHED': THE DATA, THE MEASURE, AND POPULARITY")
    print("=" * 70)
    M = mask(u, i, tr, nu, ni); H = mask(u, i, te, nu, ni)                            # training and test interactions
    print(f"training: {M.sum()} watched cells of {M.size} ({M.mean():.1%}); test: {H.sum()} held-out watched films "
          f"for {(H.sum(1) > 0).sum()} users")
    print(f"every rated film counts as watched, whatever its stars: {np.mean(r[te] <= 2):.1%} of test 'watched' films got 1-2 stars")
    pop = np.broadcast_to(M.sum(0).astype(float), M.shape)                            # popularity: how many watched each film
    rnd = np.random.RandomState(0).rand(nu, ni)                                       # random scores, for scale
    pp, rp, _ = topk_metrics(pop, M, H); pr, rr, _ = topk_metrics(rnd, M, H)          # both on test
    print(f"precision@10 / recall@10 on held-out watched films: popularity {pp:.4f} / {rp:.4f}; random {pr:.4f} / {rr:.4f}")
    Hl = mask(u, i, te[r[te] >= 4], nu, ni)                                           # Day 154's measure: liked test films only
    print(f"Day 154's measure (test films rated 4-5): popularity {topk_metrics(pop, M, Hl)[0]:.4f}")
    print("Actual output from this lesson's run.")
    return M, H, Hl                                                                   # for the later sections


def demo_ials(u, i, r, nu, ni, tr, te, fi, va, M, H, Hl):                             # section 2
    print("\n" + "=" * 70)
    print("2. IMPLICIT ALS: EVERY EMPTY CELL IS A WEAK 0")
    print("=" * 70)
    rng = np.random.RandomState(0); Cs = np.where(rng.rand(6, 5) < 0.5, 1 + 3 * rng.rand(6, 5), 0.0)  # a small check table
    Ys = rng.randn(5, 2); a = 0; j = np.flatnonzero(Cs[a]); W = np.where(Cs[a] > 0, Cs[a], 1.0)  # one user, written out in full
    full = np.linalg.solve(Ys.T @ (W[:, None] * Ys) + 0.5 * np.eye(2), Ys.T @ (W * (Cs[a] > 0)))  # the plain weighted ridge, every cell
    fast = fold_in(Ys, j, Cs[a, j], 0.5)                                              # the shortcut using only watched cells
    print(f"check: the shortcut user step against the full weighted ridge over every cell: largest difference {np.abs(full - fast).max():.1e}")
    Mf = mask(u, i, fi, nu, ni); Hv = mask(u, i, va, nu, ni)                          # training without validation, and validation
    t0 = time.time(); E = np.zeros((len(KS), len(LAMS), len(ALPHAS)))                 # validation precision@10 per setting
    for a1, k in enumerate(KS):                                                       # every factor count
        for b1, lam in enumerate(LAMS):                                               # every penalty
            for c1, al in enumerate(ALPHAS):                                          # every confidence weight
                X, Y, _ = implicit_als(conf_binary(Mf, al), k, lam)                   # fit on the training part
                E[a1, b1, c1] = topk_metrics(X @ Y.T, Mf, Hv)[0]                      # precision@10 on validation
    a1, b1, c1 = np.unravel_index(np.argmax(E), E.shape); k, lam, al = KS[a1], LAMS[b1], ALPHAS[c1]  # the best setting
    print(f"grid: {len(KS)} factor counts x {len(LAMS)} penalties x {len(ALPHAS)} confidence weights ({time.time() - t0:.0f} s); "
          f"validation precision@10 at {k} factors:")
    print(f"  {'penalty':>8} " + " ".join(f"{'alpha ' + str(x):>9}" for x in ALPHAS))
    for b2, lm in enumerate(LAMS):                                                    # one row per penalty at the chosen factor count
        print(f"  {lm:>8} " + " ".join(f"{e:>9.4f}" for e in E[a1, b2]))
    print(f"  best at each factor count: " + ", ".join(f"{kk}: {E[q].max():.4f}" for q, kk in enumerate(KS)))
    print(f"chosen: {k} factors, penalty {lam}, alpha {al} (validation {E.max():.4f})")
    C = conf_binary(M, al)                                                            # all training interactions
    obj = lambda X, Y: (implicit_objective(X, Y, C, lam),)                            # the objective after each sweep
    X, Y, h = implicit_als(C, k, lam, track=obj)                                      # refit on all training data
    print(f"objective after each sweep: {', '.join(f'{x[0]:.0f}' for x in h[:4])}, ..., {h[-1][0]:.0f}; "
          f"never rises: {all(h[q + 1][0] <= h[q][0] + 1e-6 for q in range(len(h) - 1))}")
    Es = [topk_metrics(d155.low_rank(Mf.astype(float), kk), Mf, Hv)[0] for kk in SVD_KS]  # PureSVD: SVD of the 0/1 table
    ks = SVD_KS[int(np.argmax(Es))]                                                   # its rank, chosen on validation
    R, Mr = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni)                               # the star ratings, for the explicit model
    Fe, _ = d155.als_mf(R, Mr, 100, 0.125)                                            # Day 155's explicit ALS, its chosen setting
    exp = Fe["mu"] + Fe["bu"][:, None] + Fe["bi"][None, :] + Fe["P"] @ Fe["Q"].T      # predicted stars everywhere
    Xs = X[np.random.RandomState(0).permutation(nu)]                                  # control: users' factors shuffled
    B = M.astype(float)                                                               # 0/1 table
    scores = {"popularity": np.broadcast_to(B.sum(0), M.shape),                       # every model's scores
              "co-occurrence (item-item)": B @ cooccurrence(B),                       # sum of similarities to watched films
              f"PureSVD, rank {ks}": d155.low_rank(B, ks),                            # the 0/1 table's best rank-k fit
              "explicit ALS (Day 155, stars)": exp,                                   # predicted stars
              f"implicit ALS ({k}, {lam}, {al})": X @ Y.T,                            # today's model
              "control: implicit ALS, users shuffled": Xs @ Y.T}                      # personal factors removed
    print(f"PureSVD rank chosen on validation: {ks} ({', '.join(f'{x}: {e:.4f}' for x, e in zip(SVD_KS, Es))})")
    print(f"{'test':>40} {'precision@10':>13} {'recall@10':>10} {'liked: precision@10':>20} {'median watchers':>16}")
    out = {}                                                                          # for the figure
    for name, S in scores.items():                                                    # one row per model
        p, rc, top = topk_metrics(S, M, H); pl = topk_metrics(S, M, Hl)[0]            # watched, and liked
        out[name] = (p, rc, pl, np.median(B.sum(0)[top]))                             # every measure
        print(f"{name:>40} {p:>13.4f} {rc:>10.4f} {pl:>20.4f} {out[name][3]:>16.0f}")
    print("Actual output from this lesson's run.")
    return X, Y, k, lam, al, E, a1, h, out                                            # the model and the results


def demo_stars(u, i, r, nu, ni, tr, te, fi, va, M, H, Hl, k, lam, al):                # section 3
    print("\n" + "=" * 70)
    print("3. STARS AS CONFIDENCE")
    print("=" * 70)
    Rf, Mf = d154.to_matrix(u[fi], i[fi], r[fi], nu, ni); Hv = mask(u, i, va, nu, ni)  # stars without validation
    Hvl = mask(u, i, va[r[va] >= 4], nu, ni)                                          # liked validation films
    print(f"confidence = 1 + alpha x stars instead of 1 + alpha ({k} factors, penalty {lam}); validation precision@10 watched / liked:")
    best = None                                                                       # the best star-weighted setting
    for a2 in (al / 20, al / 10, al / 5, al / 3, al / 2, al):                         # a few scales (stars average about 3.5); al/20 and al/10 added: al/5 won the first run, at the edge
        X, Y, _ = implicit_als(np.where(Mf, 1 + a2 * Rf, 0.0), k, lam)                # star-weighted confidence
        pw, pl = topk_metrics(X @ Y.T, Mf, Hv)[0], topk_metrics(X @ Y.T, Mf, Hvl)[0]   # both measures
        print(f"  alpha {a2:g} per star: {pw:.4f} / {pl:.4f}")
        if best is None or pw > best[0]:                                              # chosen on watched validation precision
            best = (pw, a2)  # the new best: score and setting
    X, Y, _ = implicit_als(conf_binary(Mf, al), k, lam)                               # the binary model, same split
    print(f"  binary, alpha {al}: {topk_metrics(X @ Y.T, Mf, Hv)[0]:.4f} / {topk_metrics(X @ Y.T, Mf, Hvl)[0]:.4f}")
    R, Mr = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni)                               # all training stars
    X, Y, _ = implicit_als(np.where(Mr, 1 + best[1] * R, 0.0), k, lam)                # refit the chosen star weighting
    print(f"test, stars as confidence (alpha {best[1]:g} per star): watched {topk_metrics(X @ Y.T, M, H)[0]:.4f}, "
          f"liked {topk_metrics(X @ Y.T, M, Hl)[0]:.4f}")
    print("Actual output from this lesson's run.")


def demo_cold_users(u, i, r, titles, nu, ni, users, k, lam, al):                     # section 4
    print("\n" + "=" * 70)
    print("4. THE OTHER COLD START: BRAND-NEW USERS")
    print("=" * 70)
    age, fem, occ = users                                                             # the users' details
    rng = np.random.RandomState(4); cold = np.zeros(nu, bool); cold[rng.permutation(nu)[:nu // 5]] = True  # 20% of users held out
    M = np.zeros((nu, ni), bool); M[u[~cold[u]], i[~cold[u]]] = True                  # training: every interaction of the other users
    A = np.zeros((nu, ni), bool); A[u[cold[u]], i[cold[u]]] = True                    # everything the new users watched
    cu = np.flatnonzero(cold)                                                         # the new users
    print(f"{cold.sum()} new users held out entirely (every one has at least {A[cu].sum(1).min()} films); "
          f"models trained on the other {nu - cold.sum()} users")
    B = M.astype(float); cnt = B.sum(0)                                               # training 0/1 table and film counts
    X, Y, _ = implicit_als(conf_binary(M, al), k, lam); S = cooccurrence(B)           # the two personal models
    band = np.digitize(age, (25, 35, 45))                                             # age bands: under 25, 25-34, 35-44, 45+
    grp = band * 2 + fem                                                              # 8 groups of age band x gender
    gpop = np.array([B[(grp == g) & ~cold].sum(0) for g in range(8)])                 # each group's own popularity
    sg = grp[cu][np.random.RandomState(6).permutation(len(cu))]                       # control: the new users' groups shuffled among them
    res = {m: [] for m in ("popularity", "age and gender popularity", "control: groups shuffled", "co-occurrence",
                           "implicit ALS fold-in")}                                   # recall per model
    for n in REVEAL:                                                                  # films known about each new user
        Rv = np.zeros((len(cu), ni), bool); rs = np.random.RandomState(5)             # what is revealed
        for q, a in enumerate(cu):                                                    # each new user
            w = np.flatnonzero(A[a]); Rv[q, rs.choice(w, n, replace=False)] = True    # n of their films, at random
        Hd = A[cu] & ~Rv                                                              # the rest are to be found
        sc = {"popularity": np.broadcast_to(cnt, Rv.shape),                           # the same list for everyone
              "age and gender popularity": gpop[grp[cu]] + 1e-3 * cnt,                # the group's favourites (overall count breaks ties)
              "control: groups shuffled": gpop[sg] + 1e-3 * cnt,                      # another new user's group's favourites
              "co-occurrence": Rv.astype(float) @ S + 1e-6 * cnt,                     # similar to what they watched
              "implicit ALS fold-in": np.array([fold_in(Y, np.flatnonzero(Rv[q]), np.full(n, 1.0 + al), lam) for q in range(len(cu))]) @ Y.T
              + 1e-6 * cnt}                                                           # their own factors from n films
        for m in res:                                                                 # score every model
            res[m].append(topk_metrics(sc[m], Rv, Hd)[1])                             # recall@10
    print(f"recall@10 for new users, by how many of their films are known (popularity breaks ties when nothing is known):")
    print(f"  {'films known':>28} " + " ".join(f"{n:>7}" for n in REVEAL))
    for m, v in res.items():                                                          # one row per model
        print(f"  {m:>28} " + " ".join(f"{x:>7.4f}" for x in v))
    print("Actual output from this lesson's run.")
    return res                                                                        # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(E, a1, h, out, res, path):                                           # six panels
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # two rows of three
    im = ax[0, 0].imshow(E[a1], cmap="viridis", aspect="auto")                        # validation precision at the chosen factor count
    ax[0, 0].set_xticks(range(len(ALPHAS))); ax[0, 0].set_xticklabels(ALPHAS)         # alpha along the bottom
    ax[0, 0].set_yticks(range(len(LAMS))); ax[0, 0].set_yticklabels(LAMS)             # penalty down the side
    for b in range(len(LAMS)):                                                        # each cell's value
        for c in range(len(ALPHAS)):  # each column
            ax[0, 0].text(c, b, f"{E[a1, b, c]:.3f}", ha="center", va="center", fontsize=8, color="w")  # the value, in white
    ax[0, 0].set_xlabel("alpha (confidence of a watched film)"); ax[0, 0].set_ylabel("penalty")  # labels
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046); ax[0, 0].set_title(f"validation precision@10, {KS[a1]} factors", fontsize=10)  # colour scale and title
    ax[0, 1].plot(range(1, len(h) + 1), [x[0] for x in h], "o-")                      # objective per sweep
    ax[0, 1].set_xlabel("sweep"); ax[0, 1].set_ylabel("objective"); ax[0, 1].set_title("implicit ALS objective falls every sweep", fontsize=10)  # labels and title
    nm = list(out); short = [x.split(" (")[0].replace("control: ", "control:\n") for x in nm]  # model names, shortened
    xs = np.arange(len(nm))                                                           # one group per model
    ax[0, 2].bar(xs - 0.2, [out[x][0] for x in nm], 0.4, label="watched")            # precision on watched films
    ax[0, 2].bar(xs + 0.2, [out[x][2] for x in nm], 0.4, label="liked (4-5 stars)")  # precision on liked films
    ax[0, 2].set_xticks(xs); ax[0, 2].set_xticklabels(short, fontsize=7, rotation=20, ha="right")  # names
    ax[0, 2].set_ylabel("test precision@10"); ax[0, 2].legend(fontsize=8); ax[0, 2].set_title("top-10 lists, test", fontsize=10)  # label, legend and title
    ax[1, 0].bar(xs, [out[x][3] for x in nm], color="C2")                             # how popular the recommended films are
    ax[1, 0].set_xticks(xs); ax[1, 0].set_xticklabels(short, fontsize=7, rotation=20, ha="right")  # model names
    ax[1, 0].set_ylabel("median watchers of recommended films"); ax[1, 0].set_title("how popular are the recommendations?", fontsize=10)  # label and title
    for q, (m, v) in enumerate(res.items()):                                          # recall for new users
        ax[1, 1].plot(REVEAL, v, "o-", color=f"C{q}", label=m)                        # one colour per model, kept in the next panel
    ax[1, 1].set_xlabel("films known about the new user"); ax[1, 1].set_ylabel("recall@10"); ax[1, 1].legend(fontsize=8)  # labels and legend
    ax[1, 1].set_title("new users: recall@10 as their history grows", fontsize=10)    # title
    base = np.array(res["popularity"])                                                # the baseline
    for q, (m, v) in enumerate(res.items()):                                          # gain over popularity
        if m != "popularity":                                                         # popularity against itself is 0
            ax[1, 2].plot(REVEAL, np.array(v) - base, "o-", color=f"C{q}", label=m)  # the same colour as the last panel
    ax[1, 2].axhline(0, color="k", lw=0.8); ax[1, 2].set_xlabel("films known about the new user")  # zero line
    ax[1, 2].set_ylabel("recall@10 minus popularity's"); ax[1, 2].legend(fontsize=8)  # labels
    ax[1, 2].set_title("new users: gain over popularity", fontsize=10)                 # title
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)                    # save to disk


def draw_diagram():                                     # explicit against implicit
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.4); ax.axis("off")  # a wide, empty canvas
    p = 0.08                                                                          # the boxes' rounded padding

    def box(x, y, w, h, text, fc):                                                    # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={p}", fc=fc, ec="k", lw=1.2))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)   # its text

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))  # from (x0, y0) to (x1, y1)
    box(0.3, 3.4, 3.0, 1.2, "explicit: stars\n(known cells only)", "#e8f0fe")         # Days 154-157
    box(0.3, 0.8, 3.0, 1.2, "implicit: watched or not\n(every cell: 1 or a weak 0)", "#fde8e8")  # today
    box(4.6, 3.4, 3.6, 1.2, "fit known cells only\n(Day 155 ALS / SGD)", "#e8f0fe")    # explicit fit
    box(4.6, 0.8, 3.6, 1.2, "weighted fit of every cell:\nwatched = 1 with weight 1 + alpha,\nempty = 0 with weight 1", "#fde8e8")  # implicit fit
    box(9.5, 2.1, 2.6, 1.2, "user and film\nfactors", "#e6f4ea")                       # shared output
    box(12.6, 2.1, 2.1, 1.2, "top-10 list", "#e6f4ea")                                 # and the list
    box(9.5, 0.2, 5.2, 1.0, "new user: solve one user step from the films\nthey watched (fold-in), or fall back on popularity", "#fff4d6")  # cold users
    arrow(3.3 + p, 4.0, 4.6 - p, 4.0); arrow(3.3 + p, 1.4, 4.6 - p, 1.4)              # into each fit
    arrow(8.2 + p, 4.0, 9.5 - p, 3.0); arrow(8.2 + p, 1.4, 9.5 - p, 2.4)              # into the factors
    arrow(12.1 + p, 2.7, 12.6 - p, 2.7)                                               # then the list
    arrow(10.8, 2.1 - p, 10.8, 1.2 + p)                                               # the film factors serve new users too
    ax.set_title("Day 158: from stars to 'watched', and recommending to a brand-new user", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "implicit_feedback_diagram.png"), dpi=120); plt.close(fig)  # save to disk


def note_on_what_comes_next():  # a preview of Day 159
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 159: evaluating recommenders. Precision@10 has been the week's only ranking measure. Tomorrow")
    print("builds precision@k, recall@k, MAP and NDCG from scratch, asks which ones reward what, and looks at")
    print("popularity bias and how sure a difference between two models really is.")


def main():                                             # runs every section in order
    u, i, r, titles, nu, ni, tr, te, fi, va, users = setup()                         # Day 154's data and split, and the users
    M, H, Hl = demo_data(u, i, r, nu, ni, tr, te, fi, va)                            # 1. the data and popularity
    X, Y, k, lam, al, E, a1, h, out = demo_ials(u, i, r, nu, ni, tr, te, fi, va, M, H, Hl)  # 2. implicit ALS
    demo_stars(u, i, r, nu, ni, tr, te, fi, va, M, H, Hl, k, lam, al)                 # 3. stars as confidence
    res = demo_cold_users(u, i, r, titles, nu, ni, users, k, lam, al)                 # 4. new users
    plot_results(E, a1, h, out, res, os.path.join(HERE, "implicit_feedback_results.png"))  # the results figure
    draw_diagram()                                                                    # the method
    print("\nsaved implicit_feedback_results.png\nsaved implicit_feedback_diagram.png")
    note_on_what_comes_next()                                                         # a preview of Day 159


if __name__ == "__main__":  # only when run directly, not when imported
    main()                                                                            # run everything
