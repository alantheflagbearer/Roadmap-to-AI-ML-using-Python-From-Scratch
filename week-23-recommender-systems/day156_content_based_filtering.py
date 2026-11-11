"""
Day 156: Content-Based Filtering -- Recommending from What the Films Are

Days 154 and 155 learned only from who rated what. A film with no ratings has no
neighbours and no factors, so neither method can place it. Content-based
filtering describes each film by its own attributes (here MovieLens 100K's 19
genre flags and the decade of release), learns from each user's ratings which
attributes that user likes (a profile), and scores any film, rated or not, by
how well its attributes match the profile.

Built from scratch in NumPy on Day 154's split (Day 154's parts are imported
from day154_exercise.py):

  1. Film content: features, content similarity, and how far it agrees with ratings.
  2. User profiles: a ridge regression per user, shrunk toward everyone's profile.
  3. Cold start: films nobody has rated, where ratings-only methods cannot help.
  4. Top-10 lists: accuracy, and how narrow content-based lists are.
  5. Explaining a recommendation from one user's profile.

Data: F. Maxwell Harper and Joseph A. Konstan (2015), The MovieLens Datasets:
History and Context, ACM TiiS 5(4). Kept outside the repository.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep the imported part quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: to print film titles with accents on any console
import time                                            # time: how long each step takes

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

if hasattr(sys.stdout, "reconfigure"):                  # only real consoles can be reconfigured
    sys.stdout.reconfigure(encoding="utf-8")            # so titles such as 'Belle de jour' print on any console
HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # so Day 154's part can be imported from here
with contextlib.redirect_stdout(io.StringIO()):         # day154_exercise prints its hand checks when imported
    import day154_exercise as d154                      # Day 154: data, split, ratings table, biases, similarity, item-based

LU, LI = 5, 2                                           # Day 154's bias penalties, chosen there on the same validation split
ITEM = dict(k=40, shrink=800)                           # Day 154's item-based setting (adjusted cosine), chosen there
LAMS = (0.3, 1, 3, 10, 30, 100, 300, 1000)              # profile penalties tried
GLAMS = (0.01, 0.1, 1, 10, 100, 1000, 10000)            # penalties tried for everyone's shared profile
KS = (5, 10, 20, 40, 80, 160, 320)                      # neighbour counts tried for content kNN
DECADES = ("before 1930", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s")  # the release-decade features


# =============================================================================
# THE PARTS (copied unchanged into day156_exercise.py)
# =============================================================================
def load_years(folder=d154.DATA):                       # NEW: each film's release year from u.item, -1 when missing
    path = os.path.join(folder, "ml-100k", "u.item")                                  # Day 154 downloaded it
    rows = open(path, encoding="latin-1").read().strip().split("\n")                  # one film per line, latin-1 like load_items
    return np.array([int(x.split("|")[2][-4:]) if x.split("|")[2] else -1 for x in rows])  # the date ends in the year


def film_features(genres, years):                       # NEW: 19 genre flags + 8 decade flags per film (27 numbers)
    dec = np.clip((years - 1920) // 10, 0, 7)                                         # 0 = before 1930, ..., 7 = 1990s
    D = np.eye(8)[dec] * (years > 0)[:, None]                                         # one decade flag; none if the year is missing
    return np.c_[genres, D].astype(float)                                             # films x 27


def content_similarity(X):                              # NEW: cosine of every pair of films' feature vectors
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)              # each row scaled to length 1
    S = Xn @ Xn.T                                                                     # every pair's cosine at once
    np.fill_diagonal(S, 0.0)                                                          # a film is never its own neighbour
    return S                                                                          # films x films


def fit_shared_profile(X, T, M, lam):                   # NEW: one ridge regression for everyone: T[u, i] ~ X[i] . w
    uu, ii = np.nonzero(M)                                                            # every known (user, film) pair
    A = X[ii]                                                                         # the film's features, one row per rating
    return np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ T[uu, ii])      # ridge solution, 27 numbers


def fit_profiles(X, T, M, lam, prior=None):             # NEW: a ridge regression per user, shrunk toward the prior (default 0)
    w0 = np.zeros(X.shape[1]) if prior is None else prior                             # what an unknown user's profile is
    W = np.tile(w0, (M.shape[0], 1))                                                  # every user starts at the prior
    for a in range(M.shape[0]):                                                       # one user at a time
        j = np.flatnonzero(M[a]); A = X[j]                                            # the films user a rated, as features
        W[a] += np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ (T[a, j] - A @ w0))  # ridge on what the prior misses
    return W                                                                          # users x 27


def content_predict(W, X, u, i):                        # NEW: profile . film features, for each (user, film) pair
    return (W[u] * X[i]).sum(1)                                                       # one number per pair


def cold_split(i, ni, frac, seed, exclude=None):        # NEW: hold out whole films: every rating of a chosen film
    pool = np.arange(ni) if exclude is None else np.flatnonzero(~exclude)              # films still available
    rng = np.random.RandomState(seed); cold = np.zeros(ni, bool)                      # seeded, so the split repeats
    cold[rng.permutation(pool)[:int(round(frac * ni))]] = True                       # frac of all films chosen
    return cold, np.flatnonzero(cold[i])                                              # the films, and their ratings' positions


# =============================================================================
# HELPERS FOR THE EXPERIMENTS
# =============================================================================
def setup():                                            # Day 154's data and split, rebuilt exactly, plus the content
    u, i, r, t = d154.load_movielens(); titles, genres, gnames = d154.load_items()    # the ratings, titles and genre flags
    nu, ni = u.max() + 1, i.max() + 1                                                 # users and films, counted from 0
    tr, te = d154.split_ratings(len(r), 0.2, seed=0)                                  # Day 154's 80/20 split
    fi, va = d154.split_ratings(len(tr), 0.1, seed=1); fi, va = tr[fi], tr[va]        # and its validation part
    years = load_years(); X = film_features(genres, years)                           # the content
    return u, i, r, titles, genres, gnames, years, X, nu, ni, tr, te, fi, va          # everything the experiments need


def score(p, r):                                        # RMSE of clipped predictions
    return d154.rmse(np.clip(p, 1, 5), r)                                             # predictions kept inside 1-5 stars


def user_only(R, M):                                    # mean + user bias, with no film bias at all
    mu, bu, _ = d154.fit_biases(R, M, LU, 1e12)                                       # a huge film penalty keeps every film bias at 0
    return mu, bu                                                                     # what a brand-new film can use


def pick(errors, grid):                                 # the setting with the lowest validation error
    return grid[int(np.argmin(errors))]                                               # first one on a tie


def top10(Q, M, liked):                                 # precision@10 and the films shown, for users with a liked test film
    Q = np.where(M, -np.inf, Q); who = np.flatnonzero(liked.sum(1) > 0)               # rated films can never be picked
    top = np.argsort(-Q, axis=1, kind="stable")[:, :10]                               # each user's 10 best-scored films
    return np.take_along_axis(liked, top, 1)[who].sum(1).mean() / 10, top[who]       # mean precision@10, and the lists


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_content(u, i, r, titles, genres, gnames, years, X, nu, ni, tr):              # section 1
    print("\n" + "=" * 70)
    print("1. FILM CONTENT: FEATURES, CONTENT SIMILARITY, AND RATINGS")
    print("=" * 70)
    k = genres.sum(1)                                                                 # genres per film
    print(f"{ni} films x {X.shape[1]} features (19 genres + 8 decades); genres per film: "
          + ", ".join(f"{c} films with {g}" for g, c in enumerate(np.bincount(k)) if c))  # how many films have 1, 2, ... genres
    print(f"release year missing for {(years < 0).sum()} film(s); years {years[years > 0].min()}-{years.max()}; "
          f"1990s films {int(X[:, -1].sum())} ({X[:, -1].mean():.0%})")              # the decade feature is lopsided
    S = content_similarity(X)                                                         # content cosine of every pair
    sw = titles.index("Star Wars (1977)")                                             # the same example as Days 154-155
    print(f"Star Wars (1977): {', '.join(g for g, f in zip(gnames, genres[sw]) if f)}, 1970s")
    top = np.argsort(-S[sw], kind="stable")                                           # films by content similarity
    ties = int((S[sw] == S[sw, top[4]]).sum())                                        # how many share the fifth value
    print("  content neighbours: " + "; ".join(f"{titles[j]} ({S[sw, j]:.2f})" for j in top[:5]))  # the five closest by content
    print(f"  ({ties} films share the fifth one's cosine; ties are broken by file order, not merit)")  # a warning about ties
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni)                                # all the training ratings
    Xr = (R - (R.sum(1) / np.maximum(M.sum(1), 1))[:, None]) * M                      # ratings minus each user's mean (adjusted cosine)
    Sr, _ = d154.similarity(Xr, M, shrink=ITEM["shrink"])                             # Day 154's rating similarity
    o = np.argsort(-Sr[sw])[:5]                                                       # Star Wars's rating neighbours
    print("  rating neighbours (Day 154): " + "; ".join(f"{titles[j]} (content {S[sw, j]:.2f})" for j in o))
    ok = np.flatnonzero(M.sum(0) >= 50)                                               # films with 50+ training ratings
    nb = np.argsort(-Sr[np.ix_(ok, ok)], axis=1)[:, :10]                              # each one's 10 rating neighbours among them
    near = S[ok[:, None], ok[nb]].ravel()                                             # content cosine with those neighbours
    rng = np.random.RandomState(0); rnd = rng.randint(0, len(ok), nb.shape)           # control: 10 random films each
    far = S[ok[:, None], ok[rnd]].ravel()                                             # content cosine with random films
    print(f"films with 50+ ratings ({len(ok)}): mean content cosine with their 10 rating neighbours {near.mean():.3f}, "
          f"with 10 random films {far.mean():.3f}; same genres and decade (cosine 1): {(near > 0.999).mean():.1%} against "
          f"{(far > 0.999).mean():.1%}")                                              # do ratings and content agree?
    print("Actual output from this lesson's run.")
    return near, far                                                                  # for the figure


def demo_profiles(u, i, r, X, nu, ni, tr, te, fi, va):                                # section 2
    print("\n" + "=" * 70)
    print("2. USER PROFILES: A RIDGE REGRESSION PER USER")
    print("=" * 70)
    Rf, Mf = d154.to_matrix(u[fi], i[fi], r[fi], nu, ni)                              # training without validation
    mu, bu = user_only(Rf, Mf); T = (Rf - mu - bu[:, None]) * Mf                      # what the user's mean leaves to explain
    mb, bub, bib = d154.fit_biases(Rf, Mf, LU, LI); B = mb + bub[:, None] + bib[None, :]  # Day 154's bias model
    Tb = (Rf - B) * Mf                                                                # what the bias model leaves
    eg = [score(mu + bu[u[va]] + X[i[va]] @ fit_shared_profile(X, T, Mf, g), r[va]) for g in GLAMS]  # shared profile, every penalty
    g = pick(eg, GLAMS); wg = fit_shared_profile(X, T, Mf, g)                         # the chosen shared profile
    wgb = fit_shared_profile(X, Tb, Mf, g)                                            # the same on the bias model's residuals
    t0 = time.time(); Ec, Eb = [], []                                                 # validation errors per penalty
    for lam in LAMS:                                                                  # every penalty
        W = fit_profiles(X, T, Mf, lam, prior=wg)                                     # content only: profiles from the user's mean
        Ec.append(score(mu + bu[u[va]] + content_predict(W, X, u[va], i[va]), r[va]))  # scored on validation
        W = fit_profiles(X, Tb, Mf, lam, prior=wgb)                                   # on top of the bias model
        Eb.append(score(B[u[va], i[va]] + content_predict(W, X, u[va], i[va]), r[va]))  # scored on validation
    print(f"validation RMSE ({time.time() - t0:.1f} s): mean + user bias {score(mu + bu[u[va]], r[va]):.4f}; "
          f"+ shared profile (penalty {g}) {min(eg):.4f}; bias model {score(B[u[va], i[va]], r[va]):.4f}")
    print(f"{'penalty':>9} {'content only':>13} {'bias + content':>15}   (per-user profiles, shrunk toward the shared one)")
    for lam, a, b in zip(LAMS, Ec, Eb):                                               # one row per penalty
        print(f"{lam:>9g} {a:>13.4f} {b:>15.4f}")
    lc, lb = pick(Ec, LAMS), pick(Eb, LAMS)                                           # the two choices
    print(f"chosen on validation: content only, penalty {lc}; bias + content, penalty {lb}")
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni)                                # all the training ratings
    mu, bu = user_only(R, M); T = (R - mu - bu[:, None]) * M                          # refit everything on them
    mb, bub, bib = d154.fit_biases(R, M, LU, LI); B = mb + bub[:, None] + bib[None, :]  # and Day 154's bias model
    Wc = fit_profiles(X, T, M, lc, prior=fit_shared_profile(X, T, M, g))              # content-only profiles
    Wb = fit_profiles(X, (R - B) * M, M, lb, prior=fit_shared_profile(X, (R - B) * M, M, g))  # profiles on the residuals
    Xs = X[np.random.RandomState(0).permutation(ni)]                                  # control: every film gets another film's content
    Ws = fit_profiles(Xs, T, M, lc, prior=fit_shared_profile(Xs, T, M, g))            # the same model on the shuffled content
    Xr = (R - (R.sum(1) / np.maximum(M.sum(1), 1))[:, None]) * M                      # Day 154's item-based model
    S, _ = d154.similarity(Xr, M, shrink=ITEM["shrink"])                              # adjusted cosine, shrunk
    c, _ = d154.predict_item_knn((R - B) * M, M, S, u[te], i[te], ITEM["k"])          # its correction to the bias model
    rows = {"mean + user bias": mu + bu[u[te]],                                       # every model's test predictions
            "content only (user profiles)": mu + bu[u[te]] + content_predict(Wc, X, u[te], i[te]),
            "control: content shuffled among films": mu + bu[u[te]] + content_predict(Ws, Xs, u[te], i[te]),
            "bias model (Day 154)": B[u[te], i[te]],
            "bias model + content profiles": B[u[te], i[te]] + content_predict(Wb, X, u[te], i[te]),
            "item-based, k 40 (Day 154)": B[u[te], i[te]] + c}
    print(f"{'test, 20,000 ratings':>40}    RMSE")
    for name, p in rows.items():                                                      # one row per model
        print(f"{name:>40}  {score(p, r[te]):.4f}")
    cnt = M.sum(0)[i[te]]; edges = (0, 1, 6, 21, 51, 151, 10 ** 6)                   # each test rating's film's training count
    names = ("0", "1-5", "6-20", "21-50", "51-150", "151+")                           # the bins' labels
    print("test RMSE by how many training ratings the film has:")
    print(f"  {'ratings':>8} {'test ratings':>13} {'content only':>13} {'bias model':>11} {'bias + content':>15} {'item-based':>11}")
    byc = {}                                                                          # for the figure
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):                              # one row per bin
        q = (cnt >= lo) & (cnt < hi)                                                  # the test ratings in this bin
        e = [score(rows[k][q], r[te][q]) for k in ("content only (user profiles)", "bias model (Day 154)",  # four models' errors in this bin
                                                   "bias model + content profiles", "item-based, k 40 (Day 154)")]
        byc[nm] = (q.sum(), e)                                                        # count and the four errors
        print(f"  {nm:>8} {q.sum():>13} {e[0]:>13.4f} {e[1]:>11.4f} {e[2]:>15.4f} {e[3]:>11.4f}")
    print("Actual output from this lesson's run.")
    return Wc, Wb, B, mu, bu, g, lc, Ec, Eb, rows, byc                                # the models and the results


def demo_cold(u, i, r, X, nu, ni, g):                                                 # section 3
    print("\n" + "=" * 70)
    print("3. COLD START: FILMS NOBODY HAS RATED")
    print("=" * 70)
    cold, te = cold_split(i, ni, 0.2, seed=2)                                         # 20% of films: all their ratings are test
    cval, va = cold_split(i, ni, 0.1, seed=3, exclude=cold)                           # another 10% of films for validation
    fi = np.flatnonzero(~cold[i] & ~cval[i])                                          # training: ratings of the other 70% of films
    print(f"films: {(~cold & ~cval).sum()} for training ({len(fi)} ratings), {cval.sum()} for validation ({len(va)}), "
          f"{cold.sum()} for test ({len(te)}); every rating of a held-out film is held out")
    R, M = d154.to_matrix(u[fi], i[fi], r[fi], nu, ni)                                # only the training films have ratings
    mu, bu = user_only(R, M); T = (R - mu - bu[:, None]) * M                          # mean + user bias
    mb, bub, bib = d154.fit_biases(R, M, LU, LI)                                      # the bias model: a new film's bias is 0
    print(f"  a held-out film's bias in Day 154's model is {np.abs(bib[cold]).max():.1f}, and it has no raters, so the "
          "bias model, item-based\n  neighbours and learned factors all fall back to mean + user bias for it")
    eg = [score(mu + bu[u[va]] + X[i[va]] @ fit_shared_profile(X, T, M, gg), r[va]) for gg in GLAMS]  # shared profile, validated on new films
    gc = pick(eg, GLAMS); wg = fit_shared_profile(X, T, M, gc)                        # the chosen shared profile
    E = [score(mu + bu[u[va]] + content_predict(fit_profiles(X, T, M, lam, prior=wg), X, u[va], i[va]), r[va]) for lam in LAMS]  # profiles, every penalty
    lam = pick(E, LAMS); W = fit_profiles(X, T, M, lam, prior=wg)                     # per-user profiles, validated on new films
    S = content_similarity(X); Ek = []                                                # content kNN: rated films that look alike
    for k in KS:                                                                      # every neighbour count
        c, _ = d154.predict_item_knn(T, M, S, u[va], i[va], k)                        # Day 154's item-based predictor, content similarity
        Ek.append(score(mu + bu[u[va]] + c, r[va]))                                   # scored on new validation films
    k = pick(Ek, KS); c, _ = d154.predict_item_knn(T, M, S, u[te], i[te], k)          # the chosen count, on test
    print("validation (new films): " + f"shared profile penalty {gc} ({min(eg):.4f}); profiles penalty {lam} ({min(E):.4f}); "
          f"content kNN k {k} ({min(Ek):.4f})")
    print("  profiles by penalty: " + ", ".join(f"{x:g}: {e:.4f}" for x, e in zip(LAMS, E)))
    print("  content kNN by k: " + ", ".join(f"{x}: {e:.4f}" for x, e in zip(KS, Ek)))
    Xs = X[np.random.RandomState(0).permutation(ni)]                                  # control: shuffled content
    Ws = fit_profiles(Xs, T, M, lam, prior=fit_shared_profile(Xs, T, M, gc))          # the same model, wrong content
    rows = {"mean + user bias (all ratings-only methods)": mu + bu[u[te]],            # the fallback
            "shared profile (genre and decade effects)": mu + bu[u[te]] + X[i[te]] @ wg,
            "user profiles (shrunk toward shared)": mu + bu[u[te]] + content_predict(W, X, u[te], i[te]),
            "content kNN": mu + bu[u[te]] + c,
            "control: profiles, content shuffled": mu + bu[u[te]] + content_predict(Ws, Xs, u[te], i[te])}
    print(f"{'test: ' + str(len(te)) + ' ratings of new films':>44}    RMSE")
    for name, p in rows.items():                                                      # one row per model
        print(f"{name:>44}  {score(p, r[te]):.4f}")
    Wz = fit_profiles(X, T, M, lam)                                                   # the same profiles shrunk toward 0 instead
    print(f"  profiles shrunk toward 0 rather than the shared profile: {score(mu + bu[u[te]] + content_predict(Wz, X, u[te], i[te]), r[te]):.4f}")
    print("Actual output from this lesson's run.")
    return rows, r[te], E, Ek                                                         # for the figure


def demo_topn(u, i, r, titles, genres, X, nu, ni, tr, te, Wc, Wb, B):                 # section 4
    print("\n" + "=" * 70)
    print("4. TOP-10 LISTS: ACCURACY AND NARROWNESS")
    print("=" * 70)
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni); cnt = M.sum(0)               # training ratings and counts
    liked = np.zeros(M.shape, bool); g = r[te] >= 4; liked[u[te][g], i[te][g]] = True  # test films rated 4-5 are hits
    lists = {"content only": Wc @ X.T, "bias model + content": B + Wb @ X.T,          # scores for every (user, film)
             "bias model": B, "popularity": np.broadcast_to(cnt.astype(float), M.shape),
             "random": np.random.RandomState(0).rand(nu, ni)}
    out = {}                                                                          # results per list
    print(f"{'':>22} {'precision@10':>13} {'median ratings':>15} {'unrated films':>14} {'distinct genres':>16}")
    for name, Q in lists.items():                                                     # one row per list
        p, top = top10(Q, M, liked)                                                   # precision and the lists themselves
        dg = np.mean([np.count_nonzero(genres[t].sum(0)) for t in top])               # genres covered by a 10-film list
        out[name] = (p, np.median(cnt[top]), (cnt[top] == 0).mean(), dg)              # every measure
        print(f"{name:>22} {p:>13.4f} {out[name][1]:>15.0f} {out[name][2]:>14.1%} {dg:>16.1f}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_explain(titles, genres, gnames, X, Wc, M_tr_counts):                         # section 5
    print("\n" + "=" * 70)
    print("5. EXPLAINING A RECOMMENDATION FROM ONE USER'S PROFILE")
    print("=" * 70)
    a = 0                                                                             # the first user in the file, not hand-picked
    names = list(gnames) + list(DECADES)                                              # the 27 features' names
    o = np.argsort(-Wc[a])                                                            # features from most to least liked
    print(f"user 1's content profile (stars above the user's mean): likes " + ", ".join(f"{names[j]} {Wc[a, j]:+.2f}" for j in o[:4]))
    print(f"  {'':>36}dislikes " + ", ".join(f"{names[j]} {Wc[a, j]:+.2f}" for j in o[::-1][:4]))
    return a, names                                                                   # the user and the feature names


def explain_list(a, titles, genres, gnames, X, Wc, rated, names):                     # section 5, continued: the list itself
    s = np.where(rated[a], -np.inf, Wc[a] @ X.T); top = np.argsort(-s, kind="stable")[:5]  # user 1's top five unrated films
    for j in top:                                                                     # each recommendation, with its reasons
        f = np.flatnonzero(X[j]); parts = sorted(((Wc[a, q], names[q]) for q in f), reverse=True)  # its features and their weights
        print(f"  {titles[j][:38]:<38} {s[j]:+.2f} = " + " ".join(f"{w:+.2f} ({n})" for w, n in parts))
    print(f"  unrated films sharing the top score: {int((s == s[top[0]]).sum())}")  # same features, same score
    print("Actual output from this lesson's run.")


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(near, far, Ec, Eb, byc, cold_rows, rc, top, Wc, names, out):        # six panels
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # two rows of three
    b = np.linspace(0, 1, 21)                                                         # cosine bins
    ax[0, 0].hist(far, bins=b, alpha=0.6, density=True, label="10 random films")      # the control
    ax[0, 0].hist(near, bins=b, alpha=0.6, density=True, label="10 rating neighbours")  # the real neighbours
    ax[0, 0].set_xlabel("content cosine"); ax[0, 0].set_ylabel("density"); ax[0, 0].legend(fontsize=8)  # labels and legend
    ax[0, 0].set_title("rating neighbours share more content than random films", fontsize=10)  # title
    ax[0, 1].semilogx(LAMS, Ec, "o-", label="content only")                           # validation error per penalty
    ax[0, 1].semilogx(LAMS, Eb, "s-", label="bias model + content")                   # the same on the residuals
    ax[0, 1].set_xlabel("profile penalty"); ax[0, 1].set_ylabel("validation RMSE"); ax[0, 1].legend(fontsize=8)  # labels
    ax[0, 1].set_title("choosing the profile penalty (Day 154's split)", fontsize=10)  # title
    xs = np.arange(len(byc)); labs = list(byc)                                        # one point per rating-count bin
    for j, nm in enumerate(("content only", "bias model", "bias + content", "item-based")):  # four models
        ax[0, 2].plot(xs, [byc[k][1][j] for k in labs], "o-", label=nm)               # test RMSE per bin
    ax[0, 2].set_xticks(xs); ax[0, 2].set_xticklabels([f"{k}\n({byc[k][0]})" for k in labs], fontsize=8)  # bin names and sizes
    ax[0, 2].set_xlabel("training ratings of the film (test ratings)"); ax[0, 2].set_ylabel("test RMSE"); ax[0, 2].legend(fontsize=8)  # labels and legend
    ax[0, 2].set_title("bias + content beats the bias model in every bin", fontsize=10)  # title
    nm = list(cold_rows); v = [score(cold_rows[k], rc) for k in nm]                   # cold-start test errors
    ax[1, 0].barh(range(len(nm)), v, color=["grey", "C0", "C1", "C2", "C3"])          # one bar per model
    ax[1, 0].set_yticks(range(len(nm))); ax[1, 0].set_yticklabels([k.split(" (")[0] for k in nm], fontsize=8)  # names
    ax[1, 0].invert_yaxis(); ax[1, 0].set_xlim(min(v) - 0.02, max(v) + 0.012)         # zoom in on the differences
    for y, x in enumerate(v):                                                         # the value at each bar
        ax[1, 0].text(x + 0.001, y, f"{x:.4f}", va="center", fontsize=8)              # written next to it
    ax[1, 0].set_xlabel("test RMSE, films never rated in training"); ax[1, 0].set_title("cold start: only content can help", fontsize=10)  # label and title
    nm = list(top); p = [top[k][0] for k in nm]; dg = [top[k][3] for k in nm]         # top-10 results
    ax[1, 1].bar(np.arange(len(nm)) - 0.2, p, 0.4, label="precision@10")             # accuracy
    a2 = ax[1, 1].twinx(); a2.bar(np.arange(len(nm)) + 0.2, dg, 0.4, color="C1", label="distinct genres in the list")  # breadth
    ax[1, 1].set_xticks(range(len(nm))); ax[1, 1].set_xticklabels(nm, fontsize=8, rotation=15)  # list names
    ax[1, 1].set_ylabel("precision@10"); a2.set_ylabel("distinct genres in a 10-film list")  # both axes
    h1, l1 = ax[1, 1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()  # one legend for both
    ax[1, 1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")                   # top left, clear of the bars
    ax[1, 1].set_title("top-10 lists: accuracy and breadth", fontsize=10)              # title
    o = np.argsort(Wc[0])                                                             # user 1's weights, sorted
    ax[1, 2].barh(range(len(o)), Wc[0][o], color=np.where(Wc[0][o] > 0, "C2", "C3"))  # liked green, disliked red
    ax[1, 2].set_yticks(range(len(o))); ax[1, 2].set_yticklabels([names[j] for j in o], fontsize=7)  # feature names
    ax[1, 2].axvline(0, color="k", lw=0.8); ax[1, 2].set_xlabel("stars above the user's mean")  # zero line and label
    ax[1, 2].set_title("user 1's content profile", fontsize=10)                       # title
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)                     # save to disk


def draw_diagram():                                     # the content-based pipeline
    fig, ax = plt.subplots(figsize=(15, 5.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.2); ax.axis("off")  # a wide, empty canvas

    def box(x, y, w, h, text, fc):                                                    # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec="k", lw=1.2))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)   # its text

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))  # from (x0, y0) to (x1, y1)
    box(0.3, 3.3, 3.0, 1.2, "film attributes\n19 genres + decade\n(u.item)", "#e8f0fe")  # the content
    box(4.4, 3.3, 3.0, 1.2, "film vectors X\n27 numbers per film", "#e8f0fe")          # the features
    box(0.3, 0.7, 3.0, 1.2, "one user's ratings\nminus their mean", "#fde8e8")         # the ratings
    box(4.4, 0.7, 3.0, 1.2, "user profile w\nridge regression,\nshrunk toward everyone's", "#fde8e8")  # the profile
    box(8.6, 2.0, 2.9, 1.2, "score = mean + user bias\n+ w . x(film)", "#e6f4ea")       # the prediction
    box(12.4, 2.0, 2.3, 1.2, "top-10 list:\nhighest scores,\nunrated films", "#e6f4ea")  # the list
    p = 0.08                                                                          # the boxes' rounded padding, so arrows meet the drawn edges
    arrow(3.3 + p, 3.9, 4.4 - p, 3.9); arrow(3.3 + p, 1.3, 4.4 - p, 1.3)              # along each row
    arrow(5.9, 3.3 - p, 5.9, 1.9 + p)                                                 # the rated films' vectors feed the regression
    arrow(7.4 + p, 3.9, 8.6 - p, 2.9); arrow(7.4 + p, 1.3, 8.6 - p, 2.3)              # into the score
    arrow(11.5 + p, 2.6, 12.4 - p, 2.6)                                               # then the list
    ax.text(8.3, 4.6, "a brand-new film has a vector the moment it is listed", fontsize=9.5, style="italic")  # the cold-start point
    ax.text(8.3, 0.5, "other users' ratings enter only through the shared profile it is shrunk toward", fontsize=9.5, style="italic")  # what else is used
    ax.set_title("Day 156: content-based filtering, from film attributes and one user's ratings", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "content_based_diagram.png"), dpi=120); plt.close(fig)  # save to disk


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 157: hybrid recommenders. Content places new films; ratings-based models are more accurate once")
    print("films have ratings. Hybrids combine them: blending predictions, switching by how much is known, or")
    print("putting content features inside the factor model itself.")


def main():                                             # runs every section in order
    u, i, r, titles, genres, gnames, years, X, nu, ni, tr, te, fi, va = setup()       # Day 154's data and split, plus content
    near, far = demo_content(u, i, r, titles, genres, gnames, years, X, nu, ni, tr)   # 1. content
    Wc, Wb, B, mu, bu, g, lc, Ec, Eb, rows, byc = demo_profiles(u, i, r, X, nu, ni, tr, te, fi, va)  # 2. profiles
    cold_rows, rc, Ecold, Ek = demo_cold(u, i, r, X, nu, ni, g)                       # 3. cold start
    top = demo_topn(u, i, r, titles, genres, X, nu, ni, tr, te, Wc, Wb, B)            # 4. top-10 lists
    R, M = d154.to_matrix(u[tr], i[tr], r[tr], nu, ni)                                # what user 1 has rated
    a, names = demo_explain(titles, genres, gnames, X, Wc, M.sum(0))                  # 5. one user's profile
    explain_list(a, titles, genres, gnames, X, Wc, M, names)                          # and their top five
    plot_results(near, far, Ec, Eb, byc, cold_rows, rc, top, Wc, names, os.path.join(HERE, "content_based_results.png"))  # the results figure
    draw_diagram()                                                                    # the method
    print("\nsaved content_based_results.png\nsaved content_based_diagram.png")
    note_on_what_comes_next()                                                         # a preview of Day 157


if __name__ == "__main__":
    main()                                                                            # run everything
