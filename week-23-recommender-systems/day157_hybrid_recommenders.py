"""
Day 157: Hybrid Recommenders -- Combining Neighbours, Factors and Content

Each model of the week so far has a blind spot. Item-based neighbours (Day 154)
and learned factors (Day 155) are the most accurate for films with ratings but
know nothing about a new film; content profiles (Day 156) can place a new film
but are weaker once ratings exist. A hybrid combines them. Three classic ways
are built from scratch in NumPy on Day 154's MovieLens split, importing the
earlier days' parts:

  1. The ingredients: five models, and whether their mistakes differ.
  2. Weighted hybrid: a linear blend fitted on validation, with two checks.
  3. Switching and feature-weighted hybrids: let the mix depend on how many
     ratings a film has.
  4. Content inside the factor model: one model for old and new films.
  5. Top-10 lists from the hybrids.

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
    import day154_exercise as d154                      # Day 154: data, split, biases, similarity, item-based
    import day155_exercise as d155                      # Day 155: ALS and the factor prediction
    import day156_exercise as d156                      # Day 156: film features, profiles, the cold split

LU, LI = 5, 2                                           # Day 154's bias penalties, chosen there on validation
ITEM = dict(k=40, shrink=800)                           # Day 154's item-based setting, chosen there
ALS = dict(k=100, lam=0.125)                            # Day 155's ALS setting, chosen there
PROF = dict(shared=1, lam=30)                           # Day 156's profile penalties, chosen there
NAMES = ("bias model", "item-based", "ALS factors", "content profiles", "bias + content")  # the five ingredients
THRESHOLDS = (0, 1, 3, 5, 10, 20, 50, 100)              # switching thresholds tried (training ratings of the film)
EPOCHS = 120                                            # passes of SGD for the hybrid factor model


# =============================================================================
# THE PARTS (copied unchanged into day157_exercise.py)
# =============================================================================
def stack_fit(P, y):                                    # NEW: blend weights by least squares: y ~ w0 + P w
    A = np.c_[np.ones(len(y)), P]                                                     # a column of 1s for the intercept
    return np.linalg.lstsq(A, y, rcond=None)[0]                                       # intercept first, then one weight per model


def stack_predict(P, w):                                # NEW: the blend's prediction
    return w[0] + P @ w[1:]                                                           # intercept + weighted sum of the models


def fw_design(P, n):                                    # NEW: feature-weighted stacking: each weight a line in log(1 + film's ratings)
    g = np.log1p(n)[:, None]                                                          # how much is known about the film
    return np.c_[g, P, P * g]                                                         # (intercept added by stack_fit) + slope, weights, weight slopes


def switch_predict(P_rich, P_poor, n, threshold):       # NEW: switching hybrid: the content model below the threshold
    return np.where(n < threshold, P_poor, P_rich)                                    # few ratings: content; enough: the ratings-based blend


def hybrid_mf(u, i, r, X, nu, ni, k, lam, lr=0.01, lr_s=0.1, lam_s=0.001, epochs=60, batch=128, seed=0, track=None):  # NEW: factors with content inside
    rng = np.random.RandomState(seed); f = X.shape[1]                                 # seeded, so a run repeats
    seen = np.bincount(i, minlength=ni) > 0                                           # films with a training rating
    F = dict(mu=r.mean(), P=0.1 * rng.randn(nu, k), Q=0.1 * rng.randn(ni, k) * seen[:, None],  # unseen films' own factors stay 0
             bu=np.zeros(nu), bi=np.zeros(ni), a=np.zeros(f), Z=0.1 * rng.randn(f, k) / np.sqrt(f))  # a: content bias; Z: content to factors
    hist = []                                                                         # measures after each epoch, if asked for
    for _ in range(epochs):                                                           # one pass over the ratings
        o = rng.permutation(len(r))                                                   # a fresh random order
        for s in range(0, len(o), batch):                                             # small batches
            b = o[s:s + batch]; a, c = u[b], i[b]; xc = X[c]                          # the batch's users, films and features
            qe = F["Q"][c] + xc @ F["Z"]                                              # each film's factors: its own + what its content says
            e = r[b] - (F["mu"] + F["bu"][a] + F["bi"][c] + xc @ F["a"] + (F["P"][a] * qe).sum(1))  # errors
            gp = e[:, None] * qe - lam * F["P"][a]                                    # downhill for the user factors
            gq = e[:, None] * F["P"][a] - lam * F["Q"][c]                             # and the films' own factors
            gz = xc.T @ (e[:, None] * F["P"][a]) / len(b) - lam_s * F["Z"]            # shared: averaged over the batch
            ga = xc.T @ e / len(b) - lam_s * F["a"]                                   # the content bias, the same way
            np.add.at(F["bu"], a, lr * (e - lam * F["bu"][a]))                        # user biases (repeats counted)
            np.add.at(F["bi"], c, lr * (e - lam * F["bi"][c]))                        # film biases
            np.add.at(F["P"], a, lr * gp); np.add.at(F["Q"], c, lr * gq)              # and the factors
            F["Z"] += lr_s * gz; F["a"] += lr_s * ga                                  # the shared content weights
        hist.append(track(F) if track else ())                                        # measures after this epoch
    return F, hist                                                                    # the model, and one record per epoch


def hybrid_predict(F, X, u, i):                         # NEW: mean + biases + content bias + user factors . (own + content factors)
    q = F["Q"][i] + X[i] @ F["Z"]                                                     # a new film's own part is 0: content alone
    return F["mu"] + F["bu"][u] + F["bi"][i] + X[i] @ F["a"] + (F["P"][u] * q).sum(1)  # one number per pair


# =============================================================================
# HELPERS FOR THE EXPERIMENTS
# =============================================================================
def setup():                                            # Day 154's data and split, plus Day 156's content
    u, i, r, t = d154.load_movielens(); titles, genres, gnames = d154.load_items()    # the ratings and the films
    nu, ni = u.max() + 1, i.max() + 1                                                 # users and films, counted from 0
    tr, te = d154.split_ratings(len(r), 0.2, seed=0)                                  # Day 154's 80/20 split
    fi, va = d154.split_ratings(len(tr), 0.1, seed=1); fi, va = tr[fi], tr[va]        # and its validation part
    X = d156.film_features(genres, d156.load_years())                                 # Day 156's 27 content flags
    return u, i, r, titles, genres, X, nu, ni, tr, te, fi, va                         # everything the experiments need


def score(p, r):                                        # RMSE of clipped predictions
    return d154.rmse(np.clip(p, 1, 5), r)                                             # kept inside 1-5 stars


def ingredients(u, i, r, X, nu, ni, fit, ev):          # the five models, trained on `fit`, predicting `ev`
    R, M = d154.to_matrix(u[fit], i[fit], r[fit], nu, ni)                             # the training table
    mb, bu, bi = d154.fit_biases(R, M, LU, LI); B = mb + bu[:, None] + bi[None, :]    # Day 154's bias model
    Xr = (R - (R.sum(1) / np.maximum(M.sum(1), 1))[:, None]) * M                      # adjusted cosine's centring
    S, _ = d154.similarity(Xr, M, shrink=ITEM["shrink"])                              # Day 154's item similarity
    c, _ = d154.predict_item_knn((R - B) * M, M, S, u[ev], i[ev], ITEM["k"])          # its correction to the bias model
    F, _ = d155.als_mf(R, M, ALS["k"], ALS["lam"])                                    # Day 155's ALS factors
    mu0, bu0, _ = d154.fit_biases(R, M, LU, 1e12); T = (R - mu0 - bu0[:, None]) * M   # mean + user bias, no film bias
    W = d156.fit_profiles(X, T, M, PROF["lam"], prior=d156.fit_shared_profile(X, T, M, PROF["shared"]))  # content profiles
    Tb = (R - B) * M                                                                  # what the bias model leaves
    Wb = d156.fit_profiles(X, Tb, M, PROF["lam"], prior=d156.fit_shared_profile(X, Tb, M, PROF["shared"]))  # profiles on top of it
    b = B[u[ev], i[ev]]                                                               # the bias model's predictions
    P = np.c_[b, b + c, d155.mf_predict(F, u[ev], i[ev]),                             # one column per model
              mu0 + bu0[u[ev]] + d156.content_predict(W, X, u[ev], i[ev]), b + d156.content_predict(Wb, X, u[ev], i[ev])]
    return P, M.sum(0)[i[ev]], dict(B=B, F=F, W=W, Wb=Wb, M=M, mu0=mu0, bu0=bu0)      # predictions, film counts, the fitted models


def top10(Q, M, liked):                                 # Day 154's precision@10 for users with a liked test film
    Q = np.where(M, -np.inf, Q); who = np.flatnonzero(liked.sum(1) > 0)               # rated films can never be picked
    top = np.argsort(-Q, axis=1, kind="stable")[:, :10]                               # each user's 10 best-scored films
    return np.take_along_axis(liked, top, 1)[who].sum(1).mean() / 10, np.median(M.sum(0)[top[who]])  # precision, median count


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_ingredients(u, i, r, X, nu, ni, tr, te, fi, va):                            # section 1
    print("\n" + "=" * 70)
    print("1. THE INGREDIENTS: FIVE MODELS, AND WHETHER THEIR MISTAKES DIFFER")
    print("=" * 70)
    t0 = time.time()                                                                  # start the clock
    Pv, nv, _ = ingredients(u, i, r, X, nu, ni, fi, va)                               # trained without validation, predicting it
    Pt, nt, fit = ingredients(u, i, r, X, nu, ni, tr, te)                             # trained on all training ratings, predicting test
    print(f"each model trained twice: without the validation ratings (to fit blends) and on all 80,000 ({time.time() - t0:.0f} s)")
    print(f"{'':>18} {'validation':>11} {'test':>8}")
    for j, n in enumerate(NAMES):                                                     # one row per model
        print(f"{n:>18} {score(Pv[:, j], r[va]):>11.4f} {score(Pt[:, j], r[te]):>8.4f}")
    C = np.corrcoef((Pt - r[te][:, None]).T)                                          # how alike their test errors are
    print("correlation of test errors:")
    print(f"{'':>18} " + " ".join(f"{n[:10]:>10}" for n in NAMES))
    for j, n in enumerate(NAMES):                                                     # one row per model
        print(f"{n:>18} " + " ".join(f"{x:>10.3f}" for x in C[j]))
    print("Actual output from this lesson's run.")
    return Pv, nv, Pt, nt, fit, C                                                     # for the later sections


def demo_blend(r, te, va, Pv, Pt):                                                    # section 2
    print("\n" + "=" * 70)
    print("2. WEIGHTED HYBRID: A LINEAR BLEND FITTED ON VALIDATION")
    print("=" * 70)
    sets = ((1, 2), (1, 2, 3), (1, 2, 3, 4), (0, 1, 2, 3, 4))                         # which models go into each blend
    out = {}                                                                          # test RMSE per blend
    print(f"{'blend of':>58} {'test':>7}   weights (intercept first)")
    print(f"{'item-based + ALS, plain average (no fitting)':>58} {score(Pt[:, [1, 2]].mean(1), r[te]):>7.4f}")
    for cols in sets:                                                                 # every blend
        w = stack_fit(Pv[:, cols], r[va])                                               # weights from validation predictions
        out[cols] = score(stack_predict(Pt[:, cols], w), r[te])                       # applied to the test predictions
        print(f"{' + '.join(NAMES[c] for c in cols):>58} {out[cols]:>7.4f}   " + " ".join(f"{x:+.3f}" for x in w))
    h = np.random.RandomState(1).permutation(len(va)); A, Bh = h[:len(h) // 2], h[len(h) // 2:]  # two halves of the validation ratings
    for cols, nm in (((1, 2, 3), "three-model"), ((0, 1, 2, 3, 4), "five-model")):   # checks on two of the blends
        w = stack_fit(Pv[:, cols], r[va])                                             # the blend as fitted above
        rng = np.random.RandomState(0)                                                # control: a column of pure noise
        wn = stack_fit(np.c_[Pv[:, cols], rng.randn(len(va))], r[va])                 # does the blend give it any weight?
        en = score(stack_predict(np.c_[Pt[:, cols], rng.randn(len(te))], wn), r[te])  # and does it hurt?
        wt = stack_fit(Pt[:, cols], r[te])                                            # a leak, for comparison only: weights fitted on test
        el = score(stack_predict(Pt[:, cols], wt), r[te])                             # the optimistic score a leak would report
        ha = score(stack_predict(Pv[Bh][:, cols], stack_fit(Pv[A][:, cols], r[va][A])), r[va][Bh])  # fit on one half, score the other
        print(f"{nm} blend: model weights sum to {w[1:].sum():.3f}; plus a noise column: noise weight {wn[-1]:+.4f}, test {en:.4f}; "
              f"weights fitted on test (a leak): {el:.4f}; fitted on half of validation, scored on the other half: {ha:.4f}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_switch(r, te, va, Pv, nv, Pt, nt):                                           # section 3
    print("\n" + "=" * 70)
    print("3. SWITCHING AND FEATURE-WEIGHTED HYBRIDS: HOW MUCH IS KNOWN ABOUT THE FILM")
    print("=" * 70)
    cols = (1, 2)                                                                     # the ratings-based blend: item-based + ALS
    w = stack_fit(Pv[:, cols], r[va]); Bv, Bt = stack_predict(Pv[:, cols], w), stack_predict(Pt[:, cols], w)  # its predictions
    Ev = [score(switch_predict(Bv, Pv[:, 4], nv, th), r[va]) for th in THRESHOLDS]   # switching to bias + content below each threshold
    th = THRESHOLDS[int(np.argmin(Ev))]                                               # chosen on validation
    print("switching to 'bias + content' for films with fewer training ratings than the threshold (validation RMSE):")
    print("  " + ", ".join(f"{x}: {e:.4f}" for x, e in zip(THRESHOLDS, Ev)))
    print(f"  chosen threshold {th}: test {score(switch_predict(Bt, Pt[:, 4], nt, th), r[te]):.4f} "
          f"(the blend alone {score(Bt, r[te]):.4f}; test films below it: {(nt < th).sum()})")
    cols = (1, 2, 3)                                                                  # feature-weighted: weights move with log(1 + count)
    w = stack_fit(fw_design(Pv[:, cols], nv), r[va])                                  # fitted on validation
    pf = stack_predict(fw_design(Pt[:, cols], nt), w)                                 # on test
    pl = stack_predict(Pt[:, cols], stack_fit(Pv[:, cols], r[va]))                    # the plain blend of the same three
    print(f"feature-weighted blend of item-based, ALS and content profiles: test {score(pf, r[te]):.4f} "
          f"(plain blend {score(pl, r[te]):.4f})")
    g = np.log1p(np.array([1, 10, 100, 400]))                                         # four example films
    ww = w[2:5][None, :] + np.outer(g, w[5:8])                                        # each model's weight at each count
    for n, row in zip((1, 10, 100, 400), ww):                                         # one line per example count
        print(f"  a film with {n:>3} training ratings: weights item-based {row[0]:+.2f}, ALS {row[1]:+.2f}, content {row[2]:+.2f}")
    edges = (0, 6, 21, 51, 151, 10 ** 6); names = ("0-5", "6-20", "21-50", "51-150", "151+")  # bins by film count
    print(f"  {'ratings':>8} {'test ratings':>13} {'plain':>7} {'weighted':>9}")
    byc = []                                                                          # for the figure
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):                              # one row per bin
        q = (nt >= lo) & (nt < hi)                                                    # the test ratings in this bin
        byc.append((nm, score(pl[q], r[te][q]), score(pf[q], r[te][q])))             # both blends' errors
        print(f"  {nm:>8} {q.sum():>13} {byc[-1][1]:>7.4f} {byc[-1][2]:>9.4f}")
    print("Actual output from this lesson's run.")
    return Ev, th, byc                                                                # for the figure


def demo_hybrid_mf(u, i, r, X, nu, ni, tr, te, fi, va):                              # section 4
    print("\n" + "=" * 70)
    print("4. CONTENT INSIDE THE FACTOR MODEL: ONE MODEL FOR OLD AND NEW FILMS")
    print("=" * 70)
    vt = lambda F: (score(hybrid_predict(F, X, u[va], i[va]), r[va]),)               # validation RMSE after each epoch
    t0 = time.time(); F, h = hybrid_mf(u[fi], i[fi], r[fi], X, nu, ni, ALS["k"], ALS["lam"], epochs=EPOCHS, track=vt)  # warm films
    H = np.array([x[0] for x in h]); ep = int(np.argmin(H)) + 1                       # early stopping on validation
    print(f"Day 154's split, {ALS['k']} factors, penalty {ALS['lam']}: best validation {H.min():.4f} at epoch {ep} of {EPOCHS} "
          f"({time.time() - t0:.0f} s); Day 155's SGD without content reached 0.9229 in Day 155's run")
    Fh, _ = hybrid_mf(u[tr], i[tr], r[tr], X, nu, ni, ALS["k"], ALS["lam"], epochs=ep)  # refit on all training ratings
    warm = score(hybrid_predict(Fh, X, u[te], i[te]), r[te])                          # test RMSE
    print(f"  refit on all training ratings for {ep} epochs: test {warm:.4f}")
    cold, ct = d156.cold_split(i, ni, 0.2, seed=2)                                    # Day 156's cold split: 20% of films
    cval, cv = d156.cold_split(i, ni, 0.1, seed=3, exclude=cold)                      # and 10% for validation
    cf = np.flatnonzero(~cold[i] & ~cval[i])                                          # ratings of the other films
    ct2 = lambda F: (score(hybrid_predict(F, X, u[cv], i[cv]), r[cv]),)               # validation on new films
    F, h = hybrid_mf(u[cf], i[cf], r[cf], X, nu, ni, ALS["k"], ALS["lam"], epochs=EPOCHS, track=ct2)  # trained without the new films
    Hc = np.array([x[0] for x in h]); epc = int(np.argmin(Hc)) + 1                    # early stopping on new validation films
    F, _ = hybrid_mf(u[cf], i[cf], r[cf], X, nu, ni, ALS["k"], ALS["lam"], epochs=epc)  # rerun to that epoch
    R, M = d154.to_matrix(u[cf], i[cf], r[cf], nu, ni)                                # the training table
    mu0, bu0, _ = d154.fit_biases(R, M, LU, 1e12); T = (R - mu0 - bu0[:, None]) * M   # Day 156's fallback and target
    W = d156.fit_profiles(X, T, M, 10, prior=d156.fit_shared_profile(X, T, M, 100))   # Day 156's cold-chosen penalties
    Xs = X[np.random.RandomState(0).permutation(ni)]                                  # control: shuffled content
    Fs, _ = hybrid_mf(u[cf], i[cf], r[cf], Xs, nu, ni, ALS["k"], ALS["lam"], epochs=epc)  # the same model, wrong content
    rows = {"mean + user bias (ratings-only fallback)": mu0 + bu0[u[ct]],              # every model's predictions on new films
            "content profiles (Day 156)": mu0 + bu0[u[ct]] + d156.content_predict(W, X, u[ct], i[ct]),
            "factors with content inside": hybrid_predict(F, X, u[ct], i[ct]),
            "control: factors, content shuffled": hybrid_predict(Fs, Xs, u[ct], i[ct])}
    print(f"new films (Day 156's cold split, {len(ct)} test ratings); epochs chosen on new validation films: {epc}")
    for name, p in rows.items():                                                      # one row per model
        print(f"  {name:>42}  {score(p, r[ct]):.4f}")
    print("Actual output from this lesson's run.")
    return Fh, H, Hc, warm, {k: score(p, r[ct]) for k, p in rows.items()}             # for the figure and section 5


def demo_topn(u, i, r, X, nu, ni, te, fit, Fh):                                       # section 5
    print("\n" + "=" * 70)
    print("5. TOP-10 LISTS FROM THE HYBRIDS")
    print("=" * 70)
    M = fit["M"]; liked = np.zeros(M.shape, bool); g = r[te] >= 4; liked[u[te][g], i[te][g]] = True  # test films rated 4-5
    F = fit["F"]; als = F["mu"] + F["bu"][:, None] + F["bi"][None, :] + F["P"] @ F["Q"].T  # ALS predictions everywhere
    bc = fit["B"] + fit["Wb"] @ X.T                                                   # bias + content everywhere
    hyb = (Fh["mu"] + Fh["bu"][:, None] + Fh["bi"][None, :] + (X @ Fh["a"])[None, :]  # factors with content inside, everywhere
           + Fh["P"] @ (Fh["Q"] + X @ Fh["Z"]).T)
    lists = {"ALS factors": als, "bias + content": bc, "average of the two": (als + bc) / 2,  # a simple blend of the two
             "factors with content inside": hyb, "popularity": np.broadcast_to(M.sum(0).astype(float), M.shape)}
    out = {}                                                                          # for the figure
    print(f"{'':>30} {'precision@10':>13} {'median ratings':>15}")
    for name, Q in lists.items():                                                     # one row per list
        out[name] = top10(Q, M, liked)                                                # precision and median count
        print(f"{name:>30} {out[name][0]:>13.4f} {out[name][1]:>15.0f}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(Pv, Pt, r_te, C, blends, Ev, byc, H, Hc, cold, top, out):           # six panels
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # two rows of three
    im = ax[0, 0].imshow(C, vmin=0.85, vmax=1.0, cmap="viridis")                      # the error correlations
    ax[0, 0].set_xticks(range(5)); ax[0, 0].set_xticklabels([n.replace(" ", "\n", 1) for n in NAMES], fontsize=7)  # model names
    ax[0, 0].set_yticks(range(5)); ax[0, 0].set_yticklabels([n.replace(" ", "\n", 1) for n in NAMES], fontsize=7)  # on both axes
    for a in range(5):                                                                # each cell's value
        for b in range(5):
            ax[0, 0].text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=7, color="w" if C[a, b] < 0.95 else "k")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046); ax[0, 0].set_title("correlation of test errors", fontsize=10)  # colour scale, title
    short = ("bias", "item", "ALS", "content", "bias+content")                        # short names for the bar labels
    names = ["item + ALS\n(average)"] + ["\n+ ".join(short[c] for c in k) for k in blends]  # one label per blend
    vals = [score(Pt[:, [1, 2]].mean(1), r_te)] + list(blends.values())               # test RMSE of each
    ax[0, 1].bar(range(len(vals)), vals, color=["grey"] + ["C0"] * len(blends))       # one bar each
    ax[0, 1].axhline(score(Pt[:, 2], r_te), color="C3", ls="--", label="ALS alone")   # the best single model
    ax[0, 1].set_xticks(range(len(vals))); ax[0, 1].set_xticklabels(names, fontsize=7)  # labels
    ax[0, 1].set_ylim(min(vals) - 0.01, score(Pt[:, 2], r_te) + 0.01); ax[0, 1].legend(fontsize=8)  # zoom in on the differences
    for x, v in enumerate(vals):                                                      # the value on each bar
        ax[0, 1].text(x, v + 0.0005, f"{v:.4f}", ha="center", fontsize=7)
    ax[0, 1].set_ylabel("test RMSE"); ax[0, 1].set_title("weighted blends (weights fitted on validation)", fontsize=10)  # labels
    ax[0, 2].plot(range(len(THRESHOLDS)), Ev, "o-")                                   # switching threshold against validation error
    ax[0, 2].set_xticks(range(len(THRESHOLDS))); ax[0, 2].set_xticklabels(THRESHOLDS)  # the thresholds
    ax[0, 2].set_xlabel("switch to content below this many film ratings"); ax[0, 2].set_ylabel("validation RMSE")  # labels
    ax[0, 2].set_title("switching hybrid: choosing the threshold", fontsize=10)        # title
    xs = np.arange(len(byc))                                                          # one group per bin
    ax[1, 0].plot(xs, [b[1] for b in byc], "o-", label="plain blend")                 # plain blend per bin
    ax[1, 0].plot(xs, [b[2] for b in byc], "s-", label="feature-weighted blend")      # feature-weighted per bin
    ax[1, 0].set_xticks(xs); ax[1, 0].set_xticklabels([b[0] for b in byc])            # bin names
    ax[1, 0].set_xlabel("training ratings of the film"); ax[1, 0].set_ylabel("test RMSE"); ax[1, 0].legend(fontsize=8)  # labels
    ax[1, 0].set_title("blends by how much is known about the film", fontsize=10)      # title
    ax[1, 1].plot(np.arange(1, len(H) + 1), H, label="old films (Day 154 split)")     # validation per epoch, warm
    ax[1, 1].plot(np.arange(1, len(Hc) + 1), Hc, label="new films (cold split)")      # and on new films
    ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel("validation RMSE"); ax[1, 1].legend(fontsize=8)  # labels
    ax[1, 1].set_title("factors with content inside: validation per epoch", fontsize=10)  # title
    nm = list(cold); v = [cold[k] for k in nm]                                        # new-film results
    ax[1, 2].barh(range(len(nm)), v, color=["grey", "C1", "C0", "C3"])                # one bar per model
    ax[1, 2].set_yticks(range(len(nm))); ax[1, 2].set_yticklabels([k.split(" (")[0] for k in nm], fontsize=8)  # names
    ax[1, 2].invert_yaxis(); ax[1, 2].set_xlim(min(v) - 0.02, max(v) + 0.015)         # zoom in
    for y, x in enumerate(v):                                                         # the value at each bar
        ax[1, 2].text(x + 0.001, y, f"{x:.4f}", va="center", fontsize=8)              # written next to it
    ax[1, 2].set_xlabel("test RMSE on films never rated in training"); ax[1, 2].set_title("new films", fontsize=10)  # labels
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)                     # save to disk


def draw_diagram():                                     # three ways to combine recommenders
    fig, ax = plt.subplots(figsize=(15, 5.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.6); ax.axis("off")  # a wide, empty canvas
    p = 0.08                                                                          # the boxes' rounded padding

    def box(x, y, w, h, text, fc):                                                    # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={p}", fc=fc, ec="k", lw=1.2))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)   # its text

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))  # from (x0, y0) to (x1, y1)
    box(0.3, 3.9, 2.6, 1.0, "item-based\n(Day 154)", "#e8f0fe"); box(0.3, 2.5, 2.6, 1.0, "ALS factors\n(Day 155)", "#e8f0fe")  # ratings-based
    box(0.3, 1.1, 2.6, 1.0, "content profiles\n(Day 156)", "#fde8e8")                 # content-based
    box(4.4, 3.9, 3.6, 1.0, "weighted: w0 + sum of w x prediction\n(weights fitted on validation)", "#e6f4ea")  # weighted hybrid
    box(4.4, 2.5, 3.6, 1.0, "switching / feature-weighted:\nthe mix depends on the film's count", "#e6f4ea")  # switching hybrid
    box(4.4, 1.1, 3.6, 1.0, "inside the model: film factors =\nown factors + content x Z", "#e6f4ea")  # in-model hybrid
    box(9.6, 2.5, 2.6, 1.0, "prediction", "#fff4d6")                                  # the output
    box(12.6, 2.5, 2.1, 1.0, "top-10 list", "#fff4d6")                                # and the list
    for y in (4.4, 3.0, 1.6):                                                         # each ingredient feeds the hybrids
        for y2 in (4.4, 3.0):                                                         # weighted and switching use all three
            arrow(2.9 + p, y, 4.4 - p, y2)
    arrow(2.9 + p, 1.6, 4.4 - p, 1.6)                                                 # content feeds the in-model hybrid
    arrow(2.9 + p, 3.0, 4.4 - p, 1.8)                                                 # and so do the factors
    for y in (4.4, 3.0, 1.6):                                                         # every hybrid gives a prediction
        arrow(8.0 + p, y, 9.6 - p, 3.0)
    arrow(12.2 + p, 3.0, 12.6 - p, 3.0)                                               # then the list
    ax.text(4.4, 0.45, "the in-model hybrid is one SGD model, trained like Day 155's, so it also works for brand-new films", fontsize=9.5, style="italic")
    ax.set_title("Day 157: three ways to combine neighbours, factors and content", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "hybrid_recommenders_diagram.png"), dpi=120); plt.close(fig)  # save to disk


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 158: implicit feedback and cold start. Most real systems never see star ratings, only who watched")
    print("or clicked what. Tomorrow turns the ratings into watched / not watched, learns from that, and looks at")
    print("the other cold start: a brand-new user.")


def main():                                             # runs every section in order
    u, i, r, titles, genres, X, nu, ni, tr, te, fi, va = setup()                      # Day 154's data and split, Day 156's content
    Pv, nv, Pt, nt, fit, C = demo_ingredients(u, i, r, X, nu, ni, tr, te, fi, va)     # 1. the five models
    blends = demo_blend(r, te, va, Pv, Pt)                                            # 2. weighted hybrid
    Ev, th, byc = demo_switch(r, te, va, Pv, nv, Pt, nt)                              # 3. switching and feature-weighted
    Fh, H, Hc, warm, cold = demo_hybrid_mf(u, i, r, X, nu, ni, tr, te, fi, va)        # 4. content inside the factors
    top = demo_topn(u, i, r, X, nu, ni, te, fit, Fh)                                  # 5. top-10 lists
    plot_results(Pv, Pt, r[te], C, blends, Ev, byc, H, Hc, cold, top, os.path.join(HERE, "hybrid_recommenders_results.png"))  # the figure
    draw_diagram()                                                                    # the method
    print("\nsaved hybrid_recommenders_results.png\nsaved hybrid_recommenders_diagram.png")
    note_on_what_comes_next()                                                         # a preview of Day 158


if __name__ == "__main__":
    main()                                                                            # run everything
