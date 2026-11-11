"""
Day 159: Evaluating Recommenders -- Precision, Recall, MAP, NDCG, and How Sure

All week, top-10 lists have been judged by one number, precision@10. Today the
measures themselves are the subject. Each is built from scratch in NumPy and
applied to the week's models on Day 154's MovieLens split, importing the
earlier days' parts:

  1. Four ways to score one list: precision, recall, average precision, NDCG
     (and reciprocal rank), worked by hand on a toy list.
  2. The week's models on every measure: do the measures agree?
  3. Beyond accuracy: how much of the catalogue is recommended, how popular
     and how alike the lists are, and how well models find less popular films.
  4. How sure is a difference? A paired bootstrap over users, a retrained
     model with a different seed, and fresh random splits, applied to Day
     158's small "stars as confidence" gain.
  5. The sampled-negatives shortcut (rank the held-out film among 100 random
     ones), used by the neural collaborative filtering paper of Day 160: does
     it tell the same story as ranking every film?

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
    import day154_exercise as d154                      # Day 154: data, split, ratings table
    import day155_exercise as d155                      # Day 155: low_rank and explicit ALS
    import day158_exercise as d158                      # Day 158: implicit ALS and co-occurrence

IALS = dict(k=50, lam=30, alpha=3)                      # Day 158's validation choice for implicit ALS
STAR_ALPHA = 0.6                                        # Day 158's validation choice for stars as confidence
SVD_K = 20                                              # Day 158's PureSVD rank
K = 10                                                  # list length throughout


# =============================================================================
# THE PARTS (copied unchanged into day159_exercise.py)
# =============================================================================
def per_user_metrics(S, seen, gain, k=10):              # NEW: precision, recall, AP, NDCG and reciprocal rank, one value per user
    S = np.where(seen, -np.inf, S); held = gain > 0                                   # seen films never listed; held = films to find
    who = np.flatnonzero(held.sum(1) > 0)                                             # only users with something to find
    order = np.argsort(-S, axis=1, kind="stable")                                     # every film, best score first
    top = order[:, :k]; hit = np.take_along_axis(held, top, 1)                        # the lists, and which places are hits
    disc = 1.0 / np.log2(np.arange(2, k + 2))                                         # NDCG's discount: 1, 0.63, 0.5, ... by place
    dcg = (np.take_along_axis(gain, top, 1) * disc).sum(1)                            # gain earned, discounted by place
    idcg = (-np.sort(-gain, axis=1)[:, :k] * disc).sum(1)                             # the best possible: biggest gains first
    nrel = held.sum(1)                                                                # films to find, per user
    ap = ((np.cumsum(hit, 1) / np.arange(1, k + 1)) * hit).sum(1) / np.maximum(np.minimum(nrel, k), 1)  # precision at each hit, averaged
    pos = np.empty_like(order); np.put_along_axis(pos, order, np.arange(S.shape[1])[None, :].repeat(len(S), 0), 1)  # each film's place
    rr = 1.0 / (np.where(held, pos, S.shape[1]).min(1) + 1)                           # 1 / place of the first film to find (whole list)
    return dict(precision=hit.sum(1)[who] / k, recall=(hit.sum(1) / np.maximum(nrel, 1))[who], map=ap[who],
                ndcg=(dcg / np.maximum(idcg, 1e-12))[who], mrr=rr[who]), top[who]    # per-user values, and the lists


def beyond_accuracy(top, pop, nu):                      # NEW: coverage, median popularity, novelty and personalisation of the lists
    L = np.zeros((len(top), len(pop))); L[np.arange(len(top))[:, None], top] = 1.0    # users x films: 1 where listed
    ov = L @ L.T / top.shape[1]; n = len(top)                                         # share of list in common, every pair of users
    return dict(coverage=len(np.unique(top)) / len(pop),                              # share of the catalogue recommended to anyone
                median_pop=float(np.median(pop[top])),                                # watchers of the typical listed film
                novelty=float(np.mean(-np.log2(np.maximum(pop[top], 1) / nu))),       # bits of surprise: rarer films score higher
                personal=1.0 - (ov.sum() - np.trace(ov)) / (n * (n - 1)))             # 1 - average overlap between two users' lists


def paired_bootstrap(a, b, B=2000, seed=0):             # NEW: how sure is mean(a) > mean(b), resampling users
    d = a - b; rng = np.random.RandomState(seed)                                      # per-user differences; seeded
    m = d[rng.randint(0, len(d), (B, len(d)))].mean(1)                                # B resamples of the users, each one's mean difference
    return d.mean(), np.percentile(m, [2.5, 97.5]), (m <= 0).mean(), m                # mean, 95% interval, share not above 0, all of them


def leave_last_out(u, t):                               # NEW: each user's latest rating held out (the NCF paper's split)
    order = np.lexsort((np.arange(len(t)), t, u))                                     # by user, then time, then position
    last = order[np.r_[np.flatnonzero(np.diff(u[order])), len(u) - 1]]               # the final entry of each user's block
    keep = np.ones(len(u), bool); keep[last] = False                                  # everything else is training
    return np.flatnonzero(keep), last                                                 # training rows, test rows (one per user)


def sampled_eval(S, seen, item, n_neg=100, k=10, seed=0):  # NEW: rank the held-out film among n_neg random unwatched films
    rng = np.random.RandomState(seed); hr, nd = [], []                                # seeded sampling; per-user results
    for a in range(len(item)):                                                        # one user at a time
        cand = np.flatnonzero(~seen[a]); cand = cand[cand != item[a]]                 # films the user has not watched, minus the target
        neg = rng.choice(cand, n_neg, replace=False)                                  # the sampled negatives
        rank = (S[a, neg] >= S[a, item[a]]).sum()                                     # how many beat or tie it: ties count against it
        hr.append(rank < k); nd.append(1.0 / np.log2(rank + 2) if rank < k else 0.0)  # a hit in the top k, and its NDCG
    return np.mean(hr), np.mean(nd)                                                   # hit rate@k and NDCG@k


# =============================================================================
# HELPERS FOR THE EXPERIMENTS
# =============================================================================
def setup():                                            # Day 154's data and split
    u, i, r, t = d154.load_movielens()                                                # the ratings
    nu, ni = u.max() + 1, i.max() + 1                                                 # users and films, counted from 0
    tr, te = d154.split_ratings(len(r), 0.2, seed=0)                                  # Day 154's 80/20 split
    return u, i, r, t, nu, ni, tr, te                                                 # everything the experiments need


def models(u, i, r, idx, nu, ni, seed=0):               # every model's scores from one set of training rows
    M = np.zeros((nu, ni), bool); M[u[idx], i[idx]] = True; B = M.astype(float)       # watched, as True / False and 0 / 1
    R, Mr = d154.to_matrix(u[idx], i[idx], r[idx], nu, ni)                            # the stars, for the explicit model and confidence
    Fe, _ = d155.als_mf(R, Mr, 100, 0.125)                                            # Day 155's explicit ALS, its chosen setting
    X, Y, _ = d158.implicit_als(np.where(M, 1.0 + IALS["alpha"], 0.0), IALS["k"], IALS["lam"], seed=seed)  # implicit ALS, binary
    Xs, Ys, _ = d158.implicit_als(np.where(M, 1.0 + STAR_ALPHA * R, 0.0), IALS["k"], IALS["lam"], seed=seed)  # stars as confidence
    return M, {"popularity": np.broadcast_to(B.sum(0), M.shape),                      # how many watched each film
               "random": np.random.RandomState(seed).rand(nu, ni),                    # random scores
               "co-occurrence": B @ d158.cooccurrence(B),                             # similarity to the films watched
               "PureSVD": d155.low_rank(B, SVD_K),                                    # best rank-20 fit of the 0/1 table
               "explicit ALS": Fe["mu"] + Fe["bu"][:, None] + Fe["bi"][None, :] + Fe["P"] @ Fe["Q"].T,  # predicted stars
               "implicit ALS": X @ Y.T,                                               # Day 158's model
               "implicit ALS, stars": Xs @ Ys.T}                                      # with stars as confidence


def held(u, i, r, idx, nu, ni, graded):                 # the gain of every test film: 1, or 2^stars - 1
    G = np.zeros((nu, ni)); G[u[idx], i[idx]] = 2.0 ** r[idx] - 1 if graded else 1.0  # 0 where nothing is to be found
    return G                                                                          # users x films


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_by_hand():                                                                   # section 1
    print("\n" + "=" * 70)
    print("1. FOUR WAYS TO SCORE ONE LIST")
    print("=" * 70)
    seen = np.zeros((2, 12), bool); G = np.zeros((2, 12)); G[:, [0, 1, 10]] = 1.0     # two users, 12 films; films 0, 1 and 10 to find
    S = np.array([np.r_[12, 11, np.arange(10, 0, -1)], np.r_[4, 3, np.arange(12, 4, -1), 2, 1]], float)  # A ranks 0 and 1 first; B ninth and tenth
    m, top = per_user_metrics(S, seen, G, k=10)                                       # every measure, per user
    print("3 films to find (0, 1 and 10); top-10 list A has films 0 and 1 in places 1-2, list B in places 9-10:")
    print(f"  list A: {top[0].tolist()}\n  list B: {top[1].tolist()}")
    for q, nm in enumerate("AB"):                                                     # one row per list
        print(f"  list {nm}: precision {m['precision'][q]:.3f}  recall {m['recall'][q]:.3f}  AP {m['map'][q]:.3f}  "
              f"NDCG {m['ndcg'][q]:.3f}  reciprocal rank {m['mrr'][q]:.3f}")
    print("by hand, A: precision 2/10; recall 2/3; AP (1/1 + 2/2)/3 = 0.667; NDCG (1 + 0.631)/(1 + 0.631 + 0.5) = 0.765; RR 1")
    print("by hand, B: precision 2/10; recall 2/3; AP (1/9 + 2/10)/3 = 0.104; NDCG (0.301 + 0.289)/2.131 = 0.277; RR 1/9")
    G2 = G.copy(); G2[:, 0] = 31.0                                                    # graded: film 0 rated 5 stars (gain 31), films 1 and 10 1 star (gain 1)
    S2 = np.array([S[0], S[0][[1, 0] + list(range(2, 12))]])                          # list A, and list A with films 0 and 1 swapped
    g, _ = per_user_metrics(S2, seen, G2, k=10)                                       # both lists, graded
    print(f"graded NDCG (gain 2^stars - 1): 5-star film first {g['ndcg'][0]:.3f}, 1-star film first {g['ndcg'][1]:.3f}; "
          f"precision, recall and AP cannot tell them apart ({g['map'][0]:.3f} and {g['map'][1]:.3f})")
    print("by hand: ideal 31 + 1 x 0.631 + 1 x 0.5 = 32.131; 5-star first (31 + 0.631)/32.131 = 0.984; "
          "1-star first (1 + 31 x 0.631)/32.131 = 0.640")
    print("Actual output from this lesson's run.")


def demo_models(u, i, r, nu, ni, tr, te):                                             # section 2
    print("\n" + "=" * 70)
    print("2. THE WEEK'S MODELS ON EVERY MEASURE")
    print("=" * 70)
    t0 = time.time(); M, sc = models(u, i, r, tr, nu, ni)                             # fit everything on Day 154's training split
    Gb, Gg = held(u, i, r, te, nu, ni, False), held(u, i, r, te, nu, ni, True)        # watched test films, and their graded gains
    print(f"7 models fitted on Day 154's training split ({time.time() - t0:.0f} s); test: every user's held-out films, every film ranked")
    print(f"{'@10':>20} {'precision':>10} {'recall':>8} {'MAP':>7} {'NDCG':>7} {'graded NDCG':>12} {'MRR':>7}")
    res, tops, per = {}, {}, {}                                                       # means, lists and per-user values
    for nm, S in sc.items():                                                          # one row per model
        m, top = per_user_metrics(S, M, Gb); mg, _ = per_user_metrics(S, M, Gg)      # watched, and graded by stars
        res[nm] = [m["precision"].mean(), m["recall"].mean(), m["map"].mean(), m["ndcg"].mean(), mg["ndcg"].mean(), m["mrr"].mean()]
        tops[nm], per[nm] = top, mg["ndcg"]                                           # kept for sections 3 and 4
        print(f"{nm:>20} " + " ".join(f"{x:>{w}.4f}" for x, w in zip(res[nm], (10, 8, 7, 7, 12, 7))))
    names = list(res)                                                                 # model order
    for c, mn in enumerate(("precision", "recall", "MAP", "NDCG", "graded NDCG", "MRR")):  # the models' order by each measure
        print(f"  order by {mn:>11}: " + " > ".join(names[q] for q in np.argsort([-res[x][c] for x in names], kind="stable")))
    print("Actual output from this lesson's run.")
    return M, sc, res, tops, per, Gb                                                  # for the later sections


def demo_beyond(u, i, r, nu, ni, tr, te, M, sc, tops, Gb):                            # section 3
    print("\n" + "=" * 70)
    print("3. BEYOND ACCURACY: COVERAGE, POPULARITY, NOVELTY, PERSONALISATION")
    print("=" * 70)
    pop = M.sum(0); head = pop >= np.sort(pop)[-int(0.1 * len(pop))]                  # watchers per film; the 10% most watched
    print(f"the {head.sum()} most-watched films (10% of {len(pop)}) hold {Gb[:, head].sum() / Gb.sum():.1%} of the held-out test watches")
    print(f"{'':>20} {'coverage':>9} {'median watchers':>16} {'novelty (bits)':>15} {'personalisation':>16} {'tail recall@10':>15}")
    Gt = Gb * ~head[None, :]                                                          # only the less popular films count
    out = {}                                                                          # for the figure
    for nm, S in sc.items():                                                          # one row per model
        b = beyond_accuracy(tops[nm], pop, nu)                                        # the four list measures
        tr10 = per_user_metrics(S, M, Gt)[0]["recall"].mean()                         # recall of less popular films only
        out[nm] = (b, tr10)                                                           # kept
        print(f"{nm:>20} {b['coverage']:>9.3f} {b['median_pop']:>16.0f} {b['novelty']:>15.2f} {b['personal']:>16.3f} {tr10:>15.4f}")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_sure(u, i, r, nu, ni, tr, te, M, per):                                      # section 4
    print("\n" + "=" * 70)
    print("4. HOW SURE IS A DIFFERENCE?")
    print("=" * 70)
    Gg = held(u, i, r, te, nu, ni, True)                                              # graded gains on Day 154's test
    X, Y, _ = d158.implicit_als(np.where(M, 1.0 + IALS["alpha"], 0.0), IALS["k"], IALS["lam"], seed=1)  # the same model, another seed
    per["implicit ALS, seed 1"] = per_user_metrics(X @ Y.T, M, Gg)[0]["ndcg"]         # its per-user graded NDCG
    print(f"paired bootstrap over {len(per['PureSVD'])} users, 2000 resamples, graded NDCG@10:")
    boots = {}                                                                        # for the figure
    for a, b in (("implicit ALS", "PureSVD"), ("implicit ALS, stars", "implicit ALS"), ("implicit ALS, seed 1", "implicit ALS"),
                 ("co-occurrence", "popularity")):                                    # four comparisons
        d, ci, p, m = paired_bootstrap(per[a], per[b]); boots[f"{a} - {b}"] = m       # the result, and every resample
        print(f"  {a:>20} - {b:<13}: {d:+.4f}, 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}], resamples with no gain {p:.1%}")
    print("fresh random 80/20 splits: graded NDCG@10, stars as confidence against binary (Day 158's settings, not re-tuned):")
    diffs = []                                                                        # one difference per split
    for s in (10, 11, 12, 13, 14):                                                    # five new splits
        tr2, te2 = d154.split_ratings(len(r), 0.2, seed=s)                            # a fresh split
        M2 = np.zeros((nu, ni), bool); M2[u[tr2], i[tr2]] = True                      # its training interactions
        R2, _ = d154.to_matrix(u[tr2], i[tr2], r[tr2], nu, ni); G2 = held(u, i, r, te2, nu, ni, True)  # its stars and test gains
        X, Y, _ = d158.implicit_als(np.where(M2, 1.0 + IALS["alpha"], 0.0), IALS["k"], IALS["lam"])  # binary
        Xs, Ys, _ = d158.implicit_als(np.where(M2, 1.0 + STAR_ALPHA * R2, 0.0), IALS["k"], IALS["lam"])  # stars
        nb, ns = per_user_metrics(X @ Y.T, M2, G2)[0]["ndcg"].mean(), per_user_metrics(Xs @ Ys.T, M2, G2)[0]["ndcg"].mean()  # both
        diffs.append(ns - nb)                                                         # stars minus binary
        print(f"  split seed {s}: binary {nb:.4f}, stars {ns:.4f}, difference {ns - nb:+.4f}")
    print(f"  mean difference {np.mean(diffs):+.4f}, smallest {min(diffs):+.4f}, largest {max(diffs):+.4f}")
    print("Actual output from this lesson's run.")
    return boots, diffs                                                               # for the figure


def demo_sampled(u, i, r, t, nu, ni):                                                 # section 5
    print("\n" + "=" * 70)
    print("5. THE SAMPLED-NEGATIVES SHORTCUT")
    print("=" * 70)
    trl, tel = leave_last_out(u, t)                                                   # each user's latest rating held out
    t0 = time.time(); M, sc = models(u, i, r, trl, nu, ni)                            # every model refitted without them
    item = np.zeros(nu, int); item[u[tel]] = i[tel]                                   # each user's held-out film
    G = np.zeros((nu, ni)); G[np.arange(nu), item] = 1.0                              # exactly one film to find per user
    print(f"leave-last-out: {len(tel)} held-out films, one per user, models refitted ({time.time() - t0:.0f} s)")
    print(f"{'':>20} {'full ranking: HR@10':>20} {'NDCG@10':>8}  {'100 sampled: HR@10':>19} {'NDCG@10':>8}  {'other negatives: HR@10':>23}")
    out = {}                                                                          # for the figure
    for nm, S in sc.items():                                                          # one row per model
        m = per_user_metrics(S, M, G)[0]; hs, ns = sampled_eval(S, M, item); h2, _ = sampled_eval(S, M, item, seed=1)  # full and sampled
        out[nm] = (m["recall"].mean(), hs)                                            # hit rate both ways
        print(f"{nm:>20} {m['recall'].mean():>20.4f} {m['ndcg'].mean():>8.4f}  {hs:>19.4f} {ns:>8.4f}  {h2:>23.4f}")
    print(f"  expected for random scores: full ranking 10/{ni - 1} = {10 / (ni - 1):.4f} (roughly), sampled 10/101 = {10 / 101:.4f}")
    other = [sampled_eval(np.random.RandomState(s).rand(nu, ni), M, item)[0] for s in (1, 2, 3, 4, 5)]  # fresh random scores, same negatives
    print(f"  check: five other draws of random scores, sampled HR@10 {', '.join(f'{x:.4f}' for x in other)} (mean {np.mean(other):.4f})")
    names = list(out)                                                                 # model order
    for c, mn in enumerate(("full ranking", "sampled")):                            # the models' order both ways
        print(f"  order by HR@10, {mn:>12}: " + " > ".join(names[q] for q in np.argsort([-out[x][c] for x in names], kind="stable")))
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(res, beyond, boots, diffs, samp, path):                              # six panels
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # two rows of three
    names = list(res); A = np.array([res[n] for n in names])                          # models x measures
    im = ax[0, 0].imshow(A / A.max(0), cmap="viridis", aspect="auto")                 # each measure as a share of its best model
    ax[0, 0].set_xticks(range(6)); ax[0, 0].set_xticklabels(["prec.", "recall", "MAP", "NDCG", "graded\nNDCG", "MRR"], fontsize=8)  # measures
    ax[0, 0].set_yticks(range(len(names))); ax[0, 0].set_yticklabels(names, fontsize=8)  # models
    for a in range(len(names)):                                                       # each cell's value
        for c in range(6):                                                            # each measure
            ax[0, 0].text(c, a, f"{A[a, c]:.3f}", ha="center", va="center", fontsize=7, color="w" if A[a, c] < 0.6 * A[:, c].max() else "k")  # white text on dark cells
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046); ax[0, 0].set_title("test @10, colour = share of the best model", fontsize=10)  # colour scale
    off = {"implicit ALS": (6, -12), "implicit ALS, stars": (6, 4), "PureSVD": (-48, -4)}  # label offsets where points crowd together
    for n in names:                                                                   # popularity of the lists against precision
        ax[0, 1].scatter(beyond[n][0]["median_pop"], res[n][0], s=40)                 # one point per model
        ax[0, 1].annotate(n, (beyond[n][0]["median_pop"], res[n][0]), fontsize=7, xytext=off.get(n, (4, 4)), textcoords="offset points")  # its name
    ax[0, 1].set_xlabel("median watchers of listed films"); ax[0, 1].set_ylabel("precision@10")  # labels
    ax[0, 1].set_title("accuracy against popularity of the lists", fontsize=10)       # title
    xs = np.arange(len(names))                                                        # one group per model
    ax[0, 2].bar(xs - 0.2, [beyond[n][0]["coverage"] for n in names], 0.4, label="coverage")  # share of catalogue listed
    ax[0, 2].bar(xs + 0.2, [beyond[n][0]["personal"] for n in names], 0.4, label="personalisation")  # how different users' lists are
    ax[0, 2].set_xticks(xs); ax[0, 2].set_xticklabels(names, fontsize=7, rotation=25, ha="right"); ax[0, 2].legend(fontsize=8)  # names
    ax[0, 2].set_title("beyond accuracy", fontsize=10)                                # title
    for q, (k_, m) in enumerate(boots.items()):                                       # each comparison's resampled differences
        ax[1, 0].hist(m, bins=40, alpha=0.6, label=k_.replace("implicit ALS", "iALS"))  # one histogram each
    ax[1, 0].axvline(0, color="k", lw=0.8); ax[1, 0].legend(fontsize=7); ax[1, 0].set_xlabel("difference in graded NDCG@10")  # zero line
    ax[1, 0].set_title("paired bootstrap over users", fontsize=10)                    # title
    ax[1, 1].bar(range(len(diffs)), diffs, color="C2"); ax[1, 1].axhline(0, color="k", lw=0.8)  # one bar per split
    ax[1, 1].set_xticks(range(len(diffs))); ax[1, 1].set_xticklabels([f"seed {s}" for s in (10, 11, 12, 13, 14)], fontsize=8)  # splits
    ax[1, 1].set_ylabel("stars minus binary, graded NDCG@10"); ax[1, 1].set_title("stars as confidence on five fresh splits", fontsize=10)  # label and title
    sn = list(samp); xs = np.arange(len(sn))                                          # models of section 5
    ax[1, 2].bar(xs - 0.2, [samp[n][0] for n in sn], 0.4, label="all films ranked")   # full ranking
    ax[1, 2].bar(xs + 0.2, [samp[n][1] for n in sn], 0.4, label="100 sampled negatives")  # the shortcut
    ax[1, 2].set_xticks(xs); ax[1, 2].set_xticklabels(sn, fontsize=7, rotation=25, ha="right"); ax[1, 2].legend(fontsize=8)  # names
    ax[1, 2].set_ylabel("hit rate@10, leave-last-out"); ax[1, 2].set_title("full ranking against sampled negatives", fontsize=10)  # label and title
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)                    # save to disk


def draw_diagram():                                     # how a list is scored
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.4); ax.axis("off")  # a wide, empty canvas
    p = 0.08                                                                          # the boxes' rounded padding

    def box(x, y, w, h, text, fc):                                                    # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={p}", fc=fc, ec="k", lw=1.2))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)   # its text

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))  # from (x0, y0) to (x1, y1)
    box(0.3, 2.1, 2.4, 1.2, "scores for\nevery film", "#e8f0fe")                       # the model's output
    box(3.6, 2.1, 2.6, 1.2, "drop films already\nwatched; take top 10", "#e8f0fe")     # the list
    box(7.1, 3.7, 3.4, 1.3, "accuracy: precision, recall,\nMAP, NDCG (graded),\nreciprocal rank", "#e6f4ea")  # accuracy measures
    box(7.1, 2.05, 3.4, 1.3, "beyond accuracy: coverage,\npopularity, novelty,\npersonalisation", "#e6f4ea")  # list measures
    box(7.1, 0.4, 3.4, 1.3, "shortcut: rank the held-out\nfilm among 100 random\nunwatched films", "#fde8e8")  # sampled evaluation
    box(11.4, 2.05, 3.3, 1.3, "how sure? bootstrap over\nusers, other seeds,\nfresh splits", "#fff4d6")  # uncertainty
    arrow(2.7 + p, 2.7, 3.6 - p, 2.7)                                                 # scores to list
    arrow(6.2 + p, 2.9, 7.1 - p, 4.3); arrow(6.2 + p, 2.7, 7.1 - p, 2.7)              # list to the measures
    arrow(6.2 + p, 2.5, 7.1 - p, 1.05)                                                # list to the shortcut
    arrow(10.5 + p, 4.3, 11.4 - p, 3.0); arrow(10.5 + p, 2.7, 11.4 - p, 2.7)          # measures to how sure
    ax.set_title("Day 159: scoring a recommender's lists, and how sure the comparison is", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "evaluating_recommenders_diagram.png"), dpi=120); plt.close(fig)  # save to disk


def note_on_what_comes_next():  # a preview of Day 160
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 160: the week's build, neural collaborative filtering. A small network learns from the same")
    print("watched / not watched data, using the week's parts, and is judged with today's measures, both by")
    print("ranking every film and by the sampled shortcut its original paper used.")


def main():                                             # runs every section in order
    u, i, r, t, nu, ni, tr, te = setup()                                              # Day 154's data and split
    demo_by_hand()                                                                    # 1. the measures on one list
    M, sc, res, tops, per, Gb = demo_models(u, i, r, nu, ni, tr, te)                  # 2. every model, every measure
    beyond = demo_beyond(u, i, r, nu, ni, tr, te, M, sc, tops, Gb)                    # 3. beyond accuracy
    boots, diffs = demo_sure(u, i, r, nu, ni, tr, te, M, per)                         # 4. how sure
    samp = demo_sampled(u, i, r, t, nu, ni)                                           # 5. the sampled shortcut
    plot_results(res, beyond, boots, diffs, samp, os.path.join(HERE, "evaluating_recommenders_results.png"))  # the results figure
    draw_diagram()                                                                    # the method
    print("\nsaved evaluating_recommenders_results.png\nsaved evaluating_recommenders_diagram.png")
    note_on_what_comes_next()                                                         # a preview of Day 160


if __name__ == "__main__":  # only when run directly, not when imported
    main()                                                                            # run everything
