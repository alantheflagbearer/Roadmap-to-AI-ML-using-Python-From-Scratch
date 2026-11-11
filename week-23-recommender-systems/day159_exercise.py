"""
Day 159 Exercise -- Evaluating Recommenders

The sixth of Week 23's parts for Day 160's neural recommender.

    per_user_metrics(S, seen, gain, k)      precision, recall, AP, NDCG and reciprocal rank, per user
    beyond_accuracy(top, pop, nu)           coverage, median popularity, novelty, personalisation
    paired_bootstrap(a, b)                  mean difference, 95% interval, share of resamples with no gain
    leave_last_out(u, t)                    each user's latest rating as the test set
    sampled_eval(S, seen, item)             hit rate and NDCG with 100 sampled negatives

Checked by hand: one list scored five ways, three tiny lists for the
beyond-accuracy measures, a bootstrap with a known answer, a leave-last-out
split with a tie, and the sampled shortcut at its best and worst.
"""
import numpy as np                                     # numpy: every array operation below is built on this


def per_user_metrics(S, seen, gain, k=10):              # NEW: Day 159's version, unchanged
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


def beyond_accuracy(top, pop, nu):                      # NEW: Day 159's version, unchanged
    L = np.zeros((len(top), len(pop))); L[np.arange(len(top))[:, None], top] = 1.0    # users x films: 1 where listed
    ov = L @ L.T / top.shape[1]; n = len(top)                                         # share of list in common, every pair of users
    return dict(coverage=len(np.unique(top)) / len(pop),                              # share of the catalogue recommended to anyone
                median_pop=float(np.median(pop[top])),                                # watchers of the typical listed film
                novelty=float(np.mean(-np.log2(np.maximum(pop[top], 1) / nu))),       # bits of surprise: rarer films score higher
                personal=1.0 - (ov.sum() - np.trace(ov)) / (n * (n - 1)))             # 1 - average overlap between two users' lists


def paired_bootstrap(a, b, B=2000, seed=0):             # NEW: Day 159's version, unchanged
    d = a - b; rng = np.random.RandomState(seed)                                      # per-user differences; seeded
    m = d[rng.randint(0, len(d), (B, len(d)))].mean(1)                                # B resamples of the users, each one's mean difference
    return d.mean(), np.percentile(m, [2.5, 97.5]), (m <= 0).mean(), m                # mean, 95% interval, share not above 0, all of them


def leave_last_out(u, t):                               # NEW: Day 159's version, unchanged
    order = np.lexsort((np.arange(len(t)), t, u))                                     # by user, then time, then position
    last = order[np.r_[np.flatnonzero(np.diff(u[order])), len(u) - 1]]               # the final entry of each user's block
    keep = np.ones(len(u), bool); keep[last] = False                                  # everything else is training
    return np.flatnonzero(keep), last                                                 # training rows, test rows (one per user)


def sampled_eval(S, seen, item, n_neg=100, k=10, seed=0):  # NEW: Day 159's version, unchanged
    rng = np.random.RandomState(seed); hr, nd = [], []                                # seeded sampling; per-user results
    for a in range(len(item)):                                                        # one user at a time
        cand = np.flatnonzero(~seen[a]); cand = cand[cand != item[a]]                 # films the user has not watched, minus the target
        neg = rng.choice(cand, n_neg, replace=False)                                  # the sampled negatives
        rank = (S[a, neg] >= S[a, item[a]]).sum()                                     # how many beat or tie it: ties count against it
        hr.append(rank < k); nd.append(1.0 / np.log2(rank + 2) if rank < k else 0.0)  # a hit in the top k, and its NDCG
    return np.mean(hr), np.mean(nd)                                                   # hit rate@k and NDCG@k


if __name__ == "__main__":                              # the checks run only when this file is run, not when Day 160 imports it
    S = np.array([[9.0, 8.0, 7.0, 6.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])             # user 0's list and user 1's
    seen = np.array([[False, True, False, False, False], [False] * 5])                # user 0 has already watched film 1
    G = np.array([[0.0, 0.0, 1.0, 0.0, 1.0], [0.0] * 5])                              # user 0 must find films 2 and 4; user 1 nothing
    m, top = per_user_metrics(S, seen, G, k=3)                                        # top 3
    print(f"user 0's top 3 skips watched film 1: {top[0].tolist()} (films 0, 2, 3); user 1 has nothing to find, so only "
          f"{len(m['precision'])} user is scored")
    print("  by hand: hit only at place 2; precision 1/3 = 0.333, recall 1/2 = 0.5, AP (1/2)/2 = 0.25, "
          "NDCG 0.631/(1 + 0.631) = 0.387, first film to find is film 2 at place 2, so RR 0.5")
    print(f"  per_user_metrics: precision {m['precision'][0]:.3f}, recall {m['recall'][0]:.3f}, AP {m['map'][0]:.3f}, "
          f"NDCG {m['ndcg'][0]:.3f}, RR {m['mrr'][0]:.3f}")
    b = beyond_accuracy(np.array([[0, 1], [0, 1], [2, 3]]), np.array([8, 4, 2, 1, 1]), 8)  # three users' lists of 2, five films, 8 users
    print("beyond_accuracy, lists (0, 1), (0, 1), (2, 3); watchers 8, 4, 2, 1, 1 out of 8 users:")
    print("  by hand: coverage 4/5 = 0.8; listed films' watchers 8, 4, 8, 4, 2, 1, median 4; novelty -log2(watchers/8) "
          "= 0, 1, 0, 1, 2, 3, mean 1.167; overlaps 1 (first two users) and 0 twice, so personalisation 1 - 1/3 = 0.667")
    print(f"  beyond_accuracy: coverage {b['coverage']:.3f}, median {b['median_pop']:.0f}, novelty {b['novelty']:.3f}, "
          f"personalisation {b['personal']:.3f}")
    a = np.random.RandomState(0).rand(200)                                            # 200 users' scores
    d0, ci0, p0, _ = paired_bootstrap(a, a); d1, ci1, p1, _ = paired_bootstrap(a + 0.1, a)  # identical, and 0.1 better for everyone
    print(f"paired_bootstrap: a model against itself {d0:+.3f} [{ci0[0]:+.3f}, {ci0[1]:+.3f}], no-gain share {p0:.0%}; "
          f"0.1 better for every user {d1:+.3f} [{ci1[0]:+.3f}, {ci1[1]:+.3f}], no-gain share {p1:.0%}")
    tr, te = leave_last_out(np.array([0, 0, 1, 1, 1]), np.array([5, 3, 1, 9, 9]))   # user 1's last two ratings tie in time
    print(f"leave_last_out, users 0, 0, 1, 1, 1 at times 5, 3, 1, 9, 9: test rows {te.tolist()} (row 0, and row 4: "
          f"a tie goes to the later row), training rows {tr.tolist()}")
    Sb = np.arange(20, dtype=float)[None, :]; seen = np.zeros((1, 20), bool)          # one user, 20 films scored 0 to 19
    hi, lo = sampled_eval(Sb, seen, np.array([19]), n_neg=10), sampled_eval(Sb, seen, np.array([0]), n_neg=10)  # best and worst case
    print(f"sampled_eval with 10 negatives: target scored highest (film 19): hit rate {hi[0]:.1f}, NDCG {hi[1]:.1f}; "
          f"lowest (film 0): hit rate {lo[0]:.1f}, NDCG {lo[1]:.1f}")
