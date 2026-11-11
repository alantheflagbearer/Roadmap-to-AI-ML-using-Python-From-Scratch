"""
Day 160: Neural Collaborative Filtering -- Week 23's Build

The week ends with a neural recommender. Neural collaborative filtering (He,
Liao, Zhang, Nie, Hu and Chua, 2017) replaces the dot product of matrix
factorisation with a small network that learns from watched / not watched
data. It has three variants: GMF (a weighted element-wise product), an MLP
over the two embeddings, and NeuMF (both side by side). Here they are built
from scratch in NumPy, with hand-written backpropagation and Adam, and judged
by everything the week built:

  Day 154  the data, the split and the ratings table
  Day 155  PureSVD (low_rank)
  Day 156  the films' genres, to measure how varied each list is
  Day 157  the weighted blend (stack_fit), to combine NeuMF with implicit ALS
  Day 158  implicit ALS and co-occurrence, the models to beat
  Day 159  every measure, the paired bootstrap, and the paper's own protocol
           (leave-last-out with 100 sampled negatives)

The week's parts are imported from their exercise files, not pasted. This is
the build day, so it has no exercise file of its own. A full run took about
two and a half hours on a laptop CPU (most of it the tuning grids).

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
sys.path.insert(0, HERE)                                # so the week's parts can be imported from here
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day154_exercise as d154                      # Day 154: data, split, ratings table
    import day155_exercise as d155                      # Day 155: low_rank (PureSVD)
    import day156_exercise as d156                      # Day 156: release years and film features (genres)
    import day157_exercise as d157                      # Day 157: the weighted blend
    import day158_exercise as d158                      # Day 158: implicit ALS and co-occurrence
    import day159_exercise as d159                      # Day 159: measures, bootstrap, leave-last-out, sampled negatives

IALS = dict(k=50, lam=30, alpha=3)                      # Day 158's validation choice for implicit ALS
STAR_ALPHA = 0.6                                        # Day 158's choice for stars as confidence
SVD_K = 20                                              # Day 158's PureSVD rank
NEG, LR, BATCH, EPOCHS = 4, 0.001, 256, 20              # the paper's settings: 4 negatives per watched film, Adam at 0.001, batches of 256
GMF_DS = (32, 64, 128)                                  # GMF embedding sizes tried
GMF_LAMS = (0.003, 0.01, 0.03)                          # GMF penalties tried (0.1 and above collapsed GMF in a trial run)
MLP_DMS = (32, 64, 128)                                 # MLP embedding sizes tried (layers 2d, d, d/2); 64 won a first run at the edge
MLP_LAMS = (0.03, 0.1, 1.0)                             # MLP penalties tried; 0.1 won a first run at the edge


# =============================================================================
# THE NETWORK (new today; the build day has no exercise file)
# =============================================================================
def ncf_init(nu, ni, d, dm, seed=0):                    # every weight of GMF, MLP and NeuMF
    rng = np.random.RandomState(seed); F = {}                                         # seeded, so a run repeats
    F["P"], F["Q"] = 0.01 * rng.randn(nu, d), 0.01 * rng.randn(ni, d)                 # GMF's user and film embeddings
    F["U"], F["V"] = 0.01 * rng.randn(nu, dm), 0.01 * rng.randn(ni, dm)               # the MLP's own user and film embeddings
    sizes = (2 * dm, 2 * dm, dm, dm // 2)                                             # input, then three ReLU layers halving in width
    for l in range(3):                                                                # each layer
        F[f"W{l}"] = rng.randn(sizes[l], sizes[l + 1]) * np.sqrt(2.0 / sizes[l])      # He initialisation, for ReLU
        F[f"b{l}"] = np.zeros(sizes[l + 1])                                           # biases start at 0
    F["h"] = 0.01 * rng.randn(d + dm // 2); F["c"] = np.zeros(1)                      # output weights over [GMF part, MLP part], and a bias
    return F                                                                          # a dictionary of arrays


def ncf_forward(F, u, i, mode):                         # the logit for each (user, film) pair, and what backprop needs
    d = F["P"].shape[1]; C = {}                                                       # GMF width; the cache
    g = F["P"][u] * F["Q"][i] if mode != "mlp" else np.zeros((len(u), d))             # GMF: element-wise product (zeros when unused)
    if mode != "gmf":                                                                 # the MLP tower
        a = np.concatenate([F["U"][u], F["V"][i]], 1); C["acts"] = [a]               # the two embeddings side by side
        for l in range(3):                                                            # three ReLU layers
            a = np.maximum(a @ F[f"W{l}"] + F[f"b{l}"], 0.0); C["acts"].append(a)     # each layer's output, kept for backprop
    else:                                                                             # GMF alone
        a = np.zeros((len(u), F["h"].shape[0] - d))                                   # the MLP part of the output is unused
    C["x"] = np.concatenate([g, a], 1)                                                # what the output layer sees
    return C["x"] @ F["h"] + F["c"][0], C                                             # logit = h . x + c


def ncf_loss_grad(F, u, i, y, mode, lam):               # binary cross-entropy + penalty on this batch's embedding rows, and its gradient
    z, C = ncf_forward(F, u, i, mode); B = len(y)                                     # logits
    p = 1.0 / (1.0 + np.exp(-z))                                                      # sigmoid: probability the user watched the film
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))          # cross-entropy, averaged over the batch
    emb = (("P", u), ("Q", i)) if mode == "gmf" else (("U", u), ("V", i)) if mode == "mlp" else (("P", u), ("Q", i), ("U", u), ("V", i))
    loss += 0.5 * lam * sum((F[k][idx] ** 2).sum() for k, idx in emb) / B              # penalty on the rows this batch uses
    G = {k: np.zeros_like(v) for k, v in F.items()}; dz = (p - y) / B                 # gradients; d loss / d logit
    G["h"] = C["x"].T @ dz; G["c"][0] = dz.sum(); dx = np.outer(dz, F["h"])           # output layer
    d = F["P"].shape[1]                                                               # GMF width
    if mode != "mlp":                                                                 # back through the product
        np.add.at(G["P"], u, dx[:, :d] * F["Q"][i]); np.add.at(G["Q"], i, dx[:, :d] * F["P"][u])  # a row can appear twice: add, do not overwrite
    if mode != "gmf":                                                                 # back through the MLP
        da = dx[:, d:]; acts = C["acts"]                                              # gradient arriving at the last layer
        for l in (2, 1, 0):                                                           # last layer first
            dl = da * (acts[l + 1] > 0)                                               # through the ReLU
            G[f"W{l}"] = acts[l].T @ dl; G[f"b{l}"] = dl.sum(0); da = dl @ F[f"W{l}"].T  # weights, biases, and the layer below
        dm = F["U"].shape[1]                                                          # MLP embedding width
        np.add.at(G["U"], u, da[:, :dm]); np.add.at(G["V"], i, da[:, dm:])            # into the two embeddings
    for k, idx in emb:                                                                # the penalty's gradient, rows in the batch only
        np.add.at(G[k], idx, lam * F[k][idx] / B)                                     # lam x row / B, once per appearance
    return loss, G                                                                    # the loss and every gradient


def ncf_train(F, M, mode, lam, epochs, lr=LR, seed=0, track=None, dense_l2=None):  # Adam on watched films + NEG sampled negatives per epoch
    rng = np.random.RandomState(seed); pu, pi = np.nonzero(M); ni = M.shape[1]         # seeded; the watched pairs
    m = {k: np.zeros_like(v) for k, v in F.items()}; v = {k: np.zeros_like(x) for k, x in F.items()}; t = 0  # Adam's moments
    hist = []                                                                         # one record per epoch
    for _ in range(epochs):                                                           # each epoch draws fresh negatives
        nu_ = np.repeat(pu, NEG); ni_ = rng.randint(0, ni, len(nu_)); bad = M[nu_, ni_]  # NEG random films per watched film
        while bad.any():                                                              # any that the user did watch are redrawn
            ni_[bad] = rng.randint(0, ni, bad.sum()); bad = M[nu_, ni_]               # until none are watched
        uu, ii = np.r_[pu, nu_], np.r_[pi, ni_]; yy = np.r_[np.ones(len(pu)), np.zeros(len(nu_))]  # positives (1) and negatives (0)
        o = rng.permutation(len(uu)); tot = 0.0                                       # shuffled order; summed loss
        for s in range(0, len(o), BATCH):                                             # one batch at a time
            b = o[s:s + BATCH]; loss, G = ncf_loss_grad(F, uu[b], ii[b], yy[b], mode, 0.0 if dense_l2 else lam)  # loss and gradients
            tot += loss * len(b); t += 1                                              # running total; Adam's step count
            for k in F:                                                               # every weight
                g = G[k] + (lam * F[k] if dense_l2 == "all" or (dense_l2 and k in ("P", "Q", "U", "V")) else 0.0)  # the first draft's decay, every step
                m[k] = 0.9 * m[k] + 0.1 * g; v[k] = 0.999 * v[k] + 0.001 * g * g      # Adam's running mean and mean square
                F[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)  # bias-corrected step
        hist.append((tot / len(o),) + (track(F) if track else ()))                    # mean loss, then measures, after this epoch
    return F, hist                                                                    # the trained weights, and the records


def ncf_scores(F, mode, nu, ni, chunk=64):              # every user's logit for every film
    S = np.zeros((nu, ni))                                                            # users x films
    for s in range(0, nu, chunk):                                                     # a few users at a time, to keep memory small
        n = min(s + chunk, nu) - s                                                    # users in this chunk
        S[s:s + n] = ncf_forward(F, np.repeat(np.arange(s, s + n), ni), np.tile(np.arange(ni), n), mode)[0].reshape(n, ni)
    return S                                                                          # the score table


def neumf_from(Fg, Fm):                                 # NeuMF started from a trained GMF and a trained MLP (the paper's pre-training)
    F = {k: x.copy() for k, x in Fm.items()}                                          # the MLP's tower and embeddings
    F["P"], F["Q"] = Fg["P"].copy(), Fg["Q"].copy()                                   # GMF's embeddings
    dg, dmm = Fg["P"].shape[1], Fm["P"].shape[1]                                      # GMF's width; the width the MLP model carried unused
    F["h"] = np.r_[0.5 * Fg["h"][:dg], 0.5 * Fm["h"][dmm:]]                           # half of each model's output weights
    F["c"] = 0.5 * (Fg["c"] + Fm["c"])                                                # and half of each bias
    return F                                                                          # ready to fine-tune


# =============================================================================
# HELPERS FOR THE EXPERIMENTS
# =============================================================================
def setup():                                            # Day 154's data and split, and the films' genres
    u, i, r, t = d154.load_movielens(); _, genres, _ = d154.load_items()             # ratings; each film's genre flags
    nu, ni = u.max() + 1, i.max() + 1                                                 # users and films, counted from 0
    tr, te = d154.split_ratings(len(r), 0.2, seed=0)                                  # Day 154's 80/20 split
    fi, va = d154.split_ratings(len(tr), 0.1, seed=1); fi, va = tr[fi], tr[va]        # and its validation part
    X = d156.film_features(genres, d156.load_years())[:, :19]                         # Day 156's features: the 19 genre flags
    return u, i, r, t, nu, ni, tr, te, fi, va, X                                      # everything the experiments need


def watched(u, i, idx, nu, ni):                         # True where the user watched the film, from a set of rows
    M = np.zeros((nu, ni), bool); M[u[idx], i[idx]] = True                            # users x films
    return M                                                                          # the table


def gains(u, i, r, idx, nu, ni, graded=False):          # films to find: 1 each, or 2^stars - 1
    G = np.zeros((nu, ni)); G[u[idx], i[idx]] = 2.0 ** r[idx] - 1 if graded else 1.0  # 0 where there is nothing to find
    return G                                                                          # users x films


def week_models(u, i, r, idx, nu, ni):                  # the week's best non-neural models, from one set of rows
    M = watched(u, i, idx, nu, ni); B = M.astype(float)                               # watched, as True / False and 0 / 1
    R, _ = d154.to_matrix(u[idx], i[idx], r[idx], nu, ni)                             # stars, for the star-weighted model
    X, Y, _ = d158.implicit_als(np.where(M, 1.0 + IALS["alpha"], 0.0), IALS["k"], IALS["lam"])  # Day 158's implicit ALS
    Xs, Ys, _ = d158.implicit_als(np.where(M, 1.0 + STAR_ALPHA * R, 0.0), IALS["k"], IALS["lam"])  # with stars as confidence
    return M, {"popularity": np.broadcast_to(B.sum(0), M.shape),                      # how many watched each film
               "co-occurrence": B @ d158.cooccurrence(B),                             # similarity to the films watched
               "PureSVD": d155.low_rank(B, SVD_K),                                    # Day 155's rank-20 fit of the 0/1 table
               "implicit ALS": X @ Y.T,                                               # Day 158's model
               "implicit ALS, stars": Xs @ Ys.T}                                      # and its star-weighted version


def zrows(S):                                           # each user's scores standardised, so models on different scales can be blended
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)         # mean 0, spread 1 per user


def val_precision(Mf, Hv, mode):                        # a tracker: validation precision@10 after each epoch
    return lambda F: (d159.per_user_metrics(ncf_scores(F, mode, *Mf.shape), Mf, Hv)[0]["precision"].mean(),)


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_check(nu, ni):                                                               # section 1
    print("\n" + "=" * 70)
    print("1. THE NETWORK, AND A GRADIENT CHECK THROUGH THE WHOLE PIPELINE")
    print("=" * 70)
    rng = np.random.RandomState(0)                                                    # for the check's data
    u, i = np.array([0, 1, 2, 3, 4, 1, 2]), np.array([0, 1, 2, 3, 4, 5, 0]); y = np.array([1, 0, 1, 0, 1, 1, 0.0])  # 7 pairs
    for mode in ("gmf", "mlp", "neumf"):                                              # every variant
        F = ncf_init(5, 6, 3, 4, seed=1)                                              # a tiny network
        for k in F:                                                                   # move every weight off its starting value
            F[k] = F[k] + 0.3 * rng.randn(*F[k].shape)
        _, G = ncf_loss_grad(F, u, i, y, mode, 0.1); worst = 0.0                      # analytic gradients, with the penalty on
        for k in F:                                                                   # every weight, one at a time
            for idx in np.ndindex(F[k].shape):                                        # every entry
                old = F[k][idx]; F[k][idx] = old + 1e-6; lp = ncf_loss_grad(F, u, i, y, mode, 0.1)[0]  # loss a little above
                F[k][idx] = old - 1e-6; lm = ncf_loss_grad(F, u, i, y, mode, 0.1)[0]; F[k][idx] = old  # and a little below
                worst = max(worst, abs((lp - lm) / 2e-6 - G[k][idx]))                 # largest disagreement so far
        print(f"  {mode:>6}: {sum(x.size for x in F.values())} weights; largest gap between backprop and finite differences {worst:.1e}")
    F = ncf_init(nu, ni, 64, 32)                                                      # the real size
    print(f"real size (GMF 64, MLP 32): {sum(x.size for x in F.values())} weights; one epoch = "
          f"{NEG + 1} x the watched films, in batches of {BATCH}")
    print("Actual output from this lesson's run.")


def demo_penalty(Mf, Hv):                                                             # section 2
    print("\n" + "=" * 70)
    print("2. A FIRST-DRAFT BUG: AN L2 PENALTY INSIDE ADAM")
    print("=" * 70)
    nu, ni = Mf.shape                                                                 # sizes
    for dense, lam, how in (("all", 1e-5, "first draft: decay 1e-5 on every weight"),  # the first draft
                            ("emb", 1e-5, "decay 1e-5 on every embedding row"),       # decay on the embeddings only
                            (None, 0.01, "penalty 0.01 on the batch's rows only")):   # the fix
        F = ncf_init(nu, ni, 16, 8)                                                   # GMF, 16 numbers per user and film
        F, h = ncf_train(F, Mf, "gmf", lam, 8, track=val_precision(Mf, Hv, "gmf"), dense_l2=dense)  # 8 epochs
        print(f"  {how:>39}: validation precision@10 by epoch {', '.join(f'{x[1]:.3f}' for x in h)}")
        print(f"  {'':>39}  largest |film embedding| {np.abs(F['Q']).max():.4f}, largest |output weight| {np.abs(F['h'][:16]).max():.4f}")
    print("Actual output from this lesson's run.")


def demo_tune(Mf, Hv):                                                                # section 3
    print("\n" + "=" * 70)
    print("3. TUNING GMF, MLP AND NEUMF ON VALIDATION")
    print("=" * 70)
    nu, ni = Mf.shape; best = {}; curves = {}                                         # sizes; winners; their learning curves
    for mode, dims, lams in (("gmf", GMF_DS, GMF_LAMS), ("mlp", MLP_DMS, MLP_LAMS)):  # the two towers
        print(f"{mode.upper()}: validation precision@10 at its best epoch (of {EPOCHS})")
        for dd in dims:                                                               # every embedding size
            row = []                                                                  # one line per size
            for lam in lams:                                                          # every penalty
                t0 = time.time(); F = ncf_init(nu, ni, dd if mode == "gmf" else 8, dd if mode == "mlp" else 8)  # the unused tower stays small
                F, h = ncf_train(F, Mf, mode, lam, EPOCHS, track=val_precision(Mf, Hv, mode))  # train, scoring every epoch
                v = [x[1] for x in h]; e = int(np.argmax(v)) + 1                       # the best epoch
                row.append(f"lam {lam:g}: {max(v):.4f} (epoch {e})")                  # for the printout
                if mode not in best or max(v) > best[mode][0]:                        # the best so far
                    best[mode] = (max(v), dd, lam, e, F); curves[mode] = h            # keep it (F is the final-epoch model)
            print(f"  size {dd:>3}: " + "; ".join(row))
        print(f"  chosen {mode.upper()}: size {best[mode][1]}, lam {best[mode][2]:g}, {best[mode][3]} epochs (validation {best[mode][0]:.4f})")
    (_, dg, lg, eg, _), (_, dmm, lm, em, _) = best["gmf"], best["mlp"]                # the two winners' settings
    Fg = ncf_train(ncf_init(nu, ni, dg, 8), Mf, "gmf", lg, eg)[0]                     # retrained to their best epochs, for pre-training
    Fm = ncf_train(ncf_init(nu, ni, 8, dmm), Mf, "mlp", lm, em)[0]
    out = {}                                                                          # NeuMF three ways
    for how, F, lr in (("from scratch", ncf_init(nu, ni, dg, dmm), LR),               # random start
                       ("pre-trained", neumf_from(Fg, Fm), LR / 10),                  # the paper fine-tunes the combination more slowly
                       ("pre-trained", neumf_from(Fg, Fm), LR / 100)):                # and slower still (lr / 10 was best at its first epoch)
        h0 = [(np.nan,) + val_precision(Mf, Hv, "neumf")(F)] if how == "pre-trained" else []  # epoch 0: the combination, untrained
        F, h = ncf_train(F, Mf, "neumf", lg, EPOCHS, lr=lr, track=val_precision(Mf, Hv, "neumf"))  # train, scoring every epoch
        h = h0 + h; v = [x[1] for x in h]; first = 0 if h0 else 1                     # epochs counted from 0 when epoch 0 was scored
        key = f"NeuMF, {how}, lr {lr:g}"; out[key] = (max(v), int(np.argmax(v)) + first, lr); curves[key] = h  # best epoch
        print(f"{key}: best validation {max(v):.4f} at epoch {out[key][1]}; by epoch from {first}: "
              f"{', '.join(f'{x:.3f}' for x in v)}")
    print(f"for reference, Day 158's validation precision@10: implicit ALS 0.1374, PureSVD 0.1313")
    print("Actual output from this lesson's run.")
    return best, out, curves                                                          # settings and curves


def fit_final(u, i, idx, nu, ni, best, out):            # GMF, MLP and NeuMF refitted on a set of rows with the chosen settings
    M = watched(u, i, idx, nu, ni)                                                    # the training table
    (_, dg, lg, eg, _), (_, dmm, lm, em, _) = best["gmf"], best["mlp"]                # chosen settings
    Fg = ncf_train(ncf_init(nu, ni, dg, 8), M, "gmf", lg, eg)[0]                      # GMF
    Fm = ncf_train(ncf_init(nu, ni, 8, dmm), M, "mlp", lm, em)[0]                     # MLP
    how = max(out, key=lambda k: out[k][0])                                           # the better NeuMF on validation
    Fn = ncf_init(nu, ni, dg, dmm) if "scratch" in how else neumf_from(Fg, Fm)        # started the same way
    Fn = ncf_train(Fn, M, "neumf", lg, out[how][1], lr=out[how][2])[0]                # and trained to its best epoch
    return {"GMF": ncf_scores(Fg, "gmf", nu, ni), "MLP": ncf_scores(Fm, "mlp", nu, ni), "NeuMF": ncf_scores(Fn, "neumf", nu, ni)}, Fn, how


def demo_final(u, i, r, nu, ni, tr, te, fi, va, X, best, out):                       # section 4
    print("\n" + "=" * 70)
    print("4. THE WEEK'S FINAL COMPARISON ON DAY 154'S TEST")
    print("=" * 70)
    t0 = time.time(); M, sc = week_models(u, i, r, tr, nu, ni)                        # the week's models on all training rows
    nn, Fn, how = fit_final(u, i, tr, nu, ni, best, out); sc.update(nn)               # and the three networks
    print(f"models refitted on all {len(tr)} training ratings ({time.time() - t0:.0f} s); chosen: {how}")
    Mf = watched(u, i, fi, nu, ni)                                                    # the blend is fitted on validation, from models without it
    Xv, Yv, _ = d158.implicit_als(np.where(Mf, 1.0 + IALS["alpha"], 0.0), IALS["k"], IALS["lam"])  # implicit ALS without validation
    nv = fit_final(u, i, fi, nu, ni, best, out)[0]["NeuMF"]                           # NeuMF without validation
    rng = np.random.RandomState(7); nu_ = np.repeat(u[va], NEG); ni_ = rng.randint(0, ni, len(nu_))  # NEG random films per validation film
    ok = ~Mf[nu_, ni_] & ~watched(u, i, va, nu, ni)[nu_, ni_]                         # keep only films the user never watched
    cu, ci = np.r_[u[va], nu_[ok]], np.r_[i[va], ni_[ok]]; y = np.r_[np.ones(len(va)), np.zeros(ok.sum())]  # cells and targets
    w = d157.stack_fit(np.c_[zrows(Xv @ Yv.T)[cu, ci], zrows(nv)[cu, ci]], y)         # Day 157's blend, by least squares
    print(f"Day 157's blend fitted on {len(y)} validation cells: intercept {w[0]:+.3f}, implicit ALS {w[1]:+.3f}, NeuMF {w[2]:+.3f}")
    sc["blend: implicit ALS + NeuMF"] = d157.stack_predict(np.c_[zrows(sc["implicit ALS"]).ravel(), zrows(sc["NeuMF"]).ravel()], w).reshape(nu, ni)
    perm = np.random.RandomState(0).permutation(nu); Fs = dict(Fn); Fs["P"], Fs["U"] = Fn["P"][perm], Fn["U"][perm]  # control: users' embeddings shuffled
    sc["control: NeuMF, users shuffled"] = ncf_scores(Fs, "neumf", nu, ni)            # personal information removed
    Gb, Gg = gains(u, i, r, te, nu, ni), gains(u, i, r, te, nu, ni, True)             # films to find, plain and graded
    pop = M.sum(0); res, per = {}, {}                                                 # watchers per film; results
    print(f"{'test @10':>31} {'precision':>9} {'recall':>7} {'NDCG':>7} {'graded':>7} {'coverage':>8} {'median pop.':>11} "
          f"{'personal.':>9} {'genres':>6}")
    for nm, S in sc.items():                                                          # one row per model
        m, top = d159.per_user_metrics(S, M, Gb); mg = d159.per_user_metrics(S, M, Gg)[0]  # Day 159's measures
        b = d159.beyond_accuracy(top, pop, nu)                                        # and the lists themselves
        gen = (X[top].max(1) > 0).sum(1).mean()                                       # distinct genres in a list (Day 156's flags)
        res[nm] = (m["precision"].mean(), m["recall"].mean(), m["ndcg"].mean(), mg["ndcg"].mean(), b["coverage"], b["median_pop"], b["personal"], gen)
        per[nm] = mg["ndcg"]                                                          # per-user graded NDCG, for the bootstrap
        print(f"{nm:>31} {res[nm][0]:>9.4f} {res[nm][1]:>7.4f} {res[nm][2]:>7.4f} {res[nm][3]:>7.4f} {res[nm][4]:>8.3f} "
              f"{res[nm][5]:>11.0f} {res[nm][6]:>9.3f} {res[nm][7]:>6.2f}")
    print("paired bootstrap over users, graded NDCG@10:")
    boots = {}                                                                        # for the figure
    for a, b_ in (("NeuMF", "implicit ALS"), ("NeuMF", "PureSVD"), ("blend: implicit ALS + NeuMF", "implicit ALS")):  # three comparisons
        d, ci, p, mm = d159.paired_bootstrap(per[a], per[b_]); boots[f"{a} - {b_}"] = mm  # Day 159's bootstrap
        print(f"  {a:>27} - {b_:<12}: {d:+.4f}, 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}], resamples with no gain {p:.1%}")
    print("Actual output from this lesson's run.")
    return res, boots                                                                 # for the figure


def demo_paper(u, i, r, t, nu, ni, best, out):                                        # section 5
    print("\n" + "=" * 70)
    print("5. THE PAPER'S PROTOCOL: LEAVE-LAST-OUT, 100 SAMPLED NEGATIVES")
    print("=" * 70)
    trl, tel = d159.leave_last_out(u, t)                                              # each user's latest rating held out
    t0 = time.time(); M, sc = week_models(u, i, r, trl, nu, ni)                       # the week's models without them
    sc.update(fit_final(u, i, trl, nu, ni, best, out)[0])                             # and the three networks
    item = np.zeros(nu, int); item[u[tel]] = i[tel]; G = np.zeros((nu, ni)); G[np.arange(nu), item] = 1.0  # one film to find per user
    print(f"{len(tel)} held-out films, models refitted ({time.time() - t0:.0f} s)")
    print(f"{'':>20} {'all films ranked: HR@10':>24} {'NDCG@10':>8}   {'100 sampled: HR@10':>19} {'NDCG@10':>8}")
    res = {}                                                                          # for the figure
    for nm, S in sc.items():                                                          # one row per model
        m = d159.per_user_metrics(S, M, G)[0]; hs, ns = d159.sampled_eval(S, M, item)  # full ranking, and the shortcut
        res[nm] = (m["recall"].mean(), hs)                                            # hit rate both ways
        print(f"{nm:>20} {m['recall'].mean():>24.4f} {m['ndcg'].mean():>8.4f}   {hs:>19.4f} {ns:>8.4f}")
    for p_ in (0.15, 0.65):                                                           # one hit or miss per user, so a hit rate is noisy
        print(f"standard error of a hit rate near {p_:.2f} over {nu} users: {np.sqrt(p_ * (1 - p_) / nu):.4f}")
    print("Actual output from this lesson's run.")
    return res                                                                        # for the figure


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(curves, res, boots, paper, path):                                    # six panels
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))                                   # two rows of three
    for q, (nm, h) in enumerate(curves.items()):                                      # every chosen model's learning curve
        e = np.arange(len(h)) + (0 if "pre-trained" in nm else 1)                     # epochs (pre-trained runs start at epoch 0)
        ax[0, 0].plot(e, [x[1] for x in h], "o-", ms=3, color=f"C{q}", label=nm.upper() if nm in ("gmf", "mlp") else nm)  # validation
        ax[0, 1].plot(e, [x[0] for x in h], "o-", ms=3, color=f"C{q}", label=nm.upper() if nm in ("gmf", "mlp") else nm)  # training loss
    ax[0, 0].axhline(0.1374, color="k", ls="--", lw=1, label="implicit ALS (Day 158)")  # the target
    ax[0, 0].set_ylim(0, 0.145); ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("validation precision@10"); ax[0, 0].legend(fontsize=7)  # labels
    ax[0, 0].set_title("learning curves: validation", fontsize=10)                    # title
    ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("training loss (cross-entropy)"); ax[0, 1].legend(fontsize=7)  # labels
    ax[0, 1].set_title("learning curves: training loss keeps falling", fontsize=10)   # title
    names = list(res); xs = np.arange(len(names))                                     # models of section 4
    short = [n.replace("blend: implicit ALS + NeuMF", "blend").replace("control: NeuMF, users shuffled", "control") for n in names]  # short names
    ax[0, 2].bar(xs, [res[n][0] for n in names], color=["C3" if n in ("GMF", "MLP", "NeuMF") else "C0" for n in names])  # test precision
    ax[0, 2].set_xticks(xs); ax[0, 2].set_xticklabels(short, fontsize=7, rotation=30, ha="right")  # names
    ax[0, 2].set_ylabel("test precision@10"); ax[0, 2].set_title("Day 154's test: networks in red", fontsize=10)  # labels
    for n, s in zip(names, short):                                                    # coverage against precision
        ax[1, 0].scatter(res[n][4], res[n][0], s=40, color="C3" if n in ("GMF", "MLP", "NeuMF") else "C0")  # one point per model
        off = {"blend": (4, -10), "MLP": (4, -10), "implicit ALS, stars": (-70, 5)}.get(s, (4, 3))  # labels nudged apart where points crowd
        ax[1, 0].annotate(s, (res[n][4], res[n][0]), fontsize=7, xytext=off, textcoords="offset points")  # its name
    ax[1, 0].set_xlabel("catalogue coverage"); ax[1, 0].set_ylabel("test precision@10")  # labels
    ax[1, 0].set_title("accuracy against coverage", fontsize=10)                      # title
    for k_, m in boots.items():                                                       # each comparison's resampled differences
        ax[1, 1].hist(m, bins=40, alpha=0.6, label=k_.replace("blend: implicit ALS + NeuMF", "blend"))  # one histogram each
    ax[1, 1].axvline(0, color="k", lw=0.8); ax[1, 1].legend(fontsize=7); ax[1, 1].set_xlabel("difference in graded NDCG@10")  # zero line
    ax[1, 1].set_title("paired bootstrap over users", fontsize=10)                    # title
    pn = list(paper); xs = np.arange(len(pn))                                         # models of section 5
    ax[1, 2].bar(xs - 0.2, [paper[n][0] for n in pn], 0.4, label="all films ranked")  # full ranking
    ax[1, 2].bar(xs + 0.2, [paper[n][1] for n in pn], 0.4, label="100 sampled negatives")  # the shortcut
    ax[1, 2].set_xticks(xs); ax[1, 2].set_xticklabels(pn, fontsize=7, rotation=30, ha="right"); ax[1, 2].legend(fontsize=8)  # names
    ax[1, 2].set_ylabel("hit rate@10, leave-last-out"); ax[1, 2].set_title("the paper's protocol against full ranking", fontsize=10)  # labels
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)                    # save to disk


def draw_diagram():                                     # NeuMF's architecture, and the week's parts around it
    fig, ax = plt.subplots(figsize=(15, 6.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis("off")  # a wide, empty canvas
    p = 0.08                                                                          # the boxes' rounded padding

    def box(x, y, w, h, text, fc):                                                    # a rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={p}", fc=fc, ec="k", lw=1.2))  # the box
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)   # its text

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two boxes
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))  # from (x0, y0) to (x1, y1)
    box(0.3, 3.9, 1.8, 0.9, "user id", "#e8f0fe"); box(0.3, 2.3, 1.8, 0.9, "film id", "#e8f0fe")  # the inputs
    box(3.0, 4.6, 2.6, 0.9, "GMF embeddings\np_u, q_i", "#e6f4ea"); box(3.0, 1.6, 2.6, 0.9, "MLP embeddings\nu_u, v_i", "#fde8e8")  # two pairs
    box(6.5, 4.6, 2.4, 0.9, "element-wise\nproduct p_u * q_i", "#e6f4ea")             # GMF
    box(6.5, 1.6, 2.4, 0.9, "concatenate, then\n3 ReLU layers", "#fde8e8")             # MLP
    box(9.8, 3.1, 2.2, 0.9, "concatenate both;\nh . x + c", "#fff4d6")                 # NeuMF output
    box(12.8, 3.1, 1.9, 0.9, "sigmoid:\nP(watched)", "#fff4d6")                        # probability
    box(3.0, 0.1, 11.7, 0.8, "trained on watched films (1) + 4 sampled unwatched (0) per epoch, cross-entropy, Adam; "
        "judged with Day 159's measures against Day 158's implicit ALS", "#f3f3f3")   # training and evaluation
    arrow(2.1 + p, 4.35, 3.0 - p, 5.0); arrow(2.1 + p, 4.25, 3.0 - p, 2.15)           # user id into both embedding tables
    arrow(2.1 + p, 2.75, 3.0 - p, 4.85); arrow(2.1 + p, 2.65, 3.0 - p, 1.95)          # film id into both
    arrow(5.6 + p, 5.05, 6.5 - p, 5.05); arrow(5.6 + p, 2.05, 6.5 - p, 2.05)          # embeddings into each tower
    arrow(8.9 + p, 5.05, 9.8 - p, 3.75); arrow(8.9 + p, 2.05, 9.8 - p, 3.35)          # towers into the output
    arrow(12.0 + p, 3.55, 12.8 - p, 3.55)                                             # output into the sigmoid
    ax.set_title("Day 160: NeuMF = GMF and an MLP side by side (GMF or MLP alone use one tower)", fontsize=12)  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "neural_cf_diagram.png"), dpi=120); plt.close(fig)  # save to disk


def note_on_what_comes_next():  # a preview of Week 24
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Week 24: model evaluation and statistics, a deep dive. Day 161 revisits the bias-variance tradeoff")
    print("with real learning curves, and the week goes on to hypothesis tests, bootstrapping, multiple")
    print("comparisons, calibration, fairness metrics and explainability.")


def main():                                             # runs every section in order
    u, i, r, t, nu, ni, tr, te, fi, va, X = setup()                                  # Day 154's data and split, and the genres
    Mf = watched(u, i, fi, nu, ni); Hv = gains(u, i, r, va, nu, ni)                   # training without validation, and validation
    demo_check(nu, ni)                                                                # 1. gradient check
    demo_penalty(Mf, Hv)                                                              # 2. the penalty bug
    best, out, curves = demo_tune(Mf, Hv)                                             # 3. tuning on validation
    res, boots = demo_final(u, i, r, nu, ni, tr, te, fi, va, X, best, out)            # 4. the final comparison
    paper = demo_paper(u, i, r, t, nu, ni, best, out)                                 # 5. the paper's protocol
    plot_results(curves, res, boots, paper, os.path.join(HERE, "neural_cf_results.png"))  # the results figure
    draw_diagram()                                                                    # the architecture
    print("\nsaved neural_cf_results.png\nsaved neural_cf_diagram.png")
    note_on_what_comes_next()                                                         # a preview


if __name__ == "__main__":  # only when run directly, not when imported
    main()                                                                            # run everything
