"""
Day 171: Early Stopping

A network trained for too long starts to memorise its training rows: the
training loss keeps falling while the loss on rows it has not seen turns and
rises. Early stopping watches a held-out "stop set" after every epoch, stops
when that loss has not improved for a while (the patience), and goes back to
the weights from the best epoch. It is the cheapest regulariser there is, and
it saves training time too. Today builds it from scratch and asks what it
really costs and saves:

  1. what over-fitting looks like on UCI Adult, at 2,000 and 18,000 training
     rows, with a control where the training labels are shuffled
  2. patience and Prechelt's generalisation-loss rule: epochs saved against
     the loss on fresh rows, measured against the best epoch in hindsight
  3. restoring the best weights against keeping the last ones
  4. how big must the stop set be? Small ones stop at the wrong epoch, and the
     loss they report at their chosen epoch is too good (a winner's curse)
  5. the stopper run for real, checked against the same rule replayed on a
     recorded curve, and what it saves in time

Data: Week 24's split of UCI Adult. Of the 24,130 fitting rows, 6,000 form the
stop set and the first 2,000 or 18,000 of the rest train; the 6,032
validation rows are the fresh rows that score every choice and choose
nothing. The test file is not touched. The network is Day 170's (Day 126's
87 -> 64 -> 32 -> 1 with Adam), imported unchanged.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import io                                              # io: StringIO, where that output goes
import os                                              # os: file paths
import sys                                             # sys: the import path
import time                                            # time: what each method costs

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, HERE)                                # this week's folder: Day 170's exercise
sys.path.insert(0, os.path.join(ROOT, "week-24-model-evaluation-statistics"))        # Day 161's data
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day170_exercise as d170                      # Day 170: bce_logits, the network's shape, and Day 126 through it
    import day161_exercise as d161                      # Day 161: the Adult data and Week 24's split
d126 = d170.d126                                        # Day 126: init_net, forward, backward, Adam

MAX_EPOCHS = 100                                        # the longest any run trains
LR = 1e-3                                               # Adam's usual rate, inside Day 170's good range (3e-4 to 1e-2)
SIZES = (2000, 18000)                                   # training rows: a small set that over-fits fast, and most of the data
STOP_ROWS = 6000                                        # the stop set
SEEDS = 10                                              # runs per training size
PATIENCES = (1, 2, 5, 10, 20)                           # epochs without improvement before stopping
GL_ALPHAS = (1.0, 2.0, 5.0)                             # Prechelt's rule: stop once the loss is this many percent above its best
HALVES = (100, 500, 3000)                               # stop-set sizes for section 4 (each draws two halves of this size)
SPLITS = 100                                            # random draws per seed and size in section 4


# =============================================================================
# THE PARTS (copied unchanged into day171_exercise.py, for Day 174's build)
# =============================================================================
def copy_net(net):                                      # an independent copy of a network's weights
    return [(W.copy(), b.copy()) for W, b in net]                                     # new arrays: later steps do not change the copy


class EarlyStopper:                                     # stop when the score has not improved for `patience` epochs
    def __init__(self, patience=5, min_delta=0.0):      # min_delta: how much better counts as better
        self.patience, self.min_delta = patience, min_delta                           # the rule
        self.best, self.best_epoch, self.best_net, self.wait = np.inf, -1, None, 0    # nothing seen yet

    def update(self, epoch, score, net=None):           # call once per epoch; returns True when training should stop
        if score < self.best - self.min_delta:                                        # a real improvement
            self.best, self.best_epoch, self.wait = score, epoch, 0                   # remember it; reset the count
            self.best_net = copy_net(net) if net is not None else None                # and the weights that gave it
        else:                                                                         # no improvement
            self.wait += 1                                                            # one more epoch without one
        return self.wait >= self.patience                                             # out of patience


def stop_at(curve, patience, min_delta=0.0):            # replay EarlyStopper on a recorded curve: (stop epoch, best epoch)
    es = EarlyStopper(patience, min_delta)                                            # the same class, so the same rule
    for e, v in enumerate(curve):                                                     # epoch by epoch
        if es.update(e, v):                                                           # it would have stopped here
            return e, es.best_epoch                                                   # where it stopped; the weights it restores
    return len(curve) - 1, es.best_epoch                                              # never stopped: ran to the end


def gl_stop_at(curve, alpha):                           # Prechelt (1998): stop once the loss is alpha percent above its best so far
    best = np.minimum.accumulate(np.asarray(curve, float))                            # the best loss up to each epoch
    gl = 100 * (np.asarray(curve, float) / best - 1)                                  # generalisation loss, in percent
    hit = np.flatnonzero(gl > alpha)                                                  # epochs where the rule fires
    e = int(hit[0]) if len(hit) else len(curve) - 1                                   # the first one, or the end
    return e, int(np.argmin(curve[:e + 1]))                                           # where it stopped; the best epoch up to then


def row_losses(net, acts, X, y):                        # each row's log loss (float64, so every caller averages it identically)
    z = d126.forward(net, X, acts)[-1][:, 0].astype(float)                            # one logit per row
    return np.logaddexp(0, z) - y * z                                                 # log(1 + e^z) - y z, without overflow


def train_epoch(net, opt, acts, Xa, ya, rng, batch=128):  # one pass over the training rows in a new random order
    o = rng.permutation(len(Xa))                                                      # the order
    for s in range(0, len(Xa), batch):                                                # each batch
        A = d126.forward(net, Xa[o[s:s + batch]], acts)                               # forward
        opt.step(net, d126.backward(net, A, acts, d170.bce_logits(A[-1], ya[o[s:s + batch]])[1])[0])  # backward and step


def train_es(Xa, ya, Xs, ys, opt_cls, lr, seed, stopper, max_epochs=MAX_EPOCHS, batch=128):  # train with early stopping
    sizes = [Xa.shape[1], *d170.HIDDEN, 1]; acts = ["relu"] * len(d170.HIDDEN) + ["linear"]  # Day 170's network
    net = d126.init_net(sizes, seed); opt = opt_cls(net, lr); rng = np.random.RandomState(seed + 1)  # as in Day 170's train_mlp
    curve = []                                                                        # the stop set's loss after each epoch
    for e in range(max_epochs):                                                       # up to the limit
        train_epoch(net, opt, acts, Xa, ya, rng, batch)                               # one epoch
        curve.append(row_losses(net, acts, Xs, ys).mean())                            # score it on the stop set
        if stopper.update(e, curve[-1], net):                                         # out of patience
            break                                                                     # stop
    return stopper.best_net, acts, np.array(curve)                                    # the best epoch's weights, not the last


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def trajectory(D, n, seed, shuffle=False):              # one full run of MAX_EPOCHS, recording everything after every epoch
    Xa, ya = D["X"][D["train"][:n]], D["y"][D["train"][:n]]                          # the first n training rows
    if shuffle:                                                                       # the control: labels with no link to the inputs
        ya = ya[np.random.RandomState(99 + seed).permutation(n)]                     # same labels, wrong rows
    sizes = [Xa.shape[1], *d170.HIDDEN, 1]; acts = ["relu"] * len(d170.HIDDEN) + ["linear"]  # the same network as train_es
    net = d126.init_net(sizes, seed); opt = d126.Adam(net, LR); rng = np.random.RandomState(seed + 1)  # the same start
    tr, st, rows, fr = [], [], [], []                                                 # training loss, stop-set loss and rows, fresh loss
    t0 = time.time()                                                                  # a clock
    for e in range(MAX_EPOCHS):                                                       # every epoch, no stopping
        train_epoch(net, opt, acts, Xa, ya, rng)                                      # one epoch
        tr.append(row_losses(net, acts, Xa, ya).mean())                               # on the rows it trains on
        ls = row_losses(net, acts, D["X"][D["stop"]], D["y"][D["stop"]])              # each stop-set row's loss
        st.append(ls.mean()); rows.append(ls.astype(np.float32))                      # averaged exactly as train_es does; rows for section 4
        fr.append(row_losses(net, acts, D["X"][D["fresh"]], D["y"][D["fresh"]]).mean())  # the fresh rows
    return dict(train=np.array(tr), stop=np.array(st), rows=np.array(rows), fresh=np.array(fr), sec=time.time() - t0)  # plain arrays


def head(title):                                        # a section banner
    print("\n" + "=" * 70); print(title); print("=" * 70)


def demo_curves(D, T, C):                               # section 1
    head("1. WHAT OVER-FITTING LOOKS LIKE (Adam, rate 1e-3, 100 epochs, 10 seeds)")       # the banner
    for n in SIZES:                                                                   # each training size
        R = T[n]                                                                      # its ten runs
        bs = [int(np.argmin(r["stop"])) for r in R]; bf = [int(np.argmin(r["fresh"])) for r in R]  # best epochs, stop set and fresh rows
        print(f"  {n:5d} training rows: best epoch on the stop set: median {np.median(bs) + 1:.0f} (range {min(bs) + 1} to "
              f"{max(bs) + 1}); on the fresh rows: median {np.median(bf) + 1:.0f}")
        print(f"         training loss at epoch 100 {np.mean([r['train'][-1] for r in R]):.4f}; fresh loss: at the stop set's best "
              f"epoch {np.mean([r['fresh'][b] for r, b in zip(R, bs)]):.4f}, at epoch 100 {np.mean([r['fresh'][-1] for r in R]):.4f}, "
              f"best in hindsight {np.mean([r['fresh'].min() for r in R]):.4f}")
    q = D["y"][D["train"][:SIZES[0]]].mean()                                          # the training rows' positive rate
    yv = D["y"][D["fresh"]]; base = -np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q))  # log loss of always predicting that rate
    bs = [int(np.argmin(r["stop"])) for r in C]                                       # the control's best epochs
    print(f"  control, {SIZES[0]} rows with shuffled labels: best epoch on the stop set: median {np.median(bs) + 1:.0f}; fresh loss "
          f"there {np.mean([r['fresh'][b] for r, b in zip(C, bs)]):.4f}, at epoch 100 {np.mean([r['fresh'][-1] for r in C]):.4f}; "
          f"always predicting the base rate {q:.3f}: {base:.4f}")
    print("Actual output from this lesson's run.")


def rule_rows(R):                                       # every stopping rule on every run: (name, epochs run, fresh loss, regret)
    rules = [(f"patience {p}", lambda c, p=p: stop_at(c, p)) for p in PATIENCES]     # patience rules
    rules += [(f"GL > {a:g}%", lambda c, a=a: gl_stop_at(c, a)) for a in GL_ALPHAS]  # Prechelt's rule
    rules += [("no stopping, best", lambda c: (len(c) - 1, int(np.argmin(c)))),       # train to the end, keep the best epoch
              ("no stopping, last", lambda c: (len(c) - 1, len(c) - 1))]              # train to the end, keep the last
    out = []                                                                          # one row per rule
    for name, rule in rules:                                                          # each rule
        se = [rule(r["stop"]) for r in R]                                             # (stop, best) for every run
        fr = [r["fresh"][b] for r, (s, b) in zip(R, se)]                              # fresh loss of the weights it keeps
        rg = [f - r["fresh"].min() for r, f in zip(R, fr)]                            # against the best epoch in hindsight
        hit = np.mean([b == int(np.argmin(r["stop"])) for r, (s, b) in zip(R, se)])  # kept the stop set's best of all 100 epochs?
        out.append((name, np.mean([s + 1 for s, b in se]), np.mean(fr), np.mean(rg), max(rg), hit))  # epochs, loss, regret, worst, found
    return out                                                                        # for printing and the figure


def demo_rules(D, T):                                   # section 2
    head("2. STOPPING RULES: EPOCHS RUN AGAINST LOSS ON FRESH ROWS (means over 10 seeds)")  # the banner
    out = {}                                                                          # per training size
    for n in SIZES:                                                                   # each size
        print(f"  {n} training rows                 epochs run   fresh loss   regret (mean / worst)   kept the overall best")
        out[n] = rule_rows(T[n])                                                      # every rule
        for name, ep, fr, rg, wr, hit in out[n]:                                      # one line each
            print(f"    {name:26s} {ep:9.1f}   {fr:10.4f}   {rg:+.4f} / {wr:+.4f}   {hit:.1f}")
    print("  (regret: the fresh loss of the kept weights minus the best fresh loss of any epoch, known only in hindsight;")
    print("   kept the overall best: the share of runs whose kept epoch is the stop set's best of all 100)")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_restore(D, T):                                 # section 3
    head("3. RESTORE THE BEST WEIGHTS, OR KEEP THE LAST ONES?")                           # the banner
    for n in SIZES:                                                                   # each size
        parts = []                                                                    # one entry per patience
        for p in PATIENCES:                                                           # each patience
            se = [stop_at(r["stop"], p) for r in T[n]]                                # (stop, best) per run
            last = np.mean([r["fresh"][s] for r, (s, b) in zip(T[n], se)])            # keep the weights at the stopping epoch
            best = np.mean([r["fresh"][b] for r, (s, b) in zip(T[n], se)])            # go back to the best epoch
            parts.append(f"p{p}: {best:.4f} / {last:.4f}")                            # restored / last
        print(f"  {n:5d} rows, fresh loss restored / last: " + ", ".join(parts))
    print("Actual output from this lesson's run.")


def demo_stopset(D, T):                                 # section 4
    head("4. HOW BIG MUST THE STOP SET BE? (patience 5; two disjoint halves A and B)")    # the banner
    out = {}                                                                          # per training size
    for n in SIZES:                                                                   # each size
        out[n] = []                                                                   # one entry per half size
        for m in HALVES:                                                              # rows in each half
            rg, opt, ctl, ep = [], [], [], []                                         # regret, optimism, its control, epochs
            for k, r in enumerate(T[n]):                                              # each run
                rng = np.random.RandomState(1000 * k + m)                             # a reproducible draw per run and size
                for _ in range(SPLITS):                                               # many draws
                    i = rng.permutation(STOP_ROWS)[:2 * m]; a, b = i[:m], i[m:]       # two disjoint halves
                    ca, cb = r["rows"][:, a].astype(float).mean(1), r["rows"][:, b].astype(float).mean(1)  # each half's curve
                    s, e = stop_at(ca, 5)                                             # stop and choose on half A only
                    rg.append(r["fresh"][e] - r["fresh"].min()); ep.append(s + 1)     # regret on fresh rows; cost
                    opt.append(cb[e] - ca[e])                                         # A's score at its own choice, against B's
                    f = int(np.argmin(r["stop"]))                                     # control: an epoch chosen without looking at A or B
                    ctl.append(cb[f] - ca[f])                                         # should average about zero
            d = np.array(opt) - np.array(ctl); se = d.std() / np.sqrt(len(d))        # paired: same halves, chosen epoch against fixed
            out[n].append((m, np.mean(rg), np.mean(opt), np.mean(ctl), np.mean(ep), d.mean()))  # means over 10 runs x 100 draws
            print(f"  {n:5d} rows, halves of {m:4d}: regret {np.mean(rg):+.4f}; optimism {np.mean(opt):+.4f}, control {np.mean(ctl):+.4f}, "
                  f"difference {d.mean():+.4f} (standard error {se:.5f}); epochs run {np.mean(ep):.1f}")
    print("  (optimism: B's loss minus A's at the epoch A chose; control: the same halves at an epoch chosen on the full stop set;")
    print("   difference: optimism minus control, draw by draw, which removes the luck of which rows fell in A)")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_online(D, T):                                  # section 5
    head("5. THE STOPPER RUN FOR REAL, AGAINST THE SAME RULE REPLAYED")                   # the banner
    for n in SIZES:                                                                   # each size
        Xa, ya = D["X"][D["train"][:n]], D["y"][D["train"][:n]]                       # seed 0's training rows
        Xs, ys = D["X"][D["stop"]], D["y"][D["stop"]]                                 # the stop set
        t0 = time.time()                                                              # a clock
        net, acts, curve = train_es(Xa, ya, Xs, ys, d126.Adam, LR, 0, EarlyStopper(5))  # for real, patience 5
        sec = time.time() - t0                                                        # its cost
        t0 = time.time(); train_es(Xa, ya, Xs, ys, d126.Adam, LR, 0, EarlyStopper(MAX_EPOCHS + 1))  # the same loop that never stops
        full = time.time() - t0                                                       # all 100 epochs, scored on the stop set only
        fr = row_losses(net, acts, D["X"][D["fresh"]], D["y"][D["fresh"]]).mean()     # the restored weights on fresh rows
        s, b = stop_at(T[n][0]["stop"], 5)                                            # the same rule on seed 0's recorded curve
        same = np.array_equal(curve, T[n][0]["stop"][:len(curve)])                    # identical losses, epoch by epoch?
        print(f"  {n:5d} rows: run for real, stopped after {len(curve)} epochs, best epoch {int(np.argmin(curve)) + 1}, fresh loss "
              f"{fr:.4f}; replayed: stopped after {s + 1}, best {b + 1}, fresh {T[n][0]['fresh'][b]:.4f}; curves identical: {same}")
        print(f"         time: {sec:.2f} s, against {full:.2f} s for the same loop with patience {MAX_EPOCHS + 1} (all {MAX_EPOCHS} epochs)")
    print("Actual output from this lesson's run.")


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(R):                                    # four panels from plain arrays
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); ax = ax.ravel()                   # two rows of two, read left to right
    for a, n in zip(ax[:2], SIZES):                                                   # one run's curves per training size
        r = R["T"][n][0]; ep = np.arange(1, MAX_EPOCHS + 1)                           # seed 0; epochs from 1
        a.plot(ep, r["train"], color="0.5", label="training rows")                    # falls for ever
        a.plot(ep, r["stop"], color="tab:blue", label="stop set (chooses)")           # turns and rises
        a.plot(ep, r["fresh"], color="tab:orange", label="fresh rows (scores)")       # the same shape, never used to choose
        s, b = stop_at(r["stop"], 5)                                                  # patience 5
        a.axvline(b + 1, color="tab:green", lw=1, label=f"best stop-set epoch ({b + 1})")  # the weights kept
        a.axvline(s + 1, color="tab:red", lw=1, ls="--", label=f"patience 5 stops ({s + 1})")  # where training ends
        a.set_ylim(0, 0.7); a.set_xlabel("epoch"); a.set_ylabel("log loss")           # axes
        a.set_title(f"{n} training rows, seed 0"); a.legend(fontsize=8)              # labels
    a = ax[2]                                                                         # rules: epochs against regret
    for n, mk in zip(SIZES, ("o", "s")):                                              # both sizes
        rows = [x for x in R["rules"][n] if x[0].startswith(("patience", "GL"))]      # the stopping rules
        a.plot([x[1] for x in rows], [x[3] for x in rows], mk, label=f"{n} rows")     # cost against regret
        lab = {}                                                                      # rules that land on the same point share one label
        for x in rows:                                                                # each rule
            lab.setdefault((round(x[1], 1), round(x[3], 5)), []).append(x[0].replace("patience ", "p"))  # grouped by position
        for (ex, ry), names in lab.items():                                           # label each point once
            a.annotate(", ".join(names), (ex, ry), fontsize=7, xytext=(3, 3), textcoords="offset points")  # label each point once
    a.margins(x=0.12); a.set_xlabel("epochs run (mean)"); a.set_ylabel("regret on fresh rows (mean)")  # room for labels; axes
    a.set_title("stopping rules: cost against regret"); a.legend(fontsize=8)          # labels
    a = ax[3]                                                                         # stop-set size
    for n, mk in zip(SIZES, ("o", "s")):                                              # both sizes
        S = R["stopset"][n]                                                           # (m, regret, optimism, control, epochs, difference)
        a.plot([x[0] for x in S], [x[5] for x in S], "-" + mk, label=f"optimism minus control, {n} rows")  # the winner's curse
        a.plot([x[0] for x in S], [x[1] for x in S], ":" + mk, label=f"regret, {n} rows")   # choosing the wrong epoch
    a.axhline(0, color="0.5", lw=0.8); a.set_xscale("log")                            # zero line; sizes on a log scale
    a.set_xlabel("rows in the half that chooses"); a.set_ylabel("log loss")          # axes
    a.set_title("a small stop set chooses worse and flatters itself"); a.legend(fontsize=8)  # labels
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "early_stopping_results.png"), dpi=110); plt.close(fig)  # save
    print("saved early_stopping_results.png")


def draw_diagram(r):                                    # one real run, annotated: best epoch, patience, stop, restore
    fig, ax = plt.subplots(figsize=(11, 5.2))                                         # one panel
    ep = np.arange(1, MAX_EPOCHS + 1); s, b = stop_at(r["stop"], 5)                   # epochs; patience 5 on this run
    ax.plot(ep, r["train"], color="0.55", label="training rows: keeps falling")       # memorising
    ax.plot(ep, r["stop"], color="tab:blue", lw=2, label="stop set: watched after every epoch")  # the signal
    ax.axvspan(b + 1, s + 1, color="tab:red", alpha=0.12, label="patience: 5 epochs without a new best")  # the waiting window
    ax.plot([b + 1], [r["stop"][b]], "o", color="tab:green", ms=8)                    # the best epoch
    ax.annotate("best epoch: its weights are kept", (b + 1, r["stop"][b]), xytext=(b + 12, r["stop"][b] - 0.12),  # the best epoch
                arrowprops=dict(arrowstyle="->"), fontsize=9)                         # restore
    ax.annotate("stop here, then go back", (s + 1, r["stop"][s]), xytext=(s + 14, r["stop"][s] + 0.12),  # the stopping epoch
                arrowprops=dict(arrowstyle="->"), fontsize=9)                         # stop
    ax.axvspan(s + 1, MAX_EPOCHS, color="0.85", alpha=0.5)                          # epochs early stopping never trains
    ax.text(60, 0.62, "never trained when stopping early: time saved\n(recorded here only to show what was avoided)",  # the epochs never trained
            fontsize=9, color="0.3", ha="center")                                    # the rest
    ax.set_ylim(0, 0.7); ax.set_xlabel("epoch"); ax.set_ylabel("log loss")            # axes
    ax.set_title(f"Day 171: early stopping on one real run ({SIZES[0]} Adult rows, seed 0)"); ax.legend(loc="lower left", fontsize=8)  # title and legend
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "early_stopping_diagram.png"), dpi=110); plt.close(fig)  # save
    print("saved early_stopping_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    tr, _ = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Week 24's data and split (the test file is not used)
    ref = {k: v[fi] for k, v in tr.items()}                                           # category codes and scaling from the fitting rows
    X = d161.onehot_matrix(tr, ref).astype(np.float32); y = tr["y"].astype(np.float32)  # standardised numbers and one-hot categories
    p = np.random.RandomState(0).permutation(fi)                                      # the fitting rows in a fixed random order
    D = dict(X=X, y=y, train=p[:-STOP_ROWS], stop=p[-STOP_ROWS:], fresh=va)           # training pool, stop set, fresh rows
    print(f"rows: {len(D['train'])} in the training pool, {STOP_ROWS} in the stop set, {len(va)} fresh")
    T = {n: [trajectory(D, n, s) for s in range(SEEDS)] for n in SIZES}               # 2 x 10 full runs of 100 epochs
    C = [trajectory(D, SIZES[0], s, shuffle=True) for s in range(SEEDS)]              # the shuffled-label control
    print(f"20 full runs and 10 control runs recorded in {time.time() - t0:.0f} s")
    demo_curves(D, T, C)                                                              # 1
    rules = demo_rules(D, T)                                                          # 2
    demo_restore(D, T)                                                                # 3
    stopset = demo_stopset(D, T)                                                      # 4
    demo_online(D, T)                                                                 # 5
    print()
    plot_results(dict(T=T, rules=rules, stopset=stopset))                             # plain arrays and dictionaries only
    draw_diagram(T[SIZES[0]][0])                                                      # the diagram
    print(f"total time {time.time() - t0:.0f} s")
    head("WHAT COMES NEXT")                                                               # the banner
    print("Day 172: cross-validation for time series and grouped data. Shuffled folds leak the future")
    print("and the same user into the test fold; a real leakage demo on CO2, electrical equipment and MovieLens.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
