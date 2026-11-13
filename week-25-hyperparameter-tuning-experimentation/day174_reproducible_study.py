"""
Day 174: Reproducibility -- Week 25's Build

One tuning study built from all six of this week's parts, then run again and
again in fresh Python processes to see what reproduces and what does not:

  Day 170  a learning-rate finder sets the range the learning rate is searched in
  Day 168  8 random trials in the unit cube, decoded onto each setting's scale
  Day 169  12 Bayesian-optimisation trials, each proposed from the log so far
  Day 172  every trial scored by 3-fold cross-validation (shuffled folds: the
           Adult rows are separate people, so shuffling is the right scheme)
  Day 171  every fold trained with early stopping on a separate stop set
  Day 173  every step tracked in a JSON Lines log, which is also what the
           Bayesian optimiser and a resumed study read their history from

Then the chosen settings are retrained on all 3,000 tuning rows and scored
once on the 6,032 validation rows and once on the official test file, which
no part of the week has touched before and which chooses nothing.

The experiments, each study in its own new Python process:
  1. the study itself
  2. the same study again from scratch: is every number identical, bit for bit?
  3. a study that crashes in its Bayesian phase, then resumes from its log
  4. what breaks it: another seed, a fold split with no seed, one BLAS thread
  5. pinning: the environment recorded and checked, and a line-ending trap
     in the code hash

Run it with `python day174_reproducible_study.py`. It starts its own child
processes (`--study NAME`) and writes their logs to day174_runs/. Every line
of code below carries its own comment, continuing Day 52's convention.
"""
import argparse                                        # argparse: the child process's options
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import hashlib                                         # hashlib: hashing the script's bytes with other line endings
import io                                              # io: StringIO, where that output goes
import json                                            # json: the final record and the environment file
import os                                              # os: paths and environment variables
import platform                                        # platform: the machine and Python version
import shutil                                          # shutil: removing an old run folder
import subprocess                                      # subprocess: each study runs in a new Python process
import sys                                             # sys: the import path and this Python's path
import time                                            # time: costs

import numpy as np                                     # numpy: every array operation below is built on this

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, HERE)                                # this week's folder: Days 168 to 173
sys.path.insert(0, os.path.join(ROOT, "week-24-model-evaluation-statistics"))        # Day 161's data
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day161_exercise as d161                      # Day 161: the Adult data and Week 24's split
    import day168_exercise as d168                      # Day 168: random_unit, decode, best_so_far
    import day169_exercise as d169                      # Day 169: propose (a GP and expected improvement)
    import day170_exercise as d170                      # Day 170: lr_range, lr_find, mlp_stepper, suggest
    import day171_exercise as d171                      # Day 171: EarlyStopper, train_es, row_losses
    import day172_exercise as d172                      # Day 172: shuffled_folds, cross_val
    import day173_exercise as d173                      # Day 173: fingerprint, file_hash, run_id, Tracker, query
d126 = d171.d126                                        # Day 126: Adam and the network

RUNS = os.path.join(HERE, "day174_runs")                # every study's log goes here
PIN = os.path.join(HERE, "day174_environment.json")     # the pinned environment
REQ = os.path.join(HERE, "day174_requirements.txt")     # pinned package versions, pip's format
CV_ROWS, STOP_ROWS = 3000, 6000                         # tuning rows (folded) and the stop set
K = 3                                                   # folds
N_RANDOM, N_BO = 8, 12                                  # Day 168's random trials, then Day 169's proposals
CANDIDATES = 2000                                       # random candidates the optimiser chooses each proposal from
FIND_STEPS = 300                                        # steps in the finder run, as on Day 170
CRASH_AT = 13                                           # the trial that crashes in study C (a Bayesian one)
LR_BELOW = 300                                          # the range reaches this far below the suggestion (see section 1)
THREADS = "4"                                           # the BLAS thread count the studies are pinned to


# =============================================================================
# ONE STUDY (runs in a child process)
# =============================================================================
def load_data():                                        # tuning rows, stop set, validation and test
    tr, te = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Week 24's data and split
    ref = {k: v[fi] for k, v in tr.items()}                                           # category codes and scaling from the fitting rows
    X = d161.onehot_matrix(tr, ref).astype(np.float32); y = tr["y"].astype(np.float32)  # 87 columns
    Xt = d161.onehot_matrix(te, ref).astype(np.float32); yt = te["y"].astype(np.float32)  # the official test file, coded the same way
    p = np.random.RandomState(0).permutation(fi)                                      # Day 171's fixed order
    return dict(Xc=X[p[:CV_ROWS]], yc=y[p[:CV_ROWS]], Xs=X[p[-STOP_ROWS:]], ys=y[p[-STOP_ROWS:]],  # the tuning rows and the stop set
                Xv=X[va], yv=y[va], Xt=Xt, yt=yt)                                     # eight arrays


def space_from(s, below=LR_BELOW):                      # the search space around the finder's suggestion s
    return (("lr", "log", s / below, s * 3), ("patience", "int", 1, 20), ("batch", "logint", 32, 256))  # Day 168's format


def cv_score(D, p, folds, seed):                        # Day 172's cross-validation of one configuration
    def fit_score(tr, te):                              # one fold: train with early stopping, score the held-out fold
        es = d171.EarlyStopper(p["patience"])                                         # Day 171's stopper
        net, acts, curve = d171.train_es(D["Xc"][tr], D["yc"][tr], D["Xs"], D["ys"], d126.Adam, p["lr"], seed, es,  # train on this fold's other rows
                                         batch=p["batch"])                            # train
        return float(d171.row_losses(net, acts, D["Xc"][te], D["yc"][te]).mean())    # held-out log loss
    return d172.cross_val(folds, fit_score)                                           # mean and per-fold scores


def trials_so_far(T):                                   # finished trials from the log, in trial order
    return sorted(d173.query(T.runs(), where=lambda r: r["params"].get("step") == "trial"), key=lambda r: r["params"]["trial"])  # sorted by trial number


def study(name, seed, crash_at=None, unseeded=False, below=LR_BELOW):  # the whole study, tracked; skips finished steps
    os.makedirs(RUNS, exist_ok=True)                                                  # the log folder, if it is missing
    T = d173.Tracker(os.path.join(RUNS, name + ".jsonl"))                             # Day 173's tracker
    D = load_data()                                                                   # the rows
    data = d173.fingerprint(*[D[k] for k in sorted(D)])                               # the data version
    code = d173.file_hash(__file__)                                                   # the code version
    fp = dict(step="lr_finder", steps=FIND_STEPS)                                     # Day 170: the finder, tracked like any run
    found = d173.query(T.runs(), where=lambda r: r["params"] == fp)                   # done already?
    if not found:                                                                     # no: run it
        with T.run(fp, seed, data, code) as r:                                        # tracked
            lrs, raw, sm = d170.lr_find(d170.mlp_stepper(D["Xc"], D["yc"], d126.Adam, seed), d170.lr_range(d170.LO, d170.HI, FIND_STEPS))  # the finder run
            r.summary.update(d170.suggest(lrs, sm), lrs=lrs.tolist(), sm=sm.tolist())  # its suggestions and curve
        found = d173.query(T.runs(), where=lambda r: r["params"] == fp)               # read it back from the log
    space = space_from(found[0]["summary"]["min_over_10"], below)                     # the search range for the learning rate
    folds = d172.shuffled_folds(CV_ROWS, K, None if unseeded else seed)               # Day 172: the same folds for every trial
    U0 = d168.random_unit(N_RANDOM, len(space), np.random.RandomState(seed))         # Day 168: the random trials' points
    for t in range(N_RANDOM + N_BO):                                                  # every trial
        done = {r["params"]["trial"]: r for r in trials_so_far(T)}                    # finished trials, from the log only
        if t in done:                                                                 # finished before (a resumed study)
            continue                                                                  # skip it
        hist = [done[i] for i in range(t)]                                            # the trials before this one, and only those
        if t < N_RANDOM:                                                              # random phase
            u = U0[t]                                                                 # the t-th random point
        else:                                                                         # Bayesian phase: Day 169
            X = np.array([r["params"]["u"] for r in hist]); y = np.array([r["summary"]["cv_loss"] for r in hist])  # trials before t
            C = np.random.RandomState(seed * 1000 + t).rand(CANDIDATES, len(space))  # this trial's candidates
            u = C[d169.propose(X, y, C)[0]]                                           # the one with the highest expected improvement
        p = dict(step="trial", trial=t, u=[float(x) for x in u], **d168.decode(u[None, :], space)[0])  # the configuration
        with T.run(p, seed, data, code) as r:                                         # tracked
            if t == crash_at:                                                         # the simulated crash
                raise RuntimeError(f"simulated crash in trial {t}")                   # logged as failed
            m, s = cv_score(D, p, folds, seed)                                        # Day 172 + Day 171
            r.summary.update(cv_loss=m, folds=s.tolist())                             # its scores
    best = min(trials_so_far(T), key=lambda r: r["summary"]["cv_loss"])["params"]     # the chosen configuration
    with T.run(dict(best, step="final"), seed, data, code) as r:                      # the final fit, tracked
        es = d171.EarlyStopper(best["patience"])                                      # early stopping again
        net, acts, curve = d171.train_es(D["Xc"], D["yc"], D["Xs"], D["ys"], d126.Adam, best["lr"], seed, es, batch=best["batch"])  # the chosen settings, on all tuning rows
        zt = d126.forward(net, D["Xt"], acts)[-1][:, 0]                               # test logits
        r.summary.update(val_loss=float(d171.row_losses(net, acts, D["Xv"], D["yv"]).mean()),  # validation log loss
                         test_loss=float(d171.row_losses(net, acts, D["Xt"], D["yt"]).mean()),  # test log loss
                         test_acc=float(np.mean((zt > 0) == (D["yt"] > 0.5))),       # test accuracy
                         weights=d173.fingerprint(*[a for W, b in net for a in (W, b)]),  # a hash of every final weight
                         best_epoch=es.best_epoch + 1)                                # where early stopping kept it


def child(argv):                                        # the entry point of a child process
    a = argparse.ArgumentParser()                                                     # its options
    a.add_argument("--study"); a.add_argument("--seed", type=int, default=0)          # name and study seed
    a.add_argument("--crash-at", type=int, default=None); a.add_argument("--unseeded", action="store_true")  # variants
    a.add_argument("--lr-below", type=float, default=LR_BELOW)                        # the first draft's narrower range
    o = a.parse_args(argv)                                                            # parse them
    study(o.study, o.seed, o.crash_at, o.unseeded, o.lr_below)                        # run it (a crash ends the process with an error)


# =============================================================================
# THE EXPERIMENTS (the parent process)
# =============================================================================
def launch(name, *args, threads=THREADS):               # one study in a new Python process, with a pinned environment
    env = dict(os.environ, OPENBLAS_NUM_THREADS=threads, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,  # the parent's variables, with BLAS threads
               PYTHONHASHSEED="0")                                                    # BLAS threads and hashing fixed
    t0 = time.time()                                                                  # a clock
    p = subprocess.run([sys.executable, __file__, "--study", name, *args], env=env, capture_output=True, text=True)  # run it
    return p.returncode, time.time() - t0, p.stderr.strip().split("\n")[-1]           # exit code, seconds, last error line


def outcome(name):                                      # what a study produced: (params, summary) of every finished step, in order
    T = d173.Tracker(os.path.join(RUNS, name + ".jsonl"))                             # its log
    R = d173.query(T.runs())                                                          # finished steps
    return {json.dumps(r["params"], sort_keys=True): r for r in R}                    # keyed by their settings


def compare(a, b):                                      # how many of a's steps b reproduced exactly
    A, B = outcome(a), outcome(b)                                                     # both studies
    keys = [k for k in A if json.loads(k).get("step") != "lr_finder"]                 # trials and the final fit
    same = [k for k in keys if k in B and B[k]["summary"] == A[k]["summary"]]         # same settings, same results to the bit
    return len(same), len(keys), A, B                                                 # counts and both outcomes


def max_gap(A, B):                                      # steps with the same settings: the largest difference in any score
    gaps = [abs(A[k]["summary"][m] - B[k]["summary"][m]) for k in A if k in B             # absolute differences
            for m in ("cv_loss", "val_loss", "test_loss") if m in A[k]["summary"]]   # every shared score
    return sum(k in B for k in A), max(gaps) if gaps else None                        # shared settings, and the largest gap


def final(O):                                           # a study's final record
    return next(r for k, r in O.items() if json.loads(k)["step"] == "final")          # the one final step


def head(title):                                        # a section banner
    print("\n" + "=" * 70); print(title); print("=" * 70)  # three lines


def environment():                                      # what the results may depend on, beyond code and data
    import matplotlib                                                                 # only for its version
    return dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,  # versions and machine
                machine=platform.machine(), system=platform.system(), blas_threads=THREADS, pythonhashseed="0")  # the facts


def demo_study():                                       # section 1
    head("1. THE STUDY: SIX PARTS, ONE TRACKED LOG")                                      # section 1
    code, sec, err = launch("A")                                                      # the reference study
    O = outcome("A"); F = final(O)                                                    # what it produced
    fd = next(r for k, r in O.items() if json.loads(k)["step"] == "lr_finder")["summary"]  # the finder
    s = fd["min_over_10"]                                                             # its suggestion
    print(f"  learning-rate finder (Day 170): min / 10 = {s:.2e}; learning rates searched from {s / LR_BELOW:.2e} to {s * 3:.2e}")
    tr = sorted([r for k, r in O.items() if json.loads(k)["step"] == "trial"], key=lambda r: r["params"]["trial"])  # trials
    sc = [r["summary"]["cv_loss"] for r in tr]                                        # their scores
    bs = d168.best_so_far(sc)                                                         # Day 168's running best
    print(f"  best 3-fold CV log loss after the {N_RANDOM} random trials {bs[N_RANDOM - 1]:.4f}; after all {len(sc)} trials {bs[-1]:.4f}")
    b = min(tr, key=lambda r: r["summary"]["cv_loss"])["params"]                      # the chosen configuration
    print(f"  chosen: trial {b['trial']} ({'random' if b['trial'] < N_RANDOM else 'Bayesian'}), lr {b['lr']:.2e}, "
          f"patience {b['patience']}, batch {b['batch']}")
    f = F["summary"]                                                                  # the final fit's results
    print(f"  final fit on all {CV_ROWS} tuning rows: stopped at best epoch {f['best_epoch']}; validation log loss "
          f"{f['val_loss']:.4f}; test log loss {f['test_loss']:.4f}, test accuracy {f['test_acc']:.4f}")
    print(f"  final weights fingerprint {f['weights']}; {len(O)} tracked steps; {sec:.0f} s in a new process (exit code {code})")
    launch("D30", "--lr-below", "30")                                                 # the first draft: a range only 30 times below
    O30 = outcome("D30"); f30 = final(O30)                                            # what it produced
    b30 = f30["params"]; c30 = min(r["summary"]["cv_loss"] for k, r in O30.items() if json.loads(k)["step"] == "trial")  # its choice
    print(f"  first draft (rates from {s / 30:.2e}): chose lr {b30['lr']:.2e}, patience {b30['patience']}, batch {b30['batch']}; "
          f"best CV {c30:.4f} against {bs[-1]:.4f}; best epoch {f30['summary']['best_epoch']}; validation "
          f"{f30['summary']['val_loss']:.4f}, test {f30['summary']['test_loss']:.4f}")
    print("Actual output from this lesson's run.")
    return O, sec                                                                     # for later sections and the figure


def demo_rerun():                                       # section 2
    head("2. THE SAME STUDY AGAIN, FROM SCRATCH, IN A NEW PROCESS")                       # section 2
    code, sec, err = launch("B")                                                      # the same command again
    n, m, A, B = compare("A", "B")                                                    # step by step
    print(f"  {n} of {m} steps identical to the bit (20 trials and the final fit); exit code {code}, {sec:.0f} s")
    ka = sorted(k for k in A if json.loads(k)["step"] == "trial")[:2]                 # two trials, to show the digits
    for k in ka:                                                                      # each
        print(f"  trial {json.loads(k)['trial']}: A {A[k]['summary']['cv_loss']!r}, B {B[k]['summary']['cv_loss']!r}")
    print(f"  final weights: A {final(A)['summary']['weights']}, B {final(B)['summary']['weights']}; "
          f"run ids identical: {[r['id'] for r in A.values()] == [B[k]['id'] for k in A]}")
    print("Actual output from this lesson's run.")


def demo_resume():                                      # section 3
    head(f"3. A STUDY THAT CRASHES IN TRIAL {CRASH_AT} (BAYESIAN PHASE), THEN RESUMES FROM ITS LOG")  # section 3
    code, sec, err = launch("C", "--crash-at", str(CRASH_AT))                         # crashes
    print(f"  first attempt: exit code {code}, last error line: {err}")
    T = d173.Tracker(os.path.join(RUNS, "C.jsonl"))                                   # its log
    print(f"  the log after the crash: {sum(r['status'] == 'finished' for r in T.runs())} finished steps "
          f"(the finder and trials 0 to {CRASH_AT - 1}), {sum(r['status'] == 'failed' for r in T.runs())} failed")
    code, sec, err = launch("C")                                                      # the same command without the crash
    n, m, A, C = compare("A", "C")                                                    # against the uninterrupted study
    print(f"  resumed: exit code {code}, {sec:.0f} s; {n} of {m} steps identical to study A, including the final weights "
          f"{final(C)['summary']['weights']}")
    print("Actual output from this lesson's run.")


def demo_breaks():                                      # section 4
    head("4. WHAT BREAKS IT: ANOTHER SEED, AN UNSEEDED FOLD SPLIT, ONE BLAS THREAD")      # section 4
    out = {}                                                                          # for the figure
    for name, args, th, what in (("S1", ("--seed", "1"), THREADS, "study seed 1 instead of 0"),  # name, options, threads, description
                                 ("U1", ("--unseeded",), THREADS, "folds with no seed, first run"),  # no seed for the folds
                                 ("U2", ("--unseeded",), THREADS, "folds with no seed, second run"),  # the same again
                                 ("T1", (), "1", "seed 0, 1 BLAS thread instead of 4")):  # four variants
        code, sec, err = launch(name, *args, threads=th)                              # run it
        n, m, A, V = compare("A", name)                                               # against A
        f, fa = final(V)["summary"], final(A)["summary"]                              # final results
        b = json.loads(next(k for k in V if json.loads(k)["step"] == "final"))        # its chosen configuration
        print(f"  {name} ({what}): {n} of {m} of A's steps reproduced; chose lr {b['lr']:.2e}, patience {b['patience']}, "
              f"batch {b['batch']}; test log loss {f['test_loss']:.4f} (A {fa['test_loss']:.4f}), weights {f['weights']}")
        sh, g = max_gap(A, V)                                                         # did anything match apart from the last bits?
        print(f"      steps with A's exact settings: {sh} of {len(A)}; largest score difference among them: "
              f"{'none shared' if g is None else f'{g:.1e}'}")
        out[name] = V                                                                 # keep it
    n, m, U1, U2 = compare("U1", "U2")                                                # the two unseeded runs against each other
    print(f"  U1 against U2: {n} of {m} steps identical; the same command gave two different studies")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_pinning():                                     # section 5
    head("5. PINNING: THE ENVIRONMENT RECORDED AND CHECKED, AND A LINE-ENDING TRAP")      # section 5
    env = environment()                                                               # this machine
    if not os.path.exists(PIN):                                                       # the first run anywhere: pin it
        json.dump(env, open(PIN, "w", encoding="utf-8"), indent=1, sort_keys=True)    # write the pin file
        print(f"  wrote {os.path.basename(PIN)}")
    pinned = json.load(open(PIN, encoding="utf-8"))                                   # read the pin
    diff = {k: (pinned.get(k), v) for k, v in env.items() if pinned.get(k) != v}       # what differs from it
    print(f"  pinned: {', '.join(f'{k} {v}' for k, v in sorted(pinned.items()))}")
    print(f"  this machine against the pin: {'no differences' if not diff else diff}")
    with open(REQ, "w", encoding="utf-8") as f:                                       # pip's format, for reinstalling
        f.write(f"numpy=={env['numpy']}\nmatplotlib=={env['matplotlib']}\n")          # the two packages used
    print(f"  {os.path.basename(REQ)}: numpy=={env['numpy']}, matplotlib=={env['matplotlib']} (python {env['python']})")
    rec = json.loads(open(os.path.join(RUNS, "A.jsonl"), encoding="utf-8").readline())  # the first record study A wrote
    print(f"  every start record carries its environment too, e.g. study A's first: {rec['env']}")
    raw = open(__file__, "rb").read()                                                 # this script's bytes, as they are on disk
    lf = raw.replace(bytes([13, 10]), bytes([10])); crlf = lf.replace(bytes([10]), bytes([13, 10]))  # LF (Unix) and CRLF (Windows) versions
    h = lambda b: hashlib.sha256(b).hexdigest()[:12]                                  # Day 173's hash, on bytes
    print(f"  code hash of this script on disk {h(raw)} (the hash every record carries); with LF line endings {h(lf)}, "
          f"with CRLF line endings {h(crlf)}: the same code, two hashes")
    print("Actual output from this lesson's run.")


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(A, V):                                 # four panels from the logs
    import matplotlib                                                                 # only for figures
    matplotlib.use("Agg")                                                             # draw to files, no window
    import matplotlib.pyplot as plt                                                   # the plotting API
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); ax = ax.ravel()                   # two rows of two, read left to right
    fd = next(r for k, r in A.items() if json.loads(k)["step"] == "lr_finder")["summary"]  # the finder's curve
    a = ax[0]; a.plot(fd["lrs"], fd["sm"], color="tab:blue"); a.set_xscale("log")     # smoothed loss against rate
    s = fd["min_over_10"]; a.axvspan(s / LR_BELOW, s * 3, color="tab:green", alpha=0.15, label="searched range")  # the range
    a.axvspan(s / 30, s * 3, fill=False, hatch="//", ec="0.5", lw=0, label="first draft's range")  # the narrower first draft
    a.axvline(s, color="tab:red", ls="--", label=f"min / 10 = {s:.1e}"); a.legend()  # the suggestion
    a.set_xlabel("learning rate"); a.set_ylabel("smoothed training loss"); a.set_title("Day 170's finder sets the search range")  # axes and title
    a = ax[1]                                                                         # the trials of each study
    for name, O, c, mk in (("A", A, "black", "o"), ("B", None, "tab:orange", "x"), ("S1", V["S1"], "tab:green", "s"),  # each study: its style
                           ("U1", V["U1"], "tab:purple", "^"), ("T1", V["T1"], "tab:red", "+")):  # A and its variants
        O = outcome(name) if O is None else O                                         # B is read here
        tr = sorted([r for k, r in O.items() if json.loads(k)["step"] == "trial"], key=lambda r: r["params"]["trial"])  # its trials in order
        a.plot([r["params"]["trial"] for r in tr], [r["summary"]["cv_loss"] for r in tr], marker=mk, ls="none", color=c,  # one point per trial
               ms=8 if mk in "x+" else 6, label=name, mfc="none" if mk in "os^" else None)  # one marker per trial
    a.axvline(N_RANDOM - 0.5, color="0.5", ls=":"); a.text(N_RANDOM - 0.3, a.get_ylim()[1], "Bayesian", va="top", fontsize=9)  # where the Bayesian phase starts
    a.set_xlabel("trial"); a.set_ylabel("3-fold CV log loss"); a.legend(fontsize=8)   # axes
    a.set_title("trials: B lands exactly on A; T1 differs below the 6th decimal")     # label
    a = ax[2]; names = ["A", "B", "C", "S1", "U1", "U2", "T1"]                        # every study
    vals = [final(outcome(n))["summary"]["test_loss"] for n in names]                 # their test log loss
    a.bar(names, vals, color=["black", "tab:orange", "tab:blue", "tab:green", "tab:purple", "tab:purple", "tab:red"])  # one bar per study
    a.set_ylim(min(vals) - 0.002, max(vals) + 0.002); a.set_ylabel("test log loss")   # a close-up of the differences
    a.set_title("the final model's test log loss, per study")                         # label
    a = ax[3]; same = [compare("A", n)[0] for n in names[1:]]                         # how many steps reproduced
    sett = [sum(k in outcome(n) for k in outcome("A") if json.loads(k)["step"] != "lr_finder")  # count A's trials and final fit found in study n
            for n in names[1:]]                                                       # steps with A's exact settings
    x = np.arange(len(names) - 1)                                                     # bar positions
    a.bar(x - 0.2, sett, 0.4, color="tab:blue", label="same settings as A")           # same configurations tried
    a.bar(x + 0.2, same, 0.4, color="tab:gray", label="identical to the bit")         # same results, every bit
    a.set_xticks(x, names[1:]); a.axhline(N_RANDOM + N_BO + 1, color="black", ls="--"); a.legend()  # out of 21
    a.set_ylabel("steps (of 21)"); a.set_title("what reproduced A")                   # labels
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "reproducibility_results.png"), dpi=110); plt.close(fig)  # save
    print("saved reproducibility_results.png")


def draw_diagram():                                     # the study's pipeline and the reproducibility checks
    import matplotlib                                                                 # only for figures
    matplotlib.use("Agg")                                                             # draw to files, no window
    import matplotlib.pyplot as plt                                                   # the plotting API
    fig, ax = plt.subplots(figsize=(12, 5.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5.6)  # a blank canvas
    box = lambda x, y, w, h, t, c: (ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="black")),  # a box and its centred text
                                    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9))  # a labelled box
    steps = (("Day 170\nLR finder\nsets lr range", "#dbe9f6"), ("Day 168\n8 random\ntrials", "#dbe9f6"),  # the six steps
             ("Day 169\n12 Bayesian\ntrials", "#dbe9f6"), ("Day 172\n3-fold CV\nper trial", "#fde9d9"),  # two more
             ("Day 171\nearly stopping\nper fold", "#fde9d9"), ("final fit\nvalidation +\ntest once", "#e2f0d9"))  # the pipeline
    for i, (t, c) in enumerate(steps):                                                # each box in a row
        box(0.2 + i * 1.95, 3.4, 1.65, 1.4, t, c)                                     # evenly spaced
        if i:                                                                         # an arrow from the previous box
            ax.annotate("", (0.2 + i * 1.95, 4.1), (0.2 + i * 1.95 - 0.3, 4.1), arrowprops=dict(arrowstyle="->"))  # an arrow
    box(0.2, 2.2, 11.5, 0.75, "Day 173 tracker: every step writes start + end records to day174_runs/NAME.jsonl; "  # the log's box
        "the optimiser and a resumed study read their history from it", "#fff2cc")    # the log under everything
    checks = (("B: same command,\nnew process\n-> identical?", "#f2f2f2"), ("C: crash in trial 13,\nresume from log\n-> identical?", "#f2f2f2"),  # the checks
              ("S1: seed 1\nU1, U2: unseeded folds\nT1: 1 BLAS thread", "#f2f2f2"), ("pin: environment file,\nrequirements, LF vs\nCRLF code hash", "#f2f2f2"))  # two more
    for i, (t, c) in enumerate(checks):                                               # the checks along the bottom
        box(0.2 + i * 2.95, 0.3, 2.65, 1.4, t, c)                                     # evenly spaced
    ax.set_title("Day 174: one study from six parts, and the checks that it reproduces")  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "reproducibility_diagram.png"), dpi=110); plt.close(fig)  # save
    print("saved reproducibility_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    if os.path.exists(RUNS):                                                          # an earlier run's logs
        shutil.rmtree(RUNS)                                                           # start every study fresh
    os.makedirs(RUNS)                                                                 # an empty folder
    A, sec = demo_study()                                                             # 1
    demo_rerun()                                                                      # 2
    demo_resume()                                                                     # 3
    V = demo_breaks()                                                                 # 4
    demo_pinning()                                                                    # 5
    print()
    plot_results(A, V)                                                                # the figure
    draw_diagram()                                                                    # the diagram
    print(f"total time {time.time() - t0:.0f} s")
    head("WHAT COMES NEXT")                                                               # the last banner
    print("Week 26, MLOps I: serving models. Day 175 compares real ways of saving and loading a trained model,")
    print("starting from a model like this one, whose weights must load back identical to the bit.")


if __name__ == "__main__":                              # only when run directly, not when imported
    if "--study" in sys.argv:                                                         # a child process
        child(sys.argv[1:])                                                           # run one study
    else:                                                                             # the parent
        main()                                                                        # run everything
