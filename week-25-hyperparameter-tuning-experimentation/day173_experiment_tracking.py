"""
Day 173: Experiment Tracking

A tuning study is dozens of runs. A month later the questions are always the
same: which settings gave that number? On which data, with which code, which
seed? Can it be reproduced? Experiment tracking answers them by writing every
run down automatically. Today builds a small tracker from scratch (no MLflow)
and puts it to work on a real study, Day 171's network on UCI Adult:

  1. a tracked random search of 40 configurations that crashes at run 17,
     and resumes without redoing the 16 finished runs
  2. querying the log: the best runs, and what each setting did
  3. reproducing logged runs from their records alone, bit for bit
  4. the seed: rerunning the top five configurations with ten new seeds
     each; how much of the winner's score was its seed?
  5. fingerprints: a silently changed dataset and a changed line of code,
     caught by their hashes; and what tracking costs in time

The log is JSON Lines (one JSON record per line, appended, never rewritten),
saved next to this script as day173_runs.jsonl. Data: Week 24's split of UCI
Adult, as on Day 171: 2,000 training rows, a 6,000-row stop set for early
stopping, and the 6,032 validation rows, which choose. The test file is not
touched. Every line of code below carries its own comment, continuing Day
52's convention.
"""
import contextlib                                      # contextlib: redirect_stdout, to keep imported checks quiet
import hashlib                                         # hashlib: sha256 fingerprints
import io                                              # io: StringIO, where that output goes
import json                                            # json: one record per line
import os                                              # os: file paths
import platform                                        # platform: the Python version, for the environment record
import sys                                             # sys: the import path
import time                                            # time: timestamps and costs

import numpy as np                                     # numpy: every array operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saved figures
matplotlib.use("Agg")                                  # draw to files, no window
import matplotlib.pyplot as plt                         # the plotting API

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
ROOT = os.path.dirname(HERE)                            # the repository
sys.path.insert(0, HERE)                                # this week's folder: Days 168 to 172
sys.path.insert(0, os.path.join(ROOT, "week-24-model-evaluation-statistics"))        # Day 161's data
with contextlib.redirect_stdout(io.StringIO()):         # each exercise prints its hand checks when imported
    import day161_exercise as d161                      # Day 161: the Adult data and Week 24's split
    import day171_exercise as d171                      # Day 171: EarlyStopper, train_es, row_losses (and Days 170 and 126 through it)
d126 = d171.d126                                        # Day 126: Adam

LOG = os.path.join(HERE, "day173_runs.jsonl")           # the study's log
N_RUNS = 40                                             # configurations in the random search
CRASH_AT = 17                                           # the run that fails the first time
TRAIN_ROWS, STOP_ROWS = 2000, 6000                      # as on Day 171
TOP, NEW_SEEDS = 5, 10                                  # configurations rerun, and new seeds each


# =============================================================================
# THE PARTS (copied unchanged into day173_exercise.py, for Day 174's build)
# =============================================================================
def fingerprint(*arrays):                               # a short hash of arrays' shapes, types and bytes
    h = hashlib.sha256()                                                              # a fresh hash
    for a in arrays:                                                                  # each array in turn
        a = np.ascontiguousarray(a); h.update(f"{a.shape}{a.dtype}".encode()); h.update(a.tobytes())  # shape, type, contents
    return h.hexdigest()[:12]                                                         # 12 hex digits are plenty to tell versions apart


def file_hash(path):                                    # a short hash of a file's bytes: the code version
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]                   # any edit changes it


def run_id(params, seed, data, code):                   # the same configuration on the same data and code -> the same id
    key = json.dumps(dict(params=params, seed=seed, data=data, code=code), sort_keys=True)  # a canonical text form
    return hashlib.sha256(key.encode()).hexdigest()[:12]                              # its hash


class Run:                                              # one run: `with tracker.run(...) as r:` records its start, metrics and end
    def __init__(self, tracker, params, seed, data, code):  # everything that defines the run
        self.t, self.rec = tracker, dict(id=run_id(params, seed, data, code), params=params, seed=seed, data=data, code=code)  # the tracker, and the run's defining record with its id
        self.history, self.summary = [], {}                                           # per-step metrics; final metrics

    def log(self, step, **metrics):                     # a metric at one step (here: one epoch)
        self.history.append(dict(step=step, **{k: float(v) for k, v in metrics.items()}))  # plain floats for JSON

    def __enter__(self):                                # the run starts: write a start record at once
        self.t0 = time.time()                                                         # a clock
        self.t.write(dict(event="start", time=round(self.t0, 3), env=self.t.env, **self.rec))  # so a crash still leaves a trace
        return self                                                                   # the object used inside the with block

    def __exit__(self, kind, err, tb):                  # the run ends, normally or by an exception
        self.t.write(dict(event="end", id=self.rec["id"], status="failed" if kind else "finished",  # the end record: the same id
                          error=repr(err) if kind else None, seconds=round(time.time() - self.t0, 3),  # the error text if it failed; the time taken
                          summary=self.summary, history=self.history))                # the outcome, whatever it was
        return False                                                                  # do not swallow the exception


class Tracker:                                          # an append-only JSON Lines log of runs
    def __init__(self, path):                           # the log file; created on the first write
        self.path = path                                                              # where records go
        self.env = dict(python=platform.python_version(), numpy=np.__version__)       # the environment, recorded with every run

    def write(self, record):                            # append one record as one line
        with open(self.path, "a", encoding="utf-8") as f:                             # append: earlier lines are never rewritten
            f.write(json.dumps(record, sort_keys=True) + "\n")                        # one JSON object per line

    def run(self, params, seed, data, code):            # a new run, for a with block
        return Run(self, params, seed, data, code)                                    # see Run

    def runs(self):                                     # every run, start and end records merged; the latest attempt wins
        out = {}                                                                      # id -> merged record
        if not os.path.exists(self.path):                                             # nothing logged yet
            return []                                                                 # no runs
        for line in open(self.path, encoding="utf-8"):                                # one record per line
            r = json.loads(line)                                                      # parse it
            if r["event"] == "start":                                                 # a run began
                out[r["id"]] = dict(r, status="unfinished")                           # unfinished until its end record appears
            else:                                                                     # a run ended
                out[r["id"]].update(status=r["status"], error=r["error"], seconds=r["seconds"],  # merge the outcome into the start record
                                    summary=r["summary"], history=r["history"])       # merge the outcome
        return list(out.values())                                                     # in order of first appearance

    def done(self, rid):                                # has this exact run already finished?
        return any(r["id"] == rid and r["status"] == "finished" for r in self.runs())  # then a resumed study can skip it


def query(runs, where=lambda r: True, by=None, n=None):  # filter runs, sort by a summary metric (smallest first), keep n
    out = [r for r in runs if r["status"] == "finished" and where(r)]                # finished runs that match
    if by is not None:                                                                # sort if asked
        out.sort(key=lambda r: r["summary"][by])                                      # e.g. by validation loss
    return out[:n] if n else out                                                      # the first n, or all


# =============================================================================
# THE STUDY
# =============================================================================
def load_data():                                        # Day 171's rows: train, stop set, validation (which chooses)
    tr, _ = d161.load_adult(); fi, va = d161.split_rows(len(tr["y"]))               # Week 24's data and split
    ref = {k: v[fi] for k, v in tr.items()}                                           # category codes and scaling from the fitting rows
    X = d161.onehot_matrix(tr, ref).astype(np.float32); y = tr["y"].astype(np.float32)  # 87 columns
    p = np.random.RandomState(0).permutation(fi)                                      # Day 171's fixed order
    return dict(Xa=X[p[:TRAIN_ROWS]], ya=y[p[:TRAIN_ROWS]], Xs=X[p[-STOP_ROWS:]], ys=y[p[-STOP_ROWS:]], Xv=X[va], yv=y[va])  # six arrays


def sample_configs(n, seed=0):                          # the random search's configurations, drawn once
    rng = np.random.RandomState(seed)                                                 # reproducible
    return [dict(lr=float(10 ** rng.uniform(-4, -1.5)), patience=int(rng.randint(1, 21)),  # a learning rate between 1e-4 and 10**-1.5
                 batch=int(rng.choice([32, 64, 128, 256]))) for _ in range(n)]       # log-uniform rate, patience 1-20, batch


def train_and_score(D, p, seed, run=None):              # one configuration and seed: train with early stopping, score validation
    es = d171.EarlyStopper(p["patience"])                                             # Day 171's stopper
    net, acts, curve = d171.train_es(D["Xa"], D["ya"], D["Xs"], D["ys"], d126.Adam, p["lr"], seed, es, batch=p["batch"])  # train
    val = float(d171.row_losses(net, acts, D["Xv"], D["yv"]).mean())                 # validation log loss of the restored weights
    if run is not None:                                                               # tracked: record the curve and the result
        for e, v in enumerate(curve):                                                 # every epoch
            run.log(e + 1, stop_loss=v)                                               # the stop set's loss
        run.summary.update(val_loss=val, best_epoch=es.best_epoch + 1, epochs=len(curve))  # the final numbers
    return val, es.best_epoch + 1, len(curve)                                         # for untracked reruns


def study(T, D, configs, data, code, crash_at=None):    # the tracked random search; skips runs already finished
    ran = skipped = 0                                                                 # counts
    for k, p in enumerate(configs):                                                   # each configuration, seed 0
        if T.done(run_id(p, 0, data, code)):                                          # already in the log as finished
            skipped += 1; continue                                                    # do not redo it
        with T.run(p, 0, data, code) as r:                                            # start record written now
            if k + 1 == crash_at:                                                     # the simulated crash
                raise RuntimeError(f"simulated crash in run {k + 1}")                 # the end record says failed
            train_and_score(D, p, 0, r); ran += 1                                     # train, log, finish
    return ran, skipped                                                               # what this call did


def head(title):                                        # a section banner
    print("\n" + "=" * 70); print(title); print("=" * 70)  # three lines


def demo_study(T, D, configs, data, code):              # section 1
    head("1. A TRACKED RANDOM SEARCH THAT CRASHES, THEN RESUMES")                         # section 1
    t0 = time.time()                                                                  # a clock
    try:                                                                              # the first attempt
        study(T, D, configs, data, code, crash_at=CRASH_AT)                           # stops at the crash
    except RuntimeError as e:                                                         # the crash
        print(f"  first attempt stopped: {e}")
    R = T.runs()                                                                      # what the log knows now
    print(f"  log after the crash: {sum(r['status'] == 'finished' for r in R)} finished, "
          f"{sum(r['status'] == 'failed' for r in R)} failed ({[r['error'] for r in R if r['status'] == 'failed'][0]})")
    ran, skipped = study(T, D, configs, data, code)                                   # resume: no crash this time
    R = T.runs()                                                                      # the log again
    print(f"  resumed: {skipped} runs skipped as already finished, {ran} run now; the log holds {len(R)} runs, "
          f"{sum(r['status'] == 'finished' for r in R)} finished")
    print(f"  {sum(1 for _ in open(LOG, encoding='utf-8'))} lines in {os.path.basename(LOG)}, "
          f"{os.path.getsize(LOG) / 1024:.0f} KB; data {data}, code {code}; ({time.time() - t0:.0f} s)")
    print("Actual output from this lesson's run.")
    return R                                                                          # for the next sections


def demo_query(R):                                      # section 2
    head("2. QUERYING THE LOG")                                                           # section 2
    print("  the five best runs by validation log loss:")
    for r in query(R, by="val_loss", n=5):                                            # the top five
        p, s = r["params"], r["summary"]                                              # settings and results
        print(f"    {r['id']}  lr {p['lr']:.2e}  patience {p['patience']:2d}  batch {p['batch']:3d}  ->  val {s['val_loss']:.4f}, "
              f"best epoch {s['best_epoch']}, {s['epochs']} epochs, {r['seconds']:.2f} s")
    F = query(R)                                                                      # every finished run
    v = np.array([r["summary"]["val_loss"] for r in F])                               # their scores
    for name, f in (("log10 lr", lambda r: np.log10(r["params"]["lr"])), ("patience", lambda r: r["params"]["patience"]),  # three settings, each as a number
                    ("log2 batch", lambda r: np.log2(r["params"]["batch"]))):        # each setting
        x = np.array([f(r) for r in F]); c = np.corrcoef(x, v)[0, 1]                 # its correlation with the score
        print(f"  {name:10s}: correlation with validation loss {c:+.2f}")
    slow = query(R, where=lambda r: r["params"]["lr"] < 3e-4)                         # a question asked after the fact
    print(f"  runs with lr < 3e-4: {len(slow)}, mean epochs {np.mean([r['summary']['epochs'] for r in slow]):.1f}, "
          f"mean val {np.mean([r['summary']['val_loss'] for r in slow]):.4f}; the rest: mean val "
          f"{np.mean([r['summary']['val_loss'] for r in F if r['params']['lr'] >= 3e-4]):.4f}")
    print("Actual output from this lesson's run.")


def demo_reproduce(R, D):                               # section 3
    head("3. REPRODUCING RUNS FROM THEIR RECORDS ALONE")                                  # section 3
    picks = query(R, by="val_loss")[::8]                                              # every eighth finished run, best first
    same = 0                                                                          # exact matches
    for r in picks:                                                                   # each chosen record
        val, best, ep = train_and_score(D, r["params"], r["seed"])                    # rerun from the record's settings and seed
        ok = val == r["summary"]["val_loss"] and best == r["summary"]["best_epoch"] and ep == r["summary"]["epochs"]  # exact?
        same += ok                                                                    # count
        print(f"  {r['id']}: logged val {r['summary']['val_loss']!r}, rerun {val!r}; identical: {ok}")
    print(f"  {same} of {len(picks)} reproduced bit for bit (Python {platform.python_version()}, numpy {np.__version__})")
    print("Actual output from this lesson's run.")


def demo_seeds(R, D):                                   # section 4
    head(f"4. THE SEED: THE TOP {TOP} CONFIGURATIONS, {NEW_SEEDS} NEW SEEDS EACH")        # section 4
    top = query(R, by="val_loss", n=TOP); out = []                                    # the five best by their single seed-0 run
    for k, r in enumerate(top):                                                       # each
        v = np.array([train_and_score(D, r["params"], s)[0] for s in range(1, NEW_SEEDS + 1)])  # seeds 1 to 10
        out.append((r["summary"]["val_loss"], v))                                     # logged score and the new ones
        print(f"  rank {k + 1} ({r['id']}): logged {r['summary']['val_loss']:.4f}; new seeds mean {v.mean():.4f} "
              f"(sd {v.std(ddof=1):.4f}, {v.min():.4f} to {v.max():.4f})")
    means = [v.mean() for _, v in out]                                                # the configurations' new-seed means
    print(f"  the logged winner's score {out[0][0]:.4f} against its new-seed mean {means[0]:.4f}: {means[0] - out[0][0]:+.4f}")
    print(f"  ranked by new-seed means, the logged ranks 1 to {TOP} become: {(np.argsort(means) + 1).tolist()}")
    mid = query(R, by="val_loss")[17:22]; gaps = []                                   # a control: ranks 18 to 22, not selected for a good score
    for r in mid:                                                                     # each
        v = np.array([train_and_score(D, r["params"], s)[0] for s in range(1, NEW_SEEDS + 1)])  # the same ten new seeds
        gaps.append(v.mean() - r["summary"]["val_loss"])                              # new-seed mean minus the logged seed-0 score
    top_gaps = [v.mean() - lg for lg, v in out]                                       # the same for the top five
    print(f"  new-seed mean minus logged score: top {TOP} {np.round(top_gaps, 4).tolist()}; control ranks 18-22 "
          f"{np.round(gaps, 4).tolist()}")
    print(f"  means: top {np.mean(top_gaps):+.4f}, control {np.mean(gaps):+.4f}; seed 0 helped the unselected controls too, "
          f"and the top {TOP} gain {np.mean(top_gaps) - np.mean(gaps):+.4f} more from being chosen")
    print("Actual output from this lesson's run.")
    return out                                                                        # for the figure


def demo_fingerprints(T, D, configs, data, code):       # section 5
    head("5. FINGERPRINTS: A SILENTLY CHANGED DATASET, A CHANGED LINE OF CODE, AND THE COST")  # section 5
    D2 = dict(D); keep = np.random.RandomState(5).rand(TRAIN_ROWS) > 0.01             # about 1% of training rows quietly dropped
    D2["Xa"], D2["ya"] = D["Xa"][keep], D["ya"][keep]                                 # the 'same' dataset, a little different
    data2 = fingerprint(D2["Xa"], D2["ya"], D2["Xs"], D2["ys"], D2["Xv"], D2["yv"])   # its fingerprint
    p = query(T.runs(), by="val_loss", n=1)[0]["params"]                              # the best configuration
    with T.run(p, 0, data2, code) as r:                                               # tracked on the changed data
        train_and_score(D2, p, 0, r)                                                  # same settings, same seed
    a, b = [x for x in T.runs() if x["params"] == p and x["seed"] == 0]               # the two runs of that configuration
    print(f"  {TRAIN_ROWS - keep.sum()} of {TRAIN_ROWS} training rows dropped: data {a['data']} -> {b['data']}; same settings and seed, "
          f"val {a['summary']['val_loss']:.4f} -> {b['summary']['val_loss']:.4f}; run ids differ: {a['id'] != b['id']}")
    X3 = D["Xa"].copy(); X3[0, 0] += 1e-6                                              # one number changed by a millionth
    print(f"  one value changed by 1e-6: fingerprint {fingerprint(X3, D['ya'], D['Xs'], D['ys'], D['Xv'], D['yv'])} "
          f"against {data}")
    src = open(__file__, "rb").read(); src2 = src.replace(b"TOP, NEW_SEEDS = 5, 10", b"TOP, NEW_SEEDS = 5, 11")  # a one-character edit
    print(f"  this script {code}; with one constant changed {hashlib.sha256(src2).hexdigest()[:12]}")
    F = query(T.runs())                                                               # every finished run
    t0 = time.time()                                                                  # time the tracker's own work
    for _ in range(20):                                                               # 20 repetitions
        T.runs(); T.done("0" * 12)                                                    # read and search the whole log
    rd = (time.time() - t0) / 20                                                      # per read
    print(f"  cost: a run took {np.mean([r['seconds'] for r in F]):.2f} s on average; reading and searching the "
          f"{len(T.runs())}-run log took {rd * 1000:.1f} ms")
    print("Actual output from this lesson's run.")


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(R, seeds):                             # four panels from plain values
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); ax = ax.ravel()                   # two rows of two, read left to right
    F = query(R)                                                                      # finished runs
    v = np.array([r["summary"]["val_loss"] for r in F])                               # scores
    a = ax[0]; sc = a.scatter([r["params"]["lr"] for r in F], v, c=[r["params"]["patience"] for r in F], cmap="viridis")  # rate
    a.set_xscale("log"); a.set_xlabel("learning rate"); a.set_ylabel("validation log loss")  # axes
    a.set_title("40 tracked runs (colour: patience)"); fig.colorbar(sc, ax=a)         # labels
    a = ax[1]                                                                         # stop-set curves of the five best
    for r in query(R, by="val_loss", n=5):                                            # from their logged histories
        h = r["history"]; a.plot([x["step"] for x in h], [x["stop_loss"] for x in h], label=f"{r['id']} (lr {r['params']['lr']:.1e})")  # this run's logged curve
    a.set_xlabel("epoch"); a.set_ylabel("stop-set log loss"); a.set_title("curves read back from the log: the five best")  # axes and title
    a.legend(fontsize=8)                                                              # labels
    a = ax[2]                                                                         # seeds
    for k, (lg, nv) in enumerate(seeds):                                              # each of the top five
        a.scatter(np.full(len(nv), k + 1), nv, color="tab:blue", alpha=0.6, s=18)     # new seeds
        a.plot([k + 0.75, k + 1.25], [lg, lg], color="tab:red", lw=2)                 # the logged seed-0 score
        a.plot([k + 0.8, k + 1.2], [nv.mean(), nv.mean()], color="black", lw=1)       # the new-seed mean
    a.set_xlabel("logged rank"); a.set_ylabel("validation log loss")                  # axes
    a.set_title("red: the logged score; blue: 10 new seeds; black: their mean")        # labels
    a = ax[3]; t = [r["seconds"] for r in F]                                          # cost per run
    a.hist(t, bins=15, color="tab:gray"); a.set_xlabel("seconds per run"); a.set_ylabel("runs")  # the distribution
    a.set_title("what each logged run cost")                                          # label
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "tracking_results.png"), dpi=110); plt.close(fig)  # save
    print("saved tracking_results.png")


def draw_diagram():                                     # what a tracker records, and the crash-and-resume flow
    fig, ax = plt.subplots(figsize=(11, 5.4)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.4)  # a blank canvas
    box = lambda x, y, w, h, t, c: (ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="black")),  # a box and its centred text
                                    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9))  # a labelled box
    box(0.2, 3.3, 2.4, 1.5, "a run\nsettings, seed\ndata, code", "#dbe9f6")           # what defines a run
    box(3.4, 3.3, 2.6, 1.5, "run id = hash of\nsettings + seed +\ndata + code", "#fde9d9")  # its identity
    box(6.8, 3.3, 3.9, 1.5, "day173_runs.jsonl (append only)\nstart: id, settings, seed,\ndata, code, Python, numpy\n"  # the log's box
        "end: status, seconds, metrics", "#e2f0d9")                                    # the log
    for x0, x1 in ((2.6, 3.4), (6.0, 6.8)):                                           # arrows between them
        ax.annotate("", (x1, 4.05), (x0, 4.05), arrowprops=dict(arrowstyle="->"))         # an arrow
    box(0.2, 0.5, 3.0, 1.8, "first attempt\nruns 1-16 finished\nrun 17 failed\n(start + end: failed)", "#f2f2f2")  # crash
    box(4.0, 0.5, 3.0, 1.8, "resume\nskip ids already finished\nrerun 17 onwards", "#f2f2f2")  # resume
    box(7.8, 0.5, 2.9, 1.8, "afterwards\nquery, compare,\nreproduce from records", "#f2f2f2")  # use
    for x0, x1 in ((3.2, 4.0), (7.0, 7.8)):                                           # arrows
        ax.annotate("", (x1, 1.4), (x0, 1.4), arrowprops=dict(arrowstyle="->"))           # an arrow
    ax.annotate("", (5.5, 2.3), (8.7, 3.3), arrowprops=dict(arrowstyle="->", color="0.4"))  # the log informs the resume
    ax.text(7.35, 2.65, "reads the log", fontsize=8, color="0.3")                     # its label
    ax.set_title("Day 173: what the tracker records, and how a crashed study resumes")  # title
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "tracking_diagram.png"), dpi=110); plt.close(fig)  # save
    print("saved tracking_diagram.png")


def main():                                             # every section in order
    t0 = time.time()                                                                  # a clock for the whole run
    if os.path.exists(LOG):                                                           # a previous run's log
        os.remove(LOG)                                                                # start the study fresh
    D = load_data()                                                                   # the rows
    data = fingerprint(D["Xa"], D["ya"], D["Xs"], D["ys"], D["Xv"], D["yv"])          # the data version
    code = file_hash(__file__)                                                        # the code version
    T = Tracker(LOG); configs = sample_configs(N_RUNS)                                # the tracker; 40 configurations
    R = demo_study(T, D, configs, data, code)                                         # 1
    demo_query(R)                                                                     # 2
    demo_reproduce(R, D)                                                              # 3
    seeds = demo_seeds(R, D)                                                          # 4
    demo_fingerprints(T, D, configs, data, code)                                      # 5
    print()
    plot_results(R, seeds)                                                            # the figure
    draw_diagram()                                                                    # the diagram
    print(f"total time {time.time() - t0:.0f} s")
    head("WHAT COMES NEXT")                                                               # the last banner
    print("Day 174: reproducibility. One tracked tuning study built from all six of this week's parts,")
    print("run twice to bit-identical results, with seeds controlled and the environment pinned.")


if __name__ == "__main__":                              # only when run directly, not when imported
    main()                                              # run everything
