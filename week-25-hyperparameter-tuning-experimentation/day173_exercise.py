"""
Day 173 Exercise -- Experiment Tracking

The sixth of Week 25's parts for Day 174's reproducible tuning study.

    fingerprint(*arrays)             a short sha256 of arrays' shapes, types and bytes: the data version
    file_hash(path)                  a short sha256 of a file: the code version
    run_id(params, seed, data, code) the same run on the same data and code always gets the same id
    Tracker(path)                    an append-only JSON Lines log; .run(...) for a with block,
                                     .runs() to read every run back, .done(id) to skip finished runs
    query(runs, where, by, n)        finished runs that match, sorted by a metric, the first n

Checked by hand in a temporary folder: fingerprints that change with one
value, a shape or a type; ids that ignore dictionary order; three runs with
one crash, each leaving a start and an end record; a resume that redoes only
the failed run; and a query that sorts the finished runs.
"""
import hashlib                                         # hashlib: sha256 fingerprints
import json                                            # json: one record per line
import os                                              # os: file paths
import platform                                        # platform: the Python version, for the environment record
import time                                            # time: timestamps and costs

import numpy as np                                     # numpy: every array operation below is built on this



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


if __name__ == "__main__":  # the hand checks run only when this file is run directly
    import tempfile                                                                   # a throwaway folder for the log
    a = np.arange(6, dtype=np.float32)                                                # six numbers
    b = a.copy(); b[0] += 1e-6                                                        # one changed by a millionth
    print(f"fingerprint: equal copies match {fingerprint(a) == fingerprint(a.copy())}, one value changed "
          f"{fingerprint(a) == fingerprint(b)}, reshaped {fingerprint(a) == fingerprint(a.reshape(2, 3))}, "
          f"float64 {fingerprint(a) == fingerprint(a.astype(np.float64))} (by hand True, False, False, False)")
    print(f"run_id ignores dictionary order: {run_id(dict(lr=0.1, b=32), 0, 'd', 'c') == run_id(dict(b=32, lr=0.1), 0, 'd', 'c')}, "
          f"but not the seed: {run_id(dict(lr=0.1), 0, 'd', 'c') == run_id(dict(lr=0.1), 1, 'd', 'c')} (by hand True, False)")
    path = os.path.join(tempfile.mkdtemp(), "runs.jsonl"); T = Tracker(path)          # an empty log

    def toy(T, crash):                                  # three runs whose score is (lr - 0.2) squared; the second may crash
        for lr in (0.1, 0.2, 0.4):                                                    # three settings
            rid = run_id(dict(lr=lr), 0, "toy", "v1")                                 # the id
            if T.done(rid):                                                           # finished before
                continue                                                              # skip it
            with T.run(dict(lr=lr), 0, "toy", "v1") as r:                             # tracked
                if crash and lr == 0.2:                                               # the simulated crash
                    raise RuntimeError("crash")                                       # recorded as failed
                r.log(1, loss=lr); r.summary["score"] = (lr - 0.2) ** 2               # one step, one score
    try:                                                                              # the first attempt
        toy(T, crash=True)                                                            # stops at lr 0.2
    except RuntimeError:                                                              # expected
        pass                                                                          # carry on
    print(f"after the crash: statuses {[r['status'] for r in T.runs()]}, lines {len(open(path).readlines())} "
          f"(by hand ['finished', 'failed']: lr 0.4 never started; 4 lines, a start and an end each)")
    toy(T, crash=False)                                                               # resume
    print(f"after resuming: statuses {[r['status'] for r in T.runs()]}, lines {len(open(path).readlines())} "
          f"(by hand three finished; 4 + 2 for lr 0.2 again + 2 for lr 0.4 = 8)")
    best = query(T.runs(), by="score")                                                # sorted, smallest first
    print(f"query by score: lr {[r['params']['lr'] for r in best]}, scores {[round(r['summary']['score'], 4) for r in best]} "
          f"(by hand 0.2 first with 0, then 0.1 with 0.01, then 0.4 with 0.04)")
