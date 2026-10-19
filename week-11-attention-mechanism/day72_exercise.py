"""
Day 72 Exercise -- Measuring an Alignment

Today's lesson read a trained model's attention weights. This exercise
isolates the three measurements it used, on weight matrices built by hand
so the right answers are known in advance:

  argmax on target   share of output steps whose largest weight is on the right input
  weight on target   the average weight placed on the right input
  entropy            how spread out each row is: 0 = one position, log(n) = even

and draws them as a heatmap (rows: output steps, columns: input positions).
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API


def alignment_metrics(W, target):                       # NEW: W (steps, n) rows of weights; target[j] = right input for step j
    j = np.arange(len(target))                            # output steps
    entropy = -(W * np.log(W + 1e-12)).sum(axis=1)        # one number per row
    return (W.argmax(axis=1) == target).mean(), W[j, target].mean(), entropy.mean()


def rows_softmax(S):                                    # each row of scores into weights
    E = np.exp(S - S.max(axis=1, keepdims=True))          # subtract each row's max, exponentiate
    return E / E.sum(axis=1, keepdims=True)               # each row sums to 1


n = 8                                                     # 8 output steps over 8 input positions
target = np.arange(n)                                     # COPY: step j should look at position j
dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])  # how far each position is from each step's target
cases = {"sharp diagonal": rows_softmax(-8.0 * dist),     # almost all weight on the target
         "blurred diagonal": rows_softmax(-1.0 * dist),   # weight leaks to neighbours
         "uniform": np.full((n, n), 1 / n),               # no preference at all
         "shifted by one": rows_softmax(-8.0 * np.abs(np.arange(n)[:, None] + 1 - np.arange(n)[None, :]))}  # looks one step late

print(f"{'matrix':<18}{'argmax on target':>18}{'weight on target':>18}{'entropy':>9}   (log n = {np.log(n):.3f})")
for name, W in cases.items():                             # each hand-built matrix
    a, w, e = alignment_metrics(W, target)                # the three measurements
    print(f"{name:<18}{a:>18.3f}{w:>18.3f}{e:>9.3f}")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))          # one heatmap per case
for ax, (name, W) in zip(axes, cases.items()):
    ax.imshow(W, cmap="Blues", vmin=0, vmax=1)              # dark = large weight
    ax.set_title(name, fontsize=9); ax.set_xlabel("input position"); ax.set_ylabel("output step")
plt.tight_layout(); plt.savefig("day72_exercise_heatmaps.png", dpi=100); plt.close()
print("saved day72_exercise_heatmaps.png")
