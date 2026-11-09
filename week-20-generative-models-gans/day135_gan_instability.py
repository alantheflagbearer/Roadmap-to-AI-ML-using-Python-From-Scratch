"""
Day 135: GAN Training Instability -- a Real, Honest Mode-Collapse Demonstration

Day 134's GANs worked, but not reliably: some 1-D runs sat on one peak for
thousands of steps, and on MNIST one setting of Adam produced the same image
for every input. That is MODE COLLAPSE: the generator covers only part of
the data. Today it is measured on data where every mode is known exactly:
eight tight Gaussians on a circle.

Everything is written from scratch in NumPy, on Day 134's GAN parts
(gan_step, generate, the two losses), imported unchanged:
  * the ring of eight Gaussians, and a count of which modes the generator
    reaches and how many of its samples are good
  * a GAN on the ring, watched every 250 steps, not only at the end
  * the experiments: five seeds; the non-saturating against the minimax
    loss; a discriminator with a head start; and the balance between the
    two players' learning rates

  1. By hand: the gradient a fake sends back, under each loss.
  2. Eight peaks, five seeds: how often is a mode lost, and is it found again?
  3. The minimax loss on the same five seeds.
  4. A discriminator with a head start: where the minimax loss should fail.
  5. The balance between the players: learning rates and extra
     discriminator steps.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier parts
import time                                            # time: how long the runs take

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts
sys.path.insert(0, os.path.join(HERE, "..", "week-19-unsupervised-learning-ii"))  # Day 126's parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, sigmoid, Adam
    import day134_exercise as d134                      # Day 134: gan_step, generate, g_loss


# =============================================================================
# A DATA SET WHOSE MODES ARE KNOWN -- NEW
# =============================================================================
def ring(n, rng, k=8, radius=2.0, sd=0.05):             # NEW: k tight Gaussians on a circle: every sample belongs to a known mode
    a = 2 * np.pi * rng.randint(0, k, n) / k                                          # which mode, as an angle
    return (np.c_[np.cos(a), np.sin(a)] * radius + sd * rng.standard_normal((n, 2))).astype(np.float32)


def mode_stats(F, k=8, radius=2.0, sd=0.05):            # NEW: which modes the samples reach, and how many samples are good
    C = np.c_[np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)] * radius  # the true centres
    D = np.sqrt(((F[:, None, :] - C[None]) ** 2).sum(-1)); near = D.argmin(1)                    # each sample's nearest centre
    good = D[np.arange(len(F)), near] < 3 * sd                                        # within 3 standard deviations: a real-looking sample
    counts = np.bincount(near[good], minlength=k)                                     # good samples per mode
    return int((counts >= 0.01 * len(F)).sum()), float(good.mean()), counts           # a mode counts if it gets 1% of the samples


def train_ring(seed, steps=6000, kind="non-saturating", lr_g=1e-3, lr_d=1e-3, d_steps=1, every=250, k=8, z=2, head=0):  # NEW: a GAN on the ring, watched as it trains
    rng = np.random.RandomState(seed); G = d126.init_net([z, 128, 128, 2], seed); D = d126.init_net([2, 128, 128, 1], seed + 100)
    oG, oD = d126.Adam(G, lr_g), d126.Adam(D, lr_d); track = []                         # (step, modes, good share, counts, D(real), D(fake))
    ev = np.random.RandomState(999).standard_normal((2000, z)).astype(np.float32)      # the same noise every time we look
    for t in range(1, steps + 1):
        oG.lr = 0.0 if t <= head else lr_g                                            # a head start: only the discriminator learns at first
        _, _, dr, df = d134.gan_step(G, D, oG, oD, ring(256, rng, k), rng, z, out="linear", kind=kind, d_steps=d_steps)
        if t % every == 0:                                                            # look during training, not only at the end
            m, g, c = mode_stats(d134.generate(G, ev, "linear")[0], k); track.append((t, m, g, c, dr, df))
    return G, track


# =============================================================================
# DEMOS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- the gradient one fake sends back to the generator, under each loss")
    print("=" * 70)
    print("   D(fake)   non-saturating -log D   minimax log(1 - D)   ratio")
    rows = []
    for p in (0.5, 0.1, 0.01, 0.001):                                                 # from undecided to sure the fake is fake
        lf = np.full((1, 1), np.log(p / (1 - p)), np.float32)                         # the logit that gives D = p
        a, b = float(d134.g_loss(lf)[1][0, 0]), float(d134.g_loss(lf, "minimax")[1][0, 0])  # d loss / d logit, one sample
        rows.append((p, a, b)); print(f"   {p:7.3f}   {a:+21.4f}   {b:+18.4f}   {a / b:5.0f}")
    print("by hand: non-saturating d/dlogit = D - 1; minimax = -D. When D(fake) is near 0 the minimax gradient is near 0 too")
    print("Actual output from this lesson's run.")
    return rows


def lost_and_found(track):                              # how often a covered mode later drops below 1%, and comes back
    have = [set(np.flatnonzero(c >= 20)) for _, _, _, c, _, _ in track]              # modes with at least 1% of 2,000 samples
    lost = sum(len(a - b) for a, b in zip(have, have[1:])); found = sum(len(b - a) for a, b in zip(have, have[1:]))
    return lost, found


def run_seeds(name, seeds=(0, 1, 2, 3, 4), **kw):       # several seeds of one setting, with a table watched over training
    marks = (500, 1000, 2000, 3000, 4000, 5000, 6000); out = {}; t0 = time.perf_counter()
    print(f"{name}: modes covered (of 8) at each step, then good share and D(fake) at the end")
    print("   seed  " + "".join(f"{m:>6,}" for m in marks) + "   good  D(fake)  lost  found  final counts per mode (of 2,000)")
    for s in seeds:
        G, tr = train_ring(s, **kw); at = {t: m for t, m, _, _, _, _ in tr}; lost, found = lost_and_found(tr)
        out[s] = (G, tr); print(f"   {s:4d}  " + "".join(f"{at[m]:6d}" for m in marks) + f"   {tr[-1][2]:.2f}  {tr[-1][5]:7.2f}  {lost:4d}  {found:5d}  "
                           + " ".join(f"{c:3d}" for c in tr[-1][3]))
    print(f"   all 8 modes at the end in {sum(tr[-1][1] == 8 for _, tr in out.values())} of {len(seeds)} runs ({time.perf_counter() - t0:.0f} s)")
    return out


def demo_ring():
    print("\n" + "=" * 70)
    print("2. EIGHT PEAKS, FIVE SEEDS -- how often is a mode lost, and is it found again?")
    print("=" * 70)
    print("data: 8 Gaussians (spread 0.05) on a circle of radius 2; a sample is good if within 3 spreads (0.15) of a centre")
    print("generator [2, 128, 128, 2], discriminator [2, 128, 128, 1], Day 126's Adam at 0.001 for both, batches of 256, 6,000 steps;")
    print("watched every 250 steps on 2,000 fixed noise vectors; a mode is covered if it gets at least 1% of them (20 samples);")
    print("lost / found = how many times a covered mode dropped below 1%, and how many times one came back, over the 24 looks")
    R = run_seeds("non-saturating loss, -log D(G(z))")
    print("Actual output from this lesson's run.")
    return R


def demo_minimax():
    print("\n" + "=" * 70)
    print("3. THE MINIMAX LOSS -- the same five seeds, generator loss log(1 - D(G(z)))")
    print("=" * 70)
    R = run_seeds("minimax loss, log(1 - D(G(z)))", kind="minimax")
    print("Actual output from this lesson's run.")
    return R


def demo_head_start():
    print("\n" + "=" * 70)
    print("4. A DISCRIMINATOR WITH A HEAD START -- the generator frozen for the first 1,000 steps")
    print("=" * 70)
    print("for 1,000 steps only the discriminator learns, so by the time the generator starts it is sure the fakes are fake")
    out = {}
    for kind in ("non-saturating", "minimax"):
        out[kind] = run_seeds(f"{kind} loss, head start 1,000 steps", kind=kind, head=1000)
        print(f"   D(fake) at step 1,000, when the generator starts: {' '.join(f'{tr[3][5]:.3f}' for _, tr in out[kind].values())}")
    print("Actual output from this lesson's run.")
    return out


def demo_balance():
    print("\n" + "=" * 70)
    print("5. THE BALANCE BETWEEN THE PLAYERS -- learning rates, and extra discriminator steps (non-saturating loss)")
    print("=" * 70)
    out = {}
    for name, kw in (("generator slower: G 0.0001, D 0.001", dict(lr_g=1e-4)), ("discriminator slower: G 0.001, D 0.0001", dict(lr_d=1e-4)),
                     ("5 discriminator steps per generator step", dict(d_steps=5))):
        out[name] = run_seeds(name, seeds=(0, 1, 2), **kw)
    print("Actual output from this lesson's run.")
    return out


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(rows, ns, mm, hs, bal, out):             # nine panels
    fig = plt.figure(figsize=(16, 13)); ev = np.random.RandomState(999).standard_normal((2000, 2)).astype(np.float32)
    real = ring(2000, np.random.RandomState(5))
    ax = fig.add_subplot(3, 4, 1); p = np.logspace(-3, np.log10(0.99), 200)            # the two gradients against D(fake)
    ax.plot(p, 1 - p, label="non-saturating |D - 1|"); ax.plot(p, p, label="minimax |-D|"); ax.set_xscale("log")
    ax.set_xlabel("D(fake)"); ax.set_ylabel("|d loss / d logit|"); ax.set_title("the gradient a fake sends back", fontsize=10); ax.legend(fontsize=8)
    worst = lambda R: min(R, key=lambda s: (R[s][1][-1][1], R[s][1][-1][2]))           # the seed with the fewest modes at the end
    best = lambda R: max(R, key=lambda s: (R[s][1][-1][1], R[s][1][-1][2]))            # and with the most
    w, wh, wm = worst(ns), worst(hs["non-saturating"]), worst(hs["minimax"])
    for j, (title, G) in enumerate(((f"non-saturating, seed {best(ns)}", ns[best(ns)][0]), (f"non-saturating, seed {w}", ns[w][0]),
                                     (f"minimax, seed {w}", mm[w][0]), (f"non-saturating, head start, seed {wh}", hs["non-saturating"][wh][0]),
                                     (f"minimax, head start, seed {wm}", hs["minimax"][wm][0]))):
        ax = fig.add_subplot(3, 4, 2 + j); F, _ = d134.generate(G, ev, "linear")     # where the generator puts its samples
        ax.scatter(real[:, 0], real[:, 1], s=2, c="lightgray", label="real"); ax.scatter(F[:, 0], F[:, 1], s=2, c="#C44E52", alpha=0.5, label="generated")
        m, g, _ = mode_stats(F); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
        ax.set_title(f"{title}\n{m} of 8 modes, good share {g:.2f}", fontsize=9)
        if j == 0:
            ax.legend(fontsize=7, loc="center", markerscale=4)
    for j, (title, R) in enumerate((("non-saturating", ns), ("minimax", mm), ("non-saturating, head start", hs["non-saturating"]),
                                     ("minimax, head start", hs["minimax"]))):
        ax = fig.add_subplot(3, 4, 7 + j)                                             # modes covered during training, every seed
        for s, (_, tr) in R.items():
            ax.plot([t for t, *_ in tr], [m for _, m, *_ in tr], lw=1.2, label=f"seed {s}")
        ax.set_ylim(-0.3, 8.5); ax.set_xlabel("step"); ax.set_ylabel("modes covered"); ax.set_title(title, fontsize=10)
        if j == 0:
            ax.legend(fontsize=7, loc="lower right")
    ax = fig.add_subplot(3, 4, 11); _, tr = ns[w]                                     # the worst run, mode by mode
    C = np.array([c for _, _, _, c, _, _ in tr]) / 2000; ts = [t for t, *_ in tr]
    for k in range(8):
        ax.plot(ts, C[:, k], lw=1, label=f"mode {k}")
    ax.set_xlabel("step"); ax.set_ylabel("share of good samples"); ax.set_title(f"non-saturating, seed {w}, mode by mode", fontsize=10)
    ax.set_ylim(0, 1.45 * C.max()); ax.legend(fontsize=6, ncol=4, loc="upper center")
    ax = fig.add_subplot(3, 4, 12); names = list(bal) + ["equal rates (section 2)"]     # the balance experiments, final coverage
    vals = [[tr[-1][1] for _, tr in bal[n].values()] for n in bal] + [[ns[s][1][-1][1] for s in (0, 1, 2)]]
    for i, v in enumerate(vals):
        ax.scatter([i] * len(v), v, s=40); ax.plot([i - 0.2, i + 0.2], [np.mean(v)] * 2, "k")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split(":")[0].replace("5 discriminator steps per generator step", "5 D steps") for n in names],
                                                         fontsize=7, rotation=15)
    ax.set_ylim(-0.3, 8.5); ax.set_ylabel("modes covered at the end"); ax.set_title("balance, seeds 0-2 (line: mean)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # three ways the game goes wrong
    fig, ax = plt.subplots(figsize=(12, 5)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9):                                           # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
    box(1.6, 2.5, 2.4, 1.2, "the GAN game:\nD learns, then G learns", "#DDEEFF")
    box(6.0, 4.0, 4.2, 1.1, "mode collapse: G finds a few modes\nthat fool D, and never returns to the rest", "#F8CBAD", 8.5)
    box(6.0, 2.5, 4.2, 1.1, "vanishing gradient: D is sure fakes are fake,\nso log(1 - D) sends almost nothing back", "#FFF2CC", 8.5)
    box(6.0, 1.0, 4.2, 1.1, "no settling: each player chases the other;\nthe loss is not a measure of quality", "#E2F0D9", 8.5)
    box(10.4, 2.5, 2.6, 2.6, "measured here on\n8 known modes:\nmodes covered,\ngood share,\nwatched during\ntraining", "#EEEEEE", 8.5)
    arrow(2.85, 2.8, 3.85, 3.85); arrow(2.85, 2.5, 3.85, 2.5); arrow(2.85, 2.2, 3.85, 1.15)
    arrow(8.15, 4.0, 9.05, 3.2); arrow(8.15, 2.5, 9.05, 2.5); arrow(8.15, 1.0, 9.05, 1.8)
    ax.set_title("Day 135: three ways GAN training goes wrong", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "gan_instability_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 136 replaces the dense players with convolutional ones (a DCGAN-style GAN), so the generator can draw strokes")
    print("instead of placing every pixel on its own.")


def main():                                             # runs every section in order
    rows = demo_hand()                                                                # 1. the two gradients, by hand
    ns = demo_ring()                                                                  # 2. eight peaks, non-saturating loss
    mm = demo_minimax()                                                               # 3. the minimax loss
    hs = demo_head_start()                                                            # 4. a discriminator with a head start
    bal = demo_balance()                                                              # 5. the balance between the players
    plot_results(rows, ns, mm, hs, bal, os.path.join(HERE, "gan_instability_results.png"))
    draw_diagram()                                                                    # the picture of what goes wrong
    print("\nsaved gan_instability_results.png\nsaved gan_instability_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
