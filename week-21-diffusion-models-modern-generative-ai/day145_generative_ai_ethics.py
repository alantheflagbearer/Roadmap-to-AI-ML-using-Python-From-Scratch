"""
Day 145: Ethical Considerations in Generative AI -- Bias, Deepfakes, Real Case Studies

The models of Days 141-144 draw handwritten digits, which harm nobody. The same
methods, scaled up, draw faces, voices and photographs, and three problems
follow them: a model can exaggerate the skews in its training data (bias), its
output can be passed off as real (deepfakes), and it can reproduce its training
images (memorisation). Each is measured here on digits, where every effect can
be counted, and the guide sets each beside a real, documented case.

Everything is written from scratch in NumPy, on Day 126's networks and Adam,
Days 129-130's VAE, Day 134's GAN, Day 138's scorer, Day 141's diffusion model,
Day 142's DDIM, Day 143's encoders and Day 144's text-conditioned model, all
imported unchanged:
  * a training set skewed on purpose, against an unskewed control, and what guidance does
  * a detector of generated images: how well it works, and how it fails
  * models trained on 10, 100 and 1,000 digits, checked for copies

  1. By hand: what a detector's accuracy means when fakes are rare.
  2. Bias: a skewed training set, and what the model makes of it.
  3. Deepfake detection: a detector, other generators, and simple edits.
  4. Memorisation: small training sets and copied digits.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import os                                              # os: paths, and the thread setting below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")      # 4 threads: measured here at about 4 times one thread's speed
import contextlib                                      # contextlib: redirect_stdout, to keep imported lessons quiet
import io                                              # io: StringIO, where that output goes
import sys                                             # sys: path, to import earlier parts
import time                                            # time: how long training takes

import numpy as np                                     # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API
from matplotlib.patches import FancyBboxPatch           # rounded boxes, for the diagram

HERE = os.path.dirname(os.path.abspath(__file__))       # the folder this script lives in
sys.path.insert(0, HERE)                                # this week's parts (Days 140-144)
for w in ("week-20-generative-models-gans", "week-19-unsupervised-learning-ii", "week-18-unsupervised-learning-i", "week-17-classical-ml-kernels-ensembles"):
    sys.path.insert(0, os.path.join(HERE, "..", w))     # earlier weeks' parts
with contextlib.redirect_stdout(io.StringIO()):         # the imports print their own checks; keep them quiet
    import day126_exercise as d126                      # Day 126: init_net, forward, backward, Adam
    import day126_autoencoders as d126m                 # Day 126: mnist (the cached loader)
    import day129_exercise as d129                      # Day 129: train_vae
    import day130_exercise as d130                      # Day 130: sample_prior
    import day134_exercise as d134                      # Day 134: AdamGAN, gan_step, generate
    import day139_gan_build_and_forward_diffusion as d139  # Day 139: train_classifier, Scorer (Day 138's)
    import day141_exercise as d141                      # Day 141: train_image_denoiser, nearest_train, to_pixels
    import day142_exercise as d142                      # Day 142: ddim_sample, net_x0_fn
    import day143_exercise as d143                      # Day 143: measure, train_clip, embed_texts, NUM
    import day144_exercise as d144                      # Day 144: train_cond, cfg_sample, ink_lean, said

SKEW = 0.75                                             # the share of training ones that lean right, after the skew (a third, before)
MEM_SIZES = (10, 100, 1000)                             # training-set sizes for the memorisation test
MEM_STEPS = 8000                                        # training steps for each of those models
NEAR = 2.0                                              # 'near copy': a sample this close (pixel distance) to a training digit


# =============================================================================
# THE PARTS -- NEW
# =============================================================================
def skew_ones(y, lean, share=SKEW, seed=0):              # NEW: training indices with most ones not leaning right removed
    rng = np.random.RandomState(seed); idx = np.arange(len(y))
    one = y == 1; right = one & (lean == 1); other = one & (lean != 1)                # ones leaning right, and every other one
    keep_other = int(right.sum() * (1 - share) / share)                               # how many other ones leave `share` of the ones leaning right
    drop = rng.permutation(idx[other])[keep_other:]                                   # the rest of them are removed
    return np.setdiff1d(idx, drop)                                                    # every other digit is kept as it was


def train_detector(R, G, epochs=10, seed=0, batch=128, lr=1e-3):  # NEW: real (0) against generated (1), Day 126's network [784, 128, 1]
    net = d126.init_net([784, 128, 1], seed); opt = d126.Adam(net, lr); rng = np.random.RandomState(seed)
    Xd = np.vstack([R, G]).astype(np.float32); yd = np.r_[np.zeros(len(R)), np.ones(len(G))]  # images and their labels
    for _ in range(epochs):
        o = rng.permutation(len(Xd))
        for s in range(0, len(Xd) - batch + 1, batch):
            b = o[s:s + batch]; A = d126.forward(net, Xd[b], ["relu", "linear"])      # one score per image
            p = 1 / (1 + np.exp(-A[-1][:, 0].astype(np.float64)))                    # the score as a probability of 'generated'
            grads, _ = d126.backward(net, A, ["relu", "linear"], ((p - yd[b]) / len(b))[:, None].astype(np.float32))  # logistic loss's gradient
            opt.step(net, grads)
    return net


def p_fake(net, F):                                     # NEW: the detector's probability that each image is generated
    return 1 / (1 + np.exp(-d126.forward(net, F.astype(np.float32), ["relu", "linear"])[-1][:, 0].astype(np.float64)))


def blur(F):                                            # NEW: each pixel replaced by the average of its 3 x 3 neighbourhood
    P = np.pad(F.reshape(-1, 28, 28), ((0, 0), (1, 1), (1, 1)))                       # a border of zeros, so edge pixels have 9 neighbours
    return (sum(P[:, i:i + 28, j:j + 28] for i in range(3) for j in range(3)) / 9).reshape(-1, 784)


def flagged_precision(tpr, fpr, prevalence):            # NEW: of the images a detector flags, the share that really are generated
    return tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence))             # Bayes' rule


# =============================================================================
# THE EXPERIMENTS
# =============================================================================
def demo_hand():
    print("=" * 70)
    print("1. BY HAND -- what a detector's accuracy means when fakes are rare")
    print("=" * 70)
    print("a detector that catches 95% of generated images (true positive rate) and wrongly flags 5% of real ones (false positive rate);")
    print("share of flagged images that really are generated = 0.95 p / (0.95 p + 0.05 (1 - p)), p = the share of images that are generated")
    print(f"{'p':>8} {'flagged that are generated':>27} {'real images flagged per generated one caught':>45}")
    for p in (0.5, 0.1, 0.01, 0.001):
        f = flagged_precision(0.95, 0.05, p)
        print(f"{p:8.3f} {f:27.3f} {0.05 * (1 - p) / (0.95 * p):45.2f}")
    print(f"by hand for p = 0.01: 0.95 x 0.01 = 0.0095; 0.05 x 0.99 = 0.0495; 0.0095 / (0.0095 + 0.0495) = {0.0095 / 0.059:.3f}")
    print("Actual output from this lesson's run.")


def demo_bias(X, y, X2, sc):
    print("\n" + "=" * 70)
    print("2. BIAS -- a skewed training set, and what the model makes of it")
    print("=" * 70)
    ink, slant, thick, lean = d143.measure(X, y)                                      # Day 143's attributes, cut points from all 60,000 training digits
    keep = skew_ones(y[:60000], lean[:60000]); one = y[keep] == 1
    print(f"training set: the 60,000 training digits with {60000 - len(keep):,} ones removed, all of them ones not leaning right;")
    print(f"ones leaning right: {np.mean(lean[:60000][y[:60000] == 1] == 1):.3f} of ones before, {np.mean(lean[keep][one] == 1):.3f} after "
          f"({one.sum():,} ones left); every other digit unchanged")
    t0 = time.perf_counter()
    _, txt, _ = d143.train_clip(X[keep], y[keep], thick[keep], lean[keep])            # Day 143's encoders, on the skewed set
    net, ema, _ = d144.train_cond(X2[keep], y[keep], thick[keep], lean[keep], txt)    # Day 144's text-conditioned model, on the skewed set
    print(f"Day 143's encoders and Day 144's model ({d144.STEPS:,} steps) trained on the skewed set: {time.perf_counter() - t0:.0f} s")
    cut_lean = {d: np.percentile(slant[:60000][y[:60000] == d], [100 / 3, 200 / 3]) for d in range(10)}  # Day 143's cut points, from the
    cut_ink = {d: np.percentile(ink[:60000][y[:60000] == d], 200 / 3) for d in range(10)}                # training digits before the skew
    print("generated: 400 samples per row from the same starting noise; 'leaning right' and 'thick' use Day 143's cut points")
    print(f"{'prompt':>22} {'w':>5} {'named as asked':>15} {'leaning right':>14} {'leaning left':>13} {'thick':>7}")

    def table(ema, txt, prompts, tag):                                                # one row per prompt and guidance scale
        rows, pics = [], {}
        for prompt, d in prompts:
            for w in (1.0, 2.0, 4.0):
                F = d144.cfg_sample(ema, d143.embed_texts(txt, [prompt.split()])[0], 400, w); lab = d144.said(sc, F)
                ik, ln = d144.ink_lean(F); k = lab == d                               # only samples that came out as the asked-for digit
                r, l_, th = np.mean(ln[k] > cut_lean[d][1]), np.mean(ln[k] < cut_lean[d][0]), np.mean(ik[k] > cut_ink[d])  # that digit's cut points
                rows.append((prompt + tag, w, r, th)); pics[(prompt + tag, w)] = F[:12]
                print(f"{prompt + tag:>22} {w:5.1f} {k.mean():15.3f} {r:14.3f} {l_:13.3f} {th:7.3f}")
        return rows, pics
    rows, pics = table(ema, txt, (("one", 1), ("seven", 7), ("a one leaning left", 1)), "")
    kk = y[keep] == 7
    print(f"{'training data':>22} {'':>5} {'':>15} {np.mean(lean[keep][one] == 1):14.3f} {np.mean(lean[keep][one] == -1):13.3f} "
          f"{np.mean(thick[keep][kk] == 1):7.3f}   (ones for the lean columns, sevens for 'thick')")
    t0 = time.perf_counter()                                                          # the control: the same training on the unskewed set
    _, txt0, _ = d143.train_clip(X[:60000], y[:60000], thick[:60000], lean[:60000])         # Day 143's encoders, on all 60,000
    _, ema0, _ = d144.train_cond(X2[:60000], y[:60000], thick[:60000], lean[:60000], txt0)  # Day 144's model, on all 60,000
    print(f"\ncontrol: the same encoders and model, same seeds, trained on all 60,000 training digits (no skew): {time.perf_counter() - t0:.0f} s")
    r0, p0 = table(ema0, txt0, (("one", 1), ("seven", 7)), " (no skew)"); rows += r0; pics.update(p0)  # the same rows, control model
    o1 = y[:60000] == 1                                                               # every training one, before the skew
    print(f"{'training data':>22} {'':>5} {'':>15} {np.mean(lean[:60000][o1] == 1):14.3f} {np.mean(lean[:60000][o1] == -1):13.3f} "
          f"{np.mean(thick[:60000][y[:60000] == 7] == 1):7.3f}   (unskewed: ones for the lean columns, sevens for 'thick')")
    print("Actual output from this lesson's run.")
    return ema, txt, rows, pics


def demo_detector(X, y, ema, txt):
    print("\n" + "=" * 70)
    print("3. DEEPFAKE DETECTION -- a detector, other generators, and simple edits")
    print("=" * 70)
    t0 = time.perf_counter()
    G = np.vstack([d144.cfg_sample(ema, d143.embed_texts(txt, [[w]])[0], 600, 2.0, seed=20 + k) for k, w in enumerate(d143.NUM)])  # 6,000 generated digits
    G = G[np.random.RandomState(4).permutation(len(G))]                               # mixed, so training and held-out sets hold every digit
    R = X[np.random.RandomState(3).permutation(60000)[:5000]]                          # 5,000 real training digits
    Rt = X[60000 + np.random.RandomState(0).permutation(10000)[:1000]]; Gt = G[5000:]  # held out: 1,000 real test digits, 1,000 generated
    det = train_detector(R, G[:5000])
    print(f"detector: Day 126's network [784, 128, 1], logistic loss, 10 epochs on 5,000 real training digits and 5,000 digits from")
    print(f"section 2's model (prompts 'zero' ... 'nine', w = 2); {time.perf_counter() - t0:.0f} s including sampling")
    enc, dec, _ = d129.train_vae(X[:60000], 16, epochs=30)                             # Day 129's VAE, as on Day 142
    V = d130.sample_prior(dec, 1000, 16, np.random.RandomState(1)).astype(np.float32)
    Gn = d126.init_net([64, 256, 784], 0); Dn = d126.init_net([784, 256, 1], 1); rng = np.random.RandomState(0)
    oG, oD = d134.AdamGAN(Gn, 2e-4, 0.5), d134.AdamGAN(Dn, 2e-4, 0.5)                   # Day 134's dense GAN, as on Day 142
    for _ in range(10):                                                               # 10 epochs (Day 142: FID 49.446)
        o = rng.permutation(60000)
        for s in range(0, 60000 - 127, 128):
            d134.gan_step(Gn, Dn, oG, oD, X[o[s:s + 128]], rng, 64)
    Ga = d134.generate(Gn, np.random.RandomState(1).standard_normal((1000, 64)).astype(np.float32))[0]
    print("share called generated (probability above 0.5); 1,000 images each")
    print(f"{'images':>44} {'called generated':>17}")
    tpr, fpr = float(np.mean(p_fake(det, Gt) > 0.5)), float(np.mean(p_fake(det, Rt) > 0.5)); res = {}
    for name, F in (("held-out generated (same model)", Gt), ("real test digits", Rt), ("VAE samples (never seen)", V),
                    ("GAN samples (never seen)", Ga)):
        res[name] = float(np.mean(p_fake(det, F) > 0.5)); print(f"{name:>44} {res[name]:17.3f}")
    edits = (("3 x 3 blur", blur), ("noise, sd 0.1", lambda F: np.clip(F + np.random.RandomState(5).normal(0, 0.1, F.shape), 0, 1)),
             ("4 grey levels", lambda F: np.round(F * 3) / 3), ("black and white (0.5)", lambda F: (F > 0.5).astype(np.float32)))
    for name, f in edits:
        for who, F in (("generated", Gt), ("real", Rt)):
            res[f"{who}, {name}"] = float(np.mean(p_fake(det, f(F)) > 0.5)); print(f"{who + ', ' + name:>44} {res[who + ', ' + name]:17.3f}")
    print(f"if 1 image in 100 were generated: of the images this detector flags, {flagged_precision(tpr, fpr, 0.01):.3f} would be generated "
          f"(true positive rate {tpr:.3f}, false positive rate {fpr:.3f})")
    print("Actual output from this lesson's run.")
    return res, G[:8], V[:8], Ga[:8], Gt[:8], Rt[:8]


def demo_memorise(X, X2):
    print("\n" + "=" * 70)
    print("4. MEMORISATION -- small training sets and copied digits")
    print("=" * 70)
    print(f"Day 141's model, {MEM_STEPS:,} steps, trained on the first n of a fixed shuffle of the training digits; 500 samples each (DDIM 100 steps);")
    print(f"for each sample, the pixel distance to the nearest training digit; 500 real test digits give the distances an honest new digit")
    print(f"lands at; 'near copies': closer than {NEAR}. Day 141's model, trained on all 60,000, had its closest sample 1.37 away")
    print(f"{'n':>6} {'median: samples':>16} {'test digits':>12} {'near copies':>12} {'test digits':>12} {'different nearest':>18} {'time':>6}")
    order = np.random.RandomState(7).permutation(60000); Te = X[60000 + np.random.RandomState(0).permutation(10000)[:500]]
    out = {}
    for n in MEM_SIZES:
        t0 = time.perf_counter(); idx = order[:n]
        net, _ = d141.train_image_denoiser(X2[idx], MEM_STEPS)                         # Day 141's model, on n digits only
        F = d141.to_pixels(d142.ddim_sample(d142.net_x0_fn(net), 500, np.random.RandomState(1), 100, 0.0))
        dF, jF = d141.nearest_train(F, X[idx]); dT, _ = d141.nearest_train(Te, X[idx])
        out[n] = (F, X[idx][jF], dF)
        print(f"{n:6,d} {np.median(dF):16.2f} {np.median(dT):12.2f} {np.mean(dF < NEAR):12.3f} {np.mean(dT < NEAR):12.3f} "
              f"{len(np.unique(jF)):>11d} of {min(n, 500):<4d} {time.perf_counter() - t0:5.0f}s")
    print("Actual output from this lesson's run.")
    return out


# =============================================================================
# FIGURES
# =============================================================================
def plot_results(rows, pics, det, mem, out):            # bias, detection, memorisation
    fig = plt.figure(figsize=(17, 13)); gs = fig.add_gridspec(3, 4)

    def grid(ax, F, r, c, title):                                                     # r x c images tiled into one picture
        ax.imshow(np.vstack([np.hstack([F[i * c + j].reshape(28, 28) for j in range(c)]) for i in range(r)]), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[0, 0:2])                                                  # bias: the share leaning right, by guidance
    for prompt, mk, ls, lab in (("one", "o", "-", "'one', skewed set"), ("one (no skew)", "o", "--", "'one', unskewed control"),
                                ("a one leaning left", "s", "-", "'a one leaning left', skewed set")):
        r = [x for x in rows if x[0] == prompt]; ax.plot([x[1] for x in r], [x[2] for x in r], marker=mk, ls=ls, label=f"{lab}: ones leaning right")
    r = [x for x in rows if x[0] == "seven"]; ax.plot([x[1] for x in r], [x[3] for x in r], marker="^", label="'seven': sevens measured thick")
    ax.axhline(SKEW, color="gray", ls=":", label=f"ones leaning right in the skewed data ({SKEW:.2f})")
    ax.axhline(1 / 3, color="gray", ls="--", label="a third: unskewed ones leaning right, sevens measured thick")
    ax.set_xlabel("guidance scale w"); ax.set_ylabel("share of samples"); ax.set_ylim(0, 1.5); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.legend(fontsize=7, loc="upper center", ncol=2)  # headroom above 1 for the legend
    ax.set_title("bias: what the model draws, against what it was shown", fontsize=10)
    grid(fig.add_subplot(gs[0, 2]), pics[("one", 2.0)], 3, 4, "'one', w = 2 (skewed training set)")
    grid(fig.add_subplot(gs[0, 3]), pics[("a one leaning left", 2.0)], 3, 4, "'a one leaning left', w = 2")
    res, G, V, Ga, Gt, Rt = det
    ax = fig.add_subplot(gs[1, 0:2]); names = list(res)                               # detection: share called generated
    ax.barh(range(len(names)), [res[k] for k in names], color=["#C44E52" if ("generated" in k or "samples" in k) else "#4C72B0" for k in names])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", ls=":"); ax.set_xlabel("share called generated (red: generated images; blue: real)")
    ax.set_title("deepfake detection: the detector, other generators, simple edits", fontsize=10)
    grid(fig.add_subplot(gs[1, 2]), np.vstack([Gt[:4], Rt[:4]]), 2, 4, "held out: generated (top), real (bottom)")
    grid(fig.add_subplot(gs[1, 3]), np.vstack([V[:4], Ga[:4]]), 2, 4, "never seen: VAE (top), GAN (bottom)")
    for j, n in enumerate(MEM_SIZES):                                                 # memorisation: samples above their nearest training digit
        F, N_, dF = mem[n]; o = np.argsort(dF)[:6]
        grid(fig.add_subplot(gs[2, j]), np.vstack([F[o], N_[o]]), 2, 6, f"n = {n:,}: 6 closest samples (top),\ntheir nearest training digits")
    ax = fig.add_subplot(gs[2, 3])
    for n in MEM_SIZES:
        ax.hist(mem[n][2], bins=40, range=(0, 10), histtype="step", lw=1.5, label=f"n = {n:,}")
    ax.axvline(NEAR, color="gray", ls=":"); ax.set_xlabel("distance to the nearest training digit"); ax.set_ylabel("samples"); ax.legend(fontsize=8)
    ax.set_title("memorisation: smaller sets, closer copies", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_diagram():                                     # three risks, each measured
    fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):                                         # a rounded box centred on (x, y)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):                                                        # an arrow between two points
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(1.4, 2.2, 2.2, 1.0, "training data\n(skews, private\nimages)", "#FFF2CC"); box(4.3, 2.2, 2.0, 1.0, "generative\nmodel", "#DDEBF7")
    arrow(2.55, 2.2, 3.25, 2.2)
    for y, risk, test, col in ((3.7, "bias: skews copied\nor exaggerated", "section 2: skewed digits,\nguidance turned up", "#F4CCCC"),
                               (2.2, "deepfakes: output\npassed off as real", "section 3: a detector,\nnew generators, edits", "#FCE4D6"),
                               (0.7, "memorisation: training\nimages reproduced", "section 4: 10 to 1,000\ntraining digits", "#E2F0D9")):
        box(7.6, y, 2.6, 0.95, risk, col); box(11.0, y, 2.9, 0.95, test, "#EEEEEE"); arrow(5.35, 2.2, 6.25, y); arrow(8.95, y, 9.5, y)
    ax.set_title("Day 145: three risks of generative models, each measured on digits", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "generative_ai_ethics_diagram.png"), dpi=120); plt.close(fig)


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("WHAT COMES NEXT")
    print("=" * 70)
    print("Day 146, the week's capstone, puts this week's parts together into one text-to-digit generator and checks it end to end.")


def main():                                             # runs every section in order
    X, y = d126m.mnist(); X2 = (X * 2 - 1).astype(np.float32)                         # Day 126's 70,000 digits; [-1, 1] for the diffusion models
    demo_hand()                                                                       # 1. base rates
    t0 = time.perf_counter(); body, head = d139.train_classifier(X[:60000], y[:60000])  # Day 138's feature network, for naming digits
    sc = d139.Scorer(body, head, X[:10000], X[10000:30000]); print(f"\n(feature network: {time.perf_counter() - t0:.0f} s)")
    ema, txt, rows, pics = demo_bias(X, y, X2, sc)                                    # 2. bias
    det = demo_detector(X, y, ema, txt)                                               # 3. detection
    mem = demo_memorise(X, X2)                                                        # 4. memorisation
    plot_results(rows, pics, det, mem, os.path.join(HERE, "generative_ai_ethics_results.png"))
    draw_diagram()                                                                    # the three risks
    print("\nsaved generative_ai_ethics_results.png\nsaved generative_ai_ethics_diagram.png")
    note_on_what_comes_next()


if __name__ == "__main__":
    main()
