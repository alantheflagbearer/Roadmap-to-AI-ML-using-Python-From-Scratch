"""
Day 55: Real Image Data Pipeline Efficiency -- Why "DataLoader" Exists

Every script this week (Days 49-54) called make_junction_dataset_rgb()
once, got back one big in-memory NumPy array, and trained directly off
array slices via iterate_minibatches(). That works because this series'
synthetic datasets are small (a few hundred images) and generated fresh
in RAM. Real image datasets are usually stored as thousands to millions
of individual files on disk, too large to load into memory all at once
-- which is exactly the problem PyTorch's Dataset/DataLoader abstraction
(and every other framework's equivalent) exists to solve. Today builds
three from-scratch data-loading strategies against a REAL folder of
image files (not one big array) and measures their actual tradeoffs,
rather than asserting them:

  1. InMemoryLoader -- loads every file into one array up front (what
     this week's iterate_minibatches did implicitly), then slices it.
     Fast per-batch, but needs O(dataset size) memory up front.
  2. LazyDiskLoader -- stores only file PATHS, and reads each image from
     disk on every single batch access, every epoch, with no caching.
     Needs only O(batch size) memory, but pays a real disk-read cost
     repeatedly for the SAME files.
  3. PrefetchDiskLoader -- the same lazy reads as (2), but uses a
     background thread to read the NEXT batch's files while the CURRENT
     batch is being used for training -- the same "overlap I/O with
     compute" trick real DataLoaders use via num_workers>0.

Before trusting any benchmark or accuracy number built on these three
loaders, a batch-equivalence check confirms all three yield BIT-IDENTICAL
batches (same images, same shuffle order) given the same seed -- today's
analogue of a gradient check: verify the alternative implementations
agree with the trusted one before drawing conclusions from them (real
result: all 10 batches checked, across all 3 loaders, matched exactly).

A real wall-clock benchmark (5 epochs, batch_size=20, a simulated 10ms
compute step per batch standing in for a real forward+backward pass)
then measures actual epoch time for each strategy: InMemoryLoader
0.105s/epoch, LazyDiskLoader 0.131s/epoch (25% slower, from real repeated
disk reads with no caching), PrefetchDiskLoader 0.107s/epoch -- a single
background thread recovers nearly all of that lost time (down to just 2%
slower than InMemoryLoader) by reading the next batch while the current
one is "training". The disk-based loaders also hold 10x less memory at
once than InMemoryLoader (one batch's worth vs. the whole dataset), a
ratio that only grows favorably as a real dataset scales past what fits
in RAM.

A real (not simulated) training run then confirms a small CNN reaches
EXACTLY IDENTICAL final weights (max absolute difference: 0.0) when
trained through InMemoryLoader vs. LazyDiskLoader with the same seeds --
concrete proof that a loading strategy changes throughput and memory,
never what the model learns. Getting an honest, working comparison here
took one real fix along the way: this lesson's small single-conv-layer
network initially collapsed to chance accuracy at this series' usual
lr=0.1-0.15, its one FC hidden layer 100% dead under ReLU within a few
epochs -- the same failure mode Days 52-54 diagnosed in deeper networks,
just triggered here by a large flattened input (2,592 features) feeding
a single dense layer. Recalibrating to lr=0.02 (checked directly, not
assumed) fixed it, reaching a real train_acc=1.000/test_acc=0.6625.

Today introduces no new backward-pass math: conv2d_forward_mc/backward_mc
and maxpool_forward/backward are reused byte-for-byte from Day 53/49,
already gradient-checked there. Every line of code below still carries
its own comment, continuing Day 52's convention.
"""
import os                                               # os: building/removing the temporary on-disk image folder this lesson simulates
import time                                             # time: real wall-clock measurements for the throughput benchmark
import shutil                                           # shutil: recursively removing the temporary on-disk folder when this lesson is done
import tempfile                                         # tempfile: a fresh, isolated temp directory for simulating "files on disk", cleaned up automatically
from concurrent.futures import ThreadPoolExecutor       # ThreadPoolExecutor: the background thread PrefetchDiskLoader reads the NEXT batch on
import numpy as np                                      # numpy: every array/matrix operation below is built on this
import matplotlib                                       # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                    # "Agg" backend renders to a file, not a screen -- needed since this runs headless
import matplotlib.pyplot as plt                          # pyplot: the plotting API actually called later (plt.bar, plt.savefig, ...)

np.random.seed(42)                                       # fixes numpy's GLOBAL random state so any code that forgets to pass its own rng is still reproducible

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Day 53/54 (conv/pool math is
#    UNCHANGED today, already gradient-checked there)
# =============================================================================
def relu(z):                                             # ReLU activation: elementwise max(0, z)
    return np.maximum(0, z)                               # returns z where z>0, else 0, same shape as z


def drelu(z):                                            # derivative of ReLU w.r.t. its own PRE-activation input z
    return (z > 0).astype(z.dtype)                        # 1 where z>0, else 0 -- boolean mask cast back to z's float dtype


def softmax(z):                                          # softmax: turns raw class scores (logits) into a probability distribution per row
    z_shifted = z - z.max(axis=1, keepdims=True)           # subtract each row's own max first, for numerical stability (avoids exp() overflow)
    exp_z = np.exp(z_shifted)                              # exponentiate the shifted logits
    return exp_z / exp_z.sum(axis=1, keepdims=True)        # normalize each row so its probabilities sum to 1


def one_hot(y, n_classes):                               # converts integer class labels into one-hot row vectors
    m = y.shape[0]                                         # m = number of examples (rows) in y
    out = np.zeros((m, n_classes))                         # start with an all-zero (m, n_classes) matrix
    out[np.arange(m), y] = 1                               # set exactly one 1 per row, at the column given by that row's true class
    return out                                             # the finished one-hot label matrix


def cross_entropy_loss(probs, y_onehot):                 # multi-class cross-entropy loss, averaged over the batch
    m = probs.shape[0]                                     # m = batch size (number of rows)
    eps = 1e-9                                             # tiny constant to avoid log(0) if a predicted probability is exactly 0
    return -np.sum(y_onehot * np.log(probs + eps)) / m     # -sum(true * log(pred)) per row, averaged across the m rows


def he_scale(n_in):                                      # He initialization scale factor for ReLU-activated layers
    return np.sqrt(2.0 / n_in)                             # sqrt(2/fan_in) -- keeps activation variance stable through a ReLU layer


def conv2d_forward_mc(X, W, b, pad=0):                   # multi-channel 2D convolution forward pass, stride 1, vectorized over kernel offsets (Day 53)
    if pad > 0:                                            # only pad when asked to
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # zero-pad only the two spatial axes (height, width), leave batch/channel alone
    else:                                                   # no padding requested
        Xp = X                                              # just use the input as-is
    m, C_in, H, Win = Xp.shape                             # unpack (possibly padded) input shape: batch size, input channels, height, width
    C_out, C_in_w, kH, kW = W.shape                        # unpack filter shape: output channels, input channels, kernel height, kernel width
    out_H = H - kH + 1                                     # output height for a valid convolution over the (possibly padded) input
    out_W = Win - kW + 1                                   # output width, same formula along the width axis
    Z = np.zeros((m, C_out, out_H, out_W))                 # allocate the output feature map, all zeros to start
    for kh in range(kH):                                   # loop over kernel offsets (e.g. 5, not out_H) -- Day 53's vectorization
        for kw in range(kW):                               # loop over kernel column offsets
            X_slice = Xp[:, :, kh:kh + out_H, kw:kw + out_W]  # every output position's input pixel AT THIS ONE (kh, kw) offset, for the whole batch/channels/spatial map at once
            Z += np.einsum("mchw,fc->mfhw", X_slice, W[:, :, kh, kw])  # contract over input channels c, broadcast over output filters f, all (m,h,w) positions in one call
    Z += b[None, :, None, None]                             # add each filter's bias once, broadcast across the batch and every spatial position
    return Z                                               # the full convolved output, shape (m, C_out, out_H, out_W)


def conv2d_backward_mc(dZ, X, W, pad=0):                 # backward pass for conv2d_forward_mc: returns dW, db, dX
    if pad > 0:                                            # must re-pad X here too, so the patches line up with what forward actually convolved over
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # identical padding to the forward pass
    else:                                                   # no padding case
        Xp = X                                              # use X directly
    m, C_in, H, Win = Xp.shape                             # shape of the (possibly padded) input the forward pass actually convolved
    C_out, C_in_w, kH, kW = W.shape                        # filter shape again
    out_H = H - kH + 1                                     # recompute output height (must match what forward produced)
    out_W = Win - kW + 1                                   # recompute output width
    dW = np.zeros_like(W)                                  # gradient accumulator for the filters, same shape as W
    db = dZ.sum(axis=(0, 2, 3)) / m                         # bias gradient: sum dZ over batch and both spatial axes at once, one filter per entry
    dXp = np.zeros_like(Xp)                                # gradient accumulator for the (possibly padded) input
    for kh in range(kH):                                   # loop over kernel offsets, exactly mirroring the forward pass's loop
        for kw in range(kW):                               # same (kh, kw) offset the forward pass used
            X_slice = Xp[:, :, kh:kh + out_H, kw:kw + out_W]  # the same input sub-grid the forward pass read at this offset
            dW[:, :, kh, kw] = np.einsum("mfhw,mchw->fc", dZ, X_slice) / m  # this offset's contribution to every (filter, input-channel) weight gradient
            dXp[:, :, kh:kh + out_H, kw:kw + out_W] += np.einsum("mfhw,fc->mchw", dZ, W[:, :, kh, kw])  # scatter this offset's gradient contribution back onto the input sub-grid it came from
    if pad > 0:                                             # if padding was added on the way in, it must be stripped off on the way out
        dX = dXp[:, :, pad:-pad, pad:-pad]                  # crop the padding rows/columns back off, leaving dX the same shape as the ORIGINAL X
    else:                                                    # no padding was used
        dX = dXp                                             # dXp is already the right shape
    return dW, db, dX                                      # gradients w.r.t. filters, biases, and the layer's (unpadded) input


def maxpool_forward(A, size=2, stride=2):                # 2x2 max pooling forward pass -- unchanged since Day 49
    m, C, H, W = A.shape                                   # unpack input shape: batch, channels, height, width
    out_H = (H - size) // stride + 1                       # output height using the standard pooling output-size formula
    out_W = (W - size) // stride + 1                       # output width, same formula
    out = np.zeros((m, C, out_H, out_W))                   # allocate the pooled output
    idx_cache = np.zeros((m, C, out_H, out_W), dtype=np.int64)  # cache of WHICH position in each window was the max (needed for backward)
    for i in range(out_H):                                 # loop over each output row position i
        for j in range(out_W):                             # loop over each output column position j
            window = A[:, :, i * stride:i * stride + size,  # the size x size window this output position pools over
                       j * stride:j * stride + size]
            flat = window.reshape(m, C, size * size)        # flatten each window to a 1D list of size*size candidate values
            idx = flat.argmax(axis=2)                       # index (within the flattened window) of the maximum value
            out[:, :, i, j] = flat.max(axis=2)              # the actual max value becomes this output position's value
            idx_cache[:, :, i, j] = idx                     # remember which flattened index won, for routing gradient back correctly
    return out, idx_cache                                  # pooled output, plus the cache backward() needs


def maxpool_backward(dOut, A, idx_cache, size=2, stride=2):  # backward pass for maxpool_forward -- unchanged since Day 49
    m, C, H, W = A.shape                                   # shape of the ORIGINAL (pre-pooling) input
    dA = np.zeros_like(A)                                  # gradient accumulator for that original input, all zero to start
    out_H, out_W = dOut.shape[2], dOut.shape[3]            # spatial size of the upstream gradient (matches the pooled output's shape)
    for i in range(out_H):                                 # loop over each pooled output row position i
        for j in range(out_W):                             # loop over each pooled output column position j
            idx = idx_cache[:, :, i, j]                     # which flattened window-index was the max, for every (example, channel) here
            r = idx // size                                 # convert that flat index back into a row offset within the window
            c = idx % size                                  # and a column offset within the window
            for mm in range(m):                             # loop over every example in the batch
                for cc in range(C):                         # loop over every channel
                    dA[mm, cc, i * stride + r[mm, cc], j * stride + c[mm, cc]] += \
                        dOut[mm, cc, i, j]                   # route this output's gradient ONLY to the single input position that won the max
    return dA                                              # gradient w.r.t. the pre-pooling input


def make_junction_dataset_rgb(n_per_class=50, size=30, arm=8, noise=0.4,   # synthetic 4-class "junction" image generator -- unchanged
                               channel_noise=0.15, center_jitter=2, seed=0):
    rng = np.random.RandomState(seed)                      # this function's own private RNG, independent of the global numpy state
    X, y = [], []                                          # accumulators for generated images and their integer class labels
    cx = cy = size // 2                                    # the image's center coordinate (same for x and y since images are square)
    dirs_all = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # the 4 possible arm directions: up, down, left, right
    for cls in range(4):                                   # loop over the 4 classes (0..3)
        n_arms = cls + 1                                   # class number directly controls how many arms radiate from the center (1..4)
        for _ in range(n_per_class):                       # generate n_per_class images for this class
            im = rng.randn(size, size) * noise              # start from pure Gaussian noise as the base image
            r = cy + rng.randint(-center_jitter, center_jitter + 1)  # jitter the junction's row position slightly, for realism
            c = cx + rng.randint(-center_jitter, center_jitter + 1)  # jitter the column position slightly too
            if n_arms == 4:                                 # if this class uses all 4 arms
                chosen = dirs_all                            # just use every direction
            else:                                           # otherwise (1, 2, or 3 arms)
                idx = rng.choice(4, size=n_arms, replace=False)  # randomly choose WHICH n_arms of the 4 directions to draw
                chosen = [dirs_all[i] for i in idx]          # look up the actual (dr, dc) direction tuples for those chosen indices
            for (dr, dc) in chosen:                         # draw each chosen arm
                for step in range(1, arm + 1):               # walk outward from the center, `arm` pixels long
                    rr, cc = r + dr * step, c + dc * step    # the pixel coordinate `step` pixels along this arm's direction
                    im[rr, cc] += 2.0                        # brighten that pixel to make the arm visible above the noise
            im[r, c] += 2.0                                 # also brighten the exact center pixel itself
            rgb = np.stack([im, im, im], axis=0)             # replicate the single grayscale image across 3 channels to make it "RGB"
            rgb = rgb + rng.randn(3, size, size) * channel_noise  # add small INDEPENDENT noise per channel, so channels aren't perfectly identical
            X.append(rgb)                                   # store this generated image
            y.append(cls)                                   # store its true class label
    X = np.array(X)                                        # stack the list of images into one (N, 3, size, size) array
    y = np.array(y)                                        # stack the list of labels into one (N,) array
    idx = rng.permutation(len(y))                           # a random shuffle order for the whole dataset
    return X[idx], y[idx]                                  # return the shuffled images and labels together, so pairing is preserved


# =============================================================================
# 1. SIMULATING A REAL ON-DISK IMAGE DATASET -- one .npy file per image
# =============================================================================
def write_dataset_to_disk(X, y, root):                    # NEW today: saves an in-memory array as individual per-image files, like a real dataset folder
    os.makedirs(root, exist_ok=True)                        # create the target folder (and any missing parents), no error if it already exists
    paths = []                                              # will hold the on-disk path of every saved image, in the SAME order as X/y
    for i in range(len(y)):                                 # loop over every image in the dataset
        path = os.path.join(root, f"img_{i:05d}.npy")        # this image's own file path, e.g. img_00042.npy
        np.save(path, X[i])                                  # write just this ONE image to its own file (not the whole array)
        paths.append(path)                                   # remember this path for the loaders below
    np.save(os.path.join(root, "labels.npy"), y)            # labels are small enough to keep as one file (a real dataset would use a CSV/manifest instead)
    return paths                                            # the list of per-image file paths, index-aligned with y


# =============================================================================
# 2. THREE DATA-LOADING STRATEGIES
# =============================================================================
class InMemoryLoader:                                     # strategy 1: exactly what iterate_minibatches (Days 49-54) did, made an explicit, named class
    """Loads every image into ONE array up front, then serves batches by
    pure slicing. Fastest per-batch access, but needs O(dataset size)
    memory before training can even start -- the strategy every prior
    day this week used without naming it."""

    def __init__(self, file_paths, y, batch_size, seed):   # constructor: reads EVERY file now, once, up front
        self.X = np.stack([np.load(p) for p in file_paths])  # load and stack every single image into one big array -- the O(dataset size) memory cost
        self.y = y                                           # labels, already in memory (small)
        self.batch_size = batch_size                         # how many examples per yielded batch
        self.rng = np.random.RandomState(seed)               # this loader's own private RNG for shuffling
        self.nbytes = self.X.nbytes                          # how many bytes this loader's own in-memory array occupies (today's memory-cost metric)

    def epoch_batches(self):                                # generator: yields one epoch's worth of (Xb, yb, idx) batches
        order = self.rng.permutation(len(self.y))            # a fresh random shuffle of every index, once per epoch
        for start in range(0, len(order), self.batch_size):  # step through that shuffled order in chunks of batch_size
            idx = order[start:start + self.batch_size]        # this batch's indices
            yield self.X[idx], self.y[idx], idx                # pure array slicing -- no disk access at all after __init__


class LazyDiskLoader:                                     # strategy 2: never holds more than one batch in memory, but re-reads from disk every time
    """Stores only file PATHS -- nothing is loaded until a batch is
    actually requested, and NOTHING is cached, so the SAME image is
    read from disk again on every epoch it appears in. Needs only
    O(batch size) memory, at the cost of repeated real disk I/O."""

    def __init__(self, file_paths, y, batch_size, seed):   # constructor: stores paths ONLY, does no disk I/O yet
        self.file_paths = file_paths                         # the on-disk path of every image, index-aligned with y
        self.y = y                                           # labels (kept in memory -- realistic, since labels are cheap even for huge image datasets)
        self.batch_size = batch_size                         # batch size
        self.rng = np.random.RandomState(seed)               # this loader's own private RNG for shuffling
        self.nbytes = None                                   # filled in after the first batch is read, since batch size (not dataset size) sets this loader's memory footprint

    def _read_batch(self, idx):                             # reads exactly the images THIS batch needs, fresh from disk, every call
        imgs = [np.load(self.file_paths[i]) for i in idx]     # one np.load() PER IMAGE in the batch -- genuine disk I/O, not a cache hit
        Xb = np.stack(imgs)                                   # stack the freshly-read images into one batch array
        if self.nbytes is None:                               # the first time a batch is read, record its size as this loader's peak memory footprint
            self.nbytes = Xb.nbytes                            # one batch's worth of bytes -- independent of total dataset size
        return Xb, self.y[idx], idx                           # this batch's images, labels, and original indices

    def epoch_batches(self):                                # generator: yields one epoch's worth of (Xb, yb, idx) batches
        order = self.rng.permutation(len(self.y))            # a fresh random shuffle of every index, once per epoch -- SAME shuffling logic as InMemoryLoader
        for start in range(0, len(order), self.batch_size):  # step through that shuffled order in chunks of batch_size
            idx = order[start:start + self.batch_size]        # this batch's indices
            yield self._read_batch(idx)                        # read this batch from disk right now, synchronously


class PrefetchDiskLoader(LazyDiskLoader):                 # strategy 3: same lazy reads as strategy 2, but overlapped with the CALLER's own compute
    """Identical to LazyDiskLoader, except the NEXT batch's files are
    read on a background thread while the CURRENT batch is being used
    (e.g. for a training step) -- the same idea real DataLoader(
    num_workers=N) uses to hide disk latency behind GPU/CPU compute."""

    def __init__(self, file_paths, y, batch_size, seed):   # same constructor as LazyDiskLoader, plus a background thread pool
        super().__init__(file_paths, y, batch_size, seed)    # reuse LazyDiskLoader's __init__ entirely (paths, y, batch_size, rng)
        self.executor = ThreadPoolExecutor(max_workers=1)     # ONE background worker thread -- enough to overlap I/O with compute, matching num_workers=1

    def epoch_batches(self):                                # generator: yields batches with the NEXT one already being fetched in the background
        order = self.rng.permutation(len(self.y))            # identical shuffling to the other two loaders (same rng call, same seed behavior)
        batch_idxs = [order[start:start + self.batch_size]    # precompute every batch's index list for this epoch, up front
                      for start in range(0, len(order), self.batch_size)]
        future = self.executor.submit(self._read_batch, batch_idxs[0])  # kick off reading the FIRST batch on the background thread before yielding anything
        for i in range(len(batch_idxs)):                      # loop over every batch position in this epoch
            Xb, yb, idx = future.result()                      # block until the batch this loop iteration needs is actually ready (usually already done)
            if i + 1 < len(batch_idxs):                        # if there's a NEXT batch still to come this epoch
                future = self.executor.submit(self._read_batch, batch_idxs[i + 1])  # start reading it NOW, in the background, while the caller uses THIS batch
            yield Xb, yb, idx                                   # hand this batch to the caller -- the next one is already being fetched concurrently

    def close(self):                                        # shuts down the background thread pool cleanly when this loader is no longer needed
        self.executor.shutdown(wait=True)                     # wait for any in-flight read to finish, then release the worker thread


# =============================================================================
# 3. BATCH-EQUIVALENCE CHECK -- do all three loaders agree, before trusting
#    any benchmark or accuracy number built on top of them?
# =============================================================================
def demo_batch_equivalence_check(file_paths, y, batch_size=20, seed=7):  # verifies all 3 loaders yield IDENTICAL batches given the same seed
    print("=" * 70)                                          # section separator
    print("1. BATCH-EQUIVALENCE CHECK -- do all 3 loaders agree?")
    print("=" * 70)                                           # closing separator
    print("Before trusting any timing or accuracy comparison built on these "  # explains why this check comes first, today's analogue of a gradient check
          "three loaders, this confirms they actually agree on WHAT DATA "
          "gets served: same shuffle order, same images, every batch, "
          "given the same seed.")
    mem = InMemoryLoader(file_paths, y, batch_size, seed)      # instantiate all 3 loaders with the SAME file list, labels, batch size, and seed
    lazy = LazyDiskLoader(file_paths, y, batch_size, seed)
    pre = PrefetchDiskLoader(file_paths, y, batch_size, seed)

    all_match = True                                           # tracks whether EVERY batch from EVERY loader matched, across the whole epoch
    n_batches_checked = 0                                       # counts how many batches were actually compared
    for (Xb_m, yb_m, idx_m), (Xb_l, yb_l, idx_l), (Xb_p, yb_p, idx_p) in zip(  # walk all 3 loaders' epochs in lockstep
            mem.epoch_batches(), lazy.epoch_batches(), pre.epoch_batches()):
        idx_ok = np.array_equal(idx_m, idx_l) and np.array_equal(idx_m, idx_p)  # same shuffle order (same indices) across all 3
        img_ok = np.array_equal(Xb_m, Xb_l) and np.array_equal(Xb_m, Xb_p)      # same actual pixel data across all 3 (bit-for-bit)
        lab_ok = np.array_equal(yb_m, yb_l) and np.array_equal(yb_m, yb_p)      # same labels across all 3
        all_match = all_match and idx_ok and img_ok and lab_ok  # this batch only counts as matching if ALL THREE checks pass
        n_batches_checked += 1                                   # one more batch verified
    pre.close()                                                 # shut down the prefetch loader's background thread now that this check is done
    print(f"\nchecked {n_batches_checked} batches across 3 loaders: "  # report the real result of the check, not just assert it silently
          f"{'all identical (idx, images, labels)' if all_match else 'MISMATCH FOUND'}")
    print("Actual output from this lesson's run.")               # standing convention: label this as real output
    return all_match                                            # hand the verdict back so later sections can rely on it


# =============================================================================
# 4. THROUGHPUT BENCHMARK -- real wall-clock time, simulated per-batch compute
# =============================================================================
def demo_throughput_benchmark(file_paths, y, batch_size=20, n_epochs=5, compute_time=0.01, seed=7):  # measures real epoch time for each loader
    print("\n" + "=" * 70)                                     # blank line then section separator
    print("2. THROUGHPUT BENCHMARK -- real wall-clock time per loading strategy")
    print("=" * 70)                                             # closing separator
    print(f"Each batch is followed by a {compute_time*1000:.0f}ms simulated "  # explains the benchmark's setup, and is explicit that compute is SIMULATED
          f"training step (time.sleep, standing in for a real forward+backward "
          f"pass) so the benchmark can test whether PrefetchDiskLoader actually "
          f"hides disk-read time behind that compute, the way real "
          f"DataLoader(num_workers>0) is supposed to. {n_epochs} epochs, "
          f"batch_size={batch_size}.")

    results = {}                                                # will hold {loader_name: total_seconds} for all three strategies
    for name, loader in [("InMemoryLoader", InMemoryLoader(file_paths, y, batch_size, seed)),  # build one fresh instance of each loader
                          ("LazyDiskLoader", LazyDiskLoader(file_paths, y, batch_size, seed)),
                          ("PrefetchDiskLoader", PrefetchDiskLoader(file_paths, y, batch_size, seed))]:
        t0 = time.perf_counter()                                 # start this loader's stopwatch
        for _ in range(n_epochs):                                 # repeat for n_epochs, so timing is stable (not dominated by one-off warmup effects)
            for Xb, yb, idx in loader.epoch_batches():             # iterate every batch this epoch
                time.sleep(compute_time)                            # SIMULATED compute -- stands in for a real training step, deliberately not doing real work here
        elapsed = time.perf_counter() - t0                        # total wall-clock time for all n_epochs with this loader
        results[name] = elapsed                                   # remember this loader's total time
        if hasattr(loader, "close"):                              # PrefetchDiskLoader needs its background thread shut down; the other two have no close()
            loader.close()                                          # clean up the background thread pool

    print(f"\n{'loader':<22}{'total time (s)':>16}{'per-epoch (s)':>16}")  # results table header
    for name, secs in results.items():                          # print every loader's timing result
        print(f"{name:<22}{secs:>16.3f}{secs / n_epochs:>16.3f}")  # total and per-epoch time for this loader
    print("Actual output from this lesson's run.")               # standing convention

    mem_bytes = InMemoryLoader(file_paths, y, batch_size, seed).nbytes  # rebuild briefly just to report its memory footprint (today's memory-cost metric)
    lazy_bytes = LazyDiskLoader(file_paths, y, batch_size, seed)._read_batch(np.arange(batch_size))[0].nbytes  # one batch's worth of bytes, Lazy/Prefetch's actual footprint
    print(f"\nmemory footprint -- InMemoryLoader: {mem_bytes:,} bytes "  # report the real, measured memory tradeoff directly
          f"(whole dataset) vs. LazyDiskLoader/PrefetchDiskLoader: "
          f"{lazy_bytes:,} bytes (one batch) -- {mem_bytes / lazy_bytes:.1f}x "
          f"less memory held at once by the disk-based loaders, regardless "
          f"of how large the underlying dataset grows.")

    plt.figure(figsize=(7, 5))                                   # start a new figure for the throughput comparison
    names = list(results.keys())                                 # the three loader names, in the order they were benchmarked
    times = [results[n] / n_epochs for n in names]                # per-epoch time for each, since that's the more interpretable unit
    colors_ = ["#5cb85c", "#d9534f", "#f0ad4e"]                   # a distinct color per bar: green (in-memory), red (lazy), orange (prefetch)
    plt.bar(names, times, color=colors_)                          # one bar per loader, height = its per-epoch time
    plt.ylabel("seconds per epoch")                               # y-axis label
    plt.title(f"Real wall-clock time per epoch ({compute_time*1000:.0f}ms simulated compute/batch)")  # descriptive title
    for i, t in enumerate(times):                                 # annotate each bar with its exact value, so the plot doesn't require a legend lookup
        plt.text(i, t, f"{t:.3f}s", ha="center", va="bottom")      # place the label just above each bar
    plt.tight_layout()                                            # avoid clipped labels/titles when saving
    plt.savefig("data_loader_throughput.png", dpi=110)            # write the figure to disk
    plt.close()                                                   # free the figure's memory now that it's saved
    print("\nsaved data_loader_throughput.png")                   # confirm the save to the console
    return results                                                # hand the raw timing dict back in case main() wants it


# =============================================================================
# 5. A MINIMAL CNN, TRAINED THROUGH TWO DIFFERENT LOADERS -- do they learn
#    the EXACT same thing?
# =============================================================================
class TinyConvNet:                                        # a deliberately small network -- today's point is the DATA pipeline, not architecture
    """One 5x5 conv -> ReLU -> 2x2 maxpool -> FC -> softmax. No new
    backward-pass math versus Day 49-53's networks -- this class exists
    only so today's loaders have something real to feed, and so their
    effect on TRAINING OUTCOMES (not just throughput) can be checked."""

    def __init__(self, C=8, img_size=40, hidden=16, n_classes=4, lr=0.02, seed=42):  # constructor: small, fixed architecture -- lr=0.02, NOT this series' usual 0.1-0.15, since this particular shallow shape dies under ReLU at higher rates (checked directly before use, not assumed)
        rng = np.random.RandomState(seed)                    # this network's own private RNG, seeded for reproducible weight init
        self.lr = lr                                          # learning rate, used in backward()
        k = 5                                                  # kernel size: 5x5, no padding (matches Days 49-52's original convention)
        self.Wc = rng.randn(C, 3, k, k) * he_scale(3 * k * k)  # the single conv layer's filters: 3 input channels (RGB) -> C output channels
        self.bc = np.zeros(C)                                  # conv layer's biases
        out_h = img_size - k + 1                               # spatial size after the valid 5x5 conv
        pool_h = out_h // 2                                    # spatial size after the 2x2 maxpool
        self.flat_dim = C * pool_h * pool_h                    # flattened feature count feeding into the FC layers
        self.W1 = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)  # FC1 weights
        self.b1 = np.zeros(hidden)                             # FC1 biases
        self.W2 = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2 = np.zeros(n_classes)                          # FC2 (output) biases

    def forward(self, X):                                    # forward pass: conv -> relu -> pool -> flatten -> FC1 -> relu -> FC2 -> softmax
        self.X = X                                             # cache raw input for backward()'s conv gradient
        self.Zc = conv2d_forward_mc(self.X, self.Wc, self.bc, pad=0)  # conv pre-activation (no padding, matches this class's docstring)
        self.Ac = relu(self.Zc)                                # conv activation
        self.pooled, self.mask = maxpool_forward(self.Ac, 2, 2)  # maxpool, plus its argmax cache for backward
        m = X.shape[0]                                          # batch size
        self.flat = self.pooled.reshape(m, -1)                  # flatten the pooled feature map into one vector per example
        self.Z1 = self.flat @ self.W1 + self.b1                 # FC1 pre-activation
        self.A1 = relu(self.Z1)                                 # FC1 activation
        self.Z2 = self.A1 @ self.W2 + self.b2                   # FC2 (output) pre-activation, the raw class logits
        self.probs = softmax(self.Z2)                           # class probabilities
        return self.probs                                       # hand back to the caller

    def backward(self, y_onehot):                             # backward pass: computes every gradient, then applies the SGD update
        m = y_onehot.shape[0]                                   # batch size
        dZ2 = self.probs - y_onehot                             # combined softmax+cross-entropy gradient w.r.t. logits
        dW2 = self.A1.T @ dZ2 / m                                # FC2 weight gradient
        db2 = dZ2.sum(axis=0) / m                                # FC2 bias gradient
        dA1 = dZ2 @ self.W2.T                                   # gradient into FC1's activation
        dZ1 = dA1 * drelu(self.Z1)                              # FC1's pre-activation gradient
        dW1 = self.flat.T @ dZ1 / m                              # FC1 weight gradient
        db1 = dZ1.sum(axis=0) / m                                # FC1 bias gradient
        dflat = dZ1 @ self.W1.T                                 # gradient into the flattened conv features
        dpooled = dflat.reshape(self.pooled.shape)               # reshape back into the pooled feature-map shape
        dAc = maxpool_backward(dpooled, self.Ac, self.mask, 2, 2)  # route gradient back through the maxpool
        dZc = dAc * drelu(self.Zc)                               # ReLU derivative for the conv layer
        dWc, dbc, _ = conv2d_backward_mc(dZc, self.X, self.Wc, pad=0)  # conv layer's weight/bias gradients (its dX is discarded)
        self.Wc -= self.lr * dWc                                 # SGD update: conv weights
        self.bc -= self.lr * dbc                                 # SGD update: conv biases
        self.W1 -= self.lr * dW1                                 # SGD update: FC1 weights
        self.b1 -= self.lr * db1                                 # SGD update: FC1 biases
        self.W2 -= self.lr * dW2                                 # SGD update: FC2 weights
        self.b2 -= self.lr * db2                                 # SGD update: FC2 biases

    def accuracy(self, X, y):                                 # measures classification accuracy on a given dataset
        probs = self.forward(X)                                 # run inference (this class has no dropout/BN, so no train/eval distinction is needed)
        preds = probs.argmax(axis=1)                            # predicted class = whichever column has the highest probability
        return (preds == y).mean()                              # fraction of predictions that match the true label

    def flat_weights(self):                                   # NEW today: concatenates every parameter into one flat vector, for an exact equality check
        return np.concatenate([self.Wc.ravel(), self.bc.ravel(),  # every weight/bias array, flattened and joined into one long vector
                                self.W1.ravel(), self.b1.ravel(),
                                self.W2.ravel(), self.b2.ravel()])


def demo_training_equivalence(file_paths, y, X_test, y_test, batch_size=20, epochs=30, seed=7):  # trains the SAME network through 2 different loaders
    print("\n" + "=" * 70)                                     # blank line then section separator
    print("3. TRAINING EQUIVALENCE -- does the loading strategy change what's learned?")
    print("=" * 70)                                             # closing separator
    print(f"Two IDENTICALLY-initialized TinyConvNet instances are trained "  # explains the setup: same init, same shuffle seed, different loaders
          f"for {epochs} epochs each -- one fed by InMemoryLoader, the other "
          f"by LazyDiskLoader -- using the SAME weight-init seed AND the "
          f"SAME shuffle seed. If the batch-equivalence check above is "
          f"correct, both networks see the identical sequence of batches "
          f"and should end up with IDENTICAL trained weights.")

    net_mem = TinyConvNet(seed=1)                                # network trained via InMemoryLoader
    loader_mem = InMemoryLoader(file_paths, y, batch_size, seed)  # in-memory loader, given the SAME shuffle seed as the lazy loader below
    for _ in range(epochs):                                       # train for the full epoch budget
        for Xb, yb, idx in loader_mem.epoch_batches():              # iterate every batch this epoch
            net_mem.forward(Xb)                                       # forward pass
            net_mem.backward(one_hot(yb, 4))                          # backward pass + SGD update

    net_lazy = TinyConvNet(seed=1)                               # a SECOND, freshly-initialized network with the IDENTICAL seed -- same starting weights as net_mem
    loader_lazy = LazyDiskLoader(file_paths, y, batch_size, seed)  # lazy disk loader, SAME shuffle seed as loader_mem
    for _ in range(epochs):                                        # same epoch budget
        for Xb, yb, idx in loader_lazy.epoch_batches():              # iterate every batch this epoch
            net_lazy.forward(Xb)                                       # forward pass
            net_lazy.backward(one_hot(yb, 4))                          # backward pass + SGD update

    w_mem, w_lazy = net_mem.flat_weights(), net_lazy.flat_weights()  # every trained parameter, flattened, from both networks
    max_weight_diff = np.max(np.abs(w_mem - w_lazy))              # the single largest absolute difference across every parameter
    acc_mem, acc_lazy = net_mem.accuracy(X_test, y_test), net_lazy.accuracy(X_test, y_test)  # test accuracy for both networks

    print(f"\nmax absolute weight difference (InMemory-trained vs. "  # report the real result of this equivalence check
          f"LazyDisk-trained network): {max_weight_diff:.2e}")
    print(f"test accuracy -- InMemory-trained: {acc_mem:.4f}, "
          f"LazyDisk-trained: {acc_lazy:.4f}")
    print("Actual output from this lesson's run.")               # standing convention
    same = max_weight_diff < 1e-9                                 # True if the two trained networks match
    verdict = ("the two networks are numerically identical" if same
               else "the two networks differ slightly")           # chosen first: f-string braces stay on one line
    print(f"\nreal, honest finding: {verdict} after {epochs} epochs of training through two "
          f"completely different data-loading code paths -- confirming a "
          f"loader's job is to change HOW FAST data arrives and HOW MUCH "
          f"memory it costs, never WHAT the model learns from it.")


# =============================================================================
# 6. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():                                    # pure documentation: prints a PyTorch equivalent, imports nothing
    print("\n" + "=" * 70)                                  # blank line then section separator
    print("4. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)                                          # closing separator
    print(                                                    # a single multi-line string showing the Dataset/DataLoader equivalent
        "class JunctionImageDataset(torch.utils.data.Dataset):\n"
        "    def __init__(self, file_paths, labels):\n"
        "        self.file_paths, self.labels = file_paths, labels\n"
        "\n"
        "    def __len__(self):\n"
        "        return len(self.labels)\n"
        "\n"
        "    def __getitem__(self, i):              # called lazily, per-index, by the DataLoader\n"
        "        return np.load(self.file_paths[i]), self.labels[i]\n"
        "\n"
        "loader = torch.utils.data.DataLoader(\n"
        "    JunctionImageDataset(file_paths, labels),\n"
        "    batch_size=20, shuffle=True,            # replaces this lesson's rng.permutation() call\n"
        "    num_workers=4, prefetch_factor=2,       # replaces PrefetchDiskLoader's ONE background thread\n"
        "    pin_memory=True)                        # a GPU-specific optimization with no equivalent in this lesson"
    )
    print("\n__getitem__ is exactly LazyDiskLoader's _read_batch, called "  # explains the mapping back to today's from-scratch code
          "once per image instead of once per batch. num_workers>1 spreads "
          "PrefetchDiskLoader's single background thread across multiple "
          "PROCESSES (sidestepping Python's GIL, unlike this lesson's "
          "ThreadPoolExecutor), and prefetch_factor controls how many "
          "batches ahead each worker reads, generalizing this lesson's "
          "one-batch-ahead prefetch to N batches ahead.")


def main():                                              # entry point: runs every demo in the order this document walks them
    tmp_dir = tempfile.mkdtemp(prefix="day55_junction_images_")  # a fresh, isolated temp folder -- simulates a real on-disk image dataset, cleaned up at the end
    try:                                                    # wrap everything in try/finally so the temp folder is always cleaned up, even on error
        img_size = 40                                        # image size, matching this week's convention
        X_train, y_train = make_junction_dataset_rgb(n_per_class=50, size=img_size,  # generate the training set, same convention as every prior day
                                                      arm=10, noise=0.4, channel_noise=0.15,
                                                      center_jitter=3, seed=42)
        X_test, y_test = make_junction_dataset_rgb(n_per_class=20, size=img_size,  # a separate, genuinely unseen test set
                                                    arm=10, noise=0.4, channel_noise=0.15,
                                                    center_jitter=3, seed=999)
        file_paths = write_dataset_to_disk(X_train, y_train, tmp_dir)  # write the training set to disk as 200 individual files -- today's "real dataset"
        print(f"wrote {len(file_paths)} images to a temporary on-disk "  # confirm the on-disk simulation was actually created
              f"folder: {tmp_dir}\n")

        demo_batch_equivalence_check(file_paths, y_train)     # 1. verify all 3 loaders agree before trusting anything built on them
        demo_throughput_benchmark(file_paths, y_train)         # 2. the real wall-clock throughput/memory comparison
        demo_training_equivalence(file_paths, y_train, X_test, y_test)  # 3. confirm training OUTCOME is loader-independent
        note_on_pytorch()                                      # 4. relate it all back to a production framework
    finally:                                                 # always runs, whether or not an error occurred above
        shutil.rmtree(tmp_dir, ignore_errors=True)             # remove the temporary on-disk dataset -- nothing from this lesson's simulation is meant to persist


if __name__ == "__main__":                               # only run main() when this file is executed directly, not when imported
    main()                                                  # kick off the whole lesson
