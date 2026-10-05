"""
Day 61 Exercise -- Embedding Lookup and Masking, Isolated

The main lesson's two new mechanisms on their own, with a six-word
vocabulary and no training:

  1. An embedding table: forward is a row lookup; backward ADDS each
     position's gradient into the row of the word that was there, so a
     word used twice collects two contributions.
  2. Masking: on a padded position the hidden state is carried forward
     unchanged, so a short sentence gives the same final state alone as
     it does padded inside a batch.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this

vocab = ["<pad>", "the", "food", "was", "not", "good"]  # id 0 is reserved for padding
stoi = {w: i for i, w in enumerate(vocab)}              # word -> id
rng = np.random.RandomState(0)                          # fixed rng
E = rng.randn(len(vocab), 3)                            # the embedding table: one 3-number vector per word
Wxh = rng.randn(3, 4) * 0.5                             # input-to-hidden weights (embedding 3 -> hidden 4)
Whh = rng.randn(4, 4) * 0.5                             # recurrent weights
bh = np.zeros(4)                                        # hidden bias


def encode(sentence, length):                           # words -> ids, padded with 0 up to `length`
    ids = [stoi[w] for w in sentence.split()]             # look every word up
    return np.array(ids + [0] * (length - len(ids))), len(ids)  # padded ids, and how many are real


def final_state(ids, n_real, use_mask=True):            # run the RNN over one sentence, return the last hidden state
    h = np.zeros(4)                                       # initial state
    for t, word_id in enumerate(ids):                     # one position at a time
        h_new = np.tanh(E[word_id] @ Wxh + h @ Whh + bh)    # the usual recurrence; E[word_id] is the embedding lookup
        is_real = 1.0 if (t < n_real or not use_mask) else 0.0  # 1 for a real word, 0 for padding
        h = is_real * h_new + (1.0 - is_real) * h           # padding: keep the old state
    return h                                              # state after the last position


# ---- 1. the embedding lookup, forward and backward -------------------------
ids, n_real = encode("the food was not the good", 6)    # "the" appears TWICE
X = E[ids]                                              # forward: (6,) ids -> (6, 3) vectors
print("ids:", ids.tolist())
print("lookup returns the table's own rows:", np.array_equal(X[0], E[stoi["the"]]) and np.array_equal(X[4], E[stoi["the"]]))

dX = np.ones((6, 3))                                    # pretend every position received a gradient of all ones
dE = np.zeros_like(E)                                   # one gradient row per vocabulary word
np.add.at(dE, ids, dX)                                  # backward: add each position's gradient into its word's row
print("gradient row for 'the'  (used twice):", dE[stoi["the"]].tolist())
print("gradient row for 'good' (used once) :", dE[stoi["good"]].tolist())
print("gradient row for '<pad>' (not used) :", dE[stoi["<pad>"]].tolist())

wrong = np.zeros_like(E)                                # the tempting one-liner that silently loses a contribution
wrong[ids] += dX                                        # fancy-index += writes repeated rows only ONCE
print("with dE[ids] += dX instead, 'the' gets:", wrong[stoi["the"]].tolist(), "<- one contribution lost")

# ---- 2. masking -------------------------------------------------------------
short_ids, short_n = encode("not good", 2)              # the sentence alone: no padding
padded_ids, padded_n = encode("not good", 8)            # the same sentence padded to length 8
h_alone = final_state(short_ids, short_n)                # reference
h_masked = final_state(padded_ids, padded_n)             # padded, with masking
h_unmasked = final_state(padded_ids, padded_n, use_mask=False)  # padded, the RNN keeps running over six <pad> steps
print("\npadded ids:", padded_ids.tolist(), " real words:", padded_n)
print(f"max |h_masked   - h_alone| = {np.max(np.abs(h_masked - h_alone)):.2e}")
print(f"max |h_unmasked - h_alone| = {np.max(np.abs(h_unmasked - h_alone)):.2e}")
