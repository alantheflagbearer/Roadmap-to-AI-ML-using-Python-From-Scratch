"""
Day 98 Exercise -- Batched Inference Helpers

The first of Week 15's parts for Day 104's build, where a retriever and a
generator both run over many texts a batch at a time.

    batched(items, size)           consecutive chunks of a list (the last may be shorter)
    run_batched(fn, items, size)   apply fn to each chunk and join the results along the first axis
    softmax(z)                     probabilities from scores, safe for very large scores

Checked: the chunks cover the list exactly once in order; batching does not
change a model's answers (beyond float rounding); softmax does not overflow.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import torch                                          # torch: the pretrained model runs on it
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # the library's model and tokenizer
from transformers.utils import logging as hf_logging  # the library's logging, kept quiet
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()  # no progress bars in the output


def batched(items, size):                               # NEW: consecutive chunks of at most `size` items
    for i in range(0, len(items), size):                  # chunk starts
        yield items[i:i + size]                             # one chunk


def run_batched(fn, items, size):                       # NEW: fn on every chunk, results stacked in order
    return np.concatenate([np.asarray(fn(chunk)) for chunk in batched(items, size)], axis=0)


def softmax(z):                                         # NEW: probabilities over the last axis
    z = z - z.max(axis=-1, keepdims=True)                 # subtracting the max changes nothing mathematically, prevents overflow
    e = np.exp(z)                                         # now every exponent is at most 1
    return e / e.sum(axis=-1, keepdims=True)


items = list(range(10))                                 # a small list
for size in (1, 3, 4, 10, 15):                          # several chunk sizes
    chunks = list(batched(items, size))
    flat = [x for c in chunks for x in c]
    print(f"size {size:>2}: {len(chunks)} chunks, sizes {[len(c) for c in chunks]}, cover the list exactly in order: {flat == items}")

z = np.array([[1000.0, 1001.0], [-5.0, 5.0]])           # huge scores: a naive exp(1000) overflows
with np.errstate(over="ignore", invalid="ignore"):      # show the naive version's failure without a warning
    naive = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
print(f"softmax of [1000, 1001]: naive {naive[0]}, safe {np.round(softmax(z)[0], 6)} (expected [0.268941, 0.731059])")

name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # Day 98's model
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForSequenceClassification.from_pretrained(name).eval()
texts = ["a gorgeous, witty, seductive movie.", "the plot is nothing but boilerplate cliches.", "it's fine.",
         "not bad at all", "an utter waste of a fine cast", "I laughed, I cried, I left early."]  # six short reviews


def scores(chunk):                                      # the model's raw scores for one chunk of texts
    with torch.no_grad():                                 # inference only
        return model(**tok(list(chunk), padding=True, return_tensors="pt")).logits.numpy()


ref = softmax(run_batched(scores, texts, 1))            # one text at a time: no padding
for size in (2, 4, 6):                                  # padded batches
    p = softmax(run_batched(scores, texts, size))
    print(f"batch {size}: largest probability difference from one-at-a-time {np.abs(p - ref).max():.1e}")
print("P(positive):", ", ".join(f"{t[:18]!r} {q:.3f}" for t, q in zip(texts, ref[:, 1])))
