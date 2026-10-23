# Week 13: Tokenization & Embeddings (Days 84–90)

Week 12 built and trained a Transformer on digits. Week 13 turns to text: how it is cut into tokens, and how tokens become vectors. It follows the two-track structure of Weeks 9–12: Days 84–89 each get the full-depth lesson (real experiments, paired study-guide PDFs) plus a short `dayNN_exercise.py` isolating one part, and Day 90 builds a tiny character-level GPT-style model from those parts and Weeks 11–12's. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 84 — Byte-pair encoding (BPE) from scratch, on real text
[`day84_byte_pair_encoding.py`](day84_byte_pair_encoding.py) — the subword tokenizer of Sennrich et al. (2016): start from characters plus an end-of-word mark `</w>`, and repeatedly merge the most frequent adjacent pair into one new symbol. Worked by hand on the paper's four-word example, then trained with 4000 merges on Week 9's 2,001 training sentences and tested on the 1,647 test sentences.

- **Lossless:** every test sentence decodes back exactly (0 of 1647 differ); the test text takes 1.454 tokens per word.
- **Vocabulary against length:** 71 symbols give 5.096 tokens per word; 4071 symbols give 1.454, with 0.788 of test words a single token. A bigger vocabulary means shorter sequences.
- **No more `<unk>`:** the 3725 test words (20.2%) outside Day 62's word vocabulary become 3.16 pieces on average instead of one unknown token. Words seen 10 or more times are 1.00 piece; words never seen, 3.46.
- **The gap that is left:** 4 characters in the test text never appear in training (in 6 test words), so they stay symbols a model has never seen. Byte-level BPE, as in GPT-2, starts from the 256 byte values and closes this gap.
- **Cost:** 4000 merges took 42.8 s here, because this simple version recounts every pair every round.

Study guide: [`day84_byte_pair_encoding_complete_guide.pdf`](day84_byte_pair_encoding_complete_guide.pdf) · Syntax walkthrough: [`day84_syntax_line_by_line.pdf`](day84_syntax_line_by_line.pdf) · [Run output](day84_run_output.txt)

Exercise: [`day84_exercise.py`](day84_exercise.py) is a whole-text BPE tokenizer in four functions (spaces are ordinary characters, so merges can cross word boundaries: "the food was " becomes one token) — the first part of Day 90's tiny GPT-style model ([run output](day84_exercise_run_output.txt)).

![BPE merges, step by step](bpe_diagram.png)
![Vocabulary size against sequence length, and pieces by word frequency](bpe_results.png)

## Day 85 — Word embeddings: one-hot vs dense, on a real vocabulary
[`day85_word_embeddings.py`](day85_word_embeddings.py) — a one-hot vector times a matrix picks out one row, so an embedding table is the same thing as a one-hot input layer, stored as a lookup. Tested on Week 9's tagging task with Day 62's 2082-word vocabulary: the same window tagger (a word and its two neighbours, 64 hidden units) fed one-hot vectors, a learned dense table, or a random table that is never trained. Learning rate chosen on validation for each kind of input; three seeds.

- **The same rows, two ways:** one-hot times the table equals the lookup exactly (0.0e+00); the product was 266 times slower in this run, and one-hot inputs for 1,000 words take 16.7 MB against 4.0 kB of ids.
- **Dense beats one-hot:** test accuracy 0.8529 with d = 32 against 0.8190 for one-hot, which has 400913 weights and took 4.70 s per epoch. Even d = 2 (0.8362) beats one-hot; d = 8, 32 and 128 are level (0.8528, 0.8529, 0.8509).
- **Learning the table matters:** a random table that is never trained reaches 0.6972.
- **What the table learned:** for the 500 most frequent words, a word's nearest neighbour has the same most common tag 0.854 of the time, against 0.136 for the random table (two random words: 0.138).
- **Not shown:** why dense wins. The likely reason, one table shared by all three window positions, was not tested. The best dense learning rate (0.25) is the smallest one tried.

Study guide: [`day85_word_embeddings_complete_guide.pdf`](day85_word_embeddings_complete_guide.pdf) · Syntax walkthrough: [`day85_syntax_line_by_line.pdf`](day85_syntax_line_by_line.pdf) · [Run output](day85_run_output.txt)

Exercise: [`day85_exercise.py`](day85_exercise.py) is an embedding layer, forward (`E[ids]`) and backward (`np.add.at`), gradient-checked with repeated tokens; the plain-assignment mistake is off by 0.90 — the second part of Day 90's model ([run output](day85_exercise_run_output.txt)).

![A one-hot vector times a table is a lookup](embeddings_diagram.png)
![Accuracy by input, and the learned table in two dimensions](embeddings_results.png)

---
[← Back to main README](../README.md)
