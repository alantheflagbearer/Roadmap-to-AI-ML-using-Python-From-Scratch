# Week 12: The Transformer (Days 77–83)

Week 11 built attention up to multi-head self-attention. Week 12 builds the Transformer around it. It follows the two-track structure of Weeks 9–11: Days 77–82 each get the full-depth lesson (real experiments, paired study-guide PDFs) plus a short `dayNN_exercise.py` isolating one part, and Day 83 trains a tiny Transformer built from those exercise scripts. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 77 — Positional encoding, from scratch
[`day77_positional_encoding.py`](day77_positional_encoding.py) — the sinusoidal encoding (`PE[p, 2i] = sin(p / 10000^(2i/d))`, `PE[p, 2i+1] = cos(...)`), added to the word vectors so that self-attention can see order. Built, its key property checked, then tested on a toy task that needs position and on Week 9's real tagging task with Day 76's multi-head layer.

- **Moving along is a rotation:** one matrix R_k built from k alone maps PE[p] to PE[p + k] for every p (to 3.71e-15), so PE[p] · PE[p + k] depends only on k. Day 74's shuffle test: 8.88e-15 without the encoding, 3.18e+01 with it.
- **Toy task (write the previous token):** 0.253 with no encoding, 1.000 with sinusoidal or learned positions on the trained lengths. Beyond them both mostly fail: at 32 tokens, never-seen positions are right 0.278 (sinusoidal) and 0.130 (learned) of the time.
- **Real tagging, chosen on validation:** added to the whole word vector ('all') the encoding barely helps (0.8063 vs 0.8036) and at strength 3 training collapses (0.1400). Added only to the queries and keys at strength 3 ('qk'): 0.8487 over three seeds, above Day 62's RNN (0.8327) and Day 68's LSTM (0.8377), below the bidirectional LSTM (0.8577).
- **How:** one head puts 0.918 of its weight on the next word, another 0.569 on the previous one. Shuffling the test sentences now costs accuracy (0.8485 → 0.7376); without an encoding nothing changes.

Study guide: [`day77_positional_encoding_complete_guide.pdf`](day77_positional_encoding_complete_guide.pdf) · Syntax walkthrough: [`day77_syntax_line_by_line.pdf`](day77_syntax_line_by_line.pdf) · [Run output](day77_run_output.txt)

Exercise: [`day77_exercise.py`](day77_exercise.py) is the encoding table as one function, with its rotation and order checks — the first part of Day 83's Transformer ([run output](day77_exercise_run_output.txt)).

![Where the encoding enters](positional_encoding_diagram.png)
![The encoding, the toy task by position, and the tagger's heads](positional_encoding_results.png)

## Day 78 — The Transformer encoder block, from scratch
[`day78_encoder_block.py`](day78_encoder_block.py) — multi-head attention (Day 76) and a position-wise feed-forward network, each wrapped in "add & norm": a residual connection and layer normalisation. Post-LN (the paper's order) and pre-LN, forward and backward from scratch, stacked and tested on Week 9's tagging task with Day 77's encoding.

- **Checks:** LayerNorm by hand; the FFN works on every position on its own (shuffled positions give shuffled outputs exactly, 0.00e+00); post-LN and pre-LN stacks and every ablation pass the gradient check. The loss is now a log-softmax: Day 62's `log(p + 1e-9)` made one entry fail the check although the analytic gradient was right.
- **What keeps the gradient alive** (8 untrained blocks): with neither residuals nor LayerNorm the gradient at the bottom fades to 1.472e-02; residuals alone make it grow to 3.022e+01; with LayerNorm it stays the same size with or without residuals (1.237e+01 and 8.498e+00).
- **One block on real tagging:** 0.8510 over three seeds (lr 2.0, chosen on validation), a little above Day 77's single layer (0.8487). Without the FFN 0.8378; without LayerNorm 0.8485 and without residuals 0.8475, within the seeds' spread.
- **Deeper did not help:** 2 and 4 blocks, post-LN or pre-LN, all tested below one block (best 0.8478 for 4 post-LN blocks against 0.8535, seed 0) and reached lower training accuracy too (0.9082 to 0.9501 against 0.9652): under plain SGD they train less well, rather than overfit.

Study guide: [`day78_encoder_block_complete_guide.pdf`](day78_encoder_block_complete_guide.pdf) · Syntax walkthrough: [`day78_syntax_line_by_line.pdf`](day78_syntax_line_by_line.pdf) · [Run output](day78_run_output.txt)

Exercise: [`day78_exercise.py`](day78_exercise.py) is one post-LN encoder block on its own, forward and backward, gradient-checked, ignoring padded positions — the second part of Day 83's Transformer ([run output](day78_exercise_run_output.txt)).

![The encoder block](encoder_block_diagram.png)
![Gradient by depth, each part removed, deeper stacks](encoder_block_results.png)

## Day 79 — LayerNorm vs BatchNorm, a real comparison
[`day79_layernorm_vs_batchnorm.py`](day79_layernorm_vs_batchnorm.py) — Day 78's encoder block with the norm as a setting: LayerNorm (each position's features), BatchNorm with padding left out of its statistics, and BatchNorm with padding counted. Same starting weights for all three, on Week 9's padded sentences.

- **Accuracy is not the difference:** with batches of 32, LayerNorm 0.8510, BatchNorm 0.8520, BatchNorm counting padding 0.8504 (three seeds). With batches of 4 both drop at the same learning rate (0.7746 / 0.7785) and both recover when it is scaled down 8 times (0.8520 / 0.8503): the learning rate, not the norm.
- **Independence is:** with BatchNorm (train mode) a sentence's output changes by up to 1.39e+00 when the other sentences in its batch change, LayerNorm's not at all (0.00e+00); counting padding in BatchNorm's statistics changes outputs by 1.38e+00.
- **Test time:** the trained BatchNorm tagger scores 0.8496 with its running averages, 0.8466 with each test batch's statistics (256 sentences) and 0.8361 with batches of 4. LayerNorm has no such choice.
- BatchNorm's backward pass with masked statistics passes the gradient check inside the block.

Study guide: [`day79_layernorm_vs_batchnorm_complete_guide.pdf`](day79_layernorm_vs_batchnorm_complete_guide.pdf) · Syntax walkthrough: [`day79_syntax_line_by_line.pdf`](day79_syntax_line_by_line.pdf) · [Run output](day79_run_output.txt)

Exercise: [`day79_exercise.py`](day79_exercise.py) is LayerNorm as one function, forward and backward, gradient-checked, beside BatchNorm — the third part of Day 83's Transformer ([run output](day79_exercise_run_output.txt)).

![Which numbers each norm averages](norms_diagram.png)
![Batches of 32, small batches, BatchNorm at test time](layernorm_vs_batchnorm.png)

## Day 80 — The Transformer decoder block, and masked self-attention
[`day80_decoder_block.py`](day80_decoder_block.py) — masked (causal) self-attention, cross-attention to an encoder's output (queries from the decoder, keys and values from the encoder) and Day 78's FFN, each with add & norm. Then a decoder-only next-word model on Week 9's sentences, trained with and without the causal mask.

- **Checks:** the mask leaves exactly zero weight on later positions; changing a later word leaves the earlier outputs unchanged (0.00e+00, against 1.61e+00 without the mask); the full decoder block, including the gradient for the encoder's output, passes the gradient check.
- **Why the mask:** without it the next-word model scores 0.5133 / 0.5167 / 0.5007 on whole sentences, as in training, because it reads the next word (0.536 of its attention is one position ahead). Predicting from the past only, as when writing text, it falls to 0.0877 / 0.1222 / 0.1102, below always guessing '<unk>' (0.1853). With the mask both scores are the same: 0.2254 / 0.2442 / 0.2326.
- **An honest limit:** one small block trained with SGD on 2,000 sentences stays below a bigram table (0.2519) on accuracy.
- **One pass = one pass per word:** with the mask, a whole-sentence pass equals feeding each prefix separately (7.11e-15), which is why decoders train on whole sentences.

Study guide: [`day80_decoder_block_complete_guide.pdf`](day80_decoder_block_complete_guide.pdf) · Syntax walkthrough: [`day80_syntax_line_by_line.pdf`](day80_syntax_line_by_line.pdf) · [Run output](day80_run_output.txt)

Exercise: [`day80_exercise.py`](day80_exercise.py) is causal self-attention and cross-attention on their own, gradient-checked — the fourth part of Day 83's Transformer ([run output](day80_exercise_run_output.txt)).

![The decoder block](decoder_block_diagram.png)
![Trained with and without the causal mask](causal_mask_results.png)

---
[← Back to main README](../README.md)
