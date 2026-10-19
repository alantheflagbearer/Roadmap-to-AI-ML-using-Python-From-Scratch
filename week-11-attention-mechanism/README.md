# Week 11: Attention Mechanism (Days 70–76)

Week 10 ended with an encoder-decoder whose decoder sees only the encoder's final state. Week 11 measures what that costs and replaces it with attention. It follows the two-track structure of Weeks 9–10: Days 70–75 each get the full-depth lesson (real experiments, paired study-guide PDFs) plus a short `dayNN_exercise.py` isolating one piece, and Day 76 (multi-head attention) is built from those exercise scripts. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 70 — The information bottleneck: where one fixed-size state stops being enough
[`day70_information_bottleneck.py`](day70_information_bottleneck.py) — Day 69's encoder-decoder, whose decoder sees only the encoder's final (h, c), pushed to longer sequences, with controls that separate "too far back" from "too much to hold". No timings today: the runs are long enough that this machine's run-to-run speed swings would make them meaningless.

- **Outside Day 69's training range** (trained on 3–8 digits): exact match 0.947 at 8 digits, 0.113 at 10, 0 from 12 — but that mixes the bottleneck with never having seen those lengths.
- **The fair test** (trained on 3–20, three seeds, 40,000 steps): 20-digit COPY exact match 0.000 / 0.004 / 0.157 at hidden 16 / 32 / 64. A bigger state fails later, not never.
- **Too far, or too much?** Every input has 20 digits; write back only the first k. Same distance, different amount: k = 4 → 1.000, 8 → 0.986, 12 → 0.761, 16 → 0.106, 20 → 0.013. The LSTM keeps 4 digits across 20 steps perfectly; it cannot keep 20 in one state.
- **What the final state holds** (linear probe per input position): the k = 4 model keeps the first digits (1.000 at position 1) and lets the rest fall to guessing; the COPY model keeps position 1 (0.990) and only part of every middle position.
- **Not just training time:** three times the budget lifts 16-digit exact match to 0.563 at 80,000 steps, then it falls back to 0.370. **Reversing** helps (0.447 vs 0.198 at 16 digits) but reaches only 0.057 at 20.

Study guide: [`day70_information_bottleneck_complete_guide.pdf`](day70_information_bottleneck_complete_guide.pdf) · Syntax walkthrough: [`day70_syntax_line_by_line.pdf`](day70_syntax_line_by_line.pdf) · [Run output](day70_run_output.txt)

Exercise: [`day70_exercise.py`](day70_exercise.py) isolates the read that replaces the bottleneck — a masked softmax over memory slots and the weighted average — and gradient-checks it ([run output](day70_exercise_run_output.txt)).

![The bottleneck](bottleneck_diagram.png)
![Length, amount, probe and longer training](bottleneck_results.png)

## Day 71 — Additive (Bahdanau) attention, from scratch
[`day71_additive_attention.py`](day71_additive_attention.py) — the decoder keeps every encoder state and, at each output step, scores each one against its previous state (`e_i = va · tanh(s Wa + h_i Ua + ba)`), turns the scores into weights with a softmax and reads their weighted average (Day 70's exercise). Its input is [previous token ; context]; the output layer sees [state ; context]. The decoder is differentiated one step at a time (`lstm_step_backward`) because each step's input depends on its own previous state.

- **Checks:** one attention step matches hand arithmetic; every entry passes the gradient check, with the decoder starting from the encoder's final state and from zeros.
- **Day 70's hardest test, with attention** (COPY 3–20 digits, hidden 32, the same test sequences, half the steps): 1.000 exact match at every length for every seed from 7,500 steps on — where Day 70's best (hidden 64) reached 0.157 at 20 digits.
- **Hidden 16 is enough** (1.000 at every length, 5898 weights); REVERSE is solved too; a decoder starting from zeros, with attention as its only view of the input, reaches 0.971 at 20 digits.
- **An honest limit:** longer than any training sequence, exact match is 0.000 at 25 digits (token accuracy 0.742) — removing the bottleneck is not the same as handling unseen lengths.
- **Cost:** a training step took 2.45 times as long as Day 69's in this run (only the ratio is meaningful).

Study guide: [`day71_additive_attention_complete_guide.pdf`](day71_additive_attention_complete_guide.pdf) · Syntax walkthrough: [`day71_syntax_line_by_line.pdf`](day71_syntax_line_by_line.pdf) · [Run output](day71_run_output.txt)

Exercise: [`day71_exercise.py`](day71_exercise.py) isolates one additive attention read, forward and backward, gradient-checked ([run output](day71_exercise_run_output.txt)).

![One decoder step with additive attention](additive_attention_diagram.png)
![Exact match during training](attention_training_curves.png)

## Day 72 — Looking inside attention: alignment heatmaps on the toy task
[`day72_attention_heatmaps.py`](day72_attention_heatmaps.py) — Day 71's models (retrained with the same seeds, so the same models) and the weights they produce at every output step: where the decoder looked.

- **The alignment a person would draw:** a diagonal for COPY, an anti-diagonal for REVERSE. The largest weight is on the input digit being written for 1.000 of COPY outputs and about 0.99 of REVERSE outputs.
- **Where the leftover weight goes:** for REVERSE mostly to the *next* encoder state (0.201–0.237) rather than the previous one. A probe explains it: the digit at position 8 is read from h_8 at 1.000 and from h_9 at 0.999, but from h_7 only at 0.131 (guessing 0.125). COPY's smaller weight on the previous state is not explained by this probe.
- **When it forms:** untrained, 0.083 of COPY outputs look at the target (chance for 12 positions); by 1,000 steps 0.442 while exact match is still 0.002; by 5,000 steps 0.998. Learning where to look comes first or together with learning what to write.
- **Where it breaks:** on 30- and 40-digit inputs the alignment collapses after about position 20, the longest training length; digits written right had 0.675 of their weight on the target, digits written wrong 0.064.

Study guide: [`day72_attention_heatmaps_complete_guide.pdf`](day72_attention_heatmaps_complete_guide.pdf) · Syntax walkthrough: [`day72_syntax_line_by_line.pdf`](day72_syntax_line_by_line.pdf) · [Run output](day72_run_output.txt)

Exercise: [`day72_exercise.py`](day72_exercise.py) computes the three alignment measurements on hand-built weight matrices with known answers ([run output](day72_exercise_run_output.txt)).

![Alignment heatmaps and alignment during training](attention_heatmaps.png)

## Day 73 — Dot-product (Luong) attention, compared like for like with additive
[`day73_dot_product_attention.py`](day73_dot_product_attention.py) — Day 71's model with the score as a setting: additive (`va · tanh(s Wa + h Ua + ba)`, Day 71's code unchanged), **dot** (`s · h`, no weights) and **general** (`(s Wg) · h`, one matrix). Same data stream, seeds, learning rate and test sequences; only the score differs.

- **Checks:** the hand example matches; with `Wg` = identity the general score equals the dot score; every entry of both new models passes the gradient check. The additive rows reproduce Day 71 exactly (0.572 at 12 digits after 2,500 steps).
- **COPY 3–20 digits, hidden 32:** all three learn it, at different speeds. At 20 digits after 2,500 steps: general 0.806, dot 0.018, additive 0.000. Every seed perfect on every length from 7,500 steps (additive), 17,500 (general), not within 20,000 (dot: 0.997 / 1.000 / 0.997).
- **A smaller state separates them:** at hidden 16, COPY exact match at 20 digits is 1.000 (additive), 0.997 (general) — and 0.000 for the dot score (0.026 at 12 digits). The general score's only extra is a 16 × 16 matrix (5066 vs 4810 weights). REVERSE at hidden 32 works for all three (dot 0.984).
- **How sharply each looks:** argmax on the right input digit 1.000 (additive, general) but 0.840 (dot); trained score spread 9.451 (general), 6.172 (additive), only 2.463 (dot). The weightless score can only sharpen as far as the LSTM states allow — a likely reason for the hidden-16 failure, not tested here.
- **Cost:** a training step took 0.53 (dot) and 0.65 (general) times the additive step in this run.

Study guide: [`day73_dot_product_attention_complete_guide.pdf`](day73_dot_product_attention_complete_guide.pdf) · Syntax walkthrough: [`day73_syntax_line_by_line.pdf`](day73_syntax_line_by_line.pdf) · [Run output](day73_run_output.txt)

Exercise: [`day73_exercise.py`](day73_exercise.py) isolates the dot and general attention reads, forward and backward, gradient-checked ([run output](day73_exercise_run_output.txt)).

![Three score functions](score_functions_diagram.png)
![Learning speed and attention sharpness](score_functions_compared.png)

## Day 74 — Self-attention: every position attends to every position of the same sequence
[`day74_self_attention.py`](day74_self_attention.py) — queries, keys and values are three projections of the *same* sequence (`Q = X Wq`, `K = X Wk`, `V = X Wv`); each position's output is a weighted average of every position's value, weights from a masked softmax over `q · k`. Tried as a tagger on Week 9's real POS data (Day 62's split), against a control that sees each word on its own.

- **Checks:** one layer matches hand arithmetic exactly; every entry passes the gradient check with padded sentences, padded keys get weight 0.0e+00 and `<pad>`'s embedding gradient is exactly zero.
- **Order-blind, proved:** shuffle the positions and the outputs shuffle with them (largest difference 1.33e-15). On the real test set, shuffling every sentence's words leaves accuracy at 0.8075 with 0.0000 of predictions changed — Day 62's RNN dropped from 0.8327 to 0.7487.
- **Context still helps** (lr chosen on validation, three seeds): 0.8070 test accuracy against 0.7726 for the no-attention control; the gain is mostly on unknown words (0.4140 vs 0.2440). It is below Day 62's RNN, which also knows word order.
- **What it attends to:** not mainly itself (0.0656 against 0.0892 for an even spread) — certain words such as 'be' (1.81 times an even share) draw weight from the whole sentence.
- **Cost:** 4.38 times the control's time per epoch in this run (only the ratio is meaningful).

Study guide: [`day74_self_attention_complete_guide.pdf`](day74_self_attention_complete_guide.pdf) · Syntax walkthrough: [`day74_syntax_line_by_line.pdf`](day74_syntax_line_by_line.pdf) · [Run output](day74_run_output.txt)

Exercise: [`day74_exercise.py`](day74_exercise.py) isolates one masked self-attention layer, forward and backward, gradient-checked, with the shuffle test ([run output](day74_exercise_run_output.txt)).

![One self-attention layer](self_attention_diagram.png)
![The self-attention tagger](self_attention_tagger.png)

---
[← Back to main README](../README.md)
