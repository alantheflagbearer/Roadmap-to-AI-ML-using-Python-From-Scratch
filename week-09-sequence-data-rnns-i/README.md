# Week 9: Sequence Data & RNNs I (Days 56–62)

The first themed week built with the new two-track structure: Days 56–61 each get **two** deliverables — the usual full-depth lesson (gradient checks, real experiments, paired study-guide PDFs) plus a short, lightweight `dayNN_exercise.py` isolating that day's one component in the simplest form possible (no PDFs — just a quick, runnable script). Day 62 combines the six exercise scripts into one final, complete, trained RNN, evaluated end-to-end. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the full week's plan.

## Day 56 — What is sequential data: why feedforward and CNN nets fail on it
[`day56_sequential_data_intro.py`](day56_sequential_data_intro.py) — a synthetic but genuine "order matters" task (a length-30 sequence with a HIGH spike and a LOW spike; label = whether high comes before low) demonstrates two distinct, measured failure modes rather than asserting them. `PlainFCNet` (dense, flattened input) is trained with both spikes confined to positions 0–14, then tested on positions 15–29 — its first-layer weights are position-specific, so it has no mechanism to transfer a learned pattern to a new location. `Conv1DNet` (one 1D conv layer + global average pooling, reusing Day 53's `conv2d_forward_mc`/`backward_mc` via a reshape trick) has translation invariance via weight sharing, but only within its kernel's own receptive field.

Real results: PlainFCNet reaches train_acc=1.000 and held-out-seen-position test_acc=0.888 (it genuinely learned the rule) — but SHIFTED-position test_acc collapses to 0.550, chance level. Conv1DNet reaches 0.885 test accuracy on spike pairs within its kernel's reach ("near") vs. 0.765 on pairs farther apart ("far") — a real, repeatable gap, though not the full collapse to chance PlainFCNet showed (global average pooling leaks a faint boundary-driven positional signal even when it can't preserve true relative order). Getting an honest reading of the second result took a real correction: the first training configuration was too undertrained to trust and produced the *opposite*, wrong pattern — retraining harder reproduced the theoretically-expected result cleanly.

Study guide: [`day56_sequential_data_intro_complete_guide.pdf`](day56_sequential_data_intro_complete_guide.pdf) · Syntax walkthrough: [`day56_syntax_line_by_line.pdf`](day56_syntax_line_by_line.pdf) · [Run output](day56_run_output.txt)

Exercise: [`day56_exercise.py`](day56_exercise.py) isolates just the PlainFCNet position-generalization result in a short, standalone script ([run output](day56_exercise_run_output.txt)) — this is the baseline Day 62's final integrated model will be measured against.

![Two real failure modes: no position invariance vs. a capped receptive field](sequential_failure_modes.png)

## Day 57 — Vanilla RNN forward pass from scratch on a real toy sequence
[`day57_rnn_forward_pass.py`](day57_rnn_forward_pass.py) — builds the RNN recurrence `h_t = tanh(x_t @ Wxh + h_{t-1} @ Whh + bh)`, the same three parameters reused at every timestep. No backward pass yet (that's Day 58), so verification takes a different form: the looped implementation is checked against an explicit hand-unrolled computation (exact match), then the same untrained network is run forward on Day 56's task at three different sequence lengths (10, 20, 30) with zero errors — direct proof against Day 56's `PlainFCNet` `ValueError`.

The headline result is a real, dramatic measurement: does information from position 0 survive to the final hidden state at long range, the way Conv1DNet's capped receptive field structurally couldn't? A finite-difference sensitivity check finds it starts at 0.70 (T=5), is still 0.44 at T=20 — then collapses to 0.000024 by T=40 and is *exactly* 0.0 by T=160 (float64 runs out of precision to represent it). There's no hard structural cutoff like Conv1DNet's kernel size, but the connection collapses in practice anyway — a forward-pass-only preview of Day 59's real topic (vanishing/exploding gradients), visible before backpropagation is even built.

Study guide: [`day57_rnn_forward_pass_complete_guide.pdf`](day57_rnn_forward_pass_complete_guide.pdf) · Syntax walkthrough: [`day57_syntax_line_by_line.pdf`](day57_syntax_line_by_line.pdf) · [Run output](day57_run_output.txt)

Exercise: [`day57_exercise.py`](day57_exercise.py) isolates just the RNN cell (`rnn_step_forward`/`rnn_forward`) and its hand-unrolled correctness check ([run output](day57_exercise_run_output.txt)) — Day 58's exercise will backprop through exactly this function.

![Does information from x_0 survive to h_T at long range?](rnn_long_range_sensitivity.png)

## Day 58 — RNN backward pass (BPTT) from scratch + gradient check
[`day58_rnn_bptt.py`](day58_rnn_bptt.py) — backpropagation through time: ordinary backprop on the unrolled network, plus one rule — the same `Wxh`, `Whh`, `bh` are used at every timestep, so their gradients are *summed* across all of them, and gradient is handed from `h_t` back to `h_{t-1}` through the same `Whh`. A numeric gradient check passes on all five parameters (max relative error 2.17e-08). Trained on Day 56's task at T=10, the RNN goes from 0.395 test accuracy (random weights) to 1.000.

Then a prediction that turned out to be wrong. Day 57's collapsing forward-pass sensitivity suggested T=40 should struggle — it doesn't: identical training reaches test_acc=0.995, holding 0.989 even when the first spike must be remembered 30–39 steps. The diagnosis, measured rather than guessed: after training, sensitivity to a *tiny* nudge at x₀ drops ~1000x, while the effect of a full-size +3 spike at x₀ grows ~12x. The network learned to ignore small noise and **latch** onto large spikes — ~20% of its final hidden units sit pinned near tanh's ±1, and `Whh`'s largest eigenvalue grew to 2.17, which tanh's bound turns into stable saturated memory rather than explosion. Day 57's tiny-epsilon derivative is a local, linear measurement; latching is a large-signal, nonlinear effect it can't see. Vanishing gradients are still real — this task just routes around them, which is why Day 59 needs one that doesn't.

Study guide: [`day58_rnn_bptt_complete_guide.pdf`](day58_rnn_bptt_complete_guide.pdf) · Syntax walkthrough: [`day58_syntax_line_by_line.pdf`](day58_syntax_line_by_line.pdf) · [Run output](day58_run_output.txt)

Exercise: [`day58_exercise.py`](day58_exercise.py) adds `rnn_backward` to Day 57's exercise cell and checks every single gradient entry numerically ([run output](day58_exercise_run_output.txt), max relative error 2.77e-10).

![Both lengths train; training suppresses noise and latches spikes](bptt_training_short_vs_long.png)

## Day 59 — Vanishing and exploding gradients: a real demonstration
[`day59_vanishing_exploding_gradients.py`](day59_vanishing_exploding_gradients.py) — Day 58's task could be latched onto, so today uses one that can't: every input is N(0,1) and the label is just the sign of the *first* input, so gradient must reach step 0. A new probe records ∂L/∂hₜ at every timestep, gradient-checked first through the ∂L/∂xₜ it implies (max relative error 1.96e-08).

- **Gradient vs. distance** (untrained, T=50, sweeping `Whh`'s spectral radius ρ): ~1e-22 of the gradient survives to step 0 at ρ=0.5; ~50,000× *more* arrives at ρ=3.0. The boundary isn't the textbook ρ=1 — with N(0,1) inputs gradient is preserved near ρ≈2, because each step also multiplies by tanh's slope; shrinking inputs so tanh stays linear moves the boundary back to ≈1.2.
- **The real cost:** the RNN learns at T=2/5/10 (test ≈0.98–0.99) and fails at chance at T=20/40. The gradient ratio *at initialization* falls 0.91 → 0.54 → 0.21 → 0.018 → 0.00033; learning survives 0.21 and dies at 0.018. The failed networks never stored x₀ — flipping it moves the final state 5–14× *less* than resampling one irrelevant input. (A post-training ratio of 0.40 at T=40 looked healthy but was measured after `Whh` had drifted into the large-radius regime.) **⚠ Corrected by Day 60:** tracking accuracy *during* training showed the T=20 network actually learned the task (0.925) and was then destroyed by a single gradient spike at step 865 — so "vanishing gradients blocked learning" was the wrong conclusion for T=20; the collapse was an exploding-gradient event, and clipping fixes it. "Never stored x₀" describes only the final, damaged networks.
- **Exploding during ordinary training**, default init, T=20: training pushes ρ from 1.25 to 2.85, crossing 2.0 at step 430; all 11 gradient spikes (up to 47× the median) fall in steps 406–868, ten after the crossing, and each jolts the loss 13× more than a typical step. From ρ=4, spikes reach 1,909 — ~4,000× the median. This run is Day 60's baseline for gradient clipping.

Study guide: [`day59_vanishing_exploding_gradients_complete_guide.pdf`](day59_vanishing_exploding_gradients_complete_guide.pdf) · Syntax walkthrough: [`day59_syntax_line_by_line.pdf`](day59_syntax_line_by_line.pdf) · [Run output](day59_run_output.txt)

Exercise: [`day59_exercise.py`](day59_exercise.py) adds one line to Day 58's BPTT loop — record ‖dh‖ at every step — and runs it at ρ = 0.5, 1.0, 3.0 ([run output](day59_exercise_run_output.txt)).

![Gradient reaching earlier timesteps, by spectral radius](gradient_vs_distance.png)
![Trainability vs. sequence length](trainability_vs_length.png)
![Exploding gradients during training](exploding_during_training.png)

## Day 60 — Gradient clipping from scratch: an honest before/after on Day 59
[`day60_gradient_clipping.py`](day60_gradient_clipping.py) — clipping by global norm: if the norm of *all* gradients together exceeds a threshold c, scale every one by the same factor c/‖g‖. Checked directly: an exact no-op below the threshold; above it the norm lands exactly on c with cosine similarity 1.00000000 to the original. Element-wise value clipping, for contrast, bends the direction (cosine 0.76–0.90).

Day 59 predicted clipping would tame the spikes but *not* rescue its T=20 run, since clipping can't enlarge vanishing gradients. **That was wrong.** Re-running Day 59's exact run (unclipped reproduces its 0.4950 exactly) while tracking accuracy *during* training showed the network had **learned** — 0.925 at step 820 — until a single gradient of 31.4 (47× the median), applied at step 865 as a step of length 1.572, knocked it to 0.463 for good. Clipped at 1.0 (no step longer than 0.05), the same run finishes at **0.955**.
- The rescue holds at every threshold from 0.5 to 10 (finals 0.953–0.975).
- Clipping also makes 4–20× larger learning rates usable at T=10: unclipped runs end at 0.49–0.69 (with `Whh`'s spectral radius reaching 1,407 at lr=1.0), clipped at 0.96–0.995.
- **Its limit:** at T=40 no threshold produces a network that learns *and stays learned* — the edge of what a vanilla RNN can train, and the motivation for the LSTM (Day 63).

This also corrects Day 59: its T=20 failure was an exploding-gradient collapse, not vanishing gradients — visible only by watching training, not its end state. Day 59's guide now carries a red update box saying so.

Study guide: [`day60_gradient_clipping_complete_guide.pdf`](day60_gradient_clipping_complete_guide.pdf) · Syntax walkthrough: [`day60_syntax_line_by_line.pdf`](day60_syntax_line_by_line.pdf) · [Run output](day60_run_output.txt)

Exercise: [`day60_exercise.py`](day60_exercise.py) isolates the clipper and its checks, and shows value clipping squashing the recurrent gradient's relative size 15× while global-norm clipping preserves it exactly ([run output](day60_exercise_run_output.txt)).

![Unclipped learns then collapses; clipped holds](clipping_before_after.png)
![Clipping: robust at T=20, enables large lr at T=10, can't fix T=40](clipping_limits.png)

## Day 61 — A many-to-one RNN classifier on a real short-text sentiment dataset
[`day61_many_to_one_sentiment.py`](day61_many_to_one_sentiment.py) — the first real text of the week: 3,000 one-sentence reviews from Amazon, IMDb and Yelp, each labelled positive or negative (Kotzias et al., KDD 2015 — the UCI "Sentiment Labelled Sentences" dataset, included in [`sentiment_labelled_sentences/`](sentiment_labelled_sentences) with its original readme). "Many-to-one" = many words in, one label out. Three new pieces, each checked: a text pipeline (tokenize → vocabulary built on training sentences only → integer ids → pad), an embedding table (forward is a row lookup, backward is a scatter-add via `np.add.at`), and masking (a padded step carries the hidden state forward unchanged). The gradient check passes on all six parameters for both readouts (max relative error 3.85e-08), and a 3-word sentence gets the identical prediction alone and inside a batch padded to 40 (difference 0.00e+00 with the mask, 1.19e-02 without).

Then the honest comparison, on a 2,000 / 400 / 600 split, against an order-blind bag-of-words logistic regression:
- **The RNN learns, then memorizes.** Best validation epoch: test 0.760; by the last epoch training accuracy is 0.993 and test has slid to 0.728. Read without the mask, the same trained network scores 0.518 — chance.
- **It does not beat the baseline.** Bag-of-words: **0.817** test. RNN reading its last state: **0.737** (five seeds, 0.677–0.773). RNN reading the mean of all states: **0.797** (0.777–0.808) — better and far steadier than last, but still not ahead of the baseline. About 14% of test tokens are words the model never learned.
- **It is not using word order.** Shuffling the words of every test sentence flips 20.5% of the last-state RNN's predictions, yet accuracy barely moves (0.737 → 0.726). Training on sentences scrambled once costs nothing (0.744); re-scrambling every epoch *helps* (0.790), acting as data augmentation that stops the network memorizing exact sequences. Hand-written negation probes show both models treating "not" as a negative word rather than a switch ("not bad" → P(positive) 0.001).

This is a negative result reported as measured: it does not show that order is unimportant for sentiment, only that this vanilla RNN, from scratch on 2,000 sentences, did not learn to use it. From this day on, each section of the complete guide opens with a plain-language **definition** and a **real-world scenario** before the specification, code and output.

Study guide: [`day61_many_to_one_sentiment_complete_guide.pdf`](day61_many_to_one_sentiment_complete_guide.pdf) · Syntax walkthrough: [`day61_syntax_line_by_line.pdf`](day61_syntax_line_by_line.pdf) · [Run output](day61_run_output.txt)

Exercise: [`day61_exercise.py`](day61_exercise.py) isolates the embedding lookup (including why `dE[ids] += dX` silently loses a repeated word's gradient) and masking, on a six-word vocabulary with no training ([run output](day61_exercise_run_output.txt)).

![Training accuracy climbs to ~1.0 while validation and test stall](sentiment_training_curve.png)
![Readout, sentence length, and word-order ablation](sentiment_readout_and_order.png)

## Day 62 — The Week 9 build: a many-to-many RNN tagger from the six exercise parts
[`day62_many_to_many_tagger.py`](day62_many_to_many_tagger.py) — two jobs in one day. The syllabus topic is the **many-to-many** RNN (an output for *every* input), and this is the week's integration day, so the final model is assembled from the six exercise scripts of Days 56–61: Day 56's helpers and `PlainFCNet` (the no-memory baseline), Day 57's cell and forward pass, Day 58's BPTT, Day 59's gradient probe, Day 60's clipper, Day 61's ids/padding/embedding. Only one idea is new, and in BPTT it is one line: `dh = dh + dH[:, t, :]` — every timestep's own output adds its gradient. The task is part-of-speech tagging on real, hand-tagged sentences from the Universal Dependencies English Web Treebank (Silveira et al., 2014; CC BY-SA 4.0 — a trimmed word/tag copy with licence and attribution is in [`ud_english_ewt_pos/`](ud_english_ewt_pos)): 2,001 training sentences, 430 validation, 1,647 test, split at document boundaries. 20.2% of test words are unknown to the model.

- **The assembled model is correct.** A gradient check through the whole pipeline passes on all six parameters. One entry showed a relative error of 3.12e-06; rather than accept or dismiss it, the entry was examined — its gradient is 4e-07 and the absolute disagreement 2.63e-12, growing as the nudge shrinks, which is the signature of rounding in the numeric estimate, not a bug.
- **It beats the no-memory baselines, modestly.** Most-frequent-tag (pure counting): 0.8016. Day 56's dense net fed one word at a time: 0.7987. RNN tagger, mean of three seeds: **0.8327**.
- **The gain is where context is needed — and memory has a cost.** Unknown words 0.4914 vs 0.3705; ambiguous words 0.8740 vs 0.8474; but *unambiguous* words slightly worse, 0.9693 vs 0.9814. The same tagger with its recurrent weights held at zero scores 0.7714. After *to*, the tagger calls an unknown word a VERB and is right 0.639 of the time (baseline 0.063); after a comma it is worse than the baseline (0.303 vs 0.460).
- **Order matters here.** Shuffling the test words drops the tagger from 0.8327 to 0.7487 — the test that changed nothing on Day 61.
- **Day 59's probe:** with a loss at every word, the gradient at the first word of a 20-word sentence is 3.07e-02; with a loss on the last word only, the same network passes back 1.05e-05. Many-to-many does not cure vanishing gradients, it avoids depending on them.
- **Day 60's clipping:** no measurable effect at this lesson's hidden size (32). At hidden 64 it decides the outcome: at lr 2.0 all three unclipped seeds collapse (mean 0.4930) while the clipped ones reach 0.8206, with clipping firing on only 0.61% of steps.
- **A limit:** the tagger reads left to right. The same tagger trained on reversed sentences scores the same (0.8353 vs 0.8338, seed 0) but disagrees on 17.9% of words, and at least one of the two is right on 90.2% — information to the right that a one-direction RNN never sees.

Study guide: [`day62_many_to_many_tagger_complete_guide.pdf`](day62_many_to_many_tagger_complete_guide.pdf) · Syntax walkthrough: [`day62_syntax_line_by_line.pdf`](day62_syntax_line_by_line.pdf) · [Run output](day62_run_output.txt)

No separate exercise script today: Day 62 *is* the week's exercise — the six earlier ones combined. The lesson script trains 31 small networks and takes roughly 10–20 minutes to run on a laptop.

![How the six exercise parts fit together](week9_parts_diagram.png)
![Tagger training curve](tagger_training_curve.png)
![Where memory helps, the gradient probe, and clipping](tagger_memory_and_parts.png)

---
[← Back to main README](../README.md)
