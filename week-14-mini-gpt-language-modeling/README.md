# Week 14: Mini-GPT & Language Modeling (Days 91–97)

Week 13 built a tiny GPT-style model on Shakespeare. Week 14 studies language modeling properly: the chain rule and honest baselines, training with real perplexity numbers, sampling, judging generated text, scaling and fine-tuning. It follows the two-track structure of Weeks 9–13: Days 91–96 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 97 builds the week's evaluation of a mini-GPT from those parts. The text is Week 13's Tiny Shakespeare ([`../week-13-tokenization-embeddings/corpora/`](../week-13-tokenization-embeddings/corpora)). See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 91 — Causal language modeling: next-token prediction from scratch
[`day91_causal_language_modeling.py`](day91_causal_language_modeling.py) — by the chain rule, a text's probability is a product of next-token predictions. Checked by enumeration, then three ways to make the predictions on Day 90's Shakespeare split (characters): counting with add-k and with interpolated Kneser-Ney smoothing (from scratch), a fixed-window neural model (Bengio et al., 2003), and Day 90's GPT. Day 90's code is imported from its script.

- **Why causal:** summed over all 81 strings of a tiny vocabulary, a causal model's probabilities give 1.000000; the same weights without the causal mask give 1.051473, so they are not probabilities at all.
- **Counting is the baseline to beat:** Kneser-Ney reaches 2.197 bits per character with 6 characters of context, where add-k stalls at 2.554 and gets worse with more context. Its discount (0.9) was chosen on held-back training text, inside the grid.
- **A fixed-window neural model:** best 2.633 at 8 characters; 16 is worse (2.685), probably undertrained at 4,000 steps (not tested further).
- **Day 90's GPT (2.525) is behind Kneser-Ney.** The target for Day 92 is below 2.197.

Study guide: [`day91_causal_language_modeling_complete_guide.pdf`](day91_causal_language_modeling_complete_guide.pdf) · Syntax walkthrough: [`day91_syntax_line_by_line.pdf`](day91_syntax_line_by_line.pdf) · [Run output](day91_run_output.txt)

Exercise: [`day91_exercise.py`](day91_exercise.py) is the Kneser-Ney character model in one class, with every context's probabilities checked to sum to 1 — the first part of Day 97's build, its counting baseline ([run output](day91_exercise_run_output.txt)).

![p(text) as next-token predictions](causal_lm_diagram.png)
![Bits per character against context, three ways](causal_lm_results.png)

## Day 92 — Training the mini-GPT on real text, with honest perplexity numbers
[`day92_training_mini_gpt.py`](day92_training_mini_gpt.py) — Day 90's GPT trained properly: a warm-up and cosine learning-rate schedule (new), the peak rate chosen by a real 300-step sweep, a bigger model (4 blocks of 128, context 128) beside Day 90's, held-out loss measured with full context (sliding windows) every 500 steps next to the same measure on training text, and perplexity reported per character and per word. The three 5,000-step runs train at the same time in separate processes. The big model is saved to `day92_mini_gpt.npz` for Days 93 and 94.

- **The sweep:** a peak of 0.006 stalls (4.639 bits); 0.003 is chosen narrowly over 0.001 (3.374 against 3.425 at step 300).
- **The schedule helps the same model:** 2.443 against 2.492 bits per character at Day 90's constant rate.
- **The big model (824129 weights) reaches 2.188** — level with Day 91's Kneser-Ney counter (2.197, scored on slightly different positions), not clearly better; its training-text loss is 1.829, a gap of 0.358 bits that is still growing.
- **Perplexity needs its unit:** 4.56 per character, 4190 per word (5.50 characters per word), for the same model.

Study guide: [`day92_training_mini_gpt_complete_guide.pdf`](day92_training_mini_gpt_complete_guide.pdf) · Syntax walkthrough: [`day92_syntax_line_by_line.pdf`](day92_syntax_line_by_line.pdf) · [Run output](day92_run_output.txt)

Exercise: [`day92_exercise.py`](day92_exercise.py) is the schedule and a full-context evaluator that takes any model as a function (a uniform model scores log2 V, an all-knowing one 0) — the second part of Day 97's build ([run output](day92_exercise_run_output.txt)).

![Measuring with full context](training_diagram.png)
![The schedule, the sweep, held-out against training text](training_results.png)

---
[← Back to main README](../README.md)
