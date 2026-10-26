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

---
[← Back to main README](../README.md)
