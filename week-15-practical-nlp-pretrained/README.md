# Week 15: Practical NLP with Pretrained Models (Days 98–104)

Weeks 11–14 built attention, the Transformer and a mini-GPT from scratch. Week 15 uses the tools practitioners use: the Hugging Face `transformers` library (on PyTorch) and models pretrained by others, measured honestly on real datasets with metrics written from scratch. It follows the two-track structure of Weeks 9–14: Days 98–103 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 104 builds a small retrieval-augmented question-answering system from those parts. Models and datasets download on first use into the Hugging Face cache outside the repository and are not committed. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 98 — A pretrained-transformers library, and a real pretrained model
[`day98_pretrained_transformers.py`](day98_pretrained_transformers.py) — the Hugging Face `transformers` library on PyTorch, with DistilBERT fine-tuned for sentiment (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`), measured on SST-2's 872 validation sentences. The from-scratch parts are NumPy: a bootstrap interval, a calibration table, and a bag-of-words logistic regression as the baseline.

- **The pipeline is just tokenizer → model → softmax:** the largest gap from doing the steps by hand is 0.0e+00. The model has 66,955,010 weights.
- **The tokenizer:** WordPiece needs 1.185 pieces per word on real reviews, never produces [UNK], and padding changes probabilities by at most 1.4e-06 thanks to the attention mask.
- **Accuracy 0.9106** (95% bootstrap interval 0.8911 to 0.9289); caveat: the model's authors may have tuned on this standard development set.
- **Overconfident:** the top confidence bin holds 806 answers with mean confidence 0.997 but accuracy 0.937; expected calibration error 0.0803.
- **What pretraining buys:** the from-scratch bag of words reaches 0.6628, 0.7821 and 0.8165 with 1,000, 10,000 and all 67,349 training items.
- **Where it fails:** mixed and ironic reviews, at confidence 0.9994 and above; sentences with a negation word 0.866 against 0.921 (intervals overlap slightly).
- **CPU speed:** best at batch 8 (52.3 sentences per second); one thread is far slower (10.4).

Study guide: [`day98_pretrained_transformers_complete_guide.pdf`](day98_pretrained_transformers_complete_guide.pdf) · Syntax walkthrough: [`day98_syntax_line_by_line.pdf`](day98_syntax_line_by_line.pdf) · [Run output](day98_run_output.txt)

Exercise: [`day98_exercise.py`](day98_exercise.py) is the batching part: `batched`, `run_batched` and a safe `softmax`, checked against one-at-a-time predictions (differences under 1e-7) — the first part of Day 104's build ([run output](day98_exercise_run_output.txt)).

![What one pipeline call does](pretrained_diagram.png)
![Calibration, the bag-of-words baseline, CPU speed](pretrained_results.png)

---
[← Back to main README](../README.md)
