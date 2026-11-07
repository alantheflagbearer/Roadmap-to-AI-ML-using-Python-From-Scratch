# Week 20: Generative Models — GANs (Days 133–139)

Week 19 learned representations of images without labels, and its variational autoencoder could already draw new ones. Week 20 is about generating data: what a generative model is, as against a discriminative one, then generative adversarial networks (a generator and a discriminator trained against each other), their instability, and how to judge what they make. Everything is written from scratch in NumPy, on MNIST (cached by Day 126) and small datasets whose true distribution is known. Days 133–138 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 139 brings the parts together. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 133 — Generative vs discriminative models: a real conceptual comparison
[`day133_generative_vs_discriminative.py`](day133_generative_vs_discriminative.py) — Gaussian class models from scratch (naive Bayes, LDA with one shared covariance, QDA with one per class, all through Cholesky factors) against logistic regression (Day 116's softmax regression), on MNIST and scikit-learn's bundled breast-cancer table: learning curves, and the three things only the generative model can do.

- **The same family, two routes:** LDA's posterior is exactly a softmax of linear scores (checked to 8.0e-14); on all 784 pixels LDA scores 0.8798, logistic regression 0.8917, but logistic regression is the more overconfident (log-loss 0.9116 against 0.5019).
- **Little data against a lot:** on MNIST (50 PCA numbers) LDA has the lower error up to 300 images (0.2922 against 0.3702 at 100), the two are level at 1,000, and logistic regression wins from 3,000 (0.0913 against 0.1278 at 60,000).
- **Generating:** digits drawn from QDA are recognised 0.8620 of the time, as often as real digits squeezed through the same 50 numbers (0.8600); with too much regularisation (1.0), samples of 1 were recognised only 0.01 of the time.
- **Half of every image hidden:** LDA, leaving the hidden pixels out, scores 0.7771; logistic regression with them filled with zeros, 0.3923.
- **Knowing what it does not know:** log p(x) separates random noise from digits perfectly (AUC 1.0000), while logistic regression is 1.0000 confident on noise.

Study guide: [`day133_generative_vs_discriminative_complete_guide.pdf`](day133_generative_vs_discriminative_complete_guide.pdf) · Syntax walkthrough: [`day133_syntax_line_by_line.pdf`](day133_syntax_line_by_line.pdf) · [Run output](day133_run_output.txt)

Exercise: [`day133_exercise.py`](day133_exercise.py) is the Gaussian class models (fit, Bayes' rule, marginals, log p(x), sampling), checked by hand — the first part of Day 139's build ([run output](day133_exercise_run_output.txt)).

![Two routes to p(class | x)](generative_discriminative_diagram.png)
![Learning curves, sampled digits, missing pixels and unusual images](generative_discriminative_results.png)

---
[← Back to main README](../README.md)
