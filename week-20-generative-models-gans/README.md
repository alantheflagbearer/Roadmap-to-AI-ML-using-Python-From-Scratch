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

## Day 134 — A GAN from scratch: generator and discriminator, a real minimax training loop
[`day134_gan_from_scratch.py`](day134_gan_from_scratch.py) — a generator and a discriminator written from scratch on Day 126's network parts, trained against each other on a 1-D two-peaked data set (whose density is known) and on MNIST, and watched during training rather than only at the end.

- **Checked three ways:** the losses at D = 0.5 match 2 log 2 and log 2 by hand; the generator's gradient, which flows back through the discriminator, matches finite differences (worst relative error 4.2e-07); a trained discriminator lands within 0.0562 of the optimal D* = p_data / (p_data + p_g), and the game's value matches −log 4 + 2 JSD (-0.6537).
- **Why the non-saturating loss:** when the discriminator is sure a fake is fake, log(1 − D) sends back -0.0025 per fake and −log D sends -0.9975.
- **The 1-D game:** two of five seeds sat on one peak for 1,000 to 3,000 steps before finding the other; after that the distance to the data kept moving between 0.104 and 0.709 instead of settling.
- **MNIST, one setting apart:** with Adam's usual momentum (b1 0.9) some digits were never generated and the commonest took 0.630 of the samples; with b1 0.5 all ten digits appear by epoch 10, and the samples are 5.74 from their nearest training image (a real test digit: 4.49), so they are not copies.

Study guide: [`day134_gan_from_scratch_complete_guide.pdf`](day134_gan_from_scratch_complete_guide.pdf) · Syntax walkthrough: [`day134_syntax_line_by_line.pdf`](day134_syntax_line_by_line.pdf) · [Run output](day134_run_output.txt)

Exercise: [`day134_exercise.py`](day134_exercise.py) is the GAN's parts (Adam with a momentum setting, both losses, the generator, its gradient through the discriminator, one round of the game), checked by hand — the second part of Day 139's build ([run output](day134_exercise_run_output.txt)).

![A GAN: two networks in one minimax game](gan_diagram.png)
![The 1-D game, the optimal discriminator, and MNIST samples](gan_results.png)

## Day 135 — GAN training instability: a real, honest mode-collapse demonstration
[`day135_gan_instability.py`](day135_gan_instability.py) — Day 134's GAN on eight tight Gaussians on a circle, where every mode is known, watched every 250 steps: how many modes the generator covers and how many of its samples are good, over five seeds, two generator losses, a discriminator head start and three imbalances between the players.

- **Collapse with the usual settings:** the non-saturating loss covered all 8 modes in 1 of 5 seeds; the others missed 1 to 3 modes while their samples stayed good (good share 0.91 to 0.95).
- **The textbook loss lost:** with equal learning rates the minimax loss covered all 8 modes in 5 of 5 seeds. Its known weakness (a gradient of about D(fake), 999 times smaller than the non-saturating one at D(fake) = 0.001) appears once the discriminator gets a 1,000-step head start: good share down to 0.06, modes lost up to 25 times; the non-saturating loss then collapses onto 3 modes in three seeds.
- **Balance matters:** a discriminator ten times slower than the generator gave good shares of 0.07 to 0.08; none of the nine imbalanced runs covered all 8 modes.

Study guide: [`day135_gan_instability_complete_guide.pdf`](day135_gan_instability_complete_guide.pdf) · Syntax walkthrough: [`day135_syntax_line_by_line.pdf`](day135_syntax_line_by_line.pdf) · [Run output](day135_run_output.txt)

Exercise: [`day135_exercise.py`](day135_exercise.py) is the ring, the mode count and the watched training loop, checked by hand — the third part of Day 139's build ([run output](day135_exercise_run_output.txt)).

![Three ways GAN training goes wrong](gan_instability_diagram.png)
![Mode coverage over training, by loss, seed and balance](gan_instability_results.png)

---
[← Back to main README](../README.md)
