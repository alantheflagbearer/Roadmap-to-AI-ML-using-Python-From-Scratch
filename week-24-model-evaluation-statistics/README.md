# Week 24: Model Evaluation & Statistics (Days 161–167)

Week 23 built recommenders and judged them. Week 24 turns judging into a toolkit: how error splits into bias and variance, how to test whether a difference is real, how sure a score is, what happens when many things are tested at once, whether a model's probabilities can be trusted, and whether it treats groups of people alike. It uses two real public datasets: UCI Adult (census incomes, 48,842 people; B. Becker and R. Kohavi, 1996) and Kevin Hillstrom's MineThatData email experiment (64,000 customers randomly assigned to an email or none, 2008). The data is not stored in this repository: the scripts download it on first run to `~/week24_data`. Every method is written from scratch in NumPy. Days 161–166 each get the full-depth lesson plus a short `dayNN_exercise.py` holding that day's tool, and Day 167 builds model explanations from scratch and uses all six tools for one model's report card. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 161 — Bias, variance and learning curves on real census data
[`day161_bias_variance_learning_curves.py`](day161_bias_variance_learning_curves.py) — Week 24 starts on UCI Adult (30162 training-file rows without missing values; is income above $50K?). Three models are compared throughout: logistic regression fitted by Newton's method, gradient boosting (Day 109's, imported unchanged), and one deep tree. The training file is split 80/20 with seed 0 into 24130 fitting rows and 6032 validation rows, and the test file is kept back for Day 167.

- **The reference fit:** Newton converged in 8 steps (largest gradient left 5.2e-17), with validation log loss 0.3410. **First draft:** Day 116's Adam at rate 0.5 stopped at 0.4070. Always predicting the base rate scores Brier 0.1869.
- **Learning curves:** logistic regression levels off at 0.110 on validation, with training and validation close together. Boosting goes from 0.142 at 250 rows to 0.092. The deep tree fits its training rows almost perfectly (0.000 at 250 rows) and is still at 0.182 on validation at the full size.
- **Bias and variance measured directly**, from up to 20 disjoint training sets per size: logistic regression's bias² + noise barely moves (0.1104 to 0.1099) while its variance shrinks from 0.0186 to 0.0005. At 8000 rows the deep tree's variance is still 0.0655.
- **A depth sweep and a first-draft error:** the first draft's bias term rose after depth 8 (0.0994 to 0.1056 at depth 20). That came from estimating with a finite number of models; subtracting variance / B fixes it (falling to 0.0973). Depth 5 is best (0.1144).
- **Extrapolation:** a + b·n^(−c), fitted on sizes up to 4000, predicted the full-size score to within +0.0040 (the tree), -0.0001 (boosting) and +0.0003 (logistic regression).

Study guide: [`day161_bias_variance_learning_curves_complete_guide.pdf`](day161_bias_variance_learning_curves_complete_guide.pdf) · Syntax walkthrough: [`day161_syntax_line_by_line.pdf`](day161_syntax_line_by_line.pdf) · [Run output](day161_run_output.txt)

Exercise: [`day161_exercise.py`](day161_exercise.py) holds the week's data loader and split, both feature matrices, the Newton logistic regression, the Brier score and the decomposition. Its hand checks: a logistic regression on two groups recovers probabilities [0.2, 0.7], and a decomposition worked on paper gives Brier 0.165 = 0.085 + 0.040. [Exercise output](day161_exercise_run_output.txt)

![Learning curves and the direct bias-variance measurement](bias_variance_diagram.png)
![Learning curves, bias and variance by size, the depth sweep and extrapolation](bias_variance_results.png)

---
[← Back to main README](../README.md)
