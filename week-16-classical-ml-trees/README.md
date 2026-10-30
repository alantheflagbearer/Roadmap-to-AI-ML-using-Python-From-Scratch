# Week 16: Classical ML I — Trees (Days 105–111)

After fifteen weeks of neural networks, Week 16 returns to classical machine learning: decision trees and the ensembles built from them, written from scratch in NumPy on real datasets. It follows the two-track structure: Days 105–110 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 111 assembles them. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 105 — Decision trees from scratch: entropy and Gini on real data
[`day105_decision_trees.py`](day105_decision_trees.py) — a decision tree in NumPy: Gini and entropy, a split search that scores every threshold of a feature at once with sorted cumulative class counts, recursive growth, 5-fold cross-validation, and bootstrap resamples. Data: the Breast Cancer Wisconsin set bundled with scikit-learn (569 tumours, 30 features; no scikit-learn models are used) and the repository's own Titanic file.

- **Checked:** the fast split search finds the brute-force best gain in 30 of 30 random problems.
- **Depth:** training accuracy reaches 1.0000 (19 leaves) while test accuracy stops at 0.9415; 5-fold CV picks depth 2 (test 0.9357, interval 0.8947 to 0.9708 — too few test tumours to separate shallow depths).
- **Gini vs entropy:** different first questions (worst radius vs worst perimeter, near-duplicates); entropy scored higher on this split, inside the noise.
- **Instability:** across 30 bootstrap resamples the first question uses five features and single trees range 0.9006 to 0.9649; their majority vote gets 0.9649 (Day 107's random forest).
- **A readable tree:** Titanic at depth 3 reads like history and beats 'women survive' (0.8022 vs 0.7724), but has a leaf made from one passenger — the case for Day 106's pruning.

Study guide: [`day105_decision_trees_complete_guide.pdf`](day105_decision_trees_complete_guide.pdf) · Syntax walkthrough: [`day105_syntax_line_by_line.pdf`](day105_syntax_line_by_line.pdf) · [Run output](day105_run_output.txt)

Exercise: [`day105_exercise.py`](day105_exercise.py) is `gini`, `entropy` and `best_split`, checked on a table small enough to solve by hand and on the edge cases — the first part of Day 111's build ([run output](day105_exercise_run_output.txt)).

![Growing a decision tree](tree_diagram.png)
![Impurity, accuracy against depth, root questions across resamples](tree_results.png)

---
[← Back to main README](../README.md)
