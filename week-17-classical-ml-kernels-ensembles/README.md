# Week 17: Classical ML II — Kernels & Ensembles (Days 112–118)

Week 16 built trees and the ensembles made from them. Week 17 covers the rest of the classical toolbox: support vector machines and the kernel trick, Naive Bayes, k-nearest neighbours, stacking, imbalanced data, and how to choose between models. Everything is written from scratch in NumPy on real datasets, with scikit-learn used only as an outside check. It follows the two-track structure: Days 112–117 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 118 assembles them. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 112 — Support vector machines from scratch: separable and non-separable data
[`day112_support_vector_machines.py`](day112_support_vector_machines.py) — a linear SVM whose dual problem is solved by SMO (two multipliers at a time, the most violating pair first), with the weights, bias and margin recovered from the multipliers. Data: iris (setosa against versicolor is separable) and breast cancer; scikit-learn's SVC runs only as an outside check.

- **Checked:** the duality gap is 0.013% of the objective, the same 35 support vectors as scikit-learn, and the same prediction on every test tumour (1.0000). The first version never finished: rounding left multipliers at about 1e-17, and snapping them onto their bounds fixed it.
- **The widest gap:** on separable irises the margin is 1.3038 cm, set by 2 support vectors; the perceptron's gap varies from 0.0109 to 1.2132 cm depending only on the order it saw the flowers.
- **Only the support vectors matter:** retraining on them alone changes w by 1.6e-04, about the solver's tolerance; without half of them, by 4.9e-01.
- **C and scaling:** C trades margin (6.2727 down to 0.0808) for training errors. Unscaled, the SVM scores 0.6696 against 0.9746, and SMO did not finish on 19 of 20 splits. Against logistic regression it is no better (-0.0023).

Study guide: [`day112_support_vector_machines_complete_guide.pdf`](day112_support_vector_machines_complete_guide.pdf) · Syntax walkthrough: [`day112_syntax_line_by_line.pdf`](day112_syntax_line_by_line.pdf) · [Run output](day112_run_output.txt)

Exercise: [`day112_exercise.py`](day112_exercise.py) is the solver alone, which only ever sees a matrix of dot products, checked by hand on four points — the first part of Day 118's build ([run output](day112_exercise_run_output.txt)).

![A support vector machine, from the problem to the line](svm_diagram.png)
![Iris margins, and C on breast cancer](svm_results.png)

## Day 113 — The kernel trick from scratch: a real RBF-kernel demonstration
[`day113_kernel_trick.py`](day113_kernel_trick.py) — linear, polynomial and RBF kernels, the explicit features a polynomial kernel stands for, Day 112's SMO unchanged with one-vs-rest for ten classes, and random Fourier features. Data: breast cancer and handwritten digits.

- **The trick, checked:** a degree-2 polynomial kernel equals the dot products of 2,145 explicit features to 5.7e-14; degree 5 would need 11,238,513.
- **Valid kernels:** tanh(x.z - 1), once widely used as a 'sigmoid kernel', has 62 negative eigenvalues on 300 images.
- **A real curved boundary:** on two breast-cancer measurements, gamma from 0.01 to 100 bends the boundary from straight to wrapping single tumours (train 0.9749, test 0.9474). A straight line is as good here (0.9591).
- **Where it helps:** on digits RBF beats linear on all 5 splits (0.9814 against 0.9633); on breast cancer the gain is small (+0.0058). Tuned by CV (the grid was extended past its first edge), digits reach 0.9870.
- **Random features:** a linear SVM on 3,000 random Fourier features matches the exact kernel (0.9566 against 0.9570).

Study guide: [`day113_kernel_trick_complete_guide.pdf`](day113_kernel_trick_complete_guide.pdf) · Syntax walkthrough: [`day113_syntax_line_by_line.pdf`](day113_syntax_line_by_line.pdf) · [Run output](day113_run_output.txt)

Exercise: [`day113_exercise.py`](day113_exercise.py) is the kernels and random features, checked by hand — the second part of Day 118's build ([run output](day113_exercise_run_output.txt)).

![The kernel trick](kernel_diagram.png)
![Boundaries for each kernel, and random features](kernel_results.png)

---
[← Back to main README](../README.md)
