# Week 18: Unsupervised Learning I (Days 119–125)

Weeks 16 and 17 covered supervised classical machine learning. Week 18 turns to data without labels: clustering (k-means, hierarchical clustering, DBSCAN, Gaussian mixtures) and dimensionality reduction (PCA, SVD, t-SNE and UMAP). Everything is written from scratch in NumPy on real datasets whose true groups are known, so every clustering can be judged; scikit-learn and SciPy are used only as outside checks. Days 119–124 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 125 brings the parts together. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 119 — k-means from scratch: the elbow method tested
[`day119_k_means.py`](day119_k_means.py) — Lloyd's loop, random and k-means++ starts with restarts, the silhouette, the adjusted Rand index (ARI) and an automatic elbow, all in NumPy. Data with known groups, used only to judge: iris and wine (standardised) and digits; scikit-learn runs only as an outside check.

- **Checked:** the inertia rose on 0 of 381 steps; the silhouette and ARI match scikit-learn (0.187974, 0.652481). From the same start the two versions end slightly apart (1167803.84 against 1167791.56) because of one exact tie at step 1; on mean-centred data, as scikit-learn computes it, the result matches exactly.
- **The elbow works twice and fails once:** it finds 3 on iris and wine, but k = 5 on digits, whose true number is 10.
- **k-means++ was no better than random starts here** (mean inertia 1193076 against 1183979); restarts are what help.
- **Scaling decides wine:** ARI 0.3711 raw, 0.8975 standardised.
- **No labels used:** 0.7896 of digit images land in a cluster led by their own digit; k-means fails on curved shapes (two moons, ARI 0.2475).

Study guide: [`day119_k_means_complete_guide.pdf`](day119_k_means_complete_guide.pdf) · Syntax walkthrough: [`day119_syntax_line_by_line.pdf`](day119_syntax_line_by_line.pdf) · [Run output](day119_run_output.txt)

Exercise: [`day119_exercise.py`](day119_exercise.py) is the loop, the ARI, the silhouette and the elbow, checked by hand — the first part of Day 125's build ([run output](day119_exercise_run_output.txt)).

![The k-means loop](kmeans_diagram.png)
![Elbows, scores, starts, the moons and the digit centres](kmeans_results.png)

---
[← Back to main README](../README.md)
