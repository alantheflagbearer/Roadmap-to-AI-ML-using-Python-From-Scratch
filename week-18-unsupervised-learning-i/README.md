# Week 18: Unsupervised Learning I (Days 119–125)

Weeks 16 and 17 covered supervised classical machine learning. Week 18 turns to data without labels: clustering (k-means, hierarchical clustering, DBSCAN, Gaussian mixtures) and dimensionality reduction (PCA, SVD, t-SNE and UMAP). Everything is written from scratch in NumPy on real datasets whose true groups are known, so every clustering can be judged; scikit-learn and SciPy are used only as outside checks. Days 119–124 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 125 brings the parts together. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 119 — k-means from scratch: the elbow method tested
[`day119_k_means.py`](day119_k_means.py) — Lloyd's loop, random and k-means++ starts with restarts, the silhouette, the adjusted Rand index (ARI) and an automatic elbow, all in NumPy. Data with known groups, used only to judge: iris and wine (standardised) and digits; scikit-learn runs only as an outside check.

- **Checked:** the inertia rose on 0 of 381 steps; the silhouette and ARI match scikit-learn (0.187974, 0.652481). From the same start the two versions end slightly apart (1167803.84 against 1167791.56) because of exact ties at step 1 (two images equally far from two centres); on mean-centred data, as scikit-learn computes it, the result matches exactly.
- **The elbow works twice and fails once:** it finds 3 on iris and wine, but k = 5 on digits, whose true number is 10.
- **k-means++ was no better than random starts here** (mean inertia 1193076 against 1183979); restarts are what help.
- **Scaling decides wine:** ARI 0.3711 raw, 0.8975 standardised.
- **No labels used:** 0.7896 of digit images land in a cluster led by their own digit; k-means fails on curved shapes (two moons, ARI 0.2475).

Study guide: [`day119_k_means_complete_guide.pdf`](day119_k_means_complete_guide.pdf) · Syntax walkthrough: [`day119_syntax_line_by_line.pdf`](day119_syntax_line_by_line.pdf) · [Run output](day119_run_output.txt)

Exercise: [`day119_exercise.py`](day119_exercise.py) is the loop, the ARI, the silhouette and the elbow, checked by hand — the first part of Day 125's build ([run output](day119_exercise_run_output.txt)).

![The k-means loop](kmeans_diagram.png)
![Elbows, scores, starts, the moons and the digit centres](kmeans_results.png)

## Day 120 — Hierarchical clustering from scratch: a real dendrogram
[`day120_hierarchical_clustering.py`](day120_hierarchical_clustering.py) — agglomerative clustering with single, complete, average and Ward linkage through one Lance–Williams update, cutting the tree, cophenetic correlation, and the dendrogram laid out and drawn by hand. Data: iris and wine (standardised) and digits, labels used only to judge; SciPy's `linkage` is an outside check, and Day 119's k-means is imported for comparison.

- **Checked:** single, average and Ward match SciPy to 7.0e-15. Complete linkage differed (2.4e-02) because iris has 1631 tied distances; with ties broken by tiny noise all four agree to 1.3e-10.
- **Ward wins everywhere:** ARI 0.6153 on iris, 0.7899 on wine, 0.7940 on digits, where it beats Day 119's k-means (0.6525) despite a higher inertia.
- **Single linkage chains:** cut into 10 clusters, digits gives one cluster of 1788 of 1797 images.
- **The biggest-gap cut** finds 3 on wine but 2 on iris and on digits.
- **Cophenetic correlation does not pick the best cut:** average linkage keeps distances best (0.8544 on iris), Ward clusters best.

Study guide: [`day120_hierarchical_clustering_complete_guide.pdf`](day120_hierarchical_clustering_complete_guide.pdf) · Syntax walkthrough: [`day120_syntax_line_by_line.pdf`](day120_syntax_line_by_line.pdf) · [Run output](day120_run_output.txt)

Exercise: [`day120_exercise.py`](day120_exercise.py) is the merge loop, the cut and the cophenetic heights, checked by hand on five points — the second part of Day 125's build ([run output](day120_exercise_run_output.txt)).

![Agglomerative clustering](hierarchical_diagram.png)
![The iris dendrogram, four linkages, and chaining on wine](hierarchical_results.png)

---
[← Back to main README](../README.md)
