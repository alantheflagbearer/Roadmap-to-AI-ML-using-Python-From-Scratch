# Week 23: Recommender Systems (Days 154–160)

Week 22 forecast a series from its own past. Week 23 predicts something different: which films a person will like, from the ratings that they and everyone else have already given. It uses one real dataset throughout, MovieLens 100K (100,000 ratings of 1,682 films by 943 users, collected by GroupLens at the University of Minnesota in 1997–98; F. M. Harper and J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015). The data is not stored in this repository: the scripts download it on first run to `~/movielens_data`. Every method is written from scratch in NumPy. Days 154–159 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 160 builds a neural recommender from those parts. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 154 — Collaborative filtering: user-based and item-based on a real ratings matrix
[`day154_collaborative_filtering.py`](day154_collaborative_filtering.py) — predicting how a user will rate a film from ratings alone: a bias model (mean + user bias + film bias), user-based neighbours (people who agreed before) and item-based neighbours (films rated alike), with centred, shrunk similarities, all written from scratch in NumPy on MovieLens 100K (100,000 ratings by 943 users of 1,682 films; 6.3% of the table filled).

- **Checked:** the vectorised similarity matches a cosine worked by hand (0.4186) and numpy's Pearson correlation (1.1e-16). The exercise works a user-based and an item-based prediction on paper.
- **Small overlaps:** pairs of films sharing 1–2 raters have |similarity| above 0.9 in 0.786 of cases, so without shrinkage Star Wars's "nearest" films were obscure films sharing one rater. With shrinkage, Return of the Jedi (1983) and The Empire Strikes Back came first.
- **Test RMSE** (settings chosen on a validation split): bias model 0.9401, user-based 0.9353, item-based 0.9153. A control that shuffled the similarities among film pairs scored 0.9748, worse than no neighbours, so the gain comes from choosing the right neighbours.
- **Where neighbours help:** heavy users (item-based 0.8990 against 0.9275). For films with fewer than 20 ratings item-based lost to the bias model (1.0557 against 1.0416).
- **Top-10 lists:** the best rating predictor ranked badly. Its lists hit liked held-out films in 0.0417 of slots, below the bias model (0.0439) and far below recommending the most-rated films (0.1381). The weighted average gives a film with one rating (total similarity 0.001) a full correction; damping it, chosen on validation, raised precision to 0.0562.

Study guide: [`day154_collaborative_filtering_complete_guide.pdf`](day154_collaborative_filtering_complete_guide.pdf) · Syntax walkthrough: [`day154_syntax_line_by_line.pdf`](day154_syntax_line_by_line.pdf) · [Run output](day154_run_output.txt)

Exercise: [`day154_exercise.py`](day154_exercise.py) is the data loaders, the split, the ratings table, the bias model, the similarity with shrinkage and the two neighbour predictors, checked by hand — the first part of Day 160's recommender ([run output](day154_exercise_run_output.txt)).

![User-based and item-based collaborative filtering](collaborative_filtering_diagram.png)
![Long tail, small-overlap similarities, choosing k, test errors and top-10 precision](collaborative_filtering_results.png)

## Day 155 — Matrix factorization from scratch: SVD and learned factors
[`day155_matrix_factorization.py`](day155_matrix_factorization.py) — describing every user and film by a short list of hidden numbers (factors) whose products rebuild the known ratings. Two routes on Day 154's MovieLens split, written from scratch in NumPy: the SVD of a filled-in table, and factors fitted to the known ratings only, by alternating least squares (ALS) and by Funk-style SGD.

- **Checked:** the SVD by subspace iteration matches numpy's to 2.3e-07 and reaches the best possible rank-20 error (219.9119). The objective's gradient matches finite differences (7.0e-09), and the ALS objective fell after every sweep.
- **Low-rank, modestly:** the top 20 directions hold 0.189 of the residual table, against 0.154 when the same values are shuffled among the known cells.
- **What goes in the empty cells:** zeros scored 2.4282 on validation. With the bias model there, rank 20 scored 0.9255 on test. Filling and refitting with no penalty overfitted: at rank 50, training 0.2279 against validation 1.0896.
- **Learned factors** (100 factors, penalty chosen on validation): ALS 0.9115 and early-stopped SGD 0.9105 on test, just ahead of Day 154's item-based 0.9153. SGD's objective was higher (63418.4 against 62689.1), so early stopping acted as extra regularisation. Giving each user another user's factors scored 0.9711.
- **Top-10 lists did not improve:** precision@10 was 0.0405, against 0.1381 for recommending the most-rated films.

Study guide: [`day155_matrix_factorization_complete_guide.pdf`](day155_matrix_factorization_complete_guide.pdf) · Syntax walkthrough: [`day155_syntax_line_by_line.pdf`](day155_syntax_line_by_line.pdf) · [Run output](day155_run_output.txt)

Exercise: [`day155_exercise.py`](day155_exercise.py) is the SVD, the rank-k approximation, fill-factorize-repeat, the prediction, the objective and its gradient, ALS and SGD, checked by hand — the second part of Day 160's recommender ([run output](day155_exercise_run_output.txt)).

![Two routes to a low-rank model of the ratings table](matrix_factorization_diagram.png)
![Spectrum against a control, SVD routes, the ALS grid, training curves and test errors](matrix_factorization_results.png)

---
[← Back to main README](../README.md)
