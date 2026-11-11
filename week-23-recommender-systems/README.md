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

---
[← Back to main README](../README.md)
