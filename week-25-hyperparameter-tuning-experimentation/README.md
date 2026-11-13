# Week 25: Hyperparameter Tuning & Experimentation (Days 168–174)

Week 24 built the tools for judging one model. Week 25 is about choosing its settings: grid against random search, Bayesian optimisation, learning-rate finders, early stopping, cross-validation for time series and grouped data, keeping a record of every experiment, and making a tuning study give identical results when it is run again. It reuses real public datasets from earlier weeks: UCI Adult (census incomes; B. Becker and R. Kohavi, 1996) from `~/week24_data`, statsmodels' CO2 and electrical-equipment series from Week 22, and MovieLens 100K from `~/movielens_data`. Every method is written from scratch in NumPy. Days 168–173 each get the full-depth lesson plus a short `dayNN_exercise.py` holding that day's tool, and Day 174 uses all six in one tracked, reproducible tuning study. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 168 — Grid search against random search: a real efficiency comparison
[`day168_grid_vs_random_search.py`](day168_grid_vs_random_search.py) — Week 25 starts with the two simplest ways to tune: every point of a grid, or random points with the same budget. Day 109's gradient boosting (100 trees) is tuned on UCI Adult over learning rate (0.01–1, log scale), depth (1–8) and min leaf (1–300, log scale). Of Week 24's 24,130 fitting rows, 6000 train each configuration and 6000 others score it; 6032 validation rows are kept fresh.

- **Known-truth benchmark:** when one of three settings matters, random search finds the better point in 0.676 / 0.715 / 0.757 of 1000 problems at 8 / 27 / 64 trials. When all three matter equally, it is the other way round (0.285 / 0.335 / 0.361).
- **Real data, the textbook claim does not hold:** the grids found 0.2973 / 0.2934 / 0.2924, and random search matched or beat them in only 0.491 / 0.344 / 0.325 of 1000 searches. **Control:** five shifted grids of 27 averaged 0.2940 (the centred grid scored 0.2934), and random search matched or beat a shifted grid in 0.410 of comparisons.
- **Why:** learning rate explains 0.293 of the spread in log loss on its own and depth 0.046, but together they explain 0.535. The settings interact, so this is not the one-setting-matters case.
- **Cost:** a depth-8 fit took 6.76 s and a depth-1 fit 0.66 s. **Timing first draft:** a first run gave identical scores, but its times were about 2.4× smaller and it ranked grid and random in the opposite order by time, so only comparisons within one run are reported.
- **Winner's curse:** tuning and fresh scores correlate 0.996. The random winner's extra optimism grows from -0.0002 to +0.0018 as the budget rises, and the pool's winner ranks 9 of 200 on fresh rows.

Study guide: [`day168_grid_vs_random_search_complete_guide.pdf`](day168_grid_vs_random_search_complete_guide.pdf) · Syntax walkthrough: [`day168_syntax_line_by_line.pdf`](day168_syntax_line_by_line.pdf) · [Run output](day168_run_output.txt)

Exercise: [`day168_exercise.py`](day168_exercise.py) holds the search space, the three scales, grid and random points in the unit cube, decoding, the running best and eta-squared importance. Its hand checks include log scale u = 0.5 → 0.1, whole numbers on a log scale u = 0.5 → 17, and an importance of [1.0, 0.0] when one setting decides everything. [Exercise output](day168_exercise_run_output.txt)

![Nine trials as a grid and at random](grid_random_diagram.png)
![The benchmark, best-so-far curves on Adult, the random pool, importance and fresh scores](grid_random_results.png)

---
[← Back to main README](../README.md)
