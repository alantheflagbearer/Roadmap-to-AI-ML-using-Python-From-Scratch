# Roadmap to AI/ML using Python — From Scratch

A daily-practice log following a 52-week (365-day) roadmap from Python fundamentals through classical ML, deep learning, and into applied AI. Most lessons don't just call a library function — they build the underlying idea (gradient descent, backprop, convolution, dropout...) from scratch in NumPy first, then show the framework equivalent. Each day is a self-contained script; from Week 1 onward, most days add a paired study guide (a concept walkthrough and a line-by-line syntax breakdown, both PDF).

**Currently on Day 63** of 365. Full syllabus: [`full_syllabus_days_1_365.md`](full_syllabus_days_1_365.md).

## How this repo is organized

- **One folder per week** (`week-01-python-foundations/` … `week-10-lstm-gru/`) — each holds that week's scripts, study-guide PDFs, diagrams, and run outputs, plus its own `README.md` with a day-by-day index. Open a week's `README.md` first; it links to everything inside.
- **`practice-drills/`** — short standalone Python-fundamentals exercises (`pfdsNN.py`), numbered independently of the daily roadmap.
- Root-level `titanic.csv`, `titanic.db`, and `students.csv` are the shared datasets several early weeks use; each is also copied into every week folder whose scripts need it, so any single week folder runs standalone without reaching outside itself.

## Week-by-week index

| Week | Days | Topic | |
|---|---|---|---|
| 1 | 2–5 | Python Foundations | [→ folder](week-01-python-foundations) |
| 2 | 6–14 | NumPy, Pandas, Statistics & SQL | [→ folder](week-02-numpy-pandas-statistics-sql) |
| 3 | 15–20 | Intro to Machine Learning | [→ folder](week-03-intro-to-machine-learning) |
| 4 | 21–27 | Feature Engineering & Boosting | [→ folder](week-04-feature-engineering-boosting) |
| 5 | 28–35 | Interpretability, Tuning & Deployment | [→ folder](week-05-interpretability-tuning-deployment) |
| 6 | 36–41 | Neural Networks from Scratch | [→ folder](week-06-neural-networks-from-scratch) |
| 7 | 42–48 | CNNs I & II | [→ folder](week-07-cnns-i-and-ii) |
| 8 | 49–55 | Classic CNN Architectures + Regularization | [→ folder](week-08-classic-cnn-architectures) |
| 9 | 56–62 | Sequence Data & RNNs I | [→ folder](week-09-sequence-data-rnns-i) |
| 10 | 63 (of 63–69) | LSTM & GRU — in progress | [→ folder](week-10-lstm-gru) |

Weeks 11 onward (attention, Transformers, tokenization, a mini-GPT, and much further — see the full syllabus) haven't been built yet. Starting Week 9, weeks whose theme is one major model get built with a new two-track structure: Days 1–6 each add a short, lightweight exercise script alongside the usual full lesson, and Day 7 combines the six exercises into one final trained model.

Starting with Week 6, the same small neural-network library — written entirely in NumPy, no framework — is extended week over week: the `net.forward()` / `net.backward()` core built on Day 36 is still what every later script, all the way through Day 62, trains against; Week 10 starts a new cell, the LSTM.

## Recent visuals

| | |
|---|---|
| ![LeNet-5 vs. baseline](week-07-cnns-i-and-ii/lenet_vs_baseline_loss.png) LeNet-5 vs. this series' baseline — Day 47 | ![Transfer strategies](week-07-cnns-i-and-ii/transfer_strategies_diagram.png) Five transfer-learning strategies compared — Day 48 |
| ![Dropout regularization comparison](week-08-classic-cnn-architectures/dropout_regularization_comparison.png) Dropout ON vs. OFF, fully converged — Day 50 | ![L1/L2 regularization sweep](week-08-classic-cnn-architectures/l1_l2_regularization_sweep.png) L1/L2 weight regularization sweep — Day 51 |
| ![VGGStyle vs. LargeKernel](week-08-classic-cnn-architectures/vgg_vs_large_kernel.png) VGG-style stacked convs vs. one large kernel — Day 52 | ![Plain vs. residual](week-08-classic-cnn-architectures/plain_vs_resnet.png) Plain vs. residual connections, 7 conv layers deep — Day 53 |
| ![Plain vs. BN vs. BN+residual](week-08-classic-cnn-architectures/batchnorm_vs_plain_vs_resnet.png) Batch norm alone beats BN+residual, same failing lr — Day 54 | ![Data loader throughput](week-08-classic-cnn-architectures/data_loader_throughput.png) In-memory vs. lazy-disk vs. prefetch loading — Day 55 |
| ![Sequential data failure modes](week-09-sequence-data-rnns-i/sequential_failure_modes.png) No position invariance (FC) vs. a capped receptive field (Conv1D) — Day 56 | ![RNN long-range sensitivity](week-09-sequence-data-rnns-i/rnn_long_range_sensitivity.png) Does x_0 survive to h_T at long range? — Day 57 |
| ![BPTT training short vs long](week-09-sequence-data-rnns-i/bptt_training_short_vs_long.png) T=40 trains anyway: noise suppressed, spikes latched — Day 58 | ![Exploding gradients during training](week-09-sequence-data-rnns-i/exploding_during_training.png) Gradient spikes as Whh's spectral radius crosses ~2 — Day 59 |
| ![Clipping before and after](week-09-sequence-data-rnns-i/clipping_before_after.png) Unclipped learns then collapses; clipping holds — Day 60 | ![Sentiment RNN readout and word order](week-09-sequence-data-rnns-i/sentiment_readout_and_order.png) Real reviews: the RNN doesn't beat bag-of-words, or use word order — Day 61 |
| ![Week 9 parts diagram](week-09-sequence-data-rnns-i/week9_parts_diagram.png) Six exercise parts, one many-to-many tagger — Day 62 | ![Tagger: memory, gradient probe, clipping](week-09-sequence-data-rnns-i/tagger_memory_and_parts.png) Where memory helps; clipping decisive at hidden 64 — Day 62 |
| ![LSTM cell](week-10-lstm-gru/lstm_cell_diagram.png) One step of an LSTM cell — Day 63 | ![LSTM memory survival](week-10-lstm-gru/lstm_memory_survival.png) Memory survival depends on the forget gate — Day 63 |

## Featured project: Titanic EDA (Day 13)

Answers one question: what determined who survived the Titanic disaster, and does it hold up across pandas, statistical tests, and raw SQL?

**Dataset:** 891 passengers, 15 columns after cleaning and feature engineering (`FamilySize`, `IsAlone`, `AgeGroup`, `Title` added; age imputed with median, cabin dropped, embarked imputed with mode).

**Key findings:**
- Survivors paid $48.40 avg fare vs. $22.12 for non-survivors (t = 6.84, p = 2.70e-11) — statistically significant.
- Female survival rate 74.2% vs. male 18.9% (χ² = 260.7, p = 1.20e-58) — statistically significant.
- Survival rate by class: 1st 63.0%, 2nd 47.3%, 3rd 24.2% (χ² = 102.9, p = 4.55e-23) — statistically significant.
- Strongest correlate of `Survived` is `Pclass` (r = -0.34) — correlation, not causation: class is a proxy for cabin location and lifeboat access, not a direct cause.

![Titanic EDA dashboard](week-02-numpy-pandas-statistics-sql/day13_dashboard.png)

Six panels: overall survival, survival by sex, survival by class, age distribution, log-scaled fare distribution, and the full correlation matrix.

**SQL cross-check:** the same survival patterns are reproduced with raw SQL (`GROUP BY`, `CASE WHEN` fare-tier bucketing, and a `RANK()` window function for top fares per class) — see [`day13_eda_portfolio.py`](week-02-numpy-pandas-statistics-sql/day13_eda_portfolio.py).

## What's next

Continuing the roadmap day by day toward Week 53 (deep learning frameworks, deployment, and applied AI projects) — see the full syllabus for the complete week-by-week breakdown.
