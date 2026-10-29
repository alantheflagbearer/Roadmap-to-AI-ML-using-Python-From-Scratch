# Roadmap to AI/ML using Python — From Scratch

A daily-practice log following a 52-week (365-day) roadmap from Python fundamentals through classical ML, deep learning, and into applied AI. Most lessons don't just call a library function — they build the underlying idea (gradient descent, backprop, convolution, dropout...) from scratch in NumPy first, then show the framework equivalent. Each day is a self-contained script; from Week 1 onward, most days add a paired study guide (a concept walkthrough and a line-by-line syntax breakdown, both PDF).

**Currently on Day 96** of 365. Full syllabus: [`full_syllabus_days_1_365.md`](full_syllabus_days_1_365.md).

## How this repo is organized

- **One folder per week** (`week-01-python-foundations/` … `week-14-mini-gpt-language-modeling/`) — each holds that week's scripts, study-guide PDFs, diagrams, and run outputs, plus its own `README.md` with a day-by-day index. Open a week's `README.md` first; it links to everything inside.
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
| 10 | 63–69 | LSTM & GRU | [→ folder](week-10-lstm-gru) |
| 11 | 70–76 | Attention Mechanism | [→ folder](week-11-attention-mechanism) |
| 12 | 77–83 | The Transformer | [→ folder](week-12-the-transformer) |
| 13 | 84–90 | Tokenization & Embeddings | [→ folder](week-13-tokenization-embeddings) |
| 14 | 91–96 (of 91–97) | Mini-GPT & Language Modeling — in progress | [→ folder](week-14-mini-gpt-language-modeling) |

Weeks 15 onward (practical NLP with pretrained models, and much further — see the full syllabus) haven't been built yet. Starting Week 9, weeks whose theme is one major model get built with a new two-track structure: Days 1–6 each add a short, lightweight exercise script alongside the usual full lesson, and Day 7 combines the six exercises into one final trained model.

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
| ![LSTM cell state growth](week-10-lstm-gru/lstm_cell_state_growth.png) With f near 1 the cell state adds up and saturates — Day 64 | ![LSTM training by forget bias](week-10-lstm-gru/lstm_training_by_forget_bias.png) First LSTM training: the forget bias decides — Day 65 |
| ![GRU cell](week-10-lstm-gru/gru_cell_diagram.png) The GRU: one state, two gates — Day 66 | ![Where the cells differ](week-10-lstm-gru/cells_where_and_how_far.png) RNN vs LSTM vs GRU on real tagging: every cell looks only a few words back — Day 67 |
| ![Deep bidirectional tagger](week-10-lstm-gru/deep_bidirectional_results.png) Reading both ways: +2 points; a second layer: nothing — Day 68 | ![Seq2seq results](week-10-lstm-gru/seq2seq_results.png) Week 10 build: an encoder-decoder from the exercise parts copies and reverses digits — Day 69 |
| ![Bottleneck results](week-11-attention-mechanism/bottleneck_results.png) One fixed-size state: 4 digits held perfectly, 20 not — Day 70 | ![Attention curves](week-11-attention-mechanism/attention_training_curves.png) Additive attention: Day 70's failing test, now 1.000 at every length — Day 71 |
| ![Attention heatmaps](week-11-attention-mechanism/attention_heatmaps.png) Attention heatmaps: the diagonal, and where it breaks — Day 72 | ![Score functions compared](week-11-attention-mechanism/score_functions_compared.png) Dot, general and additive scores, like for like — Day 73 |
| ![Self-attention tagger](week-11-attention-mechanism/self_attention_tagger.png) Self-attention: order-blind, proved on real data — Day 74 | ![Scaling results](week-11-attention-mechanism/scaling_results.png) Scaled dot-product attention: the benefit is the starting point — Day 75 |
| ![Multi-head results](week-11-attention-mechanism/multi_head_results.png) Week 11 build: one head cannot fetch two things; two heads can — Day 76 | ![Positional encoding results](week-12-the-transformer/positional_encoding_results.png) Positional encoding: strong in the scores, kept out of the word — Day 77 |
| ![Encoder block results](week-12-the-transformer/encoder_block_results.png) Encoder block: the FFN matters; depth did not help under plain SGD — Day 78 | ![LayerNorm vs BatchNorm](week-12-the-transformer/layernorm_vs_batchnorm.png) LayerNorm vs BatchNorm: same accuracy, different independence — Day 79 |
| ![Causal mask results](week-12-the-transformer/causal_mask_results.png) Decoder block: without the causal mask, a model that reads the answer — Day 80 | ![Transformer forward results](week-12-the-transformer/transformer_forward_results.png) The whole Transformer, untrained: every mask holds end to end — Day 81 |
| ![Transformer backward results](week-12-the-transformer/transformer_backward_results.png) The backward pass: every gradient checked, and the check catches a planted bug — Day 82 | ![Tiny Transformer results](week-12-the-transformer/tiny_transformer_results.png) Week 12 build: the Transformer learns COPY fast; REVERSE, not as well as the LSTM — Day 83 |
| ![BPE results](week-13-tokenization-embeddings/bpe_results.png) Byte-pair encoding: lossless, and no more unknown words — Day 84 | ![Embeddings results](week-13-tokenization-embeddings/embeddings_results.png) Word embeddings: a learned table beats one-hot inputs — Day 85 |
| ![Word2Vec results](week-13-tokenization-embeddings/word2vec_results.png) Word2Vec from raw text: word groups right, analogies mostly not — Day 86 | ![Visualization results](week-13-tokenization-embeddings/visualization_results.png) PCA and t-SNE: t-SNE keeps the word groups, not the sizes or gaps — Day 87 |
| ![Subword results](week-13-tokenization-embeddings/subword_results.png) fastText: vectors for unseen words, but shared endings dominate them — Day 88 | ![GloVe results](week-13-tokenization-embeddings/glove_results.png) Pretrained vectors: a clear help with 200 labelled sentences, little with 2,001 — Day 89 |
| ![Tiny GPT results](week-13-tokenization-embeddings/tiny_gpt_results.png) Week 13 build: a tiny GPT writes Shakespeare-shaped text, and ties a 4-character counter — Day 90 | ![Causal LM results](week-14-mini-gpt-language-modeling/causal_lm_results.png) Next-token prediction: a Kneser-Ney counter beats Day 90's GPT — Day 91 |
| ![Training results](week-14-mini-gpt-language-modeling/training_results.png) Training the mini-GPT: level with Kneser-Ney, honestly measured — Day 92 | ![Sampling results](week-14-mini-gpt-language-modeling/sampling_results.png) Sampling: greedy loops, high temperature invents words — Day 93 |
| ![Quality results](week-14-mini-gpt-language-modeling/quality_results.png) Generated text, measured: new, plausible, and still detectable — Day 94 | ![Scaling results](week-14-mini-gpt-language-modeling/scaling_results.png) Scaling: a clean power law in model size, not in data under a fixed budget — Day 95 |
| ![Fine-tuning results](week-14-mini-gpt-language-modeling/fine_tuning_results.png) Fine-tuning on an unseen play: 2.486 to 2.104 bits, replay forgets least — Day 96 | |

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
