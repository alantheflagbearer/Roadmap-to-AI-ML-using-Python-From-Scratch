# Week 12: The Transformer (Days 77–83)

Week 11 built attention up to multi-head self-attention. Week 12 builds the Transformer around it. It follows the two-track structure of Weeks 9–11: Days 77–82 each get the full-depth lesson (real experiments, paired study-guide PDFs) plus a short `dayNN_exercise.py` isolating one part, and Day 83 trains a tiny Transformer built from those exercise scripts. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 77 — Positional encoding, from scratch
[`day77_positional_encoding.py`](day77_positional_encoding.py) — the sinusoidal encoding (`PE[p, 2i] = sin(p / 10000^(2i/d))`, `PE[p, 2i+1] = cos(...)`), added to the word vectors so that self-attention can see order. Built, its key property checked, then tested on a toy task that needs position and on Week 9's real tagging task with Day 76's multi-head layer.

- **Moving along is a rotation:** one matrix R_k built from k alone maps PE[p] to PE[p + k] for every p (to 3.71e-15), so PE[p] · PE[p + k] depends only on k. Day 74's shuffle test: 8.88e-15 without the encoding, 3.18e+01 with it.
- **Toy task (write the previous token):** 0.253 with no encoding, 1.000 with sinusoidal or learned positions on the trained lengths. Beyond them both mostly fail: at 32 tokens, never-seen positions are right 0.278 (sinusoidal) and 0.130 (learned) of the time.
- **Real tagging, chosen on validation:** added to the whole word vector ('all') the encoding barely helps (0.8063 vs 0.8036) and at strength 3 training collapses (0.1400). Added only to the queries and keys at strength 3 ('qk'): 0.8487 over three seeds, above Day 62's RNN (0.8327) and Day 68's LSTM (0.8377), below the bidirectional LSTM (0.8577).
- **How:** one head puts 0.918 of its weight on the next word, another 0.569 on the previous one. Shuffling the test sentences now costs accuracy (0.8485 → 0.7376); without an encoding nothing changes.

Study guide: [`day77_positional_encoding_complete_guide.pdf`](day77_positional_encoding_complete_guide.pdf) · Syntax walkthrough: [`day77_syntax_line_by_line.pdf`](day77_syntax_line_by_line.pdf) · [Run output](day77_run_output.txt)

Exercise: [`day77_exercise.py`](day77_exercise.py) is the encoding table as one function, with its rotation and order checks — the first part of Day 83's Transformer ([run output](day77_exercise_run_output.txt)).

![Where the encoding enters](positional_encoding_diagram.png)
![The encoding, the toy task by position, and the tagger's heads](positional_encoding_results.png)

---
[← Back to main README](../README.md)
