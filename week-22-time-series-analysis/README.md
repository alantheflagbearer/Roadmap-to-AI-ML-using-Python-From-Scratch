# Week 22: Time Series Analysis (Days 147–153)

Weeks 18–21 learned from images and text. Week 22 turns to time series: measurements in time order, where the order matters and the future has to be forecast from the past. It uses two real monthly series that ship with statsmodels: carbon dioxide at Mauna Loa (1958–2001) and euro-area orders for electrical equipment (1995–2016, including the 2008–09 crash). Every method is written from scratch in NumPy, with statsmodels used for the data and as a reference to check against. Days 147–152 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 153 builds a Prophet-style forecaster from those parts. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 147 — Time-series decomposition: trend, seasonality and residual on real data
[`day147_time_series_decomposition.py`](day147_time_series_decomposition.py) — a monthly series split into a slow trend (a centred 2 × 12 moving average), a yearly season (the average of each month of the year) and what is left, written from scratch in NumPy on two real series: carbon dioxide at Mauna Loa (526 months, 1958–2001) and euro-area orders for electrical equipment (257 months, 1995–2016).

- **Checked:** the moving average returns a repeating 12-month pattern as its mean (1.1e-16) and a straight line unchanged (7.1e-15); the decomposition matches statsmodels' `seasonal_decompose` exactly (0.0e+00).
- **What it shows:** CO2's trend rose 0.86 ppm a year in the 1960s and 1.58 in the 1980s, and its season peaks in May and is lowest in October (swing 6.07 ppm). The orders drop in August and peak in September.
- **Additive or multiplicative:** in both series the seasonal swing grows with the level (correlation 0.616 and 0.703), and the multiplicative fit is closer (orders 2.4701 against 2.6607).
- **What it gets wrong:** the trend stops 6 months before the data; one fixed season cannot follow CO2's, which grew from 5.75 to 6.44 ppm between the 1960s and 1990s; and the residual is not noise (correlation one month apart 0.403, and 0.393 for statsmodels' STL). The orders' season survived the 2008–09 crash: leaving 2008–2010 out changes no monthly factor by more than 0.007.
- **Using it:** raw and seasonally adjusted month-to-month changes point opposite ways in 0.414 of months. A first forecast (a straight trend line through the last 24 trend values, plus the season) overshot by 2.425 ppm, worse than repeating last year (1.882), because those two years included a trend jump of +2.88; a 120-month line gave 0.591, a window that must not be chosen on the test months.

Study guide: [`day147_time_series_decomposition_complete_guide.pdf`](day147_time_series_decomposition_complete_guide.pdf) · Syntax walkthrough: [`day147_syntax_line_by_line.pdf`](day147_syntax_line_by_line.pdf) · [Run output](day147_run_output.txt)

Exercise: [`day147_exercise.py`](day147_exercise.py) is the two data loaders, the centred moving average, the classical decomposition, the strength measures and a January-to-December reordering, checked by hand — the first part of Day 153's forecaster ([run output](day147_exercise_run_output.txt)).

![Classical decomposition, step by step](time_series_decomposition_diagram.png)
![CO2 and electrical-equipment orders decomposed](time_series_decomposition_results.png)

---
[← Back to main README](../README.md)
