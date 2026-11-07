# Week 19: Unsupervised Learning II — Autoencoders (Days 126–132)

Week 18 found structure in data without labels: clusters and low-dimensional maps. Week 19 learns it with neural networks trained to rebuild their own input: autoencoders, denoising autoencoders, anomaly detection by reconstruction error, and variational autoencoders that can generate new data. Everything is written from scratch in NumPy, on MNIST (70,000 real handwritten digits, fetched once from openml.org and cached outside the repository). Days 126–131 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 132 brings the parts together. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 126 — Autoencoders from scratch: a real reconstruction demonstration
[`day126_autoencoders.py`](day126_autoencoders.py) — dense layers, ReLU, backpropagation through the whole stack, cross-entropy and squared-error losses, Adam and minibatch training, all in NumPy. Data: MNIST (70,000 handwritten digits, fetched once from openml.org and cached outside the repository); labels used only to judge. Day 123's PCA and Day 119's k-means are imported.

- **Checked:** backpropagation matches finite differences to 1.5e-09 and 2.2e-09 over all 77 parameters.
- **A linear autoencoder rediscovers PCA:** 0.01707 per pixel against PCA's 0.01689, its subspace within cosines 0.9878 to 0.9999 of PCA's.
- **With ReLU it beats PCA at every code size:** 0.00731 against 0.01689 with 32 numbers.
- **Little data, memorised:** on 1,000 images, training error 0.00161 against test 0.01827.
- **Not automatically better features, and a code space with gaps:** kNN on the codes 0.9155 against PCA's 0.9170; decoded random codes sit 0.2193 from real images, decoded real codes 0.1504.

Study guide: [`day126_autoencoders_complete_guide.pdf`](day126_autoencoders_complete_guide.pdf) · Syntax walkthrough: [`day126_syntax_line_by_line.pdf`](day126_syntax_line_by_line.pdf) · [Run output](day126_run_output.txt)

Exercise: [`day126_exercise.py`](day126_exercise.py) is the network's machinery, checked by hand: gradients, a line rediscovered, a falling loss — the first part of Day 132's build ([run output](day126_exercise_run_output.txt)).

![An autoencoder](autoencoder_diagram.png)
![Reconstructions, error against code size, learning curves and decoded codes](autoencoder_results.png)

## Day 127 — Denoising autoencoders: a real noise-removal demonstration
[`day127_denoising_autoencoders.py`](day127_denoising_autoencoders.py) — Gaussian noise, masking and hidden squares, PSNR, a 3 × 3 median filter and PCA as baselines, and Day 126's network trained to turn damaged MNIST images back into clean ones (damage passed through its noise hook).

- **On the damage it trained for, the denoiser wins:** Gaussian σ 0.3: 19.25 dB, against 17.55 for the median filter, 16.42 for PCA and 13.32 untouched; the plain autoencoder makes it worse (12.93).
- **Light noise is better left alone:** at σ 0.1 the untouched image (22.65 dB) beats every denoiser.
- **The wrong damage can hurt:** the masking denoiser on Gaussian noise scores 9.45 dB, below doing nothing.
- **Robust codes:** kNN on noisy test images' codes: denoiser 0.9090, plain autoencoder 0.2815.
- **Hidden squares:** filled only modestly (error 0.4354 against 0.5492 blank).

Study guide: [`day127_denoising_autoencoders_complete_guide.pdf`](day127_denoising_autoencoders_complete_guide.pdf) · Syntax walkthrough: [`day127_syntax_line_by_line.pdf`](day127_syntax_line_by_line.pdf) · [Run output](day127_run_output.txt)

Exercise: [`day127_exercise.py`](day127_exercise.py) is the three kinds of damage, PSNR and the median filter, checked by hand — the second part of Day 132's build ([run output](day127_exercise_run_output.txt)).

![A denoising autoencoder](denoising_diagram.png)
![Repairs, training against test noise, masking and hidden squares](denoising_results.png)

---
[← Back to main README](../README.md)
