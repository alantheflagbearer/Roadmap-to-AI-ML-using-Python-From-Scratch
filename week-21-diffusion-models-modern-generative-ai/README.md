# Week 21: Diffusion Models & Modern Generative AI (Days 140–146)

Week 20 trained GANs, scored them, and ended by destroying data on purpose: Day 139's forward diffusion process adds a little Gaussian noise at each of 1,000 steps until nothing of the data is left. Week 21 runs that process backwards to generate, then compares the three families of generative models built so far and looks at modern text-to-image systems and their risks. Everything is written from scratch in NumPy. Days 140–145 each get the full-depth lesson plus a short `dayNN_exercise.py` isolating one part, and Day 146 is the capstone. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for the week's plan.

## Day 140 — Reverse diffusion from scratch: a real denoising step
[`day140_reverse_diffusion.py`](day140_reverse_diffusion.py) — Day 139's noising process run backwards: start from pure noise and take 1,000 small denoising steps (Ho et al., 2020). On Day 135's ring of eight Gaussians the best possible denoiser, E[x_0 | x_t], can be written down exactly, so the reverse process and a learned denoiser are tested separately.

- **The step's formula:** the posterior q(x_(t-1) | x_t, x_0) matches a million simulated runs (t = 50: 0.03660 against 0.03634).
- **One step cannot generate:** from t = 1,000 the exact guess of x_0 is the ring's centre for every point (radius 0.0160); many small steps are needed.
- **The process works:** 1,000 steps with the exact denoiser recovered all 8 modes in every seed (good share 0.998, 0.995, 0.997), slightly too tight with the posterior's own variance (spread 0.0455 against the real 0.0509) and right with beta_t (0.0496 to 0.0501).
- **A learned denoiser:** a network with two hidden layers of 256, trained on the noise-prediction loss came within 0.0129 of the lowest possible error (0.3290) after 30,000 steps; its samples cover all 8 modes but are a little too wide (spread 0.0607). With no fresh noise at each step it fell into three modes (seed 0's counts 1059 366 0 0 0 0 0 571), where the exact denoiser kept all eight.

Study guide: [`day140_reverse_diffusion_complete_guide.pdf`](day140_reverse_diffusion_complete_guide.pdf) · Syntax walkthrough: [`day140_syntax_line_by_line.pdf`](day140_syntax_line_by_line.pdf) · [Run output](day140_run_output.txt)

Exercise: [`day140_exercise.py`](day140_exercise.py) is the posterior, the exact denoiser, the sampling loop and the noise-predicting network, checked by hand — the first part of Day 146's capstone ([run output](day140_exercise_run_output.txt)).

![Forward and reverse diffusion](reverse_diffusion_diagram.png)
![One denoising step, the whole reverse process, and a learned denoiser](reverse_diffusion_results.png)

## Day 141 — Training a tiny diffusion model: real toy image generation
[`day141_tiny_diffusion.py`](day141_tiny_diffusion.py) — a diffusion model trained on the 60,000 MNIST training digits and sampled with Day 140's 1,000-step reverse process, scored by Day 138's metrics. The network is Day 126's dense network (two hidden layers of 512) with Day 140's time embedding.

- **Two first drafts failed, and are kept:** with He initialisation the second hidden layer switched off (0.00 of its ReLUs on) and the loss sat at 1.000, the error of guessing zero; starting the output layer at zero fixed it. A network asked for the noise drew speckled noise, not digits (FID 325.88; 341.59 at twice the width); the same network asked for the clean digit gave FID 49.45 after 3,000 steps.
- **The loss hid a decline:** over 30,000 steps the test error fell from 0.1781 to 0.1623, but FID was best at 10,000 steps (36.856) and rose to 63.350 by the end.
- **Sampling choices:** with no fresh noise per step, 1782 of 2,000 samples came out as the digit 3; clipping the clean-digit guess to [-1, 1] made FID worse (95.212).
- **No copying:** the closest sample is 1.37 from a training digit, while a real test digit came as close as 0.95; the samples have 1399 different nearest training digits.

Study guide: [`day141_tiny_diffusion_complete_guide.pdf`](day141_tiny_diffusion_complete_guide.pdf) · Syntax walkthrough: [`day141_syntax_line_by_line.pdf`](day141_syntax_line_by_line.pdf) · [Run output](day141_run_output.txt)

Exercise: [`day141_exercise.py`](day141_exercise.py) is the straight-line guess, the network's output read as the clean digit or the noise, the training loop, the evaluation, the nearest-image check and the sample scorer, checked by hand — the second part of Day 146's capstone ([run output](day141_exercise_run_output.txt)).

![A tiny diffusion model: training and sampling](tiny_diffusion_diagram.png)
![Training, error by noise level, samples and the copying check](tiny_diffusion_results.png)

## Day 142 — GANs vs VAEs vs diffusion: real, honest quality/speed tradeoffs
[`day142_generative_tradeoffs.py`](day142_generative_tradeoffs.py) — Week 19's VAE, Week 20's GAN and Day 141's diffusion model, each trained on the same 60,000 MNIST digits, scored by Day 138's scorer and timed on the same machine. New parts: an averaged copy of the weights (EMA), DDIM sampling, and a direct measure of blur.

- **Quality:** FID 27.778 for diffusion (sampled from the averaged weights), 38.520 for the VAE, 52.781 for the GAN.
- **Cost:** training took 100s, 125s and 1244s; sampling 2,000 digits took 0.02s (VAE), 0.01s (GAN) and 97.06s (diffusion, 1,000 network calls each). DDIM in 20 steps took 2.42s, at FID 61.201.
- **The averaged copy fixed Day 141's decline:** FID 27.778 at 30,000 steps against 63.350 for the raw weights, though it was worse early (75.153 against 48.992 at 5,000).
- **FID missed blur:** the share of grey pixels was 0.0500 for real digits, 0.0890 for the VAE and 0.0313 for the GAN, yet the VAE beat the GAN on FID.
- **Fewer steps, worse samples:** with no fresh noise, FID went 38.086 (1,000 steps), 50.174 (50), 105.503 (5); fresh noise helped at 1,000 steps and hurt at 100 and below.
- The first run was slowed by the laptop going into standby (diffusion training timed at 8,174 s); it was rerun kept awake, and every score came out identical.

Study guide: [`day142_generative_tradeoffs_complete_guide.pdf`](day142_generative_tradeoffs_complete_guide.pdf) · Syntax walkthrough: [`day142_syntax_line_by_line.pdf`](day142_syntax_line_by_line.pdf) · [Run output](day142_run_output.txt)

Exercise: [`day142_exercise.py`](day142_exercise.py) is the averaged copy, DDIM, the guesser wrapper, the grey share and the weight count, checked by hand — the third part of Day 146's capstone ([run output](day142_exercise_run_output.txt)).

![Three generative models, and the cost of sampling](generative_tradeoffs_diagram.png)
![Samples, quality while training, and DDIM's speed against quality](generative_tradeoffs_results.png)

## Day 143 — Text-to-image models conceptually: CLIP-style joint embeddings, a real demo
[`day143_clip_joint_embeddings.py`](day143_clip_joint_embeddings.py) — an image encoder and a bag-of-words text encoder (Day 126's networks) trained with CLIP's symmetric contrastive loss, its gradient written out by hand, so that a digit and its caption land close together. MNIST has no captions, so they are written by program from each digit's label and two measured attributes: thickness and lean.

- **Zero-shot:** with no classifier trained, test digits named from words alone with accuracy 0.9805 after 45 s of training, against 0.9829 for Day 138's trained classifier.
- **The prompt barely matters here:** nine templates scored 0.9745 to 0.9805; reordered words and unknown words give exactly the same embedding (cosine 1.000000), and 'not a {}' scores 0.9805 too — a bag of words sees neither order nor negation.
- **Search by description:** 'a thick three' found threes with mean ink 0.2181 and 'a thin three' 0.0842, against 0.1433 for all test threes.
- **A combination never seen in training:** encoders trained without ever calling a seven thick still retrieved thick sevens (top-50 ink 0.1728 against 0.1149 for all sevens).
- A first draft of the retrieval check found 'right' inside 'upright' and reported 0.00 for 'a thin eight upright'; matching whole words fixed it.

Study guide: [`day143_clip_joint_embeddings_complete_guide.pdf`](day143_clip_joint_embeddings_complete_guide.pdf) · Syntax walkthrough: [`day143_syntax_line_by_line.pdf`](day143_syntax_line_by_line.pdf) · [Run output](day143_run_output.txt)

Exercise: [`day143_exercise.py`](day143_exercise.py) is the measurements, captions, bag of words, contrastive loss, training loop, embeddings and zero-shot naming, checked by hand — the fourth part of Day 146's capstone ([run output](day143_exercise_run_output.txt)).

![A CLIP-style pair of encoders](clip_joint_embeddings_diagram.png)
![Training, the shared space, retrieval and the held-out combination](clip_joint_embeddings_results.png)

## Day 144 — Prompt engineering for generative models: real experimentation
[`day144_prompt_engineering.py`](day144_prompt_engineering.py) — Day 143's text encoder now steers Day 141's diffusion model: the network gets a caption's 64-number embedding as an extra input and is trained with the caption dropped 15% of the time, so one network can draw with and without a prompt. Classifier-free guidance then pushes the captioned guess away from the uncaptioned one, and prompts were varied one change at a time, 200 drawings each.

- **The guidance scale:** with no guidance the prompt is ignored (right digit 0.107, chance); w = 1 gave 0.948, and w = 2 gave 1.000 with the best FID (25.06). Beyond that quality fell: FID 70.09 at w = 3 and 210.96 at w = 8, recall 0.527 down to 0.190 at w = 5. Guidance also thickened the digits (ink 0.124 to 0.189, against 0.132 for real digits).
- **Unknown words fail silently:** '7' and 'sept' drew no sevens (0.000), mostly eights; an unknown word becomes an empty caption (cosine 1.000000 with one), which the model never saw in training.
- **Attributes and an unseen combination:** 'a thick three' ink 0.2302 against 0.1032 for 'a thin three'; 'four leaning right' lean 0.4722 against -0.2718 for leaning left; 'a thick seven', never in training, ink 0.2111 with 198 of 200 still sevens.
- **Negative prompts:** 'three' with negative 'thick' drew ink 0.0810, thinner than 'a thin three'; the cost was the digit itself (right digit 0.975 with negative 'thin').
- **Contradictions pick a side:** 'three seven' gave three 112, seven 86, not a blend; 'thick thin seven' drew ink 0.1558, close to plain 'seven'.

Study guide: [`day144_prompt_engineering_complete_guide.pdf`](day144_prompt_engineering_complete_guide.pdf) · Syntax walkthrough: [`day144_syntax_line_by_line.pdf`](day144_syntax_line_by_line.pdf) · [Run output](day144_run_output.txt)

Exercise: [`day144_exercise.py`](day144_exercise.py) is the captioned guess, training with dropped captions, guided DDIM sampling with an optional negative prompt, the two measurements and digit naming, checked by hand — the fifth part of Day 146's capstone ([run output](day144_exercise_run_output.txt)).

![Classifier-free guidance](prompt_engineering_diagram.png)
![Training, the guidance scale, and drawings for each prompt](prompt_engineering_results.png)

## Day 145 — Ethical considerations in generative AI: bias, deepfakes, real case studies
[`day145_generative_ai_ethics.py`](day145_generative_ai_ethics.py) — three risks of generative models, each measured on digits with this week's models and set beside a documented real case: a model copying or exaggerating a skew in its training data, generated images passed off as real, and training images reproduced.

- **Bias needs a control:** ones were skewed so 75% leaned right. Against the data alone the model seemed not to copy the skew (0.414 leaning right against 0.750), but a control trained without the skew drew right-leaning ones only 0.110 of the time. Against the control the skew grew: left-leaning ones, cut to 0.131 of the data, fell from 0.217 to 0.024 in the model and to 0.003 with guidance. A first draft had no control and could not tell these apart.
- **Asking overrides the skew:** 'a one leaning left' drew ones leaning left 0.870 of the time at w = 1.
- **A detector is narrow:** trained on one diffusion model's digits it caught 0.924 of held-out fakes, but only 0.041 of a VAE's and 0.061 of a GAN's; black and white lowered detection to 0.219 and a little noise flagged every real digit (1.000). If 1 image in 100 were generated, only 0.163 of its flags would be right.
- **Memorisation:** trained on 10 digits the model copied them (near copies 0.806); on 1,000, near copies were 0.028, close to the 0.022 of real test digits.
- **Real cases:** the Bloomberg study of Stable Diffusion bias (2023), the Arup deepfake video call (HK$200 million, 2024), the New Hampshire AI robocall ($6 million FCC fine, 2024) and the extraction of training images from diffusion models (Carlini et al., 2023).

Study guide: [`day145_generative_ai_ethics_complete_guide.pdf`](day145_generative_ai_ethics_complete_guide.pdf) · Syntax walkthrough: [`day145_syntax_line_by_line.pdf`](day145_syntax_line_by_line.pdf) · [Run output](day145_run_output.txt)

Exercise: [`day145_exercise.py`](day145_exercise.py) is skewing a training set, a detector, its probability, a 3 x 3 blur and the share of flagged images that are really generated, checked by hand — the sixth part of Day 146's capstone ([run output](day145_exercise_run_output.txt)).

![Three risks of generative models](generative_ai_ethics_diagram.png)
![Bias with a control, detection, and memorisation](generative_ai_ethics_results.png)

---
[← Back to main README](../README.md)
