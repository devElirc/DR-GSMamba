# Abstract v1 (Phase 1 lock)

**Working title.**
*Class-Frequency-Aware Group-DRO for Reliable Label-Scarce Hyperspectral Image Classification.*

**Target venue.**
Pattern Recognition (Q1 / CCF-B) as primary; IEEE TNNLS as backup.

---

## Abstract (single paragraph, ~250 words)

Hyperspectral image (HSI) classification under label scarcity remains challenging because labeled pixels are rare, class distributions are highly imbalanced, and many spectral signatures sit on class boundaries. Deep models built on convolution, Transformers, graph reasoning, or state-space sequence modeling have improved average accuracy, but their *reliability* &mdash; stability across random splits, performance on rare classes, and calibration on boundary or mixed pixels &mdash; is usually only observed at evaluation time, not designed into the training objective. This paper presents a reliability-oriented framework in which two principled components shape the optimization. First, we introduce a Class-Frequency-Aware Group-DRO objective (CFA-GDRO) that solves a constrained min-max problem over the class-wise loss, with per-class adversarial caps that scale as a power of the inverse scene-level class frequency. The objective has a closed-form solution by sorted water-filling in $O(K \log K)$ per mini-batch, and a single proposition with two corollaries shows that CFA-GDRO acts as an explicit per-class upper bound on every class whose cap reaches one, recovering rare-class and worst-class dominance as special cases under verifiable conditions on the cap parameters. Second, we attach an Evidential Prototype Head (EPH) that maps prototype similarities to Dirichlet evidence, producing calibrated probabilities and a closed-form decomposition of aleatoric and epistemic (vacuity) uncertainty for every pixel. The two components share a per-sample loss, so the model is calibrated and reweighted on the same quantity. The architecture combines a spectral selective-scan encoder on raw bands, a compact spatial CNN stem, and an in-batch cross-pixel graph reasoning module &mdash; but these are design choices, not contributions. Experiments on Indian Pines, Pavia University, Salinas, and Houston 2013 with ten random seeds and four label-scarce regimes (3, 5, 10, 20 samples per class) show that the proposed framework improves rare-class accuracy, worst-class accuracy, and split stability while maintaining competitive overall accuracy, and produces calibrated uncertainty across all four datasets.

---

## Contributions (five-item list)

1. **Class-Frequency-Aware Group-DRO (CFA-GDRO) objective.**
   We formulate label-scarce HSI training as a constrained min-max problem over class losses, with per-class adversarial caps that scale as the inverse *scene-level* class frequency raised to a tunable exponent. The closed-form inner solution is a sorted water-filling procedure with $O(K \log K)$ per-batch cost. The objective recovers Sagawa-style group-DRO and class-level CVaR as special cases and reduces to uniform-cap class-CVaR when the frequency exponent is zero.

2. **Per-class upper-bound guarantees on a sliding scale.**
   We prove a per-class dominance proposition that yields two corollaries: a mild rare-class bound (active whenever $\alpha\,W(\gamma) \le \pi_{\min}^{-\gamma}$) and a strict worst-class bound (active whenever $\alpha\,W(\gamma) \le \pi_{\max}^{-\gamma}$). Together they describe a sliding scale of reliability guarantees as $\alpha$ decreases. We also situate the constraint set inside a $\chi^2$-divergence ball around the scene-level class distribution. The bounds make the title word *Reliable* a falsifiable property of the training objective, not a description of the experimental protocol.

3. **Evidential Prototype Head (EPH) for calibrated HSI predictions.**
   We design a Dirichlet-evidential head on top of cosine prototype distances that replaces both the softmax classifier and the heuristic uncertainty branch used in many recent HSI methods. EPH yields a closed-form decomposition of aleatoric and epistemic uncertainty per pixel, has lower expected calibration error than a softmax head of the same backbone, and supplies a per-sample loss that CFA-GDRO operates on, unifying calibration and robustness on a single quantity.

4. **A reproducible reliability evaluation protocol for label-scarce HSI.**
   We design and release a four-dataset, four-label-regime, ten-seed protocol that reports overall accuracy, average accuracy, Cohen's Kappa, macro-F1, per-class accuracy, worst-class accuracy, rare-class accuracy, standard deviation across seeds, coefficient of variation, worst-split accuracy, expected calibration error, mean vacuity, parameter count, FLOPs, latency, and paired t-tests against every baseline and every ablation.

5. **Empirical reliability on four public HSI benchmarks.**
   On Indian Pines, Pavia University, Salinas, and Houston 2013 (or WHU-Hi-LongKou as a fall-back), CFA-GDRO improves worst-class accuracy and split stability over cross-entropy, focal, plain CVaR, plain group-DRO, and recent HSI baselines (HybridSN, SpectralFormer, SSFTT, Nonlocal-GCN, MambaHSI). EPH lowers expected calibration error and exposes uncertain pixels on classification maps. Statistical significance is reported with paired t-tests over ten seeds.

---

## What is intentionally *not* claimed in the abstract or contributions

- We do **not** claim that the spectral selective-scan encoder is the source of accuracy gains. It is a design choice ablated against Transformer and 1D-CNN spectral encoders under the same loss.
- We do **not** claim novelty for prototype classifiers, evidential learning, or group-DRO individually. The contribution is the *frequency-aware cap structure*, the *closed-form solver*, the *worst-class upper bound*, and the *integration with an evidential prototype head* on a per-sample loss.
- We do **not** claim cross-scene or cross-domain capability. The scope is single-scene, label-scarce, multi-seed reliability.
- We do **not** claim computational efficiency. Efficiency numbers (parameters, FLOPs, latency) are reported for completeness but are not part of the contribution.

These omissions are deliberate: they shrink the reviewer's attack surface to a small number of falsifiable claims, all of which are supported by ablations or proofs.

---

## Title alternatives (kept for review, *not* the working title)

| # | Variant | Pros | Cons |
| --- | --- | --- | --- |
| A | Class-Frequency-Aware Group Distributionally Robust Learning for Label-Scarce Hyperspectral Image Classification | Fully spelled out | 13 words, two adjective phrases stacked |
| B | **Class-Frequency-Aware Group-DRO for Reliable Label-Scarce Hyperspectral Image Classification** | **Disambiguated, "Reliable" matches story, compact** | **WORKING TITLE** |
| C | Group-DRO for Reliable Label-Scarce Hyperspectral Image Classification | Shortest | Drops the frequency-awareness differentiator from Sagawa-style group-DRO |
