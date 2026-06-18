# CFA-GDRO Literature Tracking

This file records the ongoing paper-monitoring workflow for keeping CFA-GDRO current with recent hyperspectral image classification research.

## Monitoring Sources

- Minchao Ye's HSI-related paper updates, when shared by the client.
- IEEE TGRS, IEEE JSTARS, IEEE GRSL, ISPRS Journal, Pattern Recognition, ESWA, Remote Sensing, and relevant CCF-listed venues.
- Recent arXiv and GitHub releases related to HSI classification, Mamba/state-space models, graph learning, foundation models, domain adaptation, and robust learning.

## What To Track

For each relevant paper, record:

- citation and link;
- model family: CNN, Transformer, graph, Mamba/state-space, foundation model, diffusion, SSL, robust learning, domain adaptation;
- datasets used;
- training protocol and label setting;
- reported metrics: OA, AA, Kappa, Macro-F1, per-class accuracy, stability, calibration, efficiency;
- whether official code is available;
- whether it should be used as a baseline, related-work reference, or motivation only;
- direct risk to CFA-GDRO's novelty claim.

## Baseline Decision Rules

A new paper should be promoted to the baseline list when it satisfies at least one of these conditions:

- it is a recent HSI classifier in TGRS, JSTARS, Pattern Recognition, ISPRS, or another strong venue;
- it uses Mamba/state-space modeling for HSI classification;
- it uses graph reasoning for HSI classification;
- it uses uncertainty, diffusion, prototype learning, robust learning, or self-training under low-label HSI settings;
- it is a recent Transformer or foundation-style HSI method that reviewers are likely to expect;
- it reports results under low-label, few-shot, rare-class, robust, cross-scene, or multi-seed protocols.

If code is unavailable, record whether a fair reimplementation is feasible. If neither official code nor fair reimplementation is practical, cite it in related work and explain why it is not included as a direct baseline.

## Minchao Ye Update Batch: 2026-05-18

| Paper | Year | Family | Venue / DOI | Code | Role | Notes for CFA-GDRO |
| --- | --- | --- | --- | --- | --- | --- |
| Cross-Scene Diffusion-Enhanced Uncertainty Attention Network for Hyperspectral Image Classification | 2026 | Diffusion, uncertainty, attention, cross-scene transfer | IEEE TGRS, DOI: 10.1109/TGRS.2026.3677462 | Check | High-priority related work; baseline candidate if code appears | Strong novelty-risk item because it directly overlaps with uncertainty, reliability, small-sample HSI, and cross-scene alignment. CFA-GDRO must distinguish robust graph-state-space learning from diffusion-based domain adaptation and uncertainty attention. |
| Combine Meta-Learning With Feature Alignment for Cross-Domain Heterogeneous Hyperspectral Image Classification | 2026 | Meta-learning, feature alignment, few-shot, heterogeneous transfer | IEEE TGRS, DOI: 10.1109/TGRS.2026.3652354 | Author accepted manuscript available; code to check | High-priority related work; possible baseline for cross-domain protocol | Important for label-scarce framing. It is not a direct single-scene baseline, but it raises reviewer expectations for few-shot and cross-domain discussion. |
| Adaptive Graph Modeling With Self-Training for Heterogeneous Cross-Scene Hyperspectral Image Classification | 2024 | Adaptive graph learning, self-training, heterogeneous cross-scene transfer | IEEE TGRS, DOI: 10.1109/TGRS.2023.3348953 | Check | Strong graph related work; baseline candidate for cross-scene extension | Directly relevant to the graph branch. CFA-GDRO should clarify that its graph is patch-level reliability modeling, not cross-scene adaptive graph transfer, unless the project is extended to cross-scene experiments. |
| Discriminative Vision Transformer for Heterogeneous Cross-Domain Hyperspectral Image Classification | 2024 | Transformer, cross-domain alignment, knowledge distillation | IEEE TGRS, DOI: 10.1109/TGRS.2024.3482848 | Check | Transformer related work; baseline candidate if protocol matches | Useful to address reviewer concern about recent Transformer baselines. Strong candidate for related work even if direct comparison is infeasible. |
| Building Cross-Domain Mapping Chains From Multi-CycleGAN for Hyperspectral Image Classification | 2024 | CycleGAN, adversarial domain mapping, heterogeneous transfer | IEEE TGRS, DOI: 10.1109/TGRS.2024.3431460 | Check | Domain adaptation related work | Relevant to cross-domain HSI but less direct to CFA-GDRO unless we add cross-scene transfer experiments. Should be cited to show awareness of recent heterogeneous transfer methods. |

## Immediate Actions

- Add these papers to the related-work pool and reference database.
- Check whether official code is available for the 2024 graph and Transformer methods.
- Treat the 2026 diffusion-uncertainty paper as a high-risk overlap item for the uncertainty claim.
- Avoid claiming that CFA-GDRO is the first uncertainty-aware or reliability-oriented cross-scene HSI method.
- Position CFA-GDRO as explicit robust risk minimization plus graph-state-space representation under label scarcity, unless experiments are expanded to cross-scene domain adaptation.

## Local PDF Review Batch: 2026-05-20

The following papers were provided as local PDFs and copied into `papers/` for project-local reference.

| Paper | Year | Family | Venue / DOI | Datasets / Protocol | Role | Notes for CFA-GDRO |
| --- | --- | --- | --- | --- | --- | --- |
| Combine Meta-Learning With Feature Alignment for Cross-Domain Heterogeneous Hyperspectral Image Classification | 2026 | Meta-learning, task-adaptive loss, adaptive source/target weighting, Gaussian-prior feature alignment | IEEE TGRS, DOI: 10.1109/TGRS.2026.3652354 | Four public HSI datasets; source and target may have different land-cover classes and different feature dimensions | High-priority related work; possible cross-domain extension reference | Important because it treats label scarcity through cross-domain heterogeneous few-shot learning. CFA-GDRO is currently single-scene label-scarce classification, so the paper should cite CD-MFA as adjacent work and avoid implying that it solves heterogeneous transfer unless a new protocol is added. |
| Cross-Scene Hyperspectral Feature Selection via Hybrid Whale Optimization Algorithm With Simulated Annealing | 2021 | Cross-scene feature selection, hybrid whale optimization, simulated annealing, spectral-shift handling | IEEE JSTARS, DOI: 10.1109/JSTARS.2021.3056593 | Indiana and Pavia cross-scene datasets; 200 labeled samples/class from source and 5 labeled samples/class from target; source used for feature selection and target used for final classifier | Related work on cross-scene label scarcity and spectral shift; not a deep baseline | Useful motivation for the project's low-label and spectral-shift discussion. It is not a direct CFA-GDRO baseline unless we add feature-selection preprocessing or cross-scene experiments. |
| Feature Selection for Cross-Scene Hyperspectral Image Classification Using Cross-Domain I-ReliefF | 2021 | Cross-domain I-ReliefF, ranking/filter feature selection, source-target consistency, outlier robustness | IEEE JSTARS, DOI: 10.1109/JSTARS.2021.3086151 | EShanghai-EHangzhou, DPaviaU-DPaviaC, and RPaviaC-RPaviaU; 200 labeled samples/class from source and limited labeled target samples; SVM classifier on selected bands | Related work on robust cross-scene band/feature selection; not a deep baseline | Useful to show that cross-scene reliability and feature consistency have been studied before. CFA-GDRO should position itself as representation/robust-risk learning rather than feature-selection optimization. |

## Related Paper Folder Review: 2026-05-20

The `related_paper/` folder contains nine local papers. Most are not direct single-scene baselines for the current CFA-GDRO protocol, but they strongly affect the novelty story because they show that label scarcity, cross-scene reliability, graph modeling, uncertainty, and Transformer-based adaptation are already active HSI topics.

| Paper | Family | Main Mechanism | Datasets / Protocol | Relevance to CFA-GDRO | Action |
| --- | --- | --- | --- | --- | --- |
| Adaptive Graph Modeling With Self-Training for Heterogeneous Cross-Scene Hyperspectral Image Classification | Graph, self-training, heterogeneous transfer | Learnable graph weights, adaptive cutoff threshold, cross-scene graph loss, pseudo-label self-training | RPaviaU-DPaviaC, EHangzhou-RPaviaHR, Houston2018-Houston2013 | High graph novelty risk. It already uses adaptive graph modeling for small-sample cross-scene HSI, so CFA-GDRO should not claim graph learning itself as novel. | Cite in graph/cross-scene related work; use as direct baseline only if adding cross-scene experiments. |
| Building Cross-Domain Mapping Chains From Multi-CycleGAN for Hyperspectral Image Classification | GAN/domain mapping, heterogeneous transfer | Multi-CycleGAN unfolded into cross-domain mapping chains plus auxiliary classifiers | EHangzhou-RPaviaHR, RPaviaU-DPaviaC, Houston2018-Houston2013 | Shows strong recent heterogeneous-transfer baselines and source-target mapping expectations. | Cite as adjacent cross-domain work; not a direct baseline for current single-scene protocol. |
| Combine Meta-Learning With Feature Alignment for Cross-Domain Heterogeneous Hyperspectral Image Classification | Meta-learning, few-shot transfer, feature alignment | Task-adaptive inner-loop loss, adaptive source/target weighting, Gaussian-prior feature alignment | Indian Pines, University of Pavia, Salinas, and another public scene under cross-domain heterogeneous few-shot protocol | Strong label-scarce reference. It overlaps with few-shot reliability but solves heterogeneous source-target transfer, not current single-scene classification. | Cite prominently; direct baseline only if cross-domain extension is added. |
| Cross-Scene Hyperspectral Feature Selection via Hybrid Whale Optimization Algorithm With Simulated Annealing | Cross-scene feature selection | CDWOASA balances class separability and cross-scene feature consistency; controls selected-band count | Indiana and Pavia cross-scene feature-selection protocols | Supports motivation that low-label HSI is also a feature-consistency/spectral-shift problem. | Cite as classical cross-scene feature-selection work; not a deep baseline. |
| Cross-domain Hyperspectral Image Classification Based on Bi-directional Domain Adaptation | Transformer/domain adaptation | Triple-branch Transformer, semantic tokenizer, coupled multi-head cross-attention, bidirectional distillation, adaptive reinforcement under noise | Tree species cross-temporal data, Houston 2013/2018, HyRANK Dioni/Loukia | Important Transformer novelty risk for robustness/domain adaptation. It includes noise-condition robustness and code availability claim. | Add to high-priority related work; check official code before any cross-domain extension. |
| Cross-Scene Diffusion-Enhanced Uncertainty Attention Network for Hyperspectral Image Classification | Diffusion, uncertainty, attention, cross-scene transfer | Target-guided diffusion augmentation, uncertainty-aware mutual attention, multiscale contrastive fusion attention | RPaviaU-DPaviaC, EHangzhou-RPaviaHR, Houston | Very high uncertainty/reliability novelty risk. It already frames small-sample HSI around uncertainty and reliability. | Cite as high-risk related work; avoid claiming first uncertainty-aware reliability method. |
| Discriminative Vision Transformer for Heterogeneous Cross-Domain Hyperspectral Image Classification | Transformer, feature alignment, distillation | Encoder-decoder source-to-target mapping, domainwise/classwise cross-attention alignment, knowledge distillation | EHangzhou-RPaviaHR, RPaviaC-DPaviaU, Houston2018-Houston2013 | Important recent Transformer baseline if project becomes cross-domain; useful for reviewer concerns about recent Transformer HSI work. | Cite; baseline only for cross-domain protocol. |
| Feature Selection for Cross-Scene Hyperspectral Image Classification Using Cross-Domain I-ReliefF | Cross-scene feature selection | Cross-scene ReliefF update using class separability and source-target spectral consistency | EShanghai-EHangzhou, DPaviaU-DPaviaC, RPaviaC-RPaviaU | Classical evidence that source-target consistency and low-label feature selection are established. | Cite as motivation/boundary; not direct deep baseline. |
| Nonlocal Graph Convolutional Networks for Hyperspectral Image Classification | Graph, semisupervised single-scene classification | Whole-image nonlocal graph over labeled and unlabeled pixels, graph convolution, supervised CE on labeled nodes | Pavia University, Indian Pines, Salinas | Directly relevant graph baseline for single-scene HSI. It is older but TGRS and reviewer-relevant. | Keep as graph baseline family; official implementation or faithful reproduction is preferable to only a lite baseline. |

## Implications For Current Paper Direction

- These papers strengthen the argument that label scarcity, spectral shift, and source/target inconsistency are recognized HSI problems.
- They also create a boundary: CFA-GDRO should not be sold as a cross-domain heterogeneous transfer method unless the code and experiments are extended.
- For the current project, cite them in related work and motivation, but do not list them as direct baselines for single-scene Indian Pines/Pavia/Salinas experiments.
- If we later add a cross-scene protocol, CD-MFA becomes a high-priority comparison, while CDWOASA and CDIRF become classical feature-selection baselines.
- BiDA should be added to the monitored Transformer/domain-adaptation pool because it combines cross-attention, bidirectional distillation, and noise-oriented robustness. It is not a direct baseline for the current single-scene protocol, but it becomes important if the project adds cross-temporal or cross-scene robustness experiments.

## Working Table

| Paper | Year | Family | Datasets | Code | Role | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| SpectralFormer | 2022 | Transformer | Common HSI benchmarks | Check | Baseline candidate | Existing reference point for spectral Transformer comparison. |
| FactoFormer | 2024 | Transformer/SSL | Common HSI benchmarks | Check | Strong baseline candidate | Explicitly mentioned by previous TGRS reviewer. |
| HyperEAST | 2025 | Transformer/SSL | Common HSI benchmarks | Check | Strong baseline candidate | Explicitly mentioned by previous TGRS reviewer. |
| MambaMoE | Recent | Mamba/mixture | HSI benchmarks | Check | Related work or baseline candidate | Explicitly mentioned by previous TGRS reviewer. |
| CD-MFA | 2026 | Meta-learning/domain adaptation | Four public HSI datasets | Check | Related work; cross-domain baseline if protocol added | Strong adjacent work for few-shot heterogeneous transfer. |
| CDWOASA | 2021 | Cross-scene feature selection | Indiana and Pavia cross-scene datasets | Not needed for single-scene baseline | Motivation; cross-scene feature-selection baseline if protocol added | Useful for spectral-shift and limited target-label discussion. |
| CDIRF | 2021 | Cross-domain I-ReliefF feature selection | EShanghai-EHangzhou, DPaviaU-DPaviaC, RPaviaC-RPaviaU | Not needed for single-scene baseline | Motivation; cross-scene feature-selection baseline if protocol added | Useful for source-target consistency and robust feature ranking discussion. |
| Nonlocal GCN | 2020 | Graph convolution/semisupervised learning | Three benchmark HSI datasets | Paper PDF local | Graph baseline family | Motivates comparing CFA-GDRO against graph reasoning, not only CNN/Transformer baselines. A lightweight nonlocal-gcn baseline is implemented. |
| BiDA | Recent | Transformer/domain adaptation | Tree species cross-temporal data, Houston 2013/2018, HyRANK Dioni/Loukia | Code availability claimed in local PDF; verify | High-priority related work; cross-domain baseline if protocol added | Important because it explicitly targets cross-domain robustness with Transformer attention, bidirectional distillation, and noise-focused reinforcement. |

## Review Cadence

Update this file whenever the client shares Minchao Ye's latest HSI paper updates or when a new relevant method appears. Before final experiments, freeze a baseline list and justify every included or excluded recent method in the manuscript.
