# DR-GSMamba Literature Tracking

This file records the ongoing paper-monitoring workflow for keeping DR-GSMamba current with recent hyperspectral image classification research.

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
- direct risk to DR-GSMamba's novelty claim.

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

| Paper | Year | Family | Venue / DOI | Code | Role | Notes for DR-GSMamba |
| --- | --- | --- | --- | --- | --- | --- |
| Cross-Scene Diffusion-Enhanced Uncertainty Attention Network for Hyperspectral Image Classification | 2026 | Diffusion, uncertainty, attention, cross-scene transfer | IEEE TGRS, DOI: 10.1109/TGRS.2026.3677462 | Check | High-priority related work; baseline candidate if code appears | Strong novelty-risk item because it directly overlaps with uncertainty, reliability, small-sample HSI, and cross-scene alignment. DR-GSMamba must distinguish robust graph-state-space learning from diffusion-based domain adaptation and uncertainty attention. |
| Combine Meta-Learning With Feature Alignment for Cross-Domain Heterogeneous Hyperspectral Image Classification | 2026 | Meta-learning, feature alignment, few-shot, heterogeneous transfer | IEEE TGRS, DOI: 10.1109/TGRS.2026.3652354 | Author accepted manuscript available; code to check | High-priority related work; possible baseline for cross-domain protocol | Important for label-scarce framing. It is not a direct single-scene baseline, but it raises reviewer expectations for few-shot and cross-domain discussion. |
| Adaptive Graph Modeling With Self-Training for Heterogeneous Cross-Scene Hyperspectral Image Classification | 2024 | Adaptive graph learning, self-training, heterogeneous cross-scene transfer | IEEE TGRS, DOI: 10.1109/TGRS.2023.3348953 | Check | Strong graph related work; baseline candidate for cross-scene extension | Directly relevant to the graph branch. DR-GSMamba should clarify that its graph is patch-level reliability modeling, not cross-scene adaptive graph transfer, unless the project is extended to cross-scene experiments. |
| Discriminative Vision Transformer for Heterogeneous Cross-Domain Hyperspectral Image Classification | 2024 | Transformer, cross-domain alignment, knowledge distillation | IEEE TGRS, DOI: 10.1109/TGRS.2024.3482848 | Check | Transformer related work; baseline candidate if protocol matches | Useful to address reviewer concern about recent Transformer baselines. Strong candidate for related work even if direct comparison is infeasible. |
| Building Cross-Domain Mapping Chains From Multi-CycleGAN for Hyperspectral Image Classification | 2024 | CycleGAN, adversarial domain mapping, heterogeneous transfer | IEEE TGRS, DOI: 10.1109/TGRS.2024.3431460 | Check | Domain adaptation related work | Relevant to cross-domain HSI but less direct to DR-GSMamba unless we add cross-scene transfer experiments. Should be cited to show awareness of recent heterogeneous transfer methods. |

## Immediate Actions

- Add these papers to the related-work pool and reference database.
- Check whether official code is available for the 2024 graph and Transformer methods.
- Treat the 2026 diffusion-uncertainty paper as a high-risk overlap item for the uncertainty claim.
- Avoid claiming that DR-GSMamba is the first uncertainty-aware or reliability-oriented cross-scene HSI method.
- Position DR-GSMamba as explicit robust risk minimization plus graph-state-space representation under label scarcity, unless experiments are expanded to cross-scene domain adaptation.

## Local PDF Review Batch: 2026-05-20

The following papers were provided as local PDFs and copied into `papers/` for project-local reference.

| Paper | Year | Family | Venue / DOI | Datasets / Protocol | Role | Notes for DR-GSMamba |
| --- | --- | --- | --- | --- | --- | --- |
| Combine Meta-Learning With Feature Alignment for Cross-Domain Heterogeneous Hyperspectral Image Classification | 2026 | Meta-learning, task-adaptive loss, adaptive source/target weighting, Gaussian-prior feature alignment | IEEE TGRS, DOI: 10.1109/TGRS.2026.3652354 | Four public HSI datasets; source and target may have different land-cover classes and different feature dimensions | High-priority related work; possible cross-domain extension reference | Important because it treats label scarcity through cross-domain heterogeneous few-shot learning. DR-GSMamba is currently single-scene label-scarce classification, so the paper should cite CD-MFA as adjacent work and avoid implying that it solves heterogeneous transfer unless a new protocol is added. |
| Cross-Scene Hyperspectral Feature Selection via Hybrid Whale Optimization Algorithm With Simulated Annealing | 2021 | Cross-scene feature selection, hybrid whale optimization, simulated annealing, spectral-shift handling | IEEE JSTARS, DOI: 10.1109/JSTARS.2021.3056593 | Indiana and Pavia cross-scene datasets; 200 labeled samples/class from source and 5 labeled samples/class from target; source used for feature selection and target used for final classifier | Related work on cross-scene label scarcity and spectral shift; not a deep baseline | Useful motivation for the project's low-label and spectral-shift discussion. It is not a direct DR-GSMamba baseline unless we add feature-selection preprocessing or cross-scene experiments. |
| Feature Selection for Cross-Scene Hyperspectral Image Classification Using Cross-Domain I-ReliefF | 2021 | Cross-domain I-ReliefF, ranking/filter feature selection, source-target consistency, outlier robustness | IEEE JSTARS, DOI: 10.1109/JSTARS.2021.3086151 | EShanghai-EHangzhou, DPaviaU-DPaviaC, and RPaviaC-RPaviaU; 200 labeled samples/class from source and limited labeled target samples; SVM classifier on selected bands | Related work on robust cross-scene band/feature selection; not a deep baseline | Useful to show that cross-scene reliability and feature consistency have been studied before. DR-GSMamba should position itself as representation/robust-risk learning rather than feature-selection optimization. |

## Implications For Current Paper Direction

- These papers strengthen the argument that label scarcity, spectral shift, and source/target inconsistency are recognized HSI problems.
- They also create a boundary: DR-GSMamba should not be sold as a cross-domain heterogeneous transfer method unless the code and experiments are extended.
- For the current project, cite them in related work and motivation, but do not list them as direct baselines for single-scene Indian Pines/Pavia/Salinas experiments.
- If we later add a cross-scene protocol, CD-MFA becomes a high-priority comparison, while CDWOASA and CDIRF become classical feature-selection baselines.

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
| Nonlocal GCN | 2020 | Graph convolution/semisupervised learning | Three benchmark HSI datasets | Paper PDF local | Graph baseline family | Motivates comparing DR-GSMamba against graph reasoning, not only CNN/Transformer baselines. A lightweight nonlocal-gcn baseline is implemented. |

## Review Cadence

Update this file whenever the client shares Minchao Ye's latest HSI paper updates or when a new relevant method appears. Before final experiments, freeze a baseline list and justify every included or excluded recent method in the manuscript.
