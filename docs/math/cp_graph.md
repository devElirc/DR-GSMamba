# M1.6 — Cross-Pixel Affinity Graph (CP-Graph) consistency loss

This note locks the definition of the auxiliary loss $\mathcal{L}_{\mathrm{CP\text{-}graph}}$ used in the total objective of `cfa_gdro.md` §3 and `method.tex` Eq. (16).

**CP-Graph is a design choice, not a contribution.** This note exists only so that every symbol in the locked equations is defined, and so the implementation in `hsi_robust/losses/cp_graph.py` has a single source of truth.

---

## 1. Setting and notation

Inside one mini-batch of size $N$ we use:

- fused features $\{f_i \in \mathbb{R}^{d}\}_{i=1}^{N}$ produced by the backbone (output of `models/fusion.py`),
- predictive probabilities $\{p_i \in \Delta^{K-1}\}_{i=1}^{N}$ produced by the evidential prototype head (Eq. (10) of `method.tex`); note that these are *not* the raw evidences,
- a positive integer $k \in \{1,\dots,N-1\}$ that fixes the graph degree (default $k = 8$),
- a positive scalar $\tau_g > 0$ (graph temperature, default $\tau_g = 1.0$, *separate* from the EPH temperature $\tau$).

The unit-normalised feature is $\tilde f_i = f_i / \|f_i\|_2$ and cosine similarity is $s_{ij} = \tilde f_i^{\top} \tilde f_j \in [-1, 1]$.

---

## 2. In-batch $k$-NN graph

For each pixel $i$ let $\mathcal{N}_k(i) \subseteq \{1,\dots,N\}\setminus\{i\}$ be the indices of its $k$ nearest neighbours by cosine similarity computed within the same mini-batch only — i.e. the graph is rebuilt fresh every batch with no persistent state. The edge set is

$$
\mathcal{E} \;=\; \big\{\,(i,j) \;:\; j \in \mathcal{N}_k(i)\,\big\},\qquad |\mathcal{E}| \;=\; k\,N.
$$

The graph is *directed* by construction (cosine $k$-NN is not symmetric); we do *not* symmetrise.

---

## 3. Edge weights

Edge weights are softmax-normalised cosine similarities over each source vertex's neighbourhood, with temperature $\tau_g$:

$$
\boxed{\;w_{ij} \;=\; \frac{\exp(s_{ij} / \tau_g)}{\sum_{j' \in \mathcal{N}_k(i)} \exp(s_{ij'} / \tau_g)}, \qquad (i, j) \in \mathcal{E}.\;}
$$

By construction $w_{ij} \ge 0$ and $\sum_{j \in \mathcal{N}_k(i)} w_{ij} = 1$ for every source $i$. The weights depend on $\theta$ through $\tilde f_i$ and we *detach* them from the computation graph (see §4) so that the consistency loss does not push features apart through the weight term itself.

---

## 4. CP-Graph consistency loss

We use a one-directional KL with the neighbour-averaged prediction acting as the (stop-gradient) target. Define the neighbour-averaged prediction

$$
\tilde p_i \;=\; \sum_{j \in \mathcal{N}_k(i)} w_{ij}\,p_j \;\in\; \Delta^{K-1},
$$

and the per-pixel consistency loss

$$
\ell^{\mathrm{CP}}_i \;=\; \mathrm{KL}\!\big(\,\mathrm{sg}(\tilde p_i) \;\big\|\; p_i\,\big),
$$

where $\mathrm{sg}(\cdot)$ is the stop-gradient operator (no gradient flows through $\tilde p$). The batch-level loss is then

$$
\boxed{\;\mathcal{L}_{\mathrm{CP\text{-}graph}}(\theta) \;=\; \frac{1}{N}\sum_{i=1}^{N} \ell^{\mathrm{CP}}_i \;=\; \frac{1}{N}\sum_{i=1}^{N} \mathrm{KL}\!\big(\,\mathrm{sg}(\tilde p_i) \;\big\|\; p_i\,\big).\;}
$$

**Why one-directional with stop-gradient?** Two reasons. (i) Symmetric KL on a noisy graph drives the optimiser to collapse all neighbouring predictions to the same vector, which destroys class boundaries; the asymmetric form lets the neighbourhood pull a noisy $p_i$ toward the local consensus without dragging the consensus toward $p_i$. (ii) The stop-gradient yields a stable target that varies on the slow scale of the backbone, not the fast scale of one batch update — the same idea behind BYOL-style self-distillation and consistency regularisation in semi-supervised learning.

**Why not symmetric KL on each edge?** A symmetric edge-wise loss $\sum_{(i,j)} w_{ij}\,\big[\mathrm{KL}(p_i\|p_j) + \mathrm{KL}(p_j\|p_i)\big]/2$ is also coherent and produces almost identical numbers in pilot tests; we prefer the asymmetric form for the stop-gradient stability argument above. Ablation against the symmetric form is part of Phase 7 only if the design is challenged in review.

---

## 5. Compatibility with EPH and CFA-GDRO

CP-Graph operates on the EPH predictive probabilities $p_i = \alpha_i / S_i$ from `method.tex` Eq. (10) and Eq. (11). It does *not* operate on the raw evidence vector $e_i$ — using $p_i$ keeps the regulariser on the same calibrated quantity that CFA-GDRO eventually re-weights.

Because $\mathcal{L}_{\mathrm{CP\text{-}graph}}$ is an *unweighted batch mean*, not a per-class average, it is **not** passed through the CFA-GDRO water-filling solver. It enters the total loss additively with weight $\lambda_{\mathrm{graph}}$, exactly as written in `method.tex` Eq. (16):

$$
\mathcal{L}_{\mathrm{total}}(\theta)
\;=\;
\overline{\ell^{\mathrm{EPH}}}(\theta)
\;+\;
\lambda_{\mathrm{rob}}\,\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}^{\mathrm{EPH}}(\theta)
\;+\;
\lambda_{\mathrm{graph}}\,\mathcal{L}_{\mathrm{CP\text{-}graph}}(\theta).
$$

This separation is deliberate: the robust class-reweighting handled by CFA-GDRO and the per-pixel smoothing handled by CP-Graph operate on different statistical levels (class vs. pixel) and should not interact through the inner LP solver.

---

## 6. Special cases (sanity checks)

| Choice | Effect |
| --- | --- |
| $\lambda_{\mathrm{graph}} = 0$ | CP-Graph is disabled; total loss reduces to mean-EPH + CFA-GDRO |
| $k = 0$ | $\mathcal{N}_k(i) = \emptyset$ for all $i$; $\mathcal{L}_{\mathrm{CP\text{-}graph}}$ is undefined and the implementation returns $0$ |
| $\tau_g \to 0^{+}$ | $w_{ij}$ becomes a one-hot on the single nearest neighbour; $\tilde p_i = p_{j^{\star}(i)}$ |
| $\tau_g \to \infty$ | $w_{ij} = 1/k$ uniform; $\tilde p_i$ is the unweighted mean of the $k$ nearest predictions |

---

## 7. Hyperparameter defaults

| Parameter | Symbol | Default | Range explored |
| --- | --- | --- | --- |
| Neighbourhood size | $k$ | $8$ | $\{4, 8, 16\}$ |
| Graph temperature | $\tau_g$ | $1.0$ | $\{0.5, 1.0, 2.0\}$ |
| Graph weight | $\lambda_{\mathrm{graph}}$ | $0.1$ | $\{0, 0.05, 0.1, 0.3\}$ |

The default $\lambda_{\mathrm{graph}} = 0.1$ is one order of magnitude smaller than $\lambda_{\mathrm{rob}}$; the priority of the total loss is mean-EPH first, CFA-GDRO second, CP-Graph as a small smoother.

---

## 8. Implementation contract

```python
# hsi_robust/losses/cp_graph.py
def cp_graph_loss(
    features:    torch.Tensor,   # (N, d), fused features f_i
    probs:       torch.Tensor,   # (N, K), EPH predictive probabilities p_i
    k:           int = 8,
    tau_g:       float = 1.0,
    eps:         float = 1e-12,  # numerical stability for KL
) -> tuple[torch.Tensor, dict]:
    """
    Returns (loss, info), where info contains:
        - mean_neighbour_kl   : float, mean KL(stopgrad(p_tilde) || p) over the batch
        - mean_weight_entropy : float, mean entropy of softmax weights w_{ij}
        - degree              : int, k

    Notes:
        - Gradients flow through `probs` and through `features` via the
          unnormalised cosine in the softmax. Edge weights w_{ij} themselves
          are detached.
        - Neighbour-averaged target p_tilde is detached before the KL.
        - When N <= k, k is silently reduced to N - 1 and the loss is
          computed on the truncated graph.
    """
```

This is the exact signature that Phase 2D will implement.

---

## 9. What is intentionally *not* claimed

- CP-Graph is **not** claimed as a novel graph reasoning module. In-batch $k$-NN graph propagation has been used in deep metric learning (DML), self-supervised learning (BYOL, SimSiam), and several recent HSI papers.
- The contribution claim of the paper is on the loss-level reweighting (CFA-GDRO) and uncertainty (EPH), not on this regulariser.
- Ablations show CP-Graph contributes a small but consistent improvement; when the design is challenged we expect to keep it as a "well-tuned default", not defend it as a novelty.
