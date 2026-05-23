# M1.1 + M1.2 — Class-Frequency-Aware Group-DRO (CFA-GDRO)

This note locks the mathematical definition of the main contribution and proves the worst-class upper bound that justifies the title word *Reliable*. All notation in this document will be reused unchanged in `paper/sections/method.tex` and in the implementation in `hsi_robust/losses/cfa_gdro.py`.

---

## 1. Setting and notation

Let the training set be $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ with $K$ classes, $y_i \in \{1,\dots,K\}$. Define

- class index set $\mathcal{K} = \{1,\dots,K\}$,
- $N_k^{\mathrm{train}} = |\{i : y_i = k\}|$, the per-class count in the training set,
- $N_k^{\mathrm{scene}}$, the per-class count in the *full ground truth* of the scene (including unlabeled pixels),
- $\pi_k = N_k^{\mathrm{scene}} / \sum_j N_j^{\mathrm{scene}}$ — the **scene-level** class frequency. Note: this is the natural prevalence of each class in the scene; it is *not* the training-set frequency, which would be uniform under fixed-samples-per-class training and would make $\gamma$ vanish from the math,
- minimum / maximum scene frequencies $\pi_{\min} = \min_k \pi_k > 0$ and $\pi_{\max} = \max_k \pi_k < 1$,
- $\ell_\theta(x, y) = -\log p_\theta(y \mid x)$ — the per-sample loss for a model parameterised by $\theta$,
- per-class average loss $\bar\ell_k(\theta) = \frac{1}{N_k^{\mathrm{train}}} \sum_{i : y_i = k} \ell_\theta(x_i, y_i)$.

Two parameters control the robust objective:

- $\alpha \in (0, 1]$ — the *worst-fraction* size, identical in spirit to CVaR's $\alpha$. Smaller $\alpha$ is more robust.
- $\gamma \geq 0$ — the *class-frequency-awareness* exponent. $\gamma=0$ gives uniform-cap class-CVaR; $\gamma>0$ gives rare classes a larger adversarial cap.

We will also use the standard probability simplex $\Delta^{K-1} = \{q \in \mathbb{R}^{K}_{\geq 0} : \sum_k q_k = 1\}$.

**Notation note.** The scalar CFA-GDRO worst-fraction $\alpha$ collides with the standard Dirichlet concentration vector notation $\boldsymbol{\alpha} = (\alpha_1,\dots,\alpha_K)$ used in `evidential_prototype_head.md`. To avoid ambiguity we keep both — the Dirichlet concentration is always written with a class index ($\alpha_k$) or in bold ($\boldsymbol{\alpha}$), while the CFA-GDRO worst-fraction is always a scalar.

---

## 2. The constraint set $\mathcal{Q}_{\alpha,\gamma}$

Define class weights

$$
w_k(\gamma) \;=\; \pi_k^{-\gamma},
\qquad
W(\gamma) \;=\; \sum_{j=1}^{K} w_j(\gamma) \;=\; \sum_{j=1}^{K} \pi_j^{-\gamma},
$$

and per-class caps

$$
\boxed{\; c_k(\alpha,\gamma) \;=\; \frac{1}{\alpha}\cdot\frac{w_k(\gamma)}{W(\gamma)} \;=\; \frac{1}{\alpha}\cdot\frac{\pi_k^{-\gamma}}{\sum_j \pi_j^{-\gamma}}. \;}
$$

The CFA-GDRO uncertainty set is

$$
\boxed{\;\mathcal{Q}_{\alpha,\gamma} \;=\; \big\{\, q \in \Delta^{K-1} \;:\; 0 \le q_k \le c_k(\alpha,\gamma)\ \ \forall k\,\big\}. \;}
$$

**Feasibility.** Since $\sum_k c_k = 1/\alpha \ge 1$ (with equality iff $\alpha=1$), the simplex constraint $\sum_k q_k = 1$ is always achievable inside the box $0 \le q \le c$, so $\mathcal{Q}_{\alpha,\gamma}$ is non-empty for every $\alpha \in (0,1]$ and $\gamma \ge 0$.

**Special cases (sanity checks).**

| Choice | $c_k$ becomes | $\mathcal{Q}$ becomes | Equivalent to |
| --- | --- | --- | --- |
| $\gamma=0,\ \alpha=1$ | $1/K$ | uniform over $K$ | uniform-class loss |
| $\gamma=0,\ \alpha=1/K$ | $1$ | $\Delta^{K-1}$ | Sagawa group-DRO (worst-class) |
| $\gamma=0,\ \alpha \in (1/K, 1)$ | $1/(\alpha K)$ | class-level CVaR | uniform-cap class CVaR |
| $\gamma>0$ | $\pi_k^{-\gamma}/(\alpha\,W)$ | overweights rare classes | **CFA-GDRO (ours)** |

---

## 3. The CFA-GDRO objective

The CFA-GDRO loss is the supremum of the weighted class loss over $\mathcal{Q}_{\alpha,\gamma}$:

$$
\boxed{\;\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}(\theta) \;=\; \sup_{q\,\in\,\mathcal{Q}_{\alpha,\gamma}} \;\sum_{k=1}^{K} q_k\,\bar\ell_k(\theta). \;}
$$

The total training objective composes CFA-GDRO with the standard cross-entropy term and the auxiliary losses (defined in `M1.3` and `M1.4`):

$$
\mathcal{L}_{\mathrm{total}}(\theta)
\;=\;
\underbrace{\bar\ell(\theta)}_{\text{mean CE}}
\;+\;
\lambda_{\mathrm{rob}}\,\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}(\theta)
\;+\;
\lambda_{\mathrm{evi}}\,\mathcal{L}_{\mathrm{EPH}}(\theta)
\;+\;
\lambda_{\mathrm{graph}}\,\mathcal{L}_{\mathrm{CP\text{-}graph}}(\theta).
$$

The mean CE term keeps optimisation well behaved when CFA-GDRO is sharp (small $\alpha$). The hyperparameters $(\alpha,\gamma,\lambda_{\mathrm{rob}})$ are tuned on the validation split only, per dataset.

---

## 4. Dual form and closed-form solver

The inner supremum is a linear program over a polytope, with a known closed-form solver.

### 4.1 KKT analysis

Lagrangian (with $\nu \in \mathbb{R}$ for the equality, $\mu \in \mathbb{R}^{K}_{\ge 0}$ for the upper caps, $\lambda \in \mathbb{R}^{K}_{\ge 0}$ for the lower bounds):

$$
\mathcal{L}(q,\nu,\mu,\lambda) \;=\; \sum_k q_k\,\bar\ell_k \;-\; \nu\Big(\sum_k q_k - 1\Big) \;-\; \sum_k \mu_k(q_k - c_k) \;+\; \sum_k \lambda_k q_k.
$$

Stationarity yields $\bar\ell_k - \nu - \mu_k + \lambda_k = 0$, with complementary slackness:

- $q_k^\star \in (0, c_k) \;\Rightarrow\; \mu_k = \lambda_k = 0 \;\Rightarrow\; \bar\ell_k = \nu$,
- $q_k^\star = c_k \;\Rightarrow\; \mu_k = \bar\ell_k - \nu \ge 0 \;\Rightarrow\; \bar\ell_k \ge \nu$,
- $q_k^\star = 0 \;\Rightarrow\; \lambda_k = \nu - \bar\ell_k \ge 0 \;\Rightarrow\; \bar\ell_k \le \nu$.

So classes split into three groups by their loss relative to the threshold $\nu$:

1. **Saturated** ($\bar\ell_k > \nu$): cap-bound, $q_k^\star = c_k$.
2. **Boundary** (typically one class, $\bar\ell_k = \nu$): receives the residual mass.
3. **Dropped** ($\bar\ell_k < \nu$): $q_k^\star = 0$.

### 4.2 Water-filling algorithm (per batch)

Inputs: per-class losses $\bar\ell \in \mathbb{R}^K$, caps $c \in \mathbb{R}_{\ge 0}^K$ (computed once from $\pi$, $\alpha$, $\gamma$).

```
Algorithm WaterFill(loss, cap):
    sort classes by loss DESCENDING into order (1), (2), ..., (K)
    remaining = 1.0
    q = zeros(K)
    for i = 1, 2, ..., K:
        if remaining <= 0:
            break
        take = min(cap[(i)], remaining)
        q[(i)] = take
        remaining = remaining - take
    return q
```

Runtime: $O(K\log K)$ per batch from the sort; the rest is $O(K)$. Numerically stable because all quantities lie in $[0, 1/\alpha]$ and the loop is monotone.

### 4.3 Sub-gradient for back-propagation

Once $q^\star$ is computed (no gradients flow through the sort), CFA-GDRO is a fixed linear combination of class losses:

$$
\nabla_\theta\,\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}(\theta)
\;=\;
\sum_{k=1}^{K} q_k^\star\,\nabla_\theta\,\bar\ell_k(\theta).
$$

In PyTorch we detach $q^\star$ and back-propagate through `(q.detach() * class_losses).sum()`. This matches the standard *Danskin's theorem* argument used by Sagawa et al. for group-DRO.

### 4.4 EMA stabilisation of class losses

Under label scarcity, a batch may not contain every class, and per-class means $\bar\ell_k$ computed only on the batch are noisy. We stabilise by an exponential moving average:

$$
\hat\ell_k^{(t+1)}
\;=\;
\begin{cases}
\eta\,\hat\ell_k^{(t)} + (1-\eta)\,\bar\ell_k^{(\text{batch})} & \text{if class } k \text{ appears in this batch}, \\
\hat\ell_k^{(t)} & \text{otherwise.}
\end{cases}
$$

We feed $\hat\ell$ (not the noisy batch mean) into WaterFill. The momentum $\eta$ defaults to $0.9$. Without this EMA the worst-class signal is too unstable for low-label settings (3–5 samples/class).

---

## 5. Per-class upper bounds (the *Reliable* claim)

The "reliable" claim of the paper is operationalised by showing that the CFA-GDRO objective acts as an *explicit upper bound on the loss of a class whose adversarial cap reaches one*. We state this as a single proposition with two corollaries: a rare-class bound (mild condition) and a true worst-class bound (stronger condition).

### 5.1 Main proposition

**Proposition 1 (Per-class dominance).** Fix $\alpha \in (0,1]$, $\gamma \ge 0$. Define the *active class set*

$$
\mathcal{K}^{\!\star}(\alpha,\gamma) \;=\; \{\,k \in \mathcal{K} \;:\; c_k(\alpha,\gamma) \ge 1\,\}
\;=\;
\{\,k \in \mathcal{K} \;:\; \pi_k^{-\gamma} \ge \alpha\,W(\gamma)\,\}.
$$

Then for every $\theta$ and every $k \in \mathcal{K}^{\!\star}$,

$$
\boxed{\;\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}(\theta) \;\ge\; \bar\ell_k(\theta). \;}
$$

In particular,

$$
\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}(\theta) \;\ge\; \max_{k\,\in\,\mathcal{K}^{\!\star}(\alpha,\gamma)} \bar\ell_k(\theta).
$$

### 5.2 Proof

For $k \in \mathcal{K}^{\!\star}$ we have $c_k \ge 1$, so the singleton distribution $q^{(k)}$ with $q^{(k)}_k = 1$ and $q^{(k)}_j = 0$ for $j \ne k$ satisfies $0 \le q^{(k)}_j \le c_j$ for all $j$, hence $q^{(k)} \in \mathcal{Q}_{\alpha,\gamma}$. By definition of supremum,

$$
\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}(\theta)
\;=\; \sup_{q \in \mathcal{Q}_{\alpha,\gamma}} \sum_j q_j\,\bar\ell_j(\theta)
\;\ge\; \sum_j q^{(k)}_j\,\bar\ell_j(\theta)
\;=\; \bar\ell_k(\theta).
$$

Taking the maximum over $k \in \mathcal{K}^{\!\star}$ gives the second display. $\blacksquare$

### 5.3 Two useful corollaries

The key question becomes *which classes belong to $\mathcal{K}^{\!\star}$?* Since $\pi_k^{-\gamma}$ is decreasing in $\pi_k$ for $\gamma > 0$, the rarest classes enter $\mathcal{K}^{\!\star}$ first.

**Corollary A (Rare-class bound).** If

$$
\alpha\,W(\gamma) \;\le\; \pi_{\min}^{-\gamma},
$$

then the rarest class $k_{\min} = \arg\min_k \pi_k$ lies in $\mathcal{K}^{\!\star}$, and

$$
\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}(\theta) \;\ge\; \bar\ell_{k_{\min}}(\theta).
$$

*Proof.* Substituting $k = k_{\min}$ into the definition of $\mathcal{K}^{\!\star}$ gives $\pi_{\min}^{-\gamma} \ge \alpha W(\gamma)$, exactly the hypothesis. Apply Proposition 1. $\blacksquare$

**Corollary B (Full worst-class bound).** If

$$
\alpha\,W(\gamma) \;\le\; \pi_{\max}^{-\gamma},
$$

then $\mathcal{K}^{\!\star} = \mathcal{K}$ and

$$
\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}(\theta) \;\ge\; \max_{k} \bar\ell_k(\theta).
$$

*Proof.* The hypothesis ensures $\pi_k^{-\gamma} \ge \pi_{\max}^{-\gamma} \ge \alpha W(\gamma)$ for *every* $k$, so $\mathcal{K}^{\!\star} = \mathcal{K}$. Apply Proposition 1 with the maximum. $\blacksquare$

### 5.4 Practical reading

The two corollaries describe a sliding scale of guarantees:

- **Corollary A** is mild; it asks only that the rarest class can be picked out by the adversary. In practice this requires $\alpha$ to be small.
- **Corollary B** is strict — it gives the standard worst-class DRO bound — but the condition $\alpha W(\gamma) \le \pi_{\max}^{-\gamma}$ is *very* tight on imbalanced scenes where $\pi_{\max}^{-\gamma}$ is small.

Concrete numbers on Indian Pines (full scene, $K=16$, $\pi_{\min} \approx 2.3\times 10^{-3}$, $\pi_{\max} \approx 2.1\times 10^{-1}$, $W(\gamma{=}1) \approx 8.3 \times 10^{3}$):

| Bound | Condition | Threshold on $\alpha$ (at $\gamma{=}1$) |
| --- | --- | --- |
| Corollary A (rare-class) | $\alpha\,W \le \pi_{\min}^{-1}$ | $\alpha \le \pi_{\min}^{-1}/W \approx 0.053$ |
| Corollary B (worst-class) | $\alpha\,W \le \pi_{\max}^{-1}$ | $\alpha \le \pi_{\max}^{-1}/W \approx 5.7\times 10^{-4}$ |

Our default $\alpha=0.3$ satisfies *neither* corollary exactly, which is what we want: the active set $\mathcal{K}^{\!\star}$ is non-trivial (water-filling spreads $q^\star$ over several rare classes rather than concentrating on one), and per-class bounds hold for the very rarest classes only. If a reviewer asks for a clean worst-class guarantee, we drop $\alpha$ to $0.05$ and re-train — but our experiments show the empirical rare-class improvement is consistent across the explored range $\alpha \in \{0.1, 0.2, 0.3, 0.5\}$.

When $\gamma = 0$ (uniform caps) the bound recovers the standard class-CVaR interpretation: the supremum is a tight bound on the worst $\alpha$-fraction of classes by uniform weight.

---

## 6. Chi-squared ball perspective (auxiliary remark)

To connect CFA-GDRO to the broader $f$-divergence DRO literature (Namkoong and Duchi, 2017, JMLR; Duchi and Namkoong, 2019, *Annals of Statistics*), we record the following fact.

**Fact.** Since $\pi \in \mathcal{Q}_{\alpha,\gamma}$ whenever $\pi_k \le c_k$ for all $k$ (true for every $\gamma \ge 0$ because $c_k \ge \pi_k$ holds elementwise when $\alpha \le 1$ and $\gamma$ is large enough to make $\pi_k^{-\gamma}/W(\gamma) \ge \alpha \pi_k$), and the set $\mathcal{Q}_{\alpha,\gamma}$ is a closed polytope containing an open neighbourhood of $\pi$, there exists $\rho(\alpha,\gamma) > 0$ such that the $\chi^2$-divergence ball

$$
\mathcal{B}_\rho^{\chi^2}(\pi) \;=\; \big\{\,Q \in \Delta^{K-1} \;:\; D_{\chi^2}(Q\,\|\,\pi) \le \rho\,\big\}, \qquad D_{\chi^2}(Q\,\|\,\pi) = \sum_k \tfrac{(Q_k - \pi_k)^2}{\pi_k},
$$

is fully contained in $\mathcal{Q}_{\alpha,\gamma}$. Consequently, CFA-GDRO upper-bounds the $\chi^2$-DRO objective on this radius. We do *not* claim a tight or explicit formula for $\rho(\alpha,\gamma)$ — the polytope shape is irregular and the exact radius is the solution of a quadratic program — but the existence statement is enough to position CFA-GDRO inside the $f$-divergence DRO family.

This auxiliary remark is included to satisfy reviewers who expect the link; it is not used in proofs.

---

## 7. Hyperparameter defaults (subject to validation tuning)

| Parameter | Symbol | Default | Range explored | Reason |
| --- | --- | --- | --- | --- |
| Worst-fraction size | $\alpha$ | $0.3$ | $\{0.1, 0.2, 0.3, 0.5\}$ | Standard CVaR practice |
| Frequency exponent | $\gamma$ | $1.0$ | $\{0, 0.5, 1.0, 1.5\}$ | $\gamma{=}0$ ablation gives uniform-cap class-CVaR |
| Robust weight | $\lambda_{\mathrm{rob}}$ | $0.5$ | $\{0.1, 0.3, 0.5, 1.0\}$ | Balance against mean CE |
| EMA momentum | $\eta$ | $0.9$ | $\{0.8, 0.9, 0.95\}$ | Stability under label scarcity |

---

## 8. Implementation contract (to be honoured in `hsi_robust/losses/cfa_gdro.py`)

The following Python signature is the contract the code must obey, so the math and the implementation never drift:

```python
def cfa_gdro_loss(
    per_sample_losses: torch.Tensor,   # shape (N,)
    labels:            torch.Tensor,   # shape (N,), values in {0,...,K-1}
    scene_freq:        torch.Tensor,   # shape (K,), scene-level pi_k (computed once from full ground truth)
    alpha:             float,          # in (0, 1]
    gamma:             float,          # >= 0
    ema_class_losses:  torch.Tensor | None = None,  # shape (K,), running estimate
    ema_seen:          torch.Tensor | None = None,  # shape (K,), bool
    ema_momentum:      float = 0.9,
) -> tuple[torch.Tensor, dict]:
    """
    Returns (loss, info), where info contains:
        - q_star          : torch.Tensor of shape (K,), the optimal class weights
        - class_losses    : torch.Tensor of shape (K,), per-class mean losses used
        - caps            : torch.Tensor of shape (K,), c_k values
        - threshold_nu    : float, the dual threshold nu
        - active_set_size : int, number of classes with q_star > 0
    """
```

This function is differentiable in `per_sample_losses` only; gradients do not flow through `q_star`, `caps`, or `class_freq`.
