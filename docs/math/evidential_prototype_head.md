# M1.3 — Evidential Prototype Head (EPH)

This note locks the mathematical definition of the second contribution: a Dirichlet-evidential head built on prototype distances, giving closed-form aleatoric and epistemic uncertainty for hyperspectral pixels. All notation is reused unchanged in `paper/sections/method.tex` and in `hsi_robust/losses/evidential.py` + `hsi_robust/models/evidential_head.py`.

The head replaces the heuristic softmax classifier *and* the heuristic uncertainty branch used in many recent HSI papers, unifying both into one principled module.

---

## 1. Setting and notation

Let $f \in \mathbb{R}^{d}$ be the fused per-pixel feature produced by the backbone (spectral SSM + spatial CNN + CP-Graph). Let $\{m_k\}_{k=1}^{K}$ be learnable class prototypes in $\mathbb{R}^{d}$.

We will use:

- $\tilde f = f/\|f\|_2$ and $\tilde m_k = m_k/\|m_k\|_2$ — unit-normalised feature and prototype,
- $\cos(f, m_k) = \tilde f^\top \tilde m_k \in [-1, 1]$ — cosine similarity,
- $\tau > 0$ — a learnable scalar temperature, clipped to $[\tau_{\min}, \tau_{\max}] = [1, 30]$ for numerical stability,
- $\psi(\cdot)$ — the digamma function,
- $\Gamma(\cdot)$, $\log\Gamma(\cdot)$ — gamma and log-gamma functions.

---

## 2. Evidence from prototype similarity

We map prototype similarities to non-negative evidence via softplus:

$$
\boxed{\;e_k \;=\; \mathrm{softplus}\!\big(\tau\,\cos(f, m_k)\big) \;=\; \log\!\big(1 + \exp(\tau\,\tilde f^\top \tilde m_k)\big) \;\ge\; 0.\;}
$$

The softplus choice ensures (i) non-negativity, (ii) smooth gradients everywhere, (iii) approximately linear behaviour for large positive similarity (so the Dirichlet head can concentrate strongly on a confident class). The temperature $\tau$ is shared across all $K$ classes and learned end-to-end.

The Dirichlet concentration vector and its total are

$$
\alpha_k \;=\; e_k + 1, \qquad S \;=\; \sum_{k=1}^{K} \alpha_k \;=\; K + \sum_{k=1}^{K} e_k.
$$

The "$+1$" guarantees $\alpha_k \ge 1$, so the Dirichlet density is well-defined (concentration $>0$) and remains conjugate to a uniform Dirichlet prior with concentration $1$ (Sensoy et al., 2018, NeurIPS).

The predictive class probabilities are the expectations under $\mathrm{Dir}(\alpha)$:

$$
\boxed{\;p_k \;=\; \mathbb{E}_{\mathbf{p}\,\sim\,\mathrm{Dir}(\alpha)}[\mathbf{p}_k] \;=\; \alpha_k / S \;=\; (e_k + 1)/S.\;}
$$

When all $e_k \to 0$, $p \to \mathbf{1}/K$ (maximum-uncertainty prior). When one $e_{k^\star} \gg \max_{k \ne k^\star} e_k$, $p$ concentrates on $k^\star$.

---

## 3. Closed-form uncertainty decomposition

The Dirichlet evidence framework gives a clean per-pixel uncertainty triple:

$$
\boxed{\;u_{\mathrm{vac}}(f) \;=\; \frac{K}{S} \;\in\; (0, 1] \;}
\qquad\text{(vacuity / epistemic)}
$$

$$
\boxed{\;u_{\mathrm{ale}}(f) \;=\; \sum_{k=1}^{K} p_k(1 - p_k) \;\in\; [0, (K-1)/K] \;}
\qquad\text{(aleatoric)}
$$

$$
u_{\mathrm{tot}}(f) \;=\; 1 - \max_k p_k \;\in\; [0, (K-1)/K].
\qquad\text{(total predictive)}
$$

**Interpretation.** Vacuity is high when total evidence $\sum_k e_k$ is small — i.e. when the feature is not close to any prototype. Aleatoric is high when evidence is large but spread across multiple classes — i.e. when the feature is close to several prototypes simultaneously (boundary pixels, mixed pixels). The two uncertainty signals are *not* redundant, and we report both in classification maps (vacuity map and aleatoric map).

**Mixed/boundary pixels are aleatoric-high, vacuity-low.** Outlier/out-of-distribution pixels are vacuity-high. This decomposition was not available with the heuristic $1 - \max\,\mathrm{softmax}$ uncertainty.

---

## 4. Training loss

The evidential head is trained with a Bayes-risk term plus a KL regulariser on wrong-class evidence, annealed in early epochs (Sensoy et al., 2018).

### 4.1 Bayes-risk (squared-error form)

Let $\mathbf{y} \in \{0, 1\}^{K}$ be the one-hot label. Under the Dirichlet posterior, the expected squared error has a closed form:

$$
\boxed{\;\mathcal{L}_{\mathrm{lik}}(f, y) \;=\; \mathbb{E}_{\mathbf{p}\,\sim\,\mathrm{Dir}(\alpha)} \Big[\,\|\mathbf{y} - \mathbf{p}\|_2^2\,\Big]
\;=\; \sum_{k=1}^{K} \Big[\,(\,y_k - p_k\,)^2 \;+\; \frac{p_k(1 - p_k)}{S + 1}\,\Big].\;}
$$

This is the form used by Sensoy et al.; it is numerically more stable than the log-likelihood form because it does not call $\psi$ or $\log\Gamma$. The first term is the standard squared error of the mean prediction; the second term is the variance penalty of the Dirichlet posterior, which rewards the model for concentrating its evidence (large $S$) only when the prediction is correct.

### 4.2 KL regulariser on wrong-class evidence

To prevent the model from inflating evidence on wrong classes, we add a KL term that pulls the *wrong-class* Dirichlet toward the uniform prior. Define the wrong-class Dirichlet concentration:

$$
\tilde\alpha \;=\; \mathbf{y} + (1 - \mathbf{y}) \odot \alpha,
\qquad
\tilde\alpha_k \;=\; \begin{cases} 1 & \text{if } y_k = 1, \\ \alpha_k & \text{otherwise.} \end{cases}
$$

In words: keep wrong-class concentrations $\alpha_k$ as they are; reset the true-class concentration to $1$. Then penalise the divergence from the all-ones Dirichlet (uniform on the simplex):

$$
\boxed{\;\mathcal{L}_{\mathrm{KL}}(f, y) \;=\; \mathrm{KL}\!\Big(\,\mathrm{Dir}(\tilde\alpha) \,\big\|\, \mathrm{Dir}(\mathbf{1})\,\Big).\;}
$$

The KL between two Dirichlets has a closed form:

$$
\mathrm{KL}\!\big(\mathrm{Dir}(a)\,\|\,\mathrm{Dir}(b)\big) \;=\; \log\Gamma\!\Big(\!\textstyle\sum_k a_k\Big) \;-\; \sum_k \log\Gamma(a_k) \;-\; \log\Gamma\!\Big(\!\textstyle\sum_k b_k\Big) \;+\; \sum_k \log\Gamma(b_k) \;+\; \sum_k (a_k - b_k)\big(\psi(a_k) - \psi(\textstyle\sum_j a_j)\big).
$$

With $b = \mathbf{1}$ this simplifies to

$$
\mathrm{KL}\!\big(\mathrm{Dir}(\tilde\alpha)\,\|\,\mathrm{Dir}(\mathbf{1})\big)
\;=\;
\log\Gamma\!\Big(\!\textstyle\sum_k \tilde\alpha_k\Big) - \sum_k \log\Gamma(\tilde\alpha_k) - \log\Gamma(K) + \sum_k (\tilde\alpha_k - 1)\big(\psi(\tilde\alpha_k) - \psi(\textstyle\sum_j \tilde\alpha_j)\big).
$$

This is what `hsi_robust/losses/evidential.py` will compute (PyTorch provides `torch.lgamma` and `torch.digamma`).

### 4.3 Annealed combination

Define an annealing weight that ramps from $0$ to $1$ over the first $T_{\mathrm{anneal}}$ epochs:

$$
\lambda_t \;=\; \min\!\Big(1,\;\frac{t}{T_{\mathrm{anneal}}}\Big), \qquad t \in \{1,\dots,T\}.
$$

The per-pixel evidential loss is

$$
\boxed{\;\mathcal{L}_{\mathrm{EPH}}^{(t)}(f, y) \;=\; \mathcal{L}_{\mathrm{lik}}(f, y) \;+\; \lambda_t\,\mathcal{L}_{\mathrm{KL}}(f, y).\;}
$$

Early epochs let the model build evidence freely; later epochs progressively penalise wrong-class evidence. Default $T_{\mathrm{anneal}} = 10$ epochs.

---

## 5. Compatibility with CFA-GDRO

CFA-GDRO consumes a per-sample loss $\ell_\theta(x_i, y_i)$. We use the **standard cross-entropy** $\ell^{\mathrm{CE}}_i = -\log p_\theta(y_i \mid x_i)$ as that per-sample loss, where $p_\theta$ is the softmax of the EPH logits $\tau\,\cos(f, m_k)$. This choice is locked by decision D-08 in `roadmap.md` and validated by the Phase 2E smoke debug:

- the EPH Bayes-risk has a vanishing gradient at the uniform-prediction saddle (gradient on $\alpha$ at uniform is $O(1/K)$ compared with $O(1)$ for CE), so $\mathcal{L}_{\mathrm{EPH}}$ alone gets stuck and the model never escapes uniform predictions;
- using $\ell^{\mathrm{CE}}_i$ as the CFA-GDRO input lets the *same* model (same prototypes, same temperature) be re-weighted for rare-class robustness without the saddle-point pathology;
- the EPH calibration claim is still consistent end-to-end because the CE logits are derived from the EPH cos×$\tau$ pathway -- argmax of CE softmax equals argmax of EPH predictive probabilities $p_k$ when both share the temperature.

In the final training objective we therefore follow Eq. (4) of `cfa_gdro.md` §3 *verbatim*:

$$
\mathcal{L}_{\mathrm{total}}(\theta)
\;=\;
\underbrace{\overline{\ell^{\mathrm{CE}}}}_{\text{mean CE}}
\;+\;
\lambda_{\mathrm{rob}}\,\mathcal{L}_{\mathrm{CFA\text{-}GDRO}}^{\mathrm{CE}}(\theta)
\;+\;
\lambda_{\mathrm{evi}}\,\mathcal{L}_{\mathrm{EPH}}(\theta)
\;+\;
\lambda_{\mathrm{graph}}\,\mathcal{L}_{\mathrm{CP\text{-}graph}}(\theta).
$$

The EPH loss $\mathcal{L}_{\mathrm{EPH}}$ here is the **mean per-sample EPH** (Eq. (14)); it acts as an explicit calibration regulariser with weight $\lambda_{\mathrm{evi}}$ (default $1.0$). The "Reliable" claim is preserved by the chain *CE input $\rightarrow$ CFA-GDRO water-filling $\rightarrow$ rare-class upper bound* (`cfa_gdro.md` §5), and the "Calibrated" claim is preserved by the additive $\mathcal{L}_{\mathrm{EPH}}$ term, which pushes wrong-class evidence toward the uniform Dirichlet prior.

---

## 6. Calibration claim and verification protocol

The evidential head is intended to lower expected calibration error (ECE) compared with a softmax classifier of the same backbone. To make this falsifiable, we will report:

- **ECE-15** with 15 confidence bins, the standard for HSI calibration in 2024–2026 papers.
- **Reliability diagrams** per dataset (figure in experiments section).
- **Vacuity histograms** stratified by *correct* vs. *incorrect* predictions — correct predictions should have low vacuity, incorrect predictions should have high vacuity.
- **OOD-like stress test**: at test time, randomly mask 20% of input bands and check that the average vacuity rises, while a softmax head produces over-confident predictions on the corrupted input. (This experiment lives in `scripts/run_robustness.py` in Phase 7.)

If the evidential head does *not* lower ECE on at least three of the four datasets, the calibration claim in the abstract will be weakened to "uncertainty-aware" rather than "calibrated", and the OOD stress test will not be included.

---

## 7. Hyperparameter defaults

| Parameter | Symbol | Default | Range explored |
| --- | --- | --- | --- |
| Temperature clip range | $[\tau_{\min}, \tau_{\max}]$ | $[1, 30]$ | fixed |
| Annealing epochs | $T_{\mathrm{anneal}}$ | $10$ | $\{5, 10, 20\}$ |
| EPH weight in total loss | $\lambda_{\mathrm{evi}}$ | $1.0$ | $\{0.5, 1.0\}$ (when not folded into CFA-GDRO) |
| Prototype init scale | — | $0.02$ | fixed |

---

## 8. Implementation contract

Two files will own the math above:

```python
# hsi_robust/models/evidential_head.py
class EvidentialPrototypeHead(nn.Module):
    """
    Inputs:
        f : (N, d) fused features
    Outputs:
        evidence  : (N, K)
        alpha     : (N, K)
        probs     : (N, K)   p_k = alpha_k / S
        vacuity   : (N,)     K / S
        aleatoric : (N,)     sum_k p_k (1 - p_k)
    """
```

```python
# hsi_robust/losses/evidential.py
def evidential_loss(
    alpha:         torch.Tensor,   # (N, K)
    labels:        torch.Tensor,   # (N,)
    kl_weight:     float,          # lambda_t (current annealed value)
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Returns (per_sample_loss, mean_loss, info).
    per_sample_loss : shape (N,)  -> fed into CFA-GDRO solver
    mean_loss       : scalar      -> the "mean EPH" term in the total loss
    info            : dict with components ('lik', 'kl', 'mean_vacuity', ...)
    """
```

This is the exact signature we will implement in Phase 2D.
