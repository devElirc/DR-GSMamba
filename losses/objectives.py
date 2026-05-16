from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RobustRiskTracker:
    """EMA tracker for class risks used by the GroupDRO-style robust loss."""

    num_classes: int
    momentum: float = 0.9
    device: torch.device | None = None

    def __post_init__(self) -> None:
        device = self.device if self.device is not None else torch.device("cpu")
        self.risks = torch.zeros(self.num_classes, device=device)
        self.seen = torch.zeros(self.num_classes, dtype=torch.bool, device=device)

    def update(self, batch_risks: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        present = present.to(self.risks.device)
        batch_risks = batch_risks.to(self.risks.device)
        old = self.risks[present]
        new = batch_risks[present]
        initialized = self.seen[present]
        self.risks[present] = torch.where(initialized, self.momentum * old + (1.0 - self.momentum) * new, new)
        self.seen[present] = True
        return self.risks


def cvar_loss(sample_losses: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    k = max(1, int(alpha * sample_losses.numel()))
    return torch.topk(sample_losses, k=k).values.mean()


def class_dro_loss(sample_losses: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float = 0.3) -> torch.Tensor:
    """Class-level CVaR: optimize the worst class risks in the current mini-batch."""
    class_risks = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            class_risks.append(sample_losses[mask].mean())
    if not class_risks:
        return sample_losses.mean()
    risks = torch.stack(class_risks)
    k = max(1, int(alpha * risks.numel()))
    return torch.topk(risks, k=k).values.mean()


def tracked_class_dro_loss(
    sample_losses: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 0.3,
    tracker: RobustRiskTracker | None = None,
) -> torch.Tensor:
    """Class-level CVaR with optional EMA risk memory.

    Mini-batch class DRO can be unstable under label scarcity because many
    classes are absent from a batch. The tracker keeps a running class-risk
    estimate and optimizes the current worst classes among those observed so far.
    """
    batch_risks = sample_losses.new_zeros(num_classes)
    present = torch.zeros(num_classes, dtype=torch.bool, device=sample_losses.device)
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            batch_risks[cls] = sample_losses[mask].mean()
            present[cls] = True
    if tracker is None:
        risks = batch_risks[present]
        if risks.numel() == 0:
            return sample_losses.mean()
        k = max(1, int(alpha * risks.numel()))
        return torch.topk(risks, k=k).values.mean()

    with torch.no_grad():
        tracker.update(batch_risks.detach(), present.detach())
        seen = tracker.seen.detach().clone().to(sample_losses.device)
        tracked = tracker.risks.detach().clone().to(sample_losses.device)
        seen_classes = torch.nonzero(seen, as_tuple=False).flatten()
        tracked_risks = tracked[seen_classes]
        if tracked_risks.numel() > 0:
            k = max(1, int(alpha * tracked_risks.numel()))
            high_risk_classes = seen_classes[torch.topk(tracked_risks, k=k).indices]
        else:
            high_risk_classes = torch.empty(0, dtype=torch.long, device=sample_losses.device)

    selected_classes = torch.zeros(num_classes, dtype=torch.bool, device=sample_losses.device)
    selected_classes[high_risk_classes] = True
    selected_sample_mask = selected_classes[labels]
    if selected_sample_mask.any():
        return sample_losses[selected_sample_mask].mean()

    risks = batch_risks[present]
    if risks.numel() == 0:
        return sample_losses.mean()
    k = max(1, int(alpha * risks.numel()))
    return torch.topk(risks, k=k).values.mean()


def prototype_consistency_loss(linear_logits: torch.Tensor, proto_logits: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(linear_logits, dim=-1)
    q = F.softmax(proto_logits.detach(), dim=-1)
    return F.kl_div(p, q, reduction="batchmean")


def supervised_prototype_loss(features: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    target_proto = prototypes[labels]
    compact = 1.0 - (features * target_proto).sum(dim=-1)
    return compact.mean()


def uncertainty_weighted_ce(sample_losses: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    uncertainty = uncertainty.clamp_min(1e-4)
    return (torch.exp(-uncertainty) * sample_losses + uncertainty).mean()


def graph_smoothness_loss(node_logits: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    node_probs = F.softmax(node_logits, dim=-1)
    smooth = torch.bmm(adj, node_probs)
    return F.mse_loss(node_probs, smooth)


def dr_gsmamba_loss(
    outputs: dict,
    labels: torch.Tensor,
    cfg: dict,
    tracker: RobustRiskTracker | None = None,
) -> tuple[torch.Tensor, dict]:
    weights = cfg.get("loss", {})
    sample_losses = F.cross_entropy(outputs["logits"], labels, reduction="none")
    ce = sample_losses.mean()
    alpha = float(weights.get("robust_alpha", 0.3))
    sample_robust = cvar_loss(sample_losses, alpha=alpha)
    class_robust = tracked_class_dro_loss(sample_losses, labels, outputs["logits"].shape[-1], alpha=alpha, tracker=tracker)
    robust = 0.5 * sample_robust + 0.5 * class_robust
    proto = prototype_consistency_loss(outputs["linear_logits"], outputs["proto_logits"])
    proto_supervised = supervised_prototype_loss(outputs["features"], outputs["prototypes"], labels)
    uncertainty = uncertainty_weighted_ce(sample_losses, outputs["learned_uncertainty"])
    smooth = graph_smoothness_loss(outputs["node_logits"], outputs["adjacency"])
    loss = (
        ce
        + weights.get("robust_weight", 0.3) * robust
        + weights.get("prototype_weight", 0.1) * proto
        + weights.get("prototype_supervised_weight", 0.05) * proto_supervised
        + weights.get("uncertainty_weight", 0.05) * uncertainty
        + weights.get("graph_smooth_weight", 0.02) * smooth
    )
    return loss, {
        "ce": float(ce.detach().cpu()),
        "robust": float(robust.detach().cpu()),
        "prototype": float(proto.detach().cpu()),
        "prototype_supervised": float(proto_supervised.detach().cpu()),
        "uncertainty": float(uncertainty.detach().cpu()),
        "graph_smooth": float(smooth.detach().cpu()),
    }
