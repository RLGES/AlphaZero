"""AlphaZero loss and gradient steps: cross-entropy to search policy + value MSE.

Weight decay comes from the AdamW optimizer rather than an explicit L2 term.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

import torch
import torch.nn.functional as F
from torch import Tensor

from .network import PolicyValueNet, collate
from .selfplay import ReplayBuffer, Sample


def _losses(
    net: PolicyValueNet, samples: Sequence[Sample], device: str
) -> tuple[Tensor, Tensor, Tensor]:
    batch = collate([s.features for s in samples], net.spec.buckets).to(device)
    logits, values = net(batch)
    mask = torch.tensor([s.mask for s in samples], dtype=torch.bool, device=device)
    target = torch.tensor([s.policy for s in samples], dtype=torch.float32, device=device)
    z = torch.tensor([s.value for s in samples], dtype=torch.float32, device=device)
    log_probs = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    policy_loss = -(target * log_probs.masked_fill(~mask, 0.0)).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, z)
    return policy_loss + value_loss, policy_loss, value_loss


def evaluate_losses(
    net: PolicyValueNet, samples: Sequence[Sample], device: str = "cpu"
) -> dict[str, float]:
    was_training = net.training
    net.eval()
    with torch.no_grad():
        loss, policy_loss, value_loss = _losses(net, samples, device)
    net.train(was_training)
    return {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


def train_steps(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    steps: int,
    batch_size: int,
    rng: Random,
    device: str = "cpu",
    grad_clip: float = 1.0,
) -> dict[str, float]:
    net.train()
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    done = 0
    for _ in range(steps):
        samples = buffer.sample(batch_size, rng)
        if not samples:
            break
        loss, policy_loss, value_loss = _losses(net, samples, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        totals["loss"] += loss.item()
        totals["policy_loss"] += policy_loss.item()
        totals["value_loss"] += value_loss.item()
        done += 1
    return {name: total / max(done, 1) for name, total in totals.items()}
