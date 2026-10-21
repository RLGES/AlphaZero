"""Checkpoints record the language and rule order the policy head was trained on."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.language import Language

from .network import NetworkSpec, PolicyValueNet


def _identity(language: Language) -> tuple[str, list[str]]:
    return type(language).__name__, [str(action) for action in language.action_vocabulary()]


def save_checkpoint(
    path: str | Path,
    net: PolicyValueNet,
    language: Language,
    round_index: int,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    name, actions = _identity(language)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "spec": asdict(net.spec),
        "language": name,
        "actions": actions,
        "round": round_index,
        "metrics": metrics,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint behind


def load_checkpoint(path: str | Path, language: Language) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    name, actions = _identity(language)
    if payload["language"] != name or payload["actions"] != actions:
        raise ValueError(
            f"Checkpoint {path} was trained for {payload['language']} with rules "
            f"{payload['actions']}, but {name} has rules {actions}"
        )
    return payload


def load_network(path: str | Path, language: Language, device: str = "cpu") -> PolicyValueNet:
    payload = load_checkpoint(path, language)
    net = PolicyValueNet(NetworkSpec(**payload["spec"]))
    net.load_state_dict(payload["state_dict"])
    return net.to(device).eval()
