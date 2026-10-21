from collections.abc import Sequence
from typing import TypeVar

import torch

from src.language import Language

from .network import PolicyValueNet, collate

T = TypeVar("T")
A = TypeVar("A")


class NetworkEvaluator:
    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        self.net = net
        self.device = device

    def evaluate(
        self, language: Language[T, A], terms: Sequence[T]
    ) -> list[tuple[Sequence[float], float]]:
        width = len(language.action_vocabulary())
        if width != self.net.spec.num_actions:
            raise ValueError(
                f"Network scores {self.net.spec.num_actions} actions but "
                f"{type(language).__name__} has {width}"
            )
        batch = collate([language.features(term) for term in terms], self.net.spec.buckets)
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            logits, values = self.net(batch.to(self.device))
        self.net.train(was_training)
        priors = torch.softmax(logits, dim=-1).cpu().tolist()
        return list(zip(priors, values.cpu().tolist()))
