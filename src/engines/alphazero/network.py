"""Tree message-passing policy/value network in plain PyTorch.

Labels are hashed into a fixed number of buckets, so checkpoints need no
vocabulary file. Each layer mixes a node with its children (summed, with a
child-position embedding) and its parent, then a residual LayerNorm update.
The readout concatenates the root embedding with a mean pool over the graph.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .features import GraphFeatures


def label_id(label: str, buckets: int) -> int:
    return zlib.crc32(label.encode()) % buckets


@dataclass(frozen=True)
class NetworkSpec:
    num_actions: int
    buckets: int = 1024
    hidden: int = 128
    layers: int = 4
    max_children: int = 8


@dataclass
class GraphBatch:
    labels: Tensor  # [N]
    edge_index: Tensor  # [2, E] parent, child
    edge_pos: Tensor  # [E]
    graph_index: Tensor  # [N]
    roots: Tensor  # [B]
    num_graphs: int

    def to(self, device: str | torch.device) -> GraphBatch:
        return GraphBatch(
            self.labels.to(device),
            self.edge_index.to(device),
            self.edge_pos.to(device),
            self.graph_index.to(device),
            self.roots.to(device),
            self.num_graphs,
        )


def collate(features: Sequence[GraphFeatures], buckets: int) -> GraphBatch:
    labels: list[int] = []
    parents: list[int] = []
    children: list[int] = []
    positions: list[int] = []
    graph_index: list[int] = []
    roots: list[int] = []
    for graph, item in enumerate(features):
        offset = len(labels)
        labels.extend(label_id(label, buckets) for label in item.labels)
        graph_index.extend([graph] * len(item.labels))
        roots.append(offset + item.root)
        for parent, child, position in item.edges:
            parents.append(offset + parent)
            children.append(offset + child)
            positions.append(position)
    return GraphBatch(
        torch.tensor(labels, dtype=torch.long),
        torch.tensor([parents, children], dtype=torch.long).reshape(2, -1),
        torch.tensor(positions, dtype=torch.long),
        torch.tensor(graph_index, dtype=torch.long),
        torch.tensor(roots, dtype=torch.long),
        len(features),
    )


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class _TreeLayer(nn.Module):
    def __init__(self, hidden: int, max_children: int):
        super().__init__()
        self.max_children = max_children
        self.position = nn.Embedding(max_children, hidden)
        self.update = _mlp(3 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, batch: GraphBatch) -> Tensor:
        parent, child = batch.edge_index
        position = batch.edge_pos.clamp(max=self.max_children - 1)
        up = torch.zeros_like(h).index_add_(0, parent, h[child] + self.position(position))
        down = torch.zeros_like(h).index_add_(0, child, h[parent])
        return self.norm(h + self.update(torch.cat((h, up, down), dim=-1)))


class PolicyValueNet(nn.Module):
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.buckets, spec.hidden)
        self.layers = nn.ModuleList(
            _TreeLayer(spec.hidden, spec.max_children) for _ in range(spec.layers)
        )
        self.policy = _mlp(2 * spec.hidden, spec.hidden, spec.num_actions)
        self.value = _mlp(2 * spec.hidden, spec.hidden, 1)

    def forward(self, batch: GraphBatch) -> tuple[Tensor, Tensor]:
        h = self.embed(batch.labels)
        for layer in self.layers:
            h = layer(h, batch)
        pooled = torch.zeros(batch.num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch.graph_index, h)
        counts = torch.bincount(batch.graph_index, minlength=batch.num_graphs).clamp(min=1)
        readout = torch.cat((h[batch.roots], pooled / counts.unsqueeze(1)), dim=-1)
        return self.policy(readout), torch.sigmoid(self.value(readout)).squeeze(-1)
