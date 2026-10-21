import zlib

import pytest

from src.engines.alphazero.features import GraphFeatures

torch = pytest.importorskip("torch")

from src.engines.alphazero.network import (  # noqa: E402
    NetworkSpec,
    PolicyValueNet,
    collate,
    label_id,
)

A = GraphFeatures(("add", "x", "y"), ((0, 1, 0), (0, 2, 1)))
B = GraphFeatures(("x",), ())


def test_collate_offsets_edges_and_roots():
    batch = collate([A, B], buckets=64)
    assert batch.num_graphs == 2
    assert batch.roots.tolist() == [0, 3]
    assert batch.edge_index.tolist() == [[0, 0], [1, 2]]
    assert batch.edge_pos.tolist() == [0, 1]
    assert batch.graph_index.tolist() == [0, 0, 0, 1]
    assert batch.labels.tolist() == [label_id(s, 64) for s in ("add", "x", "y", "x")]


def test_forward_shapes_and_ranges():
    net = PolicyValueNet(NetworkSpec(num_actions=6, buckets=64, hidden=32, layers=2))
    logits, value = net(collate([A, B], 64))
    assert logits.shape == (2, 6)
    assert value.shape == (2,)
    assert ((value >= 0) & (value <= 1)).all()


def test_wide_nodes_clamp_positions():
    wide = GraphFeatures(("f",) + ("x",) * 12, tuple((0, i + 1, i) for i in range(12)))
    net = PolicyValueNet(NetworkSpec(num_actions=2, buckets=64, hidden=16, layers=1))
    logits, _ = net(collate([wide], 64))
    assert torch.isfinite(logits).all()


def test_label_id_stable():
    assert label_id("add", 1024) == zlib.crc32(b"add") % 1024
