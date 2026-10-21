"""AlphaZero outer loop (PPT slide 14): self-play -> train -> publish, per round.

    python -m src.engines.alphazero.train --suite arithmetic --rounds 10 --out checkpoints/arith

Each round writes ``theta_kNNNN.pt``, ``latest.pt`` (with optimizer state), the
replay buffer, and one JSON line in ``metrics.jsonl``. ``--resume`` continues
from ``latest.pt``. ``play_chunk`` is a plain picklable function so local
process pools and Modal workers run exactly the same self-play code.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Any

import torch

from benchmarks.generators import GENERATOR_SUITES, generate
from benchmarks.registry import load_suite

from .checkpoint import load_checkpoint, save_checkpoint
from .config import AlphaZeroConfig
from .engine import AlphaZeroEngine
from .inference import NetworkEvaluator
from .network import NetworkSpec, PolicyValueNet
from .selfplay import ReplayBuffer, Sample, SelfPlayConfig, play_episode
from .training import train_steps


@dataclass
class TrainConfig:
    suite: str
    rounds: int = 10
    games_per_round: int = 64
    train_steps: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    buffer_capacity: int = 50_000
    seed: int = 0
    out_dir: str = "checkpoints"
    workers: int = 1
    eval_iterations: int = 100
    depth: tuple[int, int] = (2, 5)
    device: str = "cpu"
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    spec: NetworkSpec | None = None


# A chunk of games: (suite, network spec, CPU state_dict, terms, self-play config, seed).
Chunk = tuple[str, dict, dict, list, dict, int]
PlayFn = Callable[[list[Chunk]], list[list[Sample]]]


def play_chunk(chunk: Chunk) -> list[Sample]:
    suite, spec, state_dict, terms, selfplay, seed = chunk
    torch.set_num_threads(1)
    net = PolicyValueNet(NetworkSpec(**spec))
    net.load_state_dict(state_dict)
    evaluator = NetworkEvaluator(net)
    language = load_suite(suite).language
    config = SelfPlayConfig(**selfplay)
    rng = Random(seed)
    samples: list[Sample] = []
    for term in terms:
        samples += play_episode(language, term, evaluator, config, rng)
    return samples


def play_locally(workers: int) -> PlayFn:
    def play(chunks: list[Chunk]) -> list[list[Sample]]:
        if workers <= 1:
            return [play_chunk(chunk) for chunk in chunks]
        from multiprocessing import get_context

        with get_context("spawn").Pool(workers) as pool:
            return pool.map(play_chunk, chunks)

    return play


def evaluate_held_out(suite: str, net: PolicyValueNet, iterations: int, device: str) -> dict:
    loaded = load_suite(suite)
    engine = AlphaZeroEngine(AlphaZeroConfig(iterations=iterations, network=net, device=device))
    finals = [engine.run(loaded.language, b.term).final_cost for b in loaded.benchmarks]
    initials = [loaded.language.cost(b.term) for b in loaded.benchmarks]
    improvement = [(i - f) / i if i else 0.0 for i, f in zip(initials, finals)]
    return {
        "eval_mean_final_cost": sum(finals) / len(finals),
        "eval_mean_improvement": sum(improvement) / len(improvement),
    }


class Trainer:
    def __init__(self, cfg: TrainConfig, resume: bool = False):
        if cfg.suite not in GENERATOR_SUITES:
            raise ValueError(f"Training needs a term generator; choose from {GENERATOR_SUITES}")
        self.cfg = cfg
        self.out = Path(cfg.out_dir)
        self.language = load_suite(cfg.suite).language
        width = len(self.language.action_vocabulary())
        self.round = 0
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        payload = load_checkpoint(self.out / "latest.pt", self.language) if resume else None
        spec = NetworkSpec(**payload["spec"]) if payload else cfg.spec or NetworkSpec(width)
        self.net = PolicyValueNet(spec).to(cfg.device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        if payload:
            self.net.load_state_dict(payload["state_dict"])
            if payload["optimizer"] is not None:
                self.optimizer.load_state_dict(payload["optimizer"])
            self.round = payload["round"] + 1
            buffer_path = self.out / "buffer.pkl"
            if buffer_path.exists():
                self.buffer.extend(pickle.loads(buffer_path.read_bytes()))

    def chunks(self, count: int) -> list[Chunk]:
        cfg = self.cfg
        seed = cfg.seed * 1_000_003 + self.round
        terms = generate(cfg.suite, Random(seed), cfg.games_per_round, cfg.depth)
        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        spec = asdict(self.net.spec)
        selfplay = asdict(cfg.selfplay)
        groups = [terms[i::count] for i in range(count)]
        return [
            (cfg.suite, spec, state, group, selfplay, seed + 7919 * (i + 1))
            for i, group in enumerate(groups)
            if group
        ]

    def run_round(self, play: PlayFn, chunk_count: int) -> dict[str, Any]:
        cfg = self.cfg
        started = time.perf_counter()
        results = play(self.chunks(chunk_count))
        samples = [sample for chunk in results for sample in chunk]
        self.buffer.extend(samples)
        selfplay_time = time.perf_counter() - started
        started = time.perf_counter()
        losses = train_steps(
            self.net,
            self.optimizer,
            self.buffer,
            cfg.train_steps,
            cfg.batch_size,
            Random(cfg.seed + self.round),
            cfg.device,
        )
        train_time = time.perf_counter() - started
        metrics = {
            "round": self.round,
            **losses,
            "samples": len(samples),
            "buffer": len(self.buffer),
            "mean_target_value": sum(s.value for s in samples) / max(len(samples), 1),
            **evaluate_held_out(cfg.suite, self.net, cfg.eval_iterations, cfg.device),
            "selfplay_seconds": round(selfplay_time, 2),
            "train_seconds": round(train_time, 2),
        }
        self.save(metrics)
        self.round += 1
        return metrics

    def save(self, metrics: dict[str, Any]) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        args = (self.net, self.language, self.round, metrics)
        save_checkpoint(self.out / f"theta_k{self.round:04d}.pt", *args)
        save_checkpoint(self.out / "latest.pt", *args, optimizer=self.optimizer)
        tmp = self.out / "buffer.pkl.tmp"
        tmp.write_bytes(pickle.dumps(self.buffer.items()))
        tmp.replace(self.out / "buffer.pkl")
        with (self.out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")


def outer_loop(
    cfg: TrainConfig,
    resume: bool = False,
    play: PlayFn | None = None,
    chunk_count: int | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    trainer = Trainer(cfg, resume)
    play = play or play_locally(cfg.workers)
    while trainer.round < cfg.rounds:
        metrics = trainer.run_round(play, chunk_count or max(cfg.workers, 1))
        log(
            f"round {metrics['round']}: loss={metrics['loss']:.4f} "
            f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} "
            f"samples={metrics['samples']} eval_cost={metrics['eval_mean_final_cost']:.2f} "
            f"({metrics['selfplay_seconds']}s play, {metrics['train_seconds']}s train)"
        )
    return trainer.out / "latest.pt"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AlphaZero rewrite engine.")
    parser.add_argument("--suite", choices=GENERATOR_SUITES, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--games", type=int, default=64, help="Self-play games per round")
    parser.add_argument("--steps", type=int, default=200, help="Gradient steps per round")
    parser.add_argument("--sims", type=int, default=64, help="Search simulations per move")
    parser.add_argument("--max-moves", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--eval-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/run")
    parser.add_argument("--resume", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    width = len(load_suite(args.suite).language.action_vocabulary())
    return TrainConfig(
        suite=args.suite,
        rounds=args.rounds,
        games_per_round=args.games,
        train_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out,
        workers=args.workers,
        eval_iterations=args.eval_iterations,
        device=args.device,
        selfplay=SelfPlayConfig(simulations_per_move=args.sims, max_moves=args.max_moves),
        spec=NetworkSpec(width, hidden=args.hidden, layers=args.layers),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resume and not (Path(args.out) / "latest.pt").exists():
        parser.error(f"--resume needs {Path(args.out) / 'latest.pt'}")
    latest = outer_loop(config_from_args(args), resume=args.resume)
    print(f"Latest checkpoint: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
