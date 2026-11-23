"""Train AlphaZero-GEB: AlphaZero that builds e-graphs (guided saturation).

    RLGES_ASM_LEARNED=benchmarks/asm/learned_rules_qwen_all.json PYTHONPATH=. \
      python scripts/geb_train.py --out checkpoints/geb_v1 \
        --init checkpoints/asm_v4_llm_s0/best.pt --workers 2

Each round: self-play episodes on generated training terms (PUCT over "enable one
learned rule, then one iteration of built-in + enabled rules"), train the policy on
visit counts and the value on the share of the current extracted cost still saved
by the end of the episode, then score 40 fixed validation terms (never the final
test terms) against egglog's default schedule under the same e-node limit. The
best round is kept as best.pt. Resumable from latest.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random

import torch

from benchmarks.asm.geb import Agent, EGraphEnv, SearchConfig, build, egglog_schedule
from benchmarks.asm.language import AsmLanguage
from benchmarks.generators import generate
from src.engines.alphazero.checkpoint import load_checkpoint, save_checkpoint
from src.engines.alphazero.network import NetworkSpec, PolicyValueNet
from src.engines.alphazero.selfplay import ReplayBuffer, Sample
from src.engines.alphazero.training import train_steps

VALIDATION_SEED = 515_151  # not the test seed, not the small training seeds
_CACHE: dict = {}


def _env(args) -> EGraphEnv:
    return EGraphEnv(AsmLanguage(), node_limit=args["node_limit"], max_steps=args["steps"],
                     kind="enable")


def _net(path: str):
    stamp = os.path.getmtime(path)
    if _CACHE.get("key") != (path, stamp):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        net = PolicyValueNet(NetworkSpec(**payload["spec"]))
        net.load_state_dict(payload["state_dict"])
        torch.set_num_threads(1)
        _CACHE["key"], _CACHE["net"] = (path, stamp), net.eval()
    return _CACHE["net"]


def _guard() -> None:
    import resource

    limit = 3 * 2**30
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _play(job) -> list:
    term, seed, args = job
    language = AsmLanguage()
    env = _env(args)
    agent = Agent(language, _net(args["current"]))
    cfg = SearchConfig(simulations=args["sims"], mode="az", dirichlet_epsilon=0.25)
    try:
        episode = build(term, env, agent, cfg, Random(seed), temperature_moves=2, record=True)
    except Exception:
        return []
    mask = tuple(i in set(env.valid) for i in range(len(env.names)))
    return [
        Sample(language.features(t), mask, tuple(policy), float(value))
        for t, policy, value in episode.samples
    ]


def _score(job) -> tuple[float, float, int, int]:
    term, args = job
    language = AsmLanguage()
    env = _env(args)
    egg = egglog_schedule(term, env)
    agent = Agent(language, _net(args["current"]))
    cfg = SearchConfig(simulations=args["sims"], mode="az")
    try:
        ep = build(term, env, agent, cfg, Random(0))
        az_cost, az_size = ep.best_cost, ep.final_size
    except Exception:
        az_cost, az_size = egg.initial_cost, 0
    c0 = egg.initial_cost
    return (100 * (c0 - az_cost) / c0, 100 * (c0 - egg.best_cost) / c0, az_size, egg.final_size)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--sims", type=int, default=16)
    p.add_argument("--node-limit", type=int, default=1000)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-steps", type=int, default=150)
    p.add_argument("--validation", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    language = AsmLanguage()
    start_path = out / "latest.pt" if (out / "latest.pt").exists() else Path(a.init)
    payload = load_checkpoint(start_path, language)
    net = PolicyValueNet(NetworkSpec(**payload["spec"]))
    net.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    first = payload["round"] + 1 if start_path.name == "latest.pt" else 0
    best = payload["metrics"].get("best", -1.0) if first else -1.0
    stale = payload["metrics"].get("stale", 0) if first else 0
    buffer = ReplayBuffer(20_000)
    current = out / "current.pt"
    args = {"node_limit": a.node_limit, "steps": a.steps, "sims": a.sims, "current": str(current)}
    validation = generate("asm", Random(VALIDATION_SEED), a.validation, (4, 6), split="train")
    log = (out / "train.log").open("a")

    def say(text: str) -> None:
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    say(f"=== {time.strftime('%F %T')} AZ-GEB from {start_path}, round {first}, "
        f"node limit {a.node_limit}, steps {a.steps}, sims {a.sims}")
    with ProcessPoolExecutor(a.workers, initializer=_guard) as pool:
        for rnd in range(first, a.rounds):
            t0 = time.time()
            save_checkpoint(current, net, language, rnd, {})
            terms = generate("asm", Random(10_000 + rnd), a.episodes, (3, 6))
            jobs = [(t, rnd * 1000 + i, args) for i, t in enumerate(terms)]
            samples = [s for batch in pool.map(_play, jobs) for s in batch]
            buffer.extend(samples)
            losses = train_steps(net, optimizer, buffer, a.train_steps, 64, Random(rnd))
            save_checkpoint(current, net, language, rnd, {})
            scores = list(pool.map(_score, [(t, args) for t in validation]))
            az = sum(s[0] for s in scores) / len(scores)
            egg = sum(s[1] for s in scores) / len(scores)
            az_nodes = sum(s[2] for s in scores) / len(scores)
            egg_nodes = sum(s[3] for s in scores) / len(scores)
            improved = az > best + 1e-6
            if improved:
                best, stale = az, 0
                save_checkpoint(out / "best.pt", net, language, rnd, {"score": az})
            else:
                stale += 1
            metrics = {"round": rnd, "samples": len(samples), **losses, "az": az, "egglog": egg,
                       "az_nodes": az_nodes, "egglog_nodes": egg_nodes, "best": best,
                       "stale": stale, "seconds": time.time() - t0}
            save_checkpoint(out / "latest.pt", net, language, rnd, metrics, optimizer)
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            say(f"round {rnd}: samples={len(samples)} loss={losses['loss']:.3f} "
                f"az={az:.2f} egglog={egg:.2f} nodes az={az_nodes:.0f} egglog={egg_nodes:.0f} "
                f"best={best:.2f} ({time.time() - t0:.0f}s)")
            if stale >= a.patience:
                say(f"early stop: no better validation score for {a.patience} rounds")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
