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
import multiprocessing
import os
import sys
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
                     kind=args["kind"])


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

    limit = 6 * 2**30  # torch reserves a lot of address space
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
    # "all rules, capped" baseline: always the macro action (FOLD index) in mixed mode.
    env.reset(term)
    capped = env.extract()[1]
    if env.kind == "mixed":
        while not env.terminal():
            before = env.size()
            env.apply(env.noop)
            capped = min(capped, env.extract()[1])
            if env.size() == before:
                break
    return (100 * (c0 - az_cost) / c0, 100 * (c0 - egg.best_cost) / c0, az_size, egg.final_size,
            100 * (c0 - capped) / c0)


_POOL: dict = {}


def _train_pool():
    """Training-split building blocks (never held-out/stress/test terms)."""
    if "pool" not in _POOL:
        _POOL["pool"] = generate("asm", Random(7_777), 3000, (3, 6), split="train")
        sensitive = Path("benchmarks/asm/testsets/train_sensitive.jsonl")
        if sensitive.exists():
            from benchmarks.asm.term import from_sexpr

            _POOL["sensitive"] = [from_sexpr(json.loads(x)["sexpr"]) for x in open(sensitive)]
        else:
            _POOL["sensitive"] = []
    return _POOL["pool"], _POOL["sensitive"]


def _suite_style_terms(rng: Random, n: int) -> list:
    """Training terms built like the test suites T3/T4/T5/T7, from training-split
    material only (scripts/build_testsets.py builds the test versions from test data)."""
    from benchmarks.asm.term import Const, Expr, Var

    sys.path.insert(0, "scripts")
    from testset_stats import AC_OPS, bucket

    language = AsmLanguage()
    pool, sensitive = _train_pool()
    small = [t for t in pool if language.size(t) <= 7]

    def join(parts):
        term = parts[0]
        for part in parts[1:]:
            pair = (term, part) if rng.random() < 0.5 else (part, term)
            term = Expr(rng.choice(("add", "xor")), pair)
        return term

    out = []
    while len(out) < n:
        style = rng.choice(("T3", "T4", "T5", "T7") if sensitive else ("T3", "T4", "T5"))
        if style == "T5":
            out.extend(generate("asm", rng, 1, (4, 7), split="train"))
            continue
        if style == "T4":
            op, leaves = rng.choice(sorted(AC_OPS)), []
            for _ in range(rng.randint(10, 32)):
                x, kind = Var(f"r{rng.randrange(4)}"), rng.random()
                leaves.append(Expr("mul", (x, Const(rng.choice([2, 4, 8])))) if kind < 0.25 else
                              Expr("shl", (x, Const(rng.choice([1, 2, 3])))) if kind < 0.45 else
                              Expr("add", (x, x)) if kind < 0.55 else
                              rng.choice(small) if kind < 0.75 else
                              x if kind < 0.9 else Const(rng.choice([1, 3, 7, 16, 255, 0xFFFF])))
            while len(leaves) > 1:
                i = rng.randrange(len(leaves) - 1)
                leaves[i:i + 2] = [Expr(op, (leaves[i], leaves[i + 1]))]
            term = leaves[0]
        else:
            target = rng.choice(("L", "XL"))
            parts = [rng.choice(sensitive)] if style == "T7" else []
            term = None
            while True:
                parts.append(rng.choice(pool))
                if rng.random() < 0.5 and len(parts) >= 2:
                    parts.append(rng.choice(parts))
                candidate = join(parts)
                size = language.size(candidate)
                if size > language.max_nodes:
                    break
                if bucket(size) == target or (style == "T7" and size >= 30):
                    term = candidate
                    break
        if term is not None and language.size(term) <= language.max_nodes:
            out.append(term)
    return out


def _terms(rng: Random, n: int, k: int, depth: tuple[int, int]) -> list:
    """n training-split terms; k > 0 joins k generated terms by add/xor; k < 0 builds
    terms in the style of the test suites (T3/T4/T5/T7) from training-split material."""
    if k < 0:
        return _suite_style_terms(rng, n)
    if not k:
        return generate("asm", rng, n, depth, split="train")
    from benchmarks.asm.term import Expr

    language = AsmLanguage()
    out = []
    while len(out) < n:
        parts = generate("asm", rng, k, depth, split="train")
        term = parts[0]
        for part in parts[1:]:
            term = Expr(rng.choice(["add", "xor"]), (term, part))
        if language.size(term) <= language.max_nodes:
            out.append(term)
    return out


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
    p.add_argument("--composite", type=int, default=0,
                   help="k>0: k generated terms joined by add/xor; -1: test-suite-style training terms")
    p.add_argument("--kind", default="enable", choices=["enable", "enable_all", "rule", "mixed"])
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
    args = {"node_limit": a.node_limit, "steps": a.steps, "sims": a.sims, "current": str(current),
            "kind": a.kind}
    validation = _terms(Random(VALIDATION_SEED), a.validation, a.composite, (4, 6))
    log = (out / "train.log").open("a")

    def say(text: str) -> None:
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    say(f"=== {time.strftime('%F %T')} AZ-GEB from {start_path}, round {first}, "
        f"kind {a.kind}, node limit {a.node_limit}, steps {a.steps}, sims {a.sims}")
    # spawn, not fork: forking after torch has started its thread pools deadlocks the
    # workers (first launch hung with every process at 0% CPU).
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=context, initializer=_guard) as pool:
        for rnd in range(first, a.rounds):
            t0 = time.time()
            save_checkpoint(current, net, language, rnd, {})
            terms = _terms(Random(10_000 + rnd), a.episodes, a.composite, (3, 6))
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
            capped = sum(s[4] for s in scores) / len(scores)
            improved = az > best + 1e-6
            if improved:
                best, stale = az, 0
                save_checkpoint(out / "best.pt", net, language, rnd, {"score": az})
            else:
                stale += 1
            metrics = {"round": rnd, "samples": len(samples), **losses, "az": az, "egglog": egg,
                       "az_nodes": az_nodes, "egglog_nodes": egg_nodes, "all_capped": capped, "best": best,
                       "stale": stale, "seconds": time.time() - t0}
            save_checkpoint(out / "latest.pt", net, language, rnd, metrics, optimizer)
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            say(f"round {rnd}: samples={len(samples)} loss={losses['loss']:.3f} "
                f"az={az:.2f} egglog={egg:.2f} capped={capped:.2f} nodes az={az_nodes:.0f} egglog={egg_nodes:.0f} "
                f"best={best:.2f} ({time.time() - t0:.0f}s)")
            if stale >= a.patience:
                say(f"early stop: no better validation score for {a.patience} rounds")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
