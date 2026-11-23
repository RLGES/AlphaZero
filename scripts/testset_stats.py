"""Per-term aspects for the test-set spec (docs/paper/TESTSET_SPEC.md).

    # full measurement under the qwen rules (egglog, capped schedule, MCTS@100, greedy,
    # best-known, correctness) -> <out>
    RLGES_ASM_LEARNED=benchmarks/asm/learned_rules_qwen_all.json PYTHONPATH=. \
      python scripts/testset_stats.py measure <candidates.jsonl> <out.jsonl> [workers]
    # egglog cost with built-in rules only (rule sensitivity) -> <out>
    RLGES_ASM_LEARNED=none PYTHONPATH=. \
      python scripts/testset_stats.py builtin <candidates.jsonl> <out.jsonl> [workers]

Input lines: {"id", "suite", "provenance", "sexpr"}. Output lines add the measured
fields. Structural aspects (size, depth, width, sharing, op mix, constants, AC
chains) are computed by ``structure`` and need no rules. egglog runs in a forked
child with a 2 GB address-space cap and a timeout, because egglog aborts the whole
process when an allocation fails.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import resource
import select
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from random import Random

ARITH = {"add", "sub", "mul"}
BITWISE = {"and", "or", "xor", "not", "shl", "shr", "neg"}
EXPENSIVE = {"div", "mod"}
AC_OPS = {"add", "mul", "and", "or", "xor"}
MASK32 = 0xFFFFFFFF
EGG_LIMIT = 10_000
TIMEOUT = 120.0


def bucket(size: int) -> str:
    return "S" if size < 20 else "M" if size <= 60 else "L" if size <= 150 else "XL"


def _nodes(term):
    stack = [term]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.args)


def _size(term) -> int:
    return 1 + sum(_size(a) for a in term.args)


def _depth(term) -> int:
    return 1 + max((_depth(a) for a in term.args), default=0)


def _ac_chain(term) -> int:
    """Longest flattened chain of operands under one commutative-associative op."""
    best = 0

    def operands(node, op):
        if node.op == op:
            return sum(operands(a, op) for a in node.args)
        return 1

    for node in _nodes(term):
        if node.op in AC_OPS:
            best = max(best, operands(node, node.op))
    return best


def structure(term) -> dict:
    nodes = list(_nodes(term))
    ops = Counter(n.op for n in nodes)
    levels, frontier = [], [term]
    while frontier:
        levels.append(len(frontier))
        frontier = [a for n in frontier for a in n.args]
    subtrees = Counter(n for n in nodes if _size(n) >= 3)
    consts = [int(n.value) for n in nodes if n.op == "const"]
    classes = {
        "arith": sum(ops[o] for o in ARITH),
        "bitwise": sum(ops[o] for o in BITWISE),
        "expensive": sum(ops[o] for o in EXPENSIVE),
        "memory": ops["load"],
        "control": ops["phi"],
    }
    has_arith = classes["arith"] + classes["expensive"] > 0
    has_bit = classes["bitwise"] > 0
    foldable = sum(
        1 for n in nodes
        if n.args and n.op != "load" and all(a.op == "const" for a in n.args)
    )
    size = len(nodes)
    return {
        "size": size,
        "bucket": bucket(size),
        "depth": _depth(term),
        "width": max(levels),
        "shared_subtrees": sum(1 for _, c in subtrees.items() if c >= 2),
        "op_classes": classes,
        "domain": "mixed" if has_arith and has_bit else "arith" if has_arith else "bitwise",
        "constants": len(consts),
        "pow2_constants": sum(1 for c in consts if c > 0 and c & (c - 1) == 0),
        "mask_constants": sum(1 for c in consts if c > 0 and (c + 1) & c == 0),
        "foldable": foldable,
        "ac_chain": _ac_chain(term),
        "variables": len({n.value for n in nodes if n.op == "var"}),
    }


def _forked(fn, timeout: float = TIMEOUT):
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
            data = pickle.dumps(fn())
            with os.fdopen(write, "wb") as out:
                out.write(data)
        finally:
            os._exit(0)
    os.close(write)
    chunks, deadline = [], time.monotonic() + timeout
    with os.fdopen(read, "rb") as inp:
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([inp], [], [], left)[0]:
                os.kill(pid, 9)
                break
            chunk = inp.read1(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    os.waitpid(pid, 0)
    if not chunks:
        return None
    return pickle.loads(b"".join(chunks))


def _egglog(term, limit: int = EGG_LIMIT):
    """(best cost, e-nodes, best term) of egglog's default schedule under an e-node limit."""
    from benchmarks.asm.geb import EGraphEnv, egglog_schedule
    from benchmarks.asm.language import AsmLanguage

    env = EGraphEnv(AsmLanguage(), node_limit=limit, max_steps=12, kind="enable")
    episode = egglog_schedule(term, env)
    return episode.best_cost, episode.final_size, episode.best_term


def _capped(term, limit: int = EGG_LIMIT, steps: int = 12):
    """All rules enabled, match limit 200 per step: (best cost, e-nodes, best term)."""
    from benchmarks.asm.geb import EGraphEnv
    from benchmarks.asm.language import AsmLanguage

    env = EGraphEnv(AsmLanguage(), node_limit=limit, max_steps=steps, kind="enable")
    env.reset(term)
    env.active = tuple(a for a in env.valid if a != env.noop)
    best_term, best = env.extract()
    while env.steps < steps and env.size() < env.node_limit:
        before = env.size()
        env.apply(env.noop)
        candidate, cost = env.extract()
        if cost < best:
            best, best_term = cost, candidate
        if env.size() == before:
            break
    return best, env.size(), best_term


def equivalent(a, b, trials: int = 200, seed: int = 0) -> bool:
    """Equal on ``trials`` random 32-bit inputs (edge values included)."""
    from benchmarks.asm.term import evaluate

    names = sorted({n.value for n in _nodes(a) if n.op == "var"}
                   | {n.value for n in _nodes(b) if n.op == "var"})
    rng = Random(seed)
    edges = [0, 1, 2, 31, 32, 33, MASK32, MASK32 - 1, 1 << 31]
    for i in range(trials):
        env = {name: (edges[(i + j) % len(edges)] if i < 20 else rng.getrandbits(32))
               for j, name in enumerate(names)}
        if evaluate(a, env) != evaluate(b, env):
            return False
    return True


def measure(line: dict) -> dict:
    from src.api import rewrite_detailed
    from src.engines.mcts import MCTSConfig

    from benchmarks.asm.hardness import greedy_cost
    from benchmarks.asm.language import AsmLanguage
    from benchmarks.asm.term import from_sexpr

    language = AsmLanguage()
    term = from_sexpr(line["sexpr"])
    out = dict(line)
    out.update(structure(term))
    c0 = language.cost(term)
    out["initial_cost"] = c0
    t = time.perf_counter()
    egg = _forked(lambda: _egglog(term))
    out["egglog_seconds"] = time.perf_counter() - t
    out["egglog_failed"] = egg is None
    out["egglog_cost"] = egg[0] if egg else c0
    out["egglog_enodes"] = egg[1] if egg else -1
    cap = _forked(lambda: _capped(term))
    out["capped_cost"] = cap[0] if cap else c0
    out["capped_enodes"] = cap[1] if cap else -1
    mcts = rewrite_detailed(language, term, MCTSConfig(iterations=100, seed=0))
    out["mcts100_cost"] = mcts.final_cost
    out["greedy_cost"] = greedy_cost(language, term)
    candidates = [(mcts.final_cost, mcts.final_term)]
    if egg:
        candidates.append((egg[0], egg[2]))
    if cap:
        candidates.append((cap[0], cap[2]))
    best_cost, best_term = min(candidates, key=lambda x: x[0])
    out["best_known"] = best_cost
    out["non_greedy"] = out["greedy_cost"] > best_cost
    out["best_equivalent"] = equivalent(term, best_term)
    from benchmarks.asm.term import to_sexpr

    out["best_sexpr"] = to_sexpr(best_term)
    return out


def builtin(line: dict) -> dict:
    from benchmarks.asm.term import from_sexpr

    term = from_sexpr(line["sexpr"])
    egg = _forked(lambda: _egglog(term))
    return {"id": line["id"], "egglog_builtin_cost": egg[0] if egg else None,
            "egglog_builtin_failed": egg is None}


def main() -> None:
    mode, src, dst = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    lines = [json.loads(x) for x in open(src)]
    done = set()
    if os.path.exists(dst):
        done = {json.loads(x)["id"] for x in open(dst)}
    todo = [x for x in lines if x["id"] not in done]
    fn = measure if mode == "measure" else builtin
    context = multiprocessing.get_context("spawn")
    start = time.time()
    with ProcessPoolExecutor(workers, mp_context=context) as pool, open(dst, "a") as out:
        for i, row in enumerate(pool.map(fn, todo, chunksize=1)):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(todo)} {time.time() - start:.0f}s", flush=True)
    print(f"done {len(todo)} terms in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
