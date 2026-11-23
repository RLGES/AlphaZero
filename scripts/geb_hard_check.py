"""Where does egglog struggle? Score schedules on harder (bigger) terms at several e-node limits.

    RLGES_ASM_LEARNED=... PYTHONPATH=. python scripts/geb_hard_check.py [N] [WORKERS]

Terms: generated with depth 6-8 (test split, seed 777; the held-out/stress sets are
depth <= 5). For each e-node limit: egglog's default schedule (all rules each
iteration, match limit 1000, up to 6 iterations) and "all rules, match limit 200
per step" (up to 12 steps), reporting improvement, e-nodes and seconds. Also
egglog with no node limit under a 2 GB memory cap, to see how often it blows up.
Writes checkpoints/geb_hard_check.csv.
"""

from __future__ import annotations

import csv
import multiprocessing
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from random import Random

LIMITS = (300, 1000, 3000, 10000)


def _guard() -> None:
    limit = 4 * 2**30
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _capped(env, term, steps: int = 12):
    env.reset(term)
    env.active = tuple(a for a in env.valid if a != env.noop)
    best = env.extract()[1]
    while env.steps < steps and env.size() < env.node_limit:
        before = env.size()
        env.apply(env.noop)
        best = min(best, env.extract()[1])
        if env.size() == before:
            break
    return best, env.size()


def _forked(fn, timeout: float = 300.0):
    """Run fn() in a forked child with a 2 GB cap; egglog aborts the whole process
    when an allocation fails, so it must not run in the pool worker itself."""
    import os
    import pickle
    import select

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
        raise MemoryError("egglog child died or timed out")
    return pickle.loads(b"".join(chunks))


def run(job):
    index, term = job
    from benchmarks.asm.geb import EGraphEnv, egglog_schedule
    from benchmarks.asm.language import AsmLanguage

    language = AsmLanguage()
    c0 = language.cost(term)
    rows = []
    for limit in LIMITS:
        env = EGraphEnv(language, node_limit=limit, max_steps=12, kind="enable")
        def egg(env=env):
            ep = egglog_schedule(term, env)
            return ep.best_cost, ep.final_size

        for name, fn in (("egglog-default", egg),
                         ("all-rules-capped", lambda env=env: _capped(env, term))):
            start = time.perf_counter()
            try:
                cost, size = _forked(fn)
                status = "ok"
            except Exception as exc:  # memory cap hit inside egglog
                cost, size, status = c0, -1, type(exc).__name__
            rows.append({"term": index, "limit": limit, "schedule": name,
                         "initial_cost": c0, "final_cost": cost,
                         "improvement": 100 * (c0 - cost) / c0 if c0 else 0.0,
                         "enodes": size, "seconds": time.perf_counter() - start,
                         "status": status})
    return rows


def main() -> None:
    from benchmarks.generators import generate

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    import os

    k = int(os.environ.get("COMPOSITE", "0"))
    if k:
        # Bigger "basic blocks": k held-out terms combined with add/xor (size <= 256).
        from benchmarks.asm.final_eval import test_sets
        from benchmarks.asm.language import AsmLanguage
        from benchmarks.asm.term import Expr

        pool, rng, lang, terms = [t for _, t in test_sets()["heldout"]], Random(777), AsmLanguage(), []
        while len(terms) < n:
            parts = rng.sample(pool, k)
            term = parts[0]
            for part in parts[1:]:
                term = Expr(rng.choice(["add", "xor"]), (term, part))
            if lang.size(term) <= lang.max_nodes:
                terms.append(term)
    else:
        terms = generate("asm", Random(777), n, (6, 8), split="test")
    context = multiprocessing.get_context("spawn")
    rows = []
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_guard) as pool:
        for batch in pool.map(run, list(enumerate(terms)), chunksize=2):
            rows.extend(batch)
    with open("checkpoints/geb_hard_check.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'limit':>6} {'schedule':18} {'improv':>7} {'e-nodes':>8} {'sec':>6} failed")
    for limit in LIMITS:
        for name in ("egglog-default", "all-rules-capped"):
            sel = [r for r in rows if r["limit"] == limit and r["schedule"] == name]
            ok = [r for r in sel if r["status"] == "ok"]
            print(f"{limit:>6} {name:18} {sum(r['improvement'] for r in sel)/len(sel):7.2f} "
                  f"{sum(r['enodes'] for r in ok)/max(1, len(ok)):8.0f} "
                  f"{sum(r['seconds'] for r in sel)/len(sel):6.2f} {len(sel)-len(ok)}")


if __name__ == "__main__":
    main()
