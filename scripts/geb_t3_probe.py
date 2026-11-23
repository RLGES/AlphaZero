"""Probe: on T3 terms where egglog struggled, compare at one e-node limit:
egglog default, all-rules-capped, MCTS-GEB (mixed, rollouts) and AZ-GEB (mixed,
term-network priors, untrained for e-graphs). Each run in a forked child, 2 GB cap.

    RLGES_ASM_LEARNED=... PYTHONPATH=. python scripts/geb_t3_probe.py N LIMIT WORKERS
"""
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from random import Random

sys.path.insert(0, "scripts")
from testset_stats import _forked  # noqa: E402


def run(job):
    sexpr, limit = job
    from benchmarks.asm.geb import Agent, EGraphEnv, SearchConfig, build, egglog_schedule
    from benchmarks.asm.language import AsmLanguage
    from benchmarks.asm.term import from_sexpr
    from testset_stats import _capped

    lang = AsmLanguage()
    term = from_sexpr(sexpr)
    c0 = lang.cost(term)
    out = {}

    def egg():
        env = EGraphEnv(lang, node_limit=limit, max_steps=12, kind="mixed")
        ep = egglog_schedule(term, env)
        return ep.best_cost, ep.final_size

    def cap():
        c, size, _ = _capped(term, limit=limit, steps=12)
        return c, size

    def agent(mode):
        def go():
            net = None
            if mode == "az":
                import torch

                from benchmarks.asm.final_eval import _network
                torch.set_num_threads(1)
                net = _network("checkpoints/asm_v4_llm_s0/best.pt")
            env = EGraphEnv(lang, node_limit=limit, max_steps=30, kind="mixed")
            cfg = SearchConfig(simulations=16, mode=mode, rollout_depth=6)
            ep = build(term, env, Agent(lang, net), cfg, Random(0))
            return ep.best_cost, ep.final_size
        return go

    for name, fn in (("egglog", egg), ("capped", cap), ("mcts-geb", agent("geb")),
                     ("az-geb", agent("az"))):
        t = time.perf_counter()
        r = _forked(fn, timeout=600)
        out[name] = (100 * (c0 - r[0]) / c0 if r else None, r[1] if r else None,
                     round(time.perf_counter() - t, 1))
    return out


def main():
    n, limit, workers = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    suite = sys.argv[4] if len(sys.argv) > 4 else "T3"
    which = sys.argv[5] if len(sys.argv) > 5 else "hard"  # "hard" or "all"
    path = f"benchmarks/asm/testsets/{suite}.jsonl"
    import os
    if not os.path.exists(path):
        path = f"benchmarks/asm/testsets/measured_{suite}.jsonl"
    rows = [json.loads(x) for x in open(path)]
    if which == "hard":
        rows = [r for r in rows if r.get("egglog_failed") or r.get("egglog_enodes", 0) > 3000]
    hard = rows[:n]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        results = list(pool.map(run, [(r["sexpr"], limit) for r in hard]))
    tag = f"{suite}_{which}_{limit}"
    json.dump([{"id": r.get("id"), **res} for r, res in zip(hard, results)],
              open(f"checkpoints/geb_probe_{tag}.json", "w"))
    print(f"{len(hard)} {which} {suite} terms, e-node limit {limit}")
    for name in ("egglog", "capped", "mcts-geb", "az-geb"):
        ok = [r[name] for r in results if r[name][0] is not None]
        print(f"{name:9s} improvement {sum(x[0] for x in ok)/max(1,len(ok)):6.2f} "
              f"e-nodes {sum(x[1] for x in ok)/max(1,len(ok)):7.0f} "
              f"sec {sum(x[2] for x in ok)/max(1,len(ok)):6.1f} failed {len(results)-len(ok)}")


if __name__ == "__main__":
    main()
