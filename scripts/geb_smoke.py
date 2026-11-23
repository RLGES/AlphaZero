"""Smoke test: egglog schedule vs MCTS-GEB vs AZ-GEB (term-net init) at one node limit.

    RLGES_ASM_LEARNED=... python scripts/geb_smoke.py N NODE_LIMIT SIMS CHECKPOINT
"""
import sys
import time
from random import Random

from benchmarks.asm.final_eval import _network, test_sets
from benchmarks.asm.geb import Agent, EGraphEnv, SearchConfig, build, egglog_schedule
from benchmarks.asm.language import AsmLanguage

n, limit, sims, ckpt = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
language = AsmLanguage()
env = EGraphEnv(language, node_limit=limit, max_steps=12)
terms = test_sets()["heldout"][:n]
agents = {
    "egglog": None,
    "mcts-geb": (Agent(language), SearchConfig(simulations=sims, mode="geb", rollout_depth=6)),
    "az-geb(init)": (Agent(language, _network(ckpt)), SearchConfig(simulations=sims, mode="az")),
}
for label, spec in agents.items():
    start, imps, sizes = time.time(), [], []
    for name, term in terms:
        if spec is None:
            ep = egglog_schedule(term, env)
        else:
            ep = build(term, env, spec[0], spec[1], Random(0))
        imps.append(100 * (ep.initial_cost - ep.best_cost) / ep.initial_cost)
        sizes.append(ep.final_size)
    print(f"{label:14s} improvement {sum(imps)/n:5.1f}  mean e-nodes {sum(sizes)/n:7.0f}  "
          f"{(time.time()-start)/n:6.2f} s/term", flush=True)
