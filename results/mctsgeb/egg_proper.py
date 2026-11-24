"""Stronger egg baselines on the MCTS-GEB benchmarks, with their own library (unmodified):
egg Runner with ALL rules per iteration, (a) BackoffScheduler (egg default: match limit 1000,
ban length 5) and (b) SimpleScheduler, iteration limit 30, same e-node limit, AstSize.
Same expressions as their run_egg.py / run_mcts.py (np.random seed -> gen_expr depth 5)."""
import random
import sys
import time

import numpy as np
from pes.utils.utils import get_lang, new_egraph

limit, seeds = int(sys.argv[1]), int(sys.argv[2])
print("lang,seed,limit,base_cost,egg_backoff_cost,egg_backoff_s,egg_simple_cost,egg_simple_s", flush=True)
for name in ("MATH", "PROP"):
    for seed in range(seeds):
        random.seed(seed); np.random.seed(seed)
        lang = get_lang(name)()
        expr = lang.gen_expr(p_leaf=0., depth_limit=5)
        out = [name, seed, limit]
        base = None
        for backoff in (True, False):
            eg = new_egraph(expr)
            if base is None:
                base, _ = eg.extract(expr)
                out.append(base)
            t = time.perf_counter()
            eg.run(lang.rewrite_rules(), iter_limit=30, time_limit=10.0, node_limit=limit, use_backoff=backoff)
            cost, _ = eg.extract(expr)
            out += [cost, round(time.perf_counter() - t, 3)]
        print(",".join(map(str, out)), flush=True)
