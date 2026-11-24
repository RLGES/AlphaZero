"""Export the MCTS-GEB benchmark expressions (their generator, their seeds) as JSON trees
[op, child, ...] with leaf values, so other engines get identical inputs."""
import json, random, sys
import numpy as np
from pes.utils.utils import get_lang

def tree(e):
    if hasattr(e, "_fields"):
        return [type(e).__name__] + [tree(getattr(e, f)) for f in e._fields]
    if isinstance(e, (bool, np.bool_)):
        return bool(e)
    if isinstance(e, (int, np.integer)):
        return int(e)
    return str(e)

out = []
seeds = int(sys.argv[1])
for name in ("MATH", "PROP"):
    for seed in range(seeds):
        random.seed(seed); np.random.seed(seed)
        lang = get_lang(name)()
        out.append({"lang": name, "seed": seed, "tree": tree(lang.gen_expr(p_leaf=0., depth_limit=5))})
json.dump(out, open("/root/rlges/external/results/mctsgeb_exprs.json", "w"))
print(len(out), "expressions;", out[0]["tree"][0], out[20]["tree"][0])
