"""Score a rule set by plain MCTS on training-distribution terms (never the final test terms).

    RLGES_ASM_LEARNED=<rules.json|none> python scripts/rule_select_eval.py [N] [WORKERS]
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from random import Random

from benchmarks.generators import generate
from benchmarks.registry import load_suite
from src.api import rewrite_detailed
from src.engines.mcts import MCTSConfig

VALIDATION_SEED = 424_242  # not the test seed, not the small training seeds
BUDGETS = (25, 100)


def score(term):
    language = load_suite("asm").language
    start = language.cost(term)
    out = []
    for budget in BUDGETS:
        result = rewrite_detailed(language, term, MCTSConfig(iterations=budget, seed=0))
        out.append((start - result.final_cost) / start if start else 0.0)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    terms = generate("asm", Random(VALIDATION_SEED), n, (3, 5), split="train")
    with ProcessPoolExecutor(workers) as pool:
        scores = list(pool.map(score, terms, chunksize=4))
    means = [100 * sum(s[i] for s in scores) / len(scores) for i in range(len(BUDGETS))]
    rules = os.environ.get("RLGES_ASM_LEARNED", "")
    print(rules, " ".join(f"@{b}={m:.1f}" for b, m in zip(BUDGETS, means)))
