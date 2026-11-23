"""Peak memory of egglog with all rules vs egglog on AlphaZero-chosen rules (built-in +
top-k learned + AZ's path), per held-out term, each egglog call in its own forked child.

    RLGES_ASM_LEARNED=benchmarks/asm/learned_rules_qwen_all.json PYTHONPATH=. \
      python scripts/memory_compare.py <checkpoint> <out.csv> [N] [K]

Peak memory = ru_maxrss of the egglog child (os.wait4), in MB; it includes the
forked Python process's resident baseline, which is the same for both arms. The
child has a 2 GB address-space cap; a failed call is retried with 4, 2, 1 iterations
like the final evaluation and recorded.
"""
import csv
import os
import pickle
import resource
import sys
import time


def egglog_child(language, term, rules):
    from src.api import rewrite_detailed
    from src.engines.egglog import EgglogConfig

    for iterations in (6, 4, 2, 1):
        read, write = os.pipe()
        start = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            os.close(read)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
                if rules is not None:
                    os.environ["RLGES_EGGLOG_RULES"] = ",".join(sorted(rules))
                r = rewrite_detailed(language, term, EgglogConfig(iterations=iterations))
                with os.fdopen(write, "wb") as out:
                    out.write(pickle.dumps((r.final_cost, r.statistics.get("rule_matches", 0))))
            finally:
                os._exit(0)
        os.close(write)
        with os.fdopen(read, "rb") as inp:
            data = inp.read()
        _, _, usage = os.wait4(pid, 0)
        seconds = time.perf_counter() - start
        if data:
            cost, matches = pickle.loads(data)
            return cost, matches, usage.ru_maxrss / 1024, seconds, iterations
    return language.cost(term), 0, usage.ru_maxrss / 1024, seconds, 0


def main():
    import torch

    from benchmarks.asm.final_eval import _az_rule_choice, _network, test_sets
    from benchmarks.asm.language import AsmLanguage
    from src.api import rewrite_detailed
    from src.engines.alphazero.config import AlphaZeroConfig

    torch.set_num_threads(1)
    ckpt, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1722
    os.environ["AZRULES_K"] = sys.argv[4] if len(sys.argv) > 4 else "10"
    language = AsmLanguage()
    net = _network(ckpt)
    rows = []
    for name, term in test_sets()["heldout"][:n]:
        az = rewrite_detailed(language, term, AlphaZeroConfig(iterations=100, seed=0, network=net))
        rules = _az_rule_choice(language, term, ckpt, az.trace)
        c0 = language.cost(term)
        for arm, subset in (("egglog-all", None), ("egglog-AZ-rules", rules)):
            cost, matches, mb, sec, it = egglog_child(language, term, subset)
            if arm == "egglog-AZ-rules":
                cost = min(cost, az.final_cost)
            rows.append({"term": name, "arm": arm,
                         "rules": len(language.rules) - 1 if subset is None else len(subset),
                         "improvement": 100 * (c0 - cost) / c0, "rule_matches": matches,
                         "peak_mb": round(mb, 1), "seconds": round(sec, 3), "iterations": it})
        if len(rows) % 200 == 0:
            print(len(rows) // 2, "terms", flush=True)
    with open(out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for arm in ("egglog-all", "egglog-AZ-rules"):
        sel = [r for r in rows if r["arm"] == arm]
        mbs = sorted(r["peak_mb"] for r in sel)
        print(arm, "terms", len(sel), "rules mean", round(sum(r["rules"] for r in sel) / len(sel), 1),
              "improvement", round(sum(r["improvement"] for r in sel) / len(sel), 2),
              "peak MB mean", round(sum(mbs) / len(mbs), 1), "p95", mbs[int(0.95 * len(mbs))],
              "max", mbs[-1], "matches mean", round(sum(r["rule_matches"] for r in sel) / len(sel)),
              "sec mean", round(sum(r["seconds"] for r in sel) / len(sel), 3),
              "retried", sum(1 for r in sel if r["iterations"] != 6))


if __name__ == "__main__":
    main()
