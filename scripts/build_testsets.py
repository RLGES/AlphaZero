"""Build the e-graph test suites T3, T4, T5, T7 (docs/paper/TESTSET_SPEC.md).

    PYTHONPATH=. python scripts/build_testsets.py candidates      # candidate pools
    # then measure each pool with scripts/testset_stats.py (measure + builtin)
    PYTHONPATH=. python scripts/build_testsets.py select          # suites + manifests

Files: benchmarks/asm/testsets/candidates_<suite>.jsonl, measured_<suite>.jsonl,
builtin_<suite>.jsonl -> <suite>.jsonl and <suite>_manifest.json. Seeds are >= 1,000,000
so they never meet the training seeds. The language caps terms at 256 nodes, so the
XL bucket here is 151-256 nodes (the spec's 150-500 would need a larger cap).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent))
from testset_stats import AC_OPS, bucket, structure  # noqa: E402

OUT = Path("benchmarks/asm/testsets")
SUITES = ("T3", "T4", "T5", "T7")


def _real_terms():
    from benchmarks.asm.final_eval import test_sets

    sets = test_sets()
    return [t for _, t in sets["heldout"]] + [t for _, t in sets["stress"]]


def _write(suite: str, items: list[tuple[str, object]]) -> None:
    from benchmarks.asm.term import to_sexpr

    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    with (OUT / f"candidates_{suite}.jsonl").open("w") as handle:
        for i, (provenance, term) in enumerate(items):
            text = to_sexpr(term)
            if text in seen:
                continue
            seen.add(text)
            handle.write(json.dumps({"id": f"{suite}-{i:04d}", "suite": suite,
                                     "provenance": provenance, "sexpr": text}) + "\n")
    print(suite, len(seen), "candidates")


def _join(parts, rng, ops=("add", "xor")):
    from benchmarks.asm.term import Expr

    term = parts[0]
    for part in parts[1:]:
        term = Expr(rng.choice(ops), (term, part)) if rng.random() < 0.5 else \
            Expr(rng.choice(ops), (part, term))
    return term


def t3(rng: Random, n: int = 400):
    """Real held-out/stress terms joined into blocks, half L and half XL, with sharing."""
    from benchmarks.asm.language import AsmLanguage

    language, pool, out = AsmLanguage(), _real_terms(), []
    while len(out) < n:
        target = "L" if len(out) % 2 == 0 else "XL"
        parts = []
        while True:
            parts.append(rng.choice(pool))
            if rng.random() < 0.5 and len(parts) >= 2:
                parts.append(rng.choice(parts))  # the same subterm again: shared structure
            term = _join(parts, rng)
            size = language.size(term)
            if size > language.max_nodes:
                break
            if bucket(size) == target:
                out.append(("real-blocks", term))
                break
    return out


def t4(rng: Random, n: int = 300):
    """Long chains under one commutative-associative op, with rule-undoing leaves
    (x*2 vs x<<1, x+x) and small real subterms: dense, mutually reversible matches."""
    from benchmarks.asm.language import AsmLanguage
    from benchmarks.asm.term import Const, Expr, Var

    language = AsmLanguage()
    small = [t for t in _real_terms() if language.size(t) <= 7]
    out = []
    while len(out) < n:
        op = rng.choice(sorted(AC_OPS))
        length = rng.randint(10, 32)
        leaves = []
        for _ in range(length):
            x = Var(f"r{rng.randrange(4)}")
            kind = rng.random()
            if kind < 0.25:
                leaves.append(Expr("mul", (x, Const(rng.choice([2, 4, 8])))))
            elif kind < 0.45:
                leaves.append(Expr("shl", (x, Const(rng.choice([1, 2, 3])))))
            elif kind < 0.55:
                leaves.append(Expr("add", (x, x)))
            elif kind < 0.75:
                leaves.append(rng.choice(small))
            elif kind < 0.9:
                leaves.append(x)
            else:
                leaves.append(Const(rng.choice([1, 3, 7, 16, 255, 0xFFFF])))
        rng.shuffle(leaves)
        while len(leaves) > 1:  # random tree shape, not only left-deep
            i = rng.randrange(len(leaves) - 1)
            leaves[i:i + 2] = [Expr(op, (leaves[i], leaves[i + 1]))]
        term = leaves[0]
        if language.size(term) <= language.max_nodes:
            out.append(("synthetic-ac", term))
    return out


def t5(rng: Random, n: int = 500):
    """Generated test-split terms, deeper than training (depth 4-7)."""
    from benchmarks.generators import generate

    return [("synthetic-deep", t) for t in generate("asm", rng, n, (4, 7), split="test")]


def t7(rng: Random, n_composite: int = 220, n_generated: int = 300):
    """Rule-sensitive candidates: blocks containing a held-out/stress term on which the
    qwen rules change egglog's result (from the final-eval caches), plus generated terms."""
    import csv

    from benchmarks.asm.final_eval import test_sets
    from benchmarks.asm.language import AsmLanguage
    from benchmarks.generators import generate

    language, sets, sensitive = AsmLanguage(), test_sets(), []
    for set_name in ("heldout", "stress"):
        builtin = {r["term"]: float(r["final_cost"]) for r in csv.DictReader(
            open(f"checkpoints/final_egglog_none_{set_name}.csv"))}
        qwen = {r["term"]: float(r["final_cost"]) for r in csv.DictReader(
            open(f"checkpoints/final_egglog_learned_rules_qwen_all_{set_name}.csv"))}
        terms = dict(sets[set_name])
        sensitive += [terms[k] for k in terms if qwen.get(k, 1e9) < builtin.get(k, -1)]
    print("sensitive seeds from caches:", len(sensitive))
    fillers = generate("asm", Random(1_000_003), 400, (3, 5), split="test")
    out = []
    while len(out) < n_composite:
        parts = [rng.choice(sensitive)] + rng.sample(fillers, rng.randint(1, 2))
        term = _join(parts, rng)
        if language.size(term) <= language.max_nodes:
            out.append(("real-sensitive-block", term))
    out += [("synthetic", t) for t in generate("asm", rng, n_generated, (4, 7), split="test")]
    return out


def _norm(term) -> str:
    """Structural fingerprint up to variable renaming."""
    from benchmarks.asm.term import to_sexpr

    names: dict = {}

    def walk(node):
        if node.op == "var":
            names.setdefault(node.value, f"v{len(names)}")
            return f"(var {names[node.value]})"
        if node.op == "const":
            return str(node.value)
        return "(" + node.op + " " + " ".join(walk(a) for a in node.args) + ")"

    return walk(term) if term is not None else to_sexpr(term)


def _training_fingerprints() -> set[str]:
    """Every term the run-4 trainings and their validation could have seen."""
    from benchmarks.generators import generate
    from src.engines.alphazero.evaluate import TEST_SEED

    prints = set()
    for seed in (0, 1, 2):
        for rnd in range(100):
            for term in generate("asm", Random(seed * 1_000_003 + rnd), 256, (3, 6)):
                prints.add(_norm(term))
    for term in generate("asm", Random(TEST_SEED), 450, (3, 7), split="test"):
        prints.add(_norm(term))
    return prints


ACCEPT = {
    "T3": lambda r: True,
    "T4": lambda r: True,
    "T5": lambda r: r["non_greedy"],
    "T7": lambda r: r.get("egglog_builtin_cost") is not None
    and r["egglog_cost"] < r["egglog_builtin_cost"],
}


def select() -> None:
    from benchmarks.asm.term import from_sexpr

    start = time.time()
    training = _training_fingerprints()
    existing = {_norm(t) for t in _real_terms()}
    print(f"training fingerprints {len(training)} ({time.time() - start:.0f}s)")
    used: set[str] = set()
    summary_all = {}
    for suite in SUITES:
        measured_path = OUT / f"measured_{suite}.jsonl"
        if not measured_path.exists():
            print(suite, "not measured yet")
            continue
        rows = [json.loads(x) for x in open(measured_path)]
        builtin_path = OUT / f"builtin_{suite}.jsonl"
        if builtin_path.exists():
            extra = {json.loads(x)["id"]: json.loads(x) for x in open(builtin_path)}
            for r in rows:
                r.update({k: v for k, v in extra.get(r["id"], {}).items() if k != "id"})
        kept, dropped = [], Counter()
        for r in rows:
            fp = _norm(from_sexpr(r["sexpr"]))
            if fp in training:
                dropped["in training"] += 1
            elif fp in existing:
                dropped["in T1/T2"] += 1
            elif fp in used:
                dropped["duplicate"] += 1
            elif not r["best_equivalent"]:
                dropped["best not equivalent"] += 1
            elif not ACCEPT[suite](r):
                dropped["criterion"] += 1
            else:
                used.add(fp)
                kept.append(r)
        n = len(kept)
        pct = lambda f: round(100 * sum(1 for r in kept if f(r)) / n, 1) if n else 0.0
        summary = {
            "terms": n,
            "candidates": len(rows),
            "dropped": dict(dropped),
            "buckets": dict(Counter(r["bucket"] for r in kept)),
            "provenance": dict(Counter(r["provenance"] for r in kept)),
            "domain": dict(Counter(r["domain"] for r in kept)),
            "op_mix_total": {k: sum(r["op_classes"][k] for r in kept)
                             for k in ("arith", "bitwise", "expensive", "memory", "control")},
            "pct_sharing": pct(lambda r: r["shared_subtrees"] > 0),
            "pct_ac_chain_ge6": pct(lambda r: r["ac_chain"] >= 6),
            "pct_egglog_failed": pct(lambda r: r["egglog_failed"]),
            "pct_egglog_failed_or_gt3k": pct(lambda r: r["egglog_failed"] or r["egglog_enodes"] > 3000),
            "pct_egglog_failed_or_gt5k": pct(lambda r: r["egglog_failed"] or r["egglog_enodes"] > 5000),
            "pct_non_greedy": pct(lambda r: r["non_greedy"]),
            "pct_rule_sensitive": pct(lambda r: r.get("egglog_builtin_cost") is not None
                                      and r["egglog_cost"] < r["egglog_builtin_cost"]),
            "pct_best_equivalent": pct(lambda r: r["best_equivalent"]),
            "mean_size": round(sum(r["size"] for r in kept) / n, 1) if n else 0,
            "separation": "fingerprints checked against run-4 training terms "
                          f"({len(training)}), held-out+stress, and earlier suites",
        }
        summary_all[suite] = summary
        with (OUT / f"{suite}.jsonl").open("w") as handle:
            for r in kept:
                handle.write(json.dumps({"id": r["id"], "provenance": r["provenance"],
                                         "sexpr": r["sexpr"]}) + "\n")
        json.dump({"suite": suite, "summary": summary, "terms": kept},
                  (OUT / f"{suite}_manifest.json").open("w"), indent=1)
        print(suite, json.dumps(summary))
    json.dump(summary_all, (OUT / "summary.json").open("w"), indent=1)


def main() -> None:
    if sys.argv[1] == "candidates":
        start = time.time()
        _write("T3", t3(Random(1_000_101)))
        _write("T4", t4(Random(1_000_202)))
        _write("T5", t5(Random(1_000_303)))
        _write("T7", t7(Random(1_000_404)))
        print(f"candidates built in {time.time() - start:.0f}s")
    else:
        select()


if __name__ == "__main__":
    main()
