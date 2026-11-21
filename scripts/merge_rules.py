"""Merge the accepted rules of several discovery rounds into one rule file.

    .venv/bin/python scripts/merge_rules.py --out benchmarks/asm/learned_rules_qwen12.json \
        benchmarks/asm/learned_rules_qwen.json benchmarks/asm/learned_rules_qwen2.json

Round k > 1 rule names get an ``r<k>_`` prefix (as in the night-3 ``qwen12`` file);
a rule already taken from an earlier round is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def merge(reports: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    merged = []
    for round_index, report in enumerate(reports, start=1):
        for rule in report["accepted"]:
            key = (rule["lhs"], rule["rhs"])
            if key in seen:
                continue
            seen.add(key)
            prefix = "" if round_index == 1 else f"r{round_index}_"
            merged.append({**rule, "name": prefix + rule["name"]})
    return merged


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("rounds", type=Path, nargs="+", help="Discovery reports, round 1 first")
    args = parser.parse_args(argv)
    merged = merge([json.loads(path.read_text()) for path in args.rounds])
    out = {
        "sources": [path.name for path in args.rounds],
        "note": "accepted = every round's accepted rules (round k > 1 names prefixed rk_)",
        "accepted": merged,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(merged)} rules -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
