"""Pool every verified, cost-lowering rule from the discovery rounds into one file.

    python scripts/pool_rules.py out.json in1.json in2.json ...
"""

import json
import sys

out, *inputs = sys.argv[1:]
seen: dict = {}
for path in inputs:
    report = json.load(open(path))
    for rule in report["accepted"] + report.get("verified_not_loaded", []):
        seen.setdefault((rule["lhs"], rule["rhs"]), rule)
rules = sorted(
    (r for r in seen.values() if r["rhs_cost"] < r["lhs_cost"]),
    key=lambda r: (-r.get("seed_matches", 0), r["rhs_cost"] - r["lhs_cost"]),
)
accepted = [{**r, "name": f"pool_{i}"} for i, r in enumerate(rules)]
json.dump({"sources": inputs, "note": "all verified cost-lowering rules", "accepted": accepted},
          open(out, "w"), indent=1)
print(f"{len(accepted)} rules -> {out}")
