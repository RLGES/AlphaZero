"""Paper figures from the run-4 result files (re-run any time; skips missing inputs).

    .venv/bin/python scripts/paper_figures.py   # writes docs/paper/figures/*.{png,pdf,csv}

Each figure also gets a CSV with exactly the plotted numbers (the table view).
Colours: fixed categorical order (blue, orange, aqua, yellow) from the dataviz
reference palette, validated for colour-vision deficiency as a set.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints"
OUT = ROOT / "docs" / "paper" / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name: str, header: list[str], rows: list[list]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    with (OUT / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("wrote", name)


def report(path: Path) -> dict:
    """{(engine, budget): (improvement, best_known, transitions)} from a report txt."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 7 and parts[1].isdigit():
            out[(parts[0], int(parts[1]))] = (
                float(parts[3].rstrip("%")), float(parts[4].rstrip("%")), float(parts[5]))
    return out


def mean_improvement(path: Path, budget: str | None = None) -> float | None:
    if not path.exists():
        return None
    values = [float(r["improvement"]) for r in csv.DictReader(path.open())
              if budget is None or r["budget"] == budget]
    return sum(values) / len(values) if values else None


def fig_training() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    rows = []
    for arm, colour, label in (("builtin", BLUE, "built-in rules"), ("llm", ORANGE, "+ qwen rules")):
        for i, path in enumerate(sorted(glob.glob(str(CK / f"asm_v4_{arm}_s*/metrics.jsonl")))):
            seed = re.search(r"_s(\d+)", path).group(1)
            points = [json.loads(line) for line in open(path)]
            xs = [p["round"] for p in points]
            ys = [p["score"] for p in points]
            ax.plot(xs, ys, color=colour, alpha=0.9, linewidth=1.4,
                    label=label if i == 0 else None)
            rows += [[arm, seed, x, round(y, 4)] for x, y in zip(xs, ys)]
    ax.set_xlabel("training round")
    ax.set_ylabel("validation score (improvement @25)")
    ax.legend(loc="lower right")
    ax.set_title("Term-level AlphaZero training, 3 seeds per rule set", loc="left", color=INK)
    save(fig, "fig1_training", ["arm", "seed", "round", "score"], rows)


def fig_az_vs_mcts() -> None:
    budgets = [10, 25, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    rows = []
    for ax, arm, title in zip(axes, ("builtin", "llm"), ("built-in rules", "+ qwen rules")):
        reports = [report(Path(p)) for p in sorted(glob.glob(str(CK / f"asm_v4_{arm}_s*/report_after.txt")))]
        reports = [r for r in reports if r]
        if not reports:
            continue
        series = {}
        for engine, colour in (("AlphaZero", BLUE), ("MCTS", ORANGE)):
            ys = [sum(r[(engine, b)][0] for r in reports) / len(reports) for b in budgets]
            ax.plot(budgets, ys, color=colour, marker="o", markersize=5, label=engine)
            series[engine] = ys[-1]
            rows += [[arm, engine, b, round(y, 2), len(reports)] for b, y in zip(budgets, ys)]
        # end labels: the higher value above, the lower below, so they never collide
        high = max(series, key=series.get)
        for engine, value in series.items():
            ax.annotate(f"{value:.1f}", (budgets[-1], value), textcoords="offset points",
                        xytext=(6, 5 if engine == high else -11), color=INK, fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("search budget (iterations)")
        ax.set_title(f"{title} ({len(reports)} seeds)", loc="left", color=INK)
    axes[0].set_ylabel("mean improvement (%)")
    axes[0].legend(loc="lower right")
    save(fig, "fig2_az_vs_mcts_314", ["arm", "engine", "budget", "improvement", "seeds"], rows)


def fig_transitions() -> None:
    budgets = [10, 25, 50, 100]
    reports = [report(Path(p)) for p in sorted(glob.glob(str(CK / "asm_v4_llm_s*/report_after.txt")))]
    reports = [r for r in reports if r]
    if not reports:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    rows = []
    for engine, colour in (("AlphaZero", BLUE), ("MCTS", ORANGE)):
        ys = [sum(r[(engine, b)][2] for r in reports) / len(reports) for b in budgets]
        ax.plot(budgets, ys, color=colour, marker="o", markersize=5, label=engine)
        rows += [[engine, b, round(y, 1)] for b, y in zip(budgets, ys)]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.set_xlabel("search budget (iterations)")
    ax.set_ylabel("rewrites applied per term")
    ax.legend(loc="upper left")
    ax.set_title("Search work, qwen rules", loc="left", color=INK)
    save(fig, "fig3_transitions", ["engine", "budget", "transitions"], rows)


def fig_final(set_name: str) -> None:
    q = "learned_rules_qwen_all"
    entries = [
        ("MCTS @100, built-in", CK / f"final_mcts_none_{set_name}.csv", "100"),
        ("MCTS @100, qwen", CK / f"final_mcts_{q}_{set_name}.csv", "100"),
        ("egglog, built-in", CK / f"final_egglog_none_{set_name}.csv", None),
        ("egglog, qwen", CK / f"final_egglog_{q}_{set_name}.csv", None),
        ("egglog then MCTS @100, qwen", CK / f"final_egglogmcts_{q}_{set_name}.csv", "100"),
    ]
    for seedfile, label in ((f"final_{set_name}.csv", "AlphaZero @100, qwen"),
                            (f"final_{set_name}_hybrid.csv", "egglog then AZ @100, qwen"),
                            (f"final_{set_name}_azegg.csv", "AZ @100 then egglog, qwen"),
                            (f"final_{set_name}_azrules.csv", "egglog on AZ-chosen rules, qwen")):
        paths = sorted(glob.glob(str(CK / f"asm_v4_llm_s*/{seedfile}")))
        values = []
        for p in paths:
            rows = [r for r in csv.DictReader(open(p)) if r["budget"] == "100"
                    and (seedfile != f"final_{set_name}.csv" or r["engine"] == "AlphaZero")]
            if rows:
                values.append(sum(float(r["improvement"]) for r in rows) / len(rows))
        if values:
            entries.append((f"{label} ({len(values)} seed{'s' if len(values) > 1 else ''})", sum(values) / len(values), None))
    data = []
    for label, source, budget in entries:
        value = source if isinstance(source, float) else mean_improvement(source, budget)
        if value is not None:
            data.append((label, value))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(data) + 0.8))
    labels = [d[0] for d in data][::-1]
    values = [d[1] for d in data][::-1]
    ax.barh(labels, values, color=BLUE, height=0.6)
    for y, v in enumerate(values):
        ax.text(v + 0.4, y, f"{v:.1f}", va="center", color=INK, fontsize=8)
    ax.set_xlim(0, 100)  # bars start at zero
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean improvement (%)")
    n = {"heldout": "1,722 held-out", "stress": "784 stress"}[set_name]
    ax.set_title(f"Final evaluation, {n} terms", loc="left", color=INK)
    save(fig, f"fig4_final_{set_name}", ["method", "improvement"], [[l, round(v, 2)] for l, v in data])


def main() -> None:
    fig_training()
    fig_az_vs_mcts()
    fig_transitions()
    fig_final("heldout")
    fig_final("stress")


if __name__ == "__main__":
    main()
