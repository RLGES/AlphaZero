"""Bridge to Learn-R: compile programs with its frontend into asm terms.

Learn-R's source language has ``+ - *``, parentheses, integers, assignments and
if/while. ``frontend.parse`` + ``lower_to_ir`` emit IR such as ``t0 = 5``,
``a = t0``, ``t2 = a + t1``; this module inlines those definitions into one
expression tree per program variable. Variables that are read but never
assigned (``in0``, ``in1``, ...) are free inputs, like function parameters, so
constant folding alone cannot solve the program.

``export_corpus`` turns many random programs into the training and test term
files under ``benchmarks/asm/corpus/``; training then needs no Learn-R checkout.

    LEARNR_PATH=~/Projects/RLGES/Learn-R python -m benchmarks.asm.learnr_bridge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from random import Random

from .term import Const, Expr, Var, to_sexpr

CORPUS_DIR = Path(__file__).with_name("corpus")
PROGRAMS_PATH = Path(__file__).with_name("learnr_programs.json")
DEFAULT_LEARNR = Path.home() / "Projects" / "RLGES" / "Learn-R"
TEMP = re.compile(r"^t\d+$")
OPS = {"+": "add", "-": "sub", "*": "mul"}


def learnr_frontend(learnr_path: str | Path):
    path = str(Path(learnr_path).expanduser())
    if not Path(path, "frontend").is_dir():
        raise FileNotFoundError(f"No Learn-R checkout at {path} (set LEARNR_PATH)")
    if path not in sys.path:
        sys.path.insert(0, path)
    from frontend import lower_to_ir, parse  # type: ignore[import-not-found]

    return parse, lower_to_ir


def ir_to_terms(ir: list[str]) -> tuple[dict[str, Expr], bool]:
    """Inline IR assignments; returns (user variable -> term, straight_line)."""
    env: dict[str, Expr] = {}
    straight_line = True

    def operand(text: str) -> Expr:
        text = text.strip()
        if text.lstrip("-").isdigit():
            return Const(int(text))
        return env.get(text, Var(text))

    for line in ir:
        dest, eq, rhs = line.partition(" = ")
        if not eq:
            straight_line = False  # CMP, jumps and labels: control flow
            continue
        parts = rhs.split()
        if len(parts) == 3 and parts[1] in OPS:
            env[dest.strip()] = Expr(OPS[parts[1]], (operand(parts[0]), operand(parts[2])))
        elif len(parts) == 1:
            env[dest.strip()] = operand(parts[0])
        else:
            straight_line = False
    return {name: term for name, term in env.items() if not TEMP.match(name)}, straight_line


LABEL = re.compile(r"^L\d+:$")


def ir_to_block_terms(ir: list[str]) -> list[dict[str, Expr]]:
    """Split IR into basic blocks and inline each one on its own.

    A block starts at a label and ends after a jump. Values that flow into a
    block (loop-carried or set by a branch) are free variables there, so each
    block's terms are exact for any entry state. ``CMP`` operands are kept as
    terms too (``cmp0``, ``cmp1``), since the comparison computes them.
    """
    blocks: list[list[str]] = [[]]
    for line in ir:
        line = line.strip()
        if LABEL.match(line):
            blocks.append([])
        elif line.startswith("J"):
            blocks.append([])
        elif line.startswith("CMP "):
            left, _, right = line[4:].partition(",")
            blocks[-1] += [f"cmp0 = {left.strip()}", f"cmp1 = {right.strip()}"]
        else:
            blocks[-1].append(line)
    return [terms for terms, _ in map(ir_to_terms, blocks) if terms]


def compile_program(source: str, learnr_path: str | Path) -> tuple[dict[str, Expr], bool]:
    parse, lower_to_ir = learnr_frontend(learnr_path)
    return ir_to_terms(lower_to_ir(parse(source)))


def compile_blocks(source: str, learnr_path: str | Path) -> list[dict[str, Expr]]:
    parse, lower_to_ir = learnr_frontend(learnr_path)
    return ir_to_block_terms(lower_to_ir(parse(source)))


def _atom(rng: Random, names: list[str]) -> str:
    roll = rng.random()
    if roll < 0.7 and names:
        return rng.choice(names)
    return str(rng.choice((0, 1, 2, 3, 4, 5, 8, 16, 10)))


def _expr(rng: Random, names: list[str], depth: int) -> str:
    if depth <= 0:
        return _atom(rng, names)
    a, b = _expr(rng, names, depth - 1), _expr(rng, names, depth - 1)
    shape = rng.randrange(10)
    if shape == 0:
        return f"({a} + 0)"
    if shape == 1:
        return f"({a} * 1)"
    if shape == 2:
        return f"({a} * {rng.choice((2, 4, 8, 16))})"
    if shape == 3:
        return f"({a} - {a})"
    if shape == 4:
        return f"({a} + {a})"
    if shape == 5:
        return f"({a} * 0 + {b})"
    if shape == 6:
        return f"({a} + {rng.randint(1, 9)} + {rng.randint(1, 9)})"
    return f"({a} {rng.choice('+-*')} {b})"


def _hard_expr(rng: Random, names: list[str], depth: int) -> str:
    """Redundancy that needs a non-improving step (commute, reassociate) first."""
    if depth <= 0:
        return _atom(rng, names)
    a, b = _hard_expr(rng, names, depth - 1), _hard_expr(rng, names, depth - 1)
    c1, c2 = rng.randint(1, 9), rng.randint(1, 9)
    shape = rng.randrange(12)
    if shape == 0:
        return f"(({a} + {c1}) + {c2})"  # reassociate, then fold
    if shape == 1:
        return f"(({c1} + {a}) + ({b} + {c2}))"  # commute and reassociate to fold
    if shape == 2:
        return f"(({a} * {rng.choice((2, 4))}) * {rng.choice((2, 4))})"  # fold to one shift
    if shape == 3:
        return f"({c1} * ({a} * {c2}))"
    if shape == 4:
        return f"(({a} + {b}) - ({b} + {a}))"  # commute, then sub_self
    if shape == 5:
        return f"(({a} * {b}) - ({b} * {a}))"
    if shape == 6:
        return f"({a} * ({c1} - {c1} + 1))"  # fold to mul by one
    if shape == 7:
        return f"(({a} + {b}) * ({c1} - {c1}))"  # fold to mul by zero
    if shape == 8:
        return f"({a} * ({rng.choice((1, 2, 4))} * {rng.choice((2, 4))}))"  # fold, then shift
    if shape == 9:
        return f"({a} {rng.choice('+-*')} {b})"
    return _expr(rng, names, depth)  # some classic one-step plants too


def random_program(
    rng: Random, statements: int | None = None, style: str = "classic", control_flow: float = 0.0
) -> str:
    """A Learn-R program over free inputs with planted redundancy.

    ``style="hard"`` writes longer programs whose redundancy mostly needs a
    multi-step rewrite; ``"classic"`` reproduces the original corpus. With
    ``control_flow`` > 0, that share of statements opens an ``if``/``else`` or
    ``while`` whose body holds 2-6 further statements (straight-line otherwise).
    """
    hard = style == "hard"
    expr = _hard_expr if hard else _expr
    inputs = [f"in{i}" for i in range(rng.randint(1, 3))]
    names = list(inputs)
    lines: list[str] = []
    counter = iter(range(10_000))

    def assignment(indent: str) -> None:
        name = f"v{next(counter)}"
        if names[len(inputs) :] and rng.random() < 0.2:
            lines.append(f"{indent}{name} = {rng.choice(names[len(inputs) :])}")  # copy chain
        else:
            lines.append(f"{indent}{name} = {expr(rng, names, rng.randint(1, 3))}")
        names.append(name)

    def body(indent: str) -> None:
        for _ in range(rng.randint(2, 6)):
            assignment(indent)

    for _ in range(statements or (rng.randint(6, 14) if hard else rng.randint(3, 7))):
        if control_flow and rng.random() < control_flow:
            condition = (
                f"{rng.choice(names)} {rng.choice(('<', '>', '==', '!='))} {rng.choice(names)}"
            )
            keyword = rng.choice(("if", "if", "while"))
            lines.append(f"{keyword} ({condition}) {{")
            body("    ")
            if keyword == "if" and rng.random() < 0.6:
                lines.append("} else {")
                body("    ")
            lines.append("}")
        else:
            assignment("")
    return "\n".join(lines) + "\n"


def program_terms(
    source: str, learnr_path: str | Path, max_nodes: int = 80, blocks: bool = False
) -> list[Expr]:
    from .language import AsmLanguage

    language = AsmLanguage()
    if blocks:
        found = [t for block in compile_blocks(source, learnr_path) for t in block.values()]
    else:
        found = list(compile_program(source, learnr_path)[0].values())
    keep = []
    for term in found:
        if term.op in ("var", "const") or language.size(term) > max_nodes:
            continue
        if language.applicable(term) and term not in keep:
            keep.append(term)
    return keep


def export_corpus(
    learnr_path: str | Path,
    train_programs: int = 3000,
    test_programs: int = 300,
    seed: int = 0,
    out_dir: Path = CORPUS_DIR,
    style: str = "classic",
    hard_only: bool = False,
    easy_share: float = 0.15,
    workers: int = 1,
    control_flow: float = 0.0,
    prefix: str = "",
) -> dict[str, int]:
    """Write ``train.jsonl``/``test.jsonl``.

    ``hard_only`` keeps the terms ``hardness.is_hard`` accepts (test: only
    those; train: plus ``easy_share`` of the rest for variety) and stores the
    greedy/quick/strong costs with each term. ``control_flow`` > 0 writes
    if/while programs split into basic blocks; ``prefix`` names the files
    (``cf_`` keeps them beside the straight-line corpus, see ``generator``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    max_nodes = 150 if style == "hard" else 80
    splits = {"train": (train_programs, seed), "test": (test_programs, seed + 104_729)}
    collected: dict[str, list[Expr]] = {}
    for split, (count, split_seed) in splits.items():
        rng = Random(split_seed)
        terms: list[Expr] = []
        seen = set()
        for _ in range(count):
            source = random_program(rng, style=style, control_flow=control_flow)
            for term in program_terms(source, learnr_path, max_nodes, blocks=control_flow > 0):
                if term not in seen:
                    seen.add(term)
                    terms.append(term)
        collected[split] = terms
    train = set(collected["train"])
    collected["test"] = [t for t in collected["test"] if t not in train]
    stats: dict[Expr, dict] = {}
    if hard_only:
        from .hardness import hardness, is_hard

        for split, terms in collected.items():
            print(f"hardness filter: {split}, {len(terms)} candidates", flush=True)
            if workers > 1:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(workers) as pool:
                    found = list(pool.map(hardness, terms, chunksize=16))
            else:
                found = [hardness(term) for term in terms]
            keep_rng = Random(splits[split][1] + 1)
            kept = []
            for term, term_stats in zip(terms, found):
                term_stats["hard"] = is_hard(term_stats)
                easy_ok = split == "train" and keep_rng.random() < easy_share
                if term_stats["hard"] or easy_ok:
                    stats[term] = term_stats
                    kept.append(term)
            collected[split] = kept
    for split, terms in collected.items():
        with (out_dir / f"{prefix}{split}.jsonl").open("w") as handle:
            for term in terms:
                handle.write(json.dumps({"sexpr": to_sexpr(term), **stats.get(term, {})}) + "\n")
    return {split: len(terms) for split, terms in collected.items()}


def export_programs(learnr_path: str | Path, out: Path = PROGRAMS_PATH) -> dict:
    """Compile Learn-R's own test programs into held-out benchmark terms."""
    programs = {}
    for path in sorted(Path(learnr_path).expanduser().glob("test_code*.txt")):
        terms, straight_line = compile_program(path.read_text(), learnr_path)
        programs[path.stem] = {
            "source": path.read_text(),
            "straight_line": straight_line,
            "terms": {name: to_sexpr(term) for name, term in terms.items()},
        }
    out.write_text(json.dumps(programs, indent=2) + "\n")
    return programs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Learn-R programs as asm terms.")
    parser.add_argument("--learnr", default=os.environ.get("LEARNR_PATH", str(DEFAULT_LEARNR)))
    parser.add_argument("--train-programs", type=int, default=3000)
    parser.add_argument("--test-programs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--style", choices=("classic", "hard"), default="classic")
    parser.add_argument("--hard-only", action="store_true", help="Keep terms quick search misses")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--control-flow", type=float, default=0.0, help="Share of if/while")
    parser.add_argument("--prefix", default="", help="Corpus file prefix, e.g. cf_")
    args = parser.parse_args(argv)
    counts = export_corpus(
        args.learnr,
        args.train_programs,
        args.test_programs,
        args.seed,
        style=args.style,
        hard_only=args.hard_only,
        workers=args.workers,
        control_flow=args.control_flow,
        prefix=args.prefix,
    )
    programs = export_programs(args.learnr)
    print(f"corpus terms: {counts}; Learn-R test programs: {sorted(programs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
