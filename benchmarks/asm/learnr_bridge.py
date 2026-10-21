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


def compile_program(source: str, learnr_path: str | Path) -> tuple[dict[str, Expr], bool]:
    parse, lower_to_ir = learnr_frontend(learnr_path)
    return ir_to_terms(lower_to_ir(parse(source)))


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


def random_program(rng: Random, statements: int | None = None) -> str:
    """A straight-line Learn-R program over free inputs with planted redundancy."""
    inputs = [f"in{i}" for i in range(rng.randint(1, 3))]
    names = list(inputs)
    lines = []
    for index in range(statements or rng.randint(3, 7)):
        name = f"v{index}"
        if names[len(inputs) :] and rng.random() < 0.2:
            lines.append(f"{name} = {rng.choice(names[len(inputs) :])}")  # copy chain
        else:
            lines.append(f"{name} = {_expr(rng, names, rng.randint(1, 3))}")
        names.append(name)
    return "\n".join(lines) + "\n"


def program_terms(source: str, learnr_path: str | Path, max_nodes: int = 80) -> list[Expr]:
    from .language import AsmLanguage

    language = AsmLanguage()
    terms, _ = compile_program(source, learnr_path)
    keep = []
    for term in terms.values():
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
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": (train_programs, seed), "test": (test_programs, seed + 104_729)}
    collected: dict[str, list[Expr]] = {}
    for split, (count, split_seed) in splits.items():
        rng = Random(split_seed)
        terms: list[Expr] = []
        seen = set()
        for _ in range(count):
            for term in program_terms(random_program(rng), learnr_path):
                if term not in seen:
                    seen.add(term)
                    terms.append(term)
        collected[split] = terms
    train = set(collected["train"])
    collected["test"] = [t for t in collected["test"] if t not in train]
    for split, terms in collected.items():
        with (out_dir / f"{split}.jsonl").open("w") as handle:
            for term in terms:
                handle.write(json.dumps({"sexpr": to_sexpr(term)}) + "\n")
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
    args = parser.parse_args(argv)
    counts = export_corpus(args.learnr, args.train_programs, args.test_programs, args.seed)
    programs = export_programs(args.learnr)
    print(f"corpus terms: {counts}; Learn-R test programs: {sorted(programs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
