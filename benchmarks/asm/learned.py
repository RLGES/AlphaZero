"""Import Learn-R's LLM-generated rules as general, verified pattern rules.

Learn-R stores learned rules as instruction sequences over concrete registers
(``MOV t2, start`` / ``ADD t2, t1``), so they only match the program they came
from. Here each sequence is simulated symbolically: registers read before they
are written become pattern variables, and the rule's meaning is the value of
the last register each side writes. A rule is accepted only if it is
non-trivial, binds every right-hand variable, and passes ``verify_rule``.
Accepted and rejected rules (with reasons) go to ``learned_rules.json``, which
``AsmLanguage`` loads as extra actions.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from random import Random

from benchmarks.common import TreeRule

from .patterns import parse_pattern, pattern_rule, verify_rule

LEARNED_PATH = Path(__file__).with_name("learned_rules.json")

TWO_OPERAND = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "imul": "mul",
    "and": "and",
    "or": "or",
    "xor": "xor",
    "shl": "shl",
    "sal": "shl",
    "shr": "shr",
}
ONE_OPERAND = {"neg": "neg", "not": "not"}


class UnsupportedRule(ValueError):
    pass


def _operand(text: str, registers: dict[str, str]) -> str:
    text = text.strip()
    try:
        return str(int(text, 0))
    except ValueError:
        pass
    if not text.replace("_", "").isalnum():
        raise UnsupportedRule(f"unsupported operand {text!r}")
    return registers.get(text.lower(), f"?{text.lower()}")


def simulate(sequence: list[str]) -> str:
    """Symbolic value of the last written register, as a pattern s-expression."""
    registers: dict[str, str] = {}
    last = None
    for line in sequence:
        line = line.strip().rstrip(";")
        if not line:
            continue
        opcode, _, rest = line.partition(" ")
        opcode = opcode.lower()
        operands = [part.strip() for part in rest.split(",")] if rest else []
        if opcode == "mov" and len(operands) == 2:
            value = _operand(operands[1], registers)
        elif opcode in TWO_OPERAND and len(operands) == 2:
            dest = _operand(operands[0], registers)
            value = f"({TWO_OPERAND[opcode]} {dest} {_operand(operands[1], registers)})"
        elif opcode in TWO_OPERAND and len(operands) == 3:
            a, b = (_operand(op, registers) for op in operands[1:])
            value = f"({TWO_OPERAND[opcode]} {a} {b})"
        elif opcode in ONE_OPERAND and len(operands) == 1:
            value = f"({ONE_OPERAND[opcode]} {_operand(operands[0], registers)})"
        elif opcode in ("inc", "dec") and len(operands) == 1:
            op = "add" if opcode == "inc" else "sub"
            value = f"({op} {_operand(operands[0], registers)} 1)"
        else:
            raise UnsupportedRule(f"unsupported instruction {line!r}")
        last = operands[0].strip().lower()
        registers[last] = value
    if last is None:
        raise UnsupportedRule("empty sequence")
    return registers[last]


def generalize(lhs_seq: list[str], rhs_seq: list[str], rng: Random) -> tuple[str, str]:
    """Return (lhs, rhs) patterns, or raise UnsupportedRule with the reason."""
    lhs, rhs = simulate(lhs_seq), simulate(rhs_seq)
    if not lhs.startswith("("):
        raise UnsupportedRule("left side is a bare register or constant")
    if parse_pattern(lhs) == parse_pattern(rhs):
        raise UnsupportedRule("no-op: both sides compute the same expression")
    if not verify_rule(lhs, rhs, rng):
        raise UnsupportedRule("unsound or unbound: sides disagree on some inputs")
    return lhs, rhs


INSTRUCTION = re.compile(r"^[A-Za-z]{2,7}(\s+[\w\[\]+\-, ]+)?$")


def parse_llm_text(text: str) -> list[tuple[list[str], list[str]]]:
    """Rules from Learn-R LLM output: ``LHS:``/``RHS:`` blocks or inline lists.

    ``; comments`` are dropped, and a block ends at the first line that does not
    look like an instruction (pipeline log lines, labels, explanations).
    """
    rules: list[tuple[list[str], list[str]]] = []
    side = None
    lhs: list[str] = []
    rhs: list[str] = []

    def flush() -> None:
        if lhs and rhs:
            rules.append((list(lhs), list(rhs)))
        lhs.clear()
        rhs.clear()

    for raw in text.splitlines():
        line = raw.strip()
        prefix = line[:4] if line.upper().startswith(("LHS:", "RHS:")) else ""
        body = line[len(prefix) :].strip()
        if not body.startswith("["):  # list items carry their own "; comment"
            body = body.split(";", 1)[0].strip()
        line = f"{prefix} {body}".strip()
        upper = line.upper()
        if upper.startswith("LHS:"):
            flush()
            side, rest = "lhs", line[4:].strip()
        elif upper.startswith("RHS:"):
            side, rest = "rhs", line[4:].strip()
        elif not line or upper.startswith(("CONDITION", "EXPLANATION", "RULE")):
            if side == "rhs":
                flush()
            side = None
            continue
        else:
            rest = line
        if side is None or not rest:
            continue
        target = lhs if side == "lhs" else rhs
        if rest.startswith("["):
            try:
                items = [str(item).split(";", 1)[0].strip() for item in ast.literal_eval(rest)]
                target.extend(item for item in items if item)
            except (ValueError, SyntaxError):
                pass
        elif INSTRUCTION.match(rest):
            target.append(rest)
        else:
            if side == "rhs":
                flush()
            side = None
    flush()
    return rules


def collect_rules(paths: list[Path]) -> list[tuple[list[str], list[str]]]:
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    rules = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".json":
            found = [(r["lhs_seq"], r["rhs_seq"]) for r in json.loads(text).get("rules", [])]
        else:
            found = parse_llm_text(text)
        for lhs, rhs in found:
            key = (tuple(lhs), tuple(rhs))
            if key not in seen:
                seen.add(key)
                rules.append((lhs, rhs))
    return rules


def import_rules(paths: list[Path], builtin: set[tuple[str, str]], out: Path = LEARNED_PATH):
    """Generalize and verify rules from ``paths``; write and return the report."""
    rng = Random(0)
    accepted, rejected = [], []
    for lhs_seq, rhs_seq in collect_rules(paths):
        try:
            lhs, rhs = generalize(lhs_seq, rhs_seq, rng)
            if (lhs, rhs) in builtin or any((lhs, rhs) == (a["lhs"], a["rhs"]) for a in accepted):
                raise UnsupportedRule("duplicate of an existing rule")
        except UnsupportedRule as exc:
            rejected.append({"lhs_seq": lhs_seq, "rhs_seq": rhs_seq, "reason": str(exc)})
            continue
        name = f"learned_{len(accepted)}"
        accepted.append(
            {"name": name, "lhs": lhs, "rhs": rhs, "lhs_seq": lhs_seq, "rhs_seq": rhs_seq}
        )
    report = {"sources": [str(p) for p in paths], "accepted": accepted, "rejected": rejected}
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def learned_path() -> Path | None:
    """``RLGES_ASM_LEARNED``: a rules file, or ``none`` for built-in rules only."""
    choice = os.environ.get("RLGES_ASM_LEARNED", "")
    if choice.lower() == "none":
        return None
    return Path(choice) if choice else LEARNED_PATH


def load_learned_rules(path: Path | None = LEARNED_PATH) -> tuple[TreeRule, ...]:
    if path is None or not path.exists():
        return ()
    report = json.loads(path.read_text())
    return tuple(pattern_rule(r["name"], r["lhs"], r["rhs"]) for r in report["accepted"])
