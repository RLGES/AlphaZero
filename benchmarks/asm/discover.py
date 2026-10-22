"""Discover new rewrite rules with a local LLM, then prove them with Z3.

Learn-R asks its LLM for instruction-level rules over concrete registers, which
only match the program they came from (see ``learned``). This driver instead
shows the model "stuck" terms (what MCTS with the built-in rules leaves behind)
and asks for general s-expression pattern rules. Every proposal must parse, use
known operators, bind every right-hand variable, be new (modulo variable
renaming), agree on random 32-bit inputs and have a Z3 proof; anything else is
rejected with a reason. The report keeps both lists, so acceptance rates are a
result in themselves; ``AsmLanguage`` loads ``accepted`` when pointed at it via
``RLGES_ASM_LEARNED``.

    python -m benchmarks.asm.discover --prompts 150 --out benchmarks/asm/learned_rules_qwen.json

The server is any OpenAI-compatible endpoint (LM Studio by default):
``RLGES_LLM_URL`` (http://localhost:1234/v1) and ``RLGES_LLM_MODEL``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random

from .patterns import Pattern, match, parse_pattern, variables, verify_rule
from .rules import PATTERN_RULES
from .term import BINARY, UNARY, Expr, to_sexpr

DEFAULT_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen2.5-coder-7b-instruct"
DEFAULT_OUT = Path(__file__).with_name("learned_rules_qwen.json")
OP_COST = {"mul": 3, "div": 10, "mod": 10, "load": 4}

SYSTEM = "You are an expert in compiler term rewriting. You output only JSON lines."
PROMPT = """We optimise integer expressions written as s-expressions.
Operators (32-bit unsigned wrap-around; div/mod by 0 give 0; shift counts use the low 5 bits):
  (add a b) (sub a b) (mul a b) (div a b) (mod a b) (and a b) (or a b) (xor a b)
  (shl a b) (shr a b) (neg a) (not a) (load a)
Pattern variables: ?x ?y ?z. Constants: integer literals.
Cost: mul=3, div=10, mod=10, load=4, every other node=1. A good rule lowers cost
or enables rules that lower cost (e.g. by exposing constants for folding).

FORBIDDEN: registers (eax, t3, r1 ...), cmp, test, jmp/jcc, cmov, set*, labels,
instruction syntax, conditions/preconditions. Rules must hold for ALL inputs.

Already known (do NOT repeat): x+0, x*1, x*0, x-x, x^x, x&x, x|x, identities with 0,
double neg/not, x*2^k -> shl, x/2, x/4 -> shr, x+x -> shl, commutativity of add/mul/and/or/xor,
associativity of add/mul/and/or, folding of constant operations.

These expressions are NOT simplified well by the known rules:
{terms}

Propose {count} NEW general rewrite rules that would help simplify expressions like these.
Output one JSON object per line, nothing else:
{{"name": "...", "lhs": "(mul (add ?x ?y) ?z)", "rhs": "(add (mul ?x ?z) (mul ?y ?z))"}}

Good examples:
{{"name":"factor_mul","lhs":"(add (mul ?x ?y) (mul ?x ?z))","rhs":"(mul ?x (add ?y ?z))"}}
{{"name":"add_sub_cancel","lhs":"(sub (add ?x ?y) ?y)","rhs":"?x"}}
Bad (unsound): {{"lhs":"(div (mul ?x ?y) ?y)","rhs":"?x"}}   (fails when y=0 or on overflow)"""

Complete = Callable[[str, float], str]


def http_complete(url: str, model: str, timeout: float = 180) -> Complete:
    def complete(prompt: str, temperature: float) -> str:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 700,
        }
        request = urllib.request.Request(
            f"{url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"]

    return complete


def check_server(url: str) -> list[str]:
    """Model ids the server offers; raises SystemExit when it is unreachable."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/models", timeout=5) as response:
            return [m["id"] for m in json.load(response)["data"]]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"LLM server at {url} is unreachable ({exc}); start it first") from exc


JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_proposals(text: str) -> list[dict]:
    """JSON objects with string ``lhs``/``rhs`` anywhere in the reply."""
    found = []
    for chunk in JSON_OBJECT.findall(text):
        try:
            item = json.loads(chunk)
        except ValueError:
            continue
        if (
            isinstance(item, dict)
            and isinstance(item.get("lhs"), str)
            and isinstance(item.get("rhs"), str)
        ):
            found.append(
                {"name": str(item.get("name", "")), "lhs": item["lhs"], "rhs": item["rhs"]}
            )
    return found


def _well_formed(pattern: Pattern) -> bool:
    if isinstance(pattern, (str, int)):
        return True
    op, *args = pattern
    arity = 2 if op in BINARY else 1 if op in UNARY else None
    return arity == len(args) and all(_well_formed(arg) for arg in args)


def _canonical(lhs: Pattern, rhs: Pattern) -> str:
    """Both sides with variables renamed in order of first use on the left."""
    names: dict[str, str] = {}

    def rename(pattern: Pattern) -> Pattern:
        if isinstance(pattern, str):
            return names.setdefault(pattern, f"?v{len(names)}")
        if isinstance(pattern, int):
            return pattern & 0xFFFF_FFFF
        return (pattern[0], *(rename(arg) for arg in pattern[1:]))

    return repr((rename(lhs), rename(rhs)))


def pattern_cost(pattern: Pattern) -> int:
    if not isinstance(pattern, tuple):
        return 1
    return OP_COST.get(pattern[0], 1) + sum(pattern_cost(arg) for arg in pattern[1:])


def _subterms(term: Expr):
    yield term
    for arg in term.args:
        yield from _subterms(arg)


def check_proposal(proposal: dict, known: set[str], rng: Random, prove: bool) -> tuple[str, dict]:
    """("accepted", info) or (rejection reason, info)."""
    try:
        lhs, rhs = parse_pattern(proposal["lhs"]), parse_pattern(proposal["rhs"])
    except (ValueError, IndexError):
        return "unparseable", {}
    if not (_well_formed(lhs) and _well_formed(rhs)):
        return "unknown operator or arity", {}
    if not isinstance(lhs, tuple):
        return "left side is a bare variable or constant", {}
    if not variables(rhs) <= variables(lhs):
        return "right side uses unbound variables", {}
    key = _canonical(lhs, rhs)
    if lhs == rhs:
        return "no-op", {}
    if key in known:
        return "duplicate", {}
    if not verify_rule(proposal["lhs"], proposal["rhs"], rng):
        return "unsound (random 32-bit test)", {}
    info = {"lhs_cost": pattern_cost(lhs), "rhs_cost": pattern_cost(rhs)}
    if prove:
        from .smt import prove_rule

        verdict, detail = prove_rule(proposal["lhs"], proposal["rhs"])
        if verdict != "proved":
            return f"unsound (z3 {verdict})", {**info, "z3": detail}
    return "accepted", info


def stuck_terms(count: int, seed: int = 0, budget: int = 50) -> list[Expr]:
    """Terms after MCTS with the built-in rules, the costliest residues first."""
    from benchmarks.generators import generate
    from src.api import rewrite_detailed
    from src.engines.mcts import MCTSConfig

    from .language import AsmLanguage

    language = AsmLanguage()
    rng = Random(seed)
    residues = {}
    for term in generate("asm", rng, count * 3, (3, 6)):
        final = rewrite_detailed(language, term, MCTSConfig(iterations=budget, seed=0)).final_term
        if final.op not in ("var", "const") and language.size(final) <= 40:
            residues[final] = language.cost(final)
    ranked = sorted(residues, key=residues.get, reverse=True)
    return ranked[:count]


def discover(
    complete: Complete,
    seeds: Sequence[Expr],
    prompts: int,
    *,
    per_prompt: int = 5,
    shown: int = 4,
    temperature: float = 0.8,
    workers: int = 4,
    max_accepted: int = 20,
    seed: int = 0,
    prove: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    rng = Random(seed)
    requests = []
    for index in range(prompts):
        chosen = rng.sample(list(seeds), min(shown, len(seeds)))
        text = PROMPT.format(terms="\n".join(f"  {to_sexpr(t)}" for t in chosen), count=per_prompt)
        requests.append((index, chosen, text))

    def ask(request):
        index, chosen, text = request
        try:
            return index, chosen, complete(text, temperature), None
        except Exception as exc:  # one failed request must not stop the run
            return index, chosen, "", f"{type(exc).__name__}: {exc}"

    known = {_canonical(parse_pattern(lhs), parse_pattern(rhs)) for _, lhs, rhs in PATTERN_RULES}
    stats: Counter = Counter()
    verified, rejected = [], []
    check_rng = Random(seed + 1)
    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for index, chosen, reply, error in pool.map(ask, requests):
            stats["prompts"] += 1
            if error:
                stats["request errors"] += 1
                log(f"prompt {index}: {error}")
                continue
            proposals = parse_proposals(reply)
            stats["proposals"] += len(proposals)
            if not proposals:
                stats["replies without rules"] += 1
            for proposal in proposals:
                reason, info = check_proposal(proposal, known, check_rng, prove)
                entry = {**proposal, **info, "prompt": index}
                if reason == "accepted":
                    known.add(
                        _canonical(parse_pattern(proposal["lhs"]), parse_pattern(proposal["rhs"]))
                    )
                    entry["seeds"] = [to_sexpr(t) for t in chosen]
                    verified.append(entry)
                else:
                    rejected.append({**entry, "reason": reason})
                stats[reason] += 1
            log(
                f"prompt {index + 1}/{prompts}: {len(proposals)} proposals, "
                f"{len(verified)} verified so far ({time.time() - started:.0f}s)"
            )
    usage = Counter()
    for seed_term in seeds:
        for sub in _subterms(seed_term):
            for rule in verified:
                if match(parse_pattern(rule["lhs"]), sub) is not None:
                    usage[rule["lhs"], rule["rhs"]] += 1
    for rule in verified:
        rule["seed_matches"] = usage[rule["lhs"], rule["rhs"]]
        rule["lowers_cost"] = rule["rhs_cost"] < rule["lhs_cost"]
    # Load the rules most likely to matter: those that fire on stuck terms, then cheaper ones.
    ranked = sorted(verified, key=lambda r: (-r["seed_matches"], r["rhs_cost"] - r["lhs_cost"]))
    accepted = [
        {**rule, "name": f"llm_{i}_{re.sub(r'[^a-z0-9_]', '', rule['name'].lower())[:24]}"}
        for i, rule in enumerate(ranked[:max_accepted])
    ]
    return {
        "stats": dict(stats),
        "accepted": accepted,
        "verified_not_loaded": ranked[max_accepted:],
        "rejected": rejected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM rule discovery with Z3 verification.")
    parser.add_argument("--prompts", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=120, help="Stuck terms to sample from")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-accepted", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    url = os.environ.get("RLGES_LLM_URL", DEFAULT_URL)
    model = os.environ.get("RLGES_LLM_MODEL", DEFAULT_MODEL)
    models = check_server(url)
    if model not in models:
        raise SystemExit(f"model {model!r} not served at {url}; available: {models}")
    from .smt import z3_available

    if not z3_available():
        raise SystemExit("z3-solver is not installed; install it to prove rules")
    seeds = stuck_terms(args.seeds)
    print(f"{len(seeds)} stuck seed terms; asking {model} at {url}", flush=True)
    report = discover(
        http_complete(url, model),
        seeds,
        args.prompts,
        temperature=args.temperature,
        workers=args.workers,
        max_accepted=args.max_accepted,
        log=lambda line: print(line, flush=True),
    )
    report = {"model": model, "url": url, "temperature": args.temperature, **report}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["stats"], indent=2))
    for rule in report["accepted"]:
        print(f"  {rule['lhs']} -> {rule['rhs']}  (cost {rule['lhs_cost']}->{rule['rhs_cost']})")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
