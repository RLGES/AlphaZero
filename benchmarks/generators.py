"""Training-term generation for AlphaZero self-play.

Hand-written benchmarks stay held out: generated terms equal to any of them are
discarded, as are duplicates and terms where no rule applies. Generators that
accept a ``split`` argument (asm) draw held-out evaluation terms from separate
data when called with ``split="test"``; the others rely on a distinct seed.
"""

from importlib import import_module
from inspect import signature
from random import Random

from .registry import load_suite

GENERATOR_SUITES = ("arithmetic", "boolean", "bitvector", "asm")


def generate(
    suite: str,
    rng: Random,
    count: int,
    depth: tuple[int, int] = (2, 5),
    split: str = "train",
) -> list:
    if suite not in GENERATOR_SUITES:
        raise ValueError(
            f"No term generator for '{suite}'. Choose from: {', '.join(GENERATOR_SUITES)}"
        )
    module_term = import_module(f"benchmarks.{suite}.generator").random_term
    if "split" in signature(module_term).parameters:

        def random_term(rng: Random, depth: int):
            return module_term(rng, depth, split=split)
    else:
        random_term = module_term
    loaded = load_suite(suite)
    language = loaded.language
    held_out = {benchmark.term for benchmark in loaded.benchmarks}
    terms: list = []
    seen: set = set()
    attempts = 0
    while len(terms) < count:
        attempts += 1
        if attempts > 100 * count + 1000:
            raise RuntimeError(f"Could not generate {count} distinct '{suite}' terms")
        term = random_term(rng, rng.randint(*depth))
        if term in held_out or term in seen or language.size(term) > language.max_nodes:
            continue
        if not language.applicable(term):
            continue
        seen.add(term)
        terms.append(term)
    return terms
