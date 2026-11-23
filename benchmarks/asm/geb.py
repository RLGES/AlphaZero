"""AlphaZero on e-graphs for the asm suite: MCTS-GEB with a learned policy and value.

MCTS-GEB (He, Singh, Yoneki, EuroMLSys '23) treats e-graph construction as a
planning problem: the state is an e-graph, an action applies one rewrite rule for
one iteration, the episode ends at an e-node limit, and the reward is the cost
reduction of the extracted term. It plans with UCT and random rollouts. Here:

- ``EGraphEnv`` wraps egglog. Tree search uses ``push``/``pop`` snapshots instead of
  replaying action sequences. Constant folding runs after every action.
- ``search`` is PUCT. ``mode="az"`` takes priors and a value from the term network
  (``PolicyValueNet``) evaluated on the currently extracted term; ``mode="geb"`` is
  the MCTS-GEB baseline (uniform priors, random rollouts).
- ``build`` plays one episode (real moves until the node or step limit) and returns
  the best extracted cost seen, plus training samples.
- ``egglog_schedule`` is egglog's default schedule under the same e-node limit.

The action space is ``AsmLanguage.rules`` (the same vocabulary as the term-level
AlphaZero), so a trained term network initialises the e-graph agent directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random

from egglog import EGraph, ruleset

from .egglog_adapter import AsmAdapter, learned_named
from .language import AsmLanguage
from .rules import PATTERN_RULES

_RULESETS: dict[str, object] = {}


def _rulesets(adapter: AsmAdapter, names: list[str]) -> list:
    """One egglog ruleset per action; FOLD (no pattern) maps to constant folding."""
    patterns = {name: (lhs, rhs) for name, lhs, rhs in list(PATTERN_RULES) + learned_named()}
    if "__fold__" not in _RULESETS:
        _RULESETS["__fold__"] = ruleset(*adapter.fold_rules(), name="geb_fold")
    out = []
    for i, name in enumerate(names):
        if name not in patterns:
            out.append(_RULESETS["__fold__"])
            continue
        if name not in _RULESETS:
            lhs, rhs = patterns[name]
            _RULESETS[name] = ruleset(adapter.rewrite_rule(lhs, rhs), name=f"geb_{i}")
        out.append(_RULESETS[name])
    return out


class EGraphEnv:
    """kind="rule": MCTS-GEB's action space, one action = one iteration of one rule.
    kind="enable": guided saturation, one action = enable one learned rule (or none,
    the FOLD index), then one iteration of the built-in rules plus every enabled
    learned rule. Built-in rules always run; the agent decides which learned
    (LLM-discovered) rules join and when."""

    def __init__(self, language: AsmLanguage, node_limit: int = 2000, max_steps: int = 12,
                 kind: str = "rule"):
        self.language = language
        self.adapter = AsmAdapter()
        self.names = [rule.name for rule in language.action_vocabulary()]
        self.rulesets = _rulesets(self.adapter, self.names)
        self.fold = _RULESETS["__fold__"]
        self.node_limit = node_limit
        self.max_steps = max_steps
        self.kind = kind
        builtin = {name for name, _, _ in PATTERN_RULES}
        self.noop = next(i for i, n in enumerate(self.names) if n not in builtin
                         and n not in {r[0] for r in learned_named()})
        if kind == "enable":
            if "__builtin__" not in _RULESETS:
                _RULESETS["__builtin__"] = ruleset(
                    *self.adapter.fold_rules(),
                    *(self.adapter.rewrite_rule(lhs, rhs) for _, lhs, rhs in PATTERN_RULES),
                    name="geb_builtin",
                )
            self.builtin = _RULESETS["__builtin__"]
            learned = {r[0] for r in learned_named()}
            self.valid = [i for i, n in enumerate(self.names) if n in learned] + [self.noop]
        else:
            self.valid = list(range(len(self.names)))
        self.active: tuple[int, ...] = ()

    def snapshot(self):
        self.egraph.push()
        return (self.steps, self.active)

    def restore(self, saved) -> None:
        self.egraph.pop()
        self.steps, self.active = saved

    def reset(self, term) -> None:
        self.egraph = EGraph()
        self.root = self.adapter.encode(term)
        self.egraph.register(self.root)
        self.egraph.run(self.fold)
        self.initial_cost = self.language.cost(term)
        self.steps = 0
        self.active = ()

    def apply(self, action: int) -> None:
        if self.kind == "enable":
            if action != self.noop and action not in self.active:
                self.active = self.active + (action,)
            schedule = self.builtin
            for i in self.active:
                schedule = schedule | self.rulesets[i]
            self.egraph.run(1, ruleset=schedule)
        else:
            self.egraph.run(self.rulesets[action])
            self.egraph.run(self.fold)
        self.steps += 1

    def size(self) -> int:
        return sum(size for _, size in self.egraph.all_function_sizes())

    def extract(self):
        expr = self.egraph.extract(self.root)  # static costs, no Python callback
        term = self.adapter.decode(expr)
        return term, self.language.cost(term)

    def terminal(self, size: int | None = None) -> bool:
        size = self.size() if size is None else size
        return size >= self.node_limit or self.steps >= self.max_steps


@dataclass
class _Node:
    priors: list[float] | None = None
    visits: list[int] = field(default_factory=list)
    value_sum: list[float] = field(default_factory=list)
    children: dict[int, _Node] = field(default_factory=dict)
    dead: set[int] = field(default_factory=set)  # actions that left the e-graph unchanged
    total: int = 0


def _uniform(width: int) -> list[float]:
    return [1.0 / width] * width


class Agent:
    """Priors and values for e-graph states, from the extracted term."""

    def __init__(self, language: AsmLanguage, net=None):
        self.language = language
        self.width = len(language.action_vocabulary())
        self.evaluator = None
        if net is not None:
            from src.engines.alphazero.inference import NetworkEvaluator

            self.evaluator = NetworkEvaluator(net)

    def evaluate(self, term) -> tuple[list[float], float]:
        if self.evaluator is None:
            return _uniform(self.width), 0.0
        ((prior, value),) = self.evaluator.evaluate(self.language, [term])
        return [float(p) for p in prior], float(value)


@dataclass
class SearchConfig:
    simulations: int = 32
    c_puct: float = 1.5
    mode: str = "az"  # "az" (network priors + value) or "geb" (uniform + rollouts)
    rollout_depth: int = 20
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0


class _Tracker:
    def __init__(self, cost: float, term):
        self.best_cost, self.best_term = cost, term

    def see(self, term, cost: float) -> None:
        if cost < self.best_cost:
            self.best_cost, self.best_term = cost, term


def _leaf_value(env: EGraphEnv, agent: Agent, node: _Node, cfg: SearchConfig, rng: Random,
                tracker: _Tracker) -> float:
    """Value of the current e-graph as a fraction of the episode's initial cost saved."""
    term, cost = env.extract()
    tracker.see(term, cost)
    c0 = env.initial_cost
    now = (c0 - cost) / c0 if c0 else 0.0
    size = env.size()
    if env.terminal(size):
        return now
    if cfg.mode == "az":
        prior, value = agent.evaluate(term)
        node.priors = prior
        # value: fraction of the extracted term's cost still to be saved
        return 1.0 - (cost / c0) * (1.0 - max(0.0, min(1.0, value))) if c0 else 0.0
    node.priors = _uniform(agent.width)
    best = cost
    for _ in range(cfg.rollout_depth):
        env.apply(rng.choice(env.valid))
        if env.terminal():
            break
    term, cost = env.extract()
    tracker.see(term, cost)
    best = min(best, cost)
    return (c0 - best) / c0 if c0 else 0.0


def _init(node: _Node, width: int) -> None:
    if not node.visits:
        node.visits = [0] * width
        node.value_sum = [0.0] * width


def search(env: EGraphEnv, agent: Agent, root: _Node, cfg: SearchConfig, rng: Random,
           tracker: _Tracker) -> _Node:
    """Run simulations from the current e-graph (restored after each one)."""
    width = agent.width
    if root.priors is None:
        saved = env.snapshot()
        _leaf_value(env, agent, root, cfg, rng, tracker)
        env.restore(saved)
    _init(root, width)
    if cfg.dirichlet_epsilon > 0:
        noise = [rng.gammavariate(cfg.dirichlet_alpha, 1.0) for _ in range(width)]
        total = sum(noise)
        eps = cfg.dirichlet_epsilon
        root.priors = [(1 - eps) * p + eps * n / total for p, n in zip(root.priors, noise)]
    valid = set(env.valid)
    for _ in range(cfg.simulations):
        saved = env.snapshot()
        node, path = root, []
        value = None
        while True:
            _init(node, width)
            scale = cfg.c_puct * math.sqrt(max(1, node.total))

            def score(a: int, node=node, scale=scale) -> float:
                if a in node.dead or a not in valid:
                    return -1e9
                n = node.visits[a]
                q = node.value_sum[a] / n if n else 0.0
                return q + scale * node.priors[a] / (1 + n)

            action = max(range(width), key=score)
            before = env.size()
            env.apply(action)
            after = env.size()
            path.append((node, action))
            if after == before and action not in node.children and env.kind == "rule":
                node.dead.add(action)  # MCTS-GEB's saturated-action pruning
            child = node.children.get(action)
            if child is None:
                child = node.children[action] = _Node()
                value = _leaf_value(env, agent, child, cfg, rng, tracker)
                break
            if env.terminal(after) or child.priors is None:
                term, cost = env.extract()
                tracker.see(term, cost)
                value = (env.initial_cost - cost) / env.initial_cost
                break
            node = child
        for parent, action in path:
            parent.visits[action] += 1
            parent.value_sum[action] += value
            parent.total += 1
        env.restore(saved)
    return root


@dataclass
class Episode:
    initial_cost: float
    best_cost: float
    best_term: object
    steps: int
    final_size: int
    samples: list = field(default_factory=list)  # (term, visit policy, cost at that state)


def build(term, env: EGraphEnv, agent: Agent, cfg: SearchConfig, rng: Random,
          temperature_moves: int = 0, record: bool = False) -> Episode:
    """One e-graph construction episode: plan, take the most visited action, repeat."""
    env.reset(term)
    tracker = _Tracker(env.initial_cost, term)
    root = _Node()
    samples = []
    while not env.terminal():
        root = search(env, agent, root, cfg, rng, tracker)
        visits = root.visits
        if sum(visits) == 0:
            break
        if record:
            current, cost = env.extract()
            total = sum(visits)
            samples.append((current, [v / total for v in visits], cost))
        if env.steps < temperature_moves:
            action = rng.choices(range(len(visits)), weights=visits)[0]
        else:
            action = max(range(len(visits)), key=visits.__getitem__)
        env.apply(action)
        current, cost = env.extract()
        tracker.see(current, cost)
        root = root.children.get(action) or _Node()
    final_size = env.size()
    episode = Episode(env.initial_cost, tracker.best_cost, tracker.best_term, env.steps, final_size)
    if record:
        episode.samples = [
            (t, policy, (cost - tracker.best_cost) / cost if cost else 0.0)
            for t, policy, cost in samples
        ]
    return episode


def egglog_schedule(term, env: EGraphEnv, max_iterations: int = 6) -> Episode:
    """egglog's default schedule (all rules each iteration, back-off scheduler)
    under the same e-node limit: stop at the limit, at saturation, or after
    ``max_iterations`` iterations."""
    from egglog import back_off, run

    env.reset(term)
    full = env.adapter.ruleset()
    schedule = run(full, scheduler=back_off(match_limit=1000, ban_length=2))
    tracker = _Tracker(env.initial_cost, term)
    for i in range(max_iterations):
        before = env.size()
        env.egraph.run(schedule)
        after = env.size()
        current, cost = env.extract()
        tracker.see(current, cost)
        if after >= env.node_limit or after == before:
            break
    return Episode(env.initial_cost, tracker.best_cost, tracker.best_term, i + 1, env.size())
