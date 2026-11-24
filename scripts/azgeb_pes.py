"""AZ-GEB inside the MCTS-GEB framework (their environment, egg engine, rules, expressions).

Only the planner differs from MCTS-GEB: PUCT guided by a small policy/value network instead
of UCT with random rollouts. Their code is used unmodified (pes.environment.base_env,
pes.utils.utils.get_lang); this file only adds a planner.

    cd /root/rlges/external/eqs/pes
    /root/rlges/external/venv/bin/python /root/rlges/external/azgeb_pes.py train MATH out_dir
    /root/rlges/external/venv/bin/python /root/rlges/external/azgeb_pes.py eval MATH out_dir

Training expressions: their generator (depth 5) with seeds 1000+ (the test set is seeds
0-19, as in our MCTS-GEB sweep). State features: cost / base cost, e-nodes / limit, step,
per-rule "saturated last time" flags and the last action. Value target: final size
reduction / base cost (MCTS-GEB's sparse reward, normalised).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from pes.environment.base_env import BaseEnvironment
from pes.utils.utils import get_lang

LIMIT, MAX_STEPS, C_PUCT = 500, 60, 1.5
SIMS = int(os.environ.get("SIMS", "32"))
LR = float(os.environ.get("LR", "1e-3"))
EPISODES = int(os.environ.get("EPISODES", "24"))
ROUNDS = int(os.environ.get("ROUNDS", "30"))


def reward_func(done, info, egraph, expr, base_cost):  # MCTS-GEB's sparse reward
    if done:
        cost, _ = egraph.extract(expr)
        info["ast_size"] = cost
        return max(base_cost - cost, 0)
    return 0


def make_expr(lang_name, seed):
    random.seed(seed)
    np.random.seed(seed)
    lang = get_lang(lang_name)()
    return lang, lang.gen_expr(p_leaf=0., depth_limit=5)


class Net(nn.Module):
    def __init__(self, n_rules: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(3 + 2 * n_rules, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, n_rules)
        self.value = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)


class State:
    """Planner-side bookkeeping on top of the environment (their env holds the e-graph)."""

    def __init__(self, n):
        self.saturated = [0.0] * n
        self.last = -1
        self.steps = 0
        self.done = False

    def copy(self):
        s = State(len(self.saturated))
        s.saturated, s.last, s.steps, s.done = list(self.saturated), self.last, self.steps, self.done
        return s


def features(env, st: State, n: int):
    cost, _ = env.egraph.extract(env.expr)
    enodes = env.egraph.num_enodes() if hasattr(env.egraph, "num_enodes") else 0
    last = [0.0] * n
    if st.last >= 0:
        last[st.last] = 1.0
    x = [cost / max(env.base_cost, 1), enodes / LIMIT, st.steps / MAX_STEPS] + st.saturated + last
    return torch.tensor(x, dtype=torch.float32), cost


def apply(env, st: State, a: int):
    _, _, done, info = env.step(a)
    st.saturated[a] = 1.0 if info["stop_reason"] == "SATURATED" else 0.0
    st.last, st.steps = a, st.steps + 1
    st.done = done or st.steps >= MAX_STEPS
    return info


class Node:
    def __init__(self):
        self.P = None
        self.N = None
        self.W = None
        self.children = {}


def search(env, st: State, net, n: int, sims: int, noise: bool):
    root, root_ck = Node(), env.checkpoint()
    best = [math.inf]
    for _ in range(sims):
        env.restore(root_ck)
        s, node, path = st.copy(), root, []
        while True:
            if node.P is None:  # expand + evaluate with the network (no rollout)
                x, cost = features(env, s, n)
                best[0] = min(best[0], cost)
                with torch.no_grad():
                    logits, v = net(x.unsqueeze(0))
                p = torch.softmax(logits[0], -1).numpy()
                if noise and node is root:
                    p = 0.75 * p + 0.25 * np.random.dirichlet([0.3] * n)
                node.P, node.N, node.W = p, np.zeros(n), np.zeros(n)
                value = 1 - (cost / max(env.base_cost, 1)) * (1 - float(v[0]))
                break
            if s.done:
                cost, _ = env.egraph.extract(env.expr)
                best[0] = min(best[0], cost)
                value = 1 - cost / max(env.base_cost, 1)
                break
            u = node.W / np.maximum(node.N, 1) + C_PUCT * node.P * math.sqrt(node.N.sum() + 1) / (1 + node.N)
            a = int(np.argmax(u))
            apply(env, s, a)
            path.append((node, a))
            node = node.children.setdefault(a, Node())
        for nd, a in path:
            nd.N[a] += 1
            nd.W[a] += value
    env.restore(root_ck)
    return root, best[0]


def episode(lang_name, seed, net, sims, train=True):
    lang, expr = make_expr(lang_name, seed)
    env = BaseEnvironment(lang, expr, reward_func, LIMIT)
    env.reset()
    n = env.num_rules
    st, samples, best = State(n), [], env.base_cost
    start = time.perf_counter()
    while not st.done:
        root, seen = search(env, st, net, n, sims, noise=train)
        best = min(best, seen)
        visits = root.N if root.N is not None and root.N.sum() > 0 else np.ones(n)
        x, _ = features(env, st, n)
        samples.append((x, visits / visits.sum()))
        a = int(np.random.choice(n, p=visits / visits.sum())) if train and st.steps < 5 else int(np.argmax(visits))
        apply(env, st, a)
    final, _ = env.egraph.extract(env.expr)
    best = min(best, final)
    z = (env.base_cost - best) / max(env.base_cost, 1)
    return [(x, p, z) for x, p in samples], env.base_cost, best, time.perf_counter() - start


def _worker(args):
    torch.set_num_threads(1)
    lang_name, seed, path, sims, train = args
    lang, _ = make_expr(lang_name, 0)
    net = Net(len(lang.rewrite_rules()))
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    net.eval()
    samples, base, best, sec = episode(lang_name, seed, net, sims, train)
    return [(x.numpy(), p, z) for x, p, z in samples], base, best, sec


def train(lang_name, out, rounds=ROUNDS, episodes=EPISODES, workers=3):
    os.makedirs(out, exist_ok=True)
    lang, _ = make_expr(lang_name, 0)
    n = len(lang.rewrite_rules())
    net, path = Net(n), os.path.join(out, "net.pt")
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    torch.save(net.state_dict(), path)
    buffer, ctx = [], mp.get_context("spawn")
    val = list(range(900, 920))  # validation seeds, never test (0-19) or train (1000+)
    best_val = -1
    for r in range(rounds):
        t = time.time()
        seeds = [1000 + r * episodes + i for i in range(episodes)]
        with ctx.Pool(workers) as pool:
            outs = pool.map(_worker, [(lang_name, s, path, SIMS, True) for s in seeds])
        for samples, *_ in outs:
            buffer.extend(samples)
        buffer = buffer[-20000:]
        net.train()
        for _ in range(200):
            batch = random.sample(buffer, min(64, len(buffer)))
            x = torch.tensor(np.stack([b[0] for b in batch]))
            p = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
            z = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            logits, v = net(x)
            loss = -(p * torch.log_softmax(logits, -1)).sum(-1).mean() + ((v - z) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(net.state_dict(), path)
        with ctx.Pool(workers) as pool:
            vo = pool.map(_worker, [(lang_name, s, path, SIMS, False) for s in val])
        score = float(np.mean([(b - c) / max(b, 1) for _, b, c, _ in vo]))
        if score > best_val:
            best_val = score
            torch.save(net.state_dict(), os.path.join(out, "best.pt"))
        line = (f"round {r}: loss={loss.item():.3f} val={100 * score:.2f} best={100 * best_val:.2f} "
                f"sec/episode={np.mean([o[3] for o in vo]):.1f} ({time.time() - t:.0f}s)")
        print(line, flush=True)
        open(os.path.join(out, "train.log"), "a").write(line + "\n")


def evaluate(lang_name, out, workers=3):
    path = os.path.join(out, "best.pt")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        outs = pool.map(_worker, [(lang_name, s, path, SIMS, False) for s in range(20)])
    rows = [{"lang": lang_name, "seed": s, "base_cost": b, "azgeb_cost": c, "azgeb_s": round(sec, 2)}
            for s, (_, b, c, sec) in enumerate(outs)]
    json.dump(rows, open(os.path.join(out, "eval_test.json"), "w"))
    red = np.mean([(r["base_cost"] - r["azgeb_cost"]) / max(r["base_cost"], 1) for r in rows])
    print(f"{lang_name} test: mean size reduction {100 * red:.1f}%, "
          f"{np.mean([r['azgeb_s'] for r in rows]):.1f} s per expression", flush=True)


if __name__ == "__main__":
    mode, lang_name, out = sys.argv[1:4]
    (train if mode == "train" else evaluate)(lang_name, out)
