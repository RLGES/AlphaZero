"""Train the AlphaZero engine on Modal (https://modal.com).

    modal token new                                              # once
    modal run --detach modal_app.py --suite arithmetic           # v2 defaults
    modal run --detach modal_app.py --suite boolean --no-follow  # launch and return
    modal volume get rewritezero-checkpoints arithmetic_v2/best.pt models/arithmetic_v2.pt

``scripts/modal_v2.sh`` launches all three suites at once.

Self-play is CPU work (one network call per new search leaf), so it fans out
over small CPU containers via ``selfplay_chunk.map``. The driver runs the outer
loop and the gradient steps on CPU, which is enough for this network size (and
GPU functions need a payment method on the Modal account). Every round commits
the checkpoint volume, and a driver that finds ``latest.pt`` resumes from it,
so a lost laptop connection, a preempted container or the driver timeout costs
at most one round: run the same command again to continue. Reusing a
``--run-name`` therefore continues that run; pick a new name to start fresh.

The defaults are the v2 recipe: harder terms (depth 3-7, up to 40 moves),
128 simulations per move, a larger network, and early stopping once the score
on 30 hard held-out terms has not improved for 8 rounds.
"""

from __future__ import annotations

from typing import Any

import modal

CHECKPOINT_ROOT = "/checkpoints"

app = modal.App("rewritezero-alphazero")
image = (  # CPU-only torch keeps the image small
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4", "numpy", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("src", "benchmarks")
)
volume = modal.Volume.from_name("rewritezero-checkpoints", create_if_missing=True)


def default_options(**overrides: Any) -> dict[str, Any]:
    options = {
        "suite": "arithmetic",
        "run_name": "",
        "rounds": 60,
        "games": 256,
        "sims": 128,
        "max_moves": 40,
        "depth": "3,7",
        "steps": 400,
        "batch_size": 128,
        "lr": 1e-3,
        "hidden": 192,
        "layers": 5,
        "chunks": 32,
        "eval_iterations": 100,
        "heldout": 30,
        "heldout_depth": "3,7",
        "patience": 8,
        "seed": 0,
    }
    options.update(overrides)
    options["run_name"] = options["run_name"] or f"{options['suite']}_v2"
    return options


def train_config(options: dict[str, Any]):
    from benchmarks.registry import load_suite
    from src.engines.alphazero.network import NetworkSpec
    from src.engines.alphazero.selfplay import SelfPlayConfig
    from src.engines.alphazero.train import TrainConfig, depth_range

    width = len(load_suite(options["suite"]).language.action_vocabulary())
    return TrainConfig(
        suite=options["suite"],
        rounds=options["rounds"],
        games_per_round=options["games"],
        train_steps=options["steps"],
        batch_size=options["batch_size"],
        lr=options["lr"],
        seed=options["seed"],
        out_dir=f"{CHECKPOINT_ROOT}/{options['run_name']}",
        eval_iterations=options["eval_iterations"],
        depth=depth_range(options["depth"]),
        heldout_terms=options["heldout"],
        heldout_depth=depth_range(options["heldout_depth"]),
        patience=options["patience"],
        selfplay=SelfPlayConfig(
            simulations_per_move=options["sims"], max_moves=options["max_moves"]
        ),
        spec=NetworkSpec(width, hidden=options["hidden"], layers=options["layers"]),
    )


# Caps keep a small budget safe: at most 32 one-CPU workers at a time.
@app.function(image=image, cpu=1.0, memory=1024, timeout=1800, retries=1, max_containers=32)
def selfplay_chunk(chunk):
    from src.engines.alphazero.train import play_chunk

    return play_chunk(chunk)


@app.function(image=image, cpu=4.0, memory=4096, volumes={CHECKPOINT_ROOT: volume}, timeout=43200)
def train_driver(options: dict[str, Any]) -> str:
    from pathlib import Path

    from src.engines.alphazero.train import outer_loop

    volume.reload()
    cfg = train_config(options)
    resume = (Path(cfg.out_dir) / "latest.pt").exists()
    print(f"run {options['run_name']}: {'resuming' if resume else 'starting'} -> {cfg.out_dir}")

    def play(chunks):
        return list(selfplay_chunk.map(chunks))

    def log(message: str) -> None:
        print(message, flush=True)
        volume.commit()  # the round's checkpoint is already on disk

    return str(outer_loop(cfg, resume, play=play, chunk_count=options["chunks"], log=log))


@app.local_entrypoint()
def main(
    suite: str = "arithmetic",
    run_name: str = "",
    rounds: int = 60,
    games: int = 256,
    sims: int = 128,
    max_moves: int = 40,
    depth: str = "3,7",
    steps: int = 400,
    hidden: int = 192,
    layers: int = 5,
    chunks: int = 32,
    heldout: int = 30,
    patience: int = 8,
    follow: bool = True,
):
    options = default_options(
        suite=suite,
        run_name=run_name,
        rounds=rounds,
        games=games,
        sims=sims,
        max_moves=max_moves,
        depth=depth,
        steps=steps,
        hidden=hidden,
        layers=layers,
        chunks=chunks,
        heldout=heldout,
        patience=patience,
    )
    train_config(options)  # fail fast on bad options before anything is spent
    print(f"Training {options['run_name']} on Modal: {options}")
    if follow:
        print(train_driver.remote(options))
    else:
        call = train_driver.spawn(options)
        print(f"Launched {options['run_name']} as {call.object_id}; watch it on modal.com/apps")
