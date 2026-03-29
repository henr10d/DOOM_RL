"""Central hyperparameters and environment settings (import and edit; no CLI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- PPO (Stable-Baselines3) ---
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 500_000

    # --- Vectorized training ---
    n_envs: int = 4  # Use DummyVecEnv if 1; SubprocVecEnv if >1

    # --- VizDoom / observation ---
    frame_skip: int = 4
    screen_resolution: tuple[int, int] = (84, 84)  # (width, height) for ViZDoom then resize confirm
    grayscale: bool = True
    # Path to .cfg; None = packaged `scenarios/basic.cfg`
    scenario_config_path: str | None = None

    # Discrete actions: each entry is a list of vizdoom.Button names pressed together
    action_button_sets: List[List[str]] = field(
        default_factory=lambda: [
            [],  # noop
            ["MOVE_FORWARD"],
            ["TURN_RIGHT"],
            ["TURN_LEFT"],
            ["MOVE_RIGHT"],
            ["MOVE_LEFT"],
            ["ATTACK"],
        ]
    )

    # --- Wrappers ---
    frame_stack: int = 4
    clip_reward: float | None = 1.0  # clip to [-c, c]; None disables clipping
    use_reward_shaping: bool = False  # optional small shaping in wrappers

    # --- Logging / checkpoints ---
    tensorboard_root: str = "./runs"
    # Roughly every N *simulator* timesteps (train.py converts for SB3 VecEnv semantics).
    checkpoint_freq: int = 50_000
    model_path: str = "./models/ppo_doom.zip"

    # --- Reproducibility ---
    seed: int = 42

    # --- Eval / video ---
    eval_episodes: int = 5
    record_eval_video: bool = False
    eval_video_path: str = "./videos/doom_eval.mp4"


CFG = Config()
