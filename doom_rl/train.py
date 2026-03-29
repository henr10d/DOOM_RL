"""Train PPO on VizDoom with vectorized env, TensorBoard, and checkpoints."""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from doom_rl.config import CFG, Config
from doom_rl.env import VizDoomGymEnv
from doom_rl.utils import get_tensorboard_log_dir, seed_everything
from doom_rl.wrappers import apply_wrappers


def build_train_env(cfg: Config, rank: int = 0) -> gym.Env:
    """Single process: base env + wrappers (channel-first, float32 after normalize)."""
    # Per-env NumPy stream only; global torch seed is set once in main.
    np.random.seed(cfg.seed + rank * 9973)
    env = VizDoomGymEnv(cfg, window_visible=False)
    return apply_wrappers(
        env,
        frame_stack=cfg.frame_stack,
        normalize=True,
        clip_reward=cfg.clip_reward,
        use_reward_shaping=cfg.use_reward_shaping,
    )


def make_vec_env(cfg: Config):
    """DummyVecEnv (n=1) or SubprocVecEnv (n>1)."""
    if cfg.n_envs < 1:
        raise ValueError("n_envs must be >= 1")

    def make(rank: int):
        def _init():
            return build_train_env(cfg, rank=rank)

        return _init

    if cfg.n_envs == 1:
        return DummyVecEnv([make(0)])
    return SubprocVecEnv([make(i) for i in range(cfg.n_envs)])


def main() -> None:
    cfg = CFG
    seed_everything(cfg.seed)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.model_path)) or ".", exist_ok=True)
    os.makedirs(cfg.tensorboard_root, exist_ok=True)

    vec_env = make_vec_env(cfg)
    log_dir = get_tensorboard_log_dir(cfg.tensorboard_root, run_name="ppo_doom")

    # CnnPolicy: channel-first (C,H,W). Wrappers already scale to [0,1], so disable
    # NatureCNN's default /255 to avoid double normalization.
    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        tensorboard_log=log_dir,
        policy_kwargs={"normalize_images": False},
        verbose=1,
        seed=cfg.seed,
    )

    checkpoint_dir = os.path.join(os.path.dirname(cfg.model_path) or ".", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_doom",
    )

    print(f"Training PPO for {cfg.total_timesteps} timesteps...")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"TensorBoard: {log_dir}")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=checkpoint_cb,
    )

    model.save(cfg.model_path)
    print(f"Saved model to {cfg.model_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
