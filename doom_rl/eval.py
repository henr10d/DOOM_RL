"""Load a trained PPO agent and run with a visible window; optional MP4 recording."""

from __future__ import annotations

import os
import re

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from doom_rl.config import CFG, Config
from doom_rl.env import VizDoomGymEnv
from doom_rl.utils import SimpleVideoRecorder, seed_everything
from doom_rl.wrappers import apply_wrappers


def _resolve_model_path(cfg: Config) -> str:
    """
    Prefer final save (model_path). If missing (training still running or interrupted),
    use the newest SB3 checkpoint: models/checkpoints/ppo_doom_*_steps.zip.
    """
    if os.path.isfile(cfg.model_path):
        return os.path.abspath(cfg.model_path)

    model_dir = os.path.dirname(os.path.abspath(cfg.model_path)) or os.getcwd()
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"No model at {cfg.model_path} and no checkpoint folder at {ckpt_dir}.\n"
            "Train with: python -m doom_rl.train\n"
            "The final zip is written only when training completes; during training, "
            "checkpoints appear under models/checkpoints/."
        )

    pat = re.compile(r"ppo_doom_(\d+)_steps\.zip$", re.IGNORECASE)
    candidates: list[tuple[int, str]] = []
    for name in os.listdir(ckpt_dir):
        m = pat.match(name)
        if m:
            candidates.append((int(m.group(1)), os.path.join(ckpt_dir, name)))

    if not candidates:
        raise FileNotFoundError(
            f"No model at {cfg.model_path} and no ppo_doom_*_steps.zip in {ckpt_dir}.\n"
            "Wait for the first checkpoint (see checkpoint_freq in config) or finish training."
        )

    candidates.sort(key=lambda x: x[0])
    path = candidates[-1][1]
    print(f"No {cfg.model_path!r}; loading latest checkpoint:\n  {path}")
    return path


def build_eval_env(cfg: Config) -> gym.Env:
    seed_everything(cfg.seed)
    env = VizDoomGymEnv(cfg, window_visible=True)
    return apply_wrappers(
        env,
        frame_stack=cfg.frame_stack,
        normalize=True,
        clip_reward=cfg.clip_reward,
        use_reward_shaping=cfg.use_reward_shaping,
    )


def main() -> None:
    cfg = CFG
    model_path = _resolve_model_path(cfg)

    env = build_eval_env(cfg)
    model = PPO.load(model_path, env=env)

    # Video uses raw RGB from the game (160x120) via the base env's render().
    video_writer: SimpleVideoRecorder | None = None
    if cfg.record_eval_video:
        frame = env.render()
        if frame is not None:
            h, w = frame.shape[0], frame.shape[1]
            video_writer = SimpleVideoRecorder(
                cfg.eval_video_path, fps=35.0, frame_size=(w, h)
            )

    try:
        for ep in range(cfg.eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                a = int(np.asarray(action).squeeze())
                obs, reward, term, trunc, _ = env.step(a)
                ep_reward += float(reward)
                done = term or trunc
                steps += 1

                if video_writer is not None:
                    vis = env.render()
                    if vis is not None:
                        video_writer.write_rgb(vis)

            print(f"Episode {ep + 1}/{cfg.eval_episodes}: reward={ep_reward:.3f}, steps={steps}")
    finally:
        env.close()
        if video_writer is not None:
            video_writer.close()
            print(f"Video saved to {cfg.eval_video_path}")


if __name__ == "__main__":
    main()
