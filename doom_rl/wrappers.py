"""Composable Gymnasium wrappers: frame stack, normalization, reward shaping/clipping."""

from __future__ import annotations

from typing import Any, Callable, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizeObservations(gym.ObservationWrapper):
    """Scale uint8 (or any obs) to float32 [0, 1]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = np.zeros(self.observation_space.shape, dtype=np.float32)
        high = np.ones(self.observation_space.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


class FrameStack(gym.ObservationWrapper):
    """Stack the last `n_frames` observations along channel dim -> (n * C, H, W)."""

    def __init__(self, env: gym.Env, n_frames: int = 4) -> None:
        super().__init__(env)
        self.n_frames = n_frames
        c, h, w = self.observation_space.shape
        low = np.repeat(self.observation_space.low, n_frames, axis=0)
        high = np.repeat(self.observation_space.high, n_frames, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
        self._frames: list[np.ndarray] = []

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._frames = [obs] * self.n_frames
        return self._stack(), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        self._frames = self._frames[-self.n_frames :]
        return self._stack()

    def _stack(self) -> np.ndarray:
        return np.concatenate(self._frames, axis=0)


class ClipReward(gym.RewardWrapper):
    """Clip scalar reward to [-m, m] for stabler value targets."""

    def __init__(self, env: gym.Env, clip: float) -> None:
        super().__init__(env)
        self.clip = float(clip)

    def reward(self, reward: SupportsFloat) -> float:
        r = float(reward)
        return float(np.clip(r, -self.clip, self.clip))


class RewardShapingWrapper(gym.Wrapper):
    """Optional extra shaping: bonus on positive game reward, small living penalty."""

    def __init__(
        self,
        env: gym.Env,
        positive_bonus: float = 0.0,
        step_penalty: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.positive_bonus = positive_bonus
        self.step_penalty = step_penalty

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        r = float(reward)
        if r > 0:
            r += self.positive_bonus
        r -= self.step_penalty
        return obs, r, terminated, truncated, info


def apply_wrappers(
    env: gym.Env,
    *,
    frame_stack: int = 4,
    normalize: bool = True,
    clip_reward: Optional[float] = 1.0,
    use_reward_shaping: bool = False,
    shaping_bonus: float = 0.01,
    shaping_step_penalty: float = 0.0001,
) -> gym.Env:
    """
    Apply wrappers in a fixed order (inner -> outer):
    reward shaping -> clip reward -> frame stack -> normalize.
    """
    if use_reward_shaping:
        env = RewardShapingWrapper(
            env, positive_bonus=shaping_bonus, step_penalty=shaping_step_penalty
        )
    if clip_reward is not None and clip_reward > 0:
        env = ClipReward(env, clip_reward)
    env = FrameStack(env, n_frames=frame_stack)
    if normalize:
        env = NormalizeObservations(env)
    return env
