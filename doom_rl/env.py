"""VizDoom wrapped as a Gymnasium Env: discrete actions, frame-skip, grayscale, (C,H,W) uint8."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, SupportsFloat, Tuple

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from doom_rl.config import CFG, Config


def _default_scenario_path() -> str:
    return os.path.join(os.path.dirname(vzd.__file__), "scenarios", "basic.cfg")


def _ordered_button_names(action_button_sets: Sequence[Sequence[str]]) -> List[str]:
    """Preserve first-seen order for stable make_action vectors."""
    out: List[str] = []
    for group in action_button_sets:
        for name in group:
            if name not in out:
                out.append(name)
    return out


class VizDoomGymEnv(gym.Env):
    """
    ViZDoom with Gymnasium API.
    Observation: uint8 (1, H, W) if grayscale else (3, H, W), values in [0, 255].
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        cfg: Optional[Config] = None,
        *,
        window_visible: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else CFG
        self._window_visible = window_visible

        cfg_path = self.cfg.scenario_config_path or _default_scenario_path()
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"VizDoom config not found: {cfg_path}")

        self._button_names = _ordered_button_names(self.cfg.action_button_sets)
        if not self._button_names:
            raise ValueError("action_button_sets produced no buttons; add at least one action.")

        self._buttons = [getattr(vzd.Button, n) for n in self._button_names]
        self._action_vectors = [
            [1.0 if n in set(group) else 0.0 for n in self._button_names]
            for group in self.cfg.action_button_sets
        ]

        w, h = self.cfg.screen_resolution
        self._out_w, self._out_h = int(w), int(h)
        self._frame_skip = max(1, int(self.cfg.frame_skip))

        c = 1 if self.cfg.grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c, self._out_h, self._out_w),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self._action_vectors))

        self.game = vzd.DoomGame()
        self.game.load_config(cfg_path)
        self.game.set_window_visible(window_visible)
        self.game.set_mode(vzd.Mode.PLAYER)
        # RGB24 everywhere; convert to grayscale in _get_obs for broader ViZDoom builds.
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        # Slightly larger internal res; final resize is cheap and keeps cfg simple.
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_available_buttons(self._buttons)
        self.game.init()

    def close(self) -> None:
        if hasattr(self, "game"):
            self.game.close()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # ViZDoom has limited seeding; global seed in utils covers most cases.
        self.game.new_episode()
        return self._get_obs(), {}

    def step(self, action: Any) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        if self.game.is_episode_finished():
            # Safe if step() is called after terminal (shouldn't happen in SB3);
            # return a valid terminal-like transition without crashing.
            obs = self._get_obs()
            return obs, 0.0, True, False, {"episode_finished": True}

        a = int(action)
        if a < 0 or a >= len(self._action_vectors):
            raise ValueError(f"Invalid action {action}")

        total_reward = 0.0
        terminated = False
        for _ in range(self._frame_skip):
            r = self.game.make_action(self._action_vectors[a])
            total_reward += float(r)
            terminated = self.game.is_episode_finished()
            if terminated:
                break

        obs = self._get_obs()
        truncated = False
        return obs, total_reward, terminated, truncated, {}

    def render(self) -> Optional[np.ndarray]:
        """Return RGB (H, W, 3) uint8 for logging/video; ViZDoom window handles live view."""
        st = self.game.get_state()
        if st is None:
            return None
        buf = st.screen_buffer
        if buf is None:
            return None
        if buf.ndim == 2:
            return cv2.cvtColor(buf, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(buf[:, :, :3])

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        buf = state.screen_buffer
        if buf.ndim == 2:
            buf = cv2.cvtColor(buf, cv2.COLOR_GRAY2RGB)
        rgb = np.ascontiguousarray(buf[:, :, :3])
        if self.cfg.grayscale:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(
                gray, (self._out_w, self._out_h), interpolation=cv2.INTER_AREA
            )
            chw = np.expand_dims(resized, axis=0)
        else:
            resized = cv2.resize(
                rgb, (self._out_w, self._out_h), interpolation=cv2.INTER_AREA
            )
            chw = np.transpose(resized, (2, 0, 1))

        return np.asarray(chw, dtype=np.uint8)
