"""TensorBoard paths, video recording, and global seeding."""

from __future__ import annotations

import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch


def get_tensorboard_log_dir(root: str = "./runs", run_name: str | None = None) -> str:
    """Return a unique directory under root for SB3 TensorBoard logging."""
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.abspath(root), name)
    os.makedirs(path, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and (where supported) CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: more deterministic CUDA (can slow training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class SimpleVideoRecorder:
    """Stack (H, W, C) uint8 BGR frames and write an MP4 with OpenCV."""

    def __init__(self, path: str, fps: float, frame_size: tuple[int, int]) -> None:
        """
        path: output file, e.g. ./videos/run.mp4
        frame_size: (width, height)
        """
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.path = path
        self.fps = max(1.0, float(fps))
        self.size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, self.fps, self.size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {path}")

    def write_rgb(self, frame_chw_or_hwc: np.ndarray) -> None:
        """Accept (H,W,3) RGB uint8 or (3,H,W) RGB uint8."""
        if frame_chw_or_hwc.ndim != 3:
            raise ValueError("Expected 3D image tensor")
        if frame_chw_or_hwc.shape[0] == 3:  # CHW
            hwc = np.transpose(frame_chw_or_hwc, (1, 2, 0))
        else:
            hwc = frame_chw_or_hwc
        if hwc.dtype != np.uint8:
            hwc = np.clip(hwc, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            bgr = cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(bgr)

    def close(self) -> None:
        self._writer.release()
