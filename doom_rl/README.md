# VizDoom PPO (Stable-Baselines3)

Modular Gymnasium-style pipeline for training PPO with a CNN policy on ViZDoom (`basic.cfg` by default).

## Install

1. Install [ViZDoom dependencies](https://github.com/mwydmuch/ViZDoom/blob/master/docs/Building.md) for your OS (CMake, SDL2, etc.).

2. Create a virtual environment and install Python packages from the project root (parent of `doom_rl/`):

```bash
pip install -r doom_rl/requirements.txt
```

Use a PyTorch build that matches your CUDA setup if you train on GPU ([pytorch.org](https://pytorch.org)).

## Configure

Edit `doom_rl/config.py` (`CFG` dataclass): learning rate, `total_timesteps`, `n_envs`, `frame_skip`, `screen_resolution`, `scenario_config_path`, reward clipping, etc. There is no CLI; training reads `CFG` directly.

## Train

From the directory that contains the `doom_rl` package (e.g. this repo root):

```bash
python -m doom_rl.train
```

- Writes TensorBoard logs under `./runs/`.
- Saves checkpoints under `./models/checkpoints/`.
- Saves the final policy to `./models/ppo_doom.zip`.

View logs:

```bash
tensorboard --logdir runs
```

## Evaluate

```bash
python -m doom_rl.eval
```

Runs several episodes with a visible game window and prints episode return.

Record an MP4 (path in `config.py`: `record_eval_video`, `eval_video_path`):

```python
# In doom_rl/config.py
record_eval_video: bool = True
```

Then:

```bash
python -m doom_rl.eval
```

## Project layout

| File | Role |
|------|------|
| `env.py` | ViZDoom → Gymnasium env: discrete actions, frame skip, grayscale, resize, `(C,H,W)` |
| `wrappers.py` | Frame stack, `[0,1]` normalization, optional reward shaping, reward clip |
| `config.py` | Hyperparameters and env settings |
| `train.py` | VecEnv + PPO (`CnnPolicy`) + TensorBoard + checkpoints |
| `eval.py` | Load model, play with window, optional video |
| `utils.py` | Seeds, TensorBoard path helper, OpenCV video writer |

## Notes

- Observations are channel-first `(stack * C, H, W)` float32 in `[0, 1]`, matching SB3 `CnnPolicy` with `normalize_images=False` (wrappers already scale pixels).
- Rewards are clipped by default for stabler value learning; set `clip_reward = None` in `config.py` to disable.
- Use `n_envs = 1` + `DummyVecEnv` if multiprocessing causes issues on your platform; increase `n_envs` for `SubprocVecEnv` throughput.
