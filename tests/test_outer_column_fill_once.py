from pathlib import Path

"""
Micro regression: from reset, can we guide the very first piece to touch column 0 or 9?
This catches hard "stuck-in-middle" bugs quickly.
"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_HARD_DROP

def extract_board(obs):
    import numpy as _np
    b = obs
    if isinstance(obs, dict) and 'board' in obs:
        b = obs['board']
    b = _np.asarray(b)
    if b.ndim == 3 and b.shape[-1] >= 1:
        b = b[..., 0]
    return (b > 0).astype(_np.uint8)

def try_side(env, to_left=True):
    obs, _ = env.reset(seed=777)
    mover = ACTION_LEFT if to_left else ACTION_RIGHT
    for _ in range(25):
        obs, *_ = env.step(mover)
        obs, *_ = env.step(ACTION_DOWN)
    obs, *_ = env.step(ACTION_HARD_DROP)
    b = extract_board(obs)
    return (b[:, 0].any() if to_left else b[:, -1].any())

def main():
    env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
    left_ok = try_side(env, True)
    right_ok = try_side(env, False)
    env.close()
    print(f"Left-touch: {left_ok}, Right-touch: {right_ok}")
    assert left_ok or right_ok, "Neither outer column was reached by the first piece."

if __name__ == "__main__":
    main()
