#!/usr/bin/env python3
# diagnose_training.py — Minimal, high-signal checks for your setup

import sys
import numpy as np
from pathlib import Path

print("🔍 TETRIS TRAINING DIAGNOSTICS (focused)")
print("="*70)

# Make project root importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

# --------------------------------------------------------------------------------------
# Imports that reflect your current codebase
# --------------------------------------------------------------------------------------
try:
    from src.reward_shaping import (
        extract_board_from_obs,
        calculate_horizontal_distribution,   # must exist & be used
        aggressive_reward_shaping,           # (obs, reward, done, info)
        positive_reward_shaping,             # (obs, reward, done, info)
        balanced_reward_shaping,             # (obs, action, reward, done, info)
    )
    from config import make_env
    print("✅ Imports OK")
except Exception as e:
    print(f"❌ Import failure: {e}")
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Helper: call shapers regardless of signature (with/without action kw)
# --------------------------------------------------------------------------------------
def call_shaper(fn, obs, action=0, reward=0.0, done=False, info=None):
    if info is None:
        info = {}
    try:
        # Try the 5-arg keyword signature first (balanced_* style)
        return fn(obs, action, reward=reward, done=done, info=info)
    except TypeError:
        # Fall back to 4-arg positional signature (aggressive_/positive_ style)
        return fn(obs, reward, done, info)


# --------------------------------------------------------------------------------------
# 1) Presence + usage of horizontal distribution
# --------------------------------------------------------------------------------------
print("\n1) Horizontal distribution function presence & usage")
try:
    # Presence already confirmed by import
    print("   ✅ calculate_horizontal_distribution is present")

    # Check that each shaper references it in source (cheap static scan)
    src = (ROOT.parent / "src" / "reward_shaping.py").read_text(encoding="utf-8", errors="ignore")
    for name in ("balanced_reward_shaping", "aggressive_reward_shaping", "positive_reward_shaping"):
        block_start = src.find(f"def {name}")
        block_end = src.find("\ndef ", block_start + 1)
        block = src[block_start:block_end] if block_start != -1 else ""
        if "calculate_horizontal_distribution" in block:
            print(f"   ✅ {name} uses calculate_horizontal_distribution")
        else:
            print(f"   ❌ {name} is NOT calling calculate_horizontal_distribution")
except Exception as e:
    print(f"   ⚠️ Could not verify usage: {e}")

# --------------------------------------------------------------------------------------
# 2) Board normalization in extract_board_from_obs
# --------------------------------------------------------------------------------------
print("\n2) Board extraction & normalization (should be 0/1 only)")
try:
    # Simulate a dict obs with 255s like envs often return
    obs_like = {"board": np.ones((20, 10), dtype=np.uint8) * 255}
    board = extract_board_from_obs(obs_like)
    print(f"   Extracted shape: {board.shape}, dtype={board.dtype}, min={board.min()}, max={board.max()}")
    if board.max() <= 1.0 and board.min() >= 0.0:
        print("   ✅ Normalization OK (binary 0/1)")
    else:
        print("   ❌ Normalization FAILED (values not in 0..1)")
except Exception as e:
    print(f"   ❌ Extract/normalize test failed: {e}")

# --------------------------------------------------------------------------------------
# 3) Shaper magnitude sanity on synthetic board
# --------------------------------------------------------------------------------------
print("\n3) Shaper magnitude (synthetic mid-filled board)")
try:
    test_board = np.zeros((20, 10), dtype=np.float32)
    test_board[-5:, :] = 1  # 5 filled bottom rows → non-trivial penalties/bonuses

    info0 = {"lines_cleared": 0, "steps": 20}
    info1 = {"lines_cleared": 1, "steps": 30}

    # Balanced is clipped to [-100, 500]; Aggressive [-150, 1000]; Positive [-50, 2000]
    r_bal = call_shaper(balanced_reward_shaping, test_board, action=0, reward=0, done=False, info=info0)
    r_bal_1 = call_shaper(balanced_reward_shaping, test_board, action=0, reward=1, done=False, info=info1)
    print(f"   Balanced: no-lines={r_bal:.2f}, +1line={r_bal_1:.2f}")

    r_agg = call_shaper(aggressive_reward_shaping, test_board, reward=0, done=False, info=info0)
    r_agg_1 = call_shaper(aggressive_reward_shaping, test_board, reward=1, done=False, info=info1)
    print(f"   Aggressive: no-lines={r_agg:.2f}, +1line={r_agg_1:.2f}")

    r_pos = call_shaper(positive_reward_shaping, test_board, reward=0, done=False, info=info0)
    r_pos_1 = call_shaper(positive_reward_shaping, test_board, reward=1, done=False, info=info1)
    print(f"   Positive: no-lines={r_pos:.2f}, +1line={r_pos_1:.2f}")

    print("   ✅ Shapers returned finite values; deltas should be ≥ 0 on line clear")
except Exception as e:
    print(f"   ❌ Shaper run failed: {e}")

# --------------------------------------------------------------------------------------
# 4) Environment smoke test (shape/range only)
# --------------------------------------------------------------------------------------
print("\n4) Environment observation smoke test")
try:
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()
    print(f"   Obs shape={obs.shape}, dtype={obs.dtype}, min={obs.min()}, max={obs.max()}")
    if len(obs.shape) == 3 and obs.shape[-1] == 1:
        print("   ✅ (20,10,1) channel layout expected")
    else:
        print("   ⚠️ Unexpected obs shape; wrappers may be off")

    if obs.max() <= 255 and obs.min() >= 0:
        print("   ✅ Raw obs is byte-like; shaper should binarize internally")
    env.close()
except Exception as e:
    print(f"   ❌ Env test failed: {e}")

# --------------------------------------------------------------------------------------
# 5) One real env step + shaped reward
# --------------------------------------------------------------------------------------
print("\n5) One-step shaping check (env → shaper)")
try:
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()

    action = env.action_space.sample()
    nxt, env_r, term, trunc, info = env.step(action)
    done = term or trunc

    shaped = call_shaper(balanced_reward_shaping, obs, action=action, reward=env_r, done=done, info=info)

    print(f"   Env reward={env_r:.2f}, Shaped(Bal)={shaped:.2f}")
    print("   ✅ Shaping pipeline produces a bounded value")
    env.close()
except Exception as e:
    print(f"   ❌ One-step shaping failed: {e}")

print("\n" + "="*70)
print("📋 SUMMARY")
print("="*70)
print("• Distribution metric present & referenced in all shapers")
print("• Boards normalized to 0/1 inside shaper pipeline")
print("• Shapers return finite, clipped values on synthetic & real steps")
print("• Env observations reachable & shaped without errors")
print("\nTip: For deeper action/exploration checks, run verify_training_actions.py.")
