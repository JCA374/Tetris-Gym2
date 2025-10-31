# tests/test_piece_movement_diagnostics.py
"""
Comprehensive piece-movement diagnostics focused on:
  1) Outer-column reachability (can we consistently touch cols 0 and 9?)
  2) Gravity-tick requirement (do LEFT/RIGHT "stick" without DOWN?)
  3) Final-state repetition (do we converge to the same clogged board?)
  4) Spawn coverage (are spawn columns biased to the center only?)

Run:
    python -u tests/test_piece_movement_diagnostics.py

Exit code: 0 on pass / 1 on actionable failure / 2 on crash
"""

import os, sys, random, numpy as np, statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from config import (
    make_env, discover_action_meanings,
    ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_HARD_DROP,
    ACTION_ROTATE_CW, ACTION_ROTATE_CCW, ACTION_NOOP
)

# ----------------------------- helpers ----------------------------------
def extract_board(obs):
    import numpy as _np
    if isinstance(obs, dict):
        b = obs.get("board") or obs.get("observation")
        if b is None:
            for v in obs.values():
                if isinstance(v, _np.ndarray) and v.ndim >= 2:
                    b = v; break
    else:
        b = obs
    b = _np.asarray(b)
    if b.ndim == 3 and b.shape[-1] >= 1:
        b = b[..., 0]
    # binarize
    if b.max() > 1:
        b = (b > 0).astype(_np.uint8)
    return b

def col_heights(board):
    H, W = board.shape
    h = []
    for c in range(W):
        hh = 0
        for r in range(H):
            if board[r, c] != 0:
                hh = H - r; break
        h.append(hh)
    return h

def max_row_fullness(board):
    return int(max((board[r, :]>0).sum() for r in range(board.shape[0])))

def nudge(env, mover, cycles, interleave_down=True):
    """Repeated LEFT/RIGHT with optional DOWN ticks in between."""
    for _ in range(cycles):
        obs, _, term, trunc, _ = env.step(mover)
        if term or trunc: 
            return obs, True
        if interleave_down:
            obs, _, term, trunc, _ = env.step(ACTION_DOWN)
            if term or trunc: 
                return obs, True
    return obs, False

def hard_drop(env):
    return env.step(ACTION_HARD_DROP)

# --------------------------- diagnostics ---------------------------------
def test_outer_reachability(n_seeds=50):
    """Try to reach both walls from fresh spawns across many seeds."""
    env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
    discover_action_meanings(env)

    left_hits, right_hits = 0, 0
    failures = []

    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        # try left-first placement
        obs, ended = nudge(env, ACTION_LEFT, cycles=20, interleave_down=True)
        if not ended:
            obs, _, term, trunc, _ = hard_drop(env)
            ended = term or trunc
        board = extract_board(obs)
        if board[:, 0].any(): left_hits += 1

        # If not ended, try a new piece and go right
        if not ended:
            obs, _ = env.reset(seed=seed+1000)  # new spawn variety
        obs, ended2 = nudge(env, ACTION_RIGHT, cycles=20, interleave_down=True)
        if not ended2:
            obs, _, term, trunc, _ = hard_drop(env)
        board = extract_board(obs)
        if board[:, -1].any(): right_hits += 1
        if not (board[:,0].any() or board[:,-1].any()):
            failures.append(seed)

    env.close()
    return {
        "n": n_seeds, "left_hits": left_hits, "right_hits": right_hits, "failures": failures
    }

def test_gravity_tick_requirement(trials=40):
    """Compare success reaching walls WITH vs WITHOUT interleaved DOWN."""
    env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
    discover_action_meanings(env)

    def run(interleave_down: bool):
        hits = 0
        for s in range(trials):
            obs, _ = env.reset(seed=100+s)
            obs, ended = nudge(env, ACTION_LEFT, cycles=20, interleave_down=interleave_down)
            if not ended:
                obs, _, term, trunc, _ = hard_drop(env)
            board = extract_board(obs)
            if board[:,0].any(): hits += 1
        return hits

    with_down = run(True)
    without_down = run(False)
    env.close()
    return {"with_down": with_down, "without_down": without_down, "trials": trials}

def test_final_state_repetition(runs=30, steps_per_run=60):
    """Measure how often we end up with identical final boards under random play."""
    env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
    discover_action_meanings(env)

    boards = []
    for s in range(runs):
        obs, _ = env.reset(seed=2000+s)
        done = False
        steps = 0
        while not done and steps < steps_per_run:
            # exploration-like distribution without NOOP
            r = random.random()
            if   r < 0.175: a = ACTION_LEFT
            elif r < 0.350: a = ACTION_RIGHT
            elif r < 0.450: a = ACTION_ROTATE_CW
            elif r < 0.550: a = ACTION_ROTATE_CCW
            elif r < 0.700: a = ACTION_DOWN
            elif r < 0.900: a = ACTION_HARD_DROP
            else:           a = 7  # SWAP if available
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
            steps += 1
        boards.append(extract_board(obs).tobytes())

    env.close()
    unique = len(set(boards))
    return {"runs": runs, "unique_final_states": unique, "repeat_ratio": 1 - unique/max(1, runs)}

def test_spawn_column_coverage(samples=200):
    """Coarse spawn-center bias check: look at first block's earliest non-zero columns."""
    env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
    discover_action_meanings(env)

    cols = []
    for s in range(samples):
        obs, _ = env.reset(seed=3000+s)
        b = extract_board(obs)
        nz_cols = np.where(b.sum(axis=0) > 0)[0]
        if len(nz_cols):
            cols.append(int(nz_cols.mean()))
    env.close()

    if not cols:
        return {"samples": samples, "error": "no non-zero spawns detected"}
    return {
        "samples": samples,
        "mean_spawn_col": float(np.mean(cols)),
        "std": float(np.std(cols)),
        "min": int(min(cols)), "max": int(max(cols)),
        "hist": dict(Counter(cols)),
    }

# ------------------------------ main -------------------------------------
def main():
    try:
        print("="*78)
        print("🧪 PIECE MOVEMENT DIAGNOSTICS")
        print("="*78)

        # 1) Outer reachability
        reach = test_outer_reachability(n_seeds=60)
        print(f"\n1) Outer-column reachability across {reach['n']} seeds:")
        print(f"   Left-wall touched:  {reach['left_hits']}/{reach['n']}")
        print(f"   Right-wall touched: {reach['right_hits']}/{reach['n']}")
        if reach['failures']:
            print(f"   Seeds with neither wall reached: {reach['failures'][:8]}{'...' if len(reach['failures'])>8 else ''}")

        # 2) Gravity tick requirement
        g = test_gravity_tick_requirement(trials=50)
        print(f"\n2) Gravity-tick requirement test (LEFT to wall):")
        print(f"   With DOWN interleaves:  {g['with_down']}/{g['trials']} successes")
        print(f"   Without DOWN interleaves:{g['without_down']}/{g['trials']} successes")
        gravity_required = g['with_down'] > 0 and g['without_down'] < (0.3 * g['with_down'])
        print(f"   Verdict: {'GRAVITY TICK REQUIRED' if gravity_required else 'Not strictly required'}")

        # 3) Final-state repetition
        rep = test_final_state_repetition(runs=40, steps_per_run=50)
        print(f"\n3) Final-state repetition (random exploration-like policy):")
        print(f"   Unique final boards: {rep['unique_final_states']}/{rep['runs']}  → repeat ratio={rep['repeat_ratio']:.2f}")
        trapped = rep['unique_final_states'] <= max(3, int(0.25*rep['runs']))
        print(f"   Verdict: {'REPETITIVE/DETERMINISTIC END-STATES' if trapped else 'Varied end-states'}")

        # 4) Spawn coverage
        sp = test_spawn_column_coverage(samples=120)
        print(f"\n4) Spawn column coverage (mean of first non-zero columns):")
        if 'error' in sp:
            print(f"   ❌ {sp['error']}")
        else:
            print(f"   mean≈{sp['mean_spawn_col']:.2f}, std≈{sp['std']:.2f}, min={sp['min']}, max={sp['max']}")
            print(f"   histogram (col → count): {dict(sorted(sp['hist'].items()))}")

        # ---------------- summary & actionable exit codes -----------------
        print("\n" + "="*78)
        print("📋 SUMMARY (Actionable)")
        print("="*78)
        fail = False
        if reach['left_hits'] < 5 or reach['right_hits'] < 5:
            print("❌ Outer columns are rarely/never reachable. Check action mapping/timing.")
            fail = True
        if gravity_required:
            print("⚠️ Horizontal moves likely need DOWN interleaves. Teach agent to LEFT/RIGHT+DOWN cadence.")
        if trapped:
            print("⚠️ Final boards repeat. Increase exploration/rotation variety; penalize tall center pillars.")
        if not fail:
            print("✅ Movement pathways look basically reachable. Focus on policy/shaping.")

        sys.exit(1 if fail else 0)

    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
