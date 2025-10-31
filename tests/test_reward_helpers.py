# tests/test_reward_helpers.py
"""
Test script to verify the reward shaping helper functions work correctly
against the current src/reward_shaping.py in this repo.

Run this BEFORE integrating into train.py to make sure everything works!

Usage:
    python tests/test_reward_helpers.py
"""

import numpy as np
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# ---------------------------------------------------------------------
# Imports from the project's reward_shaping module (current names)
# ---------------------------------------------------------------------
try:
    from src.reward_shaping import (
        extract_board_from_obs,
        get_column_heights,
        calculate_aggregate_height,
        get_max_height,
        count_holes,
        calculate_bumpiness,             # expects a board
        calculate_wells,                 # expects a board
        calculate_horizontal_distribution,
        aggressive_reward_shaping,       # may be (obs, reward, done, info)
        positive_reward_shaping,         # may be (obs, reward, done, info)
        balanced_reward_shaping,         # (obs, action, reward, done, info)
    )
    print("✅ Successfully imported reward shaping functions from src.reward_shaping")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure src/reward_shaping.py exists and is importable.")
    sys.exit(1)


# ---------------------------------------------------------------------
# Compatibility shim: call shaping funcs regardless of signature
# - Some functions accept (obs, action, reward, done, info)
# - Others accept (obs, reward, done, info) without action
# ---------------------------------------------------------------------
def _call_shaper(fn, obs, action_id=0, reward=0.0, done=False, info=None):
    """
    Calls a shaping function regardless of whether it expects
    keyword-only (reward, done, info) or positional args, and whether
    it includes the action parameter or not.
    """
    if info is None:
        info = {}
    try:
        # First try new-style with keywords and action
        return fn(obs, action_id, reward=reward, done=done, info=info)
    except TypeError:
        # Fall back to signature without action (obs, reward, done, info)
        return fn(obs, reward, done, info)
    except Exception:
        raise


# ---------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------
def create_test_board(scenario="empty"):
    """Create different 20x10 test boards (float32, 0/1)."""
    board = np.zeros((20, 10), dtype=np.float32)

    if scenario == "empty":
        pass

    elif scenario == "simple":
        # Simple flat bottom 3 rows filled
        board[-3:, :] = 1

    elif scenario == "holes":
        # Flat-ish with some holes
        board[-4:, :] = 1
        board[-3, 2] = 0
        board[-2, 5] = 0
        board[-3, 7] = 0

    elif scenario == "bumpy":
        heights = [2, 5, 3, 7, 2, 8, 3, 6, 2, 4]
        for col, h in enumerate(heights):
            if h > 0:
                board[-h:, col] = 1

    elif scenario == "dangerous":
        heights = [18, 17, 19, 18, 17, 18, 19, 18, 17, 18]
        for col, h in enumerate(heights):
            if h > 0:
                board[-h:, col] = 1

    elif scenario == "well":
        # Deep well in column 5
        board[-10:, :] = 1
        board[-10:, 5] = 0

    return board


def print_board(board, title="Board"):
    """Pretty print a board (top 10 rows shown)."""
    print(f"\n{title}:")
    print("─" * 22)
    for row in board[:10]:  # show top 10 rows
        print("│", end="")
        for cell in row:
            print(" █" if cell != 0 else " ·", end="")
        print(" │")
    print("─" * 22)
    print("  0 1 2 3 4 5 6 7 8 9  (columns)")


def test_scenario(name, board):
    """Compute metrics and shaped rewards for a given board."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")

    print_board(board, f"{name} Board")

    # Metrics (these functions expect the BOARD, not heights)
    heights = get_column_heights(board)
    agg_height = calculate_aggregate_height(board)
    max_h = get_max_height(board)
    holes = count_holes(board)
    bump = calculate_bumpiness(board)
    wells = calculate_wells(board)
    horiz = calculate_horizontal_distribution(board)

    print(f"\n📊 METRICS:")
    print(f"   Column Heights:    {heights}")
    print(f"   Aggregate Height:  {agg_height}")
    print(f"   Max Height:        {max_h}")
    print(f"   Holes:             {holes}")
    print(f"   Bumpiness:         {bump}")
    print(f"   Wells (depth sum): {wells}")
    print(f"   Horizontal dist.:  {horiz:.3f}")

    # Fake info for shaping
    info0 = {'lines_cleared': 0, 'steps': 0}
    info1 = {'lines_cleared': 1, 'steps': 10}

    # For shaping we pass the board as the "obs" (extractor handles raw boards)
    obs = board

    print(f"\n💰 REWARD SHAPING (no lines cleared):")
    aggr_r = _call_shaper(aggressive_reward_shaping, obs, 0, reward=0, done=False, info=info0)
    pos_r  = _call_shaper(positive_reward_shaping,   obs, 0, reward=0, done=False, info=info0)
    bal_r  = _call_shaper(balanced_reward_shaping,   obs, 0, reward=0, done=False, info=info0)

    print(f"   Aggressive: {aggr_r:+.2f}")
    print(f"   Positive:   {pos_r:+.2f}")
    print(f"   Balanced:   {bal_r:+.2f}")

    print(f"\n💎 WITH 1 LINE CLEARED:")
    aggr_r1 = _call_shaper(aggressive_reward_shaping, obs, 0, reward=1, done=False, info=info1)
    pos_r1  = _call_shaper(positive_reward_shaping,   obs, 0, reward=1, done=False, info=info1)
    bal_r1  = _call_shaper(balanced_reward_shaping,   obs, 0, reward=1, done=False, info=info1)

    print(f"   Aggressive: {aggr_r1:+.2f}  (Δ = {aggr_r1 - aggr_r:+.2f})")
    print(f"   Positive:   {pos_r1:+.2f}  (Δ = {pos_r1 - pos_r:+.2f})")
    print(f"   Balanced:   {bal_r1:+.2f}  (Δ = {bal_r1 - bal_r:+.2f})")


def test_edge_cases():
    """Edge cases to ensure helpers handle extremes gracefully."""
    print(f"\n{'='*60}")
    print("EDGE CASE TESTS")
    print(f"{'='*60}")

    # 1) Empty board
    print("\n1. Empty board")
    empty = np.zeros((20, 10), dtype=np.float32)
    print(f"   Heights:   {get_column_heights(empty)}")
    print(f"   Holes:     {count_holes(empty)}")
    print(f"   Bumpiness: {calculate_bumpiness(empty)}")
    print(f"   Wells:     {calculate_wells(empty)}")

    # 2) Full board
    print("\n2. Full board")
    full = np.ones((20, 10), dtype=np.float32)
    print(f"   Aggregate height:  {calculate_aggregate_height(full)}")
    print(f"   Holes:             {count_holes(full)}")
    print(f"   Bumpiness:         {calculate_bumpiness(full)}")
    print(f"   Wells:             {calculate_wells(full)}")

    # 3) Single column
    print("\n3. Single column filled")
    single = np.zeros((20, 10), dtype=np.float32)
    single[:, 5] = 1
    print(f"   Heights:   {get_column_heights(single)}")
    print(f"   Bumpiness: {calculate_bumpiness(single)}")
    print(f"   Wells:     {calculate_wells(single)}")

    # 4) Observation extraction sanity check
    print("\n4. Observation extraction")
    print("   - Direct numpy array:", "✅" if extract_board_from_obs(empty) is not None else "❌")
    print("   - 3D array (H, W, 1):", "✅" if extract_board_from_obs(np.zeros((20, 10, 1))) is not None else "❌")
    print("   - Dict with 'board': ", "✅" if extract_board_from_obs({'board': np.zeros((24, 18))}) is not None else "❌")


def compare_modes():
    """Compare shaping modes on a moderately bad board."""
    print(f"\n{'='*60}")
    print("REWARD SHAPING MODE COMPARISON")
    print(f"{'='*60}")

    board = create_test_board("holes")
    info_no = {'lines_cleared': 0, 'steps': 0}
    info_1  = {'lines_cleared': 1, 'steps': 10}
    info_4  = {'lines_cleared': 4, 'steps': 20}

    modes = [
        ('Aggressive', aggressive_reward_shaping),
        ('Positive',   positive_reward_shaping),
        ('Balanced',   balanced_reward_shaping),
    ]

    print("\n📋 Board stats:")
    print(f"   Holes: {count_holes(board)}")
    print(f"   Agg. height: {calculate_aggregate_height(board)}")
    print(f"   Bumpiness: {calculate_bumpiness(board)}")
    print(f"   Wells: {calculate_wells(board)}")

    for name, fn in modes:
        r0 = _call_shaper(fn, board, 0, reward=0, done=False, info=info_no)
        r1 = _call_shaper(fn, board, 0, reward=1, done=False, info=info_1)
        r4 = _call_shaper(fn, board, 0, reward=4, done=False, info=info_4)
        print(f"\n   {name} mode:")
        print(f"      No lines:  {r0:+7.1f}")
        print(f"      1 line:    {r1:+7.1f}  (Δ = +{r1-r0:.1f})")
        print(f"      TETRIS:    {r4:+7.1f}  (Δ = +{r4-r0:.1f})")


def main():
    print("="*60)
    print("TETRIS REWARD SHAPING HELPER FUNCTIONS TEST")
    print("="*60)

    scenarios = [
        ("Empty Board",         create_test_board("empty")),
        ("Simple Flat Bottom",  create_test_board("simple")),
        ("Board with Holes",    create_test_board("holes")),
        ("Bumpy Surface",       create_test_board("bumpy")),
        ("Dangerous Height",    create_test_board("dangerous")),
        ("Deep Well",           create_test_board("well")),
    ]

    for name, board in scenarios:
        test_scenario(name, board)

    test_edge_cases()
    compare_modes()

    print(f"\n{'='*60}")
    print("✅ ALL TESTS COMPLETE!")
    print(f"{'='*60}")
    print("\nIf you see this message, helper functions & shaping calls worked.\n")


if __name__ == "__main__":
    main()
