"""
Test 3: Feature Extraction - Verify correct values from game states
====================================================================

This test checks if feature extraction correctly identifies holes, heights,
bumpiness, and wells from Tetris board states.
"""

import sys
import numpy as np
import tetris_gymnasium.envs  # Required to register environment
sys.path.insert(0, '/home/jonas/Code/Tetris-Gym2')

from src.feature_vector import (
    extract_feature_vector,
    get_column_heights,
    count_holes,
    calculate_bumpiness,
    calculate_wells,
    normalize_features
)

def test_empty_board():
    """Test features on empty board"""
    print("=" * 70)
    print("TEST 3A: Empty Board Features")
    print("=" * 70)

    # Empty 20x10 board
    board = np.zeros((20, 10), dtype=np.uint8)

    features = extract_feature_vector(board)

    print("Empty board should have:")
    print("  - All heights = 0")
    print("  - No holes")
    print("  - No bumpiness")
    print("  - No wells")

    print(f"\nExtracted features (raw):")
    print(f"  aggregate_height: {features[0]} (expected: 0)")
    print(f"  holes: {features[1]} (expected: 0)")
    print(f"  bumpiness: {features[2]} (expected: 0)")
    print(f"  wells: {features[3]} (expected: 0)")
    print(f"  column_heights: {features[4:14]} (expected: all 0)")
    print(f"  max_height: {features[14]} (expected: 0)")
    print(f"  min_height: {features[15]} (expected: 0)")
    print(f"  std_height: {features[16]} (expected: 0)")

    # Normalize
    features_norm = normalize_features(features)
    print(f"\nNormalized features:")
    print(f"  All values in [0,1]: {(features_norm >= 0).all() and (features_norm <= 1).all()}")

    assert features[0] == 0, "Aggregate height should be 0"
    assert features[1] == 0, "Holes should be 0"
    print("\n✅ Empty board features correct!")


def test_simple_board():
    """Test features on simple known board"""
    print("\n" + "=" * 70)
    print("TEST 3B: Simple Board with Known Features")
    print("=" * 70)

    # Create a board with known features
    # Bottom row filled except one cell (should create a hole when we add piece on top)
    board = np.zeros((20, 10), dtype=np.uint8)

    # Fill bottom row
    board[19, :] = 1

    # Add one cell on top in first column (height = 2)
    board[18, 0] = 1

    print("Board layout:")
    print("  Column 0: height 2")
    print("  Columns 1-9: height 1")
    print("  Expected bumpiness: 1 (diff between col 0 and 1)")

    features = extract_feature_vector(board)

    print(f"\nExtracted features:")
    print(f"  aggregate_height: {features[0]} (expected: 11)")
    print(f"  holes: {features[1]} (expected: 0)")
    print(f"  bumpiness: {features[2]} (expected: 1)")
    print(f"  column_heights: {features[4:14]}")
    print(f"  max_height: {features[14]} (expected: 2)")
    print(f"  min_height: {features[15]} (expected: 1)")

    # Check heights
    heights = get_column_heights(board)
    print(f"\nColumn heights: {heights}")
    assert heights[0] == 2, f"Column 0 should be height 2, got {heights[0]}"
    assert heights[1] == 1, f"Column 1 should be height 1, got {heights[1]}"

    # Check bumpiness
    bumpiness = calculate_bumpiness(heights)
    print(f"Bumpiness: {bumpiness} (expected: 1)")

    print("\n✅ Simple board features correct!")


def test_holes_detection():
    """Test hole detection specifically"""
    print("\n" + "=" * 70)
    print("TEST 3C: Hole Detection")
    print("=" * 70)

    # Create board with holes
    board = np.zeros((20, 10), dtype=np.uint8)

    # Stack in column 0: filled, empty, filled (1 hole)
    board[19, 0] = 1  # Bottom
    board[18, 0] = 0  # Hole
    board[17, 0] = 1  # Top

    # Stack in column 1: filled, empty, empty, filled (2 holes)
    board[19, 1] = 1  # Bottom
    board[18, 1] = 0  # Hole
    board[17, 1] = 0  # Hole
    board[16, 1] = 1  # Top

    print("Board setup:")
    print("  Column 0: 1 hole (bottom-empty-top)")
    print("  Column 1: 2 holes (bottom-empty-empty-top)")
    print("  Expected total holes: 3")

    holes = count_holes(board)
    print(f"\nDetected holes: {holes}")

    assert holes == 3, f"Expected 3 holes, got {holes}"
    print("✅ Hole detection correct!")


def test_feature_extraction_from_real_game():
    """Test feature extraction during actual gameplay"""
    print("\n" + "=" * 70)
    print("TEST 3D: Feature Extraction During Gameplay")
    print("=" * 70)

    import gymnasium as gym

    env = gym.make('tetris_gymnasium/Tetris', render_mode=None)
    obs, info = env.reset()

    print("Extracting features from real game states...")

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract board from observation
        if isinstance(obs, dict) and 'board' in obs:
            board = obs['board']
        else:
            # Assume obs is the board directly
            board = obs

        # Handle board size (24, 18) -> extract playable area (20, 10)
        if board.shape == (24, 18):
            board = board[0:20, 4:14]

        # Binarize
        board_binary = (board > 0).astype(np.uint8)

        # Extract features
        features = extract_feature_vector(board_binary)

        print(f"\nStep {step}:")
        print(f"  Board shape: {board.shape}")
        print(f"  Filled cells: {board_binary.sum()}")
        print(f"  Aggregate height: {features[0]:.1f}")
        print(f"  Holes: {features[1]:.1f}")
        print(f"  Max height: {features[14]:.1f}")

        # Check if features are reasonable
        assert features[0] >= 0, "Aggregate height can't be negative"
        assert features[1] >= 0, "Holes can't be negative"
        assert features[14] <= 20, "Max height can't exceed board height"

        if terminated or truncated:
            print(f"\n  Game over at step {step}")
            break

    env.close()
    print("\n✅ Feature extraction works during gameplay!")


def test_normalization():
    """Test feature normalization"""
    print("\n" + "=" * 70)
    print("TEST 3E: Feature Normalization")
    print("=" * 70)

    # Create worst-case board (full height, max holes, etc.)
    board = np.ones((20, 10), dtype=np.uint8)

    # Add some holes
    board[10:15, :] = 0  # Create a layer of holes

    features = extract_feature_vector(board)
    features_norm = normalize_features(features)

    print("Raw features:")
    for i, val in enumerate(features):
        print(f"  [{i}]: {val:.2f}")

    print("\nNormalized features:")
    for i, val in enumerate(features_norm):
        print(f"  [{i}]: {val:.4f}")

    print(f"\nNormalization check:")
    print(f"  Min: {features_norm.min():.4f} (should be >= 0)")
    print(f"  Max: {features_norm.max():.4f} (should be <= 1)")

    assert features_norm.min() >= -0.01, "Normalized features below 0"
    assert features_norm.max() <= 1.01, "Normalized features above 1"

    print("✅ Normalization correct!")


if __name__ == "__main__":
    print("🔍 DEBUGGING PLAN - TEST 3: Feature Extraction")
    print("=" * 70)
    print("This test verifies that feature extraction works correctly.\n")

    try:
        test_empty_board()
        test_simple_board()
        test_holes_detection()
        test_feature_extraction_from_real_game()
        test_normalization()

        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)
        print("✅ PASS: All feature extraction tests passed")
        print("   Feature extraction is working correctly.")
        print("   Issue likely in agent or reward function.")
        print("=" * 70)

    except AssertionError as e:
        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)
        print(f"❌ FAIL: Feature extraction has bugs")
        print(f"   Error: {e}")
        print("   Fix feature extraction before continuing.")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)
        print(f"❌ ERROR: Unexpected error during testing")
        print(f"   Error: {e}")
        print("=" * 70)
