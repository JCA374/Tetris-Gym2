"""
Unit tests for feature heatmap generation functions.

Run with: python tests/test_feature_heatmaps.py
Or with pytest: pytest tests/test_feature_heatmaps.py -v
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_heatmaps import (
    compute_hole_heatmap,
    compute_height_map,
    compute_bumpiness_map,
    compute_well_map,
    compute_all_feature_heatmaps,
    verify_heatmap_properties
)


def test_hole_heatmap_empty_board():
    """Test that empty board has no holes."""
    board = np.zeros((20, 10), dtype=np.uint8)
    heatmap = compute_hole_heatmap(board)

    assert heatmap.shape == (20, 10)
    assert heatmap.sum() == 0, "Empty board should have no holes"
    verify_heatmap_properties(heatmap)
    print("✅ test_hole_heatmap_empty_board passed")


def test_hole_heatmap_single_hole():
    """Test detection of a single hole."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Create a hole: filled cell above empty cell
    board[19, 5] = 1  # Bottom filled
    board[18, 5] = 0  # Hole
    board[17, 5] = 1  # Top filled

    heatmap = compute_hole_heatmap(board)

    # Check that hole is detected
    assert heatmap[18, 5] == 1.0, f"Hole not detected: {heatmap[18, 5]}"
    # Check that filled cells are not marked as holes
    assert heatmap[17, 5] == 0.0, "Filled cell marked as hole"
    assert heatmap[19, 5] == 0.0, "Filled cell marked as hole"
    # Check other columns have no holes
    assert heatmap[:, :5].sum() == 0, "False holes in other columns"
    assert heatmap[:, 6:].sum() == 0, "False holes in other columns"

    verify_heatmap_properties(heatmap)
    print("✅ test_hole_heatmap_single_hole passed")


def test_hole_heatmap_multiple_holes():
    """Test detection of multiple holes in different columns."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Column 0: hole at row 18
    board[19, 0] = 1
    board[18, 0] = 0
    board[17, 0] = 1

    # Column 5: two holes at rows 18 and 16
    board[19, 5] = 1
    board[18, 5] = 0  # Hole 1
    board[17, 5] = 1
    board[16, 5] = 0  # Hole 2
    board[15, 5] = 1

    heatmap = compute_hole_heatmap(board)

    assert heatmap[18, 0] == 1.0, "Hole 1 not detected"
    assert heatmap[18, 5] == 1.0, "Hole 2 not detected"
    assert heatmap[16, 5] == 1.0, "Hole 3 not detected"
    assert heatmap.sum() == 3.0, f"Expected 3 holes, got {heatmap.sum()}"

    verify_heatmap_properties(heatmap)
    print("✅ test_hole_heatmap_multiple_holes passed")


def test_hole_heatmap_no_false_positives():
    """Test that empty cells without filled cells above are not holes."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Bottom row filled, but no cells above
    board[19, :] = 1

    heatmap = compute_hole_heatmap(board)

    # Rows 0-18 should have no holes (nothing above them)
    assert heatmap.sum() == 0, "Empty cells incorrectly marked as holes"

    verify_heatmap_properties(heatmap)
    print("✅ test_hole_heatmap_no_false_positives passed")


def test_height_map_empty_board():
    """Test height map for empty board."""
    board = np.zeros((20, 10), dtype=np.uint8)
    heatmap = compute_height_map(board)

    assert heatmap.shape == (20, 10)
    assert heatmap.sum() == 0, "Empty board should have zero heights"
    verify_heatmap_properties(heatmap)
    print("✅ test_height_map_empty_board passed")


def test_height_map_known_heights():
    """Test height map with known column heights."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Column 0: height 5 (rows 15-19)
    board[15:20, 0] = 1
    # Column 1: height 2 (rows 18-19)
    board[18:20, 1] = 1
    # Column 2: height 10 (rows 10-19)
    board[10:20, 2] = 1

    heatmap = compute_height_map(board)

    # Check normalized heights
    assert np.allclose(heatmap[:, 0], 5/20), f"Column 0 height incorrect: {heatmap[0, 0]}"
    assert np.allclose(heatmap[:, 1], 2/20), f"Column 1 height incorrect: {heatmap[0, 1]}"
    assert np.allclose(heatmap[:, 2], 10/20), f"Column 2 height incorrect: {heatmap[0, 2]}"
    assert np.allclose(heatmap[:, 3:], 0), "Other columns should be zero"

    verify_heatmap_properties(heatmap)
    print("✅ test_height_map_known_heights passed")


def test_height_map_full_column():
    """Test height map with full column."""
    board = np.zeros((20, 10), dtype=np.uint8)
    board[:, 5] = 1  # Full column

    heatmap = compute_height_map(board)

    assert np.allclose(heatmap[:, 5], 1.0), "Full column should have value 1.0"
    verify_heatmap_properties(heatmap)
    print("✅ test_height_map_full_column passed")


def test_bumpiness_map_flat_board():
    """Test bumpiness map for flat board (no bumps)."""
    board = np.zeros((20, 10), dtype=np.uint8)
    board[18:20, :] = 1  # All columns height 2

    heatmap = compute_bumpiness_map(board)

    # All columns should have zero bumpiness (all same height)
    assert np.allclose(heatmap, 0), f"Flat board should have zero bumpiness, got max {heatmap.max()}"
    verify_heatmap_properties(heatmap)
    print("✅ test_bumpiness_map_flat_board passed")


def test_bumpiness_map_alternating_heights():
    """Test bumpiness map with alternating heights."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Alternating heights: 5, 1, 5, 1, ...
    for col in range(10):
        if col % 2 == 0:
            board[15:20, col] = 1  # Height 5
        else:
            board[19:20, col] = 1  # Height 1

    heatmap = compute_bumpiness_map(board)

    # Each column should have significant bumpiness (difference of 4)
    expected_bumpiness = 4 / 20  # Difference normalized
    assert heatmap.min() > 0, "Bumpy board should have non-zero values"
    # Values should be around expected_bumpiness
    assert heatmap.max() <= 1.0, f"Bumpiness values too high: {heatmap.max()}"

    verify_heatmap_properties(heatmap)
    print("✅ test_bumpiness_map_alternating_heights passed")


def test_well_map_no_wells():
    """Test well map for board with no wells."""
    board = np.zeros((20, 10), dtype=np.uint8)
    board[18:20, :] = 1  # All columns same height

    heatmap = compute_well_map(board)

    assert heatmap.sum() == 0, "Flat board should have no wells"
    verify_heatmap_properties(heatmap)
    print("✅ test_well_map_no_wells passed")


def test_well_map_single_well():
    """Test well map with a single well."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Create a well in column 5
    board[15:20, 4] = 1  # Left wall: height 5
    board[18:20, 5] = 1  # Well: height 2
    board[15:20, 6] = 1  # Right wall: height 5

    heatmap = compute_well_map(board)

    # Column 5 should be marked as a well (depth 3)
    assert heatmap[0, 5] > 0, "Well not detected"
    # Other columns should not be wells
    assert heatmap[:, :5].sum() == 0, "False well detected"
    assert heatmap[:, 6:].sum() == 0, "False well detected"

    # Check well depth is reasonable (3/20 = 0.15)
    expected_depth = 3 / 20
    assert np.allclose(heatmap[0, 5], expected_depth), f"Well depth incorrect: {heatmap[0, 5]} != {expected_depth}"

    verify_heatmap_properties(heatmap)
    print("✅ test_well_map_single_well passed")


def test_well_map_edge_columns():
    """Test well detection at board edges."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Left edge well (column 0)
    board[19:20, 0] = 1  # Height 1
    board[15:20, 1] = 1  # Height 5

    # Right edge well (column 9)
    board[19:20, 9] = 1  # Height 1
    board[15:20, 8] = 1  # Height 5

    heatmap = compute_well_map(board)

    # Edge columns should be detected as wells
    assert heatmap[0, 0] > 0, "Left edge well not detected"
    assert heatmap[0, 9] > 0, "Right edge well not detected"

    verify_heatmap_properties(heatmap)
    print("✅ test_well_map_edge_columns passed")


def test_compute_all_feature_heatmaps():
    """Test that compute_all_feature_heatmaps returns all four heatmaps."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Create interesting board
    board[19, :] = 1      # Bottom row
    board[18, 5] = 0      # Hole in column 5
    board[17, 5] = 1      # Fill above hole
    board[15:20, 3] = 1   # Tall column

    holes, heights, bumpiness, wells = compute_all_feature_heatmaps(board)

    # Verify all heatmaps returned
    assert holes.shape == (20, 10), "Holes heatmap shape incorrect"
    assert heights.shape == (20, 10), "Heights heatmap shape incorrect"
    assert bumpiness.shape == (20, 10), "Bumpiness heatmap shape incorrect"
    assert wells.shape == (20, 10), "Wells heatmap shape incorrect"

    # Verify properties
    verify_heatmap_properties(holes)
    verify_heatmap_properties(heights)
    verify_heatmap_properties(bumpiness)
    verify_heatmap_properties(wells)

    # Verify hole was detected
    assert holes[18, 5] == 1.0, "Hole not detected in combined function"

    print("✅ test_compute_all_feature_heatmaps passed")


def test_complex_board_scenario():
    """Test all heatmaps on a complex realistic board."""
    board = np.zeros((20, 10), dtype=np.uint8)

    # Create a realistic-looking board with various features
    # Bottom layer mostly filled
    board[19, :] = 1
    board[19, 2] = 0  # Gap at bottom

    # Some columns built up
    board[16:19, 0] = 1  # Column 0: height 4
    board[17:19, 1] = 1  # Column 1: height 3
    board[18:19, 2] = 1  # Column 2: height 2 (well)
    board[16:19, 3] = 1  # Column 3: height 4

    # Create a hole
    board[18, 1] = 0  # Hole in column 1
    board[17, 1] = 1  # Filled above

    holes, heights, bumpiness, wells = compute_all_feature_heatmaps(board)

    # Verify specific features
    assert holes[18, 1] == 1.0, "Hole at (18, 1) not detected"
    assert holes[19, 2] == 1.0, "Hole at (19, 2) not detected"
    assert heights[0, 0] == 4/20, f"Column 0 height incorrect: {heights[0, 0]}"
    assert wells[0, 2] > 0, "Well at column 2 not detected"
    assert bumpiness.sum() > 0, "Bumpiness not detected on bumpy board"

    # Verify all properties
    verify_heatmap_properties(holes)
    verify_heatmap_properties(heights)
    verify_heatmap_properties(bumpiness)
    verify_heatmap_properties(wells)

    print("✅ test_complex_board_scenario passed")


def run_all_tests():
    """Run all test functions."""
    print("="*60)
    print("Running Feature Heatmap Unit Tests")
    print("="*60)

    test_functions = [
        test_hole_heatmap_empty_board,
        test_hole_heatmap_single_hole,
        test_hole_heatmap_multiple_holes,
        test_hole_heatmap_no_false_positives,
        test_height_map_empty_board,
        test_height_map_known_heights,
        test_height_map_full_column,
        test_bumpiness_map_flat_board,
        test_bumpiness_map_alternating_heights,
        test_well_map_no_wells,
        test_well_map_single_well,
        test_well_map_edge_columns,
        test_compute_all_feature_heatmaps,
        test_complex_board_scenario,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {e}")
            failed += 1

    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"❌ {failed} TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
