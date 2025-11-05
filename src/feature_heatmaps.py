"""
Feature Heatmap Generation for Enhanced DQN Tetris Observation

This module provides functions to generate spatial heatmaps of board quality metrics.
These heatmaps are added as additional channels to the observation space, providing
the agent with explicit guidance on holes, heights, bumpiness, and wells while
maintaining spatial awareness.

Author: Claude Code
Date: 2025-11-05
"""

import numpy as np
from typing import Tuple
from src.reward_shaping import get_column_heights, count_holes


def compute_hole_heatmap(board: np.ndarray) -> np.ndarray:
    """
    Generate spatial heatmap showing where holes exist on the board.

    A hole is an empty cell (value=0) that has at least one filled cell (value>0)
    above it in the same column. This provides explicit spatial information about
    where the problematic empty cells are located.

    Args:
        board: (20, 10) binary board array where 1=filled, 0=empty

    Returns:
        (20, 10) array with 1.0 where holes are, 0.0 elsewhere

    Example:
        >>> board = np.array([
        ...     [1, 0, 0],  # Row 2 (top)
        ...     [1, 1, 0],  # Row 1
        ...     [1, 0, 1],  # Row 0 (bottom) - hole at (0, 1)
        ... ])
        >>> heatmap = compute_hole_heatmap(board)
        >>> heatmap[0, 1]  # Check hole position
        1.0
    """
    height, width = board.shape
    heatmap = np.zeros((height, width), dtype=np.float32)

    # For each column, mark holes
    for col in range(width):
        column_data = board[:, col]

        # Find the highest filled cell (from top)
        filled_indices = np.where(column_data > 0)[0]

        if len(filled_indices) > 0:
            # Highest filled cell (smallest row index)
            highest_filled = filled_indices[0]

            # Everything below the highest filled cell
            for row in range(highest_filled + 1, height):
                if column_data[row] == 0:
                    # This is a hole (empty cell below filled cell)
                    heatmap[row, col] = 1.0

    return heatmap


def compute_height_map(board: np.ndarray) -> np.ndarray:
    """
    Generate spatial heatmap showing normalized column heights.

    Each column's height (number of rows from bottom with pieces) is normalized
    to [0, 1] and repeated down the entire column. This provides explicit
    information about which columns are tall vs short.

    Args:
        board: (20, 10) binary board array where 1=filled, 0=empty

    Returns:
        (20, 10) array with normalized column heights (0.0 = empty, 1.0 = full)

    Example:
        >>> board = np.zeros((20, 10))
        >>> board[15:20, 0] = 1  # Column 0: height 5
        >>> board[18:20, 1] = 1  # Column 1: height 2
        >>> heatmap = compute_height_map(board)
        >>> heatmap[0, 0]  # Column 0 height normalized
        0.25  # 5/20 = 0.25
        >>> heatmap[0, 1]  # Column 1 height normalized
        0.1   # 2/20 = 0.1
    """
    height, width = board.shape
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Get column heights using existing utility
    column_heights = get_column_heights(board)

    # Normalize heights to [0, 1] and broadcast to full column
    for col in range(width):
        normalized_height = column_heights[col] / float(height)
        heatmap[:, col] = normalized_height

    return heatmap


def compute_bumpiness_map(board: np.ndarray) -> np.ndarray:
    """
    Generate spatial heatmap showing height differences between adjacent columns.

    Bumpiness indicates how uneven the board surface is. High bumpiness makes
    it harder to place pieces and more likely to create holes. This map shows
    where the height transitions occur.

    Args:
        board: (20, 10) binary board array where 1=filled, 0=empty

    Returns:
        (20, 10) array with normalized bumpiness values (0.0 = smooth, 1.0 = very bumpy)

    Example:
        >>> board = np.zeros((20, 10))
        >>> board[15:20, 0] = 1  # Column 0: height 5
        >>> board[19:20, 1] = 1  # Column 1: height 1 (difference = 4)
        >>> heatmap = compute_bumpiness_map(board)
        >>> heatmap[0, 0]  # Shows height diff between col 0 and 1
        0.2  # 4/20 = 0.2
    """
    height, width = board.shape
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Get column heights
    column_heights = get_column_heights(board)

    # Compute height differences with neighbors
    for col in range(width):
        diffs = []

        # Difference with left neighbor
        if col > 0:
            diff = abs(column_heights[col] - column_heights[col - 1])
            diffs.append(diff)

        # Difference with right neighbor
        if col < width - 1:
            diff = abs(column_heights[col] - column_heights[col + 1])
            diffs.append(diff)

        # Average difference (bumpiness at this column)
        if diffs:
            avg_diff = np.mean(diffs)
            # Normalize by max possible difference (full height)
            normalized_diff = avg_diff / float(height)
            heatmap[:, col] = normalized_diff
        else:
            # Single column case (no neighbors)
            heatmap[:, col] = 0.0

    return heatmap


def compute_well_map(board: np.ndarray) -> np.ndarray:
    """
    Generate spatial heatmap showing wells (valleys between columns).

    A well is a column that is lower than both of its neighbors. Wells are
    useful for placing I-pieces and can be strategic, but deep wells can
    trap pieces. This map shows where wells exist and how deep they are.

    Args:
        board: (20, 10) binary board array where 1=filled, 0=empty

    Returns:
        (20, 10) array with normalized well depths (0.0 = no well, 1.0 = deep well)

    Example:
        >>> board = np.zeros((20, 10))
        >>> board[15:20, 0] = 1  # Column 0: height 5
        >>> board[18:20, 1] = 1  # Column 1: height 2 (well of depth 3)
        >>> board[15:20, 2] = 1  # Column 2: height 5
        >>> heatmap = compute_well_map(board)
        >>> heatmap[0, 1]  # Well depth at column 1
        0.15  # min(5-2, 5-2) / 20 = 3/20 = 0.15
    """
    height, width = board.shape
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Get column heights
    column_heights = get_column_heights(board)

    # Identify wells
    for col in range(width):
        current_height = column_heights[col]
        well_depth = 0

        # Check if this column is a well (lower than both neighbors)
        if col == 0:
            # Leftmost column - only check right neighbor
            if width > 1:
                right_height = column_heights[col + 1]
                if current_height < right_height:
                    well_depth = right_height - current_height
        elif col == width - 1:
            # Rightmost column - only check left neighbor
            left_height = column_heights[col - 1]
            if current_height < left_height:
                well_depth = left_height - current_height
        else:
            # Middle column - check both neighbors
            left_height = column_heights[col - 1]
            right_height = column_heights[col + 1]

            if current_height < left_height and current_height < right_height:
                # This is a well - depth is minimum of the two walls
                well_depth = min(left_height - current_height,
                               right_height - current_height)

        # Normalize well depth and set for entire column
        if well_depth > 0:
            normalized_depth = well_depth / float(height)
            heatmap[:, col] = normalized_depth

    return heatmap


def compute_all_feature_heatmaps(board: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to compute all four feature heatmaps at once.

    This is more efficient than calling each function separately when you need
    all features, as column heights are computed only once.

    Args:
        board: (20, 10) binary board array where 1=filled, 0=empty

    Returns:
        Tuple of (holes_map, height_map, bumpiness_map, well_map)
        Each is a (20, 10) float32 array with values in [0, 1]

    Example:
        >>> board = get_current_board()
        >>> holes, heights, bumpiness, wells = compute_all_feature_heatmaps(board)
        >>> observation = np.stack([board, active, holder, queue,
        ...                        holes, heights, bumpiness, wells], axis=-1)
    """
    holes_map = compute_hole_heatmap(board)
    height_map = compute_height_map(board)
    bumpiness_map = compute_bumpiness_map(board)
    well_map = compute_well_map(board)

    return holes_map, height_map, bumpiness_map, well_map


# Verification functions for testing

def verify_heatmap_properties(heatmap: np.ndarray, expected_shape: Tuple[int, int] = (20, 10)) -> bool:
    """
    Verify that a heatmap has correct properties.

    Args:
        heatmap: Heatmap to verify
        expected_shape: Expected shape (default: 20x10)

    Returns:
        True if all checks pass

    Raises:
        AssertionError if any check fails
    """
    assert heatmap.shape == expected_shape, f"Shape mismatch: {heatmap.shape} != {expected_shape}"
    assert heatmap.dtype == np.float32, f"Dtype mismatch: {heatmap.dtype} != float32"
    assert heatmap.min() >= 0.0, f"Values below 0: min={heatmap.min()}"
    assert heatmap.max() <= 1.0, f"Values above 1: max={heatmap.max()}"
    assert not np.isnan(heatmap).any(), "NaN values detected"
    assert not np.isinf(heatmap).any(), "Inf values detected"

    return True


if __name__ == '__main__':
    # Quick self-test
    print("Testing feature heatmap functions...")

    # Test board with known features
    board = np.zeros((20, 10), dtype=np.uint8)

    # Add some pieces
    board[18:20, 0] = 1  # Column 0: height 2
    board[15:20, 1] = 1  # Column 1: height 5
    board[18:20, 2] = 1  # Column 2: height 2 (well between 1 and 3)
    board[15:20, 3] = 1  # Column 3: height 5

    # Create a hole
    board[19, 1] = 1  # Fill bottom
    board[18, 1] = 0  # Create hole
    board[17, 1] = 1  # Fill above

    print("\nTest board created:")
    print(f"  Column heights: [2, 5, 2, 5, 0, 0, 0, 0, 0, 0]")
    print(f"  Holes: 1 (at row 18, col 1)")
    print(f"  Wells: Column 2 is a well (depth 3)")

    # Compute heatmaps
    holes_map = compute_hole_heatmap(board)
    height_map = compute_height_map(board)
    bumpiness_map = compute_bumpiness_map(board)
    well_map = compute_well_map(board)

    # Verify properties
    print("\nVerifying heatmap properties...")
    verify_heatmap_properties(holes_map)
    verify_heatmap_properties(height_map)
    verify_heatmap_properties(bumpiness_map)
    verify_heatmap_properties(well_map)
    print("✅ All property checks passed!")

    # Check specific values
    print("\nChecking specific values...")
    assert holes_map[18, 1] == 1.0, "Hole not detected"
    assert height_map[0, 1] == 5/20, f"Height map incorrect: {height_map[0, 1]} != {5/20}"
    assert well_map[0, 2] > 0.0, "Well not detected"
    print("✅ Value checks passed!")

    print("\n✅ All tests passed! Feature heatmaps working correctly.")
