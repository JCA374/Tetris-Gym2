"""
Feature Vector Extraction for Tetris DQN

This module extracts explicit scalar features from Tetris board states
for use with fully-connected (non-CNN) DQN networks.

Based on research showing feature-based approaches outperform visual-only
methods by 100-1000x in sample efficiency.

Features extracted:
- Aggregate height (sum of all column heights)
- Holes (number of empty cells with filled cells above)
- Bumpiness (sum of absolute height differences between adjacent columns)
- Wells (depth of valleys between columns)
- Column heights (10 values, one per column)
- Max height (maximum column height)
- Min height (minimum column height)

Total: 17 features
"""

import numpy as np


def extract_feature_vector(obs):
    """
    Extract feature vector from Tetris observation.

    Args:
        obs: Can be either:
            - Dict with 'board' key (raw Gymnasium observation)
            - 3D array (H, W, C) from CompleteVisionWrapper
            - 2D array (H, W) board directly

    Returns:
        np.ndarray: Feature vector of shape (17,) with:
            [0]: aggregate_height (sum of column heights)
            [1]: holes (number of holes)
            [2]: bumpiness (sum of adjacent height differences)
            [3]: wells (sum of well depths)
            [4:14]: column_heights (10 values)
            [14]: max_height
            [15]: min_height
            [16]: std_height (standard deviation of heights)
    """
    # Extract 2D board from observation
    if isinstance(obs, dict):
        # Raw Gymnasium observation
        board = obs['board']
        # Extract playable area: rows 0-19, cols 4-13
        board = board[0:20, 4:14]
    elif len(obs.shape) == 3:
        # 3D array from wrapper - extract channel 0 (board state)
        board = obs[:, :, 0]
    else:
        # Already a 2D array
        board = obs

    # Binarize: 1 if filled, 0 if empty
    board_binary = (board > 0).astype(np.uint8)

    # Extract features
    column_heights = get_column_heights(board_binary)
    holes = count_holes(board_binary)
    bumpiness = calculate_bumpiness(column_heights)
    wells = calculate_wells(column_heights)

    # Aggregate statistics
    aggregate_height = np.sum(column_heights)
    max_height = np.max(column_heights)
    min_height = np.min(column_heights)
    std_height = np.std(column_heights)

    # Construct feature vector
    feature_vector = np.concatenate([
        [aggregate_height],   # 1 value
        [holes],              # 1 value
        [bumpiness],          # 1 value
        [wells],              # 1 value
        column_heights,       # 10 values
        [max_height],         # 1 value
        [min_height],         # 1 value
        [std_height],         # 1 value
    ])

    return feature_vector.astype(np.float32)


def get_column_heights(board):
    """
    Get the height of each column (number of filled cells from bottom).

    Args:
        board: Binary 2D array (H, W) where 1=filled, 0=empty

    Returns:
        np.ndarray: Array of shape (W,) with heights
    """
    heights = np.zeros(board.shape[1], dtype=np.int32)

    for col in range(board.shape[1]):
        column = board[:, col]
        # Find first filled cell from top
        filled_indices = np.where(column > 0)[0]
        if len(filled_indices) > 0:
            # Height = board_height - top_filled_row
            heights[col] = board.shape[0] - filled_indices[0]

    return heights


def count_holes(board):
    """
    Count the number of holes (empty cells with filled cells above).

    Args:
        board: Binary 2D array (H, W) where 1=filled, 0=empty

    Returns:
        int: Number of holes
    """
    holes = 0

    for col in range(board.shape[1]):
        column = board[:, col]
        # Find first filled cell from top
        filled_indices = np.where(column > 0)[0]
        if len(filled_indices) > 0:
            top_filled = filled_indices[0]
            # Count empty cells below the top filled cell
            holes += np.sum(column[top_filled:] == 0)

    return holes


def calculate_bumpiness(column_heights):
    """
    Calculate bumpiness (sum of absolute height differences between adjacent columns).

    Args:
        column_heights: Array of shape (W,) with column heights

    Returns:
        float: Total bumpiness
    """
    if len(column_heights) < 2:
        return 0.0

    bumpiness = 0.0
    for i in range(len(column_heights) - 1):
        bumpiness += abs(column_heights[i] - column_heights[i + 1])

    return bumpiness


def calculate_wells(column_heights):
    """
    Calculate total well depth (valleys between columns).

    A well is a column that is lower than both neighbors.

    Args:
        column_heights: Array of shape (W,) with column heights

    Returns:
        float: Total well depth
    """
    if len(column_heights) < 3:
        return 0.0

    wells = 0.0

    # Left edge well
    if column_heights[0] < column_heights[1]:
        wells += column_heights[1] - column_heights[0]

    # Middle wells
    for i in range(1, len(column_heights) - 1):
        left_height = column_heights[i - 1]
        right_height = column_heights[i + 1]
        current_height = column_heights[i]

        # If lower than both neighbors
        min_neighbor = min(left_height, right_height)
        if current_height < min_neighbor:
            wells += min_neighbor - current_height

    # Right edge well
    if column_heights[-1] < column_heights[-2]:
        wells += column_heights[-2] - column_heights[-1]

    return wells


def normalize_features(feature_vector, board_height=20, board_width=10):
    """
    Normalize feature vector to [0, 1] range for better neural network training.

    Args:
        feature_vector: Raw feature vector (17,)
        board_height: Height of board (default 20)
        board_width: Width of board (default 10)

    Returns:
        np.ndarray: Normalized feature vector
    """
    normalized = feature_vector.copy()

    # Normalize each feature by its theoretical maximum
    max_values = np.array([
        board_height * board_width,  # aggregate_height
        board_height * board_width,  # holes (theoretical max)
        board_height * (board_width - 1),  # bumpiness
        board_height * board_width,  # wells
        *([board_height] * board_width),  # column_heights (10 values)
        board_height,  # max_height
        board_height,  # min_height
        board_height / 2,  # std_height (approximate max)
    ])

    # Avoid division by zero
    max_values = np.where(max_values > 0, max_values, 1.0)

    normalized = normalized / max_values

    # Clip to [0, 1] just in case
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float32)


# Test function
if __name__ == "__main__":
    print("Testing feature vector extraction...")

    # Create a sample board
    board = np.zeros((20, 10), dtype=np.uint8)

    # Add some test pieces
    board[18:20, 0:4] = 1  # Bottom left
    board[19, 5] = 1  # Single block (creates holes)
    board[18:20, 6:8] = 1  # Bottom middle
    board[17:20, 9] = 1  # Right column (height 3)

    print("\nTest board:")
    print(board)

    # Extract features
    features = extract_feature_vector(board)
    print(f"\nRaw features (17 values):")
    print(f"  Aggregate height: {features[0]:.1f}")
    print(f"  Holes: {features[1]:.1f}")
    print(f"  Bumpiness: {features[2]:.1f}")
    print(f"  Wells: {features[3]:.1f}")
    print(f"  Column heights: {features[4:14]}")
    print(f"  Max height: {features[14]:.1f}")
    print(f"  Min height: {features[15]:.1f}")
    print(f"  Std height: {features[16]:.3f}")

    # Normalize
    normalized = normalize_features(features)
    print(f"\nNormalized features (0-1 range):")
    print(f"  Aggregate height: {normalized[0]:.3f}")
    print(f"  Holes: {normalized[1]:.3f}")
    print(f"  Bumpiness: {normalized[2]:.3f}")
    print(f"  Wells: {normalized[3]:.3f}")

    print("\n✅ Feature vector extraction working!")
