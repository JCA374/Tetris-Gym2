"""
Feature Extraction for Simple Feature-Based DQN

This module extracts hand-crafted scalar features from Tetris board states.
These features are used by the simple feature-based DQN models as an alternative
to processing raw board pixels with CNNs.

Feature extraction is based on successful Tetris AI implementations:
- Holes: Empty cells with filled cells above
- Bumpiness: Surface roughness (height variation between adjacent columns)
- Aggregate Height: Sum of all column heights
- Completable Rows: Rows that are almost full (can be cleared soon)

Additional features can be enabled:
- Max Height: Tallest column
- Height Variance: Variance in column heights
- Wells: Deep valleys between columns
- Clean Rows: Rows with no holes
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Union
from .reward_shaping import (
    extract_board_from_obs,
    get_column_heights,
    count_holes,
    calculate_bumpiness,
    calculate_aggregate_height,
    calculate_wells
)


class FeatureExtractor:
    """
    Extracts scalar features from Tetris board observations.

    This class converts 2D or 3D board observations into a small feature vector
    suitable for simple feedforward neural networks.
    """

    def __init__(self, feature_set="basic"):
        """
        Initialize feature extractor.

        Args:
            feature_set: Which features to extract
                - "basic": 4 features (holes, bumpiness, height, completable) [DEFAULT]
                - "standard": 6 features (+ max_height, height_variance)
                - "extended": 8 features (+ wells, clean_rows)
                - "minimal": 3 features (holes, height, bumpiness)
        """
        self.feature_set = feature_set

        if feature_set == "minimal":
            self.feature_names = ['holes', 'aggregate_height', 'bumpiness']
            self.n_features = 3
        elif feature_set == "basic":
            self.feature_names = ['holes', 'bumpiness', 'aggregate_height', 'completable_rows']
            self.n_features = 4
        elif feature_set == "standard":
            self.feature_names = ['holes', 'bumpiness', 'aggregate_height',
                                 'completable_rows', 'max_height', 'height_variance']
            self.n_features = 6
        elif feature_set == "extended":
            self.feature_names = ['holes', 'bumpiness', 'aggregate_height',
                                 'completable_rows', 'max_height', 'height_variance',
                                 'wells', 'clean_rows']
            self.n_features = 8
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}. "
                           f"Use 'minimal', 'basic', 'standard', or 'extended'.")

        print(f"FeatureExtractor initialized with '{feature_set}' feature set:")
        print(f"  Features: {self.feature_names}")
        print(f"  Dimension: {self.n_features}")

    def extract(self, obs: Union[np.ndarray, Dict]) -> np.ndarray:
        """
        Extract features from observation.

        Args:
            obs: Board observation (dict, 2D array, or 3D array)

        Returns:
            Feature vector (numpy array of shape (n_features,))
        """
        # Convert observation to board
        board = extract_board_from_obs(obs)

        # Calculate all features
        features = self._calculate_features(board)

        # Select features based on feature_set
        feature_vector = np.array([features[name] for name in self.feature_names],
                                   dtype=np.float32)

        return feature_vector

    def _calculate_features(self, board: np.ndarray) -> Dict[str, float]:
        """
        Calculate all possible features from board.

        Args:
            board: Binary board array (20, 10)

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Get column heights (used by many features)
        heights = get_column_heights(board)

        # Core features
        features['holes'] = float(count_holes(board))
        features['bumpiness'] = float(calculate_bumpiness(board))
        features['aggregate_height'] = float(calculate_aggregate_height(board))

        # Completable rows (rows with 8+ filled cells and no holes)
        features['completable_rows'] = float(self._count_completable_rows(board))

        # Height statistics
        if len(heights) > 0:
            features['max_height'] = float(max(heights))
            features['min_height'] = float(min(h for h in heights if h > 0) if any(h > 0 for h in heights) else 0)
            non_zero_heights = [h for h in heights if h > 0]
            features['height_variance'] = float(np.var(non_zero_heights) if non_zero_heights else 0.0)
            features['height_std'] = float(np.std(non_zero_heights) if non_zero_heights else 0.0)
        else:
            features['max_height'] = 0.0
            features['min_height'] = 0.0
            features['height_variance'] = 0.0
            features['height_std'] = 0.0

        # Wells
        features['wells'] = float(calculate_wells(board))

        # Clean rows (rows with no holes)
        features['clean_rows'] = float(self._count_clean_rows(board))

        # Columns used
        features['columns_used'] = float(sum(1 for h in heights if h > 0))

        return features

    def _count_completable_rows(self, board: np.ndarray) -> int:
        """
        Count rows that are 8+ filled with no holes (almost ready to clear).

        A completable row is close to being cleared, which is a good intermediate state.
        """
        completable = 0
        for row in range(20):
            row_data = board[row, :]
            filled_count = np.sum(row_data)

            if filled_count >= 8:
                # Check for holes
                has_hole = False
                filled_found = False

                for col in range(10):
                    if row_data[col]:
                        filled_found = True
                    elif filled_found and col < 9 and np.any(row_data[col+1:]):
                        # Empty cell with filled cells after it = hole
                        has_hole = True
                        break

                if not has_hole:
                    completable += 1

        return completable

    def _count_clean_rows(self, board: np.ndarray) -> int:
        """
        Count rows with no holes AND at least 3 filled cells.

        Clean rows indicate good placement quality.
        """
        clean_rows = 0
        for row in range(20):
            row_data = board[row, :]
            filled_count = np.sum(row_data)

            # Empty rows don't count
            if filled_count < 3:
                continue

            # Full row (about to clear) - definitely clean
            if filled_count == 10:
                clean_rows += 1
                continue

            # Check if filled cells are contiguous (no holes)
            first_filled = -1
            last_filled = -1
            for col in range(10):
                if row_data[col]:
                    if first_filled == -1:
                        first_filled = col
                    last_filled = col

            if first_filled != -1:
                expected_filled = last_filled - first_filled + 1
                if filled_count == expected_filled:
                    # Contiguous filled cells with no holes
                    clean_rows += 1

        return clean_rows


class FeatureObservationWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper that converts board observations to feature vectors.

    This wrapper can be used to transform a Tetris environment that returns
    board observations into one that returns scalar features.

    Properly inherits from gymnasium.ObservationWrapper for compatibility.
    """

    def __init__(self, env, feature_set="basic"):
        """
        Initialize feature observation wrapper.

        Args:
            env: Tetris gymnasium environment
            feature_set: Feature set to extract (see FeatureExtractor)
        """
        # Initialize base wrapper
        super().__init__(env)

        self.extractor = FeatureExtractor(feature_set=feature_set)

        # Update observation space to be feature vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.extractor.n_features,),
            dtype=np.float32
        )

        print(f"FeatureObservationWrapper applied to environment")
        print(f"  Original obs space: {env.observation_space}")
        print(f"  New obs space: {self.observation_space}")

    def observation(self, obs):
        """
        Convert observation to feature vector.

        This is the standard ObservationWrapper method that's called
        automatically by reset() and step().

        Args:
            obs: Original observation from environment

        Returns:
            Feature vector (numpy array)
        """
        return self.extractor.extract(obs)

    def step(self, action):
        """
        Step environment and add feature info.

        Override step to add feature values to info dict for reward shaping.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert to features (via self.observation())
        feature_obs = self.observation(obs)

        # Add raw features to info for reward shaping
        board = extract_board_from_obs(obs)
        heights = get_column_heights(board)
        info['holes'] = count_holes(board)
        info['bumpiness'] = calculate_bumpiness(board)
        info['max_height'] = max(heights) if heights else 0
        info['column_heights'] = heights

        return feature_obs, reward, terminated, truncated, info


def test_feature_extraction():
    """Test feature extraction on sample boards."""
    print("Testing Feature Extraction")
    print("=" * 70)

    # Create sample boards
    boards = {
        "empty": np.zeros((20, 10), dtype=np.uint8),
        "flat": np.vstack([np.zeros((15, 10), dtype=np.uint8),
                          np.ones((5, 10), dtype=np.uint8)]),
        "with_holes": np.zeros((20, 10), dtype=np.uint8),
        "bumpy": np.zeros((20, 10), dtype=np.uint8),
    }

    # Add holes to "with_holes" board
    boards["with_holes"][15:20, :] = 1
    boards["with_holes"][17, 3:7] = 0  # Holes

    # Make "bumpy" board
    for col in range(10):
        height = 5 + (col % 3) * 2
        boards["bumpy"][20-height:20, col] = 1

    # Test all feature sets
    for feature_set in ["minimal", "basic", "standard", "extended"]:
        print(f"\n{feature_set.upper()} feature set:")
        print("-" * 70)

        extractor = FeatureExtractor(feature_set=feature_set)

        for board_name, board in boards.items():
            features = extractor.extract(board)
            print(f"  {board_name:12s}: {features}")

    print("\n" + "=" * 70)
    print("✅ Feature extraction tests completed!")


if __name__ == "__main__":
    test_feature_extraction()
