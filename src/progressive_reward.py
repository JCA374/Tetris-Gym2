"""
Progressive Reward Shaping for Tetris AI

Implements a 4-stage curriculum that teaches skills progressively:
1. Basic Placement: Focus on avoiding holes
2. Height Management: Keep board low while placing cleanly
3. Spreading: Encourage using all columns
4. Balanced: Final optimized rewards

Automatically advances based on performance thresholds.
"""

import numpy as np

try:
    # Try relative import (when used as module)
    from .reward_shaping import (
        extract_board_from_obs,
        get_column_heights,
        calculate_aggregate_height,
        count_holes,
        calculate_bumpiness,
        calculate_wells,
        calculate_horizontal_distribution
    )
except ImportError:
    # Fall back to absolute import (when run as script)
    from reward_shaping import (
        extract_board_from_obs,
        get_column_heights,
        calculate_aggregate_height,
        count_holes,
        calculate_bumpiness,
        calculate_wells,
        calculate_horizontal_distribution
    )


class ProgressiveRewardShaper:
    """
    Progressive curriculum learning for Tetris rewards.

    Stages:
    1. basic (0-200 eps): High hole penalty, learn clean placement
    2. height (200-400 eps): Add height management, small spread bonus
    3. spreading (400-600 eps): Reduce hole penalty, strong spread bonus
    4. balanced (600+ eps): Final balanced rewards
    """

    def __init__(self, stage_thresholds=None):
        """
        Args:
            stage_thresholds: Dict of stage -> (min_episodes, performance_criteria)
                             Default uses episode counts only
        """
        self.current_stage = "basic"
        self.episode_count = 0
        self.stage_history = []

        # Default thresholds based on episodes
        if stage_thresholds is None:
            self.stage_thresholds = {
                "basic": 200,      # Episodes 0-200
                "height": 400,     # Episodes 200-400
                "spreading": 600,  # Episodes 400-600
                "balanced": float('inf')  # 600+
            }
        else:
            self.stage_thresholds = stage_thresholds

        # Performance tracking for adaptive advancement
        self.recent_holes = []
        self.recent_heights = []
        self.recent_columns_used = []

    def update_stage(self, episode, holes=None, height=None, columns_used=None):
        """Update current stage based on episode and performance"""
        self.episode_count = episode

        # Track performance
        if holes is not None:
            self.recent_holes.append(holes)
            if len(self.recent_holes) > 50:
                self.recent_holes.pop(0)

        if height is not None:
            self.recent_heights.append(height)
            if len(self.recent_heights) > 50:
                self.recent_heights.pop(0)

        if columns_used is not None:
            self.recent_columns_used.append(columns_used)
            if len(self.recent_columns_used) > 50:
                self.recent_columns_used.pop(0)

        # Determine stage based on episodes
        old_stage = self.current_stage

        if episode >= self.stage_thresholds["spreading"]:
            self.current_stage = "balanced"
        elif episode >= self.stage_thresholds["height"]:
            self.current_stage = "spreading"
        elif episode >= self.stage_thresholds["basic"]:
            self.current_stage = "height"
        else:
            self.current_stage = "basic"

        # Log stage changes
        if old_stage != self.current_stage:
            self.stage_history.append({
                'episode': episode,
                'old_stage': old_stage,
                'new_stage': self.current_stage
            })
            print(f"\n{'='*80}")
            print(f"🎓 CURRICULUM ADVANCEMENT: {old_stage} → {self.current_stage}")
            print(f"   Episode: {episode}")
            if self.recent_holes:
                print(f"   Recent avg holes: {np.mean(self.recent_holes):.1f}")
            if self.recent_columns_used:
                print(f"   Recent avg columns used: {np.mean(self.recent_columns_used):.1f}")
            print(f"{'='*80}\n")

    def get_stage_info(self):
        """Get current stage information"""
        return {
            'stage': self.current_stage,
            'episode': self.episode_count,
            'avg_holes': np.mean(self.recent_holes) if self.recent_holes else None,
            'avg_height': np.mean(self.recent_heights) if self.recent_heights else None,
            'avg_columns': np.mean(self.recent_columns_used) if self.recent_columns_used else None
        }

    def shape_reward(self, obs, action, reward, done, info):
        """
        Apply progressive reward shaping based on current stage.

        Args:
            obs: Board observation
            action: Action taken
            reward: Environment reward
            done: Episode done flag
            info: Info dict with steps, lines_cleared, etc.

        Returns:
            Shaped reward value
        """
        board = extract_board_from_obs(obs)

        # Base reward
        shaped = float(reward) * 100.0

        # Calculate metrics
        agg_h = calculate_aggregate_height(board)
        holes = count_holes(board)
        bump = calculate_bumpiness(board)
        wells = calculate_wells(board)
        spread = calculate_horizontal_distribution(board)
        heights = get_column_heights(board)

        # Update performance tracking
        columns_used = sum(1 for h in heights if h > 0)
        max_height = max(heights) if heights else 0
        self.update_stage(
            self.episode_count,
            holes=holes,
            height=max_height,
            columns_used=columns_used
        )

        # Stage-specific rewards
        if self.current_stage == "basic":
            # Stage 1: Focus on clean placement (GENTLER to avoid learned helplessness)
            shaped -= 0.05 * agg_h
            shaped -= 1.0 * holes      # REDUCED from 2.0 (still encourages clean play)
            shaped -= 0.3 * bump       # REDUCED from 0.5
            shaped -= 0.05 * wells     # REDUCED from 0.10

            # STRONGER survival bonus to encourage longer episodes
            shaped += min(info.get("steps", 0) * 0.5, 30.0)  # Increased!

        elif self.current_stage == "height":
            # Stage 2: Add height management (BALANCED)
            shaped -= 0.1 * agg_h      # Stronger height penalty
            shaped -= 1.2 * holes      # REDUCED from 1.5
            shaped -= 0.4 * bump       # REDUCED from 0.5
            shaped -= 0.08 * wells     # REDUCED from 0.10
            shaped += 8.0 * spread     # INCREASED from 5.0 for more encouragement
            shaped += min(info.get("steps", 0) * 0.4, 25.0)  # STRONGER survival

        elif self.current_stage == "spreading":
            # Stage 3: Encourage spreading
            shaped -= 0.05 * agg_h
            shaped -= 0.8 * holes      # REDUCED hole penalty (agent skilled now)
            shaped -= 0.5 * bump
            shaped -= 0.10 * wells

            # Strong spreading rewards
            shaped += 25.0 * spread
            shaped += columns_used * 6.0

            outer_unused = sum(1 for c in [0,1,2,7,8,9] if heights[c] == 0)
            shaped -= outer_unused * 8.0

            height_std = float(np.std(heights)) if columns_used > 0 else 0
            shaped -= 3.0 * height_std

            shaped += min(info.get("steps", 0) * 0.2, 20.0)

        else:  # balanced
            # Stage 4: Final balanced rewards
            shaped -= 0.05 * agg_h
            shaped -= 0.75 * holes     # Balanced hole penalty
            shaped -= 0.5 * bump
            shaped -= 0.10 * wells

            # Balanced spreading rewards
            shaped += 25.0 * spread
            shaped += columns_used * 6.0

            outer_unused = sum(1 for c in [0,1,2,7,8,9] if heights[c] == 0)
            shaped -= outer_unused * 8.0

            height_std = float(np.std(heights)) if columns_used > 0 else 0
            shaped -= 3.0 * height_std

            shaped += min(info.get("steps", 0) * 0.2, 20.0)

        # Common rewards across all stages
        lines = int(info.get("lines_cleared", 0))
        if lines > 0:
            shaped += lines * 80.0
            if lines == 4:  # Tetris bonus
                shaped += 120.0

        if done:
            shaped -= 5.0  # Light death penalty

        # Clamp
        return float(np.clip(shaped, -150.0, 600.0))


# Convenience function for backward compatibility
_global_shaper = None

def progressive_reward_shaping(obs, action, reward, done, info, shaper=None):
    """
    Global function interface for progressive reward shaping.

    Uses a global shaper instance if none provided.
    """
    global _global_shaper

    if shaper is None:
        if _global_shaper is None:
            _global_shaper = ProgressiveRewardShaper()
        shaper = _global_shaper

    return shaper.shape_reward(obs, action, reward, done, info)


if __name__ == "__main__":
    # Self-test
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    print("Progressive Reward Shaper Self-Test")
    print("=" * 80)

    shaper = ProgressiveRewardShaper()

    # Test board
    board = np.zeros((20, 10), dtype=np.float32)
    board[-5:, 4:7] = 1  # Center stack

    info = {"steps": 10, "lines_cleared": 0}

    # Test each stage
    for episode in [50, 250, 450, 650]:
        shaper.episode_count = episode
        reward = shaper.shape_reward(board, 0, 0.0, False, info)
        print(f"Episode {episode:3d} ({shaper.current_stage:10s}): Reward = {reward:+7.2f}")

    print("\n✅ Progressive reward shaper initialized successfully!")
