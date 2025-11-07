# progressive_reward_improved.py
"""
Improved Progressive Reward Shaper for Tetris RL
Fixes the holes problem by using a more sophisticated curriculum
and better balanced reward components.
"""

import numpy as np
from typing import Dict, Any, Tuple

# Import existing reward shaping utilities
from .reward_shaping import (
    extract_board_from_obs,
    get_column_heights,
    calculate_aggregate_height,
    count_holes,
    calculate_bumpiness,
    calculate_wells,
    calculate_horizontal_distribution
)


class ImprovedProgressiveRewardShaper:
    """
    5-Stage Curriculum to fix the holes problem:
    
    Stage 1 (0-500): Foundation - Learn basic survival and placement
    Stage 2 (500-1000): Clean Placement - Increase holes penalty gradually
    Stage 3 (1000-2000): Spreading Foundation - Gentle spreading encouragement
    Stage 4 (2000-5000): Clean Spreading - Balance spreading with cleanliness
    Stage 5 (5000+): Line Clearing Focus - Strong line clear rewards
    """
    
    def __init__(self):
        self.episode_count = 0
        self.total_steps = 0
        self.stage_transitions = []
        self.metrics_history = []
        self.recent_hole_avg = None
        self.recent_completable_avg = None
        self.recent_clean_avg = None
        self.line_stage_unlocked = False
        self.last_reward_components = {}
        self.prev_holes = None  # Track holes for reduction bonus
        
    def get_current_stage(self) -> str:
        """Get current curriculum stage based on episode count with fallback"""
        if self.episode_count < 500:
            return "foundation"
        elif self.episode_count < 1000:
            return "clean_placement"
        elif self.episode_count < 2000:
            return "spreading_foundation"
        elif self.episode_count < 5000:
            return "clean_spreading"
        else:
            # Try performance gate first
            if not self.line_stage_unlocked and self._ready_for_line_stage():
                self.line_stage_unlocked = True
                print("\n✅ Stage 5 unlocked: Performance gate passed\n")

            # FALLBACK: Force transition after 3000 episodes in Stage 4 (episode 8000)
            if not self.line_stage_unlocked and self.episode_count >= 8000:
                self.line_stage_unlocked = True
                print("\n⏭️  Stage 5 unlocked: Fallback timer (episode 8000)\n")
                print(f"    Agent spent 3000 episodes in Stage 4 without passing gate.")
                print(f"    Forcing progression to prevent infinite stuckness.\n")

            if self.line_stage_unlocked:
                return "line_clearing_focus"

            return "clean_spreading"

    def update_curriculum_metrics(self, hole_avg=None, completable_avg=None, clean_avg=None):
        """
        Update rolling curriculum metrics so we can gate transitions based on performance.
        Called by the training loop once per episode with rolling averages.
        """
        self.recent_hole_avg = hole_avg
        self.recent_completable_avg = completable_avg
        self.recent_clean_avg = clean_avg

        if self.line_stage_unlocked:
            return

        if self._ready_for_line_stage():
            self.line_stage_unlocked = True
            print("\n✅ Curriculum gate passed: Stage 5 requirements met!")
            print(f"    Holes: {hole_avg:.1f} (≤30) ✅")
            if completable_avg is not None:
                print(f"    Completable rows: {completable_avg:.2f} (≥0.3) ✅")
            print()

    def _ready_for_line_stage(self) -> bool:
        """
        Determine whether the agent has earned the right to enter line-clearing focus.
        Requirements (RELAXED - rolling average over recent episodes):
          - holes <= 30 (relaxed from 25)
          - completable rows >= 0.3 (relaxed from 0.5, optional)
          - clean rows requirement REMOVED (was too strict)
        """
        if self.recent_hole_avg is None:
            return False

        # Primary criterion: holes must be below 30
        if self.recent_hole_avg > 30:
            return False

        # Secondary criterion: completable rows >= 0.3 (optional, more lenient)
        # If metric not tracked, ignore this requirement
        if self.recent_completable_avg is not None and self.recent_completable_avg < 0.3:
            return False

        # Clean rows requirement REMOVED - was too strict for Stage 5 entry

        return True
    
    def calculate_reward(self, obs: np.ndarray, action: int,
                        base_reward: float, done: bool,
                        info: Dict[str, Any]) -> float:
        """Calculate shaped reward based on current curriculum stage"""

        # Reset per-step component log
        self.last_reward_components = {}

        board = extract_board_from_obs(obs)
        metrics = self.calculate_metrics(board, info)
        stage = self.get_current_stage()
        
        # Log metrics for analysis
        self.metrics_history.append({
            'episode': self.episode_count,
            'stage': stage,
            'holes': metrics['holes'],
            'columns_used': metrics['columns_used'],
            'lines_cleared': metrics['lines_cleared'],
            'completable_rows': metrics['completable_rows'],
            'clean_rows': metrics['clean_rows']
        })
        
        # Apply stage-specific reward shaping
        if stage == "foundation":
            return self._foundation_reward(base_reward, metrics, done, info)
        elif stage == "clean_placement":
            return self._clean_placement_reward(base_reward, metrics, done, info)
        elif stage == "spreading_foundation":
            return self._spreading_foundation_reward(base_reward, metrics, done, info)
        elif stage == "clean_spreading":
            return self._clean_spreading_reward(base_reward, metrics, done, info)
        else:  # line_clearing_focus
            return self._line_clearing_reward(base_reward, metrics, done, info)

    def _record_components(self, stage: str, components: Dict[str, float],
                           pre_clip: float, post_clip: float):
        """Store latest reward component breakdown for diagnostics/logging."""
        record = dict(components)
        record['stage'] = stage
        record['pre_clip_reward'] = pre_clip
        record['clip_delta'] = post_clip - pre_clip
        self.last_reward_components = record
    
    def _foundation_reward(self, base_reward: float, metrics: Dict, 
                           done: bool, info: Dict) -> float:
        """
        Stage 1: Foundation (Episodes 0-500)
        Focus: Basic survival and not creating too many holes
        """
        shaped = float(base_reward) * 100.0
        
        # Gentle penalties to avoid learned helplessness
        shaped -= 0.3 * metrics['holes']  # Very gentle hole penalty
        shaped -= 0.02 * metrics['aggregate_height']  # Minimal height penalty
        shaped -= 0.1 * metrics['bumpiness']
        
        # Strong survival bonus to encourage exploration
        shaped += min(info.get('steps', 0) * 0.8, 40.0)
        
        # Small bonus for any line clears
        if metrics['lines_cleared'] > 0:
            shaped += metrics['lines_cleared'] * 50.0
        
        # Gentle death penalty
        if done:
            shaped -= 10.0
        
        return float(np.clip(shaped, -100.0, 200.0))
    
    def _clean_placement_reward(self, base_reward: float, metrics: Dict,
                                done: bool, info: Dict) -> float:
        """
        Stage 2: Clean Placement (Episodes 500-1000)
        Focus: Gradually increase cleanliness requirements
        """
        shaped = float(base_reward) * 100.0
        
        # Progressive hole penalty (increases with episode count)
        hole_penalty_factor = 0.3 + (self.episode_count - 500) / 500 * 0.7  # 0.3 to 1.0
        shaped -= hole_penalty_factor * metrics['holes']
        
        # Standard penalties
        shaped -= 0.03 * metrics['aggregate_height']
        shaped -= 0.2 * metrics['bumpiness']
        shaped -= 0.05 * metrics['wells']
        
        # Reward for clean rows (rows with no holes)
        shaped += metrics['clean_rows'] * 3.0
        
        # Survival bonus (reduced from Stage 1)
        shaped += min(info.get('steps', 0) * 0.5, 30.0)
        
        # Line clear bonuses
        lines = metrics['lines_cleared']
        if lines > 0:
            shaped += lines * 60.0
            if lines == 4:  # Tetris bonus
                shaped += 100.0
        
        if done:
            shaped -= 15.0
        
        return float(np.clip(shaped, -150.0, 300.0))
    
    def _spreading_foundation_reward(self, base_reward: float, metrics: Dict,
                                     done: bool, info: Dict) -> float:
        """
        Stage 3: Spreading Foundation (Episodes 1000-2000)
        Focus: Encourage spreading while maintaining cleanliness
        """
        shaped = float(base_reward) * 100.0
        
        # Balanced hole penalty (not too harsh to allow exploration)
        shaped -= 0.8 * metrics['holes']

        # CRITICAL: Center-stacking detection and penalty
        shaped += metrics['center_stacking_penalty']  # This is always ≤ 0

        # Height and structure penalties
        shaped -= 0.04 * metrics['aggregate_height']
        shaped -= 0.3 * metrics['bumpiness']
        shaped -= 0.08 * metrics['wells']
        
        # Spreading rewards (MASSIVELY INCREASED to fight center-stacking)
        shaped += 40.0 * metrics['spread']  # MASSIVELY INCREASED from 20.0
        shaped += metrics['columns_used'] * 8.0  # MASSIVELY INCREASED from 4.0
        shaped -= metrics['outer_unused'] * 15.0  # MASSIVELY INCREASED from 5.0 - Heavy penalty for unused outer
        
        # Reward even height distribution
        shaped -= 2.0 * metrics['height_std']
        
        # Clean rows bonus
        shaped += metrics['clean_rows'] * 4.0
        
        # Survival bonus
        shaped += min(info.get('steps', 0) * 0.3, 25.0)
        
        # Line clear bonuses
        lines = metrics['lines_cleared']
        if lines > 0:
            shaped += lines * 70.0
            if lines == 4:
                shaped += 120.0
        
        if done:
            shaped -= 20.0
        
        return float(np.clip(shaped, -200.0, 400.0))
    
    def _clean_spreading_reward(self, base_reward: float, metrics: Dict,
                                done: bool, info: Dict) -> float:
        """
        Stage 4: Clean Spreading (Episodes 2000-5000)
        Focus: Master spreading without creating holes
        """
        components: Dict[str, float] = {}

        base_component = float(base_reward) * 100.0
        components['base'] = base_component

        hole_penalty = -2.5 * metrics['holes']
        components['hole_penalty'] = hole_penalty

        center_penalty = metrics['center_stacking_penalty']
        components['center_penalty'] = center_penalty

        completable_bonus = metrics['completable_rows'] * 10.0
        components['completable_bonus'] = completable_bonus

        structure_penalty = (
            -0.05 * metrics['aggregate_height']
            -0.4 * metrics['bumpiness']
            -0.1 * metrics['wells']
        )
        components['structure_penalty'] = structure_penalty

        heights = metrics['column_heights']
        max_height = max(heights) if heights else 0
        height_penalty = 0.0
        if max_height > 15:
            height_penalty = -(max_height - 15) * 8.0
        components['height_penalty'] = height_penalty

        spread_bonus = 50.0 * metrics['spread']
        columns_bonus = metrics['columns_used'] * 12.0
        outer_penalty = -metrics['outer_unused'] * 20.0
        height_std_penalty = -3.0 * metrics['height_std']
        components['spread_bonus'] = spread_bonus
        components['columns_bonus'] = columns_bonus
        components['outer_penalty'] = outer_penalty
        components['height_std_penalty'] = height_std_penalty

        clean_rows_bonus = metrics['clean_rows'] * 5.0
        components['clean_rows_bonus'] = clean_rows_bonus

        steps = info.get('steps', 0)
        if metrics['holes'] < 15:
            survival_bonus = min(steps * 0.4, 30.0)
        elif metrics['holes'] < 30:
            survival_bonus = min(steps * 0.2, 15.0)
        else:
            survival_bonus = 0.0
        components['survival_bonus'] = survival_bonus

        lines = metrics['lines_cleared']
        quality = 0.0
        line_bonus = 0.0
        if lines > 0:
            quality = max(0.3, 1.0 - (metrics['holes'] / 50.0) - (metrics['bumpiness'] / 100.0))
            line_bonus = lines * 100.0 * quality
            if lines == 2:
                line_bonus += 30.0 * quality
            elif lines == 3:
                line_bonus += 60.0 * quality
            elif lines == 4:
                line_bonus += 200.0 * quality
        components['line_bonus'] = line_bonus

        shaped = sum(components.values())

        if done:
            if metrics['holes'] > 40:
                death_penalty = -200.0
            elif metrics['holes'] > 30:
                death_penalty = -150.0
            else:
                death_penalty = -75.0
            components['death_penalty'] = death_penalty
            shaped += death_penalty

        pre_clip = shaped
        clipped = float(np.clip(shaped, -400.0, 600.0))
        self._record_components('clean_spreading', components, pre_clip, clipped)
        return clipped
    
    def _line_clearing_reward(self, base_reward: float, metrics: Dict,
                             done: bool, info: Dict) -> float:
        """
        Stage 5: Line Clearing Focus (Episodes 5000+)
        Focus: Maximize line clears with clean, spread placement
        """
        components: Dict[str, float] = {}

        base_component = float(base_reward) * 100.0
        components['base'] = base_component

        holes = metrics['holes']
        hole_penalty = -5.0 * holes
        components['hole_penalty'] = hole_penalty

        hole_reduction_bonus = 0.0
        if self.prev_holes is not None:
            holes_reduced = self.prev_holes - holes
            if holes_reduced > 0:
                hole_reduction_bonus = 25.0 * holes_reduced
        components['hole_reduction_bonus'] = hole_reduction_bonus
        self.prev_holes = holes

        center_penalty = metrics['center_stacking_penalty']
        components['center_penalty'] = center_penalty

        completable_bonus = metrics['completable_rows'] * 45.0
        components['completable_bonus'] = completable_bonus

        structure_penalty = (
            -0.06 * metrics['aggregate_height']
            -0.5 * metrics['bumpiness']
            -0.12 * metrics['wells']
        )
        components['structure_penalty'] = structure_penalty

        heights = metrics['column_heights']
        max_height = max(heights) if heights else 0
        height_penalty = 0.0
        if max_height > 15:
            height_penalty += -(max_height - 15) * 12.0
        if max_height > 18:
            height_penalty += -50.0
        components['height_penalty'] = height_penalty

        cleanliness_scale = 1.0
        if holes > 20:
            cleanliness_scale = max(0.0, 1.0 - (holes - 20) / 20.0)
        spread_bonus = cleanliness_scale * 60.0 * metrics['spread']
        columns_bonus = cleanliness_scale * metrics['columns_used'] * 15.0
        outer_penalty = -metrics['outer_unused'] * 30.0
        height_std_penalty = -5.0 * metrics['height_std']
        components['spread_bonus'] = spread_bonus
        components['columns_bonus'] = columns_bonus
        components['outer_penalty'] = outer_penalty
        components['height_std_penalty'] = height_std_penalty

        clean_rows_bonus = metrics['clean_rows'] * 6.0
        components['clean_rows_bonus'] = clean_rows_bonus

        steps = info.get('steps', 0)
        if holes < 8:
            survival_bonus = min(steps * 0.5, 40.0)
        elif holes < 15:
            survival_bonus = min(steps * 0.3, 25.0)
        elif holes < 20:
            survival_bonus = min(steps * 0.1, 10.0)
        else:
            survival_bonus = 0.0
        components['survival_bonus'] = survival_bonus

        lines = metrics['lines_cleared']
        quality = 0.0
        line_bonus = 0.0
        if lines > 0:
            quality = max(0.3, 1.0 - (metrics['holes'] / 50.0) - (metrics['bumpiness'] / 100.0))
            line_bonus = lines * 150.0 * quality
            if lines == 2:
                line_bonus += 50.0 * quality
            elif lines == 3:
                line_bonus += 100.0 * quality
            elif lines == 4:
                line_bonus += 400.0 * quality
        components['line_bonus'] = line_bonus

        efficiency_bonus = 0.0
        if lines > 0 and info.get('pieces_placed', 1) > 0:
            efficiency = lines / info.get('pieces_placed', 1)
            efficiency_bonus = efficiency * 100.0
        components['efficiency_bonus'] = efficiency_bonus

        shaped = sum(components.values())

        if done:
            if metrics['holes'] > 40:
                death_penalty = -500.0
            elif metrics['holes'] > 30:
                death_penalty = -350.0
            else:
                death_penalty = -200.0
            components['death_penalty'] = death_penalty
            shaped += death_penalty

        pre_clip = shaped
        clipped = float(np.clip(shaped, -1000.0, 1000.0))
        self._record_components('line_clearing_focus', components, pre_clip, clipped)
        return clipped
    
    def calculate_metrics(self, board: np.ndarray, info: Dict) -> Dict[str, Any]:
        """Calculate all metrics needed for reward shaping"""
        metrics = {}

        # Basic metrics (using imported functions)
        metrics['aggregate_height'] = calculate_aggregate_height(board)
        metrics['holes'] = count_holes(board)
        metrics['bumpiness'] = calculate_bumpiness(board)
        metrics['wells'] = calculate_wells(board)
        metrics['spread'] = calculate_horizontal_distribution(board)

        # Column usage
        heights = get_column_heights(board)
        metrics['column_heights'] = heights
        metrics['columns_used'] = sum(1 for h in heights if h > 0)

        # Count unused outer columns (0,1,2,7,8,9) - CRITICAL for detecting center-stacking
        metrics['outer_unused'] = sum([
            1 for i in [0, 1, 2, 7, 8, 9] if heights[i] == 0
        ])

        # Extra metric: count completely empty outer columns (height = 0)
        # This is different from outer_unused which just counts any unused
        metrics['empty_outer_cols'] = sum([
            1 for i in [0, 1, 2, 7, 8, 9] if heights[i] == 0
        ])

        # Height statistics
        non_zero_heights = [h for h in heights if h > 0]
        if non_zero_heights:
            metrics['height_std'] = float(np.std(non_zero_heights))
            metrics['height_variance'] = float(np.var(non_zero_heights))
        else:
            metrics['height_std'] = 0.0
            metrics['height_variance'] = 0.0

        # Clean rows and completable rows (new metrics)
        metrics['clean_rows'] = self.count_clean_rows(board)
        metrics['completable_rows'] = self.count_completable_rows(board)

        # Center-stacking detection (critical for preventing main problem!)
        metrics['center_stacking_penalty'] = self.detect_center_stacking(heights)

        # Lines cleared from info
        metrics['lines_cleared'] = info.get('lines_cleared', 0)

        return metrics
    
    def count_clean_rows(self, board: np.ndarray) -> int:
        """
        Count rows with no holes AND at least 3 filled cells
        (Empty rows don't count - agent must actually place pieces cleanly)
        """
        clean_rows = 0
        for row in range(20):
            row_data = board[row, :]
            filled_count = np.sum(row_data)

            # FIXED: Empty rows don't count as clean!
            # Only count rows with at least 3 pieces
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
    
    def count_completable_rows(self, board: np.ndarray) -> int:
        """Count rows that are 8+ filled with no holes (almost ready to clear)"""
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

    def detect_center_stacking(self, heights: list) -> float:
        """
        Detect and heavily penalize center-only stacking.
        Returns a penalty value (always negative or zero).

        CRITICAL: Agent must use outer columns (0,1,2,7,8,9) not just center (3,4,5,6)
        """
        outer_cols = [0, 1, 2, 7, 8, 9]
        center_cols = [3, 4, 5, 6]

        outer_height = sum(heights[i] for i in outer_cols)
        center_height = sum(heights[i] for i in center_cols)

        total_height = outer_height + center_height
        if total_height > 0:
            center_ratio = center_height / total_height

            # If 70%+ of pieces are in center columns = center stacking!
            if center_ratio > 0.7:
                # MASSIVE LINEAR penalty to break center-stacking habit
                # 70% = -100, 80% = -150, 90% = -200, 100% = -250 PER STEP!
                penalty = -500.0 * (center_ratio - 0.5)
                return penalty
            elif center_ratio > 0.6:
                # Moderate penalty for mild center preference
                # 60% = -10, 65% = -15, 70% = -20
                penalty = -100.0 * (center_ratio - 0.5)
                return penalty

        return 0.0

    def reset(self):
        """Reset metrics for new episode"""
        self.prev_holes = None  # Reset hole tracking for new episode
    
    def update_episode(self, episode: int):
        """Update episode count and check for stage transitions"""
        old_stage = self.get_current_stage()
        self.episode_count = episode
        new_stage = self.get_current_stage()
        
        if old_stage != new_stage:
            self.stage_transitions.append({
                'episode': episode,
                'from_stage': old_stage,
                'to_stage': new_stage
            })
            print(f"\n🎓 CURRICULUM ADVANCEMENT: {old_stage} → {new_stage}")
            print(f"   Episode {episode}: Entering {new_stage} stage\n")
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about current training stage"""
        return {
            'current_stage': self.get_current_stage(),
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'transitions': self.stage_transitions
        }
