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
        self.prev_holes = None  # Track holes for reduction bonus
        
    def get_current_stage(self) -> str:
        """Get current curriculum stage based on episode count"""
        if self.episode_count < 500:
            return "foundation"
        elif self.episode_count < 1000:
            return "clean_placement"
        elif self.episode_count < 2000:
            return "spreading_foundation"
        elif self.episode_count < 5000:
            return "clean_spreading"
        else:
            return "line_clearing_focus"
    
    def calculate_reward(self, obs: np.ndarray, action: int,
                        base_reward: float, done: bool,
                        info: Dict[str, Any]) -> float:
        """Calculate shaped reward based on current curriculum stage"""

        board = extract_board_from_obs(obs)
        metrics = self.calculate_metrics(board, info)
        stage = self.get_current_stage()
        
        # Log metrics for analysis
        self.metrics_history.append({
            'episode': self.episode_count,
            'stage': stage,
            'holes': metrics['holes'],
            'columns_used': metrics['columns_used'],
            'lines_cleared': metrics['lines_cleared']
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
        shaped = float(base_reward) * 100.0

        # IMPROVED: Stronger hole penalty to prevent swiss cheese towers
        shaped -= 2.5 * metrics['holes']  # INCREASED from 1.5

        # CRITICAL: Center-stacking detection and penalty
        shaped += metrics['center_stacking_penalty']  # This is always ≤ 0

        # Bonus for rows that are almost complete (8-9 filled, no holes)
        shaped += metrics['completable_rows'] * 10.0  # INCREASED from 8.0

        # Height and structure
        shaped -= 0.05 * metrics['aggregate_height']
        shaped -= 0.4 * metrics['bumpiness']
        shaped -= 0.1 * metrics['wells']

        # NEW: Explicit height penalty - discourage tall towers
        heights = metrics['column_heights']
        max_height = max(heights) if heights else 0
        if max_height > 15:
            shaped -= (max_height - 15) * 8.0  # -8 per row above 15

        # Strong spreading rewards (MASSIVELY INCREASED to fight center-stacking)
        shaped += 50.0 * metrics['spread']  # DOUBLED from 25.0
        shaped += metrics['columns_used'] * 12.0  # DOUBLED from 6.0
        shaped -= metrics['outer_unused'] * 20.0  # INCREASED from 8.0

        # Penalty for uneven heights
        shaped -= 3.0 * metrics['height_std']

        # Clean rows bonus (balanced after fix)
        shaped += metrics['clean_rows'] * 5.0  # REDUCED from 7.0 after clean_rows fix

        # IMPROVED: Much more conditional survival bonus
        if metrics['holes'] < 15:
            shaped += min(info.get('steps', 0) * 0.4, 30.0)  # Full bonus if very clean
        elif metrics['holes'] < 30:
            shaped += min(info.get('steps', 0) * 0.2, 15.0)  # Reduced if moderate holes
        else:
            shaped += 0  # NO bonus if 30+ holes

        # Strong line clear bonuses - SCALED BY BOARD QUALITY
        lines = metrics['lines_cleared']
        if lines > 0:
            # Calculate board quality (0.3 = terrible, 1.0 = perfect)
            quality = max(0.3, 1.0 - (metrics['holes'] / 50.0) - (metrics['bumpiness'] / 100.0))

            shaped += lines * 100.0 * quality  # Scale by quality!
            if lines == 2:
                shaped += 30.0 * quality
            elif lines == 3:
                shaped += 60.0 * quality
            elif lines == 4:  # Tetris
                shaped += 200.0 * quality  # Clean boards get bigger rewards!

        if done:
            # Heavier penalty if dying with many holes (scaled for Stage 4)
            if metrics['holes'] > 40:
                shaped -= 200.0  # Heavy penalty for terrible board
            elif metrics['holes'] > 30:
                shaped -= 150.0  # Moderate penalty
            else:
                shaped -= 75.0  # Standard penalty

        return float(np.clip(shaped, -400.0, 600.0))
    
    def _line_clearing_reward(self, base_reward: float, metrics: Dict,
                             done: bool, info: Dict) -> float:
        """
        Stage 5: Line Clearing Focus (Episodes 5000+)
        Focus: Maximize line clears with clean, spread placement
        """
        shaped = float(base_reward) * 100.0

        # IMPROVED: VERY strong hole penalty - force clean play
        shaped -= 3.5 * metrics['holes']  # INCREASED from 2.0

        # NEW: Reward for reducing holes from previous step (PDF Page 2)
        if self.prev_holes is not None:
            holes_reduced = self.prev_holes - metrics['holes']
            if holes_reduced > 0:
                shaped += 25.0 * holes_reduced  # +25 per hole filled!
        self.prev_holes = metrics['holes']

        # CRITICAL: Center-stacking detection and penalty
        shaped += metrics['center_stacking_penalty']  # This is always ≤ 0

        # Strong bonus for completable rows
        shaped += metrics['completable_rows'] * 30.0  # DOUBLED from 15.0 (was 12.0)

        # Height and structure
        shaped -= 0.06 * metrics['aggregate_height']
        shaped -= 0.5 * metrics['bumpiness']
        shaped -= 0.12 * metrics['wells']

        # NEW: Explicit height penalty - strongly discourage tall towers
        heights = metrics['column_heights']
        max_height = max(heights) if heights else 0
        if max_height > 15:
            shaped -= (max_height - 15) * 12.0  # -12 per row above 15 (STRONG)
        if max_height > 18:
            shaped -= 50.0  # Extra penalty for being at the ceiling

        # Maintain spreading (MASSIVELY INCREASED to fight center-stacking)
        shaped += 60.0 * metrics['spread']  # MASSIVELY INCREASED from 25.0
        shaped += metrics['columns_used'] * 15.0  # MASSIVELY INCREASED from 6.0
        shaped -= metrics['outer_unused'] * 30.0  # TRIPLED from 10.0
        shaped -= 5.0 * metrics['height_std']  # Slightly increased

        # Clean placement is critical (balanced after clean_rows fix)
        shaped += metrics['clean_rows'] * 6.0  # REDUCED from 12.0 after clean_rows fix

        # IMPROVED: VERY conditional survival bonus - only reward clean play
        if metrics['holes'] < 10:
            # Excellent - full bonus
            shaped += min(info.get('steps', 0) * 0.5, 40.0)
        elif metrics['holes'] < 20:
            # Good - reduced bonus
            shaped += min(info.get('steps', 0) * 0.3, 25.0)
        elif metrics['holes'] < 30:
            # Acceptable - minimal bonus
            shaped += min(info.get('steps', 0) * 0.1, 10.0)
        else:
            # Too many holes - NO survival bonus
            shaped += 0

        # Massive line clear bonuses - SCALED BY BOARD QUALITY
        lines = metrics['lines_cleared']
        if lines > 0:
            # Calculate board quality (0.3 = terrible, 1.0 = perfect)
            # Lower holes and bumpiness = higher quality = bigger reward
            quality = max(0.3, 1.0 - (metrics['holes'] / 50.0) - (metrics['bumpiness'] / 100.0))

            shaped += lines * 150.0 * quality  # Scale base reward by quality!
            if lines == 2:
                shaped += 50.0 * quality
            elif lines == 3:
                shaped += 100.0 * quality
            elif lines == 4:  # Tetris
                shaped += 400.0 * quality  # Only big bonus on clean boards!

        # Efficiency bonus for clearing lines with fewer pieces
        if lines > 0 and info.get('pieces_placed', 1) > 0:
            efficiency = lines / info.get('pieces_placed', 1)
            shaped += efficiency * 100.0

        if done:
            # IMPROVED: Strengthened death penalty (PDF Page 4 rec: -200 baseline)
            if metrics['holes'] > 40:
                shaped -= 500.0  # MASSIVE penalty for dying with messy board (was -300)
            elif metrics['holes'] > 30:
                shaped -= 350.0  # Heavy penalty for dying with bad board (was -200)
            else:
                shaped -= 200.0  # PDF baseline for clean death (was -100)

        return float(np.clip(shaped, -1000.0, 1000.0))  # Allow stronger penalties and rewards
    
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
