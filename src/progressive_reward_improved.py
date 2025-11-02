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
        
        # Height and structure penalties
        shaped -= 0.04 * metrics['aggregate_height']
        shaped -= 0.3 * metrics['bumpiness']
        shaped -= 0.08 * metrics['wells']
        
        # Spreading rewards (gentle introduction)
        shaped += 15.0 * metrics['spread']  # Horizontal distribution bonus
        shaped += metrics['columns_used'] * 3.0  # Reward for using columns
        shaped -= metrics['outer_unused'] * 4.0  # Penalty for unused outer columns
        
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
        
        # Strong hole penalty now that agent knows how to spread
        shaped -= 1.5 * metrics['holes']
        
        # Bonus for rows that are almost complete (8-9 filled, no holes)
        shaped += metrics['completable_rows'] * 8.0
        
        # Height and structure
        shaped -= 0.05 * metrics['aggregate_height']
        shaped -= 0.4 * metrics['bumpiness']
        shaped -= 0.1 * metrics['wells']
        
        # Strong spreading rewards
        shaped += 25.0 * metrics['spread']
        shaped += metrics['columns_used'] * 5.0
        shaped -= metrics['outer_unused'] * 8.0
        
        # Penalty for uneven heights
        shaped -= 3.0 * metrics['height_std']
        
        # Clean rows bonus (increased)
        shaped += metrics['clean_rows'] * 6.0
        
        # Conditional survival bonus (only if maintaining low holes)
        if metrics['holes'] < 30:
            shaped += min(info.get('steps', 0) * 0.4, 30.0)
        else:
            shaped += min(info.get('steps', 0) * 0.1, 10.0)
        
        # Strong line clear bonuses
        lines = metrics['lines_cleared']
        if lines > 0:
            shaped += lines * 100.0
            if lines == 2:
                shaped += 30.0
            elif lines == 3:
                shaped += 60.0
            elif lines == 4:  # Tetris
                shaped += 200.0
        
        if done:
            shaped -= 25.0
        
        return float(np.clip(shaped, -300.0, 600.0))
    
    def _line_clearing_reward(self, base_reward: float, metrics: Dict,
                             done: bool, info: Dict) -> float:
        """
        Stage 5: Line Clearing Focus (Episodes 5000+)
        Focus: Maximize line clears with clean, spread placement
        """
        shaped = float(base_reward) * 100.0
        
        # Very strong hole penalty
        shaped -= 2.0 * metrics['holes']
        
        # Strong bonus for completable rows
        shaped += metrics['completable_rows'] * 12.0
        
        # Height and structure
        shaped -= 0.06 * metrics['aggregate_height']
        shaped -= 0.5 * metrics['bumpiness']
        shaped -= 0.12 * metrics['wells']
        
        # Maintain spreading
        shaped += 20.0 * metrics['spread']
        shaped += metrics['columns_used'] * 4.0
        shaped -= metrics['outer_unused'] * 10.0
        shaped -= 4.0 * metrics['height_std']
        
        # Clean placement is critical
        shaped += metrics['clean_rows'] * 10.0
        
        # Survival only matters if playing clean
        if metrics['holes'] < 20:
            shaped += min(info.get('steps', 0) * 0.5, 40.0)
        
        # Massive line clear bonuses
        lines = metrics['lines_cleared']
        if lines > 0:
            shaped += lines * 150.0
            if lines == 2:
                shaped += 50.0
            elif lines == 3:
                shaped += 100.0
            elif lines == 4:  # Tetris
                shaped += 400.0
        
        # Efficiency bonus for clearing lines with fewer pieces
        if lines > 0 and info.get('pieces_placed', 1) > 0:
            efficiency = lines / info.get('pieces_placed', 1)
            shaped += efficiency * 100.0
        
        if done:
            # Scaled death penalty based on performance
            if metrics['holes'] > 50:
                shaped -= 50.0  # Heavy penalty for dying with many holes
            else:
                shaped -= 30.0
        
        return float(np.clip(shaped, -400.0, 800.0))
    
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
        metrics['outer_unused'] = sum([
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

        # Lines cleared from info
        metrics['lines_cleared'] = info.get('lines_cleared', 0)

        return metrics
    
    def count_clean_rows(self, board: np.ndarray) -> int:
        """Count rows with no holes (all filled or all empty)"""
        clean_rows = 0
        for row in range(20):
            row_data = board[row, :]
            filled_count = np.sum(row_data)
            
            # Check if row has no holes
            if filled_count == 0:  # Empty row
                clean_rows += 1
            elif filled_count == 10:  # Full row (about to clear)
                clean_rows += 1
            else:
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
    
    def reset(self):
        """Reset metrics for new episode"""
        pass  # Keep history across episodes
    
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
