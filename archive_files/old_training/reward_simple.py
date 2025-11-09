"""
Simple Reward Shaping for Tetris Baseline

This module implements the simple, sparse reward functions used by successful
Tetris DQN implementations in the literature. This serves as a baseline to
compare against our complex 5-stage progressive curriculum.

Based on successful implementations:
- nuno-faria/tetris-ai: +1 per piece, +(lines²×width) for clears, -1 for death
- Simple approaches: Minimal shaping, let agent figure out what matters

Philosophy: Less is more. Strong reward signals for outcomes (line clears, death),
minimal intermediate shaping. Let the agent explore and discover strategies.
"""

import numpy as np
from typing import Dict, Any


class SimpleRewardShaper:
    """
    Simple sparse reward function for Tetris.

    This implementation uses minimal reward shaping with strong signals for:
    - Line clearing (primary objective)
    - Game over (failure state)
    - Optional light penalties for bad structure

    Complexity: 1-3 terms vs. 10+ in progressive curriculum
    """

    def __init__(self, variant="quadratic"):
        """
        Initialize simple reward shaper.

        Args:
            variant: Reward function variant
                - "quadratic": (lines_cleared)² × 10 [DEFAULT - from literature]
                - "exponential": 10 × 2^lines_cleared
                - "linear": lines_cleared × 40
                - "sparse": Only line clears and death (no intermediate rewards)
                - "light_penalty": Quadratic + light hole/height penalties
        """
        self.variant = variant
        self.episode_count = 0

        print(f"SimpleRewardShaper initialized with '{variant}' variant")
        if variant == "quadratic":
            print("  Line clearing: (lines)² × 10")
            print("  Per piece: +1")
            print("  Death: -10")
        elif variant == "exponential":
            print("  Line clearing: 10 × 2^lines (20, 40, 80, 160)")
            print("  Per piece: +1")
            print("  Death: -50")
        elif variant == "linear":
            print("  Line clearing: lines × 40")
            print("  Per piece: +1")
            print("  Death: -10")
        elif variant == "sparse":
            print("  Line clearing: (lines)² × 10")
            print("  Per piece: 0 (pure sparse)")
            print("  Death: -50")
        elif variant == "light_penalty":
            print("  Line clearing: (lines)² × 10")
            print("  Per piece: +1")
            print("  Death: -10")
            print("  Light penalties: -0.1×holes, -0.01×height")

    def calculate_reward(self, obs: np.ndarray, action: int, base_reward: float,
                        done: bool, info: Dict[str, Any]) -> float:
        """
        Calculate simple shaped reward.

        Args:
            obs: Current observation (not used in simple shaping)
            action: Action taken (not used in simple shaping)
            base_reward: Raw environment reward
            done: Episode termination flag
            info: Environment info dict with 'lines_cleared', etc.

        Returns:
            Shaped reward (float)
        """
        shaped = 0.0

        # Get lines cleared this step
        lines = int(info.get('lines_cleared', 0))

        if self.variant == "quadratic":
            # Survival bonus
            if not done:
                shaped += 1.0

            # Line clearing (quadratic - from literature)
            if lines > 0:
                shaped += (lines ** 2) * 10.0

            # Death penalty
            if done:
                shaped -= 10.0

        elif self.variant == "exponential":
            # Survival bonus
            if not done:
                shaped += 1.0

            # Line clearing (exponential growth: 1→20, 2→40, 3→80, 4→160)
            if lines > 0:
                shaped += 10.0 * (2 ** lines)

            # Strong death penalty
            if done:
                shaped -= 50.0

        elif self.variant == "linear":
            # Survival bonus
            if not done:
                shaped += 1.0

            # Line clearing (linear)
            if lines > 0:
                shaped += lines * 40.0

            # Death penalty
            if done:
                shaped -= 10.0

        elif self.variant == "sparse":
            # NO survival bonus - truly sparse

            # Line clearing only
            if lines > 0:
                shaped += (lines ** 2) * 10.0

            # Strong death penalty
            if done:
                shaped -= 50.0

        elif self.variant == "light_penalty":
            # Survival bonus
            if not done:
                shaped += 1.0

            # Line clearing (quadratic)
            if lines > 0:
                shaped += (lines ** 2) * 10.0

            # Light structure penalties (only when not clearing)
            if lines == 0 and not done:
                holes = int(info.get('holes', 0))
                max_height = int(info.get('max_height', 0))

                shaped -= 0.1 * holes
                shaped -= 0.01 * max_height

            # Death penalty
            if done:
                shaped -= 10.0

        return float(shaped)

    def update_episode(self, episode: int):
        """Update episode count (for compatibility with progressive shaper)."""
        self.episode_count = episode

    def reset(self):
        """Reset for new episode (no-op for simple shaper)."""
        pass

    def get_stage_info(self) -> Dict[str, Any]:
        """Get stage info (for compatibility with progressive shaper)."""
        return {
            'current_stage': self.variant,
            'episode': self.episode_count,
            'transitions': []
        }


class AdaptiveSimpleRewardShaper:
    """
    Simple reward shaper that adapts over training.

    Starts with dense rewards (survival bonus) and gradually shifts to
    sparse rewards (line clears only) as agent learns.

    This tests whether reducing reward frequency helps in later training.
    """

    def __init__(self, survival_decay_episodes=5000):
        """
        Initialize adaptive simple shaper.

        Args:
            survival_decay_episodes: Episodes over which to decay survival bonus
        """
        self.episode_count = 0
        self.survival_decay_episodes = survival_decay_episodes

        print(f"AdaptiveSimpleRewardShaper initialized")
        print(f"  Survival bonus decays over {survival_decay_episodes} episodes")
        print(f"  Line clearing: (lines)² × 10 (constant)")
        print(f"  Death: -10 (constant)")

    def calculate_reward(self, obs: np.ndarray, action: int, base_reward: float,
                        done: bool, info: Dict[str, Any]) -> float:
        """Calculate adaptive shaped reward."""
        shaped = 0.0

        # Get lines cleared
        lines = int(info.get('lines_cleared', 0))

        # Decaying survival bonus (1.0 → 0.0 over training)
        if not done:
            survival_factor = max(0.0, 1.0 - (self.episode_count / self.survival_decay_episodes))
            shaped += survival_factor

        # Line clearing (constant - always important)
        if lines > 0:
            shaped += (lines ** 2) * 10.0

        # Death penalty (constant)
        if done:
            shaped -= 10.0

        return float(shaped)

    def update_episode(self, episode: int):
        """Update episode count."""
        self.episode_count = episode

    def reset(self):
        """Reset for new episode."""
        pass

    def get_stage_info(self) -> Dict[str, Any]:
        """Get stage info."""
        survival_factor = max(0.0, 1.0 - (self.episode_count / self.survival_decay_episodes))
        return {
            'current_stage': f'adaptive (survival_factor={survival_factor:.2f})',
            'episode': self.episode_count,
            'transitions': []
        }


def create_simple_reward_shaper(variant="quadratic"):
    """
    Factory function to create simple reward shapers.

    Args:
        variant: Reward variant or shaper type
            - "quadratic", "exponential", "linear", "sparse", "light_penalty"
            - "adaptive": Adaptive simple shaper

    Returns:
        Initialized reward shaper
    """
    if variant == "adaptive":
        return AdaptiveSimpleRewardShaper()
    else:
        return SimpleRewardShaper(variant=variant)


def test_simple_reward_shaper():
    """Test simple reward shaper with various scenarios."""
    print("Testing Simple Reward Shapers")
    print("=" * 70)

    # Test scenarios
    scenarios = [
        {"name": "Survival", "lines": 0, "done": False, "holes": 10, "max_height": 15},
        {"name": "Single clear", "lines": 1, "done": False, "holes": 8, "max_height": 12},
        {"name": "Double clear", "lines": 2, "done": False, "holes": 5, "max_height": 10},
        {"name": "Triple clear", "lines": 3, "done": False, "holes": 3, "max_height": 8},
        {"name": "Tetris!", "lines": 4, "done": False, "holes": 0, "max_height": 5},
        {"name": "Game over", "lines": 0, "done": True, "holes": 50, "max_height": 20},
    ]

    # Test all variants
    for variant in ["quadratic", "exponential", "linear", "sparse", "light_penalty"]:
        print(f"\n{variant.upper()} variant:")
        print("-" * 70)

        shaper = SimpleRewardShaper(variant=variant)

        for scenario in scenarios:
            info = {
                'lines_cleared': scenario['lines'],
                'holes': scenario['holes'],
                'max_height': scenario['max_height']
            }

            reward = shaper.calculate_reward(
                obs=None,
                action=0,
                base_reward=0.0,
                done=scenario['done'],
                info=info
            )

            print(f"  {scenario['name']:15s}: {reward:+8.1f}")

    # Test adaptive
    print(f"\nADAPTIVE variant (at different episodes):")
    print("-" * 70)

    shaper = AdaptiveSimpleRewardShaper(survival_decay_episodes=5000)

    for episode in [0, 1000, 2500, 5000, 10000]:
        shaper.update_episode(episode)
        info = {'lines_cleared': 0, 'holes': 10, 'max_height': 15}

        reward = shaper.calculate_reward(
            obs=None, action=0, base_reward=0.0, done=False, info=info
        )

        stage_info = shaper.get_stage_info()
        print(f"  Episode {episode:5d} (survival): {reward:+6.2f} | Stage: {stage_info['current_stage']}")

    print("\n" + "=" * 70)
    print("✅ Simple reward shaper tests completed!")


if __name__ == "__main__":
    test_simple_reward_shaper()
