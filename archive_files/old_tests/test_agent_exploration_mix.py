# tests/test_agent_exploration_mix.py
"""
Sanity check: Agent exploration must *never* sample NOOP and must
hit all non-NOOP actions with sensible frequency when epsilon=1.0.
"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import Agent
import gymnasium as gym
from config import make_env

def main():
    env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
    agent = Agent(env.observation_space, env.action_space,
                  epsilon_start=1.0, epsilon_end=1.0, max_episodes=10)
    obs, _ = env.reset(seed=42)

    counts = {i:0 for i in range(env.action_space.n)}
    for _ in range(5000):
        a = agent.select_action(obs, training=True)
        counts[a] += 1

    env.close()

    print("Action counts over 5k exploratory picks:", counts)
    assert counts.get(0, 0) == 0, "NOOP should not be sampled during exploration"
    # ensure each other action appears at least 2% (~100)
    for a in range(1, env.action_space.n):
        assert counts[a] >= 100, f"Action {a} underrepresented in exploration: {counts[a]}"

if __name__ == "__main__":
    main()
