#!/usr/bin/env python3
"""Test RAW gymnasium environment with NO wrappers - sanity check"""

from tetris_gymnasium.envs import Tetris

print("="*80)
print("SANITY CHECK: Raw tetris-gymnasium with NO custom code")
print("="*80)

# Make base env with ZERO wrappers - direct import
env = Tetris(render_mode=None)

print(f"\nEnvironment: {env}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Run 5000 steps with random actions
obs, info = env.reset(seed=42)
total_lines = 0
episodes = 0
steps = 0

print(f"\nRunning random actions for up to 5000 steps...\n")

for i in range(5000):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    steps += 1

    # Check for line clears
    lines = info.get('number_of_lines', 0)
    if lines > 0:
        total_lines += lines
        print(f"✅ Step {steps}: {lines} line(s) cleared! Total: {total_lines}")
        print(f"   Reward: {reward}")

    if term or trunc:
        episodes += 1
        print(f"   Episode {episodes} ended at step {steps}")
        if i < 4999:  # Don't reset on last iteration
            obs, info = env.reset()
        steps = 0

env.close()

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total lines cleared: {total_lines}")
print(f"Episodes completed: {episodes}")
print()

if total_lines > 0:
    print("✅ tetris-gymnasium CAN clear lines!")
    print("   The problem is in OUR wrapper/action mapping code")
else:
    print("❌ No lines cleared even in raw env")
    print("   This would be a library issue")

print("="*80)
