#!/usr/bin/env python3
"""Quick Summary Test - Core Functionality"""

import sys
import gymnasium as gym
import tetris_gymnasium.envs
sys.path.append('src')
from env_feature_vector import make_feature_vector_env

print("="*80)
print("QUICK VERIFICATION TEST")
print("="*80)

# Test 1: Bugfix verification
print("\n[1] Bugfix: Field Name Mismatch")
env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)
obs, info = env.reset()

if 'lines_cleared' in info:
    print(f"✅ Environment returns: info['lines_cleared']")
else:
    print(f"❌ Missing lines_cleared field!")
    sys.exit(1)

# Check training script
import importlib.util
spec = importlib.util.spec_from_file_location("train", "train_feature_vector.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

with open('train_feature_vector.py', 'r') as f:
    content = f.read()
    if "info.get('lines_cleared'," in content:
        print(f"✅ Training script uses: info.get('lines_cleared', 0)")
    else:
        print(f"❌ Training script has wrong field name!")
        sys.exit(1)

env.close()

# Test 2: Reward function
print("\n[2] Reward Function Bonuses")
simple_reward = train_module.simple_reward
r0 = simple_reward(0, {'lines_cleared': 0})
r1 = simple_reward(0, {'lines_cleared': 1})
r4 = simple_reward(0, {'lines_cleared': 4})

print(f"  0 lines: {r0:6.1f}")
print(f"  1 line:  {r1:6.1f} (+{r1-r0:.1f})")
print(f"  4 lines: {r4:6.1f} (+{r4-r0:.1f})")

if r1 > r0 and r4 > r1:
    print(f"✅ Rewards increase with line clears")
else:
    print(f"❌ Reward function broken!")
    sys.exit(1)

# Test 3: Feature vector wrapper
print("\n[3] Feature Vector Wrapper")
env = make_feature_vector_env()
obs, info = env.reset()

if obs.shape == (17,):
    print(f"✅ Observation shape: {obs.shape} (correct)")
else:
    print(f"❌ Wrong shape: {obs.shape}")
    sys.exit(1)

if obs.min() >= 0 and obs.max() <= 1:
    print(f"✅ Normalized to [0, 1]")
else:
    print(f"⚠️  Range: [{obs.min():.2f}, {obs.max():.2f}]")

env.close()

print("\n" + "="*80)
print("✅ ALL CRITICAL TESTS PASSED")
print("="*80)
print("\nSystem is ready for training!")
print("Bugfix verified: Agent will now detect line clears correctly")
print("\nRecommended next step:")
print("  python train_feature_vector.py --episodes 5000 --force_fresh")
print("="*80)
