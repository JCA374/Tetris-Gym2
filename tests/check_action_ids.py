#!/usr/bin/env python3
"""Check actual action ID mapping"""

from tetris_gymnasium.envs import Tetris

env = Tetris()

print("Action space:", env.action_space)
print("Actions enum:", env.actions)
print()

for i in range(8):
    action_name = env.actions(i).name
    print(f"Action {i}: {action_name}")

print()
print("Specific actions:")
print(f"  move_left = {env.actions.move_left}")
print(f"  move_right = {env.actions.move_right}")
print(f"  move_down = {env.actions.move_down}")
print(f"  rotate_clockwise = {env.actions.rotate_clockwise}")
print(f"  rotate_counterclockwise = {env.actions.rotate_counterclockwise}")
print(f"  hard_drop = {env.actions.hard_drop}")
print(f"  swap = {env.actions.swap}")
print(f"  no_op = {env.actions.no_op}")

env.close()
