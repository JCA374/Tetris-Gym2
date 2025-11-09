"""Verify that lines can actually be cleared"""
import gymnasium as gym
import tetris_gymnasium.envs

env = gym.make('tetris_gymnasium/Tetris', render_mode='ansi', height=20, width=10)

print("Testing if environment can clear lines...")
print("Strategy: Drop pieces in one column to build up and clear")
print()

obs, info = env.reset()

actions_taken = []
total_lines = 0

for step in range(200):
    # Strategy: mostly drop straight down with occasional rotations
    if step % 10 == 0:
        action = 3  # Rotate
    elif step % 5 == 0:
        action = 0  # Move left
    else:
        action = 5  # Hard drop

    actions_taken.append(action)
    obs, reward, terminated, truncated, info = env.step(action)

    # Check BOTH field names
    lines_cleared = info.get('lines_cleared', 0)
    lines_number = info.get('number_of_lines', 0)

    if lines_cleared > 0 or lines_number > 0:
        total_lines += max(lines_cleared, lines_number)
        print(f"🎉 Step {step}: Lines cleared!")
        print(f"   info['lines_cleared'] = {lines_cleared}")
        print(f"   info.get('number_of_lines', 0) = {lines_number}")
        print(f"   Reward: {reward}")
        print(f"   Info dict: {info}")
        print()

    if terminated or truncated:
        print(f"Game ended at step {step}")
        break

print(f"\nTotal lines cleared: {total_lines}")
print(f"Field name in info dict: 'lines_cleared' = {info.get('lines_cleared', 'missing')}")
print(f"Field name training looks for: 'number_of_lines' = {info.get('number_of_lines', 'MISSING')}")

if total_lines > 0:
    print("\n✅ Environment CAN clear lines!")
    print("❌ But training script looks for wrong field name!")
else:
    print("\n❌ Environment cannot clear lines")
