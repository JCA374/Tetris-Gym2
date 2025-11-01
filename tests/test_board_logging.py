#!/usr/bin/env python3
"""Test board state logging"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import TrainingLogger

print("="*80)
print("TESTING BOARD STATE LOGGING")
print("="*80)

# Create a test logger
logger = TrainingLogger("test_logs", "board_log_demo")

# Create some example board states

# Example 1: Center-stacking
print("\nCreating example 1: Center-stacking pattern")
board1 = np.zeros((20, 10), dtype=np.float32)
for c in [3, 4, 5, 6]:
    board1[-19:, c] = 1

logger.log_board_state(
    episode=100,
    board=board1,
    reward=-25.5,
    steps=45,
    lines_cleared=0,
    heights=[0, 0, 0, 19, 19, 19, 19, 0, 0, 0],
    holes=5,
    bumpiness=0.0,
    max_height=19
)

# Example 2: Balanced distribution
print("Creating example 2: Balanced distribution")
board2 = np.zeros((20, 10), dtype=np.float32)
heights2 = [5, 6, 8, 10, 12, 11, 9, 7, 5, 4]
for c, h in enumerate(heights2):
    if h > 0:
        board2[-h:, c] = 1

logger.log_board_state(
    episode=500,
    board=board2,
    reward=32.8,
    steps=180,
    lines_cleared=8,
    heights=heights2,
    holes=2,
    bumpiness=12.5,
    max_height=12
)

# Example 3: Game ending soon (high stack)
print("Creating example 3: High stack (near death)")
board3 = np.zeros((20, 10), dtype=np.float32)
heights3 = [15, 16, 18, 19, 20, 19, 18, 17, 14, 12]
for c, h in enumerate(heights3):
    if h > 0:
        board3[-h:, c] = 1

# Add some holes
board3[10, 0] = 0
board3[12, 2] = 0
board3[14, 5] = 0

logger.log_board_state(
    episode=750,
    board=board3,
    reward=-45.2,
    steps=220,
    lines_cleared=3,
    heights=heights3,
    holes=3,
    bumpiness=25.8,
    max_height=20
)

print("\n" + "="*80)
print("BOARD LOG CREATED!")
print("="*80)
print(f"\nBoard states saved to: {logger.board_log_path}")
print("\nPreview of the log file:\n")

# Show the content
with open(logger.board_log_path, 'r') as f:
    content = f.read()
    print(content)

print("="*80)
print("During training, this file will be located at:")
print("  logs/<experiment_name>/board_states.txt")
print("\nBoard states are logged every LOG_FREQ episodes (default: every 10 episodes)")
print("="*80)
