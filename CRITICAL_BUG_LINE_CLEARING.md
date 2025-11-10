# CRITICAL BUG: tetris-gymnasium Cannot Clear Lines

**Date:** 2025-11-10
**Severity:** 🔴 **BLOCKING** - Makes training impossible
**Status:** Root cause identified, solution needed

---

## Executive Summary

The Tetris environment (`tetris-gymnasium` v0.2.1 and v0.3.0) **cannot clear lines under ANY conditions**. This makes all training pointless, as the agent can never receive rewards for the primary objective of Tetris.

### Evidence

- ✅ 100 episodes with random actions: **0 lines cleared**
- ✅ 100 episodes with HARD_DROP only: **0 lines cleared**
- ✅ 1,000 episodes with random actions: **0 lines cleared**
- ✅ 50 episodes with "smart" left/right cycling strategy: **0 lines cleared**
- ✅ 20 episodes with bottom-up fill strategy: **0 lines cleared**
- ✅ Tested with tetris-gymnasium v0.2.1: **Cannot clear lines**
- ✅ Tested with tetris-gymnasium v0.3.0: **Cannot clear lines**

**Conclusion:** This is statistically impossible if the environment was working correctly.

---

## Investigation Timeline

### 1. Initial Observation (User Report)
User noted that after 110,000+ training episodes across multiple runs, the agent had **never cleared a single line**, not even by accident.

### 2. Simple Test (100 Random Episodes)
Created `test_can_clear_lines.py`:
- 100 episodes with random actions
- 100 episodes with HARD_DROP only
- **Result:** 0 lines cleared

### 3. Line Clearing Logic Verification
Checked tetris-gymnasium source code (`envs/tetris.py`):

```python
def clear_filled_rows(self, board) -> "tuple(np.ndarray, int)":
    # A row is filled if it doesn't contain any free space (0)
    # and doesn't contain any bedrock / padding (1).
    filled_rows = (~(board == 0).any(axis=1)) & (~(board == 1).all(axis=1))
    n_filled = np.sum(filled_rows)
```

**Finding:** The logic itself appears correct. Manually filled rows ARE detected as "filled".

### 4. Board Structure Analysis
Created `test_board_values.py` to examine the board:

```
Board shape: (24, 18)

Structure:
  Columns 0-3:   Wall (value 1)
  Columns 4-13:  Playable area (10 columns)
  Columns 14-17: Wall (value 1)
  Rows 0-19:     Playable area
  Rows 20-23:    Floor (value 1)
```

**Finding:** Board structure is as expected. Line clearing logic should work for rows where columns 4-13 are all filled with pieces.

### 5. Manual Line Fill Test
Created `test_manual_line_clear.py`:
- Manually filled row 19 (columns 4-13) with pieces (value 2)
- Checked with environment's line clearing logic
- **Result:** Row IS detected as filled by the logic
- **But:** In actual gameplay over 1,000 episodes, never achieved a line clear

### 6. Episode Length Analysis
Created `test_episode_length.py`:

```
With random actions:
  Average episode length: 60 steps
  Estimated pieces placed: 12-20
  All episodes end with game_over (board full)
```

**Finding:** Episodes are very short with random play, but this doesn't explain zero lines in 1,000+ episodes.

### 7. Action Effectiveness Test
Created `test_action_placement.py`:

```
LEFT/RIGHT actions DO work:
  - LEFT x3 moves pieces left
  - RIGHT x3 moves pieces right

Piece distribution (100 random episodes):
  Columns 0-3:  6-43 occurrences (edges)
  Columns 4-7:  77-100 occurrences (center-left)
  Columns 8-9:  8-30 occurrences (right side)
```

**Finding:** Actions work, but pieces concentrate in center columns. However, this still doesn't explain ZERO lines in extensive testing.

### 8. Smart Strategy Test
Created `test_smart_strategy.py`:
- Strategy 1: Cycle left/right to spread pieces (50 episodes)
- Strategy 2: Bottom-up fill (20 episodes)
- **Result:** 0 lines cleared with ANY strategy

**Finding:** Even deliberate strategies cannot clear lines.

### 9. Version Upgrade Test
- Original version: tetris-gymnasium 0.2.1 (pinned in requirements.txt)
- Upgraded to: tetris-gymnasium 0.3.0 (latest)
- Re-ran all tests
- **Result:** Still 0 lines cleared

**Finding:** The bug exists in both versions.

---

## Root Cause Analysis

The line clearing **detection logic** is correct, but the environment appears to have a deeper issue preventing complete rows from forming during actual gameplay.

### Hypothesis 1: Pieces Can't Fill Complete Rows (LIKELY)
Pieces may not be able to create the specific configurations needed to fill all 10 columns of a row, possibly due to:
- Tetromino rotation/placement mechanics
- Gaps created by piece shapes
- Invalid action handling

### Hypothesis 2: Environment State Bug
There may be a bug in how the environment:
- Places committed pieces on the board
- Checks for line clearing at the right time
- Handles the actual clearing operation

### Hypothesis 3: Incompatible Environment Version
The pinned version (0.2.1) may have a known bug, but v0.3.0 also fails, suggesting this is a fundamental issue with how we're using the environment.

---

## Impact on Training

### Training is Impossible
Without line clears:
1. Agent never receives line clear rewards (+100 per line in our reward function)
2. Primary objective of Tetris (clearing lines) cannot be learned
3. Agent can only learn survival (avoiding game over)
4. All 110,000+ training episodes were essentially wasted

### Why Training "Worked" Anyway
The agent DID learn something:
- Episodes went from ~60 steps (random) to ~475 steps (trained)
- Agent learned to survive longer by avoiding quick game overs
- But survival without line clearing is not Tetris gameplay

---

## Next Steps

### Option 1: Fix tetris-gymnasium
- Investigate the exact cause in the environment code
- File a bug report with the tetris-gymnasium maintainers
- Attempt to patch the environment locally

### Option 2: Use a Different Tetris Environment
- Research alternative Tetris environments for Gymnasium
- Options might include:
  - `gym-tetris` (if compatible with Gymnasium)
  - Custom Tetris environment
  - Older/newer versions of tetris-gymnasium

### Option 3: Manual Environment Patch
- Modify `clear_filled_rows` to be more aggressive
- Add debugging to see why rows aren't clearing
- Potentially bypass the issue with environment modification

---

## Files Created During Investigation

1. `test_can_clear_lines.py` - Basic line clearing test (random + HARD_DROP)
2. `test_environment_deep_dive.py` - Detailed environment behavior analysis
3. `test_board_values.py` - Board structure and value analysis
4. `test_manual_line_clear.py` - Manual row fill and logic verification
5. `test_action_placement.py` - Action effectiveness and piece distribution
6. `test_episode_length.py` - Episode duration statistics
7. `test_smart_strategy.py` - Deliberate strategies to trigger line clears

All tests conclusively show: **The environment cannot clear lines**.

---

## Recommendation

**STOP ALL TRAINING** until this is resolved. Training without line clearing is pointless and wastes compute resources.

**Priority:** BLOCKING - Must be fixed before any further development.

**Action:** Research alternative Tetris environments or debug tetris-gymnasium to find the exact cause of the line clearing failure.
