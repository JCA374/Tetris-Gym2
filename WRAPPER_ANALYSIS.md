# CompleteVisionWrapper Issue: Thorough Analysis & Recommendations

## Executive Summary

**CRITICAL BUG IDENTIFIED**: The `CompleteVisionWrapper` incorrectly includes bottom wall rows from the raw tetris-gymnasium environment, causing the bottom 2 rows of the wrapped observation to always be completely filled.

---

## 1. Raw Environment Structure (tetris-gymnasium)

The raw `tetris_gymnasium/Tetris` environment provides a **24×18 board**:

```
Rows 0-1:   Top spawn area (partially visible, includes active piece)
Rows 2-19:  Main playable area (18 rows, only side walls filled)
Rows 20-23: Bottom wall (4 rows, COMPLETELY FILLED - not playable)

Cols 0-3:   Left wall (4 columns, always filled)
Cols 4-13:  Playable area (10 columns)
Cols 14-17: Right wall (4 columns, always filled)
```

**Visual representation:**
```
Row  0: ████....XXX...████  ← Spawn area + active piece
Row  1: ████....XXX...████  ← Spawn area + active piece
Row  2: ████..........████  ← Start of playable area
...
Row 19: ████..........████  ← End of playable area
Row 20: ██████████████████  ← WALL (not playable!)
Row 21: ██████████████████  ← WALL (not playable!)
Row 22: ██████████████████  ← WALL (not playable!)
Row 23: ██████████████████  ← WALL (not playable!)
```

---

## 2. Current Wrapper Implementation (BUGGY)

**Current extraction in `src/env_wrapper.py` line 32:**
```python
board = full_board[2:22, 4:14]  # Extract 20x10 playable area
```

This extracts:
- **Rows 2-21** (20 rows) from the raw board
- **Columns 4-13** (10 columns)

### Problem:

**Rows 20-21 from the raw board are WALLS**, not playable area!

This means the wrapped 20×10 board has:
- Rows 0-17: Actual playable area (18 rows)
- **Rows 18-19: WALLS (always completely filled)** ❌

### Impact on Tests and Training:

1. **`test_actions_simple.py` fails**: All columns [0-9] are detected as "filled" from the start due to bottom walls
2. **`investigate_line_clearing.py` detects false full rows**: Rows 18-19 are always full but never cleared
3. **Agent training**: The neural network sees bottom 2 rows as always filled, wasting observation space and potentially learning incorrect patterns
4. **Reward shaping functions**: Metrics like `calculate_aggregate_height()` and `count_holes()` include wall rows in calculations

---

## 3. Tested Extraction Strategies

| Strategy | Rows | Shape | Bottom Walls? | Status |
|----------|------|-------|---------------|--------|
| **Current** `[2:22, 4:14]` | 2-21 | 20×10 | Yes, 2 rows ❌ | **BUGGY** |
| **Option A** `[0:20, 4:14]` | 0-19 | 20×10 | No ✓ | **CORRECT** |
| **Option B** `[4:24, 4:14]` | 4-23 | 20×10 | Yes, 4 rows ❌ | Wrong |
| **Option C** `[2:20, 4:14]` | 2-19 | 18×10 | No ✓ | Wrong shape |

**Option A is the correct extraction.**

---

## 4. Recommended Fix

### 4.1 Primary Fix: Update CompleteVisionWrapper

**File:** `src/env_wrapper.py`

**Change line 32 from:**
```python
board = full_board[2:22, 4:14]  # Extract 20x10 playable area
```

**To:**
```python
board = full_board[0:20, 4:14]  # Extract 20x10 playable area (rows 0-19, no walls)
```

### 4.2 Explanation

- **Rows 0-19** include the full spawn area and playable area
- **Rows 20-23** are walls and should be excluded
- This gives us a true 20×10 playable observation without wall contamination

### 4.3 Alternative Approach (if spawn area is unwanted)

If you want to exclude the spawn area rows and only keep the visible playable area:

```python
board = full_board[2:20, 4:14]  # Extract 18x10 (visible play area only, no spawn or walls)
```

But this changes the shape to **18×10** and would require updating the observation space definition on line 15:
```python
self.observation_space = gym.spaces.Box(
    low=0, high=255, shape=(18, 10, 1), dtype=np.uint8  # Changed from (20, 10, 1)
)
```

**Recommendation:** Use Option A `[0:20, 4:14]` to keep the 20×10 shape.

---

## 5. Impact of Fix

### 5.1 Tests That Will Pass After Fix:
- ✅ `test_actions_simple.py` - Will correctly detect LEFT/RIGHT movement
- ✅ `investigate_line_clearing.py` - Will only detect actual line clears, not wall rows

### 5.2 Code That Benefits:
- ✅ Reward shaping functions will calculate correct metrics
- ✅ Neural network will use observation space efficiently
- ✅ Visual debugging will show accurate board state

### 5.3 Potential Breakage:
⚠️ **Trained models may be affected** if they learned patterns based on the buggy observation!

**Mitigation:**
1. Save current models before applying fix
2. Retrain from scratch after fix (recommended)
3. Or fine-tune existing models with new observations

---

## 6. Additional Recommendations

### 6.1 Fix test_actions_simple.py

Even after fixing the wrapper, `test_actions_simple.py` needs updates:

**Current issue:** Uses 30 cycles which places ~5-6 pieces

**Fix:** Reduce cycles or change test logic:
```python
# Option 1: Use fewer cycles (place only 1 piece)
CYCLES = 5  # Instead of BOARD_W * 3

# Option 2: Check specific rows/columns instead of comparing full column sets
left_touched_edge = np.any(board[:, 0:2] > 0)  # Check leftmost columns
right_touched_edge = np.any(board[:, 8:10] > 0)  # Check rightmost columns
```

### 6.2 Update Reward Shaping

Ensure reward shaping functions don't make assumptions about always-filled bottom rows:

**File:** `src/reward_shaping.py`

Add parameter to exclude bottom rows from calculations if needed:
```python
def calculate_aggregate_height(board, exclude_bottom_rows=0):
    """Calculate total height across all columns"""
    if exclude_bottom_rows > 0:
        board = board[:-exclude_bottom_rows, :]
    # ... rest of function
```

---

## 7. Testing Plan

### Before Fix:
1. Run diagnostic: `python tests/analyze_wrapper_deep.py`
2. Observe: Bottom 2 rows are walls ❌

### Apply Fix:
1. Update `src/env_wrapper.py` line 32
2. Change `board = full_board[2:22, 4:14]` to `board = full_board[0:20, 4:14]`

### After Fix:
1. Run diagnostic again: `python tests/analyze_wrapper_deep.py`
2. Verify: No bottom wall rows ✅
3. Run tests:
   ```bash
   python -u tests/test_piece_movement_diagnostics.py
   python -u tests/test_outer_column_fill_once.py
   python -u tests/investigate_line_clearing.py
   ```
4. Retrain model and compare performance

---

## 8. Conclusion

**Root Cause:** Incorrect row extraction range in CompleteVisionWrapper

**Impact:** High - affects observations, tests, and trained models

**Fix Difficulty:** Easy - one line change

**Fix Risk:** Medium - requires model retraining

**Status:** Ready to implement
