Great — I’ll generate the file content here for you. Just copy it into a file named **`center_stacking_fix.md`**.

````markdown
# Full-Width Movement Analysis & Fix

## ✅ True Behavior

**tetris-gymnasium v0.3.0** uses:
- Internal board: **24×18** (walls + 10×20 playable)
- Playable columns: **4–13 → corresponds to columns 0–9**
- Piece movement range: **x = 0–9**, unrestricted across the full width

So all 10 columns **are reachable**. Center-stacking arises from:
1. Insufficient LEFT/RIGHT actions in exploration.  
2. Early epsilon decay limiting movement diversity.  
3. Reward shaping not rewarding spread.

---

## 🚫 Common Misinterpretation

Earlier diagnostics saw empty outer columns because:
- The wrapper extracted `[2:22, 4:14]` correctly (playable 20×10),
- But the **agent never executed enough horizontal moves** before hard dropping.

There is **no environment limitation** — only behavioral bias.

---

## 🎯 Fix Summary

### 1. Exploration Fix (in `Agent.select_action`)
Ensure LEFT/RIGHT are sampled frequently:
```python
# 35% LEFT/RIGHT, 20% HARD_DROP, 15% DOWN, 20% ROTATE, 10% SWAP
````

### 2. Reward Fix

Encourage horizontal spread:

```python
spread = np.std(get_column_heights(board))
shaped += 4.0 * spread  # restore higher spread bonus
```

### 3. Evaluation Expectation

Balanced training should produce:

```
Column heights: [2, 4, 6, 8, 10, 9, 8, 6, 4, 2]
```

Center stacking = mild → GOOD
Full column usage = normal when exploration active.

---

## ✅ Final Notes

* Movement range: **fully 10 columns (0–9)**
* Wrapper `[2:22, 4:14]` is **correct**
* Outer columns reachable — issue was exploration, not environment

Train again with:

```bash
python train.py --episodes 3000 --reward_shaping positive --force_fresh --epsilon_decay 0.99995
```

and monitor column usage in logs.

```

If you like, I can **push this to your Git repository** or provide a **diff version** to apply to your documentation.
::contentReference[oaicite:0]{index=0}
```
