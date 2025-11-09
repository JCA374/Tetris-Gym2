# config.py
"""Configuration for Tetris RL Training - FIXED with 3D observations"""

import numpy as np
import gymnasium as gym

# Import tetris_gymnasium to register environments
import tetris_gymnasium.envs

# ✅ FIXED: Add all variables that train.py expects
ENV_NAME = 'tetris_gymnasium/Tetris'
LR = 0.0001
MAX_EPISODES = 10000
MODEL_DIR = "models/"
LOG_DIR = "logs/"

# Training Configuration
EPISODES = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
MIN_MEMORY_SIZE = 1000
TARGET_UPDATE_FREQUENCY = 1000

# Epsilon (exploration) schedule
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9999

# Model architecture
HIDDEN_UNITS = [512, 256, 128]

# Environment configuration
ENV_CONFIG = {
    'height': 20,
    'width': 10,
}

# Reward shaping modes
REWARD_SHAPING_MODE = 'balanced'

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 500
MILESTONE_INTERVAL = 1000

# FIXED: Correct action mapping for tetris-gymnasium v0.3.0
# ActionsMapping(move_left=0, move_right=1, move_down=2, rotate_clockwise=3,
#                rotate_counterclockwise=4, hard_drop=5, swap=6, no_op=7)
ACTION_LEFT=0; ACTION_RIGHT=1; ACTION_DOWN=2; ACTION_ROTATE_CW=3
ACTION_ROTATE_CCW=4; ACTION_HARD_DROP=5; ACTION_SWAP=6; ACTION_NOOP=7
ACTION_MEANINGS = {0:"LEFT",1:"RIGHT",2:"DOWN",3:"ROTATE_CW",4:"ROTATE_CCW",5:"HARD_DROP",6:"SWAP",7:"NOOP"}

def discover_action_meanings(env): 
    # set/verify the constants above if env exposes meanings
    return ACTION_MEANINGS


def make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False, use_feature_channels=True):
    """
    Create Tetris environment with optional enhanced vision

    Args:
        render_mode: Rendering mode ('rgb_array', 'human', None)
        use_complete_vision: If True, wrap to convert dict to array
        use_cnn: Not used, kept for compatibility
        use_feature_channels: If True, add explicit feature heatmap channels (8-channel mode)
                             If False, use visual-only (4-channel mode)

    Returns:
        Gymnasium environment with (20, 10, 4) or (20, 10, 8) observation space
    """
    # Create base environment
    env = gym.make(
        ENV_NAME,
        render_mode=render_mode,
        **ENV_CONFIG
    )

    print(f"✅ Environment created: {env.spec.id}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Complete vision: {use_complete_vision}")
    print(f"   Feature channels: {use_feature_channels}")

    # Wrap environment to convert dict observations to arrays
    if use_complete_vision:
        env = CompleteVisionWrapper(env, use_feature_channels=use_feature_channels)

    return env


class CompleteVisionWrapper(gym.ObservationWrapper):
    """
    Enhanced wrapper to convert dict observations to multi-channel array for CNN

    Supports two modes:
    - 4-channel mode (original): Visual information only
    - 8-channel mode (enhanced): Visual + explicit feature heatmaps

    Visual Channels (0-3):
    - Channel 0: Board state (locked pieces)
    - Channel 1: Active tetromino (falling piece)
    - Channel 2: Holder (held piece for swap)
    - Channel 3: Queue (preview of next pieces)

    Feature Channels (4-7, if enabled):
    - Channel 4: Holes heatmap (where holes exist)
    - Channel 5: Height map (normalized column heights)
    - Channel 6: Bumpiness map (height variation)
    - Channel 7: Wells map (valleys between columns)

    Output shape: (20, 10, 4) or (20, 10, 8) depending on use_feature_channels
    Output dtype: float32, values in [0, 1]
    """

    def __init__(self, env, use_feature_channels=True):
        super().__init__(env)

        self.use_feature_channels = use_feature_channels
        n_channels = 8 if use_feature_channels else 4

        # Define new observation space - 3D with 4 or 8 channels
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(20, 10, n_channels),
            dtype=np.float32  # Changed from uint8 for normalized values
        )

        if use_feature_channels:
            print("🎯 CompleteVisionWrapper initialized (8-CHANNEL HYBRID):")
            print(f"   Input space: {env.observation_space}")
            print(f"   Output space: {self.observation_space}")
            print(f"   Visual channels: Board | Active | Holder | Queue")
            print(f"   Feature channels: Holes | Heights | Bumpiness | Wells")
        else:
            print("🎯 CompleteVisionWrapper initialized (4-CHANNEL VISUAL):")
            print(f"   Input space: {env.observation_space}")
            print(f"   Output space: {self.observation_space}")
            print(f"   Channels: Board | Active Piece | Holder | Queue")
    
    def observation(self, obs_dict):
        """
        Extract multi-channel game state from observation dict.

        Tetris Gymnasium raw board structure (24x18):
        - Rows 0-1:   Top spawn area
        - Rows 2-19:  Main playable area (18 rows)
        - Rows 20-23: Bottom wall (4 rows, always filled - NOT playable)
        - Cols 0-3:   Left wall (4 columns, always filled)
        - Cols 4-13:  Playable area (10 columns)
        - Cols 14-17: Right wall (4 columns, always filled)

        Args:
            obs_dict: Dictionary with 'board', 'active_tetromino_mask', 'holder', 'queue'

        Returns:
            numpy array (20, 10, 4) or (20, 10, 8) with normalized values [0, 1]:
            Visual channels (always):
            - Channel 0: Board state (locked pieces)
            - Channel 1: Active piece (falling tetromino)
            - Channel 2: Holder (held piece for swap)
            - Channel 3: Queue (preview of next pieces)
            Feature channels (if use_feature_channels=True):
            - Channel 4: Holes heatmap
            - Channel 5: Height map
            - Channel 6: Bumpiness map
            - Channel 7: Wells map
        """
        if not isinstance(obs_dict, dict):
            # Fallback for non-dict observations
            board = np.array(obs_dict, dtype=np.float32)
            if len(board.shape) == 2:
                # Convert single 2D to multi-channel
                board_binary = (board > 0).astype(np.float32)
                n_ch = 8 if self.use_feature_channels else 4
                return np.stack([board_binary] * n_ch, axis=-1)
            elif len(board.shape) == 3:
                return board.astype(np.float32)
            return board

        # Initialize channels list
        channels = []

        # === CHANNEL 0: BOARD (locked pieces) ===
        board = obs_dict.get('board', np.zeros((24, 18), dtype=np.uint8))
        if board.shape == (24, 18):
            board_playable = board[0:20, 4:14]  # Extract rows 0-19, cols 4-13
        elif board.shape == (20, 10):
            board_playable = board
        else:
            board_playable = self._resize_to_playable(board)

        # Normalize to [0, 1]
        board_channel = (board_playable > 0).astype(np.float32)
        channels.append(board_channel)

        # === CHANNEL 1: ACTIVE TETROMINO (falling piece) ===
        mask = obs_dict.get('active_tetromino_mask', np.zeros((24, 18), dtype=np.uint8))
        if mask.shape == (24, 18):
            mask_playable = mask[0:20, 4:14]  # Same extraction as board
        elif mask.shape == (20, 10):
            mask_playable = mask
        else:
            mask_playable = self._resize_to_playable(mask)

        mask_channel = (mask_playable > 0).astype(np.float32)
        channels.append(mask_channel)

        # === CHANNEL 2: HOLDER (held piece) ===
        holder = obs_dict.get('holder', np.zeros((4, 4), dtype=np.uint8))
        holder_channel = np.zeros((20, 10), dtype=np.float32)

        # Place holder in top-left corner (scaled to 4x4 region)
        if holder.shape == (4, 4):
            holder_binary = (holder > 0).astype(np.float32)
            holder_channel[0:4, 0:4] = holder_binary

        channels.append(holder_channel)

        # === CHANNEL 3: QUEUE (next pieces preview) ===
        queue = obs_dict.get('queue', np.zeros((4, 16), dtype=np.uint8))
        queue_channel = np.zeros((20, 10), dtype=np.float32)

        # Place queue preview in top-right corner (4 pieces × 4 cells each)
        if queue.shape == (4, 16):
            queue_binary = (queue > 0).astype(np.float32)
            # Take first 10 columns (2.5 pieces) and place in top-right
            queue_preview = queue_binary[:, :10]
            queue_channel[0:4, 0:10] = queue_preview

        channels.append(queue_channel)

        # === ADD FEATURE CHANNELS (if enabled) ===
        if self.use_feature_channels:
            # Import feature heatmap functions
            from src.feature_heatmaps import compute_all_feature_heatmaps

            # Compute all 4 feature heatmaps from board state
            holes_map, height_map, bumpiness_map, well_map = compute_all_feature_heatmaps(board_playable)

            # Add to channels (already normalized to [0, 1])
            channels.extend([
                holes_map,      # Channel 4
                height_map,     # Channel 5
                bumpiness_map,  # Channel 6
                well_map        # Channel 7
            ])

        # Stack all channels: (20, 10, 4) or (20, 10, 8)
        observation = np.stack(channels, axis=-1).astype(np.float32)

        return observation

    def _resize_to_playable(self, array):
        """Helper to resize any array to (20, 10)"""
        target = np.zeros((20, 10), dtype=np.uint8)
        h, w = min(20, array.shape[0]), min(10, array.shape[1])
        target[:h, :w] = array[:h, :w]
        return target


def test_environment():
    """Test function to verify environment works"""
    print("\n🧪 Testing environment...")
    
    env = make_env(render_mode="rgb_array", use_complete_vision=True)
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"✅ Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print(f"   Observation range: [{obs.min()}, {obs.max()}]")
    
    # Verify 3D shape
    assert len(obs.shape) == 3, f"Expected 3D observation, got shape {obs.shape}"
    assert obs.shape == (20, 10, 1), f"Expected (20, 10, 1), got {obs.shape}"
    
    # Test steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Verify shape consistency
        assert obs.shape == (20, 10, 1), f"Shape changed to {obs.shape}!"
        
        if terminated or truncated:
            obs, info = env.reset()
            break
    
    print(f"✅ Environment steps work - Total reward: {total_reward:.2f}")
    print(f"✅ Shape consistency verified: all observations are (20, 10, 1)")
    
    env.close()
    return True


if __name__ == "__main__":
    # Test the configuration
    print("="*60)
    print("Testing Tetris Configuration")
    print("="*60)
    
    test_environment()
    
    print("\n✅ Configuration test passed!")
    print(f"\nConfiguration values:")
    print(f"  ENV_NAME: {ENV_NAME}")
    print(f"  LR: {LR}")
    print(f"  GAMMA: {GAMMA}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  MAX_EPISODES: {MAX_EPISODES}")
    print(f"  MODEL_DIR: {MODEL_DIR}")
    print(f"  LOG_DIR: {LOG_DIR}")