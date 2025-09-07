# Block Puzzle RL Game Implementation Plan

## Project Overview
Building a block puzzle game optimized for reinforcement learning training, with clean separation between game logic, visualization, and RL environment.

## Project Structure
```
block_puzzle_rl/
├── src/
│   ├── game/
│   │   ├── __init__.py
│   │   ├── core.py              # Core game logic
│   │   ├── pieces.py            # Block piece definitions
│   │   ├── grid.py              # Game grid management
│   │   └── rules.py             # Game rules and scoring
│   ├── env/
│   │   ├── __init__.py
│   │   ├── block_puzzle_env.py  # Gym environment wrapper
│   │   └── observations.py      # State representation utilities
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── train.py             # Training scripts
│   │   ├── evaluate.py          # Model evaluation
│   │   └── callbacks.py         # Training callbacks
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── renderer.py          # Game visualization
│   │   └── human_play.py        # Human playable interface
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── logging_utils.py     # Logging utilities
├── tests/
├── configs/                     # Training configurations
├── models/                      # Saved RL models
├── logs/                        # Training logs
├── requirements.txt
└── README.md
```

## Phase 1: Core Game Engine (Week 1)

### 1.1 Define Game Mechanics
**Decision Points:**
- Grid size (recommend 10x20 for Tetris-like, or 8x8 for puzzle variant)
- Piece types (7 standard Tetris pieces or custom set)
- Drop mechanics (gravity vs manual placement)
- Line clearing rules
- Scoring system

### 1.2 Implement Core Classes

**Grid Class (`src/game/grid.py`):**
```python
class GameGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
    
    def is_valid_position(self, piece, x, y, rotation):
        # Check collision detection
    
    def place_piece(self, piece, x, y, rotation):
        # Place piece on grid
    
    def clear_lines(self):
        # Remove complete lines, return count
    
    def get_state(self):
        # Return grid state for RL
```

**Piece Class (`src/game/pieces.py`):**
```python
class Piece:
    def __init__(self, shape, color):
        self.shape = shape  # 2D numpy array
        self.color = color
        self.rotation = 0
    
    def rotate(self, direction=1):
        # Rotate piece 90 degrees
    
    def get_rotated_shape(self, rotation):
        # Get shape at specific rotation
```

**Game Engine (`src/game/core.py`):**
```python
class BlockPuzzleGame:
    def __init__(self, config):
        self.grid = GameGrid(config.width, config.height)
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
    
    def spawn_piece(self):
        # Generate new piece
    
    def move_piece(self, dx, dy):
        # Move current piece
    
    def rotate_piece(self, direction=1):
        # Rotate current piece
    
    def drop_piece(self):
        # Drop piece one row
    
    def hard_drop(self):
        # Drop piece to bottom
    
    def update(self, action):
        # Main game update loop
        # Returns: new_state, reward, done, info
```

### 1.3 Game Rules Implementation (`src/game/rules.py`)
- Line clearing logic
- Scoring system
- Level progression
- Game over conditions

## Phase 2: Gym Environment Integration (Week 2)

### 2.1 Environment Wrapper (`src/env/block_puzzle_env.py`)

```python
import gym
from gym import spaces
import numpy as np

class BlockPuzzleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None):
        super().__init__()
        self.game = BlockPuzzleGame(config or DefaultConfig())
        
        # Define observation space
        # Option 1: Just grid
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.game.grid.height, self.game.grid.width),
            dtype=np.float32
        )
        
        # Option 2: Grid + piece info (more complex but informative)
        # self.observation_space = spaces.Dict({
        #     'grid': spaces.Box(low=0, high=1, shape=(height, width)),
        #     'current_piece': spaces.Box(low=0, high=7, shape=(4, 4)),
        #     'next_piece': spaces.Box(low=0, high=7, shape=(4, 4))
        # })
        
        # Define action space
        self.action_space = spaces.Discrete(6)  # [Left, Right, Rotate, Soft Drop, Hard Drop, No-op]
        
    def step(self, action):
        # Execute action in game
        obs, reward, done, info = self.game.update(action)
        return self._get_observation(), reward, done, info
    
    def reset(self):
        self.game.reset()
        return self._get_observation()
    
    def render(self, mode='human'):
        # Visualization for debugging
        pass
    
    def _get_observation(self):
        # Convert game state to RL observation
        return self.game.grid.grid.astype(np.float32)
```

### 2.2 Observation Design (`src/env/observations.py`)

**State Representation Options:**
1. **Simple Grid**: Just the current grid state (0s and 1s)
2. **Enhanced Grid**: Grid + current piece overlay
3. **Feature Engineering**: Height map, hole count, bumpiness, etc.
4. **Multi-channel**: Separate channels for placed blocks, current piece, ghost piece

### 2.3 Action Space Design

**Discrete Actions:**
- 0: Move Left
- 1: Move Right  
- 2: Rotate Clockwise
- 3: Rotate Counter-clockwise
- 4: Soft Drop (move down one)
- 5: Hard Drop (drop to bottom)
- 6: Hold/No-op

## Phase 3: Reward Function Design (Week 2)

### 3.1 Reward Components (`src/game/rules.py`)

```python
class RewardCalculator:
    def __init__(self, config):
        self.line_clear_reward = config.line_clear_reward
        self.placement_reward = config.placement_reward
        self.survival_reward = config.survival_reward
        self.height_penalty = config.height_penalty
        self.hole_penalty = config.hole_penalty
    
    def calculate_reward(self, prev_state, current_state, action, lines_cleared):
        reward = 0
        
        # Line clearing (primary positive reward)
        if lines_cleared > 0:
            reward += self.line_clear_reward * (lines_cleared ** 2)  # Bonus for multiple lines
        
        # Piece placement
        if self._piece_was_placed(prev_state, current_state):
            reward += self.placement_reward
        
        # Height penalty (encourage keeping stack low)
        max_height = self._get_max_height(current_state)
        reward -= self.height_penalty * max_height
        
        # Hole penalty (discourage creating holes)
        holes = self._count_holes(current_state)
        reward -= self.hole_penalty * holes
        
        # Survival reward
        reward += self.survival_reward
        
        # Game over penalty
        if self._is_game_over(current_state):
            reward -= 100
            
        return reward
```

### 3.2 Reward Tuning Strategy
- Start with sparse rewards (only line clears)
- Gradually add shaped rewards based on training performance
- A/B test different reward combinations
- Monitor for reward hacking

## Phase 4: Training Infrastructure (Week 3)

### 4.1 Training Script (`src/rl/train.py`)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

def train_model(config):
    # Create vectorized environment
    env = make_vec_env(
        lambda: BlockPuzzleEnv(config), 
        n_envs=config.n_envs
    )
    
    # Initialize model
    model = PPO(
        "MlpPolicy",  # Start with MLP, can upgrade to CNN later
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env, best_model_save_path='./models/',
        log_path='./logs/', eval_freq=10000
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, save_path='./models/checkpoints/'
    )
    
    # Train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    
    return model
```

### 4.2 Configuration Management (`src/utils/config.py`)

```python
from dataclasses import dataclass

@dataclass
class GameConfig:
    grid_width: int = 10
    grid_height: int = 20
    piece_types: int = 7
    max_episode_steps: int = 1000

@dataclass  
class RLConfig:
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_envs: int = 8
    total_timesteps: int = 1_000_000

@dataclass
class RewardConfig:
    line_clear_reward: float = 10.0
    placement_reward: float = 0.1
    survival_reward: float = 0.01
    height_penalty: float = 0.1
    hole_penalty: float = 0.5
```

## Phase 5: Visualization & Testing (Week 3)

### 5.1 Human Playable Interface (`src/visualization/human_play.py`)
- Pygame-based interface for manual testing
- Allows human vs AI comparison
- Debugging visualization for RL agent decisions

### 5.2 Rendering System (`src/visualization/renderer.py`)
- Grid visualization
- Piece rendering with colors
- Score/stats display
- RL agent action visualization

## Phase 6: Training & Optimization (Week 4+)

### 6.1 Baseline Training
1. **Environment Validation**: Test gym environment thoroughly
2. **Sanity Checks**: Train on simple rewards first
3. **Baseline Model**: Train PPO with basic reward function
4. **Performance Metrics**: Lines cleared per episode, survival time

### 6.2 Hyperparameter Optimization
- Grid search on learning rate, batch size
- Reward function coefficient tuning
- Network architecture experiments (MLP → CNN)

### 6.3 Advanced Techniques
- **Curriculum Learning**: Start with easier configurations
- **Self-Play**: If applicable to your puzzle variant
- **Feature Engineering**: Hand-crafted features vs raw grid
- **Multi-objective**: Balance different game objectives

## Development Workflow

### Sprint 1 (Days 1-3): Core Game
- [ ] Implement `GameGrid` class with basic operations
- [ ] Create `Piece` class with rotation mechanics
- [ ] Build `BlockPuzzleGame` main engine
- [ ] Write unit tests for collision detection
- [ ] Add basic console visualization

### Sprint 2 (Days 4-5): Gym Integration  
- [ ] Implement `BlockPuzzleEnv` wrapper
- [ ] Define observation and action spaces
- [ ] Test environment with `gym.Env.check()`
- [ ] Create basic reward function

### Sprint 3 (Days 6-7): Visualization
- [ ] Pygame rendering system
- [ ] Human playable interface
- [ ] RL agent visualization mode

### Sprint 4 (Week 2): RL Training
- [ ] Training pipeline with SB3
- [ ] Logging and monitoring setup
- [ ] Baseline model training
- [ ] Performance evaluation metrics

## Technical Considerations

### Performance Optimization
- Vectorized numpy operations for grid manipulation
- Efficient collision detection algorithms
- Memory-efficient state representation

### RL-Specific Design
- **Fast Environment**: Prioritize speed over visual fidelity for training
- **Deterministic**: Ensure reproducible results with random seeds
- **Configurable**: Easy to modify game parameters for experiments

### Testing Strategy
- Unit tests for game mechanics
- Integration tests for Gym environment
- Property-based testing for edge cases
- Manual testing with human play

## Potential Challenges & Solutions

### Challenge: Sparse Rewards
**Solution**: Implement reward shaping with intermediate goals

### Challenge: Large Action Space
**Solution**: Start with simplified action set, expand gradually

### Challenge: Training Instability  
**Solution**: Careful hyperparameter tuning, reward clipping

### Challenge: Sample Efficiency
**Solution**: Feature engineering, curriculum learning

## Success Metrics

### Game Implementation
- [ ] Game runs without crashes for 1000+ moves
- [ ] All piece rotations and movements work correctly
- [ ] Line clearing mechanics function properly
- [ ] Human can play and enjoy the game

### RL Integration
- [ ] Environment passes Gym validation
- [ ] Agent learns basic piece placement (>0 average reward)
- [ ] Agent achieves line clearing consistently
- [ ] Training runs stable for 1M+ timesteps

## Next Steps After Implementation

1. **Baseline Comparison**: Compare RL agent vs random policy vs human players
2. **Algorithm Comparison**: Test PPO vs DQN vs A2C performance  
3. **Architecture Experiments**: MLP vs CNN vs Transformer policies
4. **Transfer Learning**: Train on simple variant, transfer to complex
5. **Multi-Agent**: Self-play or competitive variants

## Quick Start Commands

```bash
# Setup environment
pip install -r requirements.txt

# Test game implementation
python -m src.game.core

# Validate Gym environment  
python -m src.env.block_puzzle_env

# Start training
python -m src.rl.train --config configs/default.yaml

# Play human vs AI
python -m src.visualization.human_play --model models/best_model.zip
```

## Resources & References

- **Stable Baselines 3**: [Documentation](https://stable-baselines3.readthedocs.io/)
- **OpenAI Gym**: [Environment Creation Guide](https://gymnasium.farama.org/tutorials/environment_creation/)
- **Tetris RL Papers**: Look up "Deep Q-Networks for Tetris" for inspiration
- **Reward Engineering**: "Reward Engineering for Object Placement"

## Timeline Estimate
- **Week 1**: Core game implementation and basic testing
- **Week 2**: Gym integration and reward function design  
- **Week 3**: Visualization and training pipeline setup
- **Week 4+**: Training, optimization, and experimentation

Start with the simplest possible implementation that works, then iterate and improve. The key is getting a training loop running quickly, even if the initial performance is poor.