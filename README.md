# Snake Game AI with Deep Q-Network (DQN) 🐍🤖

This repository contains a Deep Reinforcement Learning project that trains an AI agent to play the classic Snake game using PyTorch and Pygame.

## Project Structure 📁

- `main.py`: The primary entry point of the project. Run this to start the training.
- `train.py`: Contains the game simulator (`SnakeGameAI`), the replay buffer memory, the agent setup, and the training loop.
- `evaluate/`: Folder containing scripts and documentation for evaluating the best model in a new custom environment. See [evaluate/README.md](evaluate/README.md) for results.
- `src/models/base_model.py`: Defines the `DQN` neural network model.
- `experiments/`: Stores the saved model weights (`model.pth`) and the persistent highest score record (`best_score.txt`).

---

## How It Works 💡

The agent uses a **Deep Q-Network (DQN)** to learn how to play Snake via trial and error.

### 1. State Representation (32 Features)
The environment states are represented as 32 binary values (True/False):
- **Local $5 \times 5$ Grid (24 features)**: A local vision grid centered around the snake's head (excluding the head itself). This tells the agent if there is a collision (wall or body) at any of the surrounding 24 coordinates, helping the snake identify traps and avoid hitting itself ("own hits") as it grows longer.
- **Direction (4 features - Left, Right, Up, Down)**: Current movement direction of the snake.
- **Food Direction (4 features - Left, Right, Up, Down)**: Location of the food relative to the snake's head.

### 2. Actions (3 Discrete Outputs)
At each step, the agent decides one of three directions relative to its current heading:
- `[1, 0, 0]`: Go Straight
- `[0, 1, 0]`: Turn Right
- `[0, 0, 1]`: Turn Left

### 3. Rewards (with Reward Shaping)
- **Eat Food**: `+10`
- **Collision / Game Over**: `-10`
- **Moving closer to food**: `+1` (Reward shaping reward)
- **Moving further from food**: `-1` (Reward shaping penalty)

### 4. Best Model Checkpoint & Fine-Tuning 💾
- The model automatically saves the best parameters (`model.pth`) and the high score (`best_score.txt`) in the `experiments/` directory whenever a new high record is achieved.
- **Robust Loading**: When running the project, it loads the saved checkpoint. If the saved model is incompatible (e.g. from an old architecture with different dimensions), it catches the mismatch gracefully and initializes training with a fresh model without crashing.

---

## Recent Improvements & Training Results 🏆

1. **Fixed Gradient Flow Bug in Training Targets**:
   - Detached the target Q-value matrix and next-state Q-predictions from the PyTorch computation graph. Previously, backpropagating gradients through target calculations caused training instability.
2. **Upgraded State Representation (11-dim ➡️ 32-dim)**:
   - Expanded state input space to include a $5 \times 5$ surrounding grid check, providing full local visual context to prevent the snake from coiling into itself.
3. **Deepened DQN Architecture**:
   - Configured `DQN` to utilize `state_dim=32` and enhanced the dense layers (`Linear(32, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 3)`) to accommodate the richer spatial inputs.
4. **Distance-Based Reward Shaping**:
   - Incentivizes the agent at every step based on Euclidean distance change to food. Moving closer yields `+1`, while moving away incurs `-1`. This eliminates the issue of sparse reward signal.
5. **Slow Exploration Decay with Minimum Floor**:
   - Decays exploration rate (`epsilon`) slower over 280 games and maintains a $5\%$ exploration minimum (`epsilon=10`) to prevent the policy from getting trapped in infinite loops.
6. **Dynamic Obstacle Training & Loop Prevention (New)**:
   - Added **15 randomized obstacles** generated on reset in [train.py](train.py) (safeguarding the starting area) to teach the snake spatial obstacle navigation.
   - Implemented a **step penalty of `-0.05`** for every step without food to break circular loops.
   - Implemented a **loop timeout penalty of `-15`** to severely punish state lock/looping behavior.

### Training Results (300 Games)
After implementing these fixes and improvements, the model trained successfully for 300 games, achieving a **best high score record of 38** (and consistently scoring 15-25+ in the later stages).

---

## Evaluation & Explainable AI (XAI) Results 📈🧪

The trained DQN model was evaluated in brand-new environments with obstacles to test generalization, including a gradient-based XAI analysis to interpret model decisions. Detailed explanations of looping limits, saliency heatmaps, and plots can be found in the [Evaluation README](evaluate/README.md).

* **Standard Evaluation (`evaluate.py`)**:
  * **Average Score**: **`30.50`** (Improved from the earlier baseline)
  * **Max Score**: `41` | **Min Score**: `12`
* **Hard/Large Evaluation (`evaluate_hard.py`)**:
  * **Average Score**: **`13.00`**
  * **Max Score**: `40` | **Min Score**: `0`
* **XAI Saliency Evaluation (`evaluate_xai.py`) [NEW]**:
  * **Steps**: `259` | **Final Score**: `16` (in standard obstacle environment)
  * **Q-Values Over Time**: Verifies clear action preferences and dynamic adaptation near hazards.
  * **Saliency Heatmap**: Confirms the model focuses heavily on cells **directly adjacent** to the head, showing it has successfully learned spatial collision avoidance.
* **Two Foods Evaluation (`evaluate_two_foods.py`) [NEW]**:
  * Spawns **2 foods simultaneously**.
  * **Average Score**: **`27.80`** | **Max Score**: `33` | **Min Score**: `14`
  * **Analysis**: While having two resources reduces travel distance, it introduces *target oscillation* (toggling back and forth between equidistant foods), which slightly penalizes the score compared to the highly-optimized standard baseline of `30.50`.

---

## How to Run 🚀

This project uses **[uv](https://github.com/astral-sh/uv)** for fast, reliable Python package and environment management.

### 1. Install uv
If you don't have `uv` installed, you can install it using pip:
```bash
pip install uv
```
*For other installation methods (e.g., curl, Homebrew), check the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).*

### 2. Set Up Virtual Environment & Dependencies
Initialize and sync the virtual environment with the project dependencies specified in `pyproject.toml`:
```bash
uv sync
```
This command automatically creates a `.venv` directory and installs all the required packages (PyTorch, Pygame, NumPy, Matplotlib, Seaborn).

### 3. Run Training
To start training the DQN model, execute the main entrypoint:
```bash
uv run python main.py
```
A Pygame window will open showing the snake training in real-time. Training progress and best score records will be outputted to your terminal.

### 4. Run Evaluation & XAI Saliency Analysis
To evaluate the best model checkpoint (`experiments/model.pth`) in the custom evaluation environments:
```bash
# Standard evaluation
uv run python evaluate/evaluate.py

# Hard evaluation (Large board + dense barriers)
uv run python evaluate/evaluate_hard.py

# Explainable AI (XAI) Saliency evaluation (generates analysis plots)
uv run python evaluate/evaluate_xai.py

# Two Foods evaluation (spawns 2 foods simultaneously)
uv run python evaluate/evaluate_two_foods.py
```
The analysis plots will be saved inside the `evaluate/plots/` directory.
