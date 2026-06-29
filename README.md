# Snake Game AI with Deep Q-Network (DQN) 🐍🤖

This repository contains a Deep Reinforcement Learning project that trains an AI agent to play the classic Snake game using PyTorch and Pygame.

## Project Structure 📁

- `main.py`: The primary entry point of the project. Run this to start the training.
- `train.py`: Contains the game simulator (`SnakeGameAI`), the replay buffer memory, the agent setup, and the training loop.
- `src/models/base_model.py`: Defines the `DQN` neural network model.
- `experiments/`: Stores the saved model weights (`model.pth`) and the persistent highest score record (`best_score.txt`).

---

## How It Works 💡

The agent uses a **Deep Q-Network (DQN)** to learn how to play Snake via trial and error.

### 1. State Representation (11 Features)
The environment states are represented as 11 binary values (True/False):
- **Danger (Straight, Right, Left)**: Tells the agent if moving in that direction results in a collision.
- **Direction (Left, Right, Up, Down)**: Current movement direction of the snake.
- **Food Direction (Left, Right, Up, Down)**: Location of the food relative to the snake's head.

### 2. Actions (3 Discrete Outputs)
At each step, the agent decides one of three directions relative to its current heading:
- `[1, 0, 0]`: Go Straight
- `[0, 1, 0]`: Turn Right
- `[0, 0, 1]`: Turn Left

### 3. Rewards
- **Eat Food**: `+10`
- **Collision / Game Over**: `-10`
- **Other moves**: `0`

### 4. Best Model Checkpoint & Fine-Tuning 💾
- The model automatically saves the best parameters (`model.pth`) and the high score (`best_score.txt`) in the `experiments/` directory whenever a new high record is achieved.
- When running the project again, it loads the saved checkpoint and starts training from the previous highest record. This ensures you can stop training and resume (fine-tune) anytime without losing progress.

---

## How to Run 🚀

### 1. Install Dependencies
Make sure you have PyTorch and Pygame installed:
```bash
uv sync
```
*or using standard pip:*
```bash
pip install pygame torch numpy
```

### 2. Run Training
Simply run the `main.py` script:
```bash
python main.py
```
A Pygame window will pop up showing the snake learning in real-time. Training progress and record scores will be outputted in your terminal.
