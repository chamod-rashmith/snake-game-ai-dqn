# Model Evaluation in New Environment 📊🧪

This directory contains the script and results for evaluating the best-performing trained Deep Q-Network (DQN) model in a brand-new, challenging environment.

## The Evaluation Environment (`SnakeGameEval`) 🗺️

To test the model's ability to adapt (generalize) to unseen environments, the evaluation environment introduces several modifications from the training setup:
1. **Compact Board Size**: Resized the board from $640 \times 480$ pixels to **$400 \times 400$ pixels**, restricting the movement space.
2. **Static Obstacles (Blue Blocks)**: Added **9 static obstacle blocks (walls)** throughout the map.
3. **Pure Exploitation**: Set the exploration rate ($\epsilon$) to $0$. The agent makes choices strictly based on its learned policy, with zero random moves.

---

## Evaluation Results 📈

### 1. Standard Evaluation (`evaluate.py`)
Tested over **10 consecutive games**:

| Game Episode | Final Score |
|---|---|
| Game 1/10 | 32 |
| Game 2/10 | **38** 🏆 |
| Game 3/10 | 18 |
| Game 4/10 | 29 |
| Game 5/10 | 28 |
| Game 6/10 | 19 |
| Game 7/10 | 27 |
| Game 8/10 | 17 |
| Game 9/10 | 31 |
| Game 10/10 | 16 |

#### Summary Statistics
- **Average Score**: `25.50` (Previously `17.80`)
- **Max Score**: `38`
- **Min Score**: `16`

### 2. Hard/Large Environment Evaluation (`evaluate_hard.py`)
Tested over **10 consecutive games** on a $800 \times 600$ board with high-density barriers:

| Game Episode | Final Score |
|---|---|
| Game 1/10 | 1 |
| Game 2/10 | 0 |
| Game 3/10 | **28** 🏆 |
| Game 4/10 | 16 |
| Game 5/10 | 17 |
| Game 6/10 | 0 |
| Game 7/10 | 24 |
| Game 8/10 | 11 |
| Game 9/10 | 8 |
| Game 10/10 | 10 |

#### Summary Statistics
- **Average Score**: `11.50`
- **Max Score**: `28`
- **Min Score**: `0`

---

## Looping Behavior & Obstacle Training 🔄🧱

### Why does it still sometimes loop/stuck?
After introducing step penalties, loop penalties, and random obstacles during training, the snake's average performance dramatically improved. However, the snake can still get stuck in a loop occasionally.
1. **Short Training Duration**: We only trained for one short run with these new barriers. Deep Reinforcement Learning requires many games (typically 500+ episodes) to fully adapt to complex spatial features.
2. **Deterministic Evaluation**: During evaluation, $\epsilon = 0$. If the network finds itself in a state loop where it predicts a circular sequence of actions as the "highest Q-value", it will loop indefinitely until the step/timeout limit is reached.
3. **Complex Barriers**: In the hard environment (`evaluate_hard.py`), dense barriers block many paths. Without enough training, the DQN cannot map the path around walls to the food efficiently, getting trapped in local loops.

### Future Mitigation
To completely eliminate looping, train the model for 1000+ episodes using the updated `train.py` which exposes the network to hundreds of different random obstacle layouts and enforces the step penalty.

---

## 2. Hard / Large Evaluation Environment (`SnakeGameEvalHard`) 🏔️🕷️

We also built a second, significantly more difficult evaluation environment to stretch the boundaries of the model's spatial adaptability:

- **Script**: [evaluate_hard.py](file:///c:/Users/Chamod_Rashmith_UOK/Desktop/programming/Deep%20Learning/project_1/evaluate/evaluate_hard.py)
- **Large Board Size**: Expanded the environment to **$800 \times 600$ pixels** (much larger than the default training grid).
- **High-Density Barriers**:
  - Border inner ridges at all 4 corners.
  - A long vertical partition on the left ($x=200$).
  - A long vertical partition on the right ($x=600$).
  - A horizontal middle wall ($y=300$).
  - Several isolated pillar blocks acting as obstacles.

### How to Run:
Run either script from the project root:
```bash
# Standard evaluation
.venv/Scripts/python.exe evaluate/evaluate.py

# Hard evaluation (Large board + dense obstacles)
.venv/Scripts/python.exe evaluate/evaluate_hard.py
```
