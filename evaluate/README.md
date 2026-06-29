# Model Evaluation in New Environment 📊🧪

This directory contains the script and results for evaluating the best-performing trained Deep Q-Network (DQN) model in a brand-new, challenging environment.

## The Evaluation Environment (`SnakeGameEval`) 🗺️

To test the model's ability to adapt (generalize) to unseen environments, the evaluation environment introduces several modifications from the training setup:
1. **Compact Board Size**: Resized the board from $640 \times 480$ pixels to **$400 \times 400$ pixels**, restricting the movement space.
2. **Static Obstacles (Blue Blocks)**: Added **9 static obstacle blocks (walls)** throughout the map.
3. **Pure Exploitation**: Set the exploration rate ($\epsilon$) to $0$. The agent makes choices strictly based on its learned policy, with zero random moves.

---

## Evaluation Results 📈

The model was tested over **10 consecutive games** with the following scores:

| Game Episode | Final Score |
|---|---|
| Game 1/10 | **37** 🏆 |
| Game 2/10 | 19 |
| Game 3/10 | 5 |
| Game 4/10 | 19 |
| Game 5/10 | 27 |
| Game 6/10 | 14 |
| Game 7/10 | 8 |
| Game 8/10 | 14 |
| Game 9/10 | 12 |
| Game 10/10 | 23 |

### Summary Statistics
- **Average Score**: `17.80`
- **Max Score**: `37`
- **Min Score**: `5`

---

## Why the Model Performs So Well 🧠

1. **Successful Spatial Generalization**: During training, the agent **never saw blue obstacle blocks**. However, because its state representation checks a local $5 \times 5$ collision grid around the head, it treats obstacles, walls, and its own body identically as collisions. Therefore, it generalizes seamlessly to avoid obstacles.
2. **Determinism**: Since exploration is disabled ($\epsilon=0$), the agent takes optimal pathways to food without making risky, random exploratory maneuvers.

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
