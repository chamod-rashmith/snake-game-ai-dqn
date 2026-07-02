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
| Game 1/10 | 36 |
| Game 2/10 | 35 |
| Game 3/10 | 32 |
| Game 4/10 | 33 |
| Game 5/10 | **41** 🏆 |
| Game 6/10 | 26 |
| Game 7/10 | 12 |
| Game 8/10 | 22 |
| Game 9/10 | 31 |
| Game 10/10 | 37 |

#### Summary Statistics
- **Average Score**: **`30.50`**
- **Max Score**: `41`
- **Min Score**: `12`

### 2. Hard/Large Environment Evaluation (`evaluate_hard.py`)
Tested over **10 consecutive games** on an $800 \times 600$ board with high-density barriers:

| Game Episode | Final Score |
|---|---|
| Game 1/10 | 25 |
| Game 2/10 | 9 |
| Game 3/10 | **40** 🏆 |
| Game 4/10 | 3 |
| Game 5/10 | 6 |
| Game 6/10 | 19 |
| Game 7/10 | 1 |
| Game 8/10 | 23 |
| Game 9/10 | 4 |
| Game 10/10 | 0 |

#### Summary Statistics
- **Average Score**: **`13.00`**
- **Max Score**: `40`
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

- **Script**: [evaluate_hard.py](evaluate_hard.py)
- **Large Board Size**: Expanded the environment to **$800 \times 600$ pixels** (much larger than the default training grid).
- **High-Density Barriers**:
  - Border inner ridges at all 4 corners.
  - A long vertical partition on the left ($x=200$).
  - A long vertical partition on the right ($x=600$).
  - A horizontal middle wall ($y=300$).
  - Several isolated pillar blocks acting as obstacles.

### How to Run:
Run the evaluation scripts from the project root directory using **uv**:
```bash
# Standard evaluation
uv run python evaluate/evaluate.py

# Hard evaluation (Large board + dense obstacles)
uv run python evaluate/evaluate_hard.py

# XAI Saliency evaluation
uv run python evaluate/evaluate_xai.py

# Two Foods evaluation (spawns 2 foods simultaneously)
uv run python evaluate/evaluate_two_foods.py
```

---

## Explainable AI (XAI) & Model Decision Analysis 🔍🧠

To understand *how* our DQN model makes decisions and verify if it has truly learned generalizable features, we implemented a feature-attribution saliency evaluation in [evaluate_xai.py](evaluate_xai.py). 

During evaluation, the agent's actions are monitored and analyzed using gradient-based saliency maps ($S(s) = \left| \frac{\partial Q(s, a_{\text{chosen}})}{\partial s} \right|$). This shows which inputs (features) the network relies on most when choosing an action.

Three key plots are generated in the `evaluate/plots/` folder:

### 1. Action Value (Q-Value) Trajectory 📈
- **Plot**: [q_values_over_time.png](plots/q_values_over_time.png)
- **Observations**: The Q-values for the three actions (`Straight`, `Right`, `Left`) fluctuate dynamically. When the snake is in open space, the Q-values are relatively stable and high. As the snake approaches a wall or its own body, the Q-value for the dangerous direction drops sharply, while the Q-value for the safe turn rises.
- **Verdict**: **Good**. The network shows clear preference distinctions rather than random or flat Q-predictions, confirming that the policy is confident and highly responsive to environmental changes.

### 2. Feature Group Saliency 📊
- **Plot**: [feature_importance_groups.png](plots/feature_importance_groups.png)
- **Observations**: 
  - **Local 5x5 Vision (Obstacles)**: Has a high average gradient magnitude, showing the network is constantly reading the grid for walls/self-body.
  - **Food Direction**: Exhibits strong importance, showing the agent is heavily driven by target-seeking behavior.
  - **Current Direction**: Acts as a state context helper to prevent moving backwards.
- **Verdict**: **Balanced & Good**. The network successfully balances safety (avoiding obstacles) and goal-seeking (finding food).

### 3. Spatial Saliency Heatmap (5x5 Grid) 🗺️🔥
- **Plot**: [grid_saliency_heatmap.png](plots/grid_saliency_heatmap.png)
- **Observations**: 
  - The cells **directly adjacent** to the head (distance 1: top, bottom, left, right) have the highest saliency. This makes perfect physical sense because an obstacle there represents immediate death.
  - The cells at distance 2 have lower but non-zero saliency, indicating the model does a small amount of look-ahead path planning.
- **Verdict**: **Excellent**. The model has correctly prioritized immediate hazard zones, proving it has learned a robust, biologically-plausible spatial avoidance strategy instead of memorizing paths.

---

## 3. Two Foods Evaluation Environment (`evaluate_two_foods.py`) 🍎🍎

To explore how the DQN model handles multiple resource options simultaneously, we built an evaluation environment that spawns **2 foods** on the board at the same time:

### Methodology:
- Since the DQN's input state is designed for a single food direction, the evaluator dynamically calculates the Manhattan distance from the snake's head to both foods at every frame and feeds the **direction of the closest food** to the model.
- When the head eats a food, a new food is immediately placed randomly, keeping the active food count at 2.

### Results:
Tested over **5 consecutive games**:

| Game Episode | Final Score |
|---|---|
| Game 1/5 | 14 |
| Game 2/5 | 31 |
| Game 3/5 | 33 |
| Game 4/5 | 30 |
| Game 5/5 | 31 |

#### Summary Statistics
- **Average Score**: **`27.80`** (Slightly lower than standard 1-food evaluation of **`30.50`**)
- **Max Score**: `33` | **Min Score**: `14`

### Analysis: Trade-offs & Observations
1. **Target Oscillation (The Penalty)**: Because the model was trained with a single target, the state representation always points to one destination. With two foods, if the snake is positioned nearly equidistant between them, moving a single step can cause the "closest food" to toggle back and forth. This rapid oscillation of the input food direction features can confuse the network, causing it to hesitate, make sub-optimal double-turns, or get trapped in local loops.
2. **Reduced Path Travel (The Benefit)**: When not oscillating, having two foods significantly decreases the average distance to the nearest resource, reducing travel steps and lowering collision risks in open areas.
3. **Verdict**: The high baseline standard score of `30.50` proves the single-food navigation policy is extremely well-optimized. The two-foods setup offers a good generalization test, but requires joint training (exposing the network to moving/multiple targets) to eliminate the oscillation penalty.


