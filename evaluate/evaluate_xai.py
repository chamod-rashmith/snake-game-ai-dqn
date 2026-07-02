import os
import sys
import pygame
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.base_model import DQN
from evaluate import SnakeGameEval

class XAIEvaluator:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim=32, action_dim=3)
        
        # Resolve paths relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        resolved_path = None
        if model_path is not None:
            if os.path.isabs(model_path):
                resolved_path = model_path
            else:
                resolved_path = os.path.join(project_root, model_path)
        else:
            paths_to_try = [
                os.path.join(project_root, "experiments", "model.pth"),
                os.path.join(project_root, "model.pth")
            ]
            for p in paths_to_try:
                if os.path.exists(p):
                    resolved_path = p
                    break
        
        if resolved_path and os.path.exists(resolved_path):
            self.model.load_state_dict(torch.load(resolved_path, map_location=self.device))
            print(f"Loaded model from {resolved_path}")
        else:
            print(f"Warning: Model file not found. Using uninitialized model.")
            
        self.model.eval()
        self.game = SnakeGameEval()
        
    def get_state(self):
        head = self.game.head
        dir_l = self.game.direction == 3
        dir_r = self.game.direction == 1
        dir_u = self.game.direction == 0
        dir_d = self.game.direction == 2
        
        state = []
        # Local 5x5 grid around head
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                check_pt = namedtuple('Point', 'x, y')(head.x + dx * 20, head.y + dy * 20)
                state.append(self.game.is_collision(check_pt))
        
        state.extend([dir_l, dir_r, dir_u, dir_d])
        state.extend([
            self.game.food.x < self.game.head.x,
            self.game.food.x > self.game.head.x,
            self.game.food.y < self.game.head.y,
            self.game.food.y > self.game.head.y
        ])
        return np.array(state, dtype=int)

    def run_evaluation_with_xai(self, max_steps=300):
        self.game.reset()
        
        q_history = []  # To store Q-values of actions at each step
        saliency_history = []  # To store gradients of selected action w.r.t input
        state_history = []
        action_names = ["Straight", "Right", "Left"]
        
        steps = 0
        game_over = False
        
        print("Starting evaluation game. Keep Pygame window focused...")
        
        while not game_over and steps < max_steps:
            # 1. Get state representation
            state_arr = self.get_state()
            state_tensor = torch.tensor(state_arr, dtype=torch.float32, requires_grad=True)
            
            # 2. Forward pass to compute Q-values
            q_values = self.model(state_tensor.unsqueeze(0)).squeeze(0)
            q_history.append(q_values.detach().numpy().copy())
            
            # 3. Select action
            action_idx = torch.argmax(q_values).item()
            
            # Make one-hot action vector for Pygame play_step
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # 4. Compute Saliency (Gradients of the chosen Q-value w.r.t. input state)
            self.model.zero_grad()
            q_values[action_idx].backward()
            saliency = state_tensor.grad.detach().numpy().copy()
            saliency_history.append(saliency)
            state_history.append(state_arr)
            
            # 5. Take step in game
            game_over, score = self.game.play_step(action)
            steps += 1
            
        pygame.quit()
        print(f"Evaluation game finished. Steps: {steps}, Final Score: {score}")
        
        # Now convert to numpy arrays for plotting
        q_history = np.array(q_history)
        saliency_history = np.array(saliency_history)
        state_history = np.array(state_history)
        
        # Create output directory
        os.makedirs("evaluate/plots", exist_ok=True)
        
        self.plot_q_values(q_history, steps)
        self.plot_feature_importance_groups(saliency_history)
        self.plot_grid_saliency_heatmap(saliency_history)
        
        print("\nAll plots generated and saved in the 'evaluate/plots/' directory!")

    def plot_q_values(self, q_history, steps):
        plt.figure(figsize=(10, 5))
        plt.plot(q_history[:, 0], label="Straight", color="#2ca02c", linewidth=2)
        plt.plot(q_history[:, 1], label="Right", color="#1f77b4", linewidth=2)
        plt.plot(q_history[:, 2], label="Left", color="#d62728", linewidth=2)
        plt.title("Q-Value Variation of Actions Over Steps (Decisions)", fontsize=14, fontweight='bold')
        plt.xlabel("Step / Frame", fontsize=12)
        plt.ylabel("Q-Value Prediction", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("evaluate/plots/q_values_over_time.png", dpi=150)
        plt.close()

    def plot_feature_importance_groups(self, saliency_history):
        # Calculate mean absolute gradients (saliency) for each feature group
        abs_saliency = np.abs(saliency_history)
        mean_saliency = np.mean(abs_saliency, axis=0)
        
        # Feature mapping:
        # 0-23: Local 5x5 grid (24 features)
        # 24-27: Current Direction (4 features)
        # 28-31: Food Direction (4 features)
        grid_sal = np.mean(mean_saliency[0:24])
        dir_sal = np.mean(mean_saliency[24:28])
        food_sal = np.mean(mean_saliency[28:32])
        
        groups = ["Local 5x5 Vision\n(Obstacle/Body Avoidance)", "Current Direction\n(Self-movement context)", "Food Direction\n(Target orientation)"]
        importances = [grid_sal, dir_sal, food_sal]
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=groups, y=importances, palette="viridis")
        plt.title("Average Feature Group Importance (Saliency)", fontsize=14, fontweight='bold')
        plt.ylabel("Mean Absolute Gradient", fontsize=12)
        plt.grid(axis='y', linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("evaluate/plots/feature_importance_groups.png", dpi=150)
        plt.close()

    def plot_grid_saliency_heatmap(self, saliency_history):
        # Grid index to 5x5 coordinate mapper
        # Saliency history size is (Steps, 32)
        # We need the first 24 features (Local 5x5 grid)
        grid_saliencies = np.abs(saliency_history[:, 0:24])
        mean_grid_saliency = np.mean(grid_saliencies, axis=0)
        
        heatmap_grid = np.zeros((5, 5))
        
        idx = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    # Head position itself is at the center (index 2, 2)
                    heatmap_grid[2, 2] = 0
                    continue
                # Row index (dy+2), Col index (dx+2)
                heatmap_grid[dy + 2, dx + 2] = mean_grid_saliency[idx]
                idx += 1
                
        # Plot Heatmap
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(heatmap_grid, annot=True, cmap="YlOrRd", fmt=".4f", cbar_kws={'label': 'Mean Saliency Magnitude'},
                         xticklabels=[-2, -1, 0, 1, 2], yticklabels=[-2, -1, 0, 1, 2])
        
        plt.title("5x5 Local Saliency Heatmap\n(Which cells around the head affect decisions most?)", fontsize=12, fontweight='bold')
        plt.xlabel("Relative X coordinate (Grid Cells)", fontsize=11)
        plt.ylabel("Relative Y coordinate (Grid Cells)", fontsize=11)
        
        # Annotate head
        ax.text(2.5, 2.5, "HEAD", color="blue", ha="center", va="center", fontweight="bold", 
                bbox=dict(boxstyle="round,pad=0.3", fc="cyan", alpha=0.5))
                
        plt.tight_layout()
        plt.savefig("evaluate/plots/grid_saliency_heatmap.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    evaluator = XAIEvaluator()
    evaluator.run_evaluation_with_xai()
