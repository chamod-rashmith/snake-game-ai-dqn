import os
import sys
import pygame
import random
import numpy as np
import torch
from collections import namedtuple

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.base_model import DQN

# Colors
WHITE = (255, 255, 255)
RED = (220, 20, 60)
GREEN_DARK = (34, 139, 34)
GREEN_LIGHT = (50, 205, 50)
BLACK = (20, 20, 20)
GRAY = (80, 80, 80)
BLUE = (70, 130, 180)

BLOCK_SIZE = 20
SPEED = 10  # Moderate speed so the user can observe the choices!

Point = namedtuple('Point', 'x, y')

class SnakeGameTwoFoods:
    def __init__(self, w=400, h=400):
        self.w = w
        self.h = h
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game AI - Two Foods Evaluation')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 20)
        
        # Obstacles
        self.obstacles = [
            Point(100, 100), Point(100, 120), Point(100, 140),
            Point(280, 240), Point(280, 260), Point(280, 280),
            Point(200, 100), Point(200, 120), Point(200, 140)
        ]
        
        self.reset()
        
    def reset(self):
        self.direction = 1  # RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.foods = []
        self._place_food()  # Place first food
        self._place_food()  # Place second food
        self.frame_iteration = 0
        
    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            new_food = Point(x, y)
            if new_food not in self.snake and new_food not in self.obstacles and new_food not in self.foods:
                self.foods.append(new_food)
                break
            
    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        self._move(action)
        self.snake.insert(0, self.head)
        
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            return game_over, self.score
            
        # Check if head is on any of the foods
        food_eaten = False
        for food in self.foods:
            if self.head == food:
                self.score += 1
                self.foods.remove(food)
                self._place_food()  # Spawn a new food to replace the eaten one
                food_eaten = True
                break
                
        if not food_eaten:
            self.snake.pop()
            
        self._update_ui()
        self.clock.tick(SPEED)
        
        return game_over, self.score
        
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        if pt in self.obstacles:
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.obstacles:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12), 1)
            
        for i, pt in enumerate(self.snake):
            color = GREEN_LIGHT if i == 0 else GREEN_DARK
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12), 1)
            
        # Draw both foods
        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
            
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()
        
    def _move(self, action):
        clock_wise = [0, 1, 2, 3] # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == 0:
            y -= BLOCK_SIZE
        elif self.direction == 1:
            x += BLOCK_SIZE
        elif self.direction == 2:
            y += BLOCK_SIZE
        elif self.direction == 3:
            x -= BLOCK_SIZE
        self.head = Point(x, y)


class TwoFoodsEvaluator:
    def __init__(self, model_path='experiments/model.pth'):
        self.model = DQN(state_dim=32, action_dim=3)
        
        # Resolve path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resolved_path = os.path.join(project_root, model_path)
        if not os.path.exists(resolved_path):
            resolved_path = os.path.join(project_root, "model.pth")
            
        if os.path.exists(resolved_path):
            self.model.load_state_dict(torch.load(resolved_path))
            self.model.eval()
            print(f"Loaded trained model weights from {resolved_path}")
        else:
            raise FileNotFoundError(f"Model file not found. Please train a model first.")
            
    def get_state(self, game):
        head = game.head
        dir_l = game.direction == 3
        dir_r = game.direction == 1
        dir_u = game.direction == 0
        dir_d = game.direction == 2
        
        state = []
        
        # 1. Local 5x5 grid (24 binary inputs)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                check_pt = Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)
                state.append(game.is_collision(check_pt))
                
        # 2. Movement direction (4 features)
        state.extend([dir_l, dir_r, dir_u, dir_d])
        
        # 3. Target the closest food!
        # Find which food is closer using Manhattan distance
        closest_food = game.foods[0]
        min_dist = float('inf')
        for food in game.foods:
            dist = abs(head.x - food.x) + abs(head.y - food.y)
            if dist < min_dist:
                min_dist = dist
                closest_food = food
                
        # Food location relative to head for the closest food
        state.extend([
            closest_food.x < head.x, # food left
            closest_food.x > head.x, # food right
            closest_food.y < head.y, # food up
            closest_food.y > head.y  # food down
        ])
        
        return np.array(state, dtype=int)
        
    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0.unsqueeze(0))
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

def main():
    evaluator = TwoFoodsEvaluator()
    game = SnakeGameTwoFoods()
    
    num_episodes = 5
    scores = []
    
    print("\nStarting Two Foods Evaluation...")
    print("-" * 40)
    for episode in range(1, num_episodes + 1):
        game.reset()
        done = False
        while not done:
            state = evaluator.get_state(game)
            action = evaluator.get_action(state)
            done, score = game.play_step(action)
            
        scores.append(score)
        print(f"Game {episode}/{num_episodes} | Final Score: {score}")
        
    print("-" * 40)
    print("Two Foods Evaluation Complete!")
    print(f"Average Score (with 2 foods): {np.mean(scores):.2f}")
    
    pygame.quit()

if __name__ == '__main__':
    main()
