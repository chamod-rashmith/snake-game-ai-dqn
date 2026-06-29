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
SPEED = 15  # Slower speed so it's easy for the user to watch the evaluation!

Point = namedtuple('Point', 'x, y')

class SnakeGameEval:
    def __init__(self, w=400, h=400):
        # Using a new board size: 400x400 instead of 640x480
        self.w = w
        self.h = h
        # Init display
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game AI Evaluation (New Env)')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 20)
        
        # Define static obstacles/walls in the environment
        # These will be placed in the map, and because the state checks 5x5 local vision
        # for collisions, the agent should be able to avoid them!
        self.obstacles = [
            Point(100, 100), Point(100, 120), Point(100, 140),
            Point(280, 240), Point(280, 260), Point(280, 280),
            Point(200, 100), Point(200, 120), Point(200, 140)
        ]
        
        self.reset()
        
    def reset(self):
        # Init game state
        self.direction = 1 # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        # Food should not spawn on the snake or on static obstacles
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()
            
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            return game_over, self.score
            
        # 4. Place new food or move tail
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
            
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Return game over and score
        return game_over, self.score
        
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        # Hits static obstacles
        if pt in self.obstacles:
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw obstacles
        for pt in self.obstacles:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12), 1)
            
        # Draw snake
        for i, pt in enumerate(self.snake):
            color = GREEN_LIGHT if i == 0 else GREEN_DARK
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12), 1)
            
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Score text
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()
        
    def _move(self, action):
        clock_wise = [0, 1, 2, 3] # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left turn
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == 0: # UP
            y -= BLOCK_SIZE
        elif self.direction == 1: # RIGHT
            x += BLOCK_SIZE
        elif self.direction == 2: # DOWN
            y += BLOCK_SIZE
        elif self.direction == 3: # LEFT
            x -= BLOCK_SIZE
            
        self.head = Point(x, y)

class Evaluator:
    def __init__(self, model_path='experiments/model.pth'):
        self.model = DQN(state_dim=32, action_dim=3)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"Loaded trained model weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please train a model first.")

    def get_state(self, game):
        head = game.head
        
        dir_l = game.direction == 3
        dir_r = game.direction == 1
        dir_u = game.direction == 0
        dir_d = game.direction == 2
        
        state = []
        
        # Local 5x5 grid (24 binary inputs)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue  # skip head
                check_pt = Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)
                state.append(game.is_collision(check_pt))
        
        # Move direction
        state.extend([dir_l, dir_r, dir_u, dir_d])
        
        # Food location
        state.extend([
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down
        ])
        
        return np.array(state, dtype=int)

    def get_action(self, state):
        # Epsilon = 0 (Pure exploitation, no exploration during evaluation)
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0.unsqueeze(0))
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

def main():
    evaluator = Evaluator()
    game = SnakeGameEval()
    
    num_episodes = 10
    scores = []
    
    print("\nStarting Evaluation...")
    print("-" * 30)
    for episode in range(1, num_episodes + 1):
        game.reset()
        done = False
        while not done:
            state = evaluator.get_state(game)
            action = evaluator.get_action(state)
            done, score = game.play_step(action)
        
        scores.append(score)
        print(f"Game {episode}/{num_episodes} | Final Score: {score}")
        
    print("-" * 30)
    print("Evaluation Complete!")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")
    
    pygame.quit()

if __name__ == '__main__':
    main()
