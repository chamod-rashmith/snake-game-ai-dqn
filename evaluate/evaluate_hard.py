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
SPEED = 20  # Slower speed for visualization

Point = namedtuple('Point', 'x, y')

class SnakeGameEvalHard:
    def __init__(self, w=800, h=600):
        # Larger board size: 800x600
        self.w = w
        self.h = h
        # Init display
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game AI Evaluation (Hard/Large Env)')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 20)
        
        # Define significantly more static obstacles (border ridges, inner blocks, corridors)
        self.obstacles = []
        self._generate_obstacles()
        
        self.reset()
        
    def _generate_obstacles(self):
        # 1. Vertical inner partition wall on left side
        for y in range(120, 260, BLOCK_SIZE):
            self.obstacles.append(Point(200, y))
            
        # 2. Vertical inner partition wall on right side
        for y in range(340, 480, BLOCK_SIZE):
            self.obstacles.append(Point(600, y))
            
        # 3. Horizontal middle wall
        for x in range(300, 520, BLOCK_SIZE):
            self.obstacles.append(Point(x, 300))
            
        # 4. Corner barrier blocks (inner ridges)
        # Top Left corner ridges
        self.obstacles.extend([Point(100, 100), Point(120, 100), Point(100, 120)])
        # Top Right corner ridges
        self.obstacles.extend([Point(680, 100), Point(660, 100), Point(680, 120)])
        # Bottom Left corner ridges
        self.obstacles.extend([Point(100, 480), Point(120, 480), Point(100, 460)])
        # Bottom Right corner ridges
        self.obstacles.extend([Point(680, 480), Point(660, 480), Point(680, 460)])

        # 5. Scattered single pillars (obstacles)
        pillars = [
            Point(400, 120), Point(400, 140),
            Point(400, 440), Point(400, 460),
            Point(140, 300), Point(660, 300)
        ]
        self.obstacles.extend(pillars)
        
    def reset(self):
        # Init game state
        self.direction = 1 # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.head = Point(self.w/2, self.h/2)
        # Verify head is not inside an obstacle on spawn
        while self.head in self.obstacles:
            # Shift head slightly if it spawns on an obstacle
            self.head = Point(self.head.x + BLOCK_SIZE, self.head.y)
            
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
        # Epsilon = 0 (Pure exploitation)
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0.unsqueeze(0))
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

def main():
    evaluator = Evaluator()
    game = SnakeGameEvalHard()
    
    num_episodes = 10
    scores = []
    
    print("\nStarting Hard/Large Environment Evaluation...")
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
    print("Hard Evaluation Complete!")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")
    
    pygame.quit()

if __name__ == '__main__':
    main()
