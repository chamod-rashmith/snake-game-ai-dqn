import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from src.models.base_model import DQN

# Colors
WHITE = (255, 255, 255)
RED = (220, 20, 60)
GREEN_DARK = (34, 139, 34)
GREEN_LIGHT = (50, 205, 50)
BLACK = (20, 20, 20)
GRAY = (100, 100, 100)

BLOCK_SIZE = 20
SPEED = 80  # Speed of the game during training

Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Init display
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game AI - DQN')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
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
        
        # Generate 15 random obstacles
        self.obstacles = []
        for _ in range(15):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                pt = Point(x, y)
                # Keep a safe zone around the initial snake starting area
                in_safe_zone = (220 < x < 420) and (180 < y < 300)
                if pt not in self.obstacles and not in_safe_zone:
                    self.obstacles.append(pt)
                    break

        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()
            
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input (so window doesn't freeze/crash)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        # Calculate old distance to food
        old_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
        
        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
        elif self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -15  # Loop/timeout penalty
            return reward, game_over, self.score
            
        # 4. Place new food or move tail
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            # Reward shaping: reward for getting closer to food, penalty for getting further, with step penalty
            new_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
            if new_dist < old_dist:
                reward = 1 - 0.05  # Step penalty of 0.05
            else:
                reward = -1 - 0.05 # Step penalty of 0.05
            
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Return reward and game over
        return reward, game_over, self.score
        
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
        
        # Draw obstacles (blue, same as evaluation)
        for pt in self.obstacles:
            pygame.draw.rect(self.display, (70, 130, 180), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12), 1)

        # Draw snake
        for i, pt in enumerate(self.snake):
            # Head is lighter green
            color = GREEN_LIGHT if i == 0 else GREEN_DARK
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12), 1)
            
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Score text
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # Action is [straight, right, left]
        # Clockwise order: UP, RIGHT, DOWN, LEFT
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


# Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)


# Agent
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80 # randomness parameter (will decay)
        self.gamma = 0.9  # discount rate
        self.memory = ReplayMemory(100_000)
        self.model = DQN(state_dim=32, action_dim=3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
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
        # Epsilon-greedy (slower decay, keeping a min epsilon of 5% / value 10 for continuous exploration)
        self.epsilon = max(10, 80 - (self.n_games / 4))
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0.unsqueeze(0))
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
        
    def train_step(self, states, actions, rewards, next_states, dones):
        self.model.train()
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)
        
        # 1: Predicted Q values with current state
        pred = self.model(states)
        
        target = pred.clone().detach()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx].unsqueeze(0))).detach()
                
            # action index
            action_idx = torch.argmax(actions[idx]).item()
            target[idx][action_idx] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


    def save(self, record_score, file_name='model.pth'):
        import os
        model_folder_path = './experiments'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        # Save model weights
        file_name_path = os.path.join(model_folder_path, file_name)
        torch.save(self.model.state_dict(), file_name_path)
        
        # Save best score
        score_path = os.path.join(model_folder_path, 'best_score.txt')
        with open(score_path, 'w') as f:
            f.write(str(record_score))

    def load(self, file_name='model.pth'):
        import os
        file_path = os.path.join('./experiments', file_name)
        record = 0
        if os.path.exists(file_path):
            try:
                self.model.load_state_dict(torch.load(file_path))
                self.model.eval()
                print(f"Successfully loaded saved model checkpoint from: {file_path}")
                
                # Load best score
                score_path = os.path.join('./experiments', 'best_score.txt')
                if os.path.exists(score_path):
                    try:
                        with open(score_path, 'r') as f:
                            record = int(f.read().strip())
                        print(f"Successfully loaded best score record: {record}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Could not load checkpoint ({e}). Starting training with a fresh model.")
        return record


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    record = agent.load()  # Load checkpoint & score if exists
    game = SnakeGameAI()
    
    # Run for 300 games
    max_games = 300
    while agent.n_games < max_games:
        # Get old state
        state_old = agent.get_state(game)
        
        # Get move
        final_move = agent.get_action(state_old)
        
        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Store in memory
        agent.memory.push(state_old, final_move, reward, state_new, done)
        
        # Train short memory (on every step)
        agent.train_step([state_old], [final_move], [reward], [state_new], [done])
        
        if done:
            # Train long memory (replay buffer experience)
            game.reset()
            agent.n_games += 1
            
            if len(agent.memory) > 1000:
                mini_sample = agent.memory.sample(1000)
            else:
                mini_sample = agent.memory.sample(len(agent.memory))
                
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            agent.train_step(states, actions, rewards, next_states, dones)
            
            if score > record:
                record = score
                agent.save(record)  # Save the best model and record
                print(f"New Record! Model saved.")
                
            print(f'Game: {agent.n_games} | Score: {score} | Record: {record}')

if __name__ == '__main__':
    train()
