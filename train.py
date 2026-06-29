import pygame
import random
import json
import os
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
        
        # Default environment settings (overridden by curriculum configuration)
        self.num_random_obstacles = 15
        self.use_complex_barriers = False
        self.food_reward = 10
        self.collision_penalty = -10
        self.timeout_penalty = -15
        self.step_penalty = -0.05
        self.use_bfs_distance = True
        
        self.reset()
        
    def reset(self, config=None):
        if config is not None:
            new_w = config.get('w', self.w)
            new_h = config.get('h', self.h)
            if new_w != self.w or new_h != self.h:
                self.w = new_w
                self.h = new_h
                self.display = pygame.display.set_mode((self.w, self.h))
            
            self.num_random_obstacles = config.get('num_random_obstacles', 15)
            self.use_complex_barriers = config.get('use_complex_barriers', False)
            self.food_reward = config.get('food_reward', 10)
            self.collision_penalty = config.get('collision_penalty', -10)
            self.timeout_penalty = config.get('timeout_penalty', -15)
            self.step_penalty = config.get('step_penalty', -0.05)
            self.use_bfs_distance = config.get('use_bfs_distance', True)

        # Init game state
        self.direction = 1 # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.head = Point(self.w/2, self.h/2)
        
        self.obstacles = []
        if self.use_complex_barriers:
            self._generate_complex_obstacles()
        else:
            # Generate random obstacles
            for _ in range(self.num_random_obstacles):
                while True:
                    x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                    y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                    pt = Point(x, y)
                    in_safe_zone = (self.w/2 - 100 < x < self.w/2 + 100) and (self.h/2 - 60 < y < self.h/2 + 60)
                    if pt not in self.obstacles and not in_safe_zone:
                        self.obstacles.append(pt)
                        break

        # Relocate head if it spawns on an obstacle
        while self.head in self.obstacles:
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
        
    def _generate_complex_obstacles(self):
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

        # 5. Scattered single pillars
        pillars = [
            Point(400, 120), Point(400, 140),
            Point(400, 440), Point(400, 460),
            Point(140, 300), Point(660, 300)
        ]
        self.obstacles.extend(pillars)
        
        # Add random obstacles on top of complex walls
        for _ in range(self.num_random_obstacles):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                pt = Point(x, y)
                in_safe_zone = (self.w/2 - 100 < x < self.w/2 + 100) and (self.h/2 - 60 < y < self.h/2 + 60)
                if pt not in self.obstacles and not in_safe_zone:
                    self.obstacles.append(pt)
                    break

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()
            
    def _bfs_distance(self, start, target):
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            curr, dist = queue.popleft()
            if curr == target:
                return dist
            for dx, dy in [(-BLOCK_SIZE, 0), (BLOCK_SIZE, 0), (0, -BLOCK_SIZE), (0, BLOCK_SIZE)]:
                nxt = Point(curr.x + dx, curr.y + dy)
                if 0 <= nxt.x < self.w and 0 <= nxt.y < self.h:
                    if nxt not in self.obstacles and nxt not in self.snake[:-1]:
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, dist + 1))
        # Fallback to Manhattan distance if no path is found
        return abs(start.x - target.x) // BLOCK_SIZE + abs(start.y - target.y) // BLOCK_SIZE

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        # Calculate old distance to food
        if self.use_bfs_distance:
            old_dist = self._bfs_distance(self.head, self.food)
        else:
            old_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
        
        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = self.collision_penalty
            return reward, game_over, self.score
        elif self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = self.timeout_penalty
            return reward, game_over, self.score
            
        # 4. Place new food or move tail
        if self.head == self.food:
            self.score += 1
            reward = self.food_reward
            self._place_food()
        else:
            self.snake.pop()
            
            # Reward shaping
            if self.use_bfs_distance:
                new_dist = self._bfs_distance(self.head, self.food)
            else:
                new_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
                
            if new_dist < old_dist:
                reward = 1 + self.step_penalty
            else:
                reward = -1 + self.step_penalty
            
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
        
        # Draw obstacles
        for pt in self.obstacles:
            pygame.draw.rect(self.display, (70, 130, 180), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
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
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        clock_wise = [0, 1, 2, 3] # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
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
        self.epsilon = 80
        self.gamma = 0.9
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
        
        # Local 5x5 grid
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                check_pt = Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)
                state.append(game.is_collision(check_pt))
        
        state.extend([dir_l, dir_r, dir_u, dir_d])
        
        state.extend([
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ])
        
        return np.array(state, dtype=int)
        
    def get_action(self, state):
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
        
        pred = self.model(states)
        
        target = pred.clone().detach()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx].unsqueeze(0))).detach()
                
            action_idx = torch.argmax(actions[idx]).item()
            target[idx][action_idx] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def save(self, record_score, file_name='model.pth'):
        model_folder_path = './experiments'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name_path = os.path.join(model_folder_path, file_name)
        torch.save(self.model.state_dict(), file_name_path)
        
        score_path = os.path.join(model_folder_path, 'best_score.txt')
        with open(score_path, 'w') as f:
            f.write(str(record_score))

    def load(self, file_name='model.pth'):
        file_path = os.path.join('./experiments', file_name)
        record = 0
        if os.path.exists(file_path):
            try:
                self.model.load_state_dict(torch.load(file_path))
                self.model.eval()
                print(f"Successfully loaded saved model checkpoint from: {file_path}")
                
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

def load_curriculum_config():
    config_path = './experiments/curriculum_config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading curriculum config: {e}")
    return None

def get_current_phase_config(n_games, curriculum_config):
    if not curriculum_config or 'phases' not in curriculum_config:
        return None
    
    current_config = None
    for phase in sorted(curriculum_config['phases'], key=lambda x: x['start_game']):
        if n_games >= phase['start_game']:
            current_config = phase
    return current_config

def train():
    agent = Agent()
    record = agent.load()
    
    curriculum_config = load_curriculum_config()
    current_phase = get_current_phase_config(agent.n_games, curriculum_config)
    
    game = SnakeGameAI()
    if current_phase:
        game.reset(config=current_phase)
        print(f"Starting training on Phase {current_phase['phase_index']} (Game {agent.n_games})")
    
    max_games = 300
    while agent.n_games < max_games:
        # Check if phase needs to transition at start of game
        phase_config = get_current_phase_config(agent.n_games, curriculum_config)
        
        # Get old state
        state_old = agent.get_state(game)
        
        # Get move
        final_move = agent.get_action(state_old)
        
        # Perform move
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Store in memory
        agent.memory.push(state_old, final_move, reward, state_new, done)
        
        # Train short memory
        agent.train_step([state_old], [final_move], [reward], [state_new], [done])
        
        if done:
            agent.n_games += 1
            
            # Train long memory
            if len(agent.memory) > 1000:
                mini_sample = agent.memory.sample(1000)
            else:
                mini_sample = agent.memory.sample(len(agent.memory))
                
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            agent.train_step(states, actions, rewards, next_states, dones)
            
            if score > record:
                record = score
                agent.save(record)
                print(f"New Record! Model saved.")
                
            next_phase_config = get_current_phase_config(agent.n_games, curriculum_config)
            if next_phase_config and (not current_phase or next_phase_config['phase_index'] != current_phase['phase_index']):
                current_phase = next_phase_config
                print(f"\n>>> Transitioning to Phase {current_phase['phase_index']} (Game {agent.n_games}) <<<")
                
            # Reset game with configuration for the next step/game
            game.reset(config=current_phase)
            
            print(f'Game: {agent.n_games} | Score: {score} | Record: {record}')

if __name__ == '__main__':
    train()
