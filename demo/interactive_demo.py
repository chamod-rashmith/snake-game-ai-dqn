import os
import sys
import pygame
import random
import numpy as np
import torch
import time
import math
from collections import namedtuple, deque

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.base_model import DQN

# Neon Cyberpunk Color Palette
COLOR_BG_GAME = (10, 13, 26)         # Deepest Space Obsidian
COLOR_BG_DASH = (16, 20, 39)        # Tech-panel Navy
COLOR_BG_CARD = (24, 30, 59)        # Sleek slate glass panel
COLOR_GRID_DOT = (30, 38, 70)       # High-tech grid dot color
COLOR_TEXT_PRIMARY = (248, 250, 252) # Pure Off-white
COLOR_TEXT_MUTED = (120, 130, 160)   # Modern slate-blue label text

# Snake & Food Glow Themes
COLOR_SNAKE_HEAD = (57, 255, 20)    # Laser Neon Green
COLOR_SNAKE_BODY_START = (57, 255, 20)
COLOR_SNAKE_BODY_END = (6, 182, 212) # Transition to Neon Cyan
COLOR_FOOD_CORE = (255, 46, 108)    # Pulsing Pink/Red
COLOR_FOOD_GLOW = (255, 46, 108, 35)
COLOR_OBSTACLE = (15, 23, 42)       # Sleek obsidian obstacles
COLOR_OBSTACLE_BORDER = (0, 180, 216) # Cyan laser outline

# UI Highlights
COLOR_CYAN = (6, 182, 212)          # Neon Cyan
COLOR_ORANGE = (249, 115, 22)       # Amber Warning Orange
COLOR_GREEN = (16, 185, 129)        # Stable Active Green
COLOR_BORDER = (38, 48, 88)         # Subtle border line
COLOR_RED = (239, 68, 68)           # Alarm Red for danger zones

BLOCK_SIZE = 20
Point = namedtuple('Point', 'x, y')

class InteractiveSnakeGame:
    def __init__(self, w=400, h=400):
        self.w = w
        self.h = h
        
        pygame.init()
        self.display_width = 850
        self.display_height = 550
        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Snake Game AI - Premium Diagnostics')
        self.clock = pygame.time.Clock()
        
        # Load high-tech monospace fonts
        try:
            self.font_title = pygame.font.Font(pygame.font.match_font('consolas', bold=True), 20)
            self.font_section = pygame.font.Font(pygame.font.match_font('consolas', bold=True), 14)
            self.font_body = pygame.font.Font(pygame.font.match_font('consolas'), 13)
            self.font_body_bold = pygame.font.Font(pygame.font.match_font('consolas', bold=True), 13)
            self.font_small = pygame.font.Font(pygame.font.match_font('consolas'), 11)
        except Exception:
            self.font_title = pygame.font.SysFont('arial', 20, bold=True)
            self.font_section = pygame.font.SysFont('arial', 14, bold=True)
            self.font_body = pygame.font.SysFont('arial', 13)
            self.font_body_bold = pygame.font.SysFont('arial', 13, bold=True)
            self.font_small = pygame.font.SysFont('arial', 11)
            
        self.obstacles = [
            Point(100, 100), Point(100, 120), Point(100, 140),
            Point(280, 240), Point(280, 260), Point(280, 280),
            Point(200, 100), Point(200, 120), Point(200, 140)
        ]
        self.reset()
        
    def reset(self):
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
        self.flash_timer = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()
            
    def play_step(self, action):
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)
        
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            return game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            self.flash_timer = 4  
            self._place_food()
        else:
            self.snake.pop()
            
        if self.flash_timer > 0:
            self.flash_timer -= 1
            
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
        if self.direction == 0:
            y -= BLOCK_SIZE
        elif self.direction == 1:
            x += BLOCK_SIZE
        elif self.direction == 2:
            y += BLOCK_SIZE
        elif self.direction == 3:
            x -= BLOCK_SIZE
            
        self.head = Point(x, y)


class DemoManager:
    def __init__(self, model_path='experiments/model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim=32, action_dim=3)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model_loaded = True
        else:
            self.model_loaded = False
            
        self.game = InteractiveSnakeGame()
        self.ai_mode = True 
        self.speed = 10
        self.manual_direction = 1
        
        # Stats
        self.games_played = 0
        self.high_score = 0
        
        # Animations
        self.lerped_q = [0.0, 0.0, 0.0]
        self.oscilloscope_history = deque(maxlen=70)
        
    def get_state(self):
        head = self.game.head
        dir_l = self.game.direction == 3
        dir_r = self.game.direction == 1
        dir_u = self.game.direction == 0
        dir_d = self.game.direction == 2
        
        state = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                check_pt = Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)
                state.append(self.game.is_collision(check_pt))
                
        state.extend([dir_l, dir_r, dir_u, dir_d])
        state.extend([
            self.game.food.x < self.game.head.x,
            self.game.food.x > self.game.head.x,
            self.game.food.y < self.game.head.y,
            self.game.food.y > self.game.head.y
        ])
        return np.array(state, dtype=int)
        
    def get_ai_action_and_q(self, state):
        if not self.model_loaded:
            return [1, 0, 0], [0.0, 0.0, 0.0]
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0).cpu().numpy()
        
        move_idx = np.argmax(q_values)
        action = [0, 0, 0]
        action[move_idx] = 1
        return action, q_values

    def get_manual_action(self):
        curr = self.game.direction
        desired = self.manual_direction
        action = [1, 0, 0]
        if desired == curr:
            action = [1, 0, 0]
        elif desired == (curr + 1) % 4:
            action = [0, 1, 0]
        elif desired == (curr - 1) % 4:
            action = [0, 0, 1]
        return action

    def draw_panel_card(self, rect, title, is_active=False, active_color=COLOR_CYAN):
        # Translucent glass panel look
        pygame.draw.rect(self.game.display, COLOR_BG_CARD, rect, border_radius=8)
        border_color = active_color if is_active else COLOR_BORDER
        pygame.draw.rect(self.game.display, border_color, rect, 1, border_radius=8)
        
        # Glow corner ticks for premium UI
        offset = 6
        pygame.draw.line(self.game.display, active_color, (rect.x, rect.y), (rect.x + offset, rect.y), 2)
        pygame.draw.line(self.game.display, active_color, (rect.x, rect.y), (rect.x, rect.y + offset), 2)
        pygame.draw.line(self.game.display, active_color, (rect.right - 1, rect.y), (rect.right - 1 - offset, rect.y), 2)
        pygame.draw.line(self.game.display, active_color, (rect.right - 1, rect.y), (rect.right - 1, rect.y + offset), 2)
        
        pygame.draw.line(self.game.display, active_color, (rect.x, rect.bottom - 1), (rect.x + offset, rect.bottom - 1), 2)
        pygame.draw.line(self.game.display, active_color, (rect.x, rect.bottom - 1), (rect.x, rect.bottom - 1 - offset), 2)
        pygame.draw.line(self.game.display, active_color, (rect.right - 1, rect.bottom - 1), (rect.right - 1 - offset, rect.bottom - 1), 2)
        pygame.draw.line(self.game.display, active_color, (rect.right - 1, rect.bottom - 1), (rect.right - 1, rect.bottom - 1 - offset), 2)

        # Title visual chip
        accent_rect = pygame.Rect(rect.x + 12, rect.y + 12, 3, 13)
        pygame.draw.rect(self.game.display, active_color, accent_rect, border_radius=2)
        txt = self.game.font_section.render(title, True, COLOR_TEXT_PRIMARY)
        
        # Soft shadow for title
        shadow = self.game.font_section.render(title, True, (0, 0, 0))
        self.game.display.blit(shadow, (rect.x + 23, rect.y + 11))
        self.game.display.blit(txt, (rect.x + 22, rect.y + 10))

    def draw_dashboard(self, q_values, chosen_action, state_arr, inference_time_ms):
        dash_x = self.game.w + 40
        dash_w = self.game.display_width - dash_x - 20
        
        # Dashboard Panel Fill
        pygame.draw.rect(self.game.display, COLOR_BG_DASH, pygame.Rect(self.game.w + 30, 0, self.game.display_width - self.game.w - 30, self.game.display_height))
        pygame.draw.line(self.game.display, COLOR_BORDER, (self.game.w + 30, 0), (self.game.w + 30, self.game.display_height), 2)

        # Titles
        title_txt = self.game.font_title.render("SYSTEM DIAGNOSTICS", True, COLOR_CYAN)
        shadow_title = self.game.font_title.render("SYSTEM DIAGNOSTICS", True, (0, 0, 0))
        self.game.display.blit(shadow_title, (dash_x + 1, 16))
        self.game.display.blit(title_txt, (dash_x, 15))
        
        sub_txt = self.game.font_small.render("Deep Q-Network Inference Engine Status", True, COLOR_TEXT_MUTED)
        self.game.display.blit(sub_txt, (dash_x, 38))

        # Card 1: System control status
        mode_rect = pygame.Rect(dash_x, 58, dash_w, 115)
        status_color = COLOR_GREEN if self.ai_mode else COLOR_ORANGE
        self.draw_panel_card(mode_rect, "SYSTEM CONTROL STATUS", is_active=True, active_color=status_color)
        
        mode_str = "AI AUTOPLAY (STABLE)" if self.ai_mode else "MANUAL DRIVE OVERRIDE"
        self.game.display.blit(self.game.font_body_bold.render(mode_str, True, status_color), (mode_rect.x + 15, mode_rect.y + 32))
        
        action_names_lookup = ["STRAIGHT A0 ⬆️", "RIGHT TURN A1 ➡️", "LEFT TURN A2 ⬅️"]
        chosen_idx = np.argmax(chosen_action)
        decision_str = action_names_lookup[chosen_idx]
        
        cardinal_lookup = ["NORTH ⬆️", "EAST ➡️", "SOUTH ⬇️", "WEST ⬅️"]
        dir_str = cardinal_lookup[self.game.direction]
        
        lbl_dec = self.game.font_body.render("DECISION:", True, COLOR_TEXT_MUTED)
        val_dec = self.game.font_body_bold.render(decision_str, True, COLOR_CYAN)
        self.game.display.blit(lbl_dec, (mode_rect.x + 15, mode_rect.y + 60))
        self.game.display.blit(val_dec, (mode_rect.x + 100, mode_rect.y + 60))
        
        lbl_dir = self.game.font_body.render("COMPASS :", True, COLOR_TEXT_MUTED)
        val_dir = self.game.font_body_bold.render(dir_str, True, COLOR_GREEN)
        self.game.display.blit(lbl_dir, (mode_rect.x + 15, mode_rect.y + 85))
        self.game.display.blit(val_dir, (mode_rect.x + 100, mode_rect.y + 85))

        # Blinking LED
        blink = (int(time.time() * 2) % 2 == 0)
        led_color = status_color if blink else (15, 23, 42)
        pygame.draw.circle(self.game.display, led_color, (mode_rect.right - 25, mode_rect.y + 38), 5)
        
        # Card 2: Q-Values output
        q_rect = pygame.Rect(dash_x, 185, dash_w, 125)
        self.draw_panel_card(q_rect, "DQN OUTPUT TENSORS (Q-VALUES)", is_active=self.ai_mode, active_color=COLOR_CYAN)
        
        action_names = ["STRAIGHT (A0)", "RIGHT TURN (A1)", "LEFT TURN (A2)"]
        
        y_offset = q_rect.y + 38
        for i, name in enumerate(action_names):
            q_target = q_values[i]
            self.lerped_q[i] += (q_target - self.lerped_q[i]) * 0.2
            
            is_chosen = (chosen_action[i] == 1)
            text_color = COLOR_GREEN if is_chosen else COLOR_TEXT_PRIMARY
            
            lbl = self.game.font_body.render(name, True, text_color)
            val = self.game.font_body_bold.render(f"{self.lerped_q[i]:.4f}", True, text_color)
            self.game.display.blit(lbl, (q_rect.x + 15, y_offset))
            self.game.display.blit(val, (q_rect.x + 145, y_offset))
            
            bar_w = 120
            bar_h = 10
            bar_x = q_rect.x + 215
            
            # Progress bar container
            pygame.draw.rect(self.game.display, (15, 23, 42), pygame.Rect(bar_x, y_offset + 3, bar_w, bar_h), border_radius=4)
            
            # Normalize to visual slider length
            norm_q = max(0.05, min(1.0, (self.lerped_q[i] + 4.0) / 8.0))
            fill_w = int(norm_q * bar_w)
            bar_color = COLOR_GREEN if is_chosen else COLOR_CYAN
            pygame.draw.rect(self.game.display, bar_color, pygame.Rect(bar_x, y_offset + 3, fill_w, bar_h), border_radius=4)
            
            if is_chosen:
                pygame.draw.circle(self.game.display, COLOR_GREEN, (bar_x - 10, y_offset + 8), 3)
                
            y_offset += 26
            
        # 5. Local 5x5 Sensor Vision Card - Split evenly with gap
        card_gap = 12
        card_w = (dash_w - card_gap) // 2
        
        sens_rect = pygame.Rect(dash_x, 320, card_w, 135)
        self.draw_panel_card(sens_rect, "5x5 LOCAL VISION", is_active=True, active_color=COLOR_BORDER)
        
        grid_size = 14
        spacing = 2
        start_x = sens_rect.x + 12
        start_y = sens_rect.y + 35
        
        idx = 0
        for r in range(5):
            for c in range(5):
                cell_x = start_x + c * (grid_size + spacing)
                cell_y = start_y + r * (grid_size + spacing)
                
                if r == 2 and c == 2:
                    # Head - custom premium style
                    pygame.draw.rect(self.game.display, COLOR_SNAKE_HEAD, pygame.Rect(cell_x, cell_y, grid_size, grid_size), border_radius=3)
                    arrow_color = (0, 0, 0)
                    cx, cy = cell_x + grid_size//2, cell_y + grid_size//2
                    if self.game.direction == 0:   # UP
                        pygame.draw.polygon(self.game.display, arrow_color, [(cx, cy-4), (cx-3, cy+2), (cx+3, cy+2)])
                    elif self.game.direction == 1: # RIGHT
                        pygame.draw.polygon(self.game.display, arrow_color, [(cx+4, cy), (cx-2, cy-3), (cx-2, cy+3)])
                    elif self.game.direction == 2: # DOWN
                        pygame.draw.polygon(self.game.display, arrow_color, [(cx, cy+4), (cx-3, cy-2), (cx+3, cy-2)])
                    elif self.game.direction == 3: # LEFT
                        pygame.draw.polygon(self.game.display, arrow_color, [(cx-4, cy), (cx+2, cy-3), (cx+2, cy+3)])
                else:
                    is_danger = state_arr[idx]
                    idx += 1
                    cell_color = COLOR_RED if is_danger else (15, 23, 42)
                    pygame.draw.rect(self.game.display, cell_color, pygame.Rect(cell_x, cell_y, grid_size, grid_size), border_radius=2)
                    pygame.draw.rect(self.game.display, COLOR_BORDER, pygame.Rect(cell_x, cell_y, grid_size, grid_size), 1, border_radius=2)

        # Legend alongside the sensor grid (aligned to fit inside card_w)
        leg_x = sens_rect.x + 98
        leg_y = sens_rect.y + 35
        pygame.draw.rect(self.game.display, COLOR_SNAKE_HEAD, pygame.Rect(leg_x, leg_y, 8, 8), border_radius=2)
        self.game.display.blit(self.game.font_small.render("Head", True, COLOR_TEXT_PRIMARY), (leg_x + 13, leg_y - 3))
        
        leg_y += 18
        pygame.draw.rect(self.game.display, COLOR_RED, pygame.Rect(leg_x, leg_y, 8, 8), border_radius=2)
        self.game.display.blit(self.game.font_small.render("Danger", True, COLOR_TEXT_PRIMARY), (leg_x + 13, leg_y - 3))
        
        leg_y += 18
        pygame.draw.rect(self.game.display, (15, 23, 42), pygame.Rect(leg_x, leg_y, 8, 8), border_radius=2)
        pygame.draw.rect(self.game.display, COLOR_BORDER, pygame.Rect(leg_x, leg_y, 8, 8), 1, border_radius=2)
        self.game.display.blit(self.game.font_small.render("Clear", True, COLOR_TEXT_PRIMARY), (leg_x + 13, leg_y - 3))

        # 6. Oscilloscope Graph (aligned to right half)
        wave_rect = pygame.Rect(dash_x + card_w + card_gap, 320, card_w, 135)
        self.draw_panel_card(wave_rect, "OSCILLOSCOPE WAVE", is_active=True, active_color=COLOR_BORDER)
        
        self.oscilloscope_history.append(q_values[chosen_idx])
        
        plot_x = wave_rect.x + 12
        plot_y = wave_rect.y + 35
        plot_w = wave_rect.w - 24
        plot_h = wave_rect.h - 48
        
        pygame.draw.rect(self.game.display, (9, 12, 28), pygame.Rect(plot_x, plot_y, plot_w, plot_h), border_radius=4)
        pygame.draw.rect(self.game.display, COLOR_BORDER, pygame.Rect(plot_x, plot_y, plot_w, plot_h), 1, border_radius=4)
        pygame.draw.line(self.game.display, (26, 36, 74), (plot_x, plot_y + plot_h//2), (plot_x + plot_w, plot_y + plot_h//2), 1)
        
        if len(self.oscilloscope_history) > 1:
            points = []
            for i, val in enumerate(self.oscilloscope_history):
                norm_y = (val + 4.0) / 8.0
                norm_y = max(0.0, min(1.0, norm_y))
                point_x = plot_x + int((i / (self.oscilloscope_history.maxlen - 1)) * plot_w)
                point_y = plot_y + plot_h - int(norm_y * plot_h)
                points.append((point_x, point_y))
            
            # 1. Thick Neon Bloom Glow line
            pygame.draw.lines(self.game.display, (6, 182, 212, 60), False, points, 3)
            # 2. Bright Core Line
            pygame.draw.lines(self.game.display, COLOR_CYAN, False, points, 1)

        # Card 5: Metrics footer
        stat_rect = pygame.Rect(dash_x, 465, dash_w, 70)
        self.draw_panel_card(stat_rect, "SYSTEM METRICS & HELP", is_active=True, active_color=COLOR_BORDER)
        
        y_metrics = stat_rect.y + 30
        self.game.display.blit(self.game.font_small.render(f"LATENCY: {inference_time_ms:.2f}ms", True, COLOR_CYAN), (stat_rect.x + 15, y_metrics))
        self.game.display.blit(self.game.font_small.render(f"FPS: {self.speed}", True, COLOR_CYAN), (stat_rect.x + 130, y_metrics))
        self.game.display.blit(self.game.font_small.render(f"RUNS: {self.games_played}", True, COLOR_CYAN), (stat_rect.x + 200, y_metrics))
        self.game.display.blit(self.game.font_small.render(f"HIGH: {self.high_score}", True, COLOR_CYAN), (stat_rect.x + 270, y_metrics))
        self.game.display.blit(self.game.font_small.render("[SPACE] Toggle Control | [Arrow Keys] Drive Manual | [+/-] Speed", True, COLOR_TEXT_MUTED), (stat_rect.x + 15, y_metrics + 20))

    def draw_game_grid(self):
        # Draw high-tech crosshairs/dots at grid intersections instead of flat grid lines
        for x in range(20, 20 + self.game.w, BLOCK_SIZE):
            for y in range(50, 50 + self.game.h, BLOCK_SIZE):
                # Tiny tech crosshairs/dots
                pygame.draw.rect(self.game.display, COLOR_GRID_DOT, pygame.Rect(x, y, 2, 2))

    def run(self):
        running = True
        game_over = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.ai_mode = not self.ai_mode
                        self.manual_direction = self.game.direction
                    elif event.key in [pygame.K_KP_PLUS, pygame.K_EQUALS]:
                        self.speed = min(60, self.speed + 1)
                    elif event.key in [pygame.K_KP_MINUS, pygame.K_MINUS]:
                        self.speed = max(1, self.speed - 1)
                    elif event.key == pygame.K_PAGEUP:
                        self.speed = min(60, self.speed + 5)
                    elif event.key == pygame.K_PAGEDOWN:
                        self.speed = max(1, self.speed - 5)
                    elif not self.ai_mode:
                        if event.key == pygame.K_UP and self.game.direction != 2:
                            self.manual_direction = 0
                        elif event.key == pygame.K_RIGHT and self.game.direction != 3:
                            self.manual_direction = 1
                        elif event.key == pygame.K_DOWN and self.game.direction != 0:
                            self.manual_direction = 2
                        elif event.key == pygame.K_LEFT and self.game.direction != 1:
                            self.manual_direction = 3
                            
            if game_over:
                self.games_played += 1
                if self.game.score > self.high_score:
                    self.high_score = self.game.score
                pygame.time.delay(1000)
                self.game.reset()
                self.manual_direction = self.game.direction
                game_over = False
                
            state_arr = self.get_state()
            
            start_time = time.perf_counter()
            if self.ai_mode:
                action, q_values = self.get_ai_action_and_q(state_arr)
            else:
                action = self.get_manual_action()
                _, q_values = self.get_ai_action_and_q(state_arr)
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000.0
            
            game_over, score = self.game.play_step(action)
            
            # --- Rendering Phase ---
            self.game.display.fill(COLOR_BG_GAME)
            
            # Draw game screen container panel
            game_panel_rect = pygame.Rect(18, 48, self.game.w + 4, self.game.h + 4)
            pygame.draw.rect(self.game.display, COLOR_BORDER, game_panel_rect, 1, border_radius=8)
            
            # Corner accents on game panel
            offset = 8
            pygame.draw.line(self.game.display, COLOR_CYAN, (18, 48), (18 + offset, 48), 2)
            pygame.draw.line(self.game.display, COLOR_CYAN, (18, 48), (18, 48 + offset), 2)
            pygame.draw.line(self.game.display, COLOR_CYAN, (18 + self.game.w + 3, 48), (18 + self.game.w + 3 - offset, 48), 2)
            pygame.draw.line(self.game.display, COLOR_CYAN, (18 + self.game.w + 3, 48), (18 + self.game.w + 3, 48 + offset), 2)
            
            pygame.draw.rect(self.game.display, COLOR_BG_GAME, pygame.Rect(20, 50, self.game.w, self.game.h), border_radius=6)
            self.draw_game_grid()
            
            # Draw Obstacles with high-tech outlines
            for pt in self.game.obstacles:
                draw_x = 20 + pt.x
                draw_y = 50 + pt.y
                pygame.draw.rect(self.game.display, COLOR_OBSTACLE, pygame.Rect(draw_x, draw_y, BLOCK_SIZE, BLOCK_SIZE), border_radius=4)
                pygame.draw.rect(self.game.display, COLOR_OBSTACLE_BORDER, pygame.Rect(draw_x, draw_y, BLOCK_SIZE, BLOCK_SIZE), 1, border_radius=4)
                # Draw a tiny dot inside obstacle
                pygame.draw.circle(self.game.display, COLOR_CYAN, (draw_x + BLOCK_SIZE//2, draw_y + BLOCK_SIZE//2), 2)
                
            # Draw Snake - Beautiful organic overlapping spheres
            for i, pt in enumerate(self.game.snake):
                draw_x = 20 + pt.x
                draw_y = 50 + pt.y
                cx, cy = draw_x + BLOCK_SIZE//2, draw_y + BLOCK_SIZE//2
                
                if i == 0:
                    # Draw Snake Head as smooth round circle
                    pygame.draw.circle(self.game.display, COLOR_SNAKE_HEAD, (cx, cy), BLOCK_SIZE//2 + 1)
                    
                    # Laser-glow pointer arrow in direction of travel
                    arrow_color = (0, 0, 0)
                    if self.game.direction == 0:   # UP
                        pygame.draw.polygon(self.game.display, arrow_color, [(cx, cy-5), (cx-4, cy+2), (cx+4, cy+2)])
                    elif self.game.direction == 1: # RIGHT
                        pygame.draw.polygon(self.game.display, arrow_color, [(cx+5, cy), (cx-2, cy-4), (cx-2, cy+4)])
                    elif self.game.direction == 2: # DOWN
                        pygame.draw.polygon(self.game.display, arrow_color, [(cx, cy+5), (cx-4, cy-2), (cx+4, cy-2)])
                    elif self.game.direction == 3: # LEFT
                        pygame.draw.polygon(self.game.display, arrow_color, [(cx-5, cy), (cx+2, cy-4), (cx+2, cy+4)])
                else:
                    # Linear color blending along body length
                    t = i / len(self.game.snake)
                    r = int(COLOR_SNAKE_BODY_START[0] * (1 - t) + COLOR_SNAKE_BODY_END[0] * t)
                    g = int(COLOR_SNAKE_BODY_START[1] * (1 - t) + COLOR_SNAKE_BODY_END[1] * t)
                    b = int(COLOR_SNAKE_BODY_START[2] * (1 - t) + COLOR_SNAKE_BODY_END[2] * t)
                    
                    # Draw body segment as circles
                    pygame.draw.circle(self.game.display, (r, g, b), (cx, cy), BLOCK_SIZE//2 - 1)
                    # Small core highlight
                    pygame.draw.circle(self.game.display, (max(0, r-40), max(0, g-40), max(0, b-40)), (cx, cy), BLOCK_SIZE//4)
                    
            # Draw Pulsing Neon Energy Food
            food_x = 20 + self.game.food.x + BLOCK_SIZE//2
            food_y = 50 + self.game.food.y + BLOCK_SIZE//2
            
            # Beautiful concentric waves
            pulse = math.sin(time.time() * 10) * 2.5
            radius_core = max(4.0, 5.0 + pulse * 0.3)
            
            # Concentric glow rings
            for r_mult, alpha in [(2.2, 12), (1.6, 25), (1.1, 40)]:
                glow_r = int((BLOCK_SIZE//2) * r_mult + pulse * 0.5)
                if glow_r > 0:
                    glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (255, 46, 108, alpha), (glow_r, glow_r), glow_r)
                    self.game.display.blit(glow_surf, (food_x - glow_r, food_y - glow_r))
            
            # Food Core
            pygame.draw.circle(self.game.display, COLOR_FOOD_CORE, (food_x, food_y), int(radius_core))
            pygame.draw.circle(self.game.display, (255, 255, 255), (food_x - 1, food_y - 1), int(radius_core//2 - 1)) # Shiny gloss
 
            # Title
            score_lbl = self.game.font_title.render("SNAKE GAME SANDBOX", True, COLOR_CYAN)
            score_shadow = self.game.font_title.render("SNAKE GAME SANDBOX", True, (0,0,0))
            self.game.display.blit(score_shadow, (21, 16))
            self.game.display.blit(score_lbl, (20, 15))
            
            score_val = self.game.font_body_bold.render(f"SCORE: {self.game.score}", True, COLOR_TEXT_PRIMARY)
            self.game.display.blit(score_val, (310, 18))
            
            # Render Dashboard
            self.draw_dashboard(q_values, action, state_arr, inference_time_ms)
            
            pygame.display.flip()
            self.game.clock.tick(self.speed)
            
        pygame.quit()

if __name__ == "__main__":
    demo = DemoManager()
    demo.run()
