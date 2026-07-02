# 🎮 Interactive AI Snake Game Demo

An interactive dashboard built with **Pygame** to demonstrate and verify the decisions of the trained Deep Q-Network (DQN) model in real time.

---

## ✨ Features
1. **Dual Control Mode (AI vs. Manual)**:
   - Toggle between **AI Autoplay** and **Manual User Control** dynamically at the press of a key.
2. **Real-time Q-Value Visualization**:
   - Displays the expected future cumulative rewards (confidence scores) for the three possible actions: `Straight`, `Right Turn`, and `Left Turn`.
3. **5x5 Local Sensory Grid**:
   - Visualizes the 32-dimensional input state fed into the model, mapping local obstacles, walls, and the snake body in a 5x5 matrix.
4. **Decision Oscilloscope**:
   - Plots the selected action's Q-values over time, giving a scrolling waveform of the AI's confidence levels.
5. **Modern Aesthetics**:
   - Cyberpunk-themed visuals with high-tech crosshairs, glowing gradients, pulsing energy cores, and blinking LEDs.

---

## 🚀 How to Run

Launch the script from the project root using:
```bash
uv run demo/interactive_demo.py
```

### ⌨️ Keyboard Hotkeys
* **`SPACEBAR`** : Toggle between AI autoplay and Manual control.
* **`ARROW KEYS`** : Manually steer the snake when in Manual Mode.
* **`+` / `-`** (or **PageUp/PageDown**) : Increase or decrease simulation speed (FPS).
* **Close Window** : Safely exits the demo.
