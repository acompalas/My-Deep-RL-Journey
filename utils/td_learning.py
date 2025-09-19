import numpy as np
import matplotlib.pyplot as plt

class GridWorldEnv:
    def __init__(self, rows, cols, start=(0, 0), goal=None, obstacles=None):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.state = start
        self.goal = goal if goal else (rows - 1, cols - 1)
        self.obstacles = obstacles if obstacles else []

        # Actions: up, down, left, right
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        self.q_table = {
            (r, c): np.zeros(len(self.actions))
            for r in range(rows)
            for c in range(cols)
            if (r, c) not in self.obstacles
        }

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        dr, dc = self.actions[action]
        nr, nc = r + dr, c + dc

        if (0 <= nr < self.rows) and (0 <= nc < self.cols):
            next_state = (nr, nc)
        else:
            next_state = self.state

        if next_state in self.obstacles:
            next_state = self.state

        if next_state == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False

        self.state = next_state
        return next_state, reward, done

    def get_states(self):
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.obstacles
        ]


def render_gridworld(env: GridWorldEnv, path=None, color="red"):
    """
    Render grid with matplotlib.
    - White = free space
    - Black = obstacle
    - Yellow = start
    - Green = goal
    - Red line = path (if provided)
    """
    grid = np.ones((env.rows, env.cols, 3))  # RGB grid (white default)

    # Obstacles → black
    for (r, c) in env.obstacles:
        grid[r, c] = [0, 0, 0]

    # Start → yellow
    sr, sc = env.start
    grid[sr, sc] = [1, 1, 0]

    # Goal → green
    gr, gc = env.goal
    grid[gr, gc] = [0, 1, 0]

    fig, ax = plt.subplots(figsize=(env.cols, env.rows))
    ax.imshow(grid, extent=[0, env.cols, 0, env.rows])

    # Grid lines
    ax.set_xticks(np.arange(0, env.cols + 1, 1))
    ax.set_yticks(np.arange(0, env.rows + 1, 1))
    ax.grid(color="gray", linestyle="-", linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Path → red line
    if path:
        xs = [c + 0.5 for (_, c) in path]
        ys = [env.rows - r - 0.5 for (r, _) in path]  # flip y for matplotlib
        ax.plot(xs, ys, color=color, linestyle="-")

    return fig

def render_policy(env, Q, color="black"):
    """
    Render final greedy policy arrows for each state from Q-table.
    - ↑ ↓ ← → for best action
    - Obstacles are black
    - Start = yellow, Goal = green
    - Unvisited states = blank
    """
    grid = np.ones((env.rows, env.cols, 3))  # RGB grid (white default)

    # Obstacles → black
    for (r, c) in env.obstacles:
        grid[r, c] = [0, 0, 0]

    # Start → yellow
    sr, sc = env.start
    grid[sr, sc] = [1, 1, 0]

    # Goal → green
    gr, gc = env.goal
    grid[gr, gc] = [0, 1, 0]

    fig, ax = plt.subplots(figsize=(env.cols, env.rows))
    ax.imshow(grid, extent=[0, env.cols, 0, env.rows])

    # Grid lines
    ax.set_xticks(np.arange(0, env.cols + 1, 1))
    ax.set_yticks(np.arange(0, env.rows + 1, 1))
    ax.grid(color="gray", linestyle="-", linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Map actions to arrows
    action_arrows = {
        0: "↑",   # up
        1: "↓",   # down
        2: "←",   # left
        3: "→",   # right
    }

    for (r, c) in env.get_states():
        if (r, c) in [env.start, env.goal]:
            continue  # skip start/goal

        q_values = Q[(r, c)]
        if np.allclose(q_values, 0.0):
            # never visited → leave blank
            continue
        else:
            best_action = np.argmax(q_values)
            arrow = action_arrows[best_action]
            ax.text(c + 0.5, env.rows - r - 0.5, arrow,
                    ha="center", va="center", color=color, fontsize=14, weight="bold")

    return fig



class BaseAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon          # current ε
        self.epsilon_min = epsilon_min  # floor
        self.epsilon_decay = epsilon_decay  # multiplicative decay
        self.Q = {s: np.zeros(len(env.actions)) for s in env.get_states()}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.env.actions))  # explore
        return np.argmax(self.Q[state])  # exploit

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def run_episode(self, max_steps=50):
        """Implemented in child class"""
        raise NotImplementedError


class MonteCarloAgent(BaseAgent):
    def run_episode(self, max_steps=20):
        state = self.env.reset()
        episode = []
        path, total_reward = [state], 0

        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            path.append(next_state)
            total_reward += reward
            state = next_state
            if done: break

        # MC update
        G = 0
        for (s, a, r) in reversed(episode):
            G = self.gamma * G + r
            self.Q[s][a] += self.alpha * (G - self.Q[s][a])

        return path, total_reward


class SARSAAgent(BaseAgent):
    def run_episode(self, max_steps=20):
        state = self.env.reset()
        action = self.choose_action(state)
        path, total_reward = [state], 0

        for _ in range(max_steps):
            next_state, reward, done = self.env.step(action)
            path.append(next_state)
            total_reward += reward
            next_action = self.choose_action(next_state)

            self.Q[state][action] += self.alpha * (
                reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
            )
            state, action = next_state, next_action
            if done: break

        self.decay_epsilon()
        return path, total_reward


class QLearningAgent(BaseAgent):
    def run_episode(self, max_steps=20):
        state = self.env.reset()
        path, total_reward = [state], 0

        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            path.append(next_state)
            total_reward += reward

            self.Q[state][action] += self.alpha * (
                reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
            )
            state = next_state
            if done: break

        self.decay_epsilon()
        return path, total_reward

