import streamlit as st
import numpy as np

class SlotMachine:
    """
    A SlotMachine with multiple arms. Each arm has its own reward distribution.
    Can be used for manual interactive pulling or for agent-based simulations.
    """
    
    def __init__(self, n_arms, reward_distributions=None, seed=None):
        """
        Initialize the slot machine.
        """
        
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.means = []
        self.stds = []
        
        if reward_distributions is not None:
            # use what the user gave
            for i, spec in enumerate(reward_distributions):
                mu = float(spec.get("mean", 0.0))
                std = float(spec.get("std", 1.0))
                if std < 0:
                    raise ValueError(f"Arm {i}: std must be >= 0")
                self.means.append(mu)
                self.stds.append(std)
                
            # if not enough arms defined, randomize the rest
            while len(self.means) < n_arms:
                mu = self.rng.normal(0, 1)         # random mean
                std = float(self.rng.uniform(0.5, 2.0))  # random std
                self.means.append(mu)
                self.stds.append(std)
        
        else:
            # no distributions given → randomize all arms
            self.means = list(self.rng.normal(0, 1, size=n_arms))
            self.stds = list(self.rng.uniform(0.5, 2.0, size=n_arms))

        # counters for usage
        self.t = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        
        
    def reset(self):
        """
        Reset the slot machine to its initial state.
        Useful for restarting simulations.
        """
        self.t = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        
    def pull(self, arm_index):
        """
        Pull a specific arm manually.

        Args:
            arm_index (int): Index of the arm to pull.

        Returns:
            reward (float): The observed reward sampled from the arm's distribution.
        """
        
        # basic bounds check
        if not (0 <= arm_index < self.n_arms):
            raise IndexError(f"arm_index out of range: {arm_index}")
        
        mean = float(self.means[arm_index])
        std = float(self.stds[arm_index])
        
        # sample one reward
        reward = float(self.rng.normal(loc=mean, scale=std))
        
        # update counters
        self.t += 1
        self.counts[arm_index] += 1

        return reward
    
    def step(self, action: int):
        r = self.pull(action)
        return r, {}
    
class BaseAgent:
    def __init__(self, n_arms, alpha: float | None = None, seed: int | None = None):
        self.n_arms = n_arms
        self.alpha = alpha  # if None -> sample average
        self.Q = np.zeros(n_arms, dtype=float)
        self.N = np.zeros(n_arms, dtype=int)
        self.rng = np.random.default_rng(seed)

    def _update_Q(self, a, r):
        self.N[a] += 1
        if self.alpha is None:
            # sample average
            self.Q[a] += (r - self.Q[a]) / self.N[a]
        else:
            # constant step size
            self.Q[a] += self.alpha * (r - self.Q[a])
            
class ExploreAgent(BaseAgent):
    def select_action(self, t):
        return int(self.rng.integers(0, self.n_arms))
    def update(self, a, r):
        self._update_Q(a, r)
        
class ExploitAgent(BaseAgent):
    def __init__(self, n_arms, m_per_arm: int = 1, **kw):
        super().__init__(n_arms, **kw)
        self.m_per_arm = int(m_per_arm)
        self._phase = "explore"
        self._round_robin = 0
        self._commit_arm = None

    def select_action(self, t):
        if self._phase == "explore":
            a = self._round_robin % self.n_arms
            return a
        else:
            return self._commit_arm

    def update(self, a, r):
        self._update_Q(a, r)
        if self._phase == "explore":
            self._round_robin += 1
            if self._round_robin >= self.n_arms * self.m_per_arm:
                self._phase = "commit"
                self._commit_arm = int(np.argmax(self.Q))

class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, n_arms, epsilon: float = 0.1, **kw):
        super().__init__(n_arms, **kw)
        self.epsilon = float(epsilon)

    def select_action(self, t):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.Q))

    def update(self, a, r):
        self._update_Q(a, r)

class UCBAgent(BaseAgent):
    def __init__(self, n_arms, c: float = 2.0, **kw):
        super().__init__(n_arms, **kw)
        self.c = float(c)

    def select_action(self, t):
        # Ensure every arm is tried at least once
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            # deterministic order or random among untried
            return int(untried[0])
        # UCB score
        bonus = self.c * np.sqrt(np.log(max(1, t)) / self.N)  # t starts at 1 in your loop
        return int(np.argmax(self.Q + bonus))

    def update(self, a, r):
        self._update_Q(a, r)

def run_episode(env, agent, horizon: int):
    rewards = []
    for t in range(1, horizon + 1):
        a = agent.select_action(t)
        r = env.pull(a)  # or env.step(a)[0]
        agent.update(a, r)
        rewards.append(r)
    return np.array(rewards)

def regret_curve(rewards, best_mean):
    # pseudo‑regret vs best stationary arm
    # R_T = T * mu* - sum_{t=1}^T r_t
    cum_reward = np.cumsum(rewards)
    T = np.arange(1, len(rewards)+1)
    return T * best_mean - cum_reward

def run_many(env, AgentCls, agent_kwargs, episodes: int, horizon: int):
    all_rewards = []
    best_mean = float(np.max(env.means))
    for _ in range(episodes):
        env.reset()  # ✅ no reseed flag; keeps same means/stds and RNG
        agent = AgentCls(env.n_arms, **agent_kwargs)
        rewards = run_episode(env, agent, horizon)
        all_rewards.append(rewards)
    all_rewards = np.vstack(all_rewards)  # [episodes, horizon]
    mean_regret = np.mean([regret_curve(r, best_mean) for r in all_rewards], axis=0)
    return all_rewards, mean_regret

