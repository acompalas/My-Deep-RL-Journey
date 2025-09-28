import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Q-Network
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Loss Computation
# ---------------------------
def compute_loss(batch, q_net, target_net, gamma):
    states, actions, rewards, next_states, dones = batch

    # 1. Predicted Q(s,a; w) from the behavior network
    q_pred = q_net(states).gather(1, actions)  # shape: [batch, 1]

    # 2. Compute target y = r + γ max_a' Q̂(s', a'; w^-)
    with torch.no_grad():
        q_next_max = target_net(next_states).max(1, keepdim=True)[0]
        y_target = rewards + gamma * (1 - dones) * q_next_max

    # 3. Loss = MSE(y_target, q_pred)
    loss = nn.MSELoss()(q_pred, y_target)
    return loss

# ---------------------------
# Target Network Update
# ---------------------------
def update_target_network(q_net, target_net, tau=1.0):
    """
    Copy weights from q_net to target_net.
    tau=1.0 → hard update
    tau<1.0 → soft update
    """
    for target_param, param in zip(target_net.parameters(), q_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def agent_learn(batch, q_net, target_net, optimizer, gamma, tau):
    loss = compute_loss(batch, q_net, target_net, gamma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    update_target_network(q_net, target_net, tau)
    return loss.item()