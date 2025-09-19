import streamlit as st
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import tempfile, imageio

st.title("Deep Q Networks")

section = st.selectbox(
    "",
    [
        "Demo",
        "My Notes",
        "Playing Atari with Deep Reinforcement Learning"
    ],
)

import streamlit as st
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import tempfile, imageio

# ---------------------------
# Section: Demo
# ---------------------------
if section == "Demo":
    st.title("Demo: Deep Q-Learning with Experience Replay")

    st.header("Deep Q-Learning Algorithm")
    st.markdown(r"""
    This demo implements **Deep Q-Learning with Experience Replay**, inspired by  
    *Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)*.  

    - Uses a replay buffer to break correlation between consecutive transitions.  
    - Maintains a target network for stable learning.  
    - Uses $\epsilon$-greedy exploration (decaying $\epsilon$ from 1.0 → 0.1).  
    """)

    st.image("assets/deep_q-learning.png", caption="Deep Q-Learning Algorithm", use_container_width=True)

    # ---------------------------
    # Training parameters
    # ---------------------------
    st.divider()
    st.header("Training Parameters")
    col1, col2, col3 = st.columns(3)
    episodes = col1.number_input("Episodes", min_value=50, max_value=2000, value=500, step=50)
    batch_size = col2.number_input("Batch Size", min_value=32, max_value=512, value=64, step=32)
    lr = col3.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")

    col4, col5, col6 = st.columns(3)
    gamma = col4.number_input("Discount γ", min_value=0.8, max_value=0.999, value=0.99, step=0.01)
    target_update = col5.number_input("Target Update Freq", min_value=10, max_value=500, value=100, step=10)
    buffer_size = col6.number_input("Replay Buffer Size", min_value=10000, max_value=200000, value=50000, step=10000)

    col7, col8 = st.columns(2)
    eps_start = col7.number_input("Epsilon Start", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    eps_end = col8.number_input("Epsilon End", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

    train_button = st.button("Train", use_container_width=True)

    # ---------------------------
    # Placeholders
    # ---------------------------
    plot_placeholder = st.empty()
    log_placeholder = st.empty()
    progress = st.progress(0)
    first_ep_placeholder, last_ep_placeholder = st.columns(2)

    # Default blank plot
    fig, ax = plt.subplots()
    ax.plot([], [])
    ax.set_title("Average Reward (Smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    plot_placeholder.pyplot(fig, use_container_width=True)

    # ---------------------------
    # Q-Network
    # ---------------------------
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    # ---------------------------
    # Training Loop
    # ---------------------------
    if train_button:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        q_net = QNetwork(state_dim, action_dim)
        target_net = QNetwork(state_dim, action_dim)
        target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

        replay_buffer = deque(maxlen=buffer_size)
        returns, avg_returns = [], []
        log_history = deque(maxlen=10)

        first_ep_frames, last_ep_frames = None, None

        for ep in range(episodes):
            # Smooth epsilon decay across episodes
            epsilon = eps_end + (eps_start - eps_end) * (1 - ep / episodes)

            state, _ = env.reset()
            done, ep_ret = False, 0
            frames = []

            while not done:
                # Epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_vals = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                        action = q_vals.argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.append((state, action, reward, next_state, done))

                state = next_state
                ep_ret += reward

                # Record first and last episode frames
                if ep == 0 or ep == episodes - 1:
                    frames.append(env.render())

                # Train step
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.tensor(states, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                    next_states = torch.tensor(next_states, dtype=torch.float32)
                    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                    # Compute targets
                    with torch.no_grad():
                        max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                        target_q = rewards + gamma * (1 - dones) * max_next_q

                    # Compute current estimates
                    current_q = q_net(states).gather(1, actions)

                    # Loss
                    loss = nn.MSELoss()(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Update target network
            if ep % target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            # Logging
            returns.append(ep_ret)
            avg_returns.append(np.mean(returns[-50:]))
            log_history.append(f"Episode {ep+1}/{episodes} | Return: {ep_ret:.1f} | Eps {epsilon:.2f}")

            # Update plot
            fig, ax = plt.subplots()
            ax.plot(avg_returns, label="Smoothed Return (50-ep)")
            ax.axhline(200, color="red", linestyle="--", label="Solved Threshold (200)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.legend()
            plot_placeholder.pyplot(fig, use_container_width=True)

            # Update logs
            log_placeholder.code("\n".join(log_history), language="")

            # Update progress
            progress.progress((ep + 1) / episodes)

            # Save videos
            if ep == 0:
                first_ep_frames = frames
            if ep == episodes - 1:
                last_ep_frames = frames

        # ---------------------------
        # Save first & last videos
        # ---------------------------
        def save_video(frames, label, placeholder):
            if frames:
                try:
                    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    imageio.mimsave(tmpfile.name, frames, fps=30)
                    placeholder.subheader(label)
                    placeholder.video(tmpfile.name)
                except Exception:
                    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
                    imageio.mimsave(tmpfile.name, frames, fps=30)
                    placeholder.subheader(label + " (GIF fallback)")
                    placeholder.image(tmpfile.name)

        save_video(first_ep_frames, "First Episode", first_ep_placeholder)
        save_video(last_ep_frames, "Last Episode", last_ep_placeholder)

    
if section == "My Notes":
    st.title("My Notes")
    
    st.markdown(r"""
                
                
    ###
                
    ### Deep Q-Networks (DQNs)
    
    ---
    #### Recall Q-Learning
    
    Q-Learning is an **off-policy value-based** reinforcement learning method.  
    It learns the action-value function:
    
    $$
    Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big[ r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \Big]
    $$
    
    - **Off-policy**: uses a *behavior policy* (like $\epsilon$-greedy) to explore,  
      but updates towards the *target policy* (greedy w.r.t $Q$).  
    - **Sample-efficient**: one step of environment interaction updates many $Q$ estimates.  
    - Works well in **discrete state & action spaces**, but scales poorly to continuous/high-dimensional settings.
    
    ---
    #### From Discrete to Continuous
    
    - In continuous **state spaces**, we can’t tabulate $Q(s,a)$.  
      → We use **function approximation** (neural nets) to generalize.  
    - In continuous **action spaces**, $\max_a Q(s',a)$ is intractable.  
      → DQN handles *discrete actions only*. For continuous actions we need Actor–Critic variants (DDPG, SAC).
    
    ---
    #### Function Approximation
    
    Instead of a table:
    
    $$
    Q(s,a; \theta) \approx \text{Neural Network}(s,a)
    $$
    
    - Parameters $\theta$ updated by minimizing the **TD error**:
    
    $$
    L(\theta) = \Big( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a;\theta) \Big)^2
    $$
    
    - $\theta^-$ are parameters of a **target network**, updated slowly for stability.
    
    ---
    #### How Deep Q-Networks Work
    
    Key innovations that made DQN successful (Atari, 2015):
    
    1. **Replay buffer** $\mathcal{D}$  
       - Store transitions $(s,a,r,s')$.  
       - Sample minibatches uniformly → break correlation between consecutive states.  
    
    2. **Target network**  
       - Maintain a separate copy $Q(s,a;\theta^-)$, updated slowly.  
       - Reduces oscillations and divergence.
    
    3. **Neural network function approximator**  
       - Input: state (pixels or low-d features).  
       - Output: $Q$-values for all actions.  
    
    ---
    #### DQN Algorithm (High-Level)
    
    1. Initialize replay buffer $\mathcal{D}$, Q-network $\theta$, target network $\theta^-$.  
    2. For each step:  
       - Select action $a$ with $\epsilon$-greedy policy.  
       - Execute $a$, observe $(s,a,r,s')$.  
       - Store in $\mathcal{D}$.  
       - Sample minibatch, compute TD error, backprop to update $\theta$.  
       - Every $C$ steps, update target network $\theta^- \leftarrow \theta$.  
    
    ---
    #### Pros and Cons
    
    **Pros**
    - Compresses Q-table for data
    - Can handle states not seen during training
    - Q-networks act as feature extractors allowing for generalization of Q-values
    - Can leverage both on and off policy updates for sample efficiency
    
    **Cons**
    - Deterministic: cannot learn stochastic policies
    - Cannot directly be applied to continuous action spaces (need to discretize)
    - Need to separately add $\epsilon$-greedy algorithm to balance exploration and exploitation
    """)
    
if section == "Playing Atari with Deep Reinforcement Learning":
    st.title("Playing Atari with Deep Reinforcement Learning")
    
    st.markdown(r"""
                
    ### Abstract
    - First deep learning model to successfully learn control policies directly from high-dimensional sensory input 
    - **Model**: Convolutional Neural Network + Q-Learning
    - **Input**: Raw Pixels
    - **Output**: value function estimating future rewards
    - **Experiments**: Seven Atari 2600 games from the Arcade Learning Environment
    
    ### Introduction
    - Previous approaches to high dimensional sensory inputs heavily dependent on feature representations and were often linear
    - Deep Learning made it possible to extract high-level features from raw sensory data such as Convolutional Neural Networks
    - Deep Learning applications require large amounts of hand labelled data
    - RL algorithms must be able to learn from scalar reward signal that is sparse, noisy, and delayed.
    - **Architecture**: Convolutional Neural Network with Q-learning + Experience replay (randomly samples previous transitions)
    - Train single network to play seven Atari games
    - Outperforms all previous RL algorithms and outperforms humans on 3 games
             
    ### Background
    
    #### Reward
    $$
    R_t = \sum_{t' = t}^T \gamma^{t' - t} r_{t'}
    $$
    
    #### Optimal Value Function
    $$
    Q^*(s, a) = E_{s'\sim \epsilon}\big[r + \gamma \text{max}_{a'}Q^8(s', a') | s, a\big]
    $$
    
    #### Function Approximation
    
    $$
    L_i(\theta_i) = E_{s, a \sim \rho(\cdot)} \big[(y_i- Q(s, a; \theta_i)^2\big]
    $$
    
    $$
    Q_i \rightarrow Q^* \quad \text{as } i \rightarrow \infty
    $$
    
    $$
    y_i = E_{s'\sim \epsilon}\big[r + \gamma \text{max}_{a'} Q(s', a'; \theta_{i-1})|s, a \big]
    $$
    
    #### Gradient for Backprop
    
    $$
    \nabla_{\theta_i} L_i(\theta_i) = E_{s, a \sim \rho(\cdot); s' \sim \epsilon} \big[\left(r + \gamma \, \text{max}_a Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i)) \nabla_{\theta_i} Q(s, a; \theta_i\right)\big]
    $$
    
    Trained using Stochastic Gradient Descent
    
    #### Policy 
    
    Off-Policy 
    
    $$
    \underset{a}{\arg\max}\; Q(s, a; \theta)
    $$
    
    Epsilon-Greedy Target Policy

    $$
    \pi(s) =
    \begin{cases}
    \underset{a}{\arg\max}\; Q(s, a; \theta) & \text{with probability } 1-\epsilon \\
    \text{random action} & \text{with probability } \epsilon
    \end{cases}
    $$
    
    ### Related Work
    - TD-gammon previous best known success of reinforcement learning learnt by self play achieving human performance
    - Model free and used approximated value function using multi-layer perceptron with one hidden layer
    
    ### Deep Reinforcement Learning
    
    #### **Experience Replay**
    
    Store the agents epxeriences at each time-step $e_t$ in a dataset $D$ pooled over many episodes into a replay memory
    $$
    e_t = (s_t, a_t, r_t, s_{t+1}) \quad D = e_1, \dots, e_N
    $$
    
    During the inner loop of the DQN algorithm, we apply minibatch Q-learning updates using samples of experiences $e$ drawn from $D$
    $$
    e \sim D
    $$
    
    1. Each step of experience is potentiall used in many weight updates
    2. Learning directly from consecutive samples is inefficient due to strong correlations. Randomizing the smaples breaks correlations and reduces variance
    3. When learning on-policy, the current parameters determine the next data sample the parameters are trained on.
        - For example, if the maximizing action is to move left then the training samples will be dominated by samples from the left-hand side

    In practice, Deep Q-Learning only stores the last N experience tuples in the replay memory and samples uniformly at random from $D$ when performing updates.
    
    ### Experiments
    
    - Used Stochastic Gradient Descent with RMSProp with minibatches of size 32.
    - Epsilon was linearly annealled from 1 to 0.1 over the first million frames
    - Agent sees and selects actions on every kth frame instead of every frame and its last action is repeated on skipped frames
    
    ### Conclusion
    - Paper introduced new deep learning model for reinforcement learning
    - Used only raw pixles as input
    - Presented a variant of online Q-learning combining stochastic minibatch with experience replay
    """)