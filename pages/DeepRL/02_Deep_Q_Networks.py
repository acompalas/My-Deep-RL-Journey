import streamlit as st
import numpy as np
import random
import base64
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import tempfile, imageio
from utils.dqn import QNetwork, ReplayBuffer, compute_loss, update_target_network, agent_learn

st.title("Deep Q Networks")

section = st.selectbox(
    "",
    [
        "Demo",
        "My Notes",
        "Playing Atari with Deep Reinforcement Learning"
    ],
)

# ---------------------------
# Section: Demo
# ---------------------------
if section == "Demo":

    def render_episode_to_gif(env, agent=None, max_steps=500, fps=30):
        frames = []
        obs, _ = env.reset()
        done = False
        
        # Progress bar
        progress = st.progress(0)
        
        for t in range(max_steps):
            # choose action
            if agent:
                action = agent.act(obs)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(frame)
            
            done = terminated or truncated
            if done:
                break
            
            # update progress bar
            progress.progress((t + 1) / max_steps)
        
        # Clear the progress bar once done
        progress.empty()
        
        # Save GIF to temp file
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
            imageio.mimsave(tmpfile.name, frames, fps=fps)
            return tmpfile.name

    def display_gif(gif_path, width=400):
        with open(gif_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        st.markdown(
            f"<img src='data:image/gif;base64,{b64}' width='{width}' loop autoplay>",
            unsafe_allow_html=True
        )

    st.title("Demo: Deep Q-Learning with Experience Replay")

    st.header("Deep Q-Learning Algorithm")
    st.markdown(r"""
    This demo implements **Deep Q-Learning with Experience Replay**, inspired by  
    *Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)*.  
    """)

    st.image("assets/deep_q-learning.png", caption="Deep Q-Learning Algorithm", use_container_width=True)
    st.divider()
    
    # ---------------------------
    # Environment Explanation
    # ---------------------------
    st.header("The LunarLander Environment")
    st.markdown(r"""
    The [**LunarLander-v3**](https://gymnasium.farama.org/environments/box2d/lunar_lander/) task is provided by OpenAI Gym.  
    The agent controls a lunar lander and must safely land on the pad at $(0,0)$ between two flags.

    ### Action Space
    The agent has **4 discrete actions**:
    - `0`: Do nothing  
    - `1`: Fire right engine  
    - `2`: Fire main (downward) engine  
    - `3`: Fire left engine  

    ### Observation Space
    Each state is an **8-dimensional vector**:
    - $x, y$: position  
    - $\dot x, \dot y$: linear velocities  
    - $\theta$: angle (tilt of the lander)  
    - $\dot \theta$: angular velocity  
    - $l, r$: booleans (whether each leg touches the ground)  

    ### Rewards
    - Positive reward for moving closer and slower near the pad.  
    - +10 points for each leg in contact with the ground.  
    - -0.03 per step side engines fire, -0.3 per step main engine fires.  
    - +100 for a safe landing, -100 for crashing.  

    ### Termination
    - The lander crashes.  
    - The lander goes out of bounds ($|x| > 1$).  
    - The episode times out (max 1000 steps).  

    The environment is considered **solved** if the average reward ≥ 200 over 100 episodes.
    """)

    st.divider()
    # ---------------------------
    # Algorithm & Code Explanation
    # ---------------------------
    st.header("Code")
    with st.expander("Show/Hide Full Explanation"):

        # ---------------------------
        # 1. Q-Learning
        # ---------------------------
        st.markdown("### 1. Q-Learning")
        st.markdown(r"""
        Q-Learning is an **off-policy value-based** algorithm.  
        It updates state-action values according to the Bellman equation:

        $$
        Q(s_t, a_t) \leftarrow r_t + \gamma \max_{a'} Q(s_{t+1}, a')
        $$

        We can use a neural network for function approximation of Q-values
        
        $$
        Q(s, a, \theta)
        $$

        In Atari DQN:
        - CNN was used to process pixels.
        - RMSProp was used for optimization.

        In our LunarLander demo:
        - We use a simple **MLP** with two hidden layers (128 units, ReLU).  
        - We use **Adam optimizer** instead of RMSProp.
        """)

        st.code("""
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
        """, language="python")

        st.code("""
    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    update_target_network(q_net, target_net)  # initialize target as hard copy
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
        """, language="python")

        # ---------------------------
        # 2. Action Selection
        # ---------------------------
        st.markdown("### 2. Action Selection")
        st.markdown(r"""
        Actions are chosen **ε-greedily** by the behaviour Q-Network:
        
        $$
        a_t =
        \begin{cases}
        \text{random action} & \text{with prob. } \epsilon \\
        \arg\max_a Q(s_t, a; \theta) & \text{with prob. } (1-\epsilon)
        \end{cases}
        $$

        Epsilon is annealed linearly across episodes:

        $$
        \epsilon_{ep} = \epsilon_{\text{end}} + (\epsilon_{\text{start}} - \epsilon_{\text{end}}) \left(1 - \frac{ep}{N}\right)
        $$

        where $N$ = total episodes.
        """)

        st.code("""
    epsilon = eps_end + (eps_start - eps_end) * (1 - ep / episodes)

    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            q_vals = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = q_vals.argmax().item()
        """, language="python")

        # ---------------------------
        # 3. Experience Replay Buffer
        # ---------------------------
        st.markdown("### 3. Experience Replay Buffer")
        st.markdown(r"""
        We store experiences $(s, a, r, s', done)$ into a **replay buffer**.  
        Later, we randomly sample minibatches to **break correlation** between consecutive steps.
        """)

        st.code("""
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
        """, language="python")

        # ---------------------------
        # 4. Compute Loss
        # ---------------------------
        st.markdown("### 4. Compute Loss")
        st.markdown(r"""
        For each minibatch:
        1. Predict $Q(s,a; w)$ with the behavior network.  
        2. Compute target using the target network:

        $$
        y = r + \gamma (1 - done) \max_{a'} \hat{Q}(s', a'; w^-)
        $$

        3. Compute MSE loss:

        $$
        L = \frac{1}{B}\sum (y - Q(s,a; w))^2
        $$
        """)

        st.code("""
    def compute_loss(batch, q_net, target_net, gamma):
        states, actions, rewards, next_states, dones = batch

        q_pred = q_net(states).gather(1, actions)

        with torch.no_grad():
            q_next_max = target_net(next_states).max(1, keepdim=True)[0]
            y_target = rewards + gamma * (1 - dones) * q_next_max

        loss = nn.MSELoss()(q_pred, y_target)
        return loss
        """, language="python")

        # ---------------------------
        # 5. Update Gradients
        # ---------------------------
        st.markdown("### 5. Update Gradients")
        st.markdown(r"""
        - Update the **behavior network** by minimizing the loss.  
        - Update the **target network** using **soft updates** with parameter $\tau$:

        $$
        \theta^- \leftarrow \tau \theta + (1-\tau)\theta^-
        $$

        This makes the target values change slowly, improving stability.
        """)

        st.code("""
    loss = compute_loss(batch, q_net, target_net, gamma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_target_network(q_net, target_net, tau)
        """, language="python")

        st.code("""
    def update_target_network(q_net, target_net, tau=0.005):
        for target_param, param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        """, language="python")

    st.divider()
    st.header("Demo Pretrained Model")
    
    st.markdown(r"""
    Below you can demo a model I trained until an average reward of 250 for the last 100 episodes. The task is considered solved by OpenAI Gymnasium for a reward of 200.
    You can play a video of the episodes after simulating.         
    """)

    # User input for demo
    n_demo_eps = st.number_input("Number of Episodes to Simulate", min_value=1, max_value=20, value=5, step=1)
    demo_button = st.button("Run Demo", use_container_width=True)

    # Placeholders
    log_placeholder = st.empty()
    video_placeholder = st.empty()

    if demo_button:
        # Load pretrained model
        state_dim = 8  # LunarLander-v3 obs space
        action_dim = 4 # LunarLander-v3 action space
        q_net = QNetwork(state_dim, action_dim)
        q_net.load_state_dict(torch.load("models/dqn_lunarlander.pt", map_location="cpu"))
        q_net.eval()

        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        demo_frames = []
        log_history = []

        for ep in range(n_demo_eps):
            state, _ = env.reset()
            done, ep_ret = False, 0

            while not done:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = q_vals.argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                ep_ret += reward

                # Capture frame
                demo_frames.append(env.render())

            # Append log entry
            log_history.append(f"Episode {ep+1}/{n_demo_eps} | Return: {ep_ret:.1f}")
            log_placeholder.code("\n".join(log_history), language="")

        # Save as temp video
        
        gif_path = render_episode_to_gif(env)
        display_gif(gif_path)


    # # ---------------------------
    # # Training parameters
    # # ---------------------------
    # st.divider()
    # st.header("Training Parameters")
    # col1, col2, col3 = st.columns(3)
    # episodes = col1.number_input("Episodes", min_value=50, max_value=2000, value=1000, step=50)
    # batch_size = col2.number_input("Batch Size", min_value=32, max_value=512, value=64, step=32)
    # lr = col3.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")

    # col4, col5, col6 = st.columns(3)
    # gamma = col4.number_input("Discount γ", min_value=0.8, max_value=0.999, value=0.99, step=0.01)
    # tau = col5.number_input("Target Update τ", min_value=0.001, max_value=1.0, value=0.005, step=0.001, format="%.3f")
    # buffer_size = col6.number_input("Replay Buffer Size", min_value=10000, max_value=200000, value=50000, step=10000)

    # col7, col8, col9 = st.columns(3)
    # eps_start = col7.number_input("Epsilon Start", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    # eps_end = col8.number_input("Epsilon End", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    # update_freq = col9.number_input("Update Freq (steps)", min_value=1, max_value=20, value=4, step=1)

    # train_button = st.button("Train", use_container_width=True)

    # # ---------------------------
    # # Placeholders
    # # ---------------------------
    # plot_placeholder = st.empty()
    # log_placeholder = st.empty()
    # progress = st.progress(0)
    # first_ep_placeholder, last_ep_placeholder = st.columns(2)

    # # Default blank plot
    # fig, ax = plt.subplots()
    # ax.plot([], [])
    # ax.set_title("Average Reward (Smoothed)")
    # ax.set_xlabel("Episode")
    # ax.set_ylabel("Reward")
    # plot_placeholder.pyplot(fig, use_container_width=True)

    # # ---------------------------
    # # Environment + Networks
    # # ---------------------------
    # if train_button:
    #     env = gym.make("LunarLander-v3", render_mode="rgb_array")
    #     state_dim = env.observation_space.shape[0]
    #     action_dim = env.action_space.n

    #     q_net = QNetwork(state_dim, action_dim)
    #     target_net = QNetwork(state_dim, action_dim)
    #     update_target_network(q_net, target_net)  # init hard copy
    #     optimizer = optim.Adam(q_net.parameters(), lr=lr)

    #     replay_buffer = ReplayBuffer(buffer_size)

    #     # ---------------------------
    #     # Training Loop
    #     # ---------------------------
    #     returns, avg_returns, log_history = [], [], deque(maxlen=10)
    #     first_ep_frames, last_ep_frames = None, None
    #     step_count = 0
    #     epsilon = eps_start

    #     for ep in range(episodes):
    #         # epsilon = eps_end + (eps_start - eps_end) * (1 - ep / episodes)
    #         epsilon = max(eps_end, 0.995 * epsilon)
    #         state, _ = env.reset()
    #         done, ep_ret = False, 0
    #         frames = []

    #         while not done:
    #             step_count += 1

    #             # ε-greedy action
    #             if np.random.rand() < epsilon:
    #                 action = env.action_space.sample()
    #             else:
    #                 with torch.no_grad():
    #                     q_vals = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    #                     action = q_vals.argmax().item()

    #             next_state, reward, terminated, truncated, _ = env.step(action)
    #             done = terminated or truncated
    #             replay_buffer.push(state, action, reward, next_state, done)

    #             state = next_state
    #             ep_ret += reward

    #             if ep == 0 or ep == episodes - 1:
    #                 frames.append(env.render())

    #             # ✅ Training step every update_freq
    #             if step_count % update_freq == 0 and len(replay_buffer) >= batch_size:
    #                 batch = replay_buffer.sample(batch_size)
    #                 agent_learn(batch, q_net, target_net, optimizer, gamma, tau)

    #         # Logging
    #         returns.append(ep_ret)
    #         avg_return = np.mean(returns[-100:])  # ✅ smooth over 100
    #         avg_returns.append(avg_return)
    #         solved_flag = " ✅ Solved!" if avg_return >= 200 else ""
    #         log_history.append(
    #             f"Episode {ep+1}/{episodes} | Return: {ep_ret:.1f} | "
    #             f"Avg100: {avg_return:.1f} | Epsilon {epsilon:.2f}{solved_flag}"
    #         )

    #         fig, ax = plt.subplots()
    #         ax.plot(avg_returns, label="Smoothed Return (100-ep)")
    #         ax.axhline(200, color="red", linestyle="--", label="Solved Threshold (200)")
    #         ax.legend()
    #         plot_placeholder.pyplot(fig, use_container_width=True)
    #         log_placeholder.code("\n".join(log_history), language="")
    #         progress.progress((ep + 1) / episodes)

    #         if ep == 0: first_ep_frames = frames
    #         if ep == episodes - 1: last_ep_frames = frames

    #     # Save videos
    #     def save_video(frames, label, placeholder):
    #         if frames:
    #             tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    #             imageio.mimsave(tmpfile.name, frames, fps=30)
    #             placeholder.subheader(label)
    #             placeholder.video(tmpfile.name)

    #     save_video(first_ep_frames, "First Episode", first_ep_placeholder)
    #     save_video(last_ep_frames, "Last Episode", last_ep_placeholder)

    
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