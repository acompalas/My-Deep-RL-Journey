import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym
import random


st.title("Vanilla Policy Gradient")

section = st.selectbox(
    "",
    [
        "Demo",
        "My Notes"
    ],
)

if section == "My Notes":
    st.title("My Notes")
    
    st.markdown(r"""
    ### Policy Gradient Methods for Reinforcement Learning with Function Approximation (Sutton et al.)
    
    ---
    
    #### Issues with Function Approximation for Value-Based Methods
    **Value-Based** methods dominated reinforcement learning before policy gradient methods gained traction
    
    **Function Approximation** was used to estimate the value function in large or continuous state spaces, where lookup tables were infeasible.
    
    However, value-based methods can fail to converge even in simple MDPs with function approximation.
    
    **Examples** of value based methods include the on-policy and off-policy TD(0) algorithms:
    - **SARSA** 
    - **Q-Learning**.
    
    While value based methods worked well in many applications, they had several limitations
    - **Deterministic vs. Stochastic Policies**: Action-selection policies were the deterministic "$\epsilon$-greedy" policy with respect to the estimated values
    while the optimal policy may be stochastic.
    $$
    \pi(s) = \textbf{argmax}_a Q(s, a)
    $$
    - **Discontinuity**: small errors in approximation → huge changes in behavior.
    
    ---
    
    #### Policy Gradients
    
    Sutton’s paper explored policy gradient methods as an alternative to value-function-based approaches with function approximation.
    
    Instead of approximating a value function and use that to compute a deterministic policy, policy gradient methods seek ot approximate a stochastic policy directly using its own parameters.
    
    - $\theta \in \mathbb{R}^n$: vector of policy parameters (e.g., weights of a neural network).  
    - $\rho \in \mathbb{R}$: scalar performance measure of the policy.  

    Two common formulations of $\rho$:  

    1. **Average-reward formulation** (long-run average reward per step):  
    $$
    \rho(\pi_\theta) = \lim_{n \to \infty} \frac{1}{n} \, \mathbb{E}_\pi \Big[\sum_{t=1}^n r_t \Big] 
    \quad 
    Q^\pi(s,a) = \mathbb{E}_\pi \Big[ \sum_{t=1}^\infty (r_t - \rho(\pi_\theta)) \,\Big|\, s_0=s, a_0=a \Big]
    $$  

    2. **Start-state discounted formulation** (expected discounted return from $s_0$):  
    $$
    \rho(\pi_\theta) = \mathbb{E}_\pi \Big[\sum_{t=0}^\infty \gamma^t r_t \,\big|\, s_0 \Big], \quad \gamma \in [0,1) 
    \quad 
    Q^\pi(s,a) = \mathbb{E}_\pi \Big[ \sum_{k=1}^\infty \gamma^{k-1} r_{t+k} \,\Big|\, s_t=s, a_t=a \Big]
    $$  

    In both cases, the objective is to adjust $\theta$ to maximize $\rho(\pi_\theta)$. 
    
    - The **policy gradient** is the partial derivative of the scalar performance $\rho(\pi_\theta)$ with respect to the parameter vector $\theta$:  
    $$
    \nabla_\theta \rho(\pi_\theta) = \frac{\partial \rho(\pi_\theta)}{\partial \theta}
    $$  

    - The parameters $\theta$ are updated incrementally toward their optimum using **gradient ascent**:  
    $$
    \theta \;\leftarrow\; \theta + \alpha \, \nabla_\theta \rho(\pi_\theta)
    $$  

    where $\alpha > 0$ is the learning rate (step size). 
    
    --- 
    
    #### Main Points
    
    **Sutton’s contribution** was to show that William's 1992 REINFORCE algorithm when used with a learned baseline (actor–critic) can be made stable and provably convergent.

    **Theorem 1 (Policy Gradient Theorem):**  
    The performance gradient depends only on the policy and $Q^\pi(s,a)$, not on state distribution derivatives.  
    - Doing this with sampled returns leads to **Williams’ 1992 REINFORCE algorithm**.
    - William's REINFORCE algorithm has been proven to yield an unbiased estimate of $\frac{\partial \rho}{\partial \theta}$, however it suffered from slow learning and high variance. 
    - Sutton extended this with baselines (actor–critic) by learning a value function and using it to reduce variance.
    
    $$
    \nabla_{\theta} \rho \;=\;  \sum_s d^\pi(s) \sum_a \nabla_{\theta}\pi_\theta(s, a)\,Q^\pi(s,a)
    $$
    
    where $d^\pi(s)$ is the stationary (or discounted) state distribution which weights how often states are visited under $\pi$.  

    ---

    **Theorem 2 (Policy Gradient with Function Approximation):**  
    If the value function approximation $f_w(s,a)$ is *compatible* with the policy parameterization, then using $f_w$ in place of $Q^\pi$ gives the exact gradient.  

    - Sutton justifies the use of a learned value function to reduce the variance of the gradient and yield the optimal parameters.  
    - At convergence, the learned function yields the optimal solution.  

    $$
    \sum_s d^\pi(s) \sum_a \pi_\theta(s, a)\,\big(Q^\pi(s,a) - f_w(s,a)\big)\,\nabla_w f_w(s,a) \;=\; 0
    $$

    - **Compatibility condition:**  
    $$
    \nabla_w f_w(s,a) \;=\; \nabla_\theta \pi(s, a) \frac{1}{\pi(s, a)}
    $$

    - Substituting gives:  
    $$
    \sum_s d^\pi(s) \sum_a \,\big(Q^\pi(s,a) - f_w(s,a)\big)\, \nabla_\theta \pi(s, a) \;=\; 0
    $$

    because the equation above is zero, we can subtract it from the policy gradient which yields
    
    $$
    \nabla_\theta \rho(\pi_\theta) \;=\; \sum_s d^\pi(s) \sum_a \nabla_\theta \pi(s, a)\, f_w(s,a)
    $$

    ---

    **Theorem 3 (Convergence of Policy Iteration with Function Approximation):**  
    With compatible policy and value function approximators, bounded second derivatives, and proper step sizes, policy iteration converges to a **locally optimal policy**.  
    
    Then for any MDP with bounded rewards, actor-critic methods using function approximation and a policy gradient converges to an optimum with gradient ascent
    
    $$
    w_k = w \, \text{such that}\, 
    \sum_s d^\pi(s) \sum_a \pi_\theta(s, a)\,\big(Q^\pi(s,a) - f_w(s,a)\big)\,\nabla_w f_w(s,a) \;=\; 0
    $$
    
    
    $$
    \theta_{k+1} = \theta_k + \alpha_k 
    \sum_s d^{\pi_k}(s) \sum_a \nabla_\theta \pi_{\theta_k}(s,a) \, f_{w_k}(s,a)
    $$
    
    $$
    \text{lim}_{k \rightarrow \infty} \frac{\partial \rho}{\partial \theta} = 0
    $$

    As our value function approximation $f_w(s, a)$ approches the true value function $Q(s, a)$, the policy updates approach the optimum through gradient ascent updates.
    
    ---
    
    ### Reinforcement Learning (Sutton and Barto): Chapter 13  
    
    #### REINFORCE (Sutton & Barto, Chapter 13.3)

    **From the Policy Gradient Theorem (13.6):**  
    The gradient of the performance objective can be expressed as  

    $$
    \nabla J(\theta) \;\propto\; 
    \sum_s \mu(s) \sum_a q^\pi(s,a) \, \nabla_\theta \pi_\theta(a|s)
    \;=\; 
    \mathbb{E}_\pi \Bigg[ \sum_a q^\pi(S_t,a)\, \nabla_\theta \pi_\theta(a|S_t) \Bigg]
    $$  

    where $\mu(s)$ is the on-policy state distribution under $\pi$.  

    ---

    **All-Actions Update (13.7):**  
    If we sample a state $S_t$, we could update using *all actions*:  

    $$
    \theta_{t+1} \doteq \theta_t 
    + \alpha \sum_a \hat{q}(S_t,a,w)\, \nabla_\theta \pi_\theta(a|S_t)
    $$  

    where $\hat{q}$ is a learned approximation of $q^\pi$.  

    ---

    **From All-Actions to One-Action (REINFORCE):**  
    Instead of summing over actions, introduce the action actually taken, $A_t \sim \pi(\cdot|S_t)$:  

    $$
    \nabla J(\theta) \;\propto\;
    \mathbb{E}_\pi \left[ 
    \frac{q^\pi(S_t, A_t)\,\nabla_\theta \pi_\theta(A_t|S_t)}{\pi_\theta(A_t|S_t)}
    \right]
    $$  

    Since $\mathbb{E}_\pi[G_t \mid S_t, A_t] = q^\pi(S_t,A_t)$, we can replace $q^\pi(S_t,A_t)$ with the **sampled return $G_t$**:  

    $$
    \nabla J(\theta) \;\propto\;
    \mathbb{E}_\pi \left[ 
    G_t \, \frac{\nabla_\theta \pi_\theta(A_t|S_t)}{\pi_\theta(A_t|S_t)}
    \right]
    $$  
    
    ---

    **REINFORCE Update (Williams, 1992):**  
    This gives the classic Monte Carlo policy gradient update:  

    $$
    \theta_{t+1} \doteq \theta_t 
    + \alpha \, G_t \, \frac{\nabla_\theta \pi_\theta(A_t|S_t)}{\pi_\theta(A_t|S_t)}
    $$
    
    **Log-Derivative Trick**

    From calculus:  
    $$
    \nabla_\theta \ln f(x) \;=\; \frac{\nabla_\theta f(x)}{f(x)}
    $$  

    **Apply to the policy:**  
    $$
    \frac{\nabla_\theta \pi_\theta(A_t|S_t)}{\pi_\theta(A_t|S_t)} 
    \;=\; \nabla_\theta \ln \pi_\theta(A_t|S_t)
    $$  

    **Substitute into the REINFORCE update:**  
    $$
    \theta_{t+1} 
    \doteq \theta_t 
    + \alpha \, G_t \, \nabla_\theta \ln \pi_\theta(A_t|S_t)
    $$  

    This is the classic **REINFORCE gradient ascent update** written in log form.  
    
    ---
    
    **REINFORCE Algorithm Pseudocode**
    
    Below we have pseudocode for the REINFORCE algorithm.
    - Since REINFORCE is a Monte Carlo algorithm, we first generate the whole episode before updating parameters
    - Update the parameters using gradient ascent
    - Repeat for several episodes until convergence
        
    """)
    
    st.image("assets/reinforce.png", caption="The REINFORCE Algorithm")

    st.markdown(r"""
       
    ### REINFORCE with Baseline (Sutton & Barto, Chapter 13.4)

    Recall in his 1999 paper, Sutton justified that using a **function approximation of the value function** preserves the gradient direction toward convergence to a local optimum.  

    ---

    **Policy Gradient Theorem with Baseline (13.10):**  
    We can subtract any baseline $b(s)$ that does not depend on the action:  

    $$
    \nabla J(\theta) \;\propto\; \sum_s \mu(s) \sum_a \big(q^\pi(s,a) - b(s)\big)\, \nabla_\theta \pi_\theta(a|s)
    $$  

    This works because  
    $$
    \sum_a b(s)\,\nabla_\theta \pi_\theta(a|s) = b(s)\,\nabla_\theta \sum_a \pi_\theta(a|s) = b(s)\,\nabla_\theta 1 = 0.
    $$  

    ---

    **REINFORCE with Baseline Update (13.11):**  
    Following the same steps as vanilla REINFORCE, we obtain the Monte Carlo policy gradient update:  

    $$
    \theta_{t+1} \doteq \theta_t 
    + \alpha \, \big(G_t - b(S_t)\big) \, \frac{\nabla_\theta \pi_\theta(A_t|S_t)}{\pi_\theta(A_t|S_t)}
    $$  

    Using the log-derivative trick:  

    $$
    \theta_{t+1} \doteq \theta_t 
    + \alpha \, \big(G_t - b(S_t)\big) \, \nabla_\theta \ln \pi_\theta(A_t|S_t)
    $$  

    ---

    **Key insight:**  
    - The baseline leaves the **expected gradient unchanged** but can **reduce variance**.  
    - A natural choice is the **state-value function** $\hat{v}(S_t,w)$, learned with its own update rule.  
    - This turns REINFORCE into an **actor–critic architecture**, with the actor updating $\theta$ and the critic updating $w$.    
                
    """)
    
    st.image("assets/reinforce_baseline.png", caption="REINFORCE with Baseline Algorithm")
         
    st.markdown(r"""
    #### Actor–Critic Methods (Sutton & Barto, Chapter 13.5)

    - **Monte Carlo REINFORCE** uses the *full return* $G_t$ to update the policy, meaning updates happen only at the *end of an episode*.  
    - By introducing a learned value function as a baseline, we can move from Monte Carlo updates to **online, step-by-step updates**.  

    ---

    **One-step Actor–Critic (TD(0)-style):**  

    Policy update (actor):  
    $$
    \theta_{t+1} = \theta_t + \alpha \, \delta_t \, \nabla_\theta \ln \pi_\theta(A_t|S_t)
    $$  

    Critic update (state-value function):  
    $$
    w_{t+1} = w_t + \alpha_w \, \delta_t \, \nabla_w \hat{v}(S_t, w)
    $$  

    where the **TD error** is:  
    $$
    \delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1},w) - \hat{v}(S_t,w)
    $$  

    - The update now uses the *immediate TD error* instead of waiting for the full return $G_t$.  
    - This makes the algorithm **incremental and online**.  

    ---

    **Actor–Critic with Eligibility Traces (TD(λ)-style):**  

    - Generalizes the one-step method using eligibility traces for both actor and critic.  
    - Interpolates between **Monte Carlo REINFORCE ($\lambda=1$)** and **TD(0) actor–critic ($\lambda=0$)**.  
    - Provides a flexible bias–variance trade-off.  

    ---

    **Main Point:**  
    Actor–critic methods extend REINFORCE by replacing full-episode Monte Carlo returns with **online TD-style updates**, enabling incremental learning and faster adaptation.         

    """)

if section == "Demo":
    st.title("Vanilla Policy Gradient (Monte Carlo REINFORCE with Baseline)")
    
    st.markdown(r"""
    This demo implements the **Monte Carlo REINFORCE algorithm** with a learned **value function baseline**, one of the simplest policy gradient methods in reinforcement learning.  

    We train an agent on the **CartPole-v1 environment** from [Gymnasium](https://gymnasium.farama.org/):  

    - **Goal:** Keep the pole balanced upright by moving the cart left or right.  
    - **Action space:** Two discrete actions (`0 = left`, `1 = right`).  
    - **State space:** A 4-dimensional vector:  
    1. Cart position (x on the track)  
    2. Cart velocity  
    3. Pole angle  
    4. Pole angular velocity  
    - **Reward scheme:** The agent receives +1 reward for every timestep the pole stays upright. Episodes end if the pole falls beyond a threshold angle or the cart moves too far from the center.  

    We use **policy gradients** to update the policy network and a baseline value network to reduce variance in gradient estimates. Over time, the agent learns to keep the pole balanced for longer episodes.

    """)
    st.divider()
    st.header("CartPole-v1 Demo")

    # ---------------------------
    # Training parameters UI
    # ---------------------------
    col1, col2, col3 = st.columns(3)
    episodes = col1.number_input("Episodes", min_value=50, max_value=2000, value=200, step=50)
    lr = col2.number_input("Learning Rate", min_value=1e-4, max_value=1e-1, value=1e-2, format="%.4f")
    save_n = col3.number_input("Episodes to Save (video)", min_value=1, max_value=50, value=10, step=1)

    train_button = st.button("Train", use_container_width=True)
    
    # ---------------------------
    # Placeholders
    # ---------------------------
    plot_placeholder = st.empty()          # reward plot
    log_placeholder = st.empty()           # rolling log of episodes
    progress = st.progress(0)              # progress bar
    video_placeholder = st.empty()         # final training video
    
    # Default blank plot
    fig, ax = plt.subplots()
    ax.plot([], [])  # dummy line so Streamlit renders
    ax.set_title("Average Reward (Smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    plot_placeholder.pyplot(fig, use_container_width=True)
    
    # ---------------------------
    # Set seeds for reproducibility
    # ---------------------------
    SEED = 123
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # ---------------------------
    # Environment and Networks
    # ---------------------------
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset(seed=SEED)

    policy_net = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    value_net = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=lr)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=lr)

    # ---------------------------
    # Helpers
    # ---------------------------
    def run_episode(record=False):
        states, actions, rewards, log_probs = [], [], [], []
        frames = []

        obs, info = env.reset()
        done = False
        while not done:
            if record:
                frame = env.render()
                frame = np.array(frame)
                if frame.ndim == 4:
                    frame = frame[0]
                frames.append(frame)

            s = torch.as_tensor(obs, dtype=torch.float32)
            logits = policy_net(s)
            dist = Categorical(logits=logits)
            a = dist.sample()

            obs, r, terminated, truncated, info = env.step(a.item())
            done = terminated or truncated

            states.append(s)
            actions.append(a)
            rewards.append(r)
            log_probs.append(dist.log_prob(a))

        return states, actions, rewards, log_probs, frames

    def compute_returns(rewards, gamma=0.99):
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    # ---------------------------
    # Training loop
    # ---------------------------
    if train_button:
        avg_rewards = []
        window = deque(maxlen=50)
        log_history = deque(maxlen=10)
        all_frames = []

        for episode in range(episodes):
            record_video = (episode >= episodes - save_n)
            states, actions, rewards, log_probs, frames = run_episode(record=record_video)

            if record_video:
                all_frames.extend(frames)

            returns = compute_returns(rewards)
            states_tensor = torch.stack(states)
            values = value_net(states_tensor).squeeze()
            advantages = returns - values.detach()

            # Policy update
            policy_loss = -(torch.stack(log_probs) * advantages).mean()
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            # Value update
            value_loss = nn.MSELoss()(values, returns)
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

            # Logging
            ep_ret = sum(rewards)
            window.append(ep_ret)
            avg_rewards.append(np.mean(window))

            # Update plot
            fig, ax = plt.subplots()
            ax.plot(avg_rewards, label="Smoothed Return (50-ep)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.legend()
            plot_placeholder.pyplot(fig, use_container_width=True)

            # Update log
            log_history.append(f"Episode {episode+1}/{episodes} | Return: {ep_ret:.1f}")
            log_placeholder.code("\n".join(log_history), language="")

            progress.progress((episode + 1) / episodes)

        # ---------------------------
        # After training → show video
        # ---------------------------
        import tempfile, imageio

        if all_frames:
            try:
                # Try saving as mp4 (requires ffmpeg)
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                imageio.mimsave(tmpfile.name, all_frames, fps=30)
                video_placeholder.subheader(f"Training Video (last {save_n} episodes)")
                video_placeholder.video(tmpfile.name)
            except Exception as e:
                # Fallback to GIF
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
                imageio.mimsave(tmpfile.name, all_frames, fps=30)
                video_placeholder.subheader(f"Training Video (last {save_n} episodes, GIF fallback)")
                video_placeholder.image(tmpfile.name)

    st.divider()
    st.header("OpenAI Pseudocode")
    st.image("assets/vpg_pseudo.svg", caption="REINFORCE Algorithm", use_container_width=True)
    
    st.divider()
    st.subheader("Policy Gradient with Baseline")
    st.markdown(r"""
    The policy gradient update is:

    $$
    \nabla_\theta J(\pi_\theta) =
    \mathbb{E}\Bigg[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \; (G_t - V_\phi(s_t)) \Bigg]
    $$

    - $\pi_\theta$: policy network (actor)  
    - $V_\phi$: value network (critic, used as baseline)  
    - $G_t$: Monte Carlo return (reward-to-go)  
    - $(G_t - V_\phi(s_t))$: advantage estimate  
    """)

    st.divider()
    st.subheader("Code")

    st.markdown("### 1. Policy and Value Networks (actor–critic)")
    st.code("""
    policy_net = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 2),   # 2 actions: left or right
    )

    value_net = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 1),   # scalar value baseline
    )
    """, language="python")

    st.markdown("""
    - The **policy network** outputs logits for a `Categorical` distribution over actions.  
    - The **value network** predicts $V(s)$, trained via regression to Monte Carlo returns.  
    """)

    st.markdown("### 2. Sampling an Episode")
    st.code("""
    def run_episode(record=False):
        states, actions, rewards, log_probs = [], [], [], []
        obs, info = env.reset()
        done = False
        while not done:
            logits = policy_net(torch.as_tensor(obs, dtype=torch.float32))
            dist = Categorical(logits=logits)
            a = dist.sample()

            obs, r, terminated, truncated, info = env.step(a.item())
            done = terminated or truncated

            states.append(torch.as_tensor(obs, dtype=torch.float32))
            actions.append(a)
            rewards.append(r)
            log_probs.append(dist.log_prob(a))
        return states, actions, rewards, log_probs
    """, language="python")

    st.markdown("""
    - At each step, we sample from the policy distribution.  
    - We store `(state, action, reward, log_prob)` until termination.  
    """)

    st.markdown("### 3. Computing Reward-to-Go")
    st.code("""
    def compute_returns(rewards, gamma=0.99):
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)
    """, language="python")

    st.markdown("""
    This corresponds to:

    $$
    G_t = \sum_{k=t}^T \gamma^{k-t} r_k
    $$
    """)

    st.markdown("### 4. Policy and Value Losses")
    st.code("""
    values = value_net(states_tensor).squeeze()
    advantages = returns - values.detach()

    # Policy gradient loss (using the log trick)
    policy_loss = -(torch.stack(log_probs) * advantages).mean()

    # Value function regression loss
    value_loss = nn.MSELoss()(values, returns)
    """, language="python")

    st.markdown(r"""
    - First we compute the **advantage function**:
    
    $$
    A_t = G_t - V(s_t)
    $$

    This measures how much better or worse the observed return $G_t$ was compared to the baseline estimate $V(s_t)$.

    - **Policy loss**:  
    We use the **log trick** to represent the policy gradient:

    $$
    \nabla_\theta J(\pi_\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \, A_t
    $$

    In code, this is implemented by multiplying the saved log-probabilities of each action (`log_probs`) with their corresponding advantages.  
    The negative sign (`-`) ensures that PyTorch’s **gradient descent** actually performs **gradient ascent** on $J(\pi_\theta)$.

    - **Value loss**:  
    The critic (value network) is trained as a regression problem, minimizing mean squared error (MSE):

    $$
    L_V = \big(V(s_t) - G_t\big)^2
    $$

    This makes $V(s_t)$ approximate the true returns so it can serve as a good baseline in future updates.
    """)


    st.markdown("### 5. Gradient Updates")
    st.code("""
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()
    """, language="python")

    st.markdown(r"""
    - PyTorch’s autograd automatically computes the gradients of both losses w.r.t. their parameters.  
    - The **policy network** (parameters $\theta$) is updated with **gradient ascent** on the expected return:  

    $$
    \theta \;\leftarrow\; \theta + \alpha_\pi \, \nabla_\theta J(\pi_\theta)
    $$

    - The **value network** (parameters $w$) is updated with **gradient descent** on the MSE loss:  

    $$
    w \;\leftarrow\; w - \alpha_V \, \nabla_w \; \big(V_w(s_t) - G_t\big)^2
    $$

    - In code:  
    - `policy_loss.backward()` propagates $\nabla_\theta$ through the policy network.  
    - `policy_optim.step()` applies ascent (the negative sign inside the loss makes descent → ascent).  
    - `value_loss.backward()` propagates $\nabla_w$ through the value network.  
    - `value_optim.step()` applies standard descent.
    """)

    st.divider()
    st.subheader("Training Summary")

    st.markdown("""
    - Each episode generates a trajectory.  
    - Rewards are discounted into returns.  
    - Policy and value networks are updated with stochastic gradient methods.  
    - We log rolling average reward and save video of the last episodes.  
    """)
    

# if section == "Policy Gradient Methods (Sutton et al., 2000) Notes":
#     st.title("Policy Gradient Methods (Sutton et al., 2000) Notes")
    
#     st.markdown(r"""
#     These notes are from the research paper "Policy Gradient Methods for Reinforcement Learning with Function Approximation (Sutton et al., 2000)"
    
#     #### Abstract
    
#     - Value-based methods with function approximation lose stability and tractability as state–action spaces scale.
#     - Policy gradients directly parameterize the policy instead of deriving it from Q-values.
#     - REINFORCE and actor–critic are key examples of policy gradient methods.
#     - The policy gradient can be estimated from experience using $Q$ or advantage functions.
#     - Policy iteration with differentiable function approximation converges to a locally optimal policy.
    
#     #### Introduction
    
#     - Large RL problems need function approximation (NNs, trees, instance-based methods).
#     - Value-based methods approximate $V$ or $Q$ and derive policies greedily.
#     - Value-based methods are unstable:
#         - They push toward deterministic policies, though optimal ones may be stochastic.
#         - Tiny value changes can flip greedy action choice, causing discontinuities and divergence.
#         - Proven non-convergence even with simple MDPs and approximators (e.g., Q-learning, SARSA).
        
#     "In this paper we explore an alternative approach to function approximation in RL."
        
#     - Policy gradients approximate the policy directly as a stochastic function (e.g., NN outputting action probabilities).
#     - Policy parameters $\theta$ are updated via gradient ascent on performance $\rho$:
#     $$
#     \Delta \theta \approx \alpha \frac{\partial \rho}{\partial \theta}
#     $$
    
#     "In this paper we prove that an unbiased estimate of the gradient can be obtained from experience using an approximate value function satisfying certain properties."
    
#     "REINFORCE algorithm finds an unbiased estimate of the gradient, but without the assistance of a learned value function."
#     - REINFORCE estimates gradients without value functions but is high variance and slow.
    
#     "Our result also suggests a way of proving the convergence of a wide variety of algorithms based on “actor-critic” or policy-iteration architectures"
#     - Actor–critic uses value functions to reduce variance, enabling faster learning.
    
#     "In this paper we take the first step in this direction by proving for the first time that a version of policy iteration with general diﬀerentiable function approximation is convergent to a locally optimal policy."
#     - Main result: policy iteration with general differentiable function approximation converges to a locally optimal policy.
    
#     #### Policy Gradient Theorem

#     - Setup: agent interacts with an MDP with states $s$, actions $a$, rewards $r$, and transition probabilities $P^a_{ss'}$.  
#     - Policy $\pi(a|s;\theta)$ is differentiable and parameterized by vector $\theta \in \mathbb{R}^l$, with $l \ll |S|$.  
#     - Two formulations of the performance objective:  
#     - **Average reward:** $\rho(\pi)$ = long-term expected reward per step.  
#     - **Start-state (discounted):** $\rho(\pi)$ = expected discounted return from $s_0$.  
#     - Define $Q^\pi(s,a)$ accordingly under each formulation.  
#     - **Policy Gradient Theorem:**  
#     $$
#     \nabla_\theta \rho(\pi) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi(a|s;\theta)\, Q^\pi(s,a)
#     $$
#     - Key insight: the gradient has **no term $\nabla_\theta d^\pi(s)$** → state distribution shift need not be differentiated.  
#     - This makes gradient estimation feasible from sampling trajectories under $\pi$.  
#     - **Unbiased estimate:** use actual returns $R_t$ to approximate $Q^\pi(s_t,a_t)$.  
#     - Leads to **REINFORCE update:**  
#     $$
#     \Delta \theta_t \propto \nabla_\theta \log \pi(a_t|s_t;\theta)\, R_t
#     $$
#     - The $\frac{1}{\pi(a_t|s_t)}$ term corrects oversampling of actions favored by $\pi$.  

#     """)
    
# if section == "Reinforcement Learning (Sutton and Barto): Chapter 13":
    # st.title("Reinforcement Learning (Sutton and Barto): Chapter 13")