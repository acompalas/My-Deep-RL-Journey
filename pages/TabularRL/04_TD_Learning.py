import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.td_learning import GridWorldEnv, MonteCarloAgent, SARSAAgent, QLearningAgent, render_gridworld, render_policy
import inspect

def run_experiment(title, rows, cols, start, goal, obstacles, steps):
    st.header(title)

    # --- Environments ---
    env_mc = GridWorldEnv(rows=rows, cols=cols, start=start, goal=goal, obstacles=obstacles)
    env_sarsa = GridWorldEnv(rows=rows, cols=cols, start=start, goal=goal, obstacles=obstacles)
    env_q = GridWorldEnv(rows=rows, cols=cols, start=start, goal=goal, obstacles=obstacles)

    # --- Training controls ---
    st.subheader("Training Controls")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        episodes = st.number_input("Episodes", min_value=100, max_value=5000,
                                  value=500, step=100, key=f"{title}_episodes")
        alpha = st.number_input("α (Learning Rate)", min_value=0.01, max_value=1.0,
                                value=0.1, step=0.01, format="%.2f", key=f"{title}_alpha")
    with col_in2:
        gamma = st.number_input("γ (Discount Factor)", min_value=0.1, max_value=1.0,
                                value=0.99, step=0.01, format="%.2f", key=f"{title}_gamma")
        epsilon = st.number_input("ε (Initial Exploration Rate)", 0.01, 1.0, 0.1,
                          0.01, format="%.2f", key=f"{title}_epsilon")

        current_eps_placeholder = st.empty()

    train_button = st.button("Train Agents", use_container_width=True, key=f"{title}_train")
    # # --- Stats placeholders ---
    # stat_col1, stat_col2, stat_col3 = st.columns(3)
    # stat_mc = stat_col1.empty()
    # stat_sarsa = stat_col2.empty()
    # stat_q = stat_col3.empty()

    # --- Layout for grids and reward chart ---
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    grid_mc_placeholder = top_left.empty()
    grid_sarsa_placeholder = top_right.empty()
    grid_q_placeholder = bottom_left.empty()
    reward_chart_placeholder = bottom_right.empty()

    # --- Render empty visuals before training ---
    grid_mc_placeholder.pyplot(render_gridworld(env_mc, path=None, color="blue"))
    grid_sarsa_placeholder.pyplot(render_gridworld(env_sarsa, path=None, color="green"))
    grid_q_placeholder.pyplot(render_gridworld(env_q, path=None, color="red"))

    fig_empty, ax_empty = plt.subplots()
    ax_empty.set_title("Episode Rewards")
    ax_empty.set_xlabel("Episode")
    ax_empty.set_ylabel("Total Reward")
    ax_empty.plot([], [], label="Monte Carlo", color="blue")
    ax_empty.plot([], [], label="SARSA", color="green")
    ax_empty.plot([], [], label="Q-Learning", color="red")
    ax_empty.legend()
    reward_chart_placeholder.pyplot(fig_empty)

    if train_button:
        mc_agent = MonteCarloAgent(env_mc, alpha=alpha, gamma=gamma, epsilon=epsilon)
        sarsa_agent = SARSAAgent(env_sarsa, alpha=alpha, gamma=gamma, epsilon=epsilon)
        q_agent = QLearningAgent(env_q, alpha=alpha, gamma=gamma, epsilon=epsilon)

        mc_rewards, sarsa_rewards, q_rewards = [], [], []
        mc_avg, sarsa_avg, q_avg = [], [], []
        avg_mc, avg_sarsa, avg_q = 0, 0, 0

        for ep in range(1, episodes + 1):
            current_eps_placeholder.markdown(
            f"**Current ε** → SARSA: {sarsa_agent.epsilon:.3f}, Q-Learning: {q_agent.epsilon:.3f}"
            )

            path_mc, R_mc = mc_agent.run_episode(max_steps=steps)
            path_sarsa, R_sarsa = sarsa_agent.run_episode(max_steps=steps)
            path_q, R_q = q_agent.run_episode(max_steps=steps)

            mc_rewards.append(R_mc)
            sarsa_rewards.append(R_sarsa)
            q_rewards.append(R_q)

            avg_mc += (R_mc - avg_mc) / ep
            avg_sarsa += (R_sarsa - avg_sarsa) / ep
            avg_q += (R_q - avg_q) / ep

            mc_avg.append(avg_mc)
            sarsa_avg.append(avg_sarsa)
            q_avg.append(avg_q)

            if ep % 10 == 0 or ep == episodes:


                grid_mc_placeholder.pyplot(render_gridworld(env_mc, path=path_mc, color="blue"))
                grid_sarsa_placeholder.pyplot(render_gridworld(env_sarsa, path=path_sarsa, color="green"))
                grid_q_placeholder.pyplot(render_gridworld(env_q, path=path_q, color="red"))  
                      
                # Reward chart
                fig_r, ax = plt.subplots()
                ax.plot(mc_avg, label="Monte Carlo (avg)", color="blue")
                ax.plot(sarsa_avg, label="SARSA (avg)", color="green")
                ax.plot(q_avg, label="Q-Learning (avg)", color="red")
                ax.set_title("Average Episode Rewards")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Average Total Reward")
                ax.legend()
                reward_chart_placeholder.pyplot(fig_r)

        # After training: render policies
        grid_mc_placeholder.pyplot(render_policy(env_mc, mc_agent.Q, color="blue"))
        grid_sarsa_placeholder.pyplot(render_policy(env_sarsa, sarsa_agent.Q, color="green"))
        grid_q_placeholder.pyplot(render_policy(env_q, q_agent.Q, color="red"))

st.title("Temporal Difference Learning")

section = st.selectbox(
    "",
    [
        "Demo: MC vs SARSA vs Q-Learning",
        "Review",
        "Temporal Difference Learning",
        "TD Learning for Control",
        "Summary",
    ],
)

if section == "Demo: MC vs SARSA vs Q-Learning":
    st.title("Demo: MC vs SARSA vs Q-Learning")
    
    st.markdown(r"""
    Here I am recreating a GridWorld demo that I saw on YouTube from Gonkee:  
    *"The FASTEST introduction to Reinforcement Learning on the Internet"*  
    demonstrating the effectiveness of SARSA and Q-Learning over Monte Carlo.        
    """)

    algo_choice = st.radio("Choose an algorithm to review:", ["Monte Carlo", "SARSA", "Q-Learning"])

    if algo_choice == "Monte Carlo":
        st.markdown(r"""
        ## Monte Carlo Control  

        - Monte Carlo methods learn directly from **complete episodes**.  
        - They do not bootstrap (no updates mid-episode).  
        - Instead, they use the *return* $G$ observed after visiting a state-action pair.  

        The return is defined as the **discounted cumulative reward**:  

        $$
        G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}
        $$

        Update rule for a visited $(s,a)$:  

        $$
        Q(s,a) \;\leftarrow\; Q(s,a) + \alpha \big( G - Q(s,a) \big)
        $$

        - $G$ is backed up at the **end of the episode**.  
        - Requires sufficient exploration (often enforced by ε-greedy behavior).  
        """)
        st.code(inspect.getsource(MonteCarloAgent), language="python")

    elif algo_choice == "SARSA":
        st.markdown(r"""
        ## SARSA (On-Policy TD Control)  

        - SARSA = **State, Action, Reward, State, Action**.  
        - On-policy: learns the value of the **same policy it follows** (ε-greedy).  

        Update rule:  

        $$
        Q(s,a) \;\leftarrow\; Q(s,a) + \alpha \big( r + \gamma Q(s',a') - Q(s,a) \big)
        $$

        - $a'$ is the next action chosen by the **ε-greedy behavior policy**.  

        ### ε-greedy action selection
        At each state $s$:  

        $$
        \pi(a|s) =
        \begin{cases}
        1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|}, & a = \arg\max_{a'} Q(s,a') \\\\
        \frac{\varepsilon}{|\mathcal{A}|}, & \text{otherwise}
        \end{cases}
        $$

        - With probability $(1-\varepsilon)$, choose the greedy action.  
        - With probability $\varepsilon$, choose a random action.  

        ### ε decay
        After each episode, we shrink exploration to let the policy converge:  

        $$
        \varepsilon \;\leftarrow\; \max(\varepsilon_{\min}, \; \varepsilon \cdot \varepsilon_{\text{decay}})
        $$

        """)
        st.code(inspect.getsource(SARSAAgent), language="python")

    elif algo_choice == "Q-Learning":
        st.markdown(r"""
        ## Q-Learning (Off-Policy TD Control)  

        - Off-policy: learns the value of the **greedy target policy** while behaving ε-greedy.  
        - This separates **behavior policy** (exploration) from **target policy** (optimality).  

        Update rule:  

        $$
        Q(s,a) \;\leftarrow\; Q(s,a) + \alpha \big( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \big)
        $$

        ### Behavior policy (exploration)
        - Actions are chosen with ε-greedy, exactly as in SARSA.  
        - Ensures all state-action pairs are visited.  

        ### Target policy (greedy)
        - Updates always assume the **best next action** will be chosen:  
        $\max_{a'} Q(s',a')$.  
        - Even if the behavior took a random move, updates still back up the greedy estimate.  

        ### ε decay
        As in SARSA, the exploration rate shrinks each episode:  

        $$
        \varepsilon \;\leftarrow\; \max(\varepsilon_{\min}, \; \varepsilon \cdot \varepsilon_{\text{decay}})
        $$

        This ensures early exploration and eventual exploitation of the learned greedy policy.  
        """)
        st.code(inspect.getsource(QLearningAgent), language="python")

    # First experiment: 3×5
    run_experiment(
        title="3×5 GridWorld",
        rows=3, cols=5,
        start=(2,0), goal=(2,4),
        obstacles=[(1,2),(2,2)],
        steps=20
    )
    
    # Divider
    st.markdown("---")

    # Second experiment: 5×8
    run_experiment(
        title="5×8 GridWorld",
        rows=5, cols=8,
        start=(4,0), goal=(4,7),
        obstacles=[(0,2),(1,2),(2,2),(1,5),(2,5),(3,5),(4,5)],
        steps=20
    )

if section == "Review":
    st.header("Overview")
    st.markdown(r"""
    ## Topics
    - Temporal Difference Prediction
    - Batch Learning Methods
    - SARSA (on-policy control)
    - Q-Learning (off-policy control)
    
    ## Learning Objectives
    - Implement TD prediction for a given policy
    - Apply TD methods in batch and/or offline
    - Implement SARSA and Q-Learning
    - Compare and contrast properties and characteristics of on-policy vs off-policy TD learning
    
    ## Review: Monte Carlo Prediction
    
    Estimate state values for given policy $\pi$
    
    Monte Carlo methods average observed utilities from multiple episodes
    
    $$
    V^\pi(s) = \frac{1}{N}\sum_{i=1}^N G_i(s)
    $$
    
    In implmentation we have a moving average update
    
    $$
    V^\pi(s) \leftarrow \frac{N(s)V^\pi(s) + G_i(s)}{N(s) + 1}
    $$
    
    where $N(s)$ is the number of visits to state $s$ prior to current episode
    
    ### Constant-$\alpha$ Monte Carlo
    
    $$
    V^\pi(s) \leftarrow \frac{N(s)V^\pi(s) + G_i(s)}{N(s) + 1} = V^\pi(s) + \frac{1}{N(s) + 1}(G_i(s) - V^\pi(s))
    $$
    
    We can choose to give more weight to recent returns by choosing a larger $\alpha$ to replace the sample average
    """)

if section == "Temporal Difference Learning":
  st.title("Temporal Difference Learning")
  
  st.markdown(r"""
  One of the drawbacks to Monte Carlo learning is that some sort of episodic structure is required. Each string of state-observations must end in order for a utility estimation to be returned
  
  In contrast, Dynamic Programming methods do not have this requirements as values are estimated based off successor state values.
  
  ### One-step Temporal Difference Learning (TD(0))
  
  Combines the strengths of Monte Carlo and Dynamic Programming.
  
  We do this by replacing the target term with the sum of the immediate reward with the discounted successor state value
  
  $$
  V^\pi(s) \leftarrow V^\pi(s) + \alpha(r_{t+1} + \gamma V^\pi(s') - V^\pi(s))
  $$
  
  We can now make updates to state values immediately after transition instead of waiting for an episode to complete.
  
  ### TD(0) Algorithm
  
  - Given: Policy $\pi$, learning rate $\alpha$ between 0 and 1
  - Initialize $V^\pi(s) \leftarrow 0$
  - Loop:
      - Initialize starting state s
      - Generate sequence(s, \pi(s), r, s')
      $$
      V^\pi(s) \leftarrow V^\pi(s) + \alpha(r_{t+1} + \gamma V^\pi(s') - V^\pi(s))
      $$
      - $s \leftarrow s'$
      
  Now values are updated immediately after each transition
  
  ### Optimality of TD(0)
  
  - Performs updates immediately with no episodic structure
  - Useful if problems have long episodes or are continuing tasks
  - For sufficiently small $\alpha$, average values of $V^\pi$ converge to true values
  - If $\alpha$ is constant, $V^\pi$ is prone to jumping around near convergence
  - In practice, we try to decrease $\alpha$ to $0$ over time.
  
  ## Offline Prediction
  
  So far constant-$\alpha$ MC and TD(0) are online, sample-based algorithms
  
  It is also possible to use these as offline sample-based algorithms
  
  Suppose we have a fixed set of samples in the place of the underlying model
  
  We can run TD(0) on the samples over and over until values converge
  
  We can perform **batch updating**, in which we make a single TD update based on the sum of all sample TD errors in the sample set.
  
  ### Batch Update Algorithm
  
  - Given: Tranining data and learning rate $\alpha$
  $$
  s_0, a_0, r_1, ..., s_{T-1}, a_{T-1}, r_T 
  $$
  
  - Initialize $V^\pi(s) \leftarrow 0$
  - Loop until values converge:
    - Initialize $\delta \leftarrow 0$ to store the total TD error for a state $s$
    - Then we go through each state of the training data in order
    
    For each state $s_t$ in training data:
    $$
    \delta(s_t) \leftarrow \delta(s_t) + (r_{t+1} + \gamma V(s_{t+1}) - V(s_t))
    $$
    
    Once we have computed all the temporal difference errors for all states in the sample set, we update the values
    - For each state $s$:
    
    $$
    V^\pi(s) \leftarrow V^\pi(s) + \alpha \delta(s)
    $$
  
  
  
  """)
  
if section == "TD Learning for Control":
  
  st.title("TD Learning for Control")
  st.markdown(r"""
  ## TD Learning for Control
  
  We can apply TD updates to Q values $Q(s, a)$ rather than state values $V(s)$
  
  After each transition $(s, a, r)$, we apply the TD update to $Q(s, a)$
  
  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha(r + \gamma Q(s', a') - Q(s, a))
  $$
    
  Action $a$ in state $s$ has been chosen according to behaviour policy
  
  May allow for exploration to learn about different actions.
  
  We need to think carefully about which successor state action to use: $Q(s', a')$
  
  This determination is going to be based on whether we want to use on-policy or off-policy learning
  
  ## On-Policy: SARSA
  
  - Given: Step size $\alpha$, exploration rate $\epsilon$
  - Initialize $Q(s, a) \leftarrow 0$, behaviour policy $\pi$ (e.g., $\epsilon$-greedy)
  - Loop: 
    - Initialize startign state $s$, action $a = \pi(s)$ if needed
    - Generate sequence $(s, a, r, s'), a' \leftarrow \pi(s')$
    $$
    Q(s, a), \leftarrow Q(s, a) \alpha(r + \gamma Q(s', a') - Q(s, a))
    $$
    
    - $s \leftarrow s', \quad a \leftarrow a'$
           
  ## Off-Policy: Q-Learning
  
  In off-policy approaches we will be able to learn about a target policy that is different from the one that we are following (behaviour policy)
  
  - Given: Step size $\alpha$, exploration rate $\epsilon$
  - Initialize $Q(s, a) \leftarrow 0$, behaviour policy $\pi$ (e.g., $\epsilon$-greedy)
  - Loop: 
    - Initialize startign state $s$, action $a = \pi(s)$
    - Generate sequence $(s, a, r, s')$
    $$
    Q(s, a), \leftarrow Q(s, a) \alpha \left(r + \gamma \ \text{max}_a Q(s', a') - Q(s, a)\right)
    $$
    
    - $s \leftarrow s'$
    
  ## SARSA vs. Q-Learning
  | SARSA | Q-Learning | 
  |-------|------------|
  |On-Polciy Approach: learn values of the behaviour policy| - Off-Policy Approach: learn values of the target policy|
  |Agents are more "cautious" as they worry about low rewards from exploration | Agents are more optimistic as only the best (greedy) actions at each state matter | 
  | SARSA and Q-Learning are identical if behaviour and target policy are the same or if exploration is removed|
  
  SARSA may be an ideal approach if we want to learn how to optimize a policy that includes exploration or if we want the agent to be more careful
  
  Q-Learning may be ideal if we just care about the optimality of the final greedy policy without worrying about low rewards received during learning due to exploration
  """)
  
if section == "Summary":
  st.title("Summary")
  st.markdown(r"""
  
  - Temporal difference methods update values after each transition instead of at the end of episode termination
  - Uses samples like MC, but bootstraps like DP
  
  - Online Prediction: $TD(0)$ one-step for online updates
  - Offline Prediction: Batch Updating can be done offline for fixed set of samples
  
  - Control: On-Policy (SARSA) vs off-policy(Q-learning)
  - SARSA learns value of behaviour policy, utilties reflect exploration
  - Q-Learning learns values of target policy, producing optimal policy
           
              
  """)