import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.title("Markov Decision Processes")

section = st.selectbox("", [
    "The MDP Framework",
    "Policy and Value Functions",
    "Bellman Equations",
])

if section == "The MDP Framework":
    st.title("The MDP Framework")
    
    st.markdown(r"""A Markov decision process (MDP) is a model for sequential decision making when outcomes are uncertain.       
    """)
    
    st.image("assets/mdp_framework.png", caption="MDP Framework for RL", use_container_width=True)
    
    st.markdown(r"""
    In Reinforcement Learning, a Markov Decision Process is modeled in an Agent-Environment Interface.
    
    - An Agent can interact with the environment by performing Actions.
    - The result of an action can change the state of a system.
    - The agent can receive feedback in the form of rewards
    
    In a Markov Decision Processes:
    - **Markov**: refers to the memoryless relationship where the next state only depends on the previous state and action. 
    - **Decision**: refers to the fact that we select an action depending on the current state such that maximizes the reward as introduced in Multi-Armed bandits. This leads to a state transition and accumulation of rewards.
    - **Process**: emphasizes the interaction between the agent and the environment as a sequence of markov decisions and their outcomes.
    
    When optimizing Markov Decision Processes, we need to define:
    - **Agent**
    - **Environemnt**
    - **Action Space** $A(s)$
    - **State Space**: $S_t$
    - **Transition Function**: Probability of a new state given current state and action
    $$
    T(s,a,s') = Pr(s'|s,a)
    $$
    - **Rewards**: $R_t(s, a, s') \in \mathbb{R}$
                
    """)
    
if section == "Policy and Value Functions":
    st.title("Policy and Value Functions")
    
    st.markdown(r"""
    ## Utility 
    We quantify decision-making in MDPs by defining a **utility** for any given sequence of states and actions
    
    A rational agent seeks to maximize expected utility.
    
    A first approach to define utility of a state-action sequence is to just sum all the rewards.
    $$
    V([s_0, a_0, s_1, a_1 \dots, a_{T-1}, s_T]) = \sum_{t=0}^{T-1} R(s_t, a_t, s_{t+1})
    $$
    
    However, a simple sum does not count for the order of which rewards are received. 
    
    In many cases rewards are preferable now rather than later.
    
    To give more weight to immediate rewards, we apply a discount factor $0 < \gamma < 1$ to diminish future rewards:
    $$
    V([s_0, a_0, s_1, a_1 \dots, a_{T-1}, s_T]) = \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t, s_{t+1})
    $$
    
    where rewards at later timesteps discount rewards by a larger and larger factor.
    
    Suppose we have an infinite sequence of rewards all equal to 2, what is the utility of this sequence using $\gamma = 0.8$?
    
    $$
    V = \sum_{t=0}^\infty \gamma^t R = 0.8^0 (2) + 0.8^1 (2) + \dots 0.8^t (2) = 2 + 1.6 + ...
    $$
    
    We can solve this with the closed-form solution for a simple geometric sequence:
    $$
    V = \sum_{t=0}^\infty \gamma^t R = \frac{R}{1-\gamma} = \frac{2}{0.2} = 10
    $$
    
    We use this closed-form solution to define the upper bound for maximum rewards
    
    $$
    V([s_0, a_0, s_1, \dots]) = \sum_{t=0}^\infty \gamma^t R(s_t, a_t, s_{t+1}) \leq \frac{R_{max}}{1-\gamma} 
    $$
    
    ## Policy and Value Functions
    An MDP defines a decision problem. Solving an MDP means finding a policy mapping states to actions.
    
    The solution to an MDP is called a policy. A policy function 
    $$
    \pi: S \rightarrow A
    $$ 
    
    tells the agent what to do in any given state.
    
    Each policy function has a value function:
    $$
    V^\pi: S \rightarrow R
    $$
    that describes the expected utility of following $\pi$ from a given state
    $$
    V^\pi(s) = E\left[\sum_{t=0}\gamma^t r(s_t, \pi(s_t), s_{t+1})\right], s_0 = s
    $$
    We are interested in finding the optimal policy and value functions
    
    $$
    \pi^* = argmax_\pi V^\pi \quad V^* = max_\pi V^\pi
    $$
    
    both of which are defined by maximizing the value at all possible states.
    
    We can write the recursive definition of the value function as summing over the values of successors after a transition:
    
    $$
    V^\pi(s) = \sum_{s'} T(s, \pi(s), s')[R(s, \pi(s), s') + \gamma V^\pi(s')]
    $$
    
    Intuitively, this is the sum of the transition probability times the individual value.
    
    """)
    
if section == "Bellman Equations": 
    st.header("Bellman Equations")
    
    st.markdown(r"""
                
    Recall to solve an MDP we want to find an optimal policy and value function
    
    $$
    V^\pi(s) = \sum_{s'} = T(s, \pi(s), s')[R(s, \pi(s), s') + \gamma V^\pi(s')]
    $$
    
    We approximate the optimal functions by considering the set of all possible actions:
    
    $$
    V^*(s) = \text{max}_a \sum_{s'} = T(s, a, s')[R(s, a, s') + \gamma V^*(s')]
    $$
    $$
    \pi^*(s) = \text{argmax}_a \sum_{s'} = T(s, a, s')[R(s, a, s') + \gamma V^*(s')]
    $$
    
    These are called the Bellman Optimality Equations
                
    """)
    
    st.header("Dynamic Programming")
    
    st.markdown(r"""
    ## Value Iteration
    
    Value iteration is a dynamic programming approach to solving for $V^*$
    
    We define the recursive Bellman Update Equation which accounts for the number of transitions in a sequence
          
    $$
    V_{i+1}(s) = \text{max}_a \sum_{s'} T(s, a, s')[R(s, a, s') + \gamma V_i(s')]
    $$

    We assume that $V_i$ approaches $V^*$ as $i \rightarrow \infty$. That is to say the optimal Value function can be approximated with longer and longer sequences.
    - for every iteration from $i=0$
    $$
    V_{i+1}(s) \leftarrow \text{max}_a \sum_{s'} = T(s, a, s')[R(s, a, s') + \gamma V_i(s')]
    $$
    - for each state $s \in S$:
    - until $\text{max}_a | V_{i+1}(s) - V_i(s)| < \epsilon$ (small threshold)
    
    ## Policy Iteration
    
    The goal of value iteration is to eventually extract an optimal policy
    $$
    \pi^*(s) = \text{argmax}_a \sum_{s'} = T(s, a, s')[R(s, a, s') + \gamma V^*(s')]
    $$
    
    A policy can be computed at any point during value iteration.
    
    - initialize $\pi_0$ arbitratily for all states
    - Loop from $i=0$
        - Policy evaluation: Compute $V^{\pi_i}$ for policy $\pi_i$
        - Policy improvement: Given $V^{\pi_i}$ find new policy $\pi_{i+1}$
    - until $\pi_{i+1} = \pi_{i}$
    
    ### Policy Evaluation
    
    We can iteratively evaluate values for a specific policy
    - Loop from $i=0$
        - for each state $s \in S$:
    $$
    V_{i+1}^\pi(s) \leftarrow \sum_{s'} T(s, \pi(s), s')[R(s, \pi(s), s') + \gamma V_i^\pi(s')]
    $$
    - until $\text{max}_a | V_{i+1}^\pi(s) - V_i^\pi(s')| < \epsilon$ (small threshold)
    
    ### Policy Improvement
    Policy improvement changes the policy $\pi$ given the values we evaluated during Policy Evaluation
    
    $$
    \pi_{i+1(s)} = \text{argmax}_a \sum_{s'} = T(s, a, s')[R(s, a, s') + \gamma V^{\pi_i}(s')]
    $$
    - We consider taking the "greediest" action at each state
    - If $\pi_i$ already optimal then $V^{\pi_i} = V^*$ and $\pi_{i+1} = \pi_{i}$
    - Otherwise we do not have the optimal solution so we continue iterating by changing actions
    """)
    
if section == "Monte Carlo Methods":
    st.title("Monte Carlo Methods")
    
    st.markdown(r"""
    Monte Calor methods generate sampled experience and average them for different states and actions
    
    Recall the definition of the value function for a fixed policy $\pi$:
    $$
    V^{\pi}(s) = E\left[\sum_{t=0}\gamma^t R(s_t, \pi(s_t), s_{t+1})\right]
    $$
    
    The idea is to approximate the expectation by taking averages of sample reward sequences over multiple episodes
                
    Value Estimation
    
    State-action Values
    Epsilon Greedy on-policy MC Control
    
    Importance Sampling
    Monte Carlo control         
    """)
# st.markdown(r"""

# Agent Environment interface (insert image)

# Markov Decision Process definition
# - State Space and Action space for each state
# - Transition Function probability to go to another state given as state and an action
# - Reward function the reward of an action when going to a new state

# Markov Property: Transitions depend on finite number of previous states

# Math
# Utility -> maximizing rewards
# Discounting for future rewards (gamma)
# V = \sum ...

# Policy and Value Functions

# Gridworld example (3x4)
# - Starting state 
# - Obstacles
# - Terminal States
# - 

# Bellman Optimality Equations
            
# """)