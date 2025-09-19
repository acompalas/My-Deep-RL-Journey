import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.title("Multi-Armed Bandits")

section = st.selectbox("", [
    "Overview",
    "Epsilon-Greedy",
    "Upper Confidence Bound (UCB)",
    "Manual Demo",
    "Agent Demo"
])

if section == "Overview":
    st.header("Multi-Armed Bandit Problem")

    st.markdown(r"""
    The **Multi-Armed Bandit Problem** models the challenge of choosing between several uncertain actions (slot machine arms), each with an unknown reward distribution.

    The goal of the problem is to maximize total reward over time.
    
    ---
    
    ### Explore-Exploit Tradeoff
    
    The challenge this problem is to learn the reward distributions of the different arms while optmizing total payout.
    
    This balance is called the **explore–exploit tradeoff**.
    
    **Exploration:** 
    - Trying different arms to learn their payoffs.
    
    - **Explore only** may find high rewards by chance, but wastes pulls on bad arms.
    
    **Exploitation:** 
    - Choosing the best arm based on what you currently know.
    
    - **Exploit only** can still get unlucky (due to variance) and never discovers if another arm might be better.
    
    Smart strategies mix the two, improving decisions over time.
    
    ---
    
    ### Action Values
    
    Lets define choosing an arm on a slot machine as choosing an action $a$.
    
    Let $A_t$ be the action chosen and $R_t$ be the payout for that action at time $t$.
    
    The **Action Value** is the expected reward of an action
    $$
    Q^*(a) = E[R_t | A_t = a]
    $$
    
    The optimal strategy would be to pick the action $a$ with the highest expected reward $Q^*$.
    
    Since we do not initially know the action values for each arm, we try to estimate $Q^*$ by trying different actions and recording their results in the explore-exploit tradeoff.
    
    We can do this with a simple sample average
    
    $$
    Q_t(a) = \frac{\sum_{i=0}^{t} R_i \, \mathbf{1}\{A_i = a\}}{N_{t-1}(a)}
    $$
    
    where $N_{t}(a)$ is the number of times action $a$ has been chosen at time $t$.
    
    This would require storing all previous rewards leading to an $\mathcal{O}(t)$ space complexity for calculating the Action Value estimates.
    
    However using a Moving Average strategy allows us to only store the most recent reward, keeping the space in constant complexity
    
    $$
    Q_{n+1}(a) = Q_n(a) + \alpha \big( R_n - Q_n(a) \big)
    $$
    where $\alpha$ is either $1/N_n(a)$ (sample average) for stationary problems or a constant step size $0<\alpha\le1$ for non-stationary problems where reward distributions may change over time.
    
    In further sections we will define estimation strategies to balance exploration and exploitation building off the moving average strategy
    
    ---
    
    """)
    
    st.subheader("Strategies We'll Explore")
    st.markdown(r"""
    In this module, we will explain and compare three popular strategies for balancing exploration and exploitation:
    
    1. **Epsilon-Greedy:**  
       - With probability $\epsilon$, explore a random arm.  
       - With probability $1-\epsilon$, exploit the best-known arm.
    
    2. **Upper Confidence Bound (UCB):**  
       - Choose the arm with the highest optimistic estimate of its value, balancing mean reward with uncertainty.
    
    3. **Thompson Sampling:**  
       - Maintain a probability distribution over each arm's expected reward, sample from these, and choose the best sample.
    """)

elif section == "Epsilon-Greedy":
    st.header("Epsilon-Greedy")

    st.markdown(r"""
    The Epsilon-Greedy strategy for assigns probabilities for any given action selection to explore with probability $\epsilon$ **explore** (pick a random arm), otherwise **exploit** (pick the arm with the largest current estimate).
    
    This simple rule trades off discovering better arms vs. harvesting the best-known one.

    ---
    
    ### Action Selection 
    
    We know that the goal of the Multi-Bandit Problem is to maximize the overall payout for a sequence of actions.
    
    To do this, we want to select the actions that maximize the Action Values previously mentioned in the **Overview**
    
    $$
    A_t = \argmax_{a}Q^*(a)
    $$

    But since we only know $Q_t$ our Action Value estimates may not be accurate.
    
    $\epsilon$-greedy action selection selects the greedy action most of the time for exploitation but explores with some small probability $\epsilon$ picking a random action instead

    ### Example

    Suppose we have 3 arms with the following estimated Action Values:
    
    $$
    Q_t(a_1) = 3,\quad Q_t(a_2) = 2,\quad Q_t(a_3) = 5
    $$
    
    In this case:
    
    $$
    a_3 = \arg\max_a Q_t(a)
    $$
    
    So under **pure exploitation**, we would always choose $a_3$.
    
    However, the **true** reward distributions might be different:
    
    - Arm 1: $\mathcal{N}(\mu=4, \sigma=1)$  
    - Arm 2: $\mathcal{N}(\mu=2.5, \sigma=1)$  
    - Arm 3: $\mathcal{N}(\mu=3.5, \sigma=1)$
    
    Here, the true best arm is **Arm 1** (mean reward 4), but our estimates $Q_t$ are misleading due to limited or unlucky samples.
    
    ---
    
    ### **Why $\epsilon$-greedy helps:**  
    - With probability $1-\epsilon$, we exploit and choose $a_3$ for largest known $Q_t$ 
    - With probability $\epsilon$, we explore another arm updating $Q_t(a)$ and giving us the chance to discover that $a_1$ is actually better.
    
    Over time, exploration allows $Q_t$ to converge toward $Q^*(a)$, and exploitation becomes more effective.
    
    The $Q_t$ values used in $\epsilon$-greedy can be computed in different ways:

    - **Sample Average (Moving Average with $\alpha = 1 / N_t(a)$):**  
      $$
      Q_{n+1}(a) = Q_n(a) + \frac{1}{N_n(a)} \big( R_n - Q_n(a) \big)
      $$
      Here, $N_n(a)$ is the number of times arm $a$ has been selected.  
      This is equivalent to a true average of all past rewards, and works best in **stationary** environments.

    - **Exponential Recency-Weighted Average (Constant $\alpha$):**  
      $$
      Q_{n+1}(a) = Q_n(a) + \alpha \big( R_n - Q_n(a) \big)
      $$
      where $0 < \alpha \le 1$ is fixed.  
      - **Small $\alpha$** → smooth changes slowly, gives more weight to long-term history.  
      - **Large $\alpha$** → adapts quickly to recent rewards, useful in **non-stationary** environments.
    
    """)

elif section == "Upper Confidence Bound (UCB)":
    st.header("Upper Confidence Bound (UCB)")

    st.markdown(r"""
    The **Upper Confidence Bound (UCB)** strategy addresses the explore–exploit tradeoff by adding an **optimism bonus** to each action's estimated value.  
    Instead of exploring randomly, UCB explores actions that are **uncertain** but potentially rewarding.

    ---
    
    ### Action Selection Rule
    
    At time $t$, UCB chooses the action $a$ that maximizes:
    
    $$
    A_t = \arg\max_a \Bigg[ Q_t(a) + c \, \sqrt{\frac{\ln t}{N_t(a)}} \Bigg]
    $$
    
    where:
    - $Q_t(a)$ = current action-value estimate  
    - $N_t(a)$ = number of times action $a$ has been selected so far  
    - $t$ = current timestep (total pulls so far)  
    - $c > 0$ = exploration parameter controlling how much to favor uncertainty  

    ---
    
    ### Intuition
    
    - The first term $Q_t(a)$ favors actions with high estimated rewards.  
    - The second term $\;c \sqrt{\ln t / N_t(a)}\;$ is larger when:
      - $ln(t)$ grows very slowly as $t$ goes to $\infty$,  
      - $N_t(a)$ is small (arm has been tried few times → high uncertainty).  
    - Thus, arms with fewer samples get an **optimistic boost**.  
    - As $N_t(a)$ grows, the confidence bonus shrinks and UCB relies more on actual $Q_t(a)$.  
    - Because $ln(t)$ grows slowly, there may be a time to pick the least picked worst action again, but not for a very long time.


    Below is an illustration of UCB with two actions:
    
    - **Arm A** has a lower estimated mean $Q(A)$ but **high uncertainty** (wide confidence interval).  
    - **Arm B** has a higher mean $Q(B)$ but **lower uncertainty** (narrower interval). 
    """)
    
    st.image("assets/ucb.png", caption="Upper Confidence Bound illustration")
    
    st.markdown(r"""
                          
    Notice that the **upper half** of Arm A’s confidence interval reaches above the mean of Arm B.  
    This “optimistic” estimate makes UCB consider Arm A worth exploring further, even though its current average reward is lower.  
    
    This is why the method is called the **Upper Confidence Bound** — it selects actions based not only on their mean estimate $Q_t(a)$, but also on the upper edge of the plausible range of their value.
    
    ---   
    
    ### Example
    
    Suppose after $t=100$ pulls we have:
    
    - $Q_t(a_1)=3$, $N_t(a_1)=60$  
    - $Q_t(a_2)=2.8$, $N_t(a_2)=30$  
    - $Q_t(a_3)=2.5$, $N_t(a_3)=10$
    
    With $c=2$:
    
    $$
    UCB(a_1) = 3 + 2\sqrt{\frac{\ln(100)}{60}} \approx 3.34
    $$
    $$
    UCB(a_2) = 2.8 + 2\sqrt{\frac{\ln(100)}{30}} \approx 3.32
    $$
    $$
    UCB(a_3) = 2.5 + 2\sqrt{\frac{\ln(100)}{10}} \approx 3.44
    $$
    
    Even though $Q_t(a_3)$ is lowest, its **uncertainty bonus** makes it the best candidate to try next.

    ---
    
    ### Advantages of UCB
    
    - Balances exploration and exploitation **deterministically** (not random).  
    - Strong theoretical guarantees: achieves **logarithmic regret**.  
    - Naturally reduces exploration over time as estimates become more reliable.  

    ---
    
    ### Limitations
    
    - Requires careful tuning of $c$ (exploration parameter).  
    - Computationally heavier than $\epsilon$-greedy since it tracks confidence bounds.  
    - Works best when reward distributions are stationary; struggles if they shift over time.  
    """)
    
elif section == "Thompson Sampling":
    st.header("Thompson Sampling")
    
elif section == "Manual Demo":
    from utils.multi_armed_bandits import SlotMachine

    st.header("Manual Demo")

    # ------------------ session init (fixed to 3 arms) ------------------
    # ---- state defaults (set once) ----
    if "demo_seed" not in st.session_state:
        st.session_state.demo_seed = 123

    def init_specs_from_seed(seed: int):
        rng = np.random.default_rng(seed)
        means = rng.normal(0, 1, size=3)
        stds  = rng.uniform(0.5, 2.0, size=3)
        return [{"mean": float(m), "std": float(s)} for m, s in zip(means, stds)]

    if "demo_specs" not in st.session_state:
        st.session_state.demo_specs = init_specs_from_seed(st.session_state.demo_seed)

    if "demo_sm" not in st.session_state:
        st.session_state.demo_sm = SlotMachine(
            n_arms=3,
            reward_distributions=st.session_state.demo_specs,
            seed=st.session_state.demo_seed,
        )

    if "demo_history" not in st.session_state:
        st.session_state.demo_history = {0: [], 1: [], 2: []}

    sm   = st.session_state.demo_sm
    hist = st.session_state.demo_history

    # ---- top controls ----
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🔄 Initialize New Specs"):
            # change seed so specs differ next time
            st.session_state.demo_seed += 1
            # replace specs
            st.session_state.demo_specs = init_specs_from_seed(st.session_state.demo_seed)
            # build a brand-new SlotMachine with the new specs
            st.session_state.demo_sm = SlotMachine(
                n_arms=3,
                reward_distributions=st.session_state.demo_specs,
                seed=st.session_state.demo_seed,
            )
            # clear samples/counters
            st.session_state.demo_history = {0: [], 1: [], 2: []}
            sm = st.session_state.demo_sm  # refresh local reference
            hist = st.session_state.demo_history
            st.success("Initialized new specs and reset stats.")

    with c2:
        if st.button("♻️ Reset Stats (counts & samples)"):
            # do NOT touch specs or seed; just reset counters and clear sample history
            sm.reset()
            st.session_state.demo_history = {0: [], 1: [], 2: []}
            hist = st.session_state.demo_history
            st.info("Counts and samples reset (specs unchanged).")
        
    # ------------------ show arm parameters ------------------
    st.markdown("**Arm parameters (Normal distributions):**")
    colm = st.columns(3)
    for i, c in enumerate(colm):
        with c:
            st.metric(label=f"Arm {i}", value=f"μ={sm.means[i]:.2f}", delta=f"σ={sm.stds[i]:.2f}")

    st.divider()
    
    # shared x-range over all arms
    xmin = min(m - 4*s for m, s in zip(sm.means, sm.stds))
    xmax = max(m + 4*s for m, s in zip(sm.means, sm.stds))
    xs = np.linspace(xmin, xmax, 600)

    def normal_pdf(x, mu, sd):
        return (1.0 / (sd * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sd)**2)

    colors = ["C0", "C1", "C2"]
    fig = plt.figure(figsize=(8, 4.2))

    # plot true PDFs
    for i in range(3):
        ys = normal_pdf(xs, sm.means[i], sm.stds[i])
        plt.plot(xs, ys, linewidth=2, label=f"True PDF Arm {i} (μ={sm.means[i]:.2f}, σ={sm.stds[i]:.2f})", color=colors[i])

    # scatter the observed samples (small vertical jitter so points are visible)
    for i in range(3):
        if len(hist[i]) > 0:
            y_base = -0.02 - 0.015 * i
            jitter = (np.random.rand(len(hist[i])) - 0.5) * 0.01
            plt.scatter(hist[i], np.full(len(hist[i]), y_base) + jitter, s=20, color=colors[i], alpha=0.8, label=f"Samples Arm {i}")

    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("Reward")
    plt.ylabel("Density")
    plt.title("Each click draws one sample from the selected arm's Normal distribution")
    plt.legend(fontsize=8, ncol=2)
    st.pyplot(fig)

    # ------------------ three buttons = three arms ------------------
    st.markdown("**Click a button to pull an arm (sample one reward):**")
    bcols = st.columns(3)
    last_pull = st.empty()

    def do_pull(i: int):
        r = sm.pull(i)
        hist[i].append(r)
        last_pull.success(f"Pulled Arm {i} → reward = {r:.3f}")

    if bcols[0].button("Pull Arm 0"):
        do_pull(0)
    if bcols[1].button("Pull Arm 1"):
        do_pull(1)
    if bcols[2].button("Pull Arm 2"):
        do_pull(2)

    st.caption(f"Total pulls: {sm.t} | Counts per arm: {sm.counts.tolist()}")

    st.divider()
    
    # ------------------ running Q estimates + best estimated arm ------------------
    est_means = []
    cols2 = st.columns(3)
    for i, c in enumerate(cols2):
        with c:
            if len(hist[i]) == 0:
                st.write(f"Arm {i}: no pulls yet")
                est_means.append(None)
            else:
                qhat = float(np.mean(hist[i]))
                est_means.append(qhat)
                st.write(f"Arm {i}:  $\\hat Q$ = {qhat:.3f}  (n={len(hist[i])})")
                
    st.markdown(r"""
    $$
    Q_{n+1}(a) = Q_n(a) + \frac{1}{n} \big( R_n - Q_n(a) \big)
    $$
    
    $$
    A_t = \argmax_{a}Q^*(a)
    $$
    """)

    # best current estimate (Q-hat)
    valid = [(i, m) for i, m in enumerate(est_means) if m is not None]
    if valid:
        best_arm_hat, best_mean_hat = max(valid, key=lambda x: x[1])
        st.success(
            f"Best estimated arm (based on samples): "
            f"**Arm {best_arm_hat}** with $\\hat Q \\approx {best_mean_hat:.3f}$"
        )

        # true best (Q-star)
        best_arm_star = int(np.argmax(sm.means))
        best_mean_star = sm.means[best_arm_star]
        st.info(
            f"True best arm (ground truth): "
            f"**Arm {best_arm_star}** with $Q^* = {best_mean_star:.3f}$"
        )
    else:
        st.info("No pulls yet — click a button above to sample and estimate $\\hat Q$.")


    # ------------------ combined visualization: scatter samples + true PDFs ------------------
    # st.subheader("Observed samples vs. true distributions")

elif section == "Agent Demo":
    from utils.multi_armed_bandits import (
        SlotMachine,
        ExploreAgent,
        ExploitAgent,
        EpsilonGreedyAgent,
        UCBAgent,
        run_episode,
        regret_curve,
    )

    st.header("Agent Demo")

    # ---------- helpers ----------
    def init_specs_from_seed(n_arms: int, seed: int):
        rng = np.random.default_rng(seed)
        means = rng.normal(0, 1, size=n_arms)
        stds  = rng.uniform(0.5, 2.0, size=n_arms)
        return [{"mean": float(m), "std": float(s)} for m, s in zip(means, stds)]

    # ---------- session state init ----------
    if "agent_seed" not in st.session_state:
        st.session_state.agent_seed = 42
    if "agent_n_arms" not in st.session_state:
        st.session_state.agent_n_arms = 4
    if "agent_specs" not in st.session_state:
        st.session_state.agent_specs = init_specs_from_seed(
            st.session_state.agent_n_arms, st.session_state.agent_seed
        )

    # ---------- top controls: n_arms + reinit specs ----------
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        n_arms_new = st.number_input("Number of arms", min_value=2, max_value=20,
                                     value=st.session_state.agent_n_arms, step=1)
    with c2:
        if st.button("🔄 Initialize New Specs"):
            # bump seed → different specs next time
            st.session_state.agent_seed += 1
            st.session_state.agent_n_arms = int(n_arms_new)
            st.session_state.agent_specs = init_specs_from_seed(
                st.session_state.agent_n_arms, st.session_state.agent_seed
            )
            st.success("Initialized new specs (means/stds) and kept RNG-at-instantiation semantics.")

    # ---------- instantiate env from current specs/seed ----------
    env = SlotMachine(
        n_arms=st.session_state.agent_n_arms,
        reward_distributions=st.session_state.agent_specs,
        seed=st.session_state.agent_seed,  # seed fixed at instantiation
    )
    best_mean = float(np.max(env.means))

    with st.expander("Bandit details (ground truth)"):
        cols = st.columns(st.session_state.agent_n_arms)
        for i, col in enumerate(cols):
            with col:
                st.metric(f"Arm {i}", value=f"μ={env.means[i]:.2f}", delta=f"σ={env.stds[i]:.2f}")

    st.divider()

    # ------------------ Controls ------------------
    agent_options = {
        "Explore": ExploreAgent,
        "Exploit (Explore-Then-Commit)": ExploitAgent,
        "Epsilon-Greedy": EpsilonGreedyAgent,
        "UCB": UCBAgent,
    }
    selected = st.multiselect(
        "Agents to simulate",
        list(agent_options.keys()),
        default=["Epsilon-Greedy", "UCB"]
    )

    left, right = st.columns(2)
    with left:
        episodes = st.number_input("Episodes", min_value=1, max_value=5000, value=200, step=50)
        horizon  = st.number_input("Horizon (steps per episode)", min_value=1, max_value=10000, value=500, step=50)
        update_every = st.number_input("Update charts every N episodes", min_value=1, max_value=1000, value=20, step=5)
    with right:
        epsilon = st.number_input("ε (Epsilon-Greedy)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
        c_ucb   = st.number_input("c (UCB bonus)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.1f")
        m_per   = st.number_input("m per arm (Exploit ETC)", min_value=1, max_value=100, value=2, step=1)
        alpha   = st.select_slider("Value update (α): None = sample average; else constant α",
                                   options=[None, 0.05, 0.1, 0.2, 0.5, 0.9], value=None)

    # ------------------ Placeholders ------------------
    ph_info   = st.empty()
    ph_avg    = st.empty()
    ph_regret = st.empty()
    ph_table  = st.empty()
    st.divider()

    run = st.button("▶️ Simulate")

    if run:
        results = {name: [] for name in selected}

        # matplotlib helpers
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        def plot_mean_avg_reward(data_dict):
            fig, ax = plt.subplots()
            for name, mats in data_dict.items():
                if len(mats) == 0: 
                    continue
                R = np.vstack(mats)  # [episodes, horizon]
                avg_rewards = np.mean(np.cumsum(R, axis=1) / (np.arange(1, R.shape[1]+1)), axis=0)
                ax.plot(avg_rewards, label=name)
            ax.set_title("Average Reward over Time")
            ax.set_xlabel("Step t")
            ax.set_ylabel("Average Reward")
            ax.legend()
            ph_avg.pyplot(fig)

        def plot_mean_regret(data_dict):
            fig, ax = plt.subplots()
            for name, mats in data_dict.items():
                if len(mats) == 0:
                    continue
                R = np.vstack(mats)
                mean_reg = np.mean([regret_curve(r, best_mean) for r in R], axis=0)
                ax.plot(mean_reg, label=name)
            ax.set_title("Mean Regret (vs best fixed arm)")
            ax.set_xlabel("Step t")
            ax.set_ylabel("Regret")
            ax.legend()
            ph_regret.pyplot(fig)

        def make_agent(name):
            cls = agent_options[name]
            if name == "Epsilon-Greedy":
                return cls(env.n_arms, epsilon=epsilon, alpha=alpha)
            elif name == "UCB":
                return cls(env.n_arms, c=c_ucb, alpha=alpha)
            elif name == "Exploit (Explore-Then-Commit)":
                return cls(env.n_arms, m_per_arm=m_per, alpha=alpha)
            else:
                return cls(env.n_arms, alpha=alpha)

        for ep in range(1, episodes + 1):
            for name in selected:
                env.reset()
                agent = make_agent(name)
                rewards = run_episode(env, agent, horizon)
                results[name].append(rewards)

            if ep % update_every == 0 or ep == episodes:
                ph_info.info(f"Completed {ep}/{episodes} episodes")
                plot_mean_avg_reward(results)
                plot_mean_regret(results)

                rows = []
                for name, mats in results.items():
                    R = np.vstack(mats)
                    mean_total = float(np.mean(np.sum(R, axis=1)))
                    mean_final_avg = float(np.mean(np.cumsum(R, axis=1)[:,-1] / horizon))
                    mean_final_regret = float(np.mean([regret_curve(r, best_mean)[-1] for r in R]))
                    rows.append({
                        "Agent": name,
                        "Avg Total Reward": f"{mean_total:.2f}",
                        "Final Avg Reward": f"{mean_final_avg:.3f}",
                        "Final Regret": f"{mean_final_regret:.2f}"
                    })
                ph_table.dataframe(pd.DataFrame(rows))

        st.success("Simulation complete.")

    

    
