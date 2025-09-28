import streamlit as st 

# --- Deep Learning pages ---
DL_PAGES = [
    st.Page("pages/DeepLearning/01_Feedforward_Networks.py",
            title="Feedforward Networks",
            url_path="dl-feedforward"),
    st.Page("pages/DeepLearning/02_Convolutional_Neural_Networks.py",
            title="Convolutional Neural Networks",
            url_path="dl-cnn"),
    st.Page("pages/DeepLearning/03_Recurrent_Neural_Networks.py",
            title="Recurrent Neural Networks",
            url_path="dl-rnn"),
]

# --- Tabular RL pages ---
TAB_RL_PAGES = [
    st.Page("pages/TabularRL/01_Multi_armed_Bandits.py",
            title="Multi-Armed Bandits",
            url_path="rl-bandits"),
    st.Page("pages/TabularRL/02_Markov_Decision_Processes.py",
            title="Markov Decision Processes",
            url_path="rl-mdps"),
    st.Page("pages/TabularRL/03_Monte_Carlo_Methods.py",
            title="Monte Carlo Methods",
            url_path="rl-mc"),
    st.Page("pages/TabularRL/04_TD_Learning.py",
            title="TD Learning",
            url_path="rl-td"),
]

DEEP_RL_PAGES = [
    st.Page("pages/DeepRL/01_Vanilla_Policy_Gradient.py",
            title="Vanilla Policy Gradient",
            url_path="rl-vpg"),
    st.Page("pages/DeepRL/02_Deep_Q_Networks.py",
            title="Deep Q Networks",
            url_path="rl-dqn"),
    st.Page("pages/DeepRL/03_Advantage_Actor_Critic.py",
            title="Advantage Actor Critic",
            url_path="rl-a2c"),
]

# --- Home page ---
HOME = st.Page("pages/HomePage.py", title="Home", url_path="home")

# --- Sidebar toggle ---
with st.sidebar:
    choice = st.segmented_control("Show",
                                  ["Deep Learning", "Tabular RL", "Deep RL"],
                                  label_visibility="collapsed")

# --- Navigation builder ---
if choice == "Deep Learning":
    nav = st.navigation({"": [HOME], "Deep Learning": DL_PAGES})
elif choice == "Tabular RL":
    nav = st.navigation({"": [HOME], "Tabular Reinforcement Learning": TAB_RL_PAGES})
elif choice == "Deep RL":
    nav = st.navigation({"": [HOME], "Deep Reinforcement Learning": DEEP_RL_PAGES})
else:
    nav = st.navigation({
        "": [HOME],
        "Deep Learning": DL_PAGES,
        "Tabular Reinforcement Learning": TAB_RL_PAGES,
        "Deep Reinforcement Learning": DEEP_RL_PAGES
    })

nav.run()
