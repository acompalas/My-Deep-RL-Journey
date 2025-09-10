import streamlit as st 

st.title("My Deep Reinforcement Learning Journey")

st.markdown(r"""Welcome to my Deep Reinforcement Learning Journey!
In this streamlit website, I document my review of fundamental deep learning and reinforcement learning as I code demos and write notes mainly following OpenAI's 
Spinning Up as a Deep RL Researcher.          
""")


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
    st.Page("pages/TabularRL/02_TD_Learning.py",
            title="TD Learning",
            url_path="rl-td"),
]

# --- Home page ---
HOME = st.Page("pages/HomePage.py", title="Home", url_path="home")

# --- Sidebar toggle ---
with st.sidebar:
    choice = st.segmented_control("Show",
                                  ["All", "Deep Learning", "Tabular RL"],
                                  default="All",
                                  label_visibility="collapsed")

# --- Navigation builder ---
if choice == "Deep Learning":
    nav = st.navigation({"": [HOME], "Deep Learning": DL_PAGES})
elif choice == "Tabular RL":
    nav = st.navigation({"": [HOME], "Tabular Reinforcement Learning": TAB_RL_PAGES})
else:
    nav = st.navigation({
        "": [HOME],
        "Deep Learning": DL_PAGES,
        "Tabular Reinforcement Learning": TAB_RL_PAGES
    })

nav.run()
