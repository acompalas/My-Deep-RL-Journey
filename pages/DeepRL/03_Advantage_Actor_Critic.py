import streamlit as st

st.title("Advantage Actor Critic")

section = st.selectbox(
    "",
    [
        "Demo",
        "My Notes",
        "Asynchronous Methods for Deep Reinforcement Learning"
    ],
)

if section == "Demo":
    st.title("Demo: A2C and A3C on LunarLander-v3")
    
if section == "Asynchronous Methods for Deep Reinforcement Learning":
    st.title("Asynchronous Methods for Deep Reinforcement Learning")
    
    st.markdown(r"""
    
    ### Abstract
    - Asynchronous actor critic surpasses SOTA Atari (Probably DQNs)
    
    ### Introduction   
             
    """)