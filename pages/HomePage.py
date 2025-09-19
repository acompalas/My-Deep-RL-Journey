import streamlit as st 

st.title("My Deep Reinforcement Learning Journey")

st.markdown(r"""
Welcome to my Deep Reinforcement Learning Journey!  
This site documents my review of deep learning and reinforcement learning through code demos and notes.  
I mainly follow OpenAI's *Spinning Up as a Deep RL Researcher*, Andrew Ng’s *Deep Learning Specialization*,  
and Columbia's *Decision Making and Reinforcement Learning*.

---

### Demos

#### 🔹 Deep Learning
- Feedforward: MNIST (MLP)
- CNNs: LeNet (MNIST), AlexNet, ResNet (CIFAR-10/Imagenet)
- RNNs: Karpathy’s char-RNN (text generation)
- Transformers: BERT (MLM), GPT (next-token prediction)
- VAEs: MNIST latent space demo
- Diffusion: image denoising demo

#### 🔹 Tabular RL
- Multi-Armed Bandits: Slot machine demo
- Markov Decision Processes: notes only
- Monte Carlo Methods: notes only
- TD Learning: Monte Carlo, SARSA, Q-learning on GridWorld
- Function Approximation: TD(λ) with eligibility traces (GridWorld/Checkers + NN)

#### 🔹 Deep RL
- VPG: CartPole-v1
- DQN: LunarLander-v2
- SAC: BipedalWalker-v3
- PPO: Pendulum-v1 → Ant-v3 (PyBullet)
- DDPG: Reacher-v2 (robotic arm)
- Isaac Lab / PyBullet: Quadruped, Robotic Arm, Humanoid

---
""")
