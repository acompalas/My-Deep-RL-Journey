import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import ConfusionMatrixDisplay

st.title("Feed Forward Networks")

# -------------------- NAV --------------------
section = st.selectbox("", [
    "Demo",
    "Overview",
    "Multi-Layer Perceptrons",
    "Backpropagation",
    "Initializations",
    "Activations",
    "Criterions",
    "Optimizers",
    "Normalization",
    "Regularization",
])

if section == "Demo":
    st.title("Demo")
    
    st.markdown("""
    Before we dive into feedforward networks, it’s important to see the difference between 
        **Gradient Descent (GD)** and **Stochastic Gradient Descent (SGD)**.  
    """)

    st.subheader("Gradient Descent vs. Stochastic Gradient Descent")
    st.markdown(r"""
        **Full-Batch GD** uses the gradient of the **average loss across all data points**:
        
        $$
        \nabla L(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla \ell_i(\theta)
        $$
        
        **Stochastic GD** uses the gradient from **just one random data point**:
        $$
        \nabla L(\theta) = \nabla \ell_i(\theta)
        $$
        
        **Mini-Batch Stochastic GD** updates based on single datapoints in a batch
        $$
        \nabla L_B(\theta) = \frac{1}{k} \sum_{i\in B}^N \nabla \ell_i(\theta)
        $$
    """)
    with st.expander("show/hide"):
        st.markdown(r"""
        Here we use a simple **linear regression** problem to illustrate.  
        """)

        st.subheader("1. Data Generation")
        st.code("""
        N = 1000
        m_true, b_true = 2.0, 0.5
        X = np.linspace(0, 1, N).reshape(-1, 1)
        y = m_true * X + b_true + np.random.normal(0, 1, size=(N, 1))
            """, language="python")

        # -----------------------
        # Loss function
        # -----------------------

        st.subheader("2. Loss Function")
        st.code("""
        def predict(X, w, b):
            return X * w + b

        def mse_loss(y_pred, y):
            return np.mean((y_pred - y) ** 2)
            """, language="python")

        # -----------------------
        # Training functions
        # -----------------------

        st.subheader("3. Training Functions")
        
        st.markdown(r"""
        **Full Batch Gradient Descent**
        """)
        st.code("""
        for _ in range(epochs):
                y_pred = predict(X, w, b)
                dw = np.mean(2 * X * (y_pred - y))
                db = np.mean(2 * (y_pred - y))
                w -= lr * dw
                b -= lr * db
        """, language="python")
        
        st.markdown("""
        **Stochastic Gradient Descent**
        """)
        st.code("""
        for _ in range(epochs):
                indices = np.random.permutation(N)
                losses = []
                for i in indices:
                    xi, yi = X[i], y[i]
                    y_pred = predict(xi, w, b)
                    dw = 2 * xi * (y_pred - yi)
                    db = 2 * (y_pred - yi)
                    w -= lr * dw
                    b -= lr * db
        """, language="python")

        # -----------------------
        # Run training + Plot
        # -----------------------

        st.subheader("4. Loss History")
        st.markdown("Notice how **full-batch GD** yields a smooth, stable curve, while **SGD** is noisier but converges quickly.")
        st.image("assets/gd_vs_sgd.png", caption="Gradient Descent vs. Stochastic Gradient Descent", use_container_width=True)

    st.header("Feedforward Networks")
    
    with open("data/results.pkl", "rb") as f:
        results = pickle.load(f) 
    
    st.markdown(r"""
    For the following demonstrations we solve the classic classification task of MNIST handwritten digits using variations of **Mini-batch Stochastic Gradient Descent** with various optimizers in Pytorch trained on a T4 GPU in Google Colab.
    """)
    
    st.subheader("Code")
    with st.expander("show/hide"):
        st.subheader("1. Data")
        st.markdown("We download the built-in MNIST handwritten dataset from torchvision, transforming them into tensor format and load them as batches in Pytorch's Dataloader")
        st.code("""
tfm = transforms.ToTensor()
train_set = datasets.MNIST(root="data", train=True,  download=True, transform=tfm)
test_set  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

train_loader = DataLoader(train_set, batch_size=128, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=(device=="cuda"))

test_loader  = DataLoader(test_set,  batch_size=1024, shuffle=False,
                        num_workers=num_workers, pin_memory=(device=="cuda"))
            """, language="python")
        
        st.subheader("2. Basic MLP")
        st.markdown("We structure a basic MLP meant to take in the flattened $28 \times 28$ tensor image and output the logits for the 10 classes")
        st.code("""
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)   # 784 -> 256
        self.fc2 = nn.Linear(256, 128)     # 256 -> 128
        self.fc3 = nn.Linear(128, 10)      # 128 -> 10 (digits)

    def forward(self, x):
        x = self.flatten(x)        # [batch, 1, 28, 28] -> [batch, 784]
        x = F.relu(self.fc1(x))    # hidden layer 1
        x = F.relu(self.fc2(x))    # hidden layer 2
        logits = self.fc3(x)       # output logits
        return logits              # raw logits (no softmax here)

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)  # probabilities for inference
            """, language="python")
        
        st.subheader("3. Training Loop")
        st.markdown("We train and evaluate the model storing the train and validation loss history")
        st.code("""
def train(model, optimizer, train_loader, val_loader, epochs=5):
    model.to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "time": None}
    start_time = time.time()  # start timer
    for epoch in range(epochs):
      model.train()
      total_loss, total_correct, total = 0.0, 0, 0

      for x, y in train_loader:
          x, y = x.to(device), y.to(device)
          logits = model(x)
          loss = F.cross_entropy(logits, y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          total_loss += loss.item() * y.size(0)
          total_correct += (logits.argmax(1) == y).sum().item()
          total += y.size(0)

      train_loss = total_loss / total
      train_acc = total_correct / total
      val_loss, val_acc = evaluate(model, val_loader)

      history["train_loss"].append(train_loss)
      history["train_acc"].append(train_acc)
      history["val_loss"].append(val_loss)
      history["val_acc"].append(val_acc)
      
@torch.no_grad()                                         # override gradients to not update
def evaluate(model, loader):
  model.eval()
  total_loss, total_correct, total = 0.0, 0, 0
  for x, y in loader:
    x, y = x.to(device), y.to(device)
    logits = model(x)                                     # logits.shape() → [batch, classes]
    loss = F.cross_entropy(logits, y, reduction="sum")    # get total loss of batch
    total_loss += loss.item()                             # tensor → float then add
    total_correct += (logits.argmax(1) == y).sum().item() # boolean operation checking if the argmax is same index as correct class
    total += y.size(0)                                    # add number of items in batch to total
  return total_loss/total, total_correct/total            # return mean loss
            """, language="python")
        
        st.subheader("4. Usage")
        st.markdown("Below is our usage as we train and evaluate our MNIST classification model, storing the results in a dictionary. Additionally we evaluate the test set with a confusion matrix")
        st.code("""
train_loader, test_loader = make_loaders(regime="minibatch", batch_size=128)
model = MLP().to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.05)  # classic mini-batch SGD
hist = train(model, opt, train_loader, test_loader, epochs=10)
cm_data = collect_confusion_matrix(model, test_loader)
results["Vanilla SGD"] = {**hist, **cm_data}
            """, language="python")
        
        st.subheader("5. Visualizations")
        st.markdown("We can visualize our train and validation loss history per epoch as well as the confusion matrix performance on the test set")
        st.code("""
plot_losses(results)
plot_confusion_matrix(results, key="Vanilla SGD", normalize=True)
            """, language="python")
        
    st.subheader("Compare Loss Profile")
    st.markdown("We compare the training loss profile of each optimizer to compare how they converge on our MNIST classification problem.")
    selected_opts = st.multiselect(
    "Select optimizers to compare:",
    list(results.keys()),
    default=list(results.keys())[:2]  # pre-select first two
    )

    fig, ax = plt.subplots()

    for opt in selected_opts:
        history = results[opt]
        ax.plot(history["train_loss"], label=opt)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training Loss")
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Compare Confusion Matrices")

    selected_opts_cm = st.multiselect(
        "Select optimizers to compare confusion matrices:",
        list(results.keys()),
        default=list(results.keys())[:2]
    )

    cols = st.columns(2)  # 2-column layout

    for i, opt in enumerate(selected_opts_cm):
        if "confusion_matrix" in results[opt]:
            cm = results[opt]["confusion_matrix"].astype(float)
            # normalize per row (true class)
            cm = cm / cm.sum(axis=1, keepdims=True)

            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)  # consistent scale
            ax.set_title(f"{opt}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            # show values inside cells with smaller font
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, f"{cm[r, c]:.2f}",
                            ha="center", va="center", color="black", fontsize=6)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # send to correct column
            cols[i % 2].pyplot(fig)
        else:
            st.warning(f"No confusion matrix found for {opt}")
    
    st.subheader("Optimizer Theory")
    
    optimizer_choice = st.radio(
        "Select an optimizer to learn more about:",
        list(results.keys())
    )

    if optimizer_choice == "Vanilla SGD":
        st.markdown(r"""
        #### **Stochastic Gradient Descent (SGD)**  
        **Paper**: Robbins & Monro, 1951  
        
        **Formula**:  
        $$
        \theta_{t+1} = \theta_t - \eta \nabla L_i(\theta_t)
        $$
        
        **Explanation**:  
        Vanilla SGD updates parameters using the gradient of the loss on one sample or mini-batch.  
        It is simple and widely used, but it converges slowly, is sensitive to the learning rate, and can oscillate in ravines.
        """)

    elif optimizer_choice == "SGD + Momentum":
        st.markdown(r"""
        #### **SGD with Momentum**  
        **Paper**: Polyak, 1964  
        
        **Formula**:  
        $$
        v_t = \mu v_{t-1} - \eta \nabla L_t \\
        \theta_{t+1} = \theta_t + v_t
        $$
        
        **Explanation**:  
        Momentum improves upon Stochastic Gradient Descent (SGD) by adding a "velocity" term that accumulates past gradients, helping to accelerate convergence, dampen oscillations, and escape local minima by maintaining a consistent direction and building speed in consistent gradient directions. This smoother, faster trajectory leads to more stable training and often better generalization in complex loss landscapes.
        """)

    elif optimizer_choice == "Adagrad":
        st.markdown(r"""
        #### **Adagrad (Adaptive Gradient)**  
        **Paper**: Duchi, Hazan & Singer, 2011  
        
        **Formula**:  
        $$
        G_t = \sum_{\tau=1}^t (\nabla L_\tau)^2 \\
        \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \nabla L_t
        $$
        
        **Explanation**:  
        Adagrad improves upon Stochastic Gradient Descent (SGD) by dynamically adapting the learning rate for each parameter based on a sum of its past squared gradients.
        This "adaptive" nature allows Adagrad to assign larger updates to infrequent parameters (large gradients) and smaller updates to frequent ones (small gradients), leading to faster and more effective convergence, particularly in tasks with sparse data such as natural language processing and image recognition.
        """)

    elif optimizer_choice == "RMSprop":
        st.markdown(r"""
        #### **RMSProp**  
        **Paper**: Geoffrey Hinton, Coursera Lecture 2012  
        
        **Formula**:  
        $$
        v_t = \beta v_{t-1} + (1-\beta)(\nabla L_t)^2 \\
        \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla L_t
        $$
        
        **Explanation**:  
        RMSProp fixes Adagrad’s decaying learning rate problem by using an **exponential moving average** of past squared gradients.  
        preventing excessive learning rate decay and allowing it to adapt better by focusing on recent gradients. As a result, RMSProp generally converges faster and is more robust than Adagrad.
        It works especially well for RNNs and non-stationary objectives.
        """)

    elif optimizer_choice == "Adam":
        st.markdown(r"""
        #### **Adam (Adaptive Moment Estimation)**  
        **Paper**: Kingma & Ba, 2015  
        
        **Formula**:  
        $$
        m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L_t \\
        v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L_t)^2 \\
        \hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
        \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
        $$
        
        **Explanation**:  
        Adam combines **Momentum** (first moment, $m_t$) and **RMSProp** (second moment, $v_t$).  
        It adaptively adjusts learning rates and accelerates convergence.  
        Compared to previous optimizers, Adam is less sensitive to hyperparameter tuning and works well across many tasks.
        """)

    elif optimizer_choice == "AdamW":
        st.markdown(r"""
        #### **AdamW**  
        **Paper**: Loshchilov & Hutter, 2017  
        
        **Formula**:  
        Adam update with **decoupled weight decay**:  
        $$
        \theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
        $$
        
        **Explanation**:  
        AdamW improves Adam by **decoupling weight decay from the gradient update**.  
        In vanilla Adam, L2 regularization is implemented as part of the adaptive gradient step, which can lead to poor generalization.  
        AdamW separates weight decay into its own step, yielding better performance in practice — especially for large models like Transformers.
        """)
    

if section == "Overview":
    st.title("Overview")
    st.markdown(r"""
       
    ### 1958 — Perceptron  
    - **Paper:** *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*  
    - **Author:** Frank Rosenblatt  
    - **Application:** Introduced the first trainable linear classifier; inspired the birth of neural networks.  

    ---

    ### 1969 — Limits of the Perceptron  
    - **Paper:** *Perceptrons*  
    - **Authors:** Marvin Minsky & Seymour Papert  
    - **Application:** Showed single-layer perceptrons cannot solve nonlinear problems like XOR, leading to the first AI winter.  

    ---

    ### 1986 — Backpropagation, Sigmoid, Batch GD, and SGD  
    - **Paper:** *Learning Representations by Back-Propagating Errors*  
    - **Authors:** David Rumelhart, Geoffrey Hinton, Ronald Williams  
    - **Application:** Popularized efficient gradient-based training for multi-layer perceptrons, using the logistic sigmoid activation and introducing both batch gradient descent (full dataset updates) and stochastic gradient descent (single-sample updates).  

    ---
    
    ### 1989 — Tanh Activation  
    - **Paper:** *Backpropagation Applied to Handwritten Zip Code Recognition*  
    - **Authors:** Yann LeCun, Léon Bottou, Genevieve Orr, Klaus-Robert Müller  
    - **Application:** Popularized tanh as a zero-centered alternative to sigmoid, improving optimization for feedforward nets

    ---

    ### 1989 — Universal Approximation Theorem  
    - **Paper:** *Approximation by Superpositions of a Sigmoidal Function*  
    - **Author:** George Cybenko  
    - **Application:** Proved that a single hidden layer MLP can approximate any continuous function.  

    ---

    ### 1990 — Softmax for Neural Nets  
    - **Paper:** *Training Stochastic Model Recognition Algorithms as Networks Can Lead to Maximum Mutual Information Estimation of Parameters*  
    - **Author:** John S. Bridle  
    - **Application:** Introduced softmax outputs with cross-entropy loss for classification tasks.  

    ---

    ### 1992 — Weight Decay (L2 Regularization)  
    - **Paper:** *A Simple Weight Decay Can Improve Generalization*  
    - **Authors:** Krogh & Hertz  
    - **Application:** Popularized L2 penalty to reduce overfitting in neural networks.  

    ---

    ### 1996 — LASSO (L1 Regularization)  
    - **Paper:** *Regression Shrinkage and Selection via the Lasso*  
    - **Author:** Robert Tibshirani  
    - **Application:** Introduced L1 penalty to encourage sparsity in weights.  

    ---
    
    ### 1998 — “Efficient BackProp” (Tanh Recommendation)
    - **Paper:** *Efficient BackProp*
    - **Authors:** Yann LeCun, Léon Bottou, Genevieve B. Orr, Klaus-Robert Müller
    - **Application:** Systematized practical guidance (incl. preferring tanh over sigmoid), initialization, and optimization tips for MLPs.

    ---

    ### 1998 — Early Stopping  
    - **Paper:** *Early Stopping — But When?*  
    - **Author:** Lutz Prechelt  
    - **Application:** Formalized early stopping as a practical regularization strategy.  

    ---

    ### 2010 — Xavier/Glorot Initialization  
    - **Paper:** *Understanding the Difficulty of Training Deep Feedforward Neural Networks*  
    - **Authors:** Xavier Glorot & Yoshua Bengio  
    - **Application:** Proposed variance-preserving initialization to stabilize training.  

    ---

    ### 2011 — ReLU Activation  
    - **Paper:** *Deep Sparse Rectifier Neural Networks*  
    - **Authors:** Xavier Glorot, Antoine Bordes, Yoshua Bengio  
    - **Application:** Introduced ReLU, reducing vanishing gradients and enabling deeper networks.  

    ---

    ### 2011 — AdaGrad Optimizer  
    - **Paper:** *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*  
    - **Authors:** John Duchi, Elad Hazan, Yoram Singer  
    - **Application:** Adaptive learning rate per parameter, useful for sparse features.  

    ---

    ### 2012 — RMSProp Optimizer  
    - **Paper:** Hinton’s Coursera Lecture Notes (unpublished)  
    - **Author:** Geoffrey Hinton  
    - **Application:** Introduced exponentially decaying average of squared gradients, improving training stability.  

    ---

    ### 2012 — Dropout Regularization  
    - **Paper:** *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*  
    - **Authors:** Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov  
    - **Application:** Randomly drops units during training, reducing overfitting.  

    ---

    ### 2013 — Importance of Momentum & Initialization  
    - **Paper:** *On the Importance of Initialization and Momentum in Deep Learning*  
    - **Authors:** Sutskever, Martens, Dahl, Hinton  
    - **Application:** Highlighted momentum’s role in accelerating SGD and avoiding poor local minima.  

    ---
    
    ### 2013 — Leaky ReLU
    - **Paper:** *Rectifier Nonlinearities Improve Neural Network Acoustic Models*
    - **Authors:** Andrew L. Maas, Awni Y. Hannun, Andrew Y. Ng
    - **Application:** Introduced a small negative slope to keep gradients alive when activations are negative.

    ---

    ### 2014 — AdaDelta Optimizer  
    - **Paper:** *ADADELTA: An Adaptive Learning Rate Method*  
    - **Author:** Matthew Zeiler  
    - **Application:** Adaptive learning rate method requiring no manual tuning of global learning rate.  

    ---
    
    ### 2015 — ELU Activation
    - **Paper:** *Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)*
    - **Authors:** Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter
    - **Application:** Negative values with smoother saturation reduce bias shift and can speed up training.

    ---

    ### 2015 — He Initialization & PReLU  
    - **Paper:** *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*  
    - **Authors:** Kaiming He et al.  
    - **Application:** Proposed initialization tailored to ReLU and introduced parametric ReLU variants.  

    ---

    ### 2015 — Batch Normalization  
    - **Paper:** *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*  
    - **Authors:** Sergey Ioffe & Christian Szegedy  
    - **Application:** Stabilizes and accelerates training by normalizing activations within mini-batches.  

    ---

    ### 2015 — Adam Optimizer  
    - **Paper:** *Adam: A Method for Stochastic Optimization*  
    - **Authors:** Diederik Kingma & Jimmy Ba  
    - **Application:** Combines momentum and adaptive learning rates; widely adopted default optimizer.  

    ---

    ### 2016 — GELU Activation  
    - **Paper:** *Gaussian Error Linear Units (GELUs)*  
    - **Authors:** Dan Hendrycks & Kevin Gimpel  
    - **Application:** Smooth activation that combines benefits of ReLU and sigmoid; used in Transformers.  

    ---
    
    ### 2016 — Layer Normalization
    - **Paper:** *Layer Normalization*
    - **Authors:** Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    - **Application:** Normalizes per feature vector (not per batch), removing batch-size dependence; widely used in Transformers and MLPs.

    ---
    
    ### 2016 — Nesterov-Accelerated Adam (Nadam)
    - **Paper:** *Incorporating Nesterov Momentum into Adam*
    - **Author:** Timothy Dozat
    - **Application:** Combines Adam’s adaptivity with Nesterov momentum; a minor but sometimes helpful tweak.

    ---

    ### 2016 — Label Smoothing  
    - **Paper:** *Rethinking the Inception Architecture for Computer Vision*  
    - **Authors:** Christian Szegedy et al.  
    - **Application:** Regularization method that prevents models from becoming overconfident in classification.  

    ---

    ### 2017 — Cyclical Learning Rates  
    - **Paper:** *Cyclical Learning Rates for Training Neural Networks*  
    - **Author:** Leslie N. Smith  
    - **Application:** Varying learning rate cyclically improves convergence and generalization.  

    ---

    ### 2017 — SGDR (Cosine Annealing with Restarts)  
    - **Paper:** *SGDR: Stochastic Gradient Descent with Warm Restarts*  
    - **Authors:** Ilya Loshchilov & Frank Hutter  
    - **Application:** Popularized cosine annealing with scheduled restarts for improved training.  

    ---

    ### 2017 — AdamW (Decoupled Weight Decay)  
    - **Paper:** *Fixing Weight Decay Regularization in Adam*  
    - **Authors:** Ilya Loshchilov & Frank Hutter  
    - **Application:** Corrected Adam’s handling of weight decay, improving generalization.  

    ---

    ### 2017 — Swish / SiLU Activation  
    - **Paper:** *Searching for Activation Functions*  
    - **Authors:** Ramachandran, Zoph, Le  
    - **Application:** Smooth, self-gated activation; competitive with ReLU/GELU.  

    ---

    ### 2018+ — Modern Refinements  
    - **Application:** MLPs remain core components in Transformers and deep learning systems, often paired with GELU/Swish activations, AdamW optimization, and cosine LR schedules.
    """)
    
if section == "Multi-Layer Perceptrons":
    st.title("Multi-Layer Perceptrons")
    
    st.header("Summary")
    st.markdown(r"""
    The **Multi-Layer Perceptron (MLP)** is the foundation of modern deep learning, evolving from the earliest models of artificial neurons.  

    - **Perceptron (Rosenblatt, 1958):** A model of an artificial neuron that computes a weighted sum of inputs, passes it through a threshold function, and outputs a binary decision. This inspired the perceptron, which is essentially a **linear classifier**.  
    - **Limitations of Single-Layer Models (Minsky & Papert, 1969):** A single perceptron cannot represent nonlinear functions like XOR. This limitation contributed to the first "AI Winter."  
    - **Backpropagation Breakthrough (Rumelhart, Hinton & Williams, 1986):** The introduction of backpropagation, combined with gradient descent, made training **multi-layer networks** feasible by efficiently computing derivatives via the chain rule.  
    - **Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991):** A neural network with one hidden layer and sufficient width can approximate any continuous function. However, this relies on adding more neurons in a single layer, which is **inefficient** compared to using depth.  
    - **Modern MLPs:**  
        - Built as **stacks of fully connected layers**, each neuron initialized randomly to **break symmetry**.  
        - **Nonlinear activation functions** (e.g., ReLU, Tanh) prevent collapse into a single linear transformation.  
        - **Backpropagation + gradient descent** form the backbone of training.  
        - **Deeper vs. Wider Networks:**  
            - Width → number of linear regions grows **polynomially**.  
            - Depth → number of linear regions grows **exponentially** (Montúfar et al., 2014).  

    
    """)
    
    st.image("assets/neural_network.png", caption="Depth vs Width", use_container_width=True)
    
    st.subheader("Minimal PyTorch MLP")
    st.markdown(r"""
    In research we commonly use **PyTorch** to implement MLPs. An MLP is a stack of **fully connected (linear) layers** with **nonlinear activations** in between:

    $$\text{Input} \;\xrightarrow{\;W^{(1)},\,b^{(1)};\ \phi\;}\; \text{Hidden}_1 \;\xrightarrow{\;W^{(2)},\,b^{(2)};\ \phi\;}\; \cdots \;\xrightarrow{\;W^{(L)},\,b^{(L)}\;}\; \text{Output}.$$

    - **Depth** = number of layers (how many times we stack).  
    - **Width** = neurons per layer.  
    - **Tip:** For binary classification, return **logits** (no sigmoid in `forward`) and use `BCEWithLogitsLoss` for stability.

    **Imports (what they’re for):**
    - "_`super().__init__()` calls the base **nn.Module** constructor so PyTorch can register your submodules/parameters and enable core features like `.parameters()`, `.to()`, `.train()/.eval()`, `.state_dict()`, and hooks._"
    - `torch.nn`: stateful building blocks for neural nets (e.g., `nn.Linear`, `nn.Conv2d`, loss modules); holds learnable parameters.
    - `torch.nn.functional` (aliased `F`): stateless functional ops (e.g., `F.relu`, `F.softmax`) used inside `forward` when no parameters/buffers are needed.
    """)
    
    st.code('''
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleMLP(nn.Module):
        def __init__(self, in_features=3, hidden=3):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden)  # one linear
            self.fc2 = nn.Linear(hidden, hidden)       # second linear so the 2nd ReLU does work
            self.fc3 = nn.Linear(hidden, 1)            # output to 1 logit

        def forward(self, x):
            x = F.relu(self.fc1(x))   # one ReLU
            x = F.relu(self.fc2(x))   # one ReLU
            x = torch.sigmoid(self.fc3(x))  # output sigmoid
        return x
    ''', language='python')
    
if section == "Backpropagation":
    st.title("Backpropagation")
    st.markdown(r"""
    We present the backpropagation algorithm which enabled the training of deep neural networks via the chain rule.

    1. **Initialization** → break symmetry & set variance scales  
    2. **Forward Pass (Activations)** → compute hidden features & output  
    3. **Evaluation (Criterion / Loss)** → scalar objective \(L\)  
    4. **Backpropagation (Chain Rule)** → compute $\frac{\partial L}{\partial \theta}$ for **every** parameter $\theta$.  
    5. **Parameter Update** → gradient descent / optimizer step
    """)
    
    st.subheader("1) Initialization")
    st.markdown(r"""
    - **Purpose:** start neurons differently (**break symmetry**) and keep magnitudes stable so gradients don’t explode/vanish.  
    - **Practically:** random weights with sensible scaling; frameworks often default to ReLU-friendly scaling by design.
    """)

    st.subheader("2) Forward Pass")
    st.markdown(r"""
    We use a **2-hidden-layer MLP (width = 3)** with a **binary output** as an example.

    **Shapes**  
    - Input: $\mathbf{x}\in\mathbb{R}^3$  
    - Hidden 1: $W^{(1)}\in\mathbb{R}^{3\times 3},\ \mathbf{b}^{(1)}\in\mathbb{R}^{3}$  
    - Hidden 2: $W^{(2)}\in\mathbb{R}^{3\times 3},\ \mathbf{b}^{(2)}\in\mathbb{R}^{3}$  
    - Output: $W^{(3)}\in\mathbb{R}^{1\times 3},\ b^{(3)}\in\mathbb{R}$

    **Explicit matrices and vectors**

    $$
    \mathbf{x}=
    \begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix},\quad
    W^{(1)}=
    \begin{bmatrix}
    w^{(1)}_{11}&w^{(1)}_{12}&w^{(1)}_{13}\\
    w^{(1)}_{21}&w^{(1)}_{22}&w^{(1)}_{23}\\
    w^{(1)}_{31}&w^{(1)}_{32}&w^{(1)}_{33}
    \end{bmatrix},\quad
    \mathbf{b}^{(1)}=
    \begin{bmatrix}b^{(1)}_1\\b^{(1)}_2\\b^{(1)}_3\end{bmatrix}
    $$

    Layer 1 (elementwise activation $\phi$):
    $$
    \mathbf{a}^{(1)}=W^{(1)}\mathbf{x}+\mathbf{b}^{(1)},\qquad
    \mathbf{h}^{(1)}=\phi(\mathbf{a}^{(1)})
    $$

    Layer 2:
    $$
    \mathbf{a}^{(2)}=W^{(2)}\mathbf{h}^{(1)}+\mathbf{b}^{(2)},\qquad
    \mathbf{h}^{(2)}=\phi(\mathbf{a}^{(2)})
    $$

    Output (single logit) and optional link $\psi$ (e.g., sigmoid):
    $$
    W^{(3)}=
    \begin{bmatrix}w^{(3)}_{11}&w^{(3)}_{12}&w^{(3)}_{13}\end{bmatrix},\qquad
    z = W^{(3)}\mathbf{h}^{(2)} + b^{(3)},\qquad
    \hat{y}=\psi(z)
    $$
    """)

    st.subheader("3) Criterions")
    st.markdown(r"""
    - **Purpose:** convert predictions to a **single scalar** $L$ the optimizer can minimize.  
    - **Binary classification:** Binary Cross-Entropy (often with logits).  
    - **Softmax**: Multi-class classification with logits
    - **Regression / real-valued targets:** Mean Squared Error.
    """)
    
    
    st.subheader("4) Backpropagation (Chain Rule)")
    st.markdown(r"""
    **Setup**
    - Input: $\mathbf{x}\in\mathbb{R}^3$
    - Hidden 1: $\mathbf{a}^{(1)}=W^{(1)}\mathbf{x}+\mathbf{b}^{(1)}$,  $\mathbf{h}^{(1)}=\phi(\mathbf{a}^{(1)})$
    - Hidden 2: $\mathbf{a}^{(2)}=W^{(2)}\mathbf{h}^{(1)}+\mathbf{b}^{(2)}$,  $\mathbf{h}^{(2)}=\phi(\mathbf{a}^{(2)})$
    - Output (single logit): $z=W^{(3)}\mathbf{h}^{(2)}+b^{(3)}$,  $\hat{y}=\psi(z)$  
    Loss: $L(\hat{y},y)$ is a scalar.

    ---

    ### Step 1 — Output layer error $\delta^{(3)}$
    **BCE + sigmoid** ($\hat{y}=\sigma(z)$, $L=-[y\log\hat{y}+(1-y)\log(1-\hat{y})]$)

    $$
    \frac{\partial L}{\partial \hat{y}}=-\frac{y}{\hat{y}}+\frac{1-y}{1-\hat{y}},
    \qquad
    \frac{\partial \hat{y}}{\partial z}=\hat{y}(1-\hat{y})
    $$

    **Cancellation to the classic form**
    $$
    \frac{\partial L}{\partial z}
    =
    \frac{\partial L}{\partial \hat{y}}\cdot\frac{\partial \hat{y}}{\partial z}
    =
    \Big(-\frac{y}{\hat{y}}+\frac{1-y}{1-\hat{y}}\Big)\hat{y}(1-\hat{y})
    =
    \hat{y}-y
    $$

    Define the output error:
    $$
    \boxed{\ \delta^{(3)} \equiv \frac{\partial L}{\partial z} = \hat{y}-y\ }
    $$

    **Gradients for last layer**
    $$
    \boxed{\ \frac{\partial L}{\partial W^{(3)}}= \frac{\partial L}{\partial z}\cdot \frac{\partial z}{\partial W}=\delta^{(3)}(\mathbf{h}^{(2)})^\top\ },\qquad
    \boxed{\ \frac{\partial L}{\partial b^{(3)}}= \frac{\partial L}{\partial z}\cdot \frac{\partial z}{\partial b} = \delta^{(3)}\ }
    $$

    We will need the sensitivity wrt the hidden output next:
    $$
    \boxed{\ \frac{\partial L}{\partial \mathbf{h}^{(2)}}= \frac{\partial z}{\partial h^{(2)}} \cdot \frac{\partial L}{\partial z} = (W^{(3)})^\top\,\delta^{(3)}\ }\quad(\in\mathbb{R}^3)
    $$

    ---

    ### Step 2 — Backprop to hidden layer 2 
    Hidden output sensitivity:
    $$
    \boxed{\ \frac{\partial L}{\partial \mathbf{h}^{(2)}}=(W^{(3)})^\top\,\delta^{(3)}\ }
    $$

    Convert to pre-activation sensitivity via the activation derivative (elementwise):
    $$
    \boxed{\ \boldsymbol{\delta}^{(2)} \equiv \frac{\partial L}{\partial \mathbf{a}^{(2)}}=
    \Big(\frac{\partial L}{\partial \mathbf{h}^{(2)}}\Big)\ \odot\ \frac{\partial h^{(2)}}{\partial a^{(2)}}
    =
    \big((W^{(3)})^\top \delta^{(3)}\big)\ \odot\ \phi'(\mathbf{a}^{(2)})\ }
    $$

    Parameter gradients for layer 2:
    $$
    \boxed{\ \frac{\partial L}{\partial W^{(2)}}=\frac{\partial L}{\partial a^{(2)}}\cdot\frac{\partial a^{(2)}}{\partial W^{(2)}}=\boldsymbol{\delta}^{(2)}(\mathbf{h}^{(1)})^\top\ },\qquad
    \boxed{\ \frac{\partial L}{\partial \mathbf{b}^{(2)}}=\frac{\partial L}{\partial a^{(2)}}\cdot\frac{\partial a^{(2)}}{\partial b^{(2)}}=\boldsymbol{\delta}^{(2)}\ }
    $$

    We will also need the sensitivity wrt the previous hidden output:
    $$
    \boxed{\ \frac{\partial L}{\partial \mathbf{h}^{(1)}}=\frac{\partial L}{\partial a^{(2)}}\cdot\frac{\partial a^{(2)}}{\partial h^{(1)}}=(W^{(2)})^\top\,\boldsymbol{\delta}^{(2)}\ }\quad(\in\mathbb{R}^3)
    $$

    ---

    ### Step 3 — Backprop to hidden layer 1 (make $\partial L/\partial \mathbf{h}^{(1)}$ explicit)
    Hidden output sensitivity:
    $$
    \boxed{\ \frac{\partial L}{\partial \mathbf{h}^{(1)}}=(W^{(2)})^\top\,\boldsymbol{\delta}^{(2)}\ }
    $$

    Convert to pre-activation sensitivity:
    $$
    \boxed{\ \boldsymbol{\delta}^{(1)} \equiv \frac{\partial L}{\partial \mathbf{a}^{(1)}}=
    \Big(\frac{\partial L}{\partial \mathbf{h}^{(1)}}\Big)\ \odot\ \frac{\partial h^{(1)}}{\partial a^{(1)}})
    =
    \big((W^{(2)})^\top \boldsymbol{\delta}^{(2)}\big)\ \odot\ \phi'(\mathbf{a}^{(1)})\ }
    $$

    Parameter gradients for layer 1:
    $$
    \boxed{\ \frac{\partial L}{\partial W^{(1)}}=\frac{\partial L}{\partial \mathbf{h}^{(1)}}\cdot\frac{\partial h^{(1)}}{\partial \mathbf{W}^{(1)}}=\boldsymbol{\delta}^{(1)}\mathbf{x}^\top\ },\qquad
    \boxed{\ \frac{\partial L}{\partial \mathbf{b}^{(1)}}=\frac{\partial L}{\partial \mathbf{h}^{(1)}}\cdot\frac{\partial h^{(1)}}{\partial \mathbf{b}^{(1)}}=\boldsymbol{\delta}^{(1)}\ }
    $$

    (Optional) Input sensitivity (saliency/attribution):
    $$
    \boxed{\ \frac{\partial L}{\partial \mathbf{x}}=\frac{\partial L}{\partial \mathbf{a}^{(1)}}\cdot\frac{\partial a^{(1)}}{\partial \mathbf{x}^{(1)}}=(W^{(1)})^\top\,\boldsymbol{\delta}^{(1)}\ }
    $$

    ---

    ### Chain rule expanded once (then collapsed)
    For any element $w^{(1)}_{ji}$ in $W^{(1)}$:
    $$
    \frac{\partial L}{\partial w^{(1)}_{ji}}
    =
    \underbrace{\frac{\partial L}{\partial z}}_{\delta^{(3)}}
    \cdot
    \underbrace{\frac{\partial z}{\partial \mathbf{h}^{(2)}}}_{W^{(3)}}
    \cdot
    \underbrace{\frac{\partial \mathbf{h}^{(2)}}{\partial \mathbf{a}^{(2)}}}_{\mathrm{diag}(\phi'(\mathbf{a}^{(2)}))}
    \cdot
    \underbrace{\frac{\partial \mathbf{a}^{(2)}}{\partial \mathbf{h}^{(1)}}}_{W^{(2)}}
    \cdot
    \underbrace{\frac{\partial \mathbf{h}^{(1)}}{\partial a^{(1)}_j}}_{\phi'(a^{(1)}_j)}
    \cdot
    \underbrace{\frac{\partial a^{(1)}_j}{\partial w^{(1)}_{ji}}}_{x_i}
    $$

    Grouping terms:
    $$
    \frac{\partial L}{\partial w^{(1)}_{ji}}
    =
    \underbrace{\Big((W^{(2)})^\top\big((W^{(3)})^\top \delta^{(3)}\big)\ \odot\ \phi'(\mathbf{a}^{(1)})\Big)_j}_{\delta^{(1)}_j}
    \cdot x_i
    \quad\Longrightarrow\quad
    \boxed{\ \frac{\partial L}{\partial W^{(1)}}=\boldsymbol{\delta}^{(1)}\mathbf{x}^\top\ }
    $$

    """)



    st.subheader("5) Parameter Update (what happens)")
    st.markdown(r"""
        **Instead of the classic perceptron update rule**, which uses a non-differentiable step and only updates on mistakes, **we use gradient descent on a differentiable loss $L$**. This updates **every parameter, every step**, using its own gradient

                
    - **All layers update once their gradients are known** (mini-batch or per-example):
    $$
    W^{(\ell)} \leftarrow W^{(\ell)} - \eta\,\frac{\partial L}{\partial W^{(\ell)}},\qquad
    \mathbf{b}^{(\ell)} \leftarrow \mathbf{b}^{(\ell)} - \eta\,\frac{\partial L}{\partial \mathbf{b}^{(\ell)}}
    $$

    - Optimizers (SGD with momentum, Adam, etc.) keep the same principle — **move opposite the gradient**  
    but modify the step using additional state (e.g., momentum, adaptive learning rates).
    """)
    
    st.subheader("Backpropagation (PyTorch implementation)")
    st.markdown(r"""
    Below is a **single training step** in PyTorch. It follows the pipeline:

    1) **Forward**: `output = model(x)`  
    2) **Evaluate** (criterion): `loss = criterion(logits, y)`  
    3) **Backward** (chain rule via autograd): `loss.backward()`  
    4) **Update** (gradient descent): `optimizer.step()`  

    > Autograd computes **all** $\frac{\partial L}{\partial \theta}$ for every parameter in every layer during `loss.backward()`.  
    > The optimizer then applies $ \theta \leftarrow \theta - \eta\,\frac{\partial L}{\partial \theta}$ to **all** parameters.
    """)
    
    st.subheader("Backpropagation (Training Loop Skeleton)")
    st.code('''
    # Assumes: model, criterion, optimizer, train_loader, num_epochs, device
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # 1) Forward
            outputs = model(x)

            # 2) Compute loss
            loss = criterion(outputs, y)

            # 3) Backward (autograd computes all ∂L/∂θ via chain rule)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 4) Update (gradient descent / optimizer step on ALL parameters)
            optimizer.step()
    ''', language='python')

if section == "Initializations":
    st.title("Initializations")

    st.markdown(r"""
    **Why initialization matters:** it breaks symmetry between neurons and keeps signal/gradient magnitudes stable across layers so training doesn’t stall or blow up.

    Let $\mathrm{fan\_in}$ = #inputs to a neuron and $\mathrm{fan\_out}$ = #outputs from a neuron.

    ### From early practice → modern defaults

    **LeCun (Efficient BackProp; 1998/2012)** — good for `tanh` / `SELU`  
    Variance target:
    $$
    \operatorname{Var}[W_{ij}] \;=\; \frac{1}{\mathrm{fan\_in}}
    $$
    Uniform / Normal parameterizations:
    $$
    W_{ij} \sim \mathcal{U}\!\Big[-\sqrt{\tfrac{3}{\mathrm{fan\_in}}},\; \sqrt{\tfrac{3}{\mathrm{fan\_in}}}\Big]
    \quad\text{or}\quad
    W_{ij} \sim \mathcal{N}\!\Big(0,\; \tfrac{1}{\mathrm{fan\_in}}\Big)
    $$

    **Glorot/Xavier (AISTATS 2010)** — good for `tanh` / `sigmoid`  
    Variance target:
    $$
    \operatorname{Var}[W_{ij}] \;=\; \frac{2}{\mathrm{fan\_in}+\mathrm{fan\_out}}
    $$
    Uniform / Normal parameterizations:
    $$
    W_{ij} \sim \mathcal{U}\!\Big[-\sqrt{\tfrac{6}{\mathrm{fan\_in}+\mathrm{fan\_out}}},\; \sqrt{\tfrac{6}{\mathrm{fan\_in}+\mathrm{fan\_out}}}\Big]
    \quad\text{or}\quad
    W_{ij} \sim \mathcal{N}\!\Big(0,\; \tfrac{2}{\mathrm{fan\_in}+\mathrm{fan\_out}}\Big)
    $$

    **He/Kaiming (Delving Deep into Rectifiers; 2015)** — for ReLU/LeakyReLU  
    General (LeakyReLU with negative slope $a$):
    $$
    \operatorname{Var}[W_{ij}] \;=\; \frac{2}{(1+a^2)\,\mathrm{fan\_in}}
    $$
    ReLU special case ($a=0$):
    $$
    \operatorname{Var}[W_{ij}] \;=\; \frac{2}{\mathrm{fan\_in}},\qquad
    W_{ij} \sim \mathcal{U}\!\Big[-\sqrt{\tfrac{6}{\mathrm{fan\_in}}},\; \sqrt{\tfrac{6}{\mathrm{fan\_in}}}\Big]
    \ \text{or}\ 
    \mathcal{N}\!\Big(0,\; \tfrac{2}{\mathrm{fan\_in}}\Big)
    $$

    **Orthogonal (Saxe et al., 2014)** — preserves directions/scale (deep linear/residual stacks)  
    Construct $Q$ via QR of a random Gaussian matrix, then:
    $$
    W \;=\; g\,Q,\qquad Q^\top Q = I
    $$
    ($g$ is a scalar gain, e.g., $g=\sqrt{2}$ for ReLU.)

    **SELU note (Klambauer et al., 2017)** — for self-normalizing nets use **LeCun normal** (+ AlphaDropout).

    **PyTorch defaults today:** `nn.Linear`/`nn.Conv2d` weights use a **Kaiming-uniform** style by default; biases are drawn as
    $$
    b_i \sim \mathcal{U}\!\Big(-\frac{1}{\sqrt{\mathrm{fan\_in}}},\; \frac{1}{\sqrt{\mathrm{fan\_in}}}\Big).
    $$
    """)

    st.subheader("One-liners in PyTorch")
    st.code('''
    import math
    import torch.nn as nn

    layer = nn.Linear(128, 64)  # example layer

    # --- LeCun (Efficient BackProp) ---
    # LeCun normal / uniform
    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")      # Var = 1/fan_in (normal)
    # nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")   # Var = 1/fan_in (uniform)

    # --- Glorot / Xavier (tanh/sigmoid) ---
    nn.init.xavier_uniform_(layer.weight)      # a = sqrt(6/(fan_in+fan_out))
    # nn.init.xavier_normal_(layer.weight)     # std = sqrt(2/(fan_in+fan_out))

    # --- He / Kaiming (ReLU/LeakyReLU) ---
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")  # a = 0 (ReLU), U[-sqrt(6/fan_in), +sqrt(6/fan_in)]
    # nn.init.kaiming_normal_(layer.weight, nonlinearity="relu") # std = sqrt(2/fan_in)

    # LeakyReLU with negative_slope=a:
    # nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu", a=0.01)

    # --- Orthogonal (gain depends on activation) ---
    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))

    # --- Bias init ---
    nn.init.zeros_(layer.bias)
    # Or mimic default uniform bound:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(layer.bias, -bound, bound)
    ''', language='python')

if section == "Activations":
    st.title("Activations")

    # --- Feedforward Networks: Activations (PyTorch, structured) ---

    # ---- plot settings ----
    xmin, xmax, npts = -6.0, 6.0, 1000
    show_deriv = st.checkbox("Show derivative", value=False)

    def make_grid():
        return torch.linspace(xmin, xmax, npts, requires_grad=show_deriv)

    def plot_activation(x, y, title, dy=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x.detach().numpy(), y.detach().numpy(), label=title)
        if dy is not None:
            ax.plot(x.detach().numpy(), dy.detach().numpy(), "--", label=f"d/dx {title}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        st.pyplot(fig)

    def forward_and_deriv(f):
        x = make_grid()
        y = f(x)
        dy = None
        if show_deriv:
            y.sum().backward()
            dy = x.grad
        return x, y, dy

    # ========= Sigmoid =========
    st.markdown("### Sigmoid")
    st.markdown(r"""
    **History:** Popular in the **1980s–1990s** as a standard activation in early MLPs.  
    **Formula:** $\displaystyle \sigma(x)=\frac{1}{1+e^{-x}}$  
    **Derivative:** $\displaystyle \sigma'(x)=\sigma(x)(1-\sigma(x))$  
    **Notes:** saturates at large $|x|$ → vanishing gradients.  
    **When to use:** Rarely in hidden layers today; still common for **binary output probabilities**.
    """)
    x, y, dy = forward_and_deriv(torch.sigmoid)
    plot_activation(x, y, "sigmoid", dy)

    # ========= Tanh =========
    st.markdown("### Tanh")
    st.markdown(r"""
    **History:** 1980s–1990s; recommended in *LeCun, “Efficient BackProp” (1998)* for zero-centered activations.  
    **Formula:** $\tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$  
    **Derivative:** $1-\tanh^2(x)$  
    **Notes:** zero-centered but still saturates.  
    **When to use:** Sometimes in **RNNs** or when you want bounded, zero-centered outputs.
    """)
    x = make_grid()
    y = torch.tanh(x); dy=None
    if show_deriv: y.sum().backward(); dy=x.grad
    plot_activation(x, y, "tanh", dy)

    # ========= ReLU =========
    st.markdown("### ReLU")
    st.markdown(r"""
    **History:** Popularized by *Glorot & Bengio (2011)*.  
    **Formula:** $\operatorname{ReLU}(x)=\max(0,x)$  
    **Derivative:** $1$ if $x>0$, else $0$  
    **Notes:** Simple, fast, sparse; risk of “dead” neurons.  
    **When to use:** Default for **MLPs and CNNs**.
    """)
    x, y, dy = forward_and_deriv(F.relu)
    plot_activation(x, y, "relu", dy)

    # ========= Leaky ReLU =========
    st.markdown("### Leaky ReLU")
    st.markdown(r"""
    **History:** ~2013 (Maas et al.); PReLU (He et al., 2015).  
    **Formula:** $\operatorname{LReLU}(x)=\begin{cases}x,&x\ge0\\ \alpha x,&x<0\end{cases}$  
    **Derivative:** $1$ if $x\ge0$, else $\alpha$  
    **Notes:** prevents dead neurons by allowing small negative slope.  
    **When to use:** Drop-in replacement for ReLU if you see **dead units**.
    """)
    alpha=0.1
    x = make_grid()
    y = F.leaky_relu(x, negative_slope=alpha); dy=None
    if show_deriv: y.sum().backward(); dy=x.grad
    plot_activation(x, y, f"leaky_relu (α={alpha})", dy)

    # ========= ELU =========
    st.markdown("### ELU")
    st.markdown(r"""
    **History:** *Clevert et al., 2015*.  
    **Formula:** $\operatorname{ELU}(x)=\begin{cases}x,&x\ge0\\ \alpha(e^x-1),&x<0\end{cases}$  
    **Notes:** negative outputs help keep mean activations near zero.  
    **When to use:** If you want **zero-centered activations** and smoother gradients than ReLU.
    """)
    alpha_elu=1.0
    x = make_grid()
    y = F.elu(x, alpha=alpha_elu); dy=None
    if show_deriv: y.sum().backward(); dy=x.grad
    plot_activation(x, y, f"elu (α={alpha_elu})", dy)

    # ========= SELU =========
    st.markdown("### SELU")
    st.markdown(r"""
    **History:** *Klambauer et al., 2017* — “Self-Normalizing Networks”.  
    **Formula:** $\lambda x$ if $x\ge0$, else $\lambda\alpha(e^x-1)$.  
    **Notes:** designed to keep mean/variance stable with proper init.  
    **When to use:** Only in **self-normalizing networks** (special init + architecture).
    """)
    x, y, dy = forward_and_deriv(F.selu)
    plot_activation(x, y, "selu", dy)

    # ========= GELU =========
    st.markdown("### GELU")
    st.markdown(r"""
    **History:** *Hendrycks & Gimpel, 2016*; default in Transformers.  
    **Approx Formula:** $\tfrac12x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$  
    **Notes:** smooth, keeps small negatives with input-dependent gating.  
    **When to use:** **Transformers / modern NLP**; smoother alternative to ReLU.
    """)
    x, y, dy = forward_and_deriv(F.gelu)
    plot_activation(x, y, "gelu", dy)

    # ========= Softplus =========
    st.markdown("### Softplus")
    st.markdown(r"""
    **History:** *Dugas et al., 2001*.  
    **Formula:** $\operatorname{softplus}(x)=\log(1+e^x)$  
    **Derivative:** $\sigma(x)$  
    **Notes:** smooth ReLU; avoids sharp kink at 0.  
    **When to use:** If you need a **differentiable ReLU** (e.g., probabilistic models).
    """)
    x, y, dy = forward_and_deriv(F.softplus)
    plot_activation(x, y, "softplus", dy)

    st.markdown("""
    ---
    **Quick guide:**  
    - **ReLU/GELU** → best general defaults.  
    - **LeakyReLU/ELU** → if ReLU dead neurons hurt.  
    - **Sigmoid/Tanh** → outputs or gates, not hidden layers.  
    - **SELU** → only if following the self-normalizing recipe.  
    - **Softplus** → smooth ReLU alternative.  
    """)
        
if section == "Criterions":
    st.title("Criterions")

    # =====================
    # Mean Squared Error
    # =====================
    st.header("Mean Squared Error (MSE)")

    st.markdown(r"""
    **Definition:**  
    The **Mean Squared Error (MSE)** measures the average squared difference between predicted and true values:
    """)

    st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

    st.image("assets/residuals.png", 
             caption="Residuals are the vertical differences between predicted values and actual data points. MSE averages their squared values.", 
             use_container_width=True)

    st.markdown(r"""
    - Squaring residuals prevents positive/negative errors from canceling.  
    - A common variant includes $\tfrac{1}{2}$ for simpler gradients:
    """)

    st.latex(r"J(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

    st.markdown(r"""
    **Gradients:**  

    - Original:
    $$
    \frac{\partial J}{\partial \theta_j} = \frac{2}{n}\sum_{i=1}^n (\hat{y}_i - y_i)x_{ij}
    $$
    - Modified:
    $$
    \frac{\partial J}{\partial \theta_j} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)x_{ij}
    $$

    → Same minimum, but cleaner update rule.
    """)

    # ---------------------
    # Statistical view
    # ---------------------
    st.subheader("Statistical Foundations")

    st.markdown(r"""
    **Why MSE?**  
    It arises naturally from a **Gaussian likelihood assumption** on the residuals.

    1. **Likelihood model**  
       $$
       y_i \sim \mathcal{N}(\hat{y}_i, \sigma^2)
       \quad\Rightarrow\quad
       p(y_i\mid x_i,\theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\Big(-\tfrac{(y_i-\hat{y}_i)^2}{2\sigma^2}\Big)
       $$

    2. **Log-likelihood for $n$ samples**  
       $$
       \log p(\mathbf{y}\mid X,\theta) = -\tfrac{n}{2}\log(2\pi\sigma^2)\;-\;\tfrac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\hat{y}_i)^2
       $$

    3. **Negative log-likelihood (NLL)**  
       Minimizing NLL $\;\propto\;\sum (y_i-\hat{y}_i)^2$.

    **Conclusion:**  
    Minimizing MSE $\;\Leftrightarrow\;$ Maximum Likelihood Estimation under Gaussian noise.
    """)
    
    st.divider()
    # =====================
    # Binary Cross-Entropy
    # =====================
    
    st.header("Binary Cross-Entropy (BCE)")

    # ------------------
    # 1. Definition
    # ------------------
    st.markdown(r"""
    **Definition:**  
    Binary Cross-Entropy (BCE) is the standard loss for **binary classification** tasks.

    Each label $y_i \in \{0,1\}$ is compared to the predicted probability $\hat{y}_i \in (0,1)$:
    """)

    st.latex(r"""
    \text{BCE} = -\frac{1}{n}\sum_{i=1}^n \Big[ y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \Big]
    """)

    # ------------------
    # 2. Statistical Foundations
    # ------------------
    st.subheader("Statistical Foundations")

    st.markdown(r"""
    **Bernoulli likelihood:**  
    We assume
    $$
    Y_i \sim \text{Bernoulli}(p_i), \quad p_i = \hat{y}_i = \sigma(z_i).
    $$
    Likelihood of all samples:
    $$
    \mathcal{L}(\beta) = \prod_{i=1}^n p_i^{y_i}(1-p_i)^{(1-y_i)}.
    $$
    Taking the **negative log-likelihood**:
    $$
    -\log \mathcal{L}(\beta) = -\sum_{i=1}^n \Big[y_i \log(p_i) + (1-y_i)\log(1-p_i)\Big],
    $$
    which is exactly the **BCE loss**.

    **Cross-entropy view:**  
    Let $p=(y,1-y)$ and $q=(\hat{y},1-\hat{y})$.  
    Then
    $$
    H(p,q) = -\sum_{c \in \{0,1\}} p_c \log q_c,
    $$
    which reduces to the same formula as above.  
    Thus, **BCE = NLL for Bernoulli = cross-entropy between true and predicted distributions.**
    """)

    # ------------------
    # 3. Why not MSE?
    # ------------------
    st.subheader("Why Not Use MSE for Classification?")

    st.markdown(r"""
    1. **Wrong likelihood:** MSE assumes Gaussian noise, but labels are Bernoulli.  
    2. **Non-convexity:** MSE + sigmoid produces a non-convex loss in $\hat{y}$, complicating optimization.  
    3. **Weak penalties:** MSE under-penalizes confident wrong predictions, leading to slower convergence.
    """)

    # Example math
    st.markdown(r"""
    **Example:** $y=0$, $\hat{y}=0.9$

    - **MSE:** $L=(0-0.9)^2=0.81$, gradient $=1.8$  
    - **BCE:** $L=-\log(1-0.9)=2.302$, gradient $=10$

    → BCE applies a much stronger correction for confident mistakes.
    """)

    # ------------------
    # 4. Visual Comparison
    # ------------------

    y_hat = np.linspace(0.001, 0.999, 200)
    mse_0, bce_0 = (0-y_hat)**2, -np.log(1-y_hat)
    mse_1, bce_1 = (1-y_hat)**2, -np.log(y_hat)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(y_hat, mse_0, "--", color="tab:blue", label="MSE (y=0)")
    ax.plot(y_hat, bce_0, color="tab:blue", label="BCE (y=0)")
    ax.plot(y_hat, mse_1, "--", color="tab:orange", label="MSE (y=1)")
    ax.plot(y_hat, bce_1, color="tab:orange", label="BCE (y=1)")
    ax.set_title("Binary Cross-Entropy vs MSE Loss")
    ax.set_xlabel("Predicted probability $\hat{y}$")
    ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)
    
    st.divider()
    
    # =====================
    # Cross-Entropy
    # =====================

    st.header("Cross-Entropy")

    st.markdown(r"""
    ### Information Theory Foundations

    Before defining cross-entropy, let’s recall some key ideas:

    - **Information (a.k.a. “surprisal”):**  
    How surprising is an event $x$?  
    $$
    I(x) = -\log p(x)
    $$  
    - High-probability event → low surprise  
    - Low-probability event → high surprise

    - **Expectation Value:**  
    In probability, the expectation of a function $f(x)$ is:  
    $$
    \mathbb{E}_{x\sim p}[f(x)] = \sum_x p(x)f(x)
    $$

    - **Entropy:**  
    The **expected surprise** of a distribution $p$:  
    $$
    H(p) = \mathbb{E}_{x\sim p}[-\log p(x)] = -\sum_x p(x)\log p(x)
    $$  
    → Entropy measures the **uncertainty** of a distribution.

    - **Cross-Entropy:**  
    The expected surprise if the true distribution is $p$ but we predict $q$:  
    $$
    H(p,q) = \mathbb{E}_{x\sim p}[-\log q(x)] = -\sum_x p(x)\log q(x)
    $$  
    → Tells us how well $q$ approximates $p$. Lower is better.
    """)

    st.markdown(r"""
    ### Cross-Entropy and KL Divergence

    We can rewrite:
    $$
    H(p,q) = H(p) + D_{KL}(p\|q)
    $$

    - $H(p)$ is fixed (depends only on the data).  
    - Minimizing $H(p,q)$ is equivalent to minimizing the **KL divergence** between the true distribution $p$ and the model distribution $q$.  
    - This shows cross-entropy is the natural loss for probabilistic classification.
    """)
    
    st.markdown(r"""
    ### Deriving the Softmax Function

    There are two standard ways to see why softmax has the form

    $$
    q_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} \, .
    $$

    ---

    #### A) Multinomial Logistic Regression (log-odds generalization of sigmoid)

    For $K$ classes, pick a reference class (say class $K$) and model **log-odds** linearly:

    $$
    \log \frac{P(y=k\mid x)}{P(y=K\mid x)} = z_k \quad \text{for } k=1,\dots,K-1,
    $$

    with logits $z_k = w_k^\top x + b_k$.

    Exponentiate:

    $$
    \frac{P(y=k\mid x)}{P(y=K\mid x)} = e^{z_k} 
    \;\Rightarrow\; 
    P(y=k\mid x)=P(y=K\mid x)\,e^{z_k}.
    $$

    Use the fact that probabilities sum to 1:

    $$
    1=\sum_{j=1}^K P(y=j\mid x)=P(y=K\mid x)\!\left(1+\sum_{j=1}^{K-1} e^{z_j}\right).
    $$

    Thus

    $$
    P(y=K\mid x)=\frac{1}{1+\sum_{j=1}^{K-1} e^{z_j}}, \quad
    P(y=k\mid x)=\frac{e^{z_k}}{1+\sum_{j=1}^{K-1} e^{z_j}}.
    $$

    If we simply assign a logit $z_j$ to **every** class (no explicit reference),
    this becomes the familiar **softmax**:

    $$
    P(y=k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.
    $$

    > **Check:** When $K=2$, softmax reduces to the **sigmoid**.

    ---

    #### B) Maximum Likelihood with a Normalization Constraint (Lagrange multipliers)

    Given logits $z_k$ (linear scores) and a distribution $q$ over classes,
    maximize the (per-example) log-likelihood $\sum_k p_k \log q_k$
    subject to $\sum_k q_k = 1$ and $q_k>0$.  
    Using the exponential family assumption $q_k \propto e^{z_k}$ and a Lagrange multiplier $\lambda$ for the constraint:

    $$
    \mathcal{L}(q,\lambda)=\sum_{k=1}^K p_k \log q_k + \lambda\!\left(1-\sum_{k=1}^K q_k\right).
    $$

    Setting $\frac{\partial \mathcal{L}}{\partial q_k}=0$ gives $q_k \propto e^{z_k}$, and normalization yields:

    $$
    q_k=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.
    $$

    ---

    ### Useful Properties

    - **Shift invariance:** For any constant $c$,

    $$
    \operatorname{softmax}(z)_k=\operatorname{softmax}(z+c\mathbf{1})_k.
    $$

    We often subtract $\max_j z_j$ for numerical stability.

    - **Temperature (confidence control):**

    $$
    \operatorname{softmax}_\tau(z)_k=\frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}}, \quad \tau>0.
    $$

    Smaller $\tau$ → sharper (more confident) distribution; larger $\tau$ → flatter.

    - **Gradient with cross-entropy:** With one-hot $p$,

    $$
    \frac{\partial}{\partial z_k}\big[-\sum_c p_c \log q_c\big]
    = q_k - p_k.
    $$

    (This is why `CrossEntropyLoss` has such a clean gradient.)
    """)


    st.markdown(r"""
    ### Multiclass Softmax Cross-Entropy

    In classification, the true distribution $p$ is one-hot:  

    - If the correct class is $y$, then $p_c = 1$ if $c=y$, else $0$.  
    - Prediction $q$ comes from **softmax** over logits $z$:  
    $$
    q_c = \frac{e^{z_c}}{\sum_j e^{z_j}}
    $$

    Cross-entropy becomes:
    $$
    H(p,q) = -\sum_c p_c \log q_c = -\log q_y
    $$

    This is exactly the **negative log-likelihood** for a categorical distribution.
    """)

    st.markdown(r"""
    ### Summary

    - **Information:** $I(x) = -\log p(x)$ = surprise of an outcome.  
    - **Entropy:** $H(p) = \mathbb{E}_{p}[I(x)]$ = expected surprise.  
    - **Cross-Entropy:** $H(p,q) = H(p)+D_{KL}(p\|q)$ = surprise under wrong distribution $q$.  
    - **Softmax Cross-Entropy:** practical loss for multiclass classification; matches the categorical likelihood.

    ✅ **Conclusion:** Cross-entropy is not just a heuristic — it’s grounded in **information theory** and **maximum likelihood estimation**.
    """)

if section == "Optimizers":
    st.title("Optimizers")

    st.markdown(r"""
    ### Gradient Descent Families (Batch • SGD • Mini-batch)

    - **Batch Gradient Descent**  
    Uses **all** $n$ examples each step. Stable but can be slow per update.  
    Update:  
    $$
    \theta_{t+1}=\theta_t-\eta\,\nabla_\theta \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(x_i),y_i)
    $$

    - **Stochastic Gradient Descent (SGD)** *(Robbins & Monro, 1951)*  
    Uses **one example** per step: faster, noisy updates, good for escaping shallow minima.  
    $$
    \theta_{t+1}=\theta_t-\eta\,\nabla_\theta \mathcal{L}(f_\theta(x_{i_t}),y_{i_t})
    $$

    - **Mini-batch Gradient Descent**  
    Uses a **small batch** $B$ (e.g., 32–1024). Sweet spot of stability vs. speed.  
    $$
    \theta_{t+1}=\theta_t-\eta\,\nabla_\theta \frac{1}{|B|}\sum_{i\in B} \mathcal{L}(f_\theta(x_i),y_i)
    $$

    **When to use:** Start with **mini-batch** (it’s standard). Full-batch only for tiny datasets; online SGD for streaming/very large data.
    """)

    st.markdown(r"""
    ### Momentum & Nesterov

    - **Momentum** *(Polyak, 1964)*  
    Accelerates in consistent directions; damps oscillations.  
    $$
    v_{t+1}=\beta v_t+(1-\beta)\,g_t,\quad \theta_{t+1}=\theta_t-\eta\,v_{t+1}
    $$
    where $g_t=\nabla_\theta \mathcal{L}_t$ and $\beta\in[0.8,0.99]$.

    - **Nesterov Accelerated Gradient (NAG)** *(Nesterov, 1983; Sutskever et al., 2013 in DL)*  
    Looks **ahead** before computing the gradient.  
    $$
    \tilde{\theta}_t=\theta_t-\eta\beta v_t,\quad
    g_t=\nabla_\theta \mathcal{L}(\tilde{\theta}_t),\quad
    v_{t+1}=\beta v_t+(1-\beta)g_t,\quad
    \theta_{t+1}=\theta_t-\eta v_{t+1}
    $$

    **When to use:** If plain SGD is jittery/slow, try **SGD+Momentum**; use **Nesterov** for slightly better theoretical behavior in some cases.
    """)

    st.markdown(r"""
    ### Adaptive Methods

    - **AdaGrad** *(Duchi, Hazan, Singer, 2011)*  
    Per-parameter learning rates grow small for frequently-updated coordinates.  
    $$
    r_{t+1}=r_t+g_t\odot g_t,\quad
    \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{r_{t+1}}+\epsilon}\,g_t
    $$
    **When to use:** Sparse features/rare features (e.g., NLP with bags of words).

    - **RMSProp** *(Hinton, 2012)*  
    Exponentially decaying average of squared grads (fixes AdaGrad’s aggressive decay).  
    $$
    r_{t+1}=\rho r_t+(1-\rho)g_t\odot g_t,\quad
    \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{r_{t+1}}+\epsilon}\,g_t
    $$
    **When to use:** Good default for RNNs/unstable losses; handles non-stationarity.

    - **Adam** *(Kingma & Ba, 2015)*  
    Momentum on the first moment **and** RMSProp on the second, with bias correction.  
    $$
    \begin{aligned}
    m_{t+1}&=\beta_1 m_t+(1-\beta_1)g_t,\quad &\hat{m}_{t+1}&=\frac{m_{t+1}}{1-\beta_1^{t+1}}\\
    v_{t+1}&=\beta_2 v_t+(1-\beta_2)g_t\odot g_t,\quad &\hat{v}_{t+1}&=\frac{v_{t+1}}{1-\beta_2^{t+1}}\\
    \theta_{t+1}&=\theta_t-\eta\frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}}+\epsilon}
    \end{aligned}
    $$
    **When to use:** Strong general-purpose default; fast convergence, less tuning.

    - **AdamW (decoupled weight decay)** *(Loshchilov & Hutter, 2017/2019)*  
    Decouples L2 **weight decay** from the Adam gradient step for correct regularization.  
    Adam update as above, plus a **separate decay** step:  
    $$
    \theta\leftarrow\theta-\eta\,\lambda\,\theta \quad \text{(decoupled from gradient)}
    $$
    **When to use:** Most Transformer/modern models: **AdamW** with weight decay (e.g., 0.01).
    """)

    st.markdown(r"""
    ---
    ### Minimal PyTorch Snippets

    Below is a tiny **training loop**; you swap only the **optimizer** line.  
    Use `batch_size=len(dataset)` for **full-batch**, `1` for **SGD**, or any $B$ for **mini-batch**.
    """)
    
    st.code('''
    model = MLP()
    # 1) Plain SGD
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # 2) SGD + Momentum
    # opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # 3) Nesterov Momentum
    # opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

    # 4) AdaGrad
    # opt = torch.optim.Adagrad(model.parameters(), lr=0.05)

    # 5) RMSProp
    # opt = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

    # 6) Adam
    # opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    # 7) AdamW (decoupled weight decay)
    # opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    train(model, opt, loader, epochs=5)
    ''', language = "python")

if section == "Normalization":
    st.title("Normalization Layers")

    st.markdown(r"""
    ### Why Normalization?

    - Neural nets can suffer from **internal covariate shift** — distributions of hidden activations change during training.
    - This makes optimization harder and learning rates more sensitive.
    - Normalization layers stabilize training, allow higher learning rates, and act as a form of regularization.

    Two common methods:
    1. **Batch Normalization (BatchNorm)**
    2. **Layer Normalization (LayerNorm)**
    """)

    # ----------------------
    # BatchNorm
    # ----------------------
    st.header("Batch Normalization (BatchNorm)")

    st.markdown(r"""
    **History:** *Ioffe & Szegedy, 2015 — "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift".*

    **Idea:** Normalize activations **across the batch** for each feature, then learn scale ($\gamma$) and shift ($\beta$).

    **Formula (per feature channel):**

    $$
    \mu_B = \frac{1}{m}\sum_{i=1}^m x_i,
    \quad
    \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2
    $$

    $$
    \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
    $$

    $$
    y_i = \gamma \hat{x}_i + \beta
    $$

    - $\mu_B, \sigma_B^2$: mean/variance over the batch
    - $\gamma, \beta$: learned parameters (restore representation capacity)

    **When to use:** Standard in CNNs and large MLPs; works best with reasonably large batch sizes.
    """)

    # PyTorch snippet
    st.code("""
    import torch.nn as nn

    # For CNNs (normalize across N,H,W per channel)
    bn = nn.BatchNorm2d(num_features=64)

    # For MLPs (normalize across N per feature)
    bn = nn.BatchNorm1d(num_features=128)
    """, language="python")

    # ----------------------
    # LayerNorm
    # ----------------------
    st.header("Layer Normalization (LayerNorm)")

    st.markdown(r"""
    **History:** *Ba, Kiros & Hinton, 2016 — "Layer Normalization".*

    **Idea:** Normalize **within each layer per sample**, not across the batch.  
    This avoids dependence on batch size, making it stable for **RNNs and Transformers**.

    **Formula (for input vector $x \in \mathbb{R}^d$):**

    $$
    \mu = \frac{1}{d}\sum_{j=1}^d x_j,
    \quad
    \sigma^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2
    $$

    $$
    \hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}},
    \quad
    y_j = \gamma \hat{x}_j + \beta
    $$

    - Normalization is per **individual sample**, across its hidden units.
    - No dependence on other examples in the batch.

    **When to use:** Default for Transformers, RNNs, and cases with small or variable batch sizes.
    """)

    # PyTorch snippet
    st.code("""
    import torch.nn as nn

    # Normalize across features for each sample
    ln = nn.LayerNorm(normalized_shape=512)  # e.g., Transformer hidden dim
    """, language="python")

    # ----------------------
    # Comparison
    # ----------------------
    st.subheader("BatchNorm vs LayerNorm")

    st.markdown(r"""
    | Feature           | BatchNorm                           | LayerNorm                        |
    |-------------------|-------------------------------------|----------------------------------|
    | Normalization     | Across batch (per feature)          | Across features (per sample)     |
    | Depends on batch? | Yes (batch mean/var)                | No                               |
    | Best for          | CNNs, large batch MLPs              | Transformers, RNNs, variable batch sizes |
    | Parameters        | $\gamma$, $\beta$ per feature       | $\gamma$, $\beta$ per feature    |
    | Introduced        | 2015                                | 2016                             |

    **Rule of thumb:**
    - Use **BatchNorm** for CNNs/vision tasks.
    - Use **LayerNorm** for Transformers and sequence models.
    """)

    # ----------------------
    # Training loop snippet
    # ----------------------
    st.subheader("PyTorch Training Example")

    st.code("""
    class SimpleMLP(nn.Module):
        def __init__(self, d_in=20, d_hidden=64, d_out=3):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.bn1 = nn.BatchNorm1d(d_hidden)   # try swapping with LayerNorm
            self.fc2 = nn.Linear(d_hidden, d_out)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            return self.fc2(x)
    """, language="python")

    st.markdown(r"""
    ✅ **Takeaway:**  
    - **Normalization stabilizes training.**  
    - **BatchNorm** = across the batch, great for CNNs.  
    - **LayerNorm** = across features per sample, great for Transformers.  
    """)
    
if section == "Regularization":
    st.title("Regularization")

    st.markdown(r"""
    ### Why Regularization?

    Regularization methods prevent **overfitting** by discouraging the model from relying too heavily on spurious patterns in the training set.  
    They improve **generalization** to unseen data.

    ---
    """)

    # ----------------------
    # L2 Regularization
    # ----------------------
    st.header("L2 Regularization (Weight Decay)")

    st.markdown(r"""
    **History:** Ridge regression (Hoerl & Kennard, 1970s).  

    **Idea:** Penalize large weights to keep the model smoother and prevent overfitting.  

    **Formula (added to loss):**
    $$
    J(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2
    $$

    This is equivalent to a **Gaussian prior** on the weights.  

    In optimizers like AdamW, this appears as **decoupled weight decay**:
    $$
    \theta \leftarrow \theta - \eta\lambda \theta
    $$

    **When to use:** Default for most deep learning models (AdamW with weight decay).
    """)

    st.code("""
    # Example: AdamW with weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    """, language="python")

    # ----------------------
    # Dropout
    # ----------------------
    st.header("Dropout")

    st.markdown(r"""
    **History:** *Hinton et al., 2014*.  

    **Idea:** Randomly "drop" activations during training with probability $p$, preventing co-adaptation of neurons.  

    **Formula:** At training time,
    $$
    \tilde{h}_i = \frac{m_i h_i}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)
    $$
    At test time, use the full network (no dropout).  

    **When to use:** Large MLPs and CNNs; less needed in Transformers (they rely more on weight decay + normalization).
    """)

    st.code("""
    # Example: 50% dropout in hidden layer
    nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 10)
    )
    """, language="python")

    # ----------------------
    # Learning Rate Scheduling
    # ----------------------
    st.header("Learning Rate Scheduling")

    st.markdown(r"""
    **Idea:** Adjust learning rate during training to improve convergence.  

    - **Step decay:** multiply $\eta$ by factor $\gamma$ every $k$ epochs.  
    - **Exponential decay:** $\eta_t = \eta_0 \cdot e^{-\gamma t}$.  
    - **Cosine annealing** (*Loshchilov & Hutter, 2016*): smooth periodic decay, often with restarts.  

    **Formula (cosine annealing):**
    $$
    \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos(\tfrac{T_{cur}}{T_{max}}\pi))
    $$

    **When to use:** Training deep nets (esp. Transformers, CNNs) where warmup + decay stabilizes learning.
    """)

    st.code("""
    # Step decay
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    """, language="python")

    # ----------------------
    # Early Stopping
    # ----------------------
    st.header("Early Stopping")

    st.markdown(r"""
    **Idea:** Stop training once validation loss stops improving.  
    Prevents overfitting while saving compute.

    **When to use:** Always worth monitoring — especially with small datasets.
    """)

    # ----------------------
    # Data Augmentation
    # ----------------------
    st.header("Data Augmentation")

    st.markdown(r"""
    **Idea:** Artificially expand the dataset by applying transformations.  
    Helps prevent overfitting and improves robustness.  

    - **Vision:** flips, crops, color jitter, Cutout, Mixup.  
    - **Text:** synonym replacement, back-translation.  
    - **Audio:** time-shift, noise injection.

    **When to use:** Essential in vision; helpful in audio/NLP with limited data.
    """)

    # ----------------------
    # Quick Summary
    # ----------------------
    st.markdown(r"""
    ---
    ### ✅ Quick Guide

    - **L2 / Weight Decay:** Default; prevents large weights.  
    - **Dropout:** Useful in MLPs/CNNs; less common in Transformers.  
    - **LR Schedules:** Step, exponential, cosine; improves training stability.  
    - **Cosine Annealing:** Great for Transformers and large-scale training.  
    - **Early Stopping:** Always monitor validation loss.  
    - **Data Augmentation:** Critical for vision; also useful in NLP/audio.
    """)
