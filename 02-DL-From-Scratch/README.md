## 02 — Deep Learning From Scratch

> This module builds neural networks **from scratch using NumPy** 

---

## Contents

| Notebook | Task Type | Built From Scratch |
|---|---|---|
| `backpropagation_regression` | Regression  | Forward pass, Backprop, Gradient Descent |
| `backpropagation_classification2` | Binary Classification  | Forward pass, Backprop, Sigmoid, BCE Loss |

---

## 📓 Notebook Descriptions

### 1. Backpropagation — Regression

**File:** `backpropagation_regression.ipynb`


#### Concepts Covered
- Manual weight initialization
- Forward pass through multiple layers
- Mean Squared Error (MSE) loss
- Backpropagation via the chain rule
- Gradient Descent weight update
- Tracking loss across epochs

---

#### Mathematical Derivation

**Network Notation:**
- $L$ = total number of layers
- $\mathbf{W}^{[l]}$ = weight matrix of layer $l$
- $\mathbf{b}^{[l]}$ = bias vector of layer $l$
- $\mathbf{a}^{[l]}$ = activation output of layer $l$
- $\mathbf{z}^{[l]}$ = pre-activation (linear combination) of layer $l$

---

**Step 1 — Forward Pass**

For each layer $l = 1, 2, \ldots, L$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = g^{[l]}\!\left(\mathbf{z}^{[l]}\right)$$

where $g^{[l]}$ is the activation function of layer $l$.

- Hidden layers: $g(z) = \text{ReLU}(z) = \max(0, z)$
- Output layer (regression): $g(z) = z$ (linear / identity)

Final prediction: $\hat{y} = \mathbf{a}^{[L]}$

---

**Step 2 — Loss Function**

Mean Squared Error for $m$ training examples:

$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2$$

---

**Step 3 — Backward Pass (Backpropagation)**

The goal is to compute $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$ for every layer.

**Output layer error (delta):**

$$\boldsymbol{\delta}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot g'^{[L]}\!\left(\mathbf{z}^{[L]}\right)$$

For MSE with linear output:

$$\boldsymbol{\delta}^{[L]} = \frac{-2}{m}(y - \hat{y})$$

**Hidden layer error — propagated backwards:**

$$\boldsymbol{\delta}^{[l]} = \left(\mathbf{W}^{[l+1]}\right)^T \boldsymbol{\delta}^{[l+1]} \odot g'^{[l]}\!\left(\mathbf{z}^{[l]}\right)$$

where $\odot$ is element-wise multiplication (Hadamard product).

**Gradients:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} \left(\mathbf{a}^{[l-1]}\right)^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$$

---

**Step 4 — Gradient Descent Update**

$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$$

$$\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$$

where $\eta$ is the **learning rate**.

---

**ReLU Derivative (used in backprop):**

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

---

### 2. Backpropagation — Binary Classification

**File:** `backpropagation_classification2.ipynb`


#### Concepts Covered
- Sigmoid activation and its derivative
- Binary Cross-Entropy loss derivation
- Backpropagation through a Sigmoid output
- Classification decision threshold
- Accuracy evaluation

---

#### Mathematical Derivation

**Forward pass** is identical to regression, except the output layer uses:

$$\hat{y} = \sigma\!\left(\mathbf{z}^{[L]}\right) = \frac{1}{1 + e^{-\mathbf{z}^{[L]}}}$$

This squashes any real number into $(0, 1)$, interpreted as a **probability of class 1**.

---

**Loss Function — Binary Cross-Entropy:**

For a single example:

$$\mathcal{L}(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

For $m$ training examples:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

> **Why not MSE for classification?**  
> MSE with Sigmoid output produces a non-convex loss surface with vanishing gradients.  
> BCE is convex, has well-behaved gradients, and has a probabilistic interpretation via Maximum Likelihood Estimation (MLE).

---

**Output layer delta — Backprop through Sigmoid + BCE:**

A key identity: the gradient of BCE loss w.r.t. the pre-activation $z^{[L]}$ simplifies beautifully:

$$\frac{\partial \mathcal{L}}{\partial z^{[L]}} = \hat{y} - y$$

**Derivation:**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z^{[L]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \sigma'(z) = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y}) = \hat{y} - y$$

This is one of the most elegant results in neural network mathematics.

---

**Hidden layer deltas and weight updates** follow the same formulas as in the regression notebook (see above).

---

**Classification Decision:**

$$\text{Predicted class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$

---



## Regression vs Classification 

| Component | Regression | Binary Classification |
|---|---|---|
| Output activation | Linear (identity) | Sigmoid $\sigma(z)$ |
| Loss function | MSE: $\frac{1}{m}\sum(y-\hat{y})^2$ | BCE: $-\frac{1}{m}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Output delta $\delta^{[L]}$ | $\frac{-2}{m}(y - \hat{y})$ | $\hat{y} - y$ |
| Evaluation metric | MAE, R² | Accuracy, Precision, Recall, AUC |
| Output range | $(-\infty, +\infty)$ | $(0, 1)$ — probability |



