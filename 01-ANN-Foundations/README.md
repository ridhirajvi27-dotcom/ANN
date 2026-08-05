## 01 — ANN Foundations

> This module introduces the foundational concepts and practical applications of Artificial Neural Networks, implemented from a beginner-to-intermediate level using real-world datasets.

---

## 📋 Contents

| Notebook | Topic | Dataset | Task |
|---|---|---|---|
| `Admission_prediction` | Regression with ANN | Graduate Admissions | Predict admission chance |
| `BreastCancer(binary_classification)` | Binary Classification | Breast Cancer Wisconsin | Classify tumor as benign / malignant |
| `mnistexample`| Multi-class Classification | MNIST Handwritten Digits | Recognize digits 0–9 |

---

## 📓 Notebook Descriptions

### 1. Admission Prediction — Regression


A regression task that predicts the **probability of a student being admitted** to a graduate program based on academic features such as GRE score, TOEFL score, university rating, statement of purpose, letter of recommendation strength, CGPA, and research experience.

**Key Concepts Covered:**
- Regression using Artificial Neural Networks
- Feature scaling and normalization
- Model architecture design (Dense layers, ReLU, Linear output)
- Training with Mean Squared Error (MSE) loss
- Evaluating with R² score and MAE
- Visualizing predictions vs. actuals

**Dataset:** [Graduate Admissions Dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions) — 500 samples, 7 features.

---

### 2. Breast Cancer Classification — Binary Classification


A binary classification task that uses an ANN to **classify breast tumors as benign or malignant** based on 30 numerical features derived from digitized images of fine needle aspirate (FNA) of a breast mass.

**Key Concepts Covered:**
- Binary classification with ANN
- Sigmoid activation for output layer
- Binary Cross-Entropy loss
- Precision, Recall, F1-Score, and AUC-ROC evaluation
- Confusion matrix visualization
- Handling real-world medical data

**Dataset:** [Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) — 569 samples, 30 features (built into `sklearn`).

---

### 3. MNIST Digit Recognition — Multi-class Classification


The classic "Hello, World!" of deep learning — training an ANN to **recognize handwritten digits (0–9)** from the MNIST dataset of 28×28 grayscale images.

**Key Concepts Covered:**
- Image flattening and preprocessing
- Multi-class classification with Softmax output
- Categorical Cross-Entropy loss
- Accuracy and per-class evaluation
- Visualizing predictions on test samples
- Confusion matrix for 10 classes

**Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 images (60,000 train / 10,000 test), 28×28 pixels.

---

### Task Type → Output Layer Design

| Task | Output Neurons | Activation | Loss Function | Metric |
|---|---|---|---|---|
| Binary Classification | 1 | `sigmoid` | `binary_crossentropy` | `accuracy` |
| Multi-class (integer labels) | N (num classes) | `softmax` | `sparse_categorical_crossentropy` | `accuracy` |
| Regression | 1 | `linear` | `mean_squared_error` | R² score |

---

### Scalers Comparison

| Scaler | Formula | Range | When to Use |
|---|---|---|---|
| `StandardScaler` | `(x - mean) / std` | Unbounded, ~(-3, 3) | Most ML tasks, especially with Gaussian-like features |
| `MinMaxScaler` | `(x - min) / (max - min)` | [0, 1] | When you know the feature range or target is in [0, 1] |
| Divide by max (e.g., `/255`) | `x / 255` | [0, 1] | Image pixel data |

---