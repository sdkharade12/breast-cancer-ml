# 🧬 Breast Cancer Detection using Machine Learning

## 📌 Overview

This project is an **end-to-end Machine Learning application** that predicts whether a breast tumor is **Benign** or **Malignant** using diagnostic features.
It demonstrates the **full ML lifecycle** — from feature selection and model training to inference and deployment via a Streamlit web app.

The focus of this project is not just accuracy, but **clean ML engineering practices**, interpretability, and reproducibility.

---

## 🧠 Problem Statement

Early detection of breast cancer is critical for effective treatment.
The goal of this project is to build a machine learning model that can assist in **risk assessment** by analyzing tumor characteristics extracted from medical imaging.

⚠️ **Disclaimer**:
This application is for **educational purposes only** and should **not** be considered a medical diagnosis tool.

---

## 📊 Dataset

* **Source**: Breast Cancer Wisconsin Dataset
* **Provider**: `sklearn.datasets`
* **Samples**: 569
* **Features**: 30 numerical diagnostic features
* **Target**:

  * `0` → Malignant
  * `1` → Benign

The dataset is loaded programmatically using `sklearn.datasets`, so no external data files are required.

---

## ⚙️ ML Pipeline

### 1️⃣ Feature Selection

* **Technique**: Mutual Information (`mutual_info_classif`)
* **Approach**:

  * Rank all features based on information gain
  * Select **Top 10 most informative features**
* **Reason**:

  * Improves interpretability
  * Reduces noise
  * Maintains strong performance

---

### 2️⃣ Data Preprocessing

* Train–test split (80/20)
* Feature scaling using **StandardScaler**
* Preprocessing logic is centralized to ensure **training–inference consistency**

---

### 3️⃣ Model Training

Multiple classification algorithms were initially evaluated:

* Logistic Regression
* Naive Bayes
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

Models were compared using:

* **Accuracy**
* **F1 Score**

**Logistic Regression** achieved the best overall performance and was selected as the final model due to its stability and interpretability.

Model artifacts saved:

* Trained model
* Scaler
* Selected feature list

---

### 4️⃣ Cross Validation

To ensure the model generalizes well, **5-Fold Cross Validation** was applied to the selected Logistic Regression model.

This process:

* Splits the training data into 5 folds
* Trains the model on different subsets
* Evaluates performance across multiple runs

Cross validation helps verify that the model is **stable and not overfitting to a single data split**.

---

### 5️⃣ Hyperparameter Tuning

To further optimize the model, **GridSearchCV** was used to tune key Logistic Regression parameters.

Parameters tuned include:

* **C** (regularization strength)
* **penalty** (L1 / L2 regularization)
* **solver**

Grid search combined with cross-validation automatically selected the **best performing configuration**, improving model robustness.

---

### 6️⃣ Inference

* A single inference module handles:

  * Column alignment
  * Scaling
  * Prediction
  * Probability estimation
* Prevents feature mismatch errors and logic duplication

---

## 📈 Model Performance

| Metric                    | Score |
| ------------------------- | ----- |
| Accuracy                  | ~98%  |
| F1 Score                  | ~0.98 |
| Cross-Validation Accuracy | ~95%  |

These results demonstrate that a well-engineered logistic regression model can achieve **high predictive performance while remaining interpretable**.

---

## 🖥️ Web Application (Streamlit)

The Streamlit app provides:

* User-friendly input form with guided placeholders
* Prediction output (Benign / Malignant)
* Confidence score (prediction probability)
* Graceful input validation and error handling

The UI is intentionally kept **lightweight**, while all ML logic resides in the backend pipeline.

---

## 🛠 Tech Stack

* **Python**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Streamlit**
* **Joblib**
* **Git & GitHub**

---

## 📁 Project Structure

```text
breast-cancer-ml/
│
├── notebooks/
│   └── breast_cancer_eda.ipynb
│
├── src/
│   ├── feature_selection.py
│   ├── preprocessing.py
│   ├── train.py
│   └── inference.py
│
├── models/
│   ├── reduced_LR_model.pkl
│   ├── reduced_LR_scaler.pkl
│   └── reduced_LR_columns.pkl
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model

```bash
python src/train.py
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Model Behavior Notes

* The model outputs **prediction probabilities**
* Borderline cases may lean towards **Malignant**, prioritizing safety
* This behavior is intentional and appropriate for medical screening contexts

---

## 🚀 Future Improvements

* Add model explainability using **SHAP**
* Extend to multi-model comparison dashboard
* Deploy as a cloud-based application

---

## 🎯 Key Learnings

* Feature selection is as important as model choice
* Clean separation of training and inference logic prevents real-world bugs
* Simple models, when engineered well, can be highly effective
* ML engineering is about **systems**, not just algorithms

---

