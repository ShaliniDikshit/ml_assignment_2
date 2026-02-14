# Machine Learning Classification Model Comparison

Live Streamlit App:https://ml-assignment-2-shalini.streamlit.app/

GitHub Repository: https://github.com/ShaliniDikshit/ml_assignment_2

## 1. Problem Statement
The objective of this project is to implement and compare multiple machine learning classification models on a selected dataset.  
The goal is to evaluate the performance of different models using various evaluation metrics and deploy the solution as an interactive Streamlit web application.

---

## 2. Dataset Description
Dataset used: Breast Cancer Wisconsin Diagnostic Dataset

- Type: Binary Classification
- Instances: 569
- Features: 30 numerical features
- Target: Malignant (1) or Benign (0)

The dataset is used to predict whether a tumor is malignant or benign based on medical measurements.

---

## 3. Models Used

The following classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 4. Evaluation Metrics

The following metrics were used for comparison:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 |0.9861  | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 |  0.8174|
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost | 0.9561 | 0.9901 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

---

## 6. Observations

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | Performs well on linearly separable data with high accuracy. |
| Decision Tree | Easy to interpret but may overfit. |
| kNN | Sensitive to choice of k and scaling. |
| Naive Bayes | Fast and works well for probabilistic classification. |
| Random Forest | Reduces overfitting and improves generalization. |
| XGBoost | High performance boosting model with strong predictive power. |

---

## 7. Streamlit Application

Features implemented:
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion Matrix
- Classification Report

---

## 8. Deployment

The application is deployed using Streamlit Community Cloud.

