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

1. Logistic Regression : A linear classification model that predicts probability using a sigmoid function. 
2. Decision Tree Classifier : A tree-based model that splits data based on feature conditions. 
3. K-Nearest Neighbor (kNN) : A distance-based algorithm that classifies a sample based on its nearest neighbors. 
4. Naive Bayes (Gaussian) : A probabilistic model based on Bayes’ theorem with independence assumptions. 
5. Random Forest (Ensemble) : An ensemble method that combines multiple decision trees to improve accuracy. 
6. XGBoost (Ensemble) : A gradient boosting algorithm that builds trees sequentially to minimize error.

---

## 4. Evaluation Metrics

The following metrics were used for comparison:

- Accuracy : Percentage of correctly classified samples.
- AUC Score : Measures model’s ability to distinguish between classes.
- Precision : Proportion of correct positive predictions.
- Recall : Ability of the model to correctly identify actual positives.
- F1 Score : Harmonic mean of precision and recall.
- Matthews Correlation Coefficient (MCC) : Balanced metric that considers true and false positives and negatives.

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
The deployed application allows users to upload a dataset, select a classification model, and view evaluation metrics including confusion matrix and classification report.


## 9. How to run locally

**1. Clone the repository:**

git clone https://github.com/ShaliniDikshit/ml_assignment_2.git

cd ml_assignment_2

**2. Create and activate a virtual environment:**

python3 -m venv venv

source venv/bin/activate

**3. Install dependencies:**

pip install -r requirements.txt

**4. Run the Streamlit application:**

streamlit run app.py


## 10. Conclusion
The comparative evaluation of six classification models reveals that Logistic Regression outperforms the other models across nearly all performance metrics. With the highest accuracy (0.9825), AUC (0.9954), and MCC (0.9623), it demonstrates excellent discriminative power and balanced classification ability.

While ensemble models such as XGBoost and Random Forest showed strong recall and competitive AUC values, their overall performance remained marginally lower. Additionally, Logistic Regression offers better interpretability and lower model complexity, making it a more reliable and efficient choice for this dataset.

Hence, Logistic Regression is concluded to be the optimal model for the Breast Cancer classification problem in this study.
