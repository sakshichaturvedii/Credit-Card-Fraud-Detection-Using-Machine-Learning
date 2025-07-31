# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. It handles highly imbalanced data through resampling and evaluates model performance using relevant classification metrics. The focus is on building a robust and interpretable fraud detection system.

---

## Project Objectives

- Identify and classify fraudulent transactions.
- Handle class imbalance effectively using SMOTE.
- Build and evaluate multiple classification models.
- Analyze results using precision, recall, F1-score, and ROC-AUC.

---

## Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **File:** `credit.csv`
- **Description:**  
  - Contains transactions made by European cardholders in September 2013.  
  - Total: 284,807 transactions with only 492 frauds (0.17%).  
  - Features `V1` to `V28` are result of PCA transformation; `Amount` and `Time` are original features.  
  - `Class` is the target variable (0 = Non-fraud, 1 = Fraud).


---

## Technologies Used

- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn  
- **Environment:** Jupyter Notebook 

---

## Workflow

1. **Data Exploration**
   - Checked for class imbalance, missing values, and correlation
   - Visualized fraud vs. non-fraud transactions

2. **Preprocessing**
   - Normalized `Amount` using `StandardScaler`
   - Applied SMOTE to balance class distribution

3. **Model Training**
   - Compared Logistic Regression and XGBoost models
   - Tuned hyperparameters for best performance

4. **Evaluation**
   - Used metrics: Confusion Matrix, Precision, Recall, F1-score, ROC-AUC
   - XGBoost achieved ROC-AUC score of **0.94**

---

## Results

- **XGBoost** significantly outperformed Logistic Regression.
- SMOTE improved recall and overall model balance.
- Strong model performance on identifying rare fraud cases.

---

## Key Learnings

- Class imbalance handling is critical in fraud detection.
- ROC-AUC is a better evaluation metric than accuracy for imbalanced datasets.
- XGBoost combined with SMOTE is effective for rare-event classification.



