# Logistic Regression for Binary Classification

## Overview
This task is focused on binary classification using Logistic Regression.  
The goal is to train a model that can classify samples into one of two categories using the Breast Cancer Wisconsin dataset.

## Objective
To build a binary classifier using Logistic Regression, evaluate its performance using different metrics, and understand how threshold tuning affects classification results.

## Dataset
**Source:** [Breast Cancer Wisconsin Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

**Description:**  
This dataset contains features computed from breast mass images. Each sample is labeled as either:
- **M** = Malignant (cancerous)
- **B** = Benign (non-cancerous)

## Tools & Libraries Used
- Python
- pandas
- numpy
- matplotlib
- scikit-learn (sklearn)

## Steps Followed

### 1. Data Loading
The dataset was loaded into a pandas DataFrame using `pd.read_csv()`.

### 2. Data Cleaning
- Removed non-informative columns like `id` and unnamed columns.
- Converted the target variable `diagnosis` from categorical to numeric:
  - Malignant (M) → 1
  - Benign (B) → 0

### 3. Handling Missing Values
- Used **mean imputation** with `SimpleImputer` to replace missing values in feature columns with the column mean.
- This ensures that the dataset has no `NaN` values, which Logistic Regression does not accept.

### 4. Feature Scaling
- Applied standardization using `StandardScaler` to bring all features to the same scale.
- This step improves model performance and convergence.

### 5. Splitting the Dataset
- Split the data into **80% training** and **20% testing** using `train_test_split`.
- Stratified the split to maintain class balance.

### 6. Training the Model
- Used `LogisticRegression` from `sklearn.linear_model`.
- Trained the model using the standardized training data.

### 7. Model Evaluation
Evaluated the model using the following metrics:
- **Confusion Matrix:** Shows TP, TN, FP, FN.
- **Precision:** Ratio of correctly predicted positive observations to total predicted positives.
- **Recall:** Ratio of correctly predicted positives to all actual positives.
- **ROC-AUC Score:** Measures the classifier's ability to distinguish between classes.

### 8. ROC Curve
- Plotted the ROC curve to visualize the trade-off between sensitivity and specificity.

### 9. Threshold Tuning
- Demonstrated how changing the probability threshold (default is 0.5) affects the confusion matrix.
- For example, at threshold 0.3, more positive predictions may be made, which can increase recall but decrease precision.

## Key Learnings
- Logistic Regression is effective for binary classification problems.
- Data preprocessing (handling missing values and scaling) is essential.
- Evaluation metrics help in assessing model performance.
- The sigmoid function maps predictions to probability.
- Threshold tuning allows control over model sensitivity and specificity.
- 
## Repository Contents
- `logistic_regression_task4.py` – Python script with complete code.
- (Optional) `logistic_regression_task4.ipynb` – Jupyter notebook version.
- `README.md` – This file.

## References
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 


Author
Sonali18-tech

