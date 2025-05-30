# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
)

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\capl2\OneDrive\Pictures\Documents\AIML_Internship\Task4\data.csv")  # Make sure data.csv is in your working directory

# Step 2: Drop non-informative columns (like ID or unnamed index column)
df.drop(columns=['Unnamed: 32', 'id'], inplace=True, errors='ignore')

# Step 3: Convert categorical target variable to numeric (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Step 4: Separate features (X) and target (y)
X_raw = df.drop(columns='diagnosis')
y = df['diagnosis']

# Step 5: Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X_raw)  # Converts DataFrame to NumPy array

# Step 6: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Step 9: Predict class labels and probabilities
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Step 10: Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Step 11: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', linewidth=2)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.show()

# Step 12: Try different threshold (e.g., 0.3)
threshold = 0.3
y_pred_thresh = (y_prob >= threshold).astype(int)

print(f"\n Confusion Matrix at threshold = {threshold}:\n", confusion_matrix(y_test, y_pred_thresh))

# Sigmoid function visualization
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid, color='green')
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid(True)
plt.show()
