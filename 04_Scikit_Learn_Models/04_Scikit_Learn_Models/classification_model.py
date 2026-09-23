# ============================================================
# AI/ML WEEK 1 – SCIKIT-LEARN CLASSIFICATION MODEL
# ============================================================

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
classification_report
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("SCIKIT-LEARN LOGISTIC REGRESSION CLASSIFICATION")

print("\nDataset shape:")
print(X.shape)

print("\nNumber of features:")
print(X.shape[1])

print("\nNumber of classes:")
print(len(np.unique(y)))


# ------------------------------------------------------------
# 2. TRAIN/TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------------
# 3. CREATE MODEL
# ------------------------------------------------------------

model = LogisticRegression(max_iter=5000)


# ------------------------------------------------------------
# 4. TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# ------------------------------------------------------------
# 5. MAKE PREDICTIONS
# ------------------------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

print("\nFirst five predictions:")
print(predictions[:5])


# ------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1 = f1_score(y_test, predictions)

roc_auc = roc_auc_score(y_test, probabilities)


# ------------------------------------------------------------
# 7. DISPLAY RESULTS
# ------------------------------------------------------------

print("\nMODEL EVALUATION RESULTS")

print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))
print("ROC-AUC:", round(roc_auc, 3))


# ------------------------------------------------------------
# 8. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nCLASSIFICATION REPORT")

print(classification_report(y_test, predictions))


# ------------------------------------------------------------
# 9. INTERPRETATION
# ------------------------------------------------------------

print("\nINTERPRETATION")

print("Accuracy measures the proportion of correct predictions.")

print("Precision measures how many predicted positive cases")
print("were actually positive.")

print("Recall measures how many actual positive cases")
print("were correctly identified.")

print("F1-score balances precision and recall.")

print("ROC-AUC measures the ability of the model to")
print("distinguish between the two classes.")


# ------------------------------------------------------------
# 10. SUMMARY
# ------------------------------------------------------------

print("\nSUMMARY")

print("A Logistic Regression classification model was")
print("developed using Scikit-learn.")

print("The dataset was divided into training and testing sets.")

print("Multiple classification metrics were used to evaluate")
print("the model.")

print("\nClassification modelling completed successfully.")
