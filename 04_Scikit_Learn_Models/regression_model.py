# ============================================================
# AI/ML WEEK 1 – SCIKIT-LEARN REGRESSION MODEL
# ============================================================

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("============================================================")
print("SCIKIT-LEARN LINEAR REGRESSION")
print("============================================================")

print("\nDataset shape:")
print(X.shape)

print("\nNumber of features:")
print(X.shape[1])


# ------------------------------------------------------------
# 2. TRAIN/TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------------
# 3. CREATE MODEL
# ------------------------------------------------------------

model = LinearRegression()


# ------------------------------------------------------------
# 4. TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# ------------------------------------------------------------
# 5. MAKE PREDICTIONS
# ------------------------------------------------------------

predictions = model.predict(X_test)

print("\nFirst five predictions:")
print(predictions[:5])


# ------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)


# ------------------------------------------------------------
# 7. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n============================================================")
print("MODEL EVALUATION RESULTS")
print("============================================================")

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 3))


# ------------------------------------------------------------
# 8. INTERPRETATION
# ------------------------------------------------------------

print("\n============================================================")
print("INTERPRETATION")
print("============================================================")

print("MAE represents the average absolute prediction error.")

print("MSE gives greater weight to larger prediction errors.")

print("RMSE represents prediction error on the target scale.")

print("R² indicates the proportion of variation in the target")
print("explained by the regression model.")


# ------------------------------------------------------------
# 9. SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("SUMMARY")
print("============================================================")

print("A Linear Regression model was trained using Scikit-learn.")
print("The dataset was divided into training and testing sets.")
print("The model was trained on the training data and evaluated")
print("on previously unseen test data.")

print("\nRegression modelling completed successfully.")
