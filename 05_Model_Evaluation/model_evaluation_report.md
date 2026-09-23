# AI/ML Week 1 – Model Evaluation Report

## 1. Objective

The objective of this work was to understand the basic Machine Learning workflow by developing and evaluating regression and classification models using Scikit-learn.

The workflow followed was:

Data → Train/Test Split → Model Training → Prediction → Evaluation

---

## 2. Models Used

### Linear Regression

Linear Regression was used for a regression problem involving the prediction of a continuous numerical target.

The model was trained using the Scikit-learn LinearRegression algorithm.

### Logistic Regression

Logistic Regression was used for a binary classification problem.

The model was trained using the Scikit-learn LogisticRegression algorithm.

---

## 3. Regression Model Evaluation

The Linear Regression model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Metric Interpretation

**MAE:** Measures the average absolute difference between the actual and predicted values.

**MSE:** Measures the average squared prediction error and gives greater weight to larger errors.

**RMSE:** Represents prediction error on the same scale as the target variable.

**R² Score:** Indicates the proportion of variation in the target variable explained by the model.

---

## 4. Classification Model Evaluation

The Logistic Regression model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Metric Interpretation

**Accuracy:** Measures the proportion of total predictions that were correct.

**Precision:** Measures how many observations predicted as positive were actually positive.

**Recall:** Measures how many actual positive observations were correctly identified.

**F1-score:** Provides a balance between precision and recall.

**ROC-AUC:** Measures how well the model distinguishes between the two classes across different classification thresholds.

---

## 5. Train/Test Split

Both models used a train/test split so that the models could be evaluated on observations that were not used during training.

An 80:20 split was used, with 80% of observations used for training and 20% reserved for testing.

A fixed random state was used to make the results reproducible.

---

## 6. Model Evaluation Approach

The models were evaluated using multiple metrics rather than relying on a single performance measure.

For regression, the focus was on prediction error and explained variance.

For classification, accuracy was considered together with precision, recall, F1-score and ROC-AUC.

Using multiple metrics provides a more complete understanding of model performance.

---

## 7. Limitations

The models used in this exercise are basic Machine Learning models and were developed as part of a learning exercise.

Potential limitations include:

- Limited feature engineering
- Limited hyperparameter tuning
- Performance depends on the selected dataset
- Results may vary with different train/test splits
- Model performance may not generalize to every dataset
- Classification performance can be affected by class imbalance

---

## 8. Key Learning

This exercise demonstrated the complete basic Machine Learning workflow:

Data Preparation → Train/Test Split → Model Training → Prediction → Evaluation

It also demonstrated why different evaluation metrics are required for regression and classification problems.

---

## 9. Conclusion

Week 1 provided practical experience with Python, data analysis, statistics, Scikit-learn and Machine Learning model evaluation.

The regression and classification exercises established a foundation for more advanced Machine Learning and Artificial Intelligence concepts in the upcoming weeks.
