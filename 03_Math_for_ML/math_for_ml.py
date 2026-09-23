#============================================================
# AI/ML WEEK 1 – MATHEMATICS FOR MACHINE LEARNING
# ============================================================

import numpy as np

print("============================================================")
print("MATHEMATICS AND STATISTICS FOR MACHINE LEARNING")
print("============================================================")


# ------------------------------------------------------------
# 1. DATA
# ------------------------------------------------------------

data = np.array([10, 15, 20, 25, 30, 35, 40])

print("\nDataset:")
print(data)


# ------------------------------------------------------------
# 2. MEAN
# ------------------------------------------------------------

mean_value = np.mean(data)

print("\nMean:", mean_value)


# ------------------------------------------------------------
# 3. MEDIAN
# ------------------------------------------------------------

median_value = np.median(data)

print("Median:", median_value)


# ------------------------------------------------------------
# 4. VARIANCE
# ------------------------------------------------------------

variance_value = np.var(data)

print("Variance:", variance_value)


# ------------------------------------------------------------
# 5. STANDARD DEVIATION
# ------------------------------------------------------------

standard_deviation = np.std(data)

print("Standard Deviation:", standard_deviation)


# ------------------------------------------------------------
# 6. CORRELATION
# ------------------------------------------------------------

x = np.array([10, 20, 30, 40, 50])
y = np.array([12, 22, 29, 43, 48])

correlation_matrix = np.corrcoef(x, y)

correlation = correlation_matrix[0, 1]

print("\nCorrelation between X and Y:", round(correlation, 3))


# ------------------------------------------------------------
# 7. PROBABILITY
# ------------------------------------------------------------

total_outcomes = 100
favourable_outcomes = 25

probability = favourable_outcomes / total_outcomes

print("\nProbability:", probability)
print("Probability percentage:", probability * 100, "%")


# ------------------------------------------------------------
# 8. BASIC SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("INTERPRETATION")
print("============================================================")

print("Mean represents the average value of the dataset.")
print("Median represents the middle value after ordering the data.")
print("Variance measures how widely the observations are spread.")
print("Standard deviation represents the typical spread around the mean.")
print("Correlation measures the strength and direction of a relationship between variables.")
print("Probability represents the likelihood of an event occurring.")


# ------------------------------------------------------------
# 9. MACHINE LEARNING CONNECTION
# ------------------------------------------------------------

print("\n============================================================")
print("CONNECTION TO MACHINE LEARNING")
print("============================================================")

print("1. Mean and median help summarize datasets.")
print("2. Variance and standard deviation help understand data spread.")
print("3. Correlation helps identify relationships between features.")
print("4. Probability is used in probabilistic predictions.")
print("5. Statistical concepts support data preprocessing and model evaluation.")


# ------------------------------------------------------------
# END
# ------------------------------------------------------------

print("\nMathematics for Machine Learning completed successfully.")
