I/ML WEEK 1 – DATA CLEANING & EXPLORATORY DATA ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. CREATE SAMPLE DATASET
# ------------------------------------------------------------

data = {
"Age": [21, 24, 28, 32, 35, 29, 41, 38, 26, 45],
"Income": [25000, 32000, 45000, 52000, 60000,
48000, 72000, 65000, 38000, 80000],
"Credit_Score": [620, 650, 710, 735, 760,
690, 780, 745, 670, 800],
"Monthly_Spending": [12000, 15000, 18000, 22000, 25000,
19000, 28000, 24000, 16000, 30000],
"Churn": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

print("============================================================")
print("DATA CLEANING AND EXPLORATORY DATA ANALYSIS")
print("============================================================")


# ------------------------------------------------------------
# 2. VIEW DATASET
# ------------------------------------------------------------

print("\nFirst five records:")
print(df.head())


# ------------------------------------------------------------
# 3. DATASET STRUCTURE
# ------------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ------------------------------------------------------------
# 4. MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 5. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\nNumber of duplicate records:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\nDescriptive statistics:")
print(df.describe())


# ------------------------------------------------------------
# 7. DATA CLEANING
# ------------------------------------------------------------

# Remove duplicate records if present
df = df.drop_duplicates()

print("\nDataset shape after duplicate removal:")
print(df.shape)


# ------------------------------------------------------------
# 8. BASIC STATISTICAL ANALYSIS
# ------------------------------------------------------------

print("\nAverage Age:", round(df["Age"].mean(), 2))
print("Average Income:", round(df["Income"].mean(), 2))
print("Average Credit Score:", round(df["Credit_Score"].mean(), 2))
print("Average Monthly Spending:", round(df["Monthly_Spending"].mean(), 2))


# ------------------------------------------------------------
# 9. TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\nChurn distribution:")
print(df["Churn"].value_counts())

plt.figure(figsize=(7, 5))

df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. AGE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.hist(df["Age"], bins=5)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11. INCOME DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.hist(df["Income"], bins=5)

plt.title("Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12. CREDIT SCORE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.hist(df["Credit_Score"], bins=5)

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 13. CORRELATION ANALYSIS
# ------------------------------------------------------------

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)


# ------------------------------------------------------------
# 14. CORRELATION VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(correlation, cmap="Blues", aspect="auto")

plt.colorbar()

plt.xticks(
range(len(correlation.columns)),
correlation.columns,
rotation=45,
ha="right"
)

plt.yticks(
range(len(correlation.columns)),
correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. KEY EDA FINDINGS
# ------------------------------------------------------------

print("\n============================================================")
print("KEY EDA FINDINGS")
print("============================================================")

print("1. The dataset contains numerical financial and demographic variables.")
print("2. Missing values were checked and no missing values were found.")
print("3. Duplicate records were checked and removed if present.")
print("4. Descriptive statistics were used to understand the distributions.")
print("5. Visualizations were used to identify patterns in age, income and credit score.")
print("6. Correlation analysis was performed to examine relationships between variables.")


# ------------------------------------------------------------
# END
# ------------------------------------------------------------

print("\nData Cleaning and EDA completed successfully.")
