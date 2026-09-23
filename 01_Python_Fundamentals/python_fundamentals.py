01_Python_Fundamentals/python_fundamentals.py
# AI/ML WEEK 1 – PYTHON FUNDAMENTALS

# 1. Variables and Data Types
name = "AI/ML Learner"
age = 21
score = 85.5
is_learning = True

print("Name:", name)
print("Age:", age)
print("Score:", score)
print("Currently learning:", is_learning)

# 2. Lists
marks = [72, 81, 76, 89, 94]

print("\nMarks:", marks)
print("Number of marks:", len(marks))
print("Highest mark:", max(marks))
print("Lowest mark:", min(marks))
print("Average:", sum(marks) / len(marks))

# 3. Dictionary
student = {
"name": "AI/ML Learner",
"course": "AI/ML Foundations",
"week": 1
}

print("\nStudent:", student)
print("Course:", student["course"])

# 4. Conditional Statement
average = sum(marks) / len(marks)

if average >= 75:
print("\nPerformance: Strong")
elif average >= 50:
print("\nPerformance: Satisfactory")
else:
print("\nPerformance: Needs Improvement")

# 5. Loop
print("\nMarks using loop:")

for mark in marks:
print(mark)

# 6. Function
def calculate_average(values):
return sum(values) / len(values)

average_score = calculate_average(marks)

print("\nAverage using function:", round(average_score, 2))

# 7. NumPy
import numpy as np

data = np.array([10, 20, 30, 40, 50])

print("\nNumPy Mean:", np.mean(data))
print("NumPy Median:", np.median(data))
print("NumPy Variance:", np.var(data))
print("NumPy Standard Deviation:", np.std(data))

# 8. Pandas
import pandas as pd

sample_data = {
"Age": [21, 24, 28, 32, 35],
"Score": [72, 81, 76, 89, 94]
}

df = pd.DataFrame(sample_data)

print("\nPandas DataFrame:")
print(df)

print("\nSummary Statistics:")
print(df.describe())

print("\nWeek 1 Python Fundamentals completed.")
