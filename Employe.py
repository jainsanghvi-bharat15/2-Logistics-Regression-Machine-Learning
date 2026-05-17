# 1. Import required libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 2. Load dataset (Employee Retention Data)
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(2) Logistic Regression/Employee_LoR/2. Employee Retantion_LoR.csv")
print("Original Data:\n", df.head(5))

# 3. Understanding data (Optional Analysis)
    # 'left' column → 1 = employee left, 0 = employee retained
    # Crosstab shows frequency comparison between two variables
    # Example: salary vs employee leaving
pd.crosstab(df.salary, df.left).plot(kind='bar')
plt.show()

    # Example: department vs employee leaving
pd.crosstab(df.Department, df.left).plot(kind='bar')
plt.show()

# 4. Select important features
sdf = df[['satisfaction_level', 'average_montly_hours',
          'promotion_last_5years', 'salary']]
print("\nSelected Features:\n", sdf.head(5))

# 5. Convert categorical 'salary' into dummy variables
salary_dummy = pd.get_dummies(sdf.salary, prefix='salary')
print("\nSalary Dummy Variables:\n", salary_dummy.head())

# Combine dummy variables with main dataset
fdf = pd.concat([sdf, salary_dummy], axis='columns')

# Drop original 'salary' column (no longer needed)
fdf = fdf.drop('salary', axis='columns')
print("\nData after encoding:\n", fdf.head(5))

# 6. Define independent (X) and dependent (Y) variables
x = fdf            # Features
y = df.left        # Target (0 = retained, 1 = left)

# 7. Split dataset into training and testing sets
xtr, xte, ytr, yte = train_test_split(x, y, train_size=0.7, random_state=42)
print("\nData Split Sizes:")
print("x_train:", len(xtr), "x_test:", len(xte))
print("y_train:", len(ytr), "y_test:", len(yte))

# 8. Train Logistic Regression model
model = LogisticRegression()
model.fit(xtr, ytr)

# 9. Make predictions
pred = model.predict(xte)
print("\nPredicted values:\n", pred)

# 10. Evaluate model performance
print("\nTraining Accuracy:", model.score(xtr, ytr))
print("Testing Accuracy:", model.score(xte, yte))

# 11. Visualization: Actual vs Predicted comparison
plt.scatter(yte, pred, color='blue', marker='o')
plt.xlabel("Actual (0 = Retained, 1 = Left)")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Employee Retention")
plt.grid(True)
plt.show()

# 12. Visualization: Count of employees left vs retained
pd.crosstab(df.left, columns="Count").plot(kind='bar', legend=False)
plt.title("Employee Retention Distribution")
plt.xlabel("Employee Status (0 = Retained, 1 = Left)")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.show()