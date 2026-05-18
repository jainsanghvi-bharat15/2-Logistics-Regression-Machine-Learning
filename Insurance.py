# 1. Import required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as pt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 2. Load dataset (Insurance data)
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(2) Logistic Regression/Insurance_LoR/1. Insurance_LoR.csv")
print(df.head(10))  # Display first 10 rows

# 3. Visualize data (Age vs Bought Insurance)
    # bought_insurance → 0 (No), 1 (Yes)
pt.scatter(df.age, df.bought_insurance, marker='*', color='blue')
pt.xlabel("Age")
pt.ylabel("Bought Insurance (0/1)")
pt.title("Age vs Insurance Purchase")
pt.show()

# 4. Split dataset into training and testing sets
    # Features → age
    # Target → bought_insurance
x_tr, x_te, y_tr, y_te = train_test_split(df[['age']], df.bought_insurance,test_size=0.1)
print("Training data size:", len(x_tr))

# 5. Create and train Logistic Regression model
model = LogisticRegression()
model.fit(x_tr, y_tr)

# 6. Make predictions on test data
print("Predicted values for test data:", model.predict(x_te))

# 7. Check model accuracy
print("Accuracy on testing data:", model.score(x_te, y_te))
print("Accuracy on training data:", model.score(x_tr, y_tr))

# 8. Take user input and predict
n = int(input("Enter the age for which you want prediction: "))
print("Prediction (0 = No, 1 = Yes):", model.predict([[n]]))    # Model expects input in 2D array format
