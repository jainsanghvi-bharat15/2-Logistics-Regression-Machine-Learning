# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Load dataset (Loan Prediction Data)
df_loan = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(2) Logistic Regression/Loan_LoR/4. Loan_Dataset_LoR.csv")
print("Dataset Preview:\n", df_loan.head())

# 3. Basic dataset information
print("\nShape (Rows, Columns):", df_loan.shape)
print("\nMissing Values:\n", df_loan.isnull().sum())    # Check missing values

# 4. Handle missing values
df_loan = df_loan.dropna()  # Drop rows with missing values
print("\nAfter removing missing values:\n", df_loan.isnull().sum())
print("Updated Shape:", df_loan.shape)

# 5. Explore categorical data
print("\nEducation:\n", df_loan['Education'].value_counts())
print("\nGender:\n", df_loan['Gender'].value_counts())
print("\nMarried:\n", df_loan['Married'].value_counts())
print("\nDependents:\n", df_loan['Dependents'].value_counts())

# Replace '3+' with numeric value 4
df_loan.replace({'Dependents': {'3+': 4}}, inplace=True)

# 6. Data Visualization (Relationship with Loan Status)
sn.countplot(x='Education', hue='Loan_Status', data=df_loan)
plt.title("Education vs Loan Status")
plt.show()

sn.countplot(x='Dependents', hue='Loan_Status', data=df_loan)
plt.title("Dependents vs Loan Status")
plt.show()

sn.countplot(x='Married', hue='Loan_Status', data=df_loan)
plt.title("Marital Status vs Loan Status")
plt.show()

sn.countplot(x='Property_Area', hue='Loan_Status', data=df_loan)
plt.title("Property Area vs Loan Status")
plt.show()

# 7. Convert categorical values into numeric (Label Encoding)
df_loan.replace({'Loan_Status': {'Y': 1, 'N': 0}}, inplace=True)
df_loan.replace({'Married': {'Yes': 1, 'No': 0}}, inplace=True)
df_loan.replace({'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
df_loan.replace({'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}}, inplace=True)
df_loan.replace({'Gender': {'Male': 0, 'Female': 1}}, inplace=True)
df_loan.replace({'Self_Employed': {'Yes': 1, 'No': 0}}, inplace=True)
print("\nData after encoding:\n", df_loan.head())

# 8. Define independent (X) and dependent (Y) variables
    # Drop non-useful column Loan_ID
x = df_loan.drop(['Loan_ID', 'Loan_Status'], axis='columns')
y = df_loan['Loan_Status']

# 9. Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
print("\nTraining size:", len(x_train))
print("Testing size:", len(x_test))

# 10. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased iterations for convergence
model.fit(x_train, y_train)

# 11. Prediction on training data
y_pred_train = model.predict(x_train)

# Accuracy on training data
train_accuracy = accuracy_score(y_train, y_pred_train)
print("\nAccuracy on Training Data:", train_accuracy)

# 12. Prediction on testing data
y_pred_test = model.predict(x_test)

# Accuracy on testing data
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Accuracy on Testing Data:", test_accuracy)

# Alternative accuracy check
print("Model Score (Testing Data):", model.score(x_test, y_test))

# 13. Predict for a new sample input
# Input format must match feature order in X
# Example prediction:
prediction = model.predict([[0, 1, 0, 1, 1, 3000, 0.0, 66.0, 360.0, 1.0, 2]])
print("\nPrediction for new data (0 = No Loan, 1 = Loan Approved):", prediction)