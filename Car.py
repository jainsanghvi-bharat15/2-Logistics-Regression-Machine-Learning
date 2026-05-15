# 1. Import required libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics   # For R² score

# 2. Load dataset (Car data)
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(2) Logistic Regression/Car_LoR/3. Car_Data_LoR.csv")
print("Dataset Preview:\n", df.head(10))

# 3. Basic dataset information
print("\nType of dataset:", type(df))   # Check data type
print("\nShape (Rows, Columns):", df.shape)  # Number of rows & columns
print("\nDataset Info:")
df.info()

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# 4. Explore categorical data
print("\nFuel Type Count:\n", df.Fuel_Type.value_counts())
print("\nSeller Type Count:\n", df.Seller_Type.value_counts())
print("\nTransmission Count:\n", df.Transmission.value_counts())

# 5. Convert categorical data into numeric values
df.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
df.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
print("\nData after encoding:\n", df.head())

# 6. Define independent (X) and dependent (Y) variables
# Drop non-useful columns like Car_Name
x = df.drop(['Car_Name', 'Selling_Price'], axis='columns')  # Features
y = df['Selling_Price']                                     # Target

# 7. Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
print("\nTraining size:", len(x_train))
print("Testing size:", len(x_test))

# 8. Train Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# 9. Prediction on training data
y_pred_train = model.predict(x_train)

# Plot actual vs predicted (training data)
plt.scatter(y_train, y_pred_train, marker='.', color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Training Data: Actual vs Predicted Price")
plt.show()

# Calculate R² score for training data
train_score = metrics.r2_score(y_train, y_pred_train)
print("R² Score (Training Data):", train_score)

# 10. Prediction on testing data
y_pred_test = model.predict(x_test)

# Plot actual vs predicted (testing data)
plt.scatter(y_test, y_pred_test, marker='.', color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Testing Data: Actual vs Predicted Price")
plt.show()

# Calculate R² score for testing data
test_score = metrics.r2_score(y_test, y_pred_test)
print("R² Score (Testing Data):", test_score)

# 11. Model accuracy using built-in method
print("Model Score (Testing Data):", model.score(x_test, y_test))