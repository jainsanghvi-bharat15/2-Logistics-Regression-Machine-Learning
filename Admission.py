# Project: Admission Chance Predictor using Logistic Regression
# 1. Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(2) Logistic Regression/Admission_LoR/5. Admission_Data_LoR.csv")
print(df.isnull().sum())    # Optional: Check missing values

# 3. Define features (X) and target (Y)
X = df[['GRE', 'TOEFL', 'SOP', 'CGPA', 'Research']] # Features → Input variables
y = df['Admitted']  # Target → Output (0 = Not Admitted, 1 = Admitted)

# 4. Split dataset into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)# Increased iterations for better convergence
model.fit(X_train, y_train)
print("Model Trained:", model)

# 6. Make predictions on test data
y_pred = model.predict(X_test)

# 7. Evaluate model performance
    # Accuracy → percentage of correct predictions
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed performance (optional)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Take user input and predict admission
print("\n--- Predict Admission ---")
gre = float(input("Enter GRE Score: "))
toefl = float(input("Enter TOEFL Score: "))
sop = float(input("Enter SOP Score (1-5): "))
cgpa = float(input("Enter CGPA (out of 10): "))
research = int(input("Research Experience (1 = Yes, 0 = No): "))

# 9. Predict probability using predict_proba()
    # predict_proba() → returns probability for each class [P(0), P(1)]
    # [0][1] → probability of class 1 (Admitted)
admit_prob = model.predict_proba([[gre, toefl, sop, cgpa, research]])[0][1]

# 10. Predict final class (0 or 1)
admit_class = model.predict([[gre, toefl, sop, cgpa, research]])[0]

# 11. Display results
print(f"\nPredicted Chance of Admission: {admit_prob * 100:.2f}%")
print("Prediction:", "Admitted" if admit_class == 1 else "Not Admitted")

# 12. Optional: Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()