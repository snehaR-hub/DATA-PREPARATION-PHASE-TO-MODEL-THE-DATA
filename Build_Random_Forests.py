'Building Random Forests in Python can be done using the RandomForestClassifier (for classification tasks) or RandomForestRegressor (for regression tasks) from the scikit-learn library. Random Forest is an ensemble learning method that combines multiple decision trees to improve model accuracy and robustness by reducing overfitting and variance.

Example 1 :-
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize feature importances
features = data.feature_names
importances = rf_model.feature_importances_

# Create a bar plot of feature importances
plt.figure(figsize=(8, 6))
sns.barplot(x=features, y=importances)
plt.title('Feature Importances in Random Forest')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.show()
