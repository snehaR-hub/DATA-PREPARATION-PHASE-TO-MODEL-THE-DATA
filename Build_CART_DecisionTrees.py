'Building a CART  decision tree in Python can be done using the DecisionTreeClassifier for classification tasks and DecisionTreeRegressor for regression tasks, both from the scikit-learn library. CART is a non-linear algorithm that splits the dataset recursively based on the feature that maximizes some criterion (like Gini impurity for classification or mean squared error for regression) at each node.

#1. Classification Task: Using DecisionTreeClassifier

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the DecisionTreeClassifier (CART model)
model = DecisionTreeClassifier(criterion='gini', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True)
plt.show()

#2. Regression Task: Using DecisionTreeRegressor
'For regression tasks, we use the DecisionTreeRegressor. Let work with a simple regression example, where we predict a continuous target variable.

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create a simple dataset for regression
X = np.array([[i] for i in range(1, 11)])  # Feature: 1 to 10
y = np.array([2*i + 1 for i in range(1, 11)])  # Target: y = 2*x + 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the DecisionTreeRegressor (CART model)
regressor = DecisionTreeRegressor(random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

# Visualize the regression tree
plt.figure(figsize=(15, 10))
plot_tree(regressor, filled=True, feature_names=['X'], rounded=True)
plt.show()

