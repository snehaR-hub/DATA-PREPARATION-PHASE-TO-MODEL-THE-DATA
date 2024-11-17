'Partitioning data in Python refers to splitting your dataset into distinct subsets for various purposes, such as training, validation, and testing for machine learning tasks. Partitioning is essential for ensuring that models are evaluated on data that was not used for training, which helps to prevent overfitting and ensures better generalization.

#1. Random Split (Train-Test Split)
'A simple random split divides the dataset into a training set and a testing set.

import pandas as pd
from sklearn.model_selection import train_test_split

# Example data: create a DataFrame
data = {'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}

df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df.drop('Target', axis=1)  # Features
y = df['Target']  # Target variable

# Perform the train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the results
print(f"Training features:\n{X_train}\n")
print(f"Testing features:\n{X_test}\n")

#2. K-Fold Cross-Validation
'In K-Fold Cross-Validation, the data is divided into K folds. The model is trained K times, each time using K-1 folds for training and the remaining fold for testing. This is useful to reduce bias and provide a better estimate of model performance.

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load a dataset (Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Initialize a Logistic Regression model
model = LogisticRegression(max_iter=200)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Display the cross-validation scores
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean()}")


#3. Stratified Split
'In cases where the target variable is imbalanced (e.g., 90% of the samples belong to one class and 10% to another), using a stratified split ensures that the proportions of each class are preserved in both the training and test sets. This is especially important for classification problems.

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2,
                            n_classes=2, weights=[0.9, 0.1], random_state=42)

# Perform stratified split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Check the distribution of classes in both training and test sets
print(f"Training class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

#4. Time-Based Split 
'When working with time-series data, it’s important to partition the data chronologically (i.e., using past data for training and future data for testing). This prevents future data from "leaking" into the training set.

import pandas as pd
from sklearn.model_selection import train_test_split

# Example: Create a time-based dataset
date_range = pd.date_range(start='1/1/2020', periods=10, freq='D')
data = {'Date': date_range, 'Feature': range(10, 20), 'Target': range(100, 110)}

df = pd.DataFrame(data)

# Split the data chronologically: First 7 rows for training, last 3 rows for testing
train = df.iloc[:7]
test = df.iloc[7:]

print("Training Data:")
print(train)

print("\nTesting Data:")
print(test)

