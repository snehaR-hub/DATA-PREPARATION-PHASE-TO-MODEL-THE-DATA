'Balancing a training dataset is a critical task, especially when dealing with imbalanced datasets. An imbalanced dataset is one where the target variable has a disproportionate number of observations in one or more classes, which can cause models to be biased towards the majority class. Balancing the dataset helps improve model performance by ensuring that the model learns from all classes equally.

#1. Resampling Methods
'1.1. Oversampling (Increase Minority Class Instances)
'Oversampling involves increasing the number of samples from the minority class by duplicating examples or generating synthetic examples.

import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Create a synthetic imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                            n_classes=2, weights=[0.9, 0.1], random_state=42)

# Display the original class distribution
print("Original class distribution:", Counter(y))

# Perform random oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Display the class distribution after oversampling
print("Class distribution after oversampling:", Counter(y_resampled))

#1.2. Undersampling
'Undersampling involves reducing the number of instances in the majority class to match the minority class.

from imblearn.under_sampling import RandomUnderSampler

# Display the original class distribution
print("Original class distribution:", Counter(y))

# Perform random undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Display the class distribution after undersampling
print("Class distribution after undersampling:", Counter(y_resampled))

#1.3. SMOTE 
'SMOTE creates synthetic samples for the minority class by interpolating between existing instances. This can help avoid overfitting, which might occur when randomly duplicating instances.

from imblearn.over_sampling import SMOTE

# Display the original class distribution
print("Original class distribution:", Counter(y))

# Perform SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Display the class distribution after SMOTE
print("Class distribution after SMOTE:"

#2. Algorithm-Level Approaches
'Instead of resampling the dataset, you can adjust the models parameters to give more weight to the minority class. This can be particularly useful when resampling may introduce noise or if you prefer not to modify the dataset.

 from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Logistic Regression with class weights
model = LogisticRegression(class_weight='balanced', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

     
