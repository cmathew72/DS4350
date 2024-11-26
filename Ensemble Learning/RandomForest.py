import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv('train-bank-1.csv', header=None)
test_data = pd.read_csv('test-bank-1.csv', header=None)

# Attribute names from 'data-desc-bank-note-bank-1.txt'
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
           'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data.columns = columns
test_data.columns = columns

# Numerical attributes
numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

def preprocess_data(data, target='y'):
    # Replace 'unknown' with the most frequent value in categorical columns
    for col in data.columns:
        if col not in numerical_attributes and col != target:  # Only process categorical columns except target
            majority_value = data[col].mode()[0]  # Get the most frequent value
            data[col] = data[col].replace('unknown', majority_value)  # Replace "unknown" with the majority value

    # One-hot encode categorical features, but exclude the target column
    data = pd.get_dummies(data,
                          columns=[col for col in data.columns if col not in numerical_attributes and col != target])

    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Separate input features and target label
X_train = train_data.drop('y', axis=1)
y_train = train_data['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Convert to binary 0/1
X_test = test_data.drop('y', axis=1)
y_test = test_data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Random Forest Implementation
class RandomForest:
    def __init__(self, n_trees=100, max_features=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_features=self.max_features)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(tree_preds, axis=0))

# Varying number of trees and feature subsets
n_trees_range = range(1, 501)
max_features_options = [2, 4, 6]

train_errors = {2: [], 4: [], 6: []}
test_errors = {2: [], 4: [], 6: []}

count = 0

for max_features in max_features_options:
    for n_trees in n_trees_range:
        rf = RandomForest(n_trees=n_trees, max_features=max_features)
        rf.fit(X_train, y_train)

        # Training predictions and error
        y_train_pred = rf.predict(X_train)
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        train_errors[max_features].append(train_error)

        # Test predictions and error
        y_test_pred = rf.predict(X_test)
        test_error = 1 - accuracy_score(y_test, y_test_pred)
        test_errors[max_features].append(test_error)

        count += 1
        print(count)

# Plotting training and test errors
plt.figure(figsize=(12, 8))
for max_features in max_features_options:
    plt.plot(n_trees_range, train_errors[max_features], label=f'Train Error (max_features={max_features})')
    plt.plot(n_trees_range, test_errors[max_features], label=f'Test Error (max_features={max_features})')

plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Training and Test Errors vs Number of Trees')
plt.legend()
plt.show()
