import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define column names and numerical attributes based on the description file
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
           'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Preprocess data to replace "unknown" and convert numerical features
def preprocess_data(data):
    # Replace "unknown" with the majority value for categorical attributes
    for col in data.columns:
        if col not in numerical_attributes:  # Process categorical columns
            majority_value = data[col].mode()[0]
            data[col] = data[col].replace('unknown', majority_value)

    # Convert numerical attributes to binary using the median of each column
    for col in numerical_attributes:
        median = data[col].median()
        data[col] = (data[col] > median).astype(int)

    # Convert categorical columns to numerical codes
    for col in data.select_dtypes(include='object').columns:
        data[col] = pd.factorize(data[col])[0]

    return data

# Load and rename columns for both train and test datasets
train_df = pd.read_csv('train-bank-1.csv', header=None)
test_df = pd.read_csv('test-bank-1.csv', header=None)
train_df.columns = columns
test_df.columns = columns

# Preprocess both datasets
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Separate features and target variable
X_train, y_train = train_df.drop('y', axis=1).values, train_df['y'].values
X_test, y_test = test_df.drop('y', axis=1).values, test_df['y'].values

# Implement Decision Stump with Information Gain
def decision_stump(X, y, sample_weights):
    n_samples, n_features = X.shape
    best_feature, best_threshold, best_gain = None, None, -1
    best_pred = None

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            predictions = (X[:, feature] > threshold).astype(int)
            weighted_error = np.sum(sample_weights * (predictions != y))

            if weighted_error < best_gain or best_feature is None:
                best_feature = feature
                best_threshold = threshold
                best_gain = weighted_error
                best_pred = predictions

    return best_feature, best_threshold, best_pred, best_gain

# Implement AdaBoost Algorithm
def adaboost(X, y, X_test, y_test, T=500):
    n_samples, n_features = X.shape
    sample_weights = np.full(n_samples, 1 / n_samples)
    stump_weights = []
    stumps = []
    errors_train = []
    errors_test = []

    for t in range(T):
        feature, threshold, stump_pred, stump_error = decision_stump(X, y, sample_weights)
        stump_weight = 0.5 * np.log((1 - stump_error) / (stump_error + 1e-10))

        # Update sample weights
        sample_weights *= np.exp(-stump_weight * y * ((X[:, feature] > threshold) * 2 - 1))
        sample_weights /= sample_weights.sum()

        # Track stumps and their weights
        stumps.append((feature, threshold))
        stump_weights.append(stump_weight)

        # Compute aggregated predictions for training and testing sets
        y_train_pred = np.sign(
            np.dot(stump_weights, [(X[:, feature] > threshold) * 2 - 1 for feature, threshold in stumps]))
        y_test_pred = np.sign(
            np.dot(stump_weights, [(X_test[:, feature] > threshold) * 2 - 1 for feature, threshold in stumps]))

        errors_train.append(1 - accuracy_score(y, y_train_pred))
        errors_test.append(1 - accuracy_score(y_test, y_test_pred))

    return errors_train, errors_test

# Plot Training and Test Errors
def plot_error_over_iterations(errors_train, errors_test):
    # Figure 1: Overall error over AdaBoost iterations
    plt.figure(figsize=(12, 6))
    plt.plot(errors_train, label='Training Error')
    plt.plot(errors_test, label='Test Error')
    plt.xlabel('Iterations (T)')
    plt.ylabel('Error')
    plt.title('Overall Training and Test Error over AdaBoost Iterations')
    plt.legend()
    plt.show()

def plot_error_per_stump(errors_train, errors_test):
    # Figure 2: Error for each stump in AdaBoost
    plt.figure(figsize=(12, 6))
    plt.plot([e for e in errors_train], label='Training Error for Stumps', marker='o')
    plt.plot([e for e in errors_test], label='Test Error for Stumps', marker='x')
    plt.xlabel('Stump Index (T)')
    plt.ylabel('Error')
    plt.title('Training and Test Error for Each Decision Stump')
    plt.legend()
    plt.show()

# Train AdaBoost and Plot Errors
errors_train, errors_test = adaboost(X_train, y_train, X_test, y_test)
plot_error_over_iterations(errors_train, errors_test)
plot_error_per_stump(errors_train, errors_test)
