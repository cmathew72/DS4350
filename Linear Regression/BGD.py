import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the training and test data
train_data_path = 'train-concrete.csv'
test_data_path = 'test-concrete.csv'

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv(train_data_path, header=None)
test_df = pd.read_csv(test_data_path, header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Normalize the features (mean 0, std deviation 1)
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

X_train_norm = (X_train - X_train_mean) / X_train_std
X_test_norm = (X_test - X_train_mean) / X_train_std

# Initialize parameters
m, n = X_train_norm.shape  # m = number of examples, n = number of features
w = np.zeros(n)
b = 0
learning_rate = 0.1
tolerance = 1e-6
max_iters = 1000
cost_history = []
learning_rate_history = []

# Define cost function (Mean Squared Error)
def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X.dot(w) + b
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)
    return cost

# Define gradient descent update
def gradient_descent_step(X, y, w, b, learning_rate):
    m = len(y)
    predictions = X.dot(w) + b
    errors = predictions - y
    dw = (1 / m) * np.dot(X.T, errors)
    db = (1 / m) * np.sum(errors)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Perform gradient descent with learning rate adjustment
for i in range(max_iters):
    # Save previous weight to check for convergence
    w_prev = np.copy(w)

    # Gradient descent step
    w, b = gradient_descent_step(X_train_norm, y_train, w, b, learning_rate)

    # Compute cost and store it
    cost = compute_cost(X_train_norm, y_train, w, b)
    cost_history.append(cost)

    # Check convergence
    weight_diff = np.linalg.norm(w - w_prev)
    if weight_diff < tolerance:
        print(f"Converged at iteration {i + 1}")
        break

    # Tune learning rate if necessary (reduce it after every 100 iterations)
    if i > 0 and i % 100 == 0:
        # If the cost is not decreasing significantly, reduce the learning rate
        if len(cost_history) > 1 and cost_history[-2] - cost_history[-1] < 1e-4:
            learning_rate /= 2
            print(f"Iteration {i}, reducing learning rate to {learning_rate}")
            print(f"Iteration {i}, cost {cost}")
        learning_rate_history.append(learning_rate)

# Calculate the cost on the test set using the final weight vector
test_cost = compute_cost(X_test_norm, y_test, w, b)

# Plot cost function value over iterations
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Value During Gradient Descent')
plt.show()

# Output the final learned weights, bias term, and test set cost
print("Learned weight vector:", w)
print("Bias term:", b)
print("Cost on test data:", test_cost)
