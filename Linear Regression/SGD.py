import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the training and test data
train_data_path = 'train-concrete.csv'
test_data_path = 'test-concrete.csv'

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv(train_data_path, header=None)
test_df = pd.read_csv(test_data_path, header=None)

# Separate features (X) and target (y) from the training and test datasets
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
max_iters = 5000
cost_history = []

# Define cost function (Mean Squared Error)
def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X.dot(w) + b
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)
    return cost

# Define stochastic gradient descent update
def stochastic_gradient_descent_step(X, y, w, b, learning_rate, i):
    prediction = np.dot(X[i], w) + b
    error = prediction - y[i]
    dw = X[i] * error
    db = error
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Perform stochastic gradient descent
for t in range(max_iters):
    # Randomly sample an index
    i = np.random.randint(0, m)

    # Update the weights and bias
    w, b = stochastic_gradient_descent_step(X_train_norm, y_train, w, b, learning_rate, i)

    # Compute cost for the entire training set after this update
    cost = compute_cost(X_train_norm, y_train, w, b)
    cost_history.append(cost)

    # Gradually decrease learning rate if needed
    if t > 0 and t % 1000 == 0:
        learning_rate /= 2

# Calculate the cost on the test set using the final weight vector
test_cost = compute_cost(X_test_norm, y_test, w, b)

# Plot cost function value over updates
plt.plot(cost_history)
plt.xlabel('Update Step')
plt.ylabel('Cost')
plt.title('Cost Function Value During Stochastic Gradient Descent')
plt.show()

# Output the final learned weights, bias term, and test set cost
print("Learned weight vector:", w)
print("Bias term:", b)
print("Cost on test data:", test_cost)
