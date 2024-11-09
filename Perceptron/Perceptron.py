import pandas as pd
import numpy as np

# Load training and testing data
train_path = 'train-bank-note.csv'
test_path = 'test-bank-note.csv'

train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

# Separate features and labels
X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

# Convert labels to -1 and 1 for the Perceptron
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Initialize parameters
epochs = 10
weights = np.zeros(X_train.shape[1] + 1)  # Including bias term

# Standard Perceptron training function
def train_perceptron(X, y, weights, epochs):
    for epoch in range(epochs):
        for i in range(len(X)):
            x_i = np.insert(X[i], 0, 1)  # Inset bias term
            if y[i] * np.dot(weights, x_i) <= 0:
                weights += y[i] * x_i  # Update weights
    return weights

# Train the Perceptron model
learned_weights = train_perceptron(X_train, y_train, weights, epochs)

# Prediction function
def predict(X, weights):
    X_with_bias = np.insert(X, 0, 1, axis=1)  # Add bias term to each example
    return np.sign(np.dot(X_with_bias, weights))

# Make predictions and calculate the average prediction error on the test set
y_pred_test = predict(X_test, learned_weights)
test_error = np.mean(y_pred_test != y_test)

# Print the learned weights
formatted_weights = ", ".join([f"{w:.2f}" for w in learned_weights])
print(f"Learned weight vector: [{formatted_weights}]")
print(f"Average prediction error on the test dataset (Standard Perceptron): {test_error:.2f}")


# Voted Perceptron training function
def train_voted_perceptron(X, y, epochs):
    weights = np.zeros(X.shape[1] + 1)
    weight_vectors = []
    count = 1  # Start with a count of 1 for the first weight vector

    for epoch in range(epochs):
        for i in range(len(X)):
            x_i = np.insert(X[i], 0, 1)
            if y[i] * np.dot(weights, x_i) <= 0:  # Misclassified
                weight_vectors.append((weights.copy(), count))
                weights += y[i] * x_i
                count = 1  # Reset count for the new weight vector
            else:
                count += 1  # Increase count if correctly classified

    # Append the final weights and count
    weight_vectors.append((weights.copy(), count))
    return weight_vectors

# Training the Voted Perceptron model
voted_weight_vectors = train_voted_perceptron(X_train, y_train, epochs)

# Voted prediction function
def predict_voted(X, weight_vectors):
    X_with_bias = np.insert(X, 0, 1, axis=1)  # Add bias term to each example
    final_predictions = []

    for x in X_with_bias:
        vote_sum = 0
        for weights, count in weight_vectors:
            prediction = np.sign(np.dot(weights, x))
            vote_sum += count * prediction

        final_predictions.append(np.sign(vote_sum))
    return np.array(final_predictions)

# Make predictions and calculate the average prediction error on the test set for Voted Perceptron
y_pred_test_voted = predict_voted(X_test, voted_weight_vectors)
test_error_voted = np.mean(y_pred_test_voted != y_test)

# Display distinct weight vectors and their counts, and the average test error
distinct_weights_counts = [(vec.tolist(), cnt) for vec, cnt in voted_weight_vectors]

# Print results for the voted Perceptron
print("\nDistinct weight vectors and their counts (Voted Perceptron):")
for vec, cnt in distinct_weights_counts:
    formatted_vec = ", ".join([f"{v:.2f}" for v in vec])
    print(f"Weight vector: [{formatted_vec}], Count: {cnt}")

print(f"\nAverage prediction error on the test dataset (Voted Perceptron): {test_error_voted:.2f}")


# Average Perceptron training function
def train_average_perceptron(X, y, epochs):
    weights = np.zeros(X.shape[1] + 1)
    weight_sum = np.zeros(X.shape[1] + 1)  # Initialize accumulator for the average
    total_updates = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            x_i = np.insert(X[i], 0, 1)
            if y[i] * np.dot(weights, x_i) <= 0:
                weights += y[i] * x_i
            weight_sum += weights
            total_updates += 1

    # Calculate the average weight vector
    averaged_weights = weight_sum / total_updates
    return averaged_weights

# Train the Average Perceptron model
averaged_weights = train_average_perceptron(X_train, y_train, epochs)

# Prediction function for Average Perceptron
def predict_average(X, weights):
    X_with_bias = np.insert(X, 0, 1, axis=1)
    return np.sign(np.dot(X_with_bias, weights))

# Make predictions and calculate the average prediction error on the test set for Average Perceptron
y_pred_test_average = predict_average(X_test, averaged_weights)
test_error_average = np.mean(y_pred_test_average != y_test)

# Print results for the average Perceptron
formatted_averaged_weights = ", ".join([f"{w:.2f}" for w in averaged_weights])
print(f"\nLearned weight vector (Average Perceptron): [{formatted_averaged_weights}]")
print(f"Average prediction error on the test dataset (Average Perceptron): {test_error_average:.2f}")