import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Sigmoid activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# Initialize all weights and biases to zero
def initialize_weights(input_size, hidden_size, output_size):
    weights = {
        "W1": np.zeros((hidden_size, input_size)),  # Layer 1 weights
        "b1": np.zeros((hidden_size, 1)),  # Layer 1 biases
        "W2": np.zeros((hidden_size, hidden_size)),  # Layer 2 weights
        "b2": np.zeros((hidden_size, 1)),  # Layer 2 biases
        "W3": np.zeros((output_size, hidden_size)),  # Output layer weights
        "b3": np.zeros((output_size, 1)),  # Output layer bias
    }
    return weights


# Forward propagation
def forward_propagation(X, weights):
    Z1 = np.dot(weights["W1"], X.T) + weights["b1"]
    A1 = sigmoid(Z1)

    Z2 = np.dot(weights["W2"], A1) + weights["b2"]
    A2 = sigmoid(Z2)

    Z3 = np.dot(weights["W3"], A2) + weights["b3"]
    A3 = sigmoid(Z3)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache


# Backward propagation
def backward_propagation(X, Y, weights, cache):
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    Z1, Z2, Z3 = cache["Z1"], cache["Z2"], cache["Z3"]
    m = X.shape[0]

    dZ3 = (A3 - Y.T) * sigmoid_derivative(Z3)
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dZ2 = np.dot(weights["W3"].T, dZ3) * sigmoid_derivative(Z2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(weights["W2"].T, dZ2) * sigmoid_derivative(Z1)
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dW3": dW3, "db3": db3, "dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}
    return gradients


# Update weights using SGD and learning rate schedule
def update_weights(weights, gradients, learning_rate):
    for key in weights:
        weights[key] -= learning_rate * gradients["d" + key]
    return weights


# Learning rate scheduler
def learning_rate_schedule(t, gamma0, d):
    return gamma0 / (1 + (gamma0 / d) * t)


# Compute loss
def compute_loss(Y, A3):
    m = Y.shape[0]
    loss = np.mean(0.5 * (A3.T - Y) ** 2)
    return loss


# Compute accuracy
def compute_accuracy(X, Y, weights):
    A3, _ = forward_propagation(X, weights)
    predictions = (A3.T > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)
    return accuracy


# Train the network using SGD
def train_network(X_train, Y_train, X_test, Y_test, hidden_size, gamma0, d, epochs=50):
    input_size = X_train.shape[1]
    output_size = 1
    weights = initialize_weights(input_size, hidden_size, output_size)
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # Shuffle the training data
        shuffle_indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[shuffle_indices], Y_train[shuffle_indices]

        for t in range(X_train.shape[0]):
            X_batch = X_train[t, :].reshape(1, -1)
            Y_batch = Y_train[t].reshape(1, 1)

            # Forward and backward propagation
            A3, cache = forward_propagation(X_batch, weights)
            gradients = backward_propagation(X_batch, Y_batch, weights, cache)

            # Update weights
            learning_rate = learning_rate_schedule(t + 1, gamma0, d)
            weights = update_weights(weights, gradients, learning_rate)

        # Compute train and test losses
        A3_train, _ = forward_propagation(X_train, weights)
        A3_test, _ = forward_propagation(X_test, weights)
        train_loss = compute_loss(Y_train, A3_train)
        test_loss = compute_loss(Y_test, A3_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return weights, train_losses, test_losses


# Main function
def main():
    # Load data
    train_data = pd.read_csv("train-bank-note-NN.csv", header=None)
    test_data = pd.read_csv("test-bank-note-NN.csv", header=None)

    X_train, Y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values.reshape(-1, 1)
    X_test, Y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values.reshape(-1, 1)

    # Hyperparameters
    widths = [5, 10, 25, 50, 100]
    gamma0 = 0.1
    d = 50
    epochs = 50

    for width in widths:
        print(f"\nTraining with hidden layer width = {width} (Zero Initialization)")

        # Train the network
        weights, train_losses, test_losses = train_network(X_train, Y_train, X_test, Y_test, width, gamma0, d, epochs)

        # Compute final training and test errors
        train_accuracy = compute_accuracy(X_train, Y_train, weights)
        test_accuracy = compute_accuracy(X_test, Y_test, weights)

        train_error = 1 - train_accuracy
        test_error = 1 - test_accuracy

        # Print the errors
        print(f"Hidden Layer Width: {width}")
        print(f"Final Training Error: {train_error:.4f}")
        print(f"Final Test Error: {test_error:.4f}")

        # Plot the losses
        plt.plot(train_losses, label=f"Train Loss (Width {width})")
        plt.plot(test_losses, label=f"Test Loss (Width {width})")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss with Zero Initialization")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
