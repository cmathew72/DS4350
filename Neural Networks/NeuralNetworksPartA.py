import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Forward propagation
def forward_propagation(X, weights):
    # Layer 1
    Z1 = np.dot(weights["W1"], X) + weights["b1"]
    A1 = sigmoid(Z1)

    # Layer 2
    Z2 = np.dot(weights["W2"], A1) + weights["b2"]
    A2 = sigmoid(Z2)

    # Output Layer
    Z3 = np.dot(weights["W3"], A2) + weights["b3"]
    A3 = sigmoid(Z3)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache

# Backpropagation
def backward_propagation(X, Y, weights, cache):
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    Z1, Z2, Z3 = cache["Z1"], cache["Z2"], cache["Z3"]

    # Output layer error
    dZ3 = (A3 - Y) * sigmoid_derivative(Z3)
    dW3 = np.dot(dZ3, A2.T)
    db3 = np.sum(dZ3, axis=1, keepdims=True)

    # Layer 2 error
    dZ2 = np.dot(weights["W3"].T, dZ3) * sigmoid_derivative(Z2)
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    # Layer 1 error
    dZ1 = np.dot(weights["W2"].T, dZ2) * sigmoid_derivative(Z1)
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW3": dW3, "db3": db3, "dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}
    return gradients

# Initialize weights (from Table 1)
def initialize_weights():
    weights = {
        "W1": np.array([[-1, -2, -3], [1, -3, 3]]),    # Weights for Layer 1
        "b1": np.array([[-1], [1]]),                  # Biases for Layer 1
        "W2": np.array([[-1, -2], [1, 2], [2, -3]]),  # Weights for Layer 2
        "b2": np.array([[-1], [1], [2]]),            # Biases for Layer 2
        "W3": np.array([[3, 1, 2]]),                 # Weights for Output Layer
        "b3": np.array([[0]])                       # Bias for Output Layer
    }
    return weights

# Main function to perform forward and backward propagation
def compute_gradients():
    # Input and target label
    X = np.array([[1], [1], [1]])
    Y = np.array([[1]])

    # Initialize weights
    weights = initialize_weights()

    # Forward propagation
    A3, cache = forward_propagation(X, weights)

    # Compute loss
    loss = 0.5 * (A3 - Y) ** 2
    print(f"Forward Pass: Prediction: {A3[0,0]:.6f}, Loss: {loss[0,0]:.6f}")

    # Backward propagation
    gradients = backward_propagation(X, Y, weights, cache)

    # Print gradients
    print("\nGradients:")
    print("dW3:", gradients["dW3"])
    print("db3:", gradients["db3"])
    print("dW2:", gradients["dW2"])
    print("db2:", gradients["db2"])
    print("dW1:", gradients["dW1"])
    print("db1:", gradients["db1"])

if __name__ == "__main__":
    compute_gradients()
