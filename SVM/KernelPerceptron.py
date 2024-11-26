import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Gaussian kernel function
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)


# Kernel Perceptron implementation
def kernel_perceptron(X, y, gamma, max_epochs=100):
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)  # Initialize alpha (weights)
    kernel_matrix = np.zeros((n_samples, n_samples))

    # Compute kernel matrix
    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = gaussian_kernel(X[i], X[j], gamma)

    # Training
    for epoch in range(max_epochs):
        updates = 0
        for i in range(n_samples):
            prediction = np.sign(np.sum(alpha * y * kernel_matrix[:, i]))
            if prediction != y[i]:  # Update rule
                alpha[i] += 1
                updates += 1
        if updates == 0:  # Stop if no updates
            break
    return alpha, kernel_matrix


# Prediction function
def predict_kernel_perceptron(X_train, X_test, alpha, y_train, gamma):
    predictions = []
    for x_test in X_test:
        kernel_values = np.array([gaussian_kernel(x_test, x_train, gamma) for x_train in X_train])
        prediction = np.sign(np.sum(alpha * y_train * kernel_values))
        predictions.append(prediction)
    return np.array(predictions)


# Main function
def main():
    # Generate synthetic dataset
    X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=2, n_redundant=0,
                               random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert labels to {-1, 1}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gamma_values = [0.1, 0.5, 1, 5, 100]
    results = []

    for gamma in gamma_values:
        print(f"Training Kernel Perceptron with gamma={gamma}...")
        alpha, kernel_matrix = kernel_perceptron(X_train, y_train, gamma)
        y_train_pred = predict_kernel_perceptron(X_train, X_train, alpha, y_train, gamma)
        y_test_pred = predict_kernel_perceptron(X_train, X_test, alpha, y_train, gamma)

        train_error = np.mean(y_train_pred != y_train)
        test_error = np.mean(y_test_pred != y_test)

        results.append((gamma, train_error, test_error))
        print(f"Gamma: {gamma}, Train Error: {train_error}, Test Error: {test_error}")

    print("\nResults:")
    for gamma, train_error, test_error in results:
        print(f"Gamma: {gamma}, Train Error: {train_error}, Test Error: {test_error}")


if __name__ == "__main__":
    main()
