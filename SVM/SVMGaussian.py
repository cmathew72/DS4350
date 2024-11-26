import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Gaussian Kernel
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)


# Compute Kernel Matrix
def compute_kernel_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K


# Dual Objective Function
def objective_function(alpha, y, K):
    return 0.5 * np.sum(alpha[:, None] * alpha * y[:, None] * y * K) - np.sum(alpha)


# Equality Constraint for Dual Problem
def equality_constraint(alpha, y):
    return np.dot(alpha, y)


# Solve SVM Dual Problem
def solve_dual_svm(X, y, C, gamma):
    n_samples = X.shape[0]
    K = compute_kernel_matrix(X, gamma)

    # Initial alpha values
    alpha0 = np.zeros(n_samples)

    # Bounds for alpha
    bounds = [(0, C) for _ in range(n_samples)]

    # Constraints
    constraints = {
        'type': 'eq',
        'fun': equality_constraint,
        'args': (y,)
    }

    # Solve the optimization problem
    result = minimize(
        fun=objective_function,
        x0=alpha0,
        args=(y, K),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    alpha = result.x
    return alpha, K


# Predict Function
def predict(X_train, y_train, X_test, alpha, gamma, bias):
    y_pred = []
    for x in X_test:
        pred = sum(
            alpha[i] * y_train[i] * gaussian_kernel(X_train[i], x, gamma)
            for i in range(len(X_train))
        )
        y_pred.append(np.sign(pred + bias))
    return np.array(y_pred)


# Main Function
def main():
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameters
    gammas = [0.1, 0.5, 1, 5, 100]
    C_values = [100 / 873, 500 / 873, 700 / 873]

    best_gamma = None
    best_C = None
    best_accuracy = 0

    # Grid Search
    for gamma in gammas:
        for C in C_values:
            print(f"Training SVM with gamma={gamma}, C={C}...")
            alpha, K = solve_dual_svm(X_train, y_train, C, gamma)

            # Compute bias
            support_vectors = (alpha > 1e-5)
            support_alpha = alpha[support_vectors]
            support_y = y_train[support_vectors]
            support_K = K[support_vectors][:, support_vectors]

            bias = np.mean(
                support_y - np.sum(support_alpha * support_y[:, None] * support_K, axis=1)
            )

            # Predict on training and test sets
            y_train_pred = predict(X_train, y_train, X_train, alpha, gamma, bias)
            y_test_pred = predict(X_train, y_train, X_test, alpha, gamma, bias)

            # Compute errors
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

            # Update best combination
            if test_accuracy > best_accuracy:
                best_gamma = gamma
                best_C = C
                best_accuracy = test_accuracy

    print(f"Best Gamma: {best_gamma}, Best C: {best_C}, Best Test Accuracy: {best_accuracy}")


if __name__ == "__main__":
    main()
