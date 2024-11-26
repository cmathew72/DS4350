from scipy.optimize import minimize
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to calculate the dual objective
def dual_objective(alpha, K, y, epsilon=1e-6):
    """
    Dual SVM objective function to maximize (negative for minimization).
    """
    return -np.sum(alpha) + 0.5 * np.sum(alpha * alpha * y[:, None] * y[None, :] * K) + epsilon * np.sum(alpha**2)

# Function to compute weights and bias
def compute_weights_and_bias(alpha, X, y, C, tol=1e-5):
    """
    Compute weights and bias from optimized alpha values.
    """
    # Correct computation of weights
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)  # Ensure proper broadcasting

    # Identify support vector indices
    sv_indices = np.where((alpha > tol) & (alpha < C))[0]
    if len(sv_indices) == 0:
        raise ValueError("No support vectors found.")

    # Compute bias as the average over support vectors
    b = np.mean(y[sv_indices] - np.dot(X[sv_indices], w))
    return w, b

# Main SVM training function
def train_svm_dual(X, y, C):
    """
    Train SVM using the dual formulation.
    """
    n_samples = X.shape[0]
    # Compute the Gram matrix
    K = np.dot(X, X.T)  # Linear kernel
    print("Gram matrix:\n", K)

    # Initialize alpha with small positive values
    alpha_init = np.full(n_samples, 1e-6)
    bounds = [(0, C) for _ in range(n_samples)]

    # Define the equality constraint: sum(alpha_i * y_i) = 0
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}

    # Minimize the dual objective
    result = minimize(
        fun=dual_objective,
        x0=alpha_init,
        args=(K, y),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'ftol': 1e-6, 'maxiter': 1000}
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return result.x, K

# Load your data here
# Assuming X_train and y_train are loaded
# Example placeholders (replace with your actual data)
X_train = np.random.randn(100, 2)  # 100 samples, 2 features
y_train = np.random.choice([-1, 1], size=100)

# Normalize features and ensure labels are -1 or +1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = np.where(y_train == 0, -1, y_train)

# Training the SVM with multiple C values
for C in [1, 10, 100]:
    print(f"Training SVM with C = {C}...")
    try:
        alpha, K = train_svm_dual(X_train, y_train, C)
        print("Alpha values:", alpha)
        print("Number of non-zero alphas:", np.sum(alpha > 1e-5))
        w, b = compute_weights_and_bias(alpha, X_train, y_train, C)
        print(f"Weight vector: {w}, Bias: {b}")
    except Exception as e:
        print(f"Error for C={C}: {e}")
