import numpy as np
import pandas as pd

# Load the datasets
train_path = 'train-bank-note-svm.csv'
test_path = 'test-bank-note-svm.csv'

# Read datasets
train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)
test_data.iloc[:, -1] = test_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)

# Extract features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# SVM with Stochastic Sub-Gradient Descent
class SVM:
    def __init__(self, C, learning_rate_schedule, gamma_0=0.1, a=0.1, max_epochs=100):
        self.C = C
        self.learning_rate_schedule = learning_rate_schedule
        self.gamma_0 = gamma_0
        self.a = a
        self.max_epochs = max_epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.max_epochs):
            # Shuffle the training data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for t in range(1, n_samples + 1):
                eta_t = self._get_learning_rate(t)
                i = t - 1
                condition = y[i] * (np.dot(self.w, X[i]) + self.b)
                if condition <= 1:
                    self.w = (1 - eta_t) * self.w + eta_t * self.C * y[i] * X[i]
                    self.b += eta_t * self.C * y[i]
                else:
                    self.w *= (1 - eta_t)

    def _get_learning_rate(self, t):
        if self.learning_rate_schedule == "schedule_1":
            return self.gamma_0 / (1 + self.gamma_0 * self.a * t)
        elif self.learning_rate_schedule == "schedule_2":
            return self.gamma_0 / (1 + t)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def calculate_error(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)


# Load the datasets
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    # Convert labels to {1, -1}
    train_data.iloc[:, -1] = train_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)
    test_data.iloc[:, -1] = test_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # File paths
    train_path = "train-bank-note-svm.csv"
    test_path = "test-bank-note-svm.csv"

    # Load data
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    # Hyperparameters
    C_values = [100 / 873, 500 / 873, 700 / 873, 873 / 873]
    gamma_0_values = [0.01, 0.1, 1]
    a_values = [0.01, 0.1, 1]
    max_epochs = 100

    # Part (a) Results
    results_schedule_1 = []

    for C in C_values:
        for gamma_0 in gamma_0_values:
            for a in a_values:
                svm = SVM(C=C, learning_rate_schedule="schedule_1", gamma_0=gamma_0, a=a, max_epochs=max_epochs)
                svm.fit(X_train, y_train)
                train_error = svm.calculate_error(X_train, y_train)
                test_error = svm.calculate_error(X_test, y_test)
                results_schedule_1.append({
                    "C": C,
                    "gamma_0": gamma_0,
                    "a": a,
                    "Train Error": train_error,
                    "Test Error": test_error
                })

    # Part (b) Results
    results_schedule_2 = []
    for C in C_values:
        for gamma_0 in gamma_0_values:
            svm = SVM(C=C, learning_rate_schedule="schedule_2", gamma_0=gamma_0, max_epochs=max_epochs)
            svm.fit(X_train, y_train)
            train_error = svm.calculate_error(X_train, y_train)
            test_error = svm.calculate_error(X_test, y_test)
            results_schedule_2.append({
                "C": C,
                "gamma_0": gamma_0,
                "Train Error": train_error,
                "Test Error": test_error
            })

    # Part (c) Results: Comparison
    comparison_results = []
    for C in C_values:
        for gamma_0 in gamma_0_values:
            # Train using Schedule 1
            svm_schedule_1 = SVM(C=C, learning_rate_schedule="schedule_1", gamma_0=gamma_0, a=0.1, max_epochs=max_epochs)
            svm_schedule_1.fit(X_train, y_train)
            train_error_1 = svm_schedule_1.calculate_error(X_train, y_train)
            test_error_1 = svm_schedule_1.calculate_error(X_test, y_test)
            w_1, b_1 = svm_schedule_1.w, svm_schedule_1.b

            # Train using Schedule 2
            svm_schedule_2 = SVM(C=C, learning_rate_schedule="schedule_2", gamma_0=gamma_0, max_epochs=max_epochs)
            svm_schedule_2.fit(X_train, y_train)
            train_error_2 = svm_schedule_2.calculate_error(X_train, y_train)
            test_error_2 = svm_schedule_2.calculate_error(X_test, y_test)
            w_2, b_2 = svm_schedule_2.w, svm_schedule_2.b

            # Compute differences
            param_diff = np.linalg.norm(w_1 - w_2)  # L2 norm of weight differences
            bias_diff = abs(b_1 - b_2)
            train_error_diff = train_error_1 - train_error_2
            test_error_diff = test_error_1 - test_error_2

            comparison_results.append({
                "C": C,
                "gamma_0": gamma_0,
                "Param Diff (L2 Norm)": param_diff,
                "Bias Diff": bias_diff,
                "Train Error Diff": train_error_diff,
                "Test Error Diff": test_error_diff
            })

    # Display Results
    print("Results for Schedule 1 (Part a):")
    print(pd.DataFrame(results_schedule_1))

    print("\nResults for Schedule 2 (Part b):")
    print(pd.DataFrame(results_schedule_2))

    print("\nComparison of Schedules (Part c):")
    print(pd.DataFrame(comparison_results))