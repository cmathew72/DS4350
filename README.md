# DS4350
This is a machine learning library developed by Cody Mathews for CS5350/6350 at the University of Utah.

This project includes scripts for training and evaluating decision trees on the **Car Evaluation** and
**Bank Marketing**datasets, as well as implementations of ensemble learning algorithms (AdaBoost, Bagging,
Random Forest) and linear regression methods (LMS with batch-gradient and stochastic gradient). The scripts are
designed to run in a Python environment with dependencies such as `pandas`, `numpy`, and `matplotlib`.

---

## Setup

1. **Organize folders as follows:**
   - **Ensemble Learning**: Place `RandomForest.py`, `BaggedTrees.py`, and `DecisionStumps.py` here.
   - **Linear Regression**: Place `SGD.py` and `BGD.py` here.
   - **Perceptron**: Create a new folder for Perceptron-related implementations.
   - **SVM**: Place `SVM.py` here for SVM implementations (primal and dual).
   - Ensure all relevant dataset files (e.g., `train-car.csv`, `test-car.csv`, `train-bank.csv`, `test-bank.csv`,
   `train-concrete.csv`, `test-concrete.csv`) are located in the appropriate directories, or update file paths in the
   scripts as needed.
   
2. **Install required libraries**:
   ```bash
   pip install pandas numpy matplotlib
## How to Run

### Decision Trees for Car Evaluation and Bank Marketing Datasets
- **For Car Evaluation Dataset**: Run `CarTest.py`
- **For Bank Marketing Dataset**: Run `BankTest.py`

### Ensemble Learning
- **Random Forest**: Run `RandomForest.py`
- **Bagging**: Run `BaggedTrees.py`
- **AdaBoost with Decision Stumps**: Run `DecisionStumps.py`

### Linear Regression (Least-Mean-Square (LMS))
- **Stochastic Gradient Descent**: Run `SGD.py`
- **Batch Gradient Descent**: Run `BGD.py`

### Perceptron Algorithm
- **Standard Perceptron**: Run `Perceptron.py` to train and test using the standard Perceptron algorithm.
- **Voted Perceptron**: Run `Perceptron.py` to train and test using the voted Perceptron algorithm.
- **Average Perceptron**: Run `Perceptron.py` to train and test using the average Perceptron algorithm.

### Support Vector Machines (SVM)
- **Primal Domain (Stochastic Sub-Gradient Descent)**:
  - Run `SVM.py` and specify the learning rate schedule (`schedule_1` or `schedule_w`).
  - Parameters such as \(C\), \(\gamma_0), and \(a\) can be directly in the script.
- **Dual Domain (Quadratic Programming)**:
  - Run `SVM.py` with the dual mode enabled.
  - Uses `scipy.optimize.minimize` to solve the dual problem.
  - Outputs weights (\(w\)), bias (\(b\)), and training/testing errors.

---

## Setting Parameters

The scripts allow for customization of the following parameters:

### Decision Trees
- **Splitting Criteria**: Choose from `information_gain`, `majority_error`, or `gini_index`.
- **Max Depth**: Specify the maximum depth of the tree to control its complexity.
  - Adjust the `criteria_list` for the desired splitting criterion.
  - Set the range in `depth_range` for maximum tree depth.

### Ensemble Learning
- **Random Forest**:
  - Set `n_trees` and `max_features` directly in `RandomForest.py` to control the number of trees and feature subset size.
- **Bagging**:
  - Set `n_trees` in `BaggedTrees.py` to specify the number of trees in the ensemble.
- **AdaBoost**:
  - Adjust `T` (number of boosting rounds) in `DecisionStumps.py`.

### Linear Regression (LMS)
- **Stochastic Gradient Descent**:
  - Modify `learning_rate`, `max_iters`, and other parameters in `SGD.py` for learning rate and iteration count.
- **Batch Gradient Descent**:
  - Adjust `learning_rate`, `tolerance`, and `max_iters` in `BGD.py` for convergence and learning rate tuning.

### Perceptron Algorithms
- **Standard Perceptron**:
  - Set the number of epochs in `Perceptron.py` to specify the number of training iterations.

- **Voted Perceptron**:
  - The number of `epochs` can be adjusted in `Perceptron.py`.

- **Average Perceptron**:
  - Modify the number of `epochs` in `Perceptron.py`.

### SVM
- **Primal Domain**:
  - Specify learning rate schedule: `schedule_1`: \(\gamma_t = \frac{\gamma_0}{1 + \gamma_0 a t}\) or `schedule_2`: \(\gamma_t = \frac{\gamma_0}{1 + t}\)
  - Set \(C\), \(\gamma_0\), and \(a\) values directly in the script.
  - Maximum epochs can also be adjusted.
- **Dual Domain**:
  - Set \(C\) values directly in the script.
  - Automatically computes weights (\(w\)) and bias (\(b\)) from Lagrange multipliers.

---

## Output

- **Decision Trees**: The script outputs a table with **training and testing errors** for each combination of splitting criterion and tree depth, allowing assessment of model performance and optimal parameter selection.
- **Ensemble Learning**: Each ensemble method outputs error rates across varying parameters (number of trees, maximum features) for both training and test sets.
- **Linear Regression (LMS)**: The LMS scripts plot the cost function's progress over iterations and display final weights, bias term, and cost on test data.
- **Perceptron Algorithms**:
  - **Standard Perceptron**:
    - Outputs the learned weight vector and the average prediction error on the test set.
  - **Voted Perceptron**:
    - Displays the distinct weight vectors with their counts and the average prediction error.
  - **Average Perceptron**:
    - Outputs the average weight vector and the average prediction error.
- **SVM**:
  - **Primal Domain**:
    - Output weights (\(w\)) and bias (\(b\)) for each learning rate schedule and \(C\).
    - Displays training and test errors for comparison.
  - **Dual Domain**:
    - Outputs Lagrange multipliers (\(\alpha\)), weights (\(w\)), bias (\(b\)), and errors.
    - Compares primal and dual solutions to ensure correctness.