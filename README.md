# DS4350
This is a machine learning library developed by Cody Mathews for CS5350/6350 at the University of Utah.

This project includes scripts for training and evaluating decision trees on the **Car Evaluation** and **Bank Marketing** datasets, as well as implementations of ensemble learning algorithms (AdaBoost, Bagging, Random Forest) and linear regression methods (LMS with batch-gradient and stochastic gradient). The scripts are designed to run in a Python environment with dependencies such as `pandas`, `numpy`, and `matplotlib`.

---

## Setup

1. **Organize folders as follows:**
   - **Ensemble Learning**: Place `RandomForest.py`, `BaggedTrees.py`, and `DecisionStumps.py` here.
   - **Linear Regression**: Place `SGD.py` and `BGD.py` here.
   - **Perceptron**: Create a new folder for Perceptron-related implementations.
   - Ensure all relevant dataset files (e.g., `train-car.csv`, `test-car.csv`, `train-bank.csv`, `test-bank.csv`, `train-concrete.csv`, `test-concrete.csv`) are located in the appropriate directories, or update file paths in the scripts as needed.
   
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

---

## Output

- **Decision Trees**: The script outputs a table with **training and testing errors** for each combination of splitting criterion and tree depth, allowing assessment of model performance and optimal parameter selection.
- **Ensemble Learning**: Each ensemble method outputs error rates across varying parameters (number of trees, maximum features) for both training and test sets.
- **Linear Regression (LMS)**: The LMS scripts plot the cost function's progress over iterations and display final weights, bias term, and cost on test data.
