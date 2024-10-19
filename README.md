# DS4350
This is a machine learning library develpoed by Cody Mathews for CS5350/6350 in University of Utah.

This project includes scripts for training and evaluating decision trees on the Car Evaluation, Bank Marketing, and Concrete Slump Test datasets using various splitting criteria. The scripts are designed to run in a Python environment with dependencies such as pandas and numpy.

Setup
Ensure all files (train-car.csv, test-car.csv, train-bank.csv, test-bank.csv, train-concrete.csv, test-concrete.csv, CarTest.py, BankTest.py, BaggedTrees.py, DecisionStumps,py, RandomForest.py, BGD.py, SGD.py) are located in the same directory or update the file paths as needed in the scripts.
Install required libraries:
pip install pandas numpy
pip install scikit-learn

How to Run
For Car Evaluation Dataset: Run CarTest.py
For Bank Marketing Dataset: Run BankTest.py, BaggedTrees.py, DecisionStumps,py, RandomForest.py
For Concrete Sllump Test Dataset: Run BGD.py, SGD.py

Setting Parameters
The scripts allow for customization of the following parameters:

Splitting Criteria: Choose from information_gain, majority_error, or gini_index.
Max Depth: Specify the maximum depth of the tree to control its complexity.
Parameters are set directly in the code as follows:

Adjust the criteria_list for the desired splitting criterion.
Set the range in depth_range for maximum tree depth.

Output
The script outputs a table with training and testing errors for each combination of splitting criterion and tree depth. This can be used to assess the model’s performance and select optimal parameters.
