import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv('train-bank-1.csv', header=None)
test_data = pd.read_csv('test-bank-1.csv', header=None)

# Attribute names from 'data-desc-bank-note-bank-1.txt'
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
           'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data.columns = columns
test_data.columns = columns

# Numerical attributes
numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Preprocess data: Treat "unknown" as a particular value and binary transformation for numerical attributes
def preprocess_data(data):
    for col in data.columns:
        if col not in numerical_attributes:  # Only process categorical columns
            majority_value = data[col].mode()[0]  # Get the most frequent value
            data[col] = data[col].replace('unknown', majority_value)  # Replace "unknown" with the majority value
    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Decision Tree implementation (without pruning)
class Node:
    def __init__(self, attribute=None, label=None, threshold=None):
        self.attribute = attribute
        self.label = label
        self.threshold = threshold
        self.children = {}

def build_tree(data, depth=0, max_depth=None):
    labels = data['y']

    if len(labels.unique()) == 1:
        return Node(label=labels.iloc[0])

    if max_depth is not None and depth >= max_depth:
        return Node(label=labels.mode()[0])

    if len(data) == 0:
        return Node(label=labels.mode()[0])

    best_attr = best_attribute(data)

    if best_attr is None:
        return Node(label=labels.mode()[0])

    root = Node(attribute=best_attr)

    if best_attr in numerical_attributes:
        threshold = data[best_attr].median()
        root.threshold = threshold

        subset1 = data[data[best_attr] <= threshold]
        subset2 = data[data[best_attr] > threshold]

        if len(subset1) == 0 or len(subset2) == 0:
            return Node(label=labels.mode()[0])

        root.children['<='] = build_tree(subset1, depth + 1, max_depth)
        root.children['>'] = build_tree(subset2, depth + 1, max_depth)

    else:
        for value in data[best_attr].unique():
            subset = data[data[best_attr] == value]
            if len(subset) == 0:
                root.children[value] = Node(label=labels.mode()[0])
            else:
                root.children[value] = build_tree(subset, depth + 1, max_depth)

    return root

def best_attribute(data):
    attributes = [col for col in data.columns if col != 'y']
    return attributes[0] if attributes else None

def predict(tree, instance):
    if tree.label is not None:
        return tree.label
    if tree.attribute is None:
        return None
    if tree.attribute in numerical_attributes:
        if instance[tree.attribute] <= tree.threshold:
            return predict(tree.children['<='], instance)
        else:
            return predict(tree.children['>'], instance)
    else:
        attr_value = instance[tree.attribute]
        return predict(tree.children.get(attr_value, Node(label=None)), instance)

def evaluate(tree, test_data):
    correct = 0
    for _, row in test_data.iterrows():
        if predict(tree, row) == row['y']:
            correct += 1
    accuracy = correct / len(test_data)
    return 1 - accuracy

# Optimized ensemble error using vectorization
def ensemble_error_optimized(trees, data):
    label_mapping = {"yes": 1, "no": 0}

    all_preds = np.zeros((len(data), len(trees)), dtype=int)

    for i, tree in enumerate(trees):
        all_preds[:, i] = data.apply(lambda row: label_mapping[predict(tree, row)], axis=1)

    final_preds = [np.argmax(np.bincount(preds)) for preds in all_preds]

    correct = np.sum(final_preds == data['y'].map(label_mapping).values)
    accuracy = correct / len(data)

    return 1 - accuracy

# Bagging implementation with optimizations
def bagging_optimized(train_data, test_data, n_trees):
    trees = []
    train_errors = []
    test_errors = []

    for i in range(n_trees):
        bootstrap_sample = resample(train_data)
        tree = build_tree(bootstrap_sample)
        trees.append(tree)

        train_error = ensemble_error_optimized(trees, train_data)
        test_error = ensemble_error_optimized(trees, test_data)

        train_errors.append(train_error)
        test_errors.append(test_error)

        if (i + 1) % 10 == 0:
            print(f"Tree {i + 1}/{n_trees}: Training Error = {train_error}, Test Error = {test_error}")

    return train_errors, test_errors

# Run Bagging
n_trees = 500  # Adjust the number of trees for faster testing if necessary
train_errors_opt, test_errors_opt = bagging_optimized(train_data, test_data, n_trees)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_trees + 1), train_errors_opt, label='Training Error')
plt.plot(range(1, n_trees + 1), test_errors_opt, label='Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Optimized Bagging: Training and Test Errors vs. Number of Trees')
plt.legend()
plt.show()
