import pandas as pd
import numpy as np

# Load data
train_data = pd.read_csv('train-bank.csv', header=None)
test_data = pd.read_csv('test-bank.csv', header=None)

# Attribute names from 'data-desc.txt'
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
           'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data.columns = columns
test_data.columns = columns

# Numerical attributes
numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']


# Step 1: Define updated functions
def entropy(data):
    labels = data['y']
    label_counts = labels.value_counts()
    entropy_value = -sum((count / len(data)) * np.log2(count / len(data)) for count in label_counts)
    return entropy_value


def information_gain(data, attribute):
    overall_entropy = entropy(data)
    if attribute in numerical_attributes:
        # Split based on median for numerical attributes
        median_value = data[attribute].median()
        subset1 = data[data[attribute] <= median_value]
        subset2 = data[data[attribute] > median_value]
        weighted_entropy = (len(subset1) / len(data)) * entropy(subset1) + (len(subset2) / len(data)) * entropy(subset2)
    else:
        values = data[attribute].unique()
        weighted_entropy = sum(
            (len(data[data[attribute] == value]) / len(data)) * entropy(data[data[attribute] == value]) for value in
            values)

    return overall_entropy - weighted_entropy


def majority_error(data):
    if len(data) == 0:
        return 0
    labels = data['y']
    majority_label_count = labels.value_counts().max()
    me_value = 1 - (majority_label_count / len(data))
    return me_value


def majority_error_split(data, attribute):
    if attribute in numerical_attributes:
        median_value = data[attribute].median()
        subset1 = data[data[attribute] <= median_value]
        subset2 = data[data[attribute] > median_value]
        # Check for empty subsets
        if len(subset1) == 0 or len(subset2) == 0:
            return 0

        weighted_me = (len(subset1) / len(data)) * majority_error(subset1) + (
                    len(subset2) / len(data)) * majority_error(subset2)
    else:
        values = data[attribute].unique()
        weighted_me = 0
        for value in values:
            subset = data[data[attribute] == value]
            if len(subset) == 0:
                continue  # Skip empty subsets
            weighted_me += (len(subset) / len(data)) * majority_error(subset)

    return weighted_me


def gini_index(data):
    labels = data['y']
    label_counts = labels.value_counts()
    gi_value = 1 - sum((count / len(data)) ** 2 for count in label_counts)
    return gi_value


def gini_index_split(data, attribute):
    if attribute in numerical_attributes:
        median_value = data[attribute].median()
        subset1 = data[data[attribute] <= median_value]
        subset2 = data[data[attribute] > median_value]
        weighted_gi = (len(subset1) / len(data)) * gini_index(subset1) + (len(subset2) / len(data)) * gini_index(
            subset2)
    else:
        values = data[attribute].unique()
        weighted_gi = sum(
            (len(data[data[attribute] == value]) / len(data)) * gini_index(data[data[attribute] == value]) for value in
            values)

    return weighted_gi


def best_attribute(data, criteria='information_gain'):
    attributes = [col for col in data.columns if col != 'y']
    best_attr = None
    best_value = float('-inf') if criteria == 'information_gain' else float('inf')

    for attribute in attributes:
        if criteria == 'information_gain':
            value = information_gain(data, attribute)
            if value > best_value:
                best_attr = attribute
                best_value = value
        elif criteria == 'majority_error':
            value = majority_error_split(data, attribute)
            if value < best_value:
                best_attr = attribute
                best_value = value
        elif criteria == 'gini_index':
            value = gini_index_split(data, attribute)
            if value < best_value:
                best_attr = attribute
                best_value = value

    return best_attr

class Node:
    def __init__(self, attribute=None, label=None, threshold=None):
        self.attribute = attribute
        self.label = label
        self.threshold = threshold  # For numerical splits
        self.children = {}

def build_tree(data, max_depth=None, depth=0, criteria='information_gain'):
    labels = data['y']
    if len(labels.unique()) == 1:
        return Node(label=labels.iloc[0])
    if max_depth is not None and (depth >= max_depth or len(data) == 0):
        if not labels.mode().empty:  # Check if mode() returns a non-empty series
            return Node(label=labels.mode()[0])
        else:
            return Node(label="none")

    best_attr = best_attribute(data, criteria)
    if best_attr is None:
        if not labels.mode().empty:
            return Node(label=labels.mode()[0])
        else:
            return Node(label="none")

    if best_attr in numerical_attributes:
        # Numerical split based on median
        threshold = data[best_attr].median()
        root = Node(attribute=best_attr, threshold=threshold)
        subset1 = data[data[best_attr] <= threshold]
        subset2 = data[data[best_attr] > threshold]
        root.children['<='] = build_tree(subset1, max_depth, depth + 1, criteria)
        root.children['>'] = build_tree(subset2, max_depth, depth + 1, criteria)
    else:
        root = Node(attribute=best_attr)
        for value in data[best_attr].unique():
            subset = data[data[best_attr] == value]
            root.children[value] = build_tree(subset, max_depth, depth + 1, criteria)

    return root


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
    proficiency = correct / len(test_data)
    return 1 - proficiency


# Run the evaluation for depths from 1 to 16
depth_range = range(1, 17)
criteria_list = ['information_gain', 'majority_error', 'gini_index']
results = []

for criteria in criteria_list:
    for depth in depth_range:
        tree = build_tree(train_data, max_depth=depth, criteria=criteria)
        train_error = evaluate(tree, train_data)
        test_error = evaluate(tree, test_data)
        results.append([criteria, depth, train_error, test_error])

# Display the results
results_df = pd.DataFrame(results, columns=['criteria', 'depth', 'train_error', 'test_error'])
print(results_df)
