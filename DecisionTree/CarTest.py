import pandas as pd
import numpy as np

# Load data with column headers: buying,maint,doors,persons,lug_boot,safety,label
train_data = pd.read_csv('train-car.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
test_data = pd.read_csv('test-car.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])

# Check the data
# print(train_data.head(), test_data.head())

# Question 2.A
def entropy(data):
    labels = data['label'] # Get all labels
    label_counts = labels.value_counts() # Count how many of each label
    entropy_value = -sum((count / len(data)) * np.log2(count / len(data)) for count in label_counts)
    return entropy_value

def information_gain(data, attribute):
    overall_entropy = entropy(data)
    values = data[attribute].unique() # Get all unique values
    weighted_entropy = 0

    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return overall_entropy - weighted_entropy

def majority_error(data):
    labels = data['label']
    majority_label_count = labels.value_counts().max() # Get the majority label
    me_value = 1 - (majority_label_count / len(data))
    return me_value

def majority_error_split(data, attribute):
    values = data[attribute].unique()
    weighted_majority_error = 0

    for value in values:
        subset = data[data[attribute] == value]
        me_value = majority_error(subset)
        weighted_majority_error += (len(subset) / len(data)) * me_value

    return weighted_majority_error

def gini_index(data):
    labels = data['label']
    label_counts = labels.value_counts() # Tracks counts how many of each label
    gi_value = 1 - sum((count / len(data)) ** 2 for count in label_counts)
    return gi_value

def gini_index_split(data, attribute):
    values = data[attribute].unique()
    weighted_gini = 0

    for value in values:
        subset = data[data[attribute] == value]
        gini_value = gini_index(subset)
        weighted_gini += (len(subset) / len(data)) * gini_value

    return weighted_gini

def best_attribute(data, criteria='information_gain'):
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'] # Non label attributes
    best_attr = None
    best_value = float('-inf') if criteria in ['information_gain'] else float('inf')

    for attribute in attributes:
        if criteria == 'information_gain':
            value = information_gain(data, attribute)
            if value > best_value: # Higher is better
                best_attr = attribute
                best_value = value
        elif criteria == 'majority_error':
            value = majority_error_split(data, attribute)
            if value < best_value: # Lower is better
                best_attr = attribute
                best_value = value
        elif criteria == 'gini_index':
            value = gini_index_split(data, attribute)
            if value < best_value: # Lower is better
                best_attr = attribute
                best_value = value

    return best_attr

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute # Defines the attribute to split on
        self.label = label # Defines if the label is a leaf node
        self.children = {} # Defines the children


def build_tree(data, max_depth=None, depth=0, criteria='information_gain'):
    labels = data['label']

    # If all labels are the same
    if len(labels.unique()) == 1:
        return Node(label=labels.iloc[0])
    # If tree reaches max depth or greater, return majority label
    if max_depth is not None and depth >= max_depth:
        return Node(label=labels.mode()[0])
    # Split on best attribute
    best_attr = best_attribute(data, criteria)
    # If no attribute, return majority label
    if best_attr is None:
        return Node(label=labels.mode()[0])

    root = Node(attribute=best_attr)

    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        if len(subset) == 0:
            # If no more data, return majority label
            root.children[value] = Node(label=labels.mode()[0])
        else:
            root.children[value] = build_tree(subset, max_depth, depth+1, criteria) # Recursively split the data util max depth is reached

    return root

def predict(tree, instance):
    # If is a leaf node, return
    if tree.label is not None:
        return tree.label
    attr_value = instance[tree.attribute]
    if attr_value not in tree.children:
        return None
    return predict(tree.children[attr_value], instance)

# Test the new tree
def evaluate(tree, test_data):
    correct = 0
    for _, row in test_data.iterrows():
        if predict(tree, row) == row['label']:
            correct += 1
    proficiency = correct / len(test_data)
    # Return error
    return 1 - proficiency

# Test predictions
# print(evaluate(build_tree(train_data), test_data))

# Question 2.B
# Depth from 1 to 6
depth_range = range(1, 7)
criteria_list = ['information_gain', 'majority_error', 'gini_index']
results = []

for criteria in criteria_list:
    for depth in depth_range:
        tree = build_tree(train_data, max_depth=depth, criteria=criteria)
        train_error = evaluate(tree, train_data)
        test_error = evaluate(tree, test_data)
        results.append([criteria, depth, train_error, test_error])

# Display results
results_df = pd.DataFrame(results, columns=['criteria', 'depth', 'train_error', 'test_error'])
print(results_df)