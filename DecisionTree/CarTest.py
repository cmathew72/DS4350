import pandas as pd
import numpy as np

# Load data with column headers: buying,maint,doors,persons,lug_boot,safety,label
train_data = pd.read_csv('train.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
test_data = pd.read_csv('test.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])

# Check the data
# print(train_data.head(), test_data.head())

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

def gini_index(data):
    labels = data['label']
    label_counts = labels.value_counts() # Count how many of each label
    gi_value = 1 - sum((count / len(data)) ** 2 for count in label_counts)
    return gi_value

def best_attribute(data, criteria='information_gain'):
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
    best_attribute = None
    best_value = float('-inf')

    for attribute in attributes:
        if criteria == 'information_gain':
            value = information_gain(data, attribute)
        elif criteria == 'majority_error':
            value = majority_error(data)
        elif criteria == 'gini_index':
            value = gini_index(data)

        if value > best_value:
            best_attribute = attribute
            best_value = value

    return best_attribute

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute # Defines the attribute to split on
        self.label = label # Defines if the label is a leaf node
        self.children = {} # Defines the children


def build_tree(data, max_depth=None, depth=0, criteria='information_gain'):
    labels = data['label']

    if len(labels.unique()) == 1: # If all labels are the same
        return Node(label=labels.iloc[0])
    if max_depth is not None and depth >= max_depth: # If tree reaches max depth or greater
        return Node(label=labels.iloc[0]) # Return majority label
    best_attr = best_attribute(data, criteria)
    if best_attr is None:
        return Node(label=labels.iloc[0]) # Majority label

    root = Node(attribute=best_attr)

    for value in data[best_attr].unique():
        sub_data = data[data[best_attr] == value]
        child = build_tree(sub_data, max_depth, depth+1, criteria) # Recursively split the data util max depth is reached
        root.children[value] = child

    return root

def predict(tree, instance):
    if tree.label is not None: # If is a leaf node, return
        return tree.label
    attr_value = instance[tree.attribute]
    return predict(tree.children[attr_value], instance)

# Test the new tree
def evaluate(tree, test_data):
    correct = 0
    for _, row in test_data.iterrows():
        if predict(tree, row) == row['label']:
            correct += 1
    proficiency = correct / len(test_data)
    return proficiency

# Test predictions
# print(evaluate(build_tree(train_data), test_data))