import numpy as np

def entropy(data):
    classes = np.bincount(data)
    probs = classes / np.sum(classes)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def information_gain(parent, left, right):
    p_left = len(left) / len(parent)
    p_right = len(right) / len(parent)

    return entropy(parent) - (p_left * entropy(left) + p_right * entropy(right))

def gain_ratio(parent, left, right):
    p_left = len(left) / len(parent)
    p_right = len(right) / len(parent)

    split_information = -(p_left * np.log2(p_left) + p_right * np.log2(p_right))

    if split_information == 0:
        return 0

    return information_gain(parent, left, right) / split_information

def gini(data):
    classes = np.bincount(data)
    probs = classes / np.sum(classes)
    probs = probs[probs > 0]
    return 1 - np.sum(probs * probs)

def gini_gain(parent, left, right):
    p_left = len(left) / len(parent)
    p_right = len(right) / len(parent)

    return gini(parent) - np.sum(p_left * gini(left) + p_right * gini(right))


def gain(parent, left, right, criterion="gini_gain"):
    if len(left) == 0 or len(right) == 0:
        return 0
    if criterion == "gini_gain":
        return gini_gain(parent, left, right)
    elif criterion == "information_gain":
        return information_gain(parent, left, right)
    elif criterion == "gain_ratio":
        return gain_ratio(parent, left, right)
    else: return  -1