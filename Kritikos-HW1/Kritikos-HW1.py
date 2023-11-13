from collections import Counter

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances


# Topic 2.a

# Create a DataFrame from the given table
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Gender': ['male', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'male'],
    'Region': ['city', 'city', 'countryside', 'countryside', 'city', 'city', 'city', 'city', 'countryside', 'city', 'city', 'city', 'countryside', 'countryside', 'city', 'city', 'city', 'countryside', 'city', 'city'],
    'Occupation': ['student', 'teacher', 'banker', 'teacher', 'student', 'banker', 'student', 'student', 'teacher', 'student', 'student', 'student', 'banker', 'banker', 'student', 'officer', 'student', 'officer', 'teacher', 'banker'],
    'Income': ['≤ 9000', '> 21000', '> 21000', '> 21000', '≤ 9000', '9000…21000', '≤ 9000', '≤ 9000', '9000…21000', '≤ 9000', '9000…21000', '9000…21000', '9000…21000', '> 21000', '≤ 9000', '> 21000', '≤ 9000', '> 21000', '9000…21000', '> 21000'],
    'Has Laptop': ['no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no']
}

df = pd.DataFrame(data)

# Calculate entropy
def calculate_entropy(labels):
    total_samples = len(labels)
    label_counts = Counter(labels)
    entropy = 0

    for count in label_counts.values():
        probability = count / total_samples
        entropy -= probability * math.log2(probability)

    return entropy

# Calculate information gain, split information, and gain ratio
def calculate_metrics(attribute, target):
    # Calculate total entropy
    total_entropy = calculate_entropy(df[target])

    # Calculate information gain
    group_entropy = df.groupby(attribute)[target].apply(calculate_entropy)
    information_gain = total_entropy - sum(group_entropy * (df.groupby(attribute).size() / len(df)))

    # Calculate split information
    split_info = -sum((df.groupby(attribute).size() / len(df)) * np.log2(df.groupby(attribute).size() / len(df)))

    # Calculate gain ratio
    gain_ratio = information_gain / split_info if split_info != 0 else 0

    return information_gain, split_info, gain_ratio

# Calculate metrics for each attribute
attributes = ['Gender', 'Region', 'Occupation', 'Income']
target_attribute = 'Has Laptop'

for attribute in attributes:
    info_gain, split_info, gain_ratio = calculate_metrics(attribute, target_attribute)
    print(f"\nAttribute: {attribute}")
    print(f"Information Gain: {info_gain}")
    print(f"Split Information: {split_info}")
    print(f"Gain Ratio: {gain_ratio}")



# Topic 4


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

# Define the data points
data = np.array([
    [3.3, 3.6], [0.7, 4.7], [1.6, 1.5], [4.5, 0.9], [0.8, 3.3],
    [1.0, 4.1], [3.7, 0.6], [1.3, 3.5], [3.1, 0.8], [4.2, 1.1]
])

# Calculate the distance matrix
distance_matrix = euclidean_distances(data, data)

# Apply DBSCAN
eps = 1.4
min_samples = 4
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data)

# Determine core, border, and noise points
core_samples = dbscan.core_sample_indices_
border_points = [i for i in range(len(data)) if i not in core_samples and any(distance_matrix[i, core_samples] <= eps)]
noise_points = [i for i in range(len(data)) if i not in core_samples and i not in border_points]


# Print the labels and type of points for each data point
point_types = []
for i, label in enumerate(labels):
    if i in noise_points:
        point_types.append("N")  # Noise point
    elif i in core_samples:
        point_types.append("C")  # Core point
    elif i in border_points:
        point_types.append("B")  # Border point

    print(f"P{i+1}: Cluster {label}, Type: {point_types[-1]}")

# Fill in the matrices
type_matrix = np.array([point_types])
cluster_matrix = np.array([labels])

# Print the filled matrices
print("\nType Matrix:")
print(type_matrix)

print("\nCluster Matrix:")
print(cluster_matrix)