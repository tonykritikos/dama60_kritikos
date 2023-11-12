import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances


# Topic 2.a

# Function to calculate Entropy
def calculate_entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


# Function to calculate Information Gain
def calculate_information_gain(entropy_before, weights, entropies_after):
    return entropy_before - sum(w * e for w, e in zip(weights, entropies_after))


# Function to calculate Split Information
def calculate_split_information(proportions):
    return -sum(p * math.log2(p) for p in proportions if p > 0)


# Function to calculate Gain Ratio
def calculate_gain_ratio(information_gain, split_information):
    if split_information == 0:
        return 0  # Avoid division by zero
    return information_gain / split_information


# Given values from the exercise
split_info_gender = 0.9710
information_gain_region = 0.0031
split_info_occupation = 1.8150
information_gain_income = 0.0458

# Data counts for each attribute
gender_counts = {'male': 12, 'female': 8}
region_counts = {'city': 14, 'countryside': 6}
occupation_counts = {'student': 9, 'teacher': 5, 'banker': 4, 'officer': 2}
income_counts = {'≤ 9000': 7, '9000…21000': 6, '> 21000': 7}

# Calculate proportions based on counts
total_gender = sum(gender_counts.values())
weights_gender = [count / total_gender for count in gender_counts.values()]
print(weights_gender)

total_region = sum(region_counts.values())
weights_region = [count / total_region for count in region_counts.values()]
print(weights_region)

total_occupation = sum(occupation_counts.values())
weights_occupation = [count / total_occupation for count in occupation_counts.values()]
print(weights_occupation)

total_income = sum(income_counts.values())
weights_income = [count / total_income for count in income_counts.values()]
print(weights_income)

# Assuming entropy_before is 1, which is common in decision trees
entropy_before = 1

# Calculate missing values
# Gender
information_gain_gender = calculate_information_gain(entropy_before, weights_gender, [split_info_gender, split_info_gender])

# Region
split_info_region = calculate_split_information(weights_region)

# Occupation
information_gain_occupation = calculate_information_gain(entropy_before, weights_occupation, [split_info_occupation, split_info_occupation])

# Income
gain_ratio_income = calculate_gain_ratio(information_gain_income,
                                         0.0458)  # Corrected: Use Information Gain, not Split Information

# Displaying the results
print("Attribute\tInformation Gain\tSplit Information\tGain Ratio")
print(
    f"Gender\t\t{round(information_gain_gender, 4)}\t\t{round(split_info_gender, 4)}\t\t{round(information_gain_gender / split_info_gender, 4)}")
print(
    f"Region\t\t{round(information_gain_region, 4)}\t\t{round(split_info_region, 4)}\t\t{round(information_gain_region / split_info_region, 4)}")
print(
    f"Occupation\t{round(information_gain_occupation, 4)}\t\t{round(split_info_occupation, 4)}\t\t{round(information_gain_occupation / split_info_occupation, 4)}")
print(f"Income\t\t{round(information_gain_income, 4)}\t\t{0.0458}\t\t{round(gain_ratio_income, 4)}")



# Topic 4



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

# Print the labels and type of points for each data point
point_types = []
for i, label in enumerate(labels):
    if label == -1:
        point_types.append("N")  # Noise point
    elif min_samples <= np.sum(labels == label):
        point_types.append("C")  # Core point
    else:
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


# Scatter plot the data points with color based on cluster labels
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, edgecolors='k')
plt.title('DBSCAN Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
