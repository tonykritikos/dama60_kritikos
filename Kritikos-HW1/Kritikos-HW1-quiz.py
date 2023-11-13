import pandas as pd
import math
import numpy as np

# 1
print("#----- Exercise 1 -----#")

# Create a DataFrame with the given data
data = {'Favorite color': ['green', 'yellow', 'green'], 'Favorite meal': ['pitsa', 'pasta', 'pasta'],
        'Favorite team': ['Barcelona', 'Barcelona', 'Real Madrid'], 'Gender': ['Male', 'Male', 'Female']}

df = pd.DataFrame(data)


# Calculate entropy for the parent node
def entropy(labels):
	n = len(labels)
	if n <= 1:
		return 0
	value, counts = np.unique(labels, return_counts=True)
	probs = counts / n
	return -np.sum(probs * np.log2(probs))


parent_entropy = entropy(df['Gender'])

# Calculate information gain for each attribute
attributes = ['Favorite color', 'Favorite meal', 'Favorite team']
information_gains = {}

for attribute in attributes:
	grouped_data = df.groupby(attribute)['Gender'].apply(list)
	subset_entropies = grouped_data.apply(entropy)
	weighted_entropy = sum(grouped_data.apply(len) / len(df) * subset_entropies)
	information_gains[attribute] = parent_entropy - weighted_entropy

# Print information gains
for attribute, gain in information_gains.items():
	print(f"Information Gain for {attribute}: {gain}")

# Choose the attribute with the highest information gain
best_attribute = max(information_gains, key=information_gains.get)
print(f"Best Attribute to split the node: {best_attribute}\n")

# 2
print("#----- Exercise 2 -----#")

# Given values from the contingency table
support_X_and_Y = 32
support_X = 53
support_Y = 37

# Calculate confidence
confidence = support_X_and_Y / support_X
print(f"The confidence of the rule is: {confidence:.4f}")
# Given values from the contingency table
total_transactions = 75

# Calculate lift
expected_support_X_and_Y = (support_X * support_Y) / total_transactions
lift = support_X_and_Y / expected_support_X_and_Y
print(f"The lift value of the rule is: {lift:.4f}")

# Calculate support
support_rule = support_X_and_Y / total_transactions
print(f"The support of the rule is: {support_rule * 100:.2f}%\n")

# 3
print("#----- Exercise 3 -----#")

# Existing centroid
current_centroid = np.array([1, 3, 15, 7])

# Points assigned to the cluster
cluster_points = np.array([[1.2, 4, 13, 5], [1.6, 3.5, 11, 6], [1, 4.1, 12, 5.5]])

# Calculate the new centroid
new_centroid = np.mean(cluster_points, axis=0)

# Print the new centroid
print(f"New Centroid: {new_centroid}\n")

# 5
print("#----- Exercise 5 -----#")

# Given distribution
class_counts = {'Red': 4, 'Green': 2, 'Yellow': 7}

# Total number of samples
total_samples = sum(class_counts.values())

# Calculate the Gini index
gini_index = 1 - sum((count / total_samples) ** 2 for count in class_counts.values())

# Print the result
print(f"Gini Index: {gini_index:.4f}\n")

# 6
print("#----- Exercise 6 -----#")


def calculate_association_rules_count(n_items):
	total_rules = 2 ** n_items - 2
	return total_rules


# Example usage with 3 items
n_items_example = 3
result = calculate_association_rules_count(n_items_example)
print(f"The total number of association rules for {n_items_example} items is: {result}\n")

# 8
print("#----- Exercise 8 -----#")

# Number of classes
num_classes = 5

# Calculate maximum entropy
max_entropy = -sum((1 / num_classes) * math.log2(1 / num_classes) for _ in range(num_classes))

# Print the result
print(f"Maximum Entropy: {max_entropy:.4f}")
