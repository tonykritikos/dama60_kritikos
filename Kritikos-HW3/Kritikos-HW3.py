import numpy as np

# Define the transition matrix M
M = np.array([
	[0, 0, 1, 0, 0.5],
	[0.5, 0, 0, 0, 0],
	[0, 1, 0, 0, 0.5],
	[0.5, 0, 0, 0, 0],
	[0, 0, 0, 1, 0]
])

# Initial probabilities vector (all nodes have equal probability)
v = np.ones(M.shape[0]) / M.shape[0]

# Number of iterations (adjust as needed)
num_iterations = 2

print("     ")
print("Exercise 2.c")
print("     ")

# Perform iterations
for _ in range(num_iterations):
	v1 = M @ v

# Round the results to 3 decimal places
v1_rounded = np.round(v1, 3)

# Print the results
print("Values of vector v' after", num_iterations, "iterations of simplified PageRank:")
print(v1_rounded)

print("     ")
print("Exercise 2.d")
print("     ")

# Set damping factor beta
beta = 0.80

# Perform the first iteration
v_prime = beta * M @ v + (1 - beta) * np.ones(M.shape[0]) / M.shape[0]

# Perform the second iteration
v_prime_prime = beta * M @ v_prime + (1 - beta) * np.ones(M.shape[0]) / M.shape[0]

# Round the results to 3 decimal places
v_prime_prime_rounded = np.round(v_prime_prime, 3)

# Print the results
print("Values of vector v' after the 2nd iteration of PageRank with teleportation:")
print(v_prime_prime_rounded)


print("     ")
print("Exercise 3.b")
print("     ")

from collections import defaultdict

# Given transactions
transactions = [
    [1, 2, 5],
    [2, 3, 6],
    [3, 4, 5],
    [1, 3, 5],
    [2, 4, 7],
    [1, 5, 6],
    [2, 3, 4],
    [2, 4, 5, 7],
    [3, 5, 7],
    [2, 4]
]

# Initialize a hash table with seven buckets
buckets = defaultdict(set)

# Calculate support for each bucket
for transaction in transactions:
	for pair in transaction:
		for other_pair in transaction:
			if pair != other_pair:
				# Ensure pairs are considered unordered (e.g., (1, 2) and (2, 1) are the same)
				pair_tuple = min(pair, other_pair) * max(pair, other_pair)
				hash_value = hash(pair_tuple % 7)
				buckets[hash_value].add((min(pair, other_pair), max(pair, other_pair)))

# Filter buckets based on the support threshold
support_threshold = 5
result_buckets = {key: (value, len(value)) for key, value in buckets.items()}

print("Buckets with support equal to or greater than 5:")
for bucket, (sets, support) in result_buckets.items():
	print(f"Bucket {bucket}: Support {support}")
	print("Sets:", sets)
	print()

print("     ")
print("Exercise 4.a")
print("     ")

import numpy as np

# Given points
points = {
    'A': np.array([1, 7]),
    'G': np.array([1, 10]),
    'K': np.array([4, 10]),
    'L': np.array([2, 9]),
    'J': np.array([3, 8]),
    'I': np.array([2, 3]),
    'C': np.array([1, 2]),
    'F': np.array([2, 1]),
    'D': np.array([6, 4]),
    'B': np.array([6, 2]),
    'H': np.array([8, 5]),
    'E': np.array([8, 2]),
}

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to find the next centroid
def find_next_centroid(centroids, points):
    max_min_distance = -1
    next_centroid = None

    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_distance = min(distances)

        if min_distance > max_min_distance:
            max_min_distance = min_distance
            next_centroid = point

    return next_centroid

# Function to perform clustering and print centroids
def initialize_clusters(points, k):
    centroids = [list(points['A'])]  # Start with the first point (A) as the initial centroid

    for _ in range(1, k):
        next_centroid = find_next_centroid(centroids, points.values())
        centroids.append(list(next_centroid))

    print(f"Initial Centroids: {centroids}")

# Initialize clusters with k=3
initialize_clusters(points, 3)

print("     ")
print("Exercise 4.b")
print("     ")

# Cluster 0
cluster_0_points = [points['A'], points['G'], points['J'], points['K'], points['L']]
cluster_0_N = len(cluster_0_points)
cluster_0_SUM = np.sum(cluster_0_points, axis=0)
cluster_0_SUMSQ = np.sum(np.square(cluster_0_points), axis=0)

print(f"Cluster 0: N={cluster_0_N}, SUM={cluster_0_SUM}, SUMSQ={cluster_0_SUMSQ}")

# Cluster 1
cluster_1_points = [points['B'], points['D'], points['E'], points['H']]
cluster_1_N = len(cluster_1_points)
cluster_1_SUM = np.sum(cluster_1_points, axis=0)
cluster_1_SUMSQ = np.sum(np.square(cluster_1_points), axis=0)

print(f"Cluster 1: N={cluster_1_N}, SUM={cluster_1_SUM}, SUMSQ={cluster_1_SUMSQ}")

# Cluster 2
cluster_2_points = [points['C'], points['F'], points['I']]
cluster_2_N = len(cluster_2_points)
cluster_2_SUM = np.sum(cluster_2_points, axis=0)
cluster_2_SUMSQ = np.sum(np.square(cluster_2_points), axis=0)

print(f"Cluster 2: N={cluster_2_N}, SUM={cluster_2_SUM}, SUMSQ={cluster_2_SUMSQ}")


print("     ")
print("Exercise 4.c")
print("     ")


# Variance and Standard Deviation calculations for each cluster
def calculate_var_sd(cluster_points, cluster_SUM, cluster_SUMSQ):
    N = len(cluster_points)
    variance = (cluster_SUMSQ / N) - np.square(cluster_SUM / N)
    std_deviation = np.sqrt(variance)
    return variance, std_deviation

# Cluster 0
variance_0, sd_0 = calculate_var_sd(cluster_0_points, cluster_0_SUM, cluster_0_SUMSQ)

# Cluster 1
variance_1, sd_1 = calculate_var_sd(cluster_1_points, cluster_1_SUM, cluster_1_SUMSQ)

# Cluster 2
variance_2, sd_2 = calculate_var_sd(cluster_2_points, cluster_2_SUM, cluster_2_SUMSQ)

# Print the results with 3 decimal digits
print(f"Cluster 0: Variance=({variance_0[0]:.3f}, {variance_0[1]:.3f}), SD=({sd_0[0]:.3f}, {sd_0[1]:.3f})")
print(f"Cluster 1: Variance=({variance_1[0]:.3f}, {variance_1[1]:.3f}), SD=({sd_1[0]:.3f}, {sd_1[1]:.3f})")
print(f"Cluster 2: Variance=({variance_2[0]:.3f}, {variance_2[1]:.3f}), SD=({sd_2[0]:.3f}, {sd_2[1]:.3f})")


print("     ")
print("Exercise 4.c")
print("     ")

from scipy.spatial.distance import mahalanobis

# Given points
points_M = np.array([6, 8])
points_N = np.array([2, 2])
points_P = np.array([6, 5])

# Calculate Mahalanobis Distance for each point and each cluster
mahalanobis_distances = {
    'M': [mahalanobis(points_M, centroid, np.linalg.inv(np.cov(np.array(cluster_points).T))) for centroid, cluster_points in zip([cluster_0_SUM / cluster_0_N, cluster_1_SUM / cluster_1_N, cluster_2_SUM / cluster_2_N], [cluster_0_points, cluster_1_points, cluster_2_points])],
    'N': [mahalanobis(points_N, centroid, np.linalg.inv(np.cov(np.array(cluster_points).T))) for centroid, cluster_points in zip([cluster_0_SUM / cluster_0_N, cluster_1_SUM / cluster_1_N, cluster_2_SUM / cluster_2_N], [cluster_0_points, cluster_1_points, cluster_2_points])],
    'P': [mahalanobis(points_P, centroid, np.linalg.inv(np.cov(np.array(cluster_points).T))) for centroid, cluster_points in zip([cluster_0_SUM / cluster_0_N, cluster_1_SUM / cluster_1_N, cluster_2_SUM / cluster_2_N], [cluster_0_points, cluster_1_points, cluster_2_points])]
}

# Assign points to clusters or Retained Set
assignments = {}
for point_name, distances in mahalanobis_distances.items():
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]
    cluster_std_deviation = [sd_0, sd_1, sd_2][min_distance_idx]

    if np.all(min_distance < 2 * cluster_std_deviation):
        assignments[point_name] = f'Cluster {min_distance_idx}'
    else:
        assignments[point_name] = 'Retained Set'

# Print assignments
print("Assignments:")
for point_name, assignment in assignments.items():
    print(f"{point_name}: {assignment}")
