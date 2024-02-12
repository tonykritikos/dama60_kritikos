import numpy as np

import numpy as np

print("     ")
print("Quiz Q2")
print("     ")

# Define the transition matrix of the graph
transition_matrix = np.array([
    [0, 1/3, 1/3, 1/3],  # A transitions to B, C, D
    [1/2, 0, 0, 1/2],    # B transitions to A, D
    [1, 0, 0, 0],        # C transitions to A
    [0, 1/2, 1/2, 0]     # D transitions to B, C
])

# Define the starting probability vector (starting at Node A)
start_vector = np.array([1, 0, 0, 0])

# Perform three iterations to compute the probability of reaching each node
final_vector = np.linalg.matrix_power(transition_matrix, 3) @ start_vector

# The probability of reaching Node C after three steps is the third element of the final vector
probability_of_reaching_C = final_vector[2]

print("Probability of reaching Node C after three steps:", probability_of_reaching_C)


print("     ")
print("Quiz Q3")
print("     ")

import numpy as np

# Define the number of nodes
num_nodes = 4

# Initialize PageRank scores
page_rank = np.zeros(num_nodes)

# Convergence threshold
threshold = 1e-8

# Iterate until convergence
iteration = 0
while True:
	new_page_rank = np.zeros(num_nodes)
	for i in range(1, num_nodes - 1):
		new_page_rank[i] = (page_rank[i - 1] + page_rank[i + 1]) / 2

	# Check for convergence
	if np.max(np.abs(new_page_rank - page_rank)) < threshold:
		break

	page_rank = new_page_rank
	iteration += 1

print("Convergence iteration:", iteration)

print("     ")
print("Quiz Q4")
print("     ")

import numpy as np

# Define the teleportation factor (beta)
beta = 0.9

# Define the graph as a dictionary of outgoing links for each page
graph = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'D'],
    'C': ['A'],
    'D': ['B', 'C']
}

# Total number of pages
num_pages = len(graph)

# Initialize the PageRank scores
page_rank = {page: 1/num_pages for page in graph}

# Calculate the PageRank scores after the first iteration
new_page_rank = {}
for page in graph:
    new_page_rank[page] = (1 - beta) / num_pages  # Teleportation component
    for neighbor in graph:
        if page in graph[neighbor]:
            new_page_rank[page] += beta * page_rank[neighbor] / len(graph[neighbor])

# Print the distribution of the surfer's location after the first iteration
for page, score in new_page_rank.items():
    print(f"Page {page}: {score:.3f}")



print("     ")
print("Quiz Q5")
print("     ")

# Define the directed edges
edges = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'C'),
    ('C', 'D'),
    ('D', 'E'),
    ('E', 'A'),
    ('E', 'D')
]

# Define the vertices
vertices = ['A', 'B', 'C', 'D', 'E']

# Initialize the transition matrix with zeros
transition_matrix = np.zeros((len(vertices), len(vertices)))

# Construct the transition matrix
for start, end in edges:
	start_index = vertices.index(start)
	end_index = vertices.index(end)
	transition_matrix[start_index][end_index] = 1 / len([e for e in edges if e[0] == start])

# Calculate the maximum sum of column elements
max_sum = np.max(np.sum(transition_matrix, axis=0))

print("Row stochastic transition matrix:")
print(transition_matrix)
print("Maximum sum of the elements among the columns:", max_sum)


print("     ")
print("Quiz Q6")
print("     ")

import numpy as np

# Define the point
y = np.array([1, -3, 4])

# Define the inverse covariance matrix
S_inv = np.diag([1/4, 1/9, 1/25])

# Calculate the Mahalanobis distance
mahalanobis_distance = np.sqrt(np.dot(y.T, np.dot(S_inv, y)))

print("Mahalanobis distance:", mahalanobis_distance)

print("     ")
print("Quiz Q7")
print("     ")

from itertools import combinations

# Define baskets
baskets = [
    {1, 2, 3},
    {2, 3, 4},
    {1, 2, 4},
    {1, 3, 4},
    {1, 3, 5},
    {2, 3, 5},
    {2, 4, 5}
]

# Define hash function
def h(i, j):
    return (i * j) % 7

# Define support threshold
support_threshold = 3

# Initialize hash table for buckets
bucket_counts = [0] * 7

# Count pairs that collide in the buckets
for basket in baskets:
    for pair in combinations(basket, 2):
        hashed_pair = h(*pair)
        bucket_counts[hashed_pair] += 1

# Check which pairs are frequent for the second pass
frequent_pairs = []
for i in range(len(bucket_counts)):
    if bucket_counts[i] >= support_threshold:
        for j in range(i + 1, len(bucket_counts)):
            if bucket_counts[j] >= support_threshold:
                frequent_pairs.append((i, j))

print("Frequent pairs for the second pass:", frequent_pairs)
# Find the index of the most frequent bucket
most_frequent_bucket_index = max(range(len(bucket_counts)), key=bucket_counts.__getitem__)

print("Most frequent bucket index:", most_frequent_bucket_index)
# Find the index of the second most frequent bucket
second_most_frequent_bucket_index = max(range(len(bucket_counts)), key=lambda i: bucket_counts[i] if i != most_frequent_bucket_index else -1)

print("Second most frequent bucket index:", second_most_frequent_bucket_index)




print("     ")
print("Quiz Q8")
print("     ")

import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (4, 3)])

# Run HITS algorithm for 2 iterations
hits_scores = nx.hits(G, max_iter=2)

# Print the hubbiness scores of each node
print("Hubbiness scores after 2 iterations:")
print(hits_scores[0])

# Print the authority scores of each node
print("Authority scores after 2 iterations:")
print(hits_scores[1])


print("     ")
print("Quiz Q9")
print("     ")

import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
edges = [('a', 'b'), ('a', 'd'), ('b', 'd'), ('c', 'b'), ('d', 'e'), ('e', 'f'), ('f', 'b'), ('f', 'c')]
G.add_edges_from(edges)

# Calculate PageRank scores
pagerank_scores = nx.pagerank(G)

# Find the node with the smallest PageRank score
smallest_node = min(pagerank_scores, key=pagerank_scores.get)

print("PageRank scores:")
print(pagerank_scores)
print("Node with the smallest PageRank score:", smallest_node)


print("     ")
print("Quiz Q10")
print("     ")

from itertools import combinations

# Transactional database
transactions = [
    ('a', 'b'),
    ('a', 'b', 'e'),
    ('b', 'c'),
    ('b', 'c', 'd'),
    ('a', 'c', 'e')
]

# Extract all pairs of items
pairs = []
for transaction in transactions:
    pairs.extend(combinations(sorted(transaction), 2))

# Count the occurrences of each pair
pair_counts = {}
for pair in pairs:
    pair_counts[pair] = pair_counts.get(pair, 0) + 1

# Calculate the number of unique pairs
num_unique_pairs = len(pair_counts)

print("Unique pairs:", num_unique_pairs)


print("     ")
print("Exercise 2.c")
print("     ")


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
print("Exercise 3.b & 3.c")
print("     ")


def generate_pairs(items):
	pairs = []
	for i in range( len(items)):
		for j in range(i + 1, len(items)):
			pairs.append((items[i], items[j]))
	return pairs


def hash_pair(pair, num_buckets):
	return (int(pair[0]) * int(pair[1])) % num_buckets

def pcy_first_pass(transactions, num_buckets, threshold):
	item_counts = {}
	bucket_pairs = {}
	bucket_counts = [0] * num_buckets
	frequent_buckets = set()
	pair_support = {}

	for transaction in transactions:
		items = transaction.split(',')

		# Increment counts for individual items
		for item in items:
			item_counts[item] = item_counts.get(item, 0) + 1

		# Generate pairs and hash/count them
		pairs = generate_pairs(items)
		for pair in pairs:
			bucket = hash_pair(pair, num_buckets)
			bucket_counts[bucket] += 1
			if bucket not in bucket_pairs:
				bucket_pairs[bucket] = set()  # Use set to avoid duplicate pairs
			bucket_pairs[bucket].add(pair)  # Use set to avoid duplicate pairs

			if bucket_counts[bucket] >= threshold:
				frequent_buckets.add(bucket)

			# Increment pair support
			if pair not in pair_support:
				pair_support[pair] = 1
			else:
				pair_support[pair] += 1

	return item_counts, frequent_buckets, bucket_counts, bucket_pairs, pair_support


transactions = [
	"1,2,5",
	"2,3,6",
	"3,4,5",
	"1,3,5",
	"2,4,7",
	"1,5,6",
	"2,3,4",
	"2,4,5,7",
	"3,5,7",
	"2,4"
]

num_buckets = 7
threshold = 5

item_counts, frequent_buckets, bucket_counts, bucket_pairs, pair_support = pcy_first_pass(transactions, num_buckets, threshold)

print("Item Counts:")
print(item_counts)
print("\nBucket Counts:")
print(bucket_counts)
print("\nBucket Pairs:")
for bucket, pairs in bucket_pairs.items():
	print(f"Bucket {bucket}:")
	print("Pairs:")
	for pair in pairs:
		print(pair)
print("Frequent Buckets:")
print(frequent_buckets)


print("     ")
print("Exercise 3.d")
print("     ")

def pcy_second_pass(transactions, threshold):
	candidate_pairs = {}

	for transaction in transactions:
		items = transaction.split(',')
		frequent_items = [item for item in items if item_counts[item] >= threshold]
		pairs = generate_pairs(frequent_items)
		for pair in pairs:
			bucket = hash_pair(pair, num_buckets)  # Hash the pair to find its bucket
			if bucket in frequent_buckets:  # Check if the bucket is frequent
				candidate_pairs[pair] = candidate_pairs.get(pair, 0) + 1

	return candidate_pairs


candidate_pairs = pcy_second_pass(transactions, threshold)

print("Candidate Pairs:")
for pair, count in candidate_pairs.items():
	print(f"{pair}: Support {count}")



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
print("Exercise 4.d")
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
