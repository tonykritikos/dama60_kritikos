import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


print("Topic 2")


# Define the user-item rating matrix
ratings = np.array([
    [2, np.nan, 4, 2, 4],
    [4, 5, 3, 4, 2],
    [5, 3, 4, 2, 1],
    [5, 4, 2, 1, 3],
    [5, 2, 3, 4, 2]
])

# (a) Calculate the average rating for Karminio
karminio_ratings = [rating for rating in ratings[:,1] if not np.isnan(rating)]
average_karminio_rating = np.mean(karminio_ratings)
print("Average rating for Karminio:", round(average_karminio_rating, 1))

# (b) Compute item-item similarity
def compute_similarity(item1, item2, method='pearson'):
    item1_ratings = []
    item2_ratings = []
    for i in range(len(ratings)):
        if not np.isnan(ratings[i, item1]) and not np.isnan(ratings[i, item2]):
            item1_ratings.append(ratings[i, item1])
            item2_ratings.append(ratings[i, item2])
    if method == 'pearson':
        return pearsonr(item1_ratings, item2_ratings)[0]
    elif method == 'cosine':
        return 1 - cosine(item1_ratings, item2_ratings)

similarity_matrix = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        similarity_matrix[i, j] = compute_similarity(i, j)

# Print results for (b)
print("\nPearson correlation:")
print("The Last Slice, Falafellas:", round(similarity_matrix[0, 2], 4))
print("The Last Slice, Tarantino Sandwiches & Fries:", round(similarity_matrix[0, 3], 4))
print("Falafellas, Tarantino Sandwiches & Fries:", round(similarity_matrix[2, 3], 4))

print("\nCosine similarity:")
print("The Last Slice, Falafellas:", round(compute_similarity(0, 2, method='cosine'), 4))
print("The Last Slice, Tarantino Sandwiches & Fries:", round(compute_similarity(0, 3, method='cosine'), 4))
print("Falafellas, Tarantino Sandwiches & Fries:", round(compute_similarity(2, 3, method='cosine'), 4))

# (c) Predict rating for Maria for Karminio
def predict_rating(user_ratings, similarity_scores, k, item_index):
    relevant_indices = np.argsort(similarity_scores)[::-1][:k]
    numerator = np.sum(similarity_scores[relevant_indices] * user_ratings[relevant_indices])
    denominator = np.sum(np.abs(similarity_scores[relevant_indices]))
    return numerator / denominator

k = 2
user_ratings = ratings[:,1]
karminio_similarities = similarity_matrix[:,1]
predicted_rating = predict_rating(user_ratings, karminio_similarities, k, 1)
print("\nPredicted rating of Maria for Karminio:", round(predicted_rating, 1))

print(" ")
print("Topic 3")
print(" ")

import numpy as np

# Actual and predicted ratings
actual_ratings = np.array([[5, np.nan, 4, 4, 5 ],
                           [4.5, 4, 3, 4.5, np.nan],
                           [np.nan, 5, 3, 4, 3]])

predicted_ratings = np.array([[4, 4, 3, 3, 1],
                              [3, 3.5, 2, 2, 3],
                              [2, 5, 5, 2, 3]])

# Mask NaN values in actual ratings
mask = ~np.isnan(actual_ratings)

# Calculate MAE
mae = np.mean(np.abs(actual_ratings[mask] - predicted_ratings[mask]))
print("Mean Absolute Error (MAE):", mae)

# Calculate RMSE
rmse = np.sqrt(np.mean((actual_ratings[mask] - predicted_ratings[mask])**2))
print("Root Mean Squared Error (RMSE):", rmse)

a = b = c = None
# Table for Precision@N, Recall@N, and DCGpos
top_n = np.array([[3, 0, 0, 3],
                  [5, None, None, None],
                  [5, 2/3, None, a],
                  [4, None, 2/3, None],
                  [2, b, None, None],
                  [3, None, None, None],
                  [5, None, None, None],
                  [1, None, c, None]])

# Calculate relevant items
relevant_items = np.where(top_n[:, 1] == 5)[0]

# Calculate Recall@N for position 3
N_3_recall = len(np.where(top_n[1:4, 1] == 5)[0]) / len(relevant_items) if len(relevant_items) > 0 else None

# Calculate Precision@N for position 2
N_2_precision = len(np.where(top_n[1:3, 1] == 5)[0]) / 2 if top_n[2, 1] is None else top_n[2, 1]

# Calculate Precision@N for position 8
N_8_precision = len(np.where(top_n[1:8, 1] == 5)[0]) / 8 if top_n[7, 1] is None else top_n[7, 1]

# Calculate DCGpos for position 3
if top_n[2, 3] is not None:
    N_3_dcg = top_n[2, 3]
else:
    N_3_dcg = np.sum(top_n[1:4, 1].astype(float) / np.log2(np.arange(2, 5)))

# Print the calculated values
print("Calculated values:")
print("a (Recall@N for position 3):", round(N_3_recall, 4) if N_3_recall is not None else None)
print("b (Precision@N for position 2):", round(N_2_precision, 4))
print("c (Precision@N for position 8):", round(N_8_precision, 4))
print("DCGpos for position 3:", round(N_3_dcg, 4))



print(" ")
print("Topic 4")
print(" ")

import numpy as np

# Define the transition matrix
transition_matrix = np.array([
    [0, 0, 0, 0.5, 0, 0.5],  # User1
    [0, 0, 0, 0, 1, 0],       # User2
    [0, 0, 0, 0, 0, 1],       # User3
    [0.5, 0, 0, 0, 0, 0.5],   # Item1
    [0, 1, 0, 0, 0, 0],   # Item2 (adjusted transition probabilities)
    [0.5, 0, 0.5, 0, 0, 0]     # Item3
])

# Define the initial probability vector
initial_prob = np.array([0, 0, 0, 0, 1, 0])  # Only Item2 has initial probability to transition to User1 and Item3

# Parameters
beta = 0.9
num_steps = 4

# Perform Random Walk with Restart
probabilities = [initial_prob]
for step in range(num_steps):
    next_prob = beta * np.dot(transition_matrix.T, probabilities[-1]) + (1 - beta) * initial_prob
    probabilities.append(next_prob)

# Print the probabilities at each step
for i, prob in enumerate(probabilities):
    print(f"Step {i}: {np.round(prob, 4)}")


from collections import defaultdict, deque

# Define the graph connections
graph = {
    1: [5, 2, 3],
    2: [1, 3, 4],
    3: [1, 2, 4, 6],
    4: [2, 3, 6],
    5: [1],
    6: [3, 4]
}

# Function to perform BFS from a starting node
def bfs(graph, start):
    visited = {node: False for node in graph}
    shortest_paths = {node: [] for node in graph}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        if not visited[node]:
            visited[node] = True
            shortest_paths[node] = path

            for neighbor in graph[node]:
                if not visited[neighbor]:
                    queue.append((neighbor, path + [neighbor]))

    return shortest_paths

# Function to calculate the number of shortest paths passing through each edge
def edge_betweenness(graph, shortest_paths):
    edge_count = defaultdict(int)

    for node, paths in shortest_paths.items():
        for i in range(len(paths) - 1):
            edge = (min(paths[i], paths[i + 1]), max(paths[i], paths[i + 1]))
            edge_count[edge] += 1

    return edge_count

# Perform BFS from node 5
shortest_paths = bfs(graph, 5)

# Calculate edge betweenness centrality
edge_count = edge_betweenness(graph, shortest_paths)

# Define the contributions based on the provided indicative answers
edge_contributions = {
    (3, 4): 0.5,
    (1, 3): None,  # To be determined
    (2, 4): 0.5,
    (1, 2): None,  # To be determined
    (1, 5): None   # To be determined
}

# Print the results
print("Edge\tContribution")
for edge, count in edge_count.items():
    if edge in edge_contributions:
        print(f"e{edge}\t{edge_contributions[edge]}")
    else:
        print(f"e{edge}\t?")
