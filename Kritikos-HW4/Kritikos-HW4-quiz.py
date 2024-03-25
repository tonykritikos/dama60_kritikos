print(" ")
print("Quiz 1")
print(" ")

from collections import defaultdict


class Advertiser:
	def __init__(self, id, budget, keywords):
		self.id = id
		self.budgets = {keyword: budget for keyword in keywords}

	def __repr__(self):
		return f"Advertiser {self.id}"


def balance_algorithm(advertisers, queries):
	handled_queries = 0
	for query in queries:
		available_advertisers = [adv for adv in advertisers if adv.budgets.get(query, 0) > 0]
		if not available_advertisers:
			continue
		selected_advertiser = min(available_advertisers, key=lambda x: (x.budgets.get(query, 0), x.id))
		selected_advertiser.budgets[query] -= 1
		handled_queries += 1
	return handled_queries


def optimal_algorithm(advertisers, queries):
	handled_queries = 0
	advertiser_budgets = defaultdict(int)
	for adv in advertisers:
		for keyword, budget in adv.budgets.items():
			advertiser_budgets[(adv.id, keyword)] = budget

	for query in queries:
		available_advertisers = [(adv.id, adv.budgets.get(query, 0)) for adv in advertisers if
		                         adv.budgets.get(query, 0) > 0]
		if not available_advertisers:
			continue
		selected_advertiser = max(available_advertisers, key=lambda x: x[1])
		adv_id = selected_advertiser[0]
		advertisers[adv_id - 1].budgets[query] -= 1
		handled_queries += 1

	return handled_queries


# Define advertisers
advertiser1 = Advertiser(1, 6, ['k1'])
advertiser2 = Advertiser(2, 6, ['k1', 'k2'])
advertiser3 = Advertiser(3, 6, ['k1', 'k2', 'k3'])
advertiser4 = Advertiser(4, 6, ['k1', 'k2', 'k3', 'k4'])

advertisers = [advertiser1, advertiser2, advertiser3, advertiser4]

# Define queries
queries = ['k2'] * 6 + ['k1'] * 6 + ['k4'] * 6 + ['k3'] * 6

# Run algorithms
balance_queries = balance_algorithm(advertisers, queries)
optimal_queries = optimal_algorithm(advertisers, queries)

print("BALANCE algorithm handled queries:", balance_queries)
print("OPTIMAL algorithm handled queries:", optimal_queries)

print(" ")
print("Quiz 2")
print(" ")

class AdvertiserTwo:
    def __init__(self, budget, bids):
        self.budget = budget
        self.bids = bids
        self.remaining_budget = budget

def balance_algorithm_two(advertisers, queries):
    allocation_sequence = ''
    for query in queries:
        available_advertisers = [adv for adv in advertisers if query in adv.bids and adv.remaining_budget >= adv.bids[query]]
        if not available_advertisers:
            allocation_sequence += '-'
            continue
        selected_advertiser = min(available_advertisers, key=lambda x: (x.remaining_budget, x.bids[query]))
        allocation_sequence += 'A' if selected_advertiser == advertisers[0] else 'B'
        selected_advertiser.remaining_budget -= selected_advertiser.bids[query]
    return allocation_sequence

# Define advertisers
adv_A_two = AdvertiserTwo(4, {'T1': 1})
adv_B_two = AdvertiserTwo(6, {'T1': 1, 'T2': 1})
advertisers_list_two = [adv_A_two, adv_B_two]

# Define queries
queries_list_two = ['T1', 'T2', 'T1', 'T1', 'T2', 'T1']

# Possible allocation sequences
options_list_two = ['BBBABA', 'BBAABA', 'BBBABB', 'BABABA']

# Check each option
correct_option_found = False
for option in options_list_two:
    adv_A_two.remaining_budget = 4
    adv_B_two.remaining_budget = 6
    allocation_sequence = balance_algorithm_two(advertisers_list_two, queries_list_two)
    if allocation_sequence == option:
        print("Option", option, "is possible.")
        correct_option_found = True
        break

if not correct_option_found:
    print("No correct option found.")




print(" ")
print("Quiz 3")
print(" ")

import numpy as np

# Define the Laplacian matrix
L = np.array([[3, -1, -1, -1],
              [-1, 2, 0, -1],
              [-1, 0, 2, -1],
              [-1, -1, -1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(L)

# Round eigenvalues to integers
rounded_eigenvalues = np.round(eigenvalues).astype(int)

# Round eigenvectors to integers
rounded_eigenvectors = np.round(eigenvectors).astype(int)

# Display eigenvalues and eigenvectors
for i in range(len(rounded_eigenvalues)):
    print("Eigenvalue:", rounded_eigenvalues[i])
    print("Eigenvector:", rounded_eigenvectors[:, i])
    print()



print(" ")
print("Quiz 4")
print(" ")


from fractions import Fraction

# Define the graph as a dictionary of edges and their weights
graph = {
    'A': {'E': 1, 'D': 1, 'B': 1},
    'B': {'D': 1, 'C': 1, 'F': 1},
    'C': {'D': 1, 'B': 1, 'F': 1, 'G': 1},
    'D': {'A': 1, 'B': 1, 'C': 1},
    'E': {'A': 1},
    'F': {'B': 1, 'C': 1, 'H': 1},
    'G': {'C': 1},
    'H': {'F': 1}
}

# Define the sets
set1 = {'A', 'D', 'E'}
set2 = {'B', 'C', 'F', 'G', 'H'}

# Calculate the cut value
cut_value = sum(graph[node1].get(node2, 0) for node1 in set1 for node2 in set2)

# Calculate the volume of each set
volume_set1 = sum(graph[node1][node2] for node1 in set1 for node2 in graph[node1] if node2 in set1)
volume_set2 = sum(graph[node1][node2] for node1 in set2 for node2 in graph[node1] if node2 in set2)

# Calculate the normalized cut value
normalized_cut_value = (cut_value / volume_set1) + (cut_value / volume_set2)

# Convert the result to a fraction
normalized_cut_fraction = Fraction(normalized_cut_value).limit_denominator()

print("Normalized cut value:", normalized_cut_fraction)



print(" ")
print("Quiz 5")
print(" ")

import numpy as np

# Original matrix
M = np.array([[5, 8, 7, 3, 7],
              [9, 3, 0, 8, 10],
              [8, 0, 9, 4, 9]])

# Initial matrices U and V
U = np.full((3, 3), 2)  # 3x3 matrix with all elements initialized to 2
V = np.full((3, 5), 1)  # 3x5 matrix with all elements initialized to 1

# Optimize U22
U[1, 1] = (M[1, 1] - np.dot(U[1, :], V[:, 1])) / V[1, 3]

# Optimize V34
V[2, 3] = (M[2, 3] - np.dot(U[2, :], V[:, 3])) / U[2, 1]

# Print the updated matrices
print("U matrix after optimization:")
print(U)

print("\nV matrix after optimization:")
print(V)

# Value of V14
print("\nValue of element V14:", V[0, 3])



print(" ")
print("Quiz 6")
print(" ")

from collections import defaultdict

def shortest_paths(graph, source):
    queue = [(source, [source])]
    paths = defaultdict(list)
    while queue:
        (node, path) = queue.pop(0)
        for next_node in graph[node] - set(path):
            if next_node not in paths:
                queue.append((next_node, path + [next_node]))
                paths[next_node].append(path + [next_node])
            else:
                if len(paths[next_node][0]) == len(path) + 1:
                    paths[next_node].append(path + [next_node])
    return paths

def betweenness_centrality(graph):
    betweenness = defaultdict(int)  # Initialize betweenness for all possible edges
    for node in graph:
        paths = shortest_paths(graph, node)
        for target in paths:
            for path in paths[target]:
                if ('A', 'B') in [(path[i], path[i+1]) for i in range(len(path)-1)]:
                    for i in range(len(path) - 1):
                        edge = tuple(sorted([path[i], path[i + 1]]))
                        betweenness[edge] += 1
    return betweenness

# Define the graph with the missing edge FC
graph = {
    'A': {'B', 'D', 'E'},
    'B': {'A', 'C', 'F'},
    'C': {'B', 'F'},
    'D': {'A', 'E'},
    'E': {'A', 'D'},
    'F': {'B', 'C'}
}

# Calculate betweenness centrality
centrality = betweenness_centrality(graph)

# Edge whose betweenness centrality we want to find
edge = ('A', 'B')

# Print the betweenness centrality of the edge
print("Betweenness centrality of edge", edge, ":", centrality[edge])


print(" ")
print("Quiz 7")
print(" ")

# Define the edges as tuples
edges = [(1, 6), (1, 5), (2, 8), (2, 7), (3, 6), (3, 5), (4, 8), (4, 7)]

# Remove duplicate edges
unique_edges = set(edges)

# Function to find a perfect matching
def find_perfect_matching(edges):
    matched_vertices = set()
    matching = set()

    for edge in edges:
        u, v = edge
        if u not in matched_vertices and v not in matched_vertices:
            matching.add(edge)
            matched_vertices.add(u)
            matched_vertices.add(v)

    return matching

# Call the function to find the perfect matching
perfect_matching = find_perfect_matching(unique_edges)

# Print the result
if perfect_matching:
    print("Perfect matching found:")
    print(perfect_matching)
else:
    print("No perfect matching found.")



print(" ")
print("Quiz 8")
print(" ")


## Given data
num_movies_mb = 4  # Number of movies featuring Monica Bellucci
ratings_mb = [4, 5, 2, 5]  # Ratings for movies featuring Monica Bellucci
component_mb = 0.3  # Normalized and averaged value for movies featuring Monica Bellucci

# Calculate the sum of ratings for movies featuring Monica Bellucci
sum_ratings_mb = sum(ratings_mb)

# Calculate the contribution of movies featuring Monica Bellucci to the overall average
contribution_mb = component_mb * num_movies_mb

# Total sum of all ratings (including movies featuring Monica Bellucci)
total_sum_ratings = sum_ratings_mb + contribution_mb

# Calculate the average rating x for all movies
num_total_movies = num_movies_mb  # As we don't have information about other movies
average_rating_x = total_sum_ratings / num_total_movies

# Output
print("Total sum of ratings (including movies featuring Monica Bellucci):", total_sum_ratings)
print("Average rating x for all movies:", average_rating_x)


print(" ")
print("Quiz 9")
print(" ")


class AdvertiserThree:
    def __init__(self, budget, bids):
        self.budget = budget
        self.bids = bids

def optimal_offline(advertisers, queries):
    total_cost = 0
    for query in queries:
        max_bid = -1
        winning_advertiser = None
        for advertiser in advertisers:
            if query in advertiser.bids and advertiser.bids[query] > max_bid:
                max_bid = advertiser.bids[query]
                winning_advertiser = advertiser
        if winning_advertiser is not None and winning_advertiser.budget >= max_bid:
            winning_advertiser.budget -= max_bid
            total_cost += max_bid
    return total_cost

def greedy_online(advertisers, queries):
    total_cost = 0
    for query in queries:
        max_bid = -1
        winning_advertiser = None
        for advertiser in advertisers:
            if query in advertiser.bids and advertiser.bids[query] > max_bid and advertiser.budget >= advertiser.bids[query]:
                max_bid = advertiser.bids[query]
                winning_advertiser = advertiser
        if winning_advertiser is not None:
            winning_advertiser.budget -= max_bid
            total_cost += max_bid
    return total_cost

if __name__ == "__main__":
    # Create advertisers A and B
    advertiser_a = AdvertiserThree(90, {"T1": 0.10})
    advertiser_b = AdvertiserThree(60, {"T1": 0.20, "T2": 0.20})

    advertisers = [advertiser_a, advertiser_b]
    queries = ["T1", "T2"]

    # Calculate competitive ratios
    optimal_cost = optimal_offline(advertisers, queries)
    greedy_cost = greedy_online(advertisers, queries)

    competitive_ratio = greedy_cost / optimal_cost
    print("Competitive Ratio:", competitive_ratio)



print(" ")
print("Quiz 10")
print(" ")


import random

class AdvertiserFour:
    def __init__(self, budget, bid):
        self.budget = budget
        self.bid = bid

def balance_ratio(advertiser):
    return advertiser.bid / advertiser.budget

def allocate_queries(advertiser_a, advertiser_b, num_queries):
    allocations = ""
    for _ in range(num_queries):
        if advertiser_a.budget >= advertiser_a.bid and (advertiser_b.budget < advertiser_b.bid or balance_ratio(advertiser_a) > balance_ratio(advertiser_b)):
            allocations += "A"
            advertiser_a.budget -= advertiser_a.bid
        elif advertiser_b.budget >= advertiser_b.bid and (advertiser_a.budget < advertiser_a.bid or balance_ratio(advertiser_b) > balance_ratio(advertiser_a)):
            allocations += "B"
            advertiser_b.budget -= advertiser_b.bid
        else:
            # If both advertisers have the same balance ratio, break the tie randomly
            if balance_ratio(advertiser_a) == balance_ratio(advertiser_b):
                allocations += random.choice(["A", "B"])
            else:
                # Otherwise, choose the advertiser with the higher balance ratio
                if balance_ratio(advertiser_a) > balance_ratio(advertiser_b):
                    allocations += "A"
                    advertiser_a.budget -= advertiser_a.bid
                else:
                    allocations += "B"
                    advertiser_b.budget -= advertiser_b.bid
    return allocations

# Initial budgets and bids for advertisers A and B
advertiser_a = AdvertiserFour(16, 2)
advertiser_b = AdvertiserFour(12, 3)

num_queries = 5
allocations = allocate_queries(advertiser_a, advertiser_b, num_queries)
print("Sequence of allocations:", allocations)
