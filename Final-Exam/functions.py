import networkx as nx
from fractions import Fraction
import numpy as np

# # repeat exam 22-23 q10
#
# # Create the graph
# G = nx.Graph()
# edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('D', 'G'), ('D', 'E'), ('E', 'F'), ('G', 'F')]
# G.add_edges_from(edges)
#
# # Calculate edge betweenness centrality
# betweenness = nx.edge_betweenness_centrality(G)
#
# # Print the results
# for edge, centrality in betweenness.items():
#     print(f"Edge {edge}: {centrality:.2f}")
#
#
# #  final  exam 22-23 q16
#
# # Use fractions for exact results
# P = np.array([
#     [Fraction(0), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)],
#     [Fraction(1, 2), Fraction(0), Fraction(0), Fraction(1, 2)],
#     [Fraction(1), Fraction(0), Fraction(0), Fraction(0)],
#     [Fraction(0), Fraction(1, 2), Fraction(1, 2), Fraction(0)]
# ])
#
# # Teleportation factor
# beta = Fraction(1, 2)
#
# # Number of nodes
# N = 4
#
# # Initial distribution (uniform)
# pi_0 = np.array([Fraction(1, N)] * N)
#
# # Construct the teleportation matrix
# teleportation_matrix = np.ones((N, N), dtype=Fraction) / N
#
# # Compute the PageRank matrix with teleportation
# PR = beta * P + (1 - beta) * teleportation_matrix
#
# # Compute the distribution after one iteration
# pi_1 = pi_0 @ PR
#
# # Convert to fractions for exact output
# pi_1 = [Fraction(float(x)).limit_denominator() for x in pi_1]
#
# print(pi_1)
#
#
# # quiz 3 q2
#
# # Define the transition matrix P
# P = np.array([
#     [0, 1/3, 1/3, 1/3],
#     [1/2, 0, 0, 1/2],
#     [1, 0, 0, 0],
#     [0, 1/2, 1/2, 0]
# ])
#
# # Calculate P^3
# P3 = np.linalg.matrix_power(P, 3)
# print(P3)

import networkx as nx
from collections import defaultdict

# Calculate betweenness communities

def girvan_newman(graph):
    def edge_betweenness_centrality(graph):
        betweenness = defaultdict(float)
        nodes = graph.nodes()

        for s in nodes:
            # Single-source shortest-paths problem
            S = []
            P = {}
            for v in nodes:
                P[v] = []
            sigma = dict.fromkeys(nodes, 0.0)
            sigma[s] = 1.0
            D = {}
            Q = [s]
            D[s] = 0

            while Q:
                v = Q.pop(0)
                S.append(v)
                Dv = D[v]
                sigmav = sigma[v]
                for w in graph.neighbors(v):
                    if w not in D:
                        Q.append(w)
                        D[w] = Dv + 1
                    if D[w] == Dv + 1:
                        sigma[w] += sigmav
                        P[w].append(v)

            delta = dict.fromkeys(nodes, 0.0)
            while S:
                w = S.pop()
                for v in P[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    if w != s:
                        betweenness[(v, w)] += delta[w]
                        betweenness[(w, v)] += delta[w]

        for key in betweenness:
            betweenness[key] /= 2.0
        return betweenness

    def remove_max_betweenness_edge(graph, betweenness):
        edge = max(betweenness, key=betweenness.get)
        graph.remove_edge(*edge)
        return edge

    def get_communities(graph):
        return [list(c) for c in nx.connected_components(graph)]

    # Main loop
    original_graph = graph.copy()
    communities = []
    while graph.number_of_edges() > 0:
        betweenness = edge_betweenness_centrality(graph)
        remove_max_betweenness_edge(graph, betweenness)
        communities = get_communities(graph)
        if len(communities) > 1:
            break

    return communities, original_graph


def test_girvan_newman():
    # Create a simple graph for testing
    G = nx.Graph()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 0), (1, 3), (2, 4)
    ]
    G.add_edges_from(edges)

    # Run Girvan-Newman algorithm
    communities, _ = girvan_newman(G)

    # Print the detected communities
    print("Detected communities:")
    for community in communities:
        print(community)

# quiz 3 q6

import numpy as np


def mahalanobis_distance(point, mean, std_devs):
    """
    Calculate the Mahalanobis distance between a point and a mean vector given standard deviations.

    Parameters:
    point (array-like): The point for which the distance is being calculated.
    mean (array-like): The mean vector.
    std_devs (array-like): The standard deviations for each dimension.

    Returns:
    float: The Mahalanobis distance.
    """
    point = np.array(point)
    mean = np.array(mean)
    std_devs = np.array(std_devs)

    # Calculate the covariance matrix from standard deviations
    cov_matrix = np.diag(std_devs ** 2)

    # Calculate the Mahalanobis distance
    diff = point - mean
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distance = np.sqrt(diff.T @ inv_cov_matrix @ diff)

    return distance


def test_mah_distance():
    # Data from the question
    mean = [0, 0, 0]
    point = [1, -3, 4]
    std_devs = [2, 3, 5]

    # Calculate the Mahalanobis distance
    distance = mahalanobis_distance(point, mean, std_devs)
    print(distance)

# test_mah_distance()

import math


def calculate_psi(m_i, b_i, x_i):
    """
    Calculate the allocation function Psi_i(q) for an advertiser.

    Parameters:
    m_i (float): Remaining budget of the advertiser.
    b_i (float): Total budget of the advertiser.
    x_i (float): Product of the bid and the click-through rate.

    Returns:
    float: The value of the allocation function Psi_i(q).
    """
    f_i = 1 - (m_i / b_i)
    psi_i_q = x_i * (1 - math.exp(-f_i))
    return psi_i_q


def test_calculate_psi():
    """
    Test function for calculate_psi.
    """
    test_cases = [
        {"m_i": 50, "b_i": 100, "x_i": 10},
        {"m_i": 20, "b_i": 80, "x_i": 5},
        {"m_i": 0, "b_i": 100, "x_i": 20},
        {"m_i": 100, "b_i": 100, "x_i": 15},
    ]

    for idx, test_case in enumerate(test_cases):
        m_i = test_case["m_i"]
        b_i = test_case["b_i"]
        x_i = test_case["x_i"]

        expected_psi = x_i * (1 - math.exp(-(1 - (m_i / b_i))))
        calculated_psi = calculate_psi(m_i, b_i, x_i)

        assert math.isclose(calculated_psi, expected_psi, rel_tol=1e-9), f"Test case {idx + 1} failed"

        print(f"Test case {idx + 1}:")
        print(f"  m_i = {m_i}, b_i = {b_i}, x_i = {x_i}")
        print(f"  Calculated Psi = {calculated_psi}")
        print(f"  Expected Psi   = {expected_psi}")
        print()

    print("All test cases passed!")


# Run the test function
# test_calculate_psi()


import math
import numpy as np


def calculate_rmse(true_values, predicted_values):
    """
    Calculate the Root Mean Square Error (RMSE) between true and predicted values.

    Parameters:
    true_values (list or np.array): The true values.
    predicted_values (list or np.array): The predicted values.

    Returns:
    float: The RMSE value.
    """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    mse = np.mean((true_values - predicted_values) ** 2)
    rmse = math.sqrt(mse)
    print(rmse)
    return rmse


def test_calculate_rmse():
    """
    Test function for calculate_rmse.
    """
    true_values = [3.0, -0.5, 2.0, 7.0]
    predicted_values = [2.5, 0.0, 2.0, 8.0]

    # Calculate expected RMSE manually
    expected_rmse = math.sqrt(np.mean([(3.0 - 2.5) ** 2, (-0.5 - 0.0) ** 2, (2.0 - 2.0) ** 2, (7.0 - 8.0) ** 2]))

    calculated_rmse = calculate_rmse(true_values, predicted_values)

    print(f"True values: {true_values}")
    print(f"Predicted values: {predicted_values}")
    print(f"Calculated RMSE: {calculated_rmse}")
    print(f"Expected RMSE: {expected_rmse}")

    assert math.isclose(calculated_rmse, expected_rmse, rel_tol=1e-9), "Test case failed"

    print("Test case passed!")


# Run the test function
# test_calculate_rmse()
