import numpy as np


print("----- Quiz 1 -----")
M = np.array([[1, 4],
              [2, 3],
              [3, 2],
              [4, 1]])

print("Matrix M:")
print(M)

MMT = np.dot(M, M.T)

print("\nMatrix MMT:")
print(MMT)

eigenvalues, eigenvectors = np.linalg.eig(MMT)

print("\nEigenvalues of MMT:", eigenvalues)
print("Eigenvectors of MMT:", eigenvectors)

# Option a

option_a_eigenvector = np.array([1, 1, 1, 1])
option_a_eigenvalue = 0

# Option b
option_b_eigenvector = np.array([0, 0, 1, 0])
option_b_eigenvalue = 1

# Option c
option_c_eigenvector = np.array([51, 31, 21, 11])
option_c_eigenvalue = 1

# Option e
option_e_eigenvector = np.array([1, 1, 1, 1])
option_e_eigenvalue = 50

# Checking which option matches the computed eigenpairs
matched_option = None

if np.allclose(option_a_eigenvector, eigenvectors[:, 0]) and np.isclose(option_a_eigenvalue, eigenvalues[0]):
    matched_option = 'a'
elif np.allclose(option_b_eigenvector, eigenvectors[:, 0]) and np.isclose(option_b_eigenvalue, eigenvalues[0]):
    matched_option = 'b'
elif np.allclose(option_c_eigenvector, eigenvectors[:, 0]) and np.isclose(option_c_eigenvalue, eigenvalues[0]):
    matched_option = 'c'
elif np.allclose(option_e_eigenvector, eigenvectors[:, 0]) and np.isclose(option_e_eigenvalue, eigenvalues[0]):
    matched_option = 'e'

print("\nMatched option:", matched_option)

print("----- Quiz 2 -----")

# Given values
h11 = 0.4
w11 = 0.5
h21 = 0.1
w21 = 1
h31 = 0.2
w31 = 1
b1 = 0.4
yj_expected = 1

# Printing the variables
print("h11:", h11, "w11:", w11, "h21:", h21, "w21:", w21, "h31:", h31, "w31:", w31, "b1:", b1)

# Calculate z1
z1 = w11 * h11 + w21 * h21 + w31 * h31 + b1

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calculate output yj using the sigmoid activation function
yj = sigmoid(z1)

# Calculate MSE
mse = (yj - yj_expected)**2
print(yj)
print(mse)


print("----- Quiz 3 -----")

# # Constants
# C = 0.1
# learning_rate = -0.2
#
# # Initial parameters
# w = np.array([-1, 1])
# b = 0
#
# # Training set (xi, yi)
# training_set = [
#     (np.array([1, 2]), 1),
#     (np.array([3, 4]), 1),
#     (np.array([5, 2]), 1),
#     (np.array([2, 4]), -1),
#     (np.array([3, 1]), -1),
#     (np.array([7, 3]), -1)
# ]
#
# # Gradient computation
# grad_w = np.zeros_like(w)
# grad_b = 0
#
# for xi, yi in training_set:
#     if yi * (np.dot(w, xi) + b) < 1:
#         grad_w += -yi * xi
#         grad_b += -yi
#
# # Apply updates to w and b
# w -= learning_rate * C * grad_w
# b -= learning_rate * C * grad_b
#
# # Printing the gradient and updated b value
# print(grad_b)
# print(b)

print("----- Quiz 4 -----")
# Singular values
sigma1 = 8.35006
sigma2 = 7.77337
sigma3 = 1.89068
sigma4 = 1.12981

# Total energy in Σ
total_energy = sigma1**2 + sigma2**2 + sigma3**2 + sigma4**2

# Energy with the two smallest singular values removed
energy_removed_2 = sigma1**2 + sigma2**2

# Energy with the smallest singular value removed
energy_removed_1 = sigma1**2 + sigma2**2 + sigma3**2

# Percentage of energy preserved when removing two smallest singular values
percentage_preserved_2 = (energy_removed_2 / total_energy) * 100

# Percentage of energy preserved when removing the smallest singular value
percentage_preserved_1 = (energy_removed_1 / total_energy) * 100

print(total_energy, energy_removed_2, energy_removed_1, percentage_preserved_2, percentage_preserved_1)


print("----- Quiz 5 -----")


# Matrix with user ratings
ratings = np.array([
    [3, 1, 0, 0, 0],  # Maria
    [4, 2, 0, 0, 0],  # Tim
    [5, 4, 0, 0, 0],  # Sonia
    [0, 0, 1, 2, 3],  # Peter
    [0, 0, 5, 4, 3]   # Eva
])

# User names
users = ["Maria", "Tim", "Sonia", "Peter", "Eva"]

# Calculate sum of ratings for each user
ratings_sum = np.sum(ratings, axis=1)
print("Ratings Sum for each user:", dict(zip(users, ratings_sum)))

# Identify the indices of the top two users with highest sums
top_two_indices = np.argsort(-ratings_sum)[:2]  # Sort indices by highest sums and take top two
top_two_users = [users[i] for i in top_two_indices]

print(top_two_users)

print("----- Quiz 6 -----")

# Initial conditions
w1, w2, w3 = 0, 0, 0
x1, x2, x3 = 4, 1, 3
threshold = 0.5
learning_rate = 0.2
expected_output = 1  # y

# Calculate perceptron output
output = w1 * x1 + w2 * x2 + w3 * x3
predicted_output = 1 if output >= threshold else 0

# Check if the prediction matches the expected output
if predicted_output != expected_output:
    # Update weights since predicted_output is 0 and expected_output is 1
    w1 += learning_rate * (expected_output - predicted_output) * x1
    w2 += learning_rate * (expected_output - predicted_output) * x2
    w3 += learning_rate * (expected_output - predicted_output) * x3

print(w1, w2, w3)

print("----- Quiz 7 -----")

from math import sqrt

# Coefficients for the line equation
A = 1
B = 1/2
C = -5

# Points given
points = [(1, -1), (3, 4), (4, 3)]

# Function to calculate the distance from a point to the line
def distance_to_line(x1, x2, A, B, C):
    return abs(A*x1 + B*x2 + C) / sqrt(A**2 + B**2)

# Calculate the distances for each point
distances = [distance_to_line(x1, x2, A, B, C) for x1, x2 in points]

# Find the minimum distance (margin)
margin = min(distances)
print(margin)

# Re-checking distances for each point to ensure accuracy
distances = [distance_to_line(x1, x2, A, B, C) for x1, x2 in points]
print(distances)

print("----- Quiz 8 -----")

# Define the function to calculate the intersection with x-axis and y-axis
def find_intersections(w, b):
    x1_intercept = -b / w[0]  # x2 = 0
    x2_intercept = -b / w[1]  # x1 = 0
    return (x1_intercept, 0), (0, x2_intercept)

# Define the function to classify points
def classify_point(point, w, b):
    x1, x2 = point
    output = w[0] * x1 + w[1] * x2 + b
    return +1 if output > 0 else -1

# Weights and bias
w = (2, -1)
b = 4

# Points for classification
point1 = (2, 3)
point2 = (-1, 3)

# Calculate intersections
x_axis_intersection, y_axis_intersection = find_intersections(w, b)

# Classify points
label_point1 = classify_point(point1, w, b)
label_point2 = classify_point(point2, w, b)

print(x_axis_intersection, y_axis_intersection, label_point1, label_point2)
