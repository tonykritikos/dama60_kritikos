import numpy as np

# Data points and labels
X = np.array([[1, 1], [1 / 2, 3], [2, 5], [2, 1], [3, 3], [4, 2]])
y = np.array([1, 1, 1, -1, -1, -1])

print("------- Topic 2 -------")


# Perceptron Classifier
def perceptron_classifier(X, y, w_init, eta):
	w = w_init.copy()
	correctly_classified = 0

	for i in range(len(X)):
		x_i = X[i]
		y_i = y[i]

		prediction = np.sign(np.dot(w, x_i))
		if prediction == y_i:
			correctly_classified += 1
		else:
			w += eta * y_i * x_i.astype(float)

	return w, correctly_classified


# Support Vector Machines Classifier using Gradient Descent
def svm_classifier(X, y, w_init, bias_init, eta, C, iterations):
	w = w_init.copy()
	bias = bias_init
	n = len(X)

	for iter in range(iterations):
		decisions = ""
		for i in range(n):
			x_i = X[i]
			y_i = y[i]

			hinge_loss = np.maximum(0, 1 - y_i * (np.dot(w, x_i) + bias))
			gradient = -y_i * x_i if hinge_loss > 0 else 0

			w -= eta * (gradient + 1 / C * w)
			bias -= eta * np.sum(gradient)

			decision = 'o' if hinge_loss == 0 else 'x'
			decisions += decision

		print(f"Iteration {iter + 1}: Decisions regarding (ABCDEF) points: {decisions}")
		print(f"w^new = {w}, bias^new = {bias}\n")

	return w, bias


# Perceptron Classifier parameters
w_init_perceptron = np.array([-1.0, 1.0])
eta_perceptron = 1 / 4

# SVM Classifier parameters
w_init_svm = np.array([-1.0, 0.0])
bias_init_svm = -1.0
eta_svm = 1 / 4
C_svm = 2
iterations_svm = 2

# Perceptron Classifier
w_perceptron, correct_points_perceptron = perceptron_classifier(X, y, w_init_perceptron, eta_perceptron)
print(f"Perceptron Classifier:")
print(f"w^new = {w_perceptron}, Number of correctly classified points (out of 6) = {correct_points_perceptron}\n")

# SVM Classifier
w_svm, bias_svm = svm_classifier(X, y, w_init_svm, bias_init_svm, eta_svm, C_svm, iterations_svm)


print("------- Topic 3 -------")

# Original Rating Matrix
R = np.array([
    [5, 3, 5, 5, 2, 1],
    [4, 4, 4, 4, 1, 0],
    [3, 3, 4, 5, 0, 0],
    [4, 2, 5, 4, 0, 0],
    [1, 0, 1, 0, 5, 5],
    [0, 0, 1, 2, 4, 5],
    [0, 0, 0, 0, 4, 3],
    [1, 0, 1, 1, 5, 4]
])

# Singular Value Decomposition
U, s, VT = np.linalg.svd(R, full_matrices=False)
S = np.diag(s[:2])

# Approximated Rating Matrix
R_approx = U[:, :2] @ S @ VT[:2, :]

# Compute Retained Energy and MSE
total_energy = np.sum(s**2)
retained_energy = np.sum(s[:2]**2)
re_percentage = (retained_energy / total_energy) * 100
mse = np.mean((R - R_approx)**2)

print(f"Retained Energy (%): {re_percentage:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# New user ratings
George_ratings = np.array([0, 4, 5, 0, 0, 1])
Robert_ratings = np.array([0, 1, 0, 0, 4, 0])

# Representations in the approximated concept space
George_representation = George_ratings @ VT[:2, :].T
Robert_representation = Robert_ratings @ VT[:2, :].T

# Cosine similarity between George and Robert
cos_similarity = np.dot(George_representation, Robert_representation) / (np.linalg.norm(George_representation) * np.linalg.norm(Robert_representation))

print(f"George representation in the approximated concept space: {George_representation}")
print(f"Robert representation in the approximated concept space: {Robert_representation}")
print(f"Cosine Similarity between George and Robert: {cos_similarity:.2f}")

# Top-6 movie recommendation list for George
concept_space = R_approx[:, :2]
movie_scores = George_representation @ concept_space.T

# Get the movie names and ratings
movie_names = np.array(["Matrix", "Alien", "Star Wars", "The Avengers", "Love Actually", "Titanic"])
ratings = R[0]

# Sort the movies based on movie scores while preserving original order in case of ties
sorted_indices = np.lexsort((np.arange(len(movie_scores)), -movie_scores))

# Get the sorted movies and their ratings
sorted_movies = []
for i in sorted_indices:
    if i < len(movie_names):
        sorted_movies.append((movie_names[i], ratings[i]))

# Ensure only up to 6 movies are included
sorted_movies = sorted_movies[:6]

# Print the top-6 movie recommendation list for George
print("Top-6 movie recommendation list for George:")
for i, (movie, _) in enumerate(sorted_movies, start=1):
    movie_index = np.where(movie_names == movie)[0][0]
    print(f"{i}. {movie}: {R[0, movie_index]}")


print("------- Topic 4 -------")

# Given parameters
x1, x2 = 1, 5
y_hat = 1
w11, w12, w21, w22 = 0.10, 0.20, 0.05, -0.10
u1, u2 = 0.20, 0.50

# Forward pass calculations for the hidden layer
z1 = w11 * x1 + w21 * x2
z2 = w12 * x1 + w22 * x2

# Activation functions are identity, so h1 = z1 and h2 = z2
h1, h2 = z1, z2

# Output layer calculation
y = u1 * h1 + u2 * h2

# Loss function calculation (Mean Squared Error)
loss = 0.5 * (y - y_hat) ** 2

# Backpropagation steps
# Derivative of loss w.r.t output y
dloss_dy = y - y_hat

# Derivatives of output y w.r.t each weight
dy_du1 = h1
dy_du2 = h2
dy_dh1 = u1
dy_dh2 = u2

# Since h1 and h2 are identity functions of z1 and z2 respectively
dh1_dz1 = 1
dh2_dz2 = 1

# Derivative of z1 and z2 w.r.t weights
dz1_dw11 = x1
dz1_dw21 = x2
dz2_dw12 = x1
dz2_dw22 = x2

# Chain rule to find derivatives of loss w.r.t weights
dloss_dw11 = dloss_dy * dy_dh1 * dh1_dz1 * dz1_dw11
dloss_dw21 = dloss_dy * dy_dh1 * dh1_dz1 * dz1_dw21
dloss_dw12 = dloss_dy * dy_dh2 * dh2_dz2 * dz2_dw12
dloss_dw22 = dloss_dy * dy_dh2 * dh2_dz2 * dz2_dw22
dloss_du1 = dloss_dy * dy_du1
dloss_du2 = dloss_dy * dy_du2

# Learning rate
lr = 0.10

# Update weights
updated_w11 = w11 - lr * dloss_dw11
updated_w21 = w21 - lr * dloss_dw21
updated_w12 = w12 - lr * dloss_dw12
updated_w22 = w22 - lr * dloss_dw22
updated_u1 = u1 - lr * dloss_du1
updated_u2 = u2 - lr * dloss_du2

print("a ", loss)
print("b ", updated_w11, updated_w21, updated_w12, updated_w22, updated_u1, updated_u2)
