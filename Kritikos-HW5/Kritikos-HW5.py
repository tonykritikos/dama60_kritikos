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
