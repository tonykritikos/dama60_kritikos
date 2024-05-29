import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from math import log2

# Set print options for better readability
np.set_printoptions(precision=4, suppress=True)


# Winnow Algorithm Weight Update
def test_winnow_update():
	weights = [2, 2, 1, 2, 2]
	inputs = [0, 0, 1, 1, 0]
	label = -1
	updated_weights = winnow_update(weights, inputs, label)
	print("Winnow Update:", ["{:.4f}".format(w) for w in updated_weights])


def winnow_update(weights, inputs, label, alpha=2):
	prediction = sum(w * i for w, i in zip(weights, inputs))
	if prediction * label <= 0:
		for j in range(len(weights)):
			if inputs[j] == 1:
				weights[j] *= alpha
			else:
				weights[j] /= alpha
	return weights


# Perform SVD
def test_perform_svd():
	matrix = np.array([
		[1, 1, 1, 0, 0],
		[3, 3, 3, 0, 0],
		[4, 4, 4, 0, 0],
		[5, 5, 5, 0, 0],
		[0, 2, 0, 4, 4],
		[0, 0, 0, 5, 5],
		[0, 1, 0, 2, 2]
	])
	U, S, VT = perform_svd(matrix)
	print("SVD U:", U)
	print("SVD S:", ["{:.4f}".format(s) for s in S])
	print("SVD VT:", VT)


def perform_svd(matrix):
	U, S, VT = np.linalg.svd(matrix)
	return U, S, VT


# SVM Weight Update
def test_svm_update():
	weights = [0, 1, -2]
	inputs = [1, 4, 1]
	label = 1
	updated_weights = svm_update(weights, inputs, label)
	print("SVM Update:", ["{:.4f}".format(w) for w in updated_weights])


def svm_update(weights, inputs, label, learning_rate=0.1, lambda_param=0.01):
	margin = label * sum(w * i for w, i in zip(weights, inputs))
	if margin < 1:
		weights = [w + learning_rate * (label * i - lambda_param * w) for w, i in zip(weights, inputs)]
	else:
		weights = [w - learning_rate * lambda_param * w for w in weights]
	return weights


# User-based Collaborative Filtering Prediction
def test_predict_rating():
	user_ratings = {
		'Alice': [5, 3, 4, 4, 0],
		'User1': [3, 1, 2, 3, 3],
		'User2': [4, 3, 4, 3, 5],
		'User3': [3, 3, 1, 5, 4],
		'User4': [1, 5, 5, 2, 1]
	}
	target_user = 'Alice'
	target_item = 4  # Item5
	prediction = predict_rating(user_ratings, ['User1', 'User2', 'User3', 'User4'], target_user, target_item)
	print("Collaborative Filtering Prediction: {:.4f}".format(prediction))


def cosine_similarity(a, b):
	return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def predict_rating(user_ratings, other_users, target_user, target_item):
	similarities = []
	for other_user in other_users:
		similarity = cosine_similarity(user_ratings[target_user], user_ratings[other_user])
		similarities.append((similarity, user_ratings[other_user][target_item]))
	similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
	numerator = sum(sim * rating for sim, rating in similarities[:2])  # Top 2 similar users
	denominator = sum(sim for sim, rating in similarities[:2])
	return numerator / denominator if denominator != 0 else 0


# Perform DBSCAN Clustering
def test_perform_dbscan():
	points = np.array([
		[1, 2], [2, 2], [2, 3],
		[8, 7], [8, 8], [25, 80]
	])
	eps = 3
	min_samples = 2
	labels = perform_dbscan(points, eps, min_samples)
	print("DBSCAN Labels:", labels)


def perform_dbscan(points, eps, min_samples):
	clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
	return clustering.labels_


# Random Walk with Restart
def test_random_walk_with_restart():
	adj_matrix = np.array([
		[0, 1, 0, 1, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 0, 1, 0, 1],
		[0, 0, 0, 0, 0]
	])
	restart_prob = 0.15
	start_node = 0
	probabilities = random_walk_with_restart(adj_matrix, restart_prob, start_node)
	print("Random Walk with Restart:", ["{:.4f}".format(p) for p in probabilities])


def random_walk_with_restart(adj_matrix, restart_prob, start_node, max_iter=100, tol=1e-6):
	n = adj_matrix.shape[0]
	r = np.zeros(n)
	r[start_node] = 1
	p = np.zeros(n)

	for _ in range(max_iter):
		p_new = (1 - restart_prob) * np.dot(adj_matrix, p) + restart_prob * r
		if np.linalg.norm(p_new - p) < tol:
			break
		p = p_new

	return p


# Perform PCA
def test_perform_pca():
	data = np.array([
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9],
		[10, 11, 12]
	])
	n_components = 2
	principal_components, explained_variance_ratio = perform_pca(data, n_components)
	print("PCA Principal Components:", principal_components)
	print("PCA Explained Variance Ratio:", ["{:.4f}".format(evr) for evr in explained_variance_ratio])


def perform_pca(data, n_components):
	pca = PCA(n_components=n_components)
	principal_components = pca.fit_transform(data)
	return principal_components, pca.explained_variance_ratio_


# Perform K-means Clustering
def test_perform_kmeans():
	data = np.array([
		[1, 2],
		[1, 4],
		[1, 0],
		[4, 2],
		[4, 4],
		[4, 0]
	])
	n_clusters = 2
	labels, centers = perform_kmeans(data, n_clusters)
	print("K-means Labels:", labels)
	print("K-means Centers:", centers)


def perform_kmeans(data, n_clusters):
	kmeans = KMeans(n_clusters=n_clusters, n_init=10)  # Explicitly set n_init to suppress the warning
	kmeans.fit(data)
	return kmeans.labels_, kmeans.cluster_centers_


# Perform Apriori
def test_perform_apriori():
	transactions = [
		['A', 'B', 'C'],
		['A', 'B'],
		['A', 'C'],
		['B', 'C'],
		['A', 'B', 'C']
	]
	min_support = 0.6
	frequent_itemsets = perform_apriori(transactions, min_support)
	print("Apriori Frequent Itemsets:", frequent_itemsets)


def perform_apriori(transactions, min_support):
	te = TransactionEncoder()
	te_ary = te.fit(transactions).transform(transactions)
	df = pd.DataFrame(te_ary, columns=te.columns_)
	frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
	return frequent_itemsets


# Perform FP-Growth
def test_perform_fpgrowth():
	transactions = [
		['A', 'B', 'C'],
		['A', 'B'],
		['A', 'C'],
		['B', 'C'],
		['A', 'B', 'C']
	]
	min_support = 0.6
	frequent_itemsets = perform_fpgrowth(transactions, min_support)
	print("FP-Growth Frequent Itemsets:", frequent_itemsets)


def perform_fpgrowth(transactions, min_support):
	te = TransactionEncoder()
	te_ary = te.fit(transactions).transform(transactions)
	df = pd.DataFrame(te_ary, columns=te.columns_)
	frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
	return frequent_itemsets


# PageRank Algorithm
def test_pagerank():
	adj_matrix = np.array([
		[0, 0, 1, 0],
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 1, 1, 0]
	])
	pagerank_scores = pagerank(adj_matrix)
	print("PageRank Scores:", ["{:.4f}".format(pr) for pr in pagerank_scores])


def pagerank(adj_matrix, d=0.85, tol=1e-6, max_iter=100):
	n = adj_matrix.shape[0]
	pr = np.ones(n) / n
	M = np.zeros_like(adj_matrix, dtype=float)

	for i in range(n):
		if adj_matrix[:, i].sum() != 0:
			M[:, i] = adj_matrix[:, i] / adj_matrix[:, i].sum()
		else:
			M[:, i] = np.ones(n) / n  # Handle columns with sum of zero

	for _ in range(max_iter):
		pr_new = (1 - d) / n + d * M.dot(pr)
		if np.linalg.norm(pr_new - pr, 1) < tol:
			break
		pr = pr_new

	return pr


# Precision and Recall Calculation
def test_precision_recall():
	true_positives = 10
	false_positives = 5
	false_negatives = 3
	precision, recall = precision_recall(true_positives, false_positives, false_negatives)
	print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")


def precision_recall(true_positives, false_positives, false_negatives):
	precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
	recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
	return precision, recall


# Calculate Precision and Recall
def test_calculate_precision_recall():
	tp = 10
	fp = 5
	fn = 3
	precision, recall = calculate_precision_recall(tp, fp, fn)
	print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")


def calculate_precision_recall(true_positive, false_positive, false_negative):
	precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
	recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
	return precision, recall


# Matrix Factorization for Recommendation Systems
def test_matrix_factorization():
	R = np.array([
		[5, 3, 0, 1],
		[4, 0, 0, 1],
		[1, 1, 0, 5],
		[1, 0, 0, 4],
		[0, 1, 5, 4],
	])
	N = len(R)
	M = len(R[0])
	K = 2
	P = np.random.rand(N, K)
	Q = np.random.rand(M, K)
	nP, nQ = matrix_factorization(R, P, Q, K)
	nR = np.dot(nP, nQ.T)
	print("Matrix Factorization R:", nR)


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.002, beta=0.02):
	Q = Q.T
	for step in range(steps):
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - np.dot(P[i, :], Q[:, j])
					for k in range(K):
						P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		eR = np.dot(P, Q)
		e = 0
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					e += pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
					for k in range(K):
						e += (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
		if e < 0.001:
			break
	return P, Q.T


# MinHash Signature
def test_minhash_signature():
	matrix = np.array([
		[1, 0, 0, 1],
		[0, 1, 1, 0],
		[1, 1, 0, 0],
		[0, 0, 1, 1]
	])
	num_permutations = 100
	signature = minhash_signature(matrix, num_permutations)
	print("MinHash Signature:", signature)


def minhash_signature(matrix, num_permutations):
	rows, cols = matrix.shape
	sig_matrix = np.full((num_permutations, cols), np.inf)

	for perm in range(num_permutations):
		permuted_indices = np.random.permutation(rows)
		for col in range(cols):
			for row in permuted_indices:
				if matrix[row, col] == 1:
					sig_matrix[perm, col] = min(sig_matrix[perm, col], row)
					break
	return sig_matrix


# Discounted Cumulative Gain (DCG)
def test_dcg_at_k():
	scores = [3, 2, 3, 0, 1, 2]
	k = 5
	dcg = dcg_at_k(scores, k)
	print("DCG at K:", "{:.4f}".format(dcg))


def dcg_at_k(scores, k):
	return np.sum([score / np.log2(idx + 2) for idx, score in enumerate(scores[:k])])


# Normalized Discounted Cumulative Gain (nDCG)
def test_ndcg_at_k():
	true_scores = [3, 2, 3, 0, 1, 2]
	predicted_scores = [2, 1, 2, 3, 3, 0]
	k = 5
	ndcg = ndcg_at_k(true_scores, predicted_scores, k)
	print("nDCG at K:", "{:.4f}".format(ndcg))


def ndcg_at_k(true_scores, predicted_scores, k):
	ideal_dcg = dcg_at_k(sorted(true_scores, reverse=True), k)
	actual_dcg = dcg_at_k(predicted_scores, k)
	return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


# Gradient Descent for Linear Regression
def test_gradient_descent():
	X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
	y = np.array([6, 8, 9, 11])
	theta = gradient_descent(X, y)
	print("Gradient Descent Theta:", ["{:.4f}".format(t) for t in theta])


def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
	m, n = X.shape
	theta = np.zeros(n)
	for _ in range(n_iterations):
		gradients = 2 / m * X.T.dot(X.dot(theta) - y)
		theta -= learning_rate * gradients
	return theta


# Logistic Regression with Gradient Descent
def test_logistic_regression():
	X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
	y = np.array([0, 0, 1, 1])
	theta = logistic_regression(X, y)
	print("Logistic Regression Theta:", ["{:.4f}".format(t) for t in theta])


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def logistic_regression(X, y, learning_rate=0.01, n_iterations=1000):
	m, n = X.shape
	theta = np.zeros(n)
	for _ in range(n_iterations):
		gradients = X.T.dot(sigmoid(X.dot(theta)) - y) / m
		theta -= learning_rate * gradients
	return theta


# Train Random Forest
def test_train_random_forest():
	X_train = [[1, 2], [3, 4], [5, 6], [7, 8]]
	y_train = [0, 1, 0, 1]
	clf = train_random_forest(X_train, y_train)
	print("Random Forest Prediction:", clf.predict([[5, 5]]))


def train_random_forest(X_train, y_train, n_estimators=100):
	clf = RandomForestClassifier(n_estimators=n_estimators)
	clf.fit(X_train, y_train)
	return clf


# Train K-Nearest Neighbors (KNN)
def test_train_knn():
	X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
	y_train = [0, 0, 1, 1]
	knn = train_knn(X_train, y_train)
	print("KNN Prediction:", knn.predict([[3, 3]]))


def train_knn(X_train, y_train, n_neighbors=3):
	knn = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(X_train, y_train)
	return knn


# Hierarchical Clustering
def test_hierarchical_clustering():
	data = np.array([
		[1, 2],
		[2, 2],
		[1, 1],
		[2, 1],
		[8, 8],
		[8, 9],
		[9, 8],
		[9, 9]
	])
	clusters = hierarchical_clustering(data)
	print("Hierarchical Clustering:", clusters)


def hierarchical_clustering(data, method='ward', t=1.5):
	Z = linkage(data, method=method)
	clusters = fcluster(Z, t, criterion='distance')
	return clusters


# Generate Association Rules
def test_generate_association_rules():
	transactions = [
		['A', 'B', 'C'],
		['A', 'B'],
		['A', 'C'],
		['B', 'C'],
		['A', 'B', 'C']
	]
	min_support = 0.6
	frequent_itemsets = perform_apriori(transactions, min_support)
	rules = generate_association_rules(frequent_itemsets)
	print("Association Rules:", rules)


def generate_association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5):
	rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
	return rules


# Receiver Operating Characteristic (ROC) Curve and AUC
def test_compute_roc_auc():
	y_true = [0, 0, 1, 1]
	y_scores = [0.1, 0.4, 0.35, 0.8]
	fpr, tpr, auc = compute_roc_auc(y_true, y_scores)
	print(f"FPR: {[f'{x:.4f}' for x in fpr]}, TPR: {[f'{x:.4f}' for x in tpr]}, AUC: {auc:.4f}")


def compute_roc_auc(y_true, y_scores):
	fpr, tpr, thresholds = roc_curve(y_true, y_scores)
	auc = roc_auc_score(y_true, y_scores)
	return fpr, tpr, auc


# Expectation-Maximization for Gaussian Mixture Models (GMM)
def test_fit_gmm():
	X = np.array([[1, 2], [2, 3], [3, 4], [10, 10], [10, 11], [11, 10]])
	gmm = fit_gmm(X, 2)
	print("GMM Means:", gmm.means_)
	print("GMM Predictions:", gmm.predict(X))


def fit_gmm(X, n_components):
	gmm = GaussianMixture(n_components=n_components)
	gmm.fit(X)
	return gmm


# Latent Dirichlet Allocation (LDA) for Topic Modeling
def test_fit_lda():
	from sklearn.feature_extraction.text import CountVectorizer

	documents = [
		"I love programming in Python",
		"Python is great for machine learning",
		"I enjoy learning new algorithms",
		"Machine learning and data science are fascinating"
	]

	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(documents)
	lda = fit_lda(X, n_components=2)
	print("LDA Components:", lda.components_)


def fit_lda(X, n_components):
	lda = LatentDirichletAllocation(n_components=n_components)
	lda.fit(X)
	return lda


# Linear Discriminant Analysis (LDA)
def test_perform_lda():
	X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
	y = np.array([0, 0, 1, 1])
	lda = perform_lda(X, y)
	print("LDA Prediction:", lda.predict([[2, 3]]))


def perform_lda(X, y):
	lda = LinearDiscriminantAnalysis()
	lda.fit(X, y)
	return lda


# Euclidean Distance
def test_euclidean_distance():
	point1 = [1, 2]
	point2 = [4, 6]
	distance = euclidean_distance(point1, point2)
	print("Euclidean Distance:", "{:.4f}".format(distance))


def euclidean_distance(point1, point2):
	return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


# Jaccard Similarity
def test_jaccard_similarity():
	set1 = [1, 2, 3]
	set2 = [2, 3, 4]
	similarity = jaccard_similarity(set1, set2)
	print("Jaccard Similarity:", "{:.4f}".format(similarity))


def jaccard_similarity(set1, set2):
	intersection = len(set(set1) & set(set2))
	union = len(set(set1) | set(set2))
	return intersection / union


# Entropy Calculation
def test_entropy():
	labels = [1, 1, 1, 0, 0, 0]
	ent = entropy(labels)
	print("Entropy:", "{:.4f}".format(ent))


def entropy(labels):
	n_labels = len(labels)
	if n_labels <= 1:
		return 0
	value, counts = np.unique(labels, return_counts=True)
	probs = counts / n_labels
	n_classes = np.count_nonzero(probs)
	if n_classes <= 1:
		return 0
	ent = 0.
	for p in probs:
		ent -= p * log2(p)
	return ent


# Run all tests
def run_all_tests():
	test_winnow_update()
	test_perform_svd()
	test_svm_update()
	test_predict_rating()
	test_perform_dbscan()
	test_random_walk_with_restart()
	test_perform_pca()
	test_perform_kmeans()
	test_perform_apriori()
	test_perform_fpgrowth()
	test_pagerank()
	test_precision_recall()
	test_calculate_precision_recall()
	test_matrix_factorization()
	test_minhash_signature()
	test_dcg_at_k()
	test_ndcg_at_k()
	test_gradient_descent()
	test_logistic_regression()
	test_train_random_forest()
	test_train_knn()
	test_hierarchical_clustering()
	test_generate_association_rules()
	test_compute_roc_auc()
	test_fit_gmm()
	test_fit_lda()
	test_perform_lda()
	test_euclidean_distance()
	test_jaccard_similarity()
	test_entropy()


run_all_tests()
