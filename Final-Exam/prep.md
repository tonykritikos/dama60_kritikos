Based on the uploaded files, here is a short list of all the topics discussed in your lessons:

### Tutorial Meeting 1
- Frequent Pattern Mining
- Apriori Algorithm
- FP-Growth Algorithm

### Tutorial Meeting 2
- MapReduce and the New Software Stack
- Introduction to MapReduce
- Single Node and Cluster Architectures
- Challenges of Large-scale Computing
- Programming Model of MapReduce
- Examples and Applications of MapReduce
- Finding Similar Items: Locality Sensitive Hashing (LSH)
- Introduction to Hashing
- LSH for Similarity Search
- Applications of LSH

### Tutorial Meeting 3
- Link Analysis
- PageRank Algorithm
- HITS Algorithm
- Clustering
- K-means Algorithm
- Hierarchical Clustering
- Density-based Clustering (DBSCAN)

### Tutorial Meeting 4
- Advertising on the Web
- Online Algorithms
- Bipartite Matching and Web Advertising Models
- Performance-based Advertising
- Adwords Problem and Algorithms
- Recommendation Systems
- Content-based Recommendation Systems
- Collaborative Filtering (User-based and Item-based)
- Dimensionality Reduction (UV-Decomposition)
- Evaluation Methods (MAE, RMSE, Precision, Recall, ROC, nDCG)
- Mining Social Network Graphs
- Preliminaries in Graph Theory
- Community Detection Algorithms
- Node Characteristics and Similarity
- Random Walks and PageRank

### Tutorial Meeting 5
- Dimensionality Reduction
- Techniques and Applications of Dimensionality Reduction
- Large-scale Machine Learning
- Challenges and Solutions for Machine Learning at Scale
- Neural Networks and Deep Learning
- Introduction to Neural Networks
- Deep Learning Architectures and Algorithms

These topics cover a comprehensive range of foundational and advanced concepts in data science and machine learning, focusing on both theoretical and practical aspects.

## Basic Definitions

### Tutorial Meeting 1
- **Apriori Algorithm**: An algorithm for mining frequent itemsets for boolean association rules. It uses a breadth-first search strategy to count the support of itemsets and a candidate generation function which exploits the downward closure property of support.
- **FP-Growth Algorithm**: A method for finding frequent itemsets without candidate generation. It uses a special data structure called an FP-tree, which is constructed in two passes over the data.

### Tutorial Meeting 2
- **MapReduce**: A programming model for processing large data sets with a parallel, distributed algorithm on a cluster.
- **Single Node Architecture**: Refers to traditional computing where a single machine is used for processing tasks.
- **Cluster Architecture**: Involves multiple nodes (computers) connected to form a cluster, each handling part of the data and computation.
- **Large-scale Computing**: Computing on a scale that involves processing large volumes of data across many machines.
- **Hashing**: A process that takes an input (or 'message') and returns a fixed-size string of bytes.
- **Locality Sensitive Hashing (LSH)**: A method of performing probabilistic dimension reduction of high-dimensional data. It hashes input items so that similar items map to the same buckets with high probability.

### Tutorial Meeting 3
- **PageRank Algorithm**: An algorithm used by Google Search to rank web pages in their search engine results. It counts the number and quality of links to a page to determine a rough estimate of the website's importance.
- **HITS Algorithm**: The Hyperlink-Induced Topic Search algorithm identifies two types of web pages: hubs and authorities. A good hub links to many good authorities, and a good authority is linked from many good hubs.
- **K-means Algorithm**: A clustering algorithm that partitions data into K distinct clusters based on distance to the centroid of the clusters.
- **Hierarchical Clustering**: A method of cluster analysis which seeks to build a hierarchy of clusters.
- **Density-based Clustering (DBSCAN)**: Clusters data based on density, where clusters are areas of higher density than the rest of the data set.

### Tutorial Meeting 4
- **Online Algorithms**: Algorithms that process their input piece-by-piece in a serial fashion, typically in the same order as the input is fed to the algorithm.
- **Bipartite Matching**: A type of graph matching where the vertices of the graph can be divided into two disjoint sets such that no two graph vertices within the same set are adjacent.
- **Performance-based Advertising**: An advertising model where advertisers pay for specific actions taken by users, such as clicks or purchases.
- **Content-based Recommendation Systems**: Systems that recommend items based on the characteristics of the items themselves and a profile of the user's preferences.
- **Collaborative Filtering**: A method of making automatic predictions about the interests of a user by collecting preferences or taste information from many users.
  - **User-based Collaborative Filtering**: Recommendations are made based on the ratings of similar users.
  - **Item-based Collaborative Filtering**: Recommendations are made based on the similarity between items.
- **Dimensionality Reduction (UV-Decomposition)**: Techniques used to reduce the number of random variables under consideration by obtaining a set of principal variables.
- **Graphs**: Mathematical structures used to model pairwise relations between objects. In data science, graphs are used to represent networks of data points.
- **Community Detection Algorithms**: Techniques to identify groups of nodes in a graph that are more densely connected internally than with the rest of the network.
- **Random Walks and PageRank**: Algorithms used to determine the importance of nodes in a graph based on their connectivity and the probability of visiting nodes randomly.

### Tutorial Meeting 5
- **Dimensionality Reduction**: The process of reducing the number of random variables under consideration by obtaining a set of principal variables.
- **Principal Component Analysis (PCA)**: A technique used to emphasize variation and bring out strong patterns in a dataset.
- **Large-scale Machine Learning**: Techniques and tools used to handle and analyze data sets that are too large to fit into the memory of a single computer.
- **Distributed Computing**: Computing processes are spread over multiple computing devices.
- **Neural Networks**: Computing systems inspired by the biological neural networks that constitute animal brains. A neural network is a series of algorithms that attempts to recognize underlying relationships in a set of data.
- **Deep Learning**: A subset of machine learning involving neural networks with many layers. These networks are capable of learning from data that is both unstructured and unlabeled.

## Use Cases and Example Exercises

### Tutorial Meeting 1
- **Apriori Algorithm**
  - **Use Case**: Market basket analysis to identify products frequently bought together.
  - **Exercise Example**: Given a dataset of transactions, use the Apriori algorithm to find frequent itemsets with a minimum support threshold of 3.
    - **Transactions**: {A,B,C}, {A,B}, {A,C}, {B,C}, {A,B,C}
    - **Minimum Support**: 3
    - **Find all frequent itemsets**.

### Tutorial Meeting 2
- **MapReduce**
  - **Use Case**: Processing large-scale data such as log files, web indexing, and data mining.
  - **Exercise Example**: Write a MapReduce program to count the number of occurrences of each word in a large text file.
    - **Input**: Large text file.
    - **Output**: List of words and their counts.
- **Locality Sensitive Hashing (LSH)**
  - **Use Case**: Near-duplicate detection in documents, image similarity search.
  - **Exercise Example**: Given a dataset of document vectors, use LSH to find pairs of documents with a cosine similarity above 0.8.
    - **Documents**: {doc1: [1, 0, 1], doc2: [0, 1, 1], doc3: [1, 1, 0]}
    - **Find pairs with similarity > 0.8**.

### Tutorial Meeting 3
- **PageRank Algorithm**
  - **Use Case**: Ranking web pages in search engines.
  - **Exercise Example**: Implement the PageRank algorithm on a small web graph.
    - **Graph**: A -> B, B -> C, C -> A, A -> C
    - **Compute the PageRank for each page**.
- **K-means Algorithm**
  - **Use Case**: Customer segmentation, document clustering.
  - **Exercise Example**: Apply the K-means algorithm to cluster the following points into 2 clusters: (1,1), (2,1), (4,3), (5,4).
    - **Points**: (1,1), (2,1), (4,3), (5,4)
    - **Number of clusters**: 2
    - **Initial centroids**: (1,1) and (5,4)
    - **Perform K-means clustering**.

### Tutorial Meeting 4
- **Performance-based Advertising**
  - **Use Case**: Allocating ad space on web pages based on user behavior and preferences.
  - **Exercise Example**: Given a set of bids and budgets from advertisers, implement a simple greedy algorithm to maximize revenue.
    - **Advertisers**: A: $2, B: $1, C: $1.5
    - **Budgets**: A: $10, B: $5, C: $7
    - **Queries**: Q1, Q2, Q3, Q4, Q5
    - **Allocate ads to maximize revenue**.
- **Collaborative Filtering**
  - **Use Case**: Recommending products, movies, or music to users based on their preferences.
  - **Exercise Example**: Build a user-based collaborative filtering system to recommend movies.
    - **Users' ratings**: User1: {Movie1: 5, Movie2: 3}, User2: {Movie1: 4, Movie2: 2, Movie3: 5}, User3: {Movie2: 4, Movie3: 3}
    - **Predict User1's rating for Movie3**.
- **Random Walk with Restart (RWR)**
  - **Use Case**: Analyzing social networks to detect communities or influential users.
  - **Exercise Example**: Use the Random Walk with Restart (RWR) algorithm to find the most relevant nodes in a social network starting from a given node.
    - **Graph**: A -> B, B -> C, C -> A, A -> D, D -> E
    - **Starting node**: A
    - **Perform RWR to find node relevance**.

### Tutorial Meeting 5
- **Principal Component Analysis (PCA)**
  - **Use Case**: Reducing the complexity of models, visualization of high-dimensional data.
  - **Exercise Example**: Apply Principal Component Analysis (PCA) to reduce the dimensionality of a dataset with 3 features to 2 features.
    - **Dataset**: {A: [1,2,3], B: [4,5,6], C: [7,8,9]}
    - **Perform PCA to reduce to 2 features**.
- **Large-scale Machine Learning**
  - **Use Case**: Training machine learning models on very large datasets.
  - **Exercise Example**: Implement a distributed gradient descent algorithm to train a linear regression model on a large dataset.
    - **Dataset**: 1 million records with 10 features.
    - **Goal**: Train a linear regression model using MapReduce.
- **Neural Networks and Deep Learning**
  - **Use Case**: Image and speech recognition, natural language processing.
  - **Exercise Example**: Build and train a simple feedforward neural network to classify handwritten digits from the MNIST dataset.
    - **Dataset**: MNIST
    - **Model**: 3-layer neural network (input, hidden, output)
    - **Task**: Classify digits 0-9.

## Solutions to Example Exercises

### Tutorial Meeting 1
- **Apriori Algorithm**
  - **Solution**:
    - **Transactions**: {A,B,C}, {A,B}, {A,C}, {B,C}, {A,B,C}
    - **Minimum Support**: 3
    - **Step 1**: Generate candidate itemsets of size 1: {A}, {B}, {C}
      - Count: {A: 4}, {B: 4}, {C: 4}
    - **Step 2**: Generate frequent itemsets of size 1 with support >= 3: {A}, {B}, {C}
    - **Step 3**: Generate candidate itemsets of size 2: {A,B}, {A,C}, {B,C}
      - Count: {A,B: 3}, {A,C: 3}, {B,C: 3}
    - **Step 4**: Generate frequent itemsets of size 2 with support >= 3: {A,B}, {A,C}, {B,C}
    - **Step 5**: Generate candidate itemsets of size 3: {A,B,C}
      - Count: {A,B,C: 3}
    - **Step 6**: Generate frequent itemsets of size 3 with support >= 3: {A,B,C}
    - **Result**: Frequent itemsets: {A}, {B}, {C}, {A,B}, {A,C}, {B,C}, {A,B,C}

### Tutorial Meeting 2
- **MapReduce**
  - **Solution**:
    - **Map Function**:
      ```python
      def map_function(document):
          for word in document.split():
              emit(word, 1)
      ```
    - **Reduce Function**:
      ```python
      def reduce_function(word, counts):
          total = sum(counts)
          emit(word, total)
      ```
    - **Example Input**:
      ```plaintext
      document1: "the quick brown fox"
      document2: "the quick brown dog"
      ```
    - **Map Output**:
      ```plaintext
      ("the", 1), ("quick", 1), ("brown", 1), ("fox", 1)
      ("the", 1), ("quick", 1), ("brown", 1), ("dog", 1)
      ```
    - **Reduce Output**:
      ```plaintext
      ("the", 2), ("quick", 2), ("brown", 2), ("fox", 1), ("dog", 1)
      ```

### Tutorial Meeting 3
- **PageRank Algorithm**
  - **Solution**:
    - **Graph**:
      ```plaintext
      A -> B, B -> C, C -> A, A -> C
      ```
    - **PageRank Algorithm**:
      - Initialize PageRank for each node to 1/N (N = number of nodes)
      - PageRank(A) = PageRank(B) = PageRank(C) = 1/3
      - Update PageRank iteratively using the formula:
        ```plaintext
        PR(A) = (1 - d)/N + d * (PR(C)/2)
        PR(B) = (1 - d)/N + d * (PR(A)/1)
        PR(C) = (1 - d)/N + d * (PR(A)/2 + PR(B)/1)
        ```
      - Assume d = 0.85 and iterate until convergence.

### Tutorial Meeting 4
- **Performance-based Advertising**
  - **Solution**:
    - **Advertisers**:
      ```plaintext
      A: $2, B: $1, C: $1.5
      ```
    - **Budgets**:
      ```plaintext
      A: $10, B: $5, C: $7
      ```
    - **Queries**:
      ```plaintext
      Q1, Q2, Q3, Q4, Q5
      ```
    - **Greedy Algorithm**:
      - Allocate ads to maximize revenue.
      - Query 1: A (max bid)
      - Query 2: A (max bid)
      - Query 3: C (next highest bid)
      - Query 4: A (max bid)
      - Query 5: C (next highest bid)

### Tutorial Meeting 5
- **Principal Component Analysis (PCA)**
  - **Solution**:
    - **Dataset**:
      ```plaintext
      {A: [1,2,3], B: [4,5,6], C: [7,8,9]}
      ```
    - **PCA Steps**:
      - Compute the covariance matrix.
      - Find the eigenvectors and eigenvalues of the covariance matrix.
      - Choose the top 2 eigenvectors based on eigenvalues.
      - Transform the original data onto the new 2-dimensional subspace.

## Additional Examples from Files

### 1. **Winnow Algorithm Example**
- **Source**: TM5-ch12-large-scale-Machine-Learning.pdf
- **Description**: Demonstrates the Winnow algorithm for binary classification.
- **Example**:
  - **Initial weights**: \( w = [2, 2, 1, 2, 2] \)
  - **Example input**: \( b = [0, 0, 1, 1, 0] \) with label \( y_b = -1 \)
  - **Calculation**: \( w \cdot b = 2 \cdot 0 + 2 \cdot 0 + 1 \cdot 1 + 2 \cdot 1 + 2 \cdot 0 = 3 \)
  - **Update weights**: No change since the example is correctly classified.
  - **Final weights**: \( w = [1, 8, 2, 1, 4] \)

### 2. **SVD Example**
- **Source**: TM5-ch11-dimensionality-reduction.pdf
- **Description**: Shows how Singular Value Decomposition (SVD) is used for dimensionality reduction.
- **Example Matrix**:
```plaintext
1 1 1 0 0
3 3 3 0 0
4 4 4 0 0
5 5 5 0 0
0 2 0 4 4
0 0 0 5 5
0 1 0 2 2
```
- **SVD Decomposition**:
```plaintext
U * Σ * V^T
```
- **Use Case**: Reduce the dimensions of the data while retaining important information.

### 3. **Support Vector Machines (SVM) Example**
- **Source**: TM5-ch12-large-scale-Machine-Learning.pdf
- **Description**: Illustrates how SVM finds the hyperplane that best separates data into classes.
- **Example Points**:
```plaintext
Positive: (1, 4, 1), (2, 2, 1), (3, 4, 1)
Negative: (1, 1, 1), (2, 1, 1), (3, 1, 1)
```
- **Calculation of Weights**:
- **Partial derivatives**: 
  ```
  ∂f/∂w1 = w1 + 0.1 - 2 = -0.2
  ∂f/∂w2 = w2 + 0.1 - 2 = 0.8
  ∂f/∂w3 = w3 + 0.1 - 1 = -2.1
  ```
- **New weights**: 
  ```
  w1 = 0 - 0.2 * -0.2 = 0.04
  w2 = 1 - 0.2 * 0.8 = 0.84
  w3 = -2 - 0.2 * -2.1 = -1.58
  ```

### 4. **User-based Collaborative Filtering Example**
- **Source**: TM4-ch09-recommendation-systems.pdf
- **Description**: Demonstrates how collaborative filtering recommends items based on user preferences.
- **Example Ratings**:
```plaintext
Alice: {Item1: 5, Item2: 3, Item3: 4, Item4: 4, Item5: ?}
User1: {Item1: 3, Item2: 1, Item3: 2, Item4: 3, Item5: 3}
User2: {Item1: 4, Item2: 3, Item3: 4, Item4: 3, Item5: 5}
User3: {Item1: 3, Item2: 3, Item3: 1, Item4: 5, Item5: 4}
User4: {Item1: 1, Item2: 5, Item3: 5, Item4: 2, Item5: 1}
```
- **Predicted Rating for Alice on Item5**:
- Calculate the similarity between Alice and other users.
- Use weighted average of ratings from the most similar users.

### 5. **DBSCAN Algorithm Example**
- **Source**: TM1b.pdf
- **Description**: DBSCAN groups points that are closely packed together and identifies points in low-density regions as outliers.
- **Example Points and Labels**:
```plaintext
Point P2: Core (neighbors: P4, P7)
Point P4: Core (neighbors: P2, P7)
Point P6: Core (neighbors: P7, P8)
Point P7: Core (neighbors: P2, P4, P6)
```
- **Clusters**:
```plaintext
Cluster: P2, P4, P6, P7, P8
Noise: P1, P3, P5
```

### 6. **Random Walk with Restart (RWR) Example**
- **Source**: TM4-ch10-mining-social-network-graphs.pdf
- **Description**: RWR finds relevant nodes in a graph, starting from a given node and incorporating the probability of returning to the start node.
- **Example Graph**:
```plaintext
A -> B, B -> C, C -> A, A -> D, D -> E
```
- **Starting node**: A
- **Relevance Scores after iterations**:
```plaintext
r_new(A) = 0.15 + 0.85 * (r(C) / 2 + r(D) / 1)
r_new(B) = 0.85 * r(A) / 1
r_new(C) = 0.85 * r(B) / 1
r_new(D) = 0.85 * r(A) / 2
r_new(E) = 0.85 * r(D) / 1
```

### 7. **Principal Component Analysis (PCA) Example**
- **Source**: TM5-ch11-dimensionality-reduction.pdf
- **Description**: PCA is used to reduce the dimensionality of a dataset while preserving as much variability as possible.
- **Example Dataset**:
```plaintext
{A: [1,2,3], B: [4,5,6], C: [7,8,9]}
```
- **PCA Steps**:
- Compute the covariance matrix.
- Find the eigenvectors and eigenvalues of the covariance matrix.
- Choose the top 2 eigenvectors based on eigenvalues.
- Transform the original data onto the new 2-dimensional subspace.

### 8. **CUR Decomposition Example**
- **Source**: TM5-ch11-dimensionality-reduction.pdf
- **Description**: CUR decomposition is a dimensionality reduction technique that uses a subset of rows and columns to approximate the original matrix.
- **Example Matrix**:
```plaintext
1 1 1 0 0
3 3 3 0 0
4 4 4 0 0
5 5 5 0 0
0 0 0 4 4
0 0 0 5 5
0 0 0 2 2
```
- **Steps**:
- Randomly select rows and columns based on probability.
- Construct matrices \(C\) and \(R\) from the selected columns and rows.
- Compute the intersection matrix \(W\) and apply SVD.
- Compute the CUR decomposition: \(U = W^\dagger\).

### 9. **Perceptron Algorithm Example**
- **Source**: TM5-ch12-large-scale-Machine-Learning.pdf
- **Description**: The Perceptron algorithm is a supervised learning algorithm used for binary classifiers.
- **Example Process**:
- **Initial weights**: \( w = [0, 0, 0] \)
- **Example input**: \( x = [1, 1, 1] \) with label \( y = 1 \)
- **Update rule**: \( w = w + yx \)
- **Updated weights**: \( w = [1, 1, 1] \)

### 10. **FP-Growth Algorithm Example**
- **Source**: TM1a.pdf
- **Description**: FP-Growth is an efficient method of mining frequent itemsets without candidate generation.
- **Example Transactions**:
```plaintext
{A,B,C}, {A,B}, {A,C}, {B,C}, {A,B,C}
```
- **Steps**:
- Build an FP-tree.
- Traverse the FP-tree to find frequent itemsets.
- Generate rules from the frequent itemsets with minimum confidence.

## Python Functions for Calculations

```python
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import euclidean
from collections import defaultdict
from itertools import combinations
from math import log2

# Winnow Algorithm
def winnow_update(weights, feature_vector, label, alpha=2.0):
  prediction = sum(w * f for w, f in zip(weights, feature_vector)) >= len(weights) / 2
  if prediction != label:
      for i in range(len(weights)):
          if feature_vector[i] == 1:
              if label == 1:
                  weights[i] *= alpha
              else:
                  weights[i] /= alpha
  return weights

# Singular Value Decomposition (SVD)
def perform_svd(matrix):
  U, S, VT = svd(matrix)
  return U, S, VT

# Support Vector Machines (SVM) Weight Update
def svm_update(weights, learning_rate, gradient):
  return weights - learning_rate * gradient

# Collaborative Filtering Prediction
def predict_rating(user_ratings, similarity_matrix, user_id, item_id):
  numerator = 0
  denominator = 0
  for other_user_id, similarity in enumerate(similarity_matrix[user_id]):
      if other_user_id != user_id and user_ratings[other_user_id][item_id] != 0:
          numerator += similarity * user_ratings[other_user_id][item_id]
          denominator += abs(similarity)
  return numerator / denominator if denominator != 0 else 0

# DBSCAN Clustering
def perform_dbscan(data, eps, min_samples):
  db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
  return db.labels_

# Random Walk with Restart (RWR)
def random_walk_with_restart(adj_matrix, restart_prob=0.15, max_iter=100):
  n = len(adj_matrix)
  pr = np.ones(n) / n
  for _ in range(max_iter):
      pr = (1 - restart_prob) * np.dot(adj_matrix, pr) + restart_prob / n
  return pr

# Principal Component Analysis (PCA)
def perform_pca(data, n_components):
  pca = PCA(n_components=n_components)
  principal_components = pca.fit_transform(data)
  return principal_components, pca.explained_variance_ratio_

# K-means Clustering
def perform_kmeans(data, n_clusters):
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=n_clusters)
  kmeans.fit(data)
  return kmeans.labels_, kmeans.cluster_centers_

# Apriori Algorithm
def apriori(transactions, min_support):
  itemsets = defaultdict(int)
  for transaction in transactions:
      for item in transaction:
          itemsets[frozenset([item])] += 1

  def items_with_min_support(itemsets, min_support, transactions):
      return {item: support for item, support in itemsets.items() if support >= min_support}

  itemsets = items_with_min_support(itemsets, min_support, transactions)

  def join_sets(itemsets, k):
      return {i.union(j) for i in itemsets for j in itemsets if len(i.union(j)) == k}

  k = 2
  current_itemsets = itemsets
  while current_itemsets:
      candidate_itemsets = join_sets(current_itemsets, k)
      current_itemsets = defaultdict(int)
      for transaction in transactions:
          for itemset in candidate_itemsets:
              if itemset.issubset(transaction):
                  current_itemsets[itemset] += 1
      current_itemsets = items_with_min_support(current_itemsets, min_support, transactions)
      itemsets.update(current_itemsets)
      k += 1

  return itemsets

# FP-Growth Algorithm
def perform_fpgrowth(transactions, min_support):
  from mlxtend.frequent_patterns import fpgrowth
  from mlxtend.preprocessing import TransactionEncoder

  te = TransactionEncoder()
  te_ary = te.fit(transactions).transform(transactions)
  df = pd.DataFrame(te_ary, columns=te.columns_)
  return fpgrowth(df, min_support=min_support)

# PageRank Algorithm
def pagerank(graph, d=0.85, max_iter=100):
  n = len(graph)
  pr = np.ones(n) / n
  for _ in range(max_iter):
      new_pr = np.zeros(n)
      for i in range(n):
          for j in range(n):
              if graph[j][i] == 1:
                  new_pr[i] += d * pr[j] / sum(graph[j])
      new_pr = new_pr + (1 - d) / n
      pr = new_pr
  return pr

# Precision and Recall Calculation
def precision_recall(y_true, y_pred):
  precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
  return precision, recall

# Matrix Factorization for Recommendations
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.002, beta=0.02):
  Q = Q.T
  for step in range(steps):
      for i in range(len(R)):
          for j in range(len(R[i])):
              if R[i][j] > 0:
                  eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                  for k in range(K):
                      P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                      Q[k][j] = Q[k][j] + alpha (2 * eij * P[i][k] - beta * Q[k][j])
  return P, Q.T

# MinHash Signature Generation
def minhash_signature(matrix, num_hashes):
  num_rows = matrix.shape[0]
  signatures = np.full((num_hashes, matrix.shape[1]), np.inf)
  for i in range(num_hashes):
      perm = np.random.permutation(num_rows)
      for col in range(matrix.shape[1]):
          signatures[i, col] = np.min(perm[np.where(matrix[:, col] == 1)])
  return signatures

# DCG and nDCG Calculation
def dcg_at_k(r, k):
  r = np.asfarray(r)[:k]
  if r.size:
      return np.sum(r / np.log2(np.arange(2, r.size + 2)))
  return 0.

def ndcg_at_k(r, k):
  dcg_max = dcg_at_k(sorted(r, reverse=True), k)
  if not dcg_max:
      return 0.
  return dcg_at_k(r, k) / dcg_max

# Gradient Descent for Linear Regression
def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
  m = len(y)
  for _ in range(iterations):
      gradients = X.T.dot(X.dot(theta) - y) / m
      theta -= learning_rate * gradients
  return theta

# Logistic Regression Training
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
  theta = np.zeros(X.shape[1])
  m = len(y)
  for _ in range(iterations):
      gradients = X.T.dot(sigmoid(X.dot(theta)) - y) / m
      theta -= learning_rate * gradients
  return theta

# Train Random Forest
def train_random_forest(X_train, y_train, n_estimators=100):
  clf = RandomForestClassifier(n_estimators=n_estimators)
  clf.fit(X_train, y_train)
  return clf

# Train K-Nearest Neighbors (KNN)
def train_knn(X_train, y_train, n_neighbors=3):
  knn = KNeighborsClassifier(n_neighbors=n_neighbors)
  knn.fit(X_train, y_train)
  return knn

# Hierarchical Clustering
def hierarchical_clustering(data, method='ward', t=1.5):
  Z = linkage(data, method=method)
  clusters = fcluster(Z, t, criterion='distance')
  return clusters

# Generate Association Rules
def generate_association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5):
  rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
  return rules

# Receiver Operating Characteristic (ROC) Curve and AUC
def compute_roc_auc(y_true, y_scores):
  fpr, tpr, thresholds = roc_curve(y_true, y_scores)
  auc = roc_auc_score(y_true, y_scores)
  return fpr, tpr, auc

# Expectation-Maximization for Gaussian Mixture Models (GMM)
def fit_gmm(X, n_components):
  gmm = GaussianMixture(n_components=n_components)
  gmm.fit(X)
  return gmm

# Latent Dirichlet Allocation (LDA) for Topic Modeling
def fit_lda(X, n_components):
  lda = LatentDirichletAllocation(n_components=n_components)
  lda.fit(X)
  return lda

# Linear Discriminant Analysis (LDA)
def perform_lda(X, y):
  lda = LinearDiscriminantAnalysis()
  lda.fit(X, y)
  return lda

# Euclidean Distance
def euclidean_distance(point1, point2):
  return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Jaccard Similarity
def jaccard_similarity(set1, set2):
  intersection = len(set(set1) & set(set2))
  union = len(set(set1) | set(set2))
  return intersection / union

# Entropy Calculation
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
