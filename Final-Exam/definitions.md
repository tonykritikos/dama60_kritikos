<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Classification

## Base Classifiers
- **Decision Trees**: A tree-like model used to make decisions and predictions based on the values of input features.
  - **Information Gain**: \( IG(T, a) = H(T) - \sum_{v \in \text{Values}(a)} \frac{|T_v|}{|T|} H(T_v) \)
  - **Entropy**: \( H(T) = -\sum_{i=1}^{k} p_i \log p_i \)
- **Rule-based**: Classifiers that use a set of if-then rules for decision making.
- **Nearest-neighbor**: A type of instance-based learning where the model predicts the label of a new instance based on the most similar instances in the training set.
  - **Distance Metric**: \( d(x_i, x_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - x_{jk})^2} \)
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem with strong (naive) independence assumptions between the features.
  - **Probability**: \( P(C_k | x) = \frac{P(C_k) P(x | C_k)}{P(x)} \)
- **SVMs (Support Vector Machines)**: Supervised learning models that analyze data for classification and regression analysis, using a hyperplane to separate different classes.
  - **Decision Function**: \( f(x) = w^T x + b \)
  - **Hinge Loss**: \( L(y, f(x)) = \max(0, 1 - y f(x)) \)
- **Neural Networks & Deep NN**: Computational models inspired by the human brain, composed of layers of interconnected nodes, used for complex pattern recognition.
  - **Perceptron Activation**: \( \hat{y} = f(W \cdot x + b) \)
  - **Sigmoid Function**: \( \sigma(x) = \frac{1}{1 + e^{-x}} \)

## Ensemble Classifiers
- **Boosting**: A method that combines the output of several weak classifiers to create a strong classifier.
  - **AdaBoost Update Rule**: \( w_{i+1} = w_i \cdot e^{- \alpha_i y_i h_i(x_i)} \)
  - **Weight Update**: \( \alpha_t = \frac{1}{2} \ln \left( \frac{1 - e_t}{e_t} \right) \)
- **Bagging**: Short for Bootstrap Aggregating, it improves the stability and accuracy of machine learning algorithms by combining the predictions of multiple models.
  - **Bootstrap Sample**: \( X^* = \{ x_i^* \} \)
- **Random Forests**: An ensemble method that uses multiple decision trees to make a prediction, often improving accuracy and controlling overfitting.
  - **Majority Voting**: \( \hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_n) \)

## Keywords for Classification
- **Test condition**: A condition used to split data in decision trees.
- **Splitting**: The process of dividing data into subsets based on certain conditions or features.
- **Node impurity**: A measure of the homogeneity of a node in a decision tree; lower impurity means more homogeneous nodes.
- **Entropy**: A measure of randomness or disorder, used to calculate information gain in decision trees.
- **Information gain**: The reduction in entropy or impurity after a dataset is split on an attribute.
- **Gain ratio**: A modification of information gain that reduces its bias towards multi-valued attributes.

# Model Overfitting

## Classification Errors
- **Decision tree leaf nodes**: The final nodes of a decision tree that contain the class label.
- **Overfitting & underfitting**: Overfitting occurs when a model learns the training data too well, including noise, while underfitting happens when a model is too simple to capture the underlying pattern in the data.

## Model Selection
- **Using validation set**: A subset of data used to tune model parameters and prevent overfitting.
- **Incorporating model complexity**: Taking into account the complexity of the model when evaluating its performance to avoid overfitting.
- **Pre-pruning (early stopping rule)**: Stopping the growth of a decision tree early to prevent overfitting.
- **Post-pruning**: Removing branches from a fully grown tree to reduce its complexity and improve generalization.

## Regularization Techniques
- **L1 Regularization (Lasso)**:
  - **Loss Function**: \( \text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y_i})^2 + \lambda \sum_{j=1}^{p} |w_j| \)
- **L2 Regularization (Ridge)**:
  - **Loss Function**: \( \text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y_i})^2 + \lambda \sum_{j=1}^{p} w_j^2 \)

## Nearest Neighbor
- **Cross-validation**: A technique for assessing how the results of a model will generalize to an independent dataset.
  - **k-Fold Cross-Validation**: \( \text{CV}_{k} = \frac{1}{k} \sum_{i=1}^{k} \text{error}(M_i) \)

## Metrics for Performance Evaluation
- **Accuracy**: The ratio of correctly predicted instances to the total instances.
  - **Formula**: \( \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \)
- **Precision**: The ratio of true positive predictions to the total positive predictions.
  - **Formula**: \( \text{Precision} = \frac{TP}{TP + FP} \)
- **Recall**: The ratio of true positive predictions to the actual positives.
  - **Formula**: \( \text{Recall} = \frac{TP}{TP + FN} \)
- **ROC (Receiver Operating Characteristic)**: A graphical plot that illustrates the diagnostic ability of a binary classifier system.
  - **ROC Space**: Plot of \( \text{TPR} = \frac{TP}{TP + FN} \) vs \( \text{FPR} = \frac{FP}{FP + TN} \)
- **Error rate**: The ratio of incorrect predictions to the total predictions.
  - **Formula**: \( \text{Error Rate} = \frac{FP + FN}{TP + TN + FP + FN} \)
- **Specificity**: The ratio of true negative predictions to the actual negatives.
- **FP rate (False Positive rate)**: The ratio of false positives to the total actual negatives.
- **FN rate (False Negative rate)**: The ratio of false negatives to the total actual positives.
- **Power**: The probability that the test correctly rejects a false null hypothesis.

# Association Rule Mining

- **Frequent itemsets**: Sets of items that appear together frequently in a dataset.
- **Support count**: The frequency of occurrence of an itemset in a dataset.
- **Support**: The proportion of transactions in the dataset that contain the itemset.
  - **Formula**: \( \text{Support}(A) = \frac{| \{ T | A \subseteq T \} |}{| T |} \)
- **Confidence**: The likelihood of occurrence of an itemset given another itemset.
  - **Formula**: \( \text{Confidence}(A \Rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} \)
- **Lift**: A measure of the performance of an association rule at predicting the correct outcome compared to a random chance.
  - **Formula**: \( \text{Lift}(A \Rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A) \times \text{Support}(B)} \)
- **Apriori principle**: A rule that states that all non-empty subsets of a frequent itemset must also be frequent.
- **Apriori Algorithm**:
  - **Candidate Generation**: \( C_{k+1} = \{ X \cup Y \mid X, Y \in L_k, |X \cap Y| = k - 1 \} \)
  - **Pruning**: Removing candidates with infrequent subsets.
- **FP-Growth**: A frequent pattern mining algorithm that uses a compact data structure called an FP-tree.
- **FP tree**: A tree structure that stores frequent itemsets.
- **Prefix path**: A path in an FP tree that represents the prefix of an itemset.

# Cluster Analysis

## Partitional Clustering
- **K-means**: A clustering algorithm that partitions the data into K clusters, each represented by the mean of the points in the cluster.
  - **Centroid Update**: \( \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i \)
  - **Objective Function**: \( \sum_{k=1}^{K} \sum_{x_i \in C_k} || x_i - \mu_k ||^2 \)
- **Elbow Method**: A method to determine the optimal number of clusters by plotting the SSE for different values of \( k \) and selecting the \( k \) at the "elbow" point.

## Hierarchical Clustering
- **Agglomerative**: A bottom-up approach where each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
  - **Single Linkage**: \( \text{d}(C_i, C_j) = \min_{x \in C_i, y \in C_j} ||x - y|| \)
  - **Complete Linkage**: \( \text{d}(C_i, C_j) = \max_{x \in C_i, y \in C_j} ||x - y|| \)
- **Divisive**: A top-down approach where all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.
- **Cluster similarity**: A measure of how similar or dissimilar clusters are.
- **Dendrogram**: A tree-like diagram that records the sequences of merges or splits in hierarchical clustering.

## DBSCAN
- **Density-based**: A clustering method based on the density of data points.
- **Density Reachability**: A point \( p \) is directly density-reachable from \( q \) if \( p \) is within \( \epsilon \)-distance of \( q \) and \( q \) has at least MinPts within \( \epsilon \)-distance.
- **Points (border, noise, core)**: Points in DBSCAN are classified as core points, border points, or noise points based on their neighborhood density.
- **Euclidean distance**: A measure of the true straight line distance between two points in Euclidean space.

## Cluster evaluation
- **Cohesion**: A measure of how closely related the items in a cluster are.
- **Separation**: A measure of how distinct or well-separated a cluster is from other clusters.

# Map Reduce

- **Apache Spark**: An open-source unified analytics engine for big data processing, with built-in modules for streaming, SQL, machine learning, and graph processing.
- **TensorFlow**: An open-source machine learning framework developed by Google.
- **LSH (Locality-sensitive hashing)**: An algorithmic method of performing probabilistic dimension reduction of high-dimensional data.

## Hash Functions
- **Jaccard similarity**: A measure of the similarity between two sets, defined as the size of the intersection divided by the size of the union.
  - **Formula**: \( \text{Jaccard similarity}(A, B) = \frac{|A \cap B|}{|A \cup B|} \)
- **Hashing**: The process of converting an input into a fixed-size string of bytes, typically for faster data retrieval.
- **Similar items**: Items that have high similarity in a dataset.

## Shingling
- **Min-hashing**: A technique used to quickly estimate the similarity of two sets.
  - **Signatures**: A compact representation of sets used in min-hashing.
  - **Locality-sensitive hashing**: A method for approximate nearest neighbor search in high-dimensional spaces.

- **Candidate pairs**: Pairs of items that are likely to be similar, identified using locality-sensitive hashing.

## Distance Measures
- **Jaccard distance**: The complement of the Jaccard similarity, measuring dissimilarity between sets.
  - **Formula**: \( \text{Jaccard distance}(A, B) = 1 - \text{Jaccard similarity}(A, B) \)
- **Bands**: Grouping of hash functions used in locality-sensitive hashing.
- **Hashing Bands**: The process of grouping hash values into bands.
- **S-curve**: A function used to describe the probability of collision in locality-sensitive hashing.

# Data Streams

- **Queries (ad hoc, standing)**: Ad hoc queries are made for a specific task and then discarded, while standing queries are continuously executed over a data stream.
- **Sampling data**: The process of selecting a subset of data from a larger dataset for analysis.
- **Sliding windows**: A method for managing and analyzing data streams where only the most recent data is kept.
- **Filtering**: The process of removing unwanted elements from data streams.
- **Counting distinct elements**: Keeping track of the number of distinct elements in a data stream.
- **Estimating/finding frequent moments**: Identifying frequently occurring items or patterns in data streams.

## Sliding windows
- **Bloom filters**: A space-efficient probabilistic data structure used to test whether an element is a member of a set.
  - **Formula**: \( \text{P(False Positive)} \approx (1 - e^{-kn/m})^k \)
- **Flajolet-Martin algorithm**: An algorithm used for counting the number of distinct elements in a data stream.
- **Count-Min Sketch**:
  - **Frequency Estimation**: \( \hat{f}(i) = \min_{j=1}^{d} C[j, h_j(i)] \)

## Link Analysis
- **Directed / non-directed graphs**: Graphs where edges have a direction or are bidirectional.
- **PageRank**: An algorithm used by Google Search to rank web pages in their search engine results.
  - **PageRank Formula**: \( PR(A) = (1-d) + d \left( \sum_{i=1}^{n} \frac{PR(T_i)}{C(T_i)} \right) \)
    - Where \( d \) is the damping factor, \( PR(T_i) \) is the PageRank of linking page \( T_i \), and \( C(T_i) \) is the number of links on page \( T_i \).
- **Random walk**: A mathematical process that describes a path consisting of a succession of random steps.

- **Web problems (dead-ends, spider traps, etc.)**: Issues in web navigation that affect the efficiency of web crawlers and search engines.

# Page Rank Problems

- **Measures generic popularity of a page**: PageRank assesses a page's importance based on the number and quality of links to it.
- **Susceptible to link spam**: PageRank can be manipulated through artificial link schemes.
- **Uses a single measure of importance**: PageRank provides a single value to denote a page's importance.

## HITS (Hyperlink-Induced Topic Search)
- **Hubs and Authorities**: A link analysis algorithm that rates Web pages, focusing on the relationships between hubs and authorities.
  - **Hub Score**: \( h(i) = \sum_{j: (i,j) \in E} a(j) \)
  - **Authority Score**: \( a(i) = \sum_{j: (j,i) \in E} h(j) \)

## PCY Algorithm (Park-Chen-Yu)
- **Buckets**: Hash buckets used to store item pairs.
- **Multistage algorithm**: An algorithm that uses multiple passes to reduce the size of candidate sets.
- **Multihash algorithm**: An algorithm that uses multiple hash functions to reduce the chance of collisions.
- **SON algorithm (Savasere, Omiecinski, and Navathe)**: An algorithm for mining frequent itemsets in large datasets.

# Distance Metrics

- **L1 Norm (Manhattan distance)**: The sum of the absolute differences between the coordinates of two points.
  - **Formula**: \( ||x||_1 = \sum_{i=1}^{n} |x_i| \)
- **L2 Norm (Euclidean distance)**: The straight-line distance between two points in Euclidean space.
  - **Formula**: \( ||x||_2 = \left( \sum_{i=1}^{n} x_i^2 \right)^{1/2} \)
- **L∞ Norm (Maximum norm)**: The maximum absolute difference between the coordinates of two points.
  - **Formula**: \( ||x||_\infty = \max_i |x_i| \)
- **Mahalanobis Distance**: A measure of the distance between a point and a distribution.
  - **Formula**: \( D_M(x, y) = \sqrt{(x - y)^T S^{-1} (x - y)} \)

## Unsupervised Measures
- **Cohesion**: A measure of how closely related the items in a cluster are.
- **Separation**: A measure of how distinct or well-separated a cluster is from other clusters.
- **Silhouette coefficient**: A measure of how similar an object is to its own cluster compared to other clusters.
  - **Formula**: \( s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \)
    - Where \( a(i) \) is the average distance from the \( i \)-th point to the other points in the same cluster, and \( b(i) \) is the minimum average distance from the \( i \)-th point to points in a different cluster.

## k-means
- **Centroids**: The center of a cluster in K-means.

## BFR (BIRCH)
- **Discard set**: A set of data points discarded as they are outliers or do not fit any cluster well.
- **Compression set**: A set of data points compressed into a few representative points for clustering.
- **Retained set**: A set of data points that are retained for further processing.

# Online Algorithms

## Bipartite Matching
- **Greedy algorithm**: An algorithm that makes the locally optimal choice at each stage with the hope of finding a global optimum.
- **Competitive ratio**: The ratio between the performance of an online algorithm and an optimal offline algorithm.

## Web Advertising
- **AdWords**: An online advertising service developed by Google.
  - **AdWords Auction**: \( \text{CPC}_{\text{actual}} = \frac{\text{Ad Rank}_{\text{below}}}{\text{Quality Score}} + \$0.01 \)
- **Complications (budget, LTR)**: Issues in web advertising related to budget constraints and long-term return.
- **Balance algorithm**: An algorithm to balance different factors in web advertising.
- **Generalized Balance**: An extension of the balance algorithm for more complex scenarios.

## Recommendation Systems
- **Content-based recommendation**: A recommendation system that suggests items similar to those a user liked in the past.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure used to evaluate the importance of a word in a document.
- **Collaborative Filtering**: A method of making recommendations based on the preferences of similar users.
- **User-based collaborative filtering**: A type of collaborative filtering that makes recommendations based on the preferences of similar users.

- **Pearson correlation**: A measure of the linear correlation between two variables.
  - **Formula**: \( r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}} \)
- **Cosine similarity**: A measure of similarity between two vectors by calculating the cosine of the angle between them.
  - **Formula**: \( \text{cosine\_similarity}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||} \)
- **Item-based CF (rating prediction)**: Collaborative filtering based on the similarities between items.

# Dimensionality Reduction

- **UV-decomposition**: A matrix factorization technique used in dimensionality reduction.
- **SVD (Singular Value Decomposition)**: \( A = U \Sigma V^T \)
- **Accuracy of recommendation systems**: The degree to which a recommendation system correctly predicts user preferences.
- **Binary classification**: Classification with two possible outcomes.
- **ROC space**: A graphical representation of the performance of a binary classifier.
- **Precision & recall**: Measures of a model’s accuracy in identifying relevant items.
- **nDCG (normalized Discounted Cumulative Gain)**: A measure of ranking quality.

## Graph Theory
- **Adjacency matrix**: A matrix representing the connections between nodes in a graph.
- **Adjacency list**: A list representation of the connections between nodes in a graph.
- **Directed/undirected graphs**: Graphs where edges have a direction or are bidirectional.
- **Weighted/unweighted graphs**: Graphs where edges have weights or all edges are equal.

- **Graph walks**: Paths that traverse the vertices and edges of a graph.
- **Random walk**: A mathematical process that describes a path consisting of a succession of random steps.
- **Shortest path**: The path between two nodes that has the smallest total weight.

- **Community detection**: The process of finding groups of nodes in a graph that are more densely connected internally than with the rest of the graph.

# Supervised Learning

## Linear Model
- **Linear Regression**: \( y = w^T x + b \)
- **Perceptrons**: The simplest type of artificial neural network used for binary classification tasks.
- **Perceptron Update Rule**: \( w \leftarrow w + \Delta w \)
- **Winnow algorithm**: A machine learning algorithm that is used for learning linear functions.
- **Perceptron convergence**: The property that the perceptron learning algorithm will converge if the data is linearly separable.
- **SVM Objective**: \( \min_w \frac{1}{2} ||w||^2 + C \sum \max(0, 1 - y_i (w \cdot x_i + b)) \)
- **Margin**: The distance between the decision boundary and the closest data points from either class.
- **Hinge function**: The loss function used in SVMs that penalizes misclassified points and those within the margin.

# Deep Feedforward Networks

## Deep Learning
- **Neural networks**: A set of algorithms modeled loosely after the human brain that are designed to recognize patterns.
- **Input/hidden/output layer**: The layers in a neural network where input data is processed, transformations are applied, and output is produced.
- **Random/pooled/convolutional layer**: Types of layers in a convolutional neural network used for feature extraction and down-sampling.

- **Activation Functions**:
  - **ReLU (Rectified Linear Unit)**: \( f(x) = \max(0, x) \)
  - **Sigmoid Function**: \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
  - **Softmax**: \( \sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}} \)
- **Loss functions**: Functions that measure how well the model's predictions match the true data.
  - **Regression loss**: Loss function used for regression tasks.
  - **Classification loss**: Loss function used for classification tasks.
- **Back propagation**: An algorithm for training neural networks by updating weights to minimize the loss function.
  - **Back Propagation**: \( \Delta w_{ij} = -\eta \frac{\partial E}{\partial w_{ij}} \)
    - Where \( E \) is the error, \( w_{ij} \) is the weight between nodes \( i \) and \( j \), and \( \eta \) is the learning rate.
- **Gradient Descent**:
  - **Update Rule**: \( w \leftarrow w - \eta \nabla_w L(w) \)
    - Where \( \eta \) is the learning rate and \( L(w) \) is the loss function.

# Graph Clustering and Analysis

- **Node degree and Reachability**: Measures of the number of connections a node has and how far it can reach other nodes in the graph.
- **Node similarity (k-core relevance, Affinity, closeness)**: Measures of similarity between nodes based on their connections.
- **Node Proximity**: A measure of how close or far apart nodes are in a graph.

## Hierarchical Clustering
- **Betweenness**: A measure of centrality in a graph based on shortest paths.
  - **Formula**: \( C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}} \)
    - Where \( \sigma_{st} \) is the total number of shortest paths from node \( s \) to node \( t \) and \( \sigma_{st}(v) \) is the number of those paths that pass through \( v \).
- **Girvan-Newman Algorithm**: An algorithm for detecting communities in complex systems by progressively removing edges.

## Spectral Clustering
- **Partitioning & Bipartitioning**: Methods for dividing a graph into clusters or two parts.
- **Graph cuts**: A method of clustering that involves cutting the graph into smaller components.
- **Laplacian matrix**: A matrix representation of a graph that is useful in spectral clustering.
  - **Formula**: \( L = D - A \)
    - Where \( D \) is the degree matrix and \( A \) is the adjacency matrix.
- **Eigenvectors and eigenvalues**: Values and vectors that are used to analyze the structure of graphs in spectral clustering.
  - **Eigen Decomposition**: \( L \vec{v} = \lambda \vec{v} \)

# PLA (Perceptron Learning Algorithm)

- **Variance**: A measure of the spread between numbers in a data set.
  - **Formula**: \( \text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 \)
- **Covariance**: A measure of the degree to which two variables change together.
  - **Formula**: \( \text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_X)(y_i - \mu_Y) \)
- **Weight Update Rule**: \( w \leftarrow w + \eta (y_i - \hat{y_i}) x_i \)
  - Where \( \eta \) is the learning rate, \( y_i \) is the true label, and \( \hat{y_i} \) is the predicted label.
