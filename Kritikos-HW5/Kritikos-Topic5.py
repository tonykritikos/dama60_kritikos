#############################################
#
# Implementation of a logistic regression 
# classifier updating its parameters using
# gradient descent method
#
# hou/23-24/dama60/hw5/Topic 5
#
#############################################

#Required libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# read the data and print some basic information
df = pd.read_csv('Dataset.csv')
print('Typical statistical information of the used data: \n\n ', df.describe(), sep='')

# keep the first 2 numeric columns as the feature space of the input
# and use the last one as the target variable (discrete)
X = df.loc[:, ['dim1','dim2']]
y = df.loc[:, 'label']

###############################################
# Topic 5a
###############################################

# count the instances of each class
count_class_0, count_class_1 = 0, 0

for i in y:

    if i == 0:
        count_class_0 += 1
    else:
        count_class_1 += 1

print('\n\nNumber of instances for class0: {:3} and class1: {:3}'.format(count_class_0, count_class_1))

# verify that the sum of the values of the above 2 counters 
# equals to the total number of dataset's instances 
assert count_class_0 + count_class_1 == X.shape[0]

###############################################
# Topic 5b
###############################################

# split original dataset to training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True,
                                                          stratify=y, random_state=2024) 

# scale our data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# initialize the needed parameters for the Logistic Regression Classifier
lr = 0.03                               # define the learning rate
n_iter = 100                            # define the number of iterations
m = X_train.shape[0]                    # the number of rows of the training data
n = X_test.shape[0]                     # the number of rows of the test data
number_of_features = X_train.shape[1]   # the number of columns of the training or test data

# keep the weights of the Logistic Regression Classifier in a dictionary 
network_params = {}                     

# initialize the input layer coefficients as zeros (vector with dimension (number of features,))
network_params["W"] = np.zeros(number_of_features)

# initialize the bias as 0 (scalar)
network_params["b"] = 0                       

print('\n\nInitial parameters: \nW = ', network_params['W'], '\nb = ', network_params['b'],'\n\n')


losses = []

for iteration in range(n_iter): 

    # feed input X_train into the network to calculate the log-odds
    logits = np.dot(X_train, network_params['W']) + network_params['b']
    
    # apply the sigmoid function for transforming logits to probabilities
    prob = 1 / (1 + np.exp(-logits))
    
    # calculate the sum of loss for all instances based on the binary cross entropy function
    loss = np.sum(y_train * np.log(prob) + (1 - y_train) * np.log(1 - prob))

    # average the loss with the number of training instances
    loss = -1/m * loss  

    # add the loss of each iteration into a list
    losses.append(loss)

    # partial derivative of loss function with respect to weights
    dloss_dW = 1 / m * np.dot(X_train.T, (prob - y_train))

    # partial derivative of loss function with respect to bias 
    dloss_db = 1/m * np.sum(prob - y_train)
    
    # update rules based on Gradient Descend approach
    network_params['W'] -= lr * dloss_dW
    network_params['b'] -= lr * dloss_db 
    
    # starting from zero, we repeat the next process per 25 iterations;
    # we print the value of the loss function at the current iteration
    if iteration % 25 == 0:
        print('Iteration {:3} Loss function ={:6}'.\
                                                      format(iteration, losses[iteration]))


###############################################
# Topic 5c
###############################################

print('\n\nOptimized parameters: \nW = ', np.round(network_params['W'],3),\
                                  '\nb = ', np.round(network_params['b'],3))

# using the optimized parameters of the trained network
# infer that model over the unseen test subset
logits_test_set = np.dot(X_test, network_params["W"]) + network_params["b"]

# apply the sigmoid activation function on logits for transforming them to probabilities
prob_test_set = 1 / (1 + np.exp(-logits_test_set))

# measure the classification accuracy on the test subset (this should be a number in range [0,1])
y_pred = prob_test_set >= 0.5
y_pred = y_pred.astype("int")
print('\n\nClassification Accuracy (test subset) ={:6}'.\
      format(np.round(np.count_nonzero(y_pred == y_test) / n, 3)))