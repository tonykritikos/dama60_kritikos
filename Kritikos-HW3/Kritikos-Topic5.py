import numpy as np

L = np.array([
    [0, 0, 1, 0, 1],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0]
])

# print the type of variable L
print(f'Type of variable L is {type(L)}')
print('-' * 24)

# print the number of outgoing edges per Node
for node, val in enumerate(L.sum(axis=1)):
    print(f'Node {node} has {val} outgoing edges')
print('-' * 24)

# print the number of incoming edges per Node
for node, val in enumerate(L.sum(axis=0)):
    print(f'Node {node} has {val} incoming edges')
print('-' * 24)

# check if matrix L is symmetric
if (L == L.T).all():
    print('Matrix L is symmetric')
else:
    print('Matrix L is not symmetric')
print('-' * 24)

dim = L.shape

h_input = np.ones((dim[0], 1))  # create a vector of dimension 5x1 with its points being equal to 1

# we define here the number of iterations
n_iter = 5

# apply the HITS algorithm for n_iter iterations
h_history = []
for iteration in range(0, n_iter):
    a = L.T.dot(h_input)  # compute the vector of authorities before scaling
    a = a / max(a)  # apply the scaling process

    h = L.dot(a)  # compute the vector of hubbiness before scaling
    h = h / max(h)  # apply the scaling process

    print(f'iteration: {iteration + 1} a: ', np.round(a, 3))
    print(f'iteration: {iteration + 1} h: ', np.round(h, 3))
    print('-' * 24)

    # keep in a list called h_history the values of vector h of the
    # current iteration rounded to the 3rd decimal
    h_history.append(np.round(h, 3))

    # update the h_input for the next iteration with the current value of h
    h_input = h