import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Read input data from file "mountains.csv" into a pandas dataframe
dataframe = pd.read_csv("mountains.csv", sep=";")

# Extract coordinate x (longitude) and y (latitude) from the dataframe
x = dataframe['longitude']
y = dataframe['latitude']

# Plot initial data as a scatter plot
plt.scatter(x, y)  # Plot the data in x and y as a scatter plot
plt.title('Highest mountain peaks of the Himalayas')  # Add the title
plt.xlabel('Longitude')  # Add the label for the x axis
plt.ylabel('Latitude')  # Add the label for the y axis
plt.show()  # Show the plot

# Using variables x and y create a list of corresponding (x,y)
# tuples to represent 2-D coordinates. The resulting list will
# be provided as input to DBSCAN.
data = list(zip(x, y))

# A total number of 8 different values of parameter eps will be tested.
# All results will be presented together in a set of 8 subplots.
# Subplots should be arranged on a 2x4 grid. The following lists will be
# used to create all values of parameter eps and to iterate over all subplots.
i_list = [i for i in range(0, 2)]
j_list = [j for j in range(0, 4)]

# Create grid of subplots
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('DBSCAN (min_samples=4)')

for i in i_list:
    for j in j_list:
        # Calculate parameter eps (starting with a value of 1.0 and
        # increasing by 0.5 for each iteration).
        e = 1.0 + 0.5 * (i * len(j_list) + j)
        # Set desired parameters of DBSCAN algorithm.
        # Ensure that the same value of parameter MinPts (= 4) is
        # used for each run and that only the value of eps differs.
        dbscan = DBSCAN(eps=e, min_samples=4)
        # Apply algorithm with desired parameters to the input data.
        dbscan.fit(data)
        # Plot clusters of the current iteration as a scatter plot
        # in the corresponding subplot.
        ax[i, j].scatter(x, y, c=dbscan.labels_)
        ax[i, j].set_title('eps=' + str(e))
        # Create a dictionary to count how many times each cluster label appears,
        # i.e., how many points are assigned to each cluster.

for a in ax.flat:
    # Set labels for left and bottom plots.
    a.set(xlabel='Longitude', ylabel='Latitude')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    a.label_outer()

plt.show()  # Show the plot
