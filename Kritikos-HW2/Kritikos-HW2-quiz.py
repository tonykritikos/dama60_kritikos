print('------Q1------')
v1 = [0, 1, 1, 0, 0, 0, 1]
v2 = [1, 0, 1, 0, 1, 0, 0]

intersection_size = len(set(i for i in range(len(v1)) if v1[i] == 1 and v2[i] == 1))
union_size = len(set(i for i in range(len(v1)) if v1[i] == 1 or v2[i] == 1))

jaccard_distance = intersection_size / union_size

print("Jaccard distance:", jaccard_distance)

print('------Q2------')
def map_function_q2(R, S):
    intermediate_pairs = []

    for element in R:
        if element not in S:
            intermediate_pairs.append((element, 'R'))

    for element in S:
        intermediate_pairs.append((element, 'S'))

    return intermediate_pairs

# Example sets
R = {'a', 'b', 'c', 'd'}
S = {'b', 'e', 'f'}

# Generate intermediate key-value pairs
result = map_function_q2(R, S)

print("Intermediate key-value pairs:", result)


print('------Q3------')
def map_function_q3(matrix_name, matrix):
    intermediate_pairs = []

    for row in matrix[1:]:
        key = row[1]  # Assuming the join attribute is in the second column
        value = (matrix_name, row[0])  # Pairing matrix name with the value

        intermediate_pairs.append((key, value))

    return intermediate_pairs

# Matrices R and S
R = [
    ['A', 'B'],
    ['a1', 'b1'],
    ['a2', 'b1'],
    ['a3', 'b2'],
    ['a4', 'b3'],
    ['a5', 'b4']
]

S = [
    ['C', 'B'],
    ['c1', 'b2'],
    ['c2', 'b2'],
    ['c3', 'b3'],
    ['c4', 'b4']
]

# Map function for matrices R and S
map_result_R = map_function_q3('R', R)
map_result_S = map_function_q3('S', S)

# Combine the results
map_result = map_result_R + map_result_S

print("Intermediate key-value pairs:")
for pair in map_result:
    print(pair)

print('------Q4------')

def map_function_q4(text):
    words = text.split()
    intermediate_pairs = [(word, 1) for word in words]
    return intermediate_pairs

def reduce_function_q4(intermediate_pairs):
    result_dict = {}
    for key, value in intermediate_pairs:
        if key in result_dict:
            result_dict[key] += value
        else:
            result_dict[key] = value
    return result_dict.items()

# Given text
text = "I dream that I will have a dream tomorrow"

# Map phase
map_result = map_function_q4(text)

# Reduce phase
reduce_result = reduce_function_q4(map_result)

# Print the results
for key, value in reduce_result:
    print(f"Reduce('{key}', {value}) = {value}")

print('------Q5------')
def jaccard_similarity_q5(list1, list2):
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1).union(set(list2)))
    return intersection / union

# Original documents
D1 = [1, 0, 1, 0]
D2 = [0, 1, 1, 1]

# Signatures after minhashing
sign1 = [4, 3, 1, 2]
sign2 = [1, 3, 2, 4]

# Calculate Jaccard similarity
jaccard_doc = jaccard_similarity_q5(D1, D2)
jaccard_sign = jaccard_similarity_q5(sign1, sign2)

# Print the corrected results
print(f"Corrected Jaccard(D1, D2) = {jaccard_doc}")
print(f"Corrected Jaccard(sign1, sign2) = {jaccard_sign}")

print('------Q6------')

def moving_average(stream, window_size):
    averages = []
    current_window_sum = 0

    for i in range(len(stream)):
        current_window_sum += stream[i]

        if i >= window_size - 1:
            if i >= window_size:
                current_window_sum -= stream[i - window_size]
            averages.append(current_window_sum / window_size)

    return averages

def calculate_overall_average(averages):
    return sum(averages) / len(averages)

# Given stream and window size
A = [9, 2, 7, 4, 5]
N = 3

# Calculate moving averages
averages = moving_average(A, N)

# Calculate overall average
overall_average = calculate_overall_average(averages)

# Print the results
print("Moving Averages:", averages)
print("Overall Average:", round(overall_average, 2))

print('------Q7------')

def calculate_probability_q7(m, n):
    return 1 - ((1 - 1/n) ** m)

# Given values
m = 5
n = 2

# Calculate probability
probability = calculate_probability_q7(m, n)

# Print the result
print("Probability:", round(probability, 2))


print('------Q8------')


def set_bits_in_bloom_filter(x, N):
    h1 = x % N
    h2 = (3 * x + 2) % N
    return h1, h2

# Given values
x = 7
N = 8

# Calculate positions of bits to be set to 1
positions = set_bits_in_bloom_filter(x, N)

# Print the result
print("Bits to be set to 1:", positions)


print('------Q9------')

from collections import defaultdict

# Given matrix R
matrix_R = [
    ['a1', 'b1', 'c1'],
    ['a1', 'b2', 'c2'],
    ['a1', 'b3', 'c3'],
    ['a2', 'b4', 'c4'],
    ['a2', 'b5', 'c5'],
]

# Initialize a defaultdict to store the sum of values for each key
sum_values = defaultdict(int)

# Calculate the sum of values for each key
for row in matrix_R:
    key = row[0]
    value = row[1]
    # Assuming the values in column B are numeric, convert to int
    sum_values[key] += int(value[1:])

# Generate the representation γA, SUM(B)(R)
result_representation = [[key, sum_values[key]] for key in sum_values]

# Print the result
print("Result Representation:", result_representation)



print('------Q10------')

# Given values
similarity = 0.75
rows_per_band = 6
num_bands = 10

# Calculate the probability of collision
collision_probability = 1 - (1 - similarity ** rows_per_band) ** num_bands

# Print the result
print("Probability of Collision:", round(collision_probability, 2))
