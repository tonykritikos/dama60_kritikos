from bloom_filter2 import BloomFilter
import math

max_elements = 50
error_rate = 0.5

all_tokens = []

b = BloomFilter(max_elements=max_elements, error_rate=error_rate)

with open("WA2_5.txt") as my_file:
	for line in my_file:
		line_tokens = line.split()                      # split the line to get tokens
		tokens = [l.upper() for l in line_tokens]       # convert each token to uppercase
		all_tokens.extend(tokens)                       # maintain a list with all tokens seen
		for t in tokens:                                # add each of these tokens to the BF
			b.add(t)

# calculate and print the optimal size of the BF using 
# the aforementioned formula for the existing configuration:
size = max_elements * (-1.44 * math.log(error_rate, 2))
print(size)

# The list of words to check in the BF
words = ["THE", "BE", "TO", "OF", "AND", "A", "IN", "THAT", "HAVE", "I"]

TP = 0
FP = 0
TN = 0

# Check each word in the list of words for membership in the BF 
# and characterize the result accordingly.

for w in words:
	if w in b and w in all_tokens:
		TP += 1
	elif w in b and w not in all_tokens:
		FP += 1
	else:
		TN += 1

print("TP=", TP, "FP=", FP, "TN=", TN)
