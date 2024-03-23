
#############################################
#
#
# Content-based recommendation of movies
# based on their description
#
# hou/23-24/dama60/hw4/Topic 5
#
#############################################

# Required libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter # for sorting the list of tuples



###############################################
# Topic 5a
###############################################


# List of movie titles with their respective
# descriptions. This is the movie dataset where
# description is the content.
# For example, data['title'][0] has as description data['description'][0]
data = {'title': ['Predator','The Thing','The Lobster','Edge of tomorrow','Eyes wide shut','Alien', 'Godzilla minus one'],
        'description': ['action thriller scifi','horror scifi','romance thriller scifi','scifi action','thriller mystery','scifi horror', 'horror scifi'] }



# Instantiate a TF-IDF object and calculate the
# TF-IDF score for each term (word) found in
# all movie descriptions.
tfidfVectorizer = TfidfVectorizer(stop_words='english')
tfidfMatrix = tfidfVectorizer.fit_transform(data['description'])


# Display the TF-IDF score for each word in
# each description in a human friendly way.
print('a) TF-IDF scores')
# Adjust number of decimal digits for floating point
# numbers in data frame.
pd.set_option('display.float_format', '{:.4f}'.format)
print( pd.DataFrame(tfidfMatrix.toarray(), index=data['title'],
                    columns=list(tfidfVectorizer.get_feature_names_out()) ) )



###############################################
# Topic 5b
###############################################


# Calculate cosine similarities for each pair
# of movies.
similarities = cosine_similarity(tfidfMatrix)
# Adjust number of decimal digits
np.set_printoptions(precision=4)
print('\nb) Pairwise output of cosine similarities:\n{}\n'.format(similarities))



###############################################
# Topic 5c
###############################################

# Get recommendations for this movie.
# Case sensitivity matters.
targetMovie = 'The Lobster'

# Get only relevant similarities from the
# calculated similarity matrix as a list
idx = data['title'].index(targetMovie)
simList = list( similarities[:, idx ] )

# Add (movie index, similarity) to the list
# of candidate recommendations
candidateRecommendations = []
for i, s in enumerate(simList):
    # ignore target movie
    if i != idx:
       candidateRecommendations.append( (i, s) )

# Inplace sorting of movies in descending order
# of cosine similarity
candidateRecommendations.sort(key=itemgetter(1), reverse=True)


# Print out recommendations formatted using title,
# similarity in descending order of similarity.

print(f'c) Recommendations for movie "{targetMovie}":')
# Print a header
print('\t{:<20}|{:<10}'.format('Title', 'Similarity') )
print('\t{:<20}|{:<10}'.format(20*'-', 10*'-') )
# Print movie title and similarity
for m in candidateRecommendations:
    movieTitlePos, movieSimilarity = m
    print('\t{:<20}|{:<10.4f}'.format(data['title'][ movieTitlePos ], movieSimilarity ) )


