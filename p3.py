# Word Embeddings with GloVe Code Template

"""
This template will help you implement word embeddings using pre-trained GloVe vectors.
You will load the GloVe embeddings, compute cosine similarities between words,
and find the most similar words to a given word based on cosine similarity.
"""

import numpy as np
from numpy import dot
from numpy.linalg import norm


def load_glove_model(File):
    """
    Load the GloVe model from a file.

    Args:
        File (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary mapping words to their embedding vectors.
    """
    print("Loading GloVe Model")
    glove_model = {}
    # TODO: Open the GloVe file and read the embeddings
    # HINT:
    # - Each line in the file corresponds to a word and its vector
    # - Split the line into the word and the vector components
    # - Convert the vector components to a NumPy array of floats
    # - Store the word and vector in the glove_model dictionary
    # pass  # Remove this line after implementing
    with open(File, "r", encoding="utf-8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            vector = np.array(split_line[1:], dtype=float)
            glove_model[word] = vector
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def cosine_similarity(word1, word2, glove_vectors):
    """
    Compute the cosine similarity between two words using their GloVe vectors.

    Args:
        word1 (str): First word.
        word2 (str): Second word.
        glove_vectors (dict): Dictionary of GloVe vectors.

    Returns:
        float or None: Cosine similarity between word1 and word2, or None if a word is not found.
    """
    # TODO: Check if both words are in the glove_vectors
    # If they are, compute the cosine similarity between their vectors
    # HINT:
    # - Retrieve the vectors for both words
    # - Use the dot product and the norms of the vectors
    # - Formula: cosine_similarity = (v1 ⋅ v2) / (||v1|| * ||v2||)
    # - Use np.dot() and np.linalg.norm()
    # If a word is not found, return None
    # pass  # Remove this line after implementing
    if word1 not in glove_vectors or word2 not in glove_vectors:
        return None

    vec1 = glove_vectors[word1]
    vec2 = glove_vectors[word2]
    similarity = dot(vec1,vec2)/(norm(vec1)*norm(vec2))
    return similarity


def find_most_similar(word, glove_vectors, top_n=5):
    """
    Find the top-N most similar words to a given word using cosine similarity.

    Args:
        word (str): The word to find similar words for.
        glove_vectors (dict): Dictionary of GloVe vectors.
        top_n (int): Number of most similar words to return.

    Returns:
        list or None: List of tuples (word, similarity) of the top-N most similar words, or None if the word is not found.
    """
    # TODO: Check if the word is in glove_vectors
    # If not, return None
    # HINT:
    # - Retrieve the vector for the given word
    # - Initialize an empty dictionary to store similarities
    # - Loop over all other words and compute the cosine similarity
    # - Exclude the word itself from comparison
    # - Sort the words based on similarity in descending order
    # - Return the top N most similar words and their similarities
    # pass  # Remove this line after implementing
    if word not in glove_vectors:
        return None

    targetVector = glove_vectors[word]
    similarities = {}

    for currWord, vector in glove_vectors.items():
        if currWord == word:
            continue
        similarity = dot(targetVector,vector)/(norm(targetVector)*norm(vector))
        similarities[currWord] = similarity

    # Sort the words by similarity in descending order
    sorted_similar_words = sorted(similarities.items(),key=lambda item: item[1],reverse=True)
    return sorted_similar_words[:top_n]


if __name__ == "__main__":
    # Load GloVe vectors
    glove_vectors = load_glove_model("glove.6B/glove.6B.50d.txt")

    # Compute cosine similarity for the specified word pairs
    pairs = [("cat", "dog"), ("car", "bus"), ("apple", "banana")]
    for word1, word2 in pairs:
        similarity = cosine_similarity(word1, word2, glove_vectors)
        if similarity is not None:
            print(
                f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}"
            )
        else:
            print(f"One of the words '{word1}' or '{word2}' is not in the vocabulary.")

    # Find the top 5 most similar words for specified words
    words_to_check = ["king", "computer", "university"]
    for word in words_to_check:
        similar_words = find_most_similar(word, glove_vectors)
        if similar_words is not None:
            print(f"\nTop 5 most similar words to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity:.4f}")
        else:
            print(f"The word '{word}' is not in the vocabulary.")


"""
Answer the following questions based on the outputs of you program:

cosine similarity between 'cat' and 'dog': 
0.9218
cosine similarity between 'car' and 'bus': 
0.8211
cosine similarity between 'apple' and 'banana': 
0.5608

top 5 most similar words to 'king':
prince: 0.8236
queen: 0.7839
ii: 0.7746
emperor: 0.7736
son: 0.7667

top 5 most similar words to 'computer':
computers: 0.9165
software: 0.8815
technology: 0.8526
electronic: 0.8126
internet: 0.8060

top 5 most similar words to 'university':
college: 0.8745
harvard: 0.8711
yale: 0.8567
graduate: 0.8553
institute: 0.8484
"""
