# TF-IDF and PPMI Code Template

"""
This template will help you implement TF-IDF and PPMI calculations using the NLTK library and the Brown corpus.
You will preprocess the corpus, compute term frequencies, document frequencies, TF-IDF scores, and create
a word co-occurrence matrix to compute Positive Pointwise Mutual Information (PPMI) scores.
"""

import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import math
import numpy as np

# Download the Brown corpus if not already downloaded
nltk.download("brown")


# Preprocess the corpus: Tokenize, lowercase, and add start/end tokens
def preprocess(corpus):
    """
    Preprocess the corpus by tokenizing, converting to lowercase, and adding <s> and </s> tokens.

    Args:
        corpus (list): List of sentences from the corpus.

    Returns:
        list: Preprocessed and tokenized corpus.
    """
    tokenized_corpus = []
    for sentence in corpus:
        tokenized_sentence = ["<s>"] + [word.lower() for word in sentence] + ["</s>"]
        tokenized_corpus.append(tokenized_sentence)
    return tokenized_corpus


# Calculate Term Frequency (TF)
def compute_tf(corpus):
    """
    Calculate the term frequency for each word in each document.

    Args:
        corpus (list): Preprocessed corpus where each document is a list of words.

    Returns:
        dict: Term frequencies for each document.
    """
    tf = defaultdict(Counter)
    # TODO: For each document, count the occurrences of each word
    # HINT: Use enumerate to get document index and Counter to count words
    # pass  # Remove this line after implementing
    for i, document in enumerate(corpus):
        tf[i] = Counter(document)
    return tf


# Calculate Document Frequency (DF)
def compute_df(tf):
    """
    Calculate the document frequency for each word across all documents.

    Args:
        tf (dict): Term frequencies for each document.

    Returns:
        Counter: Document frequencies for each word.
    """
    df = Counter()
    # TODO: For each word, count the number of documents it appears in
    # HINT: Use a set of words for each document to avoid counting duplicates
    # pass  # Remove this line after implementing
    for doc in tf.values():
        unique_words = set(doc.keys())
        for word in unique_words:
            df[word] += 1
    return df


# Calculate TF-IDF for each word
def compute_tfidf(tf, df, num_docs):
    """
    Calculate the TF-IDF score for each word in each document.

    Args:
        tf (dict): Term frequencies for each document.
        df (Counter): Document frequencies for each word.
        num_docs (int): Total number of documents.

    Returns:
        dict: TF-IDF scores for each word in each document.
    """
    tfidf = defaultdict(dict)
    # TODO: For each document and word, calculate TF-IDF score
    # TF-IDF formula: TF(word) * log(N / (1 + DF(word)))
    # HINT: Use math.log() for logarithm
    # pass  # Remove this line after implementing
    for docId, termFreq in tf.items():
        for word, freq in termFreq.items():
            idf = math.log(num_docs / (1 + df[word]))  # IDF calculation
            tfidf[docId][word] = freq * idf  # TF-IDF score
    return tfidf


# Create a word co-occurrence matrix
def create_cooccurrence_matrix(corpus, window_size=5):
    """
    Create a word co-occurrence matrix from the corpus.

    Args:
        corpus (list): Preprocessed corpus where each document is a list of words.
        window_size (int): The size of the context window.

    Returns:
        tuple: Co-occurrence matrix, word to index mapping, and index to word mapping.
    """
    # TODO: Build the vocabulary of unique words
    # HINT: Use a set to store unique words
    # pass  # Remove this line after implementing
    vocab = set()
    for document in corpus:
        vocab.update(document)
    vocabSize = len(vocab)
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    # TODO: Initialize co-occurrence matrix
    # HINT: Use numpy to create a zero matrix of size vocabSize x vocabSize
    # pass  # Remove this line after implementing
    cooccurrence_matrix = np.zeros((vocabSize, vocabSize))

    # TODO: Fill in the co-occurrence matrix
    # HINT: For each word, consider a window of words around it
    # pass  # Remove this line after implementing
    for document in corpus:
        
        for idx, word in enumerate(document):
        
            wordId=word_to_id[word]
            start= max(0, idx-window_size)
            end =min(len(document), idx+window_size+1)
            
            for neighbor in document[start:end]:
        
                if neighbor!= word:
                    neighborId = word_to_id[neighbor]
                    cooccurrence_matrix[wordId, neighborId] += 1

    return cooccurrence_matrix, word_to_id, id_to_word


# Calculate PPMI from co-occurrence matrix
def compute_ppmi(cooccurrence_matrix):
    """
    Compute the Positive Pointwise Mutual Information (PPMI) matrix from the co-occurrence matrix.

    Args:
        cooccurrence_matrix (numpy.ndarray): Co-occurrence matrix of word counts.

    Returns:
        numpy.ndarray: PPMI matrix.
    """
    # TODO: Calculate total sum of all co-occurrences
    # HINT: Use numpy.sum()
    # pass  # Remove this line after implementing
    total_sum = np.sum(cooccurrence_matrix)

    # TODO: Calculate sum over rows (word occurrence counts)
    # HINT: Use numpy.sum() with axis=1
    # pass  # Remove this line after implementing
    sum_over_columns = np.sum(cooccurrence_matrix, axis=0)
    sum_over_rows = np.sum(cooccurrence_matrix, axis=1)

    # Initialize PPMI matrix with zeros
    ppmi_matrix = np.zeros(cooccurrence_matrix.shape)

    # TODO: Compute PPMI for each cell in the matrix
    # HINT: Use nested loops to iterate over the matrix indices
    # Remember to check if pij > 0 before computing PMI
    # pass  # Remove this line after implementing
    for i in range(cooccurrence_matrix.shape[0]):
        for j in range(cooccurrence_matrix.shape[1]):
            if cooccurrence_matrix[i, j] > 0:
                p_ij = cooccurrence_matrix[i, j] / total_sum
                p_i = sum_over_rows[i] / total_sum
                p_j = sum_over_columns[j] / total_sum
                ppmi = max(math.log(p_ij / (p_i * p_j)), 0)  # PMI calculation
                ppmi_matrix[i, j] = ppmi

    return ppmi_matrix


# Main execution
if __name__ == "__main__":
    # Load the Brown corpus as sentences
    corpus = brown.sents()[0:1000]  # Use first 1000 sentences

    # Preprocess the corpus
    processed_corpus = preprocess(corpus)

    # Number of documents in the corpus
    num_docs = len(processed_corpus)

    # Step 1: Calculate TF-IDF
    tf = compute_tf(processed_corpus)
    df = compute_df(tf)
    tfidf = compute_tfidf(tf, df, num_docs)

    # Output TF-IDF for a few words in the first document
    print("TF-IDF for word 'county' in the first document: ", tfidf[0].get("county", 0))
    print(
        "TF-IDF for word 'investigation' in the first document: ",
        tfidf[0].get("investigation", 0),
    )
    print(
        "TF-IDF for word 'produced' in the first document: ",
        tfidf[0].get("produced", 0),
    )

    # Step 2: Calculate PPMI
    window_size = 2  # You can change the window size
    cooccurrence_matrix, word_to_id, id_to_word = create_cooccurrence_matrix(
        processed_corpus, window_size=window_size
    )
    ppmi_matrix = compute_ppmi(cooccurrence_matrix)

    # Output PPMI for a few word pairs
    print("\nPPMI for a few word pairs:")
    words = [["expected", "approve"], ["mentally", "in"], ["send", "bed"]]
    for word_pair in words:
        word1, word2 = word_pair
        word1_id = word_to_id.get(word1, None)
        word2_id = word_to_id.get(word2, None)
        if word1_id is not None and word2_id is not None:
            ppmi_value = ppmi_matrix[word1_id, word2_id]
            print(f"PPMI({word1}, {word2}) = {ppmi_value}")
        else:
            print(f"Words '{word1}' or '{word2}' not found in vocabulary.")


"""
Answer the following questions based on the outputs of you program:

TF-IDF for word 'county' the first document:  
[YOUR ANSWER]
TF-IDF for word 'investigation' the first document:  
[YOUR ANSWER]
TF-IDF for word 'produced' the first doccountyument: 
[YOUR ANSWER]

PPMI for a few word pairs:
PPMI(expected, approve) = 
[YOUR ANSWER]
PPMI(mentally, in) = 
[YOUR ANSWER]
PPMI(send, bed) = 
[YOUR ANSWER]
"""
