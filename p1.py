# Bigram Language Model Template

"""
This template will help you build a bigram language model using the NLTK library.
You will preprocess the corpus, build the bigram model, calculate probabilities,
and predict the next words given a sentence prefix.
"""

import nltk
from nltk import bigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from collections import defaultdict, Counter


# Download required NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('brown')

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

# Build the bigram model: Create frequency distributions for unigrams and bigrams
def build_bigram_model(tokenized_corpus):
    """
    Build bigram and unigram frequency distributions.

    Args:
        tokenized_corpus (list): Preprocessed and tokenized corpus.

    Returns:
        tuple: bigram frequencies and unigram frequencies.
    """
    bigram_freq = defaultdict(Counter)
    unigram_freq = Counter()

    for document in tokenized_corpus:
        unigram_freq.update(document)
        # print(bigrams(document))
        for word1, word2 in bigrams(document):
            bigram_freq[word1][word2] += 1  
    return bigram_freq, unigram_freq

# Calculate bigram probability with optional smoothing
def bigram_probability(bigram_freq, unigram_freq, word1, word2, smoothing=False):
    """
    Calculate the probability of word2 given word1 using bigram frequencies.
    If smoothing is True, apply Laplace smoothing.

    Args:
        bigram_freq (dict): Bigram frequency distribution.
        unigram_freq (dict): Unigram frequency distribution.
        word1 (str): The preceding word.
        word2 (str): The current word.
        smoothing (bool): Whether to apply Laplace smoothing.

    Returns:
        float: Probability of word2 given word1.
    """
    #print("word1:" +word1 + " bigram_freq[word1]:" +bigram_freq[word1])

    bigram_count = bigram_freq[word1][word2]  
    
    unigram_count = unigram_freq.get(word1, 0)

    if smoothing:
        V = len(unigram_freq) 
        return (bigram_count + 1) / (unigram_count + V) 
    else:
        if unigram_count == 0:
            return 0.0
        return bigram_count / unigram_count

# Compute the probability of a sentence
def sentence_probability(bigram_freq, unigram_freq, sentence, smoothing=False):
    """
    Compute the probability of a sentence using the bigram model.

    Args:
        bigram_freq (dict): Bigram frequency distribution.
        unigram_freq (dict): Unigram frequency distribution.
        sentence (str): The sentence to compute the probability for.
        smoothing (bool): Whether to apply Laplace smoothing.

    Returns:
        float: Probability of the sentence.
    """
    tokens = ["<s>"] + word_tokenize(sentence.lower()) + ["</s>"]
    
    # Initialize probability to 1.0
    probability = 1.0
    
    # Iterate over the bigrams in the sentence and multiply their probabilities
    for word1, word2 in bigrams(tokens):
        probability *= bigram_probability(bigram_freq, unigram_freq, word1, word2, smoothing=smoothing)
        
    return probability

# Predict the next N words given a sentence prefix
def predict_next_words(bigram_freq, unigram_freq, sentence_prefix, N, smoothing=False):
    """
    Predict the next N words given a sentence prefix using the bigram model.

    Args:
        bigram_freq (dict): Bigram frequency distribution.
        unigram_freq (dict): Unigram frequency distribution.
        sentence_prefix (str): The sentence prefix.
        N (int): Number of words to predict.
        smoothing (bool): Whether to apply Laplace smoothing.

    Returns:
        str: The predicted next N words.
    """
    # TODO: Tokenize and lowercase the sentence prefix
    # HINT: Use word_tokenize
    # Initialize current_word with the last word in the prefix
    tokens = word_tokenize(sentence_prefix.lower())
    current_word = tokens[-1]  # Start with the last word in the prefix
    generated_words = []
    # For each word to predict:
    # - Check if current_word is in bigram_freq
    # - If it is, find the most frequent next word
    # - If not, break the loop
    # - Append the next word to generated_words
    # - Update current_word
    # - Stop if '</s>' is generated
    for _ in range(N):
        if current_word in bigram_freq:
            # Find the most frequent next word
            next_word = bigram_freq[current_word].most_common(1)[0][0]
            if next_word == "</s>":
                break  # Stop if the end token is generated
            generated_words.append(next_word)
            current_word = next_word
        else:
            break

    return ' '.join(generated_words)
    # pass  # Remove this line after implementing

# Main execution
if __name__ == "__main__":
    # Load the corpus
    corpus = brown.sents()
    
    # Preprocess the corpus
    tokenized_corpus = preprocess(corpus)
    
    # Build the bigram model
    bigram_freq, unigram_freq = build_bigram_model(tokenized_corpus)
    
    # Calculate the probability of a test sentence
    test_sentence = "The dog barked at the cat."
    probability_no_smoothing = sentence_probability(bigram_freq, unigram_freq, test_sentence, smoothing=False)
    print(f"Sentence probability without smoothing: {probability_no_smoothing}")
    
    probability_with_smoothing = sentence_probability(bigram_freq, unigram_freq, test_sentence, smoothing=True)
    print(f"Sentence probability with smoothing: {probability_with_smoothing}")
    
    # Predict the next N words
    # sentence_prefix = "I won 500"
    sentence_prefix = "I won 200 "
    N = 5
    predicted_words = predict_next_words(bigram_freq, unigram_freq, sentence_prefix, N, smoothing=True)
    print(f"Predicted next {N} words: {predicted_words}")

"""
Answer the following questions based on the outputs of your program:

1. Sentence probability without smoothing: 
0.0
2. Sentence probability with smoothing: 
1.7204128298897112e-25
3. Predicted next 5 words:
million dollars .
"""
