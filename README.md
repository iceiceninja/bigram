The University of Texas at Dallas 
CS 4395 
Human Language Technologies
Fall 2024 
Instructor: Dr. Zhiyu Zoey Chen
Grader/ Teaching Assistant: Mian Zhang
Homework 2: 100 points 
Issued Sep 16, 2024 
   Due Sep 30, 2024 before midnight 


Submission instructions:
All problems for this homework are coding practices. We provide code templates for each question:
p1-template.py
p2-template.py
p3-template.py

After you finish all the problems, change the file name and submit your program files for the three problems:
p1.py
p2.py
p3.py
Besides completing the coding tasks, be sure to answer the questions at the end of each program file.


Brown Corpus: The Brown University Standard Corpus of Present-Day American English, better known as simply the Brown Corpus, is an electronic collection of text samples of American English, the first major structured corpus of varied genres. For problem 1 and problem 2, we will use this corpus as data.


PROBLEM 1:  N-gram language models (35 points)

1. Build a bigram language model on the whole Brown corpus and calculate the probability of the sentence: "The dog barked at the cat.". (15 points)
Note: To calculate the probability of a sentence, the start token ‘<s>’ and end token ‘</s>’’ should be considered. 

2. Apply Laplace smoothing (add-one smoothing) to the bigram language model and calculate the probilitily of the sentence: “The dog barked at the cat.”. (10 points)


3. Predict the most probable next 5 words of the sentence prefix “I won 200” using the bigram model.  (10 points)


PROBLEM 2: Vector semantics (TF-IDF and PPMI vectors) (50 points)

1. Considering the first 1000 sentences of the Brown corpus (corpus[0:1000]), regard each sentence as a document and write a program to compute the TF-IDF for each word (20 points). Compute the TF-IDF for the words county, investigation and produced of the first document of the corpus. (5 points).

2.  Considering the first 1000 sentences of the Brown corpus (corpus[0:1000]), regard each sentence as a document and write a program to compute the PPMI  for each [word, context-word] pair (20 points). The  context of a word is the “window” of words consisting of (i) five words (if avalible) to the left of  the word; and (ii) five words (if avalible) to the right of the word. Compute the PPMI for three [word, context-word] pairs  [expected, approve], [mentally, in], [send, bed]  (5 points).


PROBLEM 3: Word Embeddings (15 points)

In this problem, we use the GloVe word vectors. Download the 6B-50d version here [https://nlp.stanford.edu/data/glove.6B.zip] and unzip it. Only keep the glove.6B.50d.txt file. 

Use the word vectors in the glove.6B.50d.txt to complete the following questions:
Write a function to load the GloVe word vectors. The output is a Python dictionary and for each item in the dictionary, the key is a word and the value is its corresponding word vector in the datatype of numpy.ndarray.  (5 points)
Write a function to compute the cosine similarity between the word vectors of two words and give the cosine similarity of the following pairs of words: (5 points)
("cat", "dog")
("car", "bus")
("apple", "banana")
Write a function to find the 5 most similar words for a specific word and answer what’s the top-5 most similars words for the following words:  (5 points)
"King"
"computer"
"university"

