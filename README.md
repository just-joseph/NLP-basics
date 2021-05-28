# NLP-basics

Topics covered:
Day 1:
•	High level applications of NLP
•	Theory: NLP tasks like stemming, lemmatization, stopword removal 
•	Theory: Text embeddings/ vectorization concept, tfidf and one-hot encoding (or Bag of Words) vector calculation.

Day 2:
•	Code implementation of basic text analysis along with tokenization, stemming, lemmatization, stopword removal etc. using NLTK

Day 3:
•	Theory: Intro to NLP tasks like topic modeling, text clustering, text classification, similarity comparison.
•	Theory: Average workflow of ML/ DL model for NLP tasks.
•	Code implementation: Vectorization using tfidf and spacy.
Steps for tfidf:
o	Fit a tfidf vectorizer using a corpus of text (the vectorizer learns the different words or tokens from the corpus)
tfidfvectorizer.fit(corpus)
o	Transform new sentences using the created vectorizer to generate embeddings.
tfidf_embedding = tfidfvectorizer.transform( [‘I have a car’] ).toarray()
Steps for Spacy:
•	Import spacy package to python and import the appropriate spacy model for English language (we used the sm model)
•	Pass the sentences to spacy model and extract the vectors.
sentence_vectors=[]
for doc in nlp.pipe( [‘I own a car’, ‘I am a programmer’] ):
    sentence_vectors.append(doc.vector)
•	Code implementation: Cosine similarity of two sentences (or paragraphs) after converting them to vectors.
•	Code implementation: Using generated embeddings to train a machine learning model for text classification.
Pre-requisites: 
•	What is machine learning? 
•	What is supervised and unsupervised machine learning?
•	Basic knowledge of any ML model
o	Split dataset to test and train
o	Fitting the model on the train dataset (it’d be better if one understands the math behind what happens when a mathematical model is fit- we used a Naïve Bayes model)
o	Testing the model on the test dataset using performance metrics like accuracy

Day 4:
•	Introduction to Spacy
•	Tasks Spacy can perform
•	Code: Spacy tasks
