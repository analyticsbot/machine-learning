from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import collections
import numpy as np
import os, warnings
from text_unidecode import unidecode
import pandas as pd

## read the dataset
df = pd.read_csv('train.csv')

## ignore any warnings
warnings.filterwarnings("ignore")

# read glove vectors
gloveDict = collections.defaultdict(lambda: np.zeros((300,)))

## this can be downloaded from stanfords website
## open the file to read
f = open('glove.6B.300d.txt', "rb")
for line in f:
    cols = line.strip().split()
    word = cols[0]
    embedding = np.array(cols[1:], dtype="float32")
    gloveDict[word] = embedding

## close the file
    f.close()

def getGloveoCosineSimilarity(question1, question2):
    """Function to calculate the glove similarity
    question1 and question2 are parameters"""
    questions = [question1, question2]

    ## for the sentences we need to get the count vectors
    vec = CountVectorizer(max_features=5000, stop_words=None,binary=True)
    count_vectors = vec.fit_transform(questions)

    ## get the vocabulary of words from the questions
    vocab_index = vec.vocabulary_

    ## get the index of the words and embeddings
    index_word = {v:k for k, v in vocab_index.items()}

    ## get the question vectors
    question_vectors = np.zeros((count_vectors.shape[0], 300))

    ## iterate through count vectors for each word get the embeddings
    ## for each embedding, we will then average by the number of words
    ## this will be then used for cosine similarity
    for i in range(count_vectors.shape[0]):
        row = count_vectors[i, :].toarray()
        word_ids = np.where(row > 0)[1]
        word_counts = row[:, word_ids][0]
        numWords = np.sum(word_counts)

        ## if there are no words, continue
        if numWords == 0:
            continue

        ## initialize the word embeddings to 0
        word_embeddings = np.zeros((word_ids.shape[0], 300))

        ## update the word embeddings
        for j in range(word_ids.shape[0]):
            word_id = word_ids[j]
            word_embeddings[j, :] = word_counts[j] * gloveDict[index_word[word_id]]
        question_vectors[i, :] = np.sum(word_embeddings, axis=0) / numWords

    return(cosine_similarity(question_vectors[0], question_vectors[1])[0][0])

## iterate through the dataframe and create a new column that contains the glove similarity for
## two questions
for i, x in df.iterrows():
    ## where there is an error, it will be saved as NA
    try:
            df.loc[i, 'glove_sim'] = getGloveoCosineSimilarity(unidecode(x['question1']), unidecode(x['question2']))
    except:
            print x['question1'], x['question2'], i
            df.loc[i, 'glove_sim'] = 'NA'
            pass
df.to_csv('glove_similarity.csv', index = False)
