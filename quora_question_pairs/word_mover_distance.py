from pyemd import emd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics import euclidean_distances
import os
from gensim.models.word2vec import Word2Vec
from text_unidecode import unidecode
import pandas as pd

## read the train dataset
df = pd.readata_vectorscsv('train.csv')

## print the dimensions
print df.shape

## print basic summary of the dataset
print df.describe()

## read the word2vec model from google news vectors
## can be downloaded online
wv = Word2Vec.loadata_vectorsword2vec_format(
        "data/GoogleNews-vectors-negative300.bin.gz",
        binary=True)

## read the word embeddings for the vocabulary of words present
## in the google news
W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=wv.syn0.shape)

with open("data/embed.vocab") as f:
    vocab_list = map(str.strip, f.readlines())

## generate a dictionary of word and embeddings
vocab_dictionary = {w: k for k, w in enumerate(vocab_list)}

def WMDsimilarity(q1, q2):
    """Function to calculate the word mover distance between question 1 and queston 2
    inputs = q1 is question1 and q2 is question2
    """
    ## feed the questions to countvectorizer
    vect = CountVectorizer.fit([q1, q2])
    ## get the embeddings for words in the questions
    W_value = W[[vocab_dictionary[w] if w in vocab_dictionary.keys() else 0 for w in vect.get_feature_names() ]]
    data_vectors = euclidean_distances(W_value)

    ## transform the vectors of questions1 and 2
    vector_1, vector_2 = vect.transform([q1, q2])
    vector_1 = vector_1.toarray().ravel()
    vector_2 = vector_2.toarray().ravel()

    ## pyemd needs double precision input. hence converting to double precision
    vector_1 = vector_1.astype(np.double)
    vector_2 = vector_2.astype(np.double)
    vector_1 /= vector_1.sum()
    vector_2 /= vector_2.sum()
    data_vectors = data_vectors.astype(np.double)
    data_vectors /= data_vectors.max()  
    return emd(vector_1, vector_2, data_vectors)

## iterate through the dataframe and create a new column wmd
for i, x in df.iterrows():
    ## do a try and except since there might be an error.
    ## in case of an error, the wmd is NA
    try:
        df.loc[i, 'wmd'] = WMDsimilarity(unidecode(x['question1']), unidecode(x['question2']))
    except:
        print x['question1'], x['question2'], i
        df.loc[i, 'wmd'] = 'NA'
        pass
