import gensim
import numpy as np
import pandas as pd
import scipy

## imported all modules

## load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

## index of the words in the model
index2word_set = set(model.index2word)

## read the training dataset
df = pd.read_csv('train.csv')

## function to get the average vector for a question
def featureVector(q_words, wv_model, number_features, index2word_set):

    ## words - words in the question
    ## mode - word2vec mode
    ## num_features  is the number of features to work
    featureVecctors = np.zeros((number_features,), dtype="float32")
    num_words = 0

    ## iterate through the words
    for q_word in q_words:
        if q_word in index2word_set:
            num_words = num_words+1
            featureVec = np.add(featureVecctors, wv_model[q_word])

    if(num_words>0):
        featureVecctors = np.divide(featureVecctors, num_words)
    return featureVecctors

## define a function to get the similarity based on word2vec vector embeddings
## q1 and q2 are questions. model is word2vec model
def getSim(q1, q2, model, index2word_set):
    sentence_1 = featureVector(q1.split(), model, 300, index2word_set)
    sentence_2 = featureVector(q2.split(), model, 300, index2word_set)
    return(1 - scipy.spatial.distance.cosine(sentence_1,sentence_2))


## iterate through the dataframe and create a new column wmd
for i, x in df.iterrows():
    ## do a try and except since there might be an error.
    ## in case of an error, the wmd is NA
    try:
        df.loc[i, 'word2vec_sim'] = getSim(unidecode(x['question1']), unidecode(x['question2']))
    except:
        print x['question1'], x['question2'], i
        df.loc[i, 'word2vec_sim'] = 'NA'
        pass
