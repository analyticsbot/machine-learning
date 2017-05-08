import json
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from text_unidecode import unidecode
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
## this is a muticlass classification problem
verbose = 0 ## print updates or not - boolean

if verbose:
    print 'all modules imported'

##################################################################
## STEP 1 - READING AND CLEANING DATASETS
##################################################################
train_df = pd.read_json('train.json')
test_df = pd.read_json('train.json')

## check the shape of training and test dataset
if verbose:
    print 'size of training dataset is', train_df.shape
    print 'size of test dataset is', test_df.shape

## read the ingredient list and do some cleaning
## remove digits and lower the characters. strip any whitespaces if present
all_ingredients_train = [unidecode(re.sub('\d+', '',' '.join(row['ingredients']).lower().strip())) for row in train_df]
all_cuisines = [row['cuisine'] for row in train_df]

all_ingredients_test = [unidecode(re.sub('\d+', '',' '.join(row['ingredients']).lower().strip())) for row in test_df]

## remove special characters from ingredients
all_ingredients_train = [ ing.replace("-", " ").replace("&", " ").replace("'", " ").replace("''", " ").replace("%", " ")\
                    .replace("!", " ").replace("(", " ").replace(")", " ").replace("/", " ").replace("/", " ")\
                    .replace(",", " ").replace(".", " ") for ing in all_ingredients_train]

## remove extra whitespaces
all_ingredients_train = [ re.sub('\s+', ' ', ing).strip() for ing in all_ingredients_train]

## number of unique ingredients and cuisine in the dataset
if verbose:
    print 'total number of ingedients are', len(set(all_ingredients))
    print 'total number of cusines are', len(set(all_cuisines))


##################################################################
## STEP 2 - EXTRACTING FEATURES USING TFIDF VECTORIZER
##################################################################
## initialize tfidf vectorizer and label encoder
tfidf = TfidfVectorizer()
lbl = LabelEncoder()

## fit and transform on the test and train dataset
train = tfidf.fit_transform(all_ingredients_train).astype('float32')
y = lbl.fit_transform(all_cuisines)
# Convert labels to categorical one-hot encoding
y = keras.utils.to_categorical(y, num_classes=20)
test = tfidf.transform(all_ingredients_test).astype('float32')

    
mdl = Sequential()
mdl.add(Dense(512, init='glorot_uniform', activation='relu', 
                                    input_shape=(train_feats.shape[1],)))
mdl.add(Dropout(0.5))
mdl.add(Dense(128, init='glorot_uniform', activation='relu'))
mdl.add(Dropout(0.5))
mdl.add(Dense(20, activation='softmax'))
mdl.compile(loss='categorical_crossentropy', optimizer='adadelta')
    
mdl.fit(train, y, nb_epoch=2500, batch_size=4096, show_accuracy = True)

model.predict_proba(test))

# final cuisine decision: argmax of sum of log probabilities  
print("\nPredicting...")      
preds = sum(np.log(preds))
guess = le.inverse_transform(np.argmax(preds, axis=1))
create_submission(test_ids, guess) 
