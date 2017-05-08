#######################################################################
## Support Vector Classifier for What's Cooking Competition on Kaggle #
## ####################################################################

## Steps = data read, data clean, data munging, model initiate
## model tune, model build, results

## modules necessary - pandas, json, sklearn
import json
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from text_unidecode import unidecode

## this is a muticlass classification problem
verbose = 0 ## print updates or not - boolean

if verbose:
    print 'all modules imported'

##################################################################
## STEP 1 - READING AND CLEANING DATASETS
##################################################################
train_df = pd.read_json('train.json')[:1000]
test_df = pd.read_json('train.json')[1001:1500]

## check the shape of training and test dataset
if verbose:
    print 'size of training dataset is', train_df.shape
    print 'size of test dataset is', test_df.shape

## read the ingredient list and do some cleaning
## remove digits and lower the characters. strip any whitespaces if present
all_ingredients_train = []
all_cuisines = []
for i, row in train_df.iterrows():
    all_ingredients_train.append(unidecode(re.sub('\d+', '',' '.join(row['ingredients']).lower().strip())))
    all_cuisines.append(row['cuisine'])

all_ingredients_test= []
for i, row in test_df.iterrows():
    all_ingredients_test.append(unidecode(re.sub('\d+', '',' '.join(row['ingredients']).lower().strip())))
    
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

test = tfidf.transform(all_ingredients_test).astype('float32')

##################################################################
## STEP 3 - MODEL TUNING USING GRIDSEARCHCV
##################################################################

##nfolds = 5 ## use 5-fold cross validation to check the best parameters
##C_values = [0.01, 0.1, 1, 10, 100]
##gamma_values = [0.01, 0.1, 1]
##kernel_values = ['linear', 'rbf', 'poly']
##param_grid = {'C': C_values, 'gamma' : gamma_values, 'kernel':kernel_values}
##grid_search = GridSearchCV(OneVsRestClassifier(SVC()), param_grid, cv=nfolds)
##grid_search.fit(train, y)
##print grid_search.best_params_
#### use this parameter to make the final model

##################################################################
## STEP 4 - MODEL USING PARAMETERS OBTAINED USING GRIDSEARCHCV
##################################################################

model = OneVsRestClassifier(SVC(C=100, kernel='rbf', gamma=0.1, probability=False, tol=0.001, cache_size=200,\
          verbose=True, random_state=1))
## fit the model
model.fit(train,y)

## predict using the model
cuisine_pred = model.predict(test)
cuisine_pred_labels = lbl.inverse_transform(cuisine_pred)
## take the id from the test dataframe
ids = [row['id'] for row in test_df]

## make a submission file
output = pd.DataFrame({'id': ids, 'cuisine': cuisine_pred_labels}, columns=['id', 'cuisine'])
output.to_csv('svc_submission.csv', index=False)
