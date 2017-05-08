import json
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

nfolds = 5 ## use 5-fold cross validation to check the best parameters
n_estimators_values = range(100,1000, 300)
max_features_values = ['sqrt', 'log2']
max_depth_values = range(4, 10, 3)
min_samples_split_values = range(3, 7, 2)
min_samples_leaf_values = range(1, 3, 2)
param_grid = {'n_estimators': n_estimators_values, 'max_features' : max_features_values,\
              'max_depth':max_depth_values, 'min_samples_split': min_samples_split_values,\
               'min_samples_leaf':min_samples_leaf_values}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=nfolds)
grid_search.fit(merged_train, y)
print grid_search.best_params_
## use this parameter to make the final model

##################################################################
## STEP 4 - MODEL USING PARAMETERS OBTAINED USING GRIDSEARCHCV
##################################################################

model = RandomForestClassifier(n_estimators=700, max_features='log2', \
                             max_depth=7, min_samples_split=5, min_samples_leaf=3,\
                             verbose=True, random_state=1)
## fit the model
model.fit(train,y)

## predict using the model
cuisine_pred = model.predict(test)
cuisine_pred_labels = lbl.inverse_transform(cuisine_pred)
## take the id from the test dataframe
ids = [row['id'] for row in test_df]

## make a submission file
output = pd.DataFrame({'id': ids, 'cuisine': cuisine_pred_labels}, columns=['id', 'cuisine'])
output.to_csv('random_forest_submission.csv', index=False)


