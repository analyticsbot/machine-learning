import json
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from text_unidecode import unidecode
import xgboost as xgb
from xgboost.sklearn import XGBClassifier 
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

test = tfidf.transform(all_ingredients_test).astype('float32')

##################################################################
## STEP 3 - MODEL TUNING USING GRIDSEARCHCV
##################################################################
nfolds = 5 ## use 5-fold cross validation to check the best parameters

## first step is to get the number of estimators. this training will stop once
## the error stops reducing
xgtrain = xgb.DMatrix(train, label=y)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=1000, nfold=nfolds,
            metrics='auc', early_stopping_rounds=50, show_progress=False)

## optimum number of trees is 249

## now lets start tuning the parameters
max_features_values = ['sqrt', 'log2']
max_depth_values = range(4, 10, 3)
min_child_weight_values = [6,8,10,12]
subsample_values = [i/10.0 for i in range(6,10)],
colsample_bytree_values = [i/10.0 for i in range(6,10)]
gamma_values = [i/10.0 for i in range(0,5)]
param_grid = {'max_features' : max_features_values,\
              'max_depth':max_depth_values, 'min_child_weight': min_child_weight_values,\
               'subsample':subsample_values, 'colsample_bytree': colsample_bytree_values,\
              'gamma':gamma_values}
grid_search = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=249, \
                                                      objective= 'multi:softmax', scale_pos_weight=1,seed=1), \
                           param_grid = param_grid, scoring='roc_auc',n_jobs=4,iid=False, cv=nfolds)
grid_search.fit(train, y)
print grid_search.best_params_

## again get the final set of estimators
xgtrain = xgb.DMatrix(train, label=y)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=1000, nfold=nfolds,
            metrics='auc', early_stopping_rounds=50, show_progress=False)

## final value is 217 trees
## max_depth=7
## min_child_weight=8
## gamma=0.1
## subsample=0.7,
## colsample_bytree=0.8
## use this parameter to make the final model

##################################################################
## STEP 4 - MODEL USING PARAMETERS OBTAINED USING GRIDSEARCHCV
##################################################################

model = XGBClassifier(learning_rate =0.05, n_estimators=217, max_depth=7, \
                       min_child_weight=8, gamma=0.1, subsample=0.7, colsample_bytree=0.8,\
                       objective= 'multi:softmax', scale_pos_weight=1,seed=1)
## fit the model
model.fit(train,y)

## predict using the model
cuisine_pred = model.predict(test)
cuisine_pred_labels = lbl.inverse_transform(cuisine_pred)
## take the id from the test dataframe
ids = [row['id'] for row in test_df]

## make a submission file
output = pd.DataFrame({'id': ids, 'cuisine': cuisine_pred_labels}, columns=['id', 'cuisine'])
output.to_csv('xgboost_submission.csv', index=False)


