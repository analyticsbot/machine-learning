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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

model1 = OneVsRestClassifier(SVC(C=100, kernel='rbf', gamma=0.1, probability=False, tol=0.001, cache_size=200,\
          verbose=True, random_state=1))

## random forest
model2 = RandomForestClassifier(n_estimators=700, max_features='log2', \
                             max_depth=7, min_samples_split=5, min_samples_leaf=3,\
                             verbose=True, random_state=1)


## predict using xgboost
model3 = XGBClassifier(learning_rate =0.05, n_estimators=217, max_depth=7, \
                       min_child_weight=8, gamma=0.1, subsample=0.7, colsample_bytree=0.8,\
                       objective= 'multi:softmax', scale_pos_weight=1,seed=1)

n_folds = 5
skf = list(StratifiedKFold(y, n_folds))
clfs = [model1, model2, model3]

for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((test.shape[0], 20))
        for i, (train_idx, test_idx) in enumerate(skf):
            print "Fold", i
            X_train = train[train_idx]
            y_train = y[train_idx]
            X_test = test[test_idx]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)
            dataset_blend_train[test, row_cnt: row_cnt+20] = y_submission
            dataset_blend_test_j += clf.predict_proba(X_submission)
        dataset_blend_test[:, row_cnt: row_cnt+20] = dataset_blend_test_j / 10.
        row_cnt += 20
    print
    print "Training Logistic Regression classifier."
    # C parameter here set through experimentation.
    clf = LogisticRegression(C=10)
    clf.fit(dataset_blend_train, y)

