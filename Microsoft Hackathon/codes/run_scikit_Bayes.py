from time import time
import logging
from pprint import pprint

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

###############################################################################
# Load training set

data = pd.read_csv('BingHackathonTrainingData.csv')

print("%d documents" % len(data.summary))
print("%d categories" % len(data.topic_id.unique()))
print()

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(max_df = 1.0, ngram_range = (1, 2))),
    ('tfidf', TfidfTransformer(use_idf = True)),
    ('clf', GaussianNB()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    
}

if __name__ == "__main__":
    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.summary, data.topic_id)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


##    print ('Fitting on actual data')
##    pipeline = Pipeline([
##    ('vect', CountVectorizer(max_df = 1.0, ngram_range = (1, 2))),
##    ('tfidf', TfidfTransformer()),
##    ('clf', SGDClassifier(penalty= 'elasticnet',alpha = 1e-05 )),
##])
##    pipeline.fit(data.data, data.target)
##    predicted = pipeline.predict(data.data)
##    print(metrics.classification_report(data.target, predicted, target_names=['0','1','2']))
##
