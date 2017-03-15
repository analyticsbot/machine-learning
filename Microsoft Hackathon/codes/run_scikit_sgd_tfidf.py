from time import time
import logging
from pprint import pprint

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
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
print("%d categories" % len(data.topic_id))
print()

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SGDClassifier(penalty = 'elasticnet', alpha = 1e-05)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'tfidf__use_idf': (True, False),
    'tfidf__analyzer': ('word', 'char'),
    #'tfidf__norm': ('l1', 'l2'),
    'tfidf__ngram_range': ((1,2), (1,3)),
    'tfidf__max_df' : (0.6, 0.8, 1),
    #'tfidf__min_df' : (0.1, 0.2),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
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
