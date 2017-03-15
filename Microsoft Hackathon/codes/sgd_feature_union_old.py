import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pprint import pprint
from time import time

data = pd.read_csv('BingHackathonTrainingData.csv')

print("%d documents" % len(data.summary))

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

def my_analyzer(text):
    words = text.split(';')
    for w in words:
        yield w
    
pipeline = Pipeline([
    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ('title', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('tfidf', TfidfVectorizer()),
                ('best', TruncatedSVD(n_components=10)),
            ])),

            # Pipeline for standard bag-of-words model for body
            ('summary', Pipeline([
                ('selector', ItemSelector(key='summary')),
                ('tfidf', TfidfVectorizer()),
                ('best', TruncatedSVD(n_components=50)),
            ])),

            # Pipeline for pulling ad hoc features from post's body
            ('authors', Pipeline([
                ('selector', ItemSelector(key='authors')),
                ('count_vec', CountVectorizer(analyzer=my_analyzer)),
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'title': 1.0,
            'summary': 1.0,
            'authors': 1.0,
        },
    )),

    # Use a SGD classifier on the combined features
    ('clf', SGDClassifier(penalty = 'elasticnet')),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'vect__max_df': (1.0),
    #'vect__max_features': (40000, 45000, 50000, 60000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': ( 0.000001, 0.0000001, 1e-08),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (5, 10),
}

if __name__ == "__main__":
    # find the best parameters for both the feature extraction and the
    # classifier
##    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
##
##    print("Performing grid search...")
##    print("pipeline:", [name for name, _ in pipeline.steps])
##    print("parameters:")
##    pprint(parameters)
##    t0 = time()
##    grid_search.fit(data, data.topic_id)
##    print("done in %0.3fs" % (time() - t0))
##    print()
##
##    print("Best score: %0.3f" % grid_search.best_score_)
##    print("Best parameters set:")
##    best_parameters = grid_search.best_estimator_.get_params()
##    for param_name in sorted(parameters.keys()):
##        print("\t%s: %r" % (param_name, best_parameters[param_name]))
##

##    print ('Fitting on actual data')
##    pipeline = Pipeline([
##    ('vect', CountVectorizer(max_df = 1.0, ngram_range = (1, 2))),
##    ('tfidf', TfidfTransformer()),
##    ('clf', SGDClassifier(penalty= 'elasticnet',alpha = 1e-05 )),
##])
    pipeline.fit(data, data.topic_id)
##    predicted = pipeline.predict(data.data)
##    print(metrics.classification_report(data.target, predicted, target_names=['0','1','2']))
##
