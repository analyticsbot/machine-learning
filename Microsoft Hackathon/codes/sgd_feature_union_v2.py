import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
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
from sklearn.cross_validation import train_test_split

data = pd.read_csv('BingHackathonTrainingData.csv')
y = data.pop('topic_id')
data.drop(['record_id', 'publication_year'], axis=1, inplace=True)
test = pd.read_csv('BingHackathonTestData.csv')
record_id = test.pop('record_id')
test.drop(['publication_year', 'topic_id'], axis=1, inplace=True)


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
                ('tfidf', TfidfVectorizer(norm='l1')),
                ('best', TruncatedSVD()),
            ])),

            # Pipeline for standard bag-of-words model for body
            ('summary', Pipeline([
                ('selector', ItemSelector(key='summary')),
                ('tfidf', TfidfVectorizer(norm='l2', max_df = 1.0)),
                ('best', TruncatedSVD()),
            ])),

            # Pipeline for pulling ad hoc features from post's body
            ('authors', Pipeline([
                ('selector', ItemSelector(key='authors')),
                ('countvec', CountVectorizer(analyzer=my_analyzer)),
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
    ('clf', LogisticRegression( verbose =0, penalty ='l2')),
])
#penalty = 'elasticnet'
parameters = {
    #'union__title__tfidf__norm': ('l1', 'l2'),
    #'union__title__tfidf__ngram_range': ((1,2), (1,3)),
    #'union__title__tfidf__max_df': (0.6, 0.7),
    #'union__title__tfidf__max_features': (5000,50000),
    'union__title__best__n_components': (10, 20, 50),
    
    #'union__summary__tfidf__norm': ('l1', 'l2'),
    #'union__summary__tfidf__ngram_range': ((1,2), (1,3)),
    #'union__summary__tfidf__max_df': (0.8, 1.0),
    #'union__summary__tfidf__max_features': (5000,50000),
    'union__summary__best__n_components': (100, 200, 300),

    #'union__authors__countvec__max_features': (10, 50),
    
    #'clf__alpha': ( 0.000001, 0.0000001),
    #'clf__penalty': ('l2', 'l1'),
    #'clf__n_iter': (3, 5),
    'clf__C': ( 1, 2),
    'clf__solver': ('newton-cg', 'lbfgs'),
    'clf__multi_class': ('ovr', 'multinomial'),
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))



##    ##    pipeline.fit(data, data.topic_id)
####    predicted = pipeline.predict(data)
##    pipeline.fit(X_train, y_train)
##    predicted = pipeline.predict(X_test)
##    print(metrics.classification_report(y_test, predicted, \
##                                        target_names=['0','1','2']))
##
####    for i in range(50):
##	print predicted[i], '**', data.topic_id[i]
##	print int(predicted[i]) - int(data.topic_id[i])
##	print '%%%%%%%%%%%%%%%%%'
