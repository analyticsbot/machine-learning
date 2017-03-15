import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDRegressor
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
y = data.pop('publication_year')
data.drop(['record_id', 'topic_id'], axis=1, inplace=True)
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
                ('tfidf', TfidfVectorizer(norm='l1', max_df = 1.0)),
                ('best', TruncatedSVD(n_components = 20)),
            ])),

            # Pipeline for standard bag-of-words model for body
            ('summary', Pipeline([
                ('selector', ItemSelector(key='summary')),
                ('tfidf', TfidfVectorizer(norm='l2', max_df = 0.8)),
                ('best', TruncatedSVD(n_components = 200)),
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
    ('clf', SGDRegressor(alpha = 1e-06, n_iter = 5, verbose =0, penalty = 'l1')),
])


if __name__ == "__main__":
    pipeline.fit(data, y)
    predicted = pipeline.predict(test)
    df = pd.DataFrame(columns = ['record id', 'publication year'])
    df['record id'] = record_id
    df['publication year'] = predicted
    df.to_csv('challenge2_output_topic.csv')
    
