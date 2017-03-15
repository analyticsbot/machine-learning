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
                ('best', TruncatedSVD(n_components = 50)),
            ])),

            # Pipeline for standard bag-of-words model for body
            ('summary', Pipeline([
                ('selector', ItemSelector(key='summary')),
                ('tfidf', TfidfVectorizer(norm='l2', max_df = 1.0)),
                ('best', TruncatedSVD(n_components = 100)),
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
    ('clf', LogisticRegression(C =1,multi_class='multinomial',penalty='l2',solver= 'newton-cg' )),
])


if __name__ == "__main__":
    pipeline.fit(data, y)
    predicted = pipeline.predict(test)
    df = pd.DataFrame(columns = ['record id', 'topic'])
    df['record id'] = record_id
    df['topic'] = predicted
    df.to_csv('challenge1_output_topic_3.csv')
    
