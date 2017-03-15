import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier
vectorizer = TfidfVectorizer(norm = 'l1', max_df = 1.0, max_features = 50000)

# Build a classification task using 3 informative features
data = pd.read_csv('BingHackathonTrainingData.csv')
X, y = data.summary, data.topic_id
print 'data loaded'
X = vectorizer.fit_transform(X)

# Create the RFE object and compute a cross-validated score.
clf = SGDClassifier(penalty = 'elasticnet', alpha = 1e-08, n_iter = 5)
print 'initialized classifier'
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf, step=100, cv=StratifiedKFold(y, 2),
              scoring='accuracy', verbose =1)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
