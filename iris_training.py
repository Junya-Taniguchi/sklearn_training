from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
clf = svm.SVC()
clf.fit(X, y)

import pickle

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[55:56]),y[55])
