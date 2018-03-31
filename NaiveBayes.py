import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import pandas as pd

from sklearn import metrics
from sklearn.metrics import accuracy_score

test = pd.read_csv('test.csv')

train = pd.read_csv('train.csv')

y_train = train['label']
X_train= train.drop('label', axis=1)

y_test = test['label']
X_test= test.drop('label', axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('data loaded')


print("============Result========Naive Bayes=====")


# MOdel =========
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy : ", str(accuracy_score(y_test, y_pred)*100.0)+"%")

