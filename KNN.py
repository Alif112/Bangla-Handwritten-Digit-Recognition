

from sklearn import svm
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split


test = pd.read_csv('test.csv')

train = pd.read_csv('train.csv')

y_train = train['label']
X_train= train.drop('label', axis=1)

y_test = test['label']
X_test= test.drop('label', axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('data loaded')


print("============Result======KNN=======")
# print(len(y_test))

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


clf.fit(X_train, y_train)
acc=clf.score(X_test,y_test)
# print(acc)
print("Accuracy: %.2f" % (acc*100))

import numpy as np
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, clf.predict(X_test)))




