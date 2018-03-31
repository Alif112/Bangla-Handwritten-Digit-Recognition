from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


test = pd.read_csv('test.csv')

train = pd.read_csv('train.csv')

y_train = train['label']
X_train= train.drop('label', axis=1)

y_test = test['label']
X_test= test.drop('label', axis=1)


print('data loaded')

print('===============Results Decision tree=========================')


dtc=DecisionTreeClassifier()

dtc.fit(X_train,y_train)

print(dtc.score(X_test,y_test))

