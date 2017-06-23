
import pandas as pd

import numpy as np

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn import linear_model, datasets

train = pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})


train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")


train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(train["Age"].median())

train["Pclass"] = train["Pclass"].fillna(train["Pclass"].median())
test["Pclass"] = test["Pclass"].fillna(train["Pclass"].median())

train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1,'Q':2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1,'Q':2})

target_train = train["Survived"].values
features_train = train[["Pclass", "Sex", "Age","Embarked"]].values
features_test = test[["Pclass", "Sex", "Age","Embarked"]].values

my_classifier=KNeighborsClassifier()
my_classifier.fit(features_train,target_train)
print("K nearest neighbours:")
print(my_classifier.score(features_train, target_train))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 3), random_state=1)
clf.fit(features_train,target_train)       
print("MLP:")
print(clf.score(features_train, target_train))

svm1 = SVC()
svm1.fit(features_train,target_train)
print("SVM:")
print(svm1.score(features_train, target_train))


logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(features_train,target_train)
print("Logistic Regression:")
print(logreg.score(features_train, target_train))

