from numpy import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def randomForest(trainSet, trainLabels, testSet, D, T):
    clf = RandomForestClassifier(n_estimators = T, max_depth = D, min_samples_leaf = 2)
    clf.fit(trainSet, trainLabels)
    print(clf.feature_importances_)
    return rint(clf.predict(testSet))

def adaBoost(trainSet, trainLabels, testSet, D, T):

    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=D, min_samples_leaf=2), n_estimators=T)
    clf.fit(trainSet, trainLabels)
    return rint(clf.predict(testSet))