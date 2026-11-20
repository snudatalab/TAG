"""
Accurate graph classification via two-staged contrastive curriculum learning (PLOS ONE 2024)

Authors:
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Kahyun Park (kahyun.park@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software is free of charge under research purposes.

For other purposes (e.g. commercial), please contact the authors.

evaluate_embedding.py
    - codes for evaluating graph embeddings with classifiers
"""

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import warnings


def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'dual':[True,False], 'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)

def mlp_classify(x, y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = MLPClassifier(activation='relu', hidden_layer_sizes=[])
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)

def evaluate_embedding(embeddings, labels, search=True):
    warnings.filterwarnings("ignore")
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    accuracy = linearsvc_classify(x, y, search)

    return accuracy

def evaluate_embedding_mlp(embeddings, labels):
    warnings.filterwarnings("ignore")
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    accuracy = mlp_classify(x, y)

    return accuracy
