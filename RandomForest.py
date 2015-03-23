# -*- coding: utf-8 -*-
"""
The default training and test file is under the same folder of this file. After 
running, one submission file with default name MySubmission.csv is generated.

This file uses random forest classifier. With default setting, the Kaggle score
is 1.45878.

@author: guoqiangxu
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_train_data(path='../train.csv'):
    df = pd.read_csv(path)
    X = df.values
    X_train, y_train = X[:, 1:-1], X[:, -1]
    return X_train.astype(float), y_train.astype(str)
            
def load_test_data(path='../test.csv'):
    df = pd.read_csv(path)
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)
    
def make_submission(y_prob,ids,encoder,path='MySubmission.csv'):
    with open(path, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')

print(" - Start.")

X_train, y_train = load_train_data()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
encoder = LabelEncoder().fit(y_train)

print(" - Model built.")

X_test, ids = load_test_data()
y_prob = clf.predict_proba(X_test)

print(" - Prediction made.")

make_submission(y_prob,ids,encoder)

print(" - Submission file generated.")