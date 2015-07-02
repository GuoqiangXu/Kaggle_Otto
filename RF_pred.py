# -*- coding: utf-8 -*-
"""
Make prediction using calibrated random forest classifier after the parameter is tuned.

@author: guoqiangxu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from XGBoostClassifier import XGBoostClassifier


def load_train_data(path='train.csv'):
    df = pd.read_csv(path)
    X = df.values
    #shuffle the data
    np.random.shuffle(X)
    X_train, y_train = X[:, 1:-1], X[:, -1]
    return X_train.astype(float), y_train.astype(str)
            
def load_test_data(path='test.csv'):
    df = pd.read_csv(path)
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)
    
def make_submission(y_prob,ids,encoder,path='RF_n100_weights.csv'):
    with open(path, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')

print(" - Start.")
prefix = '/Users/guoqiangxu/Desktop/PyDev/Otto/Otto/data/'
X_train, y_train = load_train_data(prefix+'train.csv')

print(" - Data preprocessed.")
param = {'bst:max_depth':10, 'gamma':.75, 'bst:min_child_weight':4, 'subsample':.9, 'colsample_bytree':.8, 'max_delta_step':0, 'objective':'multi:softprob', 'num_class':9,'silent':0}
clf = RandomForestClassifier(n_estimators=100,max_features=30,max_depth=30,min_samples_split=2,min_samples_leaf=1)

skf = cross_validation.StratifiedKFold(y_train, n_folds=5)
clf_pc = CalibratedClassifierCV(clf, cv=skf, method='isotonic')
clf_pc.fit(X_train,y_train) 
encoder = LabelEncoder().fit(y_train)

print(" - Model built.")

X_test, ids = load_test_data(prefix+'test.csv')
y_prob = clf_pc.predict_proba(X_test)

print(" - Prediction made.")

make_submission(y_prob,ids,encoder,path=prefix+'RF.csv')

print(" - Submission file generated.")
