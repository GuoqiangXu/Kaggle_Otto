# -*- coding: utf-8 -*-
"""
Make predictions with XGBoost clssifier after the parameter is tuned.
@author: guoqiangxu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn import cross_validation
import xgboost as xgb

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
X_train, y_train = load_train_data(prefix+'Expand.csv')
encoder = LabelEncoder().fit(y_train)
yt = encoder.transform(y_train)
dtrain = xgb.DMatrix(X_train,label=yt)

print(" - Data preprocessed.")

param = {'bst:max_depth':12, 'gamma':.6, 'bst:min_child_weight':4, 'subsample':.9, 'colsample_bytree':.8, 'max_delta_step':0, 'objective':'multi:softprob', 'num_class':9,'silent':0}
plst = param.items()
#plst += [('eval_metric', 'mlogloss')]
num_round = 250
clf = xgb.train(plst,dtrain,num_round)

print(" - Model built.")

X_test, ids = load_test_data(prefix+'test.csv')
dtest = xgb.DMatrix(X_test)
y_prob = clf.predict(dtest)

print(" - Prediction made.")

make_submission(y_prob,ids,encoder,path=prefix+'XGB2.csv')

print(" - Submission file generated.")
