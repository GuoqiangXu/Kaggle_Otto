# -*- coding: utf-8 -*-
"""
Tune the calibrated random forest classifier.

@author: guoqiangxu
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn import metrics
from sklearn.grid_search import ParameterGrid
from sklearn.utils.fixes import bincount
import sys
from sklearn.calibration import CalibratedClassifierCV

np.random.seed(17411)
    
def load_train_data(path='train.csv'):
    df = pd.read_csv(path)
    X = df.values
    np.random.shuffle(X)
    X_train, y_train = X[:, 1:-1], X[:, -1]
    return X_train.astype(float), y_train.astype(str)

def CV(X_train,y_train,parameters):
    n_folds = 3
    skf = cross_validation.StratifiedKFold(y_train, n_folds=n_folds)
    score = np.zeros((n_folds,))
    count = 0
    
    for train_index, test_index in skf:
        xtrain, xtest = X_train[train_index], X_train[test_index]
        ytrain, ytest = y_train[train_index], y_train[test_index]
        clf = RandomForestClassifier(**parameters)

        y_prob = Calibrate_CV(xtrain,ytrain,clf,xtest)
        
        encoder = LabelEncoder()
        y_true = encoder.fit_transform(ytest)
        
        assert (encoder.classes_ == clf.classes_).all()
        score[count] = metrics.log_loss(y_true, y_prob)
        count += 1

    return np.mean(score)

def Calibrate_CV(X,y,clf,X_valid):
    skf = cross_validation.StratifiedKFold(y, n_folds=3)
    clf_pc = CalibratedClassifierCV(clf, cv=skf, method='isotonic')
    clf_pc.fit(X,y)
    y_prob = clf_pc.predict_proba(X_valid)
    return y_prob

def tune(X_train,y_train,param_grid):
    score=[]
    count = 0
    PG = list(ParameterGrid(param_grid))
    for parameters in PG:
        score.append(CV(X_train,y_train,parameters))   
        count += 1
        
    min_score = np.min(score)
    return PG[np.where(score==min_score)[0][0]],min_score

def output(param,score,path='output.csv'):
    with open(path, 'a') as f:
        line = str(score)+','+str(param['n_estimators'])+','+str(param['max_features'])+','+str(param['max_depth'])+','+str(param['min_samples_split'])+','+str(param['min_samples_leaf'])+'\n'
        f.write(line)

def main():
    
    prefix = '/Users/guoqiangxu/Desktop/PyDev/Otto/Otto/data/'
    X_train, y_train = load_train_data(prefix+'Expand.csv')
    
    param_grid = {"n_estimators": [100],
                  "max_features": [10,20,30,40,50,60],
                  "max_depth": [2,4,6,8,10],
                  "min_samples_split": [2],4,6,8,10,
                  "min_samples_leaf": [1,2,3,4,5]}
              
    opt_param,score = tune(X_train,y_train,param_grid)
    output(opt_param,score,prefix+'output.csv')
    
if __name__ == '__main__':
    main()
