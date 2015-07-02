# -*- coding: utf-8 -*-
"""
Tune GBC model using random search
@author: guoqiangxu
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn import metrics
from sklearn.grid_search import ParameterGrid,RandomizedSearchCV
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
        clf = GradientBoostingClassifier(**parameters)
        
        clf.fit(X=xtrain, y=ytrain)
        y_prob = clf.predict_proba(xtest)
        #y_prob = Calibrate_CV(xtrain,ytrain,clf,xtest)
        
        encoder = LabelEncoder()
        y_true = encoder.fit_transform(ytest)
        
        #assert (encoder.classes_ == clf.classes_).all()
        score[count] = metrics.log_loss(y_true, y_prob)
        count += 1
        print count
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
        line = str(score)+','+str(param['learning_rate'])+','+str(param['max_features'])+','+str(param['max_depth'])+','+str(param['min_samples_split'])+','+str(param['min_samples_leaf'])+','+str(param['min_weight_fraction_leaf'])+','+str(param['subsample'])+'\n'
        f.write(line)

def main():
    
    prefix = '/Users/guoqiangxu/Desktop/PyDev/Otto/Otto/data/'
    X_train, y_train = load_train_data(prefix+'train.csv')
    
    param_grid = {"n_estimators": [250],
                  "learning_rate": [0.05,0.1,0.15,0.2],
                  "max_features": [10,20,30,40,50],
                  "max_depth": [5,10,15,20],
                  "min_samples_split": [2,4,6,8],
                  "min_samples_leaf": [4,8,12,16],
                  "min_weight_fraction_leaf":[0.05,0.1,0.15],
                  "subsample":[0.9,0.8,0.7]}
              
    clf = GradientBoostingClassifier()
    n_iter=50
    random_search = RandomizedSearchCV(clf,param_distributions=param_grid,n_iter=n_iter,scoring='log_loss',cv=cross_validation.StratifiedKFold(y_train, n_folds=3),verbose=2)
    random_search.fit(X_train,y_train)
    opt_param,score = tune(X_train,y_train,param_grid)
    for i in range(0,n_iter):
        print random_search.grid_scores_[i]
    for i in range(0,n_iter):
        score = -random_search.grid_scores_[i][1]
        param = random_search.grid_scores_[i][0]
        output(param,score,prefix+'outputGBC.csv')
    
if __name__ == '__main__':
    main()
