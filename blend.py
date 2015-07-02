'''
Created on May 6, 2015
Simple blend.

@author: guoqiangxu
'''
import pandas as pd
import numpy as np

def load_submission_data(path='train.csv'):
    df = pd.read_csv(path)
    X = df.values
    ids,y_prob = X[:,0],X[:, 1:10]
    ids = np.array(map(int,ids))
    return ids.astype(str),y_prob

def make_submission(y_prob,ids,path='RF_n100_weights.csv'):
    with open(path, 'w') as f:
        f.write('id,')
        f.write('Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9')
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    
prefix = '/Users/guoqiangxu/Desktop/PyDev/Otto/Otto/data/'
ids,y_xgb1 = load_submission_data(prefix+'XGB1.csv')
ids,y_xgb2 = load_submission_data(prefix+'XGB2.csv')
ids,y_nn1 = load_submission_data(prefix+'NN_SGD.csv')
ids,y_nn2 = load_submission_data(prefix+'NN_SGD2.csv')
ids,y_nn3 = load_submission_data(prefix+'NN_SGD3.csv')
ids,y_rf1 = load_submission_data(prefix+'RF_SMOTE.csv')
ids,y_rf2 = load_submission_data(prefix+'RF_n1000.csv')
ids,y_svm = load_submission_data(prefix+'SVM.csv')
y_prob = np.multiply(y_xgb1,0.1) + np.multiply(y_xgb2,0.1) + np.multiply(y_nn1,0.1)+np.multiply(y_nn2,0.1)+np.multiply(y_nn3,0.1)+np.multiply(y_rf1,0.1)+np.multiply(y_rf2,0.1)+np.multiply(y_svm,0.0)

make_submission(y_prob,ids,path=prefix+'blend_average3.csv')
