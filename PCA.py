'''
Created on May 5, 2015
Dimension reduction using PCA/SVD, followed by upsampling method using SMOTE.
@author: guoqiangxu
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from SMOTE import SMOTE

def load_train_data(path='train.csv'):
    df = pd.read_csv(path)
    X = df.values
    #shuffle the data
    np.random.shuffle(X)
    X_train, y_train = X[:, 1:-1], X[:, -1]
    return X_train.astype(float), y_train.astype(str)

def output(path,data):
    df = pd.DataFrame(data=data)
    df.to_csv(path)
    
print(" - Start.")
prefix = '/Users/guoqiangxu/Desktop/PyDev/Otto/Otto/data/'
X, y = load_train_data(prefix+'train.csv')

scale = StandardScaler()
Xs = scale.fit_transform(X)
#pca = PCA(n_components=3,copy=True,whiten=False)
svd = TruncatedSVD(n_components=3)

svd.fit(Xs)
Xr = svd.transform(Xs)

encoder = LabelEncoder()
yr = encoder.fit_transform(y)
yr = yr[np.newaxis].T

N = sum(yr==1)[0]
 
for i in [0,1,2,3,4,5,6,7,8]:
    X_class = Xr[(yr==i)[:,0],:]
    k = N/X_class.shape[0] + 1
    X_synthetic = SMOTE(X_class,N,k)
    X_new = scale.inverse_transform(svd.inverse_transform(X_synthetic))
    X = np.concatenate((X,X_new),axis=0)
    y = np.concatenate((y,encoder.inverse_transform([i for j in range(0,X_new.shape[0])])),axis=0)
     
 
output_data = np.concatenate((X,y[:,np.newaxis]),axis=1)

output(prefix+'Expand.csv',output_data)
