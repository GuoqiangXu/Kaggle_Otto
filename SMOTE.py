'''
Implementation of SMOTE in python. SMOTE is an oversampling method to deal with the issue of imbalanced class. 
X: minority class samples
N: number of majority class samples
k: number of nearest 

@author: guoqiangxu
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot

def SMOTE(X,N,k):
    n_samples,n_features = X.shape
    n_synthetic = N - n_samples
    
    nearest_neighbour = NearestNeighbors(n_neighbors=k+1).fit(X)
    nns = nearest_neighbour.kneighbors(X, return_distance=False)[:, 1:]
    
    random_state = np.random.RandomState(18964)
    indices = random_state.randint(0,n_samples*k,n_synthetic)
    row = indices/k
    col = np.mod(indices - row*k,k)
    delta = random_state.rand(n_synthetic,n_features)
    
    Xneigh = X[nns[row,col],:]
    X0 = X[row,:]
    diff = Xneigh - X0
    
    X_synthetic = X0 + np.multiply(diff,delta)
    
    return X_synthetic

# test     
x_major = np.random.standard_normal(1000)
y_major = np.random.standard_normal(1000)  
 
x_minor = np.random.standard_normal(500)+5
y_minor = np.random.standard_normal(500)+5  
 
pyplot.figure()
pyplot.scatter(x_major,y_major,c=u'r')
pyplot.scatter(x_minor,y_minor,s=40,c=u'b')
 
 
X = np.concatenate((x_minor[:,np.newaxis],y_minor[:,np.newaxis]),axis=1)
Xs = SMOTE(X,1000,12)
 
pyplot.scatter(Xs[:,0],Xs[:,1],c=u'g')
pyplot.show()



    
