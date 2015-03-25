# -*- coding: utf-8 -*-
"""
This script is created for visualizing the features, i.e., the distributions
of the value for different features for each class. It is used for human to
figure out, for example, whether there is any difference of the feature value
between different classes, which feature is most important for certain class...

@author: guoqiangxu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KernelDensity

def load_train_data(path='../train.csv'):
    df = pd.read_csv(path)
    X = df.values
    X_train, y_train = X[:, 1:-1], X[:, -1]
    return X_train.astype(float), y_train.astype(str)

def calClassFreq(y_train,plot=True):
    """
    Calculate the counts of each class in the dataset (default is the train set).
    Plot the counts in a bar graph. Note that the distribution is non-uniform.
    """
    encoder = LabelEncoder().fit(y_train)
    classFreq = [(y_train==x).sum() for x in encoder.classes_]
    
    if plot==True:
        plt.figure()
        plt.bar(range(0,9), classFreq,align='center', alpha=0.4)
        plt.xticks(range(0,9), range(1,10))
        plt.xlabel('Classes')
        plt.ylabel('Counts')
    
    return classFreq
    
def getFeature(feature_index,X_train,y_train,plot=True):
    """
    Get the feature with index = feature_index for each class. The data is stored
    in the feature_data, with feature_data[0] containing the feature values in
    class 1. Plot the feature values for different class
    """
    encoder = LabelEncoder().fit(y_train)
    feature_data = []
    for i in range(0,9):
        feature = X_train[y_train==encoder.classes_[i],feature_index]
        feature_data.append([])
        feature_data[i] = feature
        
    if plot == True:
        plt.figure()
        plt.boxplot(feature_data,showmeans=True)
        plt.yscale('symlog',linthreshy=0.1)
        plt.ylim((0,100))
        plt.xticks(range(0,9), range(1,10))
        plt.xlabel('Classes')
        plt.ylabel('Feature value')
    
    return feature_data

def plotFeatureDist(feature_data,class_index):
    """
    Plot the probability density of feature for classs = class_index. The
    feature to be plotted is the one with index = feature_index obtained
    from getFeature function. The limit of x-axis is max(feature).
    """
    data = feature_data[class_index-1]
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(data[:,np.newaxis])
    X_plot = np.linspace(0,max(data),100)[:,np.newaxis]
    Y_plot = kde.score_samples(X_plot)
    plt.figure()
    plt.plot(X_plot, np.exp(Y_plot))
    plt.xlabel('Feature value')
    plt.ylabel('Probability density')
    
#X_train,y_train = load_train_data()
classFreq=calClassFreq(y_train)
feature_index = 21

feature_data=getFeature(feature_index,X_train,y_train)

class_index = 1
plotFeatureDist(feature_data,class_index)
