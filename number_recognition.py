#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File:   number_recognition.py
Author: Xiang Ji
Email:  xj4hm@virginia.edu
Date:   November, 2015
Brief:  Homework 3 OCR
Usage:  python number_recognition.py model trainData testData
'''


import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA

def loadData(file):
    data = np.loadtxt(file)
    number = data[:,0]
    pixels = data[:, range(1,256)]
    return number, pixels


def decision_tree(train, test):
    y = []
    trainY, trainX = loadData(train)
    testY, testX = loadData(test)
    
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best", random_state = 0)
    clf.fit(trainX, trainY)
    y = clf.predict(testX)
    error = 1 - clf.score(testX, testY)
    print 'Test error: ' + str(error)

    
    return y

def knn(train, test):
    y = []
    trainY, trainX = loadData(train)
    testY, testX = loadData(test)
    
    neigh = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
    neigh.fit(trainX, trainY)
    y = neigh.predict(testX)
    testError = 1 - neigh.score(testX, testY)
    trainError =  1 - neigh.score(trainX, trainY)
    print 'Test error: ' + str(testError)
    print 'Training error: ' + str(trainError)
    return y

def neural_net(train, test):
    y = []
    trainY, trainX = loadData(train)
    testY, testX = loadData(test)

    neuralNet = Perceptron()
    neuralNet.fit(trainX, trainY)
    y = neuralNet.predict(testX)

    testError = 1 - neuralNet.score(testX, testY)
    print 'Test error: ' + str(testError)
    return y

def svm(train, test):
    y = []
    trainY, trainX = loadData(train)
    testY, testX = loadData(test)

    
    svmClassifier = SVC(kernel = 'poly')
    svmClassifier.fit(trainX, trainY)
    y = svmClassifier.predict(testX)
        
    testError = 1 - svmClassifier.score(testX, testY)
    print 'Test error: ' + str(testError)
    
    return y

def pca_knn(train, test):
    y = []
    trainY, trainX = loadData(train)
    testY, testX = loadData(test)

    PCA = RandomizedPCA(n_components = 64)
    #n_components = dim
    PCA.fit(trainX)
    reducedTrainX = PCA.transform(trainX)
    reducedTestX = PCA.transform(testX)
    
    neigh = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
    neigh.fit(reducedTrainX, trainY)

    y = neigh.predict(reducedTestX)
    testError = 1 - neigh.score(reducedTestX, testY)
    print 'Test error: ' + str(testError)
    
    return y

def pca_svm(train, test):
    y = []
    trainY, trainX = loadData(train)
    testY, testX = loadData(test)

    PCA = RandomizedPCA(n_components = 64)
    
    PCA.fit(trainX)
    reducedTrainX = PCA.transform(trainX)
    reducedTestX = PCA.transform(testX)

    svmClassifier = SVC(kernel = 'poly')
    svmClassifier.fit(reducedTrainX, trainY)
    y = svmClassifier.predict(reducedTestX)
        
    testError = 1 - svmClassifier.score(reducedTestX, testY)
    print 'Test error: ' + str(testError)
    
    return y

if __name__ == '__main__':
    model = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]

    if model == "dtree":
        print(decision_tree(train, test))
    elif model == "knn":
        print(knn(train, test))
    elif model == "net":
        print(neural_net(train, test))
    elif model == "svm":
        print(svm(train, test))
    elif model == "pcaknn":
        print(pca_knn(train, test))
    elif model == "pcasvm":
        print(pca_svm(train, test))
    else:
        print("Invalid method selected!")
