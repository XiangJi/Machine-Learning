#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File:   clustering.py
Author: Xiang Ji
Email:  xj4hm@virginia.edu
Date:   November 30th, 2015
Brief:  Homework 5 K mean clustering and GMM clusting
Usage:  python clustering.py /DatasetDirectory
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math

def loadData(fileDj):
    data = np.loadtxt(fileDj)
    Y = data[:,-1]
    X = data[:,range(0,len(data[0]) - 1)]
    return X, Y

def objectFunction(clusters, label, X):
    object = 0.0
    for i in range(len(X)):
        index = label[i]
        object += math.pow(np.linalg.norm(X[i] - clusters[index]), 2.0)
    return object
    
def kmeans(X, k, maxIter):
    ''' k means clustering

    Args:
    X is the input data maxtrix;
    k is the number of clusters;
    max number of the iterations, max is 1000

    Returns: The result label vector
    
    '''
    clusters = X[np.random.choice(range(len(X)), k, replace=False)]
    
    iteration = 0
    while iteration < maxIter:
        preClusters = clusters.copy()
        label = []
        for x in X:
                label.append(np.argmin([np.linalg.norm(x - c) for c in preClusters]))
        for i in range(k):
                clusters[i] = sum(X[np.array(label) == i]) / len(X[np.array(label) == i])
        if np.all(clusters == preClusters):
                break
        iteration += 1
    return np.array(label), clusters
    
def purity(labels, trueLabels):
    trueResult = 0
    totalInstance = len(labels)
    for i in range(totalInstance):
        if labels[i] == trueLabels[i]:
            trueResult += 1
    purityMetric = trueResult / float(totalInstance)
    return purityMetric

def e_step(X, mu, p, sigma, K): 
    gamma = np.zeros((len(X), K))
    for i in range(len(X)):
        x = X[i]
        accum = np.zeros(K)
        for h in range(K):
            accum[h] = np.exp(-0.5 * np.inner(np.inner(x-mu[h], np.linalg.inv(sigma)), x-mu[h])) * p[h]
            gamma[i] = accum / sum(accum)
    return gamma

def m_step(X, gamma, K):  
    mu = np.zeros((K, len(X[0])))
    p = np.zeros(K)
    for h in range(K):
        numerator = np.zeros(len(X[0]))
        denominator = 0.0
        for i in range(len(X)):
                numerator += gamma[i, h] * X[i]
                denominator += gamma[i, h]
        mu[h] = numerator / denominator
        p[h] = denominator / len(X)
    return mu, p

def gmmCluster(X, k, covType, maxIter):
    '''Gaussian mixture model clustering

    Args:
    X is the input matrix
    k is supposed to be 2
    maxIter is the max number of iterations, max is 1000
    covariance Type can be "diag" or "full"
    
    '''
    if covType == 'diag':
        sigma = np.diag(np.diag(np.cov(X.T)))
    elif covType == 'full':
        sigma = np.cov(X.T)
    else :
        print "Error: Invalid covType, either diag or full."
        sys.exit()
            
    # initiate mu, p
    mu = X[np.random.choice(range(len(X)), k, replace=False)]
    p = np.ones(k) / k
    iteration = 0
    while iteration < maxIter:
        preMu = mu.copy()
        preP = p.copy()
        gamma = e_step(X, mu, p, sigma, k)
        mu, p = m_step(X, gamma, k)
        if np.sum(abs(mu - preMu)) / np.sum(abs(preMu)) < 0.0001:
            break
        iteration += 1
    label = [np.argmax(g) for g in gamma]
    return np.array(label), mu
    
    return labels

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python clustering.py dataSetDirectoryFullPath"
        sys.exit()
		
    directory = sys.argv[1]
    dataSet1 = os.path.join(directory, "humanData.txt")
    
    X1, Y1 = loadData(dataSet1)

    resultLabel, clusters = kmeans(X1, 2, 1000)

    # Q3 plot: draw the scatter plot of two clustering with different color.
    plt.scatter(X1[:, 0], X1[:, 1], c = resultLabel, alpha = 1.0)
    plt.show()  
    
    #Q4 plot k knee-finding where k = 1, 2, 3, 4, 5, 6, draw k versus objective function value
    objects = []
    for k in range(1, 7):
        label, clusters = kmeans(X1, k, 1000)
        objects.append(objectFunction(clusters, label, X1))
    plt.plot(range(1, 7), objects, '-o', linewidth = 2)
    plt.xlabel('K')
    plt.ylabel('Objective Function')
    plt.show()
	
    print "Kmeans purity: " + str(purity(resultLabel, Y1))

    
	
    #GMM question

    GMMlabel, mu = gmmCluster(X1, 2, 'diag', 1000)
##    plt.scatter(X1[:, 0], X1[:, 1], c = GMMlabel, alpha = 1.0)
##    plt.show()
    print 'GMM for Dataset1 purity: ' + str(purity(GMMlabel, Y1))
    
##    GMMlabel2, mu = gmmCluster(X1, 2, 'full', 1000)
##    plt.scatter(X1[:, 0], X1[:, 1], c = GMMlabel2, alpha = 1.0)
##    plt.show()
    
    dataSet2 = os.path.join(directory, "audioData.txt")
    X2, Y2= loadData(dataSet2)
    
    GMMlabel2, mu = gmmCluster(X2, 2, 'diag', 1000)
##    plt.scatter(X2[:, 0], X2[:, 1], c = GMMlabel2, alpha = 1.0, edgecolors = 'face')
##    plt.show()

    
    print 'GMM for Dataset2 purity: ' + str(purity(GMMlabel2, Y2))
    
