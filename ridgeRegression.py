#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File:   ridgeRegression.py
Author: Xiang Ji
Email:  xj4hm@virginia.edu
Date:   October 2, 2015
Brief:  Homework 2 ridgeRegression and cross validation
Usage:  python 
'''
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import operator

def loadDataSet(txtfile):  
    absPath = os.path.abspath(txtfile)
    data = np.loadtxt(absPath, delimiter = "\t", dtype = 'S')
    matrix = np.zeros((len(data),4))
    for i in range(len(data)):
        matrix[i,] = np.fromstring(data[i], sep = ' ')
    xVal = matrix[:,[0, 1, 2]]
    yVal = matrix[:, 3]
    yVal = yVal[:,None]
    return xVal, yVal



'''
 function 1: ridge regression
'''
def ridgeRegress(xVal, yVal, Lambda):
    temp = np.dot(np.transpose(xVal), xVal) + Lambda * np.identity(len(xVal[1]))
    betaLR = np.dot(np.dot(np.linalg.inv(temp), np.transpose(xVal)), yVal)
    return betaLR


'''
draw the learning plane and the data points
'''
def showPlane():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    x1 = np.arange(-4.0, 4.0, 0.1)
    x2 = np.arange(-8.0, 8.0, 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    y = betaLR[0] + betaLR[1] * x1 + betaLR[2] * x2

    ax.set_xlabel('x1 Label')
    ax.set_ylabel('x2 Label')
    ax.set_zlabel('y Label')
    ax.plot_surface(x1,x2,y)
    
    ax.scatter(X[:,1], X[:,2], Y[:,0], c = 'r', s = 50, edgecolor='')
    plt.show()


def cv(xVal, yVal):
    sequence = list(range(len(xVal)))
    SEED = 37
    random.seed(SEED)
    random.shuffle(sequence)
    sequence = np.asarray(sequence)
    lambdaPoll = np.linspace(0.02,1,50)
    lossFunction = np.zeros((50,10))
    k_fold = 10
    sample_number_per_fold = int(len(xVal)/k_fold)
    for index in range(50):
        Lambda = lambdaPoll[index]
        for k in range(10):
            test_index_temp = np.linspace(k*sample_number_per_fold,(k+1)*sample_number_per_fold-1,sample_number_per_fold)
            test_index_temp = test_index_temp.astype(int)
            test_data_index = sequence[test_index_temp]
            index_temp = np.linspace(0,len(xVal)-1,len(xVal))
            index_temp = index_temp.astype(int)
            index_temp = np.delete(index_temp,test_index_temp)
            training_data_index = sequence[index_temp]
            xTraining = xVal[training_data_index,]
            yTraining = yVal[training_data_index,]
            xTesting = xVal[test_data_index,]
            yTesting = yVal[test_data_index,]
            beta = ridgeRegress(xTraining,yTraining,Lambda)
            lossFunction[index,k] = np.dot(np.transpose(yTesting-np.dot(xTesting,beta)),yTesting-np.dot(xTesting,beta))
    lossFunction = [sum(lossFunction[i]) for i in range(50)]
    bestLambdaIndex, min_loss = min(enumerate(lossFunction), key=operator.itemgetter(1))
    lambdaBest = lambdaPoll[bestLambdaIndex]
##    plt.plot(lambdaPoll, lossFunction)
##    plt.title('Lambda vs. Loss function')
##    plt.show()
    return lambdaBest


def standRegres(xVal, yVal):
    x = np.array(xVal)
    y = np.array(yVal)
    A = np.vstack([x, np.ones(len(x))]).T
    theta = np.linalg.lstsq(A, yVal)[0]
    xmin = min(xVal)
    xmax = max(xVal)
    ymin = theta[0] * float(xmin) + theta[1] 
    ymax = theta[0] * float(xmax) + theta[1]
    plt.scatter(x,y)
    plt.plot([xmin,xmax],[ymin,ymax])
    plt.title('x1 and x2 comparison')
    plt.show()
    return theta

def main():
    X, Y = loadDataSet('RRdata.txt')
    betaLR = ridgeRegress(X, Y, 0)
    print betaLR
    print cv(X, Y)

if __name__ == "__main__":
    main()
