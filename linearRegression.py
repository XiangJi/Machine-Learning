#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File:   linearRegression.py
Author: Xiang Ji
Email:  xj4hm@virginia.edu
Date:   September 4, 2015
Brief:  load two-D dataset, plot the graph and the best fit line
Usage:  python linearRegression.py <textfile>
'''

import numpy as np
import sys
import matplotlib.pyplot as plt

'''
 function 1: load the dataset and plot the figure
'''
def loadDataSet(textfile):
    dataSet = open(textfile, 'r')
    xVal = []
    yVal = []
    for line in dataSet:
        currentList = line.split('\t')
        xVal.append(currentList[1])
        yVal.append(currentList[2])
    plt.scatter(xVal, yVal)
    plt.title('Dataset')
    plt.show()
    return (xVal, yVal)


'''
 function 2: choose one of the optimization methods,
 showing the dataset and the best-fit line, present the theta
'''

def standRegres(xVal, yVal):
    x = np.array(xVal)
    y = np.array(yVal)
    A = np.vstack([x, np.ones(len(x))]).T
    theta = np.linalg.lstsq(A, yVal)[0]
    print 'The theta is: ' + str(theta)
    xmin = min(xVal)
    xmax = max(xVal)
    ymin = theta[0] * float(xmin) + theta[1] 
    ymax = theta[0] * float(xmax) + theta[1] 
    plt.scatter(x,y)
    plt.plot([xmin,xmax],[ymin,ymax])
    plt.title('Dataset and best-fit line by least squared method')
    plt.show()
    return theta
    
def main():
    xVal,yVal = loadDataSet('Q4data.txt')
    theta = standRegres(xVal,yVal)


# only executed when running as a independent program but not a module
if __name__ == "__main__":
    main()





