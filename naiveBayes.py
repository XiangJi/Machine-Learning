#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
File:   naiveBayes.py
Author: Xiang Ji
Email:  xj4hm@virginia.edu
Date:   November 5, 2015
Brief:  Homework 3 Naive Bayes Classifier 
Usage:  python naiveBayes textDataSetsDirectoryFullPath testFileDirectoryFullPath

'''
import sys
import os
import numpy as np
import nltk
from sklearn.naive_bayes import MultinomialNB



###############################################################################

def transfer(fileDj, vocabulary):
    """Transfer txt file with comment into a vector based on given vocabulary
    Args:
        fileDj txt file
        vocabulary dictionary vector
        
    Returns:
        The vector count the frequency of words occures in vocabulary
    """
    BOWDj = []
    stemmer = nltk.stem.snowball.EnglishStemmer()
    doc = open(fileDj, 'r')
    comments = doc.read().lower()
    comments = nltk.word_tokenize(comments)
    stemComments = [stemmer.stem(word) for word in comments]
    BOWDj = np.zeros(len(vocabulary))
    
    for word in stemComments:
        if word in vocabulary:
            index = vocabulary.index(word)
            BOWDj[index] = int(BOWDj[index]) + 1
        else:
            BOWDj[0] = int(BOWDj[0]) + 1

    return BOWDj


def loadData(Path):

    # stem for original vocabulary
    vocabulary = ['UNKNOWN', 'love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst', 'stupid', 'waste', 'boring', '?', '!']
    stemmer = nltk.stem.snowball.EnglishStemmer()
    for i in range(1, 14):
        vocabulary[i] = stemmer.stem(vocabulary[i])
    stemVocabulary = vocabulary

    Xtrain = []
    ytrain = []

    Xtest = []
    ytest = []

    trainData = os.path.join(Path, "training_set")
    trainPos = os.path.join(trainData, "pos")
    trainNeg = os.path.join(trainData, "neg")
    
    testData = os.path.join(Path, "test_set")
    testPos = os.path.join(testData, "pos")
    testNeg = os.path.join(testData, "neg")

    for file in os.listdir(trainPos):
            filePath = os.path.join(trainPos, file)
            Xtrain.append(transfer(filePath, stemVocabulary))
            ytrain.append(1)
            
    for file in os.listdir(trainNeg):
            filePath = os.path.join(trainNeg, file)
            Xtrain.append(transfer(filePath, stemVocabulary))
            ytrain.append(-1)
            
    for file in os.listdir(testPos):
            filePath = os.path.join(testPos, file)
            Xtest.append(transfer(filePath, stemVocabulary))
            ytest.append(1)
            
    for file in os.listdir(testNeg):
            filePath = os.path.join(testNeg, file)
            Xtest.append(transfer(filePath, stemVocabulary))
            ytest.append(-1)

    Xtrain = np.asarray(Xtrain)
    ytrain = np.asarray(ytrain)
    Xtest = np.asarray(Xtest)
    ytest = np.asarray(ytest)
      
    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    dictLength = 15
    thetaPos = np.zeros(dictLength)
    thetaNeg = np.zeros(dictLength)
    posCount = 0
    negCount = 0
    
    for i in range(len(Xtrain)):
        localCount = sum(Xtrain[i])
        if ytrain[i] == 1:
            posCount += localCount
            for j in range(dictLength):
                thetaPos[j] += Xtrain[i,j]
        elif ytrain[i] == -1:
            negCount += localCount
            for j in range(dictLength):
                thetaNeg[j] += Xtrain[i,j]

    for i in range(dictLength):
        thetaPos[i] = (thetaPos[i] + 1) / (posCount + dictLength)
        thetaNeg[i] = (thetaNeg[i] + 1) / (negCount + dictLength)
    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    testPos = np.dot(Xtest, np.log(thetaPos).T)
    testNeg = np.dot(Xtest, np.log(thetaNeg).T)
    yPredict = ytest.copy()
    for i in range(len(Xtest)):
            if testPos[i] > testNeg[i]:
                    yPredict[i] = 1
            else:
                    yPredict[i] = -1
    Accuracy = sum(yPredict == ytest) / float(len(ytest))
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    Accuracy = clf.score(Xtest, ytest)

    return Accuracy


def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg):
    vocabulary = ['UNKNOWN', 'love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst', 'stupid', 'waste', 'boring', '?', '!']
    stemmer = nltk.stem.snowball.EnglishStemmer()
    for i in range(1, 14):
        vocabulary[i] = stemmer.stem(vocabulary[i])
    stemVocabulary = vocabulary
    
    vector = transfer(path, stemVocabulary)
    pos = np.inner(vector, np.log(thetaPos))
    neg = np.inner(vector, np.log(thetaNeg))
    if pos > neg:
        yPredict = 1
    else:
        yPredict = -1
    return yPredict
    


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg):
    yPredict = []
    testPos = os.path.join(path, "pos")
    testNeg = os.path.join(path, "neg")
    ytest = []
    
    for file in os.listdir(testPos):
            filePath = os.path.join(testPos, file)
            predict = naiveBayesMulFeature_testDirectOne(filePath, thetaPos, thetaNeg)
            yPredict.append(predict)
            ytest.append(1)       
    
    for file in os.listdir(testNeg):
            filePath = os.path.join(testNeg, file)
            predict = naiveBayesMulFeature_testDirectOne(filePath, thetaPos, thetaNeg)
            yPredict.append(predict)
            ytest.append(-1)

    yPredict = np.asarray(yPredict)
    ytest = np.asarray(ytest)
    Accuracy = sum(yPredict == ytest) / float(len(ytest))
    return yPredict, Accuracy


def naiveBayesBernFeature_train(Xtrain, ytrain):
    dictLength = 15
    thetaPosTrue = np.zeros(dictLength)
    thetaNegTrue = np.zeros(dictLength)
    posCount = 0
    negCount = 0
    
    for i in range(len(Xtrain)):
        localCount = sum(Xtrain[i])
        if ytrain[i] == 1:
            posCount += localCount
            for j in range(dictLength):
                if Xtrain[i,j] >= 1:
                    thetaPosTrue[j] += 1
        elif ytrain[i] == -1:
            negCount += localCount
            for j in range(dictLength):
                if Xtrain[i,j] >= 1:
                    thetaNegTrue[j] += 1

    for i in range(dictLength):
        thetaPosTrue[i] = (thetaPosTrue[i] + 1) / (posCount + 2)
        thetaNegTrue[i] = (thetaNegTrue[i] + 1) / (negCount + 2)

    return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    testPos = np.dot(Xtest, np.log(thetaPos).T)
    testNeg = np.dot(Xtest, np.log(thetaNeg).T)
    yPredict = ytest.copy()
    for i in range(len(Xtest)):
            if testPos[i] > testNeg[i]:
                    yPredict[i] = 1
            else:
                    yPredict[i] = -1
    Accuracy = sum(yPredict == ytest) / float(len(ytest))
    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()

    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]
    

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)
    

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg

    print "--------------------"


    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    
    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"
