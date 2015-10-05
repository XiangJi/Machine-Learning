#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File:   svmIncomeClassifier.py
Author: Xiang Ji
Email:  xj4hm@virginia.edu
Date:   October 4, 2015
Brief:  Homework 2 SVM for income classifier
Usage:  python
'''

import os
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC


def loadDataSet(txtfile):  
    absPath = os.path.abspath(txtfile)
    data = np.loadtxt(absPath, delimiter = ', ', dtype = 'S')
    return data

def dataPreprocess(dataSet):
    trainSet = loadDataSet(dataSet)
    '''
    Feature preprocessing: categorical feature for discrete data, LabelEncoder 
    1.transform categories of a particular feature into unique integers.
    2. use oneHotEncoder() function for encoding the categorical features.
    '''
    # workclass
    le = preprocessing.LabelEncoder()
    le.fit(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'])
    workclass = trainSet[:, 1]
    workclass = workclass.astype(str)
    workclass = le.transform(workclass)
    workclass = workclass[:, None] # transfer to column data
    # Dealing with the missing value by mean
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(workclass)
    workclass = imp.transform(workclass)
    workclass = np.rint(workclass)

    # re-encode
    le.fit(np.unique(workclass))
    workclass = le.transform(workclass)
    enc = OneHotEncoder()
    temp = np.array(range(len(le.classes_)))
    temp = temp[:,None]
    enc.fit(temp)
    workclass = enc.transform(workclass).toarray()
    

    
    # education
    le.fit(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool","?"])
    education = trainSet[:, 3]
    education = education.astype(str)
    education = le.transform(education)
    education = education[:, None] # transfer to column data
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(education)
    education= imp.transform(education)
    education = np.rint(education)
    le.fit(np.unique(education))
    education = le.transform(education)
    temp = np.array(range(len(le.classes_)))
    temp = temp[:,None]
    enc.fit(temp)
    education = enc.transform(education).toarray()

    # marital-status
    le.fit(["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse","?"])
    marital_status = trainSet[:,5]
    marital_status = marital_status.astype(str)
    marital_status = le.transform(marital_status)
    marital_status = marital_status[:, None] # transfer to column data
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(marital_status)
    marital_status= imp.transform(marital_status)
    marital_status = np.rint(marital_status)
    le.fit(np.unique(marital_status))
    marital_status = le.transform(marital_status)
    temp = np.array(range(len(le.classes_)))
    temp = temp[:,None]
    enc.fit(temp)
    marital_status = enc.transform(marital_status).toarray()

    # occupation
    le.fit(["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces","?"])
    occupation = trainSet[:,6]
    occupation = occupation.astype(str)
    occupation = le.transform(occupation)
    occupation = occupation[:, None] # transfer to column data
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(occupation)
    occupation= imp.transform(occupation)
    occupation = np.rint(occupation)
    le.fit(np.unique(occupation))
    occupation = le.transform(occupation)
    temp = np.array(range(len(le.classes_)))
    temp = temp[:,None]
    enc.fit(temp)
    occupation = enc.transform(occupation).toarray()


    # relationship
    le.fit(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried","?"])
    relationship = trainSet[:,7]
    relationship = relationship.astype(str)
    relationship = le.transform(relationship)
    relationship = relationship[:, None] # transfer to column data
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(relationship)
    relationship= imp.transform(relationship)
    relationship = np.rint(relationship)
    le.fit(np.unique(relationship))
    relationship = le.transform(relationship)
    temp = np.array(range(len(le.classes_)))
    temp = temp[:,None]
    enc.fit(temp)
    relationship = enc.transform(relationship).toarray()

    # race
    le.fit(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black","?"])
    race = trainSet[:,8]
    race = race.astype(str)
    race = le.transform(race)
    race = race[:, None] # transfer to column data
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(race)
    race= imp.transform(race)
    race = np.rint(race)
    le.fit(np.unique(race))
    race = le.transform(race)
    temp = np.array(range(len(le.classes_)))
    temp = temp[:,None]
    enc.fit(temp)
    race = enc.transform(race).toarray()
    

    # sex
    le.fit(["Female", "Male","?"])
    sex = trainSet[:,9]
    sex = sex.astype(str)
    sex = le.transform(sex)
    sex = sex[:, None] # transfer to column data
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(sex)
    sex= imp.transform(sex)
    sex = np.rint(sex)
    le.fit(np.unique(sex))
    sex = le.transform(sex)
    temp = np.array(range(len(le.classes_)))
    temp = temp[:,None]
    enc.fit(temp)
    sex = enc.transform(sex).toarray()

    # native_country
    le.fit(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands","?"])
    native_country = trainSet[:,13]
    native_country = native_country.astype(str)
    native_country = le.transform(native_country)
    native_country = native_country[:, None] # transfer to column data
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(native_country)
    native_country= imp.transform(native_country)
    native_country = np.rint(native_country)
    le.fit(np.unique(native_country))
    native_country = le.transform(native_country)
    temp = np.array(range(len(le.classes_)))
    temp = temp[:,None]
    enc.fit(temp)
    native_country = enc.transform(native_country).toarray()
    

    # Income output
    le.fit(["<=50K",">50K","?"])
    income = trainSet[:,14]
    income = income.astype(str)
    income = le.transform(income)
    income = income[:, None] # transfer to column data
    missed = np.nonzero(le.classes_== '?')[0][0]
    imp = Imputer(missing_values=missed)
    imp.fit(income)
    income= imp.transform(income)
    income = np.rint(income)
    le.fit(np.unique(income))
    income = le.transform(income)
    income = income[:,0] # row data

    

    # Feature preprocessing: scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    age = trainSet[:,0]
    age = age.astype(float)
    age = age[:,None]
    age = min_max_scaler.fit_transform(age)
    
    fnlwgt = trainSet[:,2]
    fnlwgt = fnlwgt.astype(float)
    fnlwgt = fnlwgt[:,None]
    fnlwgt = min_max_scaler.fit_transform(fnlwgt)
    
    
    education_num = trainSet[:,4]
    education_num = education_num.astype(float)
    education_num = education_num[:,None]
    education_num = min_max_scaler.fit_transform(education_num)

    capital_gain = trainSet[:,10]
    capital_gain = capital_gain.astype(float)
    capital_gain = capital_gain[:,None]
    capital_gain = min_max_scaler.fit_transform(capital_gain)

    capital_loss = trainSet[:,11]
    capital_loss = capital_loss.astype(float)
    capital_loss = capital_loss[:,None]
    capital_loss = min_max_scaler.fit_transform(capital_loss)

    hours_per_week = trainSet[:,12]
    hours_per_week = hours_per_week.astype(float)
    hours_per_week = hours_per_week[:,None]
    hours_per_week = min_max_scaler.fit_transform(hours_per_week)
    
    trainingData = np.concatenate((age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country),axis=1)
    return trainingData, income
  
    
def processDataSet(train, test):
    # apply three different kernels, select one based your demand by commenting out
    # Using hardcoded parameters, try random few
    # RBF
    trainSet, incomeTrain = dataPreprocess(train)
    testSet, incomeTest = dataPreprocess(test)

    Penalty = 1
    Gamma = 0
    # Linear
    # Polynomial
    # RBF
    clf = SVC(C = Penalty, gamma = Gamma, kernel='rbf')
    clf.fit(trainSet, incomeTrain)
    predictions = clf.predict(testSet)

    accuracy = score(predictions, incomeTest)
    print accuracy
    
    # change prediction from boolean to string
    lookup_table = np.array(['<=50K','>50K'])
    predictions = lookup_table[predictions]
    return predictions

'''
Test the result, should be rights predictions of len(test dataset) 
'''
def score(prediction, actualIncome):
    counter = 0
    for i in prediction:
        if (prediction[i] == actualIncome[i]):
            counter = counter + 1
    print counter
    print len(actualIncome)
    accuracy = counter / float(len(actualIncome))
    return accuracy



def main():
    processDataSet('adult.data', 'adult.test')
    


# only executed when running as a independent program but not a module
if __name__ == "__main__":
    main()
