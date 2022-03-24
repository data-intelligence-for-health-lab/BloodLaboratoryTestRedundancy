# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:01:41 2022

@author: Admin
"""


#THIS VERSION INCLUDES DIAGNOSIS VALUES PLUS vitals

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import os
import datetime as dt
from scipy.stats import t
import warnings
from scipy.spatial import distance
import numpy.matlib
from scipy.stats.distributions import chi2
from sklearn.covariance import MinCovDet
import datetime
from fuzzyTSModel_discriminator import FuzzyModelDiscriminator
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression 

import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


# MACHINE LEARNING APPROACES
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing  
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import EasyEnsembleClassifier    
from sklearn.metrics import f1_score, make_scorer

import auxiliarFeatures

import numpy.matlib

from scipy.stats import iqr
from sklearn.impute import KNNImputer, SimpleImputer



############################Connection to posgres ###########

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


# Function to check if a test is normal or abnormal
def isNormal( labId, labValue, gender, age ):
     
     if labId == 'I054':
         if age >55:
             if gender =='M':
                 if labValue >= 3.0 and labValue <= 9.0:
                     return 1
             else:
                 if labValue >= 3.0 and labValue <= 8.0:
                     return 1
         else:
             if gender =='M':
                 if labValue >= 3.0 and labValue <= 8.0:
                     return 1
             else:
                 if labValue >= 2.0 and labValue <= 7.0:
                     return 1
                 
                    
     if labId == 'I055':
         if age <= 60:
             if labValue >= 23 and labValue <= 29:
                 return 1
         elif age > 60 and age <= 90:
            if labValue >= 23 and labValue <= 31:
                 return 1
         elif age >  90:
            if labValue >= 20 and labValue <= 29:
                 return 1     
         
     if labId == 'I056':
         if gender =='M':
                 if age <60:
                     if labValue >= 80 and labValue <= 115:
                         return 1
                 else:
                     if labValue >= 71 and labValue <= 115:
                         return 1
         else:
              if age <60:
                     if labValue >= 53 and labValue <= 97:
                         return 1
              else:
                     if labValue >= 53 and labValue <= 106:
                         return 1
                 
     if labId == 'I057':
         if labValue > 3.3 and labValue <= 11.0:
             return 1
     
     if labId == 'I058':
         if labValue >= 3.5 and labValue <= 5.0:
             return 1
         
     if labId == 'I059':
         if age <=90:
             if labValue >= 136 and labValue <= 145 :
                 return 1
         else:
             if labValue >= 132 and labValue <= 146 :
                 return 1
     
     if labId == 'I060':
         if age <90:
             if labValue >= 136 and labValue <= 145 :
                 return 1
         else:
             if labValue >= 132 and labValue <= 146 :
                 return 1
     
     if labId == 'I061':
         if labValue >= 35 and labValue <= 45:
             return 1
     if labId == 'I062':
         if labValue >= 4.4 and labValue <= 6.1 :
             return 1
     
     if labId == 'I063':
          if labValue > 1 :
             if labValue >= 21 and labValue <= 5 :
                 return 1
          else:
              if labValue >= .21 and labValue <= .5 :
                 return 1
     
     if labId == 'I064':
         if labValue <= (age/4 + 4):
             return 1
     
     if labId == 'I065':
         if labValue >= 7.20 and labValue <= 7.40:
             return 1
         
            
     if labId == 'I066':
         if labValue >= 70 and labValue <= 90:
             return 1
         
     if labId == 'I067':
         if gender =='M':
             if labValue >= 138 and labValue <= 172:
                return 1
         else:
             if labValue >= 121 and labValue <= 151:
                 return 1
             
             
     if labId == 'I068':
         
         if gender =='M':
             if labValue > 1 :
                 if labValue >= 41.5 and labValue <= 50.4:
                    return 1
             else:
                 if labValue >= .415 and labValue <= .504:
                    return 1
         else:
             if labValue > 1 :
                 if labValue >= 35.9 and labValue <= 44.6:
                     return 1
             else:
                 if labValue >= .359 and labValue <= .446:
                    return 1
     
     if labId == 'I069':
         if labValue >= 3.5 and labValue <= 5.1:
             return 1
     
     if labId == 'I070':
         if labValue >= 30 and labValue <= 45:
             return 1
     
     if labId == 'I071':
         if labValue >= 40 and labValue <= 120:
             return 1
         
     if labId == 'I072':
         if gender =='M':
             if labValue  < 60:
                return 1
         else:
             if labValue < 40:
                 return 1
 
     
     if labId == 'I073':
         if gender =='M':
             if labValue >= 10 and labValue <= 40:
                return 1
         else:
             if labValue >= 9 and labValue <= 32:
                 return 1
 
     
     if labId == 'I074':
         if labValue >= 1.71 and labValue <= 20.5:
             return 1
     
     if labId == 'I075':
         if gender =='M':
             if labValue < 80 :
                     return 1
         else:
             if labValue < 50:
                     return 1
     
     if labId == 'I076':
         if gender =='M':
             if labValue > 1 :
                 if labValue >= 41.5 and labValue <= 50.4:
                    return 1
             else:
                 if labValue >= .415 and labValue <= .504:
                    return 1
         else:
             if labValue > 1 :
                 if labValue >= 35.9 and labValue <= 44.6:
                     return 1
             else:
                 if labValue >= .359 and labValue <= .446:
                    return 1
             
     if labId == 'I077':
         if gender =='M':
             if labValue >= 140 and labValue <= 175:
                return 1
         else:
             if labValue >= 123 and labValue <= 153:
                 return 1
     
     if labId == 'I078':
         if gender =='M':
             if labValue >= 4.5 and labValue <= 5.9:
                return 1
         else:
             if labValue >= 4.1 and labValue <= 5.1:
                 return 1
             
 
         
     if labId == 'I079':
         if labValue >= 4.5 and labValue <= 11:
                 return 1
             
             
     if labId == 'I080':
         if labValue <= 7.8:
                 return 1
             
             
     if labId == 'I081':
         if labValue >= 38 and labValue <= 42:
             return 1
         
         
     if labId == 'I082':
         if labValue >= 1.71 and labValue <= 20.5:
                 return 1
             
             
     if labId == 'I083':
         if labValue >= 22 and labValue <= 28:
             return 1
         
         
     if labId == 'I084':
         if labValue >= 0.5 and labValue <= 1:
             return 1
         
         
     if labId == 'I085':
         if labValue >= 60 and labValue <= 83:
             return 1
         
         
     if labId == 'I086':
         if labValue >= 22 and labValue <= 28:
             return 1
         
         
     if labId == 'I087':
         if labValue >= 11 and labValue <= 32:
             return 1
         
         
     if labId == 'I088':
         if labValue <= 100:
             return 1
         
         
     if labId == 'I089':
         if labValue <= 3:
             return 1
         
         
     if labId == 'I090':
         if labValue >= 5 and labValue <= 10:
             return 1
         
         
     if labId == 'I091':
         if labValue <= 5.6:
             return 1
         
         
     if labId == 'I092':
         if labValue >= 23 and labValue <= 29:
             return 1
         
 
 
     return 0
 


#Function to evaluate the machine learning models
def gmean_loss_func(y_true, y_pred):
    
    
    TP = np.sum( (y_true==1) & (y_pred==1) )
    TN = np.sum( (y_true==0) & (y_pred==0) )
    
    TPR = TP/ np.sum(y_true==1)
    TNR = TN/ np.sum(y_true==0)
    
    
    
    return np.sqrt( TPR*TNR ) # 


def calculatingConsecutiveNormalPrevTest( dfData_lab ):
    
    #This function takes the dataframe with admission for a lab, and determine if the previous lab was normal,
    #kepping the count of consecutiveNormalTest.
    
    #First, the lab of each admission are sorted by date. 
    
    
    ########## Selecting data for prior distribution
    
    dfData_lab_prior = dfData_lab[ ['admissionID', 'datatime', 'previousLabTestValue','labValue','gender','age'] ]
    #Removing rows with at least one nan value
    dfData_lab_prior = dfData_lab_prior.dropna()
    #reset index 
    dfData_lab_prior.reset_index(drop=True, inplace=True)
    #Sorting by date
    dfData_lab_prior = dfData_lab_prior.sort_values(['admissionID', 'datatime'], ascending=[True, True])
    
    
    
    #Finding if the lab tests are abnormal or normal  
    dfData_lab_prior['isNormal'] = dfData_lab_prior.apply( lambda row : isNormal(labTestID,row['labValue'], row['gender'],row['age'] )  , axis = 1)
    
    
    #consecutive normal test
    
    dicConsecutives = {}
    conscecutiveNormal = np.zeros ( len( dfData_lab_prior ) ) 
    previousTestNormal = -1*np.ones ( len( dfData_lab_prior ) ) 
    for iRow, row in dfData_lab_prior.iterrows():
        
        if iRow == 0:
            #first row(i.e., first admission, first lab )
            #get normality of the previous test
            previousTestNormal[iRow] = isNormal(labTestID,row['previousLabTestValue'], row['gender'],row['age'] )
            
            
            conscecutiveNormal[iRow] = previousTestNormal[iRow] 
            
            
        else:
            #Check if row has the the same admin than previous one (in that case we are in the same admission)
            
            if row[ 'admissionID' ] == dfData_lab_prior.iloc[ iRow-1][ 'admissionID' ]:
                #Same admission
                
                #previous test gives normality
                if dfData_lab_prior.iloc[ iRow-1][ 'labValue' ] == row['previousLabTestValue'] :
                    previousTestNormal[iRow] = dfData_lab_prior.iloc[ iRow-1][ 'isNormal' ]
                else:
                    previousTestNormal[iRow] = isNormal(labTestID,row['previousLabTestValue'], row['gender'],row['age'] )
                    
                # Increase the counter when previous test was normal
                if previousTestNormal[iRow]  == 0 :
                    conscecutiveNormal[iRow] = 0
                else:
                    #previous lab yielded normal
                    conscecutiveNormal[iRow] = 1 + conscecutiveNormal[iRow-1]
                    
                
                
            else:
                #This is a new admission
                #get normality of previous test #
                previousTestNormal[iRow] = isNormal(labTestID,row['previousLabTestValue'], row['gender'],row['age'] )
                
                
                #This is the first row of the admission 
                conscecutiveNormal[iRow] = previousTestNormal[iRow] 
                
        #Storing the consecutive normal test before the date for each admission
        # This allows knowing for an admission and a date, how many previous test were normal for the lab
        if row[ 'admissionID' ] in dicConsecutives:
            
            dicConsecutives[ row[ 'admissionID' ] ][ str(row['datatime']) ] = conscecutiveNormal[iRow]
        else:
            
            dicConsecutives[ row[ 'admissionID' ] ] = {}
            dicConsecutives[ row[ 'admissionID' ] ][ str(row['datatime']) ] = conscecutiveNormal[iRow]
    
    
    # Adding the two new cokumns
    dfData_lab_prior['isNormalPrevious']  = previousTestNormal
    dfData_lab_prior['conscecutiveNormal'] = conscecutiveNormal
    
    return dfData_lab_prior, dicConsecutives
######################################################################################################################################


#Loading data frame with patients information
# picke file path
pathFile = '' #For security we can not provide the data
file_to_read = open(pathFile, "rb")
dfDataAll = pickle.load(file_to_read)


#There are admission with nan in diagnosis amdission ## removing them
dfDataAllDiagnosis = dfDataAll.loc[ ~pd.isna( dfDataAll['diagnosis'] ) ].copy()
diagnosisCount = dfDataAllDiagnosis['diagnosis'].value_counts()

# Admission diagnosis are strings (e.g., 'Abdomen/extremity trauma',)
le = preprocessing.LabelEncoder()
le.fit( dfDataAllDiagnosis['diagnosis'] )
le.classes_

#Turning diagnosis into nominal variable (number from 0 to N-1)
dfDataAllDiagnosis['diagnosisLabel'] = le.transform( dfDataAllDiagnosis['diagnosis'] )
 
    

dfDataSel = dfDataAll#.iloc[ np.asarray(idxDiag) ]


#Variables related the patient and admission
idVariables = [ 'admissionID', 'patientsID', 'datatime', 'labtestID' ]

# Variables
numericalVariables = ['HR','SpO','Resp' ,'Temperature',
                      'Bp', 'previousLabTestValue',
                      'totalIntravenous','totalRedCells',
                      'totalPlasma','totalPlatelets',
                       'urineOutput' ,'labValue', 'firstValueDay','age' ] 
categoricalVariables =  ['gender', 'diagnosisLabel']


#Selecting only the relevant features
dfDataSel = dfDataAllDiagnosis[ idVariables + numericalVariables + categoricalVariables ] 

# These are the codes of the 18 blood labtests
labTestIdList = ['I065','I066','I061','I058','I077','I059','I076','I079' 
            ,'I055','I056', 'I054', 'I057','I072',
             'I074','I071', 'I070', 'I073','I075']

#labTestIdList = ['I075']

 
results = {}

for labTestID in labTestIdList:
    print("********************",labTestID,"********************")
    
    #Selecting the data corresponding to the current lab test
    dfData_lab = dfDataSel[ dfDataSel['labtestID']== labTestID ]
    
    
    #Selecting first value in the morning to use as a feature    
    colPrevValue = 'firstValueDay' 
    

    # Subset for the test #core columns for both Approaches
    dfData_lab = dfData_lab[ ['admissionID', 'HR','SpO','Resp' ,'Temperature',
                          'Bp', colPrevValue, 'previousLabTestValue',
                          'urineOutput',
                          'age','gender','labValue','datatime','diagnosisLabel'] ].copy()
                          
                          
    #Date column as datatime
    dfData_lab['datatime']= pd.to_datetime(dfData_lab['datatime'])
        
    
    
    # Calculating how many previous consecutive previous test were for each admission at each day
    dfData_lab_prior, dicConsecutives = calculatingConsecutiveNormalPrevTest( dfData_lab )

    


    ##############################################################################
    
    #Before drop nan
    uniqueAdmins = dfData_lab['admissionID'].unique()
    print('total unique admission for current lab', len( uniqueAdmins ) )
    
 
    # #Removing rows with at least nan values
    dfData_lab = dfData_lab.dropna()
   #  #reset indexes
    dfData_lab.reset_index(drop=True, inplace=True)
    #after drop nan
    uniqueAdminsAfter = dfData_lab['admissionID'].unique()
    print('# admin after removing nan values', len( uniqueAdminsAfter ) )
    
    
    # Finding target class 1:abnormal 0:normal
    dfData_lab['target'] = 1 - dfData_lab.apply( lambda row : isNormal(labTestID,row['labValue'], row['gender'],row['age'] )  , axis = 1)
    
    #Sex of each patinet (1:male;0:female)
    genderPat = np.zeros( len(dfData_lab ) )
    genderPat[ dfData_lab['gender'] == 'M' ] = 1
    dfData_lab['gender'] = genderPat
    
    
    
    ############# Spliting the data into 10 folds ###################
    
    
    #This is used for the fold split
    admissionInfo = pd.DataFrame()
    # Number of labs for each admission
    totalRecords = dfData_lab[ ['admissionID','target'] ].groupby('admissionID').count()
    # Number of abnormal labs for each admission
    totalAbnormal = dfData_lab[ ['admissionID','target'] ].groupby('admissionID').sum()
    
    #Auxiliar dataframe to split the data in 10 folds
    admissionInfo['admissionID'] = totalRecords.index
    admissionInfo['totalRecords']  = totalRecords['target'].values
    admissionInfo['totalAbnormal']  = totalAbnormal['target'].values
    admissionInfo['totalNormal']  =  admissionInfo['totalRecords'] - admissionInfo['totalAbnormal'] 
    
    
    
    folds = 10
    

    # Split admissions in folds
    
    stopSplit = False
    seedIdx = 0
    
    while not stopSplit:
        #Shuffle the number of admissions
        np.random.seed(42 + seedIdx*5)
        indices = np.random.permutation( len( uniqueAdminsAfter ) )
        
        seedIdx +=1
        #Number of admission per fold
        totalAdmissionPerFold = int( np.floor ( len( uniqueAdminsAfter )/ folds ) )
    
    
        admissionPerFold = {}
        #matrix to count admission per folds -> counting how many abnormal/normal
        countSamplesPerFold = np.zeros( [ folds, len( np.unique( dfData_lab['target'].values ) ) ] )
        startIdx = 0
        foldIdx = 0
        while foldIdx < folds:
            
            endIdx = startIdx+totalAdmissionPerFold
            # selecting admissions for the current fold
            # the labs of an admission only belong to 1 fold (addmission are mutually exclusing across folds)
            if foldIdx < folds-1:
                
                foldAdmission = uniqueAdminsAfter[ indices[startIdx: endIdx ] ]
            else:
                foldAdmission = uniqueAdminsAfter[ indices[startIdx:] ]
            
            
            selAdmissions = admissionInfo[ admissionInfo['admissionID'].isin(foldAdmission) ]
            admissionPerFold[ foldIdx ] = selAdmissions['admissionID'].values
            
            #Counting how many abnormal and normal lab values are in the fold
            countSamplesPerFold[ foldIdx, 0] = np.sum(selAdmissions['totalNormal'] )
            countSamplesPerFold[ foldIdx, 1] = np.sum(selAdmissions['totalAbnormal'] )
            
            
            foldIdx+=1
            startIdx = endIdx
            
        #Check if the split is valid
        #There are not folds with 0 samples of each class
        if np.sum( countSamplesPerFold == 0 ) == 0:
            
            #proportion of each fold
            proportionFold = countSamplesPerFold[ :, 1] / np.sum( countSamplesPerFold , axis=1 )
            
            medianProp = np.median( proportionFold )
            iqrProp = iqr( proportionFold)
            fstProp = np.quantile( proportionFold, .25)
            lowOut = fstProp - 3*iqrProp
            
            #All folds have a similar class proportion - there is not low outliers for the class proportion
            if np.all( proportionFold >= lowOut ) :
                stopSplit = True
    
    
    ##############################################################################  
    
    
    
    #Selecting only features columns for the classifiers
    #Fist numerical variables
    numerical_columns = ['HR','SpO','Resp' ,'Temperature',
                    'Bp', colPrevValue, 'urineOutput' , 'age']

    #Then categorical variable
    categorical_columns = ['gender' , 'diagnosisLabel'] 
    
    finalColumns = numerical_columns +   categorical_columns
                           
    columnsForEntropy = ['HR','SpO','Resp' ,'Temperature',
                    'Bp', colPrevValue, 'urineOutput' , 'age', 'gender', 'diagnosisLabel']


    
    
    
    #Metrics for comparison
    specificityFolds = np.empty( folds)
    sensitivityFolds = np.empty( folds)
    accuracyFolds = np.empty( folds)
    precision = np.empty( folds)
    precisionFalse = np.empty( folds)
    f1_scoreFolds = np.empty( folds)
    aucFolds = np.empty( folds)
    aucPrecisionFolds = np.empty( folds)
    gmeanFolds = np.empty( folds)
    ibaFolds = np.empty( folds)
    alphaIBA = 0.1
    # dict to store selected features
    selectedFeaturesFold = {}
    
    falsePositiveFolds = np.empty( folds)
    falseNegativeFolds = np.empty( folds)
    yPred = np.empty( len( dfData_lab) )
    
    
    approach = 1
    
    if approach ==1:
        totalFeatures = len(finalColumns)
    else:
        totalFeatures = len(finalColumns) + len(columnsForEntropy) + 1
    
    foldIdx = 0    
   
    featureRanking = np.ones( [folds, totalFeatures])*totalFeatures
   
    # Iterate over the folds
    while foldIdx < folds:
    
        #Getting the admission of the current fold
        foldAdmins = admissionPerFold[ foldIdx ]
        
        #Training and test indexes
        idxTrain = np.where( ~ dfData_lab['admissionID'].isin(foldAdmins) )[0]
        idxTest = np.where( dfData_lab['admissionID'].isin(foldAdmins) )[0]
        
        # Train and test data
        X_train = dfData_lab.iloc[ idxTrain ][finalColumns]#.to_numpy()
        X_test = dfData_lab.iloc[ idxTest ][finalColumns]#.to_numpy()
        # Train and test targets
        y_train = dfData_lab.iloc[ idxTrain ]['target'].to_numpy()
        y_test = dfData_lab.iloc[ idxTest ]['target'].to_numpy()
        # Train and test admissions
        admissions_train = dfData_lab.iloc[ idxTrain ]['admissionID'].to_numpy()
        admissions_test = dfData_lab.iloc[ idxTest ]['admissionID'].to_numpy()
        # Train and test laboratory dates
        dateLab_train = dfData_lab.iloc[ idxTrain ]['datatime'].to_numpy()
        dateLab_test = dfData_lab.iloc[ idxTest ]['datatime'].to_numpy()
    

   
        if approach ==2:
        
            # Feature engineering for Appraoach 2
            
            ###### Generate prior probability using data from training set ##########
            admissionsTrain = dfData_lab_prior[  ~ dfData_lab_prior['admissionID'].isin(foldAdmins) ]
            
            daysDict, xValues,  yValues = auxiliarFeatures.getPre_probAdmission( admissionsTrain )
            
            priorTrain, priorTest = auxiliarFeatures.calculatePrior( daysDict,  dicConsecutives,
                                                      admissions_train, dateLab_train,   
                                                      admissions_test, dateLab_test)
            
            
            
            newFeatureName = 'prior_probability'
            X_train[ newFeatureName ] = priorTrain
            X_test[  newFeatureName  ] = priorTest
            if not newFeatureName in numerical_columns:
                numerical_columns.append( newFeatureName )
            
            
            
            
            totalFeat = len( columnsForEntropy)
           
            
            for varIdx in range( totalFeat ):
                trainCondEntr, testCondEntr = auxiliarFeatures.getConditionalProbability( y_train, X_train[ columnsForEntropy[ varIdx ] ].to_numpy() , 
                                                                                           X_test[ columnsForEntropy[ varIdx ] ].to_numpy() , 
                                                                                           variableIsCat = columnsForEntropy[ varIdx ] in categorical_columns   )
                
                newFeatureName = columnsForEntropy[ varIdx ]+'_entropy'
                X_train[ newFeatureName ] = trainCondEntr
                X_test[  newFeatureName  ] = testCondEntr
                if not newFeatureName in numerical_columns:
                    numerical_columns.append( newFeatureName )
    
        
            
        
    
        
        
        sortedFeatures = numerical_columns +   categorical_columns
        
        
        
        # reorder columns to make sure that categorical columns are at the end
        X_train = X_train[ sortedFeatures ]
        X_test = X_test[ sortedFeatures ]
        
        
        
        # #Initiate a fuzzy model
        fuzzyModel = FuzzyModelDiscriminator(   )
        
        # Selecting best features and number of clusters
        dictGroupParameters = fuzzyModel.findingBestFeatures_val_aux(  X_train, y_train, numerical_columns, categorical_columns )
        
        selectedFeaturesFold[foldIdx] = dictGroupParameters
        
        #Training fuzzy model
        parameterModels = fuzzyModel.trainFuzzyModelClasses(  X_train, y_train, dictGroupParameters[0], numerical_columns, categorical_columns, dictGroupParameters[1]  )
        
        _, yHat, predProb = fuzzyModel.predict( X_test,  y_test, dictGroupParameters[0], parameterModels[0], 
                                               parameterModels[1], numerical_columns, categorical_columns, np.arange( len(numerical_columns) ), 
                                               parameterModels[2], parameterModels[3]   )
            
        
      
        
        C1 = confusion_matrix( y_test ,yHat  ) 
        print( 'fold', foldIdx )
        print(  C1.astype('float') / C1.sum(axis=1)[:, np.newaxis] )
    
    
        specificityFolds[foldIdx] = C1[0,0] / np.sum( C1[0,:]) 
        sensitivityFolds[foldIdx] = C1[1,1] / np.sum( C1[1,:])  
        
    
        falsePositiveFolds[foldIdx] = C1[0,1] / np.sum( C1[0,:])  
        falseNegativeFolds[foldIdx] = C1[1,0] / np.sum( C1[1,:])  
        
        
        accuracyFolds[foldIdx] = ( C1[1,1] + C1[0,0] ) / np.sum( C1)
        precision[foldIdx] = C1[1,1] / np.sum( C1[:,1])
        precisionFalse[foldIdx] = C1[0,0] / np.sum( C1[:,0])  
        f1_scoreFolds[foldIdx] = f1_score(y_test ,yHat )
        aucFolds[foldIdx] = roc_auc_score(y_test ,predProb[:,1] )
        precisionAUC, recallAUC, _ = precision_recall_curve(y_test ,predProb[:,1] )
        aucPrecisionFolds[foldIdx] = auc(recallAUC, precisionAUC)
        gmeanFolds[foldIdx] = np.sqrt( specificityFolds[foldIdx] * sensitivityFolds[foldIdx] )
        ibaFolds[foldIdx] = (1 + alphaIBA*( sensitivityFolds[foldIdx] - specificityFolds[foldIdx] ) )* ( gmeanFolds[foldIdx]**2 )
       
    
        #### Features ##########
        selFeatures = dictGroupParameters[0]
        rankFeatures = dictGroupParameters[2] 
        
        for idxFeature in range( len( rankFeatures) ):
            featureRanking[foldIdx, selFeatures[ rankFeatures[ idxFeature] ] ] = idxFeature+1
        
            
    
        yPred[idxTest] = yHat
    
        foldIdx+=1


    CM = confusion_matrix( dfData_lab['target'] ,yPred  ) 
    cm = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]
    print( cm  )
    print( f1_score(  y_test ,yHat, average='macro'   ) )
    print("specificity: ", CM[0,0] / np.sum( CM[0,:]) ) 
    print("sensitivity: ", CM[1,1] / np.sum( CM[1,:]) ) 

    print("false positive ", CM[0,1] / np.sum( CM[0,:]) ) 
    print("false negative: ", CM[1,0] / np.sum( CM[1,:]) ) 
    print( np.trace( CM )   / np.sum( CM)  ) 
    

    redExams = ( np.sum( CM) - ( CM[0,1] + CM[1,1] ) ) / np.sum( CM)
    print( 'exam predicted as positives', ( CM[0,1] + CM[1,1] ) , 'perReduction', redExams )
    print( 'speficifity folds', np.mean( specificityFolds ), np.std( specificityFolds ) )
    print( 'sensitivity folds', np.mean( sensitivityFolds ), np.std( sensitivityFolds ) )
    print( 'false positives folds', np.mean( falsePositiveFolds ), np.std( falsePositiveFolds ) )
    print( 'false negatives folds', np.mean( falseNegativeFolds ), np.std( falseNegativeFolds ) )

    rankingFeatures = np.mean( featureRanking, axis=0 )
    idxRank = np.argsort( rankingFeatures )
    rankFeat = rankingFeatures[ idxRank ] 


    results[labTestID] = [ CM, sensitivityFolds, specificityFolds, accuracyFolds, precision,
                          precisionFalse, f1_scoreFolds, aucFolds, aucPrecisionFolds, 
                          gmeanFolds, ibaFolds, selectedFeaturesFold, [ idxRank, rankFeat] ]



    #In case is stopped in the server
    
    
# with open('yfuzzyModeling_rewwv_v1.pickle', "wb") as f:
#     pickle.dump(results, f)
                    
   
    

