# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 06:20:20 2022

@author: Admin
"""

##### IMPORTING PACKAGES ########################
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
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



######################################################################################################
#Loading data frame with patients information

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
 
    

#columnsLabs = [ colLab for colLab in dfDataAllDiagnosis.columns if colLab.startswith('I0') ]

#dfDataAllDiagnosis[columnsLabs] = dfDataAllDiagnosis[columnsLabs].fillna(-1)





#gmean_score = make_scorer(gmean_loss_func)

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
                    'Bp', colPrevValue, 'urineOutput' , 'age','gender' , 'diagnosisLabel']       
    
    #catVarList = [ len(finalColumns) - 1 ] #[ len(finalColumns) - 2 , len(finalColumns) - 1 ] # 
    

    
    
    ###### MACHINE LEARNING APPROACES
    
    
    # Class to enconde categorival variables into numerical values for 
    # the machine learning models.
    # This procedure was adapted from:
    #Lopez-Arevalo I, Aldana-Bobadilla E, Molina-Villegas A, Galeana-Zapién H, Muñiz-Sanchez V, 
    # Gausin-Valle S. A memory-efficient encoding method for processing mixed-type data 
    #on machine learning. Entropy. 2020 Dec;22(12):1391.
    class CategoricalTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            super().__init__()
    
        # Return self nothing else to do here
        def fit(self, X, y=None):
            
            categoricalVariables = X.columns
            #Iterate over the columns
            dictVariables = {}
            for colName in categoricalVariables:
                
                uniqueVal, counts = np.unique( X[colName] , return_counts=True) 
            
                
                frequencyRelative = ( counts/ np.sum(counts) ) #+np.finfo(float).eps
                
                entropyVariable = -1*np.dot(frequencyRelative, np.log2( frequencyRelative ) )
                
                dictValues = {}
                for iUnique in range( len( uniqueVal) ):
                    
                    individualValue = ( counts[ iUnique ] / np.sum(counts) ) * -np.log2( counts[ iUnique ] / np.sum(counts) ) 
                    dictValues[ uniqueVal[ iUnique ] ] = individualValue/entropyVariable
                    
                    
                    
                dictVariables[colName] = dictValues
    
            
            self.dictVariables = dictVariables
            
            return self
    

    
        # Transformer method for this transformer
        def transform(self, X, y=None):
            
            XsetModified = X.copy()
            categoricalVariables = X.columns
            for colName in categoricalVariables:
                dictValues = self.dictVariables[colName] 
        
                XsetModified[colName] = 0
                for keyValue in dictValues:
            
                    XsetModified.loc[ X[colName] == keyValue , colName] = dictValues[keyValue]
                
            return XsetModified
    
    
    # transform to scale/econded variables for the machine learning
    # standarized is used for the numerical values
    # categorical variables are enconded using a entropy technique
    preprocessor = ColumnTransformer([ ('standarizer', preprocessing.MinMaxScaler(),numerical_columns ),
                                       ('catTrans',  CategoricalTransformer(), categorical_columns )])# OneHotEncoder(handle_unknown="ignore"), categorical_columns )]) # 
    
    
    n_jobs = 10# int(os.environ['SLURM_CPUS_PER_TASK'])
    
    ## LOGISTIC REGRESSION
    
    #Definig grid search for logistic regression
    
    cSet = [  .1, .5, 1, 5, 10 , 50]
    l1Ratio = [0, .25, .5, .75,  1]
    
    lr_grid_cv = dict(lr__C = cSet, lr__l1_ratio = l1Ratio)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    
    # Defining pipeline 
    lr_pipeline = Pipeline( [ ( "standarizer", preprocessor ), 
                              ( "lr",LogisticRegression(random_state=0, max_iter= 1e4, solver='saga', penalty='elasticnet', class_weight='balanced' ) ) ] ) 
    
    # Grid search for nested-cross validation
    clf_lr = GridSearchCV( lr_pipeline, param_grid=lr_grid_cv, cv=inner_cv, scoring=make_scorer(gmean_loss_func) ,n_jobs=n_jobs)
    
    ### Random forest ###########
    n_estimators = [300, 500 , 800] 
    max_depth = [8, 15 , 25]
    min_samples_split = [5, 10]
    min_samples_leaf = [2, 5 ]
    max_features = ['sqrt', 'log2', 1] 
    
    
    p_grid_cv = dict(rf__n_estimators = n_estimators, 
                  rf__max_depth = max_depth,  
                  rf__min_samples_split = min_samples_split, 
                  rf__min_samples_leaf = min_samples_leaf,
                  rf__max_features = max_features)
    
    
    # Non_nested parameter search and scoring
    rf_pipeline = Pipeline( [ ( "standarizer",preprocessor ),
                              ( "rf",RandomForestClassifier( class_weight='balanced') ) ] ) 
    
    
    clf_rf = GridSearchCV( rf_pipeline, param_grid=p_grid_cv, cv=inner_cv, scoring=make_scorer(gmean_loss_func) ,n_jobs=n_jobs)
    

    
    ### Gradiant bost #########
    n_estimators = [300, 500 , 800] 
    learningRates = [ 0.01, 0.05 , .1]
    max_features = ['sqrt', 'log2', 1] 
    
    
    gb_grid_cv = dict(gb__n_estimators = n_estimators, 
                  gb__learning_rate = learningRates,  
                  gb__max_features = max_features)
    
    
    # Non_nested parameter search and scoring
    gb_pipeline = Pipeline( [ ( "standarizer", preprocessor ),
                              ( "sampling" , RandomOverSampler(sampling_strategy='not majority')  ),
                              ( "gb",GradientBoostingClassifier() ) ] ) 
    
    
    clf_gb = GridSearchCV( gb_pipeline, param_grid=gb_grid_cv, cv=inner_cv, scoring=make_scorer(gmean_loss_func), n_jobs=n_jobs )



    #For oversampling 
    ros = RandomOverSampler(sampling_strategy='not majority')
    scalerMaxMin = preprocessing.MinMaxScaler()
    
    # To store the performances across the folds
    yPredML = np.empty( [ len(dfData_lab),3 ])
    specificityML = np.empty( [3, folds ])
    sensitivityML = np.empty( [3, folds ])
    accuracyFolds = np.empty( [3, folds ])
    precision = np.empty( [3, folds ])
    precisionFalse = np.empty( [3, folds ])
    f1_scoreFolds = np.empty( [3, folds ])
    aucFolds = np.empty( [3, folds ])
    aucPrecisionFolds = np.empty( [3, folds ])
    gmeanFolds = np.empty( [3, folds ])
    ibaFolds = np.empty( [3, folds ])
    alphaIBA = .1
    
    areEntropyFeat = 1 
    
    approach = 2
    
    if approach ==1:
        totalFeatures = len(finalColumns)
    else:
        totalFeatures = len(finalColumns) + len(columnsForEntropy) + 1
    
    #To store the relevant features
    featuresLR = np.ones( [folds,  totalFeatures])*totalFeatures
    featuresRF = np.ones( [folds,  totalFeatures])*totalFeatures
    featuresGB = np.ones( [folds,  totalFeatures])*totalFeatures
    
    foldIdx = 0
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
            
        
        
        
        
        ## Scaling features 
        X_train_scaled = preprocessor.fit_transform( X_train )
        X_test_scaled = preprocessor.transform( X_test )
        sortedFeatures = numerical_columns +   categorical_columns
        
        #### Training model
        clf_lr.fit( X_train, y_train )
        bestParameters = clf_lr.best_params_
        print(bestParameters )
        
        C = bestParameters['lr__C']
        l1_ratio =  bestParameters['lr__l1_ratio']
        
        lr = LogisticRegression(random_state=0, max_iter= 1e9 ,C=C, l1_ratio =l1_ratio, solver='saga', penalty='elasticnet', class_weight='balanced')
        lr.fit(X_train_scaled, y_train)
        
        coefLR = lr.coef_
   
        ################### Random forest
        clf_rf.fit( X_train, y_train )
        bestParametersRF = clf_rf.best_params_
        print(bestParametersRF )
        
    
        max_depth = bestParametersRF['rf__max_depth']
        max_features = bestParametersRF['rf__max_features']
        min_samples_leaf = bestParametersRF['rf__min_samples_leaf']
        min_samples_split = bestParametersRF['rf__min_samples_split']
        n_estimators = bestParametersRF['rf__n_estimators']
    
        rf = RandomForestClassifier(max_depth=max_depth, 
                                max_features=max_features,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                n_estimators= n_estimators,
                                class_weight='balanced') 
        
        rf.fit(X_train_scaled, y_train)
    
        ######## Gradient boost
        
        clf_gb.fit( X_train, y_train )
        
        
        bestParametersGB = clf_gb.best_params_
        print(bestParametersGB )
    
        n_estimators = bestParametersGB['gb__n_estimators']
        learning_rate = bestParametersGB['gb__learning_rate']
        max_features = bestParametersGB['gb__max_features']
    
    
    
    
        X_oversampled, y_oversampled = ros.fit_resample( X_train_scaled, y_train)
        
        
        gb = GradientBoostingClassifier(learning_rate=learning_rate, 
                                    max_features=max_features,
                                    n_estimators= n_estimators ) 
        gb.fit(X_oversampled, y_oversampled)
        
        
        ########### Predicting ##########
        
        predLR = lr.predict( X_test_scaled )
        
        predRF = rf.predict( X_test_scaled )
        
        predGB = gb.predict( X_test_scaled )
        
        predLR_prob = lr.predict_proba( X_test_scaled )
        
        predRF_prob = rf.predict_proba( X_test_scaled )
        
        predGB_prob = gb.predict_proba( X_test_scaled )
        
        
        
        CM_lr = confusion_matrix( y_test, predLR )
        CM_rf = confusion_matrix( y_test, predRF )
        CM_gb = confusion_matrix( y_test, predGB )
        
        print( 'lr\n', CM_lr/ CM_lr.sum(axis=1)[:, np.newaxis] )
        
        print( 'rf\n',  CM_rf/ CM_rf.sum(axis=1)[:, np.newaxis] )
        
        print( 'gb\n',  CM_gb/ CM_gb.sum(axis=1)[:, np.newaxis] )
        
        
        ### Storing the results ######
        
        specificityML[ 0,  foldIdx ] = np.sum( ( y_test==0 ) & ( predLR==0 ) )/  np.sum( y_test==0 )
        specificityML[ 1,  foldIdx ] = np.sum( ( y_test==0 ) & ( predRF==0 ) )/  np.sum( y_test==0 )
        specificityML[ 2,  foldIdx ] = np.sum( ( y_test==0 ) & ( predGB==0 ) )/  np.sum( y_test==0 )
        
        sensitivityML[ 0,  foldIdx ] = np.sum( ( y_test==1 ) & ( predLR==1 ) )/  np.sum( y_test==1 )
        sensitivityML[ 1,  foldIdx ] = np.sum( ( y_test==1 ) & ( predRF==1 ) )/  np.sum( y_test==1 )
        sensitivityML[ 2,  foldIdx ] = np.sum( ( y_test==1 ) & ( predGB==1 ) )/  np.sum( y_test==1 )
        
        
        yPredML[ idxTest, 0 ] = predLR
        yPredML[ idxTest, 1 ] = predRF
        yPredML[ idxTest, 2 ] = predGB    
        
               
        
        accuracyFolds[0,foldIdx] = ( CM_lr[1,1] + CM_lr[0,0] ) / np.sum( CM_lr)
        accuracyFolds[1,foldIdx] = ( CM_rf[1,1] + CM_rf[0,0] ) / np.sum( CM_rf)
        accuracyFolds[2,foldIdx] = ( CM_gb[1,1] + CM_gb[0,0] ) / np.sum( CM_gb)
        
        
        precision[0,foldIdx] = CM_lr[1,1] / np.sum( CM_lr[:,1])
        precision[1,foldIdx] = CM_rf[1,1] / np.sum( CM_rf[:,1])
        precision[2,foldIdx] = CM_gb[1,1] / np.sum( CM_gb[:,1])
        
        
        precisionFalse[0,foldIdx] = CM_lr[0,0] / np.sum( CM_lr[:,0])  
        precisionFalse[1,foldIdx] = CM_rf[0,0] / np.sum( CM_rf[:,0]) 
        precisionFalse[2,foldIdx] = CM_gb[0,0] / np.sum( CM_gb[:,0]) 
        
        f1_scoreFolds[0,foldIdx] = f1_score(y_test ,predLR )
        f1_scoreFolds[1,foldIdx] = f1_score(y_test ,predRF )
        f1_scoreFolds[2,foldIdx] = f1_score(y_test ,predGB )
        
        aucFolds[0,foldIdx] = roc_auc_score(y_test ,predLR_prob[:,1] )
        aucFolds[1,foldIdx] = roc_auc_score(y_test ,predRF_prob[:,1] )
        aucFolds[2,foldIdx] = roc_auc_score(y_test ,predGB_prob[:,1] )
        
        precisionAUC_lr, recallAUC_lr, _ = precision_recall_curve(y_test ,predLR_prob[:,1] )
        precisionAUC_rf, recallAUC_rf, _ = precision_recall_curve(y_test ,predRF_prob[:,1] )
        precisionAUC_gb, recallAUC_gb, _ = precision_recall_curve(y_test ,predGB_prob[:,1] )
        
        aucPrecisionFolds[0,foldIdx] = auc(recallAUC_lr, precisionAUC_lr)
        aucPrecisionFolds[1,foldIdx] = auc(recallAUC_rf, precisionAUC_rf)
        aucPrecisionFolds[2,foldIdx] = auc(recallAUC_gb, precisionAUC_gb)
        
        gmeanFolds[0,foldIdx] = np.sqrt( sensitivityML[0,foldIdx] * specificityML[0,foldIdx] )
        gmeanFolds[1,foldIdx] = np.sqrt( sensitivityML[1,foldIdx] * specificityML[1,foldIdx] )
        gmeanFolds[2,foldIdx] = np.sqrt( sensitivityML[2,foldIdx] * specificityML[2,foldIdx] )
        
        ibaFolds[0,foldIdx] = (1 + alphaIBA*( sensitivityML[0,foldIdx] - specificityML[0,foldIdx] ) )* ( gmeanFolds[0,foldIdx]**2 )
        ibaFolds[1,foldIdx] = (1 + alphaIBA*( sensitivityML[1,foldIdx] - specificityML[1,foldIdx] ) )* ( gmeanFolds[1,foldIdx]**2 )
        ibaFolds[2,foldIdx] = (1 + alphaIBA*( sensitivityML[2,foldIdx] - specificityML[2,foldIdx] ) )* ( gmeanFolds[2,foldIdx]**2 )
    
    
        ######### Finding the most relevant features
        
        coeffFeatures = np.abs(lr.coef_)
    
               
        # #LR 
        idxSort = np.argsort( coeffFeatures )[0]
        idxSort = idxSort[-1::-1]
        
        for idxFeature in range( len( idxSort ) ):
            featuresLR[foldIdx, idxSort[ idxFeature] ] = idxFeature+1
        
        idxMean = np.mean( featuresLR, axis=0 )
        idxSortFeatures = np.argsort( idxMean ) 
        for idxFeature in idxSortFeatures:
            if idxFeature< len(sortedFeatures):
                print(sortedFeatures[idxFeature] )
        
        # RF
        importances = rf.feature_importances_
        idxSortRf = np.argsort( importances )
        idxSortRf = idxSortRf[-1::-1]
    
        for idxFeature in range( len( idxSortRf ) ):
            featuresRF[foldIdx, idxSortRf[ idxFeature] ] = idxFeature+1
            
        
    
        #GB
        feature_importance = gb.feature_importances_
        idxSortGB = np.argsort( feature_importance )
        idxSortGB = idxSortGB[-1::-1]
    
                
        for idxFeature in range( len( idxSortGB ) ):
            featuresGB[foldIdx, idxSortGB[ idxFeature] ] = idxFeature+1
            
    
        foldIdx+=1
        

    CM_lr = confusion_matrix( dfData_lab['target'].to_numpy(), yPredML[ :, 0 ] )
    CM_rf = confusion_matrix( dfData_lab['target'].to_numpy(), yPredML[ :, 1 ] )
    CM_gb = confusion_matrix( dfData_lab['target'].to_numpy(), yPredML[ :, 2 ] )
    
    print( 'final LR lr\n', CM_lr/ CM_lr.sum(axis=1)[:, np.newaxis] )
    print("specificity_lr: ", CM_lr[0,0] / np.sum( CM_lr[0,:]) ) 
    print("sensitivity_lr: ", CM_lr[1,1] / np.sum( CM_lr[1,:]) ) 
    
    print("false positive_lr ", CM_lr[0,1] / np.sum( CM_lr[0,:]) ) 
    print("false negative_lr: ", CM_lr[1,0] / np.sum( CM_lr[1,:]) ) 
     
    
    
    print("specificity_rf: ", CM_rf[0,0] / np.sum( CM_rf[0,:]) ) 
    print("sensitivity_rf: ", CM_rf[1,1] / np.sum( CM_rf[1,:]) ) 
    
    print("false positive_rf ", CM_rf[0,1] / np.sum( CM_rf[0,:]) ) 
    print("false negative_rf: ", CM_rf[1,0] / np.sum( CM_rf[1,:]) ) 
     
    
    print("specificity_gb: ", CM_gb[0,0] / np.sum( CM_gb[0,:]) ) 
    print("sensitivity_gb: ", CM_gb[1,1] / np.sum( CM_gb[1,:]) ) 
    
    print("false positive_gb ", CM_gb[0,1] / np.sum( CM_gb[0,:]) ) 
    print("false negative_gb: ", CM_gb[1,0] / np.sum( CM_gb[1,:]) ) 
     
  
    
    
    averageLR_features = np.mean( featuresLR, axis=0 )
    averageRF_features = np.mean( featuresRF, axis=0 )
    averageGB_features = np.mean( featuresGB, axis=0 )
    
    idxLR = np.argsort( averageLR_features )
    idxRF = np.argsort( averageRF_features )
    idxGB = np.argsort( averageGB_features )

    mlOutput = [ CM_lr, CM_rf, CM_gb,  
                specificityML, sensitivityML, accuracyFolds,
                precision, precisionFalse, f1_scoreFolds,
                aucFolds, aucPrecisionFolds, gmeanFolds,ibaFolds ,
                
                idxLR , idxRF, idxGB]
    
    
    
    results[labTestID] = [ mlOutput ]



#Storing the results
#with open('Ml_output.pickle', "wb") as f:
#    pickle.dump(results, f)





      
     