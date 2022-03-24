# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:00:22 2021

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.stats import iqr
import numpy.matlib
from datetime import datetime
from sklearn.impute import KNNImputer
    
import datetime as dt
    


def getProbability( inputValues, nBins ):
    
    
    #Histogram data into 20 bins    
    hist, bin_edges = np.histogram( inputValues, bins=nBins )
    centerBins = bin_edges[:-1]+.5*np.diff(bin_edges)
    frequencyRelative = ( np.finfo(float).eps + hist) / np.sum(hist) 
    
    return frequencyRelative, bin_edges


def getMatrixJoinProbability(targetValues, variableValues, targetBins, nBins ):
    
    #Target values: only two values: 0:normal 1:abnormal
    
    hist_target, edgesTarget = np.histogram( targetValues, bins=targetBins )
    
    #Histogram for the value
    hist_variable, edgesVariable= np.histogram( variableValues, bins=nBins )
        
    jointProbability = np.zeros( [len(edgesTarget)-1 , len(edgesVariable)-1  ])
        
    Xjoin = np.vstack( (targetValues, variableValues ) )
        
    
    #Iterating over the bins of target values
    for iBin in range(  len(edgesTarget)-1  ):
        
        #Checking if the bin is the last one, in that case, the las bid edge is inclusive
        #idxSel are those with the value for the target
        if iBin == len(edgesTarget)-2:
            idxSel = np.argwhere( ( Xjoin[0,: ] >= edgesTarget[ iBin ] ) &  ( Xjoin[0,: ] <= edgesTarget[ iBin + 1] ) )
        else:
            idxSel = np.argwhere( ( Xjoin[0,: ] >= edgesTarget[ iBin ] ) &  ( Xjoin[0,: ] < edgesTarget[ iBin + 1] ) )
            
            
    
        
        selPrevious = Xjoin[1, idxSel ]
        for jBin in range( len(edgesVariable)-1 ):
            if jBin == len(edgesVariable)-2:
                jointProbability[ iBin, jBin ] = np.sum( ( selPrevious >= edgesVariable[ jBin ] ) &  ( selPrevious <= edgesVariable[ jBin + 1] ) )
            else: 
                jointProbability[ iBin, jBin ] = np.sum( ( selPrevious >= edgesVariable[ jBin ] ) &  ( selPrevious < edgesVariable[ jBin + 1] ) )
    
    
    
    jointProbability = (np.finfo(float).eps+jointProbability)/np.sum(jointProbability) 
    
    
    return jointProbability, edgesTarget, edgesVariable



def getConditionalProbability( targetValues, variableValues, testValues=None, isTargetValues = True, variableIsCat = False ):
    
    uniqueValues = np.unique(variableValues)
    
    if isTargetValues:
        targetBins = len( np.unique(targetValues) )
    else:
        #bwTar =  2 * iqr(targetValues) / ( len( np.unique(targetValues) )**(1/3) )
        bwTar = np.ceil( 1+ np.log2( len( np.unique(targetValues) ) ) )
        targetBins = np.ceil( ( np.max( np.unique(targetValues) ) - np.min( np.unique(targetValues) ) ) / bwTar )
    
    
    iqrVar = iqr(variableValues)
    
    if iqrVar == 0:
        iqrVar = iqr(variableValues[variableValues>0] )
    
    bw =  2 * iqrVar / ( len(uniqueValues)**(1/3) )
    #bw = np.ceil( 1+ np.log2( len(uniqueValues) ) )
    nBins = int( np.ceil( ( np.max( uniqueValues) - np.min(uniqueValues) ) / bw ) )
    
    if variableIsCat:
        nBins =  np.hstack([ uniqueValues-.5, uniqueValues[-1]+.5])
    
    jointProbability, edgesTarget, edgesVariable = getMatrixJoinProbability(targetValues, variableValues, int(targetBins), nBins )
    

    
    frequencyRelative, bin_edges = getProbability( variableValues, nBins  )
    
    
    frequencyRelativeTarget, bin_edgesTarget = getProbability( targetValues, int(targetBins)  )
    
    contionalProbability = np.empty( jointProbability.shape )*np.nan
    conditionalEntropy = np.empty( jointProbability.shape  )*np.nan
    mutualInformation = np.empty( jointProbability.shape  )*np.nan
    
    for iVarBin in range( jointProbability.shape[1] ) :
        
        for iTarget in range( jointProbability.shape[0] ) :
            
            ratio = ( jointProbability[iTarget,iVarBin]/frequencyRelative[ iVarBin ] ) + np.finfo(float).eps
            
            
            contionalProbability[ iTarget, iVarBin ]  = ratio
        
            conditionalEntropy[ iTarget, iVarBin ]  = -ratio * np.log2(ratio)
            
            ratioMI = ( jointProbability[iTarget,iVarBin]/ (frequencyRelativeTarget[ iTarget]*frequencyRelative[ iVarBin ] ) ) + np.finfo(float).eps
            mutualInformation[ iTarget, iVarBin ]  = ( ratio * np.log2(ratio) ) - ( frequencyRelativeTarget[ iTarget] * np.log2( frequencyRelativeTarget[ iTarget] ) )

    
    #TRAIN
    contionalProbabilityDict ={}
    conditionalEntropyDict = {} 
    mutualInformationDict = {} 
    for iUnique in range( len(uniqueValues ) ):
        value = uniqueValues[ iUnique ] 
        
        if value >= edgesVariable[-2]:
            idxBin = len(edgesVariable)-2 #np.where( (value >= edgesVariable[:-1]) & (value<= edgesVariable[1:] ) )[0]
        else:
            idxBin = np.where( (value >= edgesVariable[:-2]) & (value< edgesVariable[1:-1] ) )[0]
        
        
        
        contionalProbabilityDict[ value ]  = contionalProbability[ :, idxBin ] 
    
        conditionalEntropyDict[ value ]  = conditionalEntropy[ :, idxBin ] 
        mutualInformationDict[ value ]  = mutualInformation[ :, idxBin ] 
    
    conditionalEntropyTrain = np.empty( [  int(targetBins)  ,len(targetValues) ] )*np.nan
    contionalProbabilityTrain = np.empty( [ int(targetBins)  , len(targetValues) ] )*np.nan
    mutualInformationTrain = np.empty( [  int(targetBins)  , len(targetValues) ] )*np.nan
    
    #print(  int(targetBins)  ,len(targetValues) )
    
    
    probVariableTrain = np.empty( len(targetValues) )*np.nan
    for iRow in range( len(variableValues ) ):
        #print( variableValues[iRow], conditionalEntropyDict[variableValues[iRow]] )
        conditionalEntropyTrain[:,iRow ] = np.squeeze(conditionalEntropyDict[variableValues[iRow]])
        contionalProbabilityTrain[:,iRow ] = np.squeeze(contionalProbabilityDict[variableValues[iRow]])    
        mutualInformationTrain[:,iRow ] = np.squeeze(mutualInformationDict[variableValues[iRow]])  
 
    
    
    
    # For test set
    
    contionalProbabilityDict ={}
    conditionalEntropyDict = {} 
    uniqueTestValues = np.unique(testValues)
    for iUnique in range( len(uniqueTestValues ) ):
        value = uniqueTestValues[ iUnique ] 
        
        if value >= edgesVariable[-2]:
            if value <= edgesVariable[-1] :
                idxBin = [len(edgesVariable)-2]
            else:
                idxBin = [ ]
            #np.where( (value >= edgesVariable[:-1]) & (value<= edgesVariable[1:] ) )[0]
        else:
            idxBin = np.where( (value >= edgesVariable[:-2]) & (value< edgesVariable[1:-1] ) )[0]
        
        if len( idxBin ) > 0:
            contionalProbabilityDict[ value ]  = contionalProbability[ :, idxBin ] 
        
            conditionalEntropyDict[ value ]  = conditionalEntropy[ :, idxBin ] 
        else:
            contionalProbabilityDict[ value ]  = 0
        
            conditionalEntropyDict[ value ]  = 0
    
    conditionalEntropyTest = np.empty( [int(targetBins),len(testValues) ] )*np.nan
    contionalProbabilityTest = np.empty( [int(targetBins), len(testValues) ] )*np.nan
    for iRow in range( len(testValues ) ):
        conditionalEntropyTest[:,iRow ] = np.squeeze(conditionalEntropyDict[ testValues[iRow] ])
        contionalProbabilityTest[:,iRow ] = np.squeeze(contionalProbabilityDict[ testValues[iRow] ])
        
    
    
    return np.sum(conditionalEntropyTrain,axis=0), np.sum(conditionalEntropyTest,axis=0)





def countingConsecutiveNormalTestDays( window, dateLabs, isNormalLab  ):
    
    countDays = {}
    countNormalDays = {}
    
    
    queueTestDates = []
    
    #Iterate over the lab in chronological order
    
    #It is assumed that labs are sorted by date
    for iLab in range( len( dateLabs) ):
        
        currentDate = dateLabs[iLab]
        startWindow = currentDate
        startWindow -= np.timedelta64( window*24 , 'h')
        
        #removing dates in the queue out of the window

        
        for listDate in queueTestDates:
            if listDate < startWindow:
                #remove
                queueTestDates.remove( listDate )
            else:
                break
            
        #previous date test
        if iLab > 0:
            previousDate = dateLabs[iLab-1]
        else:
            previousDate = None
        
        #If lab is not the first one and the prev exam is withing the window, how normal labs are
        if previousDate != None and previousDate >= startWindow:
            k = len( queueTestDates )
        else:
            #Not normal test before
            k = None
            
        #Increasing counter for days for previous days with normal tests 
        if k != None:
            if k in countDays:
                countDays[k] += 1
            else:
                countDays[k] = 1
        

        #check if current test is normal
        if isNormalLab[iLab]==1 :
            queueTestDates.append( currentDate )
            if k != None:
                if k in countNormalDays:
                    countNormalDays[k] += 1
                else:
                    countNormalDays[k] = 1
        else:
            #Empty queue if the current test is not normal
            queueTestDates.clear()
    

    return countDays, countNormalDays







def getPre_probAdmission( admissions ):
    

    #Iterating over each test
    

    infoLab = {}

    uniqueAdmissions = np.unique(admissions['admissionID'].values)  

    
    
    
    countDays_overAdmission = {} #Each key will have the format labX_windowY
    countNormalDays_overAdmission = {}
    
    
    infoAdmissionPreviousDays = {}
    windowDays = 100
    for iAdmin in range( len( uniqueAdmissions ) ):
        
        #Getting the labs of the current admission
        dataAdmission = admissions[ admissions['admissionID'] == uniqueAdmissions[ iAdmin ] ]
        
        #sorting by date
        dataAdmission = dataAdmission.sort_values(['datatime'], ascending=[ True])
        
        
        labAdmissionDates = np.hstack( [ dataAdmission['datatime'].values[0], 
                                        dataAdmission['datatime'].values ] )
        
        
        labAdmissionNormal = np.hstack( [ dataAdmission['isNormalPrevious'].iloc[0], 
                                          dataAdmission['isNormal'].values ] )
        
        #Add values 

        #Counting previous normal days
        countDays, countNormalDays = countingConsecutiveNormalTestDays( windowDays , labAdmissionDates, labAdmissionNormal  )
        
        #Count for all days
        for keyDay in countDays:
            #admission + window is in the dictionary 
            # count (key) is a element of the lab+window dict
            if  keyDay in countDays_overAdmission:   
                countDays_overAdmission[keyDay] += countDays[keyDay]

            else:
                  #The lab+window element is not there in the dict
                  countDays_overAdmission[keyDay] = countDays[keyDay]
                  
        #Count for normal days
        for keyDay in countNormalDays:
            if  keyDay in countNormalDays_overAdmission:
            
                countNormalDays_overAdmission[keyDay] += countNormalDays[keyDay]
             
                  
            else:
                #The lab+window element is not there in the dict
                countNormalDays_overAdmission[keyDay] = countNormalDays[keyDay]
                
        
    # adding prob for each day to the dictionaries
    yValues= []
    xValues = []
    daysDict = {}
    for key_day in countDays_overAdmission:
        
        xValues.append( int(key_day) )
        
        if key_day in countNormalDays_overAdmission:
            
            ratioDay = countNormalDays_overAdmission[key_day] / countDays_overAdmission[key_day]
            daysDict[ key_day ] = ratioDay
        else:
            ratioDay = 0
            daysDict[ key_day ] = ratioDay
            
        yValues.append(ratioDay)
    
    xValues= np.asarray( xValues )
    yValues = np.asarray( yValues )
    
    xIdxSorted = np.argsort(xValues)


    
    
    return daysDict, xValues,  yValues
    
    
    
    
def aux_calculatePrior( daysDict,  dicConsecutives, admissions, dateLab ):
        
    priorSet = np.empty( len(admissions ) )*np.nan
    
    days = []
    ratioDays = [ ]
    for idx  in  range( len(admissions ) ):
        
        dateStr = str(dateLab[idx]).replace("T", " ")
        atoms = dateStr.split('.')
        
        #print(admissions[idx], atoms[0] )
        
        previousConsTest = dicConsecutives[ admissions[idx] ][ atoms[0] ]
        
        if int(previousConsTest) in daysDict:
            priorSet[idx] = daysDict[ int(previousConsTest) ]
            ratioDays.append( priorSet[idx] )
            days.append( int(previousConsTest) )
        else:
            ratioDays.append( np.nan )
            days.append( int(previousConsTest) )
    
    #find nan
    idxNan = np.where( np.isnan(priorSet) )[0]
    #Check if there are nan
    if len(idxNan ) > 0 :
        days = np.asarray( days )
        ratioDays = np.asarray( ratioDays )
        
        joinMatrix = np.vstack( [ days, ratioDays] ).T
        imputer = KNNImputer(n_neighbors=3)
        joinMatrix = imputer.fit_transform(joinMatrix)


        
        for currentNan in range( len(idxNan) ):
            priorSet[ idxNan[ currentNan ] ] = joinMatrix[ idxNan[ currentNan ], 1 ]
            
            
    
    
        
    return priorSet
    
    
def calculatePrior( daysDict,  dicConsecutives, admissions_train, dateLab_train, admissions_test, dateLab_test):
    
    #daysDict contains information about what is the pre-test probability given a day 
    # this was calculated with the function getPre_probAdmission using only admission of the training folds
    
    # dicConsecutives contains information about the admission. For each admission and date states 
    # how many previous consecutive normal test were
    
    priorTrain = aux_calculatePrior( daysDict,  dicConsecutives, admissions_train, dateLab_train )
    priorTest = aux_calculatePrior( daysDict,  dicConsecutives, admissions_test, dateLab_test )
    
    return priorTrain , priorTest
    














