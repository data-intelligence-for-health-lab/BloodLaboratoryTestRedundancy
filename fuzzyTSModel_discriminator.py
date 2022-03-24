# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:53:40 2021

@author: Admin
"""

import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import least_squares, minimize, basinhopping, LinearConstraint, curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import scipy as sp
import os
from sklearn import metrics
import numpy.matlib
from sklearn.utils.class_weight import compute_class_weight
from scipy.interpolate import interp1d
from imblearn.over_sampling import RandomOverSampler
#This script corresponds to the parameter estimation of the TS fuzzy model
from random import randint
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import hilbert
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# Transformer method for this transformer
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
        
        
class FuzzyModelDiscriminator(object):

    
    #mCluster usually between 1.1 and 2
    #Array exponentiation applied to the membership function u_old at each iteration, where
    # a higher mCluster number will increase fuzziness, correspoding to smaller partining coefficients
    # that means fewer cluster and more overlapping between the clusters...that is useful when the 
    #distorsion want to be measured .. how much a sample look similar to another class
    mCluster = 2# 
        
    oversampling = 1
    
    # Operation for finding membership values across variables
    #minOperation 1, the min would be used...otherwise the np.prod is used
    minOperation = 0
        
    
    # Threshold for simplifying fuzzy rules
    thrSimplyRules = 1.95
    thrSimplyUniverse = 2
        
    typeMF = 1# 1: gaussian; 2:exponential

    
    def getConvexEnvelope( self, xin, mfin, norm=1, nc=1000 ):
        
        # Calculates the convex membership function that envelopes a given set of
        # data points and their corresponding membership values. 
        
        # Input:
        # Xin: N x 1 input domain (column vector)
        # MFin: N x 1correspoding membership values 
        # nc: number of alpha cut values to consider (default=101)
        # norm: optional normalization flag (0: do not normalize, 1 : normalize, 
        # default=1)
        #
        # Output:
        # mf: membership values of convex function
        # x: output domain values    
        
        # Normalize the membership values (if requested)
        if norm == 1:
            maxVal = np.max(mfin)
            if maxVal==0:
                maxVal=np.finfo(float).eps
            mfin = np.divide(mfin, maxVal)
        
        # Initialize auxilary variables
        acut = np.linspace(0,np.max(mfin),nc)
        mf= np.full(2*nc, np.nan)
        x=np.full(2*nc, np.nan)
        
        if np.any(mfin>0):
            x[0] = np.min(xin[mfin>0])
            x[nc]=np.max(xin[mfin>0])
            mf[0]=0
            mf[nc] = 0 
        
        # Determine the elements in the alpha cuts    
        for i in range(0,nc):
            if np.any(mfin>acut[i]):
                x[i]=np.min(xin[mfin>acut[i]])
                x[i+nc]=np.max(xin[mfin>acut[i]])
                mf[i]=acut[i]
                mf[i+nc]=acut[i]
                
        #Delete NaNs
        idx=np.isnan(x)
        x=x[idx==False]
        mf=mf[idx==False]  
        
        # Sort vectors based on membership value (descending order)
        indmf=mf.argsort(axis=0)
        indmf=np.flipud(indmf)
        mf=mf[indmf]
        x=x[indmf]
        
        # Find duplicate values for x and onlykeep the ones with the highest membership value
        _, ind = np.unique(x, return_index=True, return_inverse=False, return_counts=False, axis=None)
        mf=mf[ind]
        x=x[ind]
        
        # Sort vectors based on x value (ascending order)
        indx=x.argsort(axis=0)
        mf=mf[indx]
        x=x[indx]
        
        xval=np.linspace(np.min(x),np.max(x),nc)
        mf=np.interp(xval, x, mf, left=None, right=None, period=None)
        x=xval;
        return mf, x
    

    
    
   
    def findParametersExponential(self,  x, y  ):
        
     
        #Find the convex envelope
        mf, xMf = self.getConvexEnvelope( x, y, norm=1, nc=1000 )
        
    
        
        # The input is the envelop of the projections

        # Smoothing the envelop using bins
        # Taking the mean of the values inside the bin
        xMfSet = []
        mfSet = []
        medianPeriod = np.median( np.diff( np.sort(xMf) ) )
        for idx in range( len(xMf) ):
            startWind = xMf[idx] - ( medianPeriod/2 )
            endWind = xMf[idx] + ( medianPeriod/2 )
            if idx == len(xMf)-1:
                countPoints = np.sum( (x >= startWind) & (x <= endWind) )
            else:
                countPoints = np.sum( (x >= startWind) & (x < endWind) )
            
            xMfSet += [ xMf[idx] ]*countPoints
            mfSet += [ mf[idx] ]*countPoints
            
            
        
        xMfSet = np.asarray( xMfSet )
        mfSet = np.asarray( mfSet )
        

        
        if self.typeMF == 1:

            mu = sum(xMfSet * mfSet) / sum(mfSet)
            mfSet[mfSet==0] = np.finfo(np.float64).eps
            sig = np.mean(np.sqrt(-((xMfSet-mu)**2)/(2*np.log(mfSet))))
            
            
            if np.min(xMfSet ) == np.max(xMfSet ):
                param, _ = curve_fit(self._gaussmf, xMfSet, mfSet, p0 = [  np.min(xMfSet ), np.finfo(float).eps ], 
                                 bounds=( (-np.inf, np.finfo(float).eps ), 
                                         (np.inf, np.inf ) ), maxfev = 10000, jac = self.funJacExpontentialCurveOneSide)
            else:
                param, _ = curve_fit(self._gaussmf, xMfSet, mfSet, p0 = [mu, sig], 
                                 bounds=( (np.min(xMfSet ), np.finfo(float).eps ), 
                                         (np.max(xMfSet ) , np.max([ np.max(xMfSet )-np.min(xMfSet ), sig ]) ) ), maxfev = 10000,
                                 jac = self.funJacExpontentialCurveOneSide)
            
            popt = np.zeros(4)
            popt[:2]= param
            popt[2:]= param
            
            
            
            
        else:
            

            mu1 = xMf[mf>=0.97][0]
            mu2 = xMf[mf>=0.97][-1]
            xmf = xMf
            sig1 = (mu1 - xmf[0])/(np.sqrt(-2*np.log(mf[0])))
            sig2 = (xmf[-1]-mu2)/(np.sqrt(-2*np.log(mf[-1])))
            
            if sig1==0.0:
                sig1=0.1
            if sig2==0.0:
                sig2=0.1
            
           
            
            try:
                popt, pcov = curve_fit(self.evalulateExponentialFunction, xMfSet , mfSet, p0 = [mu1, sig1, mu2, sig2],
                                       bounds=((-np.inf, np.finfo(float).eps,-np.inf,  np.finfo(float).eps ), 
                                               (np.inf, np.inf, np.inf, np.inf),),
                                       maxfev=1000, 
                                       jac = self.funJacExpontentialCurve )
                
                
                
            except:
                popt, pcov = curve_fit(self.evalulateExponentialFunction, xMfSet , mfSet, p0 = [mu1, sig1, mu2, sig2],
                                       bounds=((-np.inf, np.finfo(float).eps,-np.inf,  np.finfo(float).eps ), 
                                               (np.inf, np.inf, np.inf, np.inf) ),
                                               maxfev=100000,)

        
        return popt
        
    
    def _gaussmf(self,x, mu, sigma, a=1):
        # x:  (1D array)
        # mu: Center of the bell curve (float)
        # sigma: Width of the bell curve (float)
        # a: normalizes the bell curve, for normal fuzzy set a=1 (float) 
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    
    def evalulateExponentialFunction( self, x, cL, wL, cR, wR):
        if wR==0:
            wR = np.finfo(float).eps
        if wL==0:
            wL = np.finfo(float).eps  
            
            
        yPred = np.ones( len(x) )
        if cR - cL >=  -1e-10 :
            yPred = np.ones( len(x) )
                    
            idxLowerCl = np.argwhere( x < cL )
            yPred[ idxLowerCl ] = fuzz.gaussmf( x[ idxLowerCl] , cL, wL)
            
            idxLowerCr = np.argwhere(x> cR )
            yPred[ idxLowerCr] = fuzz.gaussmf( x[ idxLowerCr] , cR, wR)
        else:
            yPred = yPred*100
        return yPred
    

        
    def funJacExpontentialCurve( self, x, cL, wL, cR, wR):
        
        if wR==0:
            wR = np.finfo(float).eps
        if wL==0:
            wL = np.finfo(float).eps    
            
        
        
        
        yPred = self.evalulateExponentialFunction( x, cL, wL, cR, wR)
        yJac = np.zeros( [ len(yPred), 4] )
        
        for idx in range( x.shape[0] ):
            currentX = x[idx]
            
            if currentX < cL:
            
                
                
                yJac[idx, 0] = yPred[idx] * ( ( currentX- cL ) / wL**2 )
                yJac[idx, 1] = yPred[idx] * ( ( currentX- cL )**2/ wL**3 )
        
            if currentX > cR:
                
                
                yJac[idx, 2] = yPred[idx] * ( ( currentX- cR )/ wR**2 )
                yJac[idx, 3] = yPred[idx] * ( ( currentX- cR )**2/ wR**3) 
        
        
        
        
        
        return yJac

    def funJacExpontentialCurveOneSide( self, x, c, w):
        
        
        if w==0:
            w = np.finfo(float).eps    
            
        
        
        
        yPred = self._gaussmf(x, c, w)
        yJac = np.zeros( [ len(yPred), 2] )
        
        for idx in range( x.shape[0] ):
            currentX = x[idx]
   
            yJac[idx, 0] = yPred[idx] * ( ( currentX- c ) / w**2 )
            yJac[idx, 1] = yPred[idx] * ( ( currentX- c )**2/ w**3 )
        
           
        return yJac
        
    
    
    
    
    
    def findingTotalClusters( self, Xtrain, yTrain, numerical_columns, categorical_columns  ):
        #First step is to estimate how many logic rules would be in the model
        
        
        #1st normalized the input
        
        Xnormalized = self.normalizingInputData( Xtrain, numerical_columns, categorical_columns )
        
        
       
        
        Z = np.hstack( (Xnormalized, yTrain[:,np.newaxis] ) )
        
        #Normalizing data
        scalerZ = MinMaxScaler() #for the response
        
        
    
            
        if self.oversampling == 1:
            #after normalizing, resample
            ros = RandomOverSampler()
            
            #At this point y is not normalized
            X_sampled, y_sampled = ros .fit_resample(Z[ :, :-1], yTrain)
            #normalizing Y to have same scale in the clustering    
            y_sampled_normalized = scalerZ.fit_transform( y_sampled[:,np.newaxis] )
            Z = np.hstack( ( X_sampled  , y_sampled_normalized ) )
            
            alldata = Z.T
            
        else:
            alldata = Z.T   
            
        
        

        
        fpcs = []
        totalCentersToCheck = 5
        for ncenters in range(2, totalCentersToCheck+1 ):
            
            #Applying the c-means using different centers
            
            #m Array exponentiation applied to the membership function u_old at each iteration, where U_new = u_old ** m.
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                alldata, ncenters, m= self.mCluster, error=1e-6, maxiter=10000, init=None)
            
            fpcs.append(fpc)
            
        #print('fpcs', fpcs)
        #Finding which center achieve the best metric
        centers = range(2, totalCentersToCheck+1 )
        bestCentersIdx = np.argmax( fpcs )
        
        
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans( alldata, 
                                                         centers[bestCentersIdx],
                                                         m=self.mCluster, error=1e-6, maxiter=10000, init=None)
        if self.oversampling == 1:
            u = u[ :,:Xtrain.shape[0] ]
        print( 'number of clusters (rules) ', centers[bestCentersIdx] )
        
        
        #Returning number of clusters
        return centers[bestCentersIdx], u



    def trainFuzzyModel( self, Xtrain, yTrain, clusters=None, plotFun=0):
        
        if clusters == None:
            #If there is not a number of clusted defined, it is determined 
            #on the training set.
            clusters, u = self.findingTotalClusters( Xtrain, yTrain)
        
        else:
            Z = np.hstack( (Xtrain, yTrain[:,np.newaxis] ) )
        
            #Normalizing data
            scalerZ = MinMaxScaler()
            normaliedData = scalerZ.fit_transform( Z )
            
            if self.oversampling == 1:
                #after normalizing, resample
                ros = RandomOverSampler()
                X_sampled, y_sampled = ros .fit_resample(normaliedData[ :, :-1], yTrain)
                    
                y_sampled_normalized = scalerZ.fit_transform( y_sampled[:,np.newaxis] )
                Z = np.hstack( ( X_sampled  , y_sampled_normalized ) )
                
                alldata = Z.T
                
            else:
                alldata = normaliedData.T   
            
            
            
            _, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans( alldata, clusters,
                                     m=self.mCluster, error=1e-6, maxiter=10000, init=None)
        
            if self.oversampling == 1:
                u = u[ :,:Xtrain.shape[0] ]
                
            scalerTrain = MinMaxScaler()
            normalizedData = scalerTrain.fit_transform( Xtrain )
        
        
        
            parametersCurves = self.findingMembershipFunctions( normalizedData, yTrain, u, plotFun )
            parametersCurves_simplify = self.simplifyMembershipFunctions( parametersCurves.copy() , .95, .99 )
        

            consequentRules = self.calculatingConsecuents(  u,  yTrain, 0 )
        
            return parametersCurves_simplify, consequentRules, scalerTrain
    
    def trainFuzzyModelClasses( self, Xtrain, yTrain, selectedFeatures, numericalVariables, categoricalVariables, clusters=None, plotFun=0):
        
        if clusters == None:
            #If there is not a number of clusted defined, it is determined 
            #on the training set.
            clusters, _ = self.findingTotalClusters( Xtrain, yTrain)
        
     
        Xnormalized = self.normalizingInputData( Xtrain, numericalVariables, categoricalVariables )
        Z = np.hstack( (Xnormalized, yTrain[:,np.newaxis] ) )
            
    
        #Normalizing data
        if self.oversampling == 1:
            #after normalizing, resample
            ros = RandomOverSampler()
            #Resampling using original labels
            X_sampled, y_sampled = ros.fit_resample(Z[ :, :-1], yTrain )
            
            scalerY = MinMaxScaler()
            y_sampled_normalized = scalerY.fit_transform( y_sampled[:,np.newaxis] )
            Z = np.hstack( ( X_sampled  , y_sampled_normalized ) )
            
            alldata = Z.T
            
        else:
            alldata = Z.T   

            
        centroids, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans( alldata, clusters,
                                      m=self.mCluster, error=1e-6, maxiter=10000, init=None)
      
        
        #Denormalizing centroids
        scalerZ = MinMaxScaler()
        scalerZ.fit_transform( Xtrain[numericalVariables] )
        #only for numerical variables
        centroidsDeNorm = np.zeros( centroids.shape)
        centroidsDeNorm[ :, :len( numericalVariables) ] = scalerZ.inverse_transform( centroids[:, : len( numericalVariables) ] )
        #minValueFeatures = scalerZ.data_min_
        

        if self.oversampling == 1:
            u = u[ :,:Xtrain.shape[0] ]
        #     print(u)
                
        
        #converting dataframe to numpy
        #Xtrain = Xtrain[numericalVariables+categoricalVariables].to_numpy()
        Xtrain = Xtrain.to_numpy()
        
        numericalIndexes = np.arange( len(numericalVariables) )

        parametersCurves, dicCategoricalVar  = self.findingMembershipFunctions( Xtrain, u, centroidsDeNorm, numericalIndexes )
        
        xMin = np.min(Xtrain, axis=0 )
        xMax = np.max( Xtrain, axis=0)
        parametersCurves_simplify = self.simplifyMembershipFunctions( parametersCurves.copy() , xMin, 
                                                                     xMax, numericalIndexes, self.thrSimplyRules, self.thrSimplyUniverse  )
    
        idxClass0 = np.where( yTrain==0 )[0]
        idxClass1 = np.where( yTrain==1 )[0]
        Xtrain0 = Xtrain[idxClass0, 5]
        Xtrain1 = Xtrain[idxClass1, 5]
        uClass0 = u[:, idxClass0]
        uClass1 = u[:, idxClass1]
  
        consequentRules = self.calculatingConsecuents(  u,  yTrain, 0 )
        
             
        _, _, activationPerClassGroup = self.predict(  Xtrain, yTrain, selectedFeatures, parametersCurves_simplify, dicCategoricalVar,
                                                     numericalVariables, categoricalVariables, numericalIndexes, consequentRules  )
    
        fpr, tpr, thresholds = roc_curve(yTrain, activationPerClassGroup[:,1])

        
        # calculate the g-mean for each threshold
        J = tpr - fpr
        ix_J = np.argmax(J)
        
        gMeans = np.sqrt(tpr*(1 - fpr))
        ix = np.argmax(gMeans)
       
                
        return parametersCurves_simplify, dicCategoricalVar, consequentRules, thresholds[ix]    

    def predictMembershipValue(  self, Xtrain, parametersCurves, dictCategorical, idxNumericalVariables, featuresIdx):
        
        totalRules, totalFeatures, totalParameters = parametersCurves.shape
        
        MembershipMatrix = np.empty([ totalRules, Xtrain.shape[0], len(featuresIdx) ])
        Betas = np.empty([ totalRules, Xtrain.shape[0]] )
        for iRule in range( totalRules ):
            idxFeature = 0
            for iVariable in featuresIdx:
                
                if iVariable in idxNumericalVariables:
        
                    #a,b,c = parametersCurves[ iRule, iVariable, : ] 
                    #, width = parametersCurves[ iRule, iVariable, : ] 
                    cL, wL, cR, wR = parametersCurves[ iRule, iVariable, : ] 
                    MembershipMatrix[iRule, :, idxFeature] = self.evalulateExponentialFunction( Xtrain[:, iVariable], cL, wL, cR, wR)
                    
                else:
                    #This is a categorical variable
                    
                    for idX in range(Xtrain.shape[0]):
                        if Xtrain[idX,iVariable] in  dictCategorical[ iVariable][iRule] :
                            #Getting the value for each sample
                            MembershipMatrix[iRule, idX, idxFeature] = dictCategorical[ iVariable][iRule][ Xtrain[idX,iVariable] ]
                        else:
                            MembershipMatrix[iRule, idX, idxFeature] = np.finfo(float).eps
                    
                idxFeature+=1
                
            Betas[iRule,:] = np.prod( MembershipMatrix[iRule, :, :], axis=1)
       
        return Betas
    
    def predict( self, Xtest, yTest, selectedFeatures, parametersCurves, dicCategoricalVar, numericalVariables, categoricalVariables, numericalIndexes, consequentRules, thresholdClass=.5  ): 
        
       
        if isinstance(Xtest, pd.DataFrame ) :
            Xtest = Xtest.to_numpy()
            
        membership = self.predictMembershipValue(Xtest, parametersCurves, dicCategoricalVar,  numericalIndexes, selectedFeatures)
        
        membership = membership/( np.finfo(float).eps+  np.sum(membership, axis=0) )
        
                
        activationPerClass = np.dot( membership.T, consequentRules ) 
  
        yHat = np.zeros( len(yTest) )
        yHat[ activationPerClass[:,1] >= thresholdClass ] = 1
              
              
            
        accuracy = metrics.f1_score( yTest, yHat, average='weighted')
        return accuracy, yHat, activationPerClass
    
    def trainFuzzyModelClassesCategorical( self, Xtrain, yTrain, dictionaryCategorical, categoricalVariables ):
        
        
           
        numericalVariables = [ iVar for iVar in range(Xtrain.shape[1]) if iVar not in categoricalVariables ]
        
        
        parameterModels = {}
        
        for key in dictionaryCategorical:
            
            if key == 'all':
                Xgroup = Xtrain[:, numericalVariables] #Only the numeric variables
                yGroup = yTrain
                
                selectedFeatures = dictionaryCategorical[key][0]
                clusters = dictionaryCategorical[key][1]
                parametersCurves_simplify, consequentRules, scalerTrain, thresholdClass = self.trainFuzzyModelClasses(  Xgroup, yGroup, selectedFeatures, clusters )
            
                # _, yHatGroup, activationPerClassGroup = self.predict(  Xgroup, yGroup, selectedFeatures, parametersCurves_simplify, consequentRules  )
                # #rocVal = roc_auc_score(y_train, activationPerClassGroup[:,1], average='weight')  
                # fpr, tpr, thresholds = roc_curve(yGroup, activationPerClassGroup[:,1])
        
                # # plt.plot( fpr, tpr, '.-')
                # # plt.show()
        
        

                # # calculate the g-mean for each threshold
                # gmeans = np.sqrt(tpr * (1-fpr))
                # ix = np.argmax(gmeans)
            
                parameterModels[key] = [ parametersCurves_simplify, consequentRules, scalerTrain, thresholdClass  ]
            else:
                splitValues = key.split('_')
            
                idxVar = 0
                idxSel = np.zeros([Xtrain.shape[0], len(categoricalVariables) ] )
                for iCatVar in categoricalVariables:
                    idxSel[ int(splitValues[ idxVar]) == Xtrain[ :, iCatVar ], idxVar ] = 1
                    idxVar+=1
                    
                idxSelGroup = np.prod(idxSel, axis=1)    
                Xgroup = Xtrain[ idxSelGroup == 1, : ]
                Xgroup = Xgroup[:, numericalVariables] #Only the numeric variables
                yGroup = yTrain[ idxSelGroup == 1 ]
                
                selectedFeatures = dictionaryCategorical[key][0]
                clusters = dictionaryCategorical[key][1]
                parametersCurves_simplify, consequentRules, scalerTrain,thresholdClass = self.trainFuzzyModelClasses(  Xgroup, yGroup, selectedFeatures, clusters )
            
                
                
                parameterModels[key] = [ parametersCurves_simplify, consequentRules, scalerTrain, thresholdClass  ]
               
        return parameterModels
    

       
    def findingMembershipFunctions( self, Xtrain, u, centroids, idxNumericalVariables, plotFun=0 ):
        
        # number of features
        totalFeatures = Xtrain.shape[1]
        
        totalRules = u.shape[0]
        
        #Normalized feature set
        normalizedData = Xtrain# scalerZ.fit_transform( Xtrain )
        
        #To store parameters of the membership fucntions
        numberParamters = 4
        parametersCurves = np.empty( [ totalRules, totalFeatures, numberParamters] )
        dictCategoricalVar = {}
        clusterPerPoint = np.argmax( u , axis= 0 )
        
        # Each cluster has a membership function for each feature
        for iRule in range( totalRules ):
            #Membership functions of the rules are determined
            #Using the only the points of the cluster 
            dataCluster = normalizedData[ clusterPerPoint==iRule, : ]
            uCluster = u[ :, clusterPerPoint==iRule ]
            
            
            
            for iVariable in range( totalFeatures ):
            
                uniqueValues = np.unique(  dataCluster[ :, iVariable ] )
            
                projectionValue = np.empty( len( uniqueValues ) )
                
                if iVariable in idxNumericalVariables:
                
                    if self.typeMF == 1:
                        iqr = (np.quantile( dataCluster[:, iVariable], .75) - np.quantile( dataCluster[:, iVariable], .25) )
                        if iqr!=0:
                            fstOut = np.quantile( dataCluster[:, iVariable], .25) - (iqr*3)
                            trdOut = np.quantile( dataCluster[:, iVariable], .75) + (iqr*3)
                            
                            
                            idxThrSamples = np.where( (dataCluster[:,iVariable] >= fstOut) &  (dataCluster[:,iVariable] <= trdOut)  )[0] 
                        else:
                         
                            
                            lowerOut = np.quantile( dataCluster[:, iVariable], .01, interpolation='lower') #- np.finfo(float).eps
                            upperOut = np.quantile( dataCluster[:, iVariable], .99, interpolation='higher') #+ np.finfo(float).eps
                           
                            idxThrSamples = np.where( (dataCluster[:,iVariable] >= lowerOut) &  (dataCluster[:,iVariable] <= upperOut)  )[0] 
                           
                        
                        expParameters2 = self.findParametersExponential( dataCluster[idxThrSamples, iVariable], uCluster[iRule,idxThrSamples] )
                        parametersCurves[ iRule, iVariable,: ] = expParameters2
                        
                       
                    else:
                        expParameters = self.findParametersExponential( dataCluster[:, iVariable], uCluster[iRule,:] )
                        parametersCurves[ iRule, iVariable,: ] = expParameters
                
                else:
                        #This is for categorical variables
                        uniqueValues = np.unique( dataCluster[:, iVariable] )
                        
                      
                        clusterPerValue = np.zeros( len( uniqueValues) ) 
                        
                        for idxUniVal in range( len( uniqueValues ) ):
                            
                            clusterPerValue[ idxUniVal ] = np.median( uCluster[ iRule, dataCluster[:, iVariable] == uniqueValues[idxUniVal]  ] ) 
                            
                        totalU = np.sum( clusterPerValue )
                        
                        clusterPerValuePer = {}
                        for idxUniVal in range( len( uniqueValues ) ):
                            clusterPerValuePer[ uniqueValues[idxUniVal] ] = clusterPerValue[ idxUniVal ] / totalU
                
                        #check that variable is not in the dic
                        if not iVariable in dictCategoricalVar:
                            #A dic for each rule
                            dictCategoricalVar[ iVariable] = {}
                            dictCategoricalVar[ iVariable][iRule] = clusterPerValuePer
                            
                            
                        else:
                            dictCategoricalVar[ iVariable][iRule] = clusterPerValuePer
                             
                
        return  parametersCurves , dictCategoricalVar #parameterCurves for numerical variables and dict for categorical variables
        
    
    

    
    def calculatingConsecuents( self, Umatrix,  yTrain, method=0 ):
        uniqueClasses = np.unique( yTrain )
        
        consequentPerRule = np.zeros( [Umatrix.shape[0] , len(uniqueClasses ) ] )
        
       
        weightClass = compute_class_weight('balanced', classes= np.unique(yTrain), y=yTrain)
        
        
        for iRule in range( Umatrix.shape[0] ):
            consequentInter = np.zeros( len(uniqueClasses )  )
            samplesPerClass  = np.zeros( len(uniqueClasses )  )
            medianPerClass  = np.zeros( len(uniqueClasses )  )
            stdPerClass = np.zeros( len(uniqueClasses )  )
            for iClass in range( len(uniqueClasses ) ):
                consequentInter[ iClass] = np.sum( Umatrix[ iRule, yTrain == uniqueClasses[ iClass ] ] )
                samplesPerClass[ iClass] = np.sum( yTrain == uniqueClasses[ iClass ] )
                medianPerClass[ iClass ] = np.median( Umatrix[ iRule, yTrain == uniqueClasses[ iClass ] ] )
                stdPerClass[ iClass ] = np.std( Umatrix[ iRule, yTrain == uniqueClasses[ iClass ] ] )
               
           
            consequentInterMedian = medianPerClass#np.multiply( consequentInter, 1/samplesPerClass  )
            consequentInter = np.multiply( consequentInter, weightClass  )
            
            maxClassIdx = np.argmax( consequentInter )
            
            method = 2
            if method ==0:
                consequentPerRule[iRule, maxClassIdx ] = 1
            elif method == 1:
                consequentPerRule[iRule, maxClassIdx ] = consequentInter[ maxClassIdx ]
            else:

                consequentPerRule[iRule, : ] = consequentInter/ np.sum( consequentInter )
                
        return consequentPerRule

    

    def simplifyMembershipFunctions( self, parametersCurves, minX,maxX, idxNumericalVariables, thrMemb, thrU ):
        
        totalRules, totalFeatures, totalParameters =parametersCurves.shape
        
        #Evaluate pair of membership functions of the same variable
        cons = [{'type': 'ineq', 'fun': lambda x:  x[2] - x[0] } ] #For the optimziation
        

        for iFeature in range(totalFeatures):
            
            if iFeature in idxNumericalVariables:
            
                xCombined = np.linspace( minX[iFeature], maxX[iFeature]  , 100 )
                
                for iRule in range(totalRules-1):
                    cL_1, wL_1, cR_1, wR_1 =    parametersCurves[ iRule, iFeature,: ] 
                    
                    
                        
                    for jRule in range(iRule+1, totalRules):
                        
                        
                        
                        
                        cL_2, wL_2, cR_2, wR_2 =    parametersCurves[ jRule, iFeature,: ] 
                        
                        yPred1 = self.evalulateExponentialFunction(xCombined, cL_1, wL_1, cR_1, wR_1)
                        
                        yPred2 = self.evalulateExponentialFunction(xCombined, cL_2, wL_2, cR_2, wR_2)
        
                        
                    
                        #calculate similarity
                        combinedPred = np.vstack( ( yPred1,yPred2) )
                        
                        
    
                            
                            
                        similarity = np.sum( np.min(combinedPred, axis=0) )/ np.sum( np.max(combinedPred, axis=0) )
                        
                        if similarity > thrMemb:
                            
                            
                        
                            
                            expParameters = self.findParametersExponential(xCombined, np.max(combinedPred, axis=0)  )
                            
                            
                           
                            parametersCurves[ iRule, iFeature, :] = expParameters
                            parametersCurves[ jRule, iFeature, :] = expParameters
                        
        
        
        
        #Removing those rules that are like the universe
        for iFeature in range(totalFeatures):
            
            if iFeature in idxNumericalVariables:
                xPoints1 = np.linspace(minX[iFeature],maxX[iFeature],100)
                for iRule in range(totalRules):
                    cL_1, wL_1, cR_1, wR_1 =    parametersCurves[ iRule, iFeature,: ] 
                    
                    yPred1 = self.evalulateExponentialFunction(xPoints1, cL_1, wL_1, cR_1, wR_1)
                    
                    #calculate similarity with the universe
                        
                    similarity = np.sum( yPred1 )/ len( xPoints1)
                    
                    if similarity > thrU:
                            
                        parametersCurves[ iRule, iFeature, :] = [ np.nan, np.nan, np.nan, np.nan]
        
       
        return  parametersCurves
       
        

    
    def normalizingInputData( self, Xinput, numerical_columns, categorical_columns ):
        
             preprocessor = ColumnTransformer([ ('standarizer', MinMaxScaler(),numerical_columns ),
                                       ('catTrans',  CategoricalTransformer(), categorical_columns )])
             
             return preprocessor.fit_transform( Xinput )
             
             
        
    def findingBestFeatures_val_aux( self, Xtrain, yTrain, numericalVariables, categoricalVariables ):     
        
        # Finding the number of clusters (rules) using all the training set
        clusters, u = self.findingTotalClusters( Xtrain, yTrain, numericalVariables, categoricalVariables  )
        
        
       
        
        #Split data in three folds
        folds = 3 
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        
        foldIdx = 0
        selectedFeaturesFolds = np.zeros( [folds,Xtrain.shape[1] ] )
        relevanceFeaturesFolds = np.ones( [folds,Xtrain.shape[1] ] )*Xtrain.shape[1] # initially this with the max num of features
        for train_index, test_index in skf.split(Xtrain, yTrain):
            X_A, X_B = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
            y_A, y_B = yTrain[train_index], yTrain[test_index]
            
            
            #Data plus labels            
           
            Xnormalized = self.normalizingInputData( X_A, numericalVariables, categoricalVariables )
            
        
        
            Z = np.hstack( (Xnormalized, y_A[:,np.newaxis] ) )
                
        
            #Normalizing data
            if self.oversampling == 1:
                #after normalizing, resample
                ros = RandomOverSampler()
                #Resampling using original labels
                X_sampled, y_sampled = ros.fit_resample(Z[ :, :-1], y_A )
                
                scalerY = MinMaxScaler()
                y_sampled_normalized = scalerY.fit_transform( y_sampled[:,np.newaxis] )
                Z = np.hstack( ( X_sampled  , y_sampled_normalized ) )
                
                alldata = Z.T
                
            else:
                alldata = Z.T   
    
            
            #Finding centroids
            centroidsA, uA, u0, d, jm, p, fpc = fuzz.cluster.cmeans( alldata, clusters,
                                          m=self.mCluster, error=1e-6, maxiter=10000, init=None)
            
                        
            #Denormalizing centroids
            scalerZ = MinMaxScaler()
            scalerZ.fit_transform( X_A[numericalVariables] )
            #only for numerical variables
            centroidsA_deNorm = np.zeros( centroidsA.shape)
            centroidsA_deNorm[ :, :len( numericalVariables) ] = scalerZ.inverse_transform( centroidsA[:, : len( numericalVariables) ] )
            #minValueFeatures = scalerZ.data_min_
            maxValueFeatures= scalerZ.data_max_  
            
           
                                             
                                             
            #Taking back the original samples     
            if self.oversampling == 1:
                uA = uA[:, :X_A.shape[0] ]
                
            # converting X_A and X_B to numpy
            # The columns should be sorted such as numerical columns are first and the categroical
            X_A = X_A.to_numpy()
            X_B = X_B.to_numpy()
            
            idxNumericalVariables = np.arange( len( numericalVariables) )
            idxCategoricallVariables = np.arange( len( numericalVariables), len( numericalVariables)+len( categoricalVariables) )
            
            
            #Min and Max for each feature
            minX = np.min( X_A, axis=0)
            maxX = np.max( X_A, axis=0)
            
            
            #Finding the parameter for the membership matrix
            parametersCurvesA, dicCategoricalA = self.findingMembershipFunctions( X_A, uA , centroidsA_deNorm, idxNumericalVariables )
            
            #Simplify membership functions
            parametersCurvesA_simplify = self.simplifyMembershipFunctions( parametersCurvesA.copy() , minX, maxX , idxNumericalVariables, self.thrSimplyRules, self.thrSimplyUniverse )
            
            #self.plotMembershipFunctions( parametersCurvesA, minX, maxX )
            #self.plotMembershipFunctions( parametersCurvesA_simplify, minX, maxX  )
            
            
            consequentRulesA = self.calculatingConsecuents(  uA,  y_A, 0 )
            
            
            # idx of the features to evaluate at each iteration
            candidates = [ iC for iC in range( Xtrain.shape[1] ) ]
            selected = [ ]
            
         
            stop = 0
            
            bestError = 1e10
            
            while stop != 1:
                
                #strctures to store the errors
                errorsA = np.ones( len(candidates) )
                errorsB = np.ones( len(candidates) )
                
                idxError = 0
                for iCandidate in candidates:
                    
                    tupleToUse = selected.copy()
                    tupleToUse.append( iCandidate )
                    
                    #Train with set A and test B
                    
                    #Get membership for set B using curves trained on set A
                    membershipB = self.predictMembershipValue(  X_B, parametersCurvesA_simplify, dicCategoricalA, idxNumericalVariables, tupleToUse )
                    membershipB = membershipB/ ( np.finfo(float).eps+ np.sum(membershipB, axis=0) )
                    
                                      
                    
                    #Calculating membership to the training set in order to calculate the consequent
                    membershipA_training = self.predictMembershipValue(  X_A, parametersCurvesA_simplify, dicCategoricalA, idxNumericalVariables, tupleToUse )
                    membershipA_training = membershipA_training/( np.finfo(float).eps+ np.sum(membershipA_training, axis=0) )
                    
                    #Using training set to calculate the threshold calibration
                    activationPerClassA =  np.dot( membershipA_training.T, consequentRulesA )
                    #False positive and true positive for the ROC curce
                    fpr, tpr, thresholds = roc_curve(y_A, activationPerClassA[:,1] ) 
                    
                    # Finding threshold for ROC curve
                    
                    #This is using Youden's statistc
                    J = tpr - fpr
                    ix_J = np.argmax(J)
                    
                    #This is using GMeans statistc
                    gMeans = np.sqrt(tpr*(1 - fpr))
                    ix = np.argmax(gMeans)
                    
                    #Distance to (0,1)
                    euDist = np.sqrt(fpr**2 + (1 - tpr)**2 )
                    ixEU = np.argmin(euDist)
                    
                    
                    
                    precision, recall, thrs_precision = precision_recall_curve( y_A, activationPerClassA[:,1] )
                    fscore = (2 * precision * recall) / (precision + recall + np.finfo(float).eps )
                    ix_fscore = np.argmax(fscore)
                    
                    
                        
                    #consequentRules has the fc..size rules x classes
                    # B samples x rules
                    # output is then samples x classes--which is the score of each discriminat
                    activationPerClassB =  np.dot( membershipB.T, consequentRulesA ) 
                                        
                    
                    #Estimating the activation for the held-out test
                    yHatB = np.zeros( len(y_B) )
                    yHatB[ activationPerClassB[:,1] >= thresholds[ix] ] = 1
                    
                    
                    sensitivityB = np.sum(y_B[y_B==1] == yHatB[y_B==1])/np.sum(y_B==1)
                    specificityB = np.sum(y_B[y_B==0] == yHatB[y_B==0])/np.sum(y_B==0)
                    
                    gmeansB = np.sqrt(sensitivityB * specificityB )
                    accuracyB  = np.sum(y_B== yHatB)/len(yHatB) 
                    
                    f1_scoreVal = f1_score( y_B, yHatB, average='weighted') 
                    
                    
                    
                    precisionAUCVal = average_precision_score(y_B, activationPerClassB[:,1], average='weighted')
                    rocVal = roc_auc_score(y_B, activationPerClassB[:,1] )#, average='weighted')  
                    
                    #dice score
                    TP = np.sum( (y_B==1) & (yHatB ==1) )
                    TN = np.sum( (y_B==0) & (yHatB ==0) )
                    FP = np.sum( (y_B==0) & (yHatB ==1) )
                    FN = np.sum( (y_B==1) & (yHatB ==0) )
                    diceScore  = ( 2*TP)/( 2*TP+FP+FN + np.finfo(float).eps )
                    
                    
                    
                    tverski = TP/( TP+ (.1*FP) +(.9*FN) + np.finfo(float).eps )   
                    
                    recall1 = TP/(TP+FN)
                    recall0 = TN/(TN+FP)
                    accuracy = (TP+TN)/(TP+FN+TN+FP)
                    recallAvg = ( 1.5*recall1 + recall0 + accuracy ) /3.5
                    
                    
                    
                    goalB =  recallAvg
                    errorsB[ idxError ] = 1 - goalB
                    
                    ###########################################
                    
                    
                    
                    
                   
                        
                        
                    print('tuple', tupleToUse, 'errorVal',  (  errorsB[ idxError ]) )
                    idxError+=1
                    
                errors = errorsB
                    
                    
                #Finding the feature with the minimum error
                minError = np.min( errors )
                minErrorIdx = np.argmin( errors )
                
                
                if np.sum( np.min(errors) == errors) > 1:
                    #more than a best error
                    #take randomly
                    idxsMinErrors = np.where( np.min(errors) == errors)[0]
                    
                    randomIdx = randint(0, len(idxsMinErrors)-1 ) 
                    minErrorIdx = idxsMinErrors[ randomIdx ]
                
                minCandidate = candidates[ minErrorIdx ]
                
                
                #If the minError is lower than the bestError
                if minError < bestError :
                    bestError = minError
                    #Add the candidate with the best error
                    selected.append( minCandidate )
                else :
                    # the minError is greater than the best (previous one)
                    stop = 1
                    
                
                
                if stop==0:
                    #remove selected and the worst error
                
                    #Max error -- remove it
                    maxErrorIdx = np.argmax( errors ) 
                    if len( candidates ) > 1 : 
                                      
                
                        if np.sum( np.max(errors) == errors) > 1:
                            #more than a worse error
                            #take randomly
                            idxsMaxErrors = np.where( np.max(errors) == errors)[0]
                            stopMaxError = 0
                            while stopMaxError == 0:
                                randomIdx = randint(0, len(idxsMaxErrors)-1 ) 
                                maxErrorIdx = idxsMaxErrors[ randomIdx ]
                                
                                #check that maxErrorIdx is not the same than minErrorIdx
                                if maxErrorIdx != minErrorIdx:
                                    stopMaxError = 1
                        
                        
                        
                        
                    maxCandidate = candidates[ maxErrorIdx ] 

                    
                    #Removing elements from candidate set
                    #Remove the selected candidate from the candidate set
                    candidates.remove(  minCandidate )
                    
                    #When there was only one element, that element would be the min and max
                    #Since the previous line remove the min, the max is only removed if there are 
                    #element in the candidate set
                    
                    if len( candidates ) > 0 : 
                        #Remove the worst candidate from the candidate set
                        candidates.remove( maxCandidate )
                    
                    
                #print( 'candidates', candidates )
                if len( candidates ) == 0 :
                    # if candidate set is empty, the algorithm stops
                    stop = 1
                    
                print( 'minError, selected set', bestError, selected)
                    
 

            selectedFeaturesFolds[foldIdx,selected]+=1
            
            orderSel = 1
            for sel in selected:
                relevanceFeaturesFolds[ foldIdx, sel ] = orderSel
                orderSel+=1
            
            
            foldIdx+=1
        
        
        selected = np.where( np.sum( selectedFeaturesFolds, axis = 0) > folds/2 )[0]
        
        if len(selected)==0:
            #No features selected - select all those chosen at least 1
            selected = np.where( np.sum( selectedFeaturesFolds, axis = 0) > 0 )[0]
        
        selected = selected.tolist()
        
        #Avarege of relevance accros folds
        averageRelevance = np.mean( relevanceFeaturesFolds, axis=0)
        selectedAverage = averageRelevance[selected]
        
        #These are the features order by relevance
        sortRelevance = np.argsort( selectedAverage )
        
        
        
        
        
        
        return selected, clusters, sortRelevance
        