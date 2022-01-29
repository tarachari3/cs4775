#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:48:39 2017

@author: tarachari
"""

import theanets
#import os
import numpy as np
from dataExport import findOccurences,getNeuralData
from sklearn.metrics import classification_report, confusion_matrix
import random
#import matplotlib.pyplot as plt

#runs neural net, to get emission probs + predictions
def runNeuralNet():
    allData, output, ids = getNeuralData()
    neurons = 540;
    
    rootdir = '/Users/tarachari/Desktop/CS/Final_Proj/cysdataset/test'
    
    #testRes = open('/Users/tarachari/Desktop/CS/Final_Proj/cysdataset/testRes.txt','w') 
    
    #cross-validation
    #for filename in os.listdir(rootdir):
    #    if "test" in filename:
    filepath = rootdir+"/"+"test4"#rootdir+"/"+"test0" or filename
    
    x = np.loadtxt(filepath,dtype='|S15')
    
    #get test set cells
    indices = []
    for s in x:
        indices += [i for i, j in enumerate(ids) if j == s]
      
    allData = np.array(allData, dtype = 'f')
    
    trainData = np.delete(allData, indices, axis=0)
    trainOutput = np.delete(output, indices, axis=0)
    size = trainOutput.shape[0]#-1
    
    #90/10% train/valid
    forValid = random.sample(range(0, size),int(round(size*0.1)))
    validData = trainData[forValid]
    validOutput = np.array(trainOutput[forValid],dtype = 'f')#np.int32(trainOutput[forValid])
    validOutput = validOutput.reshape(validOutput.shape[0],-1)
    
    trainData = np.delete(trainData, forValid, axis=0)
    trainOutput = np.delete(trainOutput, forValid, axis=0)
    trainOutput = np.array(trainOutput,dtype='f') #np.int32(trainOutput)
    trainOutput = trainOutput.reshape(trainOutput.shape[0],-1)
    
    train = (trainData, trainOutput)
    valid = (validData, validOutput)
    
    net = theanets.Regressor([neurons,1])
    #net.train([trainData, trainOutput])
    net.train(train,valid,algo='rmsprop',learning_rate=0.01,
              momentum=0.5, min_improvement = 0.001, patience = 50)
    
    #test data is from indices left out
    testData = allData[indices]
    testOutput = np.array(output[indices],dtype='f')#np.int32(output[indices])
    results = net.predict(testData)
    
    #param = net.find('hid1', 'w')
    #values = param.get_value()
    
    newRes = np.zeros([len(results), 1], dtype=int)
    for r in range(0,len(results)):
        if results[r,0] < 0.0:
            newRes[r] = -1
        else: 
            newRes[r] = 1
    
    print(classification_report(testOutput, newRes))
    #testRes.write(classification_report(testOutput, newRes)) 
    #print(confusion_matrix(testOutput, newRes2))
    #testRes.close()
    
    #get Probability that cys is -1 or 1 (value b/w 0-1)
    min = np.amin(results)
    max = np.amax(results)
    oldRange = max - min  
    newRange = 1 - 0  #convert to 0 to 1 range
    #newValue = (((i - min) * newRange) / oldRange) + 0
    normRes = [((((i - min) * newRange) / oldRange) + 0) for i in results]#[(i-min)/(max-min) for i in results]
    newRes2 = np.zeros([len(normRes), 1], dtype=int)
    new1cutoff = 1-(newRange/oldRange)*(max)
    for n in range(0,len(normRes)):
        if normRes[n][0] >= new1cutoff:
            newRes2[n] = 1
        else:
            newRes2[n] = -1
            
         
    
            
    print(classification_report(testOutput, newRes2)) #check that it's comparable
    print(confusion_matrix(testOutput, newRes2))
    
    #find % accuracy per protein, # fully correct proteins/total # proteins.
    testIDs = ids[indices]
    allPercs = []
    
    testIndices = []
    for ID in x:
        testIndices = [i for i, j in enumerate(testIDs) if j == ID]
        if(len(testIndices) > 0):
            perc = 0.0
            
            for index in testIndices:
                if testOutput[index] == newRes2[index,0]:
                    perc += 1.0
                    
            perc = perc/len(testIndices)
            
            allPercs.append([perc])
    count = 0.0
    for i in allPercs:
        if i[0] == 1.0:
            count += 1.0
    meanAcc = count/(len(allPercs))
    print("Mean protein accuracy: ")
    print(meanAcc)
    
    return normRes,newRes2,testIDs,x

