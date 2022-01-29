# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import theanets
import numpy as np
import os

#find char occurences in string
def findOccurences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

#get freqs for all aa's in window around cys's (for all proteins)
#get correct bonding values for all cys's used
def getNeuralData():
    #go through all psi0 files (only includes count matrix)
    #export, to dictionary, freq's for AA's in each protein (for each PDB ID)
    rootdir = '/Users/tarachari/Desktop/CS/Final_Proj/cysdataset/profiles'
    
    #make PDB + aaFreqs dict
    pdbDict = {}
    window = 13;
    
    for filename in os.listdir(rootdir):
        #print(filename)
        filepath = rootdir+"/"+filename
        if (filename.endswith(".psi0") and os.stat(filepath).st_size != 0):
    
            
            x = np.loadtxt(filepath, skiprows=1)
            sums = x.sum(axis=0)
            total = sum(sums[2:22])
            dictFreqs = sums[2:22]/total
                
            keys = ['V','L','I','M','F','W','Y','G','A','P','S','T','C','H','R','K','Q','E','N','D']
            aaFreqs = dict(zip(keys, dictFreqs))
                
            PDB_ID = filename[:-5]
            #print(PDB_ID)
    
            #make dict w/ key = PDB_id value = aaFreqs
            pdbDict[PDB_ID] = aaFreqs
            
    fullMatrix = []
    fullOutput = []
    fullPDB_ID = []
    rootdir = '/Users/tarachari/Desktop/CS/Final_Proj/cysdataset/seq'
    for filename in os.listdir(rootdir):
        PDB_ID = filename[:-4]
        if PDB_ID in pdbDict:
            
            filepath = rootdir+"/"+filename
            if (filename.endswith(".seq") and os.stat(filepath).st_size != 0):
        
                matrix = np.loadtxt(filepath,delimiter='\t',dtype=np.dtype([('aa', '|S1'), ('bond', 'int')]),usecols=(0,1))
                
                #get string of aa's
                aaString = ''.join(matrix['aa'])
                aaVec = matrix['aa']
                #get string of 0's and 1's (or -1's, same as 2)
                binaryVec = matrix['bond'] 
                binaryVec[:] = [-1 if b== 2 else b for b in binaryVec]
                
                
                cysPos = findOccurences(aaString, 'C')
                
                
                #get AA freqs around each cys
                for i in cysPos:
                    begin = i-window
                    end = i+window+1
                    bounds = range(begin,end)
                    allInputs = []
                    output = 0
                    #don't use cys that => windows out of bounds
                    if (end < len(aaVec) and begin >= 0):
                        output = binaryVec[i] #save 1 or -1 value for cys
                        for j in bounds:
                            
                            inNeuron = [0] * 20
                            aa = aaVec[j]
                            if aa in keys:
                                index = keys.index(aa)
                                    
                                entry = pdbDict[PDB_ID]
                                freq = entry[aa]
                                    
                                inNeuron[index] = freq
                                
                            allInputs.extend(inNeuron)
                        
                        #allInputs.append(PDB_ID) 
                        fullPDB_ID.append(PDB_ID)#save where input neurons from
                        #print(len(allInputs))
                        fullMatrix.append(allInputs)
                        fullOutput.append(output)
    
    
    fullMatrix = np.array(fullMatrix)
    fullOutput = np.array(fullOutput)
    fullPDB_ID = np.array(fullPDB_ID)
    
    return fullMatrix,fullOutput,fullPDB_ID

#inData, outData, ids = getNeuralData()
