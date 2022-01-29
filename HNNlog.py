#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:30:12 2017

@author: tarachari
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:08:31 2017

@author: tarachari
"""

from neuralNet import runNeuralNet
#from numpy import log1p
from math import exp
import numpy as np

def logSum(a, b):
    if a > b:
        return a + np.log1p(exp(b - a))
    elif a == -np.inf and b == -np.inf:
        return -np.inf
    else: 
        return b + np.log1p(exp(a - b))
    
#calculate forward probabilities
def fwd(numCys,fTrans,fEmisNN):
    numCys = numCys
  
    # ----- initialize transition/emission probs ------
    #transition probs
    fTrans = fTrans
  
    #emission probs,  no start symbol included at beginning of seq
    fEmisNN = fEmisNN
  
  #-----initialize forward matrix-----
    forward = np.zeros([fEmisNN.shape[0], numCys], dtype='f')#4 states/rows
    numStates = fEmisNN.shape[0]
    forward = np.log(forward)
    
    for state in range(0,numStates): #row 1:B1,2:B2, 3:NB1, 4:NB2 state
    
        forward[state,0] = fTrans[0,(state+1)]+fEmisNN[state,0]
    
  
  #-----fill in forward matrix-----
  
    for y in range(1,numCys):
    
        for state in range(0,numStates):
      
            b1 = forward[0,y-1]+fTrans[1,state+1] # prob coming from state b1
            b2 = forward[1,y-1]+fTrans[2,state+1] # prob coming from state b2
            nb1 = forward[2,y-1]+fTrans[3,state+1] # prob coming from state nb1
            nb2 = forward[3,y-1]+fTrans[4,state+1] # prob coming from state nb2
            sums = logSum(logSum(b1,b2),logSum(nb1,nb2))
          
            forward[state,y] = sums+fEmisNN[state,y] #logSum + log emission prob
   
    b1x= forward[0,(numCys-1)]
    b2x = forward[1,(numCys-1)]
    nb1x = forward[2,(numCys-1)]
    nb2x = forward[3,(numCys-1)] 
    Px = logSum(logSum(b1x,b2x),logSum(nb1x,nb2x))
    #print(Px)
  
    return forward


#calculate backward probabilities
def back(numCys,bTrans,bEmisNN):
    numCys = numCys
  
  # ----- initialize transition/emission probs ------
  #transition probs
    bTrans = bTrans
  
  #emission probs, no start symbol included at beginning of seq
    bEmisNN = bEmisNN
    numStates = bEmisNN.shape[0]
  
  #-----initialize forward matrix-----
    backward = np.zeros([bEmisNN.shape[0], numCys], dtype='f') # 4 states
    
    
    for state in range(0,numStates):#row 1:B1,2:B2, 3:NB1, 4:NB2 state
    
        backward[state,(numCys-1)] = 1
   
  #-----fill in backward matrix-----
    backward = np.log(backward)
    for y in range((numCys-2),-1,-1):
    
        for state in range(0,numStates):
      
            b1 = backward[0,y+1]+bTrans[1,state+1]+bEmisNN[state,y+1] # prob coming from state b1
            b2 = backward[1,y+1]+bTrans[2,state+1]+bEmisNN[state,y+1] # prob coming from state b2
            nb1 = backward[2,y+1]+bTrans[3,state+1]+bEmisNN[state,y+1]
            nb2 = backward[3,y+1]+bTrans[4,state+1]+bEmisNN[state,y+1]
      
            backward[state,y] = logSum(logSum(b1,b2),logSum(nb1,nb2)) #logSum + log emission prob
      
  
  # check overall P(x) and compare to forward
#    b1x= backward[0,0]+bEmisNN[0,0]+bTrans[0,1]
#    b2x = backward[1,0]+bEmisNN[1,0]+bTrans[0,2]
#    nb1x = backward[2,0]+bEmisNN[2,0]+bTrans[0,3]
#    nb2x = backward[3,0]+bEmisNN[3,0]+bTrans[0,4]
#    Px = logSum(logSum(b1x,b2x),logSum(nb1x,nb2x))
#    print(Px)
  
    return backward

def getEmis(seq,fwdProbs,backProbs,emis):

    E = np.zeros([emis.shape[0], emis.shape[1]], dtype='f') #New emission probabilities for each base
    E = np.log(E)
    params = [1,-1]
  
    for out in params:
            b1Total = -np.inf
            b2Total = -np.inf
            nb1Total = -np.inf
            nb2Total = -np.inf
            
            for i in range(0,len(seq)):
                    if seq[i][0] == out:
                            b1x = fwdProbs[0,i]+backProbs[0,i] #row 0 was state b1
                            b2x = fwdProbs[1,i]+backProbs[1,i]
                            nb1x = fwdProbs[2,i]+backProbs[2,i]
                            nb2x = fwdProbs[3,i]+backProbs[3,i]
                            Px= logSum(logSum(b1x,b2x),logSum(nb1x,nb2x))#fwd and back probs from states
                            
                           
                            b1Total = logSum(b1Total,(b1x-Px)) #sum(f_k(i)b_k(i) w/ k = b1) in log space
                            b2Total = logSum(b2Total,(b2x-Px))#sum(f_k(i)b_k(i) w/ k = b2) in log space
                            nb1Total = logSum(nb1Total,(nb1x-Px))
                            nb2Total = logSum(nb2Total,(nb2x-Px))
                    # if x[i] == base
            #seq for loop
            
            E[0,out] = b1Total
            E[1,out] = b2Total
            E[2,out] = nb1Total
            E[3,out] = nb2Total
            
    #for loop for bases
  
    E[0,:] = E[0,:]-logSum(E[0,0],E[0,1]) #E_k(b)/sum(E_k(b')) for k = b1
    E[1,:] = E[1,:]-logSum(E[1,0],E[1,1]) #E_k(b)/sum(E_k(b')) for k = b2
    E[2,:] = E[2,:]-logSum(E[2,0],E[2,1]) #E_k(b)/sum(E_k(b')) for k = nb1
    E[3,:] = E[3,:]-logSum(E[3,0],E[3,1]) #E_k(b)/sum(E_k(b')) for k = nb2

    return E

#get mu, probability of transitioning
def getTrans(trans,emis,seq,fwdProbs,backProbs):
    T = np.zeros([trans.shape[0], trans.shape[1]], dtype='f') 
    T[0,:] = [0,0.5,0,0.5,0] #probs from start
    T = np.log(T)
    
    for l in range(0,4):
        b1 = -np.inf
        b2 = -np.inf
        nb1 = -np.inf
        nb2 = -np.inf
        #calc A_ij counts, sum, and divide sum by (A_ij for all j's)
        for i in range(0,(len(seq)-1)): #no transition at end (so -1)
            b1x = fwdProbs[0,i]+backProbs[0,i] #row 1 was state b1
            b2x = fwdProbs[1,i]+backProbs[1,i]
            nb1x = fwdProbs[2,i]+backProbs[2,i]
            nb2x = fwdProbs[3,i]+backProbs[3,i]
            Px = logSum(logSum(b1x,b2x),logSum(nb1x,nb2x)) #fwd and back probs from states
            
            #    hx= fwdProbs[1,ncol(fwdProbs)];
            #    lx = fwdProbs[2,ncol(fwdProbs)];
            #    Px =logSum(hx,lx);
            
            #f_k(i)*a_kl*e_l(b)*b_l(i+1)/P(x) in log space calculated for all transitions
            b1 = logSum(b1,(fwdProbs[0,i]+backProbs[l,i+1]+trans[1,l+1]+emis[l,i+1])-Px)
            b2 = logSum(b2 ,(fwdProbs[1,i]+backProbs[l,i+1]+trans[2,l+1]+emis[l,i+1])-Px)
            nb1 = logSum(nb1 ,(fwdProbs[2,i]+backProbs[l,i+1]+trans[3,l+1]+emis[l,i+1])-Px)
            nb2 = logSum(nb2 , (fwdProbs[3,i]+backProbs[l,i+1]+trans[4,l+1]+emis[l,i+1])-Px)
            
        T[1,(l+1)] = b1
        T[2,(l+1)] = b2
        T[3,(l+1)] = nb1
        T[4,(l+1)] = nb2
        
      
    T[1,:] = T[1,:]-logSum(T[1,0],logSum(logSum(T[1,1],T[1,2]),logSum(T[1,3],T[1,4])))
    T[2,:] = T[2,:]-logSum(T[2,0],logSum(logSum(T[2,1],T[2,2]),logSum(T[2,3],T[2,4])))
    T[3,:] = T[3,:]-logSum(T[3,0],logSum(logSum(T[3,1],T[3,2]),logSum(T[3,3],T[3,4])))
    T[4,:] = T[4,:]-logSum(T[4,0],logSum(logSum(T[4,1],T[4,2]),logSum(T[4,3],T[4,4])))
    
    return T

#calculate viterbi path (predicted Cys bonding state)
def viterbi(numCys,trans,emisNN):
    
    numCys = numCys
    numStates = emisNN.shape[0]
    
    # ----- in all matrices row 1 = s (Start)------
    #transition probs
    trans = trans
    
    #emission probs, no star
    emisNN = emisNN
    
    #-----initialize viterbi matrix-----
    vit = np.zeros([numStates,numCys],dtype='f')
    vit = np.log(vit)
    T = np.zeros([numStates,numCys],dtype=int) #set up backpointer matrix
    
    for state in range(0,numStates):
        
        vit[state,0] = trans[0,state+1]+emisNN[state,0]
        T[state,0] = -1 #point back to start
        
        
    #-----fill in vit matrix-----
    
    for y in range(1,numCys):
            
            for state in range(0,numStates):
                    
                    b1Prob = vit[0,y-1]+trans[0,state+1]+emisNN[state,y] #probability of coming from state b1
                    b2Prob = vit[1,y-1]+trans[1,state+1]+emisNN[state,y] #probability of coming from state b2
                    nb1Prob = vit[2,y-1]+trans[2,state+1]+emisNN[state,y] #probability of coming from state nb1
                    nb2Prob = vit[3,y-1]+trans[3,state+1]+emisNN[state,y] #probability of coming from state nb2
                    
                    vec = [b1Prob,b2Prob,nb1Prob,nb2Prob]
                    index = np.argmax(vec)
                    
                    vit[state,y] = vec[index] #save max probability
                    T[state,y] = index #save max row 		
                    
                    
    #-----traceback-----
    out = []
    
    #find max of last column to get which state to start traceback at
    finIndex = np.argmax([vit[0,(numCys-1)],vit[1,(numCys-1)],vit[2,(numCys-1)],vit[3,(numCys-1)]])
    i = finIndex
    j = numCys-1
    
    if i == 0 or i == 1:
            out += [1]     
    else:
            out += [-1]

    while i!= -1 and j!= -1:    
            if T[i,j] == 0 or T[i,j] == 1: # 1/bonded
                    
                    i = T[i,j]
                    out += [1]
                    j = j-1                   
            elif T[i,j] == 2 or T[i,j] == 3: # -1/free
                    
                    i = T[i,j]
                    out += [-1]
                    j = j-1                
            else:
                    break
                    
                    
    return out
	

#------------------------------------------------------------------------------------------

normRes,numRes2,testIDs,x = runNeuralNet()
newRes2 = numRes2
allOutcomes = {}

for ID in x:
    testIndices = [i for i, j in enumerate(testIDs) if j == ID]
    param = 2 #B or F
    #[[.7,.3],[.8,.2],[.3,.7],[.3,.7]]]
    emis = np.array([[.7,.3],[.8,.2],[.3,.7],[.3,.7]])
    trans = np.array([[0,0.5,0,0.5,0],[0,0,0.6,0,0.4],[0,0.6,0,0.4,0],
                      [0,0.4,0,0.6,0],[0,0,0,0,1.0]])
    trans = np.log(trans)
    emis = np.log(emis)
        
    if(len(testIndices) > 0):
        
        numCys = len(testIndices)
        seq = newRes2[testIndices]
        likelihoods = []
        
        #while len(likelihoods)<2 or abs((likelihoods[len(likelihoods)-1]-likelihoods[len(likelihoods)-1])) > .001:
        for j in range(0,50):
            initNN = np.zeros([numCys, param], dtype='f')
            initNN = np.log(initNN)
            emisNN = np.zeros([emis.shape[0], numCys], dtype='f')
            emisNN = np.log(emisNN)
            
            for pos in range(0,numCys):
                initNN[pos,0] = normRes[testIndices[pos]]
                initNN[pos,1] = 1 - initNN[pos,0]
                initNN[pos,0] = np.log(initNN[pos,0])
                initNN[pos,1] = np.log(initNN[pos,1])
                
                
                for row in range(0,emis.shape[0]):
                    summ = logSum(emis[row,0]+initNN[pos,0], emis[row,1]+initNN[pos,1])
                    emisNN[row,pos] = summ
                    
            #emisNN = np.log(emisNN)
            
            
            forward = fwd(numCys, trans, emisNN)
            backward = back(numCys, trans, emisNN)
            
            emis = getEmis(seq,forward,backward,emis)
            trans = getTrans(trans,emisNN,seq,forward,backward)
            
            #x= logSum(np.log(0)*2,np.log(0)*.08)
            
            b1x= forward[0,(numCys-1)]
            b2x = forward[1,(numCys-1)]
            nb1x = forward[2,(numCys-1)]
            nb2x = forward[3,(numCys-1)]
            Px = logSum(logSum(b1x,b2x),logSum(nb1x,nb2x))
            
            likelihoods += [Px] #save all likelihoods, to compare
    
        
        allOutcomes[ID] = viterbi(numCys,trans,emisNN)
  
 #test HNN accuracy, usually b/w 70-80%
true = 0.0
false = 0.0
for ID in x:
    testIndices = [i for i, j in enumerate(testIDs) if j == ID]
    if(len(testIndices) > 0):
        seq = newRes2[testIndices]
        predSeq = allOutcomes[ID]
        
        for i in range(0,len(seq)):
            if seq[i] == predSeq[i]:
                true = true+1
            else:
                false = false +1
        
#        if np.all(seq == predSeq):
#            true = true+1
#        else:
#            false = false+1
#            print(ID)
        

print(true/(true+false))


