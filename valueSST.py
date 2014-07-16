# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:32:12 2014

@author: tdelaet
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import permutations
from random import sample
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os


plt.close('all')

nameMethod = 'SST'
# exactly the same as ET but in SST 1 means indicating answer as possible

numberAlternatives = 4

wrongAnswer = -1.0/(numberAlternatives-1)
correctAnswer = 1

alpha = 0.88
beta = 0.88
lambda_v = 2.25

if not os.path.exists(nameMethod):
    os.makedirs(nameMethod)

def valueFunction(x):
    #0<=alpha<=beta<=1
    value = [ np.power(xi,alpha) if xi>=0 else -lambda_v * np.power(-xi,beta) for xi in x]
    return value
    
    
def valueOfAnswer(answer,probs):
    if sum(probs)!=1:
        print("ERROR: probabilities do not sum to 1")
        
    return sum(valueFunction([correctAnswer])*answer*probs) + sum(valueFunction([wrongAnswer])*answer*(1-probs))
    
def valueOfAnswers(couplesAnswersProbs):
    values = []
    for couple in  couplesAnswersProbs:
        values.append(valueOfAnswer(couple[0],couple[1]))
    return values        

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def cartesianList(lists):
    cartList=[]
    for xi in itertools.product(*lists):
        #print xi
        cartList.append(np.asarray(xi))
    return cartList
    

    
# evenly sampled time at 200ms intervals
x = np.arange(-5, 5, 0.2)

# red dashes, blue squares and green triangles

plt.figure()
plt.plot(x, valueFunction(x), 'r--')
plt.xlabel("x")
plt.ylabel("v(x)")
plt.show()    
plt.grid(1)


answer = np.array([1,0,0,0]) # antwoord A is geëlimineerd
probsAnswer = np.array([0.1,0.9,0.0,0.0])
print valueOfAnswer(answer, probsAnswer)


## SIMULATION 1
# simulation with three answers that can be excluded so answer is [1,0,0,0]
answer = np.array([1,0,0,0]) # alles behalve antwoord A is geëlimineerd
probsAnswer = np.array([1,0,0.0,0.0])
print valueOfAnswer(answer, probsAnswer) # the value is the score of a correct answer


## SIMULATION 2
# simulation of doubt between two answers A and B [X,X,0,0]

step = 0.02
prob1 = np.arange(0,1+step,step)
prob2 = 1-prob1
prob3 = 0
prob4 = 0
probs = []
for counter in np.arange(0,len(prob1)):
    probs.append(np.asarray([prob1[counter],prob2[counter],prob3,prob4]))
    
possibleAnswers = cartesianList([[0,1],[0,1],[0],[0]])
couplesPossibleAnswersProbs = cartesianList([possibleAnswers,probs])
values = valueOfAnswers(couplesPossibleAnswersProbs)

plt.figure()


possibleAnswersFromCouples = [couple[0] for couple in couplesPossibleAnswersProbs]
probsFromCouples = [couple[1] for couple in couplesPossibleAnswersProbs]

valuesSep = []
for possibleAnswer in possibleAnswers:
    #iterate over possible answers
    # get the indices of the couples with this possible answers
    indices = [i for i,couple in enumerate(couplesPossibleAnswersProbs) if np.array_equal(couple[0],possibleAnswer)]
    #get the associated probabilities
    probAnswer1 = [probsFromCouples[index][0] for index in indices]
    valuesAnswer = [values[index] for index in indices]
    valuesSep.append(valuesAnswer)
    plt.plot(probAnswer1,valuesAnswer,label=str(possibleAnswer))

valuesArray = np.asarray(valuesSep)
maxValue = valuesArray.max(axis=0)
maxAnswer = valuesArray.argmax(axis=0)

plt.plot(probAnswer1,maxValue,'r*',label="best Answer")
plt.show()    
plt.grid(1)
plt.xlabel("P(1)")
plt.ylabel("v(answer)")
legend = plt.legend(loc='upper center', shadow=True)
plt.savefig(nameMethod + '/2D_alpha'+str(alpha)+'_beta'+str(beta)+'.png')

## SIMULATION 3
# simulation of doubt between three answers A, B and C [X,X,X,0]


step = 0.02
prob1 = np.arange(0,1+step,step)
prob2 = np.arange(0,1+step,step)
prob3 = np.arange(0,1+step,step)
probs12 = cartesianList([prob1,prob2])
probs=[]
for prob in probs12:
    if 1-sum(prob)>=0:
        probs.append(np.append(prob,[1-sum(prob),0]))

possibleAnswers = cartesianList([[0,1],[0,1],[0,1],[0]])
couplesPossibleAnswersProbs = cartesianList([possibleAnswers,probs])
values = valueOfAnswers(couplesPossibleAnswersProbs)


possibleAnswersFromCouples = [couple[0] for couple in couplesPossibleAnswersProbs]
probsFromCouples = [couple[1] for couple in couplesPossibleAnswersProbs]



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
N = len(possibleAnswers)
color_lvl = 8
rgb = np.array(list(permutations(range(0,256,color_lvl),3)))/255.0
colors = np.arange(0,1,1.0/len(possibleAnswers))+1.0/len(possibleAnswers)
counter = 0
valuesSep = []
for possibleAnswer in possibleAnswers:
    
    #iterate over possible answers
    # get the indices of the couples with this possible answers
    indices = [i for i,couple in enumerate(couplesPossibleAnswersProbs) if np.array_equal(couple[0],possibleAnswer)]
    #get the associated probabilities
    probAnswer1 = [probsFromCouples[index][0] for index in indices]
    probAnswer2 = [probsFromCouples[index][1] for index in indices]
    valuesAnswer = [values[index] for index in indices]
    valuesSep.append(valuesAnswer)
    p = ax.scatter(probAnswer1,probAnswer2,valuesAnswer,s=80,c=cm.jet(counter/(len(possibleAnswers)-1)),label=str(possibleAnswer),cmap=cm.jet)
    counter+=1

valuesArray = np.asarray(valuesSep)
maxValue = valuesArray.max(axis=0)
maxAnswer = valuesArray.argmax(axis=0)
#probAnswer1MaxAnswer = probAnswer1Sep[maxValue]

p = ax.scatter(probAnswer1,probAnswer2,maxValue,s=80,marker='*',c='r',label=str(possibleAnswer))
    
plt.grid(1)
ax.set_xlabel("P(1)")
ax.set_ylabel("P(2)")
ax.set_zlabel("v(answer)")

counter = 0
legend1=[]
legend2=[]
for answer in set(maxAnswer):
    print counter/len(possibleAnswers)
    print cm.jet(counter/len(possibleAnswers))
    legend1.append(plt.Circle((0, 0), 1, fc=cm.jet(counter/(len(possibleAnswers)-1))))
    legend2.append(str(possibleAnswers[answer]))
    counter+=1 
ax.legend(legend1, legend2)



# plot only the best decision
#in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
counter = 0
for answer in set(maxAnswer):
    indices = np.where(maxAnswer==answer)
    p1 = [probAnswer1[index] for index in indices[0]]
    p2 = [probAnswer2[index] for index in indices[0]]
    vs = [maxValue[index] for index in indices[0]]
    p = ax.scatter(p1,p2,vs,s=80,c=cm.jet(counter/(len(possibleAnswers)-1)),label=str(possibleAnswer))
    counter +=1

plt.grid(1)
ax.set_xlabel("P(1)")
ax.set_ylabel("P(2)")
ax.set_zlabel("v(answer)")
ax.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8)
ax.legend(legend1, legend2)
plt.title("maximal value and answer leading to it")
plt.savefig(nameMethod + '/maxValueAnswer_alpha'+str(alpha)+'_beta'+str(beta)+'.png')

# plot only the best decision
#in 2d
fig = plt.figure()
ax = fig.add_subplot(111)
counter = 0
for answer in set(maxAnswer):
    indices = np.where(maxAnswer==answer)
    p1 = [probAnswer1[index] for index in indices[0]]
    p2 = [probAnswer2[index] for index in indices[0]]
    p = ax.scatter(p1,p2,s=80,c=cm.jet(counter/(len(possibleAnswers)-1)),label=str(possibleAnswer))
    counter +=1

plt.grid(1)
ax.set_xlabel("P(1)")
ax.set_ylabel("P(2)")
ax.legend(legend1, legend2)
plt.title("answer leading to maximal value")
plt.savefig(nameMethod + '/maxAnswer_alpha'+str(alpha)+'_beta'+str(beta)+'.png')

# plot only the maximum value
#in 2d
fig = plt.figure()
ax = fig.add_subplot(111)
p = ax.scatter(probAnswer1,probAnswer2,s=80,c=maxValue)

plt.grid(1)
ax.set_xlabel("P(1)")
ax.set_ylabel("P(2)")
plt.title("maximal value")
cbar = plt.colorbar(p)
cbar.set_label('maximal value')
plt.savefig(nameMethod + '/maxValue_alpha'+str(alpha)+'_beta'+str(beta)+'.png')

plt.show()  