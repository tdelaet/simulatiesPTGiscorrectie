# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:32:12 2014

@author: tdelaet
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import itertools
from random import sample
import random
from numpy.random import random_sample
from math import log
import matplotlib.cm as cm

def sample_discrete(values, probabilities, size):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]
plt.close('all')

# probability that misconception: i.e. that correct answer is not among the ones doubting on
prob_misConception = 0.1   

#probability that doubt between 0,1,2,3 answers
prob_numAlternativesDoubt = [0.4,0.3,0.2,0.1]

alpha = 0.88
beta = 0.88

lambda_vector = np.arange(1,10,1)



#number of questions per exam
numQuestions = 25

#number of exams to simulate
numExams = 10

#simuate numSamples
numSamples = numQuestions* numExams

methods = ["NR","PS","FS","ET","SST","PT_Q","PT_S","PT_L"]
methodsChoice = methods[0:5]
methodsNoChoice = methods[5:len(methods)+1]

#number of alternatives
numberAlternatives = 4

scoreWrongAnswer_NR = 0
scoreCorrectAnswer_NR = 1.0
scoreBlankAnswer_NR = 0

scoreWrongAnswer_PS = 0
scoreCorrectAnswer_PS = 1.0
scoreBlankAnswer_PS = 1.0/(numberAlternatives-1)

scoreWrongAnswer_FS = -1.0/(numberAlternatives-1)
scoreCorrectAnswer_FS = 1.0
scoreBlankAnswer_FS = 0

scoreWrongAnswer_ET = -1
scoreCorrectAnswer_ET = 1.0/(numberAlternatives-1)

scoreWrongAnswer_SST = -1.0/(numberAlternatives-1)
scoreCorrectAnswer_SST = 1


def cartesianList(lists):
    cartList=[]
    for xi in itertools.product(*lists):
        #print xi
        cartList.append(np.asarray(xi))
    return cartList    



# score for different methods           
def score_NR(answer,correctanswer):
    #answer indicated as correct
    indicatedAnswer = np.where(np.asarray(answer)==1)[0]
    if indicatedAnswer.size: # there is one element indicated as correct
        if indicatedAnswer[0] == correctanswer:
            return scoreCorrectAnswer_NR
        else:
            return scoreWrongAnswer_NR
    else:
        return scoreBlankAnswer_NR
        
def score_PS(answer,correctanswer):
    #answer indicated as correct
    indicatedAnswer = np.where(np.asarray(answer)==1)[0]
    if indicatedAnswer.size: # there is one element indicated as correct
        if indicatedAnswer[0] == correctanswer:
            return scoreCorrectAnswer_PS
        else:
            return scoreWrongAnswer_PS
    else:
        return scoreBlankAnswer_PS
                
def score_FS(answer,correctanswer):
    #answer indicated as correct
    indicatedAnswer = np.where(np.asarray(answer)==1)[0]
    if indicatedAnswer.size: # there is one element indicated as correct
        if indicatedAnswer[0] == correctanswer:
            return scoreCorrectAnswer_FS
        else:
            return scoreWrongAnswer_FS
    else:
        return scoreBlankAnswer_FS
        
def score_ET(answer,correctAnswer):
    answer = tuple(answer)
    answerWrongAnswers = np.asarray(answer[0:correctAnswer] + answer[correctAnswer+1:len(answer)+1])
    return answer[correctAnswer]*scoreWrongAnswer_ET + sum(answerWrongAnswers*scoreCorrectAnswer_ET)
        
def score_SST(answer,correctAnswer):
    answer = tuple(answer)
    answerWrongAnswers = np.asarray(answer[0:correctAnswer] + answer[correctAnswer+1:len(answer)+1])
    return answer[correctAnswer]*scoreCorrectAnswer_SST + sum(answerWrongAnswers*scoreWrongAnswer_SST)
    
def score_PT_Q(probs,correctAnswer): #quadratic
    return 2*probs[correctAnswer] - np.dot(probs,probs)
      
def score_PT_S(probs,correctAnswer): #spherical
    return probs[correctAnswer] / np.sqrt(np.dot(probs,probs)  )
    
def score_PT_L(probs,correctAnswer): #spherical
    if probs[correctAnswer] == 0:
        return -10000
    else:
        return log(probs[correctAnswer])
    
# value for different methods       
def valueOfAnswer_NR(answer,probs):
    #if sum(probs)!=1:
    #    print("ERROR: probabilities " + str(probs) + " do not sum to 1")
    indicatedAnswer = np.where(np.asarray(answer)==1)[0]
    if not(indicatedAnswer.size): # there is no correct answer indicated => so answers is blank
        return scoreBlankAnswer_NR
    else:        
        return sum(valueFunction([scoreCorrectAnswer_NR])*answer*probs) + sum(valueFunction([scoreWrongAnswer_NR])*answer*(1-probs))
        
def valueOfAnswer_PS(answer,probs):
    #if sum(probs)!=1:
    #    print("ERROR: probabilities " + str(probs) + " do not sum to 1")
    indicatedAnswer = np.where(np.asarray(answer)==1)[0]
    if not(indicatedAnswer.size): # there is no correct answer indicated => so answers is blank
        return scoreBlankAnswer_PS
    else:   
        return sum(valueFunction([scoreCorrectAnswer_PS])*answer*probs) + sum(valueFunction([scoreWrongAnswer_PS])*answer*(1-probs))
        
def valueOfAnswer_FS(answer,probs):
    #if sum(probs)!=1:
    #    print("ERROR: probabilities " + str(probs) + " do not sum to 1")
    indicatedAnswer = np.where(np.asarray(answer)==1)[0]
    if not(indicatedAnswer.size): # there is no correct answer indicated => so answers is blank
        return scoreBlankAnswer_FS
    else:   
        return sum(valueFunction([scoreCorrectAnswer_FS])*answer*probs) + sum(valueFunction([scoreWrongAnswer_FS])*answer*(1-probs))
            
def valueOfAnswer_ET(answer,probs):
    #if sum(probs)!=1:
    #   print("ERROR: probabilities do not sum to 1")
        
    return sum(valueFunction([scoreWrongAnswer_ET])*answer*probs) + sum(valueFunction([scoreCorrectAnswer_ET])*answer*(1-probs))

def valueOfAnswer_SST(answer,probs):
    #if sum(probs)!=1:
    #    print("ERROR: probabilities do not sum to 1")      
    return sum(valueFunction([scoreCorrectAnswer_SST])*answer*probs) + sum(valueFunction([scoreWrongAnswer_SST])*answer*(1-probs))    

#to get values for list of answers and for specific method    
def valueOfAnswers(couplesAnswersProbs,method):
    values = []
    for couple in  couplesAnswersProbs:
        value =globals()['valueOfAnswer_'+method](couple[0],couple[1])
        values.append(value)
    return values   
    
def valueFunction(x):
    value = [ np.power(xi,alpha) if xi>=0 else -lambda_v * np.power(-xi,beta) for xi in x]
    return value
        
# create list of possible answers    
possibleAnswers_NR = list(itertools.permutations([0,0,0,1]))
possibleAnswers_NR = list(set(possibleAnswers_NR))
possibleAnswers_NR.append((0,0,0,0)) 
 
possibleAnswers_PS = possibleAnswers_NR

possibleAnswers_FS = possibleAnswers_NR

possibleAnswers_ET = cartesianList([[0,1],[0,1],[0,1],[0,1]])
possibleAnswers_SST = possibleAnswers_ET
    

for method in methods:
    globals()['average_' + method + "_exams_lambdas"] = []
    globals()['std_' + method + "_exams_lambdas"] = []
 

        
for lambda_v in lambda_vector:
    # for each question sample between how many alternatives there is doubt
    numAlternativesDoubt = sample_discrete(np.array([1,2,3,4]), prob_numAlternativesDoubt,numSamples)
    
    # if doubt between more than one value generate random numbers for probabilities
    probAlternativesSamples = []
    for sample in np.arange(numSamples):
        remainingProb =1 
        counterAlternativesToSample = numAlternativesDoubt[sample]
        counterAlternative = 0
        probAlternativeSample = np.zeros(numberAlternatives)
        #as long as there is doubt between more than one alternative
        while ( counterAlternativesToSample>1 ):
            #sample a probability for the next alternative from between 0.01 and remainingProbability
            values_prob= np.arange(0.001,remainingProb,0.01)
            alternativeProb = sample_discrete(values_prob,np.ones(len(values_prob))/len(values_prob),1) 
            probAlternativeSample[counterAlternative] = alternativeProb        
            #subtract the probability of the alternative from the remaining probability
            remainingProb-=alternativeProb
            # there is one alternative less left to sample
            counterAlternativesToSample-=1
            # we have found the probability for one alternative more
            counterAlternative+=1
        # there is only one alternative left to doubt on: this one should get the remaining probability    
        probAlternativeSample[counterAlternative] = remainingProb
        #shuffle the result such that there is equal doubt between different alternatives
        random.shuffle(probAlternativeSample)
        #append to list of probabilities    
        probAlternativesSamples.append(np.asarray(probAlternativeSample))
    
    
    # simulation where there is no misconception => correct answer is within the ones doubted on
    correctAnswers = []
    for sample in np.arange(numSamples):
        indicesAlternativesInDoubt = np.where(probAlternativesSamples[sample]!=0)[0]
        correctAnswer = sample_discrete(indicesAlternativesInDoubt,np.ones(len(indicesAlternativesInDoubt))/len(indicesAlternativesInDoubt),1)[0]
        correctAnswers.append(correctAnswer)
        
    # simulation where there is a probability for misconception => correct answer is not necessarily within the ones doubted on
    correctAnswers = []
    for sample in np.arange(numSamples):
        indicesAlternativesInDoubt = np.where(probAlternativesSamples[sample]!=0)[0]
        indicesAlternativesNotInDoubt = np.where(probAlternativesSamples[sample]==0)[0]
        # misconception=0 is no misconception
        # misconception=1 is misconception
        if len(indicesAlternativesNotInDoubt) == 0: # all is in doubt so misconception is impossible
            misconception = 0
        else:
            misconception = sample_discrete([0,1],[1-prob_misConception,prob_misConception],1)    
        if misconception: # if there is misconception pick a answer not in doubt as correct anwer
            correctAnswer = sample_discrete(indicesAlternativesNotInDoubt,np.ones(len(indicesAlternativesNotInDoubt))/len(indicesAlternativesNotInDoubt),1)[0]
        else: # if there is no misconception pick a answer still in doubt as correct anwer
            correctAnswer = sample_discrete(indicesAlternativesInDoubt,np.ones(len(indicesAlternativesInDoubt))/len(indicesAlternativesInDoubt),1)[0]
        correctAnswers.append(correctAnswer)
        
    # determine answers using particular answering strategy (maxvalue answer according to behavioural theory (prospect theory))
    for method in methods:
        globals()['scores_' + method] = []
        
    for sample in np.arange(numSamples):     
        for method in methodsChoice:
            # step 1: combine the probabilities with all possible answers  
            globals()['couplesPossibleAnswersProb_' + method] = cartesianList([globals()['possibleAnswers_' + method],[probAlternativesSamples[sample]]])
            # step 2: get the values for all these alternatives
            globals()['values_' + method] = valueOfAnswers(globals()['couplesPossibleAnswersProb_' + method],method)
            # step 3: get the answer with maximal value
            globals()['valuesArray_' + method] = np.asarray(globals()['values_' + method])
            globals()['maxAnswer_' + method] = globals()['valuesArray_' + method].argmax(axis=0)
            # step 4: calculate the score for this 
            globals()['scores_' + method].append(globals()['score_' + method](globals()['possibleAnswers_' + method][globals()['maxAnswer_' + method]],correctAnswers[sample]))
        
        for method in methodsNoChoice:
            # step 4: calculate the score for this 
            globals()['scores_' + method].append(globals()['score_' + method](probAlternativesSamples[sample],correctAnswers[sample]))
    
    for method in methods:
        globals()['totalScore_' + method] = sum( globals()['scores_' + method])/numSamples * 100
        globals()['scores_' + method + '_exams'] =   zip(*[iter(globals()['scores_' + method])]*numQuestions)
        globals()['totalScore_' + method + '_exams'] = sum(globals()['scores_' + method + '_exams'],axis=1)/numQuestions * 100    
        globals()['average_' + method + '_exams'] = np.average(globals()['totalScore_' + method + '_exams'])
        globals()['std_' + method + '_exams'] = np.std(globals()['totalScore_' + method + '_exams'])
        
    fig, ax = plt.subplots()
    plt.boxplot([totalScore_NR_exams,totalScore_PS_exams,totalScore_FS_exams,totalScore_ET_exams,totalScore_SST_exams,totalScore_PT_Q_exams,totalScore_PT_S_exams,totalScore_PT_L_exams])
    plt.ylabel("score exam")
    plt.xlabel("different scoring methods")
    fig.canvas.draw()
    labels = methods
    ax.set_xticklabels(labels)
    plt.ylim([0,100])
    plt.savefig('totalScoreDifferentMethods_lambda'+str(lambda_v)+'_pmis'+str(prob_misConception)+'.png')
    
    for method in methods:
         globals()['average_' + method + '_exams_lambdas'].append(globals()['average_' + method + '_exams'])   
         globals()['std_' + method + '_exams_lambdas'].append(globals()['std_' + method + '_exams'])   



legend1=[]
legend2=[]
counter = 0
fig, ax = plt.subplots()
colors = np.arange(0,1,1.0/len(methods))+1.0/len(methods)
for method in methods:
    average_array = np.asarray(globals()['average_' + method + '_exams_lambdas'])
    std_array = np.asarray(globals()['std_' + method + '_exams_lambdas'])
    line = plt.plot(lambda_vector,average_array)
    plt.setp(line, color=cm.jet(counter/(len(methods)-1)), linewidth=2.0, linestyle = '-')
    #lines = plt.plot(lambda_vector,average_array-std_array,lambda_vector,average_array+std_array)
    #plt.setp(lines, color=cm.jet(counter/(len(methods)-1)), linewidth=2.0, linestyle = '--')
    legend1.append(plt.Circle((0, 0), 1, fc=cm.jet(counter/(len(methods)-1))))
    counter+=1
ax.legend(legend1, labels)
plt.ylim([0,100])
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, 5))
plt.xlabel("lambda")
plt.ylabel("total score exam")
plt.grid(1)
plt.show()    
plt.savefig('totalScoreDifferentMethods_lambdas'+'_pmis'+str(prob_misConception)+'.png')


legend1=[]
legend2=[]
counter = 0
fig, ax = plt.subplots()
colors = np.arange(0,1,1.0/len(methods))+1.0/len(methods)
for method in methods:
    average_array = np.asarray(globals()['average_' + method + '_exams_lambdas'])
    std_array = np.asarray(globals()['std_' + method + '_exams_lambdas'])
    line = plt.plot(lambda_vector,std_array)
    plt.setp(line, color=cm.jet(counter/(len(methods)-1)), linewidth=2.0, linestyle = '-')
    #lines = plt.plot(lambda_vector,average_array-std_array,lambda_vector,average_array+std_array)
    #plt.setp(lines, color=cm.jet(counter/(len(methods)-1)), linewidth=2.0, linestyle = '--')
    legend1.append(plt.Circle((0, 0), 1, fc=cm.jet(counter/(len(methods)-1))))
    counter+=1
ax.legend(legend1, labels)
plt.ylim([0,100])
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, 5))
plt.xlabel("lambda")
plt.ylabel("standard deviation total score exam")
plt.grid(1)
plt.show()    
plt.savefig('stdTotalScoreDifferentMethods_lambdas'+'_pmis'+str(prob_misConception)+'.png')