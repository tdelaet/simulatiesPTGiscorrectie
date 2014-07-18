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

nameMethod = 'valueFunction'

numberAlternatives = 4


alpha = 0.88
beta = 0.88
lambda_v = 2.25

if not os.path.exists(nameMethod):
    os.makedirs(nameMethod)

def valueFunction(x):
    #0<=alpha<=beta<=1
    value = [ np.power(xi,alpha) if xi>=0 else -lambda_v * np.power(-xi,beta) for xi in x]
    return value


    
# evenly sampled time at 200ms intervals
score = np.arange(-4, 4, 0.02)

# red dashes, blue squares and green triangles

f, (ax1,ax2) = plt.subplots(1,2)
# influence of alpha and beta
step = 0.1
alpha_vector = np.arange(step,1,step)

counter = 0
legend1=[]
legend2=[]
for alpha in alpha_vector:
    beta = alpha
    line = ax1.plot(score, valueFunction(score), 'r--')
    plt.setp(line, color=cm.jet(counter/(len(alpha_vector)-1)), linewidth=2, linestyle = '-')
    legend1.append(plt.Circle((0, 0), 1, fc=cm.jet(counter/(len(alpha_vector)-1))))
    legend2.append(r'$\alpha$ = $\beta$ =' + str(alpha))
    counter+=1
ax1.legend(legend1, legend2, loc=2)    
ax1.set_xlim(score[0],score[len(score)-1])
ax1.set_ylim(-7,5)
ax1.set_xlabel("score")
ax1.set_ylabel("v(score)")
ax1.set_title(r'influence of $\alpha$ and $\beta$ on valuefunction ($\lambda$ = ' + str(lambda_v) + ')')
ax1.grid(1)


# influence of lambda
step = 0.5  
lambda_vector = np.arange(1,5,step)
alpha = 0.88
beta = 0.88

counter = 0
legend1=[]
legend2=[]
for lambda_v in lambda_vector:
    line = ax2.plot(score, valueFunction(score), 'r--')
    plt.setp(line, color=cm.jet(counter/(len(lambda_vector)-1)), linewidth=2, linestyle = '-')
    legend1.append(plt.Circle((0, 0), 1, fc=cm.jet(counter/(len(lambda_vector)-1))))
    legend2.append(r"$\lambda =$" + str(lambda_v))
    counter+=1
ax2.legend(legend1, legend2, loc=2)    
ax2.set_xlim(score[0],score[len(score)-1])
ax2.set_ylim(-7,5)
ax2.set_xlabel("score")
ax2.set_ylabel("v(score)")
ax2.set_title(r'influence of $\lambda$ on valuefunction ($\alpha = \beta$ = ' + str(alpha) + ')')
ax2.grid(1)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()   
plt.savefig(nameMethod + '/valueFunction.png')