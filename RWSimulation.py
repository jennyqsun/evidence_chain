# Created on 3/11/22 at 1:15 PM 

# Author: Jenny Sun

from numpy.random import binomial, uniform
import numpy as np
import matplotlib.pyplot as plt

# numtrials = 20
# for _ in range(numtrials):
#     x = binomial(6, 0.44, 20)
#     x0 = binomial(6, 0.56, 20)
#
#
#     initDisplay = np.random.choice(range(1,5))
#     x[0:initDisplay] = 3
#     x0[0:initDisplay] = 3
#
#     plt.plot(np.cumsum(x-3),color = 'blue')
#     plt.plot(np.cumsum(x0-3),color ='red')
# plt.show()


def simulateTrials(numTrials: int, numSteps: int, numStim: int, Bias: float, initRange:range):
    X0 = np.zeros((numTrials, numSteps))
    X1 = np.zeros((numTrials, numSteps))
    prob0 = 0.5-Bias
    prob1 = 0.5+Bias
    print(prob0,prob1)
    for i in range(numTrials):
        x0 = binomial(numStim, 0.5 - Bias, numSteps)
        x1 = binomial(numStim, 0.5 + Bias, numSteps)
        initDisplayRange = np.random.choice(initRange)
        x0 = x0.astype('float')
        x1 = x1.astype('float')
        x0[0:initDisplayRange] = numStim/2
        x1[0:initDisplayRange] = numStim/2
        X0[i,:] = x0
        X1[i,:] = x1

    return X0, X1

# def plotcumsum(x, color:str, label: str):
#     CumSum = np.cumsum(x,axis=1)
#     plt.plot(CumSum.T, color=  color, label =label)
#     return CumSum
#

def genUnbiasedSets(nset):
    UnbiasedSet = np.ones(nset)
    UnbiasedSet[0:int(nset/2)] = -1
    np.random.shuffle(UnbiasedSet)
    return UnbiasedSet

# start simulation
X0,X1 = simulateTrials(6,30,1,0.06,[0,4,8])
X0[X0==0]= -1
X1[X1==0] = -1
X0[X0==0.5]= 0
X1[X1==0.5]= 0
# for i in range(X0.shape[0]):


# X0cumsum = plotcumsum(X0, 'red','prob < 0.5')
# X1cumsum = plotcumsum(X1, 'blue','prob < 0.5')
fig, ax = plt.subplots()
Lines0 = ax.plot(np.cumsum(X0, axis=1).T,color='blue')
Lines1 = ax.plot(np.cumsum(X1, axis=1).T, ls = '--',color = 'orange')
ax.legend([Lines0[0], Lines1[0]],['prob < 0.5', 'prob > 0.5'])
fig.show()

print('opposite bound', sum(np.cumsum(X0, axis=1)[:,-1] >= 0) / X0.shape[0])
print('opposite bound', sum(np.cumsum(X1, axis=1)[:,-1] <= 0) / X1.shape[0])