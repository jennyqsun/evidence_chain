import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import binomial
import random
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 17})

seed=86693
random.seed(seed)
np.random.seed(seed)
save= False

def simulateTrials(numTrials: int, numSteps: int, numStim: int, Bias: float, initRange:range):
    X0 = np.zeros((numTrials, numSteps))
    X1 = np.zeros((numTrials, numSteps))
    initSteps = []
    prob0 = 0.5-Bias
    prob1 = 0.5+Bias
    print(prob0,prob1)
    for i in range(numTrials):
        x0 = binomial(numStim, 0.5 - Bias, numSteps)
        x1 = binomial(numStim, 0.5 + Bias, numSteps)
        initDisplayRange = np.random.choice(initRange)
        initlist = [0]* int(initDisplayRange/2) +  [1]* int(initDisplayRange/2)
        random.shuffle(initlist)
        initSteps.append(initlist)
        x0 = x0.astype('float')
        x1 = x1.astype('float')
        x0[0:initDisplayRange] = initlist
        x1[0:initDisplayRange] = initlist
        X0[i,:] = x0
        X1[i,:] = x1
    X0[X0 == 0] = -1
    X1[X1 == 0] = -1
    # X0[X0 == 0.5] = 0
    # X1[X1 == 0.5] = 0
    return X0, X1, initSteps

# DDM
import seaborn as sns
X0,X1,initSteps = simulateTrials(1000, 30, 1,0.12, [0,0,0])

cumsum_O = np.cumsum(X0,axis=1)
cumsum_X = np.cumsum(X1,axis=1)


fig, ax = plt.subplots(1,3,figsize= (15,5))

bound = 3

cumsum = np.vstack((cumsum_O,cumsum_X))
indlist =[]
val = []
for i,j in enumerate(cumsum):
    ind = np.where(np.abs(j)>=bound)[0]
    if len(ind) > 0:
        ind = ind[0]
    elif len(ind) == 1:
        pass
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((j[0:ind+1])[-1]))
    if i < 20:
        ax[0].plot(j[0:ind+1])
ax[0].set_xlabel('Number of Steps')
ax[0].axhline(bound,color='red')
ax[0].axhline(-bound,color='red',label ='boundary')
rt = [(i-1)*0.2+np.random.normal(0.3,0.05) for i in indlist]
sns.histplot(val, ax=ax[1],color='blue', discrete=True)
sns.histplot(rt, ax=ax[2],color='green')
ax[0].legend()
ax[0].set_title('Simulated Random Walk Trials')
ax[1].set_title('Simulated Histogram of Bound')
ax[1].set_xlabel('Boundary')
ax[2].set_xlabel('RT (s)')
ax[2].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to DDM')
fig.tight_layout()
if save:
    fig.savefig('hypothesis/sim_DDM.png')
fig.show()




fig, ax = plt.subplots(1,3,figsize= (15,5))

k=2
t = np.linspace(0,6,30)
# alpha = 4 * (1-k*(t/(t+0.1)))

alpha  = bound-(1-np.exp(-(t/bound)**k))*(0.5*bound+1)

cumsum = np.vstack((cumsum_O,cumsum_X))
indlist =[]
val = []

h = []
for i,j in enumerate(cumsum):
    hit = np.abs(j)>=alpha
    h.append(any(hit))
    if any(hit):
        ind = np.where(hit)[0]
        if len(ind)>1:
            ind = ind[0]
        else:
            ind = ind[0]
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((j[0:ind + 1])[-1]))
    if i < 20:
        ax[0].plot(j[0:ind + 1])
ax[0].plot(alpha,color='red')
ax[0].plot(-alpha,color='red',label='boundary')
ax[0].set_xlabel('Number of Steps')

rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
sns.histplot(val, ax=ax[1],color='blue',discrete=True)
sns.histplot(rt, ax=ax[2],color='green')
ax[0].legend()
ax[0].set_title('Simulated Random Walk Trials')
ax[1].set_title('Simulated Histogram of Bound')
ax[1].set_xlabel('Boundary')
ax[2].set_xlabel('RT (s)')
ax[2].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to cDDM')
fig.tight_layout()
if save:
    fig.savefig('hypothesis/sim_cDDM.png')
fig.show()







fig, ax = plt.subplots(1,4,figsize= (20,5))


t = np.linspace(0,6,30) * 2
# alpha = 4 * (1-k*(t/(t+0.1)))

seq = np.vstack((X0,X1))
cumsum = np.vstack((cumsum_O,cumsum_X))
indlist =[]
val = []

h = []
for i,j in enumerate(cumsum):
    new_j = j * t
    hit = np.abs(new_j)>=bound
    h.append(any(hit))
    if any(hit):
        ind = np.where(hit)[0]
        if len(ind)>1:
            ind = ind[0]
        else:
            ind = ind[0]
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((j[0:ind + 1])[-1]))
    if i < 20:
        ax[0].plot(j[0:ind + 1])
        ax[1].plot(new_j[0:ind + 1])

ax[0].xaxis.get_major_locator().set_params(integer=True)
ax[0].set_xlabel('Number of Steps')
ax[1].axhline(bound, color='red',label='boundary')
ax[1].axhline(-bound, color='red')
ax[1].xaxis.get_major_locator().set_params(integer=True)

rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
sns.histplot(val, ax=ax[2],color='blue',discrete=True)
sns.histplot(rt, ax=ax[3],color='green')
ax[1].legend()
ax[0].set_title('Simulated Random Walk Trials')
ax[1].set_title('Simulated \n Multiplicative Random Walk Trials')
ax[2].set_title('Simulated Histogram of Bound')
ax[2].set_xlabel('Boundary')
ax[3].set_xlabel('RT (s)')
ax[3].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to DDM with Urgency Signal (Slope: 0.4)')
fig.tight_layout()
if save:
    fig.savefig('hypothesis/sim_DDM_urgency.png')
fig.show()






fig, ax = plt.subplots(1,4,figsize= (20,5))


t = np.linspace(0,6,30) * 2
# alpha = 4 * (1-k*(t/(t+0.1)))

seq = np.vstack((X0,X1))
cumsum = np.vstack((cumsum_O,cumsum_X))
indlist =[]
val = []

h = []
for i,j in enumerate(seq):
    new = []
    for k, l in enumerate(j):
        if k > 0:
            new.append(0.5*(l + j[k - 1]))
        else:
            new.append(l)
    new_j = new * t
    hit = np.abs(new_j)>=bound
    h.append(any(hit))
    if any(hit):
        ind = np.where(hit)[0]
        if len(ind)>1:
            ind = ind[0]
        else:
            ind = ind[0]
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((new_j[0:ind + 1])[-1]))
    if i < 20:
        ax[0].plot(new[0:ind + 1])
        ax[1].plot(new_j[0:ind + 1])


ax[0].set_xlabel('Number of Steps')
ax[1].axhline(bound, color='red',label='boundary')
ax[1].axhline(-bound, color='red')

rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
sns.histplot(val, ax=ax[2],color='blue')
sns.histplot(rt, ax=ax[3],color='green')
ax[1].legend()
ax[0].set_title('Simulated Instantaneous Evidence \n(Low Pass Filtered)')
ax[1].set_title('Simulated Multiplicative \n Instantaneous Evidence')
ax[0].xaxis.get_major_locator().set_params(integer=True)
ax[1].xaxis.get_major_locator().set_params(integer=True)

ax[2].set_title('Simulated Histogram of Bound')
ax[2].set_xlabel('Boundary')
ax[3].set_xlabel('RT (s)')
ax[3].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to UGM with Low Pass Filter (Slope: 0.4)')
fig.tight_layout()
if save:
    fig.savefig('hypothesis/sim_UGM.png')
fig.show()




fig, ax = plt.subplots(1,3,figsize= (15,5))

seq = np.vstack((X0,X1))
cumsum = np.vstack((cumsum_O,cumsum_X))
indlist =[]
val = []

h = []

for i,j in enumerate(seq):
    indX = np.where(j==-1)[0]
    indO = np.where(j==1)[0]
    if len(indX) <= (bound-1):
        ind = indO[bound-1]
    elif len(indO) <= (bound-1):
        ind = indX[bound-1]
    else:
        ind = np.min((indX[bound-1],indO[bound-1]))
    #
    # new_j = new * t
    # hit = np.abs(new_j)>=bound
    # h.append(any(hit))
    # if any(hit):
    #     ind = np.where(hit)[0]
    #     if len(ind)>1:
    #         ind = ind[0]
    #     else:
    #         ind = ind[0]
    # else:
    #     ind = 29
    indlist.append(ind)
    val.append(np.abs((np.cumsum(j[0:ind + 1]))[-1]))
    if i < 20:
        ax[0].plot(np.cumsum(j[0:ind + 1]))



ax[0].set_xlabel('Number of Steps')

rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
sns.histplot(val, ax=ax[1],color='blue',discrete=True)
sns.histplot(rt, ax=ax[2],color='green')
ax[0].set_title('Simulated Random Walk')
ax[0].xaxis.get_major_locator().set_params(integer=True)


ax[1].set_title('Simulated Histogram of Bound')
ax[1].set_xlabel('Boundary')
ax[2].set_xlabel('RT (s)')
ax[2].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to Race Model Counting to 6')
fig.tight_layout()
if save:
    fig.savefig('hypothesis/sim_RM.png')
fig.show()






fig, ax = plt.subplots(1,3,figsize= (15,5))

seq = np.vstack((X0,X1))
cumsum = np.vstack((cumsum_O,cumsum_X))
indlist =[]
val = []

h = []

for i,j in enumerate(seq):
    indX = np.where(j==-1)[0]
    indO = np.where(j==1)[0]
    if len(indX) <= (bound-1):
        ind = indO[bound-1]
    elif len(indO) <= (bound-1):
        ind = indX[bound-1]
    else:
        ind = np.min((indX[bound-1],indO[bound-1]))
    #
    # new_j = new * t
    # hit = np.abs(new_j)>=bound
    # h.append(any(hit))
    # if any(hit):
    #     ind = np.where(hit)[0]
    #     if len(ind)>1:
    #         ind = ind[0]
    #     else:
    #         ind = ind[0]
    # else:
    #     ind = 29
    indlist.append(ind)
    val.append(np.abs((np.cumsum(j[0:ind + 1]))[-1]))
    if i < 20:
        ax[0].plot(np.cumsum(j[0:ind + 1]))



ax[0].set_xlabel('Number of Steps')

rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
sns.histplot(val, ax=ax[1],color='blue',discrete=True)
sns.histplot(rt, ax=ax[2],color='green')
ax[0].set_title('Simulated Random Walk')
ax[0].xaxis.get_major_locator().set_params(integer=True)


ax[1].set_title('Simulated Histogram of Bound')
ax[1].set_xlabel('Boundary')
ax[2].set_xlabel('RT (s)')
ax[2].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to Race Model Counting to 6')
fig.tight_layout()
if save:
    fig.savefig('hypothesis/sim_R.png')
fig.show()






fig, ax = plt.subplots(1,3,figsize= (15,5))

seq = np.vstack((X0,X1))
cumsum = np.vstack((cumsum_O,cumsum_X))
indlist =[]
val = []



for i,j in enumerate(seq):
    for ii,jj in enumerate(j):
        if ii >= 4:
            if j[ii] == j[ii-1]==j[ii-2]==j[ii-3]:
                indlist.append(ii)
                break
    ind = ii
    # if len(indX) <=1 :
    #     ind = indO[bound-1]
    # elif len(indO) <= (bound-1):
    #     ind = indX[bound-1]
    # else:
    #
    # new_j = new * t
    # hit = np.abs(new_j)>=bound
    # h.append(any(hit))
    # if any(hit):
    #     ind = np.where(hit)[0]
    #     if len(ind)>1:
    #         ind = ind[0]
    #     else:
    #         ind = ind[0]
    # else:
    #     ind = 29
    indlist.append(ind)
    val.append(np.abs((np.cumsum(j[0:ind + 1]))[-1]))
    if i < 20:
        ax[0].plot(np.cumsum(j[0:ind + 1]))



ax[0].set_xlabel('Number of Steps')

rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
sns.histplot(val, ax=ax[1],color='blue',discrete=True)
sns.histplot(rt, ax=ax[2],color='green')
ax[0].set_title('Simulated Random Walk')
ax[0].xaxis.get_major_locator().set_params(integer=True)


ax[1].set_title('Simulated Histogram of Bound')
ax[1].set_xlabel('Boundary')
ax[2].set_xlabel('RT (s)')
ax[2].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to Consecutive Run Model (run: 4)')
fig.tight_layout()
if save:
    fig.savefig('hypothesis/sim_CRM.png')
fig.show()





