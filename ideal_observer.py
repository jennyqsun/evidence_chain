# Created on 5/3/22 at 9:48 AM 

# Author: Jenny Sun
# Created on 4/5/22 at 2:01 PM

# Author: Jenny Sun
# Created on 4/5/22 at 12:44 PM

# Author: Jenny Sun
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 8
import numpy as np
import pandas as pd
import os
from os import listdir
import matplotlib.pyplot as plt
import pickle
from notebook.services.config import ConfigManager
import matplotlib.lines as mlines
import seaborn as sns
# configurate fonts
plt.rcParams.update({'font.size': 17})


save = False
# set up file
onlyfile = [f for f in listdir('data') if (('888' in f) or ('888' in f)) and ('csv' in f)]
onlyfile.sort()

# concat all the csv files
df = []
for f in onlyfile:
    df.append(pd.read_csv('data/' + f))

condPerBlock=[]
for d in df:
    condPerBlock.append(d['stimDur'].unique())

df = pd.concat(df)

def loadPKL(filename):
    myfile = open(filename, 'rb')
    f = pickle.load(myfile)
    return f

# get rid of trials without any respones

df['time'].replace('[]',np.nan, inplace =True)
print('% of null trials:', sum(df['time'].isnull()) / len(df) )
df_1 = df[df['time'].notnull()]

# unique conditions
stimDur= df_1['stimDur'].unique()
stimDur.sort()


# get important arrays
def getAllArrays(df_1):
    allData = {}
    rt = np.array(df_1['time'].astype('float'))
    count = np.array(df_1['count'])

    # clean the data
    key = []
    sequence = []
    stimDur = []

    for index, row in df_1.iterrows():
        if row['key'] == "[5]":
            k = 1
        else:
            k = 0
        key.append(k)
        seq = row['sequence'].split(".")
        l = []
        for i in seq:
            i = i.replace("[", '')
            i = i.replace("]", '')
            if '1' in i:
                l.append(int(i))
        sequence.append(l)
        stimDur.append(row['stimDur'])
    key = np.array(key)
    sequence = np.array(sequence)
    stimDur = np.array(stimDur)
    allData = {'rt':rt, 'count': count, 'key':key, 'sequence': sequence, 'stimDur': stimDur}
    return allData


allData = getAllArrays(df_1)
def getDict(allData, dur=0.2):
    condData = {}
    # index the conditions
    trialind = allData['stimDur']==dur
    rt_ = allData['rt'][trialind]
    key_= allData['key'][trialind]
    count_ = allData['count'][trialind]
    sequence_ = allData['sequence'][trialind]

    cumsum = np.cumsum(sequence_,axis=1)   # random walk of the sequence


    sequence_stop =[]   # sequence stopping at the
    cumsum_stop = []  # random walk when sequence stops
    maxind = []     # index of max evidence
    direction = []  # direction of the max ind, O is 1 and X is 0
    maxvalue = []   # value at the max evidenc
    boundary = []   # value upon stop
    for r in range(0,len(sequence_)):
        s = sequence_[r][0:count_[r]+1]
        sequence_stop.append(s)
        c = np.cumsum(s)
        maxind.append(np.argmax(np.abs(c)))
        cumsum_stop.append(c)
        boundary.append(c[-1:])
        maxvalue.append(c[maxind[-1]])
        if c[maxind[-1]] < 0:
            direction.append(0)
        else:
            direction.append(1)
    direction=np.array(direction)
    maxvalue = np.array(maxvalue)
    boundary = np.squeeze(np.array(boundary))
    maxind = np.array(maxind)
    condData = {'trialind':trialind, 'rt_':rt_, 'key_':key_, 'count_':count_,
               'sequence_':sequence_, 'cumsum':cumsum, 'sequence_stop':sequence_stop,
               'cumsum_stop':cumsum_stop, 'maxind':maxind, 'direction':direction,'maxvalue':maxvalue,
               'boundary':boundary}
    return condData,dur


def plot_sorted(d, dur):
    # sorting ind
    rt_ind = np.argsort(d['rt_'])  # sort by rt from fastest to slowest
    boundary_ind = np.argsort(d['boundary'])  # sort from lowest boundary to highest
    maxvalue_ind = np.argsort(np.abs(d['maxvalue']))
    fig, ax = plt.subplots(1, figsize=(16, 20))

    for p in np.unique(np.abs(d['maxvalue'])):
        ind_p = np.abs(d['maxvalue']) == p
        # acc = d['direction'] == d['key_']
        # acc_p = acc[ind_p]
        maxind_p = d['maxind'][ind_p]
        count_p = d['count_'][ind_p]
        maxind_sort = np.argsort(maxind_p)
        maxind_p = maxind_p[maxind_sort]
        count_p = count_p[maxind_sort]
        #     print(sum(count_p>maxind_p) == len(count_p))
        # acc_p = acc_p[np.argsort(maxind_p)]
        yvec = np.linspace(p - 0.2, p + 0.7, len(maxind_p))
        a = ax.plot(maxind_p, yvec, '^')
        col = a[0].get_color()
        ax.plot(count_p + 0.5, yvec, 'o', color=col)
        # ax.grid(visible=True, axis='x', linestyle='-')

        ax.set_xticks(np.arange(0, 31))

        for i in range(0, len(count_p)):
            x1 = maxind_p[i]
            x2 = count_p[i] + 0.5
            ax.plot((x1, x2), (yvec[i], yvec[i]), ls='--', color=col)
        ax.set_yticks(np.unique(np.abs(d['maxvalue'])))
        label = [str(i) for i in np.arange(1, 31).tolist()] + ['  end \n of trial']
        ax.set_xticks(np.arange(0, 31))
        ax.set_xticklabels(label)
        ax.set_ylabel('Magnitude of Peak Evidence Upon Reponse')
        ax.set_xlabel('Number of Stimuli')

        trig = mlines.Line2D([], [], ls='none', color='black', marker='^',
                             markersize=15, label='The Display Where the Chain Peaked Upon Response')
        circ = mlines.Line2D([], [], ls='none', color='black', marker='o',
                             markersize=15, label='The Display Where Subject Responded')
        ax.legend(handles=[trig, circ])

        plt.title('Stimulus Duration Per Display: %s ms' % dur)
    fig.show()
d,dur = getDict(allData,dur=0.2)

bound = 5





fig, ax = plt.subplots(1,3,figsize= (15,5))


seq = d['sequence_']
cumsum = d['cumsum']

indlist =[]
val = []
maxind = []
maxvalue = []
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
    maxind_t = np.argmax(np.abs((j[0:ind+1])))
    maxind.append(maxind_t)
    maxvalue.append(((j[0:ind+1]))[maxind_t])
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
d['rt_'] = np.array(rt)
d['boundary'] = np.array(val)
d['maxind'] = np.array(maxind)
d['maxvalue']=np.array(maxvalue)
d['count_'] = np.array(indlist)

plot_sorted(d,dur)



bound= 11
######################### collappsing bound ########################
fig, ax = plt.subplots(1,3,figsize= (15,5))

k=1
t = np.linspace(0,6,30)
# alpha = 4 * (1-k*(t/(t+0.1)))

alpha  = bound-(1-np.exp(-(t/bound)**k))*(0.5*bound+12)

maxind = []
maxvalue = []

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
    maxind_t = np.argmax(np.abs((j[0:ind+1])))
    maxind.append(maxind_t)
    maxvalue.append(((j[0:ind+1]))[maxind_t])
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

d['rt_'] = np.array(rt)
d['boundary'] = np.array(val)
d['maxind'] = np.array(maxind)
d['maxvalue']=np.array(maxvalue)
d['count_'] = np.array(indlist)

plot_sorted(d,dur)




### DDM with urgency signal  #######
bound=10
fig, ax = plt.subplots(1,4,figsize= (20,5))


t = np.linspace(0,6,30)*0.5
# alpha = 4 * (1-k*(t/(t+0.1)))

indlist =[]
val = []

maxind = []
maxvalue = []


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
    maxind_t = np.argmax(np.abs((j[0:ind+1])))
    maxind.append(maxind_t)
    maxvalue.append(((j[0:ind+1]))[maxind_t])

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


d['rt_'] = np.array(rt)
d['boundary'] = np.array(val)
d['maxind'] = np.array(maxind)
d['maxvalue']=np.array(maxvalue)
d['count_'] = np.array(indlist)

plot_sorted(d,dur)


#############33 PUre UGM
fig, ax = plt.subplots(1,4,figsize= (20,5))


t = np.linspace(0,6,30) * 2
# alpha = 4 * (1-k*(t/(t+0.1)))


indlist =[]
val = []
maxind = []
maxvalue = []


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
    maxind_t = np.argmax(np.abs(np.cumsum((j[0:ind+1]))))
    maxind.append(maxind_t)
    maxvalue.append((np.cumsum(j[0:ind+1]))[maxind_t])

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

d['rt_'] = np.array(rt)
d['boundary'] = np.array(val)
d['maxind'] = np.array(maxind)
d['maxvalue']=np.array(maxvalue)
d['count_'] = np.array(indlist)

plot_sorted(d,dur)



######## Race model ###########
fig, ax = plt.subplots(1,3,figsize= (15,5))

indlist =[]
val = []
maxind = []
maxvalue = []
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
    maxind_t = np.argmax(np.abs(np.cumsum((j[0:ind+1]))))
    maxind.append(maxind_t)
    maxvalue.append((np.cumsum(j[0:ind+1]))[maxind_t])


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

d['rt_'] = np.array(rt)
d['boundary'] = np.array(val)
d['maxind'] = np.array(maxind)
d['maxvalue']=np.array(maxvalue)
d['count_'] = np.array(indlist)

plot_sorted(d,dur)





# fig, ax = plt.subplots(1,3,figsize= (15,5))
#
#
# indlist =[]
# val = []
#
# h = []
#
# for i,j in enumerate(seq):
#     indX = np.where(j==-1)[0]
#     indO = np.where(j==1)[0]
#     if len(indX) <= (bound-1):
#         ind = indO[bound-1]
#     elif len(indO) <= (bound-1):
#         ind = indX[bound-1]
#     else:
#         ind = np.min((indX[bound-1],indO[bound-1]))
#     #
#     # new_j = new * t
#     # hit = np.abs(new_j)>=bound
#     # h.append(any(hit))
#     # if any(hit):
#     #     ind = np.where(hit)[0]
#     #     if len(ind)>1:
#     #         ind = ind[0]
#     #     else:
#     #         ind = ind[0]
#     # else:
#     #     ind = 29
#     indlist.append(ind)
#     val.append(np.abs((np.cumsum(j[0:ind + 1]))[-1]))
#     if i < 20:
#         ax[0].plot(np.cumsum(j[0:ind + 1]))
#
#
#
# ax[0].set_xlabel('Number of Steps')
#
# rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
# sns.histplot(val, ax=ax[1],color='blue',discrete=True)
# sns.histplot(rt, ax=ax[2],color='green')
# ax[0].set_title('Simulated Random Walk')
# ax[0].xaxis.get_major_locator().set_params(integer=True)
#
#
# ax[1].set_title('Simulated Histogram of Bound')
# ax[1].set_xlabel('Boundary')
# ax[2].set_xlabel('RT (s)')
# ax[2].set_title('Simulated Histogram of RT')
# fig.suptitle('Predicted Patterns according to Race Model Counting to 6')
# fig.tight_layout()
# if save:
#     fig.savefig('hypothesis/sim_R.png')
# fig.show()
#
#



############################# Consecutive Run Model  ########################################33
fig, ax = plt.subplots(1,3,figsize= (15,5))
indlist =[]
val = []
maxind = []
maxvalue = []
h = []


indlist =[]
val = []
run =5
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

    maxind_t = np.argmax(np.abs(np.cumsum((j[0:ind+1]))))
    maxind.append(maxind_t)
    maxvalue.append((np.cumsum(j[0:ind+1]))[maxind_t])

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
fig.suptitle('Predicted Patterns according to Consecutive Run Model (run: %s)'%run)
fig.tight_layout()
if save:
    fig.savefig('hypothesis/sim_CRM.png')
fig.show()


d['rt_'] = np.array(rt)
d['boundary'] = np.array(val)
d['maxind'] = np.array(maxind)
d['maxvalue']=np.array(maxvalue)
d['count_'] = np.array(indlist)

plot_sorted(d,dur)
