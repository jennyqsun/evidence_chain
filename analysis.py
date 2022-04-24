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

plt.rcParams.update({'font.size': 17})
onlyfile = [f for f in listdir('data') if '888' in f and 'csv' in f]
onlyfile.sort()
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



# for i, j in df.iterrows():
#     if df.iloc[i,:]['bytetime'] == '[]':
#         print('true')
#         df.iloc[i, :]['bytetime']. 'NaN'
df['time'].replace('[]',np.nan, inplace =True)
sum(df['time'].isnull()) / len(df)


df_1 = df[df['time'].notnull()]
stimDur= df_1['stimDur'].unique()
stimDur.sort()


rt = np.array(df_1['time'].astype('float'))
count = np.array(df_1['count'])

# get key, and cumsum
key = []
cumsum = []


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

### make plots
dur = 0.2
trialind = stimDur==dur
rt_ = rt[trialind]
key_= key[trialind]
count_ = count[trialind]
sequence_ = sequence[trialind]
cumsum = np.cumsum(sequence_,axis=1)
sequence_stop =[]
maxind = []
direction = []
maxvalue = []
cumsum_stop = []
boundary = []
for r in range(0,len(sequence_)):
    s = sequence_[r][0:count_[r]+1-2]
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

maxvalue = np.abs(np.array(maxvalue))
rt_ind = np.argsort(np.abs(np.squeeze(np.array(boundary))))
# rt_ind = np.argsort(np.abs(maxvalue))
maxind = np.array(maxind)
plt.figure(figsize=(10,18))
acc = direction==key_
acc = acc[rt_ind]
numT = len(acc)
plt.plot(maxind[rt_ind],np.arange(0,numT,1),'^')
plt.plot(maxind[rt_ind][acc], np.arange(0,numT)[acc],'^',color='green',label='consistent')
plt.plot(maxind[rt_ind][~acc], np.arange(0,numT)[~acc],'^',color='red',label='inconsistent')

plt.plot(count_[rt_ind][acc],np.arange(0,numT,1)[acc],'o',color='green')
plt.plot(count_[rt_ind][~acc],np.arange(0,numT,1)[~acc],'o',color='red')
for i in range(0,len(acc)):
    x1 = maxind[rt_ind][i]
    x2 = count_[rt_ind][i]
    if acc[i]:
        plt.plot((x1,x2),(i,i),color='green',ls='--')
    else:
        plt.plot((x1,x2),(i,i),color='red',ls='--')
plt.legend()
plt.xlabel('Number of Displays of Stimuli')
plt.ylabel('Trials (sorted by evidence upon response)')
plt.title('Stimulus Duration Per Display: %s ms'%dur)
# plt.savefig('evidencevalue_rt_%s'%dur + '.png')

boundary = np.squeeze(np.array(boundary))
fig, ax = plt.subplots(1,2,figsize=  (10,5))
ax[0].scatter(maxvalue[key_==1],boundary[key_==1])
ax[0].set_xlabel('maxvalue of chain before response')
ax[0].set_ylabel('Integrated Evidence towards the Choice')
ax[1].scatter(maxvalue[key_==0],-boundary[key_==0])
plt.show()
b = np.hstack((maxvalue[key_==1],-boundary[key_==0]))


# ypothetical grapoh, x axis is maxvalue
# y axis is hypoehtical situation when there is a constant bound
# distace between response and max value as a function of amount of evidence avaialbe  axis max evidence, y axis time lapsed from max evidenc 




# ax[1].hist(b,bins=15)
# ax[1].set_xlabel('Integrated Evidence')
fig.suptitle('Correlation between Evidence and RT')


fig.show()
# # synthesize RT using chain
#
# syn = []
# for i in sequence_stop:
#     pred = np.cumsum(i)[-1]
#     syn.append(np.abs(pred))
#
# plt.scatter(rt_, syn)



#
# maxcount = np.argmax(np.abs(cumsum), axis=1)
#
# fig, ax = plt.subplots(4,2, figsize=(10,15))
# count=0
#
# data = {}
# for s in stimDur:
#     df_0 = df_1[df_1['stimDur']==s]
#     data[format(s)] = dict()
#     # print(s)
#     O_list = []
#     X_list = []
#     O_chain=[]
#     X_chain=[]
#     choice = []
#     for index, row in df_0.iterrows():
#         trial_count = row['count']
#         if trial_count <1:
#             continue
#
#         key = 'O' if row['key'] == "[5]" else 'X'
#         linecolor = 'red' if key == 'O' else 'black'
#         seq = row['sequence'].split(".")
#         l = []
#         for i in seq:
#             i = i.replace("[",'')
#             i = i.replace("]",'')
#             if '1' in i:
#                 l.append(int(i))
#         ndt = round(0.3/s)
#
#
#         if key == 'O':
#             # print('0')
#             choice.append(1)
#             cumsum = np.cumsum(l[0:int(trial_count+1)])[-1]
#             O_list.append(cumsum)
#             O_chain.append(np.cumsum(l[0:int(trial_count+1)]))
#
#             # ax[0][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
#         else:
#             # print('X')
#             choice.append(-1)
#             cumsum = np.cumsum(l[0:int(trial_count+1)])[-1]
#             X_list.append(cumsum)
#             X_chain.append(np.cumsum(l[0:int(trial_count+1)]))
#
#             # ax[1][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
#     # ax[count][2].plot(O_chain)
#     ax[count][0].hist(O_list,color = 'red',label = 'Chose Os (N = %s)'%len(O_list), bins = (np.arange(-10,10,1)), align='mid')
#     ax[count][0].set_xlim([-10,10])
#     ax[count][0].set_ylim([0, 50])
#     ax[count][0].legend()
#     ax[count][1].hist(X_list, color='black', label = 'Chose Xs (N = %s)'%len(X_list), bins = (np.arange(-10,10,1)), align = 'mid')
#     ax[count][1].set_xlim([-10,10])
#     ax[count][1].set_ylim([0, 50])
#     ax[count][1].legend()
#
#     ax[count][0].set_title('Stimulus Duration: %s ms. ' % int(s*1000))
#     # ax[count][1].invert_xaxis()
#     count +=1
#
# fig.suptitle('Histograms of Integrated Values by Choice (NDT adjusted)\n binomial parameters: (0.62, 0.38)')
# fig.tight_layout()
# fig.show()
#
#
#
#
#
# fig, ax = plt.subplots(4,3, figsize=(15,15))
# count=0
# for s in stimDur:
#     df_0 = df_1[df_1['stimDur']==s]
#     print(s)
#
#     O_list = []
#     X_list = []
#     O_rt = []
#     X_rt =[]
#     for index, row in df_0.iterrows():
#         trial_count = row['count']
#         if trial_count <1:
#             continue
#
#         key = 'O' if row['key'] == "[5]" else 'X'
#         linecolor = 'red' if key == 'O' else 'black'
#         seq = row['sequence'].split(".")
#         l = []
#         for i in seq:
#             i = i.replace("[",'')
#             i = i.replace("]",'')
#             if '1' in i:
#                 l.append(int(i))
#         ndt = np.round(0.3/s)
#
#
#         if key == 'O':
#             # print('0')
#             cumsum = np.cumsum(l[int(trial_count+1-ndt)-1:int(trial_count+1-ndt)])[-1]
#             O_list.append(cumsum)
#             O_rt.append(float(row['time'])*1000)
#
#             # ax[0][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
#         else:
#             # print('X')
#
#             cumsum = np.cumsum(l[int(trial_count+1-ndt)-1:int(trial_count+1-ndt)])[-1]
#             X_list.append(cumsum)
#             X_rt.append(float(row['time'])*1000)
#
#
#             # ax[1][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
#
#
#     O_acc = sum( [1 for i in O_list if i>0]) / len(O_list)
#     X_acc = sum( [1 for i in X_list if i<0]) / len(X_list)
#     ind0 = [i for i, j in enumerate(O_list) if j>0]
#     O_rt = np.array(O_rt)
#     X_rt = np.array(X_rt)
#     X_list = np.array(X_list)
#     O_list = np.array(O_list)
#     rt_max =max(np.max(O_rt), np.max(X_rt))
#     rt_min = min(np.min(O_rt), np.min(X_rt))
#     print(O_acc, X_acc)
#     ax[count][2].bar((0,0.5),(O_acc,X_acc), width = 0.3)
#     ax[count][2].set_xticks((0,0.5))
#     ax[count][2].set_xticklabels(('O',  'X'))
#     ax[count][2].set_xlabel('Correct Choice')
#     ax[count][2].set_ylabel('Accuracy')
#     ax[count][2].set_ylim(0.5,1)
#     ax[count][0].scatter(O_rt[ind0],O_list[ind0],color = 'red',label = 'Chose Os (N = %s)'%len(O_list))
#     # ax[count][0].set_xlim(rt_min-100,rt_max+100)
#     # ax[count][0].set_ylim([-15, 15])
#     ax[count][0].set_xlabel('RT (ms)')
#     ax[count][0].set_ylabel('Integrated Value')
#
#     indX = [i for i, j in enumerate(X_list) if j<0]
#     ax[count][0].legend(loc = 'upper right')
#     ax[count][1].scatter(X_rt[indX], X_list[indX], color='black', label = 'Chose Xs (N = %s)'%len(X_list))
#     # ax[count][1].set_xlim([rt_min-100,rt_max+100])
#     # ax[count][1].set_ylim([0, 10])
#     ax[count][1].set_xlabel('RT (ms)')
#     ax[count][1].set_ylabel('Integrated Value')
#     ax[count][1].invert_yaxis()
#
#     ax[count][1].legend()
#
#     ax[count][0].set_title('Stimulus Duration: %s ms' % int(s*1000))
#     count +=1
#
# fig.suptitle('Scatterplots of Integrated Values by Choice (red: More Os, black: More Xs)\n binomial parameters: (0.62, 0.38)',fontsize = 14)
# fig.tight_layout()
# fig.show()
#
#
#
#
# ############## plot the chains ###############
