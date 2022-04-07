# Created on 4/5/22 at 2:01 PM 

# Author: Jenny Sun
# Created on 4/5/22 at 12:44 PM

# Author: Jenny Sun

import numpy as np
import pandas as pd
import os
from os import listdir
import matplotlib.pyplot as plt
import pickle

plt.rcParams.update({'font.size': 17})
onlyfile = [f for f in listdir('data') if '111' in f and 'csv' in f]
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
df['bytetime'].replace('[]',np.nan, inplace =True)
sum(df['bytetime'].isnull()) / len(df)


df_1 = df[df['bytetime'].notnull()]
stimDur= df_1['stimDur'].unique()
stimDur.sort()



fig, ax = plt.subplots(3,3, figsize=(15,15))
count=0
for s in stimDur:
    df_0 = df_1[df_1['stimDur']==s]
    # print(s)

    O_list = []
    X_list = []
    O_chain=[]
    X_chain=[]
    for index, row in df_0.iterrows():
        trial_count = row['count']
        if trial_count <1:
            continue

        key = 'O' if row['key'] == "[5]" else 'X'
        linecolor = 'red' if key == 'O' else 'black'
        seq = row['sequence'].split(".")
        l = []
        for i in seq:
            i = i.replace("[",'')
            i = i.replace("]",'')
            if '1' in i:
                l.append(int(i))
        ndt = round(0.3/s)


        if key == 'O':
            # print('0')
            cumsum = np.cumsum(l[0:int(trial_count+1-ndt)])[-1]
            O_list.append(cumsum)
            O_chain.append(np.cumsum(l[0:int(trial_count+1-ndt)]))

            # ax[0][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
        else:
            # print('X')

            cumsum = np.cumsum(l[0:int(trial_count+1-ndt)])[-1]
            X_list.append(cumsum)
            X_chain.append(np.cumsum(l[0:int(trial_count+1-ndt)]))

            # ax[1][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
    ax[count][2].plot(O_chain)
    ax[count][0].hist(O_list,color = 'red',label = 'Chose Os (N = %s)'%len(O_list), bins = (np.arange(-15,16,1)))
    ax[count][0].set_xlim([-20,20])
    ax[count][0].set_ylim([0, 25])
    ax[count][0].legend()
    ax[count][1].hist(X_list, color='black', label = 'Chose Xs (N = %s)'%len(X_list), bins = (np.arange(-15,16,1)))
    ax[count][1].set_xlim([-20,20])
    ax[count][1].set_ylim([0, 25])
    ax[count][1].legend()

    ax[count][0].set_title('Stimulus Duration: %s ms. ' % int(s*1000))
    count +=1

fig.suptitle('Histograms of Integrated Values by Choice (NDT adjusted)\n binomial parameters: (0.62, 0.38)')
fig.tight_layout()
fig.show()





fig, ax = plt.subplots(3,3, figsize=(15,15))
count=0
for s in stimDur:
    df_0 = df_1[df_1['stimDur']==s]
    print(s)

    O_list = []
    X_list = []
    O_rt = []
    X_rt =[]
    for index, row in df_0.iterrows():
        trial_count = row['count']
        if trial_count <1:
            continue

        key = 'O' if row['key'] == "[5]" else 'X'
        linecolor = 'red' if key == 'O' else 'black'
        seq = row['sequence'].split(".")
        l = []
        for i in seq:
            i = i.replace("[",'')
            i = i.replace("]",'')
            if '1' in i:
                l.append(int(i))
        ndt = np.round(0.3/s)


        if key == 'O':
            # print('0')
            cumsum = np.cumsum(l[0:int(trial_count+1-ndt)])[-1]
            O_list.append(cumsum)
            O_rt.append(int(row['bytetime']))

            # ax[0][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
        else:
            # print('X')

            cumsum = np.cumsum(l[0:int(trial_count+1-ndt)])[-1]
            X_list.append(cumsum)
            X_rt.append(int(row['bytetime']))


            # ax[1][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)


    O_acc = sum( [1 for i in O_list if i>0]) / len(O_list)
    X_acc = sum( [1 for i in X_list if i<0]) / len(X_list)

    rt_max =max(np.max(O_rt), np.max(X_rt))
    rt_min = min(np.min(O_rt), np.min(X_rt))
    print(O_acc, X_acc)
    ax[count][2].bar((0,0.5),(O_acc,X_acc), width = 0.3)
    ax[count][2].set_xticks((0,0.5))
    ax[count][2].set_xticklabels(('O',  'X'))
    ax[count][2].set_xlabel('Correct Choice')
    ax[count][2].set_ylabel('Accuracy')
    ax[count][2].set_ylim(0.5,1)
    ax[count][0].scatter(O_rt,O_list,color = 'red',label = 'Chose Os (N = %s)'%len(O_list))
    ax[count][0].set_xlim(rt_min-100,rt_max+100)
    # ax[count][0].set_ylim([-15, 15])
    ax[count][0].set_xlabel('RT (ms)')
    ax[count][0].set_ylabel('Integrated Value')

    ax[count][0].legend(loc = 'upper right')
    ax[count][1].scatter(X_rt, X_list, color='black', label = 'Chose Xs (N = %s)'%len(X_list))
    ax[count][1].set_xlim([rt_min-100,rt_max+100])
    # ax[count][1].set_ylim([0, 10])
    ax[count][1].set_xlabel('RT (ms)')
    ax[count][1].set_ylabel('Integrated Value')

    ax[count][1].legend()

    ax[count][0].set_title('Stimulus Duration: %s ms' % int(s*1000))
    count +=1

fig.suptitle('Scatterplots of Integrated Values by Choice (red: More Os, black: More Xs)\n binomial parameters: (0.62, 0.38)',fontsize = 14)
fig.tight_layout()
fig.show()




############## plot the chains ###############
