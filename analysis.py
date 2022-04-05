# Created on 4/5/22 at 2:01 PM 

# Author: Jenny Sun
# Created on 4/5/22 at 12:44 PM

# Author: Jenny Sun

import numpy as np
import pandas as pd
import os
from os import listdir
import matplotlib.pyplot as plt

onlyfile = [f for f in listdir('data') if '001' in f]
df = []
for f in onlyfile:
    df.append(pd.read_csv('data/' + f))

df = pd.concat(df)

# for i, j in df.iterrows():
#     if df.iloc[i,:]['bytetime'] == '[]':
#         print('true')
#         df.iloc[i, :]['bytetime']. 'NaN'
df['bytetime'].replace('[]',np.nan, inplace =True)
sum(df['bytetime'].isnull()) / len(df)


df_1 = df[df['bytetime'].notnull()]
stimDur= df_1['stimDur'].unique()
stimDur.sort()

fig, ax = plt.subplots(5,2, figsize=(10,20))
count=0
for s in stimDur:
    df_0 = df_1[df_1['stimDur']==s]
    print(s)

    O_list = []
    X_list = []
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
            print('0')
            cumsum = np.cumsum(l[0:int(trial_count-ndt)])[-1]
            O_list.append(cumsum)

            # ax[0][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
        else:
            print('X')

            cumsum = np.cumsum(l[0:int(trial_count-ndt)])[-1]
            X_list.append(cumsum)

            # ax[1][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
    ax[count][0].hist(O_list,color = 'red',label = 'Chose Os (N = %s)'%len(O_list))
    ax[count][0].set_xlim([-20,20])
    ax[count][0].set_ylim([0, 10])
    ax[count][0].legend()
    ax[count][1].hist(X_list, color='black', label = 'Chose Xs (N = %s)'%len(X_list))
    ax[count][1].set_xlim([-20,20])
    ax[count][1].set_ylim([0, 10])
    ax[count][1].legend()

    ax[count][0].set_title('Stimulus Duration: %s ms' % int(s*1000), fontsize = 14)
    count +=1

fig.suptitle('Histograms of Integrated Values by Choice (NDT adjusted)\n',fontsize = 14)
fig.tight_layout()
fig.show()







fig, ax = plt.subplots(5,2, figsize=(10,20))
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
        ndt = round(0.3/s)


        if key == 'O':
            print('0')
            cumsum = np.cumsum(l[0:int(trial_count-ndt)])[-1]
            O_list.append(cumsum)
            O_rt.append(int(row['bytetime']))

            # ax[0][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
        else:
            print('X')

            cumsum = np.cumsum(l[0:int(trial_count-ndt)])[-1]
            X_list.append(cumsum)
            X_rt.append(int(row['bytetime']))


            # ax[1][0].plot(np.cumsum(l[0:int(trial_count - ndt)]), color=linecolor)
    ax[count][0].scatter(O_rt,O_list,color = 'red',label = 'Chose Os (N = %s)'%len(O_list))
    # ax[count][0].set_xlim([-20,20])
    # ax[count][0].set_ylim([0, 10])
    ax[count][0].set_xlabel('RT (ms)')
    ax[count][0].set_ylabel('Integrated Value')

    ax[count][0].legend()
    ax[count][1].scatter(X_rt, X_list, color='black', label = 'Chose Xs (N = %s)'%len(X_list))
    # ax[count][1].set_xlim([-20,20])
    # ax[count][1].set_ylim([0, 10])
    ax[count][1].set_xlabel('RT (ms)')
    ax[count][1].set_ylabel('Integrated Value')

    ax[count][1].legend()

    ax[count][0].set_title('Stimulus Duration: %s ms' % int(s*1000), fontsize = 14)
    count +=1

fig.suptitle('Scatterplots of Integrated Values by Choice (red: More Os, black: More Xs)\n',fontsize = 14)
fig.tight_layout()
fig.show()




