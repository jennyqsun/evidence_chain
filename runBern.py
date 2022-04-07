import numpy
from psychopy.visual import TextStim
from psychopy import visual, data, event, core, gui, logging
from numpy.random import binomial, uniform
import numpy as np
import random
import pandas as pd
import csv
import pickle
import matplotlib.pyplot as plt
import sys

try:
    from cedrus_util import *

    # set up resposne pad
    portname = serial_ports()[0]
    print('writing to device...')
    identiy_device()
    device_id, model_id = get_model(portname)
    keymap = def_keyboard(device_id, model_id)
    s = serial.Serial(portname, 115200)
    print('serial port opened')
except:
    print('cant find response pad. Exiting now')
    sys.exit()





def generateIntro(win):
    intro = TextStim(win, text = 'Press RIGHT key if there are more Os \n Press LEFT key if there are more Xs\n\n Press any key on keypad to start', color='white', pos = (0,0))
    return intro


def generateBreak(win):
    breakPage = TextStim(win, text = 'Break Time. \n Press any key to start.', color='white', pos = (0,0))
    return breakPage


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

def randomizeTrials(X0,X1):
    trialTotal = len(X0) + len(X1)
    randlist = np.arange(0,int(trialTotal))
    allTrials = np.vstack((X0,X1))
    random.shuffle(randlist)
    allTrials = allTrials[randlist,:]
    return allTrials



def generateFixationCross(win):
    fixation = TextStim(win, text = '+', color='white', pos = (0,0), opacity = 0.8)
    fixation.height = 50
    return fixation

def genVisualStim(win,size,pos):
    img0 = visual.ImageStim(win=win, image='stim/output0.png', units="pix", size=size, pos=pos)
    imgX = visual.ImageStim(win=win, image='stim/outputX.png', units="pix", size=size, pos=pos)
    return img0, imgX

def genPC(win):
    pc = visual.ImageStim(win=win, image='stim/rect.png', units="pix", pos=(930,-230))
    return pc


def genPos(direction, distance):
    pos0 = (0,distance)
    pos1 = (distance,0)
    pos2 = (0,-distance)
    pos3 = (-distance,0)
    if direction == 'clock':
        pos = (pos0,pos1,pos2,pos3)
    else:
        pos = (pos0, pos1, pos2, pos3)
        pos = pos[::-1]
    return (pos0,pos1,pos2,pos3)

def stackPos(numSteps, direction, distance):
    posVec = genPos('clock', distance)
    repeats = np.ceil(numSteps/len(posVec))
    posAll = posVec * int(repeats)
    return posAll

def runTrial(allTrials,img0,imgX,win,stimDur,trialIndex, refreshRate,posAll,port=s,keymap=keymap,abortkey=7):
    timer = core.Clock()
    keylist=[]
    sequence = allTrials[trialIndex]
    fix = generateFixationCross(win)
    pc = genPC(win)
    endTrial = False
    count = 0
    key =[]
    press=[]
    btime=[]
    t1=[]

    for f in range(int(0.5*refreshRate)):
        fix.draw()
        win.flip()
    timer.reset()
    clear_buffer(port)
    reset_timer(port)
    while endTrial is False:
        stim = img0 if sequence[count]==1 else imgX    # if 1 show 0s, if -1 show X
        for f in range(int(refreshRate*stimDur)):
            stim.pos = posAll[count]
            stim.draw()
            fix.draw()
            pc.draw()
            win.flip()
            k = port.in_waiting
            if k != 0:
                t1 = timer.getTime()
                win.flip()
                keylist.append(port.read(port.in_waiting))
                key, press, btime = readoutput([keylist[-1]], keymap)
                if key[-1] ==abortkey:
                    win.close()
                    port.close()
                    core.quit()
                if press[0] ==1:
                    endTrial = True
                    core.wait(1)
                    return count, t1, key, press, btime, stimDur, sequence
        count +=1
        # if count == len(sequence):
        #     while endTrial is False:
        #         win.flip()
        #         k = port.in_waiting
        #         if k != 0:
        #             t1 = timer.getTime()
        #             keylist.append(port.read(port.in_waiting))
        #             key, press, btime = readoutput([keylist[-1]], keymap)
        #             if press[0] == 1:
        #                 endTrial = True
        #                 core.wait(1)
        if count == len(sequence):
            endTrial = True
            for _ in range(int(0.5*refreshRate)):
                win.flip()
                k = port.in_waiting
                if k != 0:
                    t1 = timer.getTime()
                    keylist.append(port.read(port.in_waiting))
                    key, press, btime = readoutput(keylist, keymap)
                    nonzeroInd = next((i for i, x in enumerate(press) if x), None)
                    key = [key[nonzeroInd]]
                    press = [press[nonzeroInd]]
                    btime = [btime[nonzeroInd]]

            core.wait(0.5)

    return count,t1,key,press,btime,stimDur,sequence

# def getStopChain(count, key, press, sequence, stimDur, ndt=0.2)
#
# def accCalculator(count, key, press, sequence, stimDur, ndt=0.2):
#     chain = sequence[0:count+1]
#     backCount = int(np.round(ndt/stimDur))
#     if len(chain)==1:
#     chain = chain[:-backCount]
#
#     return chain


def winThreshold(win):
    win.recordFrameIntervals = True
    win.refreshThreshold = 1 / 60 + 0.005
    logging.console.setLevel(logging.WARNING)
    return win

def clear_reset_port(port):
    clear_buffer(port)
    reset_timer(port)

# set up experiment

def beginExp(port,numTrials,numSteps,numStim,Bias,initRange, newWin=True, win=None):
    if newWin is True:
        win = visual.Window(size=(1920, 1080), units='pix',color='black')
        win = winThreshold(win)
        intro = generateIntro(win)
        intro.draw()
        win.flip()
        clear_reset_port(port)
        resp = False
        while resp is False:
            k = port.in_waiting
            if k != 0:
                resp = True
    else:
        win = win
    X0,X1,initSteps = simulateTrials(numTrials=numTrials,numSteps=numSteps,numStim=numStim,Bias=Bias,initRange = initRange)
    img0,imgX = genVisualStim(win,61,(0,0))
    allTrials = randomizeTrials(X0,X1)
    return win, allTrials, img0, imgX, X0,X1


def runBlock(port,numTrials,numSteps,numStim,Bias,initRange, stimDur, refreshRate=60, newWin = True, win=None, direction='clock', distance=0.5):
    win,allTrials,img0,imgX, X0,X1 = beginExp(port=port,numTrials=numTrials,numSteps=numSteps,numStim=numStim,Bias=Bias,initRange = initRange, newWin=newWin, win=win)
    posAll = stackPos(numSteps, direction='clock',distance= 45)
    resp =[]
    for t in range(0,len(allTrials)):
        count,t1,key,press,btime, stimDur, sequence = runTrial(allTrials= allTrials,img0=img0,imgX=imgX,win=win,stimDur=stimDur,trialIndex=t,refreshRate=refreshRate, posAll=posAll)
        try:
            btime = HexToRt(BytesListToHexList(btime))
        except IndexError:
            pass
        resp.append((t1, btime, press, count, key, stimDur, Bias, sequence))
        print(t1,btime,count)
        print('Overall, %i frames were dropped.' % win.nDroppedFrames)
    return resp, allTrials,X0,X1, win

def break_wait(win):
    breakPage = generateBreak(win)
    breakPage.draw()
    win.flip()
    clear_reset_port(s)
    resp = False
    while resp is False:
        k = s.in_waiting
        if k != 0:
            resp = True
    return win

def savePKL(array, filename):
    fileObject = open(filename, 'wb')
    pickle.dump(array, fileObject)
    fileObject.close()
    return filename

def loadPKL(filename):
    myfile = open(filename, 'rb')
    f = pickle.load(myfile)
    return f



###########################################################
subj = input('############')
trialPerBlock = 50
numSteps = 30
if trialPerBlock % 2 != 0:
    print('!!! error: please input even trials')
numTrial = int(trialPerBlock/2)

#

resp, allTrials,X0,X1, win = runBlock(port=s,numTrials=numTrial,numSteps=numSteps,numStim=1,Bias=0.12,initRange = [0,0,0], stimDur= 0.1, refreshRate=60, newWin=True)
df = pd.DataFrame(resp)
df.columns = ['time','bytetime','press','count','key','stimDur','Bias','sequence']

trialFile = savePKL(allTrials, filename='data/'+ subj + '_block_0' + '_chains')
df.to_csv('data/'+ subj + '_block_0' + '.csv', index= False)
win = break_wait(win)
win = winThreshold(win)





resp, allTrials,X0,X1, win = runBlock(port=s,numTrials=13,numSteps=numSteps,numStim=1,Bias=0.12,initRange = [0,0,0], stimDur= 0.4, refreshRate=60, newWin=False, win=win)
df = pd.DataFrame(resp)
df.columns = ['time','bytetime','press','count','key','stimDur','Bias','sequence']
trialFile = savePKL(allTrials, filename='data/'+ subj + '_block_1' + '_chains')
df.to_csv('data/'+ subj + '_block_1' + '.csv', index= False)
win = break_wait(win)
win = winThreshold(win)
#
#
resp, allTrials,X0,X1, win = runBlock(port=s,numTrials=numTrial,numSteps=numSteps,numStim=1,Bias=0.12,initRange = [0,0,0], stimDur= 0.1, refreshRate=60, newWin=False, win=win)
df = pd.DataFrame(resp)
df.columns = ['time','bytetime','press','count','key','stimDur','Bias','sequence']
trialFile = savePKL(allTrials, filename='data/'+ subj + '_block_2' + '_chains')
df.to_csv('data/'+ subj + '_block_2' + '.csv', index= False)
win = break_wait(win)
win = winThreshold(win)

#
resp, allTrials,X0,X1, win = runBlock(port=s,numTrials=13,numSteps=numSteps,numStim=1,Bias=0.12,initRange = [0,0,0], stimDur= 0.4, refreshRate=60, newWin=False, win=win)
df = pd.DataFrame(resp)
df.columns = ['time','bytetime','press','count','key','stimDur','Bias','sequence']
trialFile = savePKL(allTrials, filename='data/'+ subj + '_block_3' + '_chains')
df.to_csv('data/'+ subj + '_block_3' + '.csv', index= False)
win = break_wait(win)
win = winThreshold(win)
#
resp, allTrials,X0,X1, win = runBlock(port=s,numTrials=numTrial,numSteps=numSteps,numStim=1,Bias=0.12,initRange = [0,0,0], stimDur= 0.2, refreshRate=60, newWin=False, win=win)
df = pd.DataFrame(resp)
df.columns = ['time','bytetime','press','count','key','stimDur','Bias','sequence']
trialFile = savePKL(allTrials, filename='data/'+ subj + '_block_4' + '_chains')
df.to_csv('data/'+ subj + '_block_4' + '.csv', index= False)
win = break_wait(win)
win = winThreshold(win)
#
#
#
resp, allTrials,X0,X1, win = runBlock(port=s,numTrials=13,numSteps=numSteps,numStim=1,Bias=0.12,initRange = [0,0,0], stimDur= 0.4, refreshRate=60, newWin=False, win=win)
df = pd.DataFrame(resp)
df.columns = ['time','bytetime','press','count','key','stimDur','Bias','sequence']
trialFile = savePKL(allTrials, filename='data/'+ subj + '_block_5' + '_chains')
df.to_csv('data/'+ subj + '_block_5' + '.csv', index= False)
win = break_wait(win)
win = winThreshold(win)
#
resp, allTrials,X0,X1, win = runBlock(port=s,numTrials=numTrial,numSteps=numSteps,numStim=1,Bias=0.12,initRange = [0,0,0], stimDur= 0.2, refreshRate=60, newWin=False, win=win)
df = pd.DataFrame(resp)
df.columns = ['time','bytetime','press','count','key','stimDur','Bias','sequence']
trialFile = savePKL(allTrials, filename='data/'+ subj + '_block_6' + '_chains')
df.to_csv('data/'+ subj + '_block_6' + '.csv', index= False)
win = break_wait(win)
win = winThreshold(win)
#
resp, allTrials,X0,X1, win = runBlock(port=s,numTrials=13,numSteps=numSteps,numStim=1,Bias=0.12,initRange = [0,0,0], stimDur= 0.4, refreshRate=60, newWin=False, win=win)
df = pd.DataFrame(resp)
df.columns = ['time','bytetime','press','count','key','stimDur','Bias','sequence']
trialFile = savePKL(allTrials, filename='data/'+ subj + '_block_7' + '_chains')
df.to_csv('data/'+ subj + '_block_7' + '.csv', index= False)
win = break_wait(win)
win = winThreshold(win)
#
#
win.close()

#
# breakPage = generateBreak(win)
# breakPage.draw()
# win.flip()
# clear_reset_port(s)
# resp = False
# while resp is False:
#     k = s.in_waiting
#     if k != 0:
#         resp = True




# check the randomwalk
fig, ax = plt.subplots(2,1, figsize = (10,12))
Lines0 = ax[0].plot(np.cumsum(X0, axis=1).T,color='blue')
Lines1 = ax[0].plot(np.cumsum(X1, axis=1).T, ls = '--',color = 'orange')
ax[0].legend([Lines0[0], Lines1[0]],['prob < 0.5', 'prob > 0.5'])

# check the correlation variance
ax[1].hist(np.corrcoef(X0)[~np.eye(len(X0),len(X0),dtype='bool')].flatten(), color = 'blue',alpha=0.8)
ax[1].hist(np.corrcoef(X1)[~np.eye(len(X0),len(X0),dtype='bool')].flatten(),color='orange',alpha=0.8)
ax[1].set_title('Correlation Coefficients')
fig.show()


# # check the randomwalk
# fig, ax = plt.subplots(2,1, figsize = (10,12))
# Lines0 = ax[0].plot(np.cumsum(allTrials[0:5], axis=1).T)
# # ax[0].legend([Lines0[0]])
#
# # check the correlation variance
# ax[1].hist(np.corrcoef(allTrials[0:5,:])[~np.eye(5,5,dtype='bool')].flatten(), color = 'blue',alpha=0.8)
# ax[1].set_title('Correlation Coefficients')
# fig.show()

for index, row in df.iterrows():
    trial_count = row['count']
    print(trial_count)
    key = 'O' if row['key'] == [5] else 'X'
    linecolor = 'red' if key == 'O' else 'black'
    plt.plot(np.cumsum(row['sequence'][0:trial_count-3]), color = linecolor)
plt.title('red: 0, black: X')
plt.show()

key =[]
for i in df['key']:
    if isinstance(i, int):
        i = [i]
    if len(i)>0:
        key.append(i[0])
    else:
        key.append(0)

l = [True if i==2 else False for i in key]
seq = []
for index, row in df[l].iterrows():
    ratio0 = (sum(row['sequence'][0:row['count'] - 3] == 1)/ len(row['sequence'][0:row['count'] - 3]))
    # if ratio0>0.5:
    #     print(ratio0)
    plt.plot(np.cumsum(row['sequence'][0:row['count']-3]))

plt.show()

# block 50 trials 0.05, 50 trials of 0.08, 50 trials of 0.1
