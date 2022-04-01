import numpy
from psychopy.visual import TextStim
from psychopy import visual, data, event, core, gui, logging
from numpy.random import binomial, uniform
import numpy as np
import random
import pandas as pd

import matplotlib.pyplot as plt

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
    randlist = np.arange(0,100)
    allTrials = np.vstack((X0,X1))
    random.shuffle(randlist)
    allTrials = allTrials[randlist,:]
    return allTrials



def generateFixationCross(win):
    fixation = TextStim(win, text = '+', color='white', pos = (0,0))
    fixation.height = 50
    return fixation

def genVisualStim(win,size,pos):
    img0 = visual.ImageStim(win=win, image='stim/output0.png', units="pix", size=size, pos=pos)
    imgX = visual.ImageStim(win=win, image='stim/outputX.png', units="pix", size=size, pos=pos)
    return img0, imgX


def runTrial(allTrials,img0,imgX,win,stimDur,trialIndex, refreshRate,port=s,keymap=keymap,abortkey=7):
    keylist=[]
    sequence = allTrials[trialIndex]
    fix = generateFixationCross(win)
    endTrial = False
    count = 0
    key =[]
    press=[]
    btime=[]

    for f in range(int(0.5*refreshRate)):
        fix.draw()
        win.flip()
    timer.reset()
    clear_buffer(port)
    reset_timer(port)
    while endTrial is False:
        stim = img0 if sequence[count]==-1 else imgX
        for f in range(int(refreshRate*stimDur)):
            stim.draw()
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
                    return count, t1, key, press, btime
        count +=1
        if count == len(sequence):
            win.flip()
            endTrial = True
            t1 = timer.getTime()
    core.wait(1)
    return count,t1,key,press,btime

def winThreshold(win):
    win.recordFrameIntervals = True
    win.refreshThreshold = 1 / 60 + 0.003
    logging.console.setLevel(logging.WARNING)
    return win


win = visual.Window(size=(1920, 1080), units='pix',color='black')
win = winThreshold(win)
X0,X1,initSteps = simulateTrials(numTrials=50,numSteps=60,numStim=1,Bias=0.08,initRange = [0,4,8])
img0,imgX = genVisualStim(win,120,(0,0))
allTrials = randomizeTrials(X0,X1)
timer = core.Clock()

resp =[]
for t in range(0,5):
    count,t1,key,press,btime = runTrial(allTrials= allTrials,img0=img0,imgX=imgX,win=win,stimDur=0.1,trialIndex=t,refreshRate=60)
    try:
        btime = HexToRt(BytesListToHexList(btime))
    except IndexError:
        pass
    resp.append((t1, btime, press, count, key))
    print(t1,btime,count)
    print('Overall, %i frames were dropped.' % win.nDroppedFrames)

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


# check the randomwalk
fig, ax = plt.subplots(2,1, figsize = (10,12))
Lines0 = ax[0].plot(np.cumsum(allTrials[0:5], axis=1).T)
# ax[0].legend([Lines0[0]])

# check the correlation variance
ax[1].hist(np.corrcoef(allTrials[0:5,:])[~np.eye(5,5,dtype='bool')].flatten(), color = 'blue',alpha=0.8)
ax[1].set_title('Correlation Coefficients')
fig.show()
