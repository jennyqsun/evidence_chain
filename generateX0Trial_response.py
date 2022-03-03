import numpy
from psychopy.visual import TextStim
from psychopy import visual, data, event, core, gui, logging
from numpy.random import binomial, uniform
import numpy as np
import random
import pandas as pd

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

def instructions(win, timer):
    instructions = TextStim(win, text = 'After stimulus displays, a white fixation will appear. press F to choose to answer or J to skip the trial.\n' +
                                        'If F was selected, a black fixation will appear. This is an indication to select an answer.\n' +
                                        'Press F for majority X, press J for majority 0.' +
                                        'Press SPACE key to start.', pos = (0,0))
    
    instructions.setAutoDraw(True)
    keep_going = True
    totalFrames = 0
    #timer = core.Clock()
    startTime = timer.getTime()
    while keep_going:
        totalFrames += 1
        win.flip()
        keys = event.getKeys(keyList=['space'], timeStamped=timer)
        if len(keys) > 0:
            keep_going = False
            
   
    endTime = keys[0][1] - startTime
    instructions.setAutoDraw(False)
    
    print({'Stim Type': 'Instructions', 'Start Time (ms)': startTime * 1000,
            'Total Time (ms)': endTime * 1000, 'Total Frames': totalFrames})
            
    return {'Stim Type': 'Instructions', 'Start Time (ms)': startTime * 1000,
            'Total Time (ms)': endTime * 1000, 'Total Frames': totalFrames}

def generateFixationCross(win):
    fixation = TextStim(win, text = '+', pos = (0,0))
    fixation.height = 50
    return fixation


def generatePositionBlock(n_n, numberofItems, numberOfTrials, maxdisplay):
    ''':returns a list that contains positions of each display for each trial'''
    positionBlock = [0] * numberOfTrials
    for i in range(numberOfTrials):
        positionTrial = [0] * maxdisplay
        for j in range(maxdisplay):
            positionTrial[j] = generateGridPlacement(n_n, numberofItems)
        positionBlock[i] = positionTrial
    return positionBlock

def generateX0Block(numberOfItems, numberOfTrials, maxdisplay, probInterval, rangeOfInit = (1,5)):
    probBlock = np.random.choice(probInterval,size=numberOfTrials)
    initBlock = np.random.choice(range(rangeOfInit[0],rangeOfInit[1]),size = numberOfTrials)
    num0sBlock = [0] * numberOfTrials
    numXsBlock = [0] * numberOfTrials
    for i in range(numberOfTrials):
        init = initBlock[i]
        num0sBlock[i] = binomial(n = numberOfItems, p = probBlock[i], size= maxdisplay)
        #
        # while len(num0sBlock[i] == numberOfItems) >0 :
        #     num0sBlock[num0sBlock[i]==numberOfItems] =  binomial(n = numberOfItems, p = probBlock[i], size= len(num0sBlock[i] == numberOfItems))
        num0sBlock[i][:init] = numberOfItems//2
        numXsBlock[i] = numberOfItems-num0sBlock[i]

    return probBlock, num0sBlock, numXsBlock, initBlock

def generateColorBlock(numberOfItems, numberOfTrials, maxdisplay, probColor):
    colorBlock = [0] * numberOfTrials
    for i in range(numberOfTrials):
        colorTrial = [0] * maxdisplay
        for j in range(maxdisplay):
            colorTrial[j] = binomial(n = 1, p = probColor, size= numberOfItems)
        colorBlock[i] = colorTrial
    return colorBlock

def generateGridPlacement(n_n, numberOfItems):
    # will generate a grid of nxn dimensions.
    
    # For size=(1920, 1080)
    grid = np.array(np.meshgrid(np.linspace(-250, 250, num=n_n), np.linspace(-250, 250, num=n_n))).T.reshape(-1, 2)
    
    # used numberOfItems to select a # of random positions from grid.
    positionsGrid = grid[np.random.choice(np.arange(0, n_n ** 2, 1), size = numberOfItems, replace=False),:]
    return positionsGrid.tolist()



def generateTrial(numTrial,positionBlock,num0sBlock,colorBlock,numberOfItems,FixationDur,StimDur,refreshRate,maxdisplay,port):
    fix = generateFixationCross(win)
    positionTrial = positionBlock[numTrial]
    num0sTrial =  num0sBlock[numTrial]
    numXsTrial = numberOfItems - num0sTrial
    colorTrial = colorBlock[numTrial]

    stim = [0] * maxdisplay
    for i in range(maxdisplay):
        stimDisplay = [0] * numberOfItems
        for j in range(numberOfItems):
            if j <num0sTrial[i]:
                stim0 = TextStim(win, text='0', color=['black','white'][colorTrial[i][j]], pos=positionTrial[i][j])
                stimDisplay[j] = stim0
            else:
                stimX = TextStim(win, text='X', color=['black', 'white'][colorTrial[i][j]], pos=positionTrial[i][j])
                stimDisplay[j] = stimX
        stim[i] = stimDisplay


    timer = core.Clock()

    endTrial = False
    count = 0
    clear_buffer(port)
    reset_timer(port)
    timer.reset()

    for i in range(int(FixationDur*refreshRate)):
        fix.draw()
        win.flip()
    t0 = timer.getTime()
    while endTrial is False:
        stimDisplay = stim[count]
        [s.draw() for s in stimDisplay]
        for frame in range(int(refreshRate * StimDur)):
            win.flip()
            k = port.in_waiting
            if  k != 0:
                endTrial=True
        count +=1
    t1 = timer.getTime()
    return t1,t0





from PIL import Image


im0 = Image.open("tex0.png")
tex0 = np.flipud(np.array(im0))
imX = Image.open("texX.png")
texX = np.flipud(np.array(imX))

texX = texX.astype('int')
texX = texX/255 * 2 -1
tex0 = tex0.astype('int')
tex0 = tex0/255 * 2 -1

#
# tex0[tex0==0] = 1
#
# tex0[tex0!=1] = 0

#
# texX[texX==0] = 1
#
# texX[texX!=1] = 0



def generateTrialArray(win, numTrial,positionBlock,num0sBlock,numberOfItems,FixationDur,StimDur,refreshRate,maxdisplay,port,keymap, abortkey):
    fix = generateFixationCross(win)
    positionTrial = positionBlock[numTrial]
    num0sTrial =  num0sBlock[numTrial]
    numXsTrial = numberOfItems - num0sTrial
    # red is mapped to 0, blue is mapped to X
    keylist = []
    btime = []
    press =[]
    timer = core.Clock()

    endTrial = False


    clear_buffer(port)


    for i in range(int(FixationDur*refreshRate)):
        fix.draw()
        win.flip()
    count = 0
    reset_timer(port)
    timer.reset()
    while endTrial is False:
        num0Display = num0sTrial[count]
        if num0Display != numberOfItems:
            blue_stim = visual.ElementArrayStim(
                win=win,
                # colors=(0.0, 0.0, 0.0),
                colorSpace='rgb',
                units="pix",
                nElements=numberOfItems - num0Display,
                elementTex=texX,
                elementMask=None,
                xys=positionTrial[count][num0Display:],
                sizes=20)
        if num0Display != 0:
            red_stim = visual.ElementArrayStim(
                win=win,
                # colors=(0, 0.0, 0.0),
                colorSpace='rgb',
                units="pix",
                nElements=num0Display,
                elementTex=tex0,
                elementMask=None,
                xys=  positionTrial[count][:num0Display],
                sizes=20)

        for frame in range(int(refreshRate * StimDur)):
            red_stim.draw()
            blue_stim.draw()
            win.flip()
            k = port.in_waiting
            if k != 0:
                t1 = timer.getTime()
                keylist.append(port.read(port.in_waiting))
                key, press, btime = readoutput([keylist[-1]], keymap)
                if key[-1] ==abortkey:
                    win.close()
                    port.close()
                    core.quit()
                if press[0] ==1:
                    endTrial = True
                    break

        if count == int(maxdisplay - 1):
            endTrial = True
            t1 = timer.getTime()
        count += 1


    return t1, btime, press, count-1

maxdisplay = 20
numberOfItems = 6
numberOfTrials = 2
n_n=25
refreshRate = 60
stimDur = 0.2
abortkey = 1
fixationDur = 1
# numTrial = 3


# for _ in range(10):
#     x = binomial(6, 0.44, 20)
#     x0 = binomial(6, 0.56, 20)
#
#     plt.plot(np.cumsum(x-3),color = 'blue')
#     plt.plot(np.cumsum(x0-3),color ='black')
# plt.show()

win = visual.Window(size=(1920, 1080), units='pix')
win.recordFrameIntervals = True
win.refreshThreshold = 1/refreshRate + 0.004
logging.console.setLevel(logging.WARNING)


def runBlock(win, numTrialPerBlock,n_n, refreshRate, probInterval, rangeOfInit, maxdisplay):
    resp = []
    positionBlock = generatePositionBlock(n_n, numberofItems=numberOfItems, numberOfTrials=numTrialPerBlock, maxdisplay=maxdisplay)
    probBlock, num0sBlock, numXsBlock, initBlock = generateX0Block(numberOfItems, numTrialPerBlock, maxdisplay = maxdisplay, probInterval=probInterval, rangeOfInit=rangeOfInit)
    core.wait(2)
    win.recordFrameIntervals = True
    for trial in range(0,numTrialPerBlock):
        print('prob: ', probBlock[trial])
        t1, btime, press, count= generateTrialArray(win, trial,positionBlock,num0sBlock,
                                   numberOfItems,fixationDur,stimDur,refreshRate, maxdisplay,port=s, keymap =keymap, abortkey=abortkey)
        win.flip()
        try:
            btime = HexToRt(BytesListToHexList(btime))
        except IndexError:
            pass
        resp.append((t1, btime, press, count))

        core.wait(1)
    return positionBlock,probBlock,num0sBlock, initBlock, resp


positionBlock,probBlock,num0sBlock, initBlock,t = runBlock(win, numberOfTrials,n_n,refreshRate ,maxdisplay = maxdisplay, probInterval=[0.44, 0.56], rangeOfInit=(1,5))
win.close()
s.close()



import matplotlib.pyplot as plt
oo = np.array(num0sBlock)

plt.plot(np.arange(0, maxdisplay),(oo-numberOfItems/2).T)
plt.xlabel('display')
plt.ylabel('number of Xs (0 means half and half) ')
plt.show()

plt.plot(np.arange(0,maxdisplay), np.cumsum((oo-numberOfItems/2),axis=1).T)
plt.ylabel('cumsum')
plt.show()

print('Overall, %i frames were dropped.' % win.nDroppedFrames)
