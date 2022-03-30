import numpy
from psychopy.visual import TextStim
from psychopy import visual, data, event, core, gui, logging
from numpy.random import binomial, uniform
import numpy as np
import random


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

def generateX0Block(numberOfItems, numberOfTrials, maxdisplay, rangeOfInit = (1,5)):
    probBlock = np.random.choice([0.44, 0.56],size=numberOfTrials)
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



def generateTrial(numTrial,positionBlock,num0sBlock,colorBlock,numberOfItems,FixationDur,StimDur,refreshRate,maxdisplay):
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
        if count == int(maxdisplay-1):
            endTrial=True
        count +=1
    t1 = timer.getTime()


        # for j in range(num0sTrial[i]):
        #     stim0 = TextStim(win, text='0', color=['black','white'][colorTrial[i][j]], pos=positionTrial[i][j])
        #     stimDisplay[j] = stim0
        # for k in range(numXsTrial[i]):
        #     k= -(k+1)
        #     stimX = TextStim(win, text='X', color=['black','white'][colorTrial[i][k]], pos=positionTrial[i][k])
        #     stimDisplay[k] = stimX
        # f = [s.draw() for s in stimDisplay]


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



def generateTrialArray(win, numTrial,positionBlock,num0sBlock,numberOfItems,FixationDur,StimDur,refreshRate,maxdisplay):
    fix = generateFixationCross(win)
    positionTrial = positionBlock[numTrial]
    num0sTrial =  num0sBlock[numTrial]
    numXsTrial = numberOfItems - num0sTrial
    # red is mapped to 0, blue is mapped to X

    timer = core.Clock()

    endTrial = False

    timer.reset()

    for i in range(int(FixationDur*refreshRate)):
        fix.draw()
        win.flip()
    count = 0
    t0 = timer.getTime()
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
        if count == int(maxdisplay - 1):
            endTrial = True
        count += 1
    t1 = timer.getTime()

    return t1,t0

maxdisplay = 20
numberOfItems = 6
numberOfTrials = 1
n_n=25
refreshRate = 60
stimDur = 1
# numTrial = 3




win = visual.Window(size=(1920, 1080), units='pix')
win.recordFrameIntervals = True
win.refreshThreshold = 1/60 + 0.004
logging.console.setLevel(logging.WARNING)


def runBlock(win, numTrialPerBlock,n_n, refreshRate):
    positionBlock = generatePositionBlock(n_n, numberofItems=numberOfItems, numberOfTrials=numTrialPerBlock, maxdisplay=maxdisplay)
    probBlock, num0sBlock, numXsBlock, initBlock = generateX0Block(numberOfItems, numTrialPerBlock, maxdisplay, (1, 5))
    core.wait(2)
    win.recordFrameIntervals = True
    for trial in range(0,numTrialPerBlock):
        print('prob: ', probBlock[trial])
        t1,t0 = generateTrialArray(win, trial,positionBlock,num0sBlock,numberOfItems,0.5,stimDur,refreshRate, maxdisplay)
        win.flip()
        core.wait(1)
    return positionBlock,probBlock,num0sBlock, initBlock, (t1,t0)


positionBlock,probBlock,num0sBlock, initBlock,t = runBlock(win, numberOfTrials,n_n,refreshRate)
win.close()

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
# plt.plot(np.arange(0,maxdisplay), np.cumsum((oo-probBlock/2),axis=1).T)
# plt.ylabel('cumsum')
# plt.show()
#
#
# count = 0
# for _ in range(0,50):
#     x = binomial(20,0.52,50)
#     cs = np.cumsum(x - 20 * 0.5)
#     if cs[-1] <0:
#         plt.plot(cs,color='black')
#         count += 1
#     else:
#         plt.plot(cs, color = 'red')
# print(count/50)
# plt.show()
#
#
# count = 0
# for _ in range(0,100):
#     x = binomial(1,0.55,50)
#     cs = np.cumsum(x - 1 * 0.5)
#     if cs[-1] <0:
#         plt.plot(cs,color='black')
#         count += 1
#     else:
#         plt.plot(cs, color = 'red')
# print(count/100)
# plt.show()


# def runTrial(win,FixationDur,Fixation,refreshRate, numTrial, positionBlock, num0sBlock, colorBlock, maxdisplay):
#     stim = [0]*maxdisplay
#     positions = positionBlock[numTrial]
#     X0
#
#
#
#     for i in range(FixationDur*refreshRate):
#         Fixation.draw()
#         win.flip()
#
#
# def generateTrialDisplay(win, numberOfItems, probabilityOf0, n_n):
#     '''output:      a list of text object for each display'''
#     positionsGrid = generateGridPlacement(n_n = n_n, numberOfItems = numberOfItems)
#     # 0s are the successes with a probability p of probability Of 0s
#     num0s = binomial(n = numberOfItems, p = probabilityOf0)
#     numXs = numberOfItems - num0s
#
#     stim = [0]*numberOfItems
#     pos0 =[0]*num0s
#     posX =[0]*numXs
#
#     for i in range(num0s): # 0(n)
#         print(i)
#         pos = positionsGrid[i]
#         pos0[i] = pos
#         stim0 = TextStim(win, text = '0', color = ['black', 'white'][binomial(1, 0.5)], pos = pos)
#         # stim0.setAutoDraw(True)
#         stim[i] = stim0
#
#     for i in range(numXs): # 0(n)
#         posX[i]=pos
#         i=i+1
#         print(-i)
#         pos = positionsGrid[-i]
#         stimX = TextStim(win, text = 'X', color = ['black', 'white'][binomial(1, 0.5)], pos = pos)
#         # stimX.setAutoDraw(True)
#         stim[-i] = stimX
#     return stim, pos0, posX, num0s
#
#
#
#     totalFrames = round((stimDuration / 1000) * frameRate)
#     startTime = timer.getTime()
#     for frame in range(totalFrames):
#         win.flip()
#     endTime = timer.getTime() - startTime
#
#     data = {'Stim Type': 'X0', 'Probability of 0': probabilityOf0, 'Total 0s': num0s, 'Start Time (ms)': startTime * 1000, 'Total Time (ms)': endTime * 1000, 'Total Frames': totalFrames}
#
#     for item in stim:
#         item.setAutoDraw(False)
#
#     return data
#
#
# def generateFixationCross(win, probabilityOf0, frameRate, timer, type = 'opt'):
#     fixation = TextStim(win, text = '+', pos = (0,0))
#     fixation.height = 50
#
#     if type == 'opt':
#         fixation.color = 'white'
#     elif type == 'response':
#         fixation.color = 'black'
#
#     fixation.setAutoDraw(True)
#
#
#     startTime = timer.getTime()
#     totalFrames = 0
#     keep_going = True
#     while keep_going:
#         totalFrames += 1
#         event.clearEvents(eventType='keyboard')
#         win.flip()
#         keys = event.getKeys(keyList=['f', 'j'], timeStamped=timer)
#         if len(keys) > 0:
#             keep_going = False
#             # draw one second here.
#
#     responseTime = keys[0][1] - startTime
#
#     correct = None
#     if type == 'opt':
#         endTime = responseTime
#     elif type == 'response':
#         for frame in range(frameRate): # waits 1 second before next trial. The ISI
#             win.flip()
#         endTime = timer.getTime() - startTime # end time of this fixation presentation.
#         totalFrames += frameRate # adding the ISI frames.
#
#         if (keys[0][0] == 'j' and probabilityOf0 > 0.5) or (keys[0][0] == 'f' and probabilityOf0 < 0.5):
#             correct = True
#         else:
#             correct = False
#
#     data = {'Stim Type': type, 'Response': keys[0][0], 'Probability of 0': probabilityOf0, 'Correct': correct, 'Start Time (ms)': startTime * 1000, 'Response Time (ms)':  responseTime * 1000, 'Total Time (ms)': endTime * 1000, 'Total Frames': totalFrames}
#
#     fixation.setAutoDraw(False)
#     return keys[0][0], data
#
#
# def trial(win, numberOfItems, n_n, probVariability, stimDuration, frameRate, timer):
#     # 10 trials just to test stimulus.
#     probabilityOf0 = np.random.choice(probVariability, size = 1)[0]
#
#     repeatedStimuli = True
#     while repeatedStimuli:
#         # give 300 ms for stimulus presentation.
#         data = generateX0Trial(win, numberOfItems = numberOfItems, probabilityOf0 = probabilityOf0, n_n = n_n, stimDuration = stimDuration, frameRate = frameRate, timer = timer)
#         print(data)
#
#         # white fixation: choose to answer or opt out. f to opt, j to skip.
#         optOrSkip, data = generateFixationCross(win, probabilityOf0 = probabilityOf0, frameRate = frameRate, timer = timer, type = 'opt')
#         print(data)
#
#         # black fixation: choose answer.
#         if 'f' in optOrSkip:
#             _, data = generateFixationCross(win, probabilityOf0 = probabilityOf0, frameRate = frameRate, timer = timer, type = 'response')
#             repeatedStimuli = False
#             print(data)
#
#     return # if correct return 1. else return 0?
#
#
# def informationInputGUI():
#     exp_name = 'Letter-Biased Task'
#
#     exp_info = {'participant ID': '',
#                 'gender:': ('male', 'female'),
#                 'age': '',
#                 'left-handed': False}
#     dlg = gui.DlgFromDict(dictionary = exp_info, title = exp_name)
#
#     exp_info['date'] = data.getDateStr()
#     exp_info['exp name'] = exp_name
#
#     if dlg.OK == False:
#         core.quit() # ends process.
#
#
#     return exp_info
