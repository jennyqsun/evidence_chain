#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:43:10 2022

@author: isaacmenchaca
"""
from psychopy import visual, event, core
from generateX0Trial import *
from datetime import datetime


#addData = (sessionID, BlockID, trialID, flanker, target, resp, cond[trial], crit[trial], rt)
def experiment(numTrials, probVariability):

    #idk = informationInputGUI()
    #print(type(idk))
    
    # data: subject identifier, trial number, time of trial onset, what variablity was present, time of opt response, time of response.

    
    win = visual.Window(size=(1920, 1080), units='pix')
    
    timer = core.Clock()
    experimentStartTime = timer.getTime()
    
    data = []
    data.append(instructions(win, timer))
    for i in range(numTrials):
        # numberOfItems: total X and 0s in grid.
        # n_n: a value n which determines an nxn grid.
        # probVariability: the biased probability towards 0 in a bernoulli process.
        # stimDuration: seconds to display stimulus.
        trial(win, numberOfItems = 40, n_n = 25, probVariability = probVariability, stimDuration = 250, frameRate = 60, timer = timer)
        
    experimentEndTime = timer.getTime()
    print(data)
    win.close()
    core.quit()
    
    return
    
    
#experiment(numTrials = 5, probVariability = [0.20, 0.35, 0.45, 0.55, 0.65, 0.80])

experiment(numTrials = 5, probVariability = [0.01, .99])
