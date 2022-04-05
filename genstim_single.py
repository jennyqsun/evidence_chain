# Created on 3/29/22 at 1:21 PM 

# Author: Jenny Sun

import numpy
from psychopy.visual import TextStim
from psychopy import visual, data, event, core, gui, logging
from numpy.random import binomial, uniform
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

# generate a unit circle
def gen_unitCircle():
    r = 1
    rads = np.linspace(0,(2*np.pi), 1000)
    x = r*np.cos(rads)
    y = r*np.sin(rads)
    return x, y

def gen_unitX():
    r=np.linspace(0,1,1000)
    rads = [1/4*np.pi,3/4*np.pi,5/4 * np.pi,7/4*np.pi]
    x=[]
    y=[]
    for rad in rads:
        x.append(r*np.cos(rad))
        y.append(r*np.sin(rad))
    return x,y
def save_unitCircle(x,y,linewidth, color,figsize):
    fig, ax = plt.subplots(1, figsize=(figsize, figsize))
    ax.plot(x, y, linewidth=linewidth, color=color)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.axis("off")
    fig.savefig("stim/output0.png")
    return fig

def save_unitX(x,y,linewidth, color,figsize):
    fig, ax = plt.subplots(1, figsize=(figsize, figsize))
    for i in range(len(x)):
        ax.plot(x[i], y[i], linewidth=linewidth, color=color)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    fig.savefig("stim/outputX.png")
    return fig

def png2array(fname):
    im = Image.open(fname)
    im = np.array(im)[:,:,0:3]
    return im
def rgb2gray(rgb):
    r,g,b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
def greySwap(gray):
    gray = 255-gray
    return gray



x,y = gen_unitCircle()
fig = save_unitCircle(x=x,y=y,linewidth=11,color='black',figsize=4)
im = png2array('stim/output0.png')
g = greySwap(rgb2gray(im))
g[g == np.min(g)] = 0


x_X,y_X = gen_unitX()
fig = save_unitX(x_X,y_X,linewidth=11,color='black',figsize=4)
imX = png2array('stim/outputX.png')
gX = greySwap(rgb2gray(imX))
gX[gX == np.min(gX)] = 0
g = g*0.5
gX = gX*0.5
imX = Image.fromarray(gX)
imX.show()


g_ = g * np.sum(gX)/ np.sum(g)
im_ = Image.fromarray(g_)
im_.show()

im_.convert('RGB').save('stim/output0.png')

imX.convert('RGB').save('stim/outputX.png')

# convert RGB to greyscale


# let's balance the pixel energy
