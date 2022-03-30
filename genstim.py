
import numpy
from psychopy.visual import TextStim
from psychopy import visual, data, event, core, gui, logging
from numpy.random import binomial, uniform
import numpy as np
import random



win = visual.Window(size=(151, 151),color = 'black')
compX =  visual.TextStim(win, text='X', units='pix', height=60, color='white')
compX.draw()
win.flip()
imX = win._getFrame()
texX = np.flipud(np.array(imX))
win.close()

win = visual.Window(size=(151, 151),color = 'black')
comp0 =  visual.TextStim(win, text='O', units='pix', height=60, color='white')
comp0.draw()
win.flip()
im0 = win._getFrame()
tex0 = np.flipud(np.array(im0))
win.close()


imX.save("texX_single.png")
im0.save("tex0_single.png")

