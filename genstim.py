win = visual.Window(size=(60, 60))
compX =  visual.TextStim(win, text='X', units='pix', height=60, color='white')
compX.draw()
win.flip()
imX = win._getFrame()
texX = np.flipud(np.array(imX))
win.close()

win = visual.Window(size=(60, 60))
comp0 =  visual.TextStim(win, text='O', units='pix', height=60, color='white')
comp0.draw()
win.flip()
im0 = win._getFrame()
tex0 = np.flipud(np.array(im0))
win.close()


imX.save("texX.png")
im0.save("tex0.png")

