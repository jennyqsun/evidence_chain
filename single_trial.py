import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import binomial
import random
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 14})

seed=86693
random.seed(seed)
np.random.seed(seed)

walk = binomial(1,0.62,30)
walk[walk==0]=-1

fig, ax = plt.subplots(figsize=((10,5)))
ax.plot(np.arange(1,23),np.cumsum(walk)[:22],'o',ls='-')
ax.axvline(21,ls = '--')
ax.axvline(22,ls = '--')
ax.axhline(5,ls = '--',color='green',label ='Boundary')

ax.axhline(0,ls= '-',color='k')
ax.axvline(21.5, ls='-',color='red',label = 'Response at 6375ms')
cumsum = np.cumsum(walk)[:21]

ax.margins(x=0)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax2 = ax.twiny()
ax2.set_xticks(np.arange(0,6600,500))
ax2.set_xlabel('Time (ms)')
#l = ax.get_xticklabels()
ax.set_ylabel('Integrated Evidence')
ax.set_xticks(np.arange(0,22))
ax.set_xlabel('Number of Steps')

p = sum(walk[0:22]==1)/len(walk[0:22])
q = sum(walk[0:22]==-1)/len(walk[0:22])

drift = (p-q) * (5/6.375)
slope =  0.23809523809524
fitX = np.linspace(0,21,1000)
fitY = slope * fitX

fitX1 = np.linspace(0,21.5,1000)
fitY1 = drift * fitX

ax.plot(fitX, fitY,label = 'Fitted Slope=0.238')

# ax.plot(fitX1, fitY1,label = 'fitted drift=0.214')


ax.legend()
plt.show()

fig.savefig('Fit_Drift.png')
