""" Test script """
import numpy as np
import time
from datetime import date

import dataparser as dp
import datepath as dtp
import display as disp
import monkeyexceptions as me

NUM_TRIES = 1
CLF = int(255 / max([1, NUM_TRIES-1]))

dtp.datepath.Island = dp.Island()

# MEMORY GENERATOR, DISPLAY and SAVE
disp.display_island(index=1, show=False)
for i in range(0, NUM_TRIES):
    print("working on memory trial " + str(i))
    try:
        path = dtp.datepath.RandomMemory(dp.Fruits(), monkey=i)
        # path.date = date(2018,11,28)
        path.savecsv()
        print("saved memory path " + str(i))
        col = [i*CLF, 0, 0]
        disp.display(path.path(), show=False, color=col)
    except(me.PathOnWaterException):
        print("!!XX memory path " + str(i) + " failed (on water)")
disp.display_fruits(dp.Fruits(), fruitsize=3)


quit();

# RANDOM GENERATOR and DISPLAY
disp.display_island(index=1, show=False)
for i in range(0, NUM_TRIES):
    rdm = np.random.uniform(0,1)
    st = time.clock()
    if rdm < 0.0:
        print("working on memory trial " + str(i))
        path = dtp.datepath.RandomMemory(dp.Fruits(), monkey=i)
        col = [i*CLF, 0, 0]
    else:
        print("working on view trial " + str(i))
        path = dtp.datepath.RandomView(dp.Fruits(), monkey=i)
        col = [0, 255-i*CLF, 0]
    end = time.clock()
    print("displaying on trial " + str(i) + ", computation required " + str(end-st) + "s")
    disp.display(path.path(), show=False, color=col)
disp.display_fruits(dp.Fruits(), fruitsize=3)
