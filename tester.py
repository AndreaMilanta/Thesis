""" Test script """
import numpy as np
import pandas as pd
import time
from datetime import date
import traceback as tb

import dataparser as dp
import datepath as dtp
import display as disp
import monkeyexceptions as me
import monkeyconstants as mc
import geometry as geo
import simulation as sim

col = [255, 255, 255]
dtp.datepath.Island = dp.Island()
geo.Coordinates.Island = dp.Island()

tries = 0

#  Choose trees 
locs = np.random.randint(0, len(dp.Fruits()), 2)
locs = [500, 1000]
orig = dp.Fruits()[locs[0]]
dest = dp.Fruits()[locs[1]]

# Creat path
while tries < mc.MAX_TRIES:
    try:
        # path = sim._gotoknown(orig, dest)
        path = sim._gotounknown(orig, dest)
        print('created path of length: ' + str(len(path)))
    except me.PathOnWaterException:
        print('Failed due to path crossing water')
        tries += 1
        continue;
    break
dtpath = dtp.datepath(path, dp.Fruits())

# Display chosen trees
disp.display_island(index=1, show=False, block=False)
disp.display_fruits(orig, color='#000000', show=True, block=False)
disp.display_fruits(dest, color='#FFFFFF', show=True, block=False)
disp.display(dtpath.path, show=True, color=col)
