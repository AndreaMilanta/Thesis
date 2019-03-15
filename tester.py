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

PATH_TO_GEN = 500 * 2

col = [255, 255, 255]
dtp.datepath.Island = dp.Island()
geo.Coordinates.Island = dp.Island()
tries = 0

#  Choose trees
locs = np.random.randint(0, len(dp.Fruits()), 2)
locs = [1000, 600]
locs = [1000, 0]
orig = dp.Fruits()[locs[0]]
dest = dp.Fruits()[locs[1]]
fruits = dp.Fruits()
fruits.remove(orig)

skipmemory = True           # do not compute memory paths
skipperception = False      # do not compute perception paths

# Creat path
tot = 0
while tot < PATH_TO_GEN:
    # even loops
    while tot%2 == 0 and tries < mc.MAX_TRIES:
        if  not skipmemory:
            try:
                dtpath = dtp.datepath.RandomMemory(dp.Fruits(), monkey=0)
                print('\ncreated memory path ' + str(tot) + ' of length: ' + str(dtpath.length))
                title = "{0:02d} - Memory".format(tot)
            except (me.PathOnWaterException, me.MaxIterationsReachedException):
                print('\nFailed to create memory path ' + str(tot) + ' due to crossing water or max iterations')
                tries += 1
                continue;
        break
    # odd loops
    while tot%2 == 1 and tries < mc.MAX_TRIES:
        if not skipperception:
            try:
                dtpath = dtp.datepath.RandomView(dp.Fruits(), monkey=0)
                print('\ncreated random path ' + str(tot) + ' of length: ' + str(dtpath.length))
                title = "{0:02d} - Perception".format(tot)
            except (me.PathOnWaterException, me.MaxIterationsReachedException):
                print('\nFailed to create random path ' + str(tot) + ' due to crossing water or max iterations')
                tries += 1
                continue;
        break

    if not(tot%2==0 and skipmemory) and not(tot%2==1 and skipperception): 
        angles = dtpath.angleAeSD
        print('\tAngles [avg:{1:.0f}, sd:{2:.1f}]'.format(dtpath.length, angles[0], angles[1]))
        print('\tvisited:' + str(dtpath.visitedNum) + ', missed:' + str(dtpath.missedNum) + ', passedby:' + str(dtpath.passedbyNum))
        # disp.display_datepath(dtpath, index=tot, title=title, show=True, block=False)
    tot += 1

# block displayed pics
disp.block()
