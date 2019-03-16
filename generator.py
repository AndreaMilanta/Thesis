""" Test script """
import numpy as np
import pandas as pd
import time
from datetime import datetime
import traceback as tb

import dataparser as dp
import datepath as dtp
import display as disp
import monkeyexceptions as me
import monkeyconstants as mc
import geometry as geo

# CONSTANTS
NUM_TRIES = 1000
STEPSIZE = 50

# init static variables
dtp.datepath.Island = dp.Island()
geo.Coordinates.Island = dp.Island()
steps = int(NUM_TRIES / STEPSIZE)

# RANDOM GENERATOR and DISPLAY
rows = []
for j in range(0,steps): 
    basemonkey = STEPSIZE * j
    i = 0
    while i < STEPSIZE:
        try:
            rdm = np.random.uniform(0,1)

            rdm = 1

            start = time.process_time()
            if rdm < 0.5:
                print("working on memory trial " + str(i+basemonkey))
                path = dtp.datepath.RandomMemory(dp.Fruits(), monkey=i+basemonkey, sortmethod=mc.SortScore)
                title = '{0:02d} - Memory'.format(i+basemonkey)
            else:
                print("working on view trial " + str(i+basemonkey))
                path = dtp.datepath.RandomView(dp.Fruits(), monkey=i+basemonkey)
                title = '{0:02d} - Perception'.format(i+basemonkey)
            print("\tlength:" + str(path.length))
            finish = time.process_time()
            # disp.display_datepath(path, index=i+basemonkey, title=title, fruit_dim='', show=True, block=True)
            row = path.getDataframeRow()
            rows.append(row)
            i += 1
        except Exception as e:
            finish = time.process_time()
            print("\n")
            print("Error on trial " + str(i) + " - COMPUTATION ERROR - computation required " + str(finish-start) + "s")
            print(path.getDataframeRow())
            print(e)
            tb_str = "".join(tb.format_tb(e.__traceback__))
            print("".join(tb_str))

    # create and save dataframe
    df = pd.DataFrame(rows, columns=mc.HEADER)
    df.set_index(mc.ID)
    with open(mc.DATA+"20190316-141426.csv", 'a') as f:
        # append new dataframe. If first go add header
        # if j == 0:
        if j == -1:
            df.to_csv(f, mode='a', header=True, na_rep=None, index=False, float_format="%2.2f")
        else:
            df.to_csv(f, mode='a', header=False, na_rep=None, index=False, float_format="%2.2f")
    print(df.describe())
    rows = [];

    # save parameters if first round
    # if j == 0: mc.savecsvparams()

quit();
