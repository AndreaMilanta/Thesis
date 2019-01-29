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

NUM_TRIES = 10000
STEPSIZE = 50
CLF = int(255 / max([1, NUM_TRIES-1]))
dtp.datepath.Island = dp.Island()
rows = []
disprequired = False
basemonkey = 0    # base monkey ID

steps = int(NUM_TRIES / STEPSIZE)

# RANDOM GENERATOR and DISPLAY
# disp.display_island(index=1, show=True, block=False)
# disp.display_fruits(dp.Fruits(), fruitsize=3, show=True, block=False)
for j in range(0,steps): 
    basemonkey = STEPSIZE * j
    for i in range(0, STEPSIZE):
        try:
            rdm = np.random.uniform(0,1)
            start = time.process_time()
            if rdm < 0.5:
                print("working on memory trial " + str(i+basemonkey))
                path = dtp.datepath.RandomMemory(dp.Fruits(), monkey=i+basemonkey)
                col = [i*CLF, 0, 0]
            else:
                print("working on view trial " + str(i+basemonkey))
                path = dtp.datepath.RandomView(dp.Fruits(), monkey=i+basemonkey)
                col = [0, 255-i*CLF, i*CLF]
            finish = time.process_time()
            row = path.getDataframeRow()
            if(row[mc.CLASS] == 0 or row[mc.STR_A] is None or row[mc.STR_A] == 0):
                print("Error on trial " + str(i) + " - FEATURES are 0 - computation required " + str(finish-start) + "s")
                print("\nPATH\n")
                print(path.path)
                print("\n")
                # disp.display(path.path, show=True, color=col, block=False)
            else:
                rows.append(row)
                # path.savecsv();
        except Exception as e:
            finish = time.process_time()
            print("\n")
            print("Error on trial " + str(i) + " - COMPUTATION ERROR - computation required " + str(finish-start) + "s")
            print(path.getDataframeRow())
            print(e)
            tb_str = "".join(tb.format_tb(e.__traceback__))
            print("".join(tb_str))
            # disp.display(path.path, show=True, color=col, block=False)

    # create and save dataframe
    df = pd.DataFrame(rows, columns=mc.HEADER)
    df.set_index(mc.ID)
    with open(mc.DFPATH, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0, na_rep=None, index=False, float_format="%2.2f")
    print(df.describe())
    rows = [];

quit();



# MEMORY GENERATOR, DISPLAY and SAVE
disp.display_island(index=1, show=False)
for i in range(0, NUM_TRIES):
    print("working on memory trial " + str(i))
    try:
        path = dtp.datepath.RandomMemory(dp.Fruits(), monkey=i)
        # path.date = date(2018,11,28)
        rows.append(path.getDataframeRow())
        path.savecsv()
        print("saved memory path " + str(i))
        col = [i*CLF, 0, 0]
        # disp.display(path.path, show=False, color=col)
    except(me.PathOnWaterException):
        print("!!XX memory path " + str(i) + " failed (on water)")
# disp.display_fruits(dp.Fruits(), fruitsize=3)


