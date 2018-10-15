""" Test script """

import dataparser as dp
import datepath as dtp
import display as disp
import time

NUM_TRIES = 2

disp.display_island(index=0, show=False, fruits=dp.Fruits(), fruitsize=1)
for i in range(0, NUM_TRIES):
    print("working on trial " + str(i))
    st = time.clock()
    path = dtp.datepath.RandomMemory(dp.Fruits(), monkey=i)
    end = time.clock()
    print("displaying on trial " + str(i) + ", computation required " + str(end-st) + "s")
    disp.display(path.path(), show=False)
disp.show()
