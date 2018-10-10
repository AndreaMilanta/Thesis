""" Test script
"""

import dataparser as dp
import datepath as dtp
import display as disp

NUM_TRIES = 2

disp.display_island(index=0, show=False, fruits=dp.Fruits())
for i in range(0, NUM_TRIES):
    print("working on trial " + str(i))
    path = dtp.datepath.Random(dp.Fruits(), monkey=i)
    print("displaying on trial " + str(i))
    disp.display(path.path(), show=False)
disp.show()
