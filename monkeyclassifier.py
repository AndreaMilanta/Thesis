"""Library purpose built to classify monkey paths
"""
import sklearn as skl
import pandas as pd
from datetime import datetime
import os

import dataparser as dp
import monkeyexceptions as me
import monkeyconstants as mc
import datepath as dtp
import display as disp
import geometry as geo

# Constants
STEPSIZE = 50
FILE_PATH = mc.REAL_PATH + "Dataframe_FullDay.csv"
START = 50 * 0  # Starting point, to recover previous rounds

START_TIME = datetime.strptime('00:00:00', '%H:%M:%S').time()
END_TIME = datetime.strptime('23:59:00', '%H:%M:%S').time()

# init static variables
dtp.datepath.Island = dp.Island()
geo.Coordinates.Island = dp.Island()

#Save dataframe with rows
def save2dataframe(records):
    df = pd.DataFrame(records, columns=mc.HEADER)
    df.set_index(mc.ID)
    with open(FILE_PATH, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0, na_rep=None, index=False, float_format="%2.2f")
    print('\nDataFrame successfully saved\n')


# init vars
counter = 0;
total = START;
rows = [];

# read files availabe
files = os.listdir(mc.REAL_PATH)
files = files[:-2]

# loop on files
for i in range(START, len(files)):
    # Read file
    f = files[i];
    print("working on file #{0} - {1}".format(total, f))
    splits = (f.split('.')[0]).split('_')
    monkey = int(splits[0])
    dtm = datetime.strptime(splits[1], '%Y%m%d')
    path = dtp.datepath.FromCsv(mc.REAL_PATH + f, dp.Fruits(), date=dtm.date(), monkey=monkey, delimiter=',', quotechar='"', start=START_TIME, end=END_TIME)
    # disp.display_datepath(path, fruit_dim='')
    # quit()
    # get dataframe row
    row = path.getDataframeRow()
    rows.append(row)
    counter += 1
    total += 1
    # create and save dataframe every STEPSIZE loops
    if counter >= STEPSIZE:
        save2dataframe(rows)
        rows = [];
        counter = 0;

# save any remaining rows
if counter > 0:
    save2dataframe(rows)


