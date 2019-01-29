"""Data Parsing Library

contains a set of functions specifically written to read and properly parse all the dataset required
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import csv

import monkeyconstants as mc
import datepath as dp
import geometry as geo

# CONSTANTS
DELTA_X_FRT = -50   # shift on x axis for gps conversion of fruit tree data
DELTA_Y_FRT = -45   # shift on y axis for gps conversion off ruit tree data
DELTA_X_MON = -50   # shift on x axis for gps conversion of monkey movement data
DELTA_Y_MON = -45   # shift on y axis for gps conversion of monkey movement data

# global variables
_Island = None    # holds parsed Island
_Fruits = None    # holds parsed Fruits


# Getters
def Island():
    """ returns parsed island
    """
    global _Island
    if _Island is None:
        _Island = _parseisland()
    return _Island


def Fruits():
    """ returns parsed fruits
    """
    global _Fruits
    if _Fruits is None:
        _Fruits = _parsefruittree()
    return _Fruits


class MovementData:
    """Represents and manages movement data

    Variables:
        _moves -- dataframe with info (id), coordinates (x,y,h) and timestamp (YYYYMMDD,h,m,s)
    """

    def __init__(self, path):
        self._moves = self._parseMonkey(path)

    def monkeys(self):
        """gets monkey ids available

        Returns:
            array -- List of unique monkey ids
        """
        return self._moves.id.unique()

    def dates(self, monkey_id):
        """gets date available for given monkeys

        Arguments:
            monkey_id {int} -- id of monkey for which we want to retrieve dates

        Returns:
            list -- list of dates
        """
        return self._moves.loc[self._moves.id == monkey_id].date.unique()

    def points(self, monkey_id, date):
        """gets point of the given monkey on the given date
        Returns the points with timestamp of the given date for the given monkey

        Arguments:
            monkey_id {int} -- monkey
            date {datetime.date} -- date of interest

        Returns:
            dataframe -- points as (x,y,h,ts)
        """
        return self._moves.loc[(self._moves.id == monkey_id) & (self._moves.date == date)].loc[:, ['x', 'y', 'h', 'ts']]
        # return self._moves.loc[(self._moves.id == monkey_id) & (self._moves.date == date)].loc[:, ['coo', 'ts']]

    def _parseMonkey(self, path):
        """Reads Monkey Dataset
        Reads AllChibi file and returns it an a dataframe

        Returns:
            dataframe -- info (id), coordinates (x,y,h) and timestamp (date,ts)
        """
        fo = open(path, "r")
        lines = fo.readlines()
        fo.close()
        moves = []
        for line in lines:
            tokens = line.split(',')
            if(tokens[30] and tokens[31]):
                # id
                individual_id = int(tokens[27].replace("\"", ""))   # monkey id
                # location
                utme = float(tokens[30])    # latitude
                utmn = float(tokens[31])    # longitude
                ht = float(tokens[23])      # height
                xy = self._converttoXY(utme, utmn)
                # timestamp
                ts = tokens[34]             # local timestamp
                tokens2 = ts.split(" ")     # split date time
                datetokens = tokens2[0].split("-")  # split year-month-date
                timetokens = tokens2[1].split(":")  # split hour:minute:second
                date = datetime.date(int(datetokens[0]), int(datetokens[1]), int(datetokens[2]))
                hour = int(timetokens[0])
                minute = int(timetokens[1])
                sec = timetokens[2].split(".")
                sec = int(sec[0])
                time = datetime.time(hour, minute, sec)
                # moves.append((individual_id, geo.Coordinates(xy[0], xy[1], ht), date, time))
                moves.append((individual_id, xy[0], xy[1], ht, date, time))
        header = ['id', 'x', 'y', 'h', 'date', 'ts']
        # header = ['id', 'coo', 'date', 'ts']
        df = pd.DataFrame(moves, columns=header)
        df.reset_index()
        print(df.describe())
        return df

    def _converttoXY(self, utmE, utmN):
        """Compute xy coordinates from GPS

        Converts GPS coordinates to xy referenced to the origin of the island pic

        Arguments:
            utmE {float} -- Longitude
            utmN {float} -- Latitude
        """
        utmE1 = 624030.0137255481   # 0.18/(float(10260)/32064) * 10260
        utmN1 = 1015207.0834458455  # 0.18/(float(9850)/30780) * 9850
        utmE2 = 629801.5337255481   # 624030.0137255481 + 32064.0 * 0.18
        utmN2 = 1009666.6834458455  # 1015207.0834458455 - 30780.0 * 0.18
        totalW = 5771.52
        totalH = 5540.4
        x = (utmE - utmE1) / (utmE2 - utmE1) * totalW
        y = (utmN2 - utmN) / (utmN2 - utmN1) * totalH
        x = x + DELTA_X_MON
        y = y + DELTA_Y_MON
        return (x, y)

    def toStandard(self):
        """Returns standard notation of monkey path (list of points)
        """
        dates = []
        for m in self.monkeys().rows():
            for d in self.dates(m).rows():
                # datepath as simple list of  geo.Coordinates
                path = map(lambda p: geo.Coordinates(p.x, p.y, p.ht, time=p.ts), self.dates(m).rows)
                dates.append(dp.datepath(path, date=d, monkey=m, fruits=Fruits()))
                # # datepath as list of subpaths each split between to-view and from-view
                # single_date = []
                # single_path = []
                # for p in self.points(m, d).rows():
                #     single_path.append(geo.Coordinates(p.x, p.y, p.ht), p.ts)
                #     if single_path[-1][0].minDistance(fruits) < mc.FRUIT_RADIUS:
                #         single_date.append([None, single_path])
                #         single_path = []
                # dates.append(dp.DatePath.FullInit(m, d, single_date))
        return dates


def _parsefruittree():
    """Fruit Tree Reader (x,y)

    Read fruittree dataset and shifts it so that it maps on the island coordinates

    Returns:
        List of Coordinates -- each element is a coordinate object of a fruit tree
    """
    columns = ['x', 'y', 'extra']
    df = pd.read_csv(mc.FRUIT_PATH, sep=' ', header=None, names=columns)
    df.reset_index()
    df.x = df.x.apply(lambda x: x + DELTA_X_FRT)
    df.y = df.y.apply(lambda x: x + DELTA_Y_FRT)
    fts = df.values.tolist()
    return [geo.Coordinates(x[1], x[0], Island()[int(x[1]), int(x[0])]) for x in fts]


def _parseisland():
    """Island Image Reader

    Reads black and white image of the island
    Each pixel represents the height of the ground there (0-255 ft)
    Resolution is 1 pixel per square meter

    Returns:
        array -- matrix with height as elements
    """
    island_img = plt.imread(mc.ISLAND_PATH)
    island_img = np.flipud(island_img)
    island_img = island_img * 255
    return island_img


def getmonkey():
    """Creates movement class

    Reads AllChibi file and returns it in a MovementData class

    Returns:
        MovementData -- Class containing tha data for the monkey movements
    """
    return MovementData(mc.MONKEY_PATH)


def parse_date(filename):
    """get date and id from filename

    Returns:
        {int, datetime} - id and date
    """
    info_str = filename.split('.')[0].split('\\')[-1].split('_')
    id_val = int(info_str[0])
    date_str = info_str[1]
    date = datetime.datetime(year=int(date_str[0:4]), month=int(date_str[4:6]), day=int(date_str[6:8]))
    print('id: ' + str(id_val) + " date: " + str(date))
    return [id_val, date_str]



def parseSTDDate(file, infofile, id=None, date=None):
    """Parses file of movement of one day according to standard format

    Arguments:
        file {string} -- path of data file
        infofile {string} -- path of info file

    Keyword Arguments:
        id {int} -- id of monkey. if None, is retrieved from filename id_yyyymmdd.csv {Default: None}
        date {datetime} -- date of path. if None is retrieved from filename id_yyyymmdd.csv  {Defualt: None}

    Return:
        datepath -- instance of datepath class representing the movement of the monkey in that date
     """
    if date is None or id is None:
        info = parse_date(file)
        id = info[0]
        date = info[1]
    path = []
    currentPointer = 0
    with open(infofile, 'rt') as info:
        reader = csv.reader(info)
        infolist = list(map(tuple, reader))
        infolist = infolist[1:]                 # remove header row
    for i in infolist:
        st = int(i[0])
        if i[1] == 'None':
            vwp = st
        else:
            vwp = int(i[1])
        end = int(i[2])
        df = pd.read_csv(file, skiprows=currentPointer, nrows=(vwp - st))
        rdm = [tuple([geo.Coordinates(p[0], p[1], p[2]), p[3]]) for p in df.values]
        currentPointer = vwp
        df = pd.read_csv(file, skiprows=currentPointer, nrows=(end - vwp + 1))
        drt = [tuple([geo.Coordinates(p[0], p[1], p[2]), p[3]]) for p in df.values]
        currentPointer = end + 1
        path.append([rdm, drt])
    return dp.datepath(path)
