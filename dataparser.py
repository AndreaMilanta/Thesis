"""Data Parsing Library

contains a set of functions specifically written to read and properly parse all the dataset required
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import geometry as geo

# PATH CONSTANTS
HERE = os.path.dirname(__file__)
FRUIT_PATH = HERE + "\Data\FruitTrees.csv"
ISLAND_PATH = HERE + "\Data\IslandPic.png"
MONKEY_PATH = HERE + "\Data\AllChibi.csv"


class MovementData:
    """Represents and manages movement data

    Variables:
        _moves -- array of tuples with info (id), coordinates (x,y,h) and timestamp (YYYYMMDD,h,m,s)
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
            array -- array of tuples with info (id), coordinates (x,y,h) and timestamp (date,ts)
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
        return (x, y)


def parsefruittree():
    """Fruit Tree Reader (x,y,h)

    Read fruittree dataset and shifts it so that it maps on the island coordinates

    Returns:
        dataframe -- each row containes xy coordinates and height of a fruit tree
    """
    DELTA_X = -50
    DELTA_Y = -45
    columns = ['x', 'y', 'h']
    df = pd.read_csv(FRUIT_PATH, sep=' ', header=None, names=columns)
    df.reset_index()
    df.x = df.x.apply(lambda x: x + DELTA_X)
    df.y = df.y.apply(lambda x: x + DELTA_Y)
    return df


def parseisland():
    """Island Image Reader

    Reads black and white image of the island
    Each pixel represents the height of the ground there (0-255 ft)
    Resolution is 1 pixel per square meter

    Returns:
        array -- matrix with height as elements
    """
    island_img = plt.imread(ISLAND_PATH)
    island_img = np.flipud(island_img)
    return island_img


def getmonkey():
    """Creates movement class

    Reads AllChibi file and returns it in a MovementData class

    Returns:
        MovementData -- Class containing tha data for the monkey movements
    """
    return MovementData(MONKEY_PATH)


# MAIN
# fruits = parsefruittree()
# island = parseisland()
# correct = fruits[fruits.apply(lambda f: island[int(f.y), int(f.x)] > 0, axis=1)]
# incorrect = fruits[fruits.apply(lambda f: island[int(f.y), int(f.x)] <= 0, axis=1)]
# moves = getmonkey()
# for m in moves.monkeys():
#     print(m)
#     print(moves.dates(m).size)
#     for d in moves.dates(m):
#         print(moves.points(m, d))
#         print(str(d))

# VISUALIZATION
# plt.imshow(island, origin='lower')
# plt.scatter(fruits.x, fruits.y, s=2, c="#00FF00")
# plt.scatter(correct.x, correct.y, s=2, c="#00FF00")
# plt.scatter(incorrect.x, incorrect.y, s=2, c="#FF0000")
# plt.show()
