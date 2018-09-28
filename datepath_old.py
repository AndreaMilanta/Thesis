import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime

import monkeyconstants as mc
import geometry as geo


class datepath:
    """ Class containing the path of one monkey in one day.

        Holds path described as list of fruit-fruit subpath.
        Each fruit-fruit subpath is a [List, List], where the
        first list is the path up to the viewpoint, the second
        one is from the viewpoint to the fruit tree.

        A set of functions allow to extract features from the path.
          - strRatioList(): list of straight ratio of each path.
          - strRatioAeSD(): average and standard deviation of straight ratio in the day.
    """
    # public variables
    label = mc.cls.TBD          # classification label
    date = datetime.today()     # date
    monkey_id = -1              # id of owning monkey

    # private variables
    _path = []      # list of coordinates
    _subpaths = []  # list of tuples [start, view, end] where each element is the index of the corresponding datapoint in _path


    def __init__(self, path):
        self._path = path
        self.cls = mc.cls.TBD

    @staticmethod
    def FullInit(id, date, path):
        dp = datepath(path)
        dp.set_date(date)
        dp.set_id(id)
        return dp

    @staticmethod
    def TimelessInit(paths, startTime=mc.INIT_TIME, dt=mc.DT):
        delta = timedelta(seconds=dt)
        time = datetime.combine(datetime.today(), startTime)
        print(str(len(paths)))
        for index, seg in enumerate(paths):
            # print(str(type(seg[0])))
            i = 0
            for p in seg[0]:
                seg[0][i] = [p, time.time()]
                time = time + delta
                i += 1
            i = 0
            # for i, p in seg[1]:
            for p in seg[1]:
                seg[1][i] = [p, time.time()]
                time = time + delta
                i += 1
        dp = datepath(paths)
        return dp

    def set_date(self, d):
        self.date = d

    def set_id(self, id):
        self.id = id

    def path(self):
        return self._path

    def strRatioList(self):
        """Return the list of straight ratio for each path of the date
        """
        ratiolist = []
        for p in self._path:
            path = [x[0] for x in p[0] + p[1]]
            ratiolist.append(geo.straighRatio(path))
        return ratiolist

    def toDataframe(self):
        """returns a dataframe for the whole day

        Returns:
            dataframe -- coordinates (x,y,h) and timestamp (ts); no date
        """
        moves = []
        for seg in self._path:
            for p in seg[0]:
                moves.append((p[0].x, p[0].y, int(p[0].z), p[1]))
            for p in seg[1]:
                moves.append((p[0].x, p[0].y, int(p[0].z), p[1]))
        header = ['x', 'y', 'h', 'ts']
        return pd.DataFrame(moves, columns=header)

    def toMultipleDf(self):
        """returns a dataframe for each fruit-fruit path

        Returns:
            List[[dataframe, dataframe]] -- coordinates (x,y,h) and timestamp (ts); no date. Same standard structure
        """
        moves = []
        header = ['x', 'y', 'h', 'ts']
        for seg in self._path:
            points_rdm = []
            points_drt = []
            for p in seg[0]:
                points_rdm.append((p[0].x, p[0].y, int(p[0].z), p[1]))
            for p in seg[1]:
                points_drt.append((p[0].x, p[0].y, int(p[0].z), p[1]))
            moves.append([pd.DataFrame(points_rdm, columns=header), pd.DataFrame(points_drt, columns=header)])
        return moves

    def strRatioAeSD(self):
        """Returns average and standard deviation of straightness ratio of the subpath of the movement of the day

        Return:
            {(float, float)} -- average ([0]) and standard deviation ([1]) of straightness ratio
        """
        arr = np.array(self.strRatioList())
        return (np.mean(arr, axis=0), np.std(arr, axis=0))
