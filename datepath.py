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

# ---------PRIVATE VARIABLES-----------------------------------#
    _label = mc.cls.TBD          # classification label
    _date = datetime.today()     # date
    _monkey = -1              # id of owning monkey
    _path = []      # list of coordinates
    _subpaths = []  # list of tuples [start, view, end] where each element is the index of the corresponding datapoint in _path


# ---------CONSTRUCTORS-----------------------------------#
    def __init__(self, path):
        self._path = path
        self._label = mc.cls.TBD

    @staticmethod
    def FullInit(id, date, path):
        dp = datepath(path)
        dp.set_date(date)
        dp.set_monkey_id(id)
        return dp

    @staticmethod
    def TimelessInit(path, startTime=mc.INIT_TIME, dt=mc.DT):
        delta = timedelta(seconds=dt)
        time = datetime.combine(datetime.today(), startTime)
        # add default timing
        for p in path:
            p.set_time(time.time())
            time = time + delta
        return datepath(path)



# ---------SETTERS-----------------------------------#
    def set_date(self, d):
        self._date = d

    def set_monkey_id(self, monkey):
        self._monkey = monkey


# ---------GETTERS-----------------------------------#
    def date(self):
        return self._date

    def monkey_id(self):
        return self._monkey

    def label(self):
        return self._label

    def path(self):
        return self._path

    def toDataframe(self):
        # TODO: adapt to new format
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
        # TODO: adapt to new format
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



# --------ANALYSIS----------------------------------------------#
    def getSubpaths(self, fruits, force_recompute=False):
        """ computes subpaths from current path, given the provided list of fruit trees

        Arguments:
            fruits {List[Coordinates]} -- list of fruit trees

        Keyword Arguments:
            force_recompute {boolean} -- force recompute of subpath.  {Default: False}
        """
        # check if value already computed
        if not force_recompute and self._subpaths:
            return
        # Value initialization
        self._subpaths = [[0, None, None]]   # reset subpaths
        frt_visited = [[0, -1]]           # list of fruits visited by the path [path_index, fruit_index]

        # loop on the whole path and identify fruit trees. Ignore same fruit tree repeated sequentially
        for idx in range(1, len(self._path)):
            min = self._path[idx].minDistance(fruits)
            if min[0] < mc.FRUIT_RADIUS and min[1] != frt_visited[-1][1]:
                frt_visited.add([idx, min[1]])

        # loop on each step, distinguish hanging and moving and compute viewpoints
        frt_idx = 1
        hanging = True
        for idx in range(frt_visited[1][0] + 1, len(self._path)):
            # case hanging (check if finished)
            if hanging and self._path[idx].distance(fruits[frt_visited[frt_idx][1]]) < mc.FRT_HANG_RAD:
                hanging = False
                frt_idx = frt_idx + 1
                self._subpaths.add([idx, None, frt_visited[frt_idx][0]])
            # case not hanging and arrived to destination
            elif not hanging and idx == frt_visited[frt_idx][0]:
                hanging = True
            # case not hanging and viewpoint not found
            elif not hanging and self._subpaths[-1][1] is None:
                if self._path[idx].isVisible(self._path[frt_visited[frt_idx][0]], next=self._path[idx + 1]):
                    self._subpaths[-1][1] = idx



# ---------FEATURE EXTRACTION-----------------------------------#
    def strRatioList(self):
        # TODO: adapt to new format
        """Return the list of straight ratio for each path of the date
        """
        ratiolist = []
        for p in self._path:
            path = [x[0] for x in p[0] + p[1]]
            ratiolist.append(geo.straighRatio(path))
        return ratiolist

    def strRatioAeSD(self):
        # TODO: adapt to new format
        """Returns average and standard deviation of straightness ratio of the subpath of the movement of the day

        Return:
            {(float, float)} -- average ([0]) and standard deviation ([1]) of straightness ratio
        """
        arr = np.array(self.strRatioList())
        return (np.mean(arr, axis=0), np.std(arr, axis=0))
