import pandas as pd
import numpy as np
import random
from collections import namedtuple
import csv

import monkeyexceptions as me
import monkeyconstants as mc
import simulation as sim
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

# STATIC VARIABLES
    _ID = 0

# ---------PRIVATE VARIABLES-----------------------------------#
    # _label     # classification label
    # _date      # date
    # _monkey    # id of owning monkey
    # _path      # list of coordinates
    # _fruits    # default fruits list to use
    # _subpaths  # list of tuples [start, view, end] where each element is the index of the corresponding datapoint in _path
    # #           hang subpaths are identified by second element (view) being None


# ---------RANDOM GENERATORS------------------------------#
    @staticmethod
    def Random(fruits, orig=None, date=mc.DEFAULT_DATE, monkey=None):
        """ generate a random datepath

        Arguments:
            fruits {List[Coordinates]} -- List of fruit tree positions

        Keyword Arguments:
            orig {Coordinates} -- first point of path to be created  {Default: None}
            date {Datetime} -- date of path  {Default: mc.DEFAULT_DATE=date.today()}
            monkey {int} -- identifier of monkey. If None internal counter is used.  {Default: None}

        Returns:
            {Datepath} -- computed random datepath
        """
        if random.getrandbits(1) == 0:
            return datepath.RandomMemory(fruits, orig=orig, date=date, monkey=monkey)
        else:
            return datepath.RandomView(fruits, orig=orig, date=date, monkey=monkey)

    def RandomMemory(fruits, orig=None, date=mc.DEFAULT_DATE, monkey=None):
        """ generate a random "memory model" datepath

        Arguments:
            fruits {List[Coordinates]} -- List of fruit tree positions

        Keyword Arguments:
            orig {Coordinates} -- first point of path to be created  {Default: None}
            date {Datetime} -- date of path  {Default: mc.DEFAULT_DATE=date.today()}
            monkey {int} -- identifier of monkey. If None internal counter is used.  {Default: None}

        Returns:
            {Datepath} -- computed random memory datepath
        """
        path = sim.createMemoryDate(fruits, orig=orig)
        dtp = datepath(path, date=date, monkey=monkey, label=mc.cls.MEMORY, fruits=fruits)
        return dtp


    def RandomView(fruits, orig=None, date=mc.DEFAULT_DATE, monkey=None):
        """ generate a random "view model" datepath

        Arguments:
            fruits {List[Coordinates]} -- List of fruit tree positions

        Keyword Arguments:
            orig {Coordinates} -- first point of path to be created  {Default: None}
            date {Datetime} -- date of path  {Default: mc.DEFAULT_DATE=date.today()}
            monkey {int} -- identifier of monkey. If None internal counter is used.  {Default: None}

        Returns:
            {Datepath} -- computed random view datepath
        """
        path = sim.createViewDate(fruits, orig=orig)
        dtp = datepath(path, date=date, monkey=monkey, label=mc.cls.VIEW, fruits=fruits)
        return dtp


# ---------CONSTRUCTORS-----------------------------------#
    def __init__(self, path, date=mc.DEFAULT_DATE, monkey=None, label=mc.cls.TBD, fruits=None):
        """ constructor

        Arguments:
            path {List[Coordinates]} -- list of coordinates of the path

        Keyword Arguments:
            date {date} -- date of path.  {Default: mc.DEFAULT_DATE}
            monkey {int} -- identifier of monkey. If None internal counter is used.  {Default: None}
            label {mc.cls} -- label of path.  {Default: mc.cls.TBD}
            fruits {List[Coordinates]} -- list of fruit tree.  {Default: None}

        """
        self._path = path
        self._label = label
        self._date = date     # date
        if monkey is None:
            self.set_monkey(datepath._ID)
            datepath._ID += 1
        else:
            self.set_monkey(monkey)
        self._fruits = fruits
        self._subpaths = []


# ---------SETTERS-----------------------------------#
    def set_date(self, d):
        self._date = d

    def set_monkey(self, monkey):
        self._monkey = monkey

    def set_fruits(self, fruits):
        self._fruits = fruits


# ---------GETTERS-----------------------------------#
    def date(self):
        return self._date

    def monkey(self):
        return self._monkey

    def label(self):
        return self._label

    def path(self):
        return self._path

    def subpaths(self, fruits=None, force_recompute=False):
        """ returns list of subpath descriptors

        List of tuples [start, view, end] where each element is the index of the corresponding datapoint in path.
        Hang subpaths are identified by second element (view) being None
        """
        # compute/update subpaths
        if force_recompute or not self._subpaths:
            if fruits is None:
                raise me.OptionalParamIsNoneException()
            self._computeSubpaths(fruits)
        return self._subpaths


# --------------REPRESENTATION----------------------------------------#
    def savecsv(self, filename=None, folderpath=None):
        """ save path to file as csv

        Keywords Arguments:
            filename {string} -- name of output file. If None the file is saved as id_date.csv.  {Default: None}
            folderpath {string} -- path of file. If None default per classification is used  {Default: None}
                                   N.B. If _label is mc.cls.TBD a folderpath is required
        """
        # Filename preparation
        if folderpath is None:
            if self._label == mc.cls.MEMORY:
                folderpath = mc.MEMORY_PATH
            elif self._label == mc.cls.VIEW:
                folderpath = mc.VIEW_PATH
            else:
                raise me.OptionalParamIsNoneException()
        if filename is None:
            fname = str(self._monkey) + '_' + str(self._date.replace('-', ''))
        filename = folderpath + fname.split('.')[0] + '.csv'              # Force filetype to .csv
        fname_info = folderpath + fname.split('.')[0] + '_info.csv'
        # Write info file
        header_info = ['start', 'viewpoint', 'end']
        with open(fname_info, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header_info)
            writer.writerows(self.subpaths())
        # Write data file
        df = self.toDataframe()
        df.to_csv(filename, na_rep='None', float_format='%.3f', header=True, index=False)


    def fullsubpaths(self, fruits=None, sep_hang=False, force_recompute=False):
        """returns list of all subpath as namedtuple Subpath

        Subpath: points {List[Coordinates]}    # list of Coordinates of the subpath
                 view {int}                # index of viewpoint in subpath
                 hang {int}                # index of first hanging moment in subpath (if sep_hang this is always None)

        Keyword Arguments:
            fruits {list[coordinates]} -- list of fruit trees. If None default is used  {Default: None}
            sep_hang {boolean} -- whether hanging is returned as separate subpath or included. If True Subpath.hang = None.  {Default: False}
            force_recompute {boolean} -- force recompute of subpath.  {Default: False}

        Returns:
            subpaths {[Subpath]} -- list of subpaths
        """
        # values and type initialization
        Subpath = namedtuple('Subpath', ['points', 'view', 'hang'])
        subpaths = []
        idx = 0
        # compute/update subpaths
        if force_recompute or not self._subpaths:
            self._computeSubpaths(fruits)

        # case travel and hanging separate
        if sep_hang:
            for sbp in self._subpaths:
                points = self._path[sbp[0]:sbp[2]]
                view = sbp[1]
                subpaths.add(Subpath(points, view, None))    # if sep_hang subpath.hang is always None
        # case travel and hanging all together
        else:
            idx = 0
            # loop on pairs travel + hanging
            while idx < len(self._subpaths) - 1:
                points = self._path[self._subpaths[idx][0]:self._subpaths[idx + 1][2]]
                view = self._subpaths[idx][1]
                hang = self._subpaths[idx + 1][0]
                subpaths.add(Subpath(points, view, hang))
                idx = idx + 2
            # add last subpath if it is travel
            if len(self._subpaths) % 2 == 0:
                points = self._path[self._subpaths[-1][0]:self._subpaths[-1][2]]
                subpaths.add(Subpath(points, self._subpaths[-1][1], self._subpaths[-1][2]))
        # return list of subpaths
        return subpaths

    def toDataframe(self, int_h=True):
        """returns a dataframe for the whole day

        Keyword Arguments:
            int_h {boolean} -- whether h is returned as original value (False) or nearest integer (True).  {Default: True}

        Returns:
            dataframe -- coordinates (x,y,h) and timestamp (ts); no date
        """
        header = ['x', 'y', 'h', 'ts']
        if int_h:
            return pd.DataFrame(map(lambda x: x.xyizt, self._path), columns=header)
        else:
            return pd.DataFrame(map(lambda x: x.xyzt, self._path), columns=header)

    def toMultipleDf(self, int_h=True):
        """returns a dataframe for each fruit-fruit path

        Keyword Arguments:
            int_h {boolean} -- whether h is returned as original value (False) or nearest integer (True).  {Default: True}

        Returns:
            List[[dataframe, dataframe]] -- coordinates (x,y,h) and timestamp (ts); no date. Same standard structure
                                            if 'hanging' second dataframe is None
        """
        moves = []
        header = ['x', 'y', 'h', 'ts']
        for seg in self._subpaths:
            # case hanging
            if seg[1] is None:
                if int_h:
                    moves.append([pd.DataFrame(map(lambda x: x.xyizt, self._path[seg[0]: seg[2] - 1]), columns=header), None])
                else:
                    moves.append([pd.DataFrame(map(lambda x: x.xyzt, self._path[seg[0]: seg[2] - 1]), columns=header), None])
            # case moving to fruit tree
            else:
                if int_h:
                    moves.append([pd.DataFrame(map(lambda x: x.xyizt, self._path[seg[0]: seg[1] - 1]), columns=header),
                                  pd.DataFrame(map(lambda x: x.xyizt, self._path[seg[1]: seg[2]]), columns=header)])
                else:
                    moves.append([pd.DataFrame(map(lambda x: x.xyzt, self._path[seg[0]: seg[1] - 1]), columns=header),
                                  pd.DataFrame(map(lambda x: x.xyzt, self._path[seg[1]: seg[2]]), columns=header)])
        return moves

    def _computeSubpaths(self, fruits=None):
        """ computes subpaths from current path, given the provided list of fruit trees

        Keyword Arguments:
            fruits {list[coordinates]} -- list of fruit trees. If None default is used  {Default: None}
            force_recompute {boolean} -- force recompute of subpath.  {Default: False}
        """
        # Value initialization
        self._subpaths = [[0, None, None]]   # reset subpaths
        frt_visited = [[0, -1]]           # list of fruits visited by the path [path_index, fruit_index]
        # check which fruit list to use
        if fruits is None:
            fruits = self._fruits

        # loop on the whole path and identify fruit trees. Ignore same fruit tree repeated sequentially
        for idx in range(1, len(self._path)):
            min = self._path[idx].minDistance(fruits)
            if min[0] < mc.FRUIT_RADIUS and min[1] != frt_visited[-1][1]:
                frt_visited.add([idx, min[1]])

        # loop on each step, distinguish hanging and moving and compute viewpoints
        frt_idx = 1
        hanging = False
        for idx in range(frt_visited[1][0] + 1, len(self._path)):
            # case hanging (check if finished)
            if hanging and self._path[idx].distance(fruits[frt_visited[frt_idx][1]]) > mc.FRT_HANG_RAD:
                hanging = False
                frt_idx = frt_idx + 1
                self._subpaths[-1][2] = idx
            # case not hanging and viewpoint not found (looking for viewpoint)
            elif not hanging and self._subpaths[-1][1] is None:
                if self._path[idx].isVisible(self._path[frt_visited[frt_idx][0]], next=self._path[idx + 1]):
                    self._subpaths[-1][1] = idx
            # case not hanging (check if arrived to destination)
            elif not hanging and idx == frt_visited[frt_idx][0]:
                hanging = True
                self._subpaths.add([idx, None, None])
        # force last subpath to end on last point
        self._subpaths[-1][2] = len(self._path) - 1



# ---------FEATURE EXTRACTION-----------------------------------#
    def strRatioList(self, ignore_hang=True, sep_hang=False):
        """Return the straight ratio of the whole path and the list of straight ratio for each subpath of the date

        Keyword Arguments:
            ignore_hang {boolean} -- ignore hanging, consider only travel subpaths  {Default: True}
            sep_hang {boolean} -- consider hanging as indipendent subpaths (ignored if ignore_hang is True)  {Default: False}

        Returns:
            {(float, List[float])} -- overall straight ratio, straight ratio of each subpath
        """
        ratiolist = []

        # case hang ignored or separate
        if ignore_hang or sep_hang:
            for sbp in self._subpaths:
                if sbp[1] is None and ignore_hang:
                    continue
                ratiolist.append(geo.straighRatio(self._path[sbp[0]:sbp[2]]))
        # case hang considered but not separate
        else:
            idx = 0
            # loop on pairs travel+hanging
            while idx < len(self._subpaths) - 1:
                ratiolist.append(geo.straighRatio(self._path[self._subpaths[idx][0]:self._subpaths[idx + 1][2]]))
            # consider last travel if present
            if len(self._subpaths) % 2 == 0:
                ratiolist.append(geo.straighRatio(self._path[self._subpaths[-1][0]:self._subpaths[-1][2]]))
        # return overall ratio and list
        return (geo.straighRatio(self._path), ratiolist)



    def strRatioAeSD(self):
        """Returns average and standard deviation of straightness ratio of the subpath of the movement of the day

        Return:
            {(float, float)} -- average ([0]) and standard deviation ([1]) of straightness ratio
        """
        arr = np.array(self.strRatioList()[1])
        return (np.mean(arr, axis=0), np.std(arr, axis=0))
