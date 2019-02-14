import pandas as pd
import numpy as np
import random
from collections import namedtuple
import csv
from datetime import date, datetime, time, timedelta

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
    Island = None

# ---------PRIVATE VARIABLES-----------------------------------#
    # _label     # classification label
    # _date      # date
    # _monkey    # id of owning monkey
    # _path      # list of coordinates
    # _fruits    # default fruits list to use
    # _subpaths  # list of tuples [start, view, end] where each element is the index of the corresponding datapoint in _path
    # #           hang subpaths are identified by second element (view) being None


# --------------------------------------------------------#
# ---------RANDOM GENERATORS------------------------------#
#
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

    @staticmethod
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

    @staticmethod
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


# --------------------------------------------------------#
# ---------CONSTRUCTORS-----------------------------------#
#
    @staticmethod
    def FromCsv(csvpath, fruits, date=mc.DEFAULT_DATE, monkey=None, delimiter=',', quotechar='"'):
        path = []
        with open(csvpath, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
            dtm = datetime.strptime(csvreader[0][2], '%Y-%m-%d %H:%M:%S.%f')
            date = dtm.date
            for row in csvreader:
                x = row[12]
                y = row[13]
                z = row[14]
                dtm = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S.%f')
                path.append(x,y,z,dtm.time)

        dtp = datepath(path, date=date, monkey=monkey, fruits=fruits)
        return dtp


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
        # internal
        self._path = path
        self._label = label
        self._date = date     # date
        if monkey is None:
            self._monkey = datepath._ID
            datepath._ID += 1
        else:
            self._monkey = monkey
        self._fruits = fruits
        self._visitedTrees = dict()
        self._passedbyTrees = dict()
        self._missedTrees = dict()

        # features
        self._subdistsAvg = None
        self._subdistsSD = None
        self._subdistsFano = None
        self._visitedAvg = None
        self._visitedSD = None
        self._fruitsZtest = None


# --------------------------------------------------------#
# ---------PROPERTIES-------------------------------------#
#
    # Date
    @property
    def date(self):
        return self._date
    @date.setter
    def date(self, value):
        self._date = value

    # Monkey
    @property
    def monkey(self):
        return self._monkey
    @monkey.setter
    def monkey(self, value):
        self._monkey = value

    # ID
    @property
    def id(self):
        """ returns unique id made of monkyeyymmdd
            e.g. monkey 11 of 10/12/2018 => 1120181210
        """
        return self._date.day + self._date.month * 100 + self._date.year * 10000 + self._monkey * 100000000

    # Label
    @property
    def label(self):
        return self._label

    # path
    @property
    def path(self):
        return self._path

    # filename
    @property
    def filename(self):
        return '{:02}'.format(self._monkey) + '_' + self._date.strftime('%Y%m%d')

    # visitedtrees
    @property
    def visitedTrees(self, misseddist=mc.MISSED_MAX_DIST, minvisittimes=mc.MIN_VISIT_TIMES):
        """sets and returns visited trees 
            visitedTrees = dict(indexfruit, indexpath) -- where indexfruit is index of fruit in self._fruits and 
                      indexpath is the index in self._path of the first time the monkey reaches the fruit tree

        Keyword Arguments:
            misseddist {float} -- distance from tree within which a tree is considered missed if not visited
            minvisittimes {int} -- minimum number of visits to a tree for the tree to be considered visited and not passedby
        """
        if not self._visitedTrees:
            self._getMeaningfulTrees(misseddist, minvisittimes)
        return self._visitedTrees
    
    # missedtrees visible
    @property
    def missedTrees(self, misseddist=mc.MISSED_MAX_DIST, minvisittimes=mc.MIN_VISIT_TIMES):
        """sets and returns missed trees AKA trees close by which are visible 
            missedTrees = dict(indexfruit, mindist) -- where indexfruit is index of fruit in self._fruits and 
                                mindist is the minimum distance from the path when the tree is visible

        Keyword Arguments:
            misseddist {float} -- distance from tree within which a tree is considered missed if not visited
            minvisittimes {int} -- minimum number of visits to a tree for the tree to be considered visited and not passedby
        """
        if not self._missedTrees:
            self._getMeaningfulTrees(misseddist, minvisittimes)
        return self._missedTrees
    
    # passedby visible
    @property
    def passedbyTrees(self, misseddist=mc.MISSED_MAX_DIST, minvisittimes=mc.MIN_VISIT_TIMES):
        """sets and returns passedby trees AKA trees which were visited only briefly (less than minvisittimes) 
            visitedTrees =dict(indexfruit, indexpath) -- where indexfruit is index of fruit in self._fruits and 
                      indexpath is the index in self._path of the first time the monkey reaches the fruit tree

        Keyword Arguments:
            misseddist {float} -- distance from tree within which a tree is considered missed if not visited
            minvisittimes {int} -- minimum number of visits to a tree for the tree to be considered visited and not passedby
        """
        if not self._passedbyTrees:
            self._getMeaningfulTrees(misseddist, minvisittimes)
        return self._passedbyTrees

    #.............FEATURES PROPERTY......................#
    # visitedNum
    @property
    def visitedNum(self):
        return len(self.visitedTrees)
    
    # missedNum
    @property
    def missedNum(self):
        return len(self.missedTrees)

    # passedbyNum
    @property
    def passedbyNum(self):
        return len(self.passedbyTrees)

    # subdistsAvg
    @property
    def subdistsAvg(self):
        if self._subdistsAvg is None:
            self._getsubdistsAeSDeF()
        return self._subdistsAvg
    
    # subdistsSD
    @property
    def subdistsSD(self):
        if self._subdistsSD is None:
            self._getsubdistsAeSDeF()
        return self._subdistsSD

    # subdistsFano
    @property
    def subdistsFano(self):
        if self._subdistsFano is None:
            self._getsubdistsAeSDeF()
        return self._subdistsFano

    # visitedAvg
    @property
    def visitedAvg(self):
        if self._visitedAvg is None:
            self._getvisitedAeSD()
        return self._visitedAvg

    # visitedSD
    @property
    def visitedSD(self):
        if self._visitedSD is None:
            self._getvisitedAeSD()
        return self._visitedSD

    # visitedVar
    @property
    def visitedVar(self):
        if self._visitedSD is None:
            self._getvisitedAeSD()
        return self._visitedSD * self._visitedSD

    # visited
    @property
    def fruitsZtest(self):
        if self._fruitsZtest is None:
            self._getfruitsZtest()
        return self._fruitsZtest
    

# --------------------------------------------------------#
# ---------SETTERS and GETTERS----------------------------#
#
    def clear_label(self):
        self._label = mc.cls.TBD;

    def set_fruits(self, fruits):
        self._fruits = fruits


# --------------------------------------------------------#
# --------------REPRESENTATION----------------------------#
#
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
            filename = self.filename
        fullpath = folderpath + filename.split('.')[0] + '.csv'              # Force filetype to .csv
        fpath_info = folderpath + filename.split('.')[0] + '_info.csv'
        # Write info file
        header_info = ['start', 'viewpoint', 'end']
        df_info = pd.DataFrame(self.subpaths(), columns=header_info)
        df_info.to_csv(fpath_info, na_rep=None, header=True, index=False)
        # Write data file
        df = self.toDataframe()
        df.to_csv(fullpath, na_rep=None, float_format='%.3f', header=True, index=False)

    def getDataframeRow(self):
        """Returns row of dataframe describing datepth

        Return:
            {dict} -- row of header as dictionary
        """
        # init dictionary
        row = dict()
        # set features
        row[mc.ID] = self.filename
        row[mc.LENGTH] = len(self.path)
        row[mc.VIS_NUM] = self.visitedNum
        row[mc.MISS_NUM] = self.missedNum
        row[mc.PBY_NUM] = self.passedbyNum
        row[mc.SUBDIST_AVG] = self.subdistsAvg
        row[mc.SUBDIST_SD] = self.subdistsSD
        row[mc.SUBDIST_FANO] = self.subdistsFano
        row[mc.VISDIST_AVG] = self.visitedAvg
        row[mc.VISDIST_SD] = self.visitedSD
        row[mc.FRUIT_ZTEST] = self.fruitsZtest
        # target class        
        row[mc.CLASS] = int(self.label)
        return row

    def toDataframe(self, int_h=True):
        """returns a dataframe for the whole day

        Keyword Arguments:
            int_h {boolean} -- whether h is returned as original value (False) or nearest integer (True).  {Default: True}

        Returns:
            dataframe -- coordinates (x,y,h) and timestamp (ts); no date
        """
        header = ['x', 'y', 'h', 'ts']
        if int_h:
            return pd.DataFrame(list(map(lambda x: x.xyizt, self._path)), columns=header)
        else:
            return pd.DataFrame(list(map(lambda x: x.xyzt, self._path)), columns=header)


# --------------------------------------------------------#
# --------------SUPPORT FEATURE EXTRACTION----------------#
#
    def _getMeaningfulTrees(self, misseddist=mc.MISSED_MAX_DIST, minvisittimes=mc.MIN_VISIT_TIMES):
        """ computes visitedTrees, missedTrees, missedVisibleTrees and passbyTrees

            Keyword Arguments:
                misseddist {float} -- distance from tree within which a tree is considered missed if not visited
                minvisittimes {int} -- minimum number of visits to a tree for the tree to be considered visited and not passedby
        """ 
        # reset dicts
        self._visitedTrees = dict()
        self._passedbyTrees = dict()
        self._missedTrees = dict()
        
        print("before computation - visited:" + str(self._visitedTrees))

        # loop on points        
        for idx, point in enumerate(self._path):
            fdists = point.distance(self._fruits)
            minf, idf = min((val, idf) for (idf, val) in enumerate(fdists))
            # case tree reached
            if minf <= mc.FRUIT_RADIUS:
                # add point to fruit ref (create one if first time fruittree is met)
                if idf in self._visitedTrees:
                    self._visitedTrees[idf].append(idx)
                else:
                    self._visitedTrees[idf] = [idx];
                # remove fruit from missed
                if idf in self._missedTrees:
                    del self._missedTrees[idf]

            # case tree visible and missed (close but not yet visited)
            elif minf < misseddist and idx<len(self._path)-1 \
              and idf not in self._visitedTrees \
              and (idf not in self._missedTrees or self._path[self._missedTrees[idf]].distance(self._fruits[idf]) > minf) \
              and point.isVisible(self._fruits[idf], self.Island, next=self._path[idx+1]):
                self._missedTrees[idf] = idx

        # distinguish between visits and passby
        bypasskeys = []
        for fruit in self._visitedTrees:
            visitlen = len(self._visitedTrees[fruit])   # get number of visits
            self._visitedTrees[fruit] = min(self._visitedTrees[fruit])  # maintain only first contact
            # case passby: add to _passedbyTrees
            if visitlen < minvisittimes:
                self._passedbyTrees[fruit] = self._visitedTrees[fruit]
        # remove bypassed from visited
        for key in self._passedbyTrees.keys():
            del self._visitedTrees[key]

        print("after - visited:" + str(self._visitedTrees))
        print("\t\tmissed:" + str(self._missedTrees))
        print("\t\tpassedby:" + str(self._passedbyTrees))

    def _getsubdistsAeSDeF(self):
        """ sets and returns average, sd and fano factor (burstiness) of distances of subpaths

            Return:
                {(float, float, float)} -- average ([0]) standard deviation ([1]) and fano factor([2]) of subpaths' length
        """
        # sort visited in visiting order
        visited = list(self.visitedTrees.values()).sort()
        # init vars
        distances = np.empty(len(visited) + 1);
        counter = 0
        start = 0
        # loop on all visited trees
        for vis in visited:
            dist = 0
            for i in range(start, vis):
                dist += self._path[i].distance(self._path[i+1])
            distances[counter] = dist
            start = vis
            counter += 1
        # add distance from last visited tree to end
        for i in range(start, len(self._path)-1):
            dist += self._path[i].distance(self._path[i+1])
            distances[counter] = dist
        # compute average and standard deviation
        self._subdistsAvg = np.mean(distances, axis=0)
        self._subdistsSD = np.std(distances, axis=0)
        self._subdistsFano = std * std / mean         # fano factor
        return (self._subdistsAvg, self._subdistsSD, self._subdistsFano)

    def _getvisitedAeSD(self):
        """ sets and returns average and standard deviation of distances between visited trees

            Return:
                {(float, float)} -- average ([0]) and standard deviation ([1]) of distances between visited trees
        """
        # get visited trees from indices
        visited = [self._fruits[i] for i in self.visitedTrees.keys()]
        ntocompare = len(visited)
        distances = np.empty(mc.comb(ntocompare,2))
        # loop on all visited trees
        for idf in range(len(visited)-1):
            distances[counter:counter+ntocompare] = visited[idf].distance(visited[idf+1:])
            counter += ntocompare
            ntocompare -= 1
        # compute average and standard deviation
        self._visitedAvg = np.mean(distances, axis=0)
        self._visitedSD = np.std(distances, axis=0)
        return (self._visitedAvg, self._visitedSD)

    def _getfruitsZtest(self):
        """ sets and returns the result of the Ztest between the fruits distribution and the visited distribution

            Return:
                {float} -- result of the Z test
        """
        # computes fruit distribution
        ntocompare = len(self._fruits)
        distances = np.empty(mc.comb(ntocompare,2))
        # loop on all trees trees
        for (idf, fruit) in enumerate(self._fruits):
            if ntocompare == 0:
                break
            distances[counter:counter+ntocompare] = fruit.distance(self._fruits[idf+1:])
            counter += ntocompare
            ntocompare -= 1
        # compute average and standard deviation
        fruitAvg = np.mean(distances, axis=0)
        fruisSD = np.std(distances, axis=0)

        # compute Z test
        self._fruitsZtest = abs(self.visitedAvg - fruitAvg) / math.sqrt(fruitSD * fruitSD + self.visitedVar)
        return (self._fruitsZtest)
