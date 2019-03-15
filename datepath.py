import pandas as pd
import numpy as np
import random
import math
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
    def RandomMemory(fruits, orig=None, date=mc.DEFAULT_DATE, monkey=None, knownperc=1, max_mem_range=mc.MAX_MEM_DIST, planning_steps=mc.PLANNING_STEPS, sortmethod=mc.SortScore):
        """ generate a random "memory model" datepath

        Arguments:
            fruits {List[Coordinates]} -- List of fruit tree positions

        Keyword Arguments:
            orig {Coordinates} -- first point of path to be created  {Default: None}
            date {Datetime} -- date of path  {Default: mc.DEFAULT_DATE=date.today()}
            monkey {int} -- identifier of monkey. If None internal counter is used.  {Default: None}
            knownperc {float [0,1]} -- percentage of fruit trees known. {Default: 1}
            max_mem_range {float} -- maximum bird's-eye distance from fruit tree to next fruit tree in path (default: {2000})
            planning_steps {int} -- number of fruit trees considered for shortest path {Default: mc.PLANNING_STEPS}
            sortmethod {function_handle} -- method to sort the known trees

        Returns:
            {Datepath} -- computed random memory datepath
        """
        trees = random.sample(fruits, int(len(fruits) * knownperc))[:]
        path = sim.createMemoryDate(trees, orig=orig, max_mem_range=max_mem_range, planning_steps=planning_steps, sortmethod=sortmethod)
        dtp = datepath(path, date=date, monkey=monkey, label=mc.cls.MEMORY, fruits=fruits)
        return dtp

    @staticmethod
    def RandomView(fruits, orig=None, date=mc.DEFAULT_DATE, monkey=None, max_range=mc.VIEW_MAX_RANGE):
        """ generate a random "view model" datepath

        Arguments:
            fruits {List[Coordinates]} -- List of fruit tree positions

        Keyword Arguments:
            orig {Coordinates} -- first point of path to be created  {Default: None}
            date {Datetime} -- date of path  {Default: mc.DEFAULT_DATE=date.today()}
            monkey {int} -- identifier of monkey. If None internal counter is used.  {Default: None}
            max_range {float} -- maximum visibility. Actual visibility is a probabiltiy squared with the distance

        Returns:
            {Datepath} -- computed random view datepath
        """
        path = sim.createViewDate(fruits, orig=orig, max_range=max_range)
        dtp = datepath(path, date=date, monkey=monkey, label=mc.cls.VIEW, fruits=fruits)
        return dtp


# --------------------------------------------------------#
# ---------CONSTRUCTORS-----------------------------------#
#
    @staticmethod
    def FromCsv(csvpath, fruits, has_header=True, date=None, monkey=None, delimiter=',', quotechar='"', start=None, end=None):
        """ Create datepath from CSV

            Arguments:
                csvpath {string} -- path to csv
                fruits {List[Tree]} -- fruit trees

            Keyword Arguments:
                has_header {boolean} -- whether the csv file has header
                date {datetime.date} -- date of path. If None infer from filename, else mc.DEFAULT_DATE.  {Default: None}
                monkey {int} -- id of monkey. If None set next available id.  {Default: None}
                delimiter {char} -- character  used to separate fields.  {Default ','}
                quotechar {char} -- character used to quote fields containing special characters, such as the delimiter \
                                    or quotechar, or which contain new-line characters.  {Default: '"'}
                start {datetime.time} -- initial time to consider. Ignore points before. If None begin from first available point. {Default: None}
                end {datetime.time} -- end time to consider. Ignore points after. If None end at last available point. {Default: None}

            Returns:
                dtp {datepath} -- generated datepath
        """
        path = []
        # get id and date from filename
        if date is None or monkey is None:
            filename = csvpath.split('\\')[-1]   # get filename with extensione
            filename = filename.split('.')[0]    # remove exstension
            fileparts = filename.split('_')
            if monkey is None:
                try:
                    monkey = int(fileparts[0])
                except:
                    monkey = None
            if date is None:
                try:
                    date = datetime.strptime(fileparts[1], '%Y%m%d').date()
                except:
                    date = None
        with open(csvpath, 'rt') as csvfile:
            # read csv file
            csvreader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
            if has_header:
                next(csvreader)
            for row in csvreader:
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
                dtm = datetime.strptime(row[3], '%H:%M:%S.%f')
                # add file only if within given timeframe
                if (start is None or dtm.time() > start) and \
                   (end is None or dtm.time() < end):
                    path.append(geo.Coordinates(x,y,z,dtm))

        dtp = datepath(path, date=date, monkey=monkey, fruits=fruits)
        return dtp


    def __init__(self, path, date=None, monkey=None, label=mc.cls.TBD, fruits=None):
        """ constructor

        Arguments:
            path {List[Coordinates]} -- list of coordinates of the path

        Keyword Arguments:
            date {date} -- date of path. If None use mc.DEFAULT_DATE.  {Default: None}
            monkey {int} -- identifier of monkey. If None internal counter is used.  {Default: None}
            label {mc.cls} -- label of path.  {Default: mc.cls.TBD}
            fruits {List[Coordinates]} -- list of fruit tree.  {Default: None}

        """
        # internal
        self._path = path
        self._label = label
        if date is None:
            self._date = mc.DEFAULT_DATE
        else:
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
        self._speedAeSD = None
        self._angleAeSD = None



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

    # length
    @property
    def length(self):
        return len(self._path)
    

    # filename
    @property
    def filename(self):
        return '{:04}'.format(self._monkey) + '_' + self._date.strftime('%Y%m%d')

    # speed distribution
    @property
    def speedAeSD(self):
        if self._speedAeSD is None:
            self._computeSpeedsAngles()
        return self._speedAeSD

    # angle distribution
    @property
    def angleAeSD(self):
        if self._angleAeSD is None:
            self._computeSpeedsAngles()
        return self._angleAeSD

    # first
    @property
    def first(self):
        return self._path[0]

    # last
    @property
    def last(self):
        return self._path[-1]
    
    

    # visitedtrees
    @property
    def visitedIndex(self):
        """sets and returns visited trees 
            visitedTrees = dict(indexfruit, indexpath) -- where indexfruit is index of fruit in self._fruits and 
                      indexpath is the index in self._path of the first time the monkey reaches the fruit tree
        """
        if not self._visitedTrees:
            self._getMeaningfulTrees()
        return self._visitedTrees
    
    # missedtrees visible
    @property
    def missedIndex(self):
        """sets and returns missed trees AKA trees close by which are visible 
            missedTrees = dict(indexfruit, mindist) -- where indexfruit is index of fruit in self._fruits and 
                                mindist is the minimum distance from the path when the tree is visible
        """
        if not self._missedTrees:
            self._getMeaningfulTrees()
        return self._missedTrees
    
    # passedby visible
    @property
    def passedbyIndex(self):
        """sets and returns passedby trees AKA trees which were visited only briefly (less than minvisittimes) 
            visitedTrees =dict(indexfruit, indexpath) -- where indexfruit is index of fruit in self._fruits and 
                      indexpath is the index in self._path of the first time the monkey reaches the fruit tree
        """
        if not self._passedbyTrees:
            self._getMeaningfulTrees()
        return self._passedbyTrees

    # visitedtrees
    @property
    def visitedTrees(self):
        """sets and returns visited trees 
            visitedTrees {List[Trees]} -- list of fruit trees visited
        """
        if not self._visitedTrees:
            self._getMeaningfulTrees()
        return [self._fruits[i] for i in self._visitedTrees.keys()]
    
    # missedtrees visible
    @property
    def missedTrees(self):
        """sets and returns missed trees AKA trees close by which are visible 
            missedTrees {List[Trees]} -- list of fruit trees missed
        """
        if not self._missedTrees:
            self._getMeaningfulTrees()
        return [self._fruits[i] for i in self._missedTrees.keys()]
    
    # passedby visible
    @property
    def passedbyTrees(self):
        """sets and returns passedby trees AKA trees which were visited only briefly (less than minvisittimes) 
            visitedTrees {List[Trees]} -- list of fruit trees passed by
        """
        if not self._passedbyTrees:
            self._getMeaningfulTrees()
        return [self._fruits[i] for i in self._passedbyTrees.keys()]

    #.............FEATURES PROPERTY......................#
    # visitedNum
    @property
    def visitedNum(self):
        return len(self.visitedIndex)
    
    # missedNum
    @property
    def missedNum(self):
        return len(self.missedIndex)

    # passedbyNum
    @property
    def passedbyNum(self):
        return len(self.passedbyIndex)

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
        fullpath = folderpath + filename.split('.')[0] + '.csv'              # Forceg filetype to .csv
        # Write data file
        df = self.toDataframe()
        df['ts'] = df['ts'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])
        df.to_csv(fullpath, na_rep=None, float_format='%.1f', header=True, index=False)

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
        
        # print("before computation - visited:" + str(self._visitedTrees))

        # loop on points        
        for idx, point in enumerate(self._path):
            fdists = point.distance(self._fruits)
            minf, idf = min((val, idf) for (idf, val) in enumerate(fdists))
            # case tree reached
            if minf <= self._fruits[idf].radius * mc.VICINITY_FACTOR:
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
              and idf not in self._missedTrees: \
              # and (idf not in self._missedTrees or self._path[self._missedTrees[idf]].distance(self._fruits[idf]) > minf): \
              # and point.isVisible(self._fruits[idf], self.Island, next=self._path[idx+1]):
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
            self._missedTrees[key] = self._passedbyTrees[key]

        # print("after - visited:" + str(self._visitedTrees))
        # print("\t\tmissed:" + str(self._missedTrees))
        # print("\t\tpassedby:" + str(self._passedbyTrees))

    def _getsubdistsAeSDeF(self):
        """ sets and returns average, sd and fano factor (burstiness) of distances of subpaths

            Return:
                {(float, float, float)} -- average ([0]) standard deviation ([1]) and fano factor([2]) of subpaths' length
        """
        # sort visited in visiting order
        visited = list(self.visitedIndex.values())
        visited.sort()
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
        dist = 0
        for i in range(start, len(self._path)-1):
            dist += self._path[i].distance(self._path[i+1])
            distances[counter] = dist
        # compute average and standard deviation
        mean = np.mean(distances, axis=0)
        std = np.std(distances, axis=0)
        self._subdistsAvg = mean
        self._subdistsSD = std
        self._subdistsFano = std * std / mean         # fano factor
        return (self._subdistsAvg, self._subdistsSD, self._subdistsFano)

    def _getvisitedAeSD(self):
        """ sets and returns average and standard deviation of distances between visited trees

            Return:
                {(float, float)} -- average ([0]) and standard deviation ([1]) of distances between visited trees
        """
        # get visited trees from indices
        visited = self.visitedTrees
        ntocompare = len(visited)
        if ntocompare <= 1:
            self._visitedAvg = 0
            self._visitedSD = 0
        elif ntocompare == 2:
            self._visitedAvg = visited[0].distance(visited[1])
            self._visitedSD = 0
        else:
            combinations = mc.comb(ntocompare,2)
            distances = np.empty(combinations)
            counter = 0
            # loop on all visited trees
            for idf in range(ntocompare):
                distances[counter:counter+ntocompare-1] = visited[idf].distance(visited[idf+1:])
                ntocompare -= 1
                counter += ntocompare
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
        visited = self.visitedTrees
        ntocompare = len(self._fruits)
        combinations = mc.comb(ntocompare,2)
        distances = np.empty(combinations)
        counter = 0
        # loop on all trees trees
        for (idf, fruit) in enumerate(self._fruits):
            if ntocompare == 0:
                break
            distances[counter:counter+ntocompare-1] = fruit.distance(self._fruits[idf+1:])
            ntocompare -= 1
            counter += ntocompare
        # compute average and standard deviation
        fruitAvg = np.mean(distances, axis=0)
        fruitsSD = np.std(distances, axis=0)

        # compute Z test
        self._fruitsZtest = abs(self.visitedAvg - fruitAvg) / math.sqrt(fruitsSD * fruitsSD + self.visitedVar)
        return (self._fruitsZtest)

    def _computeSpeedsAngles(self):
        angles = np.zeros(len(self._path)-2)
        speeds = np.zeros(len(self._path)-2)
        for i in range(self.length-2):
            current = self._path[i]
            first = self._path[i+1]
            second = self._path[i+2]
            dt = first.time - current.time
            try:
                speeds[i] = current.distance(first) / dt.total_seconds()
            except Exception as e:
                print('WRONG round ' + str(i) + ' : current:' + str(current) + ', first:' + str(first) + ', prev:' + str(self._path[i-1]))
                raise e
            sincos = current.angle(second, first, ignore_z=True)
            angles[i] = math.atan2(sincos[0], sincos[1]) * 180 / math.pi
        # set instance variables
        self._speedAeSD = (np.mean(speeds, axis=0), np.std(speeds, axis=0))
        self._angleAeSD = (np.mean(angles, axis=0), np.std(angles, axis=0))
