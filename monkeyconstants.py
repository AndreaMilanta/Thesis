"""Contains constants common to the project
"""

import os
from enum import Enum
from datetime import time
from datetime import date
from math import factorial
import numpy as np

import scipy as scp

def comb(n, k):
    return int(factorial(n) / (factorial(k) * factorial(n - k)))

class cls(Enum):
    """classification enum

    TBD
    MEMORY
    VIEW
    """
    TBD = 0
    MEMORY = 1
    VIEW = 2

    def __int__(self):
        return self.value

    def __str__(self):
        if self.value == MEMORY:
            return "Memory"
        elif self.value == VIEW:
            return "View"
        else:
            return "Undefined"

# PATH
HERE = os.path.dirname(__file__)
DATA = HERE + "\\Data\\"
FRUIT_PATH = DATA + "FruitTrees.csv"
ISLAND_PATH = DATA + "IslandPic.png"
MONKEY_PATH = DATA + "AllChibi.csv"
SIM_PATH = DATA + "Simulated\\"
MEMORY_PATH = SIM_PATH + "Memory\\"
VIEW_PATH = SIM_PATH + "View\\"
REAL_PATH = DATA + "\\Real\\"

# COMMON
FRUIT_RADIUS = 20       # Radius of fruit tree
REDUCTION_RADIUS = 5        # radius of cilinder for reducing paths
DATE_DURATION_MIN = 3 * 60  # Duration of day in minutes
MISSED_MAX_DIST = 5 * FRUIT_RADIUS # distance within which a tree is considered missed 
MIN_VISIT_TIMES = 2         # number of times the monkey must be in the tree to be considered a visit
VICINITY_FACTOR = 2         # radius multiplier to consider a point within the tree
DEFAULT_DATE = date.today()

MAX_MEM_DIST = 1000          # Maximum distance of next fruit tree in memory model (as the crow flies)
PLANNING_STEPS = 10  # Number of next steps for which shortest path is computed
MIN_FRUIT_DIST = 50  # Minimum distance between two consecutive fruit tree on a path

DO_ITERATIVELY = False

# LINE OF SIGHT
HEIGHT_MARGIN = 1       # margin on LOS height obstruction
VIEW_MAX_RANGE = 120    # maximum distance a monkey can see
VIEW_MIN_RANGE = 20     # minimum distance a monkey can see
FOV = 180               # Field of View of a monkey
# ---------------------------------------------------------#
# -------------------------SIMULATION----------------------#

# TIME
DT = 60             # delta t between points (in seconds)
INIT_TIME = time(hour=8, minute=0, second=0)  # initial time of data acquisition -- 08:00:00 am

#--------------------
# SPEEDS (m/s)
REAL_VEL_EV = 0.726    # REAL speed Expected value
REAL_VEL_SD = 1.311    # Real Speed Standard Deviation

DRT_VEL_EV = REAL_VEL_EV     # Expected Value of velocity for direct case (m/s)
DRT_VEL_SD = REAL_VEL_SD    # Standard Deviation of velocity for direct case (m/s)
RDM_VEL_EV = REAL_VEL_EV    # Expected Value of velocity for random case (m/s)
RDM_VEL_SD = REAL_VEL_SD    # Standard Deviation of velocity for random case (m/s)
HNG_VEL_EV = 0.1    # Expected Value of velocity during hanging (m/s)
HNG_VEL_SD = 0.01   # Standard Deviation of velocity during hanging (m/s)

DEF_VEL_EV = REAL_VEL_EV    # Default Expected Value of velocity (m/s)
DEF_VEL_SD = REAL_VEL_SD    # Default Standard Deviation of velocity (m/s)

RDM_DIST = 400      # Distance between random points to reach
RDM_SPEED = RDM_DIST / DT  # Defaut speed to compute next random point to reach

# ANGLES (deg)
DRT_ANG_SD = 54    # next target max angle for direct case (deg)
RDM_ANG_SD = 54    # Standard Deviation of angles for random case (deg)
RDM_ANGLE = 54     # max angle either way for next target

DEF_ANG_EV = 115    # Default Expected Value of angles (deg)
DEF_ANG_SD = 54     # Default Standard Deviation of angles (deg)
#-------------------------

# HANGING
FRT_HANG_MINTIME = 10       # minimum number of minutes the monkey hangs at a fruit tree
FRT_HANG_MAXTIME = 30        # maximum of minutes the monkey hangs at a fruit tree
FRT_HANG_RAD = FRUIT_RADIUS * 1.5  # radius of hanging zone for the monkey

# WATER AVOIDANCE
WATER_SHIFT = 15        # Shift in direction from the standard to try to get around the water
MAX_WATER_TRIES = 5     # Number of tries in one direction when avoiding water


# ----------------------------------------------------------#
# ---------------------PERFORMANCE--------------------------#

# SIMULATION
MAX_TRIES = 10  # Maximum number of failed tries due to exceptions before giving up
MAX_ITERATIONS = 10000  # Maximum number of iteration to try and complete a path before giving up
DEF_LOAD_SIZE = 10  # default amount of speed and angle data to compute at once


# ----------------------------------------------------------#
# ---------------------HEADERS------------------------------#

DFNAME = "main" + str(int(DATE_DURATION_MIN/60))+"h.csv"
DFPATH = DATA + DFNAME

# Dataframe Dictionare
ID = 'id'
LENGTH = 'Length'
CLASS = 'y'
VIS_NUM = 'FruitVisited'
MISS_NUM = 'FruitMissed'
PBY_NUM = 'FruitPassedby'
SUBDIST_AVG = 'SubdistancesAvg'
SUBDIST_SD = 'SubdistancesSD'
SUBDIST_FANO = 'SubdistancesFano'
VISDIST_AVG = 'VisitedDistancesAvg'
VISDIST_SD = 'VisitedDistancesSD'
FRUIT_ZTEST = 'FruitZtest'

HEADER = [ID, LENGTH, VIS_NUM, MISS_NUM, PBY_NUM, SUBDIST_AVG, SUBDIST_SD, SUBDIST_FANO, VISDIST_AVG, VISDIST_SD, FRUIT_ZTEST, CLASS]


# ----------------------------------------------------------#
# ---------------------FUNCTIONS----------------------------#


def Random(point, trees):
    """ return random tree to go to from list

        Arguments:
            point {Coordinates} -- point where the decision is being taken. Ignored, present for compatibility
            trees {List[Tree]} -- list of available trees

        Returns:
            target {Tree} -- Tree to go to
    """
    if len(trees) == 1:
        return trees[0]
    randidx = np.random.randint(len(trees))
    return trees[randidx]


def RadiusDistRatio(point, trees):
    """ return best tree to go to from list based on radius/distance ratio.
        Best tree is the one with minimum radius/distance ratio

        Arguments:
            point {Coordinates} -- point where the decision is being taken.
            trees {List[Tree]} -- list of available trees

        Returns:
            target {Tree} -- Tree to go to
    """
    if len(trees) == 1:
        return trees[0]
    return min(trees, key=lambda x: x.radius/point.distance(x))


def AreaDistRatio(point, trees):
    """ return best tree to go to from list based on area/distance ratio
        Best tree is the one with minimum radius*radius/distance ratio

        Arguments:
            point {Coordinates} -- point where the decision is being taken.
            trees {List[Tree]} -- list of available trees

        Returns:
            target {Tree} -- Tree to go to
    """
    if len(trees) == 1:
        return trees[0]
    return min(trees, key=lambda x: x.radius*x.radius/point.distance(x))


def _sortshortest(point, trees, method, maxsorts=None, reverse=False):
    """sorts according to a given method, select best and then computes shortest path between them

        Arguments:
            point {Coordinates} -- point where the decision is being taken.
            trees {List[Tree]} -- list of available trees
            method {function_handle(Coordinats, Tree)} -- function to sort the trees 

        Keyword Arguments:
            maxsorts {int} -- sorting steps to execute. Order the best maxsort trees.
            reverse {boolean} -- False for Ascending order, True for descending. {Default:False}

        Returns:
            target {List[Tree]} -- Sorted list of tree to go to
    """
    if maxsorts is None:
        maxsorts = float('inf')

    # select best targets
    trees.sort(key=lambda x: method(point, x), reverse=reverse)
    if maxsorts < len(trees):
        sorted = trees[0:maxsorts]
    else:
        sorted = trees
    return _sortiterative(point, sorted, method=lambda p,x: p.distance(x))



def _sortiterative(point, trees, method, maxsorts=None, reverse=False):
    """sorts according to a given method. Iteratively chooses the best starting from the previous

        Arguments:
            point {Coordinates} -- point where the decision is being taken.
            trees {List[Tree]} -- list of available trees
            method {function_handle(Coordinats, Tree)} -- function to sort the trees 

        Keyword Arguments:
            maxsorts {int} -- sorting steps to execute. Order the best maxsort trees.
            reverse {boolean} -- False for Ascending order, True for descending. {Default: False}

        Returns:
            target {List[Tree]} -- Sorted list of tree to go to
    """
    if maxsorts is None:
        maxsorts = float('inf')

    sorted = []
    old = trees
    s = point
    count = 0
    while old and count < maxsorts:
        old.sort(key=lambda x: method(s, x), reverse=reverse)
        sorted.append(old[0])
        s = old.pop(0)
        count += 1
    return sorted


def SortScore(point, trees, maxsorts=None, iterative=DO_ITERATIVELY):
    """ sorts trees according to score

        Arguments:
            point {Coordinates} -- point where the decision is being taken. Ignored, needed for consistency
            trees {List[Tree]} -- list of available trees

        Keyword Arguments:
            maxsort {int} -- sorting steps to execute. Order the best maxsort trees.
            iterative {boolean} -- compute best iteratively. {Default: True}

        Returns:
            target {List[Tree]} -- Sorted list of tree to go to
    """
    if iterative:
        return _sortiterative(point, trees, method=lambda p,t: t.score, maxsorts=maxsorts, reverse=True)
    else:
        return _sortshortest(point, trees, method=lambda p,t: t.score, maxsorts=maxsorts, reverse=True)


def SortDistScoresq(point, trees, maxsorts=None, iterative=DO_ITERATIVELY):
    """ sorts trees according to squared score over distance

        Arguments:
            point {Coordinates} -- point where the decision is being taken.
            trees {List[Tree]} -- list of available trees

        Keyword Arguments:
            maxsorts {int} -- sorting steps to execute. Order the best maxsort trees.
            iterative {boolean} -- compute best iteratively. {Default: True}

        Returns:
            target {List[Tree]} -- Sorted list of tree to go to
    """
    if iterative:
        return _sortiterative(point, trees, method=lambda p,t: t.score**2/t.distance(p), maxsorts=maxsorts, reverse=True)
    else:
        print("in SortDistScoresq")
        return _sortshortest(point, trees, method=lambda p,t: t.score**2/t.distance(p), maxsorts=maxsorts, reverse=True)


def SortDistScore(point, trees, maxsorts=None, iterative=DO_ITERATIVELY):
    """ sorts trees according to score over distance

        Arguments:
            point {Coordinates} -- point where the decision is being taken.
            trees {List[Tree]} -- list of available trees

        Keyword Arguments:
            maxsorts {int} -- sorting steps to execute. Order the best maxsort trees. {Default:Inf}
            iterative {boolean} -- compute best iteratively. {Default: True}

        Returns:
            target {List[Tree]} -- Sorted list of tree to go to
    """
    if iterative:
        return _sortiterative(point, trees, method=lambda p,t: t.score/t.distance(p), maxsorts=maxsorts, reverse=True)
    else:
        return _sortshortest(point, trees, method=lambda p,t: t.score/t.distance(p), maxsorts=maxsorts, reverse=True)


def SortDistsqScore(point, trees, maxsorts=None, iterative=DO_ITERATIVELY):
    """ sorts trees according to score over squared distance

        Arguments:
            point {Coordinates} -- point where the decision is being taken.
            trees {List[Tree]} -- list of available trees

        Keyword Arguments:
            maxsorts {int} -- sorting steps to execute. Order the best maxsort trees.
            iterative {boolean} -- compute best iteratively. {Default: True}

        Returns:
            target {List[Tree]} -- Sorted list of tree to go to
    """
    if iterative:
        return _sortiterative(point, trees, method=lambda p,t: t.score/(t.distance(p)**2), maxsorts=maxsorts, reverse=True)
    else:
        return _sortshortest(point, trees, method=lambda p,t: t.score/(t.distance(p)**2), maxsorts=maxsorts, reverse=True)


def SortDistance(point, trees, maxsorts=None, iterative=DO_ITERATIVELY):
    """sorts according to distance

        Arguments:
            point {Coordinates} -- point where the decision is being taken.
            trees {List[Tree]} -- list of available trees

        Keyword Arguments:
            maxsorts {int} -- sorting steps to execute. Order the best maxsort trees.
            iterative {boolean} -- compute best iteratively. {Default: True}

        Returns:
            target {List[Tree]} -- Sorted list of tree to go to
    """
    if iterative:
        return _sortiterative(point, trees, method=lambda p,t: t.distance(p), maxsorts=maxsorts, reverse=False)
    else:
        return _sortshortest(point, trees, method=lambda p,t: t.distance(p), maxsorts=maxsorts, reverse=False)

