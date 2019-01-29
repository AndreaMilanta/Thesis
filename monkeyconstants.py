"""Contains constants common to the project
"""

import os
from enum import Enum
from datetime import time
from datetime import date


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

# COMMON
FRUIT_RADIUS = 15       # Radius of fruit tree
REDUCTION_RADIUS = 5        # radius of cilinder for reducing paths
DATE_DURATION_MIN = 4 * 60  # Duration of day in minutes
MAX_MEM_DIST = 500          # Maximum distance of next fruit tree in memory model (as the crow flies)
DEFAULT_DATE = date.today()

# LINE OF SIGHT
HEIGHT_MARGIN = 1       # margin on LOS height obstruction
VIEW_MAX_RANGE = 200    # maximum distance a monkey can see
VIEW_MIN_RANGE = 20     # minimum distance a monkey can see
FOV = 360               # Field of View of a monkey

# ---------------------------------------------------------#
# -------------------------SIMULATION----------------------#

# TIME
DT = 8*60             # delta t between points (in seconds)
INIT_TIME = time(hour=8, minute=0, second=0)  # initial time of data acquisition -- 08:00:00 am

# SPEEDS (m/s)
DRT_VEL_EV = 0.5      # Expected Value of velocity for direct case (m/s)
DRT_VEL_SD = 0.05    # Standard Deviation of velocity for direct case (m/s)
RDM_VEL_EV = 0.3    # Expected Value of velocity for random case (m/s)
RDM_VEL_SD = 0.1    # Standard Deviation of velocity for random case (m/s)
HNG_VEL_EV = 0.02   # Expected Value of velocity during hanging (m/s)
HNG_VEL_SD = 0.0   # Standard Deviation of velocity during hanging (m/s)

# ANGLES (deg)
DRT_ANG_EV = 180    # Expected Value of angles for direct case
DRT_ANG_SD = 20     # Standard Deviation of angles for direct case
RDM_ANG_SD = 180    # Standard Deviation of angles for random case

PLANNING_STEPS = 50  # Number of next steps for which shortest path is computed
MIN_FRUIT_DIST = 50  # Minimum distance between two consecutive fruit tree on a path

# HANGING
FRT_HANG_MINTIME = 30       # minimum number of minutes the monkey hangs at a fruit tree
FRT_HANG_MAXTIME = 60        # maximum of minutes the monkey hangs at a fruit tree
FRT_HANG_RAD = FRUIT_RADIUS * 1.5  # radius of hanging zone for the monkey

# WATER AVOIDANCE
WATER_SHIFT = 15        # Shift in direction from the standard to try to get around the water
MAX_WATER_TRIES = 5     # Number of tries in one direction when avoiding water


# ----------------------------------------------------------#
# ---------------------PERFORMANCE--------------------------#

# SIMULATION
MAX_ITERATIONS = 100  # Maximum number of iteration to try and complete a path before giving up
DEF_LOAD_SIZE = 10  # default amount of speed and angle data to compute at once


# ----------------------------------------------------------#
# ---------------------HEADERS------------------------------#

DFNAME = "main" + str(int(DT/60))+"h.csv"
DFPATH = DATA + DFNAME

# Dataframe Dictionare
ID = 'id'
LENGTH = 'Length'
SUBNUM = 'Subnumber'
STR_A = 'StraightnessAvg'
STR_SD = 'StraightnessSD'
SPD_A = 'SpeedAvg'
SPD_SD = 'SpeedSD'
LEN_A = 'SublengthAvg'
LEN_SD = 'SublengthSD'
CLASS = 'y'

HEADER = [ID, LENGTH, SUBNUM, STR_A, STR_SD, \
          SPD_A, SPD_SD, LEN_A, LEN_SD, CLASS]
