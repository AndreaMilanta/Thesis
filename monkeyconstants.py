"""Contains constants common to the project
"""

import os
from enum import Enum
from datetime import time


class cls(Enum):
    """classification enum

    TBD
    MEMORY
    VIEW
    """
    TBD = 0
    MEMORY = 1
    VIEW = 2


# PATH
HERE = os.path.dirname(__file__)
FRUIT_PATH = HERE + "\Data\FruitTrees.csv"
ISLAND_PATH = HERE + "\Data\IslandPic.png"
MONKEY_PATH = HERE + "\Data\AllChibi.csv"
SIM_PATH = HERE + "\Data\Simulated\\"
MEMORY_PATH = SIM_PATH + "Memory\\"
VIEW_PATH = SIM_PATH + "View\\"

# COMMON
FRUIT_RADIUS = 15       # Radius of fruit tree
REDUCTION_RADIUS = 5        # radius of cilinder for reducing paths
DATE_DURATION_MIN = 1 * 60  # Duration of day in minutes
MAX_MEM_DIST = 500          # Maximum distance of next fruit tree in memory model (as the crow flies)

# LINE OF SIGHT
HEIGHT_MARGIN = 1       # margin on LOS height obstruction
VIEW_MAX_RANGE = 200    # maximum distance a monkey can see
VIEW_MIN_RANGE = 20     # minimum distance a monkey can see
FOV = 360               # Field of View of a monkey

# ---------------------------------------------------------#
# -------------------------SIMULATION----------------------#

# TIME
DT = 120             # delta t between points (in seconds)
INIT_TIME = time(hour=8, minute=0, second=0)  # initial time of data acquisition -- 08:00:00 am

# SPEEDS (m/s)
DRT_VEL_EV = 1      # Expected Value of velocity for direct case
DRT_VEL_SD = 0.1    # Standard Deviation of velocity for direct case
RDM_VEL_EV = 0.5    # Expected Value of velocity for random case
RDM_VEL_SD = 0.2    # Standard Deviation of velocity for random case
HNG_VEL_EV = 0.01   # Expected Value of velocity during hanging
HNG_VEL_SD = 0.2   # Standard Deviation of velocity during hanging

# ANGLES (deg)
DRT_ANG_EV = 180    # Expected Value of angles for direct case
DRT_ANG_SD = 20     # Standard Deviation of angles for direct case
RDM_ANG_SD = 180    # Standard Deviation of angles for random case

PLANNING_STEPS = 50  # Number of next steps for which shortest path is computed
MIN_FRUIT_DIST = 50  # Minimum distance between two consecutive fruit tree on a path

# HANGING
FRT_HANG_MINTIME = 10        # minimum number of minutes the monkey hangs at a fruit tree
FRT_HANG_MAXTIME = 30        # maximum of minutes the monkey hangs at a fruit tree
FRT_HANG_RAD = FRUIT_RADIUS  # radius of hanging zone for the monkey

# WATER AVOIDANCE
WATER_SHIFT = 15        # Shift in direction from the standard to try to get around the water
MAX_WATER_TRIES = 5     # Number of tries in one direction when avoiding water


# ----------------------------------------------------------#
# ---------------------PERFORMANCE--------------------------#

# SIMULATION
MAX_ITERATIONS = 10000  # MAximum number of iteration to try and complete a path before giving up
DEF_LOAD_SIZE = 10  # default amount of speed and angle data to compute at once
