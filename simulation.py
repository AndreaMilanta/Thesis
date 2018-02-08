"""Simulation Script

Creates the simulated dataset

Variables:
    island {array} -- island with height info for each coordinate
"""
import numpy as np
from enum import Enum
import random
import math

import dataparser as dp
import geometry as geo

DT = 120             # delta t between points (in seconds)

# All velocities are in m/s and angles are in deg

DRT_VEL_EV = 1      # Expected Value of velocity for direct case
DRT_VEL_SD = 0.1    # Standard Deviation of velocity for direct case
DRT_ANG_EV = 180    # Expected Value of angles for direct case
DRT_ANG_SD = 0     # Standard Deviation of angles for direct case
RDM_VEL_EV = 0.5    # Expected Value of velocity for random case
RDM_VEL_SD = 0.2    # Standard Deviation of velocity for random case
RDM_ANG_SD = 180    # Standard Deviation of angles for random case

WATER_SHIFT = 15        # Shift in direction from the standard to try to get around the water
MAX_WATER_TRIES = 5     # Number of tries in one direction when avoiding water


class Distr(Enum):
    """Distributions enum

        UNIFORM
        NORMAL
        CLUSTERED
    """
    UNIFORM = 0
    NORMAL = 1
    CLUSTERED = 2


island = dp.parseisland()


def _checkFruitTree(p):
    """Checks whether the given tree is valid or not
    """
    if p.x < 0:
        return False
    elif p.y < 0:
        return False
    elif p.x > island.shape[0] - 1:
        return False
    elif p.y > island.shape[1] - 1:
        return False
    elif p.z == 0:
        return False
    else:
        return True


def _checkMinDist(p, centers, min_dist):
    """checks if p is at least at min_dist from all the points in centers
    """
    for c in centers:
        if p.distance(c) < min_dist:
            return False
    return True


def _createCoor(xy):
    """Creates a Coordinates object from a set of points
    if the points are outside the island z=0
    """
    try:
        return geo.Coordinates(xy[0], xy[1], island[int(xy[0]), int(xy[1])])
    except IndexError:
        return geo.Coordinates(xy[0], xy[1], 0)


def buildFruitTree(density, distr, sd_factor=3, mean=None, num_clast=4, min_dist=2000):
    """Creates the fruit trees on the island

    Creates and distributes fruit tree according to parameters
    Arguments:
        density {float} -- density of fruit tree in trees/km^2
        distr {Distr} -- distribution of fruit trees on the island (UNIFORM, NORMAL, CLUSTERED)
            NORMAL: median on the center of the island or mean, standard deviation = mean/sd_factor
            CLUSTERED: creates num_clast clasters randomly (uniform) distributed over the island. each cluster has a normal distribution with sd = islandside/sd_factor*num_clast
        sd_factor {float} -- factor representing mean wrt standard deviation, used to compute sd {default=3}
        mean {[int, int]} -- mean of normal distribution {default=None}
        num_clast {int} -- number of cluster {default=4}
        min_dist {int} -- minimum distance between cluster centers {default=2000}


    Returns:
        List of Coordinates -- each row containes the coordinates of a fruit tree
    """
    fruits = []
    area = np.count_nonzero(island)
    size = int(area / 1000000 * density)     # number of trees to compute
    # Compute Trees
    if distr == Distr.UNIFORM:
        while len(fruits) < size:
            f = [random.randint(0, island.shape[0] - 1), random.randint(0, island.shape[1] - 1)]
            if island[f[0], f[1]] > 0:
                fruits.append(geo.Coordinates(f[0], f[1], island[f[0], f[1]]))
    elif distr == Distr.NORMAL:
        if mean is None or mean[0] < 0 or mean[1] < 0 or mean[0] > island.shape[0] - 1 or mean[1] > island.shape[1] - 1:
            mean = [island.shape[0] / 2, island.shape[1] / 2]
        cov = [[math.pow(mean[0] / sd_factor, 2), 0], [0, math.pow(mean[1] / sd_factor, 2)]]    # covariance matrix. Axis are independent
        fruits = np.random.multivariate_normal(mean, cov, size)
        fruits = fruits.astype(int)
        fruits = [_createCoor(x) for x in fruits]
        fruits = [x for x in fruits if _checkFruitTree(x)]
    elif distr == Distr.CLUSTERED:
        clst = []
        while len(clst) < num_clast:
            c = [random.randint(0, island.shape[0] - 1), random.randint(0, island.shape[1] - 1)]
            if island[c[0], c[1]] > 0:
                p = geo.Coordinates(c[0], c[1], island[c[0], c[1]])
                if not _checkMinDist(p, clst, min_dist):
                    continue
                clst.append(p)
        for c in clst:
            c_fruits = []
            mean = [c.x, c.y]
            cov = [[math.pow(island.shape[0] / (sd_factor * num_clast), 2), 0], [0, math.pow(island.shape[1] / (sd_factor * num_clast), 2)]]    # covariance matrix. Axis are independent
            c_fruits = np.random.multivariate_normal(mean, cov, int(size / num_clast))
            c_fruits = c_fruits.astype(int)
            c_fruits = [_createCoor(x) for x in c_fruits]
            c_fruits = [x for x in c_fruits if _checkFruitTree(x)]
            fruits.extend(c_fruits)
    return fruits


def _createP2RPath(orig, speed, angle, fruits, ignores):
    """Creates a random path up to seeing a fruit tree

    Arguments:
        orig {Coordinates} -- Origin point
        speed {[float, float]} -- speed average and standard deviation [avg, std]
        angle {float} -- standard deviation of angles wrt destination
        fruits {List[Coordinates]} -- List of fruit trees Coordinates
        ignores {List[Coordinates]} -- List of fruit trees to ignore: Already visited

    Returns:
        [List[Coordinates], Coordinate] -- List of points of the path and seen fruit tree
    """
    LOAD_SIZE = 0
    p = orig
    path = [orig]
    spds = np.array([])
    agls = np.array([])
    index = LOAD_SIZE
    angl_delta = WATER_SHIFT
    water_index = 0
    counter = 0
    valid_cnt = 0
    f = p.sees(fruits, p, ignores)
    while(f is None and counter < 10000):
        if index == LOAD_SIZE:
            LOAD_SIZE = int(np.random.normal(100, 50, 1))
            if(LOAD_SIZE < 10):
                LOAD_SIZE = 10
            agl = np.random.normal(0, 360, 1)
            spds = np.random.normal(speed[0], speed[1], LOAD_SIZE)
            agls = np.random.normal(agl, angle, LOAD_SIZE)
            index = 0
        delta = geo.unit()
        dist = DT * spds[index]
        delta.scale(dist / delta.mag())
        delta = delta.rotate(agls[index])
        nxt = p.add(delta)
        counter = counter + 1                                                        # DEBUG!!!
        try:
            nxt.z = geo.ISLAND[int(nxt.x), int(nxt.y)]
            if nxt.z == 0 or p.inWater(nxt):
                print("WARNING: direct path goes into water")
                if water_index > 180 / WATER_SHIFT:
                    angl_delta = -WATER_SHIFT
                    water_index = 0
                else:
                    angl_delta = angl_delta + WATER_SHIFT
                water_index = water_index + 1
                agls[index] = agls[index] + angl_delta
                continue
            path.append(nxt)
        except IndexError:
            print("ERROR: random path out of bound")
            continue
        water_index = 0
        index = index + 1
        path.append(nxt)
        f = nxt.sees(fruits, p, ignores)
        p = nxt
        valid_cnt = valid_cnt + 1
    # print("Random Counter " + str(valid_cnt))
    if f is None:
        print("No tree visible")
        # f = path[-1]
    return [path, f]


def _createP2PPath(orig, dest, speed, angle):
    """Creates a path from orig to dest

    Arguments:
        orig {Coordinates} -- Origin point
        dest {Coordinates} -- Destination point
        speed {[float, float]} -- speed average and standard deviation [avg, std]
        angle {float} -- standard deviation of angles wrt destination

    Returns:
        List[Coordinates] -- Listo fo points of the path
    """
    LOAD_SIZE = 100
    p = orig
    path = [orig]
    spds = np.array([])
    agls = np.array([])
    index = LOAD_SIZE
    angl_delta = WATER_SHIFT
    water_index = 0
    counter = 0                                                                     # DEBUG!!!!
    while(p.distance(dest) > DT * speed[0] and counter < 10000):                    # DEBUG!!!!
        if index == LOAD_SIZE:
            spds = np.random.normal(speed[0], speed[1], LOAD_SIZE)
            agls = np.random.normal(0, angle, LOAD_SIZE)
            index = 0
        delta = p.diff(dest)
        dist = DT * spds[index]
        delta.scale(dist / delta.mag())
        delta = delta.rotate(agls[index])
        nxt = p.add(delta)
        counter = counter + 1                                                        # DEBUG!!!
        try:
            nxt.z = geo.ISLAND[int(nxt.x), int(nxt.y)]
            if nxt.z == 0 or p.inWater(nxt):
                print("WARNING: direct path goes into water")
                # if water_index > 180 / WATER_SHIFT:
                #     angl_delta = -WATER_SHIFT
                #     water_index = 0
                # else:
                angl_delta = angl_delta + WATER_SHIFT
                water_index = water_index + 1
                agls[index] = agls[index] + angl_delta
                continue
            path.append(nxt)
        except IndexError:
            print("ERROR: direct path out of bound")
            continue
        water_index = 0
        p = nxt
        index = index + 1
    # print("Direct Counter " + str(counter))
    path.append(dest)
    return path


def _createDirect(orig, dest):
    """Created direct path from orig to dest
    """
    return _createP2PPath(orig, dest, [DRT_VEL_EV, DRT_VEL_SD], DRT_ANG_SD)


def _createRandom(orig, fruits, ignores):
    """Create random path from orig to first tree seen
    """
    return _createP2RPath(orig, [RDM_VEL_EV, RDM_VEL_SD], RDM_ANG_SD, fruits, ignores)


def createViewPath(orig, fruits, ignores):
    """Creates a view-model path

    Arguments:
        orig {Coordinates} -- starting point
        fruits {List[Coordinates]} -- set of fruit trees coordinates
        ignores {List[Coordinates]} -- set of fruit trees to ignore: already visited

    Returns:
        [List[Coordinates], List[Coordinates]] -- tuple containing the random path to the viewpoint and the direct path to the fruit tree
    """
    rdm = _createRandom(orig, fruits, ignores)
    if rdm[1] is None:
        return rdm
    path = rdm[0]
    dest = rdm[1]
    viewpoint = path[-1]
    direct = _createDirect(viewpoint, dest)
    return [path, direct]


def createViewDate(orig, fruits, totaltime):
    """Computes the path for a date according to the view model

    Arguments:
        orig {Coordinates} -- starting point
        fruits {List[Coordinates]} -- set of fruit trees
        totaltime {float} -- duration of day in minutes

    Returns:
        List[[List[Coordinates], List[Coordinates]]] -- List of tuples containing each path to the next fruit tree (tuple[1] is None for format consistency)
    """
    tot_steps = totaltime * 60 / DT  # Duration of day expressed as number of datapoints
    path = []
    curr_steps = 0
    start = orig
    ignores = [start]
    while curr_steps < tot_steps:
        curr_path = createViewPath(start, fruits, ignores)
        if curr_path[1] is None:
            continue
        start = curr_path[1][-1]
        ignores.append(start)
        curr_steps = curr_steps + len(curr_path)
        path.append(curr_path)
    print("Date split into " + str(len(path)) + " paths")
    return path


def createMemoryDate(orig, fruits, totaltime, max_mem_range=2000):
    """Computes the path for a date according to the memory model

    Arguments:
        orig {Coordinates} -- starting point
        fruits {List[Coordinates]} -- set of fruit trees
        totaltime {float} -- duration of day in minutes

    Keyword Arguments:
        max_mem_range {float} -- maximum bird's-eye distance from fruit tree to next fruit tree in path (default: {2000})

    Returns:
        List[[List[Coordinates], None]] -- List of tuples containing each path to the next fruit tree (tuple[1] is None for format consistency)
    """
    tot_steps = totaltime * 60 / DT  # Duration of day expressed as number of datapoints
    path = []
    curr_steps = 0
    start = orig
    rdm_index = int(np.random.uniform(0, len(fruits), 1))
    ignores = [start]
    while curr_steps < tot_steps:
        while start.distance(fruits[rdm_index]) > max_mem_range or fruits[rdm_index] in ignores:
            rdm_index = int(np.random.uniform(0, len(fruits), 1))
        curr_path = _createDirect(start, fruits[rdm_index])
        start = curr_path[-1]
        ignores.append(start)
        curr_steps = curr_steps + len(curr_path)
        path.append([curr_path, None])
    print("Date split into " + str(len(path)) + " paths")
    return path
