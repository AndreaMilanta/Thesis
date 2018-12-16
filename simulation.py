"""Simulation Script

Creates the simulated dataset.
Each monkey date is generated as a list of Coordinates

Variables:
    dp.Island() {array} -- dp.Island() with height info for each coordinate
"""
import pdb

import numpy as np
from enum import Enum
import random
import math
from datetime import date, datetime, time, timedelta

import monkeyconstants as mc
import dataparser as dp
import geometry as geo
import monkeyexceptions as me


# All velocities are in m/s and angles are in deg


class Distr(Enum):
    """Distributions enum

        UNIFORM
        NORMAL
        CLUSTERED
    """
    UNIFORM = 0
    NORMAL = 1
    CLUSTERED = 2


def _checkFruitTree(p):
    """Checks whether the given tree is valid or not
    """
    if p.x < 0:
        return False
    elif p.y < 0:
        return False
    elif p.x > dp.Island().shape[0] - 1:
        return False
    elif p.y > dp.Island().shape[1] - 1:
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
    if the points are outside the dp.Island() z=0
    """
    try:
        return geo.Coordinates(xy[0], xy[1], dp.Island()[int(xy[0]), int(xy[1])])
    except IndexError:
        return geo.Coordinates(xy[0], xy[1], 0)


def buildFruitTree(density, distr, sd_factor=3, mean=None, num_clast=4, min_dist=2000):
    """Creates the fruit trees on the dp.Island()

    Creates and distributes fruit tree according to parameters
    Arguments:
        density {float} -- density of fruit tree in trees/km^2
        distr {Distr} -- distribution of fruit trees on the dp.Island() (UNIFORM, NORMAL, CLUSTERED)
            NORMAL: median on the center of the dp.Island() or mean, standard deviation = mean/sd_factor
            CLUSTERED: creates num_clast clasters randomly (uniform) distributed over the dp.Island(). each cluster has a normal distribution with sd = dp.Island()side/sd_factor*num_clast
        sd_factor {float} -- factor representing mean wrt standard deviation, used to compute sd {default=3}
        mean {[int, int]} -- mean of normal distribution {default=None}
        num_clast {int} -- number of cluster {default=4}
        min_dist {int} -- minimum distance between cluster centers {default=2000}

    Returns:
        List of Coordinates -- each row containes the coordinates of a fruit tree
    """
    fruits = []
    area = np.count_nonzero(dp.Island())
    size = int(area / 1000000 * density)     # number of trees to compute
    # Compute Trees
    if distr == Distr.UNIFORM:
        while len(fruits) < size:
            f = [random.randint(0, dp.Island().shape[0] - 1), random.randint(0, dp.Island().shape[1] - 1)]
            if dp.Island()[f[0], f[1]] > 0:
                fruits.append(geo.Coordinates(f[0], f[1], dp.Island()[f[0], f[1]]))
    elif distr == Distr.NORMAL:
        if mean is None or mean[0] < 0 or mean[1] < 0 or mean[0] > dp.Island().shape[0] - 1 or mean[1] > dp.Island().shape[1] - 1:
            mean = [dp.Island().shape[0] / 2, dp.Island().shape[1] / 2]
        cov = [[math.pow(mean[0] / sd_factor, 2), 0], [0, math.pow(mean[1] / sd_factor, 2)]]    # covariance matrix. Axis are independent
        fruits = np.random.multivariate_normal(mean, cov, size)
        fruits = fruits.astype(int)
        fruits = [_createCoor(x) for x in fruits]
        fruits = [x for x in fruits if _checkFruitTree(x)]
    elif distr == Distr.CLUSTERED:
        clst = []
        while len(clst) < num_clast:
            c = [random.randint(0, dp.Island().shape[0] - 1), random.randint(0, dp.Island().shape[1] - 1)]
            if dp.Island()[c[0], c[1]] > 0:
                p = geo.Coordinates(c[0], c[1], dp.Island()[c[0], c[1]])
                if not _checkMinDist(p, clst, min_dist):
                    continue
                clst.append(p)
        for c in clst:
            c_fruits = []
            mean = [c.x, c.y]
            cov = [[math.pow(dp.Island().shape[0] / (sd_factor * num_clast), 2), 0], [0, math.pow(dp.Island().shape[1] / (sd_factor * num_clast), 2)]]    # covariance matrix. Axis are independent
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
    # values initialization
    p = orig.clone()                      # set last point of path to origin
    path = [orig]                 # path only contains origin
    spds = np.array([])           # initialize speeds array
    agls = np.array([])           # initialize angles array
    load_index = mc.DEF_LOAD_SIZE          # initialize loop counter for load size (decrementing)
    angl_delta = mc.WATER_SHIFT   # additional angle delta to avoid water
    water_index = 0               # counter of water avoidance trials
    counter = 0                   # counter of all points computed
    valid_cnt = 0                 # counter of valid points computed

    # generation loop
    f = p.sees(fruits, dp.Island(), p, ignores)
    # generation loop (continues until a tree is seen)
    while(f is None and counter < mc.MAX_ITERATIONS):
        # Compute speed and angles for the next LOAD_SIZE rounds.
        if load_index == mc.DEF_LOAD_SIZE:
            agl = np.random.normal(0, 360, 1)
            spds = np.random.normal(speed[0], speed[1], mc.DEF_LOAD_SIZE)
            agls = np.random.normal(agl, angle, mc.DEF_LOAD_SIZE)
            load_index = 0
        # Compute next point
        delta = geo.unit()
        dist = mc.DT * spds[load_index]
        delta.scale(dist / delta.mag())
        delta = delta.rotate(agls[load_index])
        nxt = p.add(delta, inplace=False)
        counter = counter + 1                                                        # DEBUG!!!
        # Check if next point is on water or path goes through water
        try:
            nxt.z = dp.Island()[int(nxt.x), int(nxt.y)]
        except IndexError:
            nxt.z = 0
        if nxt.z == 0 or p.inWater(nxt, dp.Island()):
            print("WARNING: random path goes into water from " + str(orig))
            raise me.PathOnWaterException()                                 # If on water "Cancel"
            # if water_index > 180 / mc.WATER_SHIFT:
            #     angl_delta = -mc.WATER_SHIFT
            #     water_index = 0
            # else:
            #     angl_delta = angl_delta + mc.WATER_SHIFT
            # water_index = water_index + 1
            # agls[load_index] = agls[load_index] + angl_delta
            # continue
        # add point to path
        path.append(nxt)                    # append new valid point to path
        f = nxt.sees(fruits, dp.Island(), p, ignores)    # check if fruit tree is visible from new point
        # handle loop variables
        p = nxt
        water_index = 0
        load_index += 1
        valid_cnt += 1

    # if max iterations reached throw exception
    if counter == mc.MAX_ITERATIONS:
        raise me.MaxIterationsReachedException()
    # if no tree found
    if f is None:
        print("No tree visible")
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
    # value initialization
    LOAD_SIZE = 100
    p = orig
    path = [orig]
    spds = np.array([])
    agls = np.array([])
    index = LOAD_SIZE
    angl_delta = mc.WATER_SHIFT
    water_index = 0
    counter = 0                                                                     # DEBUG!!!!
    # generation loop until almost there or MAX_ITERATIONS
    while(p.distance(dest) > mc.DT * speed[0] / 2 and counter < mc.MAX_ITERATIONS):
        # Compute speed and angles for the next LOAD_SIZE rounds.
        if index == LOAD_SIZE:
            spds = np.random.normal(speed[0], speed[1], LOAD_SIZE)
            agls = np.random.normal(0, angle, LOAD_SIZE)
            index = 0
        # Compute next point
        delta = p.diff(dest, inplace=False)
        dist = mc.DT * spds[index]
        delta.scale(dist / delta.mag())
        delta.rotate(agls[index])
        nxt = p.add(delta, inplace=False)
        counter = counter + 1                                                        # DEBUG!!!
        # Check if next point is on water or path goes through water
        nxt.z = dp.Island()[int(nxt.x), int(nxt.y)]
        # if nxt.z == 0 or p.inWater(nxt, dp.Island()):
        if nxt.z == 0 or p.inWater(nxt, dp.Island()):
            # if nxt.z == 0:
            #     print("WARNING: direct path into water from " + str(orig) + " towards " + str(nxt))
            # else:
            #     print("WARNING: direct path into water from " + str(orig))
            # print("WARNING: direct path into water")
            raise me.PathOnWaterException()                     # Raise path on water Exception and stop path creation
            # if water_index > 180 / WATER_SHIFT:
            #     angl_delta = -WATER_SHIFT
            #     water_index = 0
            # else:
            # angl_delta = angl_delta + mc.WATER_SHIFT
            # water_index = water_index + 1
            # agls[index] = agls[index] + angl_delta
            # continue
        # add point to path
        path.append(nxt)
        # handle loop variables
        water_index = 0
        p = nxt
        index = index + 1

    # print("Direct Counter " + str(counter))                                       # DEBUG!!!
    # if max iterations reached throw exception
    if counter == mc.MAX_ITERATIONS:
        raise me.MaxIterationsReachedException()
    # path.append(dest)
    return path


def _createDirect(orig, dest):
    """Created direct path from orig to dest
    """
    trials = 0
    while trials < 10:
        try:
            return _createP2PPath(orig, dest, [mc.DRT_VEL_EV, mc.DRT_VEL_SD], mc.DRT_ANG_SD)
        except me.PathOnWaterException:
            trials += 1
    raise me.PathOnWaterException()

def _createRandom(orig, fruits, ignores):
    """Create random path from orig to first tree seen
    """
    return _createP2RPath(orig, [mc.RDM_VEL_EV, mc.RDM_VEL_SD], mc.RDM_ANG_SD, fruits, ignores)

def createViewPath(orig, fruits, ignores, split=False):
    """Creates a view-model path

        orig {Coordinates} -- starting point
        fruits {List[Coordinates]} -- set of fruit trees coordinates
        ignores {List[Coordinates]} -- set of fruit trees to ignore: already visited

    Keyword Arguments:
        split {boolean} -- return either a unique path (false) or two paths split on viewpoin (true)  {Default:False}

    Returns:
        [List[Coordinates], List[Coordinates]] -- tuple containing the random path to the viewpoint and the direct path to the fruit tree
    """
    [path, dest] = _createRandom(orig, fruits, ignores)
    viewpoint = path[-1]
    direct = _createDirect(viewpoint, dest)
    path.extend(direct)
    return path


def createViewDate(fruits, orig=None, totaltime=mc.DATE_DURATION_MIN, startTime=mc.INIT_TIME):
    """Computes the path for a date according to the view model

    Arguments:
        orig {Coordinates} -- starting point
        fruits {List[Coordinates]} -- set of fruit trees
        totaltime {float} -- duration of day in minutes
        startTime {time} -- timestamp of first point.   {Default: mc.INIT_TIME}

    Returns:
        List[Coordinates] -- List of coordinates
    """
    # Compute origin if none
    if orig is None:
        p = np.random.normal(2500, 500, [1, 2])[0]
        orig = geo.Coordinates(p[0], p[1], dp.Island()[int(p[0]), int(p[1])])

    # Variable initialization
    tot_steps = int(totaltime * 60 / mc.DT)  # Duration of day expressed as number of datapoints
    path = []           # path initially empty
    curr_steps = 0      # step (datapoint) counter
    start = orig
    ignores = [start]   # add start to list of destination to ignore
    counter = 0

    # generation loop
    print("orig is " + str(orig))
    while curr_steps < tot_steps  and counter < mc.MAX_ITERATIONS:
        try:
            curr_path = createViewPath(start, fruits, ignores)      # generate a path
        # if path goes on water
        except me.PathOnWaterException:
            #  return None            # GIVE UP
            counter += 1
            continue                # try again
        # add new subpath to path and fruit tree to ignores
        path.extend(curr_path)
        ignores.append(start)
        start = curr_path[-1 ]
        # add hanging around fruit tree
        curr_path = hangAround(start, start, fruits=fruits, expand=False)
        path.extend(curr_path)
        ignores.append(start)
        start = curr_path[-1]
        # handle loop values
        curr_steps = curr_steps + len(curr_path)

    # if max iterations reached throw exception
    if counter == mc.MAX_ITERATIONS:
        raise me.MaxIterationsReachedException()

    # add timing
    delta = timedelta(seconds=mc.DT)
    dtime = datetime.combine(date.today(),startTime)
    for p in path[0:tot_steps]:
        p.set_time(dtime.time())
        dtime = dtime + delta
    # cut to tot_steps
    return path[0:tot_steps]


def createMemoryDate(fruits, orig=None, totaltime=mc.DATE_DURATION_MIN, startTime=mc.INIT_TIME, max_mem_range=2000):
    """Computes the path for a date according to the memory model

    Arguments:
        fruits {List[Coordinates]} -- set of fruit trees

    Keyword Arguments:
        orig {Coordinates} -- starting point of hanging. If None choose random.  {Default: None}
        totaltime {float} -- duration of day in minutes.  {Default: mc.DATE_DURATION_MIN}
        startTime {time} -- timestamp of first point.   {Default: mc.INIT_TIME}
        max_mem_range {float} -- maximum bird's-eye distance from fruit tree to next fruit tree in path (default: {2000})

    Returns:
        List[Coordinates] -- List of coordinates
    """
    # Compute origin if none
    if orig is None:
        p = np.random.normal(2000, 500, [1, 2])[0]
        orig = geo.Coordinates(p[0], p[1], dp.Island()[int(p[0]), int(p[1])])
    # Variale initialization
    tot_steps = int(totaltime * 60 / mc.DT)  # Duration of day expressed as number of datapoints
    path = []       # path is initially empty
    start = orig
    ignores = [start]   # add start to list of destination to ignore
    curr_plan = 0       # counter of planned steps (shortest path)
    rdm_index = int(np.random.uniform(0, len(fruits), 1))   # get random fruit tree index

    # print("tot_steps = " + str(tot_steps))                # DEBUG!!!

    # external loop. Required if mc.PLANNING_STEPS are not enough to reach tot_steps
    while True:
        # randomly draw mc.PLANNING_STEPS fruit trees that will be visited
        curr_fruits = [start]   # list of fruit tree that I will visit
        for curr_plan in range(0, mc.PLANNING_STEPS):
            while fruits[rdm_index].minDistance(curr_fruits)[0] > max_mem_range \
                    or fruits[rdm_index].minDistance(curr_fruits)[0] < mc.MIN_FRUIT_DIST or fruits[rdm_index] in ignores:
                rdm_index = int(np.random.uniform(0, len(fruits), 1))
            curr_fruits.append(fruits[rdm_index])
            ignores.append(fruits[rdm_index])
        curr_fruits.remove(start)
        # print("curr_fruits: " + str(curr_fruits))
        curr_fruits = geo.shortestPath(start, curr_fruits)      # sort fruits along shortest path
        # create sequential path to each fruit tree
        for f in curr_fruits:
            try:
                curr_path = _createDirect(start, f)
            # if path goes on water
            except me.PathOnWaterException:
                #  return None            # GIVE UP overall
                continue                # skip tree
            # add new subpath to path and fruit tree to ignores
            # print("path: " + str(len(curr_path)))
            path.extend(curr_path)
            # ignores.append(start)
            start = curr_path[-1]
            # add hanging around fruit tree
            curr_path = hangAround(start, start, fruits=fruits, expand=False)
            path.extend(curr_path[0:-2])
            ignores.append(start)
            start = curr_path[-1]
            if len(path) > tot_steps:
                break
        if len(path) > tot_steps:
            break

    # print("ignores: " + str(ignores))

    # add timing
    delta = timedelta(seconds=mc.DT)
    dtime = datetime.combine(date.today(),startTime)

    validpath = path[0:tot_steps]
    for p in validpath:
        p.set_time(dtime.time())
        print(str(p) + " - dtime: " + str(dtime.time()))
        dtime = dtime + delta
    print("After time assignment")
    for p in validpath:
        print(p)
    # cut to tot_steps
    # return path[0:tot_steps]
    return validpath;


def hangAround(orig, fruit, maxradius=mc.FRT_HANG_RAD, time=[mc.FRT_HANG_MINTIME, mc.FRT_HANG_MAXTIME],
               vel=[mc.HNG_VEL_EV, mc.HNG_VEL_SD], expand=True, fruits=None):
    """ Create hanging pattern around fruit tree

    Arguments:
        orig {Coordinates} -- starting point of hanging.
        fruit {Coordinates} -- fruit tree around which to hang

    Keyword Arguments:
        maxradius {int} -- maximum distance from fruit tree reachable while hanging  {Default: mc.FRT_HANG_RAD}
        maxtime {[int,int]} -- minimum and maximum duration (minutes) of hanging  {Default: [mc.FRT_HANG_MINTIME, mc.FRT_HANG_MAXTIME]}
        vel {[float, float]} -- expected value and standard deviation of velocity during hanging  {Default: [mc.HNG_VEL_EV, mc.HNG_VEL_SD]}
        expand {boolean} -- allow hanging on neighbour trees  {Default: True}
        fruits {List[Coordinates]} -- list of fruit trees. Required if expand, ignored otherwise  {Default: None}

    Returns:
        List[Coordinates] -- hanging path
    """
    # Values initialization
    hangtime = int(np.random.uniform(time[0], time[1], 1))   # get random hang duration
    tot_steps = int(hangtime * 60 / mc.DT)  # Duration of hanging expressed as number of datapoints
    path = []       # hanging path is initially empty
    curr_steps = 0      # step (datapoint) counter
    current = orig
    trials = 0      # number of trials. Used to avoid infinite loops

    # generate randomic params
    spds = np.random.normal(vel[0], vel[1], tot_steps)  # list of speeds for each datapoint
    agls = np.random.uniform(0, 360, tot_steps)         # list of angles for each datapoint
    # noise = np.random.uniform(0, 2, tot_steps)          # random noise

    noise = 1      # null noise. Currently deemed not necessary

    # generation loop
    while curr_steps < tot_steps and trials < mc.MAX_ITERATIONS:
        # compute next point
        delta = geo.unit()
        delta.scale(mc.DT * spds[curr_steps] * noise)
        delta.rotate(agls[curr_steps] * noise)
        nxt = current.add(delta, inplace=False)
        # Check if next point is close to fruit tree or if next point is close to any fruit tree and expand
        if nxt.distance(fruit) <= mc.FRT_HANG_RAD \
                or (expand and filter(lambda x: x < mc.FRT_HANG_RAD, current.distance(fruits))):
            current = nxt
            path.append(current)
            curr_steps = curr_steps + 1
        else:
            # regenerate random params
            spds[curr_steps] = np.random.uniform(vel[0], vel[1])
            agls[curr_steps] = np.random.uniform(0, 360)
            # noise[curr_steps] = np.random.uniform(0, 2)
        # update iteration counter
        trials = trials + 1

    # check if overflow of iterations
    if trials >= mc.MAX_ITERATIONS:
        import pdb; pdb.set_trace()  # breakpoint 763b01c8 //
        raise me.MaxIterationsReachedException()
    # print("hang: " + str(len(path)))
    return path



# -----------------------------------------------------------------------#
# ---------------------------OLD DATE GENERATOR--------------------------#


# def createMemoryDate(orig, fruits, totaltime, max_mem_range=2000):
#     """Computes the path for a date according to the memory model

#     Arguments:
#         orig {Coordinates} -- starting point
#         fruits {List[Coordinates]} -- set of fruit trees
#         totaltime {float} -- duration of day in minutes

#     Keyword Arguments:
#         max_mem_range {float} -- maximum bird's-eye distance from fruit tree to next fruit tree in path (default: {2000})

#     Returns:
#         datepath
#     """
#     # Variale initialization
#     tot_steps = totaltime * 60 / mc.DT  # Duration of day expressed as number of datapoints
#     path = []       # path is initially empty
#     curr_steps = 0      # step (datapoint) counter
#     start = orig
#     ignores = [start]   # add start to list of destination to ignore
#     curr_plan = 0       # counter of planned steps (shortest path)
#     rdm_index = int(np.random.uniform(0, len(fruits), 1))   # get random fruit tree index

#     # print("tot_steps = " + str(tot_steps))                # DEBUG!!!

#     # external loop. Required if mc.PLANNING_STEPS are not enough to reach tot_steps
#     while True:
#         # randomly draw mc.PLANNING_STEPS fruit trees that will be visited
#         curr_fruits = [start]   # list of fruit tree that I will visit
#         for curr_plan in range(0, mc.PLANNING_STEPS):
#             while fruits[rdm_index].minDistance(curr_fruits)[0] > max_mem_range \
#                     or fruits[rdm_index].minDistance(curr_fruits)[0] < mc.MIN_FRUIT_DIST or fruits[rdm_index] in ignores:
#                 rdm_index = int(np.random.uniform(0, len(fruits), 1))
#             curr_fruits.append(fruits[rdm_index])
#             ignores.append(fruits[rdm_index])
#         curr_fruits.remove(start)
#         curr_fruits = geo.shortestPath(start, curr_fruits)      # sort fruits along shortest path
#         # create sequential path to each fruit tree
#         for f in curr_fruits:
#             curr_path = _createDirect(start, f)
#             # if path goes on water
#             if curr_path is None:
#                 #  return None            # GIVE UP
#                 continue                # try again
#             #
#             # TODO: add hanging around fruit tree
#             #
#             # handle loop values
#             start = curr_path[-1]
#             curr_steps = curr_steps + len(curr_path)
#             # print("curr_steps = " + str(curr_steps))          # DEBUG!!!
#             path.append([[], curr_path])
#             if curr_steps > tot_steps:
#                 break
#         if curr_steps > tot_steps:
#             break
#     print("Date split into " + str(len(path)) + " paths")       # DEBUG!!
#     return dtpt.datepath.TimelessInit(paths=path)


# def createViewDate(orig, fruits, totaltime):
#     """Computes the path for a date according to the view model

#     Arguments:
#         orig {Coordinates} -- starting point
#         fruits {List[Coordinates]} -- set of fruit trees
#         totaltime {float} -- duration of day in minutes

#     Returns:
#         List[[List[Coordinates], List[Coordinates]]] -- List of tuples containing each path to the next fruit tree (tuple[1] is None for format consistency)
#     """
#     # Variable initialization
#     tot_steps = totaltime * 60 / mc.DT  # Duration of day expressed as number of datapoints
#     path = []           # path initially empty
#     curr_steps = 0      # step (datapoint) counter
#     start = orig
#     ignores = [start]   # add start to list of destination to ignore
#     # generation loop
#     while curr_steps < tot_steps:
#         curr_path = createViewPath(start, fruits, ignores, split=True)      # generate a path
#         # if path goes on water
#         if curr_path[1] is None:
#             #  return None            # GIVE UP
#             continue                # try again
#         # add new subpath to path and fruit tree to ignores
#         path.append(curr_path)
#         ignores.append(start)
#         #
#         # TODO: add hanging around fruit tree
#         #
#         # handle loop values
#         start = curr_path[1][-1]
#         curr_steps = curr_steps + len(curr_path)
#     # transform path to datepath
#     print("Date split into " + str(len(path)) + " paths")
#     return dtpt.datepath.TimelessInit(path)
