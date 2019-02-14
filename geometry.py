"""Library for math operation on paths and related
"""
import numpy as np
import math
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import monkeyconstants as mc


class Coordinates:
    Island = None;

    """Represents a point coordinate with time information
    """
    def __init__(self, x, y, z, time=None):
        self.x = x
        self.y = y
        self.z = z
        self.set_time(time)
        
    @property
    def iz(self):
        return int(self.z)

    @property
    def xy(self):
        return [self.x, self.y]

    @property
    def xz(self):
        return [self.x, self.z]

    @property
    def yz(self):
        return [self.x, self.z]

    @property
    def xiz(self):
        return [self.x, int(self.z)]

    @property
    def yiz(self):
        return [self.x, int(self.z)]

    @property
    def xyz(self):
        return [self.x, self.y, self.z]

    @property
    def xyiz(self):
        return [self.x, self.y, int(self.z)]

    @property
    def xyzt(self):
        return [self.x, self.y, self.z, self.time]

    @property
    def xyizt(self):
        return [self.x, self.y, int(self.z), self.time]

    def __eq__(self, other):
        return self.equals(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if (self.time is None):
            return str(f"[{self.x:.2f}, {self.y:.2f}, {self.z:.2f}]")
        else:
            return str(f"[{self.x:.2f}, {self.y:.2f}, {self.z:.2f} - {self.time}]")

    def clone(self):
        return Coordinates(self.x, self.y, self.z, time=self.time)

    def resetz(self):
        """recomputes z from Island
        """
        if Coordinates.Island is not None:
            try:
                self.z = Coordinates.Island[int(self.x), int(self.y)]
            except IndexError:
                self.z = 0

    def set_time(self, time):
        """Assign a timestamp to point
        """
        self.time = time

    def equals(self, p, ignore_z=False, ignore_t=True):
        """determines whether 2 points are equals

        Keyword Arguments:
            ignore_z {bool} -- ignore z in the comparison (default: {False})
        """
        if p is None:
            return False
        return self.x == p.x and self.y == p.y and (ignore_z or self.z == p.z) and (ignore_t or self.time == p.time)

    def distance(self, p, ignore_z=False, noneValue=None):
        """ Euclidean distance from the parameter point
        Arguments:
            p {Coordinates / [Coordinates]} -- point or list of points to measure the distance from 

        Keyword Arguments:
            ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)
            noneValue {float}  -- value to return if parameter point is None. (default: None)

        Returns:
            distances {float / [float]} -- distance from self to p. If p is list, then distances is a list of distances
                                           from self to all points in p
        """
        # Case p is list of points
        if p is None:
            return noneValue;
        if isinstance(p, list):
            distances = []                      # array of distances
            for point in p:
                distances.append(self.distance(point, ignore_z, noneValue))
            return distances
        # case p is single point
        else:
            x2 = math.pow((self.x - p.x), 2)
            y2 = math.pow((self.y - p.y), 2)
            if (ignore_z is False):
                z2 = math.pow((self.z - p.z), 2)
                return math.sqrt(x2 + y2 + z2)
            else:
                return math.sqrt(x2 + y2)

    def minDistance(self, points, ignore_z=False):
        """Return minimum distance between the point and the set of points given as parameter

        Arguments:
            points {List[Coordinates]} -- points considered for minimum distance

        Keyword Arguments:
            ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)

        Return:
            min_dist {[float, Coordinates]} -- point of the set at minimum distance and distance
        """
        min_dist = [math.inf, None]
        for p in points:
            d = self.distance(p, ignore_z)
            if d < min_dist[0]:
                min_dist = [d, p]
        return min_dist

    def maxDistance(self, points, ignore_z=False):
        """Return maximum distance between the point and the set of points given as parameter

        Arguments:
            points {List[Coordinates]} -- points considered for maximum distance

        Keyword Arguments:
            ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)

        Return:
            max_dist {[float, Coordinates]} -- point of the set at maximum distance and distance
        """
        max_dist = [0, None]
        for p in points:
            d = self.distance(p, ignore_z)
            if d > max_dist[0]:
                max_dist = [d, p]
        return max_dist

    def dot(self, p, ignore_z=False):
        """dot product between caller and p

        Keyword Arguments:
            ignore_z {bool} -- Ignore z in the computation (default: {False})
        """
        if ignore_z:
            return self.x * p.x + self.y * p.y
        else:
            return self.x * p.x + self.y * p.y + self.z * p.z

    def mag(self, ignore_z=False):
        """Returns the magnitude of the point vector

        Keyword Arguments:
            ignore_z {bool} -- ignore z in the computation (default: {False})
        """
        if ignore_z:
            return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))
        else:
            return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2) + math.pow(self.z, 2))

    def angle(self, p, o, ignore_z=False):
        """returns sin and cos of the angle between two points wrt o

        if o is equals to either the caller or p, [0, 0] is returned

        Keyword Arguments:
            ignore_z {bool} -- ignore z in the computation (default: {False})
        """
        if self.equals(o) or p.equals(o):
            return [0, 0]
        s = Coordinates(self.x - o.x, self.y - o.y, self.z - o.z)
        d = Coordinates(p.x - o.x, p.y - o.y, p.z - o.z)
        cos = s.dot(d, ignore_z) / (s.mag(ignore_z) * d.mag(ignore_z))
        if cos >= 1 or cos <= -1:
            return [0, 1]
        elif cos == 0:
            return [1, 0]
        sin = math.sqrt(1 - math.pow(cos, 2))
        return [sin, cos]

    def diff(self, p, ignore_z=False, inplace=True):
        """Return the vector of the difference between p and caller

        Keyword Arguments:
            ignore_z {bool} -- ignore z in the computation (default: {False})
            inplace {bool} -- modify caller  {Default: True}
        """
        if ignore_z:
            if inplace:
                self.x = p.x - self.x
                self.y = p.y - self.y
                self.z = 0
            return Coordinates(p.x - self.x, p.y - self.y, 0)
        else:
            if inplace:
                self.x = p.x - self.x
                self.y = p.y - self.y
                self.z = p.z - self.z
            return Coordinates(p.x - self.x, p.y - self.y, p.z - self.z)

    def add(self, p, ignore_z=False, inplace=True):
        """Return the point resulting in the sum of p and caller

        Keyword Arguments:
            ignore_z {bool} -- ignore z in the computation (default: {False})
            inplace {bool} -- modify caller  {Default: True}
        """
        if ignore_z:
            if inplace:
                self.x += p.x
                self.y += p.y
            return Coordinates(p.x + self.x, p.y + self.y, 0)
        else:
            if inplace:
                self.x += p.x
                self.y += p.y
                self.z += p.z
            return Coordinates(p.x + self.x, p.y + self.y, p.z + self.z)

    def scale(self, factor, inplace=True):
        """Scales the caller's coordinates of factor

        Arguments:
            factor {float} -- scaling factor

        Keyword Arguments:
            inplace {bool} -- modify caller  {Default: True}
        """
        if inplace:
            self.x = self.x * factor
            self.y = self.y * factor
            self.z = self.z * factor
            return Coordinates(self.x, self.y, self.z)
        else:
            return Coordinates(self.x * factor, self.y * factor, self.z * factor)

    def rotate(self, deg, inplace=True):
        """Rotates the vector of deg degrees (0-360°)

        Positive deg is counterclockwise rotation
        Negative deg is clockwise rotation

        Arguments:
            deg {float} -- rotation angle

        Keyword Arguments:
            inplace {bool} -- modify caller  {Default: True}
        """
        sin = math.sin(math.radians(-deg))
        cos = math.cos(math.radians(-deg))
        matrix = np.array([[cos, -sin], [sin, cos]])
        coor = np.array([self.x, self.y])
        rot_coor = matrix.dot(coor)
        if inplace:
            self.x = rot_coor[0]
            self.y = rot_coor[1]
        return Coordinates(rot_coor[0], rot_coor[1], self.z)

    def within(self, pts, dest, radius, ignore_z=False, stop_on_found=True):
        """Checks whether the point p is strictyl within radius distance from the line connecting the calling point and dist

        Arguments:
            pts {Coordinates} -- point considered OR list of point to be considered
            dest {Coordinates} -- destination point
            radius {float} -- maximum distance from the line orig-dist

        Keyword Arguments:
            ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)
            stop_on_found {Boolean} -- stop when a point not within is found, or continue until all points are checked (default: True)

        Returns:
            {bool, [index]} --  True -> all points are between, False -> at least one point is not within
                                if stop_on_found the index of the first point not within is returned (list with 1 item)
                                else the indices of all the points not within is returned as a list
        """
        out = True
        i = 0
        index = []
        if type(pts) is Coordinates:
            pts = [pts]
        for p in pts:
            # print('pts: ' + str(p.x) + ', ' + str(p.y) + ', ' + str(p.z))
            if(self.equals(p) | dest.equals(p)):
                continue
            dist_orig = self.distance(p, ignore_z)
            d = dist_orig * p.angle(dest, self, ignore_z)[0]
            out = out & (d < radius)
            if d >= radius:
                if stop_on_found:
                    return [False, [i]]
                else:
                    index.append(i)
            i = i + 1
        return [out, index]

    def sees(self, fruits, island, previous, ignores):
        """Checks if any fruit tree within range and in FOV with respect to movement direcction can be seen by the caller

        Arguments:
            fruits {List[Coordinates]} -- List of fruit tree Coordinates
            island {img[nxmxk]} -- island
            previous {Coordinates} -- previous point, used to computed direction of movement
            ignores {List[Coordinates]} -- List of fruit trees to ignore: Already visited

        Library Variables:
            VIEW_MAX_RANGE {float} -- maximum distance a fruit tree can be seen (default: {200})
            VIEW_MIN_RANGE {float} -- distance within which a tree is seen by default (default: {20})
            FOV {float} -- field of view of the monkey. It can only see trees which are in its field of view {default: {100}}

        Returns
            fruit {Coordinates} -- Coordinate of closes fruit tree which can be seen
        """
        # sin = math.sin(mc.FOV / 2)
        cos = math.cos(mc.FOV / 2)
        cont = self.add(previous.diff(self, inplace=False), inplace=False)
        # lst = filter(lambda x: self.distance(x) < mc.VIEW_MAX_RANGE, fruits)
        # lst = filter(lambda x: self.distance(x) < mc.VIEW_MIN_RANGE or
        #              (self.distance(x) < mc.VIEW_MAX_RANGE and abs(cont.angle(x, self, ignore_z=True)[0]) < sin and
        #               cont.angle(x, self, ignore_z=True)[1] < -cos), fruits)
        lst = filter(lambda x: self.distance(x) < mc.VIEW_MIN_RANGE or
                     (self.distance(x) < mc.VIEW_MAX_RANGE and cont.angle(x, self, ignore_z=True)[1] < -cos), fruits)
        index = 0
        for e in lst:
            index = index + 1
            if e in ignores:
                continue
            if self.isVisible(e, island):
                return e
        return None

    def isVisible(self, p, island, min_range=mc.VIEW_MIN_RANGE, next=None, FOV=mc.FOV, max_range=mc.VIEW_MAX_RANGE):
        """Checks if p is visible from the caller

        Requires island to be set to the corrisponding matrix

        Arguments:
            p {Coordinate / [Coordinates]} -- Point to be seen. may be a list of points
            island {img[nxmxk]} -- island

        Keyword Arguments:
            min_range {float} -- distance within which a tree is seen by default.  {Default: mc.VIEW_MIN_RANGE}
            next {Coordinates} -- previous point. Establishes looking direction. if None look all around {Default: None}
            FOV {int} -- Horizontal field of view in degrees. Ignored if previous is None.  {Default: mc.FOV}
            max_range {float} -- maximum visibility. Actual visibility is a probabiltiy squared with the distance

        Returns:
            visible {Boolean / [Boolean]} -- The point is visible. If p is list, return boolean list
        """

        # case p is None
        if p is None:
            return False
        # case p is list
        if isinstance(p, list):
            visible = []
            for point in p:
                visible.append(self.isVisible(point, island, min_range, next, FOV))
            return visible
        
        # distance evaluation
        dist = self.distance(p)
        sqrdmax = random.randint(min_range * min_range, max_range * max_range) 
        # case fruit tree too close
        if (dist < min_range):
            return True
        # case fruit tree too far
        if (dist*dist > sqrdmax):
            return false

        # check if p in FOV if next available
        if next is not None:
            cos = math.cos(FOV / 2)
            cont = self.add(self.diff(next, inplace=False), inplace=False)
            if not cont.angle(p, self, ignore_z=True)[1] < -cos:
                return False


        # check if visible
        dist -= min_range
        unit = delta.scale(1 / dist, inplace=False)
        steps = 0
        temp = self
        while steps < dist:
            steps += 1
            temp = temp.add(unit)
            tl = [math.ceil(temp.x), math.floor(temp.y)]
            tr = [math.ceil(temp.x), math.ceil(temp.y)]
            br = [math.floor(temp.x), math.ceil(temp.y)]
            bl = [math.floor(temp.x), math.floor(temp.y)]
            try:
                if(island[tl[0], tl[1]] > temp.z + mc.HEIGHT_MARGIN):
                    tl = False
                else:
                    tl = True
            except IndexError:
                tl = False
            try:
                if(island[tr[0], tr[1]] > temp.z + mc.HEIGHT_MARGIN):
                    tr = False
                else:
                    tr = True
            except IndexError:
                tr = False
            try:
                if(island[br[0], br[1]] > temp.z + mc.HEIGHT_MARGIN):
                    br = False
                else:
                    br = True
            except IndexError:
                br = False
            try:
                if(island[bl[0], bl[1]] > temp.z + mc.HEIGHT_MARGIN):
                    bl = False
                else:
                    bl = True
            except IndexError:
                bl = False
            if not (tl and tr and br and bl):
                return False
        # print("Tree Found of height=" + str(p.z) + " from height=" + str(self.z))         # DEBUG!!!!
        return True

    def inWater(self, p, island):
        """Checks if path from self to p goes through water

        Arguments:
            p {Coordinates} -- destination
            island {img[nxmxk]} -- island

        Returns:
            bool -- true if path goes through water
        """
        delta = self.diff(p, inplace=False)
        dist = delta.mag()
        delta.scale(1 / dist)
        steps = 0
        temp = self
        while steps < dist:
            steps = steps + 1
            temp.add(delta, inplace=True)
            tl = [math.ceil(temp.x), math.floor(temp.y)]
            tr = [math.ceil(temp.x), math.ceil(temp.y)]
            br = [math.floor(temp.x), math.ceil(temp.y)]
            bl = [math.floor(temp.x), math.floor(temp.y)]
            try:
                if(island[tl[0], tl[1]] > 0):
                    tl = False
                else:
                    tl = True
            except IndexError:
                tl = True
            try:
                if(island[tr[0], tr[1]] > 0):
                    tr = False
                else:
                    tr = True
            except IndexError:
                tr = True
            try:
                if(island[br[0], br[1]] > 0):
                    br = False
                else:
                    br = True
            except IndexError:
                br = True
            try:
                if(island[bl[0], bl[1]] > 0):
                    bl = False
                else:
                    bl = True
            except IndexError:
                bl = True
            if tl or tr or br or bl:
                return True
        return False

    def nextfrom(self, prev, speed, angle):
        """ return next point considering direction from a given point and movement with given speed and angle.

            Arguments:
                prev {coordinates} -- previous point. If 'None' direction from the origin is considered
                speed {float} -- speed of movement (4m/s)
                angle {float} -- relative angle of movement in deg

            Returns:
                next {Coordinates} -- next point
        """
        # Compute delta vector
        if prev is None:
            delta = self    
        else:
            delta = prev.diff(self, inplace=False)
        # scale for speed
        dist = abs(mc.DT * speed)
        delta.scale(dist / delta.mag(), inplace=True)
        # rotate
        delta.rotate(angle, inplace=True)
        return self.add(delta, inplace=False).resetz()

    def nexttowards(self, target, speed, angle):
        """ return next point considering direction towards a given point and movement with given speed and angle.

            Arguments:
                target {coordinates} -- target point. If 'None' direction from the origin is considered
                speed {float} -- speed of movement (4m/s)
                angle {float} -- relative angle of movement in deg

            Returns:
                next {Coordinates} -- next point
        """
        # Compute delta vector
        if prev is None:
            delta = self    
        else:
            delta = self.diff(target, inplace=False)
        # scale for speed
        dist = abs(mc.DT * speed)
        delta.scale(dist / delta.mag(), inplace=True)
        # rotate
        delta.rotate(angle, inplace=True)
        return self.add(delta, inplace=False).resetz()




def unit():
    """Vector (1, 0, 0)
    """
    return Coordinates(1, 0, 0)


def shortestPath(start, points, ignore_z=False):
    """sorts according to minimum distance from start

    Arguments:
        start {Coordinates} -- starting point of shortest path
        points {[Coordinates]} -- list of points that need to be visited -> ordered

    Keyword Arguments:
        ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)

    Return:
        points {[Coordinates]} -- List of sorted points
    """
    shortest = []
    old = points
    s = start
    while old:
        old.sort(key=lambda x: s.distance(x, ignore_z))
        shortest.append(old[0])
        s = old.pop(0)
    return shortest


def reduce_path(pts, radius, ignore_z=False):
    """reduces path datapoints

    The path is reduced eliminating all the points that are within radius from the reduced segments

    Arguments:
        pts {dataframe} -- standard dataframe for movements
        radius {int} -- maximum distance within which a point is eliminated

    Keyword Arguments:
        ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)

    Returns:
        dataframe -- reduced set in the original format
    """
    reduced = []
    pts_list = pts[['x', 'y', 'h']].apply(lambda p: Coordinates(p.x, p.y, p.h), axis=1).values.tolist()
    if len(pts_list) == 0:
        return pts
    curr_orig = 0
    while curr_orig < len(pts_list) - 1:
        curr_dest = curr_orig + 1
        while curr_dest < len(pts_list) and pts_list[curr_orig].within(pts_list[curr_orig: curr_dest], pts_list[curr_dest], radius, ignore_z)[0]:
            curr_dest = curr_dest + 1
        reduced.append(curr_orig)
        curr_orig = curr_dest - 1
    reduced.append(curr_orig)
    return pts.iloc[reduced]


def pathActLength(path, ignore_z=False):
    """returns overall length of the path
    """
    distance = 0
    pre = path[0]
    for p in path:
        distance += pre.distance(p, ignore_z)
        pre = p
    return distance


def pathCrowLenght(path, ignore_z=False):
    """returns length of the path as the crow flies - aka distance between first and last point

    Keyword Arguments:
        ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)
    """
    lgt = path[0].distance(path[-1], ignore_z)
    if lgt > 0:
        return lgt
    else:
        return 1

def straighRatio(path, ignore_z=False):
    """Returns the degree of straightness of the path given as the ratio between the crow distance and the actual path distance
     if 1 - direct path
     if 0 - limit case, path never reaches the end

    Arguments:
        path {List[Coordinates]} -- path

    Keyword Arguments:
        ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)
    """
    # check minimum path length
    if len(path) < 2:
        return 0
    try:
        return pathCrowLenght(path, ignore_z) / pathActLength(path, ignore_z)
    except:
        print("pathActLength error")
        print("\tpath: " + str(path))
        return 0

def getSpeeds(path, dt=mc.DT, ignore_z=False):
    """Returns speed of monkey (m/s) at each step 

    Arguments:
        path {List[Coordinates]} -- path

    Keyword Arguments:
        dt {float} -- time distance between two consecutive points in seconds  (default: mc.DT)
        ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)

    Returns:
        {List[float]} -- list of speed. Length(speed) = Length(path) - 1
    """
    # check minimum path length
    if len(path) < 2:
        return [0]
    # init empty list
    speeds = np.empty((len(path) - 1))
    # loop on each path from 0 to len-1
    for i in range(0,len(path)-1):
        speeds[i] = path[i].distance(path[i+1]) / dt
    return speeds

def speedAeSD(path, dt=mc.DT, ignore_z=False):
    """Returns average and standard deviation of speed of monkey (m/s) at each step 

    Arguments:
        path {List[Coordinates]} -- path

    Keyword Arguments:
        dt {float} -- time distance between two consecutive points in seconds  (default: mc.DT)
        ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)

    Returns:
        {(float, float)} -- average ([0]) and standard deviation ([1]) of path speed
    """
    # check minimum path length
    if len(path) < 2:
        return [0, 0]
    # compute
    spds = np.array(getSpeeds(dt=dt, ignore_z=ignore_z))
    return (np.mean(spds, axis=0), np.std(spds, axis=0))

