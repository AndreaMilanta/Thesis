"""Library for math operation on paths and related
"""
import numpy as np
import pandas as pd
import math
import datetime

from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

ISLAND = None
HEIGHT_MARGIN = 2
VIEW_MAX_RANGE = 200
VIEW_MIN_RANGE = 20
FOV = 100


class Coordinates:
    """Represents a point coordinate
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.xy = [x, y]
        self.xz = [x, z]
        self.yz = [y, z]
        self.xyz = [x, y, z]

    def __eq__(self, other):
        return self.equals(other)

    def equals(self, p, ignore_z=False):
        """determines whether 2 points are equals

        Keyword Arguments:
            ignore_z {bool} -- ignore z in the comparison (default: {False})
        """
        return self.x == p.x and self.y == p.y and (ignore_z or self.z == p.z)

    def distance(self, p, ignore_z=False):
        """Euclidean distance from the parameter point

        Keyword Arguments:
            ignore_z {Boolean} -- ignore z dimension and work only on xy plane projection (default: False)
        """
        x2 = math.pow((self.x - p.x), 2)
        y2 = math.pow((self.y - p.y), 2)
        if (ignore_z is False):
            z2 = math.pow((self.z - p.z), 2)
            return math.sqrt(x2 + y2 + z2)
        else:
            return math.sqrt(x2 + y2)

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

    def diff(self, p, ignore_z=False):
        """Return the vector of the difference between p and caller

        Keyword Arguments:
            ignore_z {bool} -- ignore z in the computation (default: {False})

        Returns:
            [Coordinates] -- Point representing the vector of the difference
        """
        if ignore_z:
            return Coordinates(p.x - self.x, p.y - self.y, 0)
        else:
            return Coordinates(p.x - self.x, p.y - self.y, p.z - self.z)

    def add(self, p, ignore_z=False):
        """Return the point resulting in the sum of p and caller

        Keyword Arguments:
            ignore_z {bool} -- ignore z in the computation (default: {False})
        """
        if ignore_z:
            return Coordinates(p.x + self.x, p.y + self.y, 0)
        else:
            return Coordinates(p.x + self.x, p.y + self.y, p.z + self.z)

    def scale(self, factor):
        """Scales the caller's coordinates of factor
        """
        self.x = self.x * factor
        self.y = self.y * factor
        self.z = self.z * factor
        return Coordinates(self.x, self.y, self.z)

    def rotate(self, deg):
        """Rotates the vector of deg degrees (0-360°)

        Positive deg is counterclockwise rotation
        Negative deg is clockwise rotation
        """
        sin = math.sin(math.radians(-deg))
        cos = math.cos(math.radians(-deg))
        matrix = np.array([[cos, -sin], [sin, cos]])
        coor = np.array([self.x, self.y])
        rot_coor = matrix.dot(coor)
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

    def sees(self, fruits, previous, ignores):
        """Checks if any fruit tree within range and in FOV with respect to movement direcction can be seen by the caller

        Arguments:
            fruits {List[Coordinates]} -- List of fruit tree Coordinates
            previous {Coordinates} -- previous points, used to computed direction of movement
            ignores {List[Coordinates]} -- List of fruit trees to ignore: Already visited

        Library Variables:
            VIEW_MAX_RANGE {float} -- maximum distance a fruit tree can be seen (default: {200})
            VIEW_MIN_RANGE {float} -- distance within which a tree is seen by default (default: {20})
            FOV {float} -- field of view of the monkey. It can only see trees which are in its field of view {default: {100}}

        Returns
            fruit {Coordinates} -- Coordinate of closes fruit tree which can be seen
        """
        sin = math.sin(FOV / 2)
        cos = math.cos(FOV / 2)
        cont = self.add(previous.diff(self))
        lst = filter(lambda x: self.distance(x) < VIEW_MIN_RANGE or (self.distance(x) < VIEW_MAX_RANGE and
                            abs(cont.angle(x, self, ignore_z=True)[0]) < sin and cont.angle(x, self, ignore_z=True)[1] < cos), fruits)
        index = 0
        for e in lst:
            index = index + 1
            if e in ignores:
                continue
            if self.isVisible(e, VIEW_MIN_RANGE):
                return e
        return None

    def isVisible(self, p, min_range):
        """Checks if p is visible from the caller

        Requires ISLAND to be set to the corrisponding matrix

        Arguments:
            p {Coordinate} -- Point to be seen
            min_range {float} -- distance within which a tree is seen by default
            margin {float} -- margin for height comparison {default: {2}}

        Returns:
            visible {Boolean} -- The point is visible
        """

        # VISUALIZATION for DEBUG ONLY
        draw = False
        if draw:
            fig = plt.figure()
            # plt.imshow(island.transpose())
            ny, nx = ISLAND.shape
            x = np.linspace(0, nx, nx)
            y = np.linspace(0, ny, ny)
            xv, yv = np.meshgrid(x, y)
            sf = fig.add_subplot(111, projection='3d')
            sf.plot([self.x, p.x], [self.y, p.y], [self.z, p.z], c="#000000")
            sf.plot_surface(xv, yv, ISLAND, cmap="winter", linewidth=2)
            plt.show()

        delta = self.diff(p)
        dist = delta.mag()
        if(dist < min_range):
            # print("Tree too close found")             # DEBUG!!!!
            return True
        unit = delta.scale(1 / dist)
        steps = 0
        temp = self
        while steps < dist:
            steps = steps + 1
            temp = temp.add(unit)
            tl = [math.ceil(temp.x), math.floor(temp.y)]
            tr = [math.ceil(temp.x), math.ceil(temp.y)]
            br = [math.floor(temp.x), math.ceil(temp.y)]
            bl = [math.floor(temp.x), math.floor(temp.y)]
            try:
                if(ISLAND[tl[0], tl[1]] > temp.z + HEIGHT_MARGIN):
                    tl = False
                else:
                    tl = True
            except IndexError:
                tl = True
            try:
                if(ISLAND[tr[0], tr[1]] > temp.z + HEIGHT_MARGIN):
                    tr = False
                else:
                    tr = True
            except IndexError:
                tr = True
            try:
                if(ISLAND[br[0], br[1]] > temp.z + HEIGHT_MARGIN):
                    br = False
                else:
                    br = True
            except IndexError:
                br = True
            try:
                if(ISLAND[bl[0], bl[1]] > temp.z + HEIGHT_MARGIN):
                    bl = False
                else:
                    bl = True
            except IndexError:
                bl = True
            if not (tl and tr and br and bl):
                return False
        # print("Tree Found of height=" + str(p.z) + " from height=" + str(self.z))         # DEBUG!!!!
        return True

    def inWater(self, p):
        """Checks if path from self to p goes through water

        Arguments:
            p {Coordinates} -- destination

        Returns:
            bool -- true if path goes through water
        """
        delta = self.diff(p)
        dist = delta.mag()
        unit = delta.scale(1 / dist)
        steps = 0
        temp = self
        while steps < dist:
            steps = steps + 1
            temp = temp.add(unit)
            tl = [math.ceil(temp.x), math.floor(temp.y)]
            tr = [math.ceil(temp.x), math.ceil(temp.y)]
            br = [math.floor(temp.x), math.ceil(temp.y)]
            bl = [math.floor(temp.x), math.floor(temp.y)]
            try:
                if(ISLAND[tl[0], tl[1]] > 0):
                    tl = False
                else:
                    tl = True
            except IndexError:
                tl = True
            try:
                if(ISLAND[tr[0], tr[1]] > 0):
                    tr = False
                else:
                    tr = True
            except IndexError:
                tr = True
            try:
                if(ISLAND[br[0], br[1]] > 0):
                    br = False
                else:
                    br = True
            except IndexError:
                br = True
            try:
                if(ISLAND[bl[0], bl[1]] > 0):
                    bl = False
                else:
                    bl = True
            except IndexError:
                bl = True
            if tl or tr or br or bl:
                return True
        return False


def unit():
    """Vector (1, 0, 0)
    """
    return Coordinates(1, 0, 0)


def getList(points, id, datetime0, dt):
    """returns a dataframe representing the path

    Arguments:
        points {list [Coordinates]} -- List of points
        id {int} -- identifier of the monkey
        date {datetime.datetime} -- Date and time of the firts point
        dt {int} -- number of seconds between samples (points)

    Returns:
        dataframe -- info (id), coordinates (x,y,h) and timestamp (date,ts)
    """
    moves = []
    t = datetime0
    for p in points:
        moves.append((id, p.x, p.y, p.z, t.date(), t.time()))
        t = t + datetime.timedelta(0, dt)
    header = ['id', 'x', 'y', 'h', 'date', 'ts']
    return pd.DataFrame(moves, columns=header)


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
    pts_list = pts[['x', 'y', 'h']].apply(lambda p: Coordinates(p.x, p.y, p.h), axis=1).tolist()
    curr_orig = 0
    while curr_orig < len(pts_list) - 1:
        curr_dest = curr_orig + 1
        while curr_dest < len(pts_list) and pts_list[curr_orig].within(pts_list[curr_orig: curr_dest], pts_list[curr_dest], radius, ignore_z)[0]:
            curr_dest = curr_dest + 1
        reduced.append(curr_orig)
        curr_orig = curr_dest - 1
    reduced.append(curr_orig)
    return pts.iloc[reduced]
