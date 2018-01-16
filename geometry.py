"""Library for math operation on paths and related
"""
import math


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
        dist = self.distance(dest, ignore_z)
        if type(pts) is Coordinates:
            pts = [pts]
        for p in pts:
            if(self.equals(p) | dest.equals(p)):
                continue
            dist_orig = self.distance(p, ignore_z)
            dist_dest = p.distance(dest, ignore_z)
            proportion = dist_orig / (dist_orig + dist_dest)
            proj_orig = dist * proportion
            # print("dist: " + str(dist))
            # print("orig: " + str(dist_orig))
            d = math.sqrt(math.pow(dist_orig, 2) - math.pow(proj_orig, 2))
            out = out & (d < radius)
            if d >= radius:
                if stop_on_found:
                    return [False, [i]]
                else:
                    index.append(i)
            i = i + 1
        return [out, index]


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
    return pts.iloc[reduced]
