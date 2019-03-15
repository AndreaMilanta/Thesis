import csv
import pandas as pd
from datetime import datetime

import geometry as geo
import datepath as dtp
import dataparser as dp
import monkeyconstants as mc
import display as disp

# Path to real dataset
ORIG_PATH = 'D:\\OneDrive - Politecnico di Milano\\Documents\\University\\Thesis\\Data\\Original Data.csv'
DISTR_FILE = mc.REAL_PATH + 'Distributions.txt'

MIN_VALID_LEN = 900     # Minimum length for a path to be valid
SPECIES = ["Ateles geoffroyi"]  # Species of interest

# set global static variables
dtp.datepath.Island = dp.Island()
geo.Coordinates.Island = dp.Island()


def _converttoXY(lon, lat):
    """Compute xy coordinates from GPS

    Converts GPS coordinates to xy referenced to the origin of the island pic

    Arguments:
        lon {float} -- Longitude
        lat {float} -- Latitude
    """

    # Fixed Pixel-GPS Matches
    Px =  [[3800,4400], \
           [4280,1790], \
           [355,1855]]

    GPS = [[9.142064,-79.836224], \
           [9.165617,-79.831682], \
           [9.165397,-79.867388]]

    # Compute transform
    b = (((Px[2][0] - Px[0][0]) / (GPS[2][0] - GPS[0][0])) + ((Px[0][0] - Px[1][0]) / (GPS[1][0] - GPS[0][0]))) / \
        (((GPS[0][1] - GPS[1][1]) / (GPS[1][0] - GPS[0][0])) + ((GPS[2][1] - GPS[0][1]) / (GPS[2][0] - GPS[0][0])))
    a = (Px[1][0] - Px[0][0] + b*GPS[0][1] - b*GPS[1][1]) / (GPS[1][0] - GPS[0][0])
    c = Px[0][0] - a*GPS[0][0] - b*GPS[0][1]
    e = (((Px[2][1] - Px[0][1]) / (GPS[2][0] - GPS[0][0])) + ((Px[0][1] - Px[1][1]) / (GPS[1][0] - GPS[0][0]))) / \
        (((GPS[0][1] - GPS[1][1]) / (GPS[1][0] - GPS[0][0])) + ((GPS[2][1] - GPS[0][1]) / (GPS[2][0] - GPS[0][0])))
    d = (Px[1][1] - Px[0][1] + e*GPS[0][1] - e*GPS[1][1]) / (GPS[1][0] - GPS[0][0])
    f = Px[0][1] - d*GPS[0][0] - e*GPS[0][1]

    # transform coordinates
    x = (lat*a + lon*b) + c
    y = (lat*d + lon*e) + f
    return (x, y)


def readcsvrow(filename, has_header=False, skipheader=False, delimiter=',', quotechar='"'):
    """ Returns csv row

        Arguments:
            filename {string} -- file name

        Keyword Arguments:
            has_header {boolean} -- whether the cs has the header. {Default: False}
            skipheader {boolean} -- if True return header as first row. Ignored if has_header==False.  {Default: False}
            delimiter {char} -- character  used to separate fields.  {Default ','}
            quotechar {char} -- character used to quote fields containing special characters, such as the delimiter \
                                or quotechar, or which contain new-line characters.  {Default: '"'}

        Returns:
            row
    """
    with open(filename, "rt") as csvfile:
        # create reader
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        # skip header if required
        if has_header: 
            if skipheader:
                next(reader)        # skip header row
            else:
                yield next(reader)  # yield the header row
        # return each row
        for row in reader:
            yield row



def readpath(filename, delimiter=',', quotechar='"', minlength=3, species=None):
    """ Returns datepath read from csv

        Arguments:
            filename {string} -- file name

        Keyword Arguments:
            skipheader boolean} -- if True return eader as first row (if present)
            delimiter {char} -- character  used to separate fields.  {Default ','}
            quotechar {char} -- character used to quote fields containing special characters, such as the delimiter \
                                or quotechar, or which contain new-line characters.  {Default: '"'}
            minlength {int} -- minimum length for a path to be considered valid.  {Default: 900}
            species {List[string]} -- list of accepeted species. If None take all.  {Default: None}

        Returns:
            row
    """
    # force minimum minlength to 3
    if minlength < 3:
        minlength = 3
    # init variables
    path = []
    currmonkey = -1
    currdate = None

    invalidcount = 0

    # loop on every point
    for row in readcsvrow(filename, has_header=True, skipheader=True, delimiter=delimiter, quotechar=quotechar):
        try:
            # parse row
            specie = row[23]
            monkey = int(row[24])
            [x, y] = _converttoXY(float(row[3]), float(row[4]))
            dtm = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S.%f')
            new = geo.Coordinates(x, y, 0, dtm)
            new.resetz()
        except Exception as e:
            # if row parsing failrs jump to next
            print('invalid row ' + str(invalidcount) + ': ' + str(row))
            invalidcount +=1
            continue
        if species is not None and specie not in species:
            continue
        # case same path
        if monkey == currmonkey and dtm.date() == currdate:
            path.append(new)
        else:
        # case new path
            # create and return new datepath
            if len(path) >= minlength:
                yield dtp.datepath(path, date=currdate, monkey=currmonkey, fruits=dp.Fruits())
            # prepare for new path
            currdate = dtm.date()
            currmonkey = monkey
            path = [new]
    print('Overall found ' + str(invalidcount) + 'invalid rows')
    # return last hanging path
    if len(path) > MIN_VALID_LEN:
        return dtp.datepath(path, date=currdate, monkey=currmonkey, fruits=dp.Fruits())



def splitpaths(origfile, minlength=3, species=None):
    """ split paths from huge unique file to different files separate per monkey and dates
        Also computes avg speed and angle distribution of given paths

        Arguments:
            origfile {string} -- full path to file with all paths

        Keyword Arguments:
            minlength {int} -- minimum length for a path to be considered valid.  {Default: 3}
            species {List[string]} -- list of accepeted species. If None take all.  {Default: None}
    """
    # compute
    with open(DISTR_FILE,'wt') as txtfile:
        txtfile.write('Speed_Avg,Speed_SD,Angle_Avg,Angle_SD,Length\n')
        for dtpath in readpath(ORIG_PATH, minlength=minlength, species=species):
            # disp.display_datepath(dtpath, show=True, block=True)
            dtpath.savecsv(folderpath=mc.REAL_PATH)
            spds = dtpath.speedAeSD
            agls = dtpath.angleAeSD
            line = '{0:.3f},{1:.3f},{2:.2f},{3:.2f},{4}\n'.format(spds[0], spds[1], agls[0], agls[1], dtpath.length)
            txtfile.write(line)
    # save on file overall distributions
    getoveralldistributions(DISTR_FILE)


def getoveralldistributions(distrfile):
    """ returns avg of overall distributions of speed and angle

        Arguments:
            distrfile {string} -- full path to distributions file
    """ 
    # set labels
    speedA = 'Speed Avg'; speedSD = ' Speed SD'
    angleA = ' Angle Avg'; angleSD = ' Angle SD'
    # Read distribution file
    df = pd.read_csv(distrfile)
    # try compute ovg. If it fails it means that has already been computed
    try:
        # Set SD as percentage of Avg
        df[speedSD] = df.apply(lambda row: row[speedSD] / float(row[speedA]), axis=1)
        df[angleSD] = df.apply(lambda row: row[angleSD] / row[angleA], axis=1)
        # Compute means
        spdA = df[speedA].mean()
        spdSD = df[speedSD].mean() * spdA
        aglA = df[angleA].mean()
        aglSD = df[angleSD].mean() * aglA
        # add values to last row
        with open(DISTR_FILE,'a') as txtfile:
            txtfile.write('--------,--------,----------,-----------------,\n')
            meanline = '{0:.3f},{1:.3f},{2:.2f},{3:.2f}\n'.format(spdA, spdSD, aglA, aglSD)
            txtfile.write(meanline)
        print(meanline)
    except:
        print(df.tail(1))


# EXECUTE
getoveralldistributions(DISTR_FILE)
