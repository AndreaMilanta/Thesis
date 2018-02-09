"""Visualization Script for testing

Displays trees and paths. Used for testing

Variables:
    fruits {dataframe} -- fruit tree coords and height
    island {array} -- island with height info for each coordinate
"""
from matplotlib import pyplot as plt
import datetime as dt
# import math
import numpy as np
from colormap import rgb2hex

import geometry as geo
import dataparser as dp
import simulation as sim
import monkeyfile as mf

REDUCTION_RADIUS = 5   # radius of cilinder for reducing paths
DATE_DURATION_MIN = 4 * 60  # Duration of day in minutes
MAX_MEM_DIST = 500          # Maximum distance of next fruit tree in memory model (as the crow flies)

geo.HEIGHT_MARGIN = 1
geo.VIEW_MAX_RANGE = 200
geo.VIEW_MIN_RANGE = 20
geo.FOV = 180
date = dt.datetime(1995, 1, 19, 8, 0, 0, 0)


def display_memory(path, index):
    plt.figure(index)
    ax = plt.gca()
    plt.imshow(island.transpose())
    color = [0, 0, 0]
    color_step = 255 / len(path)
    for mvs in paths:
        drt = geo.getDataframe(mvs[0], dt.datetime.now(), sim.DT)
        drt = geo.reduce_path(drt, REDUCTION_RADIUS)
        orig = mvs[0][0]
        dest = mvs[0][-1]
        plt.axis('scaled')
        plt.scatter(orig.x, orig.y, s=50, c="#0000FF")
        plt.scatter(dest.x, dest.y, s=50, c="#FF0000")
        plt.plot(drt.x, drt.y, c=rgb2hex(int(color[0]), int(color[1]), int(color[2])))
        color[0] = color[0] + color_step
        color[1] = color[1] + color_step
        color[2] = color[2] + color_step
    for f in fruits:
        c = plt.Circle((f.x, f.y), radius=7, color="#FFFFFF")
        ax.add_patch(c)
    plt.draw()


def display_view(path, index):
    plt.figure(index)
    ax = plt.gca()
    plt.imshow(island.transpose())
    color_step = 255 / len(path)
    color = [color_step, color_step, color_step]
    for mvs in paths:
        rdm = geo.getDataframe(mvs[0], dt.datetime.now(), sim.DT)
        drt = geo.getDataframe(mvs[1], dt.datetime.now(), sim.DT)
        rdm = geo.reduce_path(rdm, REDUCTION_RADIUS)
        drt = geo.reduce_path(drt, REDUCTION_RADIUS)
        orig = mvs[0][0]
        view = mvs[1][0]
        dest = mvs[1][-1]
        plt.axis('scaled')
        plt.scatter(orig.x, orig.y, s=50, c="#0000FF")
        plt.scatter(dest.x, dest.y, s=50, c="#FF0000")
        plt.scatter(view.x, view.y, s=50, c="#00FF00")
        plt.plot(rdm.x, rdm.y, c=rgb2hex(int(color[0]), int(color[1]), int(color[2])))
        plt.plot(drt.x, drt.y, c=rgb2hex(int(color[0]), 0, 0))
        color[0] = color[0] + color_step
        color[1] = color[1] + color_step
        color[2] = color[2] + color_step
    for f in fruits:
        c = plt.Circle((f.x, f.y), radius=7, color="#FFFFFF")
        ax.add_patch(c)
    plt.draw()


island = dp.parseisland()
geo.ISLAND = island
# fruits = sim.buildFruitTree(10, sim.Distr.UNIFORM)
fruits = dp.parsefruittree()
# print("fruit tree density: " + '{:3f}'.format(len(fruits) * 1000000 / np.count_nonzero(island)) + " tree / skm")

# monkey = moves.monkeys()[0]
# dates = moves.dates(monkey)

# Display all dates of first monkey
# step = 255 / dates.size
# c = 0
# for d in dates:
#     points = moves.points(monkey, d)
#     plt.scatter(points.x, points.y, s=1, c="#%02X%02X%02X" % (c, c, c))
#     c = int(c + step)

# mvs = moves.points(monkey, dates[2])
# red = geo.reduce_path(mvs, REDUCTION_RADIUS)

# p1 = geo.Coordinates(2000, 1000, island[2000][1000])
# # p2 = geo.Coordinates(4000, 4000, 0)
# # p = geo.Coordinates(3, 3, 0)
# orig = p1

pts = np.random.normal(2500, 900, [1, 2])
cts = []
invalid = 0
for p in pts:
    try:
        if p[0] < 0 or p[1] < 0 or island[int(p[0]), int(p[1])] <= 0:
            invalid = invalid + 1
            continue
        cts.append(geo.Coordinates(p[0], p[1], island[int(p[0]), int(p[1])]))
    except IndexError:
        invalid = invalid + 1
print("Invalid = " + str(invalid))
index = 0
view = False
for c in cts:
    index = index + 1
    if view:
        print("\nworking on " + str(index) + " - VIEW")
        paths = sim.createViewDate(c, fruits, DATE_DURATION_MIN)
        mf.path_to_csv(paths, 10, date, sim.DT)
        display_view(paths, index)
        view = False
    else:
        print("\nworking on " + str(index) + " - MEMORY")
        paths = sim.createMemoryDate(c, fruits, DATE_DURATION_MIN, max_mem_range=MAX_MEM_DIST)
        mf.path_to_csv(paths, 11, date, sim.DT)
        display_memory(paths, index)
        # view = True
plt.show()
