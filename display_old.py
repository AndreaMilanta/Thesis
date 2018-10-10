"""Visualization Script for testing

Displays trees and paths. Used for testing

Variables:
    fruits {dataframe} -- fruit tree coords and height
    island {array} -- island with height info for each coordinate
"""
from matplotlib import pyplot as plt
from datetime import date
# import math
import numpy as np
from colormap import rgb2hex

import monkeyconstants as mc
import monkeyexceptions as me
import geometry as geo
import dataparser as dp
import simulation as sim
import monkeyfile as mf

NUM_TRIES = 2
FIRST_IS_VIEW = False


# date = dt.datetime(1995, 1, 19, 8, 0, 0, 0)


def display_memory(path, index, newfig=False):
    if newfig:
        plt.figure(index)
    ax = plt.gca()
    plt.imshow(island.transpose())
    color = [0, 0, 0]
    color_step = 255 / len(path.path())
    for mvs in path.toMultipleDf():
        drt = geo.reduce_path(mvs[1], mc.REDUCTION_RADIUS)
        orig = mvs[1].iloc[0]
        dest = mvs[1].iloc[-1]
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


def display_view(path, index, newfig=False):
    if newfig:
        plt.figure(index)
    ax = plt.gca()
    plt.imshow(island.transpose())
    color_step = 255 / len(path.path())
    color = [color_step, color_step, color_step]
    i = 0
    for mvs in path.toMultipleDf():
        rdm = mvs[0]
        drt = mvs[1]
        rdm = geo.reduce_path(rdm, mc.REDUCTION_RADIUS)
        drt = geo.reduce_path(drt, mc.REDUCTION_RADIUS)
        orig = mvs[1].iloc[0]
        dest = mvs[1].iloc[-1]
        view = mvs[1].iloc[0]
        plt.axis('scaled')
        plt.scatter(orig.x, orig.y, s=50, c="#0000FF")
        plt.scatter(dest.x, dest.y, s=50, c="#FF0000")
        plt.scatter(view.x, view.y, s=50, c="#00FF00")
        plt.plot(rdm.x, rdm.y, c=rgb2hex(int(color[0]), int(color[1]), int(color[2])))
        plt.plot(drt.x, drt.y, c=rgb2hex(int(color[0]), 0, 0))
        color[0] = color[0] + color_step
        color[1] = color[1] + color_step
        color[2] = color[2] + color_step
        i += 1
    for f in fruits:
        c = plt.Circle((f.x, f.y), radius=7, color="#FFFFFF")
        ax.add_patch(c)
    plt.draw()


island = dp.parseisland()
geo.ISLAND = island
# fruits = sim.buildFruitTree(30, sim.Distr.UNIFORM)
fruits = dp.parsefruittree()
# print("fruit tree density: " + '{:3f}'.format(len(fruits) * 1000000 / np.count_nonzero(island)) + " tree / skm")

# DISPLAY ISLAND ONLY
plt.figure(100)
ax = plt.gca()
plt.imshow(island.transpose())
for f in fruits:
    c = plt.Circle((f.x, f.y), radius=7, color="#FFFFFF")
    ax.add_patch(c)

plt.draw()
data = dp.getmonkey()
days = data.toStandard(fruits)
display_memory(days[0])
# monkeys = data.monkeys()
# print("found " + str(len(monkeys)) + " monkeys")
# days = data.dates(monkeys[0])
# print("found " + str(len(days)) + " days")
# points = data.points(monkeys[0], days[0])
# print("found " + str(len(points)) + " points")


# plt.show()

# invalid = 0
# index = 0
# view = FIRST_IS_VIEW
# while(index < NUM_TRIES):
#     try:
#         p = np.random.normal(2000, 500, [1, 2])[0]
#         c = geo.Coordinates(p[0], p[1], island[int(p[0]), int(p[1])])
#         if view:
#             paths = sim.createViewDate(c, fruits, mc.DATE_DURATION_MIN)
#             print("\nworking on " + str(index) + " - VIEW")
#             paths.set_id(index)
#             paths.set_date(date.today())
#             mf.path_to_csv(paths, mc.VIEW_PATH)
#             display_view(paths, index)
#             view = False
#         else:
#             paths = sim.createMemoryDate(c, fruits, mc.DATE_DURATION_MIN, max_mem_range=mc.MAX_MEM_DIST)
#             print("\nworking on " + str(index) + " - MEMORY")
#             paths.set_id(index)
#             paths.set_date(date.today())
#             mf.path_to_csv(paths, mc.MEMORY_PATH)
#             display_memory(paths, index)
#             view = True
#         index += 1
#     except IndexError:
#         print("Invalid = " + str(invalid) + " :Path out of bound")
#         invalid += 1
#     except me.PathOnWaterException:
#         print("Invalid = " + str(invalid) + " :Path on water")
#         invalid += 1

# strMem = []
# strRDM = []
# for i in range(0, NUM_TRIES):
#     if i % 2 == 0:
#         path = mc.MEMORY_PATH + str(i) + "_20180320"
#     else:
#         path = mc.VIEW_PATH + str(i) + "_20180320"
#     file = path + ".csv"
#     infofile = path + "_info.csv"
#     dtpath = dp.parseSTDDate(file, infofile)
#     if i % 2 == 0:
#         strMem.append(dtpath.strRatioAeSD())
#     else:
#         strRDM.append(dtpath.strRatioAeSD())

# arrMem = np.array(strMem)
# print("Memory: " + str(np.mean(arrMem, axis=0)) + ', ' + str(np.std(arrMem, axis=0)))
# arrView = np.array(strRDM)
# print("View: " + str(np.mean(arrView, axis=0)) + ', ' + str(np.std(arrView, axis=0)))

# print("straighratio: " + str(dtpath.strRatioAeSD()))
# print('finished computing read one')
# if(FIRST_IS_VIEW):
#     display_view(dtpath, index)
# else:
#     display_memory(dtpath, index)
# print('finished displaying read one')

plt.show()
