"""Simulation Script

Creates the simulated dataset

Variables:
    fruits {dataframe} -- fruit tree coords and height
    island {array} -- island with height info for each coordinate
"""
from matplotlib import pyplot as plt
import dataparser as dp
import geometry as geo

fruits = dp.parsefruittree()
island = dp.parseisland()
moves = dp.getmonkey()

monkey = moves.monkeys()[0]
dates = moves.dates(monkey)

# Display all dates of first monkey
# step = 255 / dates.size
# c = 0
# for d in dates:
#     points = moves.points(monkey, d)
#     plt.scatter(points.x, points.y, s=1, c="#%02X%02X%02X" % (c, c, c))
#     c = int(c + step)

mvs = moves.points(monkey, dates[0])
red = geo.reduce_path(mvs, 20)

plt.imshow(island)
plt.plot(mvs.x, mvs.y, c="#FF0000")
plt.plot(red.x, red.y, c="#00FF00")
plt.show()
