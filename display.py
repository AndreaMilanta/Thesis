"""Visualization Script for testing

Displays trees and paths. Used for testing

Variables:
    Fruits {dataframe} -- fruit tree coords and height
    Island {array} -- island with height info for each coordinate
"""
from matplotlib import pyplot as plt
from colormap import rgb2hex
import dataparser as dp
import monkeyconstants as mc


def pause(time):
    """ pause execution for a variable amount of time. Allows figure to be displayed
    
        Arguments:
            time {float} -- amount of time to pause in seconds.
    """
    plt.pause(time)


def showfig(block=True):
    """ shows waiting figures

        Keyword Arguments_
            block {boolean} -- whether the figure blocks the excution.  {Default: True}
    """
    if block:
        plt.ioff()
    else:
        plt.ion()
    plt.show()
    pause(0.01)


def display(path, index=None, color='#FFFFFF', show=True, block=True):
    """ displays a path

    if a valid index is passed, the image is created with the island as background

    Arguments:
        path {List[Coordinates]} -- List of coordinates of the path

    Keyword Arguments;
        index {int} -- index of  figure. If is None no figure is created and Island is set as background.  {Default: None}
        [color {'hex'} -- color of displayed path.  {Default: '#FFFFFF'}]
        [color {[int,int,int]} -- color of displayed path. is converted to HEX]
        show {boolean} -- whether to immediately show the figure or not.  {Default: True}
        block {boolean} -- whether the figure blocks the excution.  {Default: True}
    """
    # reset block
    plt.ioff()
    # create figure if requested and add Island background
    if index is not None:
        display_island(index, show=False)
    # convert color to hex
    if not isinstance(color, str):
        color = rgb2hex(color[0], color[1], color[2])
    # display
    px = list(p.x for p in path)
    py = list(p.y for p in path)
    plt.plot(px, py, color)
    plt.draw()
    # optional
    if show:
        showfig(block)


def display_island(index=None, f_color='#FFFFFF', show=True, fruits=None, fruitsize=mc.FRUIT_RADIUS, block=True):
    """ displays the island with fruit trees

    Keyword Arguments;
        index {int} -- index of  figure. If it is None no figure is created.  {Default: None}
        [f_color {'hex'} -- color of fruit trees.  {Default: '#FFFFFF'}]
        [f_color {[int,int,int]} -- color of fruit trees. is converted to HEX]
        show {boolean} -- whether to immediately show the figure or not.  {Default: True}
        fruits {List[Coordinates]} -- list of fruits
        block {boolean} -- whether the figure blocks the excution.  {Default: True}
    """
    # reset block
    plt.ioff()
    # create figure if requested
    if index is not None:
        plt.figure(index)
    # convert color to hex
    if not isinstance(f_color, str):
        f_color = rgb2hex(f_color[0], f_color[1], f_color[2])
    #  display island
    plt.imshow(dp.Island().transpose())
    # display fruits
    if fruits is not None:
        display_fruits(fruits, color=f_color, show=False, fruitsize=fruitsize, block=False)
    plt.axis('scaled')
    plt.draw()
    # optional
    if show:
        showfig(block)


def display_fruits(fruits, index=None, color='#FFFFFF', show=True, fruitsize=mc.FRUIT_RADIUS*2, block=True):
    """ displays fruit trees

        Arguments:
            fruits {Coordinates / List[Coordinates]} -- fruit/fruits to be displayed

        Keyword Arguments:
            index {int} -- index of  figure. If it is None no figure is created.  {Default: None}
            [f_color {'hex'} -- color of fruit trees.  {Default: '#FFFFFF'}]
            [f_color {[int,int,int]} -- color of fruit trees. is converted to HEX]
            show {boolean} -- whether to immediately show the figure or not.  {Default: True}
            block {boolean} -- whether the figure blocks the excution.  {Default: True}
    """
    # reset block
    plt.ioff()
    # create figure if requested
    if index is not None:
        plt.figure(index)
    # convert color to hex
    if not isinstance(color, str):
        color = rgb2hex(color[0], color[1], color[2])
    # display fruits
    if fruits is not None:
        ax = plt.gca()
        if fruits is list:
            for f in fruits:
                c = plt.Circle((f.x, f.y), radius=fruitsize/2, color=color)
                ax.add_patch(c)
        else:
            c = plt.Circle((fruits.x, fruits.y), radius=fruitsize/2, color=color)
            ax.add_patch(c)
    plt.axis('scaled')
    plt.draw()
    # optional
    if show:
        showfig(block)


# def display_memory(path, index, newfig=False):
#     if newfig:
#         plt.figure(index)
#     ax = plt.gca()
#     plt.imshow(island.transpose())
#     color = [0, 0, 0]
#     color_step = 255 / len(path.path())
#     for mvs in path.toMultipleDf():
#         drt = geo.reduce_path(mvs[1], mc.REDUCTION_RADIUS)
#         orig = mvs[1].iloc[0]
#         dest = mvs[1].iloc[-1]
#         plt.axis('scaled')
#         plt.scatter(orig.x, orig.y, s=50, c="#0000FF")
#         plt.scatter(dest.x, dest.y, s=50, c="#FF0000")
#         plt.plot(drt.x, drt.y, c=rgb2hex(int(color[0]), int(color[1]), int(color[2])))
#         color[0] = color[0] + color_step
#         color[1] = color[1] + color_step
#         color[2] = color[2] + color_step
#     for f in fruits:
#         c = plt.Circle((f.x, f.y), radius=7, color="#FFFFFF")
#         ax.add_patch(c)
#     plt.draw()


# def display_view(path, index, newfig=False):
#     if newfig:
#         plt.figure(index)
#     ax = plt.gca()
#     plt.imshow(island.transpose())
#     color_step = 255 / len(path.path())
#     color = [color_step, color_step, color_step]
#     i = 0
#     for mvs in path.toMultipleDf():
#         rdm = mvs[0]
#         drt = mvs[1]
#         rdm = geo.reduce_path(rdm, mc.REDUCTION_RADIUS)
#         drt = geo.reduce_path(drt, mc.REDUCTION_RADIUS)
#         orig = mvs[1].iloc[0]
#         dest = mvs[1].iloc[-1]
#         view = mvs[1].iloc[0]
#         plt.axis('scaled')
#         plt.scatter(orig.x, orig.y, s=50, c="#0000FF")
#         plt.scatter(dest.x, dest.y, s=50, c="#FF0000")
#         plt.scatter(view.x, view.y, s=50, c="#00FF00")
#         plt.plot(rdm.x, rdm.y, c=rgb2hex(int(color[0]), int(color[1]), int(color[2])))
#         plt.plot(drt.x, drt.y, c=rgb2hex(int(color[0]), 0, 0))
#         color[0] = color[0] + color_step
#         color[1] = color[1] + color_step
#         color[2] = color[2] + color_step
#         i += 1
#     for f in fruits:
#         c = plt.Circle((f.x, f.y), radius=7, color="#FFFFFF")
#         ax.add_patch(c)
#     plt.draw()


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
