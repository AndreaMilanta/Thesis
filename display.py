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

def block():
    """ blocks previously shown figures
    """
    plt.ioff()
    plt.show()
    pause(0.01)


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


def display_island(index=None, show=True, block=False):
    """ displays the island with fruit trees

        Keyword Arguments;
            index {int} -- index of  figure. If it is None no figure is created.  {Default: None}
            show {boolean} -- whether to immediately show the figure or not.  {Default: True}
            block {boolean} -- whether the figure blocks the excution.  {Default: True}

        Returns:
            {int} -- number of drawn figure. None if figure has been displayed and blocks
    """
    # reset block
    plt.ioff()
    # create figure if requested
    if index is not None:
        fig = plt.figure(index)

    #  display island
    plt.imshow(dp.Island().transpose())
    plt.axis('scaled')
    plt.draw()
    # optional
    if show:
        showfig(block)
    # return current figure number
    if not block:
        return plt.gcf().number
    else:
        return None


def display_fruits(fruits=None, index=None, color='#FFFFFF', dim='', show=True, block=True):
    """ displays fruit trees

        Keyword Arguments:
            index {int} -- index of  figure. If it is None no figure is created.  {Default: None}
            fruits {Tree / List[Trees]} -- fruit/fruits to be displayed
            color {'hex' / [int,int,int]} -- color of fruit trees.  {Default: '#FFFFFF'}]
            show {boolean} -- whether to immediately show the figure or not.  {Default: True}
            block {boolean} -- whether the figure blocks the excution.  {Default: True}
            dim {string} -- dimension proportionate to radius (default) or score ('score'). {Default: ''}

        Returns:
            {int} -- number of drawn figure. None if figure has been displayed and blocks
    """
    # reset block
    plt.ioff()
    # check fruits
    if fruits is None:
        fruits = dp.Fruits()
    # create figure if requested
    if index is not None:
        plt.figure(index)
    # convert color to hex
    if not isinstance(color, str):
        color = rgb2hex(color[0], color[1], color[2])
    # display fruits
    if fruits is not None:
        ax = plt.gca()
        if isinstance(fruits, list):
            for f in fruits:
                if dim == 'score':
                    c = plt.Circle((f.x, f.y), radius=f.score, color=color)
                else:
                    c = plt.Circle((f.x, f.y), radius=f.radius, color=color)
                ax.add_patch(c)
        else:
            if dim == 'score':
                c = plt.Circle((fruits.x, fruits.y), radius=fruits.score, color=color)
            else:
                c = plt.Circle((fruits.x, fruits.y), radius=fruits.radius, color=color)
            ax.add_patch(c)
    plt.axis('scaled')
    plt.draw()
    # optional
    if show:
        showfig(block)
    # return current figure number
    if not block:
        return plt.gcf().number
    else:
        return None


def display_path(path, index=None, color='#FFFFFF', show=True, block=True):
    """ displays a path if a valid index is passed, the image is created with the island as background

        Arguments:
            path {List[Coordinates]} -- List of coordinates of the path

        Keyword Arguments;
            index {int} -- index of  figure. If is None no figure is created and Island is set as background.  {Default: None}
            color {'hex' / [int,int,int]} -- color of displayed path.  {Default: '#FFFFFF'}]
            show {boolean} -- whether to immediately show the figure or not.  {Default: True}
            block {boolean} -- whether the figure blocks the excution.  {Default: True}

        Returns:
            {int} -- number of drawn figure. None if figure has been displayed and blocks
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
    # return current figure number
    if not block:
        return plt.gcf().number
    else:
        return None


def display_points(points, index=None, color='#FFFFFF', radius=mc.FRUIT_RADIUS, show=True, block=True):
    """ displays fruit trees

        Arguments:
            points {Coordinates / List[Coordinates]} -- Point/Points to be displayed

        Keyword Arguments:
            index {int} -- index of  figure. If it is None no figure is created.  {Default: None}
            color {'hex' / [int,int,int]} -- color of displayed points.  {Default: '#FFFFFF'}]
            show {boolean} -- whether to immediately show the figure or not.  {Default: True}
            block {boolean} -- whether the figure blocks the excution.  {Default: True}

        Returns:
            {int} -- number of drawn figure. None if figure has been displayed and blocks
    """
    # reset block
    plt.ioff()
    # create figure if requested
    if index is not None:
        plt.figure(index)
    # convert color to hex
    if not isinstance(color, str):
        color = rgb2hex(color[0], color[1], color[2])
    # display points
    if points is not None:
        ax = plt.gca()
        if isinstance(points, list):
            for p in points:
                c = plt.Circle((p.x, p.y), radius=radius, color=color)
                ax.add_patch(c)
        else:
            c = plt.Circle((points.x, points.y), radius=radius, color=color)
            ax.add_patch(c)
    plt.axis('scaled')
    plt.draw()
    # optional
    if show:
        showfig(block)
    # return current figure number
    if not block:
        return plt.gcf().number
    else:
        return None


def display_datepath(dtpath, index=None, title=None, radius=mc.FRUIT_RADIUS, show=True, block=True, fruit_c='#00FF00', fruit_dim='', \
                     path_c='#FFFFFF', visited_c='#FF0000', missed_c='#FFFF00', passedby_c='#000000', first_c='#0000FF', last_c='#00FFFF'):
    """ displays datepath in new figure

        Arguments:
            dtpath {datepaht} -- datepath to be displayed

        Keyword Arguments:
            index {int} -- index of  figure. If it is None a new figure is created.  {Default: None}
            title {string} -- title to display. If None show no title. Ignored if figure already exists.  {Default: None}
            fruit_c {'hex' / [int,int,int]} -- color of fruit trees.  {Default: '#00FF00'}]
            fruit_dim {string} -- dimension proportionate to radius (default) or score ('score'). {Default: ''}
            path_c {'hex' / [int,int,int]} -- color of fruit trees.  {Default: '#FFFFFF'}]
            visited_c {'hex' / [int,int,int]} -- color of fruit trees.  {Default: '#FF0000'}]
            missed_c {'hex' / [int,int,int]} -- color of fruit trees.  {Default: '#FFFF00'}]
            passedby_c {'hex' / [int,int,int]} -- color of fruit trees.  {Default: '#000000'}]
            first_c {'hex' / [int,int,int]} -- color of fruit trees.  {Default: '#0000FF'}]
            last_c {'hex' / [int,int,int]} -- color of fruit trees.  {Default: '#00FFFF'}]

            show {boolean} -- whether to immediately show the figure or not.  {Default: True}
            block {boolean} -- whether the figure blocks the excution.  {Default: True}

        Returns:
            {int} -- number of drawn figure. None if figure has been displayed and blocks
    """
    # check if figure exists, if no create and display island and fruit trees
    if index is not None and plt.fignum_exists(index):
        plt.figure(index)
    else:
        display_island(index=index, show=False, block=False)
        display_fruits(color=fruit_c, show=False, block=False)
        if title is not None:
            plt.gcf().suptitle(title)


    # display path and properties
    display_path(dtpath.path, color=path_c, show=False, block=False)
    display_points(dtpath.first, color=first_c, show=False, block=False)
    display_points(dtpath.last, color=last_c, show=False, block=False)
    display_fruits(dtpath.visitedTrees, color=visited_c, dim=fruit_dim, show=False)
    display_fruits(dtpath.missedTrees, color=missed_c, dim=fruit_dim, show=False)
    display_fruits(dtpath.passedbyTrees, color=passedby_c, dim=fruit_dim, show=show, block=block)

    # return current figure number
    if not block:
        return plt.gcf().number
    else:
        return None
