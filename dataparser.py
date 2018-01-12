import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#PATH CONSTANTS
HERE = os.path.dirname(__file__)
FRUIT_PATH = HERE + "\Data\FruitTrees.csv"
ISLAND_PATH = HERE + "\Data\IslandPic.png"

def parsefruittree():
    """Fruit Tree Reader (x,y,h)
    
    Read fruittree dataset and shifts it so that it maps on the island coordinates
    
    Returns:
        dataframe -- each row containes xy coordinates and height of a fruit tree
    """
    DELTA_X = -50
    DELTA_Y = -45
    columns = ['x', 'y', 'h']
    df = pd.read_csv(FRUIT_PATH, sep=' ', header=None, names=columns)
    df.reset_index()
    df.x = df.x.apply(lambda x : x + DELTA_X)
    df.y = df.y.apply(lambda x : x + DELTA_Y)
    return df

#Island Image Reader
def parseisland():
    island_img = plt.imread(ISLAND_PATH)
    island_img = np.flipud(island_img)
    return island_img

#MAIN
fruits = parsefruittree()
# island = parseisland()
# correct = fruits[fruits.apply(lambda f : island[int(f.y), int(f.x)] > 0, axis = 1)]
# incorrect = fruits[fruits.apply(lambda f : island[int(f.y), int(f.x)] <= 0, axis = 1)]


#visualization
# plt.imshow(island, origin='upper')
# plt.scatter(fruits.x, fruits.y,s=2, c="#00FF00")
# plt.scatter(correct.x, correct.y, s=2, c="#00FF00")
# plt.scatter(incorrect.x, incorrect.y,s=2, c="#FF0000")
# plt.show()