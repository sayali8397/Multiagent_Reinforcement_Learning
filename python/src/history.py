import numpy as np
from constants import HOME, Constants, Tiletype
from colors import color
from ast import literal_eval
from collections import Counter

def find_max_footfall(array):
    max_footfall = np.amax(array)
    max_footfall = max(1, max_footfall)
    # _, max_footfall = Counter(history).most_common(1)[0]
    return max_footfall

def show(array, location=(-1, -1)):
    max_footfall = find_max_footfall(array)

    color_array = np.full(array.shape, -1)
    for pos, value in np.ndenumerate(array):
        if value != -1:
            grayscale_value = value / max_footfall * 255
            color_value = grayscale_value
            color_array[pos] = color_value
    color_string = ''
    n_rows, n_cols = color_array.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if (i, j) == location:
                color_string += color(' ', bg=(0, 0, 255))
            elif color_array[i, j] == -1:
                color_string += color(' ', bg=(0, 255, 0))
            else:
                color_string += color(' ', bg=tuple(3 * [color_array[i, j]]))
        color_string += '\n'
    print(max_footfall)
    print(color_string)

with open('data/agent_history') as f:
    lines = f.readlines()
    history = [literal_eval(line.strip()) for line in lines]

with open('data/map_obstacles') as f:
    lines = f.readlines()
    obstacles = [literal_eval(line.strip()) for line in lines]

grid = np.zeros(Constants.WORLD_DIMENSION)
for obstacle in obstacles:
    grid[obstacle] = -1
show(grid)
for location in history:
    grid[location] += 1
    show(grid, location)
    input()