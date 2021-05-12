import operator
import numpy as np
import networkx as nx
from constants import Tiletype, Dir_to_coords, Actions
from colors import color

def action_to_pos(curr_pos, action):
        """Find next position from action applied on current position."""
        attach_or_detach = 0
        new_pos = curr_pos
        if action in Actions.move:
            new_pos = predict_move(curr_pos, action)
        elif action == 'attach':
            attach_or_detach = 1
        elif action == 'detach':
            attach_or_detach = -1
        return attach_or_detach, new_pos

def array_to_string(array):
    """Convert 1d or 2d np.array into a formatted string."""
    assert (len(array.shape) <= 2), 'array_to_string can only convert 1d and 2d numpy arrays'
    lines = []
    for i in range(array.shape[0]):
        line = ''.join([str(x) for x in array[i]])
        lines.append(line)
    string = '\n'.join(lines)
    return string

def colorize_string(string):
    """Add color to string for printing to terminal."""
    color_string = string
    for t, value in Tiletype.to_color.items():
        color_string = color_string.replace(t, color(' ', bg=value))
    return color_string

def show_color_array(array):
    """Convert np.array to colored string and show it."""
    string = array_to_string(array)
    color_string = colorize_string(string)
    print(color_string)

def map_to_array(some_map, dimension):
    """Convert a dict of nx.Graph to a 2d np.array."""
    array = np.full(dimension, Tiletype.UNEXPLORED, dtype=np.unicode)
    for t in Tiletype.print_order:
        if t not in some_map:
            continue
        graph = some_map[t]
        for node in graph.nodes():
            array[node] = t
    return array

def colorize_map(some_map, dimension):
    """Convert a dictionary of nx.Graph into a colored string."""
    array = map_to_array(some_map, dimension)
    string = array_to_string(array)
    color_string = colorize_string(string)
    return color_string

def show_color_map(some_map, dimension):
    """Colorize map and print to terminal."""
    color_string = colorize_map(some_map, dimension)
    print(color_string)

def sub_tuples(tuple1, tuple2):
    """Subtract two tuples and return the result."""
    result = tuple(map(operator.sub, tuple1, tuple2))
    return result

def add_tuples(tuple1, tuple2):
    """Add two tuples and return the result."""
    result = tuple(map(operator.add, tuple1, tuple2))
    return result

def shift_origin(curr_pos, new_origin):
    """Calculate the new coordinates of a position after shifting origin."""
    new_pos = sub_tuples(curr_pos, new_origin)
    return new_pos

def predict_move(curr_pos, direction):
    """Predict next position by moving in given direction from current position."""
    predicted_position = add_tuples(curr_pos, Dir_to_coords.quad[direction])
    return predicted_position

def is_neighbor(point1, point2):
    """Check whether two points are at unit distance from each other."""
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) == 1

def get_neighbors(point, points, actions=Dir_to_coords.quad):
    """Return the neighbors of point in given list of points."""
    my_neighbors = [add_tuples(point, direction) for direction in actions.values()]
    my_neighbors = [n for n in my_neighbors if n in points]
    return my_neighbors

def action_from_positions(curr_pos, next_pos):
    """Return the action required to move from curr_pos to next_pos."""
    for direction in Actions.move:
        if next_pos == predict_move(curr_pos, direction):
            return direction
    return None

def array_to_map(array):
    """Convert a np.array to a dict of nx.Graph."""    
    symbols = set([value for pos, value in np.ndenumerate(array)])
    my_map = dict((symbol, nx.Graph()) for symbol in symbols)
    for pos, value in np.ndenumerate(array):
        graph = my_map[value]
        graph.add_node(pos)
        edges = [(pos, n) for n in get_neighbors(pos, graph.nodes())]
        graph.add_edges_from(edges)
    return my_map

def map_to_string(mymap, dimension):
    """Convert a dict of nx.Graph to a string"""
    return array_to_string(map_to_array(mymap, dimension))

def euclidean_distance(pos1, pos2):
    """Return the Euclidean Distance between the 2 positions."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))