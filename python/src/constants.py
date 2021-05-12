import os
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
os.chdir('..')
HOME = os.getcwd() + '/'
data_dir = HOME + 'data/'

with open(data_dir + 'config.json') as f:
    config = json.load(f)

class ProgramParameters:
    p = config['program parameters']
    USE_GUI = True if p['use gui'] == 'True' else False
    IS_TEST = True if p['is test'] == 'True' else False
    READ_WORLD_FROM_FILE = True if p['read world from file'] == 'True' else False
    USE_EXPERIENCE_REPLAY = True if p['use experience replay'] == 'True' else False
    EXCHANGE_GRAPHS = True if p['exchange graphs'] == 'True' else False
    NO_DETACH = True if p['no detach'] == 'True' else False
    REVISIT_PENALTY = True if p['revisit penalty'] == 'True' else False
    N_HIDDEN_LAYERS = p['n hidden layers']
    SURFACE_ATTACH = True if p['surface attach'] == 'True' else False
    PARTIAL_DELIVERY = True if p['partial delivery'] == 'True' else False
    PAUSE_TIME = p['pause time']
    STEP_LIMIT = p['step limit']
    del p
    
class Constants:
    BOX_WEIGHT = config['box weight']
    AGENT_LIFTING_CAPACITY = config['agent lifting capacity']
    world = config['world']
    WORLD_DIMENSION = world['rows'], world['cols']
    del world
    class Density:
        densities = config['densities']
        AGENT = densities['agent']
        BOX = densities['box']
        OBSTACLE= densities['obstacle']
        del densities

class Actions:
    move = ['u', 'd', 'l', 'r']
    explore = move + ['attach']
    attached = move + ['detach']
    to_int = {
        'u':0,
        'l':1,
        'd':2,
        'r':3,
        'attach':4,
        'detach':4,
    }

class Dir_to_coords:
    quad = {
        'u':(-1, 0),
        'l':(0, -1),
        'd':(1, 0),
        'r':(0, 1),
    }
    diag = {
        'ul':(-1, -1),
        'ur':(-1, 1),
        'dl':(1, -1),
        'dr':(1, 1),
    }
    oct = {
        **quad,
        **diag,
    }

class Tiletype:
    tiletype = config['tiletype']
    AGENT = tiletype['agent']['symbol']
    BOX = tiletype['box']['symbol']
    FREE = tiletype['free']['symbol']
    OBSTACLE = tiletype['obstacle']['symbol']
    UNEXPLORED = tiletype['unexplored']['symbol']
    to_int = {
        OBSTACLE:0,
        FREE:1,
        AGENT:2,
        BOX:3,
    }
    print_order = [OBSTACLE, FREE, AGENT, BOX]
    to_color = {
        AGENT: tiletype['agent']['color'],
        BOX: tiletype['box']['color'],
        FREE: tiletype['free']['color'],
        OBSTACLE: tiletype['obstacle']['color'],
        UNEXPLORED: tiletype['unexplored']['color'],
    }
    del tiletype