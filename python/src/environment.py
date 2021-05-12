from functools import reduce
import operator
from itertools import groupby
import numpy as np
import networkx as nx
import random

from constants import Tiletype, Actions, Dir_to_coords, Constants, HOME, ProgramParameters
from reward_system import calculate_reward
from utility_functions import array_to_map, predict_move, get_neighbors, colorize_map, map_to_array, shift_origin, show_color_map, map_to_string
from copy import deepcopy
from agent import Agent
from box import Box

class Environment:
    def __init__(self, networks):
        if ProgramParameters.READ_WORLD_FROM_FILE:
            self.world_map, self.dimension = self.read_world_map(HOME + 'data/testing/world_map')
            self.free_agents = self.read_agents_data(HOME + 'data/testing/agents_data', networks)
            self.remaining_boxes = self.read_boxes_data(HOME + 'data/testing/boxes_data')
        else:
            self.dimension = Constants.WORLD_DIMENSION
            self.obstacle_density = Constants.Density.OBSTACLE
            self.area = reduce(operator.mul, self.dimension)
            self.world_map = self.create_random_world_map()
            self.free_agents = self.create_agents(networks)
            self.remaining_boxes = self.create_boxes()
        self.all_agents = self.free_agents[:]
        self.write_current_world()
        self.delivered_boxes = []

    def __repr__(self):
        return '{}\n{}\n{}'.format(
            self.free_agents,
            self.remaining_boxes,
            self.delivered_boxes,
        )
        
    def __str__(self):
        observation_map = self.create_observation_map()
        return colorize_map(observation_map, self.dimension)

    def create_agents(self, networks):
        """Create Agents using Constants and return them."""
        agents = []
        n_agents = int(Constants.Density.AGENT * self.area)
        free_points = list(self.world_map[Tiletype.FREE].nodes())
        for _ in range(n_agents):
            position = random.choice(free_points)
            a = Agent(position, Constants.AGENT_LIFTING_CAPACITY, networks)
            a.env = self
            agents.append(a)
        return agents
    
    def create_boxes(self):
        """Create Boxes using Constants and return them."""
        boxes = []
        n_boxes = int(Constants.Density.BOX * self.area)
        free_points = list(self.world_map[Tiletype.FREE].nodes())
        for _ in range(n_boxes):
            start_position = random.choice(free_points)
            goal_position = random.choice(free_points)
            while start_position == goal_position:
                goal_position = random.choice(free_points)            
            b = Box(start_position, goal_position, Constants.BOX_WEIGHT)
            b.env = self
            boxes.append(b)
        return boxes
        
    def create_random_world_map(self):
        """Shuffle obstacles in obstacle map while maintaining one connected component."""
        rows, cols = self.dimension
        world_map = {
            Tiletype.FREE: nx.grid_graph(dim=[cols, rows]),
            Tiletype.OBSTACLE: nx.Graph(),
        }
        required_n_obstacles = int(self.obstacle_density * self.area)
        while len(world_map[Tiletype.OBSTACLE].nodes()) < required_n_obstacles:
            node = random.choice(list(world_map[Tiletype.FREE].nodes()))
            world_map[Tiletype.FREE].remove_node(node)
            if not nx.is_connected(world_map[Tiletype.FREE]):
                free_neighbors = get_neighbors(node, world_map[Tiletype.FREE].nodes())
                free_edges = [(n, node) for n in free_neighbors]        
                world_map[Tiletype.FREE].add_edges_from(free_edges)
                continue
            world_map[Tiletype.OBSTACLE].add_node(node)
            obstacle_neighbors = get_neighbors(node, world_map[Tiletype.OBSTACLE].nodes())
            obstacle_edges = [(n, node) for n in obstacle_neighbors]
            world_map[Tiletype.OBSTACLE].add_edges_from(obstacle_edges)
        return world_map

    def exchange_graphs(self):
        """Have all agents in a location share their internal graphs."""
        def get_agent_position(agent):
            return agent.pos
    
        agents = sorted(self.all_agents, key=get_agent_position)
        for _, g in groupby(agents, get_agent_position):
            group = list(g)
            graphs = [a.internal_graph for a in group]
            group_graph = nx.compose_all(graphs)
            for a in group:
                a.internal_graph = group_graph.copy()

    def generate_percept(self, pos):
        """Generate 3x3 percept using position in your observation map."""
        percept = np.full((3, 3), Tiletype.OBSTACLE, dtype=np.unicode)
        for t, graph in self.world_map.items():
            global_neighbors = get_neighbors(pos, graph.nodes(), Dir_to_coords.oct)
            agent_local_neighbors = [shift_origin(n, pos) for n in global_neighbors]
            numpy_local_neighbors = [shift_origin(n, (-1, -1)) for n in agent_local_neighbors]
            for n in numpy_local_neighbors:
                percept[n] = t
        for a in self.free_agents:
            loc = shift_origin(shift_origin(a.pos, pos), (-1, -1))
            if self.is_pos_in_map(loc, percept.shape):
                percept[loc] = Tiletype.AGENT
        if sum(1 for a in self.free_agents if pos == a.pos) > 1:
            percept[1, 1] = Tiletype.AGENT
        else:
            percept[1, 1] = Tiletype.FREE
        for b in self.remaining_boxes:
            loc = shift_origin(shift_origin(b.pos, pos), (-1, -1))
            if self.is_pos_in_map(loc, percept.shape):
                percept[loc] = Tiletype.BOX
        return percept
    
    def get_box_by_pos(self, pos):
        """Find a Box by its position and return it."""
        for box in self.remaining_boxes:
            if box.pos == pos:
                return box
        return None
   
    def is_pos_in_map(self, pos, dimension=None):
        """Return whether position is within map limits."""
        if dimension is None:
            dimension = self.dimension
        return all(0 <= p < d for p, d in zip(pos, dimension))

    def is_pos_valid(self, pos):
        """Return whether any object can move to position."""
        return self.is_pos_in_map(pos) and pos not in self.world_map[Tiletype.OBSTACLE].nodes()
    
    def move(self, thing, direction):
        """Move thing to destination if valid."""
        new_pos = predict_move(thing.pos, direction)
        if self.is_pos_valid(new_pos):
            thing.pos = new_pos
        return thing.pos
    
    def read_agents_data(self, filepath, networks):
        """Read Agents' data from file, create them and return them."""
        with open(filepath) as f:
            agents_data = [line for line in f.readlines() if line[0] != '@']
        agents = []
        for line in agents_data:
            parameters = [int(x) for x in line.strip().split()]
            start_position = parameters[:2]
            lifting_capacity = parameters[2]
            agent = Agent(start_position, lifting_capacity, networks)
            agent.env = self
            agents.append(agent)
        return agents

    def read_boxes_data(self, filepath):
        """Read Boxes' data from file, create them and return them."""
        with open(filepath) as f:
            boxes_data = [line for line in f.readlines() if line[0] != '@']
        boxes = []
        for line in boxes_data:
            parameters = [int(x) for x in line.strip().split()]
            start_position = parameters[:2]
            goal_position = parameters[2:4]
            weight = parameters[4]
            box = Box(start_position, goal_position, weight)
            box.env = self
            boxes.append(box)
        return boxes
    
    def read_world_map(self, filepath):
        """Read world map from file and return it."""
        with open(filepath) as f:
            file_data = [list(line.strip()) for line in f.readlines() if line[0] != '@']
        world_array = np.array(file_data)
        dimension = world_array.shape
        world_map = array_to_map(world_array)
        return world_map, dimension
        
    def respond_to_agent(self, agent):
        """Respond to Agent's action and return reward."""
        action = agent.action
        agent.abstract_action = None
        box = None
        if agent.check_if_attached():
            box = agent.box
            if action == 'stay':
                agent.abstract_action = 'stay with box'
            elif action in Actions.move:
                if agent.box.is_voting_step:
                    agent.box.submit_vote(action, agent.confidence)
                    agent.abstract_action = 'stay with box'
                else:
                    self.move(agent, action)
                    if agent.pos == agent.box.goal:
                        agent.abstract_action = 'delivery successful'
                    else:
                        agent.abstract_action = 'move with box'
                agent.increment_attachment_parameters()
            else:
                agent.reset_attachment_parameters()
                agent.abstract_action = 'detach'
        else:
            if action in Actions.move:
                new_pos = self.move(agent, action)
                if new_pos in agent.location_history[-Agent.CACHE_LIMIT:]:
                    agent.abstract_action = 'revisit cached location'
                else:
                    agent.abstract_action = 'move alone'
            else:
                box = self.get_box_by_pos(agent.pos)
                if box is not None:
                    agent.initialize_attachment_parameters(box)
                    agent.abstract_action = 'attach'
                else:
                    agent.abstract_action = 'move alone'
        n_new_edges = agent.n_new_edges
        agent.n_new_edges = 0
        reward = calculate_reward(agent, box, n_new_edges)

        return reward
    
    def create_observation_map(self):
        """Update your world and observation maps."""
        observation_map = deepcopy(self.world_map)
        observation_map[Tiletype.AGENT] = nx.Graph()
        observation_map[Tiletype.BOX] = nx.Graph()
        for agent in self.free_agents:
            observation_map[Tiletype.AGENT].add_node(agent.pos)
        for box in self.remaining_boxes:
            observation_map[Tiletype.BOX].add_node(box.pos)
        return observation_map

    def write_current_world(self):
        with open(HOME + 'data/current_world/world_map', 'w') as f:
            observation_map = self.create_observation_map()
            string = map_to_string(observation_map, self.dimension)
            for tiletype in [Tiletype.AGENT, Tiletype.BOX]:
                string = string.replace(tiletype, Tiletype.FREE)
            print(string, file=f)

        with open(HOME + 'data/current_world/agents_data', 'w') as f:
            print('@ Start position, lifting capacity', file=f)
            for a in self.all_agents:
                print(*a.pos, a.lifting_capacity, file=f)
        
        with open(HOME + 'data/current_world/boxes_data', 'w') as f:
            print('@ Start position, goal position, lifting capacity', file=f)
            for b in self.remaining_boxes:
                print(*b.pos, *b.goal, b.weight, file=f)