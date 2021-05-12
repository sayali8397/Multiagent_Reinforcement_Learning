import networkx as nx
import random
import numpy as np
from scipy.linalg import sqrtm

from constants import Tiletype, Actions, Constants, ProgramParameters
from utility_functions import get_neighbors, action_from_positions, predict_move, show_color_map, action_to_pos, show_color_array

class Agent:
    id_counter = 0
    CACHE_LIMIT = 10
    
    def __init__(self, position, lifting_capacity, networks):
        self.id = Agent.id_counter
        Agent.id_counter += 1
        self.pos = tuple(position)
        self.lifting_capacity = lifting_capacity

        self.percept = None
        self.state = None
        self.action = None
        self.internal_graph = nx.Graph()
        self.n_new_edges = 0
        self.map_gaussian = None
        self.env = None

        self.box = None
        self.n_lifters = None
        self.how_long_attached = None
        self.box_pickup_location = None

        self.networks = networks

        self.location_history = [self.pos]

    def __repr__(self):
        return '{}\t{}\t{}'.format(
            self.id,
            str(self.pos),
            self.lifting_capacity,
        )
    
    def __str__(self):
        return 'Id: {}\tPosition: {}\tLiftCap: {}'.format(
            self.id,
            str(self.pos),
            self.lifting_capacity,
        )
    
    def act(self, network_input=None):
        """
        Perform action in environment.

        Interact with environment.
        Collect reward.
        Store experience (choice and outcome).
        """   
        network_type = 'box' if self.check_if_attached() else 'explore'
        network = self.networks[network_type]
        if network_input is None:
            network_state_input = self.create_network_state_input()
            network_input = (*network_state_input, *action_to_pos(self.pos, self.action))
        actual_reward = self.interact()
        expected_network_output = (actual_reward, )
        if self.action != 'stay':
            network.add_to_database(network_input, expected_network_output)    

    def calculate_confidence_from_gaussian(self, goal_position):
        """Return confidence between 0 and 1 (using probability)."""
        mu, sigma, _ = self.map_gaussian
        n = mu.shape[0]
        sigma_det = np.linalg.det(sigma)
        if sigma_det == 0:
            confidence = 0
            return confidence
        sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt((2*np.pi)**n * sigma_det)
        diff = goal_position - mu
        fac = np.dot(np.dot(diff.transpose(), sigma_inv), diff)
        confidence = np.exp(-fac / 2) / N
        confidence = max(0.001, confidence)
        return confidence

    def create_network_state_input(self):
        """
        Return a summary of your state.

        NOTE: Do not add action to state summary. 
        """
        common_parameters = [
            self.pos,
            self.percept,
            self.lifting_capacity,
            self.map_gaussian,
            len(self.internal_graph.edges()),
        ]
        if self.is_on_box():
            box = self.env.get_box_by_pos(self.pos)
            try:
                common_parameters += list(box.read_data())
            except AttributeError:
                common_parameters += 4 * [None]            
        else:
            common_parameters += 4 * [None]
        if self.check_if_attached():
            extra_box_paramters = [self.how_long_attached]
            network_state_input = tuple(common_parameters +  extra_box_paramters)
        else:
            extra_explore_parameters = [self.location_history[-Agent.CACHE_LIMIT:]]
            network_state_input = tuple(common_parameters + extra_explore_parameters)
        return network_state_input    

    def explore(self):
        """
        Explore the environment.

        Perceive environment.
        Choose an action to maximize expected reward.
        Carry out action and store experience.
        """
        self.get_percept()
        network_state_input = self.create_network_state_input()
        network = self.networks['explore']
        available_actions = self.find_available_actions()
        self.action, _ = network.choose_best_action(self.pos, network_state_input, available_actions)
        network_input = (*network_state_input, *action_to_pos(self.pos, self.action))
        self.act(network_input)
        if not self.check_if_attached():
            self.location_history.append(self.pos)

    def find_available_actions(self):
        """Use percept to find available actions and return them."""
        agent_pos_in_percept = 1, 1
        available_actions = []
        for action in Actions.move:
            if self.env.is_pos_valid(predict_move(self.pos, action)):
                available_actions.append(action)
        if self.check_if_attached():
            available_actions.append('detach')
        elif self.percept[agent_pos_in_percept] == Tiletype.BOX:
            available_actions.append('attach')
        return available_actions

    def get_percept(self):
        """Get percept from the Environment."""
        self.percept = self.env.generate_percept(self.pos)
        self.update_internal_graph()        
    
    def increment_attachment_parameters(self):
        """Increment parameters for each step spent attached to Box"""
        self.how_long_attached += 1

    def initialize_attachment_parameters(self, box):
        """Set parameters while attaching to Box."""
        self.n_lifters = len(box.agents)
        self.box = box
        self.env.free_agents.remove(self)
        self.box.agents.append(self)
        self.how_long_attached = 0
        self.box_pickup_location = self.pos
        
    def interact(self):
        """Interact with Environment and receive reward."""
        reward = self.env.respond_to_agent(self)
        return reward

    def check_if_attached(self):
        """Return whether you are attached to a Box."""
        return self.box is not None
        
    def is_on_box(self):
        """Return whether you perceive a Box at current position."""
        return self.percept[1, 1] == Tiletype.BOX
    
    def reset_attachment_parameters(self):
        """Reset parameters while detaching from Box"""
        self.how_long_attached = None
        self.box.agents.remove(self)
        self.env.free_agents.append(self)
        self.box = None
        self.n_lifters = None

    def show_internal_graph(self):
        """Generate internal map and print it."""
        internal_graph = np.full(Constants.WORLD_DIMENSION, Tiletype.UNEXPLORED)
        for i, j in self.internal_graph.nodes():
            internal_graph[i, j] = Tiletype.FREE
        internal_graph[self.pos] = Tiletype.AGENT
        show_color_array(internal_graph)
    
    def update_gaussian(self):
        """Recalculate gaussian using visited points."""
        datapoints = [np.array(x[:2]) for x in self.internal_graph.nodes()]
        mu = np.mean(datapoints, axis=0)
        sigma_square = sum(np.outer(d - mu, d - mu) for d in datapoints) / len(datapoints) 
        sigma = sqrtm(sigma_square)
        if any(np.isnan(sigma.flatten())):
            return
        self.map_gaussian = mu, sigma, np.array([len(datapoints)])

    def update_internal_graph(self):
        """Use your percept and position to update internal map."""
        points = []
        percept_range = np.array(self.percept.shape)
        for pos, value in np.ndenumerate(self.percept):
            if value == Tiletype.OBSTACLE:
                continue
            position_in_percept = np.array(pos)
            position_relative_to_agent = position_in_percept - np.array(0.5 * percept_range, dtype=np.int)
            global_position = self.pos + position_relative_to_agent
            points.append(tuple(global_position))

        for point in points:
            neighbors = get_neighbors(point, points)
            for n in neighbors:
                if (point, n) not in self.internal_graph.edges():
                    self.n_new_edges += 1
                self.internal_graph.add_edge(point, n)
        self.update_gaussian()

    def vote(self):
        """
        Cast vote for action on Box and your confidence.

        If you choose to detach form Box, perform action in environment.
        """
        if ProgramParameters.NO_DETACH:
            return self.vote_detach_not_allowed()
        else:
            return self.vote_detach_allowed()

    def vote_detach_allowed(self):
        network_state_input = self.create_network_state_input()
        available_actions = self.find_available_actions()
        self.action, expected_reward = self.networks['box'].choose_best_action(self.pos, network_state_input, available_actions)
        confidence = expected_reward
        if self.action == 'detach':
            network_input = (*network_state_input, *action_to_pos(self.pos, self.action))
            self.act(network_input)
        elif self.box.goal in self.internal_graph.nodes():
            path = nx.shortest_path(self.internal_graph, self.pos, self.box.goal)
            self.action = action_from_positions(*path[:2])
            confidence = 1 + 1 / len(path)
            expected_reward = confidence
        return self.action, confidence

    def vote_detach_not_allowed(self):
        network_state_input = self.create_network_state_input()
        if self.box.goal in self.internal_graph.nodes():
            path = nx.shortest_path(self.internal_graph, self.pos, self.box.goal)
            self.action = action_from_positions(*path[:2])
            confidence = 1 + 1 / len(path)
            expected_reward = confidence
        else:
            available_actions = self.find_available_actions()
            self.action, expected_reward = self.networks['box'].choose_best_action(self.pos, network_state_input, available_actions)
        if self.action == 'detach':
            network_input = (*network_state_input, *action_to_pos(self.pos, self.action))
            self.act(network_input)
        confidence = expected_reward
        return self.action, confidence

if __name__ == '__main__':
    pass
