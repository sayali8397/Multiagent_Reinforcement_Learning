import pickle
import random
from collections import OrderedDict
from datetime import datetime
import os
import numpy as np
import torch
from torch.autograd import Variable
from copy import deepcopy

from agent import Agent
from constants import Tiletype, Actions, HOME, ProgramParameters
from utility_functions import predict_move, action_to_pos

class Network:
    common_parameters = OrderedDict([
        ('pos', lambda x: np.array(x, dtype=np.float)), # 2 integers
        ('percept', lambda x: np.array(list(map(lambda p: Tiletype.to_int[p], x.flatten())), dtype=np.float)), # 9 characters
        ('lifting_capacity', lambda x: np.array([x], dtype=np.float)), # 1 integer
        ('map_gaussian', lambda x: np.concatenate([x[0], x[1].flatten(), x[2]]) if x is not None else np.array([-50., -50., 1., 0., 0., 1., 1])), # mu = 2 floats, sigma = 4 floats, n_datapoints = 1 int
        ('internal_graph_size', lambda x: np.array([x], dtype=np.float) if x is not None else np.array([0.])), # 1 integer
        ('goal', lambda x: np.array(x, dtype=np.float) if x is not None else np.array([-50., -50.])), # 2 integers
        ('n_lifters', lambda x: np.array([x], dtype=np.float) if x is not None else np.array([0.])), # 1 integer
        ('remaining_weight', lambda x: np.array([x], dtype=np.float) if x is not None else np.array([0.])), # 1 integer
        ('peer_confidence', lambda x: np.array([x], dtype=np.float) if x is not None else np.array([0.])), # 1 float
    ])
    extra_explore_parameters = OrderedDict([
        ('cached_locations', lambda x: np.array([np.array(point, dtype=np.float) for point in x] + [np.array((-1.,-1.))] * (Agent.CACHE_LIMIT - len(x)), dtype=np.float).flatten()), # Agent.CACHE_LIMIT number of tuples
    ])
    extra_box_paramters = OrderedDict([
        ('how_long_attached', lambda x: np.array([x], dtype=np.float)), # 1 integer
    ])
    action_parameters = OrderedDict([
        # ('action', lambda x: np.array([Actions.to_int[x]], dtype=np.float)), # 1 string
        ('attach_or_detach', lambda x: np.array([x], dtype=np.float)), # 1 integer
        ('new_pos', lambda x: np.array(x, dtype=np.float)), # 2 integers
    ])
    
    def __init__(self, network_type):
        self.network_type = network_type
        self.parameters = OrderedDict()
        self.add_parameters(Network.common_parameters)
        if self.network_type == 'explore':
            self.add_parameters(Network.extra_explore_parameters)
            self.actions = Actions.explore
        elif self.network_type == 'box':
            self.add_parameters(Network.extra_box_paramters)
            self.actions = Actions.attached
        else:
            raise Exception('Invalid network type')
        self.add_parameters(Network.action_parameters)
        self.database_filepath = HOME + 'data/database/' + network_type
        self.database = []
        self.model_filepath = HOME + 'data/model_state_dict/' + network_type
        self.model = None
        self.experiences_filepath = HOME + 'data/experiences/' + network_type + '/'

    def __repr__(self):
        return '{}\n{}\n{}\n{}\n{}\n'.format(
            self.network_type,
            self.parameters,
            self.actions,
            self.database_filepath,
            self.model_filepath,
        )
    
    def __str__(self):
        return 'Type: {}\nParameters: {}\nActions: {}\nDb path: {}\nModel path: {}'.format(
            self.network_type,
            self.parameters,
            self.actions,
            self.database_filepath,
            self.model_filepath,
        )

    def add_parameters(self, parameters_to_be_added):
        """Add parameters to your parameter dictionary."""
        for key, value in parameters_to_be_added.items():
            self.parameters[key] = value
   
    def add_to_database(self, network_input, expected_network_output):
        """Add a training instance to database."""
        network_input = self.normalize_network_input(network_input)
        self.database.append((network_input, expected_network_output))

    def choose_best_action(self, agent_pos, network_state_input, available_actions):
        """Choose the action which is believed to yield maximum reward."""
        possible_actions = {}
        for action in available_actions:
            network_input = (*network_state_input, *action_to_pos(agent_pos, action))
            reward = self.evaluate(network_input)
            possible_actions[action] = reward
        std_dev = np.std(np.array(list(possible_actions.values())))
        avg_reward = sum(reward for reward in possible_actions.values()) / len(possible_actions)
        max_reward_action = max(possible_actions, key=(lambda x: possible_actions[x]))
        if possible_actions[max_reward_action] > 2 * std_dev + avg_reward:
            chosen_action = max_reward_action
        else:
            chosen_action = random.choice(list(possible_actions.keys()))
        reward = possible_actions[chosen_action]
        return chosen_action, reward 

    def evaluate(self, network_input):
        """Evaluate the model on given network input and return output of model."""
        data_input = self.normalize_network_input(network_input)
        x = Variable(torch.Tensor(np.array(data_input)))

        try:
            y = self.model(x)
        except TypeError:
            self.setup(data_input)
            y = self.model(x)

        expected_reward = float(y[0])
        return expected_reward

    def experience_replay(self, database):
        """Train from random, shuffled experiences from database."""
        try:
            experiences = [random.choice(database) for _ in range(1000)]
            random.shuffle(experiences)
        except IndexError:
            # no experience yet
            return -1
        loss = self.train(experiences)
        return loss

    def normalize_network_input(self, network_input):
        """Return a normalized, flattened network input."""
        data_input = tuple(map(lambda x, y: x(y), self.parameters.values(), network_input))
        data_input = np.concatenate(data_input)
        return data_input

    def read_database_from_disk(self, filepath):
        """Load database from a file and return it."""
        try:
            database = pickle.load(open(filepath, 'rb'))
        except FileNotFoundError:
            database = []
        return database

    def read_model_from_disk(self):
        """Retrieve model from a file and return it."""
        self.model.load_state_dict(torch.load(self.model_filepath))

    def read_random_experience_file(self):
        """Return database read from random file from experiences directory"""
        file_list = [f for f in os.listdir(self.experiences_filepath) if 'exp' in f]
        try:
            experience_file = random.choice(file_list)
        except IndexError:
            print('No experiences for' + self.network_type + 'network')
            raise
        random_experience_database = self.read_database_from_disk(self.experiences_filepath + experience_file)
        return random_experience_database

    def setup(self, data_input):
        n_in = len(data_input)
        n_hidden = n_in
        n_out = 1 # ouptut is the expected reward (1 float)
        input_layers = [
            torch.nn.Linear(n_in, n_hidden),
            torch.nn.ReLU(),
        ]
        hidden_layers =  [
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),            
        ]
        output_layers = [           
            torch.nn.Linear(n_hidden, n_out),
        ]
        sequential_parameters = input_layers
        for _ in range(ProgramParameters.N_HIDDEN_LAYERS):
            sequential_parameters += deepcopy(hidden_layers)
        sequential_parameters += output_layers
        self.model = torch.nn.Sequential(*sequential_parameters)
        try:
            self.read_model_from_disk()
        except FileNotFoundError:
            pass

    def train(self, experiences):
        """Train the neural network using experiences."""
        try:
            inputs, outputs = zip(*experiences)
        except ValueError:
            print('Experiences for', self.network_type, 'network is empty.')
            return

        # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
        x = Variable(torch.Tensor(np.array(inputs)))        
        y = Variable(torch.Tensor(np.array(outputs)), requires_grad=False)

        # Use the nn package to define our model and loss function.
        loss_fn = torch.nn.MSELoss(size_average=False)
        
        if self.model is None:
            self.setup(inputs[0])
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for t in range(500):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.model(x)

            # Compute loss.
            loss = loss_fn(y_pred, y)
            print('\r\t', self.network_type, 'loss:', float(loss), end='')

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
        print()
        self.write_model_to_disk()
        return float(loss)
    
    def train_from_database(self):
        """Train the neural network using instances from the database."""
        loss = self.train(self.database)
        return loss
    
    def update_database_in_disk(self):
        full_database = self.read_database_from_disk(self.database_filepath)
        full_database += self.database
        self.write_database_to_disk(full_database, self.database_filepath)
        return full_database
    
    def write_current_experience_to_file(self):
        """Save database to experience directory"""
        filepath = self.experiences_filepath + 'exp' + str(datetime.now())
        self.write_database_to_disk(self.database, filepath)

    def write_database_to_disk(self, database, filepath):
        """Save database to a file."""
        pickle.dump(database, open(filepath, 'wb'))

    def write_model_to_disk(self):
        """Save model to a file."""
        torch.save(self.model.state_dict(), self.model_filepath)

if __name__ == 'main':
    train_from_database()
