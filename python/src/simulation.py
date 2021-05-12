import os
import time
import random
from collections import OrderedDict

import numpy as np
import networkx as nx

from agent import Agent
from box import Box
from constants import Tiletype, HOME, Constants, ProgramParameters
from environment import Environment
from network import Network
from utility_functions import array_to_map, show_color_map

class Simulation:
    def __init__(self, pause_time=ProgramParameters.PAUSE_TIME, step_limit=ProgramParameters.STEP_LIMIT):
        self.pause_time = pause_time
        self.step_limit = step_limit
        self.networks = OrderedDict([
            ('explore', Network('explore')),
            ('box', Network('box')),
        ])
        self.env = Environment(self.networks)

    def __repr__(self):
        return '{}\n{}\n{}\n'.format(
            self.pause_time,
            self.networks,
            self.env,
        )
    
    def __str__(self):
        return 'Real step time: {}\nNetworks:\n{}\nEnv:\n{}\n'.format(
            self.pause_time,
            '\n'.join(str(n) for n in self.networks.values()),
            repr(self.env),
        )
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    
    def find_overall_efficiency(self):
        if not self.env.delivered_boxes:
            return 0
        efficiencies = [box.find_overall_efficiency() for box in self.env.delivered_boxes]
        average_efficiency = sum(efficiencies) / len(efficiencies)
        return average_efficiency

    def randomize(self):
        """Reset yourself and create new Env."""
        for network in self.networks.values():
            network.database = []
        self.env = Environment(self.networks)

    def run(self):
        """
        Run the Simulation.

        Until all Boxes are delivered, loop simulate all Agents and Boxes repeatedly.
        Once all Boxes are delivered, store training instances to a file.
        """
        counter = 0
        self.clear_screen()
        while self.env.remaining_boxes and counter < self.step_limit:
            if ProgramParameters.USE_GUI:
                self.clear_screen()
                print(self.env)
            counter += 1
            print('\r', counter, len(self.env.delivered_boxes), self.find_overall_efficiency(), end='')
            if ProgramParameters.USE_GUI:
                self.wait()
            for agent in self.env.free_agents:
                agent.explore()
            for box in self.env.remaining_boxes:
                box.simulate()
            if ProgramParameters.EXCHANGE_GRAPHS:
                self.env.exchange_graphs()
        print('\r', ' ' * 80, '\rTime taken:', counter)
        print('Efficiency:', self.find_overall_efficiency())
        if not self.env.remaining_boxes:
            print('They did it!')
        # train on all examples of current experience
        training_losses = []
        print('Instantaneous learing')
        for network in self.networks.values():
            training_loss = network.train_from_database()
            training_losses.append(training_loss)
        # write it to file
        with open(HOME + 'data/losses', 'a+') as f:
            print(counter, file=f, end=' ')
            print(*training_losses, file=f, sep='\t')
        if ProgramParameters.USE_EXPERIENCE_REPLAY:
            # read instances for experience replay from file
            replay_losses = []
            print('Experience replay')
            # add current experiences to pickled file of all experiences
            for network in self.networks.values():
                network.write_current_experience_to_file()
                try:
                    replay_database = network.read_random_experience_file()
                    replay_loss = network.experience_replay(replay_database)
                    replay_losses.append(replay_loss)
                except IndexError:
                    pass
            # write it to file
            with open(HOME + 'data/replay_losses', 'a+') as f:
                print(counter, file=f, end=' ')
                print(*replay_losses, file=f, sep='\t')
    
    def wait(self):
        """Sleep for prescribed time."""
        time.sleep(self.pause_time)
