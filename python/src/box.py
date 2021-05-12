import operator
from functools import reduce
from constants import Actions, Tiletype

import networkx as nx

class Box:
    id_counter = 0

    def __init__(self, start_position, goal_position, weight):
        self.id = Box.id_counter
        Box.id_counter += 1
        self.start = tuple(start_position)
        self.pos = self.start
        self.goal = tuple(goal_position)
        self.weight = weight
        self.remaining_weight = self.weight
        self.agents = []
        self.votes = dict()
        self.is_voting_step = True
        self.env = None
        self.steps_moved = 0

    def __repr__(self):
        return '{}\t{}\t{}\t{}'.format(
            self.id, 
            self.goal, 
            self.pos, 
            self.remaining_weight,
        )
    
    def __str__(self):
        return 'Id: {}\tGoal: {}\tPos: {}\tRem wt: {}'.format(
            self.id,
            self.goal,
            self.pos,
            self.remaining_weight,
        )
    
    def add_vote(self, action, confidence):
        """If Agent votes to move, add it to your records."""
        if action in Actions.move:
            self.votes[action].append(confidence)
    
    def can_be_moved(self):
        """Return whether lifting capacity of your Agents is sufficient to move you."""
        total_lifting_capacity = sum(a.lifting_capacity for a in self.agents)
        if total_lifting_capacity >= self.weight:
            self.remaining_weight = 0
            return True
        else:
            self.remaining_weight = self.weight - total_lifting_capacity
            return False
 
    def find_best_choice(self):
        options = [(action, sum(confidence)) for action, confidence in self.votes.items()]
        choice = max(options, key=(lambda x: x[1]))
        return choice

    def find_ideal_distance(self):
        path = nx.shortest_path(self.env.world_map[Tiletype.FREE], self.start, self.goal)
        ideal_distance = len(path) - 1
        return ideal_distance
    
    def find_overall_efficiency(self):
        if not self.pos == self.goal:
            return 0
        efficiency = self.find_ideal_distance() / self.steps_moved
        return efficiency

    def majority_direction(self):
        """Return the movement action with highest confidence."""
        choice = self.find_best_choice()
        direction = choice[0]
        return direction

    def max_confidence(self):
        """Return max confidence among all attached Agents."""
        try:
            highest_confidence = max([c for item in self.votes.values() for c in item])
        except ValueError:
            return 0
        return highest_confidence
    
    def min_confidence(self):
        """Return min confidence among all attached Agents."""        
        try:
            lowest_confidence = max([c for item in self.votes.values() for c in item])
        except ValueError:
            return 0
        return lowest_confidence

    def read_data(self):
        """Return public data."""
        return self.goal, len(self.agents), self.remaining_weight, self.max_confidence()
    
    def reset_votes(self):
        """Reset all votes for possible actions."""
        self.votes = dict()
        for action in Actions.move:
            self.votes[action] = []

    def simulate(self):
        """
        Simulate you existence.

        Call for votes from Agents and move accordingly.
        """
        if self.is_voting_step:
            self.reset_votes()
            for agent in self.agents:
                self.add_vote(*agent.vote())
        else:
            group_action = self.majority_direction() if self.can_be_moved() else 'stay'
            if group_action != 'stay':
                self.env.move(self, group_action)
                self.steps_moved += 1
            for agent in self.agents:
                agent.action = group_action
                agent.act()
                agent.get_percept()
            if self.pos == self.goal:
                self.env.remaining_boxes.remove(self)
                self.env.delivered_boxes.append(self)
                while self.agents:
                    a = self.agents[0]
                    a.reset_attachment_parameters()
        self.is_voting_step = not self.is_voting_step