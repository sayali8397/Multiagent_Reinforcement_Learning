from utility_functions import euclidean_distance
from constants import ProgramParameters

rewards = {
    'move alone': -1, # per step
    # 'hit obstacle': -10, # per step
    'move with box': -2, # -0.5, # half of explore cost
    'stay with box': -1, # -0.25, # quarter of explore cost
    'delivery successful': None, # one time reward proportional to the euclidean distance between box source and destination
    'attach': None, # one time reward
    'detach': None, # one time reward
    'revisit cached location': (-1000 if ProgramParameters.REVISIT_PENALTY else -10), # per step
}

profit_factors = {
    'delivery successful': 1000,
    'detach': 10,
    'add new edge': 0.1,
}

def calculate_box_reward(box, abstract_action):
    """
    Return the reward related to Box based on Agent's abstract action.
    
    The reward is proportional to the euclidean distance between Box's source and destination.
    This scalable reward will motivate Agents (to carry a Box over large distances) better than a constant reward.
    """
    reward = profit_factors[abstract_action] * euclidean_distance(box.start, box.goal)
    return reward

def calculate_reward(agent, box=None, n_new_edges=0):
    """Return reward to an Agent based on its abstract action."""
    if agent.abstract_action == 'attach':
        max_peer_confidence = box.max_confidence()
        box_remaining_weight = box.remaining_weight
        confidence = agent.calculate_confidence_from_gaussian(box.goal)
        reward = attachment_reward(confidence, max_peer_confidence, box_remaining_weight)
    elif ProgramParameters.PARTIAL_DELIVERY and agent.abstract_action == 'detach':
        distance_remaining = euclidean_distance(agent.pos, box.goal)
        total_distance_to_goal = euclidean_distance(agent.box_pickup_location, box.goal)
        reward = -10 + profit_factors['detach'] * (total_distance_to_goal - distance_remaining)
    elif agent.abstract_action in ['delivery successful', 'detach']:
        if agent.abstract_action == 'detach':
            reward = -10
        else:
            try:
                reward = calculate_box_reward(box, agent.abstract_action)
            except AttributeError:
                raise Exception('While calculating box reward, box object was None')
    else:
        reward = rewards[agent.abstract_action]
    reward += profit_factors['add new edge'] * n_new_edges
    return reward

def attachment_reward(confidence, max_peer_confidence, box_remaining_weight):
    if not ProgramParameters.SURFACE_ATTACH:
        return 10

    reward_upper_limit = 500
    normalization_coef = 1 / 100

    def exp_pos(x, y):
        c = 4
        try:
            reward = -1 + 2 ** (c * x * (x - y) ** 0.2)
        except OverflowError:
            reward = reward_upper_limit
        return reward

    x, y = confidence, max_peer_confidence
    if box_remaining_weight > 0:
        # even function
        x, y = max(x, y), min(x, y)
        reward = exp_pos(x, y)
    else:
        # odd function
        if x > y:
            reward = exp_pos(x, y)
        else:
            reward = -1 * exp_pos(y, x)
    reward = max(reward, -reward_upper_limit)
    reward = min(reward, reward_upper_limit)
    scaled_reward = normalization_coef * reward
    return scaled_reward
