from datetime import datetime

from constants import HOME, Tiletype, ProgramParameters
from simulation import Simulation

n_success = 0
s = Simulation()

i = 0
while True:
    start_time = datetime.now()
    s.run()
    end_time = datetime.now()
    time_taken = end_time - start_time
    with open(HOME + 'data/results', 'a+') as f:
        print(n_success, i + 1, file=f, sep='\t')
    with open(HOME + 'data/iter_times', 'a+') as f:
        print(i + 1, time_taken, file=f, sep='\t')
    with open(HOME + 'data/agent_histories/history' + str(i), 'w') as f:
        a = s.env.all_agents[0]
        print(*a.location_history, file=f, sep='\n')        
    with open(HOME + 'data/map_obstacles', 'w') as f:
        o = s.env.world_map[Tiletype.OBSTACLE].nodes()
        print(*o, file=f, sep='\n')
    if ProgramParameters.IS_TEST:
        break
    s.randomize()
    i += 1