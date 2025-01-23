#!/usr/bin/env python3

from coup_env import CoupEnv
from collections import defaultdict

import numpy as np





env = CoupEnv(n_players=2)
env.reset(seed=42)

print(env._action_space_map)

for agent in env.agent_iter():

    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action_mask = observation['action_mask']
        action = env.action_space(agent).sample(action_mask) 

    env.step(action)
env.close()
    
    
    
    

    
    
    
if __name__ == "__main__":
    main()