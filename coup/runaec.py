#!/usr/bin/env python3

from coup_env.coup_env import CoupEnv
from collections import defaultdict

import numpy as np





env = CoupEnv(n_players=2)
env.reset(seed=42)


for agent in env.agent_iter():
    print("\n\n")
    observation, reward, termination, truncation, info = env.last()
    if reward == 1:
        print("Game  End")
        assert False
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action_mask = info['action_mask']
        action = env.action_space(agent).sample(action_mask) 
    env.step(action)
env.close()
    
    
    
    

    
    
    
# if __name__ == "__main__":
#     main()