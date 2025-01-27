# A Multi-agent RL environment for COUP

This repo contains a [pettingzoo](https://pettingzoo.farama.org/index.html) and [gymnasium](https://gymnasium.farama.org) compliant environment for multi-agent reinforcement learning

The step function for the environment leverages previously designed assets from my object-orientated instantiation of the game's logic [found here](https://github.com/AlexAdrian-Hamazaki/COUP)

### Usage

Basic usage of the environment is as follows. This pattern is compliant with pettingzoo environment requirements.

```
from coup_env.coup_env import CoupEnv
env = CoupEnv(n_players=2)
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action_mask = info['action_mask']
        action = env.action_space(agent).sample(action_mask) 
    env.step(action)
env.close()
```

### Future Development

For a 100% correct implementation of the game, some features need to be added (see issues)

