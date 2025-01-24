#!/usr/bin/env python3



from pettingzoo.test import api_test
from coup_env.coup_env import CoupEnv

env = CoupEnv(2)
api_test(env, num_cycles=1000)