#!/usr/bin/env python3

from coup_env import CoupEnv

def main():

    env = CoupEnv(4)
    env.reset()

    action = 9
    observation, reward, terminated, truncated, info = env.step(9)
    
    

    
    
    
if __name__ == "__main__":
    main()