#!/usr/bin/env python3

from coup_env import CoupEnv

def main():
    print("Game Begin")

    env = CoupEnv(4)
    env._get_obs()
    
    
    
if __name__ == "__main__":
    main()