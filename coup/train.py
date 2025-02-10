########################
# This script is for multi-agent training using AgileRLs HPO multi-agent training/
# Broadly speaking I follow this guide https://docs.agilerl.com/en/latest/multi_agent_training/index.html
########################

# Importing Coup Env
from coup_env.coup_env import CoupEnv
from train_multi_agent import MultiAgentTrainer

import copy
import os
import random
from collections import deque
from datetime import datetime
import yaml

# Utility Imports
import pandas as pd
import numpy as np
from tqdm import trange
import torch
import sys
import copy
import os
import random
from collections import deque
from datetime import datetime
import tensordict
# Agile Imports
import agilerl

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import create_population, observation_space_channels_to_first
from agilerl.components.replay_buffer import ReplayBuffer

# Gymnasium Imports
from gymnasium.spaces.utils import flatten_space, flatdim, flatten
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


def init_net_conf():

    # Define the network configuration
    # Define the network configuration
    NET_CONFIG = {

        "head_config": {
            "hidden_size": [64, 64],  # Actor head hidden size
        },
    }

    return  NET_CONFIG

def init_hyper_conf():
    # Define the initial hyperparameters
        # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 6,
        # "ALGO": "Rainbow DQN",  # Algorithm
        "ALGO": "DQN",  # Algorithm
        "DOUBLE": True,
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "BATCH_SIZE": 256,  # Batch size
        "LR": 1e-4,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 1000,  # Max memory buffer size
        "LEARN_STEP": 1,  # Learning frequency
        "CUDAGRAPHS": False,  # Use CUDA graphs
        "N_STEP": 1,  # Step number to calculate td error
        "PER": False,  # Use prioritized experience replay buffer
        "ALPHA": 0.6,  # Prioritized replay buffer parameter
        "TAU": 0.01,  # For soft update of target parameters
        "BETA": 0.4,  # Importance sampling coefficient
        "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
        "NUM_ATOMS": 51,  # Unit number of support
        "V_MIN": 0.0,  # Minimum value of support
        "V_MAX": 200.0,  # Maximum value of support
    }
    return INIT_HP


def init_hp_config():
        # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=64, dtype=int),
        learn_step=RLParameter(
            min=1, max=120, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )
    return hp_config


 
def main():
    ##############################################################################
    ### Initialization section
    ##############################################################################
    
    ### INIT DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Curriculum Learning Demo =====")
    print(device)
    
    ### Initiate config dictionary for network parameters
    NET_CONFIG = init_net_conf()
    
    ### Initiate config dictionary for training parameters
    INIT_HP = init_hyper_conf()
    
    ### hp config
    hp_config = init_hp_config()    

    num_envs = 8
    ### INSTANTIATE ENVIRONMENT
    # for now just with 2 players
    env = CoupEnv(n_players=2)
    env.reset(seed=42)
    
        
    ### Configure observation Space for training
    
    # Configure the algo input arguments

    action_spaces = [env.action_space(agent) for agent in env.agents]

    assert flatten(env.observation_space(0), env.observation_space(0).sample()) in flatten_space(env.observation_space(0))

    observation_space = flatten_space(env.observation_space(0)['observation'])
    action_space = env.action_space(0)


    print(f"Observation space dim for each agent {observation_space}")
    print(f"Action space dim for each agent {action_space}")
    print("Obs Space")
    print(observation_space)
    print("Action Space")
    print(action_space)

    
    ##############################################################################
    ### Create a population of agents for evolutionary hyper-parameter optimization
    ##############################################################################
    # Create a population ready for evolutionary hyper-parameter optimisation
    pop = create_population(
        INIT_HP["ALGO"],
        observation_space,
        action_spaces[0],
        NET_CONFIG,
        INIT_HP,
        hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        device=device,
    )

    
    
    ##############################################################################
    ### Create a multi agent replay buffer
    ##############################################################################
    
    # each population of agents actually shares memories, its more efficient for exploration
    
    # MultiAgentReplayBuffer.save_to_memory() saves current experience
    # MultiAgentReplayBuffer.sample() samples saved experiences
    
    field_names = ["state", "action", "reward", "next_state", "termination"]
    memory = ReplayBuffer(
        memory_size=INIT_HP["MEMORY_SIZE"],  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
        )
    
    print(f"Instantiated memory buffer class {memory}")


    ##############################################################################
    ### Create a Tournament Selection Class
    ##############################################################################
    
    # This helper class lets us select the ELITE agent in the population, that one survives the epoch
    # then this class also helps us re-fill the population of agents for the next epoch
    
    # TournamentSelection.select()
    
    
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )
    
    print(f"Instantiated tournament class {tournament}")
    
    ##############################################################################
    ### Create a Mutation Class instance
    ##############################################################################
    
    # The mutation class helps us periodically explore the hyperparameter space turing training
    # mutation class will change hyperparameter space for agents in a population
    # if certain hyper parameters prove beneficial in training, then that agent will be more likely to survive the tournament, so they will likely be witheld by the mutation
    
    # you can do cool things like change layers/nodes or modify node weights with noise, or change activation layer
    
    # this is a super usefull class 
    
    # Mutations.mutation() returns a mutated population
    
    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        no_mutation=0.2,  # Probability of no mutation
        architecture=0,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,
        device=device,
    )
    print(f"Instantiated mutations class {mutations}")
    
    
    ##############################################################################
    ### Training loop
    ##############################################################################
        
    with open("/home/aadrian/Documents/RL_projects/RL_COUP/coup/coup_env/lesson1.yaml") as file:
        LESSON = yaml.safe_load(file)
    
    
    multi_agent_trainer = MultiAgentTrainer(
            env=env,  # Pettingzoo-style environment
            pop=pop,  # Population of agents
            elite = pop[0],  # Assign a placeholder "elite" agent
            memory=memory,  # Replay buffer
            INIT_HP=INIT_HP,  # IINIT_HP dictionary
            net_config=NET_CONFIG,  # Network configuration
            tournament=tournament,  # Tournament selection object
            mutations=mutations,  # Mutations object
            action_space=action_space,
            observation_space=observation_space,
            n_players = LESSON['n_players'],
            max_steps=100,  # Max number of training steps
            max_episodes = 10,  # Total episodes
            episodes_per_epoch = 100,
            evo_epochs = 20, # Evolution frequency
            evo_loop = 50, # Number of evaluation episodes
            epsilon = 1.0,  # Starting epsilon value
            eps_end = 0.1,  # Final epsilon value
            eps_decay = 0.9998,  # Epsilon decays
            opp_update_counter = 0,
            env_name='COUP_v0.1',  # Environment name
            algo="DQN",  # Algorithm
            LESSON=LESSON
    )
    
    
    
    print(f"Instantiated Multi agent trainer class {multi_agent_trainer}")

    multi_agent_trainer.train_multi_agent()
    
    
    # trained_pop, pop_fitnesses = train_multi_agent(
    #     env=env,  # Pettingzoo-style environment
    #     env_name='coup',  # Environment name
    #     algo=INIT_HP["ALGO"],  # Algorithm
    #     pop=pop,  # Population of agents
    #     memory=memory,  # Replay buffer
    #     INIT_HP=INIT_HP,  # IINIT_HP dictionary
    #     net_config=NET_CONFIG,  # Network configuration
    #     # swap_channels=INIT_HP['CHANNELS_LAST'],  # Swap image channel from last to first
    #     max_steps=2000000,  # Max number of training steps
    #     evo_steps=10000,  # Evolution frequency
    #     eval_steps=None,  # Number of steps in evaluation episode
    #     eval_loop=1,  # Number of evaluation episodes
    #     learning_delay=1000,  # Steps before starting learning
    #     target=200.,  # Target score for early stopping
    #     tournament=tournament,  # Tournament selection object
    #     mutation=mutations,  # Mutations object
    #     wb=False,  # Weights and Biases tracking
    # )
        
    assert False
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
    

if __name__ == "__main__":
    main()
