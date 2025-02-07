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

# Agile Imports
from agilerl.utils.utils import create_population
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
# from agilerl.training.train_multi_agent import train_multi_agent

# Gymnasium Imports
from gymnasium.spaces.utils import flatten_space, flatdim, flatten
import gymnasium as gym


def init_net_conf():
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [32, 32], } # Actor hidden size}
    return  NET_CONFIG

def init_hyper_conf():
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
    "MEMORY_SIZE": 100000,  # Max memory buffer size
    "LEARN_STEP": 1,  # Learning frequency
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
    
    ### INSTANTIATE ENVIRONMENT
    # for now just with 2 players
    env = CoupEnv(n_players=2)
    env.reset(seed=42)
    
    ### Configure observation Space for training
    obs_space_dim = [flatdim(env.observation_space(agent)["observation"]) for agent in env.agents][0]
    obs_space_dim = [obs_space_dim] # make it a tuple
    ### Configure action space for training
    act_space_dim = [flatdim(env.action_space(agent)) for agent in env.agents][0]
    
    print(f"Observation space dim for each agent {obs_space_dim}")
    print(f"Action space dim for each agent {act_space_dim}")
    

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.n_players
    INIT_HP["AGENT_IDS"] = env.agents
    
    
    ##############################################################################
    ### Create a population of agents for evolutionary hyper-parameter optimization
    ##############################################################################

    # Best model will be selected from population to keep
    pop = create_population(
                            algo = INIT_HP["ALGO"],
                            state_dim = obs_space_dim,
                            action_dim = act_space_dim,
                            one_hot = False,
                            net_config = NET_CONFIG,
                            INIT_HP = INIT_HP,
                            population_size = INIT_HP['POPULATION_SIZE'],
                            device=device
                            )
    
    
    ##############################################################################
    ### Create a multi agent replay buffer
    ##############################################################################
    
    # each population of agents actually shares memories, its more efficient for exploration
    
    # MultiAgentReplayBuffer.save_to_memory() saves current experience
    # MultiAgentReplayBuffer.sample() samples saved experiences
    
    field_names = ["state", "action", "reward", "next_state", "termination"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
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
        algo=INIT_HP["ALGO"],
        no_mutation=0.2,  # Probability of no mutation
        architecture=0,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        rl_hp_selection=[
                "lr",
                "learn_step",
                "batch_size",
        ],  # RL hyperparams selected for mutation
        mutation_sd=0.1,  # Mutation strength
        # Define search space for each hyperparameter
        min_lr=0.0001,
        max_lr=0.01,
        min_learn_step=1,
        max_learn_step=120,
        min_batch_size=8,
        max_batch_size=64,
        arch=NET_CONFIG["arch"],  # MLP or CNN
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
            action_dim=act_space_dim,
            state_dim=obs_space_dim,
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
