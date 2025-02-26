
# Importing Coup Env
from coup_env.coup_env import CoupEnv
from coup_env.coup_player import CoupPlayer
from opponent import Opponent, RandomOpponent

import copy
import os
import random
from collections import deque
from datetime import datetime

# Utility Imports
import json
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import torch
import sys
import copy
import os
import random
from collections import deque
from datetime import datetime
from gymnasium.spaces.utils import flatten, unflatten


# Agile imports
from agilerl.algorithms.dqn import DQN

class MultiAgentTrainer:
    def __init__(self,
                env,  # Pettingzoo-style environment
                pop=None,  # Population of agents
                elite=None,  # Assign a placeholder "elite" agent
                memory=None,  # Replay buffer
                INIT_HP=None,  # IINIT_HP dictionary
                net_config=None,  # Network configuration
                hp_config=None,
                tournament=None,  # Tournament selection object
                mutations=None,  # Mutations object
                action_space=None,
                observation_space=None,
                LESSON=None,
                n_players=None,
                device=None,
            ):
        
        self.env = env
        self.pop = pop
        self.elite = elite
        self.memory = memory
        self.INIT_HP = INIT_HP
        self.net_config = net_config
        self.hp_config = hp_config
        self.tournament = tournament
        self.mutations = mutations
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_players = n_players
        self.device = device
        
        self.LESSON = LESSON
        self.epochs = LESSON["epochs"]
        self.episodes_per_epoch = LESSON["episodes_per_epoch"]
        self.evo_epochs = LESSON["evo_epochs"]
        self.n_evaluations = LESSON["n_evaluations"]
        self.epsilon = LESSON["epsilon"]
        self.eps_end = LESSON["eps_end"]
        self.eps_decay = LESSON["eps_decay"]
        self.env_name = LESSON["env_name"]
        self.algo = LESSON["algo"]
        
        # For training
        self.opponent_pool = []
        self.i_epoch = 0
        self.game_ticker = 0
        
        
    
    # def fill_memory_buffer_with_gameplay(self, real_agent, opponent):
    #     """Play games for this agent against the opponent until we reach our max steps per episode
    #     adding to memory buffer as we go on
        
    #     # TODO ENABLE MULTIPLE OPPONENTS

    #     Args:
    #         agent (AgentClass): Current agent we are training from agent pool
    #         opponent (AgentClass): Frozen (or from opponent pool in case of self) agent we play against
    #     """
        
    #     self.env.reset()
    #     # randomly select what position agent plays in
    #     agent_position = np.random.randint(0, self.n_players)
    #     self.agent_position = agent_position
    #     ### Replace the pettingzoo environment's  "agents" list (which is just a list of ints)
    #     # with actual agent DQNs
    #     lo_agents = []
    #     for agent_int in self.env.agents:
    #         if agent_int == agent_position:
    #             lo_agents.append(real_agent)
    #         else:
    #             lo_agents.append(opponent)

    #     for agent in lo_agents:

    #         assert (isinstance(agent, DQN))
            
    #     # Get Obs space for flattening
    #     obs_space = self.env.observation_space_dict['observation']
         
    #     step_counter = 0
    #     # assert False
    #     for i_agent in self.env.agent_iter(): 
    #         step_counter +=1 # uptick step counter
            
    #         agent = lo_agents[i_agent]
    #         assert isinstance(agent, DQN)
            
    #         observation, _, termination, _, _= self.env.last() # reads the observation of the last state from current agent's POV
    #         state = observation['observation']
    

    #         ########### FLATTEN STATE #############
    #         try:
    #             state_flat = flatten(obs_space, state)
    #         except IndexError as e:
    #             assert False
            
    #         ###### SELECT ACTION #######
    #         if termination:
    #             action = None
    #             assert False # this shouldnt't be gotten to anymore because termination flag is handled after step now

    #         else:
    #             # Agent or opponet samples action according to policy
    #             action_mask = observation['action_mask']
    #             action = agent.get_action(state_flat, self.epsilon, action_mask)[0]
                
    #         ######### STEP ############
    #         self.env.step(action) # current agent steps
    #         print(action)
    #         assert False
            
    #         ######## SEE CONSEQUENCE OF STEP ##########
    #         next_observation, reward, termination, _, info = self.env.last() # reads the observation of the last state from current agent's POV
    #         next_state = next_observation['observation']
            
    #         ########### FLATTEN STATES #############
    #         try:
    #             assert (next_state in obs_space)
    #         except AssertionError:
    #             print(next_state)
    #             assert False
    #         try:
    #             next_state_flat = flatten(obs_space, next_state)
    #         except IndexError as e:
    #             assert False

    #         ########### FILL NONE with Pass action #############
    #         try:
    #             assert state is not None, "Error: state is None!"
    #             assert action is not None, "Error: action is None!"
    #             assert reward is not None, "Error: reward is None!"
    #             assert next_state is not None, "Error: next_state is None!"
    #             assert termination is not None, "Error: termination is None!"
    #             assert info is not None, "Error: info is None!"
    #         except AssertionError as e:
    #             print(e)
    #             print("state:", state)
    #             print("action:", action)
    #             print("reward:", reward)
    #             print("next_state:", next_state)
    #             print("termination:", termination)
    #             print("info:", info)
    #             raise  # Re-raise the exception to catch the issue

            
    #         # Save experiences to replay buffer if its the agent that did the action
    #         if real_agent == agent:
    #             # if reward == -5:
    #                 # print(state)
    #                 # print(f"ACtion that lost life {action} at next_action_type {state['next_action_type']}")

    #             self.memory.save_to_memory(
    #                 state_flat,
    #                 action,
    #                 reward,
    #                 next_state_flat,
    #                 termination,
    #                 is_vectorised=False,
    #             )

    #         ### If game ended reset it
    #         ### If game ended or max steps reached, exit loop
    #         if info['next_action_type'] == "win":
    #             # print("Game Ended")
    #             # print(f"Agent id is {agent_position}")
    #             # print(f'Reward {reward}')
    #             # print(f"Accumulated reward {self.env._cumulative_rewards[agent_position]}")
    #             # print(f"Accumulated reward {self.env._cumulative_rewards}")
    #             # print('')
    #             self.env.agent_reward = self.env._cumulative_rewards[agent_position]
                
    #             if real_agent == agent:
    #                 self.env.agent_win = 1 # if agent wins, have this 
    #             else:
    #                 self.env.agent_win = 0

    #             break
            
    def play_game(self, real_agent, opponent, fill_memory_buffer:bool, save_actions:bool=True):
        """Play games for this agent against the opponent until we reach our max steps per episode
        adding to memory buffer as we go on
        
        # TODO ENABLE MULTIPLE OPPONENTS

        Args:
            agent (AgentClass): Current agent we are training from agent pool
            opponent (AgentClass): Frozen (or from opponent pool in case of self) agent we play against
        """
        
        self.env.reset()
        # randomly select what position agent plays in
        agent_position = 0
        self.agent_position = agent_position
        ### Replace the pettingzoo environment's  "agents" list (which is just a list of ints)
        # with actual agent DQNs
        lo_agents = []
        for agent_int in self.env.agents:
            if agent_int == agent_position:
                lo_agents.append(real_agent)
            else:
                lo_agents.append(opponent)

        for agent in lo_agents:
            assert (isinstance(agent, DQN) or isinstance(agent, RandomOpponent))
        
        # Get Obs space for flattening
        obs_space = self.env.observation_space_dict['observation']
         
        step_counter = 0
        
        
        
        for i_agent in self.env.agent_iter(): 
            step_counter +=1 # uptick step counter
            
            
            observation, _, termination, _, _= self.env.last() # reads the observation of the last state from current agent's POV
            state = observation['observation']
    

            ##### SELECT AGENT ########
            agent = lo_agents[i_agent]

            ########### FLATTEN STATE #############
            try:
                state_flat = flatten(obs_space, state)
            except IndexError as e:
                assert False
            
            ###### SELECT ACTION #######
            if termination:
                action = None
                assert False # this shouldnt't be gotten to anymore because termination flag is handled after step now

            else:
                # Agent or opponet samples action according to policy
                action_mask = observation['action_mask']
                if fill_memory_buffer:
                    action = agent.get_action(state_flat, 1, action_mask)[0] # pick random actions
                else:
                    action = agent.get_action(state_flat, self.epsilon, action_mask)[0] # pick action according to eps
                
            ######### STEP ############
            self.env.step(action) # current agent steps
                        
    
            ######## SEE CONSEQUENCE OF STEP ##########
            next_observation, _, termination, _, info = self.env.last() # reads the observation of the last state from current agent's POV
            reward = self.env.rewards.get(i_agent, 0) # get the reward of the agent that just ACTED (self.env.last steps iter)    
            print(f"Observed reward in multistep {reward}")
                
            next_state = next_observation['observation']

            
            ########### FLATTEN STATES #############
            try:
                assert (next_state in obs_space)
            except AssertionError:
                print(obs_space)
                print(next_state)
                print("Obs is not in obs space")
                assert False
            try:
                next_state_flat = flatten(obs_space, next_state)
            except IndexError as e:
                assert False

            ########### FILL NONE with Pass action #############
            try:
                assert state is not None, "Error: state is None!"
                assert action is not None, "Error: action is None!"
                assert reward is not None, "Error: reward is None!"
                assert next_state is not None, "Error: next_state is None!"
                assert termination is not None, "Error: termination is None!"
                assert info is not None, "Error: info is None!"
            except AssertionError as e:
                print(e)
                print("state:", state)
                print("action:", action)
                print("reward:", reward)
                print("next_state:", next_state)
                print("termination:", termination)
                print("info:", info)
                raise  # Re-raise the exception to catch the issue

            
            # Save experiences to replay buffer if its the agent that did the action
            # print(f"Current DQN {agent}")
            # print(f"Real Agent {real_agent}")
            # print(f"Equal? {real_agent == agent}")
            # print("\n")

            
            # assert False
            if (real_agent == agent) & (fill_memory_buffer==True):
                self.memory.save_to_memory(
                    state_flat,
                    action,
                    reward,
                    next_state_flat,
                    termination,
                    is_vectorised=False,
                )
                
            action_metrics = {}

            if save_actions:
                state = convert_np_to_list(state)
                next_state = convert_np_to_list(next_state)
                action_metrics["game_id"] = int(self.game_ticker)
                action_metrics["action"] = int(action)
                action_metrics["action_mask"] = convert_np_to_list(action_mask)

                action_metrics["state"] = dict(state)
                action_metrics["reward"] = float(reward)
                action_metrics["next_state"] = dict(next_state)

                # # Assuming `rewards` is a dictionary
                os.makedirs('metrics/actions', exist_ok=True)
                lesson_name = self.LESSON['lesson_name']
                with open(f"metrics/actions/{lesson_name}_actions.jsonl", "a") as f:
                    json.dump(action_metrics, f)  # `indent=4` makes it readable
                    f.write("\n")
            

            
            
            if step_counter == self.LESSON['max_steps']:
                agent_reward = self.env._cumulative_rewards[agent_position]
                self.game_ticker +=1
                return agent_reward, 0

            ### If game ended or max steps reached, exit loop
            if info['next_action_type'] == "win":
                # print("Game Ended")
                # print(f"Agent id is {agent_position}")
                # print(f'Reward {reward}')
                # print(f"Accumulated reward {self.env._cumulative_rewards[agent_position]}")
                # print(f"Accumulated reward {self.env._cumulative_rewards}")
                agent_reward = self.env._cumulative_rewards[agent_position]
            
                if real_agent == lo_agents[agent_position]:
                    agent_win = 1 # if agent wins, tick this up for win rate counter
                else:
                    agent_win = 0
                    
                self.game_ticker +=1
                    
                return agent_reward, agent_win

        
    def train_one_epoch(self, agent):
        """Fully trains one agent in the population for one epoch number of episodes (as defined by self.episodes_per_epoch)

        Args:
            agent (_type_): _description_
        """
        # =====================================           
        # Metrics to keep track of within this epoch
        # ===================================== 
        agent_reward_per_episode = []
        agent_win_in_episode = 0
        # =====================================           
        # Setting up Opponent
        # ===================================== 
        if self.opponent_pool: # if we have opponent pol we are training against self
            # Randomly choose opponent from opponent pool if using self-play
            opponent = random.choice(self.opponent_pool)
        else:
            # Create opponent of desired difficulty
            opponent = self.opponent
        
        # =====================================           
        # Train over X number of episodes (1 epoch)
        # ===================================== 
        # for n_games in trange(self.episodes_per_epoch, desc="Training Epoch"): # one eposode is one training of an agent in the pop. this is for one epoch
        for n_games in range(self.episodes_per_epoch): # one eposode is one training of an agent in the pop. this is for one epoch
            
            # This agent plays 1 game. Filling memory buffer
            agent_reward, agent_win = self.play_game(agent, opponent, fill_memory_buffer=True)
            # Collect some metrics
            agent_reward_per_episode.append(agent_reward) # something like this.
            agent_win_in_episode += agent_win
        
    
        
        # =====================================           
        # UPDATE MODEL AT THE END OF EPOCH
        # =====================================
        
        experiences = self.memory.sample(agent.batch_size)# Sample replay buffer
        agent.learn(experiences)# Learn according to agent"s RL algorithm
        

        # =====================================           
        # SAVE METRICS
        # =====================================
        agent_reward_per_epoch = np.mean(agent_reward_per_episode) # mean agent cumulative reward for the epoch
        self.agent_reward_per_epoch = agent_reward_per_epoch# mean agent cumulative reward for the epoch
        self.win_rate = agent_win_in_episode/self.episodes_per_epoch # mean agent win rate epoch
        
    
        # =====================================          
        # EPOCH IS DONE
        # =====================================
    
    def evolve_opp(self):
        # ============================================
        #  Remove one Opponent and add current Elite agent
        # ============================================
        # If we are playing self, here you can make clone yourself and make the new opponent pool an upgraded version of agent
        # If we are playing self, clone yourself and upgrade the opponent pool
        to_remove = random.choice(self.opponent_pool)  # Pick a random element
        self.opponent_pool.remove(to_remove)  # Remove it
        
        elite_opp = DQN(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hp_config=self.hp_config,
            net_config=self.net_config,
            batch_size=self.INIT_HP["BATCH_SIZE"],
            lr=self.INIT_HP["LR"],
            learn_step=self.INIT_HP["LEARN_STEP"],
            gamma=self.INIT_HP["GAMMA"],
            tau=self.INIT_HP["TAU"],
            double=self.INIT_HP["DOUBLE"],
            cudagraphs=self.INIT_HP["CUDAGRAPHS"],
            device=self.device,
        )
        
        # Load the neural net graph galuves from the elite agent
        elite_opp.actor.load_state_dict(self.elite.actor.state_dict())

        elite_opp.actor.eval() # set to eval mode
    
        self.opponent_pool.append(elite_opp)  # Directly modify self.opponent_pool
        
        print("Evolved Opponent Pool")

    def evolve_pop(self):
        """
        Evolve the population by evaluating the DQNs in the current pop
        and keeping the best one
        
        Then mutate DQN hyperparameters
        """

        # ====================================
        # INIT opponent for evaluating against
        # ====================================
        if self.LESSON['eval_opponent'] == 'random': # eval against random opponent
            opponent = RandomOpponent() # TODO NOt tested
        elif self.LESSON['eval_opponent'] == 'self': # eval against self
            opponent = self.elite # TODO not tested
        else:
            # print("Loaded opponvolve_poent agent for evaluation")
            opponent = Opponent(dqn_path=self.LESSON['eval_opponent'], device = self.device)
            opponent = opponent.model

        # ====================================
        # EVALUATE fitness of agents in pop
        # ====================================
        lo_agent_mean_fitness = []
        lo_agent_wrs = []
        for agent in tqdm(self.pop, desc="Evaluating Agent Fitness"):
            lo_fitnesses_for_agent = [] # fitness of agent across the games
            n_wins = 0
            with torch.no_grad():
                for i in range(self.LESSON['n_evaluations']):
                    agent_reward, agent_win = self.play_game(agent, opponent, fill_memory_buffer=False)
                    lo_fitnesses_for_agent.append(agent_reward)
                    n_wins += agent_win
                                        
            # calc mean fitness for  this agent's evaluations
            mean_fit_for_agent = np.round(np.mean(lo_fitnesses_for_agent),2)
            # win rate
            wr = n_wins/self.LESSON['n_evaluations']
            lo_agent_wrs.append(wr)
            
            lo_agent_mean_fitness.append(mean_fit_for_agent)  
            agent.fitness.append(mean_fit_for_agent)

        # Tournament selection and population mutation
        # print(self.elite)
        # print(self.elite.fitness)
        
        elite, pop = self.tournament.select(self.pop)
        self.elite = elite
        self.pop = self.mutations.mutation(pop)
        
        # print(self.elite)
        # print(self.elite.fitness)
        # [print(ag.fitness) for ag in self.pop]
        
        # ====================================
        # INIT Metrics for evaluating fitness of agents in pop
        # ====================================
        eval_metrics = {}
        eval_metrics['epoch'] = self.i_epoch
        eval_metrics['agent_mean_fitness'] = lo_agent_mean_fitness
        eval_metrics['pop_mean_fitness'] = np.round(np.mean(lo_agent_mean_fitness),2)
        eval_metrics['agent_mean_wrs'] = np.round(lo_agent_wrs,2).tolist()
        eval_metrics['pop_mean_wr'] = np.round(np.mean(lo_agent_wrs),2)
        eval_metrics['elite_mean_fitness'] = np.round(np.max(lo_agent_mean_fitness),2)
        
        # Assuming `rewards` is a dictionary
        os.makedirs('metrics/eval', exist_ok=True)
        lesson_name = self.LESSON['lesson_name']
        with open(f"metrics/eval/{lesson_name}_eval.jsonl", "a") as f:
            json.dump(eval_metrics, f)  # `indent=4` makes it readable
            f.write("\n")
        




    def setup_opponent_pool(self):
        """
        If agent is training against itself, create
        a pool of opponents that are equal to itself
        """
        print("Setting up Opponent pool of type Self")
        if self.LESSON["opponent"] == "self":
            population = []
            for idx in range(self.LESSON["opponent_pool_size"]):
                opp = DQN(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    index=idx,
                    hp_config=self.hp_config,
                    net_config=self.net_config,
                    batch_size=self.INIT_HP["BATCH_SIZE"],
                    lr=self.INIT_HP["LR"],
                    learn_step=self.INIT_HP["LEARN_STEP"],
                    gamma=self.INIT_HP["GAMMA"],
                    tau=self.INIT_HP["TAU"],
                    double=self.INIT_HP["DOUBLE"],
                    cudagraphs=self.INIT_HP["CUDAGRAPHS"],
                    device=self.device,
                )
                
                # Load the neural net graph galuves from the elite agent
                opp.actor.load_state_dict(self.elite.actor.state_dict())
                
                for key in opp.actor.state_dict():
                    if not torch.allclose(opp.actor.state_dict()[key], self.elite.actor.state_dict()[key], atol=1e-6):
                        print(f"Mismatch in key: {key}")
                        break
                else:
                    print("All weights match")
                
                opp.actor.eval() # set to eval mode
                population.append(opp)
        self.opponent_pool = population
                
    def warmup(self):
        """
        Warmup the agent by training it on opponents that make random decisions
        """
        
        # Create a random opponent
        opponent = RandomOpponent()
        
        # Fill replay buffer with transitions
        print("Filling replay buffer ...")

        pbar = tqdm(total=self.memory.memory_size)
        max_buffer_size = int(self.memory.memory_size)
        while int(len(self.memory.memory)) < int(max_buffer_size):
            for agent in self.pop:
    
                self.play_game(agent,
                            opponent,
                            fill_memory_buffer=True)
                if len(self.memory.memory) % 100 ==0:                 
                    pbar.update(len(self.memory.memory)-pbar.n)
                
                if int(len(self.memory.memory)) >= int(max_buffer_size):
                    print(int(len(self.memory.memory)))
                    break

        print("Replay buffer filled")
            
        
        
        if self.LESSON["agent_warm_up"] > 0: # number of epochs to warmup on
            print("Warming up agents ...")
            
            # Train on randomly collected samples
            for epoch in trange(self.LESSON["agent_warm_up"]):
                experiences = self.memory.sample(self.elite.batch_size)
                
                # print(experiences)
                # Print "shape" (number of fields)
                # print(self.elite.batch_size)
                # print(f"Shape: (1, {len(experiences)})")  # 1 row, N columns

                # Print headers (field names)
                # print("Element types:", tuple(type(x).__name__ for x in experiences))
                
                # Print tensor shapes
                # tensor_shapes = tuple(t.shape for t in experiences)

                # print("Tuple shape:", (1, len(experiences)))  # (1, 5) structure
                # print("Tensor shapes:", tensor_shapes)
                
                #### TREACH AGENT
                self.elite.learn(experiences) # Train the agent on the sampled experiences (UPDATING Q-s)
                            
            # Create our population, which is a population of agents trained on random experiences
            self.pop = [self.elite.clone() for _ in self.pop]
            # self.elite = self.elite # Already taken care of with class
            print("Agent population warmed up.")
        else: 
            print("Not warming up agent population")

    def train_multi_agent(self):
        """
        Main function of MultiAgentTrainer
        
        Trains multi-agent for a DQN
        """
        


        
        self_train = True
        ### Handle opponent initiation # needs to be done outside of epoch training loop for self-case
        if self.LESSON["opponent"] == "self":
            self.setup_opponent_pool()
            print(f"Setup opponent pool for self-training {self.opponent_pool}")
        elif self.LESSON['opponent'] == None:
            print("Not Self-training")
            self_train = False
        elif self.LESSON['opponent'] !='self': # load a DQN from somewhere else
            self.opponent = Opponent(dqn_path = self.LESSON['opponent'], device=self.device)
            self.opponent = self.opponent.model
            assert self.opponent != self.elite
            print(f"Loaded Opponent from {self.opponent}")
            
        # Perform buffer and agent warmups if desired
        if self.LESSON["buffer_warm_up"]:
            self.warmup()
        # pbar for tracking epochs
        
        if self.epochs:
            pbar = trange(self.epochs, position=0, desc= "Epoch Trainings")
        
            # One Epoch of training (1 epoch is an X number of episodes)
            print("==========================================")
            print("Beginning Training Against DQN Agents")
            print("==========================================")        

            for epoch in pbar:
                if self_train == False:
                    break
                

                rewards_epoch = {} # epoch: lo_rewards_for_agents
                lo_rewards_for_agents = [] # holds reward of each agent in pop
                lo_win_rates = [] # holds win rate of each agent in pop
                
                # Train each agent in our pop for this epoch
                for agent in self.pop:  # Loop through population and train each one individually
                    self.train_one_epoch(agent) # Train this agent on an epoch number of episodes
                    # Update epsilon for exploration
                    self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
                    
                    lo_rewards_for_agents.append(self.agent_reward_per_epoch)
                    lo_win_rates.append(self.win_rate)
                    
                self.i_epoch+=1

                                        
                
                
                if epoch % self.LESSON['evo_epochs'] == 0: # Choosing how frequently to evolve pop
                    self.evolve_pop() # evolve population after one epoch
                    
                if (epoch % self.LESSON['evo_opp_epochs'] == 0) and (self.LESSON['opponent'] == 'self'):  # Choosing how frequently we evolve opponents
                    self.evolve_opp()
                    
                # ======================================
                # Metrics to track across epochs
                # ======================================
                rewards_epoch['epoch'] = self.i_epoch
                rewards_epoch['eps'] = np.round(self.epsilon,2)
                rewards_epoch['mean_training_reward'] = np.round(lo_rewards_for_agents,2).tolist()
                
                rewards_epoch['pop_mean_training_reward'] = np.round(np.mean(lo_rewards_for_agents),2)
                
                rewards_epoch['mean_wr'] = np.round(lo_win_rates,2).tolist()
                rewards_epoch['pop_mean_wr'] = np.round(np.mean(lo_win_rates),2)
                
                
                # Assuming `rewards` is a dictionary
                os.makedirs('metrics/train', exist_ok=True)
                lesson_name = self.LESSON['lesson_name']
                with open(f"metrics/train/{lesson_name}_rewards.jsonl", "a") as f:
                    json.dump(rewards_epoch, f)  # `indent=4` makes it readable
                    f.write("\n")
                    
                
        print("==========================================")
        print("Finished Self-training")
        print("==========================================")
        # Save the trained agent
        save_path = self.LESSON["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.elite.save_checkpoint(save_path)
        print(f"Elite agent saved to '{save_path}'.")

    def load_lesson(self,LESSON):
        self.LESSON = LESSON
        self.epochs = LESSON["epochs"]
        self.episodes_per_epoch = LESSON["episodes_per_epoch"]
        self.evo_epochs = LESSON["evo_epochs"]
        self.n_evaluations = LESSON["n_evaluations"]
        self.epsilon = LESSON["epsilon"]
        self.eps_end = LESSON["eps_end"]
        self.eps_decay = LESSON["eps_decay"]
        self.env_name = LESSON["env_name"]
        self.algo = LESSON["algo"]
        self.env.LESSON = LESSON
        
        
        
class GamePlayer(MultiAgentTrainer):
    def __init__(self, env, real_agent, opponent):
        self.env = env
        self.LESSON = self.env.LESSON
        self.real_agent = real_agent
        self.opponent = opponent
        self.fill_memory_buffer = False
        self.game_ticker = 0
        
    
        
def convert_np_to_list(obj):
    if isinstance(obj, np.ndarray):  
        return obj.tolist()
    elif isinstance(obj, dict):  
        return {k: convert_np_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):  
        return [convert_np_to_list(v) for v in obj]
    return obj