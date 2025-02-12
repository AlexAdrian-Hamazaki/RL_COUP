
# Importing Coup Env
from coup_env.coup_env import CoupEnv
from coup_env.coup_player import CoupPlayer
from opponent import Opponent

import copy
import os
import random
from collections import deque
from datetime import datetime

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
from gymnasium.spaces.utils import flatten


# Agile imports
from agilerl.algorithms.dqn import DQN

class MultiAgentTrainer:
    def __init__(self,
                env,  # Pettingzoo-style environment
                pop,  # Population of agents
                elite,  # Assign a placeholder "elite" agent
                memory,  # Replay buffer
                INIT_HP,  # IINIT_HP dictionary
                net_config,  # Network configuration
                tournament,  # Tournament selection object
                mutations,  # Mutations object
                action_space,
                observation_space,
                LESSON,
                n_players,
                device,
                

                
            ):
        
        self.env = env
        self.pop = pop
        self.elite = elite
        self.memory = memory
        self.INIT_HP = INIT_HP
        self.net_config = net_config
        self.tournament = tournament
        self.mutations = mutations
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_players = n_players
        self.device = device
        
        self.LESSON = LESSON
        self.max_steps = LESSON["max_steps"]
        self.max_episodes = LESSON["max_episodes"]
        self.episodes_per_epoch = LESSON["episodes_per_epoch"]
        self.evo_epochs = LESSON["evo_epochs"]
        self.evo_loop = LESSON["evo_loop"]
        self.epsilon = LESSON["epsilon"]
        self.eps_end = LESSON["eps_end"]
        self.eps_decay = LESSON["eps_decay"]
        self.opp_update_counter = LESSON["opp_update_counter"]
        self.env_name = LESSON["env_name"]
        self.algo = LESSON["algo"]
        
        # For training
        self.opponent_pool = []
        
        
    
    def play_game(self, real_agent, opponent):
        """Play games for this agent against the opponent until we reach our max steps per episode
        adding to memory buffer as we go on
        
        # TODO ENABLE MULTIPLE OPPONENTS

        Args:
            agent (AgentClass): Current agent we are training from agent pool
            opponent (AgentClass): Frozen (or from opponent pool in case of self) agent we play against
        """
        
        self.env.reset()
        # randomly select what position agent plays in
        agent_position = np.random.randint(0, self.n_players)
        
        ### Replace the pettingzoo environment's  "agents" list (which is just a list of ints)
        # with actual agent DQNs
        lo_agents = []
        for agent_int in self.env.agents:
            if agent_int == agent_position:
                lo_agents.append(real_agent)
            else:
                lo_agents.append(opponent.model)

        for agent in lo_agents:
            assert (isinstance(agent, DQN))
            
        
        
        # Get Obs space for flattening
        obs_space = self.env.observation_space_dict['observation']
         
        step_counter = 0
        # assert False
        for agent in self.env.agent_iter(): #NOTE: This we are iterating over actual DQN classes
            step_counter +=1 # uptick step counter
        
            
            agent = lo_agents[agent]
            assert isinstance(agent, DQN)
            
            
            
            observation, reward, termination, _, info = self.env.last() # reads the observation of the last state from current agent's POV
            state = observation['observation']
                    
            
            ########### FLATTEN STATE #############
            try:

                state = flatten(obs_space, state)
            except IndexError as e:
                assert False
            
            ###### SELECT ACTION #######
            if termination:
                action = None
                assert False # this shouldnt't be gotten to anymore because termination flag is handled after step now

            else:
                # Agent or opponet samples action according to policy
                action_mask = observation['action_mask']
                action = agent.get_action(state, self.epsilon, action_mask)[0]
                
            ######### STEP ############
            self.env.step(action) # current agent steps
            
            
            ######## SEE CONSEQUENCE OF STEP ##########
            next_observation, reward, termination, _, info = self.env.last() # reads the observation of the last state from current agent's POV
            next_state = next_observation['observation']
            
            ########### FLATTEN STATES #############
            try:
                assert (next_state in obs_space)
            except AssertionError:
                print(next_state)
                assert False
            try:
                next_state = flatten(obs_space, next_state)
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
            if real_agent == agent:
                self.memory.save_to_memory(
                    state,
                    action,
                    reward,
                    next_state,
                    termination,
                    is_vectorised=False,
                )
                
            ### If game ended reset it
            ### If game ended or max steps reached, exit loop
            if info['next_action_type'] == "win" or step_counter >= self.LESSON['max_steps']:
                break

        
    def train_one_epoch(self, agent):
        """Fully trains one agent in the population for one epoch number of episodes (as defined by self.episodes_per_epoch)

        Args:
            agent (_type_): _description_
        """
        
        agent_reward_per_episode = []
        
        
        if self.opponent_pool: # if we have opponent pol we are training against self
            # Randomly choose opponent from opponent pool if using self-play
            opponent = random.choice(self.opponent_pool)
        else:
            # Create opponent of desired difficulty
            opponent = self.opponent
        
        
        for n_games in range(self.episodes_per_epoch): # one eposode is one training of an agent in the pop. this is for one epoch
            
            # Right now, just make our agent always go first # TODO enable variable start
            # This agent plays 1 game. Filling memory buffer
            self.play_game(agent, opponent)

            # Collect some metrics
            print(self.env._cumulative_rewards)
            agent_reward_per_episode.append(self.env._cumulative_rewards) # something like this. number of turns in the game# TODO
        
        
        #### METRICS TO PRINT
        agent_reward_per_epoch = np.mean(agent_reward_per_episode)
        print(agent_reward_per_epoch)
        
        
                     
        # Learn according to learning frequency
        if (memory.counter % agent.learn_step == 0) and (
            len(memory) >= agent.batch_size
        ):
            # Sample replay buffer
            # Learn according to agent"s RL algorithm
            experiences = memory.sample(agent.batch_size)
            agent.learn(experiences)
            
            
        # If we are playing self, here you can make clone yourself and make the new opponent pool an upgraded version of agent
        self.make_opponents_me()
        if LESSON["opponent"] == "self":
            if (total_episodes % LESSON["opponent_upgrade"] == 0) and (
                    (idx_epi + 1) > evo_epochs
            ):
                    elite_opp, _, _ = tournament._elitism(pop)
                    elite_opp.actor.eval()
                    opponent_pool.append(elite_opp)
                    opp_update_counter += 1
        ######
        ### EPOCH IS DONE
        ######

    def evolve_pop():
        mean_turns = np.mean(turns_per_episode)

        # Now evolve population if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population vs random actions
            fitnesses = []
            win_rates = []
            eval_actions_hist = [0] * action_dim  # Eval actions histogram
            eval_turns = 0  # Eval turns counter
            for agent in pop:
                with torch.no_grad():
                    rewards = []
                    for i in range(evo_loop):
                        env.reset()  # Reset environment at start of episode
                        observation, cumulative_reward, done, truncation, _ = (
                            env.last()
                        )

                        player = -1  # Tracker for which player"s turn it is

                        # Create opponent of desired difficulty
                        opponent = Opponent(env, difficulty=LESSON["eval_opponent"])

                        # Randomly decide whether agent will go first or second
                        if random.random() > 0.5:
                            opponent_first = False
                        else:
                            opponent_first = True

                        score = 0

                        for idx_step in range(max_steps):
                            action_mask = observation["action_mask"]
                            if player < 0:
                                if opponent_first:
                                    if LESSON["eval_opponent"] == "random":
                                        action = opponent.get_action(action_mask)
                                    else:
                                        action = opponent.get_action(player=0)
                                else:
                                    state = np.moveaxis(
                                        observation["observation"], [-1], [-3]
                                    )
                                    state = np.expand_dims(state, 0)
                                    action = agent.get_action(
                                        state, 0, action_mask
                                    )[
                                        0
                                    ]  # Get next action from agent
                                    eval_actions_hist[action] += 1
                            if player > 0:
                                if not opponent_first:
                                    if LESSON["eval_opponent"] == "random":
                                        action = opponent.get_action(action_mask)
                                    else:
                                        action = opponent.get_action(player=1)
                                else:
                                    state = np.moveaxis(
                                        observation["observation"], [-1], [-3]
                                    )
                                    state[[0, 1], :, :] = state[[1, 0], :, :]
                                    state = np.expand_dims(state, 0)
                                    action = agent.get_action(
                                        state, 0, action_mask
                                    )[
                                        0
                                    ]  # Get next action from agent
                                    eval_actions_hist[action] += 1

                            env.step(action)  # Act in environment
                            observation, cumulative_reward, done, truncation, _ = (
                                env.last()
                            )

                            if (player > 0 and opponent_first) or (
                                player < 0 and not opponent_first
                            ):
                                score = cumulative_reward

                            eval_turns += 1

                            if done or truncation:
                                break

                            player *= -1

                        rewards.append(score)
                mean_fit = np.mean(rewards)
                agent.fitness.append(mean_fit)
                fitnesses.append(mean_fit)

            eval_turns = eval_turns / len(pop) / evo_loop

            pbar.set_postfix_str(
                f"    Train Mean Score: {np.mean(agent.scores[-episodes_per_epoch:])}   Train Mean Turns: {mean_turns}   Eval Mean Fitness: {np.mean(fitnesses)}   Eval Best Fitness: {np.max(fitnesses)}   Eval Mean Turns: {eval_turns}   Total Steps: {total_steps}"
            )
            pbar.update(0)

            # Format action histograms for visualisation
            train_actions_hist = [
                freq / sum(train_actions_hist) for freq in train_actions_hist
            ]
            eval_actions_hist = [
                freq / sum(eval_actions_hist) for freq in eval_actions_hist
            ]
            train_actions_dict = {
                f"train/action_{index}": action
                for index, action in enumerate(train_actions_hist)
            }
            eval_actions_dict = {
                f"eval/action_{index}": action
                for index, action in enumerate(eval_actions_hist)
            }


            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)


    def setup_opponent_pool(self):
        """
        If agent is training against itself, create
        a pool of opponents that are equal to itself
        """
        if self.LESSON["opponent"] == "self":
            # Create initial pool of opponents
            opponent_pool = deque(maxlen=self.LESSON["opponent_pool_size"])
            for _ in range(self.LESSON["opponent_pool_size"]):
                opp = copy.deepcopy(self.elite)
                opp.actor.load_state_dict(self.elite.actor.state_dict())
                opp.actor.eval()
                opponent_pool.append(opp)
                
            self.opponent_pool = opponent_pool
    
    def warmup(self):
        """
        Warmup the agent by training it on opponents that make random decisions
        """
        
        # Fill replay buffer with transitions
        self.memory = CoupPlayer.fill_replay_buffer(memory=self.memory,
                                                    n_players=self.n_players,
                                                    obs_space = self.env.observation_space_dict['observation'])        
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

    def train_multi_agent(self):
        """
        Main function of MultiAgentTrainer
        
        Trains multi-agent for a DQN
        """
        
        # Perform buffer and agent warmups if desired

        if self.LESSON["buffer_warm_up"]:
            self.warmup()
            
        ### Handle opponent initiation # needs to be done outside of epoch training loop for self-case
        if self.LESSON["opponent"] == "self":
            self.setup_opponent_pool()
            print(f"Setup opponent pool for self-training {self.opponent_pool}")
        elif self.LESSON['opponent'] !='self':
            self.opponent = Opponent(difficulty=self.LESSON['opponent'], device=self.device)
            print(f"Loaded Opponent from {self.opponent}")
        
    
        # pbar for tracking epochs
        pbar = trange(int(self.max_episodes / self.episodes_per_epoch))

        # One Epoch of training (1 epoch is an X number of episodes)
        print("==========================================")
        print("Beginning Self-training")
        print("==========================================")
        
        for epoch in pbar:
            # Train each agent in our pop for this epoch
            for agent in self.pop:  # Loop through population and train each one individually
                self.train_one_epoch(agent) # Train this agent on an epoch number of episodes
                # Update epsilon for exploration
                self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
                 
            self.evolve_pop() # evolve population after one epoch
            
            
            
        print("==========================================")
        print("Finished Self-training")
        print("==========================================")
        # Save the trained agent
        save_path = self.LESSON["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.elite.save_checkpoint(save_path)
        print(f"Elite agent saved to '{save_path}'.")
