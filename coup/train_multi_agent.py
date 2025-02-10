
# Importing Coup Env
from coup_env.coup_env import CoupEnv
from coup_env.coup_player import CoupPlayer

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
                
                max_steps = 100, # Max turns in a game do we step through 
                max_episodes = 100,  # How many games to play, aka filling of nemory buffer
                episodes_per_epoch = 10, # How many games you play before updating Q network with memory buffer
                evo_epochs = 20, # How frequently we evaluate the HPOs and mutate them via tournament->mutation loop
                evo_loop = 50, # Number of evaluation episodes for evaluationg which hyperparameters work best
                
                epsilon = 1.0,  # Starting epsilon value
                eps_end = 0.1,  # Final epsilon value
                eps_decay = 0.9998,  # Epsilon decays
                opp_update_counter = 0,
                env_name='COUP_v0.1',  # Environment name
                algo="DQN",  # Algorithm,
                
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
        
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.episodes_per_epoch = episodes_per_epoch
        self.evo_epochs = evo_epochs
        self.evo_loop = evo_loop
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.opp_update_counter = opp_update_counter
        self.env_name = env_name
        self.algo = algo
        self.LESSON = LESSON
        
        
    
    def play_game(self, agent, observation, cumulative_reward,
                  terminations, truncations, info):
        
        action_mask = observation["action_mask"]
        state = observation['observation']
        
        
        action = agent.get_action(state, self.epsilon, action_mask)[0]  # Get next action from agent
        train_actions_hist[action] += 1

        env.step(p0_action)  # Act in environment
        observation, cumulative_reward, done, truncation, _ = env.last()
       
        
    def train_one_epoch(self, agent, opponent_pool=[]):
        """Fully trains one agent in the population for one epoch number of episodes (as defined by self.episodes_per_epoch)

        Args:
            agent (_type_): _description_
        """
        turns_per_episode = []
        agent_reward_per_episode = []
        
        
        if self.opponent == "self":
            # Randomly choose opponent from opponent pool if using self-play
            opponent = random.choice(opponent_pool)
        else:
            # Create opponent of desired difficulty
            opponent = Opponent(self.env, difficulty='random')
        
        
        for n_episode in range(self.episodes_per_epoch): # one eposode is one training of an agent in the pop. this is for one epoch
            
            self.env.reset()  # Reset environment at start of episode
            
            # Right now, just make our agent always go first # TODO enable variable start

            self.play_game(agent, self.env)

            # Collect some metrics            
            turns_per_episode.append(self.env.n_turns) # something like this. number of turns in the game# TODO
            agent_reward_per_episode.append(self.env.reward) # something like this. number of turns in the game# TODO
            
            
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
        
        if self.LESSON["opponent"] == "self":
            self.setup_opponent_pool()


        # Perform buffer and agent warmups if desired
        if self.LESSON["buffer_warm_up"]:
            self.warmup()
            
        
        
        # pbar for tracking epochs
        pbar = trange(int(self.max_episodes / self.episodes_per_epoch))

        # One Epoch of training (1 epoch is an X number of episodes)
        print("==========================================")
        print("Beginning Self-training")
        print("==========================================")
        
        for n_epo in pbar:  #
            pass
        #     self.turns_per_episode = []
        #     self.train_actions_hist = [0] * self.action_dim # Make sure to reset these after
            
        #     for agent in self.pop:  # Loop through population and train each one individually
        #         self.train_one_epoch(agent) # Train this agent on an epoch number of episodes
        #         # Update epsilon for exploration
        #         self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
                 
        #     self.evolve_pop() # evolve population after one epoch
            
            
        ### Fully Completed Training
        
        # Save the trained agent
        save_path = self.LESSON["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.elite.save_checkpoint(save_path)
        print(f"Elite agent saved to '{save_path}'.")
