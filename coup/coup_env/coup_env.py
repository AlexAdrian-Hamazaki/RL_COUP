from pettingzoo import AECEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, Text, Sequence, Dict, Tuple, MultiDiscrete, MultiBinary, OneOf
import random
import functools
from copy import copy
import numpy as np
from classes.game import Game
import itertools

class CoupEnv(gym.Env):
    metadata = {
        "name": "coup_env_v0",
    }

    def __init__(self, n_players):
        """
        Defines the following attributes
        
        A Game of coup object with N_players 
        
        Coins and cards will be dealt
        """
        self.n_players = n_players
        self.player_ints = range(n_players)
        self.game = Game(self.n_players)
        
        self.players = list(range(n_players))  # [0, 1, 2, ..., n_players-1]
        self.turn_order_permutations = [self.players[i:] + self.players[:i] for i in range(self.n_players)]
        self.turn_order_permutation_map = dict(zip([n for n in range(len(self.turn_order_permutations))], self.turn_order_permutations))
        
        # not sure if needed
        self._card_names = ['assassin','ambassador', 'duke', 'contessa', 'captain']
        self._card_names_ints = [0, 1, 2, 3, 4]
        self._card_name_map = dict(zip(self._card_names, self._card_names_ints))
        
        deck_size = len(self.game.deck.deck)
        
        
        # action types
        self.action_type_map = {0:'normal_action',
                                1:'challenge_action',
                                2:'block_action'}
        
        
        self.observation_space = Dict({
            'mycards': MultiDiscrete([5, 5]), # pairs of cards here
            "mymoney": Discrete(n=14), # can have 0-13 coins
            "myclaims": MultiBinary([5]),
            "my_deck_knowledge": MultiDiscrete([6] * deck_size, start=[-1]*deck_size), #Order o deck, -1 indicates we do not know what the card is. # TODO figure out how to limit this to only correct observations like (-1,-1,2,4
            # Maybe will need to throw out bad obs?
            "others_claims": Dict(dict(zip([f'player{n}' for n in range(self.n_players)], 
                                      [MultiBinary([5]) for _ in range(self.n_players)]))), # Player_int: Text # dict of text spaces for what others are claiming,
            
            "others_n_cards": Dict(dict(zip([f'player{n}' for n in range(self.n_players)], 
                                      [Discrete(2) for _ in range(self.n_players)]))),
            
            "others_money": Dict(dict(zip([f'player{n}' for n in range(self.n_players)],
                                          [Discrete(14) for _ in range(self.n_players)]))), # Player_int: Discrete
            
            "revealed": Dict(dict(zip(self._card_names_ints, 
                                      [Discrete(3) for _ in range(5)] ))), # Card name, number revealed)
                
            "turn_order": Discrete(len(self.turn_order_permutations)), # see turn_order_permutation map to look at what the turn order exactly is
            "action_player": Discrete(self.n_players), 
            "target_player": Discrete(self.n_players), # may not need this
            "action_type": Discrete(3), # for action masking
        })
        
        
        # action space stays sime. Masking happens later
        self._actions = list(set(self.game.actions.ALLOWED_ACTIONS + list(self.game.actions.CHALLENGABLE_ACTIONS) + ["challenge"] + ['pass']))
        self._actions.sort()
        self._action_space_map = dict(zip([n for n in range(len(self._actions))],
                                          [action for action in self._actions]))
        self.action_space = MultiDiscrete([len(self._actions), self.n_players+1], start=[0,-1]) # first is action, second is target player, -1= No target
        
        print(f"Action map")
        print(self._action_space_map)
    
        
    def _get_obs(self):
        """Function that actually returns an observation given the state of the game

        Returns:
            _type_: _description_
        """
        
        action_type = self.game.turn.action_type # what type of action is able to be selected here
        mycards = self.game.agent.cards # need to 
        mymoney = self.game.agent.coins
        myclaims = self.game.agent.claimed_cards
        my_deck_knowledge = self.game.agent.knowledge.deck_knowledge 
        others_claims = self.game.agent.knowledge.other_player_claims # turn this into cards instead of actions
        others_n_cards = self.game.agent.knowledge.other_player_n_cards # turn this into cards instead of actions
        others_money = self.game.agent.knowledge.other_player_n_coins
        
        revealed = self.game.revealed_cards
        turn_order = self.game.turn.turn_order
        
        current_base_player= self.game.turn.current_base_player.name
        if self.game.turn.current_base_action_instance:
            current_claimed_card = self.game.turn.current_base_action_instance.card
        else:
            current_claimed_card = None
        

        base_action_target_player = -1 # no target # todo fix WHEN USING TARGET
            
            
        # Compute action mask
        action_mask = self._compute_action_mask()
        
        observation = {
            "action_type": action_type, #base_action, challenge_action, or block_action
            "mycards": mycards,
            "mymoney": mymoney,
            "myclaims": myclaims,
            "my_deck_knowledge": my_deck_knowledge,
            "others_claims": others_claims,
            "others_n_cards": others_n_cards,
            "others_money": others_money,
            "revealed": revealed,
            "turn_order": turn_order,
            "current_base_player": current_base_player,
            "current_claimed_card": current_claimed_card,
            "base_action_target_player": base_action_target_player,
            'action_mask': action_mask
            }
        
        return observation
    
    def _compute_action_mask(self):
        action_type = self.game.turn.action_type # what type of action is able to be selected here
        mymoney = self.game.turn.current_base_player.coins
        current_base_action = self.game.turn.current_base_action # none ifaction type is base_action
        
        # init mask of 0s to represent valid actions
        mask = np.array([0] * len(self._actions))
        # init mask of 0s to represent valid targets, -1th integer is -1 target which represents no target
        t_mask = np.array([0] * int(self.n_players+1)) 
        if action_type == 'base_action':
            if mymoney>10:
                good_indexes = [6,0]
                t_mask[2:] = 1
            elif mymoney >=7:
                # if at base action, enable base actions
                good_indexes = [7,8,9,10,11,0,6, 12]
                t_mask[2:] = 1
                t_mask[-1] = 1
            elif mymoney >=3:
                good_indexes = [7,8,9,10,11,0, 12]
                t_mask[2:] = 1
                t_mask[-1] = 1
            else:
                good_indexes = [7,8,9,10,11, 12]
                t_mask[-1] = 1
            mask[good_indexes] = 1
            return mask, t_mask
                
        elif action_type == "challenge_action":
            good_indexes = [5,10]
            t_mask[self.game.turn.current_base_player.name] = 1
            mask[good_indexes] = 1
            return mask, t_mask
        
        
        elif action_type == "block_action":
            if current_base_action == 'steal':
                good_indexes = [3,4]
                mask[good_indexes]
                return mask
            elif current_base_action == 'foreign_aid':
                good_indexes = [2]
                mask[good_indexes]
                return mask
            elif current_base_action == 'assassinate':
                good_indexes = [1]
                mask[good_indexes]
                return mask
            
    def _is_action_valid(self, action):
        real_action = self._compute_action_mask()[0][action[0]] == 1
    
        real_target = self._compute_action_mask()[1][action[1]] == 1
        return real_action * real_target
    
    def sample_valid_action(self):
        mask = self._compute_action_mask()
        # Sample valid actions for each instance in the mask
        valid_actions = []
        for row in mask:
            # Get indices of valid actions for this row
            valid_indices = np.flatnonzero(row)
            if valid_indices.size == 0:
                raise ValueError("No valid actions available for some instance in the mask.")
            # Sample a valid action from the valid indices
            sampled_action = np.random.choice(valid_indices)
            valid_actions.append(sampled_action)
        return np.array(valid_actions)
    
    def reset(self, seed=None, options=None):
        """Resets the game to a fresh game with freshly dealt cards

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        """    
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
    
        self.game = Game(n_players=self.n_players)
        observation = self._get_obs()
        
        return observation
        
    
    def step(self):
        """Takes an action by the current agent
        returns observation, reward, termination_flag, truncation_flag, and info dict
        Args:
            actions (int): action to take
        """
        
        action = self.sample_valid_action()
        print(action)
        if self.game.turn.action_type == 'base_action':
            action = [12,-1]
            
        print(f"Selected Action {action}")
        
        # check if action is valid
        if not self._is_action_valid(action):
            raise ValueError(f"Invalid action {action} for the current state.\n {self.game.turn.action_type}\n{self.game.turn.current_base_player}")

        action_map = self._action_space_map
        #first make the step through the action
        
        self.last = self._get_obs().copy()
        print("~~~~Prestep Obs~~~~")
        print(self.last)
        self.game.step(action, action_map)

        
        # then get the observation of the new environment
        observation = self._get_obs()
        
        # then get info on if we've terminated
        terminated = self.game.win
        # truncated if agent is dead
        truncated = self.game.lost 
        
        # then get the reward for stepping/terminating
        reward = self.get_reward(terminated, truncated)
        # print(f"Reward {reward}, terminated {terminated}, truncated {truncated}")
        
        # then get any additional info and truncation flag
        return observation, reward, terminated, truncated, {}
        
    def get_reward(self, terminated, truncated):        
        last = self.last
        obs = self._get_obs()
        reward = 0    
    
        # +1 for every money we gained
        money_reward = obs["mymoney"] - last["mymoney"]
    
        # if another bot has 1 less life, this is +5    
        diff_lives = {bot: last['others_n_cards'][bot] - obs['others_n_cards'][bot] for bot in last['others_n_cards'].keys()}
        kills = sum(diff_lives.values())
        kills_reward = abs(kills*10)
    
        # if we've terminated and we have won this is +50
        if terminated: # agent won
            win_reward = 50
        elif truncated: #agent died
            win_reward = -50
        else:
            win_reward = 0
        
        # if we've died this is -50
        reward += money_reward + kills_reward + win_reward
        
        return reward
    
    def bot_step(self):
        """
        For this bot. make a random action (depends on action_type)
        
        if action_type == base_action:
        If agent just did a base action
            Bot makes their base action
            
            action_type is changed to challenge_action
            
            Then observation is returned
            
        if action_type == challenge_action
        If agent just did a challenge action (challenged bot or not)
            The challenge action is executed (or not if there was no challenge)
            action goes through if no challenge or chal failed
            
            action_type is changed to block_action(future) or
            action_type is changed to base_action, and next bot goes
        
        # TODO
        if action_type == block_action
        If agent just did a block action, on the action of the bot
            Bot makes decision about challenge randomly?
        """
        
        if self.game.turn.action_type =="base_action": # if agent just did their base action. BOt makes a base action
            self.step() 
            print(f"Bot claims {self.game.turn.current_base_action}")
            print(f"Agent may challenge")
            observation = self._get_obs()
            
            # then get info on if we've terminated
            terminated = self.game.win
            # truncated if agent is dead
            truncated = self.game.lost 
            
            # then get the reward for stepping/terminating
            reward = self.get_reward(terminated, truncated)
        
            return observation, reward, terminated, truncated, {}
        
        if self.game.turn.action_type =="challenge_action": 
            raise ValueError("Bot is not yet able to make challenge actions")
            

    
    

    def render(self):
        pass

