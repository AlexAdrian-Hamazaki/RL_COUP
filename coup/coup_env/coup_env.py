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
                                1:'block_action',
                                2:'challenge_action'}
        
        
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
        self._actions = list(set(self.game.actions.ALLOWED_ACTIONS + list(self.game.actions.CHALLENGABLE_ACTIONS) + ["challenge"]))
        self._actions.sort()
        self._action_space_map = dict(zip([n for n in range(len(self._actions))],
                                          [action for action in self._actions]))
        self.action_space = Discrete(len(self._actions))
        
        print(self._action_space_map)
        
    def _get_obs(self):
        """Function that actually returns an observation given the state of the game

        Returns:
            _type_: _description_
        """
        
        action_type = self.game.turn.action_type # what type of action is able to be selected here
        mycards = self.game.turn.current_chooser.cards # need to 
        mymoney = self.game.turn.current_chooser.coins
        myclaims = self.game.turn.current_chooser.claimed_cards
        my_deck_knowledge = self.game.turn.current_chooser.knowledge.deck_knowledge 
        others_claims = self.game.turn.current_chooser.knowledge.other_player_claims # turn this into cards instead of actions
        others_n_cards = self.game.turn.current_chooser.knowledge.other_player_n_cards # turn this into cards instead of actions
        others_money = self.game.turn.current_chooser.knowledge.other_player_n_coins
        
        revealed = self.game.revealed_cards
        turn_order = self.game.turn.turn_order
        
        base_action_player = self.game.turn.current_base_player.name
        current_base_action = self.game.turn.current_base_action
        if self.game.turn.action_type == "challenge_action" or self.game.turn.action_type == "block_action":
            base_action_target_player = self.game.turn.current_base_action.target_player
        else:
            base_action_target_player = None
            
            
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
            "base_action_player": base_action_player,
            "current_base_action": current_base_action,
            "base_action_target_player": base_action_target_player,
            'action_mask': action_mask
            }
        
        print(observation)
        return observation
    
    def _compute_action_mask(self):
        action_type = self.game.turn.action_type # what type of action is able to be selected here
        mymoney = self.game.turn.current_chooser.coins
        current_base_action = self.game.turn.current_base_action

        
        # init mask of 0s to represent valid actions
        mask = np.array([0] * len(self._actions))
        
        if action_type == 'base_action':
            if mymoney>10:
                good_indexes = [6,0]
            elif mymoney >=7:
                # if at base action, enable base actions
                good_indexes = [7,8,9,10,11,0,6]
            elif mymoney >=3:
                good_indexes = [7,8,9,10,11,0]
            else:
                good_indexes = [7,8,9,10,11]
            mask[good_indexes] = 1
            return mask
                
        elif action_type == "challenge_action":
            good_indexes = [5]
            mask[good_indexes] = 1
            return mask
        
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
        return self._compute_action_mask()[action] == 1
    
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
        
    
    def step(self, action):
        """Takes an action by the current agent
        returns observation, reward, termination_flag, truncation_flag, and info dict
        Args:
            actions (int): action to take
        """
        # check if action is valid
        if not self._is_action_valid(action):
            raise ValueError(f"Invalid action {action} for the current state.")

        
        #first make the step through the action
        self.game.turn.step(action)
        
        # then get the observation of the new environment
        observation = self._get_obs()
        
        # then get info on if we've terminated
        terminated = not self.game.on
        
        # then get the reward for stepping/terminating
        # Reward Rules
            # +10 if we win (termination flag)
            # +1 if we get money
            # +5 if we kill someone
        reward = self.get_reward()
        
        # then get any additional info and truncation flag
        return observation, reward, terminated, False, {}
        
    def get_reward(self):
        reward = 0
        return reward
    
    

    def render(self):
        pass
